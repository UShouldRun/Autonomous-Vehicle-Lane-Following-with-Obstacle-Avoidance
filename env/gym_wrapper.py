from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import cv2

from gymnasium import spaces

from env.webots_env import WebotsEnv
from env.reward import dense_reward, sparse_reward
from utils.metrics import EpisodeStats
from utils.observation import preprocess_obs, build_observation_space


class WebotsLaneEnv(gym.Env):
    """
      - Define observation_space and action_space
      - Implement reset() and step() per the Gym API
      - Compute reward and detect episode termination
      - Delegate all hardware interaction to WebotsEnv
      - Accumulate per-episode statistics (EpisodeStats) and expose the
        running and completed episodes for monitoring callbacks.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, render_mode=None):
        super().__init__()
        self.config      = config
        self.render_mode = render_mode

        self._reward_cfg  = config["reward"]
        self._action_type = config["action_space"]["type"]

        # Monitoring thresholds (optional section in config.yaml).
        mon_cfg                   = config.get("monitoring", {}) or {}
        self._near_miss_threshold = float(mon_cfg.get("near_miss_threshold", 1.0))
        lap_departure             = float(mon_cfg.get("lap_departure_distance", 20.0))
        lap_return                = float(mon_cfg.get("lap_return_distance", 5.0))

        # Truncation thresholds (optional section in config.yaml).
        ep_cfg                      = config.get("episode", {}) or {}
        self._max_steps             = int(ep_cfg.get("max_steps", 2000))
        self._stall_speed_threshold = float(ep_cfg.get("stall_speed_threshold", 0.1))
        self._stall_steps           = int(ep_cfg.get("stall_steps", 100))

        # Per-episode step / stall counters (reset each reset()).
        self._step_count: int    = 0
        self._stall_counter: int = 0
        # Previous cumulative distance, for the v2 distance-delta progress term.
        self._prev_distance: float = 0.0

        # Obstacle config: optional `obstacles` section in config.yaml.
        # When omitted or {"enabled": False}, the obstacle subsystem is
        # fully inert and behaviour matches the pre-obstacle baseline.
        obstacles_cfg = config.get("obstacles", {}) or {}

        # Hardware driver — the only place Webots is touched
        self._hw = WebotsEnv(
            near_miss_threshold    = self._near_miss_threshold,
            collision_threshold    = float(config["reward"].get("collision_threshold", 0.3)),
            lap_departure_distance = lap_departure,
            lap_return_distance    = lap_return,
            obstacles_cfg          = obstacles_cfg,
        )

        # ── Observation space ─────────────────────────────────────
        # Bounds/dtypes must match the output of preprocess_obs().
        cam_h, cam_w = config["env"]["camera_resolution"]
        lidar_size   = self._hw.lidar_size

        self.observation_space = build_observation_space(
            cam_h=cam_h, cam_w=cam_w, lidar_size=lidar_size, config=config,
        )

        if self._action_type == "continuous":
            # steering in [-1, 1], throttle in [0, 1] (forward-only).
            # Clamping throttle to [0, 1] eliminates reverse driving entirely:
            # - PPO initialises near the action-space centre → throttle≈0.5 → ~25 km/h
            #   so the car moves forward from the very first episode.
            # - Halves the throttle exploration space, making learning faster.
            # - Reverse is not needed for lane-following; allowing it only teaches
            #   the agent to exploit backward movement to avoid collisions.
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0,  1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            # 0=hard-left  1=hard-right  2=gentle-left  3=gentle-right  4=straight
            self.action_space = spaces.Discrete(5)

        self._current_stats: Optional[EpisodeStats]  = None
        self._completed_episodes: List[EpisodeStats] = []
        self._episode_start_time: float              = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Remove any dynamically spawned barrels (node.remove()) BEFORE
        # the teleport-based reset, then clear Python-side bookkeeping.
        self._hw.clear_obstacles()
        self._hw.reset()

        self._current_stats      = EpisodeStats()
        self._episode_start_time = float(self._hw.driver.getTime())
        self._step_count         = 0
        self._stall_counter      = 0
        self._prev_distance      = self._hw.get_forward_distance()
        self._prev_theta: float = None  # for theta-improvement bonus

        return self._get_obs(), {}

    def step(self, action):
        if self._action_type == "continuous":
            self._hw.apply_continuous(float(action[0]), float(action[1]))
        else:
            self._hw.apply_discrete(int(action))

        self._hw.step()
        # Recycle barrels behind the car and conditionally spawn a new one
        # ahead. No-op when obstacles are disabled, so this is safe to call
        # every step regardless of mode.
        self._hw.update_obstacles()

        # Raw sensor read — used by reward and stats (physical units).
        raw_obs = self._get_raw_obs()

        # Signed forward displacement this step — drives the v2 distance-based
        # progress term. get_forward_distance() projects displacement onto the
        # car's forward axis: positive when moving forward, negative when
        # reversing. The v2 reward then pays nothing for reverse-driving escapes.
        # (Path-length get_distance_travelled() is still tracked separately
        # and reported in EpisodeStats / safety_score for visibility.)
        current_distance    = self._hw.get_forward_distance()
        distance_delta      = current_distance - self._prev_distance
        self._prev_distance = current_distance

        # Step / stall bookkeeping.
        self._step_count += 1
        if float(raw_obs["state"][0]) < self._stall_speed_threshold:
            self._stall_counter += 1
        else:
            self._stall_counter = 0

        theta, line_visible = self._hw.get_alignment_angle()
        line_lost = self._yellow_line_is_not_visible(line_visible)

        # Termination check — physics-based touch sensor, no LiDAR threshold.
        terminated = self._is_collision() or line_lost
        truncated  = False

        # Truncation checks (after collision, before reward).
        if terminated:
            termination_reason = "collision" if self._is_collision() else "yellow line not visible"
        elif self._max_steps > 0 and self._step_count >= self._max_steps:
            truncated = True
            termination_reason = "max_steps"
        else:
            termination_reason = ""

        reward = self._compute_reward(
            raw_obs, terminated, line_lost, distance_delta,
            theta, self._prev_theta
        )

        self._prev_theta = float(theta)
        spd = float(raw_obs["state"][0])

        print(f"\rstep={self._step_count:5d}  line={'YES' if not line_lost else ' NO'}  "
              f"theta={theta:+.3f}  spd={spd:5.2f}m/s  rew={reward:+8.3f}   ", end="", flush=True)

        self._update_episode_stats(raw_obs, terminated, truncated, theta, reward)
        obs = preprocess_obs(raw_obs, self.config)

        return obs, reward, terminated, truncated, {"termination_reason": termination_reason}

    def render(self):
        pass  # Webots renders its own GUI window

    def close(self):
        pass

    def _get_raw_obs(self) -> dict:
        """Raw sensor dict with physical units and defensive resize.

        Used internally for reward, collision and statistics, which all
        need unit-bearing values. ``get_alignment_angle()`` reads the raw
        RGB image in HSV space to locate the yellow line, so this must
        always run against the un-normalised camera frame.
        """
        cam_h, cam_w = self.config["env"]["camera_resolution"]
        raw = self._hw.get_camera_image()
        img = cv2.resize(raw, (cam_w, cam_h))
        return {
            "camera": img,
            "lidar":  self._hw.get_lidar_scan(),
            "state":  np.array(
                [self._hw.get_forward_speed()],
                dtype=np.float32,
            ),
        }

    def _get_obs(self) -> dict:
        """Normalised observation for the agent (matches observation_space)."""
        return preprocess_obs(self._get_raw_obs(), self.config)

    def _is_collision(self) -> bool:
        """Physics-based collision detection via touch sensor."""
        return self._hw.is_collision()

    def _yellow_line_is_not_visible(self, yellow_score: float) -> bool:
        return yellow_score < 0.5
    
    def _compute_reward(self, obs: dict, terminated: bool, line_lost: bool, distance_delta: float, theta: float,
                        prev_theta: float = None) -> float:
        state         = obs["state"]
        v             = float(state[0])
        cfg           = self._reward_cfg

        reward_type   = cfg.get("type", "dense")
        lap_completed = self._hw.get_lap_completed()

        if reward_type == "dense":
            return dense_reward(v, theta, line_lost,
                            lap_completed, terminated, cfg,
                            distance_delta=distance_delta,
                            near_miss=self._hw.is_near_miss(),
                            prev_theta=prev_theta)
        elif reward_type == "sparse":
            return sparse_reward(checkpoint=False, collision=terminated)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")


    def _update_episode_stats(
        self, obs: dict, terminated: bool, truncated: bool, theta: float, reward: float = 0.0
    ) -> None:
        """Fold the latest step into the current EpisodeStats. On episode end,
        finalise the record and append it to the completed-episodes list."""
        stats = self._current_stats
        if stats is None:
            return

        stats.total_steps  += 1
        stats.total_reward += float(reward)
        stats.cross_track_errors.append(abs(theta))
        stats.distance_travelled = self._hw.get_distance_travelled()

        if self._hw.is_near_miss():
            stats.near_misses += 1

        if self._hw.get_lap_completed():
            last_lap = self._hw.get_last_lap_time()
            if last_lap is not None:
                stats.lap_times.append(float(last_lap))

        if terminated:
            stats.collisions += 1

        if terminated or truncated:
            self._completed_episodes.append(stats)
            self._current_stats = None

    @property
    def current_stats(self) -> Optional[EpisodeStats]:
        """Stats for the episode currently in progress, or None between episodes."""
        return self._current_stats

    @property
    def completed_episodes(self) -> List[EpisodeStats]:
        """All episodes that have terminated or been truncated so far."""
        return self._completed_episodes

