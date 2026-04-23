from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import cv2

from gymnasium import spaces

from env.webots_env import WebotsEnv
from env.reward import dense_reward, ttc_reward, sparse_reward
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

        self._collision_threshold = config["reward"].get("collision_threshold", 0.3)
        self._reward_cfg          = config["reward"]
        self._action_type         = config["action_space"]["type"]

        # Monitoring thresholds (optional section in config.yaml).
        mon_cfg                   = config.get("monitoring", {}) or {}
        self._near_miss_threshold = float(mon_cfg.get("near_miss_threshold", 1.0))
        lap_departure             = float(mon_cfg.get("lap_departure_distance", 20.0))
        lap_return                = float(mon_cfg.get("lap_return_distance", 5.0))

        # Truncation thresholds (optional section in config.yaml).
        ep_cfg                        = config.get("episode", {}) or {}
        self._max_steps               = int(ep_cfg.get("max_steps", 2000))
        self._stall_speed_threshold   = float(ep_cfg.get("stall_speed_threshold", 0.1))
        self._stall_steps             = int(ep_cfg.get("stall_steps", 100))

        # Per-episode step / stall counters (reset each reset()).
        self._step_count: int    = 0
        self._stall_counter: int = 0

        # Hardware driver — the only place Webots is touched
        self._hw = WebotsEnv(
            near_miss_threshold     = self._near_miss_threshold,
            collision_threshold     = self._collision_threshold,
            lap_departure_distance  = lap_departure,
            lap_return_distance     = lap_return,
        )

        # ── Observation space ─────────────────────────────────────
        # Bounds/dtypes must match the output of preprocess_obs().
        cam_h, cam_w = config["env"]["camera_resolution"]
        lidar_size   = self._hw.lidar_size

        self.observation_space = build_observation_space(
            cam_h=cam_h, cam_w=cam_w, lidar_size=lidar_size, config=config,
        )

        if self._action_type == "continuous":
            # [steering, throttle] both in [-1, 1]
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            # 0=left  1=right  2=straight  3=brake
            self.action_space = spaces.Discrete(4)

        self._current_stats: Optional[EpisodeStats] = None
        self._completed_episodes: List[EpisodeStats] = []
        self._episode_start_time: float              = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._hw.reset()

        self._current_stats      = EpisodeStats()
        self._episode_start_time = float(self._hw.robot.getTime())
        self._step_count         = 0
        self._stall_counter      = 0

        return self._get_obs(), {}

    def step(self, action):
        if self._action_type == "continuous":
            self._hw.apply_continuous(float(action[0]), float(action[1]))
        else:
            self._hw.apply_discrete(int(action))

        self._hw.step()

        # 3. Raw sensor read — used by reward, collision and stats because
        #    those expect physical units (m/s, metres). The agent will
        #    instead receive the normalised version returned below.
        raw_obs = self._get_raw_obs()

        # 4. Step / stall bookkeeping (raw speed in m/s from state[0]).
        self._step_count += 1
        if float(raw_obs["state"][0]) < self._stall_speed_threshold:
            self._stall_counter += 1
        else:
            self._stall_counter = 0

        # 5. Termination check (raw lidar in metres) — collision only
        terminated = self._is_collision(raw_obs["lidar"])
        truncated  = False

        # 6. Truncation checks (AFTER collision, BEFORE reward). Collision
        #    wins; otherwise max_steps or stalled may truncate the episode.
        if terminated:
            termination_reason = "collision"
        elif self._max_steps > 0 and self._step_count >= self._max_steps:
            truncated = True
            termination_reason = "max_steps"
        elif self._stall_counter >= self._stall_steps:
            truncated = True
            termination_reason = "stalled"
        else:
            termination_reason = ""

        # 7. Reward (raw speed in m/s, raw lidar in metres)
        reward = self._compute_reward(raw_obs, terminated)

        # 8. Episode stats (raw values)
        self._update_episode_stats(raw_obs, terminated, truncated, reward)

        # 9. Normalised observation for the agent
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
                [self._hw.get_forward_speed(), self._hw.get_alignment_angle()],
                dtype=np.float32,
            ),
        }

    def _get_obs(self) -> dict:
        """Normalised observation for the agent (matches observation_space)."""
        return preprocess_obs(self._get_raw_obs(), self.config)

    def _is_collision(self, lidar: np.ndarray) -> bool:
        return bool(lidar.min() < self._collision_threshold)

    def _compute_reward(self, obs: dict, terminated: bool) -> float:
        v     = float(obs["state"][0])   # speed
        theta = float(obs["state"][1])   # alignment angle
        # TODO: `d` is a proxy for the lateral distance to the yellow line —
        # it is the normalised image offset |theta| ∈ [0, 1], NOT a metric
        # cross-track error.
        d     = abs(theta)
        cfg   = self._reward_cfg

        reward_type = cfg.get("type", "dense")

        if reward_type == "dense":
            return dense_reward(v, theta, d, terminated, cfg)
        elif reward_type == "ttc":
            d_min = float(obs["lidar"].min())
            return ttc_reward(v, theta, d, d_min, terminated, cfg)
        elif reward_type == "sparse":
            return sparse_reward(checkpoint=False, collision=terminated)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    def _update_episode_stats(
        self, obs: dict, terminated: bool, truncated: bool, reward: float = 0.0
    ) -> None:
        """Fold the latest step into the current EpisodeStats. On episode end,
        finalise the record and append it to the completed-episodes list."""
        stats = self._current_stats
        if stats is None:
            return

        # Step / reward accumulation.
        stats.total_steps += 1
        stats.total_reward += float(reward)

        stats.cross_track_errors.append(abs(float(obs["state"][1])))

        # Distance travelled is cumulative in the HW driver — overwrite.
        stats.distance_travelled = self._hw.get_distance_travelled()

        # Near-miss: counted once per step that registers one.
        if self._hw.is_near_miss():
            stats.near_misses += 1

        # Lap completed this step → record its duration.
        if self._hw.get_lap_completed():
            last_lap = self._hw.get_last_lap_time()
            if last_lap is not None:
                stats.lap_times.append(float(last_lap))

        # Collision is terminal — count it exactly once on the terminating step.
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
