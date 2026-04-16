"""
gym_wrapper.py — Gymnasium interface for the RL agent.

Knows nothing about Webots devices directly.
Delegates all hardware reads/writes to WebotsEnv.
This is what stable-baselines3 talks to.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.webots_env import WebotsEnv
from env.reward import dense_reward, ttc_reward, sparse_reward


class WebotsLaneEnv(gym.Env):
    """
      - Define observation_space and action_space
      - Implement reset() and step() per the Gym API
      - Compute reward and detect episode termination
      - Delegate all hardware interaction to WebotsEnv
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, render_mode=None):
        super().__init__()
        self.config      = config
        self.render_mode = render_mode

        # Hardware driver — the only place Webots is touched
        self._hw = WebotsEnv()

        self._collision_threshold = config["reward"].get("collision_threshold", 0.3)
        self._reward_cfg          = config["reward"]
        self._action_type         = config["action_space"]["type"]

        # ── Observation space ─────────────────────────────────────
        cam_h, cam_w = config["env"]["camera_resolution"]
        lidar_size   = self._hw.lidar_size

        self.observation_space = spaces.Dict({
            "camera": spaces.Box(0, 255, shape=(cam_h, cam_w, 3), dtype=np.uint8),
            "lidar":  spaces.Box(0.0, 10.0, shape=(lidar_size,),  dtype=np.float32),
            "state":  spaces.Box(-np.inf, np.inf, shape=(2,),     dtype=np.float32),
            # state[0] = speed (m/s)
            # state[1] = alignment angle in [-1, 1]
        })

        # ── Action space ──────────────────────────────────────────
        if self._action_type == "continuous":
            # [steering, throttle] both in [-1, 1]
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            # 0=left  1=right  2=straight  3=brake
            self.action_space = spaces.Discrete(4)

    # ── Gym API ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._hw.reset()
        return self._get_obs(), {}

    def step(self, action):
        # 1. Send action to hardware
        if self._action_type == "continuous":
            self._hw.apply_continuous(float(action[0]), float(action[1]))
        else:
            self._hw.apply_discrete(int(action))

        # 2. Advance simulation by one timestep
        self._hw.step()

        # 3. Read new state
        obs = self._get_obs()

        # 4. Termination check
        terminated = self._is_collision(obs["lidar"])

        # 5. Reward
        reward = self._compute_reward(obs, terminated)

        return obs, reward, terminated, False, {}

    def render(self):
        pass  # Webots renders its own GUI window

    def close(self):
        pass

    # ── Observation ───────────────────────────────────────────────

    def _get_obs(self) -> dict:
        return {
            "camera": self._hw.get_camera_image(),
            "lidar":  self._hw.get_lidar_scan(),
            "state":  np.array(
                [self._hw.get_speed(), self._hw.get_alignment_angle()],
                dtype=np.float32,
            ),
        }

    # ── Termination ───────────────────────────────────────────────

    def _is_collision(self, lidar: np.ndarray) -> bool:
        return bool(lidar.min() < self._collision_threshold)

    # ── Reward ────────────────────────────────────────────────────

    def _compute_reward(self, obs: dict, terminated: bool) -> float:
        v     = float(obs["state"][0])   # speed
        theta = float(obs["state"][1])   # alignment angle
        d     = abs(theta)               # lateral distance proxy
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
