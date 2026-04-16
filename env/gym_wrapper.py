import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WebotsLaneEnv(gym.Env):
    """Gymnasium wrapper around the Webots E-puck environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, render_mode=None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        cam_h, cam_w = config["env"]["camera_resolution"]

        # Observation: RGB frame + LiDAR + [speed, alignment_angle]
        self.observation_space = spaces.Dict({
            "camera": spaces.Box(low=0, high=255, shape=(cam_h, cam_w, 3), dtype=np.uint8),
            "lidar":  spaces.Box(low=0.0, high=10.0, shape=(360,), dtype=np.float32),
            "state":  spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        })

        if config["action_space"]["type"] == "continuous":
            # [steering, throttle] ∈ [-1, 1]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            # 0=left, 1=right, 2=straight, 3=brake
            self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._apply_action(action)
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._check_collision()
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> dict:
        # TODO: read camera, LiDAR, speed, alignment_angle from Webots
        raise NotImplementedError

    def _apply_action(self, action):
        # TODO: send velocity commands to E-puck motors
        raise NotImplementedError

    def _compute_reward(self, obs) -> float:
        # TODO: delegate to reward.py
        raise NotImplementedError

    def _check_collision(self) -> bool:
        # TODO: check LiDAR min distance against threshold
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass
