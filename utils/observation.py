"""
observation.py — Observation pre-processing utilities.

Converts the raw sensor dict produced by ``WebotsLaneEnv._get_raw_obs()`` into
a normalised dict suitable for feeding into a neural-network policy, and
builds the Gym ``Dict`` space that matches this normalised layout.

All transforms are pure: their input arrays are not mutated.

Raw → normalised channel summary
--------------------------------
camera : uint8   (H, W, 3)   in [0, 255]   → float32 (H, W, 1|3) in [0, 1]
lidar  : float32 (lidar_size,) in [0, max_range]
                                             → float32 (lidar_size,) in [0, 1]
state  : float32 (2,)        [speed m/s, alignment ∈ [-1, 1]]
                                             → float32 (2,) [speed/max_speed clipped
                                                             to [0, 1], alignment]
"""

from typing import Dict

import numpy as np
import cv2
from gymnasium import spaces


# ── Individual normalisers ──────────────────────────────────────────────────

def normalize_camera(img: np.ndarray, grayscale: bool = False) -> np.ndarray:
    """Convert a uint8 RGB image to float32 in [0, 1].

    If ``grayscale`` is True the RGB channels are collapsed and a trailing
    singleton dimension is kept, so the output shape is (H, W, 1) — keeping
    the array compatible with CNN feature extractors that expect a channel
    axis.
    """
    if grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return (gray.astype(np.float32) / 255.0)[..., None]
    return img.astype(np.float32) / 255.0


def normalize_state(state: np.ndarray, max_speed: float) -> np.ndarray:
    """Normalise the vehicle-state vector.

    state[0] — speed in m/s, divided by ``max_speed`` and clipped to [0, 1].
    state[1] — alignment angle proxy already in [-1, 1]; passes through unchanged.
    """
    out = state.astype(np.float32).copy()
    denom = max(float(max_speed), 1e-6)
    out[0] = float(np.clip(out[0] / denom, 0.0, 1.0))
    return out


def normalize_lidar(scan: np.ndarray, max_range: float) -> np.ndarray:
    """Scale a LiDAR scan from [0, max_range] metres to [0, 1]."""
    denom = max(float(max_range), 1e-6)
    return np.clip(scan.astype(np.float32) / denom, 0.0, 1.0)


# ── Combined transforms ─────────────────────────────────────────────────────

def preprocess_obs(obs: Dict[str, np.ndarray], config: dict) -> Dict[str, np.ndarray]:
    """Apply all three normalisations according to the ``observation`` section
    of ``config``. Returns a new dict; ``obs`` is not mutated.
    """
    obs_cfg    = config.get("observation", {}) or {}
    grayscale  = bool(obs_cfg.get("grayscale", False))
    max_speed  = float(obs_cfg.get("max_speed", 15.0))
    norm_lidar = bool(obs_cfg.get("normalize_lidar", True))
    lidar_max  = float(obs_cfg.get("lidar_max_range", 10.0))

    out = {
        "camera": normalize_camera(obs["camera"], grayscale=grayscale),
        "state":  normalize_state(obs["state"], max_speed=max_speed),
        "lidar":  (
            normalize_lidar(obs["lidar"], max_range=lidar_max)
            if norm_lidar
            else obs["lidar"].astype(np.float32)
        ),
    }
    return out


def build_observation_space(
    cam_h: int, cam_w: int, lidar_size: int, config: dict
) -> spaces.Dict:
    """Build the Gym ``Dict`` space matching the output of ``preprocess_obs``.

    The returned bounds and dtypes are kept in lock-step with
    ``preprocess_obs``: if you change one you must change the other.
    """
    obs_cfg    = config.get("observation", {}) or {}
    grayscale  = bool(obs_cfg.get("grayscale", False))
    norm_lidar = bool(obs_cfg.get("normalize_lidar", True))
    lidar_max  = float(obs_cfg.get("lidar_max_range", 10.0))

    cam_channels = 1 if grayscale else 3
    camera_space = spaces.Box(
        low=0.0, high=1.0,
        shape=(cam_h, cam_w, cam_channels),
        dtype=np.float32,
    )

    if norm_lidar:
        lidar_space = spaces.Box(
            low=0.0, high=1.0, shape=(lidar_size,), dtype=np.float32,
        )
    else:
        lidar_space = spaces.Box(
            low=0.0, high=float(lidar_max), shape=(lidar_size,), dtype=np.float32,
        )

    # state[0] = speed/max_speed ∈ [0, 1]; state[1] = alignment angle ∈ [-1, 1]
    state_space = spaces.Box(
        low=np.array([0.0, -1.0], dtype=np.float32),
        high=np.array([1.0,  1.0], dtype=np.float32),
        dtype=np.float32,
    )

    return spaces.Dict({
        "camera": camera_space,
        "lidar":  lidar_space,
        "state":  state_space,
    })
