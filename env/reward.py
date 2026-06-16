"""
rewards.py — Reward engineering functions for vehicle policy training.

Provides dense tracking rewards with non-flatlining exponential gradients
and sparse checkpoint fallback functions.
"""

import numpy as np

# Speed sensor returns m/s. MAX_SPEED_MS = 50 km/h in m/s.
MAX_SPEED_MS = 50.0 / 3.6   # ≈ 13.89 m/s

def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))

def dense_reward(
    forward_speed: float,
    theta_norm: float,
    line_lost: bool,
    lap_completed: bool,
    collision: bool,
    cfg: dict,
    distance_delta: float = 0.0,
    near_miss: bool = False,
    prev_theta_norm: float = None,
) -> float:
    """
    Dense tracking reward with alignment as the primary learning signal.

    Key design choices:
    - Progress is GATED by alignment_factor: car earns forward reward only
      when roughly centred on the line. This prevents "drive fast off-road".
    - A theta-improvement bonus rewards reducing |theta| each step,
      giving an immediate gradient toward centering.
    - Alignment weights dominate over speed so the policy prioritises
      staying on the line over raw velocity.
    """
    w_progress          = float(cfg.get("w_progress", 2.0))
    w_speed             = float(cfg.get("w_speed", 1.0))
    w_alignment_penalty = float(cfg.get("w_alignment_penalty", 3.0))
    w_alignment_bonus   = float(cfg.get("w_alignment_bonus", 3.0))
    w_theta_improve     = float(cfg.get("w_theta_improve", 3.0))
    w_line_lost         = float(cfg.get("w_line_lost", 20.0))
    w_collision         = float(cfg.get("w_collision", 100.0))
    w_lap               = float(cfg.get("w_lap", 50.0))
    w_existence         = float(cfg.get("w_existence", 0.01))
    w_near_miss         = float(cfg.get("w_near_miss", 5.0))
    w_sigmoid           = float(cfg.get("w_sigmoid", 10.0))

    if collision:
        return w_sigmoid * (sigmoid(float(-w_collision)) - 1/2)
    if line_lost:
        return w_sigmoid * (sigmoid(float(-w_line_lost)) - 1/2)

    if prev_theta_norm is not None:
        theta_improvement = np.clip(abs(prev_theta_norm) - abs(theta_norm), -0.5, 0.5)
        theta_improve_bonus = w_theta_improve * theta_improvement
    else:
        theta_improve_bonus = 0.0

    alignment_penalty = w_alignment_penalty * np.exp(- 2 * (1 - abs(theta_norm) ** 1/2))
    alignment_bonus   = w_alignment_bonus * np.exp(- 2 * abs(theta_norm) ** 1/2)

    gated_progress = w_progress * distance_delta * (1 + alignment_bonus)

    normalized_speed = np.clip(forward_speed / MAX_SPEED_MS, 0.0, 1.0)
    speed_bonus = w_speed * normalized_speed * (1 + alignment_bonus)
    
    lap_bonus         = w_lap if lap_completed else 0.0
    near_miss_penalty = w_near_miss if near_miss else 0.0
    existence_bonus   = w_existence

    reward = (
        existence_bonus
        + gated_progress
        + speed_bonus
        + alignment_bonus
        + theta_improve_bonus
        + lap_bonus
        - alignment_penalty
        - near_miss_penalty
    )

    return w_sigmoid * (sigmoid(float(reward)) - 1/2)

def sparse_reward(checkpoint: bool, collision: bool) -> float:
    """+1 on checkpoint/lap, -1 on collision, 0 otherwise."""
    if collision:
        return -1.0
    if checkpoint:
        return 1.0
    return 0.0


REWARD_FNS = {
    "dense":  dense_reward,
    "sparse": sparse_reward,
}

