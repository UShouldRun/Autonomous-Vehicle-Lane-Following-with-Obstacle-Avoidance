import numpy as np


def dense_reward(v: float, theta: float, d: float, collision: bool, cfg: dict) -> float:
    """
    Reward = Progress - Alignment Penalty - Collision Penalty
    """
    w1, w2, w3 = cfg["w1"], cfg["w2"], cfg["w3"]
    progress          = w1 * v * np.cos(theta)
    alignment_penalty = w2 * abs(d)
    collision_penalty = w3 * 100.0 if collision else 0.0
    return progress - alignment_penalty - collision_penalty


def ttc_reward(v: float, theta: float, d: float, d_min: float,
               collision: bool, cfg: dict) -> float:
    """
    Reward = Progress - Deviation Penalty - Safety Term - Terminal Penalty
    """
    w1, w2, w3, w4 = cfg["w1"], cfg["w2"], cfg["w3"], cfg["w4"]
    d_safe = cfg["d_safe"]

    progress         = w1 * v * np.cos(theta)
    deviation        = w2 * d ** 2
    safety           = w3 * max(0.0, 1.0 - d_min / d_safe) ** 2
    terminal         = w4 if collision else 0.0
    return progress - deviation - safety - terminal


def sparse_reward(checkpoint: bool, collision: bool) -> float:
    """
    +1 on checkpoint/lap, -1 on collision, 0 otherwise.
    Used with curriculum learning (Stage 1 → 2 → 3).
    """
    if collision:
        return -1.0
    if checkpoint:
        return 1.0
    return 0.0


REWARD_FNS = {
    "dense":  dense_reward,
    "ttc":    ttc_reward,
    "sparse": sparse_reward,
}
