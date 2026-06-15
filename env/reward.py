import numpy as np


def dense_reward(v: float, theta: float, d: float, collision: bool, cfg: dict) -> float:
    """
    Reward = Progress - Alignment Penalty - Collision Penalty
    """
    w1, w2, w3 = cfg["w1"], cfg["w2"], cfg["w3"]
    progress          = w1 * v / abs(v)
    alignment         = w2 * (min(1 / abs(d) if d != 0 else 2, 2) - abs(d))
    collision_penalty = w3 * abs(v) if collision else 0.0
    return progress + alignment - collision_penalty


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


# ── v2 rewards ────────────────────────────────────────────────────────────
# Redesigned after Round 1 (see prompts/round2_reward_v2.md). Three changes
# vs the v1 dense/ttc rewards above:
#   1. Progress is distance-based (metres travelled this step), not v*cos(θ).
#      θ is an image-plane lateral-offset proxy in [-1, 1], NOT an angle, so
#      feeding it to cos() was a semantic error. Here it only modulates the
#      progress via a linear alignment factor.
#   2. Losing the yellow line is an explicit, distinct state (line_lost) with
#      its own per-step penalty — no longer aliased onto "line at the extreme".
#   3. A lap-completion bonus gives the sparse terminal signal the policy was
#      missing, and the collision penalty matches the dense one (w_collision),
#      removing the v1 ttc reward-hack where a 10× weaker terminal made early
#      collisions profitable.

def dense_v2(
    distance_delta: float,
    theta_norm: float,
    line_lost: bool,
    lap_completed: bool,
    collision: bool,
    cfg: dict,
) -> float:
    """Distance-based progress modulated by alignment, plus terminal signals.

    Parameters
    ----------
    distance_delta : float
        Metres travelled this step (Δ of get_distance_travelled()). Always ≥ 0.
    theta_norm : float
        Lateral-offset proxy ∈ [-1, 1]; only meaningful when not line_lost.
    line_lost : bool
        True when no yellow pixels were visible this frame.
    lap_completed : bool
        True on the single step a lap closes.
    collision : bool
        True on the terminal collision step.
    cfg : dict
        The ``reward`` config section. Weights: w_progress, w_line_lost,
        w_lap, w_collision.

    The alignment factor is linear, ``(1 - |theta_norm|)``: it keeps a constant
    corrective gradient all the way to the frame edge (where the line is about
    to be lost), unlike a quadratic/exponential factor that flattens there.
    """
    w_progress  = float(cfg.get("w_progress", 1.0))
    w_line_lost = float(cfg.get("w_line_lost", 1.0))
    w_lap       = float(cfg.get("w_lap", 50.0))
    w_collision = float(cfg.get("w_collision", 100.0))

    if line_lost:
        progress = -w_line_lost
    else:
        alignment = 1.0 - abs(theta_norm)          # 1 centred → 0 at the edge
        progress  = w_progress * distance_delta * alignment

    lap_bonus         = w_lap if lap_completed else 0.0
    collision_penalty = -w_collision if collision else 0.0

    return progress + lap_bonus + collision_penalty


def ttc_v2(
    distance_delta: float,
    theta_norm: float,
    line_lost: bool,
    lap_completed: bool,
    collision: bool,
    d_min_lidar: float,
    cfg: dict,
) -> float:
    """``dense_v2`` plus a calibrated proximity-safety term.

    The safety term is a linear ramp that only switches on inside ``d_safe_v2``
    (default 0.5 m — half the v1 radius, so it no longer fires against distant
    walls every step and stops smothering exploration). It reads ``d_safe_v2``
    rather than the legacy ``d_safe`` (1.0 m) on purpose.
    """
    base = dense_v2(distance_delta, theta_norm, line_lost, lap_completed, collision, cfg)

    d_safe   = float(cfg.get("d_safe_v2", 0.5))
    w_safety = float(cfg.get("w_safety", 2.0))

    safety_penalty = 0.0
    if d_min_lidar < d_safe:
        safety_penalty = -w_safety * (1.0 - d_min_lidar / d_safe)

    return base + safety_penalty


REWARD_FNS = {
    "dense":    dense_reward,
    "ttc":      ttc_reward,
    "sparse":   sparse_reward,
    "dense_v2": dense_v2,
    "ttc_v2":   ttc_v2,
}
