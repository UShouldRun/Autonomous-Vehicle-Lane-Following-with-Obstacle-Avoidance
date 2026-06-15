MAX_SPEED = 13.89  # m/s (50 km/h) — matches MAX_SPEED in webots_env.py


def dense_reward(
    forward_speed: float,
    theta_norm: float,
    line_lost: bool,
    lap_completed: bool,
    collision: bool,
    cfg: dict,
    distance_delta: float = 0.0,
) -> float:
    """
    reward = progress_term + speed_bonus + alignment_bonus
             - alignment_penalty - collision_penalty + lap_bonus

    Increases with:  net forward progress, high forward speed, good alignment
                     with yellow line, lap completions.
    Penalizes:       misalignment (quadratic), losing the line, collision,
                     and net backward movement (via the signed progress term).

    Parameters
    ----------
    forward_speed : float
        Signed forward speed in m/s. Positive = forward.
    theta_norm : float
        Lateral-offset proxy in [-2, 2]. 0 = centred, ±1 = frame edge,
        ±2 = line not visible.
    line_lost : bool
        True when no yellow pixels were visible this frame.
    lap_completed : bool
        True on the single step a lap closes.
    collision : bool
        True on the terminal collision step.
    cfg : dict
        reward section of config.yaml.
        Keys: w_progress, w_speed, w_alignment, w_alignment_bonus,
        w_line_lost, w_collision, w_lap.
    distance_delta : float
        Signed forward-axis displacement (m) since the previous step
        (WebotsEnv.get_forward_distance() delta). Positive when the car made
        net forward progress, negative when it net-reversed. This is the
        term that actually penalises backward driving — forward_speed alone
        does not, since it is clamped to >= 0 by speed_bonus.
    """
    w_progress       = float(cfg.get("w_progress",       1.0))
    w_speed          = float(cfg.get("w_speed",          1.0))
    w_alignment      = float(cfg.get("w_alignment",      1.0))
    w_alignment_bonus = float(cfg.get("w_alignment_bonus", 0.0))
    w_line_lost      = float(cfg.get("w_line_lost",      0.5))
    w_collision      = float(cfg.get("w_collision",      50.0))
    w_lap            = float(cfg.get("w_lap",            20.0))

    # Progress term: signed, so reversing yields a NEGATIVE reward instead
    # of merely "no bonus". This is what keeps the agent from treating
    # reverse-driving as a free/neutral action.
    progress_term = w_progress * distance_delta 

    # Speed bonus: normalized to [-1, 1]
    speed_bonus = w_speed * min(1.0, max(-1.0, forward_speed / MAX_SPEED))

    # Alignment: penalize when offset or line lost (existing behaviour),
    # AND separately reward being close to centred (new). The bonus uses
    # (1 - |theta_norm|) clamped to [0, 1] so it's 1.0 when perfectly
    # centred, 0 at the frame edge, and 0 (not negative) when the line is
    # lost or theta_norm is at/past the edge — it never fights the penalty
    # below, it just adds a positive incentive on top of it.
    if line_lost:
        alignment_penalty = w_line_lost
        alignment_bonus   = 0.0
    else:
        alignment_penalty = w_alignment * (theta_norm ** 4)  # 0 centred → grows toward edge
        alignment_bonus   = w_alignment_bonus * max(0.0, 1.0 - theta_norm ** 4)

    collision_penalty = w_collision if collision else 0.0
    lap_bonus         = w_lap if lap_completed else 0.0

    print(f"progress_term: {progress_term:+.4f}, speed_bonus: {speed_bonus:+.4f}, alignment_bonus: {alignment_bonus:+.4f}, lap_bonus: {lap_bonus:+.4f}")
    print(f"alignment_penalty: {alignment_penalty:+.4f}, collision_penalty: {collision_penalty:+.4f}")
    print(f"line_lost: {line_lost:+.1f} theta: {theta_norm:+.4f}, speed: {forward_speed:+.4f}, distance: {distance_delta:+.4f}")
    print(f"alignment_bonus - alignment_penalty: {(alignment_bonus - alignment_penalty):+.4f}")

    return (
        progress_term + speed_bonus + alignment_bonus + lap_bonus
        - alignment_penalty - collision_penalty
    )


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
