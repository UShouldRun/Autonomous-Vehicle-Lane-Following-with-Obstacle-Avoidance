"""Evaluation metrics as defined in the project proposal (Section 4.1)."""
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class EpisodeStats:
    collisions: int = 0
    lap_times: List[float] = field(default_factory=list)
    cross_track_errors: List[float] = field(default_factory=list)  # metres
    distance_travelled: float = 0.0
    near_misses: int = 0
    total_steps: int = 0
    total_reward: float = 0.0


def success_rate(episodes: List[EpisodeStats]) -> float:
    """Percentage of episodes completed without any collision."""
    return 100.0 * sum(e.collisions == 0 for e in episodes) / len(episodes)


def mean_collisions(episodes: List[EpisodeStats]) -> float:
    return np.mean([e.collisions for e in episodes])


def mean_cross_track_error(episodes: List[EpisodeStats]) -> float:
    all_cte = [cte for e in episodes for cte in e.cross_track_errors]
    return float(np.mean(all_cte)) if all_cte else float("nan")


def mean_lap_time(episodes: List[EpisodeStats]) -> float:
    all_laps = [t for e in episodes for t in e.lap_times]
    return float(np.mean(all_laps)) if all_laps else float("nan")


def safety_score(episodes: List[EpisodeStats]) -> float:
    """Ratio of total distance to total near-misses (higher = safer)."""
    total_dist = sum(e.distance_travelled for e in episodes)
    total_nm   = sum(e.near_misses for e in episodes) or 1
    return total_dist / total_nm


def mean_steps(episodes: List[EpisodeStats]) -> float:
    return np.mean([e.total_steps for e in episodes])


def mean_reward(episodes: List[EpisodeStats]) -> float:
    return np.mean([e.total_reward for e in episodes])


def summarise(episodes: List[EpisodeStats]) -> dict:
    return {
        "success_rate_%":       success_rate(episodes),
        "mean_collisions":      mean_collisions(episodes),
        "mean_cross_track_error_m": mean_cross_track_error(episodes),
        "mean_lap_time_s":      mean_lap_time(episodes),
        "safety_score":         safety_score(episodes),
        "mean_steps":           mean_steps(episodes),
        "mean_reward":          mean_reward(episodes),
    }
