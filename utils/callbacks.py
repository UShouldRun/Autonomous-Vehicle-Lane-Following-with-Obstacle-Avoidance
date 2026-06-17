from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from env.gym_wrapper import WebotsLaneEnv
from utils.metrics import summarise

import csv
import os
import numpy as np

class MetricsCallback(BaseCallback):
    """
    SB3 callback that reads EpisodeStats accumulated by the WebotsLaneEnv
    and, every `summary_every` completed episodes, prints a one-line
    summary of the evaluation metrics defined in ``utils.metrics``
    (Section 4.1 of the project proposal):
      - success_rate_%            — episodes with zero collisions
      - mean_collisions           — per episode
      - mean_cross_track_error_m  — image-plane proxy, NOT metres
      - mean_lap_time_s           — across all completed laps
      - safety_score              — total distance / total near-misses

    If ``csv_path`` is given, each summary row is also appended to a CSV
    file for later plotting.
    """

    _CSV_FIELDS = [
        "n_episodes",
        "success_rate_%",
        "mean_collisions",
        "mean_cross_track_error_m",
        "mean_lap_time_s",
        "mean_laps_per_episode",
        "safety_score",
        "mean_steps",
        "mean_reward",
    ]

    def __init__(
        self,
        lane_env: WebotsLaneEnv,
        summary_every: int = 10,
        csv_path: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._lane_env          = lane_env
        self._summary_every     = max(int(summary_every), 1)
        self._csv_path          = csv_path
        self._last_reported     = 0

    def _on_training_start(self) -> None:
        if self._csv_path:
            # (Re)create the CSV and write the header row.
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._CSV_FIELDS)

    def _on_step(self) -> bool:
        episodes = self._lane_env.completed_episodes
        n = len(episodes)

        if n >= self._last_reported + self._summary_every:
            self._emit_summary(n, summarise(episodes))
            self._last_reported = n

        return True

    def _on_training_end(self) -> None:
        # Flush a final summary if any episodes closed after the last tick.
        episodes = self._lane_env.completed_episodes
        n = len(episodes)

        if n > self._last_reported:
            self._emit_summary(n, summarise(episodes))
            self._last_reported = n

    def _emit_summary(self, n: int, summary: dict) -> None:
        print(
            f"[metrics] episodes={n}  "
            f"success={summary['success_rate_%']:.1f}%  "
            f"collisions={summary['mean_collisions']:.2f}  "
            f"CTE={summary['mean_cross_track_error_m']:.3f}  "
            f"lap={summary['mean_lap_time_s']:.2f}s  "
            f"safety={summary['safety_score']:.2f}  "
            f"steps={summary['mean_steps']:.1f}  "
            f"reward={summary['mean_reward']:.2f}"
        )
        if self._csv_path:
            with open(self._csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    n,
                    summary["success_rate_%"],
                    summary["mean_collisions"],
                    summary["mean_cross_track_error_m"],
                    summary["mean_lap_time_s"],
                    summary["mean_laps_per_episode"],
                    summary["safety_score"],
                    summary["mean_steps"],
                    summary["mean_reward"],
                ])


class CheckpointCallback(BaseCallback):
    """
    Periodically save the model while training.

    Files are written as ``{save_path}/{name_prefix}_{num_timesteps}.zip`` —
    e.g. ``results/<run-tag>/checkpoint_100000.zip``. A trigger fires every
    ``save_freq`` training steps; ``save_freq <= 0`` disables the callback
    entirely and the caller should just not construct it in that case.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._save_freq   = max(int(save_freq), 1)
        self._save_path   = save_path
        self._name_prefix = name_prefix

    def _on_training_start(self) -> None:
        os.makedirs(self._save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self._save_freq == 0:
            # `model.save` appends the .zip extension itself.
            path = os.path.join(
                self._save_path, f"{self._name_prefix}_{self.num_timesteps}"
            )
            self.model.save(path)

            if self.verbose > 0:
                print(f"[checkpoint] saved → {path}.zip")

        return True


class BestModelCallback(BaseCallback):
    """
    Overwrite ``<save_path>/best.zip`` whenever the rolling mean reward over
    the last ``window`` completed episodes beats the previous best.

    Checked every ``check_every`` completed episodes, aligned with the
    MetricsCallback cadence. The first save can only happen once at least
    ``window`` episodes have been recorded, so the rolling window is full.
    """

    def __init__(
        self,
        lane_env: WebotsLaneEnv,
        save_path: str,
        window: int = 30,
        check_every: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._lane_env    = lane_env
        self._save_path   = save_path
        self._window      = max(int(window), 1)
        self._check_every = max(int(check_every), 1)
        self._last_checked = 0
        self._best_mean    = float("-inf")

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self._save_path) or ".", exist_ok=True)

    def _on_step(self) -> bool:
        episodes = self._lane_env.completed_episodes
        n = len(episodes)

        if n < self._window:
            return True

        if n < self._last_checked + self._check_every:
            return True

        self._last_checked = n

        window_eps  = episodes[-self._window:]
        mean_reward = float(np.mean([e.total_reward for e in window_eps]))

        if mean_reward > self._best_mean:
            self._best_mean = mean_reward
            self.model.save(self._save_path)
            print(
                f"[best] new best at step={self.num_timesteps} "
                f"eps={n} mean_reward_{self._window}={mean_reward:.2f}"
            )

        return True
