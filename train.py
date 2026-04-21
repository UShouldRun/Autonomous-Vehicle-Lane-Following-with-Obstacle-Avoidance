"""
train.py — Main training entry point.

This script IS the Webots robot controller.
Set it as the controller of the E-puck Supervisor node in your .wbt world.

Usage (Webots sets WEBOTS_ROBOT_NAME and calls this automatically):
    # From the Webots controller selector, point to this file.
    # Or launch manually for debugging (Webots must already be running):
    python train.py --agent ppo --reward dense --timesteps 50000
"""
import argparse
import csv
import os
import sys
from typing import Optional

import yaml

# ── Graceful import check ──────────────────────────────────────────────────
try:
    from controller import Supervisor  # noqa: F401  (just a presence check)
except ModuleNotFoundError:
    sys.exit(
        "[ERROR] 'controller' module not found.\n"
        "  train.py must be run as a Webots robot controller, not from a plain "
        "Python interpreter.\n"
        "  Set this file as the controller in your .wbt world, or ensure "
        "WEBOTS_HOME/lib/controller/python is on PYTHONPATH."
    )

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from env.gym_wrapper import WebotsLaneEnv
from utils.metrics import summarise


# ── Metrics callback ──────────────────────────────────────────────────────

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
                    summary["safety_score"],
                    summary["mean_steps"],
                    summary["mean_reward"],
                ])


# ── Checkpoint callback ──────────────────────────────────────────────────

class CheckpointCallback(BaseCallback):
    """
    Periodically save the model while training.

    Files are written as ``{save_path}/{name_prefix}_{num_timesteps}.zip`` —
    e.g. ``results/ppo_dense_checkpoint_50000.zip``. A trigger fires every
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


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent",     choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--reward",    choices=["dense", "ttc", "sparse"], default="dense")
    p.add_argument("--config",    default="config.yaml")
    p.add_argument("--timesteps", type=int, default=None,
                   help="Override total_timesteps from config")
    p.add_argument("--resume", default=None, metavar="PATH",
                   help="Path to a saved .zip model to continue training from")
    p.add_argument("--check-env", action="store_true",
                   help="Run SB3 env checker then exit (useful for first-run debugging)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────
    config_path = args.config
    if not os.path.exists(config_path):
        sys.exit(
            f"[ERROR] Config file not found: '{config_path}'\n"
            "  Copy config.yaml next to train.py (or pass --config <path>)."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI flags override config values
    cfg["reward"]["type"]       = args.reward
    cfg["action_space"]["type"] = "continuous" if args.agent == "ppo" else "discrete"

    # ── Output directory ──────────────────────────────────────────
    os.makedirs("results", exist_ok=True)

    # ── Build environment ─────────────────────────────────────────
    print(f"[train] Building environment  agent={args.agent}  reward={args.reward}")
    env = WebotsLaneEnv(cfg)

    # Optional: validate the env against the Gym API before training
    if args.check_env:
        print("[train] Running SB3 env checker …")
        check_env(env, warn=True)
        print("[train] Env check passed.")
        env.close()
        return

    # ── Build agent ───────────────────────────────────────────────
    timesteps = args.timesteps or cfg["training"]["total_timesteps"]

    agent_kwargs = dict(
        policy          = "MultiInputPolicy",
        env             = env,
        verbose         = 1,
        learning_rate   = cfg["agent"]["learning_rate"],
        batch_size      = cfg["agent"]["batch_size"],
        gamma           = cfg["agent"]["gamma"],
    )

    AgentCls = PPO if args.agent == "ppo" else DQN

    if args.resume:
        if not os.path.isfile(args.resume):
            sys.exit(
                f"[ERROR] --resume file not found: '{args.resume}'\n"
                "  Pass the full path to a saved .zip model."
            )
        print(f"[train] Resuming from '{args.resume}' …")
        # Loading with env=env rebinds the saved policy to the current
        # environment (equivalent to calling set_env afterwards). The saved
        # hyperparameters inside the zip take precedence over agent_kwargs.
        model = AgentCls.load(args.resume, env=env)
    else:
        model = AgentCls(**agent_kwargs)

    # ── Metrics callback ──────────────────────────────────────────
    mon_cfg       = cfg.get("monitoring", {}) or {}
    summary_every = int(mon_cfg.get("summary_every_n_episodes", 10))
    csv_log       = bool(mon_cfg.get("csv_log", True))
    csv_path      = (
        f"results/{args.agent}_{args.reward}_metrics.csv" if csv_log else None
    )
    metrics_cb = MetricsCallback(
        lane_env      = env,
        summary_every = summary_every,
        csv_path      = csv_path,
    )

    # ── Checkpoint callback ───────────────────────────────────────
    callbacks = [metrics_cb]
    ckpt_every = int(cfg.get("training", {}).get("checkpoint_every_n_steps", 0))
    if ckpt_every > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq   = ckpt_every,
                save_path   = "results",
                name_prefix = f"{args.agent}_{args.reward}_checkpoint",
                verbose     = 1,
            )
        )

    # ── Train ─────────────────────────────────────────────────────
    save_path = f"results/{args.agent}_{args.reward}_model"
    print(f"[train] Starting training for {timesteps} timesteps …")
    print(f"[train] Model will be saved to '{save_path}.zip'")
    if csv_path:
        print(f"[train] Metrics CSV → '{csv_path}' (every {summary_every} episodes)")
    if ckpt_every > 0:
        print(f"[train] Checkpoints every {ckpt_every} timesteps → 'results/'")

    try:
        model.learn(total_timesteps=timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[train] Interrupted — saving partial model …")
    finally:
        model.save(save_path)
        print(f"[train] Model saved → {save_path}.zip")

    env.close()
    print("[train] Done.")


if __name__ == "__main__":
    main()
