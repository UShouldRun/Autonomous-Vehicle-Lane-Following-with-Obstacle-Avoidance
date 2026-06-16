import argparse
import csv
import os
import sys
from typing import Optional

import numpy as np
import yaml

try:
    from controller import Supervisor  # noqa: F401  (just a presence check)
except ModuleNotFoundError:
    sys.exit(
        "[ERROR] 'controller' module not found.\n"
        "  train.py must be run as a Webots robot controller, not from a plain "
        "Python interpreter.\n"
        "  Set this file as the controller in your .wbt world, or ensure "
        "WEBOTS_HOME/lib/controller/python3 is on PYTHONPATH."
    )

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import get_linear_fn

from env.gym_wrapper import WebotsLaneEnv
from utils.metrics import summarise


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


def print_help() -> None:
    """Print a friendly, detailed help message and exit."""
    help_text = """
╔══════════════════════════════════════════════════════════════════════════╗
║                          train.py — Help                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Webots robot controller for lane-keeping RL training.                   ║
║  Must be run as a Webots controller (not a plain Python interpreter).    ║
╚══════════════════════════════════════════════════════════════════════════╝

USAGE
  python3 train.py [OPTIONS]

OPTIONS
  --agent   {ppo,dqn}              RL algorithm to use.
                                   • ppo  → Proximal Policy Optimisation
                                           (continuous action space)
                                   • dqn  → Deep Q-Network
                                           (discrete action space)
                                   Default: ppo

  --reward  {dense,sparse}         Reward shaping strategy.
                                   • dense  → per-step v·cos(θ) signal
                                   • sparse → reward only at episode end
                                   Default: dense

  --config  PATH                   Path to YAML config file.
                                   Default: config.yaml

  --timesteps INT                  Total training timesteps.
                                   Overrides total_timesteps in config.yaml.

  --resume  PATH                   Path to a saved .zip model to resume
                                   training from (e.g. results/<old-run>/best.zip).

  --check-env                      Run the SB3 environment checker then exit.
                                   Useful for validating your setup before a
                                   full training run.

  --obstacles                      Enable dynamic barrel-obstacle spawning.
                                   Barrels appear ahead of the car at random
                                   intervals (configured under the
                                   `obstacles` section of config.yaml) and
                                   are recycled once the car drives past
                                   them. Disabled by default — behaviour is
                                   then identical to the pre-obstacle code.

  --run-tag NAME                   REQUIRED. Identifier for this run. All
                                   outputs are written to results/<NAME>/.
                                   Pick a unique tag per run — train.py
                                   refuses to overwrite an existing run
                                   directory that already has files.

  --lr-schedule {fixed,linear}     Learning-rate schedule.
                                   • fixed  → constant learning_rate from
                                              config.yaml (default)
                                   • linear → linearly anneal from the
                                              configured learning_rate down
                                              to 0 over the full run
                                   Default: fixed

  --help, -h                       Show this help message and exit.

EXAMPLES
  # Basic PPO training with dense rewards (uses config.yaml defaults)
  python3 train.py --run-tag ppo_dense_baseline

  # DQN with sparse rewards for 100 000 timesteps
  python3 train.py --agent dqn --reward sparse --timesteps 100000 \
      --run-tag dqn_sparse_smoke

  # Resume a previous PPO run
  python3 train.py --agent ppo --resume results/ppo_dense_baseline/best.zip \
      --run-tag ppo_dense_resume

  # Validate environment setup before training
  python3 train.py --check-env --run-tag tmp_check

  # Train with dynamic barrel obstacles spawning ahead of the car
  python3 train.py --agent ppo --obstacles --timesteps 500 --run-tag ppo_obs

  # Use a custom config file
  python3 train.py --config configs/my_experiment.yaml --agent ppo \
      --run-tag my_experiment

  # Linear learning-rate decay
  python3 train.py --agent ppo --lr-schedule linear --run-tag ppo_dense_lrsched

OUTPUTS
  results/<run-tag>/best.zip                   Best model so far (rolling-window mean reward).
  results/<run-tag>/checkpoint_<step>.zip      Periodic checkpoints (if configured).
  results/<run-tag>/model.zip                  Final saved model.
  results/<run-tag>/metrics.csv                Per-episode metrics log (if enabled).
  results/<run-tag>/tb/                        TensorBoard logs.

NOTES
  • config.yaml must exist at the path given by --config.
  • train.py must be set as the Webots robot controller, or
    WEBOTS_HOME/lib/controller/python must be on PYTHONPATH.
  • Interrupt training at any time with Ctrl+C — the model will be saved.
"""
    print(help_text)
    sys.exit(0)


def parse_args():
    p = argparse.ArgumentParser(add_help=False)  # We handle --help ourselves
    p.add_argument("--agent",     choices=["ppo", "dqn"], default=None)
    p.add_argument("--reward",    choices=["dense", "sparse"],
                   default=None)
    p.add_argument("--config",    default=None)
    p.add_argument("--timesteps", type=int, default=None,
                   help="Override total_timesteps from config")
    p.add_argument("--resume",    default=None, metavar="PATH",
                   help="Path to a saved .zip model to continue training from")
    p.add_argument("--check-env", action="store_true",
                   help="Run SB3 env checker then exit (useful for first-run debugging)")
    p.add_argument("--obstacles", action="store_true",
                   help="Enable dynamic barrel obstacle spawning (see config.yaml: obstacles)")
    p.add_argument("--run-tag", dest="run_tag", default=None, metavar="NAME",
                   help="REQUIRED. Identifier for this run. Outputs go to results/<NAME>/.")
    p.add_argument("--lr-schedule", dest="lr_schedule",
                   choices=["fixed", "linear"], default="fixed",
                   help="Learning-rate schedule: 'fixed' (default) or 'linear' decay to 0.")
    p.add_argument("--help", "-h", action="store_true",
                   help="Show this help message and exit")

    args = p.parse_args()

    # Explicit --help / -h flag → always show help
    if args.help:
        print_help()

    # No arguments at all → show help with a short nudge
    if not sys.argv[1:]:
        print(
            "[train] No arguments provided — showing help.\n"
            "  Pass your chosen options, or use --help for full details.\n"
        )
        print_help()

    # Apply defaults for optional flags not supplied by the user
    if args.agent is None:
        args.agent = "ppo"
    if args.reward is None:
        args.reward = "dense"
    if args.config is None:
        args.config = "config.yaml"

    # --run-tag is required — fail early with a clear message.
    if not args.run_tag:
        sys.exit(
            "[ERROR] --run-tag is required.\n"
            "  Pick a unique identifier for this run, e.g.\n"
            "    python train.py --agent ppo --reward dense --run-tag ppo_dense_baseline\n"
            "  All outputs will be written to results/<run-tag>/."
        )

    return args


def main():
    args = parse_args()

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
    if args.obstacles:
        cfg.setdefault("obstacles", {})["enabled"] = True

    # ── Output directory (per-run isolation) ──────────────────────
    run_dir = os.path.join("results", args.run_tag)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        sys.exit(
            f"[ERROR] Run directory '{run_dir}' already exists and is not empty.\n"
            "  Pick a different --run-tag, or delete the directory manually if\n"
            "  you are sure you want to discard the previous run's outputs."
        )
    os.makedirs(run_dir, exist_ok=True)
    tb_dir = os.path.join(run_dir, "tb")

    # ── Build environment ─────────────────────────────────────────
    obstacles_on = bool(cfg.get("obstacles", {}).get("enabled", False))
    print(
        f"[train] Building environment  agent={args.agent}  "
        f"reward={args.reward}  obstacles={'on' if obstacles_on else 'off'}  "
        f"run_tag={args.run_tag}"
    )
    env = WebotsLaneEnv(cfg)

    # Optional: validate the env against the Gym API before training
    if args.check_env:
        print("[train] Running SB3 env checker …")
        check_env(env, warn=True)
        print("[train] Env check passed.")
        env.close()
        return

    # ── Build agent ───────────────────────────────────────────────
    timesteps  = args.timesteps or cfg["training"]["total_timesteps"]
    agent_cfg  = cfg["agent"]
    initial_lr = float(agent_cfg["learning_rate"])

    if args.lr_schedule == "linear":
        # get_linear_fn(start, end, end_fraction) returns a callable that
        # receives the remaining progress fraction (1.0 → 0.0 over training)
        # and linearly interpolates start → end over the full course of training.
        learning_rate = get_linear_fn(
            initial_lr,
            0.0,
            end_fraction=1.0,
        )
    else:
        learning_rate = initial_lr

    # Shared kwargs accepted by both PPO and DQN
    common_kwargs = dict(
        policy          = "MultiInputPolicy",
        env             = env,
        verbose         = 1,
        learning_rate   = learning_rate,
        batch_size      = agent_cfg["batch_size"],
        gamma           = agent_cfg["gamma"],
        tensorboard_log = tb_dir,
    )

    AgentCls = PPO if args.agent == "ppo" else DQN

    if AgentCls is PPO:
        agent_kwargs = {
            **common_kwargs,
            "n_steps":   agent_cfg.get("n_steps", 2048),
            "gae_lambda": agent_cfg.get("gae_lambda", 0.95),
            "n_epochs":  agent_cfg.get("n_epochs", 10),
            "clip_range": agent_cfg.get("clip_range", 0.2),
            "ent_coef":  agent_cfg.get("ent_coef", 0.0),
        }
    else:  # DQN
        agent_kwargs = {
            **common_kwargs,
            "buffer_size": agent_cfg.get("buffer_size", 10000),
        }

    if args.resume:
        if not os.path.isfile(args.resume):
            sys.exit(
                f"[ERROR] --resume file not found: '{args.resume}'\n"
                "  Pass the full path to a saved .zip model."
            )
        print(f"[train] Resuming from '{args.resume}' …")
        model = AgentCls.load(args.resume, env=env, tensorboard_log=tb_dir)
    else:
        model = AgentCls(**agent_kwargs)

    mon_cfg       = cfg.get("monitoring", {}) or {}
    summary_every = int(mon_cfg.get("summary_every_n_episodes", 10))
    csv_log       = bool(mon_cfg.get("csv_log", True))
    csv_path      = os.path.join(run_dir, "metrics.csv") if csv_log else None
    metrics_cb = MetricsCallback(
        lane_env      = env,
        summary_every = summary_every,
        csv_path      = csv_path,
    )

    callbacks = [metrics_cb]
    ckpt_every = int(cfg.get("training", {}).get("checkpoint_every_n_steps", 0))
    if ckpt_every > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq   = ckpt_every,
                save_path   = run_dir,
                name_prefix = "checkpoint",
                verbose     = 1,
            )
        )

    best_cb = BestModelCallback(
        lane_env    = env,
        save_path   = os.path.join(run_dir, "best"),
        window      = 30,
        check_every = summary_every,
        verbose     = 1,
    )
    callbacks.append(best_cb)

    save_path = os.path.join(run_dir, "model")
    print(f"[train] Starting training for {timesteps} timesteps …")
    print(f"[train] LR schedule: {args.lr_schedule} (initial={initial_lr})")
    print(f"[train] Run dir: '{run_dir}/'")
    print(f"[train] Final model will be saved to '{save_path}.zip'")
    if csv_path:
        print(f"[train] Metrics CSV → '{csv_path}' (every {summary_every} episodes)")
    if ckpt_every > 0:
        print(f"[train] Checkpoints every {ckpt_every} timesteps → '{run_dir}/'")
    print(f"[train] TensorBoard logs → '{tb_dir}/'")

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
