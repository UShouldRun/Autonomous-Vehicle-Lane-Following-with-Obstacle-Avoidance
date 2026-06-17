from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import get_linear_fn
from env.gym_wrapper import WebotsLaneEnv
from env.model import LaneCNN
from utils.callbacks import MetricsCallback, CheckpointCallback, BestModelCallback

import argparse
import os
import sys
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

    # Apply defaults for optional flags not supplied by the user
    if args.agent is None:
        args.agent = "ppo"
    if args.reward is None:
        args.reward = "dense"
    if args.config is None:
        args.config = "config.yaml"

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

    cfg["reward"]["type"]       = args.reward
    cfg["action_space"]["type"] = "continuous" if args.agent == "ppo" else "discrete"
    if args.obstacles:
        cfg.setdefault("obstacles", {})["enabled"] = True

    run_dir = os.path.join("results", args.run_tag)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        sys.exit(
            f"[ERROR] Run directory '{run_dir}' already exists and is not empty.\n"
            "  Pick a different --run-tag, or delete the directory manually if\n"
            "  you are sure you want to discard the previous run's outputs."
        )
    os.makedirs(run_dir, exist_ok=True)
    tb_dir = os.path.join(run_dir, "tb")

    obstacles_on = bool(cfg.get("obstacles", {}).get("enabled", False))
    print(
        f"[train] Building environment  agent={args.agent}  "
        f"reward={args.reward}  obstacles={'on' if obstacles_on else 'off'}  "
        f"run_tag={args.run_tag}"
    )
    env = WebotsLaneEnv(cfg)

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
        cnn_features_dim = int(agent_cfg.get("cnn_features_dim", 512))
        policy_kwargs = dict(
            features_extractor_class  = LaneCNN,
            features_extractor_kwargs = dict(features_dim=cnn_features_dim),
            net_arch = dict(pi=[256, 128], vf=[256, 128]),
        )
        agent_kwargs = {
            **common_kwargs,
            "n_steps":      agent_cfg.get("n_steps", 2048),
            "gae_lambda":   agent_cfg.get("gae_lambda", 0.95),
            "n_epochs":     agent_cfg.get("n_epochs", 10),
            "clip_range":   agent_cfg.get("clip_range", 0.2),
            "ent_coef":     agent_cfg.get("ent_coef", 0.0),
            "policy_kwargs": policy_kwargs,
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
