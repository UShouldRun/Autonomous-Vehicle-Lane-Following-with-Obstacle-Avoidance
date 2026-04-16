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
import os
import sys
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
from stable_baselines3.common.env_checker import check_env

from env.gym_wrapper import WebotsLaneEnv


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent",     choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--reward",    choices=["dense", "ttc", "sparse"], default="dense")
    p.add_argument("--config",    default="config.yaml")
    p.add_argument("--timesteps", type=int, default=None,
                   help="Override total_timesteps from config")
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

    if args.agent == "ppo":
        model = PPO(**agent_kwargs)
    else:
        model = DQN(**agent_kwargs)

    # ── Train ─────────────────────────────────────────────────────
    save_path = f"results/{args.agent}_{args.reward}_model"
    print(f"[train] Starting training for {timesteps} timesteps …")
    print(f"[train] Model will be saved to '{save_path}.zip'")

    try:
        model.learn(total_timesteps=timesteps)
    except KeyboardInterrupt:
        print("\n[train] Interrupted — saving partial model …")
    finally:
        model.save(save_path)
        print(f"[train] Model saved → {save_path}.zip")

    env.close()
    print("[train] Done.")


if __name__ == "__main__":
    main()
