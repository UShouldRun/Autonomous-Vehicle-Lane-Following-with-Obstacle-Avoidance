"""
evaluate.py — Inference-only evaluation of a trained SB3 model.

Loads a saved PPO or DQN model (algorithm auto-detected), runs N episodes
against ``WebotsLaneEnv`` and writes per-episode metrics to a CSV file.
No ``model.learn()`` is called — weights are never updated.

This script IS a Webots robot controller. Set it as the controller of the
Supervisor node in your .wbt world, or launch it from a Webots-aware
Python environment (same rule as train.py).

Usage:
    python evaluate.py --model results/ppo_dense_model.zip --episodes 100
"""
import argparse
import csv
import os
import sys

import numpy as np
import yaml

# ── Graceful import check ──────────────────────────────────────────────────
try:
    from controller import Supervisor  # noqa: F401  (just a presence check)
except ModuleNotFoundError:
    sys.exit(
        "[ERROR] 'controller' module not found.\n"
        "  evaluate.py must be run as a Webots robot controller, not from a plain "
        "Python interpreter.\n"
        "  Set this file as the controller in your .wbt world, or ensure "
        "WEBOTS_HOME/lib/controller/python is on PYTHONPATH."
    )

from gymnasium import spaces
from stable_baselines3 import PPO, DQN

from env.gym_wrapper import WebotsLaneEnv
from utils.metrics import summarise


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained PPO/DQN model (inference only)."
    )
    p.add_argument("--model",      required=True,
                   help="Path to a saved SB3 .zip model")
    p.add_argument("--config",     default="config.yaml",
                   help="YAML config used to build the environment")
    p.add_argument("--episodes",   type=int, default=100,
                   help="Number of evaluation episodes to run")
    p.add_argument("--max-steps",  type=int, default=10000,
                   help="Max steps per episode before truncation (100s at 10ms/step)")
    p.add_argument("--no-lap-stop", action="store_true",
                   help="Do NOT terminate episodes on lap completion")
    p.add_argument("--stochastic", action="store_true",
                   help="Use deterministic=False in model.predict()")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────

def load_model(path: str):
    """Try PPO first, fall back to DQN. Returns ``(model, algo_name)``."""
    errors = []
    for cls, name in [(PPO, "PPO"), (DQN, "DQN")]:
        try:
            return cls.load(path, device="auto"), name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    sys.exit(
        f"[ERROR] Could not load '{path}' as PPO or DQN:\n  "
        + "\n  ".join(errors)
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Config + model file sanity ────────────────────────────────
    if not os.path.exists(args.config):
        sys.exit(
            f"[ERROR] Config file not found: '{args.config}'\n"
            "  Copy config.yaml next to evaluate.py (or pass --config <path>)."
        )
    if not os.path.exists(args.model):
        sys.exit(f"[ERROR] Model file not found: '{args.model}'")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load model (algorithm auto-detected) ──────────────────────
    print(f"[eval] Loading model '{args.model}' …")
    model, algo_name = load_model(args.model)
    print(f"[eval] Detected algorithm: {algo_name}")

    # The env must match the model's action space, so override config.
    if isinstance(model.action_space, spaces.Discrete):
        cfg["action_space"]["type"] = "discrete"
    else:
        cfg["action_space"]["type"] = "continuous"

    # ── Build environment and bind to the model ──────────────────
    reward_type = cfg.get("reward", {}).get("type", "dense")
    print(f"[eval] Building environment  "
          f"action_space={cfg['action_space']['type']}  "
          f"reward={reward_type}")
    env = WebotsLaneEnv(cfg)
    model.set_env(env)

    deterministic = not args.stochastic

    # ── Output path ───────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    csv_path   = f"results/eval_{model_name}_{args.episodes}ep.csv"

    csv_fields = [
        "episode", "collisions", "lap_completed", "lap_time_s",
        "mean_cte", "total_distance_m", "near_misses",
        "total_steps", "total_reward", "termination_reason",
    ]

    # ── Evaluation loop ───────────────────────────────────────────
    print(f"[eval] Running {args.episodes} episodes  "
          f"max_steps={args.max_steps}  deterministic={deterministic}")
    print(f"[eval] CSV → '{csv_path}'")

    episode_stats   = []
    episode_rewards = []

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields)

            for ep in range(1, args.episodes + 1):
                obs, _ = env.reset()
                total_reward       = 0.0
                step_count         = 0
                terminated         = False
                termination_reason = ""

                while True:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    step_count   += 1

                    if terminated:
                        termination_reason = info.get("termination_reason", "collision")
                        break
                    if truncated:
                        # Env-level truncation (max_steps or stalled from gym_wrapper)
                        termination_reason = info.get("termination_reason", "truncated")
                        break
                    if not args.no_lap_stop and env._hw.get_lap_completed():
                        termination_reason = "lap_complete"
                        break
                    if step_count >= args.max_steps:
                        termination_reason = "eval_max_steps"
                        break

                # When terminated or truncated by the env (collision, stalled,
                # env max_steps), stats have been moved to completed_episodes
                # and current_stats is None. For evaluate-only breaks
                # (lap_complete, eval_max_steps), stats still in current_stats.
                if terminated or truncated:
                    stats = env.completed_episodes[-1]
                else:
                    stats = env.current_stats

                episode_stats.append(stats)
                episode_rewards.append(total_reward)

                lap_completed = (termination_reason == "lap_complete")
                last_lap      = env._hw.get_last_lap_time()
                lap_time_s    = float(last_lap) if last_lap is not None else float("nan")
                mean_cte      = (
                    float(np.mean(stats.cross_track_errors))
                    if stats.cross_track_errors
                    else float("nan")
                )

                writer.writerow([
                    ep,
                    int(stats.collisions),
                    bool(lap_completed),
                    lap_time_s,
                    mean_cte,
                    float(stats.distance_travelled),
                    int(stats.near_misses),
                    int(step_count),
                    float(total_reward),
                    termination_reason,
                ])
                f.flush()

                print(
                    f"[eval] ep {ep:3d}/{args.episodes}  "
                    f"steps={step_count:4d}  "
                    f"reward={total_reward:8.2f}  "
                    f"reason={termination_reason:<12s}  "
                    f"dist={stats.distance_travelled:6.2f}m  "
                    f"nm={stats.near_misses}"
                )
    except KeyboardInterrupt:
        print("\n[eval] Interrupted — partial CSV preserved.")

    # ── Summary ───────────────────────────────────────────────────
    n_eps = len(episode_stats)
    if n_eps == 0:
        print("[eval] No episodes completed.")
        env.close()
        return

    summary   = summarise(episode_stats)
    laps_done = sum(1 for s in episode_stats if s.lap_times)
    laps_pct  = 100.0 * laps_done / n_eps
    mean_r    = float(np.mean(episode_rewards))

    print("")
    print(f"[eval] ────── Summary ({n_eps} episodes) ──────")
    print(f"[eval] success_rate        : {summary['success_rate_%']:.1f} %")
    print(f"[eval] mean_collisions     : {summary['mean_collisions']:.2f}")
    print(f"[eval] mean_cross_track_err: {summary['mean_cross_track_error_m']:.3f}")
    print(f"[eval] mean_lap_time       : {summary['mean_lap_time_s']:.2f} s")
    print(f"[eval] safety_score        : {summary['safety_score']:.2f}")
    print(f"[eval] laps_completed      : {laps_done}/{n_eps} ({laps_pct:.1f} %)")
    print(f"[eval] mean_total_reward   : {mean_r:.2f}")
    print(f"[eval] CSV written         : {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
