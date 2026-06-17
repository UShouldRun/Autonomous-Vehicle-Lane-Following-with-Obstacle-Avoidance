"""
hyper.py — Hyperparameter search for PPO and DQN
============================================================
Uses Optuna (TPE sampler) to find good hyperparameters for both agents.

• 25 trials per algorithm
• 10 000 timesteps per trial
• Optimises for mean episode reward over the last REWARD_WINDOW completed episodes

KEY DESIGN: The Webots Driver is a singleton — it is initialised once when the
controller process starts and cannot be re-created.  To avoid the
  AttributeError: 'Driver' object has no attribute 'devices'
error that occurs when WebotsLaneEnv.__init__ is called more than once, we
create ONE env per agent and reuse it across all trials.  Between trials we:
  1. Clear the completed-episodes list so each trial is scored independently.
  2. Call env.reset() to put the car back at the start.
  3. Build a fresh SB3 model (cheap — just Python/PyTorch objects).

Usage (must be run as a Webots robot controller):
    python3 hyper.py --agent ppo
    python3 hyper.py --agent dqn
    python3 hyper.py --agent both   # PPO first, then DQN

Outputs (written to tuning/):
    <agent>_best.yaml    — best params in config.yaml-compatible format
    <agent>_results.csv  — trial-by-trial summary
    <agent>_study.pkl    — full Optuna study object (re-loadable)
"""

import argparse
import copy
import csv
import os
import pickle
import sys
import warnings

import numpy as np
import yaml

# ── Webots controller guard ────────────────────────────────────────────────
try:
    from controller import Supervisor  # noqa: F401

except ModuleNotFoundError:
    sys.exit(
        "[ERROR] 'controller' module not found.\n"
        "  hyper.py must be run as a Webots robot controller.\n"
        "  Set this file as the controller in your .wbt world, or ensure\n"
        "  WEBOTS_HOME/lib/controller/python3 is on PYTHONPATH."
    )

import optuna
from optuna.samplers import TPESampler
from stable_baselines3 import DQN, PPO

from env.gym_wrapper import WebotsLaneEnv

warnings.filterwarnings("ignore")   # silence SB3 / gym deprecation noise

# ── Defaults (overridable via CLI) ─────────────────────────────────────────
N_TRIALS      = 25
TIMESTEPS     = 10_000
REWARD_WINDOW = 10      # last N completed episodes used as the objective score
OUT_DIR       = "results/tuning"
CONFIG_PATH   = "config.yaml"

def sample_ppo_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps":       trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma":         trial.suggest_float("gamma", 0.90, 0.999),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.90, 0.99),
        "clip_range":    trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef":      trial.suggest_float("ent_coef", 1e-8, 0.05, log=True),
        "n_epochs":      trial.suggest_int("n_epochs", 3, 10),
    }


def sample_dqn_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate":           trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size":              trial.suggest_categorical("batch_size", [32, 64, 128]),
        "gamma":                   trial.suggest_float("gamma", 0.90, 0.999),
        "buffer_size":             trial.suggest_categorical("buffer_size", [5000, 10000, 50000]),
        "learning_starts":         trial.suggest_categorical("learning_starts", [500, 1000, 2000]),
        "target_update_interval":  trial.suggest_categorical("target_update_interval", [500, 1000, 2000]),
        "exploration_fraction":    trial.suggest_float("exploration_fraction", 0.05, 0.3),
        "exploration_final_eps":   trial.suggest_float("exploration_final_eps", 0.01, 0.1),
        "tau":                     trial.suggest_float("tau", 0.005, 1.0),
    }


def make_objective(agent: str, env: WebotsLaneEnv):
    """
    Return a closure that Optuna calls as objective(trial).

    IMPORTANT: `env` is shared across all trials.  We never close or
    recreate it — only reset() between runs and wipe completed_episodes.
    """
    sampler  = sample_ppo_params if agent == "ppo" else sample_dqn_params
    AgentCls = PPO               if agent == "ppo" else DQN

    def objective(trial: optuna.Trial) -> float:
        env._completed_episodes.clear()
        env.reset()

        params = sampler(trial)

        model = AgentCls(
            policy="MultiInputPolicy",
            env=env,
            verbose=0,
            **params,
        )

        try:
            model.learn(total_timesteps=TIMESTEPS)

        except Exception as exc:
            raise optuna.exceptions.TrialPruned(f"Training crashed: {exc}")

        episodes = env.completed_episodes
        if not episodes:
            return float("-inf")

        window = episodes[-REWARD_WINDOW:]
        return float(np.mean([e.total_reward for e in window]))

    return objective


def tune(agent: str, base_cfg: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Tuning {agent.upper()} — {N_TRIALS} trials × {TIMESTEPS:,} timesteps")
    print(f"{'='*60}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = copy.deepcopy(base_cfg)
    cfg["action_space"]["type"] = "continuous" if agent == "ppo" else "discrete"

    print(f"[tune] Creating WebotsLaneEnv for {agent.upper()} (action_space="
          f"{cfg['action_space']['type']}) …")
    env = WebotsLaneEnv(cfg)
    print("[tune] Environment ready.\n")

    csv_path = os.path.join(OUT_DIR, f"{agent}_results.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["trial", "reward", "state", "params"])

    def logging_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        state  = trial.state.name
        value  = f"{trial.value:.3f}" if trial.value is not None else "—"
        symbol = "✓" if trial.state == optuna.trial.TrialState.COMPLETE else "✗"

        print(f"  [{symbol}] Trial {trial.number:02d}/{N_TRIALS - 1}  "
              f"reward={value}  {trial.params}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [trial.number, trial.value, state, trial.params]
            )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=f"{agent}_tuning",
    )
    study.optimize(
        make_objective(agent, env),
        n_trials=N_TRIALS,
        callbacks=[logging_callback],
        show_progress_bar=False,
    )

    study_path = os.path.join(OUT_DIR, f"{agent}_study.pkl")
    with open(study_path, "wb") as f:
        pickle.dump(study, f)

    best = study.best_trial
    yaml_path = os.path.join(OUT_DIR, f"{agent}_best.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(
            {"agent": agent, "best_mean_reward": best.value,
             "hyperparameters": best.params},
            f, default_flow_style=False,
        )

    print(f"\n[tune] ── {agent.upper()} results ──────────────────────────")
    print(f"[tune] Best reward : {best.value:.3f}")
    print(f"[tune] Best params :")
    for k, v in best.params.items():
        print(f"         {k}: {v}")
    print(f"[tune] Saved → {yaml_path}")
    print(f"[tune] Saved → {csv_path}")
    print(f"[tune] Saved → {study_path}\n")

    # NOTE: we do NOT call env.close() here — the Driver must stay alive
    # for the remainder of the controller process.  If tuning both agents,
    # the DQN phase will create its own env after this returns.


def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter search for PPO / DQN in the Webots lane-keeping project."
    )
    p.add_argument("--agent",     choices=["ppo", "dqn", "both"], default="both")
    p.add_argument("--config",    default=CONFIG_PATH)
    p.add_argument("--trials",    type=int, default=N_TRIALS)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    return p.parse_args()


def main():
    args = parse_args()

    global N_TRIALS, TIMESTEPS
    N_TRIALS  = args.trials
    TIMESTEPS = args.timesteps

    if not os.path.exists(args.config):
        sys.exit(f"[ERROR] Config file not found: '{args.config}'")

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    agents = ["ppo", "dqn"] if args.agent == "both" else [args.agent]
    for agent in agents:
        tune(agent, base_cfg)

    print("[tune] All done.")


if __name__ == "__main__":
    main()
