"""
experiments/run_experiments.py
================================
Orchestrator that launches every training run required by the three
experiments described in the project proposal (§4.2 – §4.4).

Each run is a thin wrapper around ``train.py``, so all Webots / SB3 /
controller constraints remain exactly as they are.  This script merely
builds the right argument lists and invokes train.py in a subprocess.

Usage
-----
    # Full suite (all 7 runs × default timesteps from each config)
    python experiments/run_experiments.py

    # Quick smoke-test — 500 timesteps per run
    python experiments/run_experiments.py --timesteps 500

    # Only a specific experiment
    python experiments/run_experiments.py --exp 1
    python experiments/run_experiments.py --exp 2
    python experiments/run_experiments.py --exp 3

    # Dry run — print commands without executing
    python experiments/run_experiments.py --dry-run

Important
---------
This script must be executed from the project ROOT directory (the folder
that contains train.py), e.g.:

    cd /path/to/Autonomous-Vehicle-Lane-Following-with-Obstacle-Avoidance
    python experiments/run_experiments.py

Webots must be running with the appropriate .wbt world loaded, and
train.py must be set as the robot controller — the same prerequisites
that apply when running train.py directly.

Results
-------
All CSVs and .zip model files land in ``results/`` as usual.  The run
label (used in filenames) follows the pattern:
    <agent>_<reward>_<tag>
where <tag> is a short string identifying the experimental variant
(e.g. "exp1", "exp2_sparse", "exp3_fog").
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Run specification ──────────────────────────────────────────────────────

@dataclass
class Run:
    """One training run."""
    label:      str           # used in result filenames
    agent:      str           # "ppo" | "dqn"
    reward:     str           # "dense" | "ttc" | "sparse"
    config:     str           # path relative to project root
    obstacles:  bool  = False # pass --obstacles flag?
    extra_args: List[str] = field(default_factory=list)
    experiment: int   = 0     # which experiment this run belongs to (1/2/3)


# ── Experiment definitions ─────────────────────────────────────────────────

RUNS: List[Run] = [
    # ── Experiment 1: Action Space Comparison (§4.2) ──────────────
    # Discrete action space (DQN) vs continuous (PPO), both dense reward,
    # no obstacles.
    Run(
        label="exp1_dqn_dense",
        agent="dqn",
        reward="dense",
        config="configs/exp1_dqn_dense.yaml",
        obstacles=False,
        experiment=1,
    ),
    Run(
        label="exp1_ppo_dense",
        agent="ppo",
        reward="dense",
        config="configs/exp1_ppo_dense.yaml",
        obstacles=False,
        experiment=1,
    ),

    # ── Experiment 2: Reward Function Impact (§4.3) ───────────────
    # 2×2 matrix: {dense, sparse} × {DQN, PPO}
    # Dense × DQN is shared with Exp 1 (same config) — re-run or reuse.
    # Dense × PPO is shared with Exp 1 — re-run or reuse.
    Run(
        label="exp2_dqn_sparse",
        agent="dqn",
        reward="sparse",
        config="configs/exp2_dqn_sparse.yaml",
        obstacles=False,
        experiment=2,
    ),
    Run(
        label="exp2_ppo_sparse",
        agent="ppo",
        reward="sparse",
        config="configs/exp2_ppo_sparse.yaml",
        obstacles=False,
        experiment=2,
    ),
    # Bonus: TTC reward with obstacles (best demonstrates the safety term)
    Run(
        label="exp2_ppo_ttc",
        agent="ppo",
        reward="ttc",
        config="configs/exp2_ppo_ttc.yaml",
        obstacles=True,
        experiment=2,
    ),

    # ── Experiment 3: Camera Distortion / FOV (§4.4) ─────────────
    # The distortion is applied via monkey-patching in camera_distortion.py.
    # These runs use the standard PPO+dense config; a helper script
    # (experiments/train_with_distortion.py) wraps train.py and applies the
    # patch before building the env.
    Run(
        label="exp3_ppo_fog",
        agent="ppo",
        reward="dense",
        config="configs/exp1_ppo_dense.yaml",
        obstacles=False,
        extra_args=["--distortion", "fog", "--fov", "medium"],
        experiment=3,
    ),
    Run(
        label="exp3_ppo_rain",
        agent="ppo",
        reward="dense",
        config="configs/exp1_ppo_dense.yaml",
        obstacles=False,
        extra_args=["--distortion", "rain", "--fov", "medium"],
        experiment=3,
    ),
    Run(
        label="exp3_ppo_lowlight",
        agent="ppo",
        reward="dense",
        config="configs/exp1_ppo_dense.yaml",
        obstacles=False,
        extra_args=["--distortion", "low_light", "--fov", "medium"],
        experiment=3,
    ),
    Run(
        label="exp3_ppo_fov_short",
        agent="ppo",
        reward="dense",
        config="configs/exp1_ppo_dense.yaml",
        obstacles=False,
        extra_args=["--distortion", "clean", "--fov", "short"],
        experiment=3,
    ),
    Run(
        label="exp3_ppo_fov_long",
        agent="ppo",
        reward="dense",
        config="configs/exp1_ppo_dense.yaml",
        obstacles=False,
        extra_args=["--distortion", "clean", "--fov", "long"],
        experiment=3,
    ),
]


# ── Command builder ────────────────────────────────────────────────────────

def build_command(run: Run, timesteps: Optional[int] = None) -> List[str]:
    """
    Build the subprocess command for one run.

    Exp 3 runs are routed through ``experiments/train_with_distortion.py``
    which monkey-patches the camera before the env is built.  All other
    runs call ``train.py`` directly.
    """
    is_exp3 = bool(run.extra_args)  # exp3 runs carry --distortion / --fov

    if is_exp3:
        script = "experiments/train_with_distortion.py"
    else:
        script = "train.py"

    cmd = [sys.executable, script,
           "--agent",  run.agent,
           "--reward", run.reward,
           "--config", run.config]

    if run.obstacles:
        cmd.append("--obstacles")

    if timesteps is not None:
        cmd += ["--timesteps", str(timesteps)]

    cmd += run.extra_args
    return cmd


def run_training(
    run: Run,
    timesteps: Optional[int],
    dry_run: bool,
) -> int:
    """Execute (or print) one training run. Returns the process exit code."""
    cmd = build_command(run, timesteps)
    print(f"\n{'='*60}")
    print(f"[runner] Exp {run.experiment}  label={run.label}")
    print(f"[runner] CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        return 0

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    return result.returncode


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run all (or a subset of) experiment training runs."
    )
    p.add_argument(
        "--exp", type=int, choices=[1, 2, 3], default=None,
        help="Run only experiment 1, 2, or 3. Omit to run all.",
    )
    p.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total_timesteps for every run (useful for quick tests).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    runs = RUNS if args.exp is None else [r for r in RUNS if r.experiment == args.exp]

    if not runs:
        print(f"[runner] No runs found for experiment {args.exp}.")
        return

    print(f"[runner] {len(runs)} run(s) scheduled.")
    if args.dry_run:
        print("[runner] DRY RUN — no processes will be started.\n")

    failures = []
    for run in runs:
        rc = run_training(run, args.timesteps, args.dry_run)
        if rc != 0:
            failures.append((run.label, rc))
            print(f"[runner] WARNING: run '{run.label}' exited with code {rc}")

    print("\n" + "="*60)
    if failures:
        print(f"[runner] {len(failures)} run(s) FAILED:")
        for label, rc in failures:
            print(f"  {label}  (exit {rc})")
        sys.exit(1)
    else:
        print(f"[runner] All {len(runs)} run(s) completed successfully.")


if __name__ == "__main__":
    main()
