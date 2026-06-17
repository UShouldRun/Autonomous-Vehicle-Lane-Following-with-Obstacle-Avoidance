"""
experiments/train_with_distortion.py
======================================
Thin wrapper around ``train.py`` that monkey-patches
``WebotsEnv.get_camera_image`` with the chosen FOV mode and distortion
filter **before** the Gym environment is constructed.

This is the entry-point used for Experiment 3 runs.

Usage (same constraints as train.py — must run as a Webots controller)
-----------------------------------------------------------------------
    python experiments/train_with_distortion.py \\
        --agent ppo --reward dense --config configs/exp1_ppo_dense.yaml \\
        --distortion fog --fov medium --intensity 0.5 --timesteps 200000

Arguments
---------
All train.py arguments are forwarded transparently plus:

  --distortion {clean,fog,rain,low_light}
      Visual filter to apply to every camera frame. Default: clean
  --fov {short,medium,long}
      Field-of-view proxy. Default: medium
  --intensity FLOAT
      Filter strength in [0, 1]. Default: 0.5

How it works
------------
1. Parse the extra --distortion / --fov / --intensity args and strip them
   from sys.argv before importing train.py's ``main()``.
2. Apply ``camera_distortion.patch_env()`` (monkey-patch).
3. Call ``train.main()`` — which builds the env, the model, and runs
   ``model.learn()``.  The patch is already in place so every camera read
   goes through the distortion pipeline.

The result files follow the same naming as a direct train.py invocation
(i.e. ``results/<agent>_<reward>_model.zip``).  It is the caller's
responsibility (``run_experiments.py``) to pass a unique ``--config``
or result directory per distortion variant if you don't want files to
overwrite each other.
"""
from __future__ import annotations

import argparse
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)


def parse_distortion_args():
    """
    Extract the three distortion-specific flags from sys.argv, returning
    them as a namespace and leaving the remaining args in sys.argv for
    train.py's own argparse call.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--distortion", default="clean",
                   choices=["clean", "fog", "rain", "low_light"])
    p.add_argument("--fov", default="medium",
                   choices=["short", "medium", "long"])
    p.add_argument("--intensity", type=float, default=0.5)

    known, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining   # strip our flags from sys.argv
    return known


def main():
    dist_args = parse_distortion_args()

    # Apply the camera patch before train.main() imports the env.
    from experiments.camera_distortion import patch_env
    patch_env(
        fov_mode    = dist_args.fov,
        filter_name = dist_args.distortion,
        intensity   = dist_args.intensity,
    )

    # Delegate everything else to the standard train.main()
    import train
    train.main()


if __name__ == "__main__":
    main()
