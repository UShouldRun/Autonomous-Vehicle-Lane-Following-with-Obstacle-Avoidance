# Autonomous Vehicle Lane Following with Obstacle Avoidance

**Group A** — João Ferreira (up202306717) · Henrique Teixeira (up202306640) · Miguel Almeida (up202303926)

## Overview

Train an autonomous vehicle using Reinforcement Learning to follow the yellow centre line on a road and avoid static (barrels) and dynamic (pedestrians, vehicles) obstacles. The agent processes Camera + LiDAR data and outputs steering/throttle commands.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure Webots (2023b+) is installed and WEBOTS_HOME is set
export WEBOTS_HOME=/usr/local/webots
```

## Usage

```bash
# Train a baseline agent
python exps/train.py --agent ppo --reward dense

# Evaluate a saved model
python exps/eval.py --checkpoint results/ppo_dense_best.zip
```

## Experiments

| # | Name | Goal |
|---|------|------|
| 1 | Action Space | Discrete (DQN) vs Continuous (PPO) |
| 2 | Reward Function | Dense vs Sparse (2×2 matrix) |
| 3 | Camera Robustness | FOV distance + distortion filters (fog, rain, low-light) |

Each experiment is run **100 times** and results are averaged.

## Evaluation Metrics

- **Success Rate (%)** — trials completed without collision
- **Collisions** — total collisions per run
- **Cross-Track Error (m)** — average lateral distance from yellow line
- **Mean Lap Time (s)** — time per circuit
- **Safety Score** — distance travelled / near-misses
