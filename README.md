# Autonomous Vehicle Lane Following with Obstacle Avoidance

**Group A** — João Ferreira (up202306717) · Henrique Teixeira (up202306640) · Miguel Almeida (up202303926)

## Overview

Train an autonomous vehicle using Reinforcement Learning to follow the yellow centre line on a road and avoid static (barrels) and dynamic (pedestrians, vehicles) obstacles. The agent processes Camera + LiDAR data and outputs steering/throttle commands.

## Project Structure

```
av-lane-following/
├── env/
│   ├── __init__.py
│   ├── webots_env.py        # Webots simulation interface
│   ├── gym_wrapper.py       # Gymnasium-compatible wrapper
│   └── reward.py            # Reward function definitions
├── agents/
│   ├── __init__.py
│   ├── dqn.py               # DQN agent (discrete action space)
│   └── ppo.py               # PPO agent (continuous action space)
├── experiments/
│   ├── __init__.py
│   ├── exp1_action_space.py # Experiment 1: discrete vs continuous
│   ├── exp2_reward.py       # Experiment 2: dense vs sparse reward
│   └── exp3_camera.py       # Experiment 3: FOV & distortion filters
├── utils/
│   ├── __init__.py
│   ├── observation.py       # Camera + LiDAR preprocessing
│   └── metrics.py           # Evaluation metrics (collision rate, CTE, etc.)
├── tests/
│   └── test_env.py
├── results/
│   └── plots/
├── train.py                 # Main training entry point
├── evaluate.py              # Evaluation script
├── config.yaml              # Hyperparameters and experiment config
├── requirements.txt
└── README.md
```

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
python train.py --agent ppo --reward dense

# Run an experiment
python experiments/exp1_action_space.py

# Evaluate a saved model
python evaluate.py --checkpoint results/ppo_dense_best.zip
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

## Key Dates

| Date | Milestone |
|------|-----------|
| May 14 | Intermediate checkpoint |
| June 19 | Final submission |
