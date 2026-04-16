"""
Main training entry point.

Usage:
    python train.py --agent ppo --reward dense
    python train.py --agent dqn --reward sparse
"""
import argparse
import yaml
from stable_baselines3 import PPO, DQN
from env.gym_wrapper import WebotsLaneEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent",    choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--reward",   choices=["dense", "ttc", "sparse"], default="dense")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--timesteps", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["reward"]["type"] = args.reward
    cfg["action_space"]["type"] = "continuous" if args.agent == "ppo" else "discrete"

    env = WebotsLaneEnv(cfg)
    timesteps = args.timesteps or cfg["training"]["total_timesteps"]

    if args.agent == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1,
                    learning_rate=cfg["agent"]["learning_rate"],
                    batch_size=cfg["agent"]["batch_size"],
                    gamma=cfg["agent"]["gamma"])
    else:
        model = DQN("MultiInputPolicy", env, verbose=1,
                    learning_rate=cfg["agent"]["learning_rate"],
                    batch_size=cfg["agent"]["batch_size"],
                    gamma=cfg["agent"]["gamma"])

    model.learn(total_timesteps=timesteps)
    model.save(f"results/{args.agent}_{args.reward}_model")
    print("Training complete.")


if __name__ == "__main__":
    main()
