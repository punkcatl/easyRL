"""Experiment 1: Sparse vs Dense reward comparison.

Trains PPO with identical hyperparameters on the same environment,
varying only the reward function (sparse or dense).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import gymnasium as gym
import numpy as np

from rewards.sparse import SparseRewardWrapper
from rewards.dense import DenseRewardWrapper
from config import config
from algorithms.ppo.agent import PPOAgent
from experiments.utils import train_ppo


def run_experiment(env_id: str, env_type: str, n_episodes: int):
    """Run sparse vs dense comparison on one environment."""
    results = {}

    for reward_type, WrapperClass in [("sparse", SparseRewardWrapper),
                                       ("dense", DenseRewardWrapper)]:
        print(f"\n  Training PPO with {reward_type} reward on {env_id}...")
        if env_type == "highway":
            env = gym.make(env_id, config={"action": {"type": "ContinuousAction"}})
        else:
            env = gym.make(env_id)
        env = WrapperClass(env, env_type)

        obs, _ = env.reset()
        state_dim = obs.flatten().shape[0]
        action_dim = env.action_space.shape[0]
        hidden_dim = (config["mujoco_hidden_dim"] if env_type == "mujoco"
                      else config["highway_hidden_dim"])

        agent = PPOAgent(
            state_dim=state_dim, action_dim=action_dim,
            lr=config["lr"], gamma=config["gamma"],
            clip_eps=config["clip_eps"], epochs=config["epochs"],
            batch_size=config["batch_size"], hidden_dim=hidden_dim,
            gae_lambda=config["gae_lambda"],
            max_grad_norm=config["max_grad_norm"],
        )

        returns = train_ppo(env, agent, n_episodes, label=f"{reward_type} ")
        results[reward_type] = returns
        env.close()

    return results


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Experiment 1: Sparse vs Dense (Ant-v4)")
    print("=" * 60)
    mujoco_results = run_experiment("Ant-v4", "mujoco", config["mujoco_episodes"])
    np.save(str(results_dir / "sparse_vs_dense_ant.npy"), mujoco_results)

    print("\n" + "=" * 60)
    print("Experiment 1: Sparse vs Dense (highway-v0)")
    print("=" * 60)
    highway_results = run_experiment("highway-v0", "highway", config["highway_episodes"])
    np.save(str(results_dir / "sparse_vs_dense_highway.npy"), highway_results)

    print("\nExperiment 1 complete. Results saved to results/")


if __name__ == "__main__":
    main()
