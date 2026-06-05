"""Experiment 2: Potential-based reward shaping.

Compares sparse-only reward vs sparse + potential-based shaping (Ng 1999).
Shaping accelerates convergence without changing the optimal policy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import gymnasium as gym
import numpy as np

from rewards.sparse import SparseRewardWrapper
from rewards.potential_based import (
    PotentialShapingWrapper, mujoco_x_potential, highway_speed_potential
)
from config import config
from algorithms.ppo.agent import PPOAgent
from experiments.utils import train_ppo


def run_experiment(env_id: str, env_type: str, potential_fn, n_episodes: int):
    """Compare sparse-only vs sparse+shaping."""
    results = {}

    for name, use_shaping in [("sparse_only", False), ("sparse_shaped", True)]:
        print(f"\n  Training PPO: {name} on {env_id}...")
        if env_type == "highway":
            env = gym.make(env_id, config={"action": {"type": "ContinuousAction"}})
        else:
            env = gym.make(env_id)
        env = SparseRewardWrapper(env, env_type)
        if use_shaping:
            env = PotentialShapingWrapper(env, potential_fn, gamma=config["shaping_gamma"])

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

        returns = train_ppo(env, agent, n_episodes, label=f"{name} ")
        results[name] = returns
        env.close()

    return results


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Experiment 2: Potential-based Shaping (Ant-v4)")
    print("=" * 60)
    mujoco_results = run_experiment(
        "Ant-v4", "mujoco", mujoco_x_potential, config["mujoco_episodes"]
    )
    np.save(str(results_dir / "potential_shaping_ant.npy"), mujoco_results)

    print("\n" + "=" * 60)
    print("Experiment 2: Potential-based Shaping (highway-v0)")
    print("=" * 60)
    highway_results = run_experiment(
        "highway-v0", "highway", highway_speed_potential, config["highway_episodes"]
    )
    np.save(str(results_dir / "potential_shaping_highway.npy"), highway_results)

    print("\nExperiment 2 complete. Results saved to results/")


if __name__ == "__main__":
    main()
