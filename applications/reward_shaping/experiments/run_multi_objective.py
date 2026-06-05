"""Experiment 3: Multi-objective weighted reward + weight sensitivity analysis.

Sweeps individual weights while keeping others at defaults to show
how reward weighting affects learned behavior.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import gymnasium as gym
import numpy as np

from rewards.multi_objective import MultiObjectiveRewardWrapper, HighwayMultiObjectiveWrapper
from config import config
from algorithms.ppo.agent import PPOAgent
from experiments.utils import train_ppo


def train_with_weights(env_id: str, weights: dict, n_episodes: int, env_type: str = "mujoco"):
    """Train PPO with specific multi-objective weights."""
    if env_type == "highway":
        env = gym.make(env_id, config={"action": {"type": "ContinuousAction"}})
    else:
        env = gym.make(env_id)

    if env_type == "mujoco":
        env = MultiObjectiveRewardWrapper(env, **weights)
    else:
        env = HighwayMultiObjectiveWrapper(env, **weights)

    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = config["mujoco_hidden_dim"] if env_type == "mujoco" else config["highway_hidden_dim"]

    agent = PPOAgent(
        state_dim=state_dim, action_dim=action_dim,
        lr=config["lr"], gamma=config["gamma"],
        clip_eps=config["clip_eps"], epochs=config["epochs"],
        batch_size=config["batch_size"], hidden_dim=hidden_dim,
        gae_lambda=config["gae_lambda"],
        max_grad_norm=config["max_grad_norm"],
    )

    returns = train_ppo(env, agent, n_episodes, verbose_interval=0)
    env.close()
    return returns


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    default_weights = {"w_speed": 1.0, "w_alive": 0.5, "w_energy": 0.01, "w_posture": 0.1}

    # --- MuJoCo: Sweep w_speed ---
    print("=" * 60)
    print("Experiment 3: Weight sensitivity sweep (w_speed)")
    print("=" * 60)
    speed_sweep = {}
    for w_speed in config["weight_sweep_values"]:
        print(f"\n  Training with w_speed={w_speed}...")
        weights = {**default_weights, "w_speed": w_speed}
        returns = train_with_weights("Ant-v4", weights, config["mujoco_episodes"])
        speed_sweep[f"w_speed_{w_speed}"] = returns
        avg = np.mean(returns[-50:])
        print(f"    Final avg return: {avg:.2f}")

    np.save(str(results_dir / "multi_objective_speed_sweep.npy"), speed_sweep)

    # --- MuJoCo: Sweep w_posture ---
    print("\n" + "=" * 60)
    print("Experiment 3: Weight sensitivity sweep (w_posture)")
    print("=" * 60)
    posture_sweep = {}
    for w_posture in config["weight_sweep_values"]:
        print(f"\n  Training with w_posture={w_posture}...")
        weights = {**default_weights, "w_posture": w_posture}
        returns = train_with_weights("Ant-v4", weights, config["mujoco_episodes"])
        posture_sweep[f"w_posture_{w_posture}"] = returns
        avg = np.mean(returns[-50:])
        print(f"    Final avg return: {avg:.2f}")

    np.save(str(results_dir / "multi_objective_posture_sweep.npy"), posture_sweep)

    # --- Highway: Sweep collision penalty ---
    print("\n" + "=" * 60)
    print("Experiment 3: Highway collision weight sweep")
    print("=" * 60)
    highway_default_weights = {"w_speed": 1.0, "w_collision": -10.0,
                               "w_comfort": 0.1, "w_lane": 0.5}
    collision_sweep = {}
    for w_collision in config["collision_sweep_values"]:
        print(f"\n  Training highway with w_collision={w_collision}...")
        weights = {**highway_default_weights, "w_collision": w_collision}
        returns = train_with_weights(
            "highway-v0", weights, config["highway_episodes"], env_type="highway"
        )
        collision_sweep[f"w_collision_{w_collision}"] = returns
        avg = np.mean(returns[-50:])
        print(f"    Final avg return: {avg:.2f}")

    np.save(str(results_dir / "multi_objective_collision_sweep.npy"), collision_sweep)

    print("\nExperiment 3 complete. Results saved to results/")


if __name__ == "__main__":
    main()
