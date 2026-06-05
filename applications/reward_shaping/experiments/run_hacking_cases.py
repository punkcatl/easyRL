"""Experiment 4: Reward hacking cases.

For each case:
1. Train with broken reward -> observe undesired behavior
2. Train with fixed reward -> show corrected behavior
3. Compare training curves
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import gymnasium as gym
import numpy as np

from config import config
from hacking.ant_rolling import AntRollingBrokenReward, AntRollingFixedReward
from hacking.hopper_jumping import HopperJumpingBrokenReward, HopperJumpingFixedReward
from hacking.humanoid_sliding import HumanoidSlidingBrokenReward, HumanoidSlidingFixedReward
from hacking.highway_lane_spam import HighwayLaneSpamBrokenReward, HighwayLaneSpamFixedReward
from hacking.highway_parking import HighwayParkingBrokenReward, HighwayParkingFixedReward
from algorithms.ppo.agent import PPOAgent
from experiments.utils import train_ppo


HACKING_CASES = [
    ("Ant Rolling", "Ant-v4", AntRollingBrokenReward, AntRollingFixedReward),
    ("Hopper Jumping", "Hopper-v4", HopperJumpingBrokenReward, HopperJumpingFixedReward),
    ("Humanoid Sliding", "Humanoid-v4", HumanoidSlidingBrokenReward, HumanoidSlidingFixedReward),
    ("Highway Lane Spam", "highway-v0", HighwayLaneSpamBrokenReward, HighwayLaneSpamFixedReward),
    ("Highway Parking", "highway-v0", HighwayParkingBrokenReward, HighwayParkingFixedReward),
]


def train_hacking_case(case_name, env_id, BrokenWrapper, FixedWrapper, n_episodes):
    """Train broken and fixed versions, return comparison."""
    results = {}

    for label, Wrapper in [("broken", BrokenWrapper), ("fixed", FixedWrapper)]:
        print(f"  Training {case_name} ({label})...")
        if "highway" in env_id:
            env = gym.make(env_id, config={"action": {"type": "ContinuousAction"}})
        else:
            env = gym.make(env_id)
        env = Wrapper(env)

        obs, _ = env.reset()
        state_dim = obs.flatten().shape[0]
        action_dim = env.action_space.shape[0]
        hidden_dim = config["mujoco_hidden_dim"] if "v4" in env_id else config["highway_hidden_dim"]

        agent = PPOAgent(
            state_dim=state_dim, action_dim=action_dim,
            lr=config["lr"], gamma=config["gamma"],
            clip_eps=config["clip_eps"], epochs=config["epochs"],
            batch_size=config["batch_size"], hidden_dim=hidden_dim,
            gae_lambda=config["gae_lambda"],
            max_grad_norm=config["max_grad_norm"],
        )

        returns = train_ppo(env, agent, n_episodes, label=f"{case_name} ({label}) ")
        results[label] = returns
        env.close()

    return results


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = {}
    for case_name, env_id, BrokenW, FixedW in HACKING_CASES:
        print(f"\n{'=' * 60}")
        print(f"Case: {case_name}")
        print(f"{'=' * 60}")
        results = train_hacking_case(
            case_name, env_id, BrokenW, FixedW, config["hacking_episodes"]
        )
        all_results[case_name] = results

    np.save(str(results_dir / "hacking_cases.npy"), all_results)
    print("\nAll hacking cases complete. Results saved to results/")


if __name__ == "__main__":
    main()
