import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from algorithms.dqn.agent import DQNAgent
from algorithms.dqn.config import config
from envs.highway_lane_keeping import make_lane_keeping_env


def evaluate():
    """Load trained DQN model and run greedy policy with visualization."""
    results_dir = str(Path(__file__).resolve().parent / "results")
    model_path = f"{results_dir}/dqn_highway.pth"

    env = make_lane_keeping_env(render_mode="human")
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon=0.0,
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
    )
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    n_episodes = 10
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = obs.flatten()
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {ep + 1}/{n_episodes} | Reward: {total_reward:.2f}")

    env.close()
    print(f"\nMean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


if __name__ == "__main__":
    evaluate()
