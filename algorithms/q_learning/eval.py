import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import gymnasium as gym

from algorithms.q_learning.agent import QLearningAgent
from algorithms.q_learning.config import config


def evaluate(q_table_path: str = None, n_episodes: int = 10):
    """Load trained Q-table and evaluate with greedy policy (epsilon=0)."""
    env = gym.make("CliffWalking-v0", render_mode="human")

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon=0.0,
    )

    if q_table_path is None:
        q_table_path = str(Path(__file__).resolve().parent / "results" / "q_table.npy")
    agent.q_table = np.load(q_table_path)
    print(f"Loaded Q-table from {q_table_path}")

    rewards = []
    lengths = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.take_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"Episode {episode + 1}/{n_episodes} | Reward: {total_reward:.2f} | Length: {steps}")

    print(f"\n=== Evaluation Results ===")
    print(f"Episodes: {n_episodes}")
    print(f"Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Avg Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")

    env.close()


if __name__ == "__main__":
    evaluate()
