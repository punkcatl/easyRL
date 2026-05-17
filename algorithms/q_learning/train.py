import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import gymnasium as gym
import numpy as np

from algorithms.q_learning.agent import QLearningAgent
from algorithms.q_learning.config import config
from utils.logger import Logger
from utils.plotting import plot_training_curves


def train():
    """Train Q-Learning agent on CliffWalking-v0."""
    env = gym.make("CliffWalking-v0")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon=config["epsilon_start"],
    )

    results_dir = str(Path(__file__).resolve().parent / "results")
    logger = Logger(log_dir=results_dir)

    n_episodes = config["n_episodes"]
    epsilon = config["epsilon_start"]

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(config["epsilon_end"], epsilon * config["epsilon_decay"])
        agent.epsilon = epsilon

        # Log reward
        logger.log("episode_reward", total_reward, episode)

        # Print progress
        if (episode + 1) % 50 == 0:
            recent_rewards = [v for _, v in logger.get_data("episode_reward")[-50:]]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward (last 50): {avg_reward:.2f} | "
                  f"Epsilon: {epsilon:.4f}")

    # Save results
    logger.save()
    logger.close()

    plot_training_curves(
        log_dir=results_dir,
        tags=["episode_reward"],
        save_path=f"{results_dir}/training_curve.png",
    )

    print(f"\nTraining complete. Results saved to {results_dir}/")
    env.close()


if __name__ == "__main__":
    train()
