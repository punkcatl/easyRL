import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os
import numpy as np

from algorithms.sac.agent import SACAgent
from algorithms.sac.config import config
from envs.highway_lane_keeping import make_continuous_lane_keeping_env


def train():
    """Train SAC on highway-env continuous lane keeping."""
    env = make_continuous_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config["alpha"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        auto_alpha=config["auto_alpha"],
    )

    results_dir = str(Path(__file__).resolve().parent / "results")
    os.makedirs(results_dir, exist_ok=True)

    total_steps = 0
    rewards_history = []
    n_episodes = config["n_episodes"]

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        episode_reward = 0
        done = False

        while not done:
            if total_steps < config["start_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs.flatten()
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)

            if total_steps >= config["start_steps"]:
                agent.learn()

            state = next_state
            episode_reward += reward
            total_steps += 1

        rewards_history.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg(10): {avg_reward:.1f} | "
                  f"Steps: {total_steps}")

    env.close()
    agent.save(f"{results_dir}/sac_highway.pth")
    print(f"\nTraining complete. Model saved to {results_dir}/")


if __name__ == "__main__":
    train()
