import sys

sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import argparse

import gymnasium as gym
import numpy as np

from algorithms.sac.agent import SACAgent
from algorithms.sac.config import config


def train_pendulum(n_episodes: int = None):
    """Train SAC on Pendulum-v1 environment."""
    if n_episodes is None:
        n_episodes = config["n_episodes"]

    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
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

    total_steps = 0
    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Use random actions for initial exploration
            if total_steps < config["start_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                # Scale action to environment action space
                action = action * float(env.action_space.high[0])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)

            # Learn after start_steps
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
    agent.save("/home/lihongl/Desktop/myRL/easyRL/algorithms/sac/pendulum_sac.pth")
    print("Training complete. Model saved.")
    return rewards_history


def train_highway(n_episodes: int = None):
    """Train SAC on highway-env continuous lane keeping environment."""
    if n_episodes is None:
        n_episodes = config["n_episodes"]

    from envs.highway_lane_keeping import make_continuous_lane_keeping_env

    env = make_continuous_lane_keeping_env()
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
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

    total_steps = 0
    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        state = state.flatten()
        episode_reward = 0
        done = False

        while not done:
            if total_steps < config["start_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()
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
    agent.save("/home/lihongl/Desktop/myRL/easyRL/algorithms/sac/highway_sac.pth")
    print("Training complete. Model saved.")
    return rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent")
    parser.add_argument(
        "--env",
        type=str,
        choices=["pendulum", "highway", "both"],
        default="both",
        help="Environment to train on",
    )
    args = parser.parse_args()

    if args.env in ("pendulum", "both"):
        print("=" * 50)
        print("Training SAC on Pendulum-v1")
        print("=" * 50)
        train_pendulum()

    if args.env in ("highway", "both"):
        print("=" * 50)
        print("Training SAC on Highway Continuous Lane Keeping")
        print("=" * 50)
        train_highway()
