import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import argparse

import gymnasium as gym
import numpy as np

from algorithms.policy_gradient.agent import REINFORCEAgent
from algorithms.policy_gradient.config import config
from utils.logger import Logger
from utils.plotting import plot_training_curves


def train_cartpole():
    """Train REINFORCE agent on CartPole-v1."""
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        hidden_dim=config["hidden_dim"],
    )

    results_dir = "/home/lihongl/Desktop/myRL/easyRL/algorithms/policy_gradient/results/cartpole"
    logger = Logger(log_dir=results_dir)

    n_episodes = config["n_episodes"]

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
            total_reward += reward

        # Update policy at end of episode
        loss = agent.update()

        # Log reward
        logger.log("episode_reward", total_reward, episode)

        # Print progress
        if (episode + 1) % 50 == 0:
            recent_rewards = [v for _, v in logger.get_data("episode_reward")[-50:]]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward (last 50): {avg_reward:.2f} | "
                  f"Loss: {loss:.4f}")

    # Save results
    logger.save()
    logger.close()

    # Plot training curve
    plot_training_curves(
        log_dir=results_dir,
        tags=["episode_reward"],
        save_path=f"{results_dir}/training_curve.png",
    )

    # Save model
    agent.save(f"{results_dir}/reinforce_cartpole.pth")
    print(f"\nCartPole training complete. Results saved to {results_dir}/")
    env.close()


def train_highway():
    """Train REINFORCE agent on highway-env lane keeping."""
    from envs.highway_lane_keeping import make_lane_keeping_env

    env = make_lane_keeping_env()
    # Observation is a 2D array (vehicles_count x features), flatten it
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        hidden_dim=config["hidden_dim"],
    )

    results_dir = "/home/lihongl/Desktop/myRL/easyRL/algorithms/policy_gradient/results/highway"
    logger = Logger(log_dir=results_dir)

    n_episodes = config["n_episodes"]

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            obs, reward, done, truncated, _ = env.step(action)
            agent.store_reward(reward)
            state = obs.flatten()
            total_reward += reward

        # Update policy at end of episode
        loss = agent.update()

        # Log reward
        logger.log("episode_reward", total_reward, episode)

        # Print progress
        if (episode + 1) % 50 == 0:
            recent_rewards = [v for _, v in logger.get_data("episode_reward")[-50:]]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward (last 50): {avg_reward:.2f} | "
                  f"Loss: {loss:.4f}")

    # Save results
    logger.save()
    logger.close()

    # Plot training curve
    plot_training_curves(
        log_dir=results_dir,
        tags=["episode_reward"],
        save_path=f"{results_dir}/training_curve.png",
    )

    # Save model
    agent.save(f"{results_dir}/reinforce_highway.pth")
    print(f"\nHighway training complete. Results saved to {results_dir}/")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REINFORCE agent")
    parser.add_argument(
        "--env",
        type=str,
        choices=["cartpole", "highway", "both"],
        default="both",
        help="Environment to train on",
    )
    args = parser.parse_args()

    if args.env == "cartpole":
        train_cartpole()
    elif args.env == "highway":
        train_highway()
    else:
        train_cartpole()
        train_highway()
