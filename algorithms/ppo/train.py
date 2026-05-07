import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import argparse
import numpy as np
import gymnasium as gym
from algorithms.ppo.agent import PPOAgent
from algorithms.ppo.config import config


def train_cartpole():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        clip_eps=config["clip_eps"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        gae_lambda=config["gae_lambda"],
    )

    rewards_history = []

    for episode in range(config["n_episodes"]):
        state, _ = env.reset()
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_reward = 0

        done = False
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            state = next_state
            episode_reward += reward

        # Bootstrap value for last state (0 if done)
        if done:
            next_value = 0.0
        else:
            _, _, next_value = agent.select_action(state)

        agent.update(states, actions, rewards, log_probs, values, dones, next_value=next_value)
        rewards_history.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}/{config['n_episodes']}, Avg Reward (last 50): {avg_reward:.2f}")

    env.close()
    agent.save("ppo_cartpole.pth")
    print("Training complete. Model saved to ppo_cartpole.pth")
    return rewards_history


def train_highway():
    import highway_env  # noqa: F401

    env = gym.make("highway-v0")
    env.configure({"observation": {"type": "Kinematics", "flatten": True}})
    env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        clip_eps=config["clip_eps"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        gae_lambda=config["gae_lambda"],
    )

    rewards_history = []

    for episode in range(config["n_episodes"]):
        state, _ = env.reset()
        state = state.flatten()
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_reward = 0

        done = False
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_state.flatten()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            state = next_state
            episode_reward += reward

        if done:
            next_value = 0.0
        else:
            _, _, next_value = agent.select_action(state)

        agent.update(states, actions, rewards, log_probs, values, dones, next_value=next_value)
        rewards_history.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}/{config['n_episodes']}, Avg Reward (last 50): {avg_reward:.2f}")

    env.close()
    agent.save("ppo_highway.pth")
    print("Training complete. Model saved to ppo_highway.pth")
    return rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training")
    parser.add_argument("--env", type=str, choices=["cartpole", "highway", "both"], default="both")
    args = parser.parse_args()

    if args.env == "cartpole":
        train_cartpole()
    elif args.env == "highway":
        train_highway()
    else:
        train_cartpole()
        train_highway()
