import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os
import numpy as np

from algorithms.ppo.agent import PPOAgent
from algorithms.ppo.config import config
from envs.highway_lane_keeping import make_continuous_lane_keeping_env
from utils.hud import patch_viewer_for_hud, update_hud


def train():
    """Train PPO agent on highway-env continuous lane keeping."""
    # render_mode="human" enables real-time visualization; set to None to disable for faster training
    env = make_continuous_lane_keeping_env(render_mode="human")
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]

    patch_viewer_for_hud(env)

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

    results_dir = str(Path(__file__).resolve().parent / "results")
    os.makedirs(results_dir, exist_ok=True)

    rewards_history = []
    n_episodes = config["n_episodes"]

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_reward = 0

        update_hud(episode + 1, n_episodes, 0.0, 0.0)

        done = False
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_obs.flatten()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            state = next_state
            episode_reward += reward
            update_hud(episode + 1, n_episodes, 0.0, episode_reward)

        # Bootstrap value for last state
        if done:
            next_value = 0.0
        else:
            _, _, next_value = agent.select_action(state)

        agent.update(states, actions, rewards, log_probs, values, dones, next_value=next_value)
        rewards_history.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward (last 50): {avg_reward:.2f}")

    env.close()
    agent.save(f"{results_dir}/ppo_highway.pth")
    print(f"\nTraining complete. Model saved to {results_dir}/")


if __name__ == "__main__":
    train()
