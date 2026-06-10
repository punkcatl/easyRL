import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os
import numpy as np
import torch

from algorithms.sac.agent import SACAgent
from algorithms.sac.config import config
from envs.highway_lane_keeping import make_racetrack_env
from utils.logger import Logger
from utils.hud import patch_viewer_for_hud, update_hud


def train():
    """Train SAC on racetrack-v0 continuous lateral control."""
    # env = make_racetrack_env(render_mode="human")
    env = make_racetrack_env(render_mode=None)
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]

    patch_viewer_for_hud(env)

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
    logger = Logger(log_dir=results_dir, use_tensorboard=True)
    logger.add_graph(agent.policy, torch.randn(1, state_dim).to(agent.device))

    total_steps = 0
    rewards_history = []
    n_episodes = config["n_episodes"]

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        episode_reward = 0
        done = False

        update_hud(episode + 1, n_episodes, 0.0, 0.0)

        while not done:
            if total_steps < config["start_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.take_action(state)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs.flatten()
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)

            if total_steps >= config["start_steps"]:
                agent.update()

            state = next_state
            episode_reward += reward
            total_steps += 1
            update_hud(episode + 1, n_episodes, 0.0, episode_reward)

        rewards_history.append(episode_reward)
        logger.log("episode_reward", episode, episode_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg(50): {avg_reward:.1f} | "
                  f"Steps: {total_steps}")

    logger.save()
    logger.close()
    env.close()
    agent.save(f"{results_dir}/sac_racetrack.pth")
    print(f"\nTraining complete. Model saved to {results_dir}/")


if __name__ == "__main__":
    train()
