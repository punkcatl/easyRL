import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from algorithms.policy_gradient.agent import REINFORCEAgent
from algorithms.policy_gradient.config import config
from envs.highway_lane_keeping import make_lane_keeping_env
from utils.logger import Logger
from utils.plotting import plot_training_curves


def train():
    """Train REINFORCE agent on highway-env lane keeping (discrete actions)."""
    env = make_lane_keeping_env()
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

    results_dir = str(Path(__file__).resolve().parent / "results")
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

    plot_training_curves(
        log_dir=results_dir,
        tags=["episode_reward"],
        save_path=f"{results_dir}/training_curve.png",
    )

    agent.save(f"{results_dir}/reinforce_highway.pth")
    print(f"\nTraining complete. Results saved to {results_dir}/")
    env.close()


if __name__ == "__main__":
    train()
