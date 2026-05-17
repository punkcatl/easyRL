import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from algorithms.dqn.agent import DQNAgent
from algorithms.dqn.config import config
from envs.highway_lane_keeping import make_lane_keeping_env
from utils.logger import Logger
from utils.plotting import plot_training_curves
from utils.hud import patch_viewer_for_hud, update_hud


def train():
    """Train DQN agent on highway-env lane keeping (discrete actions)."""
    # render_mode="human" enables real-time visualization; set to None to disable for faster training
    env = make_lane_keeping_env(render_mode="human")
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n

    patch_viewer_for_hud(env)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon=config["epsilon_start"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        tau=config["tau"],
    )

    results_dir = str(Path(__file__).resolve().parent / "results")
    logger = Logger(log_dir=results_dir)

    n_episodes = config["n_episodes"]
    epsilon = config["epsilon_start"]

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        done = False
        truncated = False

        update_hud(episode + 1, n_episodes, epsilon, 0.0)

        while not (done or truncated):
            action = agent.select_action(state)
            obs, reward, done, truncated, _ = env.step(action)
            next_state = obs.flatten()
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            update_hud(episode + 1, n_episodes, epsilon, total_reward)

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

    agent.save(f"{results_dir}/dqn_highway.pth")
    print(f"\nTraining complete. Results saved to {results_dir}/")
    env.close()


if __name__ == "__main__":
    train()
