import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
import os

import numpy as np

from algorithms.dqn.agent import DQNAgent
from algorithms.dqn.config import config as dqn_config
from algorithms.ppo.agent import PPOAgent
from algorithms.ppo.config import config as ppo_config
from algorithms.sac.agent import SACAgent
from algorithms.sac.config import config as sac_config
from envs.highway_lane_keeping import make_lane_keeping_env, make_continuous_lane_keeping_env
from experiments.evaluate import evaluate_agent
from utils.logger import Logger


def train_and_evaluate_dqn(n_episodes: int = 300):
    """Train DQN on highway-env lane keeping (discrete) and evaluate."""
    env = make_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=dqn_config["lr"],
        gamma=dqn_config["gamma"],
        epsilon=dqn_config["epsilon_start"],
        buffer_size=dqn_config["buffer_size"],
        batch_size=dqn_config["batch_size"],
        hidden_dim=dqn_config["hidden_dim"],
        tau=dqn_config["tau"],
    )

    log_dir = str(ROOT_DIR / "experiments" / "results" / "dqn")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir)
    epsilon = dqn_config["epsilon_start"]

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_obs.flatten()
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        epsilon = max(dqn_config["epsilon_end"], epsilon * dqn_config["epsilon_decay"])
        agent.epsilon = epsilon

        logger.log("reward", total_reward, episode)

        if (episode + 1) % 50 == 0:
            recent = [v for _, v in logger.get_data("reward")[-50:]]
            print(f"[DQN] Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {np.mean(recent):.2f} | Eps: {epsilon:.4f}")

    logger.save()
    logger.close()

    agent.epsilon = 0.0
    metrics = evaluate_agent(env, agent, n_episodes=20, flatten_obs=True)
    env.close()

    print(f"[DQN] Evaluation: mean_reward={metrics['mean_reward']:.2f}, "
          f"lateral_mean={metrics['lateral_mean']:.4f}")
    return metrics, logger


def train_and_evaluate_ppo(n_episodes: int = 300):
    """Train PPO on highway-env continuous lane keeping and evaluate."""
    env = make_continuous_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=ppo_config["lr"],
        gamma=ppo_config["gamma"],
        clip_eps=ppo_config["clip_eps"],
        epochs=ppo_config["epochs"],
        batch_size=ppo_config["batch_size"],
        hidden_dim=ppo_config["hidden_dim"],
        gae_lambda=ppo_config["gae_lambda"],
    )

    log_dir = str(ROOT_DIR / "experiments" / "results" / "ppo")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_reward = 0
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

        if done:
            next_value = 0.0
        else:
            _, _, next_value = agent.select_action(state)

        agent.update(states, actions, rewards, log_probs, values, dones, next_value=next_value)
        logger.log("reward", episode_reward, episode)

        if (episode + 1) % 50 == 0:
            recent = [v for _, v in logger.get_data("reward")[-50:]]
            print(f"[PPO] Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {np.mean(recent):.2f}")

    logger.save()
    logger.close()

    metrics = evaluate_agent(env, agent, n_episodes=20, flatten_obs=True)
    env.close()

    print(f"[PPO] Evaluation: mean_reward={metrics['mean_reward']:.2f}, "
          f"lateral_mean={metrics['lateral_mean']:.4f}")
    return metrics, logger


def train_and_evaluate_sac(n_episodes: int = 300):
    """Train SAC on highway-env continuous lane keeping and evaluate."""
    env = make_continuous_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=sac_config["lr"],
        gamma=sac_config["gamma"],
        tau=sac_config["tau"],
        alpha=sac_config["alpha"],
        buffer_size=sac_config["buffer_size"],
        batch_size=sac_config["batch_size"],
        hidden_dim=sac_config["hidden_dim"],
        auto_alpha=sac_config["auto_alpha"],
    )

    log_dir = str(ROOT_DIR / "experiments" / "results" / "sac")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir)
    total_steps = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        episode_reward = 0
        done = False

        while not done:
            if total_steps < sac_config["start_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_obs.flatten()

            agent.store_transition(state, action, reward, next_state, done)

            if total_steps >= sac_config["start_steps"]:
                agent.learn()

            state = next_state
            episode_reward += reward
            total_steps += 1

        logger.log("reward", episode_reward, episode)

        if (episode + 1) % 50 == 0:
            recent = [v for _, v in logger.get_data("reward")[-50:]]
            print(f"[SAC] Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {np.mean(recent):.2f} | Steps: {total_steps}")

    logger.save()
    logger.close()

    class DeterministicSACWrapper:
        def __init__(self, sac_agent):
            self._agent = sac_agent

        def select_action(self, state):
            return self._agent.select_action(state, deterministic=True)

    deterministic_agent = DeterministicSACWrapper(agent)
    metrics = evaluate_agent(env, deterministic_agent, n_episodes=20, flatten_obs=True)
    env.close()

    print(f"[SAC] Evaluation: mean_reward={metrics['mean_reward']:.2f}, "
          f"lateral_mean={metrics['lateral_mean']:.4f}")
    return metrics, logger


def main():
    """Run full comparison experiment."""
    results_dir = str(ROOT_DIR / "experiments" / "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("COMPARISON EXPERIMENT: DQN vs PPO vs SAC on Highway Lane Keeping")
    print("=" * 60)

    print("\n--- Training DQN (discrete actions) ---")
    dqn_metrics, _ = train_and_evaluate_dqn(n_episodes=300)

    print("\n--- Training PPO (continuous actions) ---")
    ppo_metrics, _ = train_and_evaluate_ppo(n_episodes=300)

    print("\n--- Training SAC (continuous actions) ---")
    sac_metrics, _ = train_and_evaluate_sac(n_episodes=300)

    results = {
        "DQN": dqn_metrics,
        "PPO": ppo_metrics,
        "SAC": sac_metrics,
    }

    results_path = os.path.join(results_dir, "comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 60)
    print(f"{'Metric':<22} {'DQN':>10} {'PPO':>10} {'SAC':>10}")
    print("-" * 60)
    for metric in ["mean_reward", "std_reward", "lateral_mean", "lateral_std",
                   "heading_mean", "steering_smoothness"]:
        print(f"{metric:<22} "
              f"{dqn_metrics[metric]:>10.4f} "
              f"{ppo_metrics[metric]:>10.4f} "
              f"{sac_metrics[metric]:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
