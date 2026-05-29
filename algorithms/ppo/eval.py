import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from algorithms.ppo.agent import PPOAgent
from algorithms.ppo.config import config
from envs.highway_lane_keeping import make_continuous_lane_keeping_env


def evaluate(model_path: str = None, n_episodes: int = 10):
    """Load trained PPO agent and evaluate with deterministic policy (use mean, no sampling)."""
    env = make_continuous_lane_keeping_env(render_mode="human")
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        hidden_dim=config["hidden_dim"],
        clip_eps=config["clip_eps"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        gae_lambda=config["gae_lambda"],
    )

    if model_path is None:
        model_path = str(Path(__file__).resolve().parent / "results" / "ppo_highway.pth")
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    rewards = []
    lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            # 确定性策略：直接用 actor 输出的 mean，不采样
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                mean, _ = agent.actor(state_t)
            action = mean.clamp(-1.0, 1.0).cpu().numpy().flatten()

            obs, reward, done, truncated, _ = env.step(action)
            state = obs.flatten()
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"Episode {episode + 1}/{n_episodes} | Reward: {total_reward:.2f} | Length: {steps}")

    print(f"\n=== Evaluation Results ===")
    print(f"Episodes: {n_episodes}")
    print(f"Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Avg Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")

    env.close()


if __name__ == "__main__":
    evaluate()
