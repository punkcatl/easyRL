import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from algorithms.sac.agent import SACAgent
from algorithms.sac.config import config
from envs.highway_lane_keeping import make_continuous_lane_keeping_env


def evaluate(model_path: str = None, n_episodes: int = 10):
    """Load trained SAC agent and evaluate with deterministic policy."""
    env = make_continuous_lane_keeping_env(render_mode="human")
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        hidden_dim=config["hidden_dim"],
        tau=config["tau"],
        alpha=config["alpha"],
        buffer_size=100,
        batch_size=1,
    )

    if model_path is None:
        model_path = str(Path(__file__).resolve().parent / "results" / "sac_highway.pth")
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
            # 确定性策略：deterministic=True 直接用 mean，不加噪声
            action = agent.take_action(state, deterministic=True)
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
