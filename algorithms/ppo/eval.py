import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from algorithms.ppo.agent import PPOAgent
from algorithms.ppo.config import config
from envs.highway_lane_keeping import make_racetrack_env


def evaluate(model_path: str = None, n_episodes: int = 10):
    """加载训练好的PPO模型，用确定性策略（直接使用mean，不采样）进行评估。

    评估时不采样而用mean的原因：
    - 训练时采样是为了探索，评估时只关心学到的最优策略
    - mean是高斯分布的众数，代表策略认为最好的动作
    """
    env = make_racetrack_env(render_mode="human")
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]

    # 创建agent并加载训练好的模型参数
    # 评估时只需要网络结构参数（state_dim, action_dim, hidden_dim）
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        hidden_dim=config["hidden_dim"],
        clip_eps=config["clip_eps"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
    )

    if model_path is None:
        model_path = str(Path(__file__).resolve().parent / "results" / "ppo_racetrack.pth")
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

        while not done:
            # 确定性策略：直接用Actor输出的mean作为动作，不从分布中采样
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                mean, _ = agent.actor(state_t)
            action = mean.clamp(-1.0, 1.0).cpu().numpy().flatten()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = obs.flatten()
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"Episode {episode + 1}/{n_episodes} | Reward: {total_reward:.2f} | Length: {steps}")

    # 打印评估统计结果
    print(f"\n=== Evaluation Results ===")
    print(f"Episodes: {n_episodes}")
    print(f"Avg Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Avg Length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")

    env.close()


if __name__ == "__main__":
    evaluate()
