import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os
import numpy as np
import torch

from algorithms.ppo.agent import PPOAgent
from algorithms.ppo.config import config
from envs.highway_lane_keeping import make_racetrack_env
from utils.logger import Logger
from utils.hud import patch_viewer_for_hud, update_hud


def train():
    """训练PPO智能体在racetrack-v0赛道上进行连续横向控制。

    训练流程（on-policy + rollout buffer）：
    1. 用当前策略与环境交互，累积rollout_steps步数据到buffer
    2. 对buffer中的数据计算GAE优势估计
    3. 用PPO clip目标函数多轮更新策略
    4. 清空buffer，回到步骤1
    """
    # 创建赛道环境，render_mode="human"开启实时可视化
    env = make_racetrack_env(render_mode="human")
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]    # 观测维度：6 (x, y, vx, vy, cos_h, sin_h)
    action_dim = env.action_space.shape[0]  # 动作维度：1 (steering)

    # 在渲染窗口上叠加HUD信息（当前轮次、奖励等）
    patch_viewer_for_hud(env)

    # 初始化PPO智能体
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
        entropy_coef=config["entropy_coef"],
    )

    # 初始化日志记录器（CSV + TensorBoard）
    results_dir = str(Path(__file__).resolve().parent / "results")
    os.makedirs(results_dir, exist_ok=True)
    logger = Logger(log_dir=results_dir, use_tensorboard=True)
    logger.add_graph(agent.actor, torch.randn(1, state_dim).to(agent.device))

    n_episodes = config["n_episodes"]
    rollout_steps = config["rollout_steps"]

    # 训练状态
    state = obs.flatten()
    episode_reward = 0
    episode_count = 0
    rewards_history = []

    update_hud(1, n_episodes, 0.0, 0.0)

    while episode_count < n_episodes:
        # ===== 收集 rollout buffer =====
        # 累积足够步数的数据后再更新，解决单episode太短导致样本不足的问题
        states, actions_raw, rewards, log_probs, values, dones = [], [], [], [], [], []

        for _ in range(rollout_steps):
            # 策略采样：从高斯分布N(mean, std)中采样动作
            action_clipped, action_raw, log_prob, value = agent.take_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated or truncated

            states.append(state)
            actions_raw.append(action_raw)  # 存原始动作，保证log_prob一致性
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            state = next_obs.flatten()
            episode_reward += reward
            update_hud(episode_count + 1, n_episodes, 0.0, episode_reward)

            if done:
                episode_count += 1
                rewards_history.append(episode_reward)
                logger.log("episode_reward", episode_count, episode_reward)

                if episode_count % 50 == 0:
                    avg_reward = np.mean(rewards_history[-50:])
                    print(f"Episode {episode_count}/{n_episodes} | "
                          f"Avg Reward (last 50): {avg_reward:.2f}")

                episode_reward = 0
                if episode_count >= n_episodes:
                    break
                obs, _ = env.reset()
                state = obs.flatten()
                update_hud(episode_count + 1, n_episodes, 0.0, 0.0)

        # ===== 计算 next_value 用于 bootstrap =====
        # terminated时value=0；truncated或rollout中断时用Critic估计剩余价值
        if dones[-1]:
            next_value = 0.0
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                next_value = agent.critic(state_t).item()

        # ===== PPO更新 =====
        agent.update(states, actions_raw, rewards, log_probs, values, dones,
                     next_value=next_value)

    logger.save()
    logger.close()
    env.close()
    agent.save(f"{results_dir}/ppo_racetrack.pth")
    print(f"\nTraining complete. Model saved to {results_dir}/")


if __name__ == "__main__":
    train()
