import sys
from pathlib import Path

# 把项目根目录加入 Python 的模块搜索路径，使得 import algorithms.policy_gradient.agent 这类导入能找到。
ROOT_DIR = Path(__file__).resolve().parent.parent.parent 
# 告诉 Python："找模块时先到ROOT_DIR这里找"
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from algorithms.policy_gradient.agent import REINFORCEAgent
from algorithms.policy_gradient.config import config
from envs.highway_lane_keeping import make_lane_keeping_env
from utils.logger import Logger
from utils.plotting import plot_training_curves
from utils.hud import patch_viewer_for_hud, update_hud

# '''Tensorboard可视化'''
# # 终端1：启动后不用管，一直开着
# tensorboard --logdir=algorithms/policy_gradient/results

# 浏览器打开 http://localhost:6006 进入TensorBoard界面

# # 终端2：随时启动/停止训练
# python algorithms/policy_gradient/train.py

def train():
    """Train REINFORCE agent on highway-env lane keeping (discrete actions)."""
    # render_mode="human" enables real-time visualization; set to None to disable for faster training
    # env = make_lane_keeping_env(render_mode="human") # 开启可视化
    env = make_lane_keeping_env(render_mode=None) # 关闭可视化
    torch.manual_seed(0)
    obs, _ = env.reset(seed=0) # obs是初始观察状态
    state_dim = obs.flatten().shape[0] # flatten()拍平成一维，shape[0]取第0维的大小（元素总数）作为网络输入维度
    action_dim = env.action_space.n # highway-env DiscreteMetaAction有5个高层决策动作（底层由PID控制器执行）: 0=左换道(车道-1), 1=保持, 2=右换道(车道+1), 3=加速(+5m/s), 4=减速(-5m/s)

    patch_viewer_for_hud(env) # 在环境的viewer.display方法上打补丁，在每次渲染前显示当前训练进度（训练轮数episode/总轮数n_episodes, 探索率epsilon、累计奖励rewards）

    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"], # 1e-3
        gamma=config["gamma"], # 0.99
        hidden_dim=config["hidden_dim"], # 128
    )

    results_dir = str(Path(__file__).resolve().parent / "results")
    logger = Logger(log_dir=results_dir, use_tensorboard=True)

    # 把网络结构记录到 TensorBoard，方便可视化查看模型图。
    # torch.randn(1, state_dim) — 一个随机的假输入，TensorBoard 需要跑一次前向传播来追踪网络结构（哪些层、怎么连接）
    logger.add_graph(agent.policy, torch.randn(1, state_dim).to(agent.device))

    n_episodes = config["n_episodes"]

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        done = False
        truncated = False

        update_hud(episode + 1, n_episodes, 0.0, 0.0)

        steps = 0 # 记录每个episode活了多久
        while not (done or truncated):
            action = agent.take_action(state)
            obs, reward, done, truncated, _ = env.step(action)
            agent.store_reward(reward)
            state = obs.flatten()
            total_reward += reward
            steps += 1
            update_hud(episode + 1, n_episodes, 0.0, total_reward)

        # Update policy at end of episode
        loss = agent.update()

        # Log metrics
        logger.log("episode_reward", episode, total_reward)
        logger.log("loss", episode, loss)
        logger.log("episode_length", episode, steps)

        # Print progress
        if (episode + 1) % 50 == 0:
            # 取最近50轮的奖励数据，计算平均奖励作为近期表现指标
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
