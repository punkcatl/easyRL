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


'''Tensorboard可视化'''
# # 终端1：启动后不用管，一直开着
# tensorboard --logdir=algorithms/dqn/results

# 浏览器打开 http://localhost:6006 进入TensorBoard界面

# # 终端2：随时启动/停止训练
# python algorithms/dqn/train.py

def train():
    """Train DQN agent on highway-env lane keeping (discrete actions)."""
    # render_mode="human" enables real-time visualization; set to None to disable for faster training
    env = make_lane_keeping_env(render_mode="human")
    obs, _ = env.reset() # env.reset()返回obs和info，obs是初始观察状态，info包含额外信息但这里不需要，所以用_占位符忽略它
    # obs可能是多维的如(5,5)：5辆车，每辆5个特征。flatten()拍平成一维(25,)，shape[0]取长度作为网络输入维度
    # 因为nn.Linear只接受一维向量输入，不能直接处理二维矩阵
    state_dim = obs.flatten().shape[0]
    # highway-env DiscreteMetaAction有5个高层决策动作（底层由PID控制器执行）:
    # 0=左换道(车道-1), 1=保持, 2=右换道(车道+1), 3=加速(+5m/s), 4=减速(-5m/s)
    action_dim = env.action_space.n

    # 在环境的viewer.display方法上打补丁，在每次渲染前显示当前训练进度（训练轮数episode/总轮数n_episodes, 探索率epsilon、累计奖励rewards）
    patch_viewer_for_hud(env)

    # Q网络结构: [25] →Linear→ [128] →ReLU→ [128] →ReLU→ [5]
    # 输入25维(5车×5特征flatten), 两个128维隐藏层, 输出5维(每个动作的Q值)
    # 参数量计算（每层 = 输入×输出 + 偏置）:
    #   Linear1: 25×128 + 128 = 3,328
    #   Linear2: 128×128 + 128 = 16,512
    #   Linear3: 128×5 + 5 = 645
    #   单个网络总计: 20,485  Q网络+目标网络共两份: 40,970
    agent = DQNAgent(
        state_dim=state_dim, # 25
        action_dim=action_dim, # 5
        lr=config["lr"], # 1e-3
        gamma=config["gamma"],  # 0.99
        epsilon=config["epsilon_start"], # 1.0
        buffer_size=config["buffer_size"], # 50000
        batch_size=config["batch_size"], # 64
        hidden_dim=config["hidden_dim"], # 128
        tau=config["tau"], # 0.005
    )

    # 设置results目录路径
    # 保存训练日志和模型参数, Path(__file__)是当前文件路径, .resolve()把相对路径转成绝对路径，parent取上级目录，/ "results"拼接出results目录的绝对路径
    results_dir = str(Path(__file__).resolve().parent / "results") 
    
    # Logger类负责记录训练过程中的数据（如每轮奖励），并保存到results目录下的日志文件中，供后续分析和绘图使用
    # use_tensorboard=True时会同时写入TensorBoard事件文件，可用tensorboard命令实时查看训练曲线
    logger = Logger(log_dir=results_dir, use_tensorboard=config["use_tensorboard"])

    # 将网络结构写入TensorBoard，可在GRAPHS标签页中查看
    import torch
    logger.add_graph(agent.q_net, torch.randn(1, state_dim).to(agent.device))

    n_episodes = config["n_episodes"] #总训练轮数
    epsilon = config["epsilon_start"] #探索率，设置为1.0表示初始阶段完全随机探索，随着训练进行会逐渐衰减到config["epsilon_end"]（0.01），使得智能体在训练后期更多地利用学到的知识而不是随机探索

    for episode in range(n_episodes): # 500轮训练，每轮从环境重置开始，直到done或truncated为True（表示一轮结束），每轮记录累计奖励并更新HUD显示当前进度。truncated是新引入的环境终止条件，表示因为达到最大步数限制而结束一轮，而不是智能体本身的行为导致的done（如撞车）。
        # 重置环境，返回初始观察状态obs和info（这里不需要，所以用_占位符忽略它）
        # reset 的作用是把环境恢复到初始状态：车辆回到起始位置、周围车重新随机生成、速度归零等。就像游戏里"重新开始一局"。
        obs, _ = env.reset() 
        
        state = obs.flatten()
        total_reward = 0
        episode_losses = []
        episode_steps = 0
        done = False
        truncated = False

        update_hud(episode + 1, n_episodes, epsilon, 0.0) # 初始累计奖励为0.0，HUD显示当前轮数、总轮数、探索率和累计奖励

        while not (done or truncated):
            # agent.take_action(state)根据当前状态state选择一个动作action，可能是随机选择（以epsilon概率）或根据Q网络预测选择（以1-epsilon概率）
            action = agent.take_action(state)
            
            # env.step()执行智能体选择的动作，返回新的观察状态obs、奖励reward、是否结束done、是否达到最大步数truncated和额外信息_（这里不需要，所以用_占位符忽略它）
            obs, reward, done, truncated, _ = env.step(action) 
            
            # obs是环境返回的新状态(5,5)，flatten()拍平成一维(25,)作为下一个状态next_state，供agent.store_transition()存储到经验回放池中
            next_state = obs.flatten()
            
            # agent.store_transition()把当前状态state、动作action、奖励reward、下一个状态next_state和是否结束done存储到经验回放池中，供后续训练使用
            agent.store_transition(state, action, reward, next_state, done)
            
            # agent.update()从经验回放池中采样一个批次的经验，计算当前Q网络的Q值和目标Q网络的目标Q值，计算损失并更新Q网络参数，同时软更新目标网络参数。
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward
            episode_steps += 1
            update_hud(episode + 1, n_episodes, epsilon, total_reward) #这里其实只有total_reward变化了

        # Decay epsilon
        epsilon = max(config["epsilon_end"], epsilon * config["epsilon_decay"]) # epsilon_end：最小探索率[数值是0.01]
        agent.epsilon = epsilon

        # Log metrics
        logger.log("episode_reward", episode, total_reward)
        logger.log("epsilon", episode, epsilon)
        logger.log("episode_steps", episode, episode_steps)
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            logger.log("loss", episode, avg_loss)

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

# __name__ 是 Python 自动给每个文件设置的一个变量
if __name__ == "__main__": 
    train()
