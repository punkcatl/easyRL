try:
    import gymnasium as gym
except ImportError:
    import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import rl_utils

'''定义策略网络'''
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 输出动作概率分布，需要softmax归一化
        return F.softmax(self.fc2(x), dim=1) # dim=1 是沿动作维度（列方向）归一化，dim=0是batch维度

# 策略网络训练：
# loss = -log_prob * G_t
# 梯度让"好动作的 logit 变大，坏动作的变小"
# → 输出的是"哪个动作更该选"的相对偏好，经过softmax处理后变成概率分布

# Q 网络训练：
# loss = (Q_predicted - (reward + γ * Q_target))²
# 梯度让"输出逼近真实回报"
# → 输出的是"选这个动作能拿多少分"

# 类比：同样一支笔，用来画画就产生画，用来写字就产生字。笔的结构一样，用途不同结果就不同。
# 神经网络就是那支笔——同样的 Linear(state_dim, action_dim) 结构：
# 用策略梯度 loss 训练 → 输出变成动作偏好分数
# 用 TD loss 训练 → 输出变成动作价值估计
# 结构决定能力（能拟合什么复杂度的函数），loss 决定含义（拟合的目标是什么）。

'''定义价值网络'''
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # 输出一个标量值
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 输出价值标量，不需要softmax
        return self.fc2(x) 
    
'''定义 Actor-Critic 算法，主要包含采取动作、更新网络参数两个函数'''
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        # 网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.device = device
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device) # 转成张量并加 batch 维
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs) # 创建离散概率分布对象
        action = action_dist.sample() # 从分布中采样动作
        return action.item() # 返回动作索引
    
    def update(self, transition_dict):
        # 标量数据要转成列向量（shape [batch_size, 1]）
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device) # view(-1,1)自动推断第一维并转成列向量。动作索引本身是int64不需要转成float。
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # 时序差分目标：TD_target = r + γ * V(next_state) * (1 - done)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states) # 时序差分误差
        # 从动作概率分布中抽取当前动作的概率，然后取对数得到log_prob
        log_probs = torch.log(self.actor(states).gather(1, actions)) # gather(dim, index)，gather 的作用是沿第dim维按索引index从张量中抽取指定位置的值。
        
        # 策略网络的 loss
        # 1. td_delta — TD 优势估计，即 A(s,a) ≈ r + γV(s') - V(s)
        # 表示"这个动作比平均好多少"。正 = 好动作，负 = 差动作。
        
        # 2. .detach() — 切断梯度
        # td_delta 是通过 Critic 网络算出来的，带有计算图。detach() 把它变成纯数值常数，这样 backward 时梯度只流向 Actor，不会影响 Critic。（Actor 和 Critic 各自有独立的 loss，分开更新）
        
        # 3. -log_probs * td_delta — 策略梯度核心
        # 和 REINFORCE 的 -log π(a|s) * Gₜ 一样，只是把 Gₜ 换成了 td_delta（优势函数）。
        # td_delta > 0 → 好动作 → loss 为负 → 增大该动作概率
        # td_delta < 0 → 差动作 → loss 为正 → 减小该动作概率
        
        # 4. torch.mean() — 对 batch 取平均
        # REINFORCE 用 .sum()，这里用 .mean()——效果等价（只是梯度缩放一个常数倍），mean 让 loss 大小不依赖 episode 长度，更稳定。
        actor_loss = torch.mean(-log_probs * td_delta.detach()) # detach() 切断 td_delta 的梯度传播，使其作为常数参与 actor_loss 的计算
        
        # 价值网络的 loss
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        # 1. self.critic(states) — Critic 网络对当前状态的价值估计 V(s)，形状 [batch, 1]

        # 2. td_target — 真实目标值 = r + γ·V(s')，表示"这个状态实际上值多少"

        # 3. .detach() — 切断梯度
        # detach 后把目标值当作固定常数——类似 DQN 里目标网络的作用。
        # 如果不 detach，backward 时梯度会经过两条路径：V(s)→θ（正确）和 td_target→V(s')→θ（有害），
        # 两条路径叠加在同一组参数上，等于同时"靠近目标"和"推远目标"，训练无法收敛。
        # 注意：detach 只解决单次 backward 内不自相矛盾，靶子跨 episode 仍会移动（Critic 每轮都更新），
        # 但只要 lr 合适，Critic 越来越准，靶子越动越接近真实值，最终收敛。

        # 4. F.mse_loss(...) — 均方误差
        # MSE = (V(s) - td_target)², 让 Critic 的预测 V(s) 逼近真实目标 td_target。

        # 5. torch.mean(...) — 对 batch 取平均
        # 实际上 F.mse_loss 默认已经取了 mean（reduction='mean'），外面再套一层 torch.mean 是多余的，不影响结果（标量的 mean 等于自身）。

        # 本质：Critic 的训练目标就是"让 V(s) 尽可能接近 r + γ·V(s')"
        
        # 清除梯度
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # 反向传播计算梯度
        actor_loss.backward()
        critic_loss.backward()
        # 更新网络参数
        self.actor_optimizer.step()
        self.critic_optimizer.step()

'''开始训练'''
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v1'
# env = gym.make(env_name)
env = gym.make(env_name, render_mode='human')
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

# train_on_policy_agent: 封装了 on-policy 算法通用的训练循环
# 循环 num_episodes 轮，每轮：reset环境 → 收集完整轨迹到 transition_dict → 调 agent.update() → 记录 episode return
# Actor-Critic / PPO / TRPO 等 on-policy 算法的训练流程完全一样，只是 agent.update() 内部逻辑不同
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

