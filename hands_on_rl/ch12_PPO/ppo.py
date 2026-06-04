import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

'''
PPO 是 on-policy 算法，每个回合的流程：

用当前策略在环境里跑一条轨迹（采样）
用这条轨迹的数据更新 epochs 次
丢弃旧数据，回到第 1 步用更新后的策略重新采样
每次都要重新采样，因为 PPO 的理论保证建立在"数据来自当前策略"的前提上。clip 机制允许你复用同一条轨迹多次（epochs 次），但不能跨回合复用旧轨迹——策略变了之后旧数据就不能再用了。

这也是 on-policy 方法相比 off-policy（如 DQN 用 replay buffer）的主要劣势：数据利用率低，每条轨迹用完就扔。
'''


'''定义策略网络'''
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1) # dim=-1 表示“沿最后一维计算”，对 [batch_size, action_dim] 来说就是对每个样本的动作向量做归一化。
       
'''定义价值网络'''
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        
        # 价值网络要预测的是一个标量价值通，常表示当前状态的状态价值V(s)
        # - 策略网络 PolicyNet 需要为每个离散动作给概率，所以输出是 action_dim
        # - 价值网络 ValueNet 只需要回答“这个状态有多好”，所以输出是单个分数（size=1）
        # 如果 batch 输入形状是 [batch_size, state_dim]，那价值网络输出就是 [batch_size, 1]
        self.fc2 = torch.nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
'''定义 PPO-截断 算法，主要包含采取动作、更新网络参数两个函数'''
class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # 网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 超参数
        self.gamma = gamma
        self.lmbda = lmbda # ？
        self.epochs = epochs # 一条序列的数据用来训练的轮数
        self.eps = eps # PPO截断范围参数
        self.device = device
        
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device) # 转成张量并加上batch维度
        probs = self.actor(state) # [1, action_dim]
        action_dist = torch.distributions.Categorical(probs) # 创建一个离散分布对象
        action = action_dist.sample() # 从分布中采样一个动作
        return action.item() # 返回动作的整数索引
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device) # [batch_size, state_dim]，transition_dict['actions'] 存的是采样时每步选的动作编号
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device) # [batch_size, 1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device) # [batch_size, 1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device) # [batch_size, state_dim]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device) # [batch_size, 1]
        td_target = rewards + self.gamma * self.critic(next_states).detach() * (1 - dones) # [batch_size, 1]
        td_delta = (td_target - self.critic(states)).detach() # [batch_size, 1]
        # 计算GAE优势函数
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device) #.cpu()把张量移到CPU上，为了在 compute_advantage 中调用 numpy() 方法
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach() # [batch_size, 1]， gather(dim, index)沿指定维度按 index 取值，这里dim=1 表示沿列（动作维度）取实际执行动作的概率

        for _ in range(self.epochs):
            # 计算新概率
            log_probs = torch.log(self.actor(states).gather(1, actions)) # [batch_size, 1]
            
            # 计算概率比率
            # exp(log(new_prob) - log(old_prob)) = exp(log(new_prob/old_prob)) = new_prob / old_prob
            ratio = torch.exp(log_probs - old_log_probs) # [batch_size, 1]
            
            # 计算截断的目标函数
            surr1 = ratio * advantage # [batch_size, 1]
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage # [batch_size, 1]
            
            # PPO的目标函数是取两者的最小值，求平均作为损失
            # 因为前面的 torch.min(surr1, surr2) 得到的是 batch 中每个样本各自的损失，shape 是 (batch_size, 1)。但优化器需要一个标量（单个数字）才能做反向传播，所以用 .mean() 把整个 batch 的损失平均成一个值。
            # 这也是 mini-batch 梯度下降的标准做法：对 batch 内所有样本的 loss 取平均，得到这批数据的期望损失。
            # 这里的负号：是因为我们要最大化 PPO 的目标函数，但优化器默认是最小化损失，所以加个负号把最大化问题转成最小化问题。
            actor_loss = torch.mean(-torch.min(surr1, surr2)) 
            
            # self.critic(states) — 价值网络对当前 state 的估计 V(s)
            # td_target — 更好的估计值：r + gamma * V(s_next) * (1 - done)，即"走一步后的实际收益 + 对未来的估计"
            # F.mse_loss(...) — 均方误差，让 V(s) 尽量接近 td_target
            # .detach() — 把 td_target 从计算图中断开，当作固定常数。因为 td_target 里也用了 self.critic(next_states)，如果不 detach，反向传播时梯度会通过 td_target 再流回 critic，导致训练不稳定（相当于目标在追着自己跑）
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())
            
            # 更新策略网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 更新价值网络
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
'''车杆环境训练PPO'''
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500 # num_episodes 是整个训练过程中的总回合数，一个回合一条数据序列
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10 # epochs 是每个回合（每条序列数据用来）训练的轮数，所以总的梯度更新次数是 num_episodes * epochs。
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {} (Moving Average)'.format(env_name))
plt.show()

'''可视化训练好的 agent'''
env_vis = gym.make(env_name, render_mode='human')
state, _ = env_vis.reset()
done = False
total_reward = 0
while not done:
    action = agent.take_action(state)
    state, reward, terminated, truncated, _ = env_vis.step(action)
    done = terminated or truncated
    total_reward += reward
print(f'Visualization episode reward: {total_reward}')
env_vis.close()
