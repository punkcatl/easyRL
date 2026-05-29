import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("/home/lihongl/Desktop/myRL/easyRL/hands_on_rl")
import rl_utils

'''策略网络'''
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        # 张量形状变化：
        # x:          (batch_size, state_dim)
        #   → fc1:    (batch_size, hidden_dim)
        #   → ReLU:   (batch_size, hidden_dim)   负值置0，增加非线性
        #   → fc2:    (batch_size, action_dim)    每个动作的原始分数(logits)
        #   → softmax: (batch_size, action_dim)   归一化为概率分布 π(a|s)
        #              dim=1 沿动作维度（列方向）归一化，每行求和=1
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

'''REINFORCE智能体'''
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate) #使用adam优化器
        self.gamma = gamma
        self.device = device
        
    def take_action(self, state):
        # 输入: state 是 gym 环境返回的 1D numpy 数组, shape (state_dim,)
        # 目标: 根据策略网络输出的概率分布 π(a|s) 随机采样一个动作
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # 训练时 batch=N → shape (N, state_dim); 推理单步 → shape (1, state_dim)
        # [state]: gym 环境返回的 state 是一个 1D 数组，用列表包一层, 将 shape (state_dim,) 升为 (1, state_dim)
        # dtype=torch.float: gym 返回 float64, 网络权重是 float32, 必须对齐
        probs = self.policy_net(state) # 用网络进行一次前向传播，得到的是每个动作被选择的概率，后续会用这个概率分布做随机采样，选出本步执行的动作
        action_dist = torch.distributions.Categorical(probs) # Categorical 是 PyTorch 提供的离散概率分布对象，用概率向量初始化后，可以从中采样或计算对数概率。
        action = action_dist.sample() # 这里会按概率随机抽一个动作
        return action.item() # 返回一个整数，表示动作的索引
    
    def update(self, transition_dict):
        reward_list = transition_dict['rewards'] # 从 transition_dict 中提取 reward_list，reward_list 是一个列表，包含了一个 episode 中每一步的奖励值。
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        
        G = 0
        self.optimizer.zero_grad() # 梯度清零
        for i in reversed(range(len(reward_list))): #从最后一步算起，逆序通过累加得到每一步的回报 G_t = r_t + γ * G_{t+1}，节省了重复计算的时间，属于动态规划的思想。
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device) # .view(-1, 1) 就是把张量强制变成"N行1列"的竖列形状。第一个 -1 表示：这一维的大小由 PyTorch 自动推断。
            # 从动作概率分布中抽取当前动作的概率，然后取对数得到 log_prob
            log_prob = torch.log(self.policy_net(state).gather(1, action)) # gather(dim, index)，gather 的作用是沿第dim维按索引index从张量中抽取指定位置的值。
            G = self.gamma * G + reward # 计算当前时间步的回报 G_t = r_t + γ * G_{t+1}
            loss = -log_prob * G # 最大化回报G，等价于最小化loss，所以加上负号
            # G 是常数(Python float)，不在计算图中
            # backward() 求导: ∂loss/∂θ = -G · ∇_θ log π(a|s)
            # 等价于数学公式: 先对 log π 求梯度再乘 G，常数提前乘后效果相同
            loss.backward() # 反向传播，计算梯度
        self.optimizer.step() # 根据梯度更新网络参数
        
'''训练主函数'''
learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v1'
# env = gym.make(env_name, render_mode='human')
env = gym.make(env_name)
torch.manual_seed(0) #设置 PyTorch 随机数种子，确保实验可复现，影响的是 dist.sample()、torch.randn() 等随机操作。
state_dim = env.observation_space.shape[0] # 状态空间维度
action_dim = env.action_space.n # 动作空间维度
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

return_list = []
for i in range(10): # 10批训练，每批100个episodes
    print()
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0 # 用于记录当前 episode 的总奖励（回报），每执行一步就把奖励累加到 episode_return 中
            transition_dict = { # 用于存储一个 episode 中的所有步的状态、动作、奖励等信息，供后续 update 函数使用
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state, _ = env.reset(seed=0)
            done = False
            while not done:
                action = agent.take_action(state) # 根据当前状态选择一个动作
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated # 执行动作，获得下一个状态、奖励和是否结束的标志
                # 将当前状态、动作、奖励等信息存储到 transition_dict 中
                transition_dict['states'].append(state) 
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state # 状态更新为下一个状态
                episode_return += reward
            return_list.append(episode_return) # 将当前 episode 的总奖励添加到 return_list 中
            agent.update(transition_dict) # 使用当前 episode 的 transition_dict 来更新策略网络
            if (i_episode + 1) % 10 == 0: # 每10个 episode 更新一次进度条显示
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1), # 显示当前 episode 的编号
                    'return':
                    '%.3f' % np.mean(return_list[-10:]) # 显示最近10个 episode 的平均回报。-10 表示倒数第10个，: 表示"到末尾"
                })
            pbar.update(1) # 更新进度条，表示又完成了一个 episode 的训练
            
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on ' + env_name)
plt.show()

mv_return = rl_utils.moving_average(return_list, 9) # 计算 return_list 的移动平均，窗口大小为 9
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on ' + env_name + ' Moving Average')
plt.show()
            
            
        
        

            

    
    
        
        
        