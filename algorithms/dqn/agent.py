import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

'''1.定义Q网络, 包括初始化一个三层的MLP多层感知机网络、前向传播函数'''
class QNetwork(nn.Module):
    """3-layer MLP for Q-value approximation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # nn.Sequential: 将多层按顺序串联，输入依次经过每一层，等价于在forward中逐层调用
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 前向传播函数，输入状态张量x，输出对应动作的Q值张量
        return self.net(x)

'''2.定义经验回放池'''
class ReplayBuffer:
    """Fixed-size replay buffer for experience replay."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch) # 解压批次数据，分别得到状态、动作、奖励、下一个状态和是否结束的标志
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def size(self):
        return len(self.buffer)

'''3.定义DQN算法'''
class DQNAgent:
    """DQN agent with replay buffer and target network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float, # 学习率，控制网络权重更新的步长，值越大更新越快但可能不稳定，值越小更新更稳定但收敛更慢，通常设置在1e-4到1e-3之间
        gamma: float, # 折扣因子，控制未来奖励的权重，值越接近1表示更重视长期奖励，通常设置在0.9到0.99之间
        epsilon: float, # 探索率
        buffer_size: int,
        batch_size: int,
        hidden_dim: int = 128,
        tau: float = 0.005, # 软更新系数，控制目标网络参数向训练网络参数更新的速度，值越小更新越慢，通常设置在0.001到0.01之间
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.tau = tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def take_action(self, state: np.ndarray) -> int: #返回的是动作编号（整数）
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor([state]).to(self.device)  # [state]加一维变成(1,4)，网络要求输入为(batch_size, state_dim)，选动作时只有一个状态所以batch_size=1
        with torch.no_grad():  # PyTorch默认前向传播时会记录每步操作（为backward准备），这里只是选动作不会backward，关掉记录省内存
            q_values = self.q_net(state_t)
        # q_values形状如[[1.2, 3.4]]，第0维是batch size，第1维是每个动作的Q值
        # argmax(dim=1)返回的是最大值的索引(动作编号)，不是Q值本身。如[[1.2, 3.4]] → tensor([1])
        # .item()把tensor([1])这个单元素张量取出来变成普通Python数字1，注意取出的是argmax的结果(索引)不是Q值
        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done): #把一条经验（state, action, reward, next_state, done）存入 replay buffer,供后续训练时采样使用
        self.buffer.add(state, action, reward, next_state, done)

    def update(self) -> float: # 核心训练过程： 从 replay buffer 中采样一个批次的经验，计算当前 Q 网络的 Q 值和目标 Q 网络的目标 Q 值，计算损失并更新 Q 网络参数，同时软更新目标网络参数。返回当前批次的损失值。
        if self.buffer.size() < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # 所有参与运算的张量都必须遵循 (batch_size, 数据本身的维度) 的格式
        # states 每个样本有多个特征，从 buffer 取出时天然是 (64, 4)，已经二维，不需要 unsqueeze
        states_t = torch.FloatTensor(states).to(self.device)
        # actions 每个样本只有一个整数，取出时是 (64,)，缺少第二维，需要 unsqueeze(1) → (64, 1)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        # rewards/dones 同 actions，每个样本一个标量，需要 unsqueeze(1) 和 q_values(64,1) 对齐
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # self.q_net(states_t)输出所有动作的Q值，形状(64,2)如[[1.2,3.4],[2.1,0.5],...]
        # actions_t是当时实际选的动作编号，形状(64,1)如[[1],[0],...]
        # .gather(1, actions_t)在第1维(列)按actions_t索引取值，只保留实际选择的那个动作的Q值，结果(64,1)
        q_values = self.q_net(states_t).gather(1, actions_t)

        # 用目标网络计算下一状态的最大Q值作为TD目标，目标网络不参与梯度更新所以用no_grad
        with torch.no_grad():
            # self.target_net(next_states_t) — 用目标网络算下一状态所有动作的 Q 值，得到的形状是(batch_size, action_dim)
            # .max(1, keepdim=True) — 在第 1 维（动作维度）取最大值，返回 (values, indices)，keepdim=True 让结果保持 (64, 1) 而不是变成 (64,)
            # [0] — 取 values（最大 Q 值本身），不要 indices，最终得到的是一列最大动作Q值
            max_next_q_values = self.target_net(next_states_t).max(1, keepdim=True)[0]
            # TD目标：r + gamma * max_Q(s') * (1-done)，done时不考虑未来奖励
            targets = rewards_t + self.gamma * max_next_q_values * (1 - dones_t)

        loss = nn.SmoothL1Loss()(q_values, targets) #SmoothL1Loss又叫Huber Loss, 当误差<1时表现为L2损失[1/2*(x^2)]，误差>=1时表现为L1损失[|x| - 1/2]

        """标准训练循环"""
        self.optimizer.zero_grad() # 梯度清零，PyTorch默认梯度是累积的，每次反向传播前都要清零，否则会把多次反向传播的梯度加在一起
        loss.backward() # 反向传播计算梯度，计算过程中会自动构建计算图并求导，得到每个参数的梯度值
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0) # 梯度裁剪，防止梯度爆炸，max_norm是梯度的最大范数，如果超过这个值就按比例缩小所有梯度，使得总范数不超过max_norm
        self.optimizer.step() # 更新网络参数，使用优化器根据计算得到的梯度调整参数值，完成一次训练迭代
        """标准训练循环"""
        
        # 软更新目标网络：target_param = tau * param + (1 - tau) * target_param，tau越小更新越慢，保持目标网络稳定
        # zip把两个网络的参数按位置配对(fc1.weight对fc1.weight, fc1.bias对fc1.bias...)，不管多少层都能通用处理
        for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
            # .data取出底层纯数值张量，它没有梯度追踪功能，所以对它做运算PyTorch不会记录
            # 如果不用.data，运算会被记入计算图，backward时梯度可能意外流过这条路径污染Q网络的梯度，且内存无法释放
            # .copy_()原地赋值，直接改数值而不创建新张量
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # SmoothL1Loss默认reduction='mean'，把64个样本的误差平均成一个标量，所以loss只有一个数
        # .item()把这个单元素张量如tensor(0.03)转成普通Python浮点数0.03，避免返回张量带着计算图占内存
        return loss.item()

    def save(self, path: str):
        # state_dict()返回Q网络所有参数(权重和偏置)的字典，torch.save序列化保存到文件
        # 只保存了Q网络，足够推理用；若要断点续训还需保存optimizer状态和epsilon等
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        # map_location=self.device 解决保存和加载设备不一致的问题
        # 比如在GPU机器上训练保存了模型，换到没有GPU的机器上加载，默认会报错（因为参数标记着"我属于cuda"）
        # 指定map_location后，PyTorch会把参数统一搬到self.device上，GPU保存的模型在CPU上也能正常加载，反之亦然
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
