import random  # Python的内置模块，提供了生成随机数和执行随机操作的函数，如random.randint()用于生成随机整数，random.choice()用于从序列中随机选择一个元素等。
import gymnasium as gym  # 流行的强化学习库，提供了各种环境和工具，用于开发和测试强化学习算法。它包含了许多经典的强化学习环境，如CartPole、MountainCar等，方便我们进行实验和比较不同算法的性能。
import numpy as np  # 流行的Python库，提供了高效的多维数组对象和各种数学函数，用于科学计算和数据分析。它是许多机器学习和深度学习库的基础，常用于处理和操作数据。
import collections  # Python内置模块，提供了许多有用的数据结构，如deque（双端队列）和Counter（计数器）等。
from tqdm import (
    tqdm,
)  # Python库，用于显示循环的进度条，常用于长时间运行的任务中，以提供用户友好的反馈。
import torch  # 流行的深度学习框架，提供了强大的张量计算和自动微分功能，广泛用于构建和训练神经网络模型。
import torch.nn.functional as F  # PyTorch中的一个模块，提供了许多常用的神经网络函数，如激活函数、损失函数等，方便我们在构建和训练神经网络时使用。
import matplotlib.pyplot as plt  # Python库，用于绘制各种图表和可视化数据，常用于数据分析和结果展示。
import sys

sys.path.append("/home/lihongl/Desktop/myRL/easyRL/hands_on_rl")
import rl_utils  # 自定义的模块，包含一些强化学习相关的工具函数，如经验回放、策略更新等。


class ReplayBuffer:
    """经验回放池，用于存储智能体在环境中经历的状态、动作、奖励和下一个状态等信息，以便在训练过程中进行采样和学习。"""

    def __init__(self, capacity):
        self.buffer = collections.deque(
            maxlen=capacity
        )  # 使用双端队列来存储经验，当达到容量上限时，最旧的经验会被自动删除。即先进先出。

    def add(
        self, state, action, reward, next_state, done
    ):  # 将一个经验（状态、动作、奖励、下一个状态和是否结束）添加到经验回放池buffer中。
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size
    ):  # 从经验回放池(buffer)中随机采样一个批次的经验，用于训练智能体。数据量为batch_size
        transitions = random.sample(
            self.buffer, batch_size
        )  # 从buffer中随机采样batch_size个经验，返回一个列表，每个元素是一个经验元组（状态、动作、奖励、下一个状态和是否结束）。
        state, action, reward, next_state, done = zip(
            *transitions
        )  # 解包转置：从"每条经验一行"变为"每个字段一列"，便于批量计算
        return (
            np.array(state),
            action,
            reward,
            np.array(next_state),
            done,
        )  # 将状态和下一个状态转换为NumPy数组，并返回所有字段的批次数据。CartPole 环境的状态空间是四维的:位置、速度、杆的角度和角速度。动作是标量（0向左推或1向右推）

    def size(self):
        return len(self.buffer)  # 返回经验回放池中当前存储的经验数量，即buffer的长度。


class Qnet(torch.nn.Module):
    """只有一层隐藏层的Q网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()  # 调用父类的构造函数，初始化神经网络模块
        self.fc1 = torch.nn.Linear(
            state_dim, hidden_dim
        )  # 定义第一层全连接层，将输入的状态维度映射到隐藏层
        self.fc2 = torch.nn.Linear(
            hidden_dim, action_dim
        )  # 定义第二层全连接层，将隐藏层的输出映射到动作维度

    def forward(self, x):  # 定义前向传播函数，x是输入的状态
        x = F.relu(
            self.fc1(x)
        )  # 对输入x进行第一层全连接变换，并应用ReLU激活函数，得到隐藏层的输出。ReLU函数将输入中的负值置为0，正值保持不变，增加了网络的非线性表达能力。
        return self.fc2(x)  # 通过第二层全连接层，输出每个动作的Q值


class DQN:
    """DQN算法"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon,
        target_update,  # 每隔多少步更新一次目标网络
        device,
    ):
        self.action_dim = action_dim
        self.q_net = Qnet(
            state_dim, hidden_dim, self.action_dim
        ).to(
            device
        )  # .to(device)把网络的所有参数（权重和偏置）搬到指定设备上。device 通常是 "cpu" 或 "cuda"（GPU）。搬到 GPU 上计算会更快。
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=learning_rate,  # .parameters() 返回 Q 网络中所有可训练的权重和偏置。对于这个 Qnet，具体就是：fc1.weight, fc1.bias, fc2.weight, fc2.bias
        )  #  torch.optim.Adam: Adam优化算法, .parameters(): Q 网络中所有可训练参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:  # 以epsilon的概率选择随机动作，进行探索
            action = np.random.randint(
                self.action_dim
            )  # 随机选择一个动作， .randint()生成一个随机整数
        else:  # 否则选择当前Q网络预测的最优动作，进行利用
            state = torch.tensor(
                [state], dtype=torch.float
            ).to(
                self.device
            )  # 将状态转换为PyTorch张量，并搬到指定设备上。状态被包装在一个列表中，以便形成一个批次（batch）输入。
            action = (
                self.q_net(state).argmax().item()
            )  # 通过Q网络计算每个动作的Q值，并选择具有最大Q值的动作。argmax()返回最大值的索引，item()将其转换为Python标量。
        return action

    def update(self, transition_dict):  # 核心训练步骤
        states = torch.tensor(  # 把状态转换为PyTorch张量
            transition_dict["states"], dtype=torch.float
        ).to(self.device)
        actions = (
            torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        )  # .view(-1, 1) — 将形状重塑为列向量（N×1），-1 表示自动推断行数。这是为了后续用 gather 按动作索引取出对应的 Q 值

        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)

        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        q_values = self.q_net(states).gather(
            1, actions
        )  # 取出当前状态下，智能体选择的动作对应的Q值
        # gather(1, actions) 的意思是：
        # 在第1维（列维度）上做选择 → 即每一行里，左右挑一个列。
        # ￼
        # 第0行: [1.2, 3.4] → index=1 → 往右挑 → 取 3.4
        # 第1行: [2.1, 0.5] → index=0 → 取最左 → 取 2.1
        # 第2行: [0.8, 1.9] → index=1 → 往右挑 → 取 1.9
        # 举例：
        # 简单记忆： gather(1, ...) = 固定行，选列；gather(0, ...) = 固定列，选行。

        max_next_q_values = (
            self.q_net(next_states).max(1)[0].view(-1, 1)
        )  # 把下一个状态的最大Q值取出来，转成列的形式
        # .max(1) — 沿第1维（列维度）取最大值，返回一个命名元组 (values, indices)
        # [0] — 取 values，即每行的最大 Q 值，形状为 (batch_size,)（一维）
        # 举例：
        # q_net输出:           max(1):
        # [[1.2, 3.4],    →   values=[3.4, 2.1, 1.9]   → view(-1,1) →  [[3.4],
        #  [2.1, 0.5],        indices=[1, 0, 1]                         [2.1],
        #  [0.8, 1.9]]                                                  [1.9]]

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = F.mse_loss(
            q_values, q_targets
        )  # mse_loss()函数计算当前 Q 值和目标 Q 值之间的均方误差,即mean((q_values-q_targets)^2)
        """标准训练循环"""
        self.optimizer.zero_grad()  # 1.清除之前的梯度信息
        dqn_loss.backward()  # 2.反向传播算梯度
        self.optimizer.step()  # 3.更新模型参数（沿新的梯度迈一步，调用step()后，fc1.weight、fc2.bias 等参数的数值就被修改了，Q 网络的预测会更接近 TD 目标）
        """标准训练循环"""
        if (
            self.count % self.target_update == 0
        ):  # 每隔 target_update 次更新一次目标网络
            self.target_q_net.load_state_dict(
                self.q_net.state_dict()  # state_dict() 返回一个字典，包含了网络的所有参数（权重和偏置），注意这里不是强化学习里的"状态"，而是指模型的状态——即所有参数的当前值。
            )  # 将当前 Q 网络的参数复制到目标网络中，进行一次同步。
        self.count += 1  # 记录update() 被调用了多少次，当 count 是 target_update 的整数倍时，同步一次目标网络。


"""训练循环"""
lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env_name = "CartPole-v1"
RENDER_TRAINING = False  # 训练过程可视化开关，True=实时渲染（慢），False=不渲染（快）
RENDER_RESULT = True     # 训练完成后展示开关，True=展示训练效果，False=跳过
env = gym.make(env_name, render_mode="human" if RENDER_TRAINING else None)
random.seed(0)
np.random.seed(0)
torch.manual_seed(
    0
)  # 固定pytorch中的随机操作比如网络权重初始化（神经网络的权重在训练前需要赋初始值，PyTorch 默认用随机数初始化）、dropout（一种防止过拟合的正则化技术。训练时随机丢弃一部分神经元（将其输出置为0）等）

replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[
    0
]  # shape[0]是取出观测空间的形状的第一个维度，即状态的维度。CartPole环境的状态空间是四维的:位置、速度、杆的角度和角速度，所以 state_dim = 4
action_dim = env.action_space.n  # n是动作空间的离散动作数量。CartPole环境有两个离散动作：0（向左推）和1（向右推），所以 action_dim = 2
agent = DQN(
    state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device
)

return_list = []
for i in range(10):  # 将500episodes划分成10份，分别显示进度条
    print()  # 在每个进度条之间加一个空行
    with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 对于每一份的50条数据
            episode_return = 0
            # state, _ = env.reset(seed=0) #env.reset(seed=0) 每次都用相同的种子，导致每个 episode 的初始状态完全相同，智能体只学会了应对一种初始条件，泛化能力差。
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                        batch_size
                    )  # 从回放池里取出batch_size的随机经验
                    transition_dict = {
                        "states": b_s,
                        "actions": b_a,
                        "next_states": b_ns,
                        "rewards": b_r,
                        "dones": b_d,
                    }
                    agent.update(transition_dict)
            return_list.append(
                episode_return
            )  # 记录每一份（50条数据）的总回报。外层10*内层50,return_list一共会记录500个episodes的回报。
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix(
                    {  # 每 10 个 episode 更新一次进度条后缀信息：显示当前的全局 episode 编号和最近 10 个 episode 的平均回报
                        "episode": "%d " % (num_episodes / 10 * i + i_episode + 1),
                        "return": "%.3f" % np.mean(return_list[-10:]),
                    }
                )
            pbar.update(1)  # 每完成一个 episode，进度条前进一格

"""绘制结果"""
episodes_list = list(range(len(return_list)))  # 生成0～499的整数列表
plt.plot(episodes_list, return_list)  # 横轴为episodes，纵轴为回报
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title("Rewards: DQN on {}".format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title("Moving Average Rewards: DQN on {}".format(env_name))
plt.show()

"""训练完成后展示"""
if RENDER_RESULT:
    agent.epsilon = 0  # 展示时纯利用训练好的Q网络，不做随机探索
    episodes_to_display = 5
    env_show = gym.make(env_name, render_mode="human")
    for _ in range(episodes_to_display):  # 展示10个episode
        state, _ = env_show.reset()
        done = False
        episode_return = 0
        while not done:
            action = agent.take_action(state)
            state, reward, terminated, truncated, _ = env_show.step(action)
            done = terminated or truncated
            episode_return += reward
        print(f"展示 episode 回报: {episode_return}")
    env_show.close()
