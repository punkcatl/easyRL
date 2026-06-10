import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class Actor(nn.Module):
    """策略网络（Actor）：输出连续动作的高斯分布参数。

    网络结构：state → 两层Tanh全连接 → mean_head输出均值
    标准差通过可学习参数log_std控制，与状态无关（全局共享）。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # 特征提取层，使用Tanh激活函数
        # Tanh比ReLU更适合小网络+连续控制：输出有界、梯度流平滑、无dead neuron问题
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # 均值输出头：将隐层特征映射到动作空间维度
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # log_std是可学习参数，用exp()保证std>0
        # mean移动到最优动作位置，log_std收窄探索范围，
        # 但分布形状永远是关于mean对称的钟形
        # 初始化为-0.5（std≈0.6），使大部分采样落在有效动作范围[-1,1]内
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, x):
        x = self.net(x)
        mean = self.mean_head(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class Critic(nn.Module):
    """价值网络（Critic）：估计状态价值V(s)。

    输入状态，输出标量值，用于计算优势函数A(s,a) = Q(s,a) - V(s)。
    为Actor提供基线（baseline），降低策略梯度的方差。
    """

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # 输出单个标量V(s)
        )

    def forward(self, x):
        return self.net(x)


class PPOAgent:
    """PPO (Proximal Policy Optimization) 智能体，适用于连续动作空间。

    核心思想：通过clip机制限制策略更新幅度，防止单次更新过大导致性能崩溃。
    采用Actor-Critic架构：Actor输出动作分布，Critic评估状态价值。
    """

    def __init__(self, state_dim, action_dim, lr, gamma, clip_eps, epochs,
                 batch_size, hidden_dim=64, gae_lambda=0.95, max_grad_norm=0.5,
                 entropy_coef=0.005):
        self.gamma = gamma              # 折扣因子，控制对未来奖励的重视程度
        self.clip_eps = clip_eps        # PPO clip范围，限制策略更新幅度
        self.epochs = epochs            # 每批数据重复训练的轮数
        self.batch_size = batch_size    # 小批量大小
        self.gae_lambda = gae_lambda    # GAE的lambda参数，平衡偏差与方差
        self.max_grad_norm = max_grad_norm  # 梯度裁剪阈值，防止梯度爆炸
        self.entropy_coef = entropy_coef    # 熵奖励系数，鼓励探索
        self.action_dim = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        # Critic学习率设为Actor的3倍，使value估计更快收敛，减少GAE噪声
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr * 3)

    def take_action(self, state: np.ndarray):
        """根据当前策略采样动作。

        流程：state → Actor输出(mean, std) → 构造高斯分布 → 采样
        返回：(clamp后的动作用于环境执行, 原始动作用于buffer存储, log_prob, value)
        log_prob基于原始未截断动作计算，与evaluate()中的计算基准一致。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.actor(state_t)
            value = self.critic(state_t)
        dist = Normal(mean, std)
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        # clamp到[-1,1]用于环境执行，但存储原始action保证log_prob一致性
        action_clipped = action_raw.clamp(-1.0, 1.0)
        return (action_clipped.cpu().numpy().flatten(),
                action_raw.cpu().numpy().flatten(),
                log_prob.item(),
                value.item())

    def evaluate(self, states_t, actions_t):
        """用当前策略重新评估历史动作。

        PPO的关键步骤：用新策略计算旧动作的log_prob，
        与采集时的old_log_prob对比得到importance ratio。
        actions_t是原始未截断的动作，保证与采集时log_prob计算基准一致。
        """
        mean, std = self.actor(states_t)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions_t).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()  # 熵奖励，鼓励探索
        values = self.critic(states_t).squeeze(-1)
        return log_probs, entropy, values

    def update(self, states, actions_raw, rewards, log_probs, values, dones, next_value):
        """PPO策略更新。

        步骤：
        1. 计算GAE优势估计
        2. 多轮epoch复用同一批数据（PPO的样本效率优势）
        3. 用clipped surrogate objective更新Actor
        4. 用MSE loss更新Critic

        参数:
            actions_raw: 原始未截断的动作（与log_probs计算基准一致）
            next_value: 最后一个状态的value估计（truncated时非零）
        """
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions_raw)).to(self.device)
        old_log_probs_t = torch.FloatTensor(log_probs).to(self.device)

        # ===== GAE (Generalized Advantage Estimation) =====
        # 通过lambda加权多步TD误差，平衡偏差（lambda→0）与方差（lambda→1）
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  （单步TD误差）
        # A_t = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...
        values_ext = values + [next_value]
        n = len(rewards)
        advantages = np.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        # returns = advantages + values，作为Critic的学习目标
        returns_t = advantages_t + torch.FloatTensor(values).to(self.device)

        # 优势标准化：减小不同episode间奖励尺度差异的影响
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ===== PPO多轮更新 =====
        # 同一批数据重复使用epochs轮，每轮随机打乱后分mini-batch训练
        for _ in range(self.epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                # 用当前策略重新评估旧动作
                new_log_probs, entropy, new_values = self.evaluate(batch_states, batch_actions)

                # ===== Clipped Surrogate Objective =====
                # ratio = pi_new(a|s) / pi_old(a|s)，衡量策略变化程度
                # clip限制ratio在[1-eps, 1+eps]，防止单次更新太大
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                # 取min确保：优势为正时不过度增加概率，优势为负时不过度减少概率
                # 减去entropy项（负号变加号）鼓励策略保持一定探索性
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Critic损失：预测的V(s)逼近实际回报returns
                critic_loss = nn.MSELoss()(new_values, batch_returns)

                # 更新Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 更新Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

    def save(self, path):
        """保存模型参数（Actor + Critic）"""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        """加载模型参数"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
