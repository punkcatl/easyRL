import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

'''策略网络'''
class PolicyNetwork(nn.Module):
    """Policy network that outputs action probabilities."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # 网络结构：
        # 输入层 state_dim
        # -> 全连接层 Linear(state_dim, hidden_dim) + ReLU
        # -> 全连接层 Linear(hidden_dim, hidden_dim) + ReLU
        # -> 输出层 Linear(hidden_dim, action_dim)
        # -> Softmax(dim=-1)，将 action_dim 维的 logits 转为动作概率分布 pi(a|s)
        self.net = nn.Sequential( # nn.Sequential封装，更紧凑的 PyTorch 写法
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1), # 输出动作概率分布，dim=-1 表示沿最后一个维度（动作维度）进行 softmax 归一化
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

'''REINFORCE智能体'''
class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float,
        gamma: float,
        hidden_dim: int = 128,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.log_probs = []
        self.rewards = []

    def take_action(self, state: np.ndarray) -> int:
        """Select action by sampling from the policy distribution."""
        # unsqueeze(0) 是在第 0 维（最前面）插入一个大小为 1 的维度，相当于加 batch 维
        # unsqueeze(0) 比 [state] 包 list 更通用——对任意维度的输入都能正确加 batch 维
        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device) 
        probs = self.policy(state_t) # 输出动作概率分布，形状为 [1, action_dim]
        dist = Categorical(probs) # 创建离散概率分布对象
        action = dist.sample() # 从分布中采样动作，形状为 [1]
        
        # 和ch09 policy_gradient.py的区别： 这里会存log_prob, 等update时直接用。 ch09里是在update时重新计算log_prob。
        # dist.log_prob(action) — 计算所采样动作的对数概率 log π(a|s)。这是 REINFORCE 算法梯度公式 ∇J = E[log π(a|s) · G_t] 中的核心项。
        # .squeeze() — 去掉多余的维度，把形状从 [1] 压成标量 []，方便后续直接求和。
        # self.log_probs.append(...) — 存入列表，等一个 episode 结束后在 update() 里和对应的回报 G_t 相乘，计算策略梯度。
        self.log_probs.append(dist.log_prob(action).squeeze())
        
        return action.item()

    def store_reward(self, reward: float):
        """Store reward for the current timestep."""
        self.rewards.append(reward)

    def update(self) -> float:
        """Compute discounted returns and update the policy."""
        # Compute discounted returns
        returns = []
        G = 0
        # 从最后一步开始逆序计算每一步的回报 G_t = r_t + γ * G_{t+1}，这样可以避免重复计算，效率更高。最终得到整个episode的回报序列。
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G) # insert(0, G) 是在列表开头插入 G，保持 returns 与 rewards 的时间顺序一致。
            # 之所以要收集完整列表而不是像 ch09 那样逐步 backward，是为了后续做 normalize（减均值除标准差），降低方差使训练更稳定。

        # 转为 PyTorch 张量，并移动到 agent 的设备上（CPU 或 GPU）
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize returns（z-score 标准化）
        # 减均值（解决方向问题）：使 G_t 有正有负，高于均值的动作被鼓励，低于均值的被抑制，相当于简单 baseline。这是核心，只减均值就够用。
        # 除以 std（解决幅度问题）：不同环境 reward 尺度差异巨大（CartPole~500, Atari~5000），
        #   减均值后的数值可能相差百倍，导致同一个 lr 在某些环境梯度过大而炸掉。除以 std 后统一缩放到均值=0、标准差=1（减自己的均值→均值归零；除以自己的标准差→标准差归一），换环境不用重调 lr。
        # + 1e-8 防止 std=0 时除以零（如 episode 只有一步）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) # returns.std() — 所有 G_t 的标准差。
        # [tips] 方差：均值与所有值的差的平方的平均值。标准差：方差的平方根。

        # Compute policy loss (vectorized)
        # 1.对数概率和回报输入
        # log_probs_t = [log π(a₀|s₀), log π(a₁|s₁), ..., log π(aₜ|sₜ)]  # 形状 [T]
        # returns     = [G₀,           G₁,           ..., Gₜ]              # 形状 [T]
        # 2.逐元素相乘
        # log_probs_t * returns = [log π(a₀|s₀)·G₀, log π(a₁|s₁)·G₁, ..., log π(aₜ|sₜ)·Gₜ]
        # 3 .sum() 把 T 个值加起来得到一个标量
        # 取负号得到最终整个episode的loss总和
        
        # 将列表中的 log_prob 张量堆叠成一个张量，形状为 [episode_length]。 torch.stack()把多个相同形状的张量沿新维度拼接成一个张量
        log_probs_t = torch.stack(self.log_probs) 
        # loss = -Σₜ log π(aₜ|sₜ) · Gₜ，负号因为 PyTorch 最小化 loss 等价于最大化期望回报
        loss = -(log_probs_t * returns).sum()

        # Backprop
        # zero_grad 清的是梯度（方向指示），不是参数（网络学到的东西）。
        # 参数通过 step() 一直在累积更新，训练效果不会丢失；梯度每次要重新算，否则会和上次残留混在一起导致方向错误。
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪：若梯度总范数超过 0.5 则等比缩小，方向不变但步幅受限，防止单 episode 方差大时梯度爆炸。
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step() # 更新网络参数

        # Clear episode data
        self.log_probs = []
        self.rewards = []

        return loss.item() 

    def save(self, path: str):
        """Save policy network state dict."""
        torch.save(self.policy.state_dict(), path) # 只保存网络参数，方便后续加载和推理

    def load(self, path: str):
        """Load policy network state dict."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device)) # 加载网络参数
