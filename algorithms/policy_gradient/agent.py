import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Policy network that outputs action probabilities."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

    def select_action(self, state: np.ndarray) -> int:
        """Select action by sampling from the policy distribution."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state_t)
        dist = Categorical(probs)
        action = dist.sample()
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
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss (vectorized)
        log_probs_t = torch.stack(self.log_probs)
        loss = -(log_probs_t * returns).sum()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Clear episode data
        self.log_probs = []
        self.rewards = []

        return loss.item()

    def save(self, path: str):
        """Save policy network state dict."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        """Load policy network state dict."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
