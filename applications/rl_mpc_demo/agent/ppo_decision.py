import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPODecisionAgent:
    """Discrete PPO agent for high-level driving decisions."""

    def __init__(self, state_dim: int, action_dim: int = 5, hidden_dim: int = 128,
                 lr: float = 1e-3, gamma: float = 0.98, lmbda: float = 0.95,
                 eps: float = 0.2, epochs: int = 10):
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def take_action(self, state: np.ndarray) -> int:
        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def update(self, transition_dict: dict):
        states = torch.tensor(
            np.array(transition_dict['states']), dtype=torch.float
        ).to(self.device)
        actions = torch.tensor(
            transition_dict['actions']
        ).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            transition_dict['rewards'], dtype=torch.float
        ).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            np.array(transition_dict['next_states']), dtype=torch.float
        ).to(self.device)
        # Use terminations (not dones) so truncated episodes preserve the bootstrap value
        terminations = torch.tensor(
            transition_dict.get('terminations', transition_dict['dones']), dtype=torch.float
        ).view(-1, 1).to(self.device)

        # TD targets and advantages — mask zeros out bootstrap only on natural termination
        td_target = rewards + self.gamma * self.critic(next_states).detach() * (1 - terminations)
        td_delta = (td_target - self.critic(states)).detach()
        advantage = self._compute_advantage(td_delta)
        old_log_probs = torch.log(
            self.actor(states).gather(1, actions) + 1e-8
        ).detach()

        for _ in range(self.epochs):
            probs = self.actor(states)
            log_probs = torch.log(probs.gather(1, actions) + 1e-8)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            dist = torch.distributions.Categorical(probs)
            entropy = dist.entropy().mean()
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

    def _compute_advantage(self, td_delta: torch.Tensor) -> torch.Tensor:
        td_delta_np = td_delta.cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta_np[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(
            np.array(advantage_list), dtype=torch.float
        ).to(self.device)

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
