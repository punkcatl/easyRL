import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Standard Actor-Critic for continuous locomotion control."""

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        features = self.actor(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def act(self, obs):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return actions, log_probs, value, entropy

    def evaluate(self, obs, actions):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_probs, value, entropy


class AsymmetricActorCritic(nn.Module):
    """Actor uses obs only; Critic uses obs + privileged info (teacher mode)."""

    def __init__(self, obs_dim, privileged_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim + privileged_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor(self, obs):
        features = self.actor_net(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        return mean, std

    def forward_critic(self, obs, privileged):
        full = torch.cat([obs, privileged], dim=-1)
        return self.critic_net(full).squeeze(-1)

    def act(self, obs, privileged):
        mean, std = self.forward_actor(obs)
        dist = Normal(mean, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.forward_critic(obs, privileged)
        entropy = dist.entropy().sum(dim=-1).mean()
        return actions, log_probs, values, entropy

    def evaluate(self, obs, privileged, actions):
        mean, std = self.forward_actor(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.forward_critic(obs, privileged)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_probs, values, entropy
