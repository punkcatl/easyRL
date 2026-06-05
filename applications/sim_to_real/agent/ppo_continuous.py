import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        batch = np.asarray(batch)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class GaussianActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, x):
        mean, std = self.forward(x)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PPOContinuous:
    """PPO for continuous actions with obs normalization and reward scaling."""

    def __init__(self, obs_dim, action_dim, config):
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_eps = config["clip_eps"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.max_grad_norm = config["max_grad_norm"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_dim = config["hidden_dim"]

        self.actor = GaussianActor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["lr"])

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.reward_rms = RunningMeanStd(shape=(1,))

    def normalize_obs(self, obs):
        self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def scale_reward(self, rewards):
        self.reward_rms.update(rewards.reshape(-1, 1))
        return rewards / (np.sqrt(self.reward_rms.var[0]) + 1e-8)

    def act(self, obs_normalized):
        obs_t = torch.FloatTensor(obs_normalized).to(self.device)
        with torch.no_grad():
            dist = self.actor.get_dist(obs_t)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
            values = self.critic(obs_t).squeeze(-1)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def compute_gae_single(self, rewards, values, dones, next_value):
        """Compute GAE for a single environment trajectory.

        Args:
            rewards: (n_steps,) rewards for one env
            values: (n_steps,) value estimates for one env
            dones: (n_steps,) done flags for one env
            next_value: scalar bootstrap value for the last state

        Returns:
            advantages: (n_steps,)
            returns: (n_steps,)
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages.astype(np.float32), returns.astype(np.float32)

    def update(self, obs, actions, log_probs, advantages, returns):
        """PPO update with pre-computed advantages and returns.

        Args:
            obs: (N, obs_dim) flattened observations
            actions: (N, action_dim) flattened actions
            log_probs: (N,) flattened log probs
            advantages: (N,) pre-computed GAE advantages
            returns: (N,) pre-computed returns
        """

        # To tensors
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = len(obs)
        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_obs = obs_t[idx]
                batch_actions = actions_t[idx]
                batch_old_lp = old_log_probs_t[idx]
                batch_adv = advantages_t[idx]
                batch_ret = returns_t[idx]

                # Actor loss
                dist = self.actor.get_dist(batch_obs)
                new_lp = dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(new_lp - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                ) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                new_values = self.critic(batch_obs).squeeze(-1)
                critic_loss = nn.MSELoss()(new_values, batch_ret)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "obs_rms_mean": self.obs_rms.mean,
                "obs_rms_var": self.obs_rms.var,
                "obs_rms_count": self.obs_rms.count,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.obs_rms.mean = ckpt["obs_rms_mean"]
        self.obs_rms.var = ckpt["obs_rms_var"]
        self.obs_rms.count = ckpt["obs_rms_count"]
