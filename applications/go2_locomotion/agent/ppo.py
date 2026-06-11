import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import AsymmetricActorCritic


class PPOTrainer:
    """PPO with GAE for Go2 locomotion (asymmetric actor-critic)."""

    def __init__(self, config):
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_eps = config["clip_eps"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.max_grad_norm = config["max_grad_norm"]
        self.entropy_coef = config["entropy_coef"]
        self.value_loss_coef = config["value_loss_coef"]
        self.privileged_dim = config["privileged_dim"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = AsymmetricActorCritic(
            obs_dim=config["obs_dim"],
            privileged_dim=config["privileged_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"],
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config["lr"])

    def act(self, obs: np.ndarray, privileged: np.ndarray):
        """Sample actions for vectorized envs. Returns numpy arrays."""
        obs_t = torch.FloatTensor(obs).to(self.device)
        priv_t = torch.FloatTensor(privileged).to(self.device)
        with torch.no_grad():
            actions, log_probs, values, _ = self.network.act(obs_t, priv_t)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute GAE advantages and returns for a single env's trajectory."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        values_ext = np.append(values, next_value)

        for t in reversed(range(n)):
            delta = (rewards[t]
                     + self.gamma * values_ext[t + 1] * (1 - float(dones[t]))
                     - values_ext[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - float(dones[t])) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, rewards, dones, log_probs, values, next_value,
               privileged=None):
        """PPO update. Computes GAE internally if called with flat arrays."""
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        old_lp_t = torch.FloatTensor(log_probs).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)

        if privileged is None:
            privileged = np.zeros((len(states), self.privileged_dim), dtype=np.float32)
        priv_t = torch.FloatTensor(privileged).to(self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(states)
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                b = idx[start:end]

                new_lp, new_val, entropy = self.network.evaluate(
                    states_t[b], priv_t[b], actions_t[b]
                )
                ratio = torch.exp(new_lp - old_lp_t[b])
                s1 = ratio * adv_t[b]
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[b]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = nn.MSELoss()(new_val, ret_t[b])
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
