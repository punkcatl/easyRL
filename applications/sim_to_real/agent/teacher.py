import numpy as np
import torch
from .ppo_continuous import PPOContinuous, RunningMeanStd


class TeacherAgent:
    """Teacher policy: PPO with proprioception + privileged information."""

    def __init__(self, obs_dim: int, privileged_dim: int, action_dim: int, config: dict):
        self.obs_dim = obs_dim
        self.privileged_dim = privileged_dim
        total_input_dim = obs_dim + privileged_dim

        self.ppo = PPOContinuous(total_input_dim, action_dim, config)

    def get_input(self, obs: np.ndarray, privileged: np.ndarray) -> np.ndarray:
        """Concatenate proprioception and privileged info."""
        if obs.ndim == 1:
            return np.concatenate([obs, privileged])
        return np.concatenate([obs, privileged], axis=-1)

    def act(self, obs: np.ndarray, privileged: np.ndarray):
        """Get action from Teacher. Returns (actions, log_probs, values)."""
        full_input = self.get_input(obs, privileged)
        full_input_norm = self.ppo.normalize_obs(full_input)
        return self.ppo.act(full_input_norm)

    def update(self, obs, privileged, actions, log_probs, advantages, returns):
        """PPO update with pre-computed advantages and returns.

        Args:
            obs: (N, obs_dim) flattened observations
            privileged: (N, privileged_dim) flattened privileged info
            actions: (N, action_dim) flattened actions
            log_probs: (N,) flattened log probs
            advantages: (N,) pre-computed GAE advantages
            returns: (N,) pre-computed returns
        """
        full_input = self.get_input(obs, privileged)
        # Use read-only normalization: RMS was already updated during rollout via act()
        full_input_norm = self.ppo.obs_rms.normalize(full_input)
        self.ppo.update(
            full_input_norm, actions, log_probs,
            advantages, returns,
        )

    def get_action_deterministic(self, obs: np.ndarray, privileged: np.ndarray) -> np.ndarray:
        """Deterministic action (mean) for distillation data collection.

        Does NOT update running stats -- inference only.
        """
        full_input = self.get_input(obs, privileged)
        full_input_norm = self.ppo.obs_rms.normalize(full_input)
        obs_t = torch.FloatTensor(full_input_norm).to(self.ppo.device)
        with torch.no_grad():
            mean, _ = self.ppo.actor(obs_t)
        return mean.cpu().numpy()

    def save(self, path):
        self.ppo.save(path)

    def load(self, path):
        self.ppo.load(path)
