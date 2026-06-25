"""Student policy: history encoder + action head for G1 (15 DOF)."""
import torch
import torch.nn as nn


class StudentPolicy(nn.Module):
    """Student network: encodes obs history into latent, then predicts action.

    Architecture:
        obs_history (history_length, obs_dim) -> flatten -> MLP encoder -> latent (latent_dim)
        [latent, current_obs] -> MLP policy -> action (action_dim)
    """

    def __init__(
        self,
        obs_dim: int = 87,
        action_dim: int = 12,
        history_length: int = 20,
        latent_dim: int = 32,
        encoder_hidden: int = 256,
        policy_hidden: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.latent_dim = latent_dim

        encoder_input_dim = history_length * obs_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, encoder_hidden),
            nn.ELU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.ELU(),
            nn.Linear(encoder_hidden, latent_dim),
        )

        policy_input_dim = latent_dim + obs_dim
        self.policy = nn.Sequential(
            nn.Linear(policy_input_dim, policy_hidden),
            nn.ELU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.ELU(),
            nn.Linear(policy_hidden, action_dim),
        )

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs_history: [B, history_length, obs_dim] tensor

        Returns:
            actions: [B, action_dim] tensor in [-1, 1]
        """
        batch_size = obs_history.shape[0]
        current_obs = obs_history[:, -1, :]

        flat_history = obs_history.reshape(batch_size, -1)
        latent = self.encoder(flat_history)

        policy_input = torch.cat([latent, current_obs], dim=1)
        actions = self.policy(policy_input)
        return actions

    def get_latent(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Get latent encoding only (for analysis)."""
        batch_size = obs_history.shape[0]
        flat_history = obs_history.reshape(batch_size, -1)
        return self.encoder(flat_history)
