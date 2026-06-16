"""Teacher-Student RMA network architectures for G1 locomotion.

AdaptationModule: obs_history -> latent z (implicit environment parameters)
StudentPolicy: (obs_current, z) -> action
StudentONNXWrapper: fused module for ONNX export
"""

import torch
import torch.nn as nn


class AdaptationModule(nn.Module):
    """Encodes observation history into a latent representation of environment parameters.

    Input: obs_history of shape (batch, history_length * obs_dim)
    Output: latent z of shape (batch, latent_dim)
    """

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.net(obs_history)


class StudentPolicy(nn.Module):
    """Student policy that takes current obs + latent z and outputs actions.

    Input: obs_current (batch, obs_dim) + latent z (batch, latent_dim)
    Output: action (batch, action_dim)
    """

    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs_current: torch.Tensor, latent_z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_current, latent_z], dim=-1)
        return self.net(x)


class StudentONNXWrapper(nn.Module):
    """Fused AdaptationModule + StudentPolicy for ONNX export.

    Single forward pass: (obs_history, obs_current) -> action
    """

    def __init__(self, adaptation: AdaptationModule, policy: StudentPolicy):
        super().__init__()
        self.adaptation = adaptation
        self.policy = policy

    def forward(self, obs_history: torch.Tensor, obs_current: torch.Tensor) -> torch.Tensor:
        latent_z = self.adaptation(obs_history)
        action = self.policy(obs_current, latent_z)
        return action
