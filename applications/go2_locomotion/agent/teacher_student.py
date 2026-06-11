"""RMA (Rapid Motor Adaptation) student distillation components."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class AdaptationModule(nn.Module):
    """obs history -> latent z (implicit system identification).

    Replaces privileged info at deployment time by inferring environment
    parameters from the robot's own response history.
    """

    def __init__(self, obs_dim: int, history_length: int, latent_dim: int):
        super().__init__()
        input_dim = obs_dim * history_length
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, obs_history_flat: torch.Tensor) -> torch.Tensor:
        """obs_history_flat: (batch, obs_dim * history_length)"""
        return self.net(obs_history_flat)


class StudentPolicy(nn.Module):
    """current obs + latent z -> deterministic action."""

    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, z], dim=-1))


class StudentAgent:
    """Full student: AdaptationModule + StudentPolicy, trained via BC from teacher."""

    def __init__(self, config: dict):
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.history_length = config["student_history_length"]
        self.latent_dim = config["student_latent_dim"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adaptation = AdaptationModule(
            self.obs_dim, self.history_length, self.latent_dim
        ).to(self.device)
        self.policy = StudentPolicy(
            self.obs_dim, self.latent_dim, self.action_dim
        ).to(self.device)

        params = list(self.adaptation.parameters()) + list(self.policy.parameters())
        self.optimizer = optim.Adam(params, lr=config["student_lr"])

    def get_action(self, obs_history_flat: np.ndarray, obs_current: np.ndarray) -> np.ndarray:
        """Inference: obs_history -> z -> action (numpy in, numpy out)."""
        with torch.no_grad():
            hist_t = torch.FloatTensor(obs_history_flat).unsqueeze(0).to(self.device)
            obs_t = torch.FloatTensor(obs_current).unsqueeze(0).to(self.device)
            z = self.adaptation(hist_t)
            action = self.policy(obs_t, z)
        return action.cpu().numpy().flatten()

    def train_step(self, obs_history_batch: np.ndarray,
                   obs_current_batch: np.ndarray,
                   teacher_actions_batch: np.ndarray) -> float:
        """One BC training step. Returns loss value."""
        hist_t = torch.FloatTensor(obs_history_batch).to(self.device)
        obs_t = torch.FloatTensor(obs_current_batch).to(self.device)
        target_t = torch.FloatTensor(teacher_actions_batch).to(self.device)

        z = self.adaptation(hist_t)
        pred = self.policy(obs_t, z)
        loss = nn.MSELoss()(pred, target_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path: str):
        torch.save({
            "adaptation": self.adaptation.state_dict(),
            "policy": self.policy.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.adaptation.load_state_dict(ckpt["adaptation"])
        self.policy.load_state_dict(ckpt["policy"])
