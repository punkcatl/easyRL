import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class AdaptationModule(nn.Module):
    """RMA Adaptation Module: history of observations -> latent z.

    Implicitly performs system identification by observing how the robot
    responds to actions over time.
    """

    def __init__(self, obs_dim: int, history_length: int, latent_dim: int):
        super().__init__()
        input_dim = obs_dim * history_length
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, obs_history):
        """obs_history: (batch, history_length * obs_dim) flattened."""
        return self.net(obs_history)


class BasePolicy(nn.Module):
    """Student base policy: current obs + latent z -> action."""

    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, z):
        x = torch.cat([obs, z], dim=-1)
        return self.net(x)


class StudentAgent:
    """Student with RMA: Adaptation Module + Base Policy, trained via BC."""

    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_length = config["history_length"]
        self.latent_dim = config["latent_dim"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.adaptation = AdaptationModule(
            obs_dim, self.history_length, self.latent_dim
        ).to(self.device)

        self.base_policy = BasePolicy(
            obs_dim, self.latent_dim, action_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.adaptation.parameters()) + list(self.base_policy.parameters()),
            lr=config["student_lr"],
        )

        self._history_buffer = None

    def reset_history(self):
        """Reset history buffer for a new episode."""
        self._history_buffer = np.zeros(
            (self.history_length, self.obs_dim), dtype=np.float32
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Get deterministic action. Maintains internal history buffer."""
        if self._history_buffer is None:
            self.reset_history()

        # Shift history and append current obs
        self._history_buffer = np.roll(self._history_buffer, -1, axis=0)
        self._history_buffer[-1] = obs

        history_flat = self._history_buffer.flatten()

        with torch.no_grad():
            history_t = torch.FloatTensor(history_flat).unsqueeze(0).to(self.device)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            z = self.adaptation(history_t)
            action = self.base_policy(obs_t, z)

        return action.cpu().numpy().flatten()

    def train_step(self, obs_history_batch, obs_current_batch, action_teacher_batch):
        """One training step of BC distillation.

        Args:
            obs_history_batch: (batch, history_length * obs_dim)
            obs_current_batch: (batch, obs_dim)
            action_teacher_batch: (batch, action_dim)

        Returns:
            loss value (float)
        """
        history_t = torch.FloatTensor(obs_history_batch).to(self.device)
        obs_t = torch.FloatTensor(obs_current_batch).to(self.device)
        target_t = torch.FloatTensor(action_teacher_batch).to(self.device)

        z = self.adaptation(history_t)
        action_pred = self.base_policy(obs_t, z)
        loss = nn.MSELoss()(action_pred, target_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        torch.save(
            {
                "adaptation": self.adaptation.state_dict(),
                "base_policy": self.base_policy.state_dict(),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.adaptation.load_state_dict(ckpt["adaptation"])
        self.base_policy.load_state_dict(ckpt["base_policy"])
