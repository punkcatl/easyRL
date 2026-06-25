import torch
from mjlab.envs import ManagerBasedRlEnv


def phase(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
    """Bipedal gait phase clock: [sin(2*pi*t/period), cos(2*pi*t/period)]. Shape [B, 2]."""
    t = env.episode_length_buf * env.step_dt
    angle = 2 * torch.pi * t / period
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
