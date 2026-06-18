"""Go2 reward, termination, and observation functions for mjlab.

All functions take env: ManagerBasedRlEnv as first arg, return [B] tensor.
Contact-based rewards use the "feet_ground_contact" sensor.
"""
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# --- Velocity tracking rewards ---

def track_lin_vel_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel = asset.data.root_link_lin_vel_b[:, :2]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    error = torch.sum((vel - cmd) ** 2, dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_z = asset.data.root_link_ang_vel_b[:, 2]
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]
    error = (ang_vel_z - cmd_yaw) ** 2
    return torch.exp(-error / std**2)


def forward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 0]


# --- Contact-based gait rewards ---

def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
) -> torch.Tensor:
    """Reward feet spending time in the air (above threshold). Triggered on landing."""
    sensor = env.scene[sensor_name]
    air_time = sensor.data.current_air_time  # [B, 4]
    contact = sensor.data.found      # [B, 4] bool — currently in contact
    first_contact = (contact > 0.5) & (air_time > 0)
    reward_per_foot = torch.clamp(air_time - threshold, min=0.0) * first_contact.float()
    return reward_per_foot.sum(dim=1)


class gait_schedule_reward:
    """Reward feet following a trot gait schedule using real contact data."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.step_dt = env.step_dt
        self.time = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        trot_period: float,
    ) -> torch.Tensor:
        self.time += self.step_dt
        sensor = env.scene[sensor_name]
        contact = sensor.data.found.float()  # [B, 4]: FL, FR, RL, RR
        phase = (self.time / trot_period) % 1.0

        # Trot: FL+RR in phase, FR+RL in anti-phase
        swing_mask = (phase < 0.5).float()

        # Desired: when swing_mask=1 → FL=swing(0), FR=stance(1), RL=stance(1), RR=swing(0)
        desired = torch.stack([
            1 - swing_mask,  # FL
            swing_mask,      # FR
            swing_mask,      # RL
            1 - swing_mask,  # RR
        ], dim=1)

        match = 1.0 - torch.abs(contact - desired)
        return match.mean(dim=1)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.time[env_ids] = 0.0


def gait_symmetry_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Reward diagonal pair symmetry (FL+RR same, FR+RL same, cross opposite)."""
    sensor = env.scene[sensor_name]
    contact = sensor.data.found.float()  # [B, 4]
    # Diagonal pairs should be same state
    pair1_same = 1.0 - torch.abs(contact[:, 0] - contact[:, 3])  # FL == RR
    pair2_same = 1.0 - torch.abs(contact[:, 1] - contact[:, 2])  # FR == RL
    # Cross pairs should be opposite
    cross_opp = torch.abs(contact[:, 0] - contact[:, 1])          # FL != FR
    return (pair1_same + pair2_same + cross_opp) / 3.0


def all_feet_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalty when all four feet are in contact (standing/lurching)."""
    sensor = env.scene[sensor_name]
    contact = sensor.data.found  # [B, 4] bool
    return (contact > 0.5).all(dim=1).float()


# --- Base stability rewards ---

def base_height_reward(
    env: ManagerBasedRlEnv,
    target: float,
    sigma: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]
    error = (height - target) ** 2
    return torch.exp(-error / sigma**2)


def flat_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    return gravity_b[:, 0] ** 2 + gravity_b[:, 1] ** 2


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty for vertical body velocity (bouncing)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 2] ** 2


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty for roll/pitch angular velocity."""
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_link_ang_vel_b[:, :2]
    return torch.sum(ang_vel**2, dim=1)


# --- Joint penalties ---

def torque_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty for applied torques."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(asset.data.actuator_force**2, dim=1)


def joint_acc_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty for joint accelerations."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(asset.data.joint_acc**2, dim=1)


# --- Termination functions ---

def terminate_height_low(
    env: ManagerBasedRlEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height
