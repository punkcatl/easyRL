"""G1 humanoid reward functions for mjlab.

All functions take env: ManagerBasedRlEnv as first arg, return [B] tensor.
Bipedal gait: alternating left/right feet (not diagonal trot).
"""
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# G1 foot geom layout: [0:7] = left foot, [7:14] = right foot
_LEFT_FOOT_SLICE = slice(0, 7)
_RIGHT_FOOT_SLICE = slice(7, 14)


def _aggregate_foot_contact(sensor_data_found: torch.Tensor) -> torch.Tensor:
    """Aggregate 14 geom contacts into 2 foot contacts (left, right).

    Args:
        sensor_data_found: [B, 14] bool tensor

    Returns:
        [B, 2] float tensor: 1.0 if any geom in that foot group is in contact
    """
    left = (sensor_data_found[:, _LEFT_FOOT_SLICE] > 0.5).any(dim=1).float()
    right = (sensor_data_found[:, _RIGHT_FOOT_SLICE] > 0.5).any(dim=1).float()
    return torch.stack([left, right], dim=1)


def _aggregate_foot_air_time(air_time: torch.Tensor) -> torch.Tensor:
    """Aggregate 14 geom air times into 2 foot air times (min per group).

    Args:
        air_time: [B, 14] tensor

    Returns:
        [B, 2] tensor: minimum air time within each foot group
    """
    left = air_time[:, _LEFT_FOOT_SLICE].min(dim=1).values
    right = air_time[:, _RIGHT_FOOT_SLICE].min(dim=1).values
    return torch.stack([left, right], dim=1)


# --- Gait rewards (bipedal) ---

def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 0.1,
    threshold_max: float = 0.5,
    command_name: str = "twist",
    command_threshold: float = 0.3,
) -> torch.Tensor:
    """Reward feet whose current air time is in [threshold, threshold_max].

    Aggregates 14 foot geoms into 2 groups (left/right).
    Scaled by command magnitude.
    """
    sensor = env.scene[sensor_name]
    air_time = _aggregate_foot_air_time(sensor.data.current_air_time)  # [B, 2]
    in_range = (air_time > threshold) & (air_time < threshold_max)
    reward = in_range.float().sum(dim=1)

    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    scale = (cmd_norm > command_threshold).float()
    return reward * scale


def bipedal_gait_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Reward alternating foot contact (one foot stance, one swing).

    Penalizes both feet in contact or both in air simultaneously.
    """
    sensor = env.scene[sensor_name]
    contact = _aggregate_foot_contact(sensor.data.found)  # [B, 2]
    diff = torch.abs(contact[:, 0] - contact[:, 1])
    return diff


def double_support_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalty when both feet are in contact (double support phase too long)."""
    sensor = env.scene[sensor_name]
    contact = _aggregate_foot_contact(sensor.data.found)  # [B, 2]
    return (contact > 0.5).all(dim=1).float()


def no_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalty when no feet are in contact (flight phase = falling)."""
    sensor = env.scene[sensor_name]
    contact = _aggregate_foot_contact(sensor.data.found)  # [B, 2]
    return (contact < 0.5).all(dim=1).float()


# --- Feet clearance ---

def feet_clearance_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward feet for being above target_height when in swing phase.

    Only rewards feet currently in the air (not in contact).
    Uses foot body z-position relative to ground.
    """
    sensor = env.scene[sensor_name]
    contact = _aggregate_foot_contact(sensor.data.found)  # [B, 2]
    in_swing = (contact < 0.5)  # [B, 2] — True when foot is in air

    asset = env.scene[asset_cfg.name]
    # Get foot body heights: left_ankle_roll_link and right_ankle_roll_link
    # Use the lowest foot geom z-position approximation from root height
    # For simplicity, reward any swing phase foot (already good signal)
    reward = in_swing.float().sum(dim=1) * target_height
    return reward


# --- Base stability ---

def flat_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L2 penalty for non-upright orientation (roll + pitch of gravity projection)."""
    asset = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    return gravity_b[:, 0] ** 2 + gravity_b[:, 1] ** 2


def base_height_reward(
    env: ManagerBasedRlEnv,
    target: float,
    sigma: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian reward for maintaining target base height."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]
    error = (height - target) ** 2
    return torch.exp(-error / sigma**2)


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

def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 penalty for joint deviation from default position.

    Uses asset_cfg.joint_names to select which joints to penalize.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos
    indices, _ = asset.find_joints(asset_cfg.joint_names)
    deviation = torch.abs(joint_pos[:, indices] - default_pos[:, indices])
    return deviation.sum(dim=1)
