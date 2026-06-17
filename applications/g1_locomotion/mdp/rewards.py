"""Custom reward terms for G1 humanoid locomotion.

Includes rewards adapted from unitree_rl_lab (feet_clearance, feet_gait, energy)
plus our own additions (gait_symmetry, base_height_reward).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing a target height off the ground.

    From unitree_rl_lab. Combines foot height error with foot velocity:
    only penalizes low clearance when the foot is moving (swinging phase).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward feet for following a periodic gait schedule.

    From unitree_rl_lab. Uses a clock-driven phase to determine when each foot
    should be in stance vs swing, rewards agreement with actual contact state.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize energy consumption (|torque| * |velocity| per joint).

    From unitree_rl_lab.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def bipedal_gait_symmetry(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward symmetric alternating contact between left and right feet."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]
    left_contact = torch.norm(net_forces[:, 0, :], dim=-1) > 1.0
    right_contact = torch.norm(net_forces[:, 1, :], dim=-1) > 1.0
    symmetry = (left_contact ^ right_contact).float()
    return symmetry


def base_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining target base height using exp kernel."""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    error = (base_height - target_height) ** 2
    return torch.exp(-error / (sigma**2))


def feet_symmetry_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize asymmetric foot heights — left and right should lift similarly."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    left_h = foot_heights[:, 0]
    right_h = foot_heights[:, 1]
    asymmetry = torch.square(left_h - right_h)
    return asymmetry


def forward_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Raw forward velocity reward — faster is always better."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0]
