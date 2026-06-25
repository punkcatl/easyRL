"""Custom events for G1 locomotion."""
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


_ARM_TARGETS = {
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.1,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.5,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.1,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.5,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    # Waist: hold at zero (upright)
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
}


def set_arm_hold_targets(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Set joint_pos_target for arm joints to natural hanging pose.

    Hardcoded targets because mjlab's INIT_STATE mapping has index issues.
    """
    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    for name, target_val in _ARM_TARGETS.items():
        joint_ids, _ = asset.find_joints((name,))
        if joint_ids:
            target_tensor = torch.full(
                (len(env_ids),), target_val, device=asset.data.joint_pos_target.device
            )
            asset.set_joint_position_target(target_tensor, joint_ids=joint_ids, env_ids=env_ids)
