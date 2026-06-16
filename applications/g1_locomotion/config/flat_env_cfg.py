"""G1 flat-terrain locomotion environment configuration.

Round 7: Stronger gait schedule + smoother actions for natural walking.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)

from applications.g1_locomotion.mdp import rewards as custom_mdp


@configclass
class G1FlatLocomotionEnvCfg(G1FlatEnvCfg):
    """Round 3: Round 1 baseline + feet_clearance."""

    def __post_init__(self):
        super().__post_init__()

        # --- Scene ---
        self.scene.num_envs = 1024

        # --- Commands (same as Round 1) ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # --- Rewards: Round 1 baseline ---
        # Velocity tracking (stronger to drive larger strides)
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.4
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Gait quality (Round 1 values)
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_slide.weight = -0.1

        # Stability (Round 4: stronger to prevent crouching/leaning)
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.lin_vel_z_l2.weight = -0.5

        # Smoothness (slightly stronger for natural gait)
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_torques_l2.weight = -2.0e-6

        # Joint deviation (hip stronger to fix splayed legs)
        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_hip.weight = -0.5
        self.rewards.joint_deviation_torso.weight = -0.1

        # Limits and termination (Round 1 values)
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.termination_penalty.weight = -200.0

        # === Round 3: feet_clearance to fix shuffle ===
        self.rewards.feet_clearance = RewTerm(
            func=custom_mdp.foot_clearance_reward,
            weight=1.0,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.1,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )

        # === Round 4: base_height to prevent crouching ===
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-5.0,
            params={"target_height": 0.74},
        )

        # === Round 5+7: gait schedule (stronger weight for regular stride) ===
        self.rewards.gait_schedule = RewTerm(
            func=custom_mdp.feet_gait,
            weight=1.0,
            params={
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            },
        )


@configclass
class G1FlatLocomotionEnvCfg_PLAY(G1FlatLocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
