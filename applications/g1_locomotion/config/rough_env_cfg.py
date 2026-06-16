"""G1 rough-terrain locomotion environment configuration."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)

from applications.g1_locomotion.mdp import rewards as custom_mdp


@configclass
class G1RoughLocomotionEnvCfg(G1RoughEnvCfg):
    """Custom G1 rough env — used after flat-terrain walking is stable."""

    def __post_init__(self):
        super().__post_init__()

        # --- Scene ---
        self.scene.num_envs = 1024

        # --- Commands: wider range for rough terrain ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # --- Rewards: same structure as flat but adjusted weights ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_slide.weight = -0.1

        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_torques_l2.weight = -1.5e-7

        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_hip.weight = -0.1
        self.rewards.joint_deviation_torso.weight = -0.1
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.termination_penalty.weight = -200.0

        # Custom rewards
        self.rewards.gait_symmetry = RewTerm(
            func=custom_mdp.bipedal_gait_symmetry,
            weight=0.3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.base_height = RewTerm(
            func=custom_mdp.base_height_reward,
            weight=0.3,
            params={"target_height": 0.74, "sigma": 0.08},
        )


@configclass
class G1RoughLocomotionEnvCfg_PLAY(G1RoughLocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
