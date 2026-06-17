"""G1 rough-terrain locomotion environment configuration.

Round 10: Rough terrain + DR (building on Round 9c flat DR success).
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as isaac_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)

from applications.g1_locomotion.mdp import rewards as custom_mdp


@configclass
class G1RoughLocomotionEnvCfg(G1RoughEnvCfg):
    """Round 10: Rough terrain with proven rewards from R8b + moderate DR from R9c."""

    def __post_init__(self):
        super().__post_init__()

        # --- Scene ---
        self.scene.num_envs = 1024

        # --- Commands ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # --- Rewards (proven from R8b flat training) ---
        # Velocity tracking
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.4
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Gait quality
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_slide.weight = -0.2

        # Stability
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.lin_vel_z_l2.weight = -0.5

        # Smoothness (relaxed for rough terrain — needs more dynamic movement)
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_torques_l2.weight = -2.0e-6

        # Joint deviation (from R8b)
        self.rewards.joint_deviation_arms.weight = -0.2
        self.rewards.joint_deviation_hip.weight = -0.5
        self.rewards.joint_deviation_torso.weight = -0.1
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.termination_penalty.weight = -200.0

        # Custom rewards
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

        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-2.0,
            params={"target_height": 0.68},
        )

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

        # Symmetry: penalize left/right foot height asymmetry (fix limping)
        self.rewards.feet_symmetry = RewTerm(
            func=custom_mdp.feet_symmetry_height,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )

        # --- Domain Randomization (moderate, from R9c) ---
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.8)

        self.events.push_robot = EventTerm(
            func=isaac_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

        self.events.base_external_force_torque = EventTerm(
            func=isaac_mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "force_range": (-20.0, 20.0),
                "torque_range": (-2.0, 2.0),
            },
        )

        self.events.add_base_mass = EventTerm(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_distribution_params": (-2.0, 2.0),
                "operation": "add",
            },
        )

        self.events.randomize_actuator = EventTerm(
            func=isaac_mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
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
        self.events.randomize_actuator = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
