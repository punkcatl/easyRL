"""G1 humanoid locomotion environment — R10 configuration.

Based on R1 (proven to walk stably), with minimal additions to fix:
- body lean-back (upright weight strengthened)
- lateral sway (ang_vel_xy + joint_deviation_hip added)
"""
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.robots import get_g1_robot_cfg
from src.robots.g1_cfg import ACTION_JOINT_NAMES, FOOT_GEOM_NAMES
from src.tasks.velocity.mdp import rewards as local_rewards
from src.tasks.velocity.mdp.observations import phase as phase_obs
from src.tasks.velocity.mdp.events import set_arm_hold_targets
from mjlab.envs.mdp import dr

from mjlab.tasks.velocity import mdp


def make_g1_flat_env_cfg(num_envs: int = 2048, play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = get_g1_robot_cfg()

    feet_contact_sensor = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Actor observations
    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "phase": ObservationTermCfg(func=phase_obs, params={"period": 0.6}),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    critic_terms = {
        **actor_terms,
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # === REWARDS: R1 base +姿态修复 ===
    rewards = {
        # --- Velocity tracking (R11 values) ---
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=1.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=1.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        # --- Gait (R1 values) ---
        "feet_air_time": RewardTermCfg(
            func=local_rewards.feet_air_time,
            weight=0.3,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold": 0.15,
                "threshold_max": 0.5,
                "command_name": "twist",
                "command_threshold": 0.3,
            },
        ),
        "bipedal_gait": RewardTermCfg(
            func=local_rewards.bipedal_gait_reward,
            weight=0.5,
            params={"sensor_name": "feet_ground_contact"},
        ),
        # --- Stability: STRENGTHENED vs R1 ---
        "upright": RewardTermCfg(
            func=local_rewards.flat_orientation,
            weight=-10.0,
        ),
        "base_height": RewardTermCfg(
            func=local_rewards.base_height_reward,
            weight=1.0,
            params={"target": 0.74, "sigma": 0.05},
        ),
        "lin_vel_z": RewardTermCfg(
            func=local_rewards.lin_vel_z_l2,
            weight=-1.0,
        ),
        "ang_vel_xy": RewardTermCfg(
            func=local_rewards.ang_vel_xy_l2,
            weight=-0.5,
        ),
        # --- Joint deviation: NEW vs R1 ---
        "joint_deviation_hip": RewardTermCfg(
            func=local_rewards.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_yaw_joint", ".*_hip_roll_joint"))},
        ),
        # --- Penalties (R1 values — termination KEEP at -20, NOT -200) ---
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "termination_penalty": RewardTermCfg(
            func=mdp.is_terminated,
            weight=-20.0,  # 保持 R1 的 -20，不用 -200
        ),
    }

    # Terminations
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(45.0)},
        ),
    }

    # Events (R1: light push only)
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "set_arm_targets": EventTermCfg(
            func=set_arm_hold_targets,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                )),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 10.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
        "randomize_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
                "operation": "abs",
                "ranges": (0.5, 1.3),
                "shared_random": True,
            },
        ),
        "randomize_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "operation": "scale",
                "ranges": (0.85, 1.15),
            },
        ),
        "randomize_pd_gains": EventTermCfg(
            mode="startup",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.85, 1.15),
                "kd_range": (0.85, 1.15),
                "operation": "scale",
            },
        ),
    }

    # Commands: R1 range
    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            heading_command=False,
            rel_standing_envs=0.05,
            resampling_time_range=(3.0, 8.0),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.3, 1.0),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.5, 0.5),
            ),
        ),
    }

    # Actions
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=ACTION_JOINT_NAMES,
            scale=0.25,
            use_default_offset=True,
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            entities={"robot": robot_cfg},
            sensors=(feet_contact_sensor,),
            terrain=TerrainEntityCfg(terrain_type="plane"),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        metrics={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="pelvis",
            distance=4.0,
        ),
        sim=SimulationCfg(
            njmax=200,
            nconmax=100,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
