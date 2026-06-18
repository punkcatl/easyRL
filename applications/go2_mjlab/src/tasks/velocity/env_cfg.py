"""Go2 flat terrain velocity tracking environment configuration.

Round 4: Simplified reward following mjlab's proven Go1 pattern.
Key insight: velocity tracking (weight=2.0) must dominate over all penalties.
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

from src.robots import get_go2_robot_cfg
from src.tasks.velocity.mdp import rewards as local_rewards
from src.tasks.velocity.mdp.observations import phase as phase_obs

from mjlab.tasks.velocity import mdp


_FOOT_GEOM_NAMES = ("FL_foot_collision", "FR_foot_collision", "RL_foot_collision", "RR_foot_collision")


def make_go2_flat_env_cfg(num_envs: int = 1024, play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = get_go2_robot_cfg()

    feet_contact_sensor = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=_FOOT_GEOM_NAMES, entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Actor obs — using builtin sensors from MJCF
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
        "phase": ObservationTermCfg(func=phase_obs, params={"period": 0.5}),
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

    # Critic = actor + privileged
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

    # Rewards — simplified, following mjlab Go1 proven pattern
    # Velocity tracking dominates, minimal penalties
    rewards = {
        # Primary: velocity tracking (weight 2.0 each — same as Go1)
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        # Stability
        "upright": RewardTermCfg(
            func=local_rewards.flat_orientation,
            weight=-1.0,
        ),
        "base_height": RewardTermCfg(
            func=local_rewards.base_height_reward,
            weight=0.5,
            params={"target": 0.34, "sigma": 0.05},
        ),
        # Penalties (light — don't overwhelm tracking signal)
        "termination_penalty": RewardTermCfg(
            func=mdp.is_terminated,
            weight=-10.0,
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "lin_vel_z": RewardTermCfg(
            func=local_rewards.lin_vel_z_l2,
            weight=-0.2,
        ),
    }

    # Terminations — minimal, like Go1 (only time_out + fell_over)
    # No height termination: let RL explore even if it falls
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(60.0)},
        ),
    }

    # Events
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "yaw": (-3.14, 3.14),
                },
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
        # push_robot disabled for initial training — enable after robot learns to walk
        # "push_robot": EventTermCfg(
        #     func=mdp.push_by_setting_velocity,
        #     mode="interval",
        #     interval_range_s=(5.0, 10.0),
        #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        # ),
    }

    # Commands — wider range than before to give more signal
    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            heading_command=False,
            rel_standing_envs=0.1,
            resampling_time_range=(3.0, 8.0),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.5, 0.5),
            ),
        ),
    }

    # Actions — use per-joint scale like Go1
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.4,
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
            body_name="base_link",
            distance=3.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
