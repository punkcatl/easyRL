"""Domain randomization events with 3-phase curriculum."""
from mjlab.envs.mdp import dr, events as mdp_events
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

_FOOT_GEOMS = ("FR_foot_collision", "FL_foot_collision", "RR_foot_collision", "RL_foot_collision")

DR_PHASE1_END = 500
DR_PHASE2_END = 1500


def make_dr_events_phase1() -> dict[str, EventTermCfg]:
    """No DR — only reset events."""
    return {}


def make_dr_events_phase2() -> dict[str, EventTermCfg]:
    """Light DR: conservative randomization ranges."""
    return {
        "randomize_friction_light": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.8, 1.1),
                "shared_random": True,
            },
        ),
        "randomize_mass_light": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "operation": "scale",
                "ranges": (0.95, 1.05),
            },
        ),
        "push_robot_light": EventTermCfg(
            mode="interval",
            func=mdp_events.push_by_setting_velocity,
            interval_range_s=(7.0, 10.0),
            params={
                "velocity_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                },
            },
        ),
    }


def make_dr_events_phase3() -> dict[str, EventTermCfg]:
    """Full DR: aggressive randomization."""
    return {
        "randomize_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.5, 1.25),
                "shared_random": True,
            },
        ),
        "randomize_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "operation": "scale",
                "ranges": (0.8, 1.2),
            },
        ),
        "randomize_pd_gains": EventTermCfg(
            mode="startup",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_range": (0.8, 1.2),
                "damping_range": (0.8, 1.2),
                "operation": "scale",
            },
        ),
        "push_robot": EventTermCfg(
            mode="interval",
            func=mdp_events.push_by_setting_velocity,
            interval_range_s=(5.0, 10.0),
            params={
                "velocity_range": {
                    "x": (-0.8, 0.8),
                    "y": (-0.8, 0.8),
                },
            },
        ),
        "randomize_motor_strength": EventTermCfg(
            mode="startup",
            func=dr.effort_limits,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "ranges": (0.9, 1.1),
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)),
            },
        ),
    }
