"""Unitree Go2 robot configuration for mjlab.

Based on unitree_rl_mjlab's proven Go2 config.
"""
from pathlib import Path

import mujoco

from src import ASSETS_DIR
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

GO2_XML: Path = ASSETS_DIR / "go2" / "go2.xml"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(GO2_XML))


GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
    target_names_expr=(".*hip_.*",),
    stiffness=20.0,
    damping=1.0,
    effort_limit=23.5,
    armature=0.01,
)

GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*thigh_.*",),
    stiffness=20.0,
    damping=1.0,
    effort_limit=23.5,
    armature=0.01,
)

GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
    target_names_expr=(".*calf_.*",),
    stiffness=40.0,
    damping=2.0,
    effort_limit=45.0,
    armature=0.02,
)

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.32),
    joint_pos={
        ".*thigh_joint": 0.9,
        ".*calf_joint": -1.8,
        ".*R_hip_joint": 0.1,
        ".*L_hip_joint": -0.1,
    },
    joint_vel={".*": 0.0},
)

_foot_regex = ".*_foot_collision"

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_foot_regex: 3, ".*_collision": 1},
    priority={_foot_regex: 1},
    friction={_foot_regex: (0.6,)},
    solimp={_foot_regex: (0.9, 0.95, 0.023)},
    contype=1,
    conaffinity=1,
)

GO2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(GO2_ACTUATOR_HIP, GO2_ACTUATOR_THIGH, GO2_ACTUATOR_CALF),
    soft_joint_pos_limit_factor=0.9,
)

GO2_GAIT_PHASE_OFFSETS = [0.0, 0.5, 0.5, 0.0]


def get_go2_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=GO2_ARTICULATION,
    )
