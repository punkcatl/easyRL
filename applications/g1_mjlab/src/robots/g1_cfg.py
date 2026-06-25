"""Unitree G1 humanoid robot configuration for mjlab.

15 DOF locomotion: 12 legs + 3 waist.
Arms held at default pose via high-stiffness PD (not in action space).
"""
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_XML,
    FULL_COLLISION,
    KNEES_BENT_KEYFRAME,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    ACTUATOR_5020,
    ACTUATOR_7520_14,
    ACTUATOR_7520_22,
)

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(G1_XML))


# --- Actuators for the 15 DOF action space ---

# Hip pitch + hip yaw (7520-14 motor)
G1_ACTUATOR_HIP_PITCH_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    effort_limit=ACTUATOR_7520_14.effort_limit,
    armature=ARMATURE_7520_14,
)

# Hip roll + knee (7520-22 motor, strongest)
G1_ACTUATOR_HIP_ROLL_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
    stiffness=STIFFNESS_7520_22,
    damping=DAMPING_7520_22,
    effort_limit=ACTUATOR_7520_22.effort_limit,
    armature=ARMATURE_7520_22,
)

# Ankle pitch + ankle roll (dual 5020 linkage)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    stiffness=STIFFNESS_5020 * 2,
    damping=DAMPING_5020 * 2,
    effort_limit=ACTUATOR_5020.effort_limit * 2,
    armature=ARMATURE_5020 * 2,
)

# Waist (yaw uses 7520-14, pitch/roll use dual 5020 linkage)
G1_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    effort_limit=ACTUATOR_7520_14.effort_limit,
    armature=ARMATURE_7520_14,
)

G1_ACTUATOR_WAIST_PITCHROLL = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
    stiffness=STIFFNESS_5020 * 2,
    damping=DAMPING_5020 * 2,
    effort_limit=ACTUATOR_5020.effort_limit * 2,
    armature=ARMATURE_5020 * 2,
)

# Arms: high stiffness PD to hold at default (NOT in action space)
G1_ACTUATOR_ARMS_HOLD = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_yaw_joint",
    ),
    stiffness=STIFFNESS_5020 * 3,
    damping=DAMPING_5020 * 3,
    effort_limit=ACTUATOR_5020.effort_limit,
    armature=ARMATURE_5020,
)

# Articulation with all actuators (arms included for PD hold)
G1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        G1_ACTUATOR_HIP_PITCH_YAW,
        G1_ACTUATOR_HIP_ROLL_KNEE,
        G1_ACTUATOR_ANKLE,
        G1_ACTUATOR_WAIST_YAW,
        G1_ACTUATOR_WAIST_PITCHROLL,
        G1_ACTUATOR_ARMS_HOLD,
    ),
    soft_joint_pos_limit_factor=0.9,
)

# Collision: feet only for faster simulation
G1_FEET_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)

# Initial state: knees slightly bent, arms at natural pose
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.76),
    joint_pos={
        # Legs
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        # Arms: natural hanging pose
        # Note: mjlab swaps shoulder_yaw ↔ elbow in qpos mapping,
        # so we put elbow target (0.5) in shoulder_yaw slot
        ".*_shoulder_yaw_joint": 0.5,  # actual effect: elbow bends 0.5 rad
        ".*_elbow_joint": 0.0,         # actual effect: shoulder_yaw stays 0
        "left_shoulder_roll_joint": 0.1,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.1,
        "right_shoulder_pitch_joint": 0.2,
    },
    joint_vel={".*": 0.0},
)

# 12 joints controlled by the RL policy (legs only, waist PD-held)
ACTION_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

# Foot geom names for contact sensor
FOOT_GEOM_NAMES = (
    "left_foot1_collision",
    "left_foot2_collision",
    "left_foot3_collision",
    "left_foot4_collision",
    "left_foot5_collision",
    "left_foot6_collision",
    "left_foot7_collision",
    "right_foot1_collision",
    "right_foot2_collision",
    "right_foot3_collision",
    "right_foot4_collision",
    "right_foot5_collision",
    "right_foot6_collision",
    "right_foot7_collision",
)


def get_g1_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(G1_FEET_COLLISION,),
        spec_fn=get_spec,
        articulation=G1_ARTICULATION,
    )
