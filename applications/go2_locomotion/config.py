# applications/go2_locomotion/config.py
import numpy as np

# Go2 joint ordering: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#                     RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
DEFAULT_JOINT_ANGLES = np.array([
    0.0, 0.8, -1.5,   # FL: hip, thigh, calf
    0.0, 0.8, -1.5,   # FR
    0.0, 1.0, -1.5,   # RL
    0.0, 1.0, -1.5,   # RR
], dtype=np.float32)

config = {
    # === Environment ===
    "sim_dt": 0.005,            # 200 Hz physics
    "control_dt": 0.02,         # 50 Hz policy (decimation=4)
    "episode_length_s": 20.0,
    "num_envs": 128,
    "vec_env_type": "async",
    "num_workers": 8,

    # === Observation (48D) ===
    "obs_dim": 48,
    "privileged_dim": 7,        # friction(1) + mass_scale(1) + ext_force(3) + motor_strength(2)
    "obs_normalize": True,      # RunningMeanStd normalization

    # === Action (12D) ===
    "action_dim": 12,
    "action_scale": 0.25,       # 0.35->0.25: paired with higher kp for same torque range
    "kp": np.array([20, 35, 35, 20, 35, 35, 20, 35, 35, 20, 35, 35], dtype=np.float32),
    "kd": np.array([1.0] * 12, dtype=np.float32),
    "default_joint_angles": DEFAULT_JOINT_ANGLES,

    # === Reward: Round 5 — forward_progress driven, no alive_bonus ===
    "reward_scales": {
        # Velocity tracking (exp kernel)
        "lin_vel_tracking": 3.0,
        "ang_vel_tracking": 0.5,
        # Forward progress (linear, monotonic gradient from 0 to target speed)
        "forward_progress": 2.0,
        # Feet (gated by body_speed to prevent hopping in place)
        "feet_air_time_reward": 1.5,
        # Base height (exp kernel around nominal)
        "base_height_reward": 1.0,
        # Termination (one-time penalty on terminated=True)
        "termination_penalty": -10.0,
        # Orientation (light)
        "flat_orientation_penalty": -0.5,
        "lin_vel_z_penalty": -2.0,
        "ang_vel_xy_penalty": -0.05,
        # Joint penalties (very light)
        "action_rate_penalty": -0.01,
        "torque_penalty": -0.00005,
        "joint_acc_penalty": -2.5e-7,
        # Safety
        "collision_penalty": -1.0,
        # REMOVED: alive_bonus, joint_pos_penalty, low_speed_penalty
    },
    "tracking_sigma": 0.25,             # 0.15->0.25 (Legged Gym default, smoother gradient)
    "base_height_target": 0.34,
    "base_height_sigma": 0.01,
    "feet_air_time_threshold": 0.1,

    # === Command ===
    "command_range": {
        "lin_vel_x": [0.3, 0.6],
        "lin_vel_y": [-0.1, 0.1],
        "ang_vel_yaw": [-0.3, 0.3],
    },
    "command_limit": {
        "lin_vel_x": [-1.0, 1.5],
        "lin_vel_y": [-0.5, 0.5],
        "ang_vel_yaw": [-1.0, 1.0],
    },
    "cmd_curriculum_threshold": 0.5,
    "cmd_curriculum_delta": 0.1,
    "rel_standing_envs": 0.0,
    "command_resample_interval": 100,

    # === PPO ===
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs": 5,                # 10->5: with 128 envs, data reuse ratio is healthier
    "batch_size": 512,
    "n_steps": 48,
    "max_grad_norm": 1.0,
    "entropy_coef": 0.02,       # 0.01->0.02: more exploration early, with decay
    "value_loss_coef": 1.0,
    "hidden_dim": 128,
    "n_iterations": 5000,

    # === Domain Randomization (curricularized) ===
    "dr_curriculum": True,
    "dr_phase1_end": 500,       # iter 0-500: no DR
    "dr_phase2_end": 1500,      # iter 500-1500: light DR
    "dr_friction_range": [0.5, 1.25],
    "dr_friction_range_light": [0.8, 1.1],
    "dr_mass_scale_range": [0.8, 1.2],
    "dr_mass_scale_range_light": [0.95, 1.05],
    "dr_ext_force_range": [0.0, 3.0],
    "dr_ext_force_range_light": [0.0, 1.0],
    "dr_push_interval": [5.0, 10.0],
    "dr_motor_strength_range": [0.9, 1.1],
    "dr_kp_range": [0.8, 1.2],
    "dr_kd_range": [0.8, 1.2],

    # === Initial State Randomization ===
    "init_state_randomize": True,
    "init_joint_pos_noise": 0.2,        # +/- rad
    "init_base_height_range": [0.30, 0.38],
    "init_base_lin_vel_range": [-0.3, 0.3],
    "init_base_ang_vel_range": [-0.2, 0.2],
    "init_joint_vel_range": [-1.0, 1.0],

    # === Terrain Curriculum ===
    "terrain_types": ["flat", "rough", "slope", "stairs"],
    "terrain_difficulty_range": [0.0, 1.0],
    "curriculum_start_difficulty": 0.0,

    # === Teacher-Student ===
    "student_history_length": 20,
    "student_latent_dim": 16,
    "student_lr": 1e-3,
    "student_epochs": 200,
    "student_batch_size": 256,
    "distill_dataset_size": 500_000,
    "student_val_ratio": 0.1,
    "student_early_stop_patience": 15,

    # === Termination ===
    "terminate_on_body_contact": True,
    "max_body_height": 0.45,        # 0.50->0.45: slightly tighter
    "min_body_height": 0.20,        # 0.15->0.20: slightly tighter

    # === Export ===
    "onnx_opset_version": 17,
}
