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
    "num_envs": 32,

    # === Observation (48D) ===
    "obs_dim": 48,
    "privileged_dim": 7,        # friction(1) + mass_scale(1) + ext_force(3) + motor_strength(2)

    # === Action (12D) ===
    "action_dim": 12,
    "action_scale": 0.25,
    "kp": np.array([20.0] * 12, dtype=np.float32),
    "kd": np.array([0.5] * 12, dtype=np.float32),
    "default_joint_angles": DEFAULT_JOINT_ANGLES,

    # === Reward ===
    "reward_scales": {
        "lin_vel_tracking": 1.0,
        "ang_vel_tracking": 0.5,
        "lin_vel_z_penalty": -2.0,
        "ang_vel_xy_penalty": -0.05,
        "torque_penalty": -0.0002,
        "action_rate_penalty": -0.01,
        "joint_acc_penalty": -2.5e-7,
        "feet_air_time_reward": 1.0,
        "collision_penalty": -1.0,
        "alive_bonus": 0.1,             # small positive signal to survive early training
    },
    "tracking_sigma": 0.25,
    "feet_air_time_threshold": 0.2,     # feet air time > 0.2s triggers reward (was 0.5s = never)

    # === Command ===
    "command_range": {
        "lin_vel_x": [-1.0, 1.0],
        "lin_vel_y": [-0.5, 0.5],
        "ang_vel_yaw": [-1.0, 1.0],
    },
    "command_resample_interval": 200,

    # === PPO ===
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs": 5,
    "batch_size": 512,          # n_steps(128) * num_envs(32) = 4096 >> batch_size
    "n_steps": 128,             # 128 * 32 envs = 4096 steps per update
    "max_grad_norm": 1.0,
    "entropy_coef": 0.01,
    "value_loss_coef": 1.0,
    "hidden_dim": 128,
    "n_iterations": 3000,

    # === Domain Randomization ===
    "dr_friction_range": [0.5, 1.25],
    "dr_mass_scale_range": [0.8, 1.2],
    "dr_ext_force_range": [0.0, 3.0],
    "dr_push_interval": [5.0, 10.0],
    "dr_motor_strength_range": [0.9, 1.1],
    "dr_kp_range": [0.8, 1.2],
    "dr_kd_range": [0.8, 1.2],
    # note: action delay randomization is not yet implemented in domain_randomization.py

    # === Terrain Curriculum ===
    "terrain_types": ["flat", "rough", "slope", "stairs"],
    "terrain_difficulty_range": [0.0, 1.0],
    "curriculum_start_difficulty": 0.0,

    # === Teacher-Student ===
    "student_history_length": 20,       # 20 * 48D = 960D input (was 50*48=2400D, too large)
    "student_latent_dim": 16,
    "student_lr": 1e-3,
    "student_epochs": 200,              # more epochs with early stopping
    "student_batch_size": 256,
    "distill_dataset_size": 500_000,
    "student_val_ratio": 0.1,           # 10% held out for validation + early stopping
    "student_early_stop_patience": 15,  # stop if val loss doesn't improve for 15 epochs

    # === Termination ===
    "terminate_on_body_contact": True,
    "max_body_height": 0.5,
    "min_body_height": 0.15,

    # === Export ===
    "onnx_opset_version": 17,
}
