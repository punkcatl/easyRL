ENV_CONFIGS = {
    "Ant-v4": {"obs_dim": 27, "action_dim": 8},
    "Hopper-v4": {"obs_dim": 11, "action_dim": 3},
    "Humanoid-v4": {"obs_dim": 376, "action_dim": 17},
    "Pusher-v4": {"obs_dim": 23, "action_dim": 7},
}

PRIVILEGED_DIM = 7  # friction(1) + mass(1) + ext_force(3) + actuator(2)

config = {
    # Environment
    "env_id": "Ant-v4",
    "num_envs": 16,

    # Domain Randomization - initial ranges
    "dr_mass_range_init": [0.95, 1.05],
    "dr_mass_range_final": [0.7, 1.3],
    "dr_inertia_range_init": [0.95, 1.05],
    "dr_inertia_range_final": [0.7, 1.3],
    "dr_friction_range_init": [0.9, 1.1],
    "dr_friction_range_final": [0.5, 1.5],
    "dr_force_range_init": [0.0, 5.0],
    "dr_force_range_final": [0.0, 50.0],
    "dr_force_interval_init": 200,
    "dr_force_interval_final": 100,
    "dr_gain_range_init": [0.95, 1.05],
    "dr_gain_range_final": [0.8, 1.2],
    "dr_delay_range_init": [0, 1],
    "dr_delay_range_final": [0, 3],

    # Curriculum
    "curriculum_end_fraction": 0.5,

    # PPO
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs": 10,
    "batch_size": 4096,
    "n_steps_per_env": 2048,
    "max_grad_norm": 0.5,
    "hidden_dim": 256,
    "n_iterations": 1000,

    # Teacher
    "teacher_episodes": 2000,

    # Student (RMA)
    "history_length": 50,
    "latent_dim": 16,
    "student_lr": 1e-3,
    "student_epochs": 100,
    "student_batch_size": 256,
    "distill_dataset_size": 1_000_000,

    # Evaluation
    "eval_episodes": 100,
    "eval_perturbation_force": 50.0,
    "eval_perturbation_interval": 50,
    "eval_ood_factor": 1.3,
}
