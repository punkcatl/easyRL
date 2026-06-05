config = {
    # PPO (shared across all experiments for fair comparison)
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs": 10,
    "batch_size": 64,
    "max_grad_norm": 0.5,

    # MuJoCo experiments
    "mujoco_hidden_dim": 256,
    "mujoco_episodes": 1000,

    # highway-env experiments
    "highway_hidden_dim": 128,
    "highway_episodes": 500,

    # Multi-objective weight sweep
    "weight_sweep_values": [0.1, 0.5, 1.0, 2.0, 5.0],
    "collision_sweep_values": [-1.0, -5.0, -10.0, -20.0, -50.0],

    # Potential-based shaping
    "shaping_gamma": 0.99,

    # Hacking experiments
    "hacking_episodes": 500,
}
