config = {
    # Environment
    "env_id": "highway-v0",
    "vehicles_count": 10,
    "observation_vehicles": 5,
    "duration": 60,
    "policy_frequency": 5,

    # PPO
    "ppo_lr": 1e-3,
    "ppo_gamma": 0.98,
    "ppo_lmbda": 0.95,
    "ppo_eps": 0.2,
    "ppo_epochs": 10,
    "ppo_hidden_dim": 128,
    "n_episodes": 1000,

    # Action Mapper
    "delta_v": 5.0,         # m/s per FASTER/SLOWER
    "v_max": 40.0,          # m/s
    "v_min": 0.0,           # m/s
    "lane_width": 4.0,      # m

    # Longitudinal MPC
    "lon_N": 20,            # horizon steps
    "lon_dt": 0.1,          # s
    "lon_Q_v": 10.0,        # velocity tracking weight
    "lon_Q_a": 1.0,         # acceleration penalty
    "lon_R_j": 0.1,         # jerk penalty
    "a_min": -4.0,          # m/s^2
    "a_max": 2.0,           # m/s^2
    "j_min": -5.0,          # m/s^3
    "j_max": 5.0,           # m/s^3

    # Lateral MPC
    "lat_N": 15,            # horizon steps
    "lat_dt": 0.1,          # s
    "lat_Q_y": 10.0,        # lateral position tracking
    "lat_Q_psi": 5.0,       # heading tracking
    "lat_R_delta": 1.0,     # steering effort
    "delta_min": -0.5,      # rad
    "delta_max": 0.5,       # rad
    "delta_dot_max": 0.3,   # rad/s
    "wheelbase": 2.5,       # m (L)
}
