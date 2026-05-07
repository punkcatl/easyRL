config = {
    "n_episodes": 300,
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "buffer_size": 10000,
    "batch_size": 64,
    "target_update_freq": 10,
    "hidden_dim": 128,
}
