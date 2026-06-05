config = {
    # Export settings
    "opset_version": 17,

    # Models to export
    "models": {
        "ppo_discrete": {
            "input_shape": (1, 25),
            "output_names": ["action_probs"],
            "description": "RL+MPC discrete PPO policy network",
        },
        "student_adaptation": {
            "input_shape": (1, 1350),
            "output_names": ["latent_z"],
            "description": "Sim-to-Real RMA adaptation module",
        },
        "student_base_policy": {
            "input_shape": (1, 43),
            "output_names": ["action"],
            "description": "Sim-to-Real student base policy",
        },
    },

    # Verification
    "verify_n_samples": 1000,
    "verify_atol_fp32": 1e-5,
    "verify_atol_fp16": 1e-3,

    # Benchmark
    "benchmark_n_runs": 1000,
    "benchmark_warmup": 100,
}
