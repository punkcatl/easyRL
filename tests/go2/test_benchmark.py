# tests/go2/test_benchmark.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np


def test_apply_force_changes_velocity():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config

    env = Go2Env(config)
    env.reset()

    # Walk for 50 steps to reach stable velocity
    action = np.zeros(12, dtype=np.float32)
    for _ in range(50):
        env.step(action)

    vel_before = env._get_base_linear_velocity().copy()

    # Apply large lateral force for 5 steps
    env.apply_force(np.array([0.0, 20.0, 0.0]), duration_steps=5)
    for _ in range(5):
        env.step(action)

    vel_after = env._get_base_linear_velocity().copy()

    assert abs(vel_after[1] - vel_before[1]) > 0.01, (
        f"Force had no effect: vy before={vel_before[1]:.4f}, after={vel_after[1]:.4f}"
    )
    env.close()


def test_apply_force_clears_after_duration():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config

    env = Go2Env(config)
    env.reset()

    env.apply_force(np.array([0.0, 10.0, 0.0]), duration_steps=3)

    action = np.zeros(12, dtype=np.float32)
    for _ in range(3):
        env.step(action)

    base_id = env._base_body_id
    force_remaining = np.linalg.norm(env.data.xfrc_applied[base_id, :3])
    assert force_remaining < 1e-6, f"Force not cleared after duration: {force_remaining}"
    env.close()


def test_run_unit_tests_returns_results():
    from applications.go2_locomotion.benchmark import run_unit_tests
    from applications.go2_locomotion.agent.ppo import PPOTrainer
    from applications.go2_locomotion.config import config

    trainer = PPOTrainer(config)

    def policy_fn(obs):
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
        with torch.no_grad():
            mean, _ = trainer.network.forward_actor(obs_t)
        return mean.cpu().numpy().flatten()

    results = run_unit_tests(policy_fn, render=False)

    assert isinstance(results, dict)
    assert "forward_slow" in results
    assert "survived" in results["forward_slow"]
    assert "rmse_vx" in results["forward_slow"]
    assert "rmse_vy" in results["forward_slow"]
    assert "rmse_yaw" in results["forward_slow"]


def test_run_sequence_test_returns_results():
    from applications.go2_locomotion.benchmark import run_sequence_test, DEFAULT_SEQUENCE
    from applications.go2_locomotion.agent.ppo import PPOTrainer
    from applications.go2_locomotion.config import config

    trainer = PPOTrainer(config)

    def policy_fn(obs):
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
        with torch.no_grad():
            mean, _ = trainer.network.forward_actor(obs_t)
        return mean.cpu().numpy().flatten()

    # Use short sequence for speed
    short_seq = [
        ([1.0, 0.0, 0.0], 0.5, "forward"),
        ([0.0, 0.0, 0.0], 0.2, "brake"),
    ]
    results = run_sequence_test(policy_fn, sequence=short_seq, render=False)

    assert "survived" in results
    assert "avg_rmse" in results
    assert "per_segment" in results
    assert len(results["per_segment"]) == 2
    assert "rmse" in results["per_segment"][0]


def test_run_perturbation_tests_returns_results():
    from applications.go2_locomotion.benchmark import run_perturbation_tests
    from applications.go2_locomotion.agent.ppo import PPOTrainer
    from applications.go2_locomotion.config import config

    trainer = PPOTrainer(config)

    def policy_fn(obs):
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
        with torch.no_grad():
            mean, _ = trainer.network.forward_actor(obs_t)
        return mean.cpu().numpy().flatten()

    results = run_perturbation_tests(policy_fn, render=False)

    assert "single_shock" in results
    assert "max_force" in results
    assert "survived" in results["single_shock"]
    assert "recovery_steps" in results["single_shock"]
    assert "max_survived_force" in results["max_force"]


def test_save_results_creates_json():
    from applications.go2_locomotion.benchmark import save_results
    import tempfile, os, json

    results = {
        "tag": "test",
        "timestamp": "2026-06-11T00:00:00",
        "model": "dummy.pth",
        "unit_tests": {"forward_slow": {"survived": True, "rmse_vx": 0.1,
                                         "rmse_vy": 0.0, "rmse_yaw": 0.0}},
        "sequence_test": {"survived": True, "avg_rmse": 0.1, "per_segment": []},
        "perturbation": {
            "single_shock": {"force_n": 8.0, "survived": True, "recovery_steps": 10},
            "max_force": {"max_survived_force": 8.0},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_results(results, results_dir=tmpdir)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["tag"] == "test"
        assert "unit_tests" in loaded
