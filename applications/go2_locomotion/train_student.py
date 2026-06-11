"""Phase 2: Distill Teacher into Student via Behavior Cloning (RMA).

Collects (obs_history, obs_current, teacher_action) dataset from trained teacher,
then trains Student policy via supervised BC.

Usage:
    python applications/go2_locomotion/train_student.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from applications.go2_locomotion.config import config
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.agent.ppo import PPOTrainer
from applications.go2_locomotion.agent.teacher_student import StudentAgent
from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer


def collect_teacher_data(teacher_path: str, dataset_size: int, cfg: dict):
    """Run teacher deterministically, collect (history, obs, action) tuples."""
    env = Go2Env(cfg)
    dr = Go2DomainRandomizer(env, cfg, seed=42)

    trainer = PPOTrainer(cfg)
    trainer.load(teacher_path)
    trainer.network.eval()

    obs_dim = cfg["obs_dim"]
    history_length = cfg["student_history_length"]

    obs_history_data = []
    obs_current_data = []
    action_data = []

    obs, _ = env.reset()
    dr.randomize()
    env.kp = cfg["kp"] * dr.get_kp_scale()
    env.kd = cfg["kd"] * dr.get_kd_scale()

    history = np.zeros((history_length, obs_dim), dtype=np.float32)
    collected = 0

    print(f"Collecting {dataset_size} steps from teacher...")
    while collected < dataset_size:
        history = np.roll(history, -1, axis=0)
        history[-1] = obs

        priv = dr._get_privileged_info()
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
        priv_t = torch.FloatTensor(priv).unsqueeze(0).to(trainer.device)

        with torch.no_grad():
            mean, _ = trainer.network.forward_actor(obs_t)
            action = mean.cpu().numpy().flatten()

        obs_history_data.append(history.flatten().copy())
        obs_current_data.append(obs.copy())
        action_data.append(action.copy())
        collected += 1

        if collected % 50000 == 0:
            print(f"  {collected}/{dataset_size} collected")

        obs, _, terminated, truncated, _ = env.step(action)
        dr.step(cfg["control_dt"])

        if terminated or truncated:
            obs, _ = env.reset()
            priv = dr.randomize()
            env.kp = cfg["kp"] * dr.get_kp_scale()
            env.kd = cfg["kd"] * dr.get_kd_scale()
            history = np.zeros((history_length, obs_dim), dtype=np.float32)

    env.close()
    return (
        np.array(obs_history_data[:dataset_size], dtype=np.float32),
        np.array(obs_current_data[:dataset_size], dtype=np.float32),
        np.array(action_data[:dataset_size], dtype=np.float32),
    )


def train_student(cfg=None):
    if cfg is None:
        cfg = config

    results_dir = Path(__file__).resolve().parent / "results"
    teacher_path = str(results_dir / "teacher_final.pth")

    if not Path(teacher_path).exists():
        print(f"Teacher model not found at {teacher_path}. Run train_teacher.py first.")
        return

    obs_history, obs_current, actions = collect_teacher_data(
        teacher_path, cfg["distill_dataset_size"], cfg
    )
    print(f"Dataset: history={obs_history.shape}, obs={obs_current.shape}, actions={actions.shape}")

    student = StudentAgent(cfg)
    batch_size = cfg["student_batch_size"]
    n_epochs = cfg["student_epochs"]
    n_samples = len(obs_history)
    loss_history = []

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = indices[start:end]
            loss = student.train_step(obs_history[idx], obs_current[idx], actions[idx])
            epoch_losses.append(loss)
        avg = float(np.mean(epoch_losses))
        loss_history.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs} | Loss: {avg:.6f}")

    student.save(str(results_dir / "student_final.pth"))
    np.save(str(results_dir / "student_losses.npy"), np.array(loss_history))
    print("Student training complete.")
    return loss_history


if __name__ == "__main__":
    train_student()
