"""Phase 2: Distill Teacher into Student (RMA) via Behavior Cloning.

Usage:
    python train_student.py --env Ant-v4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import numpy as np
from tqdm import tqdm

from config import config, ENV_CONFIGS, PRIVILEGED_DIM
from envs.vectorized_env import make_vec_env, VecEnvHelper
from agent.teacher import TeacherAgent
from agent.student import StudentAgent


def collect_distillation_data(teacher, helper, obs_dim, action_dim, dataset_size, history_length):
    """Run Teacher in DR env and collect (history, obs, action) tuples."""
    print("Collecting distillation data from Teacher...")

    obs_history_data = []
    obs_current_data = []
    action_data = []

    obs, privileged = helper.reset()
    num_envs = obs.shape[0]
    histories = np.zeros((num_envs, history_length, obs_dim), dtype=np.float32)

    collected = 0
    pbar = tqdm(total=dataset_size, desc="Collecting data")

    while collected < dataset_size:
        # Shift history and append current obs
        histories = np.roll(histories, -1, axis=1)
        histories[:, -1, :] = obs

        # Teacher deterministic action
        actions = teacher.get_action_deterministic(obs, privileged)

        # Store transitions
        prev_collected = collected
        for i in range(num_envs):
            if collected >= dataset_size:
                break
            obs_history_data.append(histories[i].flatten())
            obs_current_data.append(obs[i])
            action_data.append(actions[i])
            collected += 1

        pbar.update(collected - prev_collected)

        next_obs, _, dones, next_privileged = helper.step(actions)

        # Reset histories for terminated envs
        for i in range(num_envs):
            if dones[i]:
                histories[i] = 0.0

        obs = next_obs
        privileged = next_privileged

    pbar.close()

    return (
        np.array(obs_history_data[:dataset_size]),
        np.array(obs_current_data[:dataset_size]),
        np.array(action_data[:dataset_size]),
    )


def train_student(env_id: str):
    env_cfg = ENV_CONFIGS[env_id]
    obs_dim = env_cfg["obs_dim"]
    action_dim = env_cfg["action_dim"]

    results_dir = Path(__file__).resolve().parent / "results"

    # Load trained Teacher
    teacher_path = str(results_dir / f"teacher_{env_id.replace('-', '_')}.pth")
    teacher = TeacherAgent(obs_dim, PRIVILEGED_DIM, action_dim, config)
    teacher.load(teacher_path)
    print(f"Loaded Teacher from {teacher_path}")

    # Create DR env for data collection
    vec_env = make_vec_env(env_id, config["num_envs"], config, use_dr=True)
    helper = VecEnvHelper(vec_env, config["num_envs"])

    # Collect dataset
    obs_history, obs_current, actions_teacher = collect_distillation_data(
        teacher, helper, obs_dim, action_dim,
        config["distill_dataset_size"], config["history_length"],
    )
    helper.close()

    print(
        f"Dataset: history={obs_history.shape}, "
        f"obs={obs_current.shape}, actions={actions_teacher.shape}"
    )

    # Train Student via BC
    student = StudentAgent(obs_dim, action_dim, config)
    batch_size = config["student_batch_size"]
    n_epochs = config["student_epochs"]
    n_samples = len(obs_history)

    loss_history = []

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        epoch_losses = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = indices[start:end]

            loss = student.train_step(
                obs_history[idx], obs_current[idx], actions_teacher[idx]
            )
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.6f}")

    # Save
    save_path = str(results_dir / f"student_{env_id.replace('-', '_')}.pth")
    student.save(save_path)
    np.save(str(results_dir / f"student_loss_{env_id.replace('-', '_')}.npy"), loss_history)

    print(f"Student training complete. Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Student (RMA distillation)")
    parser.add_argument("--env", default=config["env_id"], help="Gymnasium env id")
    args = parser.parse_args()
    train_student(env_id=args.env)
