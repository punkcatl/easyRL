"""Phase 2 Step 1: Collect teacher rollout data for student distillation.

Runs the frozen teacher in a DR environment and records
(obs_history, obs_current, teacher_action) tuples.

Usage:
    python applications/g1_locomotion/scripts/collect_teacher_data.py \
        --task G1-Flat-Custom-v0 --load_run <run_dir> --checkpoint teacher_final.pt \
        --num_steps 500000
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect teacher data for student distillation.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--task", type=str, default="G1-Flat-Custom-v0", help="Task name.")
parser.add_argument("--load_run", type=str, required=True, help="Run folder with teacher checkpoint.")
parser.add_argument("--checkpoint", type=str, default="teacher_final.pt", help="Teacher checkpoint.")
parser.add_argument("--num_steps", type=int, default=500000, help="Total transitions to collect.")
parser.add_argument("--history_length", type=int, default=50, help="Observation history length.")
parser.add_argument("--output", type=str, default=None, help="Output .npz path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after AppLauncher."""

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import applications.g1_locomotion.config  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path


def main():
    env_cfg: ManagerBasedRLEnvCfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()

    env_cfg.scene.num_envs = args_cli.num_envs

    log_root_path = os.path.join(
        os.path.dirname(__file__), "..", "results", agent_cfg.experiment_name
    )
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    print(f"[INFO] Loading teacher from: {resume_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load teacher
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # Determine obs_dim from environment
    obs, _ = env.get_observations()
    obs_dim = obs.shape[-1]
    num_envs = args_cli.num_envs
    history_length = args_cli.history_length
    total_steps = args_cli.num_steps
    steps_per_env = total_steps // num_envs + 1

    print(f"[INFO] obs_dim={obs_dim}, num_envs={num_envs}, history_length={history_length}")
    print(f"[INFO] Collecting {total_steps} transitions ({steps_per_env} steps/env)...")

    # Rolling history buffer [num_envs, history_length, obs_dim]
    obs_history_buf = torch.zeros(num_envs, history_length, obs_dim, device="cuda:0")

    # Storage (CPU to save GPU memory)
    all_obs_history = []
    all_obs_current = []
    all_actions = []
    collected = 0

    for step in range(steps_per_env):
        # Update history buffer (shift left, append current obs)
        obs_history_buf = torch.roll(obs_history_buf, shifts=-1, dims=1)
        obs_history_buf[:, -1, :] = obs

        # Get deterministic teacher action (use mean, no sampling)
        with torch.no_grad():
            actions = policy(obs)

        # Store data (skip first history_length steps to fill buffer)
        if step >= history_length:
            all_obs_history.append(obs_history_buf.reshape(num_envs, -1).cpu().numpy())
            all_obs_current.append(obs.cpu().numpy())
            all_actions.append(actions.cpu().numpy())
            collected += num_envs

            if collected % 50000 < num_envs:
                print(f"  collected {collected}/{total_steps} transitions")

        if collected >= total_steps:
            break

        # Step environment
        obs, _, _, _, _ = env.step(actions)

    # Save
    output_path = args_cli.output or os.path.join(
        log_root_path, "teacher_distill_data.npz"
    )
    obs_history_arr = np.concatenate(all_obs_history, axis=0)[:total_steps]
    obs_current_arr = np.concatenate(all_obs_current, axis=0)[:total_steps]
    actions_arr = np.concatenate(all_actions, axis=0)[:total_steps]

    np.savez_compressed(
        output_path,
        obs_history=obs_history_arr,
        obs_current=obs_current_arr,
        teacher_actions=actions_arr,
    )
    print(f"[INFO] Saved {obs_history_arr.shape[0]} transitions to: {output_path}")
    print(f"  obs_history shape: {obs_history_arr.shape}")
    print(f"  obs_current shape: {obs_current_arr.shape}")
    print(f"  actions shape: {actions_arr.shape}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
