"""Phase 2 Step 3: Sim2Sim validation of student policy.

Runs trained student in clean environment (no DR) and reports:
- avg_reward vs teacher baseline
- survival rate
- tracking ratio

Usage:
    python applications/g1_locomotion/student/evaluate.py \
        --task G1-Flat-Custom-Play-v0 --student_path results/.../student/student_best.pt \
        --episodes 20
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate G1 student policy (Sim2Sim).")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments.")
parser.add_argument("--task", type=str, default="G1-Flat-Custom-Play-v0", help="Task name (PLAY variant).")
parser.add_argument("--student_path", type=str, required=True, help="Path to student checkpoint.")
parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to evaluate.")
parser.add_argument("--history_length", type=int, default=50, help="Observation history length.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after AppLauncher."""

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import applications.g1_locomotion.config  # noqa: F401

from applications.g1_locomotion.student.networks import (
    AdaptationModule,
    StudentPolicy,
)


def main():
    device = torch.device("cuda:0")

    # Load student
    print(f"[INFO] Loading student from: {args_cli.student_path}")
    ckpt = torch.load(args_cli.student_path, map_location=device)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    history_dim = ckpt["history_dim"]
    latent_dim = ckpt["latent_dim"]

    adaptation = AdaptationModule(input_dim=history_dim, latent_dim=latent_dim).to(device)
    policy = StudentPolicy(obs_dim=obs_dim, latent_dim=latent_dim, action_dim=action_dim).to(device)
    adaptation.load_state_dict(ckpt["adaptation"])
    policy.load_state_dict(ckpt["policy"])
    adaptation.eval()
    policy.eval()

    # Create environment (PLAY variant: no DR, no push)
    env_cfg: ManagerBasedRLEnvCfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=True)

    num_envs = args_cli.num_envs
    history_length = args_cli.history_length

    # Rolling history buffer
    obs_history_buf = torch.zeros(num_envs, history_length, obs_dim, device=device)

    # Metrics
    episode_rewards = []
    episode_lengths = []
    max_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    completed_episodes = 0

    obs, _ = env.get_observations()
    current_rewards = torch.zeros(num_envs, device=device)
    current_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)

    print(f"[INFO] Evaluating {args_cli.episodes} episodes (max_steps={max_steps})...")

    while completed_episodes < args_cli.episodes:
        # Update history
        obs_history_buf = torch.roll(obs_history_buf, shifts=-1, dims=1)
        obs_history_buf[:, -1, :] = obs

        # Student inference
        with torch.no_grad():
            hist_flat = obs_history_buf.reshape(num_envs, -1)
            z = adaptation(hist_flat)
            actions = policy(obs, z)

        # Step
        obs, rewards, dones, truncated, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        # Check for done episodes
        done_mask = dones.bool() | truncated.bool() if truncated is not None else dones.bool()
        if done_mask.any():
            for i in done_mask.nonzero(as_tuple=True)[0]:
                episode_rewards.append(current_rewards[i].item())
                episode_lengths.append(current_lengths[i].item())
                completed_episodes += 1
                # Reset tracking for this env
                current_rewards[i] = 0.0
                current_lengths[i] = 0
                obs_history_buf[i] = 0.0

                if completed_episodes >= args_cli.episodes:
                    break

    env.close()

    # Report
    rewards_arr = np.array(episode_rewards[:args_cli.episodes])
    lengths_arr = np.array(episode_lengths[:args_cli.episodes])
    survival_rate = np.mean(lengths_arr >= max_steps * 0.95) * 100

    print("\n" + "=" * 50)
    print("G1 Student Sim2Sim Evaluation Results")
    print("=" * 50)
    print(f"  Episodes:       {len(rewards_arr)}")
    print(f"  Avg Reward:     {rewards_arr.mean():.2f} +/- {rewards_arr.std():.2f}")
    print(f"  Avg Length:     {lengths_arr.mean():.1f} / {max_steps} steps")
    print(f"  Survival Rate:  {survival_rate:.1f}%")
    print(f"  Min Reward:     {rewards_arr.min():.2f}")
    print(f"  Max Reward:     {rewards_arr.max():.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
    simulation_app.close()
