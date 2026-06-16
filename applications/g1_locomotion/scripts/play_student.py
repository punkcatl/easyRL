"""Visualize trained G1 student policy.

Usage:
    python applications/g1_locomotion/scripts/play_student.py \
        --student_path results/g1_flat_locomotion/student/student_best.pt
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play G1 student policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="G1-Flat-Custom-Play-v0", help="Task name.")
parser.add_argument("--student_path", type=str, required=True, help="Path to student checkpoint.")
parser.add_argument("--num_steps", type=int, default=50000, help="Number of sim steps.")
parser.add_argument("--history_length", type=int, default=50, help="Observation history length.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after AppLauncher."""

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import applications.g1_locomotion.config  # noqa: F401

from applications.g1_locomotion.student.networks import AdaptationModule, StudentPolicy


def main():
    device = torch.device("cuda:0")

    # Load student
    ckpt = torch.load(args_cli.student_path, map_location=device)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    history_dim = ckpt["history_dim"]
    latent_dim = ckpt["latent_dim"]

    adaptation = AdaptationModule(input_dim=history_dim, latent_dim=latent_dim).to(device)
    student_policy = StudentPolicy(obs_dim=obs_dim, latent_dim=latent_dim, action_dim=action_dim).to(device)
    adaptation.load_state_dict(ckpt["adaptation"])
    student_policy.load_state_dict(ckpt["policy"])
    adaptation.eval()
    student_policy.eval()

    # Create environment
    env_cfg: ManagerBasedRLEnvCfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0, 2.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    env_cfg.commands.base_velocity.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    num_envs = args_cli.num_envs
    history_length = args_cli.history_length

    # Rolling history buffer
    obs_history_buf = torch.zeros(num_envs, history_length, obs_dim, device=device)

    # Run
    obs_td = env.get_observations()
    obs_tensor = obs_td["policy"] if "policy" in obs_td.keys() else list(obs_td.values())[0]

    print(f"[INFO] Running student for {args_cli.num_steps} steps. Press Ctrl+C to stop.")
    try:
        for step in range(args_cli.num_steps):
            obs_history_buf = torch.roll(obs_history_buf, shifts=-1, dims=1)
            obs_history_buf[:, -1, :] = obs_tensor

            with torch.no_grad():
                hist_flat = obs_history_buf.reshape(num_envs, -1)
                z = adaptation(hist_flat)
                actions = student_policy(obs_tensor, z)

            obs_td, _, _, _ = env.step(actions)
            obs_tensor = obs_td["policy"] if "policy" in obs_td.keys() else list(obs_td.values())[0]

            if step % 1000 == 0:
                print(f"  step {step}/{args_cli.num_steps}")
    except KeyboardInterrupt:
        pass

    print("[INFO] Done.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
