"""Visualize trained G1 locomotion policy.

Usage:
    python applications/g1_locomotion/scripts/play.py --task G1-Flat-Custom-Play-v0 \
        --load_run 2026-06-15_18-52-06 --checkpoint teacher_final.pt
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play G1 locomotion policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="G1-Flat-Custom-Play-v0", help="Task name.")
parser.add_argument("--load_run", type=str, required=True, help="Run folder name.")
parser.add_argument("--checkpoint", type=str, default="model_1500.pt", help="Checkpoint file.")
parser.add_argument("--num_steps", type=int, default=50000, help="Number of sim steps to run (~16 min).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after AppLauncher."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import applications.g1_locomotion.config  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path


def infer_hidden_dims_from_checkpoint(checkpoint_path: str) -> list[int]:
    """Infer actor hidden dims from checkpoint state_dict."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    dims = []
    i = 0
    while f"actor.{i}.weight" in state_dict:
        dims.append(state_dict[f"actor.{i}.weight"].shape[0])
        i += 2  # skip bias, Linear layers are at 0, 2, 4...
    # Last dim is action_dim, not hidden
    return dims[:-1] if dims else [512, 256, 128]


def main():
    env_cfg: ManagerBasedRLEnvCfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()

    env_cfg.scene.num_envs = args_cli.num_envs

    # Free camera — user can control view with mouse
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0, 2.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    env_cfg.viewer.resolution = (1920, 1080)

    # Hide debug visualizations
    env_cfg.commands.base_velocity.debug_vis = False
    if env_cfg.scene.height_scanner is not None:
        env_cfg.scene.height_scanner.debug_vis = False
    if env_cfg.scene.contact_forces is not None:
        env_cfg.scene.contact_forces.debug_vis = False

    # Determine log path
    log_root_path = os.path.join(
        os.path.dirname(__file__), "..", "results", agent_cfg.experiment_name
    )
    log_root_path = os.path.abspath(log_root_path)

    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    print(f"[INFO] Loading policy from: {resume_path}")

    # Infer network dims from checkpoint to avoid size mismatch
    hidden_dims = infer_hidden_dims_from_checkpoint(resume_path)
    print(f"[INFO] Inferred hidden dims: {hidden_dims}")
    agent_cfg.policy.actor_hidden_dims = hidden_dims
    agent_cfg.policy.critic_hidden_dims = hidden_dims

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner and load
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(resume_path)

    # Get policy
    policy = runner.get_inference_policy(device="cuda:0")

    # Run visualization loop
    obs = env.get_observations()
    num_steps = args_cli.num_steps
    print(f"[INFO] Running policy for {num_steps} steps (~{num_steps * 0.02:.0f}s). Press Ctrl+C to stop.")
    try:
        for step in range(num_steps):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if step % 1000 == 0:
                print(f"  step {step}/{num_steps}")
    except KeyboardInterrupt:
        pass

    print("[INFO] Done.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
