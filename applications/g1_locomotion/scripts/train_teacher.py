"""Phase 1: Train G1 locomotion teacher policy with rsl_rl PPO.

Usage:
    conda activate env_isaaclab
    python applications/g1_locomotion/scripts/train_teacher.py --task G1-Flat-Custom-v0 --num_envs 1024
    python applications/g1_locomotion/scripts/train_teacher.py --task G1-Rough-Custom-v0 --num_envs 1024
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 locomotion teacher.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--task", type=str, default="G1-Flat-Custom-v0", help="Task name.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max iterations.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after AppLauncher."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register our custom environments
import applications.g1_locomotion.config  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path


def main():
    # Load config from registry
    env_cfg: ManagerBasedRLEnvCfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()

    # Override from CLI
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed

    # Log directory
    log_root_path = os.path.join(
        os.path.dirname(__file__), "..", "results", agent_cfg.experiment_name
    )
    log_root_path = os.path.abspath(log_root_path)
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    # Resume if requested
    if args_cli.resume and args_cli.load_run:
        resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")
        runner.load(resume_path)

    # Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Save final model
    final_path = os.path.join(log_dir, "teacher_final.pt")
    runner.save(final_path)
    print(f"[INFO] Final model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
