"""Sim2Sim evaluation: run trained policy in clean env (no DR) to verify generalization.

Usage:
    # Evaluate teacher
    python applications/go2_locomotion/evaluate.py --mode teacher --episodes 20

    # Evaluate student
    python applications/go2_locomotion/evaluate.py --mode student --episodes 20

    # With rendering
    python applications/go2_locomotion/evaluate.py --mode teacher --render
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import time
import numpy as np
import torch

from applications.go2_locomotion.config import config
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.agent.ppo import PPOTrainer
from applications.go2_locomotion.agent.teacher_student import StudentAgent


def evaluate_teacher(model_path: str, n_episodes: int = 20, render: bool = False) -> list:
    """Evaluate teacher policy deterministically (use mean action, no sampling)."""
    env = Go2Env(config, render_mode="human" if render else None)
    trainer = PPOTrainer(config)
    trainer.load(model_path)
    trainer.network.eval()

    rewards_list = []
    steps_list = []
    survival_count = 0
    max_steps = int(config["episode_length_s"] / config["control_dt"])

    # Open viewer before first episode so it's ready
    if render:
        env.reset()
        env.render()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            step_start = time.perf_counter()

            obs_norm = trainer.normalize_obs(obs[None])[0]
            obs_t = torch.FloatTensor(obs_norm).unsqueeze(0).to(trainer.device)
            priv_t = torch.zeros(1, config["privileged_dim"]).to(trainer.device)
            with torch.no_grad():
                mean, _ = trainer.network.forward_actor(obs_t)
                action = mean.cpu().numpy().flatten()

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if render:
                env.render()
                # throttle to real-time (control_dt = 0.02s = 50Hz)
                elapsed = time.perf_counter() - step_start
                sleep_time = config["control_dt"] - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if terminated or truncated:
                break

        rewards_list.append(ep_reward)
        steps_list.append(ep_steps)
        if ep_steps >= max_steps:
            survival_count += 1

    env.close()

    print(f"\n{'='*50}")
    print(f"Teacher Evaluation ({n_episodes} episodes, no DR)")
    print(f"  Avg Reward : {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
    print(f"  Avg Steps  : {np.mean(steps_list):.0f} / {max_steps}")
    print(f"  Survival % : {survival_count / n_episodes * 100:.0f}%")
    print(f"{'='*50}")
    return rewards_list


def evaluate_student(model_path: str, n_episodes: int = 20, render: bool = False) -> list:
    """Evaluate student policy using obs history for adaptation."""
    env = Go2Env(config, render_mode="human" if render else None)
    student = StudentAgent(config)
    student.load(model_path)

    obs_dim = config["obs_dim"]
    history_length = config["student_history_length"]
    rewards_list = []
    steps_list = []
    max_steps = int(config["episode_length_s"] / config["control_dt"])
    survival_count = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        history = np.zeros((history_length, obs_dim), dtype=np.float32)
        ep_reward = 0.0
        ep_steps = 0

        while True:
            step_start = time.perf_counter()

            history = np.roll(history, -1, axis=0)
            history[-1] = obs
            action = student.get_action(history.flatten(), obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if render:
                env.render()
                elapsed = time.perf_counter() - step_start
                sleep_time = config["control_dt"] - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if terminated or truncated:
                break

        rewards_list.append(ep_reward)
        steps_list.append(ep_steps)
        if ep_steps >= max_steps:
            survival_count += 1

    env.close()

    print(f"\n{'='*50}")
    print(f"Student Evaluation ({n_episodes} episodes, no DR)")
    print(f"  Avg Reward : {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
    print(f"  Avg Steps  : {np.mean(steps_list):.0f} / {max_steps}")
    print(f"  Survival % : {survival_count / n_episodes * 100:.0f}%")
    print(f"{'='*50}")
    return rewards_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim2Sim evaluation for Go2 locomotion")
    parser.add_argument("--mode", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model .pth file (default: results/<mode>_final.pth)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent / "results"
    if args.model is None:
        args.model = str(results_dir / f"{args.mode}_final.pth")

    if args.mode == "teacher":
        evaluate_teacher(args.model, args.episodes, args.render)
    else:
        evaluate_student(args.model, args.episodes, args.render)
