"""Phase 1: Train Teacher policy with PPO + Curriculum Domain Randomization.

Usage:
    python train_teacher.py --env Ant-v4
    python train_teacher.py --env Hopper-v4 --no-dr   # baseline without DR
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
from agent.ppo_continuous import PPOContinuous


def train_teacher(env_id: str, use_dr: bool = True):
    env_cfg = ENV_CONFIGS[env_id]
    obs_dim = env_cfg["obs_dim"]
    action_dim = env_cfg["action_dim"]
    num_envs = config["num_envs"]
    n_steps = config["n_steps_per_env"]
    n_iterations = config["n_iterations"]

    vec_env = make_vec_env(env_id, num_envs, config, use_dr=use_dr)
    helper = VecEnvHelper(vec_env, num_envs)

    if use_dr:
        teacher = TeacherAgent(obs_dim, PRIVILEGED_DIM, action_dim, config)
    else:
        # Baseline: standard PPO without privileged info
        teacher = PPOContinuous(obs_dim, action_dim, config)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    reward_history = []
    mode = "Teacher (DR)" if use_dr else "Baseline (no DR)"

    for iteration in tqdm(range(n_iterations), desc=f"{mode} Training ({env_id})"):
        all_obs, all_priv, all_actions = [], [], []
        all_log_probs, all_values, all_rewards, all_dones = [], [], [], []

        obs, privileged = helper.reset()

        for step in range(n_steps):
            if use_dr:
                actions, log_probs, values = teacher.act(obs, privileged)
            else:
                obs_norm = teacher.normalize_obs(obs)
                actions, log_probs, values = teacher.act(obs_norm)

            next_obs, rewards, dones, next_privileged = helper.step(actions)

            all_obs.append(obs)
            all_priv.append(privileged)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_rewards.append(rewards)
            all_dones.append(dones)

            obs = next_obs
            privileged = next_privileged

        # Bootstrap value for final state
        if use_dr:
            _, _, next_values = teacher.act(obs, privileged)
        else:
            obs_norm = teacher.normalize_obs(obs)
            _, _, next_values = teacher.act(obs_norm)

        # Convert to arrays: (n_steps, num_envs, ...)
        all_obs = np.array(all_obs)          # (n_steps, num_envs, obs_dim)
        all_priv = np.array(all_priv)        # (n_steps, num_envs, priv_dim)
        all_actions = np.array(all_actions)  # (n_steps, num_envs, action_dim)
        all_log_probs = np.array(all_log_probs)  # (n_steps, num_envs)
        all_values = np.array(all_values)    # (n_steps, num_envs)
        all_rewards = np.array(all_rewards)  # (n_steps, num_envs)
        all_dones = np.array(all_dones)      # (n_steps, num_envs)
        # next_values: (num_envs,) - one bootstrap value per env

        # Compute GAE per environment, then flatten for minibatch update
        if use_dr:
            # Normalize and scale per env before GAE
            all_advantages = np.zeros_like(all_rewards)
            all_returns = np.zeros_like(all_rewards)

            for e in range(num_envs):
                all_advantages[:, e], all_returns[:, e] = \
                    teacher.ppo.compute_gae_single(
                        all_rewards[:, e], all_values[:, e],
                        all_dones[:, e], next_values[e],
                    )

            # Flatten: (n_steps, num_envs, ...) -> (n_steps*num_envs, ...)
            flat_obs = all_obs.reshape(-1, obs_dim)
            flat_priv = all_priv.reshape(-1, PRIVILEGED_DIM)
            flat_actions = all_actions.reshape(-1, action_dim)
            flat_log_probs = all_log_probs.reshape(-1)
            flat_advantages = all_advantages.reshape(-1)
            flat_returns = all_returns.reshape(-1)

            teacher.update(
                flat_obs, flat_priv, flat_actions, flat_log_probs,
                flat_advantages, flat_returns,
            )
        else:
            # Scale rewards, normalize obs, then compute GAE per env
            scaled_rewards = teacher.scale_reward(all_rewards.reshape(-1))
            scaled_rewards = scaled_rewards.reshape(n_steps, num_envs)
            # Read-only normalization: RMS was already updated per-step during rollout
            obs_norm_all = teacher.obs_rms.normalize(all_obs.reshape(-1, obs_dim))
            obs_norm_all = obs_norm_all.reshape(n_steps, num_envs, obs_dim)

            # Bootstrap values need normalization too
            next_obs_norm = teacher.obs_rms.normalize(obs.reshape(-1, obs_dim))
            import torch as _torch
            with _torch.no_grad():
                _nv = teacher.critic(
                    _torch.FloatTensor(next_obs_norm).to(teacher.device)
                ).squeeze(-1).cpu().numpy()

            all_advantages = np.zeros((n_steps, num_envs))
            all_returns = np.zeros((n_steps, num_envs))

            for e in range(num_envs):
                all_advantages[:, e], all_returns[:, e] = \
                    teacher.compute_gae_single(
                        scaled_rewards[:, e], all_values[:, e],
                        all_dones[:, e], _nv[e],
                    )

            flat_obs_norm = obs_norm_all.reshape(-1, obs_dim)
            flat_actions = all_actions.reshape(-1, action_dim)
            flat_log_probs = all_log_probs.reshape(-1)
            flat_advantages = all_advantages.reshape(-1)
            flat_returns = all_returns.reshape(-1)

            teacher.update(
                flat_obs_norm, flat_actions, flat_log_probs,
                flat_advantages, flat_returns,
            )

        episode_reward = all_rewards.sum() / num_envs
        reward_history.append(episode_reward)

        if (iteration + 1) % 50 == 0:
            avg = np.mean(reward_history[-50:])
            print(f"  Iter {iteration + 1}/{n_iterations} | Avg Reward: {avg:.2f}")

    # Save
    env_tag = env_id.replace("-", "_")
    if use_dr:
        save_path = str(results_dir / f"teacher_{env_tag}.pth")
        reward_path = str(results_dir / f"teacher_rewards_{env_tag}.npy")
    else:
        save_path = str(results_dir / f"baseline_{env_tag}.pth")
        reward_path = str(results_dir / f"baseline_rewards_{env_tag}.npy")

    if use_dr:
        teacher.save(save_path)
    else:
        teacher.save(save_path)

    np.save(reward_path, reward_history)
    helper.close()
    print(f"Training complete. Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Teacher / Baseline")
    parser.add_argument("--env", default=config["env_id"], help="Gymnasium env id")
    parser.add_argument("--no-dr", action="store_true", help="Train baseline without DR")
    args = parser.parse_args()
    train_teacher(env_id=args.env, use_dr=not args.no_dr)
