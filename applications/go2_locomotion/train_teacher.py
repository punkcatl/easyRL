"""Phase 1: Train Teacher policy with PPO + Domain Randomization.

Teacher uses asymmetric actor-critic:
  actor: obs(48D) only
  critic: obs(48D) + privileged(7D) = 55D

Usage:
    python applications/go2_locomotion/train_teacher.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from applications.go2_locomotion.config import config
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.agent.ppo import PPOTrainer
from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer


class DRVecGo2Env:
    """Vectorized Go2 env where each instance has its own DomainRandomizer."""

    def __init__(self, cfg):
        self.num_envs = cfg["num_envs"]
        self.cfg = cfg
        self.envs = [Go2Env(cfg) for _ in range(self.num_envs)]
        self.drs = [Go2DomainRandomizer(env, cfg, seed=i)
                    for i, env in enumerate(self.envs)]

    def reset(self):
        obs_list, priv_list = [], []
        for env, dr in zip(self.envs, self.drs):
            obs, _ = env.reset()
            priv = dr.randomize()
            env.kp = self.cfg["kp"] * dr.get_kp_scale()
            env.kd = self.cfg["kd"] * dr.get_kd_scale()
            obs_list.append(obs)
            priv_list.append(priv)
        return np.array(obs_list, np.float32), np.array(priv_list, np.float32)

    def step(self, actions):
        obs_list, reward_list, done_list, priv_list = [], [], [], []
        for i, (env, dr) in enumerate(zip(self.envs, self.drs)):
            dr.step(self.cfg["control_dt"])
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated
            if done:
                obs, _ = env.reset()
                priv = dr.randomize()
                env.kp = self.cfg["kp"] * dr.get_kp_scale()
                env.kd = self.cfg["kd"] * dr.get_kd_scale()
            else:
                priv = dr._get_privileged_info()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            priv_list.append(priv)
        return (
            np.array(obs_list, np.float32),
            np.array(reward_list, np.float32),
            np.array(done_list, bool),
            np.array(priv_list, np.float32),
        )

    def close(self):
        for env in self.envs:
            env.close()


def train_teacher(cfg=None):
    if cfg is None:
        cfg = config

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    num_envs = cfg["num_envs"]
    n_steps = cfg["n_steps"]
    n_iterations = cfg["n_iterations"]

    vec_env = DRVecGo2Env(cfg)
    trainer = PPOTrainer(cfg)

    reward_history = []
    obs, privileged = vec_env.reset()

    for iteration in range(n_iterations):
        all_obs, all_priv, all_actions = [], [], []
        all_log_probs, all_values, all_rewards, all_dones = [], [], [], []

        for _ in range(n_steps):
            actions, log_probs, values = trainer.act(obs, privileged)
            next_obs, rewards, dones, next_priv = vec_env.step(actions)

            all_obs.append(obs.copy())
            all_priv.append(privileged.copy())
            all_actions.append(actions.copy())
            all_log_probs.append(log_probs.copy())
            all_values.append(values.copy())
            all_rewards.append(rewards.copy())
            all_dones.append(dones.copy())

            obs = next_obs
            privileged = next_priv

        # Bootstrap next values
        obs_t = torch.FloatTensor(obs).to(trainer.device)
        priv_t = torch.FloatTensor(privileged).to(trainer.device)
        with torch.no_grad():
            next_values = trainer.network.forward_critic(obs_t, priv_t).cpu().numpy()

        # Arrays: (n_steps, num_envs, ...)
        rewards_arr = np.array(all_rewards)    # (n_steps, num_envs)
        values_arr = np.array(all_values)      # (n_steps, num_envs)
        dones_arr = np.array(all_dones)        # (n_steps, num_envs)
        obs_arr = np.array(all_obs)            # (n_steps, num_envs, obs_dim)
        priv_arr = np.array(all_priv)          # (n_steps, num_envs, priv_dim)
        actions_arr = np.array(all_actions)    # (n_steps, num_envs, action_dim)
        lp_arr = np.array(all_log_probs)       # (n_steps, num_envs)

        # GAE per env, then flatten
        all_adv = np.zeros((n_steps, num_envs), np.float32)
        all_ret = np.zeros((n_steps, num_envs), np.float32)
        for e in range(num_envs):
            adv, ret = trainer.compute_gae(
                rewards_arr[:, e], values_arr[:, e], dones_arr[:, e], float(next_values[e])
            )
            all_adv[:, e] = adv
            all_ret[:, e] = ret

        flat_obs = obs_arr.reshape(-1, cfg["obs_dim"])
        flat_priv = priv_arr.reshape(-1, cfg["privileged_dim"])
        flat_actions = actions_arr.reshape(-1, cfg["action_dim"])
        flat_lp = lp_arr.reshape(-1)
        flat_adv = all_adv.reshape(-1)
        flat_ret = all_ret.reshape(-1)
        flat_dones = dones_arr.reshape(-1)

        # PPO update (pass pre-computed advantages directly)
        import torch.nn as nn
        states_t = torch.FloatTensor(flat_obs).to(trainer.device)
        actions_t = torch.FloatTensor(flat_actions).to(trainer.device)
        old_lp_t = torch.FloatTensor(flat_lp).to(trainer.device)
        adv_t = torch.FloatTensor(flat_adv).to(trainer.device)
        ret_t = torch.FloatTensor(flat_ret).to(trainer.device)
        priv_t2 = torch.FloatTensor(flat_priv).to(trainer.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(flat_obs)
        for _ in range(cfg["epochs"]):
            idx = np.random.permutation(n)
            for start in range(0, n, cfg["batch_size"]):
                end = min(start + cfg["batch_size"], n)
                b = idx[start:end]
                if len(b) == 0:
                    continue
                new_lp, new_val, entropy = trainer.network.evaluate(
                    states_t[b], priv_t2[b], actions_t[b]
                )
                ratio = torch.exp(new_lp - old_lp_t[b])
                s1 = ratio * adv_t[b]
                s2 = torch.clamp(ratio, 1 - trainer.clip_eps, 1 + trainer.clip_eps) * adv_t[b]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = nn.MSELoss()(new_val, ret_t[b])
                loss = (policy_loss
                        + trainer.value_loss_coef * value_loss
                        - trainer.entropy_coef * entropy)
                trainer.optimizer.zero_grad()
                loss.backward()
                import torch as _torch
                _torch.nn.utils.clip_grad_norm_(trainer.network.parameters(), trainer.max_grad_norm)
                trainer.optimizer.step()

        avg_reward = float(rewards_arr.mean())
        reward_history.append(avg_reward)

        if (iteration + 1) % 50 == 0:
            avg50 = float(np.mean(reward_history[-50:]))
            print(f"Iter {iteration + 1}/{n_iterations} | Avg Reward: {avg50:.4f}")

        if (iteration + 1) % 500 == 0:
            trainer.save(str(results_dir / f"teacher_iter{iteration + 1}.pth"))

    trainer.save(str(results_dir / "teacher_final.pth"))
    np.save(str(results_dir / "teacher_rewards.npy"), np.array(reward_history))
    vec_env.close()
    print("Teacher training complete.")
    return reward_history


if __name__ == "__main__":
    train_teacher()
