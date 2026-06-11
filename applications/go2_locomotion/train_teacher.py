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
import torch.nn as nn
from applications.go2_locomotion.config import config
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.envs.vectorized import AsyncDRVecGo2Env
from applications.go2_locomotion.agent.ppo import PPOTrainer
from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer


def run_eval_episode(cfg, trainer, cmd_vx=0.5, max_steps=500):
    """Run one deterministic episode with fixed command. Returns avg_vx (ground truth)."""
    eval_cfg = dict(cfg)
    eval_cfg["command_range"] = {
        "lin_vel_x": [cmd_vx, cmd_vx],
        "lin_vel_y": [0.0, 0.0],
        "ang_vel_yaw": [0.0, 0.0],
    }
    eval_cfg["init_state_randomize"] = False
    env = Go2Env(eval_cfg)
    obs, _ = env.reset()
    vx_list = []
    for _ in range(max_steps):
        obs_norm = trainer.normalize_obs(obs[None])[0]
        obs_t = torch.FloatTensor(obs_norm[None]).to(trainer.device)
        with torch.no_grad():
            mean, _ = trainer.network.forward_actor(obs_t)
            action = mean.cpu().numpy()[0]
        obs, _, terminated, truncated, _ = env.step(action)
        vx_list.append(float(obs[0]))
        if terminated or truncated:
            break
    env.close()
    return float(np.mean(vx_list)), len(vx_list)


class DRVecGo2Env:
    """Vectorized Go2 env with DomainRandomizer, command curriculum, and DR curriculum."""

    def __init__(self, cfg):
        self.num_envs = cfg["num_envs"]
        self.cfg = cfg
        self.envs = [Go2Env(cfg) for _ in range(self.num_envs)]
        self.drs = [Go2DomainRandomizer(env, cfg, seed=i)
                    for i, env in enumerate(self.envs)]

        # Command curriculum state
        self._cmd_range = {
            "lin_vel_x": list(cfg["command_range"]["lin_vel_x"]),
            "lin_vel_y": list(cfg["command_range"]["lin_vel_y"]),
            "ang_vel_yaw": list(cfg["command_range"]["ang_vel_yaw"]),
        }
        self._cmd_limit = cfg.get("command_limit", {
            "lin_vel_x": [-1.0, 1.5],
            "lin_vel_y": [-0.5, 0.5],
            "ang_vel_yaw": [-1.0, 1.0],
        })
        self._curriculum_threshold = cfg.get("cmd_curriculum_threshold", 0.5)
        self._curriculum_delta = cfg.get("cmd_curriculum_delta", 0.1)
        self._last_tracking_ratio = 0.0
        self._last_pct_positive = 0.0
        self._curriculum_stable_count = 0

        # DR curriculum state
        self._dr_curriculum = cfg.get("dr_curriculum", False)
        self._dr_phase1_end = cfg.get("dr_phase1_end", 500)
        self._dr_phase2_end = cfg.get("dr_phase2_end", 1500)
        self._current_iteration = 0

    def _get_dr_config_for_iteration(self, iteration):
        """Return DR ranges based on current training phase."""
        if not self._dr_curriculum:
            return self.cfg
        if iteration < self._dr_phase1_end:
            # Phase 1: no DR
            override = dict(self.cfg)
            override["dr_friction_range"] = [1.0, 1.0]
            override["dr_mass_scale_range"] = [1.0, 1.0]
            override["dr_ext_force_range"] = [0.0, 0.0]
            override["dr_kp_range"] = [1.0, 1.0]
            override["dr_kd_range"] = [1.0, 1.0]
            return override
        elif iteration < self._dr_phase2_end:
            # Phase 2: light DR
            override = dict(self.cfg)
            override["dr_friction_range"] = self.cfg.get("dr_friction_range_light", [0.8, 1.1])
            override["dr_mass_scale_range"] = self.cfg.get("dr_mass_scale_range_light", [0.95, 1.05])
            override["dr_ext_force_range"] = self.cfg.get("dr_ext_force_range_light", [0.0, 1.0])
            return override
        else:
            return self.cfg

    def set_iteration(self, iteration):
        self._current_iteration = iteration

    def _update_curriculum(self, avg_tracking_ratio: float):
        if avg_tracking_ratio > self._curriculum_threshold:
            self._curriculum_stable_count += 1
        else:
            self._curriculum_stable_count = max(0, self._curriculum_stable_count - 1)
        # Only expand after 10 consecutive passes (stable tracking, not a spike)
        if self._curriculum_stable_count >= 10:
            self._curriculum_stable_count = 0
            delta = self._curriculum_delta
            lx = self._cmd_range["lin_vel_x"]
            lim = self._cmd_limit["lin_vel_x"]
            new_lo = max(lim[0], lx[0] - delta)
            new_hi = min(lim[1], lx[1] + delta)
            if new_lo != lx[0] or new_hi != lx[1]:
                self._cmd_range["lin_vel_x"] = [new_lo, new_hi]
                for env in self.envs:
                    env.cmd_range["lin_vel_x"] = [new_lo, new_hi]

    def get_cmd_range(self):
        return self._cmd_range

    def get_tracking_ratio(self):
        return self._last_tracking_ratio

    def reset(self):
        obs_list, priv_list = [], []
        dr_cfg = self._get_dr_config_for_iteration(self._current_iteration)
        for env, dr in zip(self.envs, self.drs):
            dr.config = dr_cfg
            obs, _ = env.reset()
            priv = dr.randomize()
            env.kp = self.cfg["kp"] * dr.get_kp_scale()
            env.kd = self.cfg["kd"] * dr.get_kd_scale()
            obs_list.append(obs)
            priv_list.append(priv)
        return np.array(obs_list, np.float32), np.array(priv_list, np.float32)

    def step(self, actions):
        obs_list, reward_list, done_list, priv_list = [], [], [], []
        tracking_ratios = []
        dr_cfg = self._get_dr_config_for_iteration(self._current_iteration)
        for i, (env, dr) in enumerate(zip(self.envs, self.drs)):
            dr.config = dr_cfg
            dr.step(self.cfg["control_dt"])
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated

            cmd_vx = env.command[0]
            actual_vx = obs[0]
            if abs(cmd_vx) > 0.1:
                # Clip to [0, 1.5]: negative ratios (moving backwards) count as 0
                ratio = float(np.clip(actual_vx / cmd_vx, 0.0, 1.5))
                tracking_ratios.append(ratio)

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

        if tracking_ratios:
            self._last_tracking_ratio = float(np.mean(tracking_ratios))
            self._last_pct_positive = float(np.mean([r > 0.3 for r in tracking_ratios]))
        else:
            self._last_tracking_ratio = 0.0
            self._last_pct_positive = 0.0
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

    vec_env_type = cfg.get("vec_env_type", "sync")
    if vec_env_type == "async":
        num_workers = cfg.get("num_workers", min(cfg["num_envs"], 8))
        print(f"Using AsyncDRVecGo2Env ({cfg['num_envs']} envs, {num_workers} workers)")
        vec_env = AsyncDRVecGo2Env(cfg)
    else:
        print(f"Using DRVecGo2Env (sync, {cfg['num_envs']} envs)")
        vec_env = DRVecGo2Env(cfg)
    trainer = PPOTrainer(cfg)

    # Resume from checkpoint if specified
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        resume_path = sys.argv[2] if len(sys.argv) > 2 else str(results_dir / "teacher_iter1500.pth")
        print(f"Resuming from {resume_path}")
        trainer.load(resume_path)

    # LR and entropy decay setup
    lr_start = cfg["lr"]
    lr_end = cfg.get("lr_end", lr_start)
    ent_start = cfg["entropy_coef"]
    ent_end = cfg.get("entropy_coef_end", ent_start)

    reward_history = []
    obs, privileged = vec_env.reset()

    for iteration in range(n_iterations):
        # Linear decay of lr and entropy
        progress = iteration / max(n_iterations - 1, 1)
        current_lr = lr_start + (lr_end - lr_start) * progress
        current_ent = ent_start + (ent_end - ent_start) * progress
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = current_lr
        trainer.entropy_coef = current_ent

        if hasattr(vec_env, 'set_iteration'):
            vec_env.set_iteration(iteration)

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

        # Update obs normalization statistics
        obs_arr = np.array(all_obs)  # (n_steps, num_envs, obs_dim)
        if trainer.obs_rms is not None:
            trainer.obs_rms.update(obs_arr.reshape(-1, cfg["obs_dim"]))

        # Bootstrap next values (with normalized obs)
        obs_norm = trainer.normalize_obs(obs)
        obs_t = torch.FloatTensor(obs_norm).to(trainer.device)
        priv_t = torch.FloatTensor(privileged).to(trainer.device)
        with torch.no_grad():
            next_values = trainer.network.forward_critic(obs_t, priv_t).cpu().numpy()

        # Arrays: (n_steps, num_envs, ...)
        rewards_arr = np.array(all_rewards)
        values_arr = np.array(all_values)
        dones_arr = np.array(all_dones)
        priv_arr = np.array(all_priv)
        actions_arr = np.array(all_actions)
        lp_arr = np.array(all_log_probs)

        # GAE per env, then flatten
        all_adv = np.zeros((n_steps, num_envs), np.float32)
        all_ret = np.zeros((n_steps, num_envs), np.float32)
        for e in range(num_envs):
            adv, ret = trainer.compute_gae(
                rewards_arr[:, e], values_arr[:, e], dones_arr[:, e], float(next_values[e])
            )
            all_adv[:, e] = adv
            all_ret[:, e] = ret

        # Normalize obs for PPO update
        flat_obs_raw = obs_arr.reshape(-1, cfg["obs_dim"])
        flat_obs = trainer.normalize_obs(flat_obs_raw)
        flat_priv = priv_arr.reshape(-1, cfg["privileged_dim"])
        flat_actions = actions_arr.reshape(-1, cfg["action_dim"])
        flat_lp = lp_arr.reshape(-1)
        flat_adv = all_adv.reshape(-1)
        flat_ret = all_ret.reshape(-1)

        # PPO update
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
                torch.nn.utils.clip_grad_norm_(trainer.network.parameters(), trainer.max_grad_norm)
                trainer.optimizer.step()

        avg_reward = float(rewards_arr.mean())
        reward_history.append(avg_reward)

        # Command curriculum
        if hasattr(vec_env, '_update_curriculum'):
            vec_env._update_curriculum(vec_env.get_tracking_ratio())

        if (iteration + 1) % 50 == 0:
            avg50 = float(np.mean(reward_history[-50:]))
            ratio = vec_env.get_tracking_ratio() if hasattr(vec_env, 'get_tracking_ratio') else 0.0
            pct = vec_env._last_pct_positive if hasattr(vec_env, '_last_pct_positive') else 0.0
            cmd_range = vec_env.get_cmd_range() if hasattr(vec_env, 'get_cmd_range') else {}
            vx_range = cmd_range.get('lin_vel_x', [0, 0])
            dr_phase = "no-DR" if iteration < cfg.get("dr_phase1_end", 0) else (
                "light-DR" if iteration < cfg.get("dr_phase2_end", 0) else "full-DR")
            print(f"Iter {iteration + 1}/{n_iterations} | reward: {avg50:.3f} | "
                  f"track: {ratio:.2f} pct30: {pct:.0%} | "
                  f"vx: [{vx_range[0]:.2f},{vx_range[1]:.2f}] | lr: {current_lr:.1e} | {dr_phase}")

        # Ground-truth eval every 200 iter
        if (iteration + 1) % 200 == 0:
            eval_vx, eval_steps = run_eval_episode(cfg, trainer, cmd_vx=0.5)
            print(f"  [EVAL] iter {iteration+1}: avg_vx={eval_vx:.3f} m/s "
                  f"(cmd=0.5, det, {eval_steps} steps)")

        if (iteration + 1) % 500 == 0:
            trainer.save(str(results_dir / f"teacher_iter{iteration + 1}.pth"))

    trainer.save(str(results_dir / "teacher_final.pth"))
    np.save(str(results_dir / "teacher_rewards.npy"), np.array(reward_history))
    vec_env.close()
    print("Teacher training complete.")
    return reward_history


if __name__ == "__main__":
    train_teacher()
