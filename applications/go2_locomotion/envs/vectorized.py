import numpy as np
import multiprocessing as mp
from .go2_env import Go2Env


# ---------------------------------------------------------------------------
# Worker process: runs one Go2Env + DomainRandomizer in a separate process
# ---------------------------------------------------------------------------

def _get_dr_config_for_iteration(cfg, iteration):
    """Return DR config overrides based on training phase (used by workers)."""
    if not cfg.get("dr_curriculum", False):
        return cfg
    phase1_end = cfg.get("dr_phase1_end", 500)
    phase2_end = cfg.get("dr_phase2_end", 1500)
    if iteration < phase1_end:
        override = dict(cfg)
        override["dr_friction_range"] = [1.0, 1.0]
        override["dr_mass_scale_range"] = [1.0, 1.0]
        override["dr_ext_force_range"] = [0.0, 0.0]
        override["dr_kp_range"] = [1.0, 1.0]
        override["dr_kd_range"] = [1.0, 1.0]
        return override
    elif iteration < phase2_end:
        override = dict(cfg)
        override["dr_friction_range"] = cfg.get("dr_friction_range_light", [0.8, 1.1])
        override["dr_mass_scale_range"] = cfg.get("dr_mass_scale_range_light", [0.95, 1.05])
        override["dr_ext_force_range"] = cfg.get("dr_ext_force_range_light", [0.0, 1.0])
        return override
    else:
        return cfg


def _batch_worker(conn, cfg, env_ids):
    """Worker process: owns multiple Go2Env + Go2DomainRandomizer instances.

    Runs env_ids (list of int) as a serial mini-batch inside one process.
    This limits the number of OS processes while still parallelising across workers.

    Protocol over pipe:
        recv ("reset",)                       -> send list of (obs, priv)
        recv ("step", actions)                -> send list of (obs, reward, done, priv, cmd_vx)
        recv ("update_cmd_range", range_dict) -> send "ok"
        recv ("set_iteration", int)           -> send "ok"
        recv ("close",)                       -> exit
    """
    import numpy as np
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer

    envs = [Go2Env(cfg) for _ in env_ids]
    drs  = [Go2DomainRandomizer(env, cfg, seed=eid)
            for env, eid in zip(envs, env_ids)]
    current_iteration = 0

    while True:
        msg = conn.recv()
        cmd = msg[0]

        if cmd == "reset":
            dr_cfg = _get_dr_config_for_iteration(cfg, current_iteration)
            results = []
            for env, dr in zip(envs, drs):
                dr.config = dr_cfg
                obs, _ = env.reset()
                priv = dr.randomize()
                env.kp = cfg["kp"] * dr.get_kp_scale()
                env.kd = cfg["kd"] * dr.get_kd_scale()
                results.append((obs, priv))
            conn.send(results)

        elif cmd == "step":
            actions = msg[1]   # shape (len(env_ids), action_dim)
            dr_cfg = _get_dr_config_for_iteration(cfg, current_iteration)
            results = []
            for i, (env, dr) in enumerate(zip(envs, drs)):
                dr.config = dr_cfg
                dr.step(cfg["control_dt"])
                obs, reward, terminated, truncated, info = env.step(actions[i])
                done = terminated or truncated
                cmd_vx = float(env.command[0])
                if done:
                    obs, _ = env.reset()
                    priv = dr.randomize()
                    env.kp = cfg["kp"] * dr.get_kp_scale()
                    env.kd = cfg["kd"] * dr.get_kd_scale()
                else:
                    priv = dr._get_privileged_info()
                results.append((obs, float(reward), done, priv, cmd_vx))
            conn.send(results)

        elif cmd == "update_cmd_range":
            new_range = msg[1]  # dict like {"lin_vel_x": [...], ...}
            for env in envs:
                for key, val in new_range.items():
                    env.cmd_range[key] = list(val)
            conn.send("ok")

        elif cmd == "set_iteration":
            current_iteration = msg[1]
            conn.send("ok")

        elif cmd == "close":
            for env in envs:
                env.close()
            conn.close()
            break


class AsyncDRVecGo2Env:
    """Multiprocessing vectorized Go2 env with per-env DomainRandomizer.

    Spawns num_workers processes, each running num_envs/num_workers envs serially.
    Limits CPU usage while still parallelising across workers.

    Supports command curriculum and DR curriculum via pipe messages:
        - set_iteration(n): updates DR phase in all workers
        - _update_curriculum(ratio): expands cmd_range if tracking ratio > threshold

    Config keys:
        num_envs    (int): total number of environments
        num_workers (int): number of worker processes (default: min(num_envs, 6))
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs   = cfg["num_envs"]
        self.num_workers = cfg.get("num_workers", min(self.num_envs, 6))
        self.obs_dim    = cfg["obs_dim"]
        self.action_dim = cfg["action_dim"]
        self.privileged_dim = cfg["privileged_dim"]

        # Command curriculum state (mirrors DRVecGo2Env)
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
        self._curriculum_stable_count = 0
        self._current_iteration = 0

        # Assign env indices to workers as evenly as possible
        all_ids = list(range(self.num_envs))
        self._worker_env_ids = [all_ids[i::self.num_workers]
                                for i in range(self.num_workers)]

        self._conns = []
        self._procs = []
        ctx = mp.get_context("spawn")
        for env_ids in self._worker_env_ids:
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_batch_worker,
                args=(child_conn, cfg, env_ids),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)

        print(f"AsyncDRVecGo2Env: {self.num_envs} envs across {self.num_workers} workers "
              f"({[len(ids) for ids in self._worker_env_ids]} envs/worker)")

    def set_iteration(self, iteration):
        """Send iteration number to all workers for DR curriculum."""
        self._current_iteration = iteration
        for conn in self._conns:
            conn.send(("set_iteration", iteration))
        for conn in self._conns:
            conn.recv()  # wait for "ok"

    def _update_curriculum(self, avg_tracking_ratio: float):
        """Expand command range after stable tracking (10 consecutive passes)."""
        if avg_tracking_ratio > self._curriculum_threshold:
            self._curriculum_stable_count += 1
        else:
            self._curriculum_stable_count = max(0, self._curriculum_stable_count - 1)
        if self._curriculum_stable_count >= 10:
            self._curriculum_stable_count = 0
            delta = self._curriculum_delta
            lx = self._cmd_range["lin_vel_x"]
            lim = self._cmd_limit["lin_vel_x"]
            new_lo = max(lim[0], lx[0] - delta)
            new_hi = min(lim[1], lx[1] + delta)
            if new_lo != lx[0] or new_hi != lx[1]:
                self._cmd_range["lin_vel_x"] = [new_lo, new_hi]
                for conn in self._conns:
                    conn.send(("update_cmd_range", self._cmd_range))
                for conn in self._conns:
                    conn.recv()

    def get_cmd_range(self):
        return self._cmd_range

    def get_tracking_ratio(self):
        return self._last_tracking_ratio

    def reset(self):
        for conn in self._conns:
            conn.send(("reset",))
        all_results = []
        for conn in self._conns:
            all_results.extend(conn.recv())
        obs  = np.array([r[0] for r in all_results], dtype=np.float32)
        priv = np.array([r[1] for r in all_results], dtype=np.float32)
        return obs, priv

    def step(self, actions):
        # Slice actions per worker and send
        offset = 0
        for conn, env_ids in zip(self._conns, self._worker_env_ids):
            n = len(env_ids)
            conn.send(("step", actions[offset:offset + n]))
            offset += n

        all_results = []
        for conn in self._conns:
            all_results.extend(conn.recv())

        obs     = np.array([r[0] for r in all_results], dtype=np.float32)
        rewards = np.array([r[1] for r in all_results], dtype=np.float32)
        dones   = np.array([r[2] for r in all_results], dtype=bool)
        priv    = np.array([r[3] for r in all_results], dtype=np.float32)

        # Compute tracking ratio from cmd_vx and actual_vx (obs[0])
        tracking_ratios = []
        for r in all_results:
            cmd_vx = r[4]
            actual_vx = r[0][0]  # obs[0] is base_lin_vel_x
            if abs(cmd_vx) > 0.1:
                tracking_ratios.append(actual_vx / cmd_vx)
        self._last_tracking_ratio = float(np.mean(tracking_ratios)) if tracking_ratios else 0.0

        return obs, rewards, dones, priv

    def close(self):
        for conn in self._conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)


class VecGo2Env:
    """Synchronous vectorized Go2 environment.

    Each env is an independent Go2Env instance.
    On done, auto-resets and returns the new obs (standard vec env convention).
    """

    def __init__(self, config):
        self.num_envs = config["num_envs"]
        self.envs = [Go2Env(config) for _ in range(self.num_envs)]
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.privileged_dim = config["privileged_dim"]

    def reset(self):
        obs_list, priv_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            priv_list.append(info["privileged_obs"])
        return (
            np.array(obs_list, dtype=np.float32),
            {"privileged_obs": np.array(priv_list, dtype=np.float32)},
        )

    def step(self, actions):
        obs_list, reward_list, done_list, trunc_list, priv_list = [], [], [], [], []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated

            if done:
                obs, reset_info = env.reset()
                info["privileged_obs"] = reset_info["privileged_obs"]

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            trunc_list.append(truncated)
            priv_list.append(info["privileged_obs"])

        return (
            np.array(obs_list, dtype=np.float32),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            {"privileged_obs": np.array(priv_list, dtype=np.float32)},
        )

    def close(self):
        for env in self.envs:
            env.close()
