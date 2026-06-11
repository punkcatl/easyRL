import numpy as np
import multiprocessing as mp
from .go2_env import Go2Env


# ---------------------------------------------------------------------------
# Worker process: runs one Go2Env + DomainRandomizer in a separate process
# ---------------------------------------------------------------------------

def _batch_worker(conn, cfg, env_ids):
    """Worker process: owns multiple Go2Env + Go2DomainRandomizer instances.

    Runs env_ids (list of int) as a serial mini-batch inside one process.
    This limits the number of OS processes while still parallelising across workers.

    Protocol over pipe:
        recv ("reset",)           -> send list of (obs, priv)
        recv ("step", actions)    -> send list of (obs, reward, done, priv)
        recv ("close",)           -> exit
    """
    import numpy as np
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer

    envs = [Go2Env(cfg) for _ in env_ids]
    drs  = [Go2DomainRandomizer(env, cfg, seed=eid)
            for env, eid in zip(envs, env_ids)]

    while True:
        msg = conn.recv()
        cmd = msg[0]

        if cmd == "reset":
            results = []
            for env, dr in zip(envs, drs):
                obs, _ = env.reset()
                priv = dr.randomize()
                env.kp = cfg["kp"] * dr.get_kp_scale()
                env.kd = cfg["kd"] * dr.get_kd_scale()
                results.append((obs, priv))
            conn.send(results)

        elif cmd == "step":
            actions = msg[1]   # shape (len(env_ids), action_dim)
            results = []
            for i, (env, dr) in enumerate(zip(envs, drs)):
                dr.step(cfg["control_dt"])
                obs, reward, terminated, truncated, info = env.step(actions[i])
                done = terminated or truncated
                if done:
                    obs, _ = env.reset()
                    priv = dr.randomize()
                    env.kp = cfg["kp"] * dr.get_kp_scale()
                    env.kd = cfg["kd"] * dr.get_kd_scale()
                else:
                    priv = dr._get_privileged_info()
                results.append((obs, float(reward), done, priv))
            conn.send(results)

        elif cmd == "close":
            for env in envs:
                env.close()
            conn.close()
            break


class AsyncDRVecGo2Env:
    """Multiprocessing vectorized Go2 env with per-env DomainRandomizer.

    Spawns num_workers processes, each running num_envs/num_workers envs serially.
    Limits CPU usage while still parallelising across workers.

    Config keys:
        num_envs    (int): total number of environments
        num_workers (int): number of worker processes (default: min(num_envs, 6))
    """

    def __init__(self, cfg):
        self.num_envs   = cfg["num_envs"]
        self.num_workers = cfg.get("num_workers", min(self.num_envs, 6))
        self.obs_dim    = cfg["obs_dim"]
        self.action_dim = cfg["action_dim"]
        self.privileged_dim = cfg["privileged_dim"]

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
