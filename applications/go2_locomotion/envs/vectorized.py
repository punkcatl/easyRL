import numpy as np
from .go2_env import Go2Env


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
