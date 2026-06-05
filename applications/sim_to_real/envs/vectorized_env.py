import gymnasium as gym
import numpy as np
from .domain_randomization import DomainRandomizationWrapper


def make_vec_env(env_id: str, num_envs: int, config: dict, use_dr: bool = True):
    """Create vectorized MuJoCo environments with optional DR.

    Uses gymnasium.vector.AsyncVectorEnv for parallel stepping.
    """

    def make_env(seed):
        def _init():
            env = gym.make(env_id)
            if use_dr:
                env = DomainRandomizationWrapper(env, config, seed=seed)
            return env

        return _init

    env_fns = [make_env(seed=i) for i in range(num_envs)]
    vec_env = gym.vector.AsyncVectorEnv(env_fns)
    return vec_env


class VecEnvHelper:
    """Helper to manage vectorized env rollouts and privileged info collection."""

    def __init__(self, vec_env, num_envs: int):
        self.vec_env = vec_env
        self.num_envs = num_envs

    def reset(self):
        obs, infos = self.vec_env.reset()
        privileged = self._extract_privileged(infos)
        return obs, privileged

    def step(self, actions):
        obs, rewards, terminateds, truncateds, infos = self.vec_env.step(actions)
        dones = np.logical_or(terminateds, truncateds)
        privileged = self._extract_privileged(infos)
        return obs, rewards, dones, privileged

    def _extract_privileged(self, infos):
        """Extract privileged_info from vectorized info dict."""
        if "privileged_info" in infos:
            return np.array(infos["privileged_info"])
        return np.zeros((self.num_envs, 7), dtype=np.float32)

    def close(self):
        self.vec_env.close()
