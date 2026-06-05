import gymnasium as gym


class PotentialShapingWrapper(gym.Wrapper):
    """Add potential-based shaping reward: F(s,s') = gamma * Phi(s') - Phi(s).

    Preserves optimal policy (Ng et al. 1999).
    The potential_fn receives the unwrapped env and returns a scalar.
    """

    def __init__(self, env, potential_fn, gamma: float = 0.99):
        super().__init__(env)
        self.potential_fn = potential_fn
        self.gamma = gamma
        self._prev_potential = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self.potential_fn(self.unwrapped)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Zero potential at terminal states so shaping is conservative (Ng et al. 1999)
        current_potential = 0.0 if terminated else self.potential_fn(self.unwrapped)
        shaping = self.gamma * current_potential - self._prev_potential
        self._prev_potential = current_potential

        shaped_reward = reward + shaping
        info["original_reward"] = reward
        info["shaping_reward"] = shaping

        return obs, shaped_reward, terminated, truncated, info


# --- Predefined potential functions ---

def mujoco_x_potential(env):
    """Potential = x position (rewards forward progress)."""
    return env.data.qpos[0]


def highway_speed_potential(env):
    """Potential = normalized speed."""
    return env.vehicle.speed / 40.0
