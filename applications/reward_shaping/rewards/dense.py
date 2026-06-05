import gymnasium as gym


class DenseRewardWrapper(gym.Wrapper):
    """Replace environment reward with dense reward.

    MuJoCo: forward displacement per step (delta_x).
    Highway: speed / max_speed per step.
    """

    def __init__(self, env, env_type: str = "mujoco", max_speed: float = 40.0):
        super().__init__(env)
        self.env_type = env_type
        self.max_speed = max_speed
        self._prev_x = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.env_type == "mujoco":
            self._prev_x = self.unwrapped.data.qpos[0]
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        if self.env_type == "mujoco":
            x_pos = self.unwrapped.data.qpos[0]
            reward = x_pos - self._prev_x
            self._prev_x = x_pos
        elif self.env_type == "highway":
            speed = self.unwrapped.vehicle.speed
            reward = speed / self.max_speed
        else:
            raise ValueError(f"Unknown env_type: {self.env_type}")

        info["dense_reward"] = reward
        return obs, reward, terminated, truncated, info
