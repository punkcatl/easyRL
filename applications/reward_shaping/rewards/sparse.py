import gymnasium as gym


class SparseRewardWrapper(gym.Wrapper):
    """Replace environment reward with sparse reward.

    MuJoCo: +1 only when x_position > threshold, else 0.
    Highway: +1 at destination, -1 on collision, else 0.
    """

    def __init__(self, env, env_type: str = "mujoco", threshold: float = 100.0):
        super().__init__(env)
        self.env_type = env_type
        self.threshold = threshold

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        if self.env_type == "mujoco":
            x_pos = self.unwrapped.data.qpos[0]
            reward = 1.0 if x_pos > self.threshold else 0.0
        elif self.env_type == "highway":
            if info.get("crashed", False):
                reward = -1.0
            elif (terminated or truncated) and not info.get("crashed", False):
                reward = 1.0
            else:
                reward = 0.0
        else:
            raise ValueError(f"Unknown env_type: {self.env_type}")

        info["sparse_reward"] = reward
        return obs, reward, terminated, truncated, info
