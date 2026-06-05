import gymnasium as gym


class HumanoidSlidingBrokenReward(gym.Wrapper):
    """BROKEN: Only forward velocity. Agent slides on belly for less friction."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.unwrapped.data.qvel[0]
        info["hack_type"] = "sliding"
        return obs, reward, terminated, truncated, info


class HumanoidSlidingFixedReward(gym.Wrapper):
    """FIXED: Forward velocity + minimum height constraint prevents sliding."""

    def __init__(self, env, w_speed=1.0, min_height=1.0, height_penalty=5.0):
        super().__init__(env)
        self.w_speed = w_speed
        self.min_height = min_height
        self.height_penalty = height_penalty

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        forward_vel = self.unwrapped.data.qvel[0]
        z_pos = self.unwrapped.data.qpos[2]

        height_violation = max(0.0, self.min_height - z_pos)
        reward = self.w_speed * forward_vel - self.height_penalty * height_violation

        info["hack_type"] = "fixed_sliding"
        info["z_pos"] = z_pos
        return obs, reward, terminated, truncated, info
