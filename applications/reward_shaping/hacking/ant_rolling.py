import gymnasium as gym


class AntRollingBrokenReward(gym.Wrapper):
    """BROKEN: Only rewards forward velocity. Agent learns to roll instead of walk."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.unwrapped.data.qvel[0]
        info["hack_type"] = "rolling"
        return obs, reward, terminated, truncated, info


class AntRollingFixedReward(gym.Wrapper):
    """FIXED: Forward velocity + posture penalty prevents rolling."""

    def __init__(self, env, w_speed=1.0, w_posture=2.0, target_z=0.75):
        super().__init__(env)
        self.w_speed = w_speed
        self.w_posture = w_posture
        self.target_z = target_z

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        forward_vel = self.unwrapped.data.qvel[0]
        z_pos = self.unwrapped.data.qpos[2]
        posture_penalty = (z_pos - self.target_z) ** 2

        reward = self.w_speed * forward_vel - self.w_posture * posture_penalty
        info["hack_type"] = "fixed_rolling"
        info["z_pos"] = z_pos
        return obs, reward, terminated, truncated, info
