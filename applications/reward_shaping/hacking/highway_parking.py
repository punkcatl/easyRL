import gymnasium as gym


class HighwayParkingBrokenReward(gym.Wrapper):
    """BROKEN: Huge collision penalty dominates. Agent stops moving (v=0)."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        speed_reward = self.unwrapped.vehicle.speed / 40.0
        collision_penalty = -100.0 if self.unwrapped.vehicle.crashed else 0.0
        reward = speed_reward + collision_penalty
        info["hack_type"] = "parking"
        return obs, reward, terminated, truncated, info


class HighwayParkingFixedReward(gym.Wrapper):
    """FIXED: Balanced collision penalty + minimum speed enforcement."""

    def __init__(self, env, min_speed=5.0, min_speed_penalty=1.0):
        super().__init__(env)
        self.min_speed = min_speed
        self.min_speed_penalty = min_speed_penalty

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        speed = self.unwrapped.vehicle.speed
        speed_reward = speed / 40.0
        collision_penalty = -10.0 if self.unwrapped.vehicle.crashed else 0.0

        slow_penalty = 0.0
        if speed < self.min_speed:
            slow_penalty = -self.min_speed_penalty * (self.min_speed - speed) / self.min_speed

        reward = speed_reward + collision_penalty + slow_penalty
        info["hack_type"] = "fixed_parking"
        return obs, reward, terminated, truncated, info
