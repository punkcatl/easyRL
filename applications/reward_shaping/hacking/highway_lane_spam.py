import gymnasium as gym


class HighwayLaneSpamBrokenReward(gym.Wrapper):
    """BROKEN: Positive reward for lane changes. Agent oscillates left/right."""

    def __init__(self, env):
        super().__init__(env)
        self._prev_lane = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_lane = self.unwrapped.vehicle.lane_index[2]
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        speed_reward = self.unwrapped.vehicle.speed / 40.0
        lane_change_bonus = 1.0 if current_lane != self._prev_lane else 0.0
        self._prev_lane = current_lane

        reward = speed_reward + lane_change_bonus
        info["hack_type"] = "lane_spam"
        info["lane_changes"] = lane_change_bonus
        return obs, reward, terminated, truncated, info


class HighwayLaneSpamFixedReward(gym.Wrapper):
    """FIXED: Lane change penalty + cooldown prevents oscillation."""

    def __init__(self, env, cooldown_steps=20):
        super().__init__(env)
        self._prev_lane = None
        self._cooldown = 0
        self._cooldown_steps = cooldown_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_lane = self.unwrapped.vehicle.lane_index[2]
        self._cooldown = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        speed_reward = self.unwrapped.vehicle.speed / 40.0

        lane_change_penalty = 0.0
        if current_lane != self._prev_lane:
            if self._cooldown > 0:
                lane_change_penalty = -2.0
            self._cooldown = self._cooldown_steps
        self._cooldown = max(0, self._cooldown - 1)
        self._prev_lane = current_lane

        collision_penalty = -10.0 if self.unwrapped.vehicle.crashed else 0.0
        reward = speed_reward + lane_change_penalty + collision_penalty

        info["hack_type"] = "fixed_lane_spam"
        return obs, reward, terminated, truncated, info
