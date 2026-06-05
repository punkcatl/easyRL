import gymnasium as gym
import numpy as np


class MultiObjectiveRewardWrapper(gym.Wrapper):
    """Multi-objective weighted reward for MuJoCo locomotion.

    reward = w_speed * forward_velocity
           + w_alive * alive_bonus
           + w_energy * (-energy_cost)
           + w_posture * (-posture_penalty)
    """

    def __init__(self, env, w_speed=1.0, w_alive=0.5, w_energy=0.01, w_posture=0.1,
                 target_z=0.75):
        super().__init__(env)
        self.w_speed = w_speed
        self.w_alive = w_alive
        self.w_energy = w_energy
        self.w_posture = w_posture
        self.target_z = target_z

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        forward_velocity = self.unwrapped.data.qvel[0]
        alive = 0.0 if terminated else 1.0
        energy = np.sum(np.square(action))

        z_pos = self.unwrapped.data.qpos[2] if len(self.unwrapped.data.qpos) > 2 else 0.0
        posture_penalty = (z_pos - self.target_z) ** 2

        reward = (self.w_speed * forward_velocity
                  + self.w_alive * alive
                  + self.w_energy * (-energy)
                  + self.w_posture * (-posture_penalty))

        info["reward_components"] = {
            "speed": forward_velocity,
            "alive": alive,
            "energy": energy,
            "posture": posture_penalty,
        }

        return obs, reward, terminated, truncated, info


class HighwayMultiObjectiveWrapper(gym.Wrapper):
    """Multi-objective weighted reward for highway-env.

    reward = w_speed * speed_reward
           + w_collision * collision_flag
           + w_comfort * (-acceleration)
           + w_lane * lane_keeping
    """

    def __init__(self, env, w_speed=1.0, w_collision=-10.0, w_comfort=0.1, w_lane=0.5):
        super().__init__(env)
        self.w_speed = w_speed
        self.w_collision = w_collision
        self.w_comfort = w_comfort
        self.w_lane = w_lane
        self._prev_speed = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_speed = self.unwrapped.vehicle.speed
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        vehicle = self.unwrapped.vehicle
        speed_reward = vehicle.speed / 40.0
        collision = 1.0 if vehicle.crashed else 0.0

        # Speed delta is acceleration (not jerk which is d(acceleration)/dt)
        current_speed = vehicle.speed
        acceleration = abs(current_speed - self._prev_speed) if self._prev_speed is not None else 0.0
        self._prev_speed = current_speed

        # Lane keeping: penalize lateral deviation from lane center
        # vehicle.position[1] is the lateral (y) coordinate
        # vehicle.lane.start[1] gives the lane center y-coordinate
        try:
            lane = vehicle.target_lane_index
            lane_center_y = vehicle.road.network.get_lane(lane).position(
                vehicle.lane_distance, 0
            )[1]
            lateral_deviation = abs(vehicle.position[1] - lane_center_y)
            lane_keeping = max(0.0, 1.0 - lateral_deviation / 2.0)
        except (AttributeError, TypeError):
            # Fallback: use 1.0 if on road, 0.0 if crashed
            lane_keeping = 0.0 if vehicle.crashed else 1.0

        reward = (self.w_speed * speed_reward
                  + self.w_collision * collision
                  + self.w_comfort * (-acceleration)
                  + self.w_lane * lane_keeping)

        info["reward_components"] = {
            "speed": speed_reward,
            "collision": collision,
            "comfort": acceleration,
            "lane": lane_keeping,
        }

        return obs, reward, terminated, truncated, info
