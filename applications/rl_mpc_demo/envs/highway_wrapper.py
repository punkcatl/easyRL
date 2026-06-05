import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
import highway_env  # noqa: F401

from .base_wrapper import BaseEnvWrapper
from config import config as global_config


SUPPORTED_ENVS = [
    "highway-v0",
    "merge-v0",
    "roundabout-v0",
    "intersection-v0",
    "intersection-v1",
    "racetrack-v0",
    "racetrack-large-v0",
]


class HighwayEnvWrapper(BaseEnvWrapper):
    """Unified wrapper for highway-env scenarios with ContinuousAction output."""

    def __init__(self, env_id: str = "highway-v0", render_mode: str = None,
                 config_overrides: dict = None):
        assert env_id in SUPPORTED_ENVS, (
            f"Unsupported env: {env_id}. Choose from {SUPPORTED_ENVS}"
        )
        self.env_id = env_id
        self._env = gym.make(env_id, render_mode=render_mode)

        obs_features = ["x", "y", "vx", "vy", "heading"]
        obs_vehicles = global_config["observation_vehicles"]
        self._observation_dim = obs_vehicles * len(obs_features)

        env_config = {
            "action": {"type": "ContinuousAction"},
            "observation": {
                "type": "Kinematics",
                "features": obs_features,
                "vehicles_count": obs_vehicles,
                "absolute": False,
            },
            "vehicles_count": global_config["vehicles_count"],
            "duration": global_config["duration"],
            "policy_frequency": global_config["policy_frequency"],
        }
        if config_overrides:
            env_config.update(config_overrides)
        self._env.configure(env_config)

        # Patch _rewards for envs that have the action-in-list bug
        if env_id in ("merge-v0", "roundabout-v0"):
            self._patch_rewards()

        self._env = FlattenObservation(self._env)

    def _patch_rewards(self):
        """Fix merge/roundabout _rewards bug with ContinuousAction."""
        original_rewards = self._env.unwrapped._rewards

        def patched_rewards(action):
            try:
                rewards = original_rewards(action)
            except (ValueError, TypeError):
                rewards = original_rewards(1)  # IDLE equivalent
            return rewards

        self._env.unwrapped._rewards = patched_rewards

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        return obs.astype(np.float32)

    def step(self, steering: float, acceleration: float):
        action = np.array([steering, acceleration], dtype=np.float32)
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return obs.astype(np.float32), reward, done, info, terminated, truncated

    def get_ego_state(self) -> dict:
        vehicle = self._env.unwrapped.vehicle
        return {
            "x": vehicle.position[0],
            "y": vehicle.position[1],
            "vx": vehicle.speed * np.cos(vehicle.heading),
            "vy": vehicle.speed * np.sin(vehicle.heading),
            "speed": vehicle.speed,
            "heading": vehicle.heading,
            "lane_index": vehicle.lane_index,
        }

    def get_lane_center_y(self, lane_offset: int) -> float:
        """Get y-coordinate of lane center with offset from current lane.

        Args:
            lane_offset: -1 for left, 0 for current, +1 for right.
        """
        vehicle = self._env.unwrapped.vehicle
        road = self._env.unwrapped.road
        current_lane = vehicle.lane_index
        target_lane_idx = current_lane[2] + lane_offset
        # Clamp to valid lane range
        lanes_count = len(
            road.network.graph.get(current_lane[0], {}).get(current_lane[1], [])
        )
        target_lane_idx = max(0, min(target_lane_idx, lanes_count - 1))
        target_lane = road.network.get_lane(
            (current_lane[0], current_lane[1], target_lane_idx)
        )
        return target_lane.position(vehicle.position[0], 0)[1]

    def get_road_heading(self) -> float:
        """Get road tangent heading (rad) at the ego vehicle's current position."""
        vehicle = self._env.unwrapped.vehicle
        road = self._env.unwrapped.road
        lane = road.network.get_lane(vehicle.lane_index)
        return float(lane.heading_at(vehicle.position[0]))

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    def close(self):
        self._env.close()
