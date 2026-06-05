from abc import ABC, abstractmethod
import numpy as np


class BaseEnvWrapper(ABC):
    """Abstract base for all driving environment wrappers."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment, return flattened observation."""
        ...

    @abstractmethod
    def step(self, steering: float, acceleration: float):
        """Execute one step. Returns (obs, reward, done, info, terminated, truncated)."""
        ...

    @abstractmethod
    def get_ego_state(self) -> dict:
        """Return ego vehicle state: {x, y, vx, vy, speed, heading, lane_index}."""
        ...

    @abstractmethod
    def get_lane_center_y(self, lane_offset: int) -> float:
        """Get y-coordinate of lane center with offset from current lane."""
        ...

    @abstractmethod
    def get_road_heading(self) -> float:
        """Get road tangent heading angle (rad) at the ego vehicle's current position."""
        ...

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Dimension of the flattened observation vector."""
        ...

    @abstractmethod
    def close(self):
        ...
