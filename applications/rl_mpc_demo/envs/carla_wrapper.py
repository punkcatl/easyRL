import numpy as np
from .base_wrapper import BaseEnvWrapper


class CarlaEnvWrapper(BaseEnvWrapper):
    """CARLA environment wrapper stub. To be implemented for CARLA integration.

    Prerequisites for implementation:
    - CARLA server running (carla-simulator)
    - carla Python API installed (pip install carla)
    - Sensor configuration (camera, lidar, etc.) defined
    """

    def __init__(self, town: str = "Town01", **kwargs):
        raise NotImplementedError(
            "CARLA wrapper not yet implemented. "
            "Install CARLA and implement this class following BaseEnvWrapper interface."
        )

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, steering: float, acceleration: float):
        raise NotImplementedError

    def get_ego_state(self) -> dict:
        raise NotImplementedError

    def get_lane_center_y(self, lane_offset: int) -> float:
        raise NotImplementedError

    def get_road_heading(self) -> float:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
