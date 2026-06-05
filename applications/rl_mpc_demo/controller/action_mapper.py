import numpy as np


# Action indices matching highway-env DiscreteMetaAction convention
LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4

ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


class ActionMapper:
    """Maps discrete PPO actions to (v_ref, y_ref) for MPC controllers."""

    def __init__(self, delta_v: float, v_min: float, v_max: float, lane_width: float):
        self.delta_v = delta_v
        self.v_min = v_min
        self.v_max = v_max
        self.lane_width = lane_width
        self.v_ref = 25.0
        self.y_ref = 0.0

    def reset(self, initial_speed: float, initial_y: float, lane_center_fn=None):
        self.v_ref = initial_speed
        self.y_ref = lane_center_fn(0) if lane_center_fn is not None else initial_y

    def map(self, action: int, current_y: float = None, lane_center_fn=None) -> tuple:
        """Convert discrete action to (v_ref, y_ref).

        Args:
            action: integer action index (0-4)
            current_y: current lateral position (used as fallback)
            lane_center_fn: callable(offset) -> y that returns lane center y
                           for lane_offset (-1=left, 0=current, +1=right)

        Returns:
            (v_ref, y_ref) tuple
        """
        if action == FASTER:
            self.v_ref = np.clip(self.v_ref + self.delta_v, self.v_min, self.v_max)
        elif action == SLOWER:
            self.v_ref = np.clip(self.v_ref - self.delta_v, self.v_min, self.v_max)
        elif action == LANE_LEFT:
            if lane_center_fn is not None:
                self.y_ref = lane_center_fn(-1)
            else:
                self.y_ref -= self.lane_width
        elif action == LANE_RIGHT:
            if lane_center_fn is not None:
                self.y_ref = lane_center_fn(1)
            else:
                self.y_ref += self.lane_width
        # IDLE: no change

        return self.v_ref, self.y_ref
