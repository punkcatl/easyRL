import numpy as np


class Go2RewardComputer:
    """Compute per-step reward for Go2 locomotion with component tracking."""

    def __init__(self, config):
        self.scales = config["reward_scales"]
        self.sigma = config["tracking_sigma"]

    def compute(self, state: dict) -> tuple:
        """
        state keys: base_lin_vel(3), base_ang_vel(3), command(3),
                    torques(12), actions(12), last_actions(12),
                    joint_acc(12), feet_air_time(4), body_contacts(bool),
                    projected_gravity(3)
        Returns: (total_reward float, components dict)
        """
        components = {}

        lin_vel_error = np.sum((state["command"][:2] - state["base_lin_vel"][:2]) ** 2)
        components["lin_vel_tracking"] = np.exp(-lin_vel_error / self.sigma)

        ang_vel_error = (state["command"][2] - state["base_ang_vel"][2]) ** 2
        components["ang_vel_tracking"] = np.exp(-ang_vel_error / self.sigma)

        components["lin_vel_z_penalty"] = state["base_lin_vel"][2] ** 2
        components["ang_vel_xy_penalty"] = np.sum(state["base_ang_vel"][:2] ** 2)
        components["torque_penalty"] = np.sum(state["torques"] ** 2)
        components["action_rate_penalty"] = np.sum(
            (state["actions"] - state["last_actions"]) ** 2
        )
        components["joint_acc_penalty"] = np.sum(state["joint_acc"] ** 2)
        components["feet_air_time_reward"] = np.sum(
            np.clip(state["feet_air_time"] - 0.5, 0.0, None)
        )
        components["collision_penalty"] = float(state["body_contacts"])

        total = 0.0
        for key, value in components.items():
            scale = self.scales.get(key, 0.0)
            components[key] = float(value * scale)
            total += components[key]

        return float(total), components
