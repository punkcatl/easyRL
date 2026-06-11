import numpy as np


class Go2RewardComputer:
    """Compute per-step reward for Go2 locomotion.

    Round 5 design: forward_progress provides monotonic gradient,
    exp-kernel tracking for precision, no alive_bonus.
    """

    def __init__(self, config):
        self.scales = config["reward_scales"]
        self.sigma = config["tracking_sigma"]
        self.feet_air_time_threshold = config.get("feet_air_time_threshold", 0.1)
        self.base_height_target = config.get("base_height_target", 0.34)
        self.base_height_sigma = config.get("base_height_sigma", 0.01)

    def compute(self, state: dict) -> tuple:
        """
        state keys: base_lin_vel(3), base_ang_vel(3), command(3),
                    torques(12), actions(12), last_actions(12),
                    joint_pos(12), joint_vel(12), default_joint_angles(12),
                    joint_acc(12), feet_air_time(4), base_height(float),
                    body_contacts(bool), projected_gravity(3),
                    terminated(bool)
        Returns: (total_reward float, components dict)
        """
        components = {}
        body_speed = np.linalg.norm(state["base_lin_vel"][:2])

        # --- Velocity tracking (exp kernel) ---
        lin_vel_error = np.sum((state["command"][:2] - state["base_lin_vel"][:2]) ** 2)
        components["lin_vel_tracking"] = np.exp(-lin_vel_error / self.sigma)

        ang_vel_error = (state["command"][2] - state["base_ang_vel"][2]) ** 2
        components["ang_vel_tracking"] = np.exp(-ang_vel_error / self.sigma)

        # --- Forward progress (linear, clip to prevent reward from moving backward) ---
        cmd_dir = state["command"][:2]
        cmd_norm = np.linalg.norm(cmd_dir)
        if cmd_norm > 0.1:
            unit_cmd = cmd_dir / cmd_norm
            vel_in_cmd_dir = np.dot(state["base_lin_vel"][:2], unit_cmd)
            components["forward_progress"] = float(np.clip(vel_in_cmd_dir, -0.5, 2.0))
        else:
            components["forward_progress"] = 0.0

        # --- Base height (exp kernel around target) ---
        height_error = (state["base_height"] - self.base_height_target) ** 2
        components["base_height_reward"] = np.exp(-height_error / self.base_height_sigma)

        # --- Feet air time (gated by body_speed to prevent hopping in place) ---
        raw_air = np.sum(
            np.clip(state["feet_air_time"] - self.feet_air_time_threshold, 0.0, None)
        )
        speed_gate = float(np.clip(body_speed / 0.3, 0.0, 1.0))
        components["feet_air_time_reward"] = raw_air * speed_gate

        # --- Base stability ---
        components["lin_vel_z_penalty"] = state["base_lin_vel"][2] ** 2
        components["ang_vel_xy_penalty"] = np.sum(state["base_ang_vel"][:2] ** 2)

        # --- Flat orientation ---
        proj_grav = state["projected_gravity"]
        components["flat_orientation_penalty"] = np.sum(np.square(proj_grav[:2]))

        # --- Joint penalties ---
        components["torque_penalty"] = np.sum(state["torques"] ** 2)
        components["action_rate_penalty"] = np.sum(
            (state["actions"] - state["last_actions"]) ** 2
        )
        components["joint_acc_penalty"] = np.sum(state["joint_acc"] ** 2)

        # --- Contact penalty ---
        components["collision_penalty"] = float(state["body_contacts"])

        # --- Termination penalty (one-time) ---
        components["termination_penalty"] = 1.0 if state.get("terminated", False) else 0.0

        total = 0.0
        for key, value in components.items():
            scale = self.scales.get(key, 0.0)
            components[key] = float(value * scale)
            total += components[key]

        return float(total), components
