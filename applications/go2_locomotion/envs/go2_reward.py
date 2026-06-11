import numpy as np


class Go2RewardComputer:
    """Compute per-step reward for Go2 locomotion.

    Round 8: remove speed gates (bootstrap trap), sharper signals, trot schedule.
    """

    def __init__(self, config):
        self.scales = config["reward_scales"]
        self.sigma = config["tracking_sigma"]
        self.feet_air_time_threshold = config.get("feet_air_time_threshold", 0.1)
        self.base_height_target = config.get("base_height_target", 0.34)
        self.base_height_sigma = config.get("base_height_sigma", 0.05)
        self.trot_period = config.get("trot_period", 0.5)  # seconds

    def compute(self, state: dict) -> tuple:
        """
        state keys: base_lin_vel(3), base_ang_vel(3), command(3),
                    torques(12), actions(12), last_actions(12),
                    joint_pos(12), joint_vel(12), default_joint_angles(12),
                    joint_acc(12), feet_air_time(4), feet_contact(4),
                    base_height(float), body_contacts(bool),
                    projected_gravity(3), terminated(bool), time(float)
        Returns: (total_reward float, components dict)
        """
        components = {}

        # --- Velocity tracking (exp kernel, sharp sigma) ---
        lin_vel_error = np.sum((state["command"][:2] - state["base_lin_vel"][:2]) ** 2)
        components["lin_vel_tracking"] = np.exp(-lin_vel_error / self.sigma)

        ang_vel_error = (state["command"][2] - state["base_ang_vel"][2]) ** 2
        components["ang_vel_tracking"] = np.exp(-ang_vel_error / self.sigma)

        # --- Forward progress (pure linear, no baseline subtraction) ---
        cmd_dir = state["command"][:2]
        cmd_norm = np.linalg.norm(cmd_dir)
        if cmd_norm > 0.1:
            unit_cmd = cmd_dir / cmd_norm
            vel_in_cmd_dir = np.dot(state["base_lin_vel"][:2], unit_cmd)
            components["forward_progress"] = float(np.clip(vel_in_cmd_dir, -0.5, 2.0))
        else:
            components["forward_progress"] = 0.0

        # --- Base height (NO speed gate — let it work from step 1) ---
        height_error = (state["base_height"] - self.base_height_target) ** 2
        components["base_height_reward"] = float(np.exp(-height_error / self.base_height_sigma))

        # --- Feet air time (capped per foot to prevent fall/bounce exploit) ---
        raw_air = np.sum(
            np.clip(state["feet_air_time"] - self.feet_air_time_threshold, 0.0, 0.3)
        )
        components["feet_air_time_reward"] = float(raw_air)

        # --- Trot gait schedule (only valid when upright) ---
        upright = float(state["projected_gravity"][2] < -0.5)  # 1 if upright, 0 if fallen
        t = state.get("time", 0.0)
        trot_phase = np.sin(2 * np.pi * t / self.trot_period)
        feet_contact = state.get("feet_contact", np.zeros(4, dtype=bool))
        if trot_phase > 0:
            desired = np.array([True, False, False, True])
        else:
            desired = np.array([False, True, True, False])
        gait_match = np.mean(feet_contact == desired)
        components["gait_schedule"] = float(gait_match) * upright

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
