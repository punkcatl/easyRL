import numpy as np


class Go2RewardComputer:
    """Compute per-step reward for Go2 locomotion.

    Round 11: kill lurching exploit via tracking-fraction progress,
    competitive gait rewards, and all-feet-down penalty.
    """

    def __init__(self, config):
        self.scales = config["reward_scales"]
        self.sigma = config["tracking_sigma"]
        self.feet_air_time_threshold = config.get("feet_air_time_threshold", 0.1)
        self.base_height_target = config.get("base_height_target", 0.34)
        self.base_height_sigma = config.get("base_height_sigma", 0.05)
        self.trot_period = config.get("trot_period", 0.5)

    def compute(self, state: dict) -> tuple:
        components = {}

        upright = float(state["projected_gravity"][2] < -0.5)
        body_speed = np.linalg.norm(state["base_lin_vel"][:2])

        # --- Velocity tracking (exp kernel) ---
        lin_vel_error = np.sum((state["command"][:2] - state["base_lin_vel"][:2]) ** 2)
        components["lin_vel_tracking"] = np.exp(-lin_vel_error / self.sigma)

        ang_vel_error = (state["command"][2] - state["base_ang_vel"][2]) ** 2
        components["ang_vel_tracking"] = np.exp(-ang_vel_error / self.sigma)

        # --- Forward progress: raw velocity in command direction ---
        # Rewards going faster unconditionally — combined with exp tracking for accuracy
        cmd_dir = state["command"][:2]
        cmd_norm = np.linalg.norm(cmd_dir)
        if cmd_norm > 0.1:
            unit_cmd = cmd_dir / cmd_norm
            vel_in_cmd_dir = np.dot(state["base_lin_vel"][:2], unit_cmd)
            components["forward_progress"] = float(np.clip(vel_in_cmd_dir, -0.5, 2.0))
        else:
            components["forward_progress"] = 0.0

        # --- Base height (ungated) ---
        height_error = (state["base_height"] - self.base_height_target) ** 2
        components["base_height_reward"] = float(np.exp(-height_error / self.base_height_sigma))

        # --- Feet air time (capped per foot) ---
        raw_air = np.sum(
            np.clip(state["feet_air_time"] - self.feet_air_time_threshold, 0.0, 0.3)
        )
        components["feet_air_time_reward"] = float(raw_air)

        # --- Trot gait schedule (moving-gated, upright-gated) ---
        moving = float(np.clip(body_speed / 0.2, 0.0, 1.0))
        t = state.get("time", 0.0)
        trot_phase = np.sin(2 * np.pi * t / self.trot_period)
        feet_contact = state.get("feet_contact", np.zeros(4, dtype=bool))
        if trot_phase > 0:
            desired = np.array([True, False, False, True])   # FL+RR stance
        else:
            desired = np.array([False, True, True, False])   # FR+RL stance
        gait_match = np.mean(feet_contact == desired)
        components["gait_schedule"] = float(gait_match) * upright * moving

        # --- Gait symmetry: diagonal pairs anti-phase ---
        # FL=0, FR=1, RL=2, RR=3
        diag1_match = float(feet_contact[0] == feet_contact[3])   # FL==RR
        diag2_match = float(feet_contact[1] == feet_contact[2])   # FR==RL
        anti_phase = float(feet_contact[0] != feet_contact[1])    # FL != FR
        symmetry = (diag1_match + diag2_match + anti_phase) / 3.0
        components["gait_symmetry"] = float(symmetry) * upright * moving

        # --- All-feet-contact penalty (penalizes static lurch ground phase) ---
        n_feet_down = float(np.sum(feet_contact.astype(float)))
        all_down_frac = max(0.0, (n_feet_down - 2.0) / 2.0)
        components["all_feet_contact_penalty"] = all_down_frac

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
