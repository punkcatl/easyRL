import gymnasium as gym
import numpy as np
from collections import deque


class DomainRandomizationWrapper(gym.Wrapper):
    """Curriculum Domain Randomization wrapper for MuJoCo environments.

    Randomizes: mass, inertia, friction, external forces, actuator gain/delay.
    Ranges grow linearly from initial to final over the first half of training.
    """

    def __init__(self, env, config, seed=None):
        super().__init__(env)
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.current_episode = 0
        self.total_episodes = config["teacher_episodes"]
        self.curriculum_end = int(self.total_episodes * config["curriculum_end_fraction"])

        self.max_delay = int(config["dr_delay_range_final"][1])
        self.action_buffer = deque(maxlen=self.max_delay + 1)
        self.current_delay = 0

        self.privileged_info = np.zeros(7, dtype=np.float32)

        self._original_mass = None
        self._original_inertia = None
        self._original_friction = None
        self._original_gain = None

    def _get_progress(self):
        return min(self.current_episode / max(self.curriculum_end, 1), 1.0)

    def _interpolate_range(self, init_range, final_range):
        progress = self._get_progress()
        low = init_range[0] + progress * (final_range[0] - init_range[0])
        high = init_range[1] + progress * (final_range[1] - init_range[1])
        return [low, high]

    def _randomize(self):
        model = self.unwrapped.model

        if self._original_mass is None:
            self._original_mass = model.body_mass.copy()
            self._original_inertia = model.body_inertia.copy()
            self._original_friction = model.geom_friction.copy()
            if hasattr(model, "actuator_gainprm"):
                self._original_gain = model.actuator_gainprm.copy()

        # Mass
        mass_range = self._interpolate_range(
            self.config["dr_mass_range_init"], self.config["dr_mass_range_final"]
        )
        mass_scale = self.rng.uniform(mass_range[0], mass_range[1])
        model.body_mass[:] = self._original_mass * mass_scale

        # Inertia
        inertia_range = self._interpolate_range(
            self.config["dr_inertia_range_init"], self.config["dr_inertia_range_final"]
        )
        inertia_scale = self.rng.uniform(inertia_range[0], inertia_range[1])
        model.body_inertia[:] = self._original_inertia * inertia_scale

        # Friction
        friction_range = self._interpolate_range(
            self.config["dr_friction_range_init"], self.config["dr_friction_range_final"]
        )
        friction_scale = self.rng.uniform(friction_range[0], friction_range[1])
        model.geom_friction[:] = self._original_friction * friction_scale

        # Actuator gain
        gain_range = self._interpolate_range(
            self.config["dr_gain_range_init"], self.config["dr_gain_range_final"]
        )
        gain_scale = self.rng.uniform(gain_range[0], gain_range[1])
        if self._original_gain is not None:
            model.actuator_gainprm[:] = self._original_gain * gain_scale

        # Actuator delay
        delay_range = self._interpolate_range(
            self.config["dr_delay_range_init"], self.config["dr_delay_range_final"]
        )
        self.current_delay = int(self.rng.integers(int(delay_range[0]), int(delay_range[1]) + 1))
        self.action_buffer = deque(maxlen=self.current_delay + 1)

        # External force (applied during step)
        force_range = self._interpolate_range(
            self.config["dr_force_range_init"], self.config["dr_force_range_final"]
        )
        self._force_magnitude = self.rng.uniform(force_range[0], force_range[1])
        interval_range = self._interpolate_range(
            [self.config["dr_force_interval_init"]] * 2,
            [self.config["dr_force_interval_final"]] * 2,
        )
        self._force_interval = max(int(interval_range[0]), 1)
        self._step_count = 0
        self._current_force = np.zeros(3, dtype=np.float32)

        # Store privileged info: [friction, mass, Fx, Fy, Fz, gain, delay]
        self.privileged_info[0] = friction_scale
        self.privileged_info[1] = mass_scale
        self.privileged_info[2:5] = 0.0
        self.privileged_info[5] = gain_scale
        self.privileged_info[6] = float(self.current_delay)

    def reset(self, **kwargs):
        self.current_episode += 1
        self._randomize()
        obs, info = self.env.reset(**kwargs)
        info["privileged_info"] = self.privileged_info.copy()
        return obs, info

    def step(self, action):
        # Actuator delay via FIFO buffer
        self.action_buffer.append(action)
        if len(self.action_buffer) > self.current_delay:
            delayed_action = self.action_buffer[0]
        else:
            delayed_action = np.zeros_like(action)

        # External force applied periodically
        self._step_count += 1
        if self._step_count % self._force_interval == 0:
            direction = self.rng.standard_normal(3).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self._current_force = direction * self._force_magnitude
            self.privileged_info[2:5] = self._current_force
            # Apply to torso (body index 1 in most MuJoCo locomotion envs)
            self.unwrapped.data.xfrc_applied[1, :3] = self._current_force
        else:
            self.unwrapped.data.xfrc_applied[1, :3] = 0.0
            self._current_force = np.zeros(3, dtype=np.float32)
            self.privileged_info[2:5] = 0.0

        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        info["privileged_info"] = self.privileged_info.copy()
        return obs, reward, terminated, truncated, info
