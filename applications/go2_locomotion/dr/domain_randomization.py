import numpy as np
import mujoco


class Go2DomainRandomizer:
    """Domain Randomization for Go2 environment.

    Randomizes: friction, body mass scale, external push forces, motor gains.
    Returns privileged information vector (7D) for asymmetric critic.

    privileged_obs = [friction_scale(1), mass_scale(1), ext_force(3), motor_strength(2)]
    """

    def __init__(self, env, config, seed=None):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Cache original physics params
        self._original_mass = env.model.body_mass.copy()
        self._original_friction = env.model.geom_friction.copy()

        self._friction_scale = 1.0
        self._mass_scale = 1.0
        self._ext_force = np.zeros(3, dtype=np.float32)
        self._motor_strength = np.ones(2, dtype=np.float32)  # [kp_scale, kd_scale]

        self._push_timer = 0.0
        self._next_push_time = self._sample_push_interval()

    def randomize(self):
        """Apply randomization at episode reset. Returns privileged info (7D)."""
        model = self.env.model

        fr_range = self.config["dr_friction_range"]
        self._friction_scale = float(self.rng.uniform(fr_range[0], fr_range[1]))
        model.geom_friction[:] = self._original_friction * self._friction_scale

        m_range = self.config["dr_mass_scale_range"]
        self._mass_scale = float(self.rng.uniform(m_range[0], m_range[1]))
        model.body_mass[:] = self._original_mass * self._mass_scale

        kp_range = self.config["dr_kp_range"]
        kd_range = self.config["dr_kd_range"]
        self._motor_strength[0] = self.rng.uniform(kp_range[0], kp_range[1])
        self._motor_strength[1] = self.rng.uniform(kd_range[0], kd_range[1])

        self._ext_force = np.zeros(3, dtype=np.float32)
        self._push_timer = 0.0
        self._next_push_time = self._sample_push_interval()

        return self._get_privileged_info()

    def step(self, dt):
        """Called each control step. Applies random pushes at intervals."""
        self._push_timer += dt
        if self._push_timer >= self._next_push_time:
            self._apply_random_push()
            self._push_timer = 0.0
            self._next_push_time = self._sample_push_interval()

    def _apply_random_push(self):
        force_range = self.config["dr_ext_force_range"]
        force_mag = float(self.rng.uniform(force_range[0], force_range[1]))
        direction = self.rng.standard_normal(3).astype(np.float32)
        direction[2] = 0.0  # horizontal only
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        self._ext_force = (direction * force_mag).astype(np.float32)

        # Apply to base body (body id 1 = first non-world body = base)
        base_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id >= 0:
            self.env.data.xfrc_applied[base_id, :3] = self._ext_force

    def get_kp_scale(self):
        return float(self._motor_strength[0])

    def get_kd_scale(self):
        return float(self._motor_strength[1])

    def _get_privileged_info(self):
        return np.concatenate([
            [self._friction_scale],
            [self._mass_scale],
            self._ext_force,
            self._motor_strength,
        ]).astype(np.float32)

    def _sample_push_interval(self):
        interval_range = self.config["dr_push_interval"]
        return float(self.rng.uniform(interval_range[0], interval_range[1]))
