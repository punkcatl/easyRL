import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

from .go2_reward import Go2RewardComputer


class Go2Env(gym.Env):
    """Unitree Go2 locomotion env with PD position control.

    Obs (48D): base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
               + joint_pos_rel(12) + joint_vel(12) + last_action(12) + command(3)
    Action (12D): joint position targets (scaled offset from default angles)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config, render_mode=None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        xml_path = Path(__file__).resolve().parent.parent / "assets" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # Override timestep to match config
        self.model.opt.timestep = config["sim_dt"]

        self.decimation = int(config["control_dt"] / config["sim_dt"])
        self.max_steps = int(config["episode_length_s"] / config["control_dt"])
        self.action_scale = config["action_scale"]
        self.default_angles = config["default_joint_angles"].copy()
        self.kp = config["kp"].copy()
        self.kd = config["kd"].copy()

        self.observation_space = spaces.Box(-np.inf, np.inf, (config["obs_dim"],), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (config["action_dim"],), np.float32)

        self.last_action = np.zeros(config["action_dim"], dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.feet_air_time = np.zeros(4, dtype=np.float32)
        self.last_air_time = np.zeros(4, dtype=np.float32)      # last completed air phase duration
        self.last_contact_time = np.zeros(4, dtype=np.float32)  # last completed contact phase duration
        self._feet_in_contact = np.zeros(4, dtype=bool)
        self._last_joint_vel = np.zeros(12, dtype=np.float32)

        self.reward_computer = Go2RewardComputer(config)
        self.cmd_range = config["command_range"]
        self.cmd_resample_interval = config["command_resample_interval"]

        # Cache body/geom IDs for fast lookup
        self._calf_body_ids = self._get_calf_body_ids()
        self._foot_geom_ids = self._get_foot_geom_ids()
        self._floor_geom_id = self._get_floor_geom_id()

        self._viewer = None

        # External force injection (for benchmark perturbation tests)
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self._pending_force = np.zeros(3, dtype=np.float32)
        self._pending_force_steps = 0

    def _get_calf_body_ids(self):
        ids = []
        for name in ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                ids.append(bid)
        return ids

    def _get_foot_geom_ids(self):
        ids = []
        for name in ["FL", "FR", "RL", "RR"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                ids.append(gid)
        return ids

    def _get_floor_geom_id(self):
        fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        return fid  # may be -1 if unnamed, fallback handled in contact check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[7:19] = self.default_angles
        self.data.qpos[2] = 0.34

        if self.config.get("init_state_randomize", False):
            noise = self.config["init_joint_pos_noise"]
            self.data.qpos[7:19] += np.random.uniform(-noise, noise, 12)
            h_range = self.config["init_base_height_range"]
            self.data.qpos[2] = np.random.uniform(h_range[0], h_range[1])
            v_range = self.config["init_base_lin_vel_range"]
            self.data.qvel[:3] = np.random.uniform(v_range[0], v_range[1], 3)
            av_range = self.config["init_base_ang_vel_range"]
            self.data.qvel[3:6] = np.random.uniform(av_range[0], av_range[1], 3)
            jv_range = self.config["init_joint_vel_range"]
            self.data.qvel[6:18] = np.random.uniform(jv_range[0], jv_range[1], 12)

        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        self.feet_air_time = np.zeros(4, dtype=np.float32)
        self.last_air_time = np.zeros(4, dtype=np.float32)
        self.last_contact_time = np.zeros(4, dtype=np.float32)
        self._feet_in_contact = np.zeros(4, dtype=bool)
        self._last_joint_vel = np.zeros(12, dtype=np.float32)
        self._pending_force = np.zeros(3, dtype=np.float32)
        self._pending_force_steps = 0
        if self._base_body_id >= 0:
            self.data.xfrc_applied[self._base_body_id, :3] = 0.0
        self._resample_command()

        obs = self._get_obs()
        info = {"privileged_obs": self._get_privileged_obs()}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_angles = self.action_scale * action + self.default_angles

        # Apply pending external force (for perturbation tests)
        if self._pending_force_steps > 0:
            self.data.xfrc_applied[self._base_body_id, :3] = self._pending_force
            self._pending_force_steps -= 1
            if self._pending_force_steps == 0:
                self.data.xfrc_applied[self._base_body_id, :3] = 0.0
        else:
            self.data.xfrc_applied[self._base_body_id, :3] = 0.0

        for _ in range(self.decimation):
            joint_pos = self.data.qpos[7:19].astype(np.float32)
            joint_vel = self.data.qvel[6:18].astype(np.float32)
            torques = self.kp * (target_angles - joint_pos) - self.kd * joint_vel
            self.data.ctrl[:] = np.clip(torques, -33.5, 33.5)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        joint_vel_now = self.data.qvel[6:18].astype(np.float32)
        joint_acc = (joint_vel_now - self._last_joint_vel) / self.config["control_dt"]
        self._last_joint_vel = joint_vel_now.copy()

        self._update_feet_air_time()

        if self.step_count % self.cmd_resample_interval == 0:
            self._resample_command()

        terminated = self._check_termination()

        reward_state = {
            "base_lin_vel": self._get_base_linear_velocity(),
            "base_ang_vel": self._get_base_angular_velocity(),
            "command": self.command,
            "torques": self.data.ctrl[:12].astype(np.float32),
            "actions": action,
            "last_actions": self.last_action,
            "joint_pos": self._get_joint_positions(),
            "joint_vel": joint_vel_now,
            "default_joint_angles": self.default_angles,
            "joint_acc": joint_acc,
            "feet_air_time": self.feet_air_time.copy(),
            "base_height": float(self.data.qpos[2]),
            "body_contacts": self._get_body_contacts(),
            "projected_gravity": self._get_projected_gravity(),
            "terminated": terminated,
        }
        reward, reward_components = self.reward_computer.compute(reward_state)
        truncated = self.step_count >= self.max_steps

        self.last_action = action.copy()
        obs = self._get_obs()
        info = {
            "privileged_obs": self._get_privileged_obs(),
            "reward_components": reward_components,
        }
        return obs, reward, terminated, truncated, info

    def _get_body_rotation_matrix(self):
        quat = self.data.qpos[3:7].copy()  # w,x,y,z
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        return rot.reshape(3, 3)

    def _get_base_linear_velocity(self):
        world_vel = self.data.qvel[:3].astype(np.float32)
        R = self._get_body_rotation_matrix()
        return (R.T @ world_vel).astype(np.float32)

    def _get_base_angular_velocity(self):
        world_ang = self.data.qvel[3:6].astype(np.float32)
        R = self._get_body_rotation_matrix()
        return (R.T @ world_ang).astype(np.float32)

    def _get_projected_gravity(self):
        gravity_world = np.array([0.0, 0.0, -1.0])
        R = self._get_body_rotation_matrix()
        return (R.T @ gravity_world).astype(np.float32)

    def _get_joint_positions(self):
        return self.data.qpos[7:19].astype(np.float32)

    def _get_joint_velocities(self):
        return self.data.qvel[6:18].astype(np.float32)

    def _get_privileged_obs(self):
        return np.zeros(self.config["privileged_dim"], dtype=np.float32)

    def _get_obs(self):
        return np.concatenate([
            self._get_base_linear_velocity(),    # 3
            self._get_base_angular_velocity(),   # 3
            self._get_projected_gravity(),       # 3
            self._get_joint_positions() - self.default_angles,  # 12
            self._get_joint_velocities(),        # 12
            self.last_action,                    # 12
            self.command,                        # 3
        ]).astype(np.float32)

    def _update_feet_air_time(self):
        contacts = self._get_foot_contacts()
        dt = self.config["control_dt"]
        for i in range(4):
            was_in_contact = self._feet_in_contact[i]
            now_in_contact = contacts[i]
            if now_in_contact:
                if not was_in_contact:
                    # foot just landed: record completed air phase
                    self.last_air_time[i] = self.feet_air_time[i]
                self.feet_air_time[i] = 0.0
                self.last_contact_time[i] += dt
            else:
                if was_in_contact:
                    # foot just lifted: record completed contact phase
                    self.last_contact_time[i] = 0.0
                self.feet_air_time[i] += dt
            self._feet_in_contact[i] = now_in_contact

    def _get_foot_contacts(self):
        """Returns (4,) bool: whether each foot geom is in contact."""
        contacts = np.zeros(4, dtype=bool)
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            for i, gid in enumerate(self._foot_geom_ids):
                if c.geom1 == gid or c.geom2 == gid:
                    contacts[i] = True
        return contacts

    def _get_body_contacts(self):
        """True if any non-calf body (besides world/floor) is in contact with floor."""
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            # world body id = 0
            for b_robot, b_other in [(b1, b2), (b2, b1)]:
                if b_other == 0 and b_robot != 0 and b_robot not in self._calf_body_ids:
                    return True
        return False

    def _check_termination(self):
        height = self.data.qpos[2]
        if height < self.config["min_body_height"] or height > self.config["max_body_height"]:
            return True
        if self._get_projected_gravity()[2] > -0.5:
            return True
        return False

    def _resample_command(self):
        self.command[0] = np.random.uniform(*self.cmd_range["lin_vel_x"])
        self.command[1] = np.random.uniform(*self.cmd_range["lin_vel_y"])
        self.command[2] = np.random.uniform(*self.cmd_range["ang_vel_yaw"])

    def apply_force(self, force_vec: np.ndarray, duration_steps: int):
        """Apply external force to base body for duration_steps control steps.

        force_vec: (3,) array in world frame [Fx, Fy, Fz] in Newtons.
        Called before step(); force is applied during the next duration_steps steps.
        """
        self._pending_force = np.array(force_vec, dtype=np.float32)
        self._pending_force_steps = int(duration_steps)

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
