import numpy as np
import mujoco
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

        xml_path = Path(__file__).resolve().parent.parent / "assets" / "go2_scene.xml"
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
        self._last_joint_vel = np.zeros(12, dtype=np.float32)

        self.reward_computer = Go2RewardComputer(config)
        self.cmd_range = config["command_range"]
        self.cmd_resample_interval = config["command_resample_interval"]

        # Cache body/geom IDs for fast lookup
        self._calf_body_ids = self._get_calf_body_ids()
        self._floor_geom_id = self._get_floor_geom_id()

        self._viewer = None

    def _get_calf_body_ids(self):
        ids = []
        for name in ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                ids.append(bid)
        return ids

    def _get_floor_geom_id(self):
        fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        return fid  # may be -1 if unnamed, fallback handled in contact check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[7:19] = self.default_angles
        self.data.qpos[2] = 0.34
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        self.feet_air_time = np.zeros(4, dtype=np.float32)
        self._last_joint_vel = np.zeros(12, dtype=np.float32)
        self._resample_command()

        obs = self._get_obs()
        info = {"privileged_obs": self._get_privileged_obs()}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_angles = self.action_scale * action + self.default_angles

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

        reward_state = {
            "base_lin_vel": self._get_base_linear_velocity(),
            "base_ang_vel": self._get_base_angular_velocity(),
            "command": self.command,
            "torques": self.data.ctrl[:12].astype(np.float32),
            "actions": action,
            "last_actions": self.last_action,
            "joint_acc": joint_acc,
            "feet_air_time": self.feet_air_time.copy(),
            "body_contacts": self._get_body_contacts(),
            "projected_gravity": self._get_projected_gravity(),
        }
        reward, reward_components = self.reward_computer.compute(reward_state)

        terminated = self._check_termination()
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
        for i in range(4):
            if contacts[i]:
                self.feet_air_time[i] = 0.0
            else:
                self.feet_air_time[i] += self.config["control_dt"]

    def _get_foot_contacts(self):
        """Returns (4,) bool: whether each calf body is in contact."""
        contacts = np.zeros(4, dtype=bool)
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            for i, bid in enumerate(self._calf_body_ids):
                if b1 == bid or b2 == bid:
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

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
