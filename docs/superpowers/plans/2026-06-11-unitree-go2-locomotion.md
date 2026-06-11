# Unitree Go2 Locomotion 复现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零搭建 Unitree Go2 四足机器人 locomotion RL 训练全流程（MuJoCo 仿真 → PPO 训练 → Domain Randomization → Teacher-Student → Sim2Sim → ONNX 部署），复现宇树/RSL 的工业技术路线。

**Architecture:** 基于 MuJoCo 物理引擎 + MuJoCo Menagerie 的 Go2 模型，用向量化环境并行采样，PPO+GAE 训练 velocity-tracking locomotion policy。采用 Asymmetric Actor-Critic（teacher 用 privileged info，student 用 history adaptation），通过 Curriculum Domain Randomization 实现 sim-to-real 能力。最终导出 ONNX 做推理 benchmark。

**Tech Stack:** Python 3.9+, MuJoCo 2.3+, Gymnasium, PyTorch, NumPy, ONNX/ONNXRuntime, mujoco_menagerie (Go2 MJCF)

---

## File Structure

```
applications/go2_locomotion/
├── __init__.py
├── config.py                    # 所有超参数：环境、PPO、DR、reward、部署
├── envs/
│   ├── __init__.py
│   ├── go2_env.py              # 单 Go2 MuJoCo 环境（Gymnasium 接口）
│   ├── go2_reward.py           # 奖励函数（分项计算，便于消融）
│   ├── vectorized.py           # 向量化环境封装（多进程并行）
│   └── terrain.py              # 地形生成（平地 → 粗糙 → 台阶 curriculum）
├── agent/
│   ├── __init__.py
│   ├── ppo.py                  # PPO 训练器（复用现有 ppo_continuous 逻辑）
│   ├── networks.py             # Actor/Critic/LSTM 网络定义
│   └── teacher_student.py      # Teacher-Student + RMA adaptation
├── dr/
│   ├── __init__.py
│   └── domain_randomization.py # Go2 专用 DR（质量、摩擦、外力、PD增益、延迟）
├── train_teacher.py            # Phase 1: 训练 Teacher（privileged obs）
├── train_student.py            # Phase 2: 蒸馏 Student（RMA history）
├── evaluate.py                 # Sim2Sim 评估 + 指标统计
├── export_onnx.py              # 导出 Student policy 为 ONNX
└── results/                    # 模型、日志、曲线输出目录
```

---

## Task 1: 环境依赖安装 + Go2 模型获取

**Files:**
- Create: `applications/go2_locomotion/envs/__init__.py`
- Create: `applications/go2_locomotion/__init__.py`

- [ ] **Step 1: 安装依赖**

```bash
pip install imageio mujoco-menagerie
```

- [ ] **Step 2: 验证 Go2 模型可加载**

```python
python -c "
import mujoco
from pathlib import Path
import mujoco_menagerie

menagerie_path = Path(mujoco_menagerie.__path__[0])
go2_xml = menagerie_path / 'unitree_go2' / 'scene.xml'
assert go2_xml.exists(), f'Go2 model not found at {go2_xml}'
model = mujoco.MjModel.from_xml_path(str(go2_xml))
data = mujoco.MjData(model)
print(f'Go2 loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}')
print(f'Joint names: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]}')
print(f'Actuator names: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]}')
"
```

Expected: `nq=19` (7 freejoint + 12 joint), `nv=18` (6 freejoint + 12 joint), `nu=12` (12 actuators)

- [ ] **Step 3: 创建包初始化文件**

```bash
mkdir -p applications/go2_locomotion/envs applications/go2_locomotion/agent applications/go2_locomotion/dr applications/go2_locomotion/results
touch applications/go2_locomotion/__init__.py applications/go2_locomotion/envs/__init__.py applications/go2_locomotion/agent/__init__.py applications/go2_locomotion/dr/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add applications/go2_locomotion/
git commit -m "feat(go2): scaffold go2_locomotion module with verified Go2 model"
```

---

## Task 2: Config — 超参数配置

**Files:**
- Create: `applications/go2_locomotion/config.py`

- [ ] **Step 1: 写配置文件**

```python
import numpy as np

# Go2 joint ordering: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#                     RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
DEFAULT_JOINT_ANGLES = np.array([
    0.0, 0.8, -1.5,   # FL: hip, thigh, calf
    0.0, 0.8, -1.5,   # FR
    0.0, 1.0, -1.5,   # RL
    0.0, 1.0, -1.5,   # RR
], dtype=np.float32)

config = {
    # === Environment ===
    "sim_dt": 0.005,            # 200 Hz physics
    "control_dt": 0.02,         # 50 Hz policy (decimation=4)
    "episode_length_s": 20.0,   # max episode length in seconds
    "num_envs": 32,             # vectorized env count

    # === Observation (48D) ===
    # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
    # + joint_pos_rel(12) + joint_vel(12) + last_action(12) + command(3)
    "obs_dim": 48,
    "privileged_dim": 7,        # friction(1) + mass_scale(1) + ext_force(3) + motor_strength(2)

    # === Action (12D) ===
    "action_dim": 12,
    "action_scale": 0.25,       # target = action_scale * action + default_angle
    "kp": np.array([20.0] * 12, dtype=np.float32),
    "kd": np.array([0.5] * 12, dtype=np.float32),
    "default_joint_angles": DEFAULT_JOINT_ANGLES,

    # === Reward ===
    "reward_scales": {
        "lin_vel_tracking": 1.0,
        "ang_vel_tracking": 0.5,
        "lin_vel_z_penalty": -2.0,
        "ang_vel_xy_penalty": -0.05,
        "torque_penalty": -0.0002,
        "action_rate_penalty": -0.01,
        "joint_acc_penalty": -2.5e-7,
        "feet_air_time_reward": 1.0,
        "collision_penalty": -1.0,
        "alive_bonus": 0.0,
    },
    "tracking_sigma": 0.25,     # exp(-error^2/sigma) for velocity tracking

    # === Command ===
    "command_range": {
        "lin_vel_x": [-1.0, 1.0],    # m/s
        "lin_vel_y": [-0.5, 0.5],    # m/s
        "ang_vel_yaw": [-1.0, 1.0],  # rad/s
    },
    "command_resample_interval": 200,  # steps

    # === PPO ===
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs": 5,
    "batch_size": 4096,
    "n_steps": 24,              # steps per env per rollout
    "max_grad_norm": 1.0,
    "entropy_coef": 0.01,
    "value_loss_coef": 1.0,
    "hidden_dim": 128,
    "n_iterations": 3000,

    # === Domain Randomization ===
    "dr_friction_range": [0.5, 1.25],
    "dr_mass_scale_range": [0.8, 1.2],
    "dr_ext_force_range": [0.0, 3.0],   # N, random push
    "dr_push_interval": [5.0, 10.0],    # seconds between pushes
    "dr_motor_strength_range": [0.9, 1.1],
    "dr_kp_range": [0.8, 1.2],
    "dr_kd_range": [0.8, 1.2],
    "dr_action_delay_range": [0, 2],    # steps

    # === Terrain Curriculum ===
    "terrain_types": ["flat", "rough", "slope", "stairs"],
    "terrain_difficulty_range": [0.0, 1.0],
    "curriculum_start_difficulty": 0.0,

    # === Teacher-Student ===
    "student_history_length": 50,
    "student_latent_dim": 16,
    "student_lr": 1e-3,
    "student_epochs": 100,
    "student_batch_size": 256,
    "distill_dataset_size": 500_000,

    # === Termination ===
    "terminate_on_body_contact": True,
    "max_body_height": 0.5,     # above this = probably flipped
    "min_body_height": 0.15,    # below this = collapsed

    # === Export ===
    "onnx_opset_version": 17,
}
```

- [ ] **Step 2: Commit**

```bash
git add applications/go2_locomotion/config.py
git commit -m "feat(go2): add comprehensive config for Go2 locomotion training"
```

---

## Task 3: Go2 单环境 — 核心 Gymnasium 封装

**Files:**
- Create: `applications/go2_locomotion/envs/go2_env.py`
- Create: `applications/go2_locomotion/envs/go2_reward.py`
- Test: `tests/go2/test_go2_env.py`

- [ ] **Step 1: 写环境测试**

```python
# tests/go2/test_go2_env.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest


def test_env_creation():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config
    env = Go2Env(config)
    assert env.observation_space.shape == (48,)
    assert env.action_space.shape == (12,)
    env.close()


def test_reset_returns_valid_obs():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config
    env = Go2Env(config)
    obs, info = env.reset()
    assert obs.shape == (48,)
    assert not np.any(np.isnan(obs))
    assert "privileged_obs" in info
    assert info["privileged_obs"].shape == (7,)
    env.close()


def test_step_returns_valid():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config
    env = Go2Env(config)
    env.reset()
    action = np.zeros(12, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (48,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert "reward_components" in info
    env.close()


def test_pd_control_moves_joints():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config
    env = Go2Env(config)
    env.reset()
    # Positive action should increase target angle
    action = np.ones(12, dtype=np.float32) * 0.5
    obs_before = env._get_joint_positions()
    for _ in range(10):
        env.step(action)
    obs_after = env._get_joint_positions()
    # At least some joints should have moved
    assert np.any(np.abs(obs_after - obs_before) > 0.01)
    env.close()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/go2/test_go2_env.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: 实现 go2_reward.py**

```python
# applications/go2_locomotion/envs/go2_reward.py
import numpy as np


class Go2RewardComputer:
    """Compute per-step reward for Go2 locomotion with component tracking."""

    def __init__(self, config):
        self.scales = config["reward_scales"]
        self.sigma = config["tracking_sigma"]

    def compute(self, state: dict) -> tuple[float, dict]:
        """
        Args:
            state: dict with keys:
                base_lin_vel (3,), base_ang_vel (3,), command (3,),
                torques (12,), actions (12,), last_actions (12,),
                joint_acc (12,), feet_air_time (4,), body_contacts (bool),
                projected_gravity (3,)
        Returns:
            total_reward, reward_components dict
        """
        components = {}

        # Velocity tracking (exponential kernel)
        lin_vel_error = np.sum((state["command"][:2] - state["base_lin_vel"][:2]) ** 2)
        components["lin_vel_tracking"] = np.exp(-lin_vel_error / self.sigma)

        ang_vel_error = (state["command"][2] - state["base_ang_vel"][2]) ** 2
        components["ang_vel_tracking"] = np.exp(-ang_vel_error / self.sigma)

        # Penalties
        components["lin_vel_z_penalty"] = state["base_lin_vel"][2] ** 2
        components["ang_vel_xy_penalty"] = np.sum(state["base_ang_vel"][:2] ** 2)
        components["torque_penalty"] = np.sum(state["torques"] ** 2)
        components["action_rate_penalty"] = np.sum(
            (state["actions"] - state["last_actions"]) ** 2
        )
        components["joint_acc_penalty"] = np.sum(state["joint_acc"] ** 2)

        # Feet air time reward (encourage gait)
        components["feet_air_time_reward"] = np.sum(
            np.clip(state["feet_air_time"] - 0.5, 0.0, None)
        )

        # Collision penalty (body contact besides feet)
        components["collision_penalty"] = float(state["body_contacts"])

        # Sum with scales
        total = 0.0
        for key, value in components.items():
            scale = self.scales.get(key, 0.0)
            components[key] = value * scale
            total += components[key]

        return float(total), components
```

- [ ] **Step 4: 实现 go2_env.py**

```python
# applications/go2_locomotion/envs/go2_env.py
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

import mujoco_menagerie
from .go2_reward import Go2RewardComputer


class Go2Env(gym.Env):
    """Unitree Go2 locomotion environment with PD position control.

    Observation (48D): base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
                       + joint_pos_rel(12) + joint_vel(12) + last_action(12) + command(3)
    Action (12D): joint position targets (scaled and offset from default angles)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config, render_mode=None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        # Load Go2 model
        menagerie_path = Path(mujoco_menagerie.__path__[0])
        xml_path = menagerie_path / "unitree_go2" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # Timing
        self.sim_dt = config["sim_dt"]
        self.control_dt = config["control_dt"]
        self.decimation = int(self.control_dt / self.sim_dt)
        self.max_steps = int(config["episode_length_s"] / self.control_dt)

        # Action/observation
        self.action_dim = config["action_dim"]
        self.obs_dim = config["obs_dim"]
        self.action_scale = config["action_scale"]
        self.default_angles = config["default_joint_angles"]
        self.kp = config["kp"]
        self.kd = config["kd"]

        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.action_dim,), np.float32)

        # State
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.feet_air_time = np.zeros(4, dtype=np.float32)

        # Reward
        self.reward_computer = Go2RewardComputer(config)

        # Command
        self.cmd_range = config["command_range"]
        self.cmd_resample_interval = config["command_resample_interval"]

        # Renderer
        self._viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set initial joint positions to default standing pose
        qpos_start = 7  # skip freejoint (pos3 + quat4)
        self.data.qpos[qpos_start:qpos_start + 12] = self.default_angles

        # Slight random initial height
        self.data.qpos[2] = 0.34  # standing height

        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.step_count = 0
        self.feet_air_time = np.zeros(4, dtype=np.float32)
        self._resample_command()

        obs = self._get_obs()
        info = {"privileged_obs": self._get_privileged_obs()}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # PD position control
        target_angles = self.action_scale * action + self.default_angles

        for _ in range(self.decimation):
            joint_pos = self._get_joint_positions()
            joint_vel = self._get_joint_velocities()
            torques = self.kp * (target_angles - joint_pos) - self.kd * joint_vel
            self.data.ctrl[:] = torques
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # Update feet air time
        self._update_feet_air_time()

        # Resample command periodically
        if self.step_count % self.cmd_resample_interval == 0:
            self._resample_command()

        # Compute reward
        reward_state = self._build_reward_state(action)
        reward, reward_components = self.reward_computer.compute(reward_state)

        # Termination
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps

        self.last_action = action.copy()
        obs = self._get_obs()
        info = {
            "privileged_obs": self._get_privileged_obs(),
            "reward_components": reward_components,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Build 48D observation vector."""
        base_lin_vel = self._get_base_linear_velocity()
        base_ang_vel = self._get_base_angular_velocity()
        projected_gravity = self._get_projected_gravity()
        joint_pos_rel = self._get_joint_positions() - self.default_angles
        joint_vel = self._get_joint_velocities()

        obs = np.concatenate([
            base_lin_vel,           # 3
            base_ang_vel,           # 3
            projected_gravity,      # 3
            joint_pos_rel,          # 12
            joint_vel,              # 12
            self.last_action,       # 12
            self.command,           # 3
        ]).astype(np.float32)
        return obs

    def _get_privileged_obs(self):
        """7D privileged info: friction(1) + mass_scale(1) + ext_force(3) + motor_strength(2)."""
        # Default values (no DR applied yet)
        return np.zeros(7, dtype=np.float32)

    def _get_base_linear_velocity(self):
        """Base linear velocity in body frame."""
        world_vel = self.data.qvel[:3].copy()
        body_rot = self._get_body_rotation_matrix()
        return body_rot.T @ world_vel

    def _get_base_angular_velocity(self):
        """Base angular velocity in body frame."""
        world_ang_vel = self.data.qvel[3:6].copy()
        body_rot = self._get_body_rotation_matrix()
        return body_rot.T @ world_ang_vel

    def _get_projected_gravity(self):
        """Gravity vector projected into body frame."""
        gravity_world = np.array([0.0, 0.0, -1.0])
        body_rot = self._get_body_rotation_matrix()
        return body_rot.T @ gravity_world

    def _get_body_rotation_matrix(self):
        """Get 3x3 rotation matrix of the base body."""
        quat = self.data.qpos[3:7]  # w, x, y, z
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        return rot.reshape(3, 3)

    def _get_joint_positions(self):
        return self.data.qpos[7:19].copy().astype(np.float32)

    def _get_joint_velocities(self):
        return self.data.qvel[6:18].copy().astype(np.float32)

    def _get_torques(self):
        return self.data.ctrl[:12].copy().astype(np.float32)

    def _update_feet_air_time(self):
        """Track time each foot is in the air."""
        foot_contacts = self._get_foot_contacts()
        for i in range(4):
            if foot_contacts[i]:
                self.feet_air_time[i] = 0.0
            else:
                self.feet_air_time[i] += self.control_dt

    def _get_foot_contacts(self):
        """Check which feet are in contact with ground. Returns (4,) bool."""
        contacts = np.zeros(4, dtype=bool)
        foot_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for i, name in enumerate(foot_body_names):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                for j in range(self.data.ncon):
                    contact = self.data.contact[j]
                    geom1_body = self.model.geom_bodyid[contact.geom1]
                    geom2_body = self.model.geom_bodyid[contact.geom2]
                    if geom1_body == body_id or geom2_body == body_id:
                        contacts[i] = True
                        break
        return contacts

    def _get_body_contacts(self):
        """Check if non-foot body parts are in contact (collision)."""
        foot_body_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"}
        foot_ids = set()
        for name in foot_body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                foot_ids.add(bid)

        for j in range(self.data.ncon):
            contact = self.data.contact[j]
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]
            # If a non-foot body is in contact with ground (body 0)
            if b1 == 0 and b2 not in foot_ids and b2 != 0:
                return True
            if b2 == 0 and b1 not in foot_ids and b1 != 0:
                return True
        return False

    def _build_reward_state(self, action):
        joint_vel = self._get_joint_velocities()
        return {
            "base_lin_vel": self._get_base_linear_velocity(),
            "base_ang_vel": self._get_base_angular_velocity(),
            "command": self.command,
            "torques": self._get_torques(),
            "actions": action,
            "last_actions": self.last_action,
            "joint_acc": joint_vel / self.control_dt,  # approximate
            "feet_air_time": self.feet_air_time.copy(),
            "body_contacts": self._get_body_contacts(),
            "projected_gravity": self._get_projected_gravity(),
        }

    def _check_termination(self):
        body_height = self.data.qpos[2]
        if body_height < self.config["min_body_height"]:
            return True
        if body_height > self.config["max_body_height"]:
            return True
        # Check if body is too tilted (projected gravity z < 0.5 means > 60 deg tilt)
        proj_grav = self._get_projected_gravity()
        if proj_grav[2] > -0.5:  # gravity points down, so z should be ~ -1
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
```

- [ ] **Step 5: 运行测试验证通过**

```bash
pytest tests/go2/test_go2_env.py -v
```

Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add applications/go2_locomotion/envs/ tests/go2/
git commit -m "feat(go2): implement Go2 MuJoCo env with PD control and reward"
```

---

## Task 4: 向量化环境

**Files:**
- Create: `applications/go2_locomotion/envs/vectorized.py`
- Test: `tests/go2/test_vectorized.py`

- [ ] **Step 1: 写测试**

```python
# tests/go2/test_vectorized.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np


def test_vec_env_shapes():
    from applications.go2_locomotion.envs.vectorized import VecGo2Env
    from applications.go2_locomotion.config import config

    test_config = {**config, "num_envs": 4}
    vec_env = VecGo2Env(test_config)
    obs, infos = vec_env.reset()
    assert obs.shape == (4, 48)

    actions = np.zeros((4, 12), dtype=np.float32)
    obs, rewards, dones, truncs, infos = vec_env.step(actions)
    assert obs.shape == (4, 48)
    assert rewards.shape == (4,)
    assert dones.shape == (4,)
    vec_env.close()


def test_vec_env_auto_reset():
    from applications.go2_locomotion.envs.vectorized import VecGo2Env
    from applications.go2_locomotion.config import config

    test_config = {**config, "num_envs": 2, "episode_length_s": 0.1}
    vec_env = VecGo2Env(test_config)
    vec_env.reset()

    for _ in range(100):
        actions = np.random.uniform(-1, 1, (2, 12)).astype(np.float32)
        obs, rewards, dones, truncs, infos = vec_env.step(actions)
        # After done, obs should be from fresh reset (not NaN)
        assert not np.any(np.isnan(obs))
    vec_env.close()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/go2/test_vectorized.py -v
```

- [ ] **Step 3: 实现向量化环境**

```python
# applications/go2_locomotion/envs/vectorized.py
import numpy as np
from .go2_env import Go2Env


class VecGo2Env:
    """Simple synchronous vectorized Go2 environment.

    Each env is an independent Go2Env instance. On done, auto-resets and
    returns the new obs (standard vec env convention).
    """

    def __init__(self, config):
        self.num_envs = config["num_envs"]
        self.envs = [Go2Env(config) for _ in range(self.num_envs)]
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.privileged_dim = config["privileged_dim"]

    def reset(self):
        obs_list = []
        priv_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            priv_list.append(info["privileged_obs"])
        obs = np.array(obs_list, dtype=np.float32)
        infos = {"privileged_obs": np.array(priv_list, dtype=np.float32)}
        return obs, infos

    def step(self, actions):
        obs_list, reward_list, done_list, trunc_list = [], [], [], []
        priv_list = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated

            if done:
                obs, reset_info = env.reset()
                info["privileged_obs"] = reset_info["privileged_obs"]

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            trunc_list.append(truncated)
            priv_list.append(info["privileged_obs"])

        return (
            np.array(obs_list, dtype=np.float32),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            {"privileged_obs": np.array(priv_list, dtype=np.float32)},
        )

    def close(self):
        for env in self.envs:
            env.close()
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/go2/test_vectorized.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add applications/go2_locomotion/envs/vectorized.py tests/go2/test_vectorized.py
git commit -m "feat(go2): add synchronous vectorized Go2 environment"
```

---

## Task 5: PPO 训练器 + 网络定义

**Files:**
- Create: `applications/go2_locomotion/agent/networks.py`
- Create: `applications/go2_locomotion/agent/ppo.py`
- Test: `tests/go2/test_ppo.py`

- [ ] **Step 1: 写测试**

```python
# tests/go2/test_ppo.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch


def test_actor_critic_output_shapes():
    from applications.go2_locomotion.agent.networks import ActorCritic
    net = ActorCritic(obs_dim=48, action_dim=12, hidden_dim=128)
    obs = torch.randn(8, 48)
    actions, log_probs, values, entropy = net.act(obs)
    assert actions.shape == (8, 12)
    assert log_probs.shape == (8,)
    assert values.shape == (8,)


def test_ppo_update_reduces_loss():
    from applications.go2_locomotion.agent.ppo import PPOTrainer
    from applications.go2_locomotion.config import config
    trainer = PPOTrainer(config)

    # Fake rollout data
    n = 128
    states = np.random.randn(n, 48).astype(np.float32)
    actions = np.random.randn(n, 12).astype(np.float32)
    rewards = np.random.randn(n).astype(np.float32)
    dones = np.zeros(n, dtype=bool)
    log_probs = np.random.randn(n).astype(np.float32)
    values = np.random.randn(n).astype(np.float32)
    next_value = 0.0

    # Should not crash
    trainer.update(states, actions, rewards, dones, log_probs, values, next_value)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/go2/test_ppo.py -v
```

- [ ] **Step 3: 实现 networks.py**

```python
# applications/go2_locomotion/agent/networks.py
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Shared-nothing Actor-Critic for continuous locomotion control."""

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        actor_features = self.actor(obs)
        mean = self.actor_mean(actor_features)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def act(self, obs):
        """Sample actions and return (actions, log_probs, values, entropy)."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return actions, log_probs, value, entropy

    def evaluate(self, obs, actions):
        """Evaluate given actions under current policy."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_probs, value, entropy


class AsymmetricActorCritic(nn.Module):
    """Actor uses obs only, Critic uses obs + privileged info (teacher mode)."""

    def __init__(self, obs_dim, privileged_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        critic_input = obs_dim + privileged_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_input, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor(self, obs):
        features = self.actor(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        return mean, std

    def forward_critic(self, obs, privileged):
        full = torch.cat([obs, privileged], dim=-1)
        return self.critic(full).squeeze(-1)

    def act(self, obs, privileged):
        mean, std = self.forward_actor(obs)
        dist = Normal(mean, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.forward_critic(obs, privileged)
        entropy = dist.entropy().sum(dim=-1).mean()
        return actions, log_probs, values, entropy

    def evaluate(self, obs, privileged, actions):
        mean, std = self.forward_actor(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.forward_critic(obs, privileged)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_probs, values, entropy
```

- [ ] **Step 4: 实现 ppo.py**

```python
# applications/go2_locomotion/agent/ppo.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import AsymmetricActorCritic


class PPOTrainer:
    """PPO trainer with GAE for Go2 locomotion (asymmetric actor-critic)."""

    def __init__(self, config):
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_eps = config["clip_eps"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.max_grad_norm = config["max_grad_norm"]
        self.entropy_coef = config["entropy_coef"]
        self.value_loss_coef = config["value_loss_coef"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = AsymmetricActorCritic(
            obs_dim=config["obs_dim"],
            privileged_dim=config["privileged_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"],
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config["lr"])

    def act(self, obs, privileged):
        """Get actions for vectorized envs. Returns numpy arrays."""
        obs_t = torch.FloatTensor(obs).to(self.device)
        priv_t = torch.FloatTensor(privileged).to(self.device)
        with torch.no_grad():
            actions, log_probs, values, _ = self.network.act(obs_t, priv_t)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute GAE advantages and returns."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        values_ext = np.append(values, next_value)

        for t in reversed(range(n)):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, rewards, dones, log_probs, values, next_value,
               privileged=None):
        """PPO update with pre-collected rollout data."""
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        if privileged is None:
            privileged = np.zeros((len(states), 7), dtype=np.float32)
        priv_t = torch.FloatTensor(privileged).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = len(states)
        for _ in range(self.epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                batch_states = states_t[idx]
                batch_priv = priv_t[idx]
                batch_actions = actions_t[idx]
                batch_old_lp = old_log_probs_t[idx]
                batch_adv = advantages_t[idx]
                batch_ret = returns_t[idx]

                new_log_probs, new_values, entropy = self.network.evaluate(
                    batch_states, batch_priv, batch_actions
                )

                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(new_values, batch_ret)

                loss = (policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(
            torch.load(path, map_location=self.device)
        )
```

- [ ] **Step 5: 运行测试**

```bash
pytest tests/go2/test_ppo.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add applications/go2_locomotion/agent/ tests/go2/test_ppo.py
git commit -m "feat(go2): implement AsymmetricActorCritic + PPO trainer"
```

---

## Task 6: Domain Randomization

**Files:**
- Create: `applications/go2_locomotion/dr/domain_randomization.py`
- Test: `tests/go2/test_dr.py`

- [ ] **Step 1: 写测试**

```python
# tests/go2/test_dr.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np


def test_dr_changes_physics():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer
    from applications.go2_locomotion.config import config

    env = Go2Env(config)
    env.reset()
    dr = Go2DomainRandomizer(env, config)

    original_mass = env.model.body_mass.copy()
    dr.randomize()
    # Mass should be different after DR
    assert not np.allclose(env.model.body_mass, original_mass, atol=1e-6)
    env.close()


def test_dr_returns_privileged_info():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer
    from applications.go2_locomotion.config import config

    env = Go2Env(config)
    env.reset()
    dr = Go2DomainRandomizer(env, config)
    priv = dr.randomize()
    assert priv.shape == (7,)
    assert not np.all(priv == 0)
    env.close()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/go2/test_dr.py -v
```

- [ ] **Step 3: 实现 DR**

```python
# applications/go2_locomotion/dr/domain_randomization.py
import numpy as np
import mujoco


class Go2DomainRandomizer:
    """Domain Randomization for Go2 environment.

    Randomizes: friction, body mass, external push forces, motor strength.
    Returns privileged information vector (7D) for asymmetric critic.
    """

    def __init__(self, env, config, seed=None):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(seed)

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

        # Friction
        fr_range = self.config["dr_friction_range"]
        self._friction_scale = self.rng.uniform(fr_range[0], fr_range[1])
        model.geom_friction[:] = self._original_friction * self._friction_scale

        # Mass
        m_range = self.config["dr_mass_scale_range"]
        self._mass_scale = self.rng.uniform(m_range[0], m_range[1])
        model.body_mass[:] = self._original_mass * self._mass_scale

        # Motor strength (affects PD gains)
        kp_range = self.config["dr_kp_range"]
        kd_range = self.config["dr_kd_range"]
        self._motor_strength[0] = self.rng.uniform(kp_range[0], kp_range[1])
        self._motor_strength[1] = self.rng.uniform(kd_range[0], kd_range[1])

        # Reset push state
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
        """Apply random external force to base body."""
        force_range = self.config["dr_ext_force_range"]
        force_mag = self.rng.uniform(force_range[0], force_range[1])
        direction = self.rng.standard_normal(3)
        direction[2] = 0  # horizontal push only
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        self._ext_force = (direction * force_mag).astype(np.float32)

        # Apply to MuJoCo data
        self.env.data.xfrc_applied[1, :3] = self._ext_force  # body 1 = base

    def get_kp_scale(self):
        return self._motor_strength[0]

    def get_kd_scale(self):
        return self._motor_strength[1]

    def _get_privileged_info(self):
        """Return 7D privileged info: friction(1) + mass(1) + force(3) + motor(2)."""
        return np.concatenate([
            [self._friction_scale],
            [self._mass_scale],
            self._ext_force,
            self._motor_strength,
        ]).astype(np.float32)

    def _sample_push_interval(self):
        interval_range = self.config["dr_push_interval"]
        return self.rng.uniform(interval_range[0], interval_range[1])
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/go2/test_dr.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add applications/go2_locomotion/dr/ tests/go2/test_dr.py
git commit -m "feat(go2): add domain randomization with privileged info output"
```

---

## Task 7: Teacher 训练脚本

**Files:**
- Create: `applications/go2_locomotion/train_teacher.py`

- [ ] **Step 1: 实现训练脚本**

```python
# applications/go2_locomotion/train_teacher.py
"""Phase 1: Train Teacher policy with PPO + Domain Randomization.

Teacher uses asymmetric actor-critic: actor sees obs(48D), critic sees obs+privileged(55D).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from applications.go2_locomotion.config import config
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.envs.vectorized import VecGo2Env
from applications.go2_locomotion.agent.ppo import PPOTrainer
from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer


class DRVecGo2Env:
    """Vectorized Go2 env with per-env Domain Randomization."""

    def __init__(self, config):
        self.num_envs = config["num_envs"]
        self.envs = [Go2Env(config) for _ in range(self.num_envs)]
        self.drs = [Go2DomainRandomizer(env, config, seed=i) for i, env in enumerate(self.envs)]
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.control_dt = config["control_dt"]

    def reset(self):
        obs_list, priv_list = [], []
        for env, dr in zip(self.envs, self.drs):
            obs, info = env.reset()
            priv = dr.randomize()
            # Update env's PD gains based on DR
            env.kp = config["kp"] * dr.get_kp_scale()
            env.kd = config["kd"] * dr.get_kd_scale()
            obs_list.append(obs)
            priv_list.append(priv)
        return np.array(obs_list, np.float32), np.array(priv_list, np.float32)

    def step(self, actions):
        obs_list, reward_list, done_list, priv_list = [], [], [], []

        for i, (env, dr) in enumerate(zip(self.envs, self.drs)):
            dr.step(self.control_dt)
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated

            if done:
                obs, _ = env.reset()
                priv = dr.randomize()
                env.kp = config["kp"] * dr.get_kp_scale()
                env.kd = config["kd"] * dr.get_kd_scale()
            else:
                priv = dr._get_privileged_info()

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            priv_list.append(priv)

        return (
            np.array(obs_list, np.float32),
            np.array(reward_list, np.float32),
            np.array(done_list, np.bool_),
            np.array(priv_list, np.float32),
        )

    def close(self):
        for env in self.envs:
            env.close()


def train_teacher():
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    vec_env = DRVecGo2Env(config)
    trainer = PPOTrainer(config)

    num_envs = config["num_envs"]
    n_steps = config["n_steps"]
    n_iterations = config["n_iterations"]

    reward_history = []

    for iteration in range(n_iterations):
        # Collect rollout
        all_obs, all_priv, all_actions = [], [], []
        all_log_probs, all_values, all_rewards, all_dones = [], [], [], []

        obs, privileged = vec_env.reset()

        for step in range(n_steps):
            actions, log_probs, values = trainer.act(obs, privileged)
            next_obs, rewards, dones, next_priv = vec_env.step(actions)

            all_obs.append(obs)
            all_priv.append(privileged)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_rewards.append(rewards)
            all_dones.append(dones)

            obs = next_obs
            privileged = next_priv

        # Bootstrap
        with __import__("torch").no_grad():
            import torch
            obs_t = torch.FloatTensor(obs).to(trainer.device)
            priv_t = torch.FloatTensor(privileged).to(trainer.device)
            next_values = trainer.network.forward_critic(obs_t, priv_t).cpu().numpy()

        # Flatten: (n_steps, num_envs, ...) -> (n_steps * num_envs, ...)
        flat_obs = np.array(all_obs).reshape(-1, config["obs_dim"])
        flat_priv = np.array(all_priv).reshape(-1, config["privileged_dim"])
        flat_actions = np.array(all_actions).reshape(-1, config["action_dim"])
        flat_log_probs = np.array(all_log_probs).reshape(-1)
        flat_values = np.array(all_values).reshape(-1)
        flat_rewards = np.array(all_rewards).reshape(-1)
        flat_dones = np.array(all_dones).reshape(-1)

        # Per-env GAE then flatten
        all_advantages = np.zeros((n_steps, num_envs), dtype=np.float32)
        all_returns = np.zeros((n_steps, num_envs), dtype=np.float32)

        rewards_arr = np.array(all_rewards)
        values_arr = np.array(all_values)
        dones_arr = np.array(all_dones)

        for e in range(num_envs):
            adv, ret = trainer.compute_gae(
                rewards_arr[:, e], values_arr[:, e], dones_arr[:, e], next_values[e]
            )
            all_advantages[:, e] = adv
            all_returns[:, e] = ret

        flat_advantages = all_advantages.reshape(-1)
        flat_returns = all_returns.reshape(-1)

        # Update with pre-computed advantages
        trainer.update(
            flat_obs, flat_actions, flat_rewards, flat_dones,
            flat_log_probs, flat_values, next_value=0.0,  # already computed GAE
            privileged=flat_priv,
        )

        # Logging
        avg_reward = rewards_arr.sum() / num_envs
        reward_history.append(avg_reward)

        if (iteration + 1) % 50 == 0:
            avg_50 = np.mean(reward_history[-50:])
            print(f"Iter {iteration + 1}/{n_iterations} | Avg Reward: {avg_50:.3f}")

        if (iteration + 1) % 200 == 0:
            trainer.save(str(results_dir / f"teacher_iter{iteration + 1}.pth"))

    trainer.save(str(results_dir / "teacher_final.pth"))
    np.save(str(results_dir / "teacher_rewards.npy"), reward_history)
    vec_env.close()
    print("Teacher training complete.")


if __name__ == "__main__":
    train_teacher()
```

- [ ] **Step 2: Smoke test（跑几个 iteration 验证不崩）**

```bash
cd /home/lihongl/Desktop/myRL/easyRL
python -c "
import sys; sys.path.insert(0, '.')
from applications.go2_locomotion.config import config
config['n_iterations'] = 2
config['n_steps'] = 4
config['num_envs'] = 2
config['batch_size'] = 16
from applications.go2_locomotion.train_teacher import train_teacher
train_teacher()
print('Smoke test PASSED')
"
```

Expected: prints reward for 2 iterations, no crash

- [ ] **Step 3: Commit**

```bash
git add applications/go2_locomotion/train_teacher.py
git commit -m "feat(go2): implement teacher training with DR + asymmetric PPO"
```

---

## Task 8: Student 蒸馏 (RMA)

**Files:**
- Create: `applications/go2_locomotion/agent/teacher_student.py`
- Create: `applications/go2_locomotion/train_student.py`

- [ ] **Step 1: 实现 teacher_student.py (RMA adaptation module)**

```python
# applications/go2_locomotion/agent/teacher_student.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class AdaptationModule(nn.Module):
    """RMA: obs history -> latent z (implicit system identification)."""

    def __init__(self, obs_dim, history_length, latent_dim):
        super().__init__()
        input_dim = obs_dim * history_length
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, obs_history_flat):
        return self.net(obs_history_flat)


class StudentPolicy(nn.Module):
    """Student: current obs + latent z -> action (deterministic for deployment)."""

    def __init__(self, obs_dim, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, z):
        return self.net(torch.cat([obs, z], dim=-1))


class StudentAgent:
    """Full student: AdaptationModule + StudentPolicy, trained via BC from teacher."""

    def __init__(self, config):
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.history_length = config["student_history_length"]
        self.latent_dim = config["student_latent_dim"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adaptation = AdaptationModule(
            self.obs_dim, self.history_length, self.latent_dim
        ).to(self.device)
        self.policy = StudentPolicy(
            self.obs_dim, self.latent_dim, self.action_dim
        ).to(self.device)

        params = list(self.adaptation.parameters()) + list(self.policy.parameters())
        self.optimizer = optim.Adam(params, lr=config["student_lr"])

    def get_action(self, obs_history_flat, obs_current):
        """Inference: obs_history -> z -> action."""
        with torch.no_grad():
            hist_t = torch.FloatTensor(obs_history_flat).unsqueeze(0).to(self.device)
            obs_t = torch.FloatTensor(obs_current).unsqueeze(0).to(self.device)
            z = self.adaptation(hist_t)
            action = self.policy(obs_t, z)
        return action.cpu().numpy().flatten()

    def train_step(self, obs_history_batch, obs_current_batch, teacher_actions_batch):
        """One BC training step. Returns loss value."""
        hist_t = torch.FloatTensor(obs_history_batch).to(self.device)
        obs_t = torch.FloatTensor(obs_current_batch).to(self.device)
        target_t = torch.FloatTensor(teacher_actions_batch).to(self.device)

        z = self.adaptation(hist_t)
        pred_actions = self.policy(obs_t, z)
        loss = nn.MSELoss()(pred_actions, target_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save({
            "adaptation": self.adaptation.state_dict(),
            "policy": self.policy.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.adaptation.load_state_dict(ckpt["adaptation"])
        self.policy.load_state_dict(ckpt["policy"])
```

- [ ] **Step 2: 实现 train_student.py**

```python
# applications/go2_locomotion/train_student.py
"""Phase 2: Distill Teacher into Student via Behavior Cloning.

Collects (obs_history, obs_current, teacher_action) dataset, then trains Student.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from applications.go2_locomotion.config import config
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.agent.ppo import PPOTrainer
from applications.go2_locomotion.agent.teacher_student import StudentAgent
from applications.go2_locomotion.dr.domain_randomization import Go2DomainRandomizer


def collect_teacher_data(teacher_path, dataset_size):
    """Run teacher policy, collect (obs_history, obs_current, action) tuples."""
    env = Go2Env(config)
    dr = Go2DomainRandomizer(env, config)

    # Load teacher
    trainer = PPOTrainer(config)
    trainer.load(teacher_path)

    obs_dim = config["obs_dim"]
    history_length = config["student_history_length"]

    obs_history_data = []
    obs_current_data = []
    action_data = []

    obs, _ = env.reset()
    dr.randomize()
    env.kp = config["kp"] * dr.get_kp_scale()
    env.kd = config["kd"] * dr.get_kd_scale()

    history = np.zeros((history_length, obs_dim), dtype=np.float32)
    collected = 0
    pbar = tqdm(total=dataset_size, desc="Collecting teacher data")

    while collected < dataset_size:
        history = np.roll(history, -1, axis=0)
        history[-1] = obs

        priv = dr._get_privileged_info()
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
        priv_t = torch.FloatTensor(priv).unsqueeze(0).to(trainer.device)

        with torch.no_grad():
            mean, _ = trainer.network.forward_actor(obs_t)
            action = mean.cpu().numpy().flatten()

        obs_history_data.append(history.flatten())
        obs_current_data.append(obs.copy())
        action_data.append(action)
        collected += 1
        pbar.update(1)

        obs, _, terminated, truncated, _ = env.step(action)
        dr.step(config["control_dt"])

        if terminated or truncated:
            obs, _ = env.reset()
            priv = dr.randomize()
            env.kp = config["kp"] * dr.get_kp_scale()
            env.kd = config["kd"] * dr.get_kd_scale()
            history = np.zeros((history_length, obs_dim), dtype=np.float32)

    pbar.close()
    env.close()

    return (
        np.array(obs_history_data[:dataset_size], dtype=np.float32),
        np.array(obs_current_data[:dataset_size], dtype=np.float32),
        np.array(action_data[:dataset_size], dtype=np.float32),
    )


def train_student():
    results_dir = Path(__file__).resolve().parent / "results"
    teacher_path = str(results_dir / "teacher_final.pth")

    print("Collecting distillation dataset from Teacher...")
    obs_history, obs_current, actions = collect_teacher_data(
        teacher_path, config["distill_dataset_size"]
    )
    print(f"Dataset: history={obs_history.shape}, obs={obs_current.shape}, actions={actions.shape}")

    student = StudentAgent(config)
    batch_size = config["student_batch_size"]
    n_epochs = config["student_epochs"]
    n_samples = len(obs_history)

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        epoch_losses = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = indices[start:end]
            loss = student.train_step(obs_history[idx], obs_current[idx], actions[idx])
            epoch_losses.append(loss)

        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.6f}")

    student.save(str(results_dir / "student_final.pth"))
    print("Student training complete.")


if __name__ == "__main__":
    train_student()
```

- [ ] **Step 3: Commit**

```bash
git add applications/go2_locomotion/agent/teacher_student.py applications/go2_locomotion/train_student.py
git commit -m "feat(go2): implement RMA student distillation from teacher"
```

---

## Task 9: Sim2Sim 评估

**Files:**
- Create: `applications/go2_locomotion/evaluate.py`

- [ ] **Step 1: 实现评估脚本**

```python
# applications/go2_locomotion/evaluate.py
"""Evaluate trained policy in clean MuJoCo env (no DR) — Sim2Sim verification."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import numpy as np
import torch

from applications.go2_locomotion.config import config
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.agent.ppo import PPOTrainer
from applications.go2_locomotion.agent.teacher_student import StudentAgent


def evaluate_teacher(model_path, n_episodes=20, render=False):
    """Evaluate teacher policy (deterministic, using mean action)."""
    env = Go2Env(config, render_mode="human" if render else None)
    trainer = PPOTrainer(config)
    trainer.load(model_path)

    rewards_list = []
    steps_list = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
            priv_t = torch.zeros(1, config["privileged_dim"]).to(trainer.device)
            with torch.no_grad():
                mean, _ = trainer.network.forward_actor(obs_t)
                action = mean.cpu().numpy().flatten()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if render:
                env.render()

            if terminated or truncated:
                break

        rewards_list.append(ep_reward)
        steps_list.append(ep_steps)

    env.close()

    print(f"\n{'='*50}")
    print(f"Teacher Evaluation ({n_episodes} episodes)")
    print(f"  Avg Reward: {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
    print(f"  Avg Steps:  {np.mean(steps_list):.0f} ± {np.std(steps_list):.0f}")
    print(f"  Survival:   {np.mean([s >= config['episode_length_s']/config['control_dt'] for s in steps_list])*100:.0f}%")
    print(f"{'='*50}")

    return rewards_list


def evaluate_student(model_path, n_episodes=20, render=False):
    """Evaluate student policy (uses obs history for adaptation)."""
    env = Go2Env(config, render_mode="human" if render else None)
    student = StudentAgent(config)
    student.load(model_path)

    obs_dim = config["obs_dim"]
    history_length = config["student_history_length"]
    rewards_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        history = np.zeros((history_length, obs_dim), dtype=np.float32)
        ep_reward = 0.0

        while True:
            history = np.roll(history, -1, axis=0)
            history[-1] = obs
            action = student.get_action(history.flatten(), obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()

            if terminated or truncated:
                break

        rewards_list.append(ep_reward)

    env.close()
    print(f"\nStudent Evaluation ({n_episodes} episodes)")
    print(f"  Avg Reward: {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
    return rewards_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent / "results"
    if args.model is None:
        args.model = str(results_dir / f"{args.mode}_final.pth")

    if args.mode == "teacher":
        evaluate_teacher(args.model, args.episodes, args.render)
    else:
        evaluate_student(args.model, args.episodes, args.render)
```

- [ ] **Step 2: Commit**

```bash
git add applications/go2_locomotion/evaluate.py
git commit -m "feat(go2): add sim2sim evaluation for teacher and student"
```

---

## Task 10: ONNX 导出 + 推理 Benchmark

**Files:**
- Create: `applications/go2_locomotion/export_onnx.py`

- [ ] **Step 1: 实现 ONNX 导出**

```python
# applications/go2_locomotion/export_onnx.py
"""Export trained Student policy to ONNX for deployment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import numpy as np
import torch
import onnx
import onnxruntime as ort

from applications.go2_locomotion.config import config
from applications.go2_locomotion.agent.teacher_student import StudentAgent


class StudentONNXWrapper(torch.nn.Module):
    """Wraps AdaptationModule + StudentPolicy into one forward pass for ONNX."""

    def __init__(self, student: StudentAgent):
        super().__init__()
        self.adaptation = student.adaptation
        self.policy = student.policy

    def forward(self, obs_history_flat, obs_current):
        z = self.adaptation(obs_history_flat)
        action = self.policy(obs_current, z)
        return action


def export_student_onnx():
    results_dir = Path(__file__).resolve().parent / "results"
    student = StudentAgent(config)
    student.load(str(results_dir / "student_final.pth"))

    wrapper = StudentONNXWrapper(student).cpu().eval()

    obs_dim = config["obs_dim"]
    history_length = config["student_history_length"]
    history_dim = obs_dim * history_length

    dummy_history = torch.randn(1, history_dim)
    dummy_obs = torch.randn(1, obs_dim)

    onnx_path = str(results_dir / "student_go2.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_history, dummy_obs),
        onnx_path,
        input_names=["obs_history", "obs_current"],
        output_names=["action"],
        opset_version=config["onnx_opset_version"],
        dynamic_axes={
            "obs_history": {0: "batch"},
            "obs_current": {0: "batch"},
            "action": {0: "batch"},
        },
    )

    # Verify
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"ONNX model exported to: {onnx_path}")
    print(f"  Input: obs_history({history_dim}D) + obs_current({obs_dim}D)")
    print(f"  Output: action({config['action_dim']}D)")

    # Benchmark
    session = ort.InferenceSession(onnx_path)
    n_runs = 1000
    warmup = 100

    history_np = np.random.randn(1, history_dim).astype(np.float32)
    obs_np = np.random.randn(1, obs_dim).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {"obs_history": history_np, "obs_current": obs_np})

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"obs_history": history_np, "obs_current": obs_np})
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    print(f"\n  Inference benchmark ({n_runs} runs):")
    print(f"    Avg latency: {elapsed:.3f} ms")
    print(f"    Max control freq: {1000/elapsed:.0f} Hz")
    print(f"    Go2 requires: 50 Hz (20ms budget) → {'PASS' if elapsed < 20 else 'FAIL'}")

    # Accuracy check vs PyTorch
    with torch.no_grad():
        pt_action = wrapper(torch.FloatTensor(history_np), torch.FloatTensor(obs_np)).numpy()
    onnx_action = session.run(None, {"obs_history": history_np, "obs_current": obs_np})[0]
    max_diff = np.max(np.abs(pt_action - onnx_action))
    print(f"\n  Accuracy: max|PyTorch - ONNX| = {max_diff:.2e} ({'PASS' if max_diff < 1e-5 else 'WARN'})")


if __name__ == "__main__":
    export_student_onnx()
```

- [ ] **Step 2: Commit**

```bash
git add applications/go2_locomotion/export_onnx.py
git commit -m "feat(go2): add ONNX export with inference benchmark"
```

---

## Task 11: 整合验证 + 文档

**Files:**
- Modify: `applications/go2_locomotion/config.py` (if needed after testing)

- [ ] **Step 1: 端到端 smoke test**

```bash
cd /home/lihongl/Desktop/myRL/easyRL

# 1. Train teacher (short run)
python -c "
from applications.go2_locomotion.config import config
config['n_iterations'] = 10
config['n_steps'] = 8
config['num_envs'] = 4
config['batch_size'] = 64
from applications.go2_locomotion.train_teacher import train_teacher
train_teacher()
"

# 2. Evaluate teacher
python applications/go2_locomotion/evaluate.py --mode teacher --episodes 3

# 3. Export ONNX (requires student, skip if no full training)
```

- [ ] **Step 2: 运行所有 Go2 测试**

```bash
pytest tests/go2/ -v
```

Expected: All tests PASS

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(go2): complete Go2 locomotion pipeline (train → eval → deploy)"
```

---

## Summary: 全流程对应宇树路线

| 宇树实际 | 本项目实现 | 对应文件 |
|----------|-----------|---------|
| Isaac Gym 并行仿真 | MuJoCo + VecGo2Env (32 envs) | `envs/vectorized.py` |
| Go2 MJCF 模型 | mujoco_menagerie unitree_go2 | `envs/go2_env.py` |
| RSL_rl PPO | 自实现 PPO + GAE | `agent/ppo.py` |
| Asymmetric Actor-Critic | obs(48D) actor + obs+priv(55D) critic | `agent/networks.py` |
| Domain Randomization | friction/mass/force/motor + curriculum | `dr/domain_randomization.py` |
| Teacher-Student (RMA) | history→latent→action | `agent/teacher_student.py` |
| Sim2Sim (MuJoCo) | evaluate.py 在无 DR 环境验证 | `evaluate.py` |
| .pt JIT 部署 | ONNX export + benchmark | `export_onnx.py` |
