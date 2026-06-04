# RL+MPC Demo Implementation Plan

**Goal:** Build a closed-loop RL+MPC demo where PPO makes discrete driving decisions and CasADi-based MPC controllers (longitudinal + lateral) execute them across 7 highway-env scenarios.

**Architecture:** PPO observes flattened Kinematics (25-dim), outputs 1 of 5 discrete actions, action mapper converts to (v_ref, y_ref), longitudinal MPC (triple integrator) tracks v_ref outputting acceleration, lateral MPC (kinematic bicycle) tracks y_ref outputting steering. Both MPC controllers use CasADi+IPOPT.

**Tech Stack:** Python 3.9, PyTorch, CasADi 3.7, gymnasium, highway-env 1.10, numpy, matplotlib

**Known Issue:** `merge-v0` and `roundabout-v0` have a bug in `_rewards()` where `action in [0, 2]` fails with ContinuousAction (numpy array). The env wrapper must override `_rewards` or patch this.

---

## File Structure

```
applications/vehicle_control/rl_mpc/
├── __init__.py
├── config.py                   ← all hyperparameters
├── envs/
│   ├── __init__.py
│   ├── base_wrapper.py         ← abstract BaseEnvWrapper
│   ├── highway_wrapper.py      ← unified wrapper for 7 highway-env scenarios
│   └── carla_wrapper.py        ← CARLA stub
├── controller/
│   ├── __init__.py
│   ├── lon_mpc.py              ← longitudinal MPC
│   ├── lat_mpc.py              ← lateral MPC
│   └── action_mapper.py        ← discrete action → (v_ref, y_ref)
├── agent/
│   ├── __init__.py
│   └── ppo_decision.py         ← discrete PPO agent
├── train.py                    ← training loop
├── eval.py                     ← evaluation + plotting
└── docs/
    └── theory.md               ← bilingual design doc
```

---

### Task 1: Project Skeleton + Config

**Files:**
- Create: `applications/vehicle_control/rl_mpc/__init__.py`
- Create: `applications/vehicle_control/rl_mpc/config.py`
- Create: `applications/vehicle_control/rl_mpc/envs/__init__.py`
- Create: `applications/vehicle_control/rl_mpc/controller/__init__.py`
- Create: `applications/vehicle_control/rl_mpc/agent/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p applications/vehicle_control/rl_mpc/{envs,controller,agent,docs}
```

- [ ] **Step 2: Create `__init__.py` files**

```bash
touch applications/vehicle_control/rl_mpc/__init__.py
touch applications/vehicle_control/rl_mpc/envs/__init__.py
touch applications/vehicle_control/rl_mpc/controller/__init__.py
touch applications/vehicle_control/rl_mpc/agent/__init__.py
```

- [ ] **Step 3: Write config.py**

```python
# applications/vehicle_control/rl_mpc/config.py

config = {
    # Environment
    "env_id": "highway-v0",
    "vehicles_count": 10,
    "observation_vehicles": 5,
    "duration": 60,
    "policy_frequency": 5,

    # PPO
    "ppo_lr": 1e-3,
    "ppo_gamma": 0.98,
    "ppo_lmbda": 0.95,
    "ppo_eps": 0.2,
    "ppo_epochs": 10,
    "ppo_hidden_dim": 128,
    "n_episodes": 1000,

    # Action Mapper
    "delta_v": 5.0,         # m/s per FASTER/SLOWER
    "v_max": 40.0,          # m/s
    "v_min": 0.0,           # m/s
    "lane_width": 4.0,      # m

    # Longitudinal MPC
    "lon_N": 20,            # horizon steps
    "lon_dt": 0.1,          # s
    "lon_Q_v": 10.0,        # velocity tracking weight
    "lon_Q_a": 1.0,         # acceleration penalty
    "lon_R_j": 0.1,         # jerk penalty
    "a_min": -4.0,          # m/s²
    "a_max": 2.0,           # m/s²
    "j_min": -5.0,          # m/s³
    "j_max": 5.0,           # m/s³

    # Lateral MPC
    "lat_N": 15,            # horizon steps
    "lat_dt": 0.1,          # s
    "lat_Q_y": 10.0,        # lateral position tracking
    "lat_Q_psi": 5.0,       # heading tracking
    "lat_R_delta": 1.0,     # steering effort
    "delta_min": -0.5,      # rad
    "delta_max": 0.5,       # rad
    "delta_dot_max": 0.3,   # rad/s
    "wheelbase": 2.5,       # m (L)
}
```

- [ ] **Step 4: Commit**

```bash
git add applications/vehicle_control/rl_mpc/
git commit -m "feat(rl_mpc): add project skeleton and config"
```

---

### Task 2: Environment Base Wrapper + Highway Wrapper

**Files:**
- Create: `applications/vehicle_control/rl_mpc/envs/base_wrapper.py`
- Create: `applications/vehicle_control/rl_mpc/envs/highway_wrapper.py`

- [ ] **Step 1: Write base_wrapper.py**

```python
# applications/vehicle_control/rl_mpc/envs/base_wrapper.py
from abc import ABC, abstractmethod
import numpy as np


class BaseEnvWrapper(ABC):
    """Abstract base for all driving environment wrappers."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment, return flattened observation."""
        ...

    @abstractmethod
    def step(self, steering: float, acceleration: float):
        """Execute one step. Returns (obs, reward, done, info)."""
        ...

    @abstractmethod
    def get_ego_state(self) -> dict:
        """Return ego vehicle state: {x, y, vx, vy, heading, lane_index}."""
        ...

    @abstractmethod
    def close(self):
        ...
```

- [ ] **Step 2: Write highway_wrapper.py**

```python
# applications/vehicle_control/rl_mpc/envs/highway_wrapper.py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
import highway_env  # noqa: F401

from .base_wrapper import BaseEnvWrapper


SUPPORTED_ENVS = [
    "highway-v0",
    "merge-v0",
    "roundabout-v0",
    "intersection-v0",
    "intersection-v1",
    "racetrack-v0",
    "racetrack-large-v0",
]


class HighwayEnvWrapper(BaseEnvWrapper):
    """Unified wrapper for highway-env scenarios with ContinuousAction output."""

    def __init__(self, env_id: str = "highway-v0", render_mode: str = None, config_overrides: dict = None):
        assert env_id in SUPPORTED_ENVS, f"Unsupported env: {env_id}. Choose from {SUPPORTED_ENVS}"
        self.env_id = env_id
        self._env = gym.make(env_id, render_mode=render_mode)

        env_config = {
            "action": {"type": "ContinuousAction"},
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy", "heading"],
                "vehicles_count": 5,
                "absolute": False,
            },
        }
        if config_overrides:
            env_config.update(config_overrides)
        self._env.configure(env_config)

        # Patch _rewards for envs that have the action-in-list bug
        if env_id in ("merge-v0", "roundabout-v0"):
            self._patch_rewards()

        self._env = FlattenObservation(self._env)

    def _patch_rewards(self):
        """Fix merge/roundabout _rewards bug with ContinuousAction."""
        original_rewards = self._env.unwrapped._rewards

        def patched_rewards(action):
            # Convert continuous action to dummy int for reward calc
            rewards = {}
            try:
                rewards = original_rewards(action)
            except (ValueError, TypeError):
                rewards = original_rewards(1)  # IDLE equivalent
            return rewards

        self._env.unwrapped._rewards = patched_rewards

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        return obs.astype(np.float32)

    def step(self, steering: float, acceleration: float):
        action = np.array([steering, acceleration], dtype=np.float32)
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return obs.astype(np.float32), reward, done, info

    def get_ego_state(self) -> dict:
        vehicle = self._env.unwrapped.vehicle
        return {
            "x": vehicle.position[0],
            "y": vehicle.position[1],
            "vx": vehicle.speed * np.cos(vehicle.heading),
            "vy": vehicle.speed * np.sin(vehicle.heading),
            "speed": vehicle.speed,
            "heading": vehicle.heading,
            "lane_index": vehicle.lane_index,
        }

    def get_lane_center_y(self, lane_offset: int) -> float:
        """Get y-coordinate of lane center with offset from current lane.
        lane_offset: -1 for left, 0 for current, +1 for right.
        """
        vehicle = self._env.unwrapped.vehicle
        road = self._env.unwrapped.road
        current_lane = vehicle.lane_index
        target_lane_idx = current_lane[2] + lane_offset
        # Clamp to valid lane range
        lanes_count = len(road.network.graph.get(current_lane[0], {}).get(current_lane[1], []))
        target_lane_idx = max(0, min(target_lane_idx, lanes_count - 1))
        target_lane = road.network.get_lane((current_lane[0], current_lane[1], target_lane_idx))
        return target_lane.position(vehicle.position[0], 0)[1]

    @property
    def observation_dim(self) -> int:
        return 25  # 5 vehicles * 5 features

    def close(self):
        self._env.close()
```

- [ ] **Step 3: Verify wrapper works**

```bash
cd applications/vehicle_control/rl_mpc
python3 -c "
from envs.highway_wrapper import HighwayEnvWrapper
for env_id in ['highway-v0', 'intersection-v0', 'racetrack-v0']:
    w = HighwayEnvWrapper(env_id)
    obs = w.reset()
    obs2, r, done, info = w.step(0.0, 0.0)
    print(f'{env_id}: obs={obs.shape}, reward={r:.2f}')
    w.close()
"
```

Expected: all 3 print obs shape (25,) and a reward value without error.

- [ ] **Step 4: Commit**

```bash
git add applications/vehicle_control/rl_mpc/envs/
git commit -m "feat(rl_mpc): add highway-env wrapper with 7 scenario support"
```

---

### Task 3: CARLA Wrapper Stub

**Files:**
- Create: `applications/vehicle_control/rl_mpc/envs/carla_wrapper.py`

- [ ] **Step 1: Write carla_wrapper.py**

```python
# applications/vehicle_control/rl_mpc/envs/carla_wrapper.py
import numpy as np
from .base_wrapper import BaseEnvWrapper


class CarlaEnvWrapper(BaseEnvWrapper):
    """CARLA environment wrapper stub. To be implemented for CARLA integration.

    Prerequisites for implementation:
    - CARLA server running (carla-simulator)
    - carla Python API installed (pip install carla)
    - Sensor configuration (camera, lidar, etc.) defined
    """

    def __init__(self, town: str = "Town01", **kwargs):
        raise NotImplementedError(
            "CARLA wrapper not yet implemented. "
            "Install CARLA and implement this class following BaseEnvWrapper interface."
        )

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, steering: float, acceleration: float):
        raise NotImplementedError

    def get_ego_state(self) -> dict:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
```

- [ ] **Step 2: Commit**

```bash
git add applications/vehicle_control/rl_mpc/envs/carla_wrapper.py
git commit -m "feat(rl_mpc): add CARLA wrapper stub"
```

---

### Task 4: Action Mapper

**Files:**
- Create: `applications/vehicle_control/rl_mpc/controller/action_mapper.py`

- [ ] **Step 1: Write action_mapper.py**

```python
# applications/vehicle_control/rl_mpc/controller/action_mapper.py
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
        self.v_ref = 25.0  # initial reference speed (m/s)
        self.y_ref = 0.0   # initial reference lateral position

    def reset(self, initial_speed: float, initial_y: float):
        self.v_ref = initial_speed
        self.y_ref = initial_y

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
```

- [ ] **Step 2: Quick test**

```bash
python3 -c "
from controller.action_mapper import ActionMapper, FASTER, SLOWER, LANE_LEFT, IDLE
am = ActionMapper(delta_v=5.0, v_min=0.0, v_max=40.0, lane_width=4.0)
am.reset(25.0, 0.0)
print(am.map(FASTER))    # (30.0, 0.0)
print(am.map(FASTER))    # (35.0, 0.0)
print(am.map(FASTER))    # (40.0, 0.0) - clamped
print(am.map(SLOWER))    # (35.0, 0.0)
print(am.map(LANE_LEFT)) # (35.0, -4.0)
print(am.map(IDLE))      # (35.0, -4.0)
"
```

Expected: values matching comments above.

- [ ] **Step 3: Commit**

```bash
git add applications/vehicle_control/rl_mpc/controller/action_mapper.py
git commit -m "feat(rl_mpc): add action mapper (discrete → v_ref/y_ref)"
```

---

### Task 5: Longitudinal MPC

**Files:**
- Create: `applications/vehicle_control/rl_mpc/controller/lon_mpc.py`

- [ ] **Step 1: Write lon_mpc.py**

```python
# applications/vehicle_control/rl_mpc/controller/lon_mpc.py
import casadi as ca
import numpy as np


class LonMPC:
    """Longitudinal MPC using triple integrator model.

    State: [s, v, a] (position, velocity, acceleration)
    Control: j (jerk)
    Output: a_des (desired acceleration)
    """

    def __init__(self, N: int, dt: float, Q_v: float, Q_a: float, R_j: float,
                 a_min: float, a_max: float, j_min: float, j_max: float,
                 v_min: float, v_max: float):
        self.N = N
        self.dt = dt

        # Build CasADi optimization problem
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(3, N + 1)  # states [s; v; a] over horizon
        U = opti.variable(1, N)      # control (jerk) over horizon
        p_v_ref = opti.parameter()   # reference velocity
        p_x0 = opti.parameter(3)     # initial state

        # Cost function
        cost = 0
        for k in range(N):
            cost += Q_v * (X[1, k] - p_v_ref) ** 2  # velocity tracking
            cost += Q_a * X[2, k] ** 2               # acceleration penalty
            cost += R_j * U[0, k] ** 2               # jerk penalty
        cost += Q_v * (X[1, N] - p_v_ref) ** 2       # terminal velocity cost

        opti.minimize(cost)

        # Dynamics constraints (exact discretization of triple integrator)
        for k in range(N):
            s_next = X[0, k] + dt * X[1, k] + 0.5 * dt**2 * X[2, k] + (1.0/6.0) * dt**3 * U[0, k]
            v_next = X[1, k] + dt * X[2, k] + 0.5 * dt**2 * U[0, k]
            a_next = X[2, k] + dt * U[0, k]
            opti.subject_to(X[0, k+1] == s_next)
            opti.subject_to(X[1, k+1] == v_next)
            opti.subject_to(X[2, k+1] == a_next)

        # Initial state constraint
        opti.subject_to(X[:, 0] == p_x0)

        # Box constraints
        opti.subject_to(opti.bounded(v_min, X[1, :], v_max))
        opti.subject_to(opti.bounded(a_min, X[2, :], a_max))
        opti.subject_to(opti.bounded(j_min, U[0, :], j_max))

        # Solver options
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.warm_start_init_point": "yes"}
        opti.solver("ipopt", opts)

        self._opti = opti
        self._X = X
        self._U = U
        self._p_v_ref = p_v_ref
        self._p_x0 = p_x0
        self._sol = None

    def solve(self, s: float, v: float, a: float, v_ref: float) -> float:
        """Solve MPC and return desired acceleration.

        Args:
            s: current position
            v: current velocity
            a: current acceleration
            v_ref: reference velocity

        Returns:
            a_des: desired acceleration for next step
        """
        self._opti.set_value(self._p_x0, [s, v, a])
        self._opti.set_value(self._p_v_ref, v_ref)

        # Warm start from previous solution
        if self._sol is not None:
            self._opti.set_initial(self._X, self._sol.value(self._X))
            self._opti.set_initial(self._U, self._sol.value(self._U))

        try:
            self._sol = self._opti.solve()
            # Return acceleration at next step (after applying first jerk)
            a_des = float(self._sol.value(self._X[2, 1]))
        except RuntimeError:
            # Solver failed — return current acceleration (safe fallback)
            a_des = a
            self._sol = None

        return a_des
```

- [ ] **Step 2: Verify solver works**

```bash
python3 -c "
from controller.lon_mpc import LonMPC
mpc = LonMPC(N=20, dt=0.1, Q_v=10.0, Q_a=1.0, R_j=0.1,
             a_min=-4.0, a_max=2.0, j_min=-5.0, j_max=5.0,
             v_min=0.0, v_max=40.0)
# Start at v=20, want v_ref=30
a_des = mpc.solve(s=0.0, v=20.0, a=0.0, v_ref=30.0)
print(f'a_des = {a_des:.3f} m/s² (should be positive, accelerating)')
# Start at v=30, want v_ref=20
a_des = mpc.solve(s=0.0, v=30.0, a=0.0, v_ref=20.0)
print(f'a_des = {a_des:.3f} m/s² (should be negative, decelerating)')
"
```

Expected: first call gives positive acceleration (~2.0), second gives negative (~-4.0 or less aggressive).

- [ ] **Step 3: Commit**

```bash
git add applications/vehicle_control/rl_mpc/controller/lon_mpc.py
git commit -m "feat(rl_mpc): add longitudinal MPC (CasADi, triple integrator)"
```

---

### Task 6: Lateral MPC

**Files:**
- Create: `applications/vehicle_control/rl_mpc/controller/lat_mpc.py`

- [ ] **Step 1: Write lat_mpc.py**

```python
# applications/vehicle_control/rl_mpc/controller/lat_mpc.py
import casadi as ca
import numpy as np


class LatMPC:
    """Lateral MPC using kinematic bicycle model.

    State: [x, y, psi] (position_x, position_y, heading)
    Control: delta (front wheel steering angle)
    """

    def __init__(self, N: int, dt: float, L: float,
                 Q_y: float, Q_psi: float, R_delta: float,
                 delta_min: float, delta_max: float, delta_dot_max: float):
        self.N = N
        self.dt = dt
        self.L = L

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(3, N + 1)  # [x, y, psi]
        U = opti.variable(1, N)      # delta (steering)

        # Parameters
        p_x0 = opti.parameter(3)     # initial state [x0, y0, psi0]
        p_v = opti.parameter()       # longitudinal velocity (from lon controller)
        p_y_ref = opti.parameter()   # reference lateral position
        p_psi_ref = opti.parameter() # reference heading
        p_delta_prev = opti.parameter()  # previous steering angle (for rate limit)

        # Cost
        cost = 0
        for k in range(N):
            cost += Q_y * (X[1, k] - p_y_ref) ** 2
            cost += Q_psi * (X[2, k] - p_psi_ref) ** 2
            cost += R_delta * U[0, k] ** 2
        cost += Q_y * (X[1, N] - p_y_ref) ** 2
        cost += Q_psi * (X[2, N] - p_psi_ref) ** 2

        opti.minimize(cost)

        # Kinematic bicycle dynamics (nonlinear)
        for k in range(N):
            x_next = X[0, k] + dt * p_v * ca.cos(X[2, k])
            y_next = X[1, k] + dt * p_v * ca.sin(X[2, k])
            psi_next = X[2, k] + dt * p_v / L * ca.tan(U[0, k])
            opti.subject_to(X[0, k+1] == x_next)
            opti.subject_to(X[1, k+1] == y_next)
            opti.subject_to(X[2, k+1] == psi_next)

        # Initial state
        opti.subject_to(X[:, 0] == p_x0)

        # Steering angle bounds
        opti.subject_to(opti.bounded(delta_min, U[0, :], delta_max))

        # Steering rate constraint
        opti.subject_to(opti.bounded(
            -delta_dot_max * dt, U[0, 0] - p_delta_prev, delta_dot_max * dt))
        for k in range(N - 1):
            opti.subject_to(opti.bounded(
                -delta_dot_max * dt, U[0, k+1] - U[0, k], delta_dot_max * dt))

        # Solver
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.warm_start_init_point": "yes"}
        opti.solver("ipopt", opts)

        self._opti = opti
        self._X = X
        self._U = U
        self._p_x0 = p_x0
        self._p_v = p_v
        self._p_y_ref = p_y_ref
        self._p_psi_ref = p_psi_ref
        self._p_delta_prev = p_delta_prev
        self._sol = None
        self._delta_prev = 0.0

    def solve(self, x: float, y: float, psi: float, v: float,
              y_ref: float, psi_ref: float = 0.0) -> float:
        """Solve lateral MPC and return steering angle.

        Args:
            x: current x position
            y: current y position
            psi: current heading angle
            v: current longitudinal velocity
            y_ref: reference lateral position
            psi_ref: reference heading (default 0 = straight)

        Returns:
            delta: front wheel steering angle (rad)
        """
        # Avoid division by zero at very low speed
        v_safe = max(v, 1.0)

        self._opti.set_value(self._p_x0, [x, y, psi])
        self._opti.set_value(self._p_v, v_safe)
        self._opti.set_value(self._p_y_ref, y_ref)
        self._opti.set_value(self._p_psi_ref, psi_ref)
        self._opti.set_value(self._p_delta_prev, self._delta_prev)

        # Warm start
        if self._sol is not None:
            self._opti.set_initial(self._X, self._sol.value(self._X))
            self._opti.set_initial(self._U, self._sol.value(self._U))

        try:
            self._sol = self._opti.solve()
            delta = float(self._sol.value(self._U[0, 0]))
        except RuntimeError:
            delta = self._delta_prev
            self._sol = None

        self._delta_prev = delta
        return delta

    def reset(self):
        """Reset internal state for new episode."""
        self._sol = None
        self._delta_prev = 0.0
```

- [ ] **Step 2: Verify solver works**

```bash
python3 -c "
from controller.lat_mpc import LatMPC
mpc = LatMPC(N=15, dt=0.1, L=2.5, Q_y=10.0, Q_psi=5.0, R_delta=1.0,
             delta_min=-0.5, delta_max=0.5, delta_dot_max=0.3)
# At y=0, want y_ref=4 (one lane to the right)
delta = mpc.solve(x=0.0, y=0.0, psi=0.0, v=25.0, y_ref=4.0)
print(f'delta = {delta:.4f} rad (should be positive, steering right)')
# At y=4, want y_ref=4 (on target)
delta = mpc.solve(x=25.0, y=4.0, psi=0.0, v=25.0, y_ref=4.0)
print(f'delta = {delta:.4f} rad (should be ~0, on target)')
"
```

Expected: first positive steering, second near zero.

- [ ] **Step 3: Commit**

```bash
git add applications/vehicle_control/rl_mpc/controller/lat_mpc.py
git commit -m "feat(rl_mpc): add lateral MPC (CasADi, kinematic bicycle)"
```

---

### Task 7: PPO Decision Agent

**Files:**
- Create: `applications/vehicle_control/rl_mpc/agent/ppo_decision.py`

- [ ] **Step 1: Write ppo_decision.py**

```python
# applications/vehicle_control/rl_mpc/agent/ppo_decision.py
import numpy as np
import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPODecisionAgent:
    """Discrete PPO agent for high-level driving decisions."""

    def __init__(self, state_dim: int, action_dim: int = 5, hidden_dim: int = 128,
                 lr: float = 1e-3, gamma: float = 0.98, lmbda: float = 0.95,
                 eps: float = 0.2, epochs: int = 10):
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def take_action(self, state: np.ndarray) -> int:
        state_t = torch.tensor([state], dtype=torch.float).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def update(self, transition_dict: dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # TD targets and advantages
        td_target = rewards + self.gamma * self.critic(next_states).detach() * (1 - dones)
        td_delta = (td_target - self.critic(states)).detach()
        advantage = self._compute_advantage(td_delta)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def _compute_advantage(self, td_delta: torch.Tensor) -> torch.Tensor:
        td_delta_np = td_delta.cpu().detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta_np[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
```

- [ ] **Step 2: Verify agent creation and action**

```bash
python3 -c "
from agent.ppo_decision import PPODecisionAgent
import numpy as np
agent = PPODecisionAgent(state_dim=25, action_dim=5)
obs = np.random.randn(25).astype(np.float32)
action = agent.take_action(obs)
print(f'action={action} (should be int 0-4)')
assert 0 <= action <= 4
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add applications/vehicle_control/rl_mpc/agent/
git commit -m "feat(rl_mpc): add discrete PPO decision agent"
```

---

### Task 8: Training Loop

**Files:**
- Create: `applications/vehicle_control/rl_mpc/train.py`

- [ ] **Step 1: Write train.py**

```python
# applications/vehicle_control/rl_mpc/train.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config
from envs.highway_wrapper import HighwayEnvWrapper
from controller.lon_mpc import LonMPC
from controller.lat_mpc import LatMPC
from controller.action_mapper import ActionMapper
from agent.ppo_decision import PPODecisionAgent


def train(env_id: str = None, n_episodes: int = None, render: bool = False):
    env_id = env_id or config["env_id"]
    n_episodes = n_episodes or config["n_episodes"]

    # Environment
    render_mode = "human" if render else None
    env = HighwayEnvWrapper(env_id, render_mode=render_mode)

    # PPO agent
    agent = PPODecisionAgent(
        state_dim=env.observation_dim,
        action_dim=5,
        hidden_dim=config["ppo_hidden_dim"],
        lr=config["ppo_lr"],
        gamma=config["ppo_gamma"],
        lmbda=config["ppo_lmbda"],
        eps=config["ppo_eps"],
        epochs=config["ppo_epochs"],
    )

    # MPC controllers
    lon_mpc = LonMPC(
        N=config["lon_N"], dt=config["lon_dt"],
        Q_v=config["lon_Q_v"], Q_a=config["lon_Q_a"], R_j=config["lon_R_j"],
        a_min=config["a_min"], a_max=config["a_max"],
        j_min=config["j_min"], j_max=config["j_max"],
        v_min=config["v_min"], v_max=config["v_max"],
    )

    lat_mpc = LatMPC(
        N=config["lat_N"], dt=config["lat_dt"], L=config["wheelbase"],
        Q_y=config["lat_Q_y"], Q_psi=config["lat_Q_psi"], R_delta=config["lat_R_delta"],
        delta_min=config["delta_min"], delta_max=config["delta_max"],
        delta_dot_max=config["delta_dot_max"],
    )

    # Action mapper
    mapper = ActionMapper(
        delta_v=config["delta_v"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        lane_width=config["lane_width"],
    )

    # Training
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    return_list = []

    for episode in tqdm(range(n_episodes), desc=f"Training PPO+MPC on {env_id}"):
        obs = env.reset()
        ego = env.get_ego_state()
        mapper.reset(ego["speed"], ego["y"])
        lat_mpc.reset()

        transition_dict = {
            'states': [], 'actions': [], 'next_states': [],
            'rewards': [], 'dones': []
        }
        episode_reward = 0
        done = False
        a_current = 0.0  # track current acceleration for lon MPC

        while not done:
            # PPO decision
            action = agent.take_action(obs)

            # Map to references
            v_ref, y_ref = mapper.map(
                action,
                current_y=ego["y"],
                lane_center_fn=env.get_lane_center_y,
            )

            # MPC control
            a_des = lon_mpc.solve(s=ego["x"], v=ego["speed"], a=a_current, v_ref=v_ref)
            delta = lat_mpc.solve(
                x=ego["x"], y=ego["y"], psi=ego["heading"],
                v=ego["speed"], y_ref=y_ref,
            )

            # Normalize to highway-env action range [-1, 1]
            steering_normalized = np.clip(delta / config["delta_max"], -1.0, 1.0)
            accel_normalized = np.clip(a_des / abs(config["a_min"]), -1.0, 1.0)

            # Step environment
            next_obs, reward, done, info = env.step(steering_normalized, accel_normalized)

            # Store transition
            transition_dict['states'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_obs)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            obs = next_obs
            ego = env.get_ego_state()
            a_current = a_des
            episode_reward += reward

        # Update PPO
        agent.update(transition_dict)
        return_list.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg = np.mean(return_list[-50:])
            print(f"  Episode {episode+1}/{n_episodes} | Avg Reward (50): {avg:.2f}")

    # Save
    agent.save(str(results_dir / f"ppo_mpc_{env_id.replace('-', '_')}.pth"))
    np.save(str(results_dir / f"returns_{env_id.replace('-', '_')}.npy"), return_list)

    env.close()
    print(f"Training complete. Results saved to {results_dir}/")
    return return_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None, help="Environment ID")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(env_id=args.env, n_episodes=args.episodes, render=args.render)
```

- [ ] **Step 2: Smoke test (5 episodes, no render)**

```bash
python3 train.py --env highway-v0 --episodes 5
```

Expected: runs 5 episodes without crashing, prints average reward.

- [ ] **Step 3: Commit**

```bash
git add applications/vehicle_control/rl_mpc/train.py
git commit -m "feat(rl_mpc): add training loop (PPO + MPC closed-loop)"
```

---

### Task 9: Evaluation + Visualization

**Files:**
- Create: `applications/vehicle_control/rl_mpc/eval.py`

- [ ] **Step 1: Write eval.py**

```python
# applications/vehicle_control/rl_mpc/eval.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from config import config
from envs.highway_wrapper import HighwayEnvWrapper
from controller.lon_mpc import LonMPC
from controller.lat_mpc import LatMPC
from controller.action_mapper import ActionMapper, ACTION_NAMES
from agent.ppo_decision import PPODecisionAgent


def evaluate(env_id: str = None, model_path: str = None, n_episodes: int = 5, render: bool = True):
    env_id = env_id or config["env_id"]
    results_dir = Path(__file__).resolve().parent / "results"

    if model_path is None:
        model_path = str(results_dir / f"ppo_mpc_{env_id.replace('-', '_')}.pth")

    render_mode = "human" if render else None
    env = HighwayEnvWrapper(env_id, render_mode=render_mode)

    agent = PPODecisionAgent(
        state_dim=env.observation_dim, action_dim=5,
        hidden_dim=config["ppo_hidden_dim"],
    )
    agent.load(model_path)

    lon_mpc = LonMPC(
        N=config["lon_N"], dt=config["lon_dt"],
        Q_v=config["lon_Q_v"], Q_a=config["lon_Q_a"], R_j=config["lon_R_j"],
        a_min=config["a_min"], a_max=config["a_max"],
        j_min=config["j_min"], j_max=config["j_max"],
        v_min=config["v_min"], v_max=config["v_max"],
    )

    lat_mpc = LatMPC(
        N=config["lat_N"], dt=config["lat_dt"], L=config["wheelbase"],
        Q_y=config["lat_Q_y"], Q_psi=config["lat_Q_psi"], R_delta=config["lat_R_delta"],
        delta_min=config["delta_min"], delta_max=config["delta_max"],
        delta_dot_max=config["delta_dot_max"],
    )

    mapper = ActionMapper(
        delta_v=config["delta_v"], v_min=config["v_min"],
        v_max=config["v_max"], lane_width=config["lane_width"],
    )

    all_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        ego = env.get_ego_state()
        mapper.reset(ego["speed"], ego["y"])
        lat_mpc.reset()

        episode_reward = 0
        done = False
        a_current = 0.0

        # Logging for plots
        log_v, log_v_ref = [], []
        log_y, log_y_ref = [], []
        log_actions = []

        while not done:
            action = agent.take_action(obs)
            v_ref, y_ref = mapper.map(action, current_y=ego["y"], lane_center_fn=env.get_lane_center_y)

            a_des = lon_mpc.solve(s=ego["x"], v=ego["speed"], a=a_current, v_ref=v_ref)
            delta = lat_mpc.solve(x=ego["x"], y=ego["y"], psi=ego["heading"], v=ego["speed"], y_ref=y_ref)

            steering_normalized = np.clip(delta / config["delta_max"], -1.0, 1.0)
            accel_normalized = np.clip(a_des / abs(config["a_min"]), -1.0, 1.0)

            next_obs, reward, done, info = env.step(steering_normalized, accel_normalized)

            # Log
            log_v.append(ego["speed"])
            log_v_ref.append(v_ref)
            log_y.append(ego["y"])
            log_y_ref.append(y_ref)
            log_actions.append(action)

            obs = next_obs
            ego = env.get_ego_state()
            a_current = a_des
            episode_reward += reward

        all_rewards.append(episode_reward)
        print(f"Episode {ep+1}: reward = {episode_reward:.2f}")

        # Plot tracking performance for last episode
        if ep == n_episodes - 1:
            _plot_tracking(log_v, log_v_ref, log_y, log_y_ref, log_actions, env_id)

    env.close()
    print(f"\nAverage reward over {n_episodes} episodes: {np.mean(all_rewards):.2f}")


def _plot_tracking(log_v, log_v_ref, log_y, log_y_ref, log_actions, env_id):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = np.arange(len(log_v))

    axes[0].plot(t, log_v, label="v_actual")
    axes[0].plot(t, log_v_ref, "--", label="v_ref")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend()
    axes[0].set_title(f"RL+MPC Tracking Performance ({env_id})")

    axes[1].plot(t, log_y, label="y_actual")
    axes[1].plot(t, log_y_ref, "--", label="y_ref")
    axes[1].set_ylabel("Lateral Position (m)")
    axes[1].legend()

    axes[2].step(t, log_actions, where="post")
    axes[2].set_ylabel("PPO Action")
    axes[2].set_xlabel("Step")
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(ACTION_NAMES)

    plt.tight_layout()
    plt.savefig(f"results/tracking_{env_id.replace('-', '_')}.png", dpi=150)
    plt.show()


def plot_training_curve(env_id: str = None):
    env_id = env_id or config["env_id"]
    results_dir = Path(__file__).resolve().parent / "results"
    returns = np.load(str(results_dir / f"returns_{env_id.replace('-', '_')}.npy"))

    plt.figure(figsize=(10, 5))
    plt.plot(returns, alpha=0.3, label="Episode Return")
    # Moving average
    window = 50
    if len(returns) >= window:
        ma = np.convolve(returns, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(returns)), ma, label=f"Moving Avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"PPO+MPC Training Curve ({env_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(results_dir / f"training_curve_{env_id.replace('-', '_')}.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--plot-training", action="store_true")
    args = parser.parse_args()

    if args.plot_training:
        plot_training_curve(args.env)
    else:
        evaluate(env_id=args.env, model_path=args.model,
                 n_episodes=args.episodes, render=not args.no_render)
```

- [ ] **Step 2: Commit**

```bash
git add applications/vehicle_control/rl_mpc/eval.py
git commit -m "feat(rl_mpc): add evaluation and visualization"
```

---

### Task 10: Theory Documentation

**Files:**
- Create: `applications/vehicle_control/rl_mpc/docs/theory.md`

- [ ] **Step 1: Write theory.md**

Write bilingual (English first, then Chinese) documentation covering:
- Why RL+MPC: separation of decision and control
- PPO for discrete decisions: action space design
- Longitudinal MPC: triple integrator model, cost function, constraints
- Lateral MPC: kinematic bicycle model, steering rate limits
- Action mapping strategy
- Architecture diagram (same as in spec)

The document should follow the bilingual-markdown format (English section → `---` → Chinese section).

- [ ] **Step 2: Commit**

```bash
git add applications/vehicle_control/rl_mpc/docs/theory.md
git commit -m "docs(rl_mpc): add theory documentation"
```

---

### Task 11: Integration Test (Full Pipeline)

- [ ] **Step 1: Run full training for 20 episodes on highway-v0**

```bash
cd applications/vehicle_control/rl_mpc
python3 train.py --env highway-v0 --episodes 20
```

Expected: completes without error, prints average rewards, saves model file.

- [ ] **Step 2: Run evaluation**

```bash
python3 eval.py --env highway-v0 --episodes 3 --no-render
```

Expected: loads saved model, runs 3 episodes, prints rewards, generates tracking plot.

- [ ] **Step 3: Test on merge-v0 (verifies reward patch)**

```bash
python3 train.py --env merge-v0 --episodes 10
```

Expected: completes without the `ValueError: truth value of array` error.

- [ ] **Step 4: Test on racetrack-v0**

```bash
python3 train.py --env racetrack-v0 --episodes 10
```

Expected: runs without error. Racetrack has curves so lateral MPC gets exercised.

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A applications/vehicle_control/rl_mpc/
git commit -m "fix(rl_mpc): integration test fixes"
```

---

## Execution Order Summary

| Task | Component | Depends On |
|------|-----------|------------|
| 1 | Skeleton + Config | — |
| 2 | Highway Wrapper | 1 |
| 3 | CARLA Stub | 1 |
| 4 | Action Mapper | 1 |
| 5 | Longitudinal MPC | 1 |
| 6 | Lateral MPC | 1 |
| 7 | PPO Agent | 1 |
| 8 | Training Loop | 2, 4, 5, 6, 7 |
| 9 | Evaluation | 8 |
| 10 | Theory Docs | — |
| 11 | Integration Test | 8, 9 |

Tasks 2–7 can be done in parallel. Task 8 depends on all of them. Tasks 3 and 10 are independent of everything else.
