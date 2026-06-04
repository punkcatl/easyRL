# Reward Shaping Implementation Plan

**Goal:** Build a reward shaping tutorial with 4 experiments (sparse vs dense, potential-based shaping, multi-objective weighted, reward hacking cases) on MuJoCo and highway-env, with comparison plots and theory documentation.

**Architecture:** Reward functions implemented as pluggable wrappers. Each experiment trains PPO with different reward configurations, logs results, and produces comparison plots. Reward hacking cases reproduce bad behavior then demonstrate fixes.

**Tech Stack:** Python 3.9, PyTorch, gymnasium[mujoco], highway-env, numpy, matplotlib

---

## File Structure

```
applications/reward_shaping/
├── __init__.py
├── config.py
├── rewards/
│   ├── __init__.py
│   ├── sparse.py
│   ├── dense.py
│   ├── potential_based.py
│   └── multi_objective.py
├── hacking/
│   ├── __init__.py
│   ├── ant_rolling.py
│   ├── hopper_jumping.py
│   ├── humanoid_sliding.py
│   ├── highway_lane_spam.py
│   └── highway_parking.py
├── experiments/
│   ├── run_sparse_vs_dense.py
│   ├── run_potential_shaping.py
│   ├── run_multi_objective.py
│   ├── run_hacking_cases.py
│   └── plot_comparison.py
├── results/
└── docs/
    └── theory.md
```

---

### Task 1: Project Skeleton + Config

**Files:**
- Create: `applications/reward_shaping/__init__.py`
- Create: `applications/reward_shaping/config.py`
- Create: `applications/reward_shaping/rewards/__init__.py`
- Create: `applications/reward_shaping/hacking/__init__.py`
- Create: `applications/reward_shaping/experiments/__init__.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p applications/reward_shaping/{rewards,hacking,experiments,results,docs}
```

- [ ] **Step 2: Create `__init__.py` files**

```bash
touch applications/reward_shaping/__init__.py
touch applications/reward_shaping/rewards/__init__.py
touch applications/reward_shaping/hacking/__init__.py
touch applications/reward_shaping/experiments/__init__.py
```

- [ ] **Step 3: Write config.py**

```python
# applications/reward_shaping/config.py

config = {
    # PPO (shared across all experiments for fair comparison)
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs": 10,
    "batch_size": 64,
    "max_grad_norm": 0.5,

    # MuJoCo experiments
    "mujoco_hidden_dim": 256,
    "mujoco_episodes": 1000,

    # highway-env experiments
    "highway_hidden_dim": 128,
    "highway_episodes": 500,

    # Multi-objective weight sweep
    "weight_sweep_values": [0.1, 0.5, 1.0, 2.0, 5.0],
    "collision_sweep_values": [-1.0, -5.0, -10.0, -20.0, -50.0],

    # Potential-based shaping
    "shaping_gamma": 0.99,

    # Hacking experiments
    "hacking_episodes": 500,
}
```

- [ ] **Step 4: Commit**

```bash
git add applications/reward_shaping/
git commit -m "feat(reward_shaping): add project skeleton and config"
```

---

### Task 2: Reward Wrappers — Sparse and Dense

**Files:**
- Create: `applications/reward_shaping/rewards/sparse.py`
- Create: `applications/reward_shaping/rewards/dense.py`

- [ ] **Step 1: Write sparse.py**

```python
# applications/reward_shaping/rewards/sparse.py
import gymnasium as gym
import numpy as np


class SparseRewardWrapper(gym.Wrapper):
    """Replace environment reward with sparse reward."""

    def __init__(self, env, env_type: str = "mujoco"):
        super().__init__(env)
        self.env_type = env_type

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        if self.env_type == "mujoco":
            # Sparse: +1 only when x_position > 100
            x_pos = self.unwrapped.data.qpos[0]
            reward = 1.0 if x_pos > 100.0 else 0.0
        elif self.env_type == "highway":
            # Sparse: +1 at destination, -1 on collision, else 0
            if info.get("crashed", False):
                reward = -1.0
            elif terminated and not info.get("crashed", False):
                reward = 1.0
            else:
                reward = 0.0

        return obs, reward, terminated, truncated, info
```

- [ ] **Step 2: Write dense.py**

```python
# applications/reward_shaping/rewards/dense.py
import gymnasium as gym
import numpy as np


class DenseRewardWrapper(gym.Wrapper):
    """Replace environment reward with dense reward."""

    def __init__(self, env, env_type: str = "mujoco"):
        super().__init__(env)
        self.env_type = env_type
        self._prev_x = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.env_type == "mujoco":
            self._prev_x = self.unwrapped.data.qpos[0]
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        if self.env_type == "mujoco":
            # Dense: forward displacement per step
            x_pos = self.unwrapped.data.qpos[0]
            reward = x_pos - self._prev_x
            self._prev_x = x_pos
        elif self.env_type == "highway":
            # Dense: speed / max_speed per step
            speed = self.unwrapped.vehicle.speed
            max_speed = 40.0
            reward = speed / max_speed

        return obs, reward, terminated, truncated, info
```

- [ ] **Step 3: Verify**

```bash
cd applications/reward_shaping
python3 -c "
import gymnasium as gym
from rewards.sparse import SparseRewardWrapper
from rewards.dense import DenseRewardWrapper

env = gym.make('Ant-v4')
sparse_env = SparseRewardWrapper(env, 'mujoco')
obs, _ = sparse_env.reset()
obs, r, _, _, _ = sparse_env.step(sparse_env.action_space.sample())
print(f'Sparse reward: {r}')  # likely 0.0
sparse_env.close()

env = gym.make('Ant-v4')
dense_env = DenseRewardWrapper(env, 'mujoco')
obs, _ = dense_env.reset()
obs, r, _, _, _ = dense_env.step(dense_env.action_space.sample())
print(f'Dense reward: {r:.4f}')  # small positive or negative
dense_env.close()
"
```

- [ ] **Step 4: Commit**

```bash
git add applications/reward_shaping/rewards/sparse.py applications/reward_shaping/rewards/dense.py
git commit -m "feat(reward_shaping): add sparse and dense reward wrappers"
```

---

### Task 3: Potential-based Shaping Wrapper

**Files:**
- Create: `applications/reward_shaping/rewards/potential_based.py`

- [ ] **Step 1: Write potential_based.py**

```python
# applications/reward_shaping/rewards/potential_based.py
import gymnasium as gym
import numpy as np


class PotentialShapingWrapper(gym.Wrapper):
    """Add potential-based shaping reward: F(s,s') = gamma*Phi(s') - Phi(s).

    This preserves the optimal policy (Ng 1999).
    """

    def __init__(self, env, potential_fn, gamma: float = 0.99):
        super().__init__(env)
        self.potential_fn = potential_fn
        self.gamma = gamma
        self._prev_potential = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self.potential_fn(self.unwrapped)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_potential = self.potential_fn(self.unwrapped)
        shaping = self.gamma * current_potential - self._prev_potential
        self._prev_potential = current_potential

        shaped_reward = reward + shaping
        info["original_reward"] = reward
        info["shaping_reward"] = shaping

        return obs, shaped_reward, terminated, truncated, info


# Predefined potential functions

def mujoco_x_potential(env):
    """Potential = x position (reward forward progress)."""
    return env.data.qpos[0]


def highway_speed_potential(env):
    """Potential = normalized speed."""
    return env.vehicle.speed / 40.0
```

- [ ] **Step 2: Verify**

```bash
python3 -c "
import gymnasium as gym
from rewards.sparse import SparseRewardWrapper
from rewards.potential_based import PotentialShapingWrapper, mujoco_x_potential

env = gym.make('Ant-v4')
env = SparseRewardWrapper(env, 'mujoco')
env = PotentialShapingWrapper(env, mujoco_x_potential, gamma=0.99)
obs, _ = env.reset()
obs, r, _, _, info = env.step(env.action_space.sample())
print(f'Shaped reward: {r:.4f} (original={info[\"original_reward\"]}, shaping={info[\"shaping_reward\"]:.4f})')
env.close()
"
```

- [ ] **Step 3: Commit**

```bash
git add applications/reward_shaping/rewards/potential_based.py
git commit -m "feat(reward_shaping): add potential-based shaping wrapper (Ng 1999)"
```

---

### Task 4: Multi-objective Reward Wrapper

**Files:**
- Create: `applications/reward_shaping/rewards/multi_objective.py`

- [ ] **Step 1: Write multi_objective.py**

```python
# applications/reward_shaping/rewards/multi_objective.py
import gymnasium as gym
import numpy as np


class MultiObjectiveRewardWrapper(gym.Wrapper):
    """Multi-objective weighted reward for MuJoCo locomotion."""

    def __init__(self, env, w_speed=1.0, w_alive=0.5, w_energy=0.01, w_posture=0.1):
        super().__init__(env)
        self.w_speed = w_speed
        self.w_alive = w_alive
        self.w_energy = w_energy
        self.w_posture = w_posture

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Forward velocity
        forward_velocity = self.unwrapped.data.qvel[0]

        # Alive bonus
        alive = 0.0 if terminated else 1.0

        # Energy cost
        energy = np.sum(np.square(action))

        # Posture penalty (deviation from upright)
        z_pos = self.unwrapped.data.qpos[2] if len(self.unwrapped.data.qpos) > 2 else 0
        z_target = 0.75  # approximate standing height for Ant
        posture_penalty = (z_pos - z_target) ** 2

        reward = (self.w_speed * forward_velocity
                  + self.w_alive * alive
                  + self.w_energy * (-energy)
                  + self.w_posture * (-posture_penalty))

        info["reward_components"] = {
            "speed": forward_velocity,
            "alive": alive,
            "energy": energy,
            "posture": posture_penalty,
        }

        return obs, reward, terminated, truncated, info


class HighwayMultiObjectiveWrapper(gym.Wrapper):
    """Multi-objective weighted reward for highway-env."""

    def __init__(self, env, w_speed=1.0, w_collision=-10.0, w_comfort=0.1, w_lane=0.5):
        super().__init__(env)
        self.w_speed = w_speed
        self.w_collision = w_collision
        self.w_comfort = w_comfort
        self.w_lane = w_lane
        self._prev_speed = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_speed = self.unwrapped.vehicle.speed
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        vehicle = self.unwrapped.vehicle
        speed_reward = vehicle.speed / 40.0
        collision = 1.0 if vehicle.crashed else 0.0

        # Jerk approximation (speed change as comfort proxy)
        current_speed = vehicle.speed
        jerk = abs(current_speed - self._prev_speed) if self._prev_speed else 0.0
        self._prev_speed = current_speed

        # Lane keeping (0 if centered, increases with deviation)
        lane_keeping = 1.0 if not vehicle.crashed else 0.0

        reward = (self.w_speed * speed_reward
                  + self.w_collision * collision
                  + self.w_comfort * (-jerk)
                  + self.w_lane * lane_keeping)

        info["reward_components"] = {
            "speed": speed_reward,
            "collision": collision,
            "comfort": jerk,
            "lane": lane_keeping,
        }

        return obs, reward, terminated, truncated, info
```

- [ ] **Step 2: Commit**

```bash
git add applications/reward_shaping/rewards/multi_objective.py
git commit -m "feat(reward_shaping): add multi-objective reward wrappers"
```

---

### Task 5: Reward Hacking Cases

**Files:**
- Create: `applications/reward_shaping/hacking/ant_rolling.py`
- Create: `applications/reward_shaping/hacking/hopper_jumping.py`
- Create: `applications/reward_shaping/hacking/humanoid_sliding.py`
- Create: `applications/reward_shaping/hacking/highway_lane_spam.py`
- Create: `applications/reward_shaping/hacking/highway_parking.py`

- [ ] **Step 1: Write ant_rolling.py**

```python
# applications/reward_shaping/hacking/ant_rolling.py
import gymnasium as gym
import numpy as np


class AntRollingBrokenReward(gym.Wrapper):
    """BROKEN: Only rewards forward velocity. Agent learns to roll instead of walk."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.unwrapped.data.qvel[0]  # only forward velocity
        info["hack_type"] = "rolling"
        return obs, reward, terminated, truncated, info


class AntRollingFixedReward(gym.Wrapper):
    """FIXED: Forward velocity + posture penalty prevents rolling."""

    def __init__(self, env, w_speed=1.0, w_posture=2.0, target_z=0.75):
        super().__init__(env)
        self.w_speed = w_speed
        self.w_posture = w_posture
        self.target_z = target_z

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        forward_vel = self.unwrapped.data.qvel[0]
        z_pos = self.unwrapped.data.qpos[2]
        posture_penalty = (z_pos - self.target_z) ** 2

        reward = self.w_speed * forward_vel - self.w_posture * posture_penalty
        info["hack_type"] = "fixed_rolling"
        info["z_pos"] = z_pos
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 2: Write hopper_jumping.py**

```python
# applications/reward_shaping/hacking/hopper_jumping.py
import gymnasium as gym
import numpy as np


class HopperJumpingBrokenReward(gym.Wrapper):
    """BROKEN: Large alive bonus + small speed. Agent jumps in place to stay alive."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        alive_bonus = 10.0  # too large
        speed_reward = 0.1 * self.unwrapped.data.qvel[0]
        reward = alive_bonus + speed_reward
        info["hack_type"] = "jumping"
        return obs, reward, terminated, truncated, info


class HopperJumpingFixedReward(gym.Wrapper):
    """FIXED: Balanced alive bonus and speed reward."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        alive_bonus = 0.5  # reduced
        speed_reward = 2.0 * self.unwrapped.data.qvel[0]  # increased
        reward = alive_bonus + speed_reward
        info["hack_type"] = "fixed_jumping"
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 3: Write humanoid_sliding.py**

```python
# applications/reward_shaping/hacking/humanoid_sliding.py
import gymnasium as gym
import numpy as np


class HumanoidSlidingBrokenReward(gym.Wrapper):
    """BROKEN: Only forward velocity. Agent slides on belly."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.unwrapped.data.qvel[0]
        info["hack_type"] = "sliding"
        return obs, reward, terminated, truncated, info


class HumanoidSlidingFixedReward(gym.Wrapper):
    """FIXED: Forward velocity + minimum height constraint."""

    def __init__(self, env, w_speed=1.0, min_height=1.0, height_penalty=5.0):
        super().__init__(env)
        self.w_speed = w_speed
        self.min_height = min_height
        self.height_penalty = height_penalty

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        forward_vel = self.unwrapped.data.qvel[0]
        z_pos = self.unwrapped.data.qpos[2]

        height_violation = max(0, self.min_height - z_pos)
        reward = self.w_speed * forward_vel - self.height_penalty * height_violation

        info["hack_type"] = "fixed_sliding"
        info["z_pos"] = z_pos
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 4: Write highway_lane_spam.py**

```python
# applications/reward_shaping/hacking/highway_lane_spam.py
import gymnasium as gym
import numpy as np


class HighwayLaneSpamBrokenReward(gym.Wrapper):
    """BROKEN: Positive reward for lane changes. Agent oscillates."""

    def __init__(self, env):
        super().__init__(env)
        self._prev_lane = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_lane = self.unwrapped.vehicle.lane_index[2]
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        speed_reward = self.unwrapped.vehicle.speed / 40.0
        lane_change_bonus = 1.0 if current_lane != self._prev_lane else 0.0
        self._prev_lane = current_lane

        reward = speed_reward + lane_change_bonus  # bonus encourages spam
        info["hack_type"] = "lane_spam"
        info["lane_changes"] = lane_change_bonus
        return obs, reward, terminated, truncated, info


class HighwayLaneSpamFixedReward(gym.Wrapper):
    """FIXED: Lane change as penalty + cooldown."""

    def __init__(self, env, cooldown_steps=20):
        super().__init__(env)
        self._prev_lane = None
        self._cooldown = 0
        self._cooldown_steps = cooldown_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_lane = self.unwrapped.vehicle.lane_index[2]
        self._cooldown = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        current_lane = self.unwrapped.vehicle.lane_index[2]
        speed_reward = self.unwrapped.vehicle.speed / 40.0

        lane_change_penalty = 0.0
        if current_lane != self._prev_lane:
            if self._cooldown > 0:
                lane_change_penalty = -2.0  # penalize rapid changes
            self._cooldown = self._cooldown_steps
        self._cooldown = max(0, self._cooldown - 1)
        self._prev_lane = current_lane

        collision_penalty = -10.0 if self.unwrapped.vehicle.crashed else 0.0
        reward = speed_reward + lane_change_penalty + collision_penalty

        info["hack_type"] = "fixed_lane_spam"
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 5: Write highway_parking.py**

```python
# applications/reward_shaping/hacking/highway_parking.py
import gymnasium as gym
import numpy as np


class HighwayParkingBrokenReward(gym.Wrapper):
    """BROKEN: Huge collision penalty dominates. Agent stops moving."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        speed_reward = self.unwrapped.vehicle.speed / 40.0
        collision_penalty = -100.0 if self.unwrapped.vehicle.crashed else 0.0
        reward = speed_reward + collision_penalty  # penalty too large
        info["hack_type"] = "parking"
        return obs, reward, terminated, truncated, info


class HighwayParkingFixedReward(gym.Wrapper):
    """FIXED: Balanced collision penalty + minimum speed enforcement."""

    def __init__(self, env, min_speed=5.0, min_speed_penalty=1.0):
        super().__init__(env)
        self.min_speed = min_speed
        self.min_speed_penalty = min_speed_penalty

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        speed = self.unwrapped.vehicle.speed
        speed_reward = speed / 40.0
        collision_penalty = -10.0 if self.unwrapped.vehicle.crashed else 0.0

        # Penalize going too slow
        slow_penalty = 0.0
        if speed < self.min_speed:
            slow_penalty = -self.min_speed_penalty * (self.min_speed - speed) / self.min_speed

        reward = speed_reward + collision_penalty + slow_penalty
        info["hack_type"] = "fixed_parking"
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 6: Commit**

```bash
git add applications/reward_shaping/hacking/
git commit -m "feat(reward_shaping): add 5 reward hacking cases with fixes"
```

---

### Task 6: Experiment Scripts

**Files:**
- Create: `applications/reward_shaping/experiments/run_sparse_vs_dense.py`
- Create: `applications/reward_shaping/experiments/run_potential_shaping.py`
- Create: `applications/reward_shaping/experiments/run_multi_objective.py`
- Create: `applications/reward_shaping/experiments/run_hacking_cases.py`
- Create: `applications/reward_shaping/experiments/plot_comparison.py`

- [ ] **Step 1: Write run_sparse_vs_dense.py**

```python
# applications/reward_shaping/experiments/run_sparse_vs_dense.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from rewards.sparse import SparseRewardWrapper
from rewards.dense import DenseRewardWrapper
from config import config

# Import PPO from project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from algorithms.ppo.agent import PPOAgent


def run_experiment(env_id: str, env_type: str, n_episodes: int):
    """Run sparse vs dense comparison on one environment."""
    results = {}

    for reward_type, WrapperClass in [("sparse", SparseRewardWrapper), ("dense", DenseRewardWrapper)]:
        print(f"\n  Training PPO with {reward_type} reward on {env_id}...")
        env = gym.make(env_id)
        env = WrapperClass(env, env_type)

        obs, _ = env.reset()
        state_dim = obs.flatten().shape[0]
        action_dim = env.action_space.shape[0]

        agent = PPOAgent(
            state_dim=state_dim, action_dim=action_dim,
            lr=config["lr"], gamma=config["gamma"],
            clip_eps=config["clip_eps"], epochs=config["epochs"],
            batch_size=config["batch_size"],
            hidden_dim=config["mujoco_hidden_dim"] if env_type == "mujoco" else config["highway_hidden_dim"],
        )

        returns = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            state = obs.flatten()
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            done = False

            while not done:
                action, log_prob, value = agent.take_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                state = next_obs.flatten()

            next_value = 0.0 if done else agent.take_action(state)[2]
            agent.update(states, actions, rewards, log_probs, values, dones, next_value)
            returns.append(sum(rewards))

            if (ep + 1) % 100 == 0:
                print(f"    Episode {ep+1}/{n_episodes} | Avg Return: {np.mean(returns[-100:]):.2f}")

        results[reward_type] = returns
        env.close()

    return results


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # MuJoCo
    mujoco_results = run_experiment("Ant-v4", "mujoco", config["mujoco_episodes"])
    np.save(str(results_dir / "sparse_vs_dense_ant.npy"), mujoco_results)

    # highway-env
    highway_results = run_experiment("highway-v0", "highway", config["highway_episodes"])
    np.save(str(results_dir / "sparse_vs_dense_highway.npy"), highway_results)

    print("\nExperiment complete. Results saved.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write run_potential_shaping.py**

```python
# applications/reward_shaping/experiments/run_potential_shaping.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from rewards.sparse import SparseRewardWrapper
from rewards.potential_based import PotentialShapingWrapper, mujoco_x_potential, highway_speed_potential
from config import config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from algorithms.ppo.agent import PPOAgent


def run_experiment(env_id, env_type, potential_fn, n_episodes):
    """Compare sparse-only vs sparse+shaping."""
    results = {}

    for name, use_shaping in [("sparse_only", False), ("sparse_shaped", True)]:
        print(f"\n  Training PPO: {name} on {env_id}...")
        env = gym.make(env_id)
        env = SparseRewardWrapper(env, env_type)
        if use_shaping:
            env = PotentialShapingWrapper(env, potential_fn, gamma=config["shaping_gamma"])

        obs, _ = env.reset()
        state_dim = obs.flatten().shape[0]
        action_dim = env.action_space.shape[0]
        hidden_dim = config["mujoco_hidden_dim"] if env_type == "mujoco" else config["highway_hidden_dim"]

        agent = PPOAgent(
            state_dim=state_dim, action_dim=action_dim,
            lr=config["lr"], gamma=config["gamma"],
            clip_eps=config["clip_eps"], epochs=config["epochs"],
            batch_size=config["batch_size"], hidden_dim=hidden_dim,
        )

        returns = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            state = obs.flatten()
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            done = False

            while not done:
                action, log_prob, value = agent.take_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                state = next_obs.flatten()

            next_value = 0.0 if done else agent.take_action(state)[2]
            agent.update(states, actions, rewards, log_probs, values, dones, next_value)
            returns.append(sum(rewards))

        results[name] = returns
        env.close()

    return results


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    mujoco_results = run_experiment("Ant-v4", "mujoco", mujoco_x_potential, config["mujoco_episodes"])
    np.save(str(results_dir / "potential_shaping_ant.npy"), mujoco_results)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write run_multi_objective.py**

```python
# applications/reward_shaping/experiments/run_multi_objective.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from rewards.multi_objective import MultiObjectiveRewardWrapper
from config import config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from algorithms.ppo.agent import PPOAgent


def train_with_weights(env_id, weights, n_episodes):
    """Train PPO with specific multi-objective weights."""
    env = gym.make(env_id)
    env = MultiObjectiveRewardWrapper(env, **weights)

    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        state_dim=state_dim, action_dim=action_dim,
        lr=config["lr"], gamma=config["gamma"],
        clip_eps=config["clip_eps"], epochs=config["epochs"],
        batch_size=config["batch_size"], hidden_dim=config["mujoco_hidden_dim"],
    )

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        done = False

        while not done:
            action, log_prob, value = agent.take_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            state = next_obs.flatten()

        next_value = 0.0 if done else agent.take_action(state)[2]
        agent.update(states, actions, rewards, log_probs, values, dones, next_value)
        returns.append(sum(rewards))

    env.close()
    return returns


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Weight sensitivity sweep on w_speed
    sweep_results = {}
    for w_speed in config["weight_sweep_values"]:
        print(f"\n  Training with w_speed={w_speed}...")
        weights = {"w_speed": w_speed, "w_alive": 0.5, "w_energy": 0.01, "w_posture": 0.1}
        returns = train_with_weights("Ant-v4", weights, config["mujoco_episodes"])
        sweep_results[f"w_speed_{w_speed}"] = returns

    np.save(str(results_dir / "multi_objective_sweep.npy"), sweep_results)
    print("\nWeight sweep complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write run_hacking_cases.py**

```python
# applications/reward_shaping/experiments/run_hacking_cases.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from config import config

from hacking.ant_rolling import AntRollingBrokenReward, AntRollingFixedReward
from hacking.hopper_jumping import HopperJumpingBrokenReward, HopperJumpingFixedReward
from hacking.humanoid_sliding import HumanoidSlidingBrokenReward, HumanoidSlidingFixedReward
from hacking.highway_lane_spam import HighwayLaneSpamBrokenReward, HighwayLaneSpamFixedReward
from hacking.highway_parking import HighwayParkingBrokenReward, HighwayParkingFixedReward

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from algorithms.ppo.agent import PPOAgent


HACKING_CASES = [
    ("Ant Rolling", "Ant-v4", AntRollingBrokenReward, AntRollingFixedReward),
    ("Hopper Jumping", "Hopper-v4", HopperJumpingBrokenReward, HopperJumpingFixedReward),
    ("Humanoid Sliding", "Humanoid-v4", HumanoidSlidingBrokenReward, HumanoidSlidingFixedReward),
    ("Highway Lane Spam", "highway-v0", HighwayLaneSpamBrokenReward, HighwayLaneSpamFixedReward),
    ("Highway Parking", "highway-v0", HighwayParkingBrokenReward, HighwayParkingFixedReward),
]


def train_hacking_case(case_name, env_id, BrokenWrapper, FixedWrapper, n_episodes):
    """Train broken and fixed versions, return comparison."""
    results = {}

    for label, Wrapper in [("broken", BrokenWrapper), ("fixed", FixedWrapper)]:
        print(f"  Training {case_name} ({label})...")
        env = gym.make(env_id)
        env = Wrapper(env)

        obs, _ = env.reset()
        state_dim = obs.flatten().shape[0]
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
        is_continuous = hasattr(env.action_space, 'shape')
        hidden_dim = config["mujoco_hidden_dim"] if "v4" in env_id else config["highway_hidden_dim"]

        agent = PPOAgent(
            state_dim=state_dim, action_dim=action_dim,
            lr=config["lr"], gamma=config["gamma"],
            clip_eps=config["clip_eps"], epochs=config["epochs"],
            batch_size=config["batch_size"], hidden_dim=hidden_dim,
        )

        returns = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            state = obs.flatten()
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            done = False

            while not done:
                action, log_prob, value = agent.take_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                state = next_obs.flatten()

            next_value = 0.0 if done else agent.take_action(state)[2]
            agent.update(states, actions, rewards, log_probs, values, dones, next_value)
            returns.append(sum(rewards))

        results[label] = returns
        env.close()

    return results


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = {}
    for case_name, env_id, BrokenW, FixedW in HACKING_CASES:
        print(f"\n{'='*60}")
        print(f"Case: {case_name}")
        print(f"{'='*60}")
        results = train_hacking_case(case_name, env_id, BrokenW, FixedW, config["hacking_episodes"])
        all_results[case_name] = results

    np.save(str(results_dir / "hacking_cases.npy"), all_results)
    print("\nAll hacking cases complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write plot_comparison.py**

```python
# applications/reward_shaping/experiments/plot_comparison.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_sparse_vs_dense(results_dir):
    """Plot Experiment 1 results."""
    for name in ["ant", "highway"]:
        filepath = results_dir / f"sparse_vs_dense_{name}.npy"
        if not filepath.exists():
            continue
        data = np.load(str(filepath), allow_pickle=True).item()

        plt.figure(figsize=(10, 5))
        for label, returns in data.items():
            plt.plot(moving_average(returns), label=label)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title(f"Sparse vs Dense Reward ({name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(results_dir / f"plot_sparse_vs_dense_{name}.png"), dpi=150)
        plt.close()


def plot_potential_shaping(results_dir):
    """Plot Experiment 2 results."""
    filepath = results_dir / "potential_shaping_ant.npy"
    if not filepath.exists():
        return
    data = np.load(str(filepath), allow_pickle=True).item()

    plt.figure(figsize=(10, 5))
    for label, returns in data.items():
        plt.plot(moving_average(returns), label=label)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Potential-based Shaping (Ant-v4)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_potential_shaping.png"), dpi=150)
    plt.close()


def plot_hacking_cases(results_dir):
    """Plot Experiment 4 results."""
    filepath = results_dir / "hacking_cases.npy"
    if not filepath.exists():
        return
    all_data = np.load(str(filepath), allow_pickle=True).item()

    n_cases = len(all_data)
    fig, axes = plt.subplots(1, n_cases, figsize=(5*n_cases, 4))
    if n_cases == 1:
        axes = [axes]

    for ax, (case_name, results) in zip(axes, all_data.items()):
        for label, returns in results.items():
            ax.plot(moving_average(returns), label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title(case_name)
        ax.legend()

    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_hacking_cases.png"), dpi=150)
    plt.close()


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    plot_sparse_vs_dense(results_dir)
    plot_potential_shaping(results_dir)
    plot_hacking_cases(results_dir)
    print("All plots generated.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add applications/reward_shaping/experiments/
git commit -m "feat(reward_shaping): add all experiment scripts and plotting"
```

---

### Task 7: Theory Documentation

**Files:**
- Create: `applications/reward_shaping/docs/theory.md`

- [ ] **Step 1: Write theory.md (bilingual)**

Content:
1. Why Reward Design is Hard
2. Sparse vs Dense trade-offs
3. Potential-based Shaping theorem (full Ng 1999 proof)
4. Multi-objective weight tuning methodology
5. Reward Hacking patterns and prevention
6. Interview FAQ with reference answers

- [ ] **Step 2: Commit**

```bash
git add applications/reward_shaping/docs/theory.md
git commit -m "docs(reward_shaping): add theory tutorial with interview content"
```

---

### Task 8: Integration Test

- [ ] **Step 1: Run sparse vs dense (short, 50 episodes)**

```bash
cd applications/reward_shaping
python3 -c "
from experiments.run_sparse_vs_dense import run_experiment
results = run_experiment('Ant-v4', 'mujoco', 50)
print('Sparse final avg:', sum(results['sparse'][-10:])/10)
print('Dense final avg:', sum(results['dense'][-10:])/10)
"
```

- [ ] **Step 2: Run one hacking case**

```bash
python3 -c "
from experiments.run_hacking_cases import train_hacking_case
from hacking.ant_rolling import AntRollingBrokenReward, AntRollingFixedReward
results = train_hacking_case('Ant Rolling', 'Ant-v4', AntRollingBrokenReward, AntRollingFixedReward, 50)
print('Broken avg:', sum(results['broken'][-10:])/10)
print('Fixed avg:', sum(results['fixed'][-10:])/10)
"
```

- [ ] **Step 3: Generate plots**

```bash
python3 experiments/plot_comparison.py
```

- [ ] **Step 4: Final commit if fixes needed**

```bash
git add -A applications/reward_shaping/
git commit -m "fix(reward_shaping): integration test fixes"
```

---

## Execution Order Summary

| Task | Component | Depends On |
|------|-----------|------------|
| 1 | Skeleton + Config | — |
| 2 | Sparse/Dense wrappers | 1 |
| 3 | Potential-based wrapper | 1 |
| 4 | Multi-objective wrapper | 1 |
| 5 | Hacking cases (5 files) | 1 |
| 6 | Experiment scripts | 2, 3, 4, 5 |
| 7 | Theory docs | — |
| 8 | Integration test | 6 |

Tasks 2-5 and 7 are independent and can be parallelized.
