# Sim-to-Real Pipeline Implementation Plan

**Goal:** Build a complete sim-to-real locomotion pipeline: Curriculum DR + Teacher (privileged info) + Student (RMA adaptation module) + sim-to-sim validation, across 4 MuJoCo environments.

**Architecture:** Domain Randomization wrapper randomizes physics params with curriculum scheduling. Teacher PPO trains with proprioception + privileged info. Student distills Teacher via BC using an Adaptation Module (50-frame history → 16-dim latent) + Base Policy. Validated by sim-to-sim transfer tests comparing Baseline / DR-only / Full-pipeline.

**Tech Stack:** Python 3.9, PyTorch, gymnasium[mujoco], numpy, matplotlib

**Prerequisites:** `pip install gymnasium[mujoco]` (requires mujoco binary or `MUJOCO_PATH` set)

---

## File Structure

```
applications/sim_to_real/
├── __init__.py
├── config.py
├── envs/
│   ├── __init__.py
│   ├── domain_randomization.py
│   └── vectorized_env.py
├── agent/
│   ├── __init__.py
│   ├── ppo_continuous.py
│   ├── teacher.py
│   └── student.py
├── train_teacher.py
├── train_student.py
├── evaluate.py
├── results/
└── docs/
    └── theory.md
```

---

### Task 1: Project Skeleton + Config

**Files:**
- Create: `applications/sim_to_real/__init__.py`
- Create: `applications/sim_to_real/config.py`
- Create: `applications/sim_to_real/envs/__init__.py`
- Create: `applications/sim_to_real/agent/__init__.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p applications/sim_to_real/{envs,agent,results,docs}
```

- [ ] **Step 2: Create `__init__.py` files**

```bash
touch applications/sim_to_real/__init__.py
touch applications/sim_to_real/envs/__init__.py
touch applications/sim_to_real/agent/__init__.py
```

- [ ] **Step 3: Write config.py**

```python
# applications/sim_to_real/config.py

ENV_CONFIGS = {
    "Ant-v4": {"obs_dim": 27, "action_dim": 8},
    "Hopper-v4": {"obs_dim": 11, "action_dim": 3},
    "Humanoid-v4": {"obs_dim": 376, "action_dim": 17},
    "Pusher-v4": {"obs_dim": 23, "action_dim": 7},
}

PRIVILEGED_DIM = 7  # friction(1) + mass(1) + ext_force(3) + actuator(2)

config = {
    # Environment
    "env_id": "Ant-v4",
    "num_envs": 16,

    # Domain Randomization - initial ranges
    "dr_mass_range_init": [0.95, 1.05],
    "dr_mass_range_final": [0.7, 1.3],
    "dr_inertia_range_init": [0.95, 1.05],
    "dr_inertia_range_final": [0.7, 1.3],
    "dr_friction_range_init": [0.9, 1.1],
    "dr_friction_range_final": [0.5, 1.5],
    "dr_force_range_init": [0.0, 5.0],
    "dr_force_range_final": [0.0, 50.0],
    "dr_force_interval_init": 200,
    "dr_force_interval_final": 100,
    "dr_gain_range_init": [0.95, 1.05],
    "dr_gain_range_final": [0.8, 1.2],
    "dr_delay_range_init": [0, 1],
    "dr_delay_range_final": [0, 3],

    # Curriculum
    "curriculum_end_fraction": 0.5,  # reach final range at 50% of training

    # PPO
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs": 10,
    "batch_size": 4096,
    "n_steps_per_env": 2048,
    "max_grad_norm": 0.5,
    "hidden_dim": 256,
    "n_iterations": 1000,

    # Teacher
    "teacher_episodes": 2000,

    # Student (RMA)
    "history_length": 50,
    "latent_dim": 16,
    "student_lr": 1e-3,
    "student_epochs": 100,
    "student_batch_size": 256,
    "distill_dataset_size": 1_000_000,

    # Evaluation
    "eval_episodes": 100,
    "eval_perturbation_force": 50.0,
    "eval_perturbation_interval": 50,
    "eval_ood_factor": 1.3,  # 30% beyond training range
}
```

- [ ] **Step 4: Commit**

```bash
git add applications/sim_to_real/
git commit -m "feat(sim2real): add project skeleton and config"
```

---

### Task 2: Domain Randomization Wrapper

**Files:**
- Create: `applications/sim_to_real/envs/domain_randomization.py`

- [ ] **Step 1: Write domain_randomization.py**

```python
# applications/sim_to_real/envs/domain_randomization.py
import gymnasium as gym
import numpy as np
from collections import deque


class DomainRandomizationWrapper(gym.Wrapper):
    """Curriculum Domain Randomization wrapper for MuJoCo environments.

    Randomizes: mass, inertia, friction, external forces, actuator gain/delay.
    Ranges grow linearly from initial to final over curriculum_end_fraction of training.
    """

    def __init__(self, env, config, seed=None):
        super().__init__(env)
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.current_episode = 0
        self.total_episodes = config["teacher_episodes"]
        self.curriculum_end = int(self.total_episodes * config["curriculum_end_fraction"])

        # Actuator delay buffer
        max_delay = config["dr_delay_range_final"][1]
        self.action_buffer = deque(maxlen=max_delay + 1)
        self.current_delay = 0

        # Current DR parameters (privileged info)
        self.privileged_info = np.zeros(7, dtype=np.float32)

        # Store original model parameters
        self._original_mass = None
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

        # Save originals on first call
        if self._original_mass is None:
            self._original_mass = model.body_mass.copy()
            self._original_friction = model.geom_friction.copy()
            if hasattr(model, 'actuator_gainprm'):
                self._original_gain = model.actuator_gainprm.copy()

        # Mass randomization
        mass_range = self._interpolate_range(
            self.config["dr_mass_range_init"], self.config["dr_mass_range_final"])
        mass_scale = self.rng.uniform(mass_range[0], mass_range[1])
        model.body_mass[:] = self._original_mass * mass_scale

        # Friction randomization
        friction_range = self._interpolate_range(
            self.config["dr_friction_range_init"], self.config["dr_friction_range_final"])
        friction_scale = self.rng.uniform(friction_range[0], friction_range[1])
        model.geom_friction[:] = self._original_friction * friction_scale

        # Actuator gain randomization
        gain_range = self._interpolate_range(
            self.config["dr_gain_range_init"], self.config["dr_gain_range_final"])
        gain_scale = self.rng.uniform(gain_range[0], gain_range[1])
        if self._original_gain is not None:
            model.actuator_gainprm[:] = self._original_gain * gain_scale

        # Actuator delay
        delay_range = self._interpolate_range(
            self.config["dr_delay_range_init"], self.config["dr_delay_range_final"])
        self.current_delay = int(self.rng.uniform(delay_range[0], delay_range[1]))
        self.action_buffer.clear()

        # External force parameters (applied during step)
        force_range = self._interpolate_range(
            self.config["dr_force_range_init"], self.config["dr_force_range_final"])
        self._force_magnitude = self.rng.uniform(force_range[0], force_range[1])
        interval_range = self._interpolate_range(
            [self.config["dr_force_interval_init"]] * 2,
            [self.config["dr_force_interval_final"]] * 2)
        self._force_interval = int(interval_range[0])
        self._step_count = 0
        self._current_force = np.zeros(3, dtype=np.float32)

        # Store privileged info
        self.privileged_info[0] = friction_scale
        self.privileged_info[1] = mass_scale
        self.privileged_info[2:5] = 0.0  # force updated during step
        self.privileged_info[5] = gain_scale
        self.privileged_info[6] = float(self.current_delay)

    def reset(self, **kwargs):
        self.current_episode += 1
        self._randomize()
        obs, info = self.env.reset(**kwargs)
        info["privileged_info"] = self.privileged_info.copy()
        return obs, info

    def step(self, action):
        # Apply actuator delay
        self.action_buffer.append(action)
        if len(self.action_buffer) > self.current_delay:
            delayed_action = self.action_buffer[0]
        else:
            delayed_action = np.zeros_like(action)

        # Apply external force periodically
        self._step_count += 1
        if self._step_count % self._force_interval == 0:
            direction = self.rng.standard_normal(3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self._current_force = direction * self._force_magnitude
            self.privileged_info[2:5] = self._current_force
            # Apply force to torso
            self.unwrapped.data.xfrc_applied[1, :3] = self._current_force
        else:
            self.unwrapped.data.xfrc_applied[1, :3] = 0.0
            self._current_force = np.zeros(3, dtype=np.float32)
            self.privileged_info[2:5] = 0.0

        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        info["privileged_info"] = self.privileged_info.copy()
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 2: Verify wrapper (requires mujoco installed)**

```bash
cd applications/sim_to_real
python3 -c "
import gymnasium as gym
from envs.domain_randomization import DomainRandomizationWrapper
from config import config

env = gym.make('Ant-v4')
env = DomainRandomizationWrapper(env, config)
obs, info = env.reset()
print(f'obs shape: {obs.shape}')
print(f'privileged_info: {info[\"privileged_info\"]}')
obs2, r, term, trunc, info2 = env.step(env.action_space.sample())
print(f'step ok, reward={r:.3f}')
env.close()
"
```

- [ ] **Step 3: Commit**

```bash
git add applications/sim_to_real/envs/domain_randomization.py
git commit -m "feat(sim2real): add curriculum domain randomization wrapper"
```

---

### Task 3: Vectorized Environment

**Files:**
- Create: `applications/sim_to_real/envs/vectorized_env.py`

- [ ] **Step 1: Write vectorized_env.py**

```python
# applications/sim_to_real/envs/vectorized_env.py
import gymnasium as gym
import numpy as np
from .domain_randomization import DomainRandomizationWrapper


def make_vec_env(env_id: str, num_envs: int, config: dict, use_dr: bool = True):
    """Create vectorized MuJoCo environments with optional DR.

    Uses gymnasium.vector.AsyncVectorEnv for parallel stepping.
    """
    def make_env(seed):
        def _init():
            env = gym.make(env_id)
            if use_dr:
                env = DomainRandomizationWrapper(env, config, seed=seed)
            return env
        return _init

    env_fns = [make_env(seed=i) for i in range(num_envs)]
    vec_env = gym.vector.AsyncVectorEnv(env_fns)
    return vec_env


class VecEnvHelper:
    """Helper to manage vectorized env rollouts and privileged info collection."""

    def __init__(self, vec_env, num_envs: int):
        self.vec_env = vec_env
        self.num_envs = num_envs

    def reset(self):
        obs, infos = self.vec_env.reset()
        privileged = self._extract_privileged(infos)
        return obs, privileged

    def step(self, actions):
        obs, rewards, terminateds, truncateds, infos = self.vec_env.step(actions)
        dones = np.logical_or(terminateds, truncateds)
        privileged = self._extract_privileged(infos)
        return obs, rewards, dones, privileged

    def _extract_privileged(self, infos):
        """Extract privileged_info from vectorized info dict."""
        if "privileged_info" in infos:
            return np.array(infos["privileged_info"])
        return np.zeros((self.num_envs, 7), dtype=np.float32)

    def close(self):
        self.vec_env.close()
```

- [ ] **Step 2: Verify**

```bash
python3 -c "
from envs.vectorized_env import make_vec_env, VecEnvHelper
from config import config

vec_env = make_vec_env('Ant-v4', num_envs=4, config=config, use_dr=True)
helper = VecEnvHelper(vec_env, num_envs=4)
obs, priv = helper.reset()
print(f'obs shape: {obs.shape}')  # (4, 27)
print(f'priv shape: {priv.shape}')  # (4, 7)
actions = vec_env.action_space.sample()
obs2, rewards, dones, priv2 = helper.step(actions)
print(f'step ok, rewards shape: {rewards.shape}')
helper.close()
"
```

- [ ] **Step 3: Commit**

```bash
git add applications/sim_to_real/envs/vectorized_env.py
git commit -m "feat(sim2real): add vectorized environment with DR support"
```

---

### Task 4: PPO Continuous Agent (Enhanced)

**Files:**
- Create: `applications/sim_to_real/agent/ppo_continuous.py`

- [ ] **Step 1: Write ppo_continuous.py**

```python
# applications/sim_to_real/agent/ppo_continuous.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class RunningMeanStd:
    """Tracks running mean and std for normalization."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        batch = np.asarray(batch)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class GaussianActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        features = self.net(x)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, x):
        mean, std = self.forward(x)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PPOContinuous:
    """PPO for continuous actions with obs normalization and reward scaling."""

    def __init__(self, obs_dim, action_dim, config):
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_eps = config["clip_eps"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.max_grad_norm = config["max_grad_norm"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_dim = config["hidden_dim"]

        self.actor = GaussianActor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["lr"])

        # Observation normalization
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        # Reward scaling
        self.reward_rms = RunningMeanStd(shape=(1,))

    def normalize_obs(self, obs):
        self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def scale_reward(self, rewards):
        self.reward_rms.update(rewards.reshape(-1, 1))
        return rewards / (np.sqrt(self.reward_rms.var[0]) + 1e-8)

    def act(self, obs_normalized):
        obs_t = torch.FloatTensor(obs_normalized).to(self.device)
        with torch.no_grad():
            dist = self.actor.get_dist(obs_t)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
            values = self.critic(obs_t).squeeze(-1)
        return (actions.cpu().numpy(), log_probs.cpu().numpy(),
                values.cpu().numpy())

    def update(self, obs, actions, log_probs, values, rewards, dones, next_values):
        """PPO update with GAE."""
        # Compute GAE
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_values
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Mini-batch updates
        n = len(obs)
        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_obs = obs_t[idx]
                batch_actions = actions_t[idx]
                batch_old_lp = old_log_probs_t[idx]
                batch_adv = advantages_t[idx]
                batch_ret = returns_t[idx]

                # Actor loss
                dist = self.actor.get_dist(batch_obs)
                new_lp = dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(new_lp - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                new_values = self.critic(batch_obs).squeeze(-1)
                critic_loss = nn.MSELoss()(new_values, batch_ret)

                # Update
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.obs_rms.mean = ckpt["obs_rms_mean"]
        self.obs_rms.var = ckpt["obs_rms_var"]
        self.obs_rms.count = ckpt["obs_rms_count"]
```

- [ ] **Step 2: Verify agent creation**

```bash
python3 -c "
from agent.ppo_continuous import PPOContinuous
from config import config
agent = PPOContinuous(obs_dim=27, action_dim=8, config=config)
import numpy as np
obs = np.random.randn(16, 27).astype(np.float32)
obs_norm = agent.normalize_obs(obs)
actions, lp, vals = agent.act(obs_norm)
print(f'actions: {actions.shape}, log_probs: {lp.shape}, values: {vals.shape}')
"
```

- [ ] **Step 3: Commit**

```bash
git add applications/sim_to_real/agent/ppo_continuous.py
git commit -m "feat(sim2real): add enhanced PPO with obs norm and reward scaling"
```

---

### Task 5: Teacher Policy

**Files:**
- Create: `applications/sim_to_real/agent/teacher.py`

- [ ] **Step 1: Write teacher.py**

```python
# applications/sim_to_real/agent/teacher.py
import numpy as np
from .ppo_continuous import PPOContinuous, RunningMeanStd


class TeacherAgent:
    """Teacher policy: PPO with proprioception + privileged information."""

    def __init__(self, obs_dim: int, privileged_dim: int, action_dim: int, config: dict):
        self.obs_dim = obs_dim
        self.privileged_dim = privileged_dim
        total_input_dim = obs_dim + privileged_dim

        self.ppo = PPOContinuous(total_input_dim, action_dim, config)
        self.obs_rms_proprio = RunningMeanStd(shape=(obs_dim,))

    def get_input(self, obs: np.ndarray, privileged: np.ndarray) -> np.ndarray:
        """Concatenate proprioception and privileged info."""
        if obs.ndim == 1:
            return np.concatenate([obs, privileged])
        return np.concatenate([obs, privileged], axis=-1)

    def act(self, obs: np.ndarray, privileged: np.ndarray):
        """Get action from Teacher.

        Returns: (actions, log_probs, values)
        """
        full_input = self.get_input(obs, privileged)
        full_input_norm = self.ppo.normalize_obs(full_input)
        return self.ppo.act(full_input_norm)

    def update(self, obs, privileged, actions, log_probs, values, rewards, dones, next_values):
        full_input = self.get_input(obs, privileged)
        full_input_norm = self.ppo.normalize_obs(full_input)
        scaled_rewards = self.ppo.scale_reward(rewards)
        self.ppo.update(full_input_norm, actions, log_probs, values, scaled_rewards, dones, next_values)

    def get_action_for_obs(self, obs: np.ndarray, privileged: np.ndarray) -> np.ndarray:
        """Deterministic action (mean) for distillation data collection."""
        full_input = self.get_input(obs, privileged)
        full_input_norm = self.ppo.normalize_obs(full_input)
        import torch
        obs_t = torch.FloatTensor(full_input_norm).to(self.ppo.device)
        with torch.no_grad():
            mean, _ = self.ppo.actor(obs_t)
        return mean.cpu().numpy()

    def save(self, path):
        self.ppo.save(path)

    def load(self, path):
        self.ppo.load(path)
```

- [ ] **Step 2: Verify**

```bash
python3 -c "
from agent.teacher import TeacherAgent
from config import config
teacher = TeacherAgent(obs_dim=27, privileged_dim=7, action_dim=8, config=config)
import numpy as np
obs = np.random.randn(16, 27).astype(np.float32)
priv = np.random.randn(16, 7).astype(np.float32)
actions, lp, vals = teacher.act(obs, priv)
print(f'Teacher output: actions={actions.shape}')
"
```

- [ ] **Step 3: Commit**

```bash
git add applications/sim_to_real/agent/teacher.py
git commit -m "feat(sim2real): add Teacher policy with privileged info"
```

---

### Task 6: Student Policy (RMA)

**Files:**
- Create: `applications/sim_to_real/agent/student.py`

- [ ] **Step 1: Write student.py**

```python
# applications/sim_to_real/agent/student.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class AdaptationModule(nn.Module):
    """RMA Adaptation Module: history of observations → latent z."""

    def __init__(self, obs_dim: int, history_length: int, latent_dim: int):
        super().__init__()
        input_dim = obs_dim * history_length
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, obs_history):
        """obs_history: (batch, history_length * obs_dim) flattened."""
        return self.net(obs_history)


class BasePolicy(nn.Module):
    """Student base policy: current obs + latent z → action."""

    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, z):
        x = torch.cat([obs, z], dim=-1)
        return self.net(x)


class StudentAgent:
    """Student with RMA: Adaptation Module + Base Policy, trained via BC."""

    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_length = config["history_length"]
        self.latent_dim = config["latent_dim"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.adaptation = AdaptationModule(
            obs_dim, self.history_length, self.latent_dim
        ).to(self.device)

        self.base_policy = BasePolicy(
            obs_dim, self.latent_dim, action_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.adaptation.parameters()) + list(self.base_policy.parameters()),
            lr=config["student_lr"],
        )

        # Observation history buffer (for inference)
        self._history_buffer = None

    def reset_history(self):
        """Reset history buffer for new episode."""
        self._history_buffer = np.zeros(
            (self.history_length, self.obs_dim), dtype=np.float32
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Get deterministic action from Student.

        Maintains internal history buffer.
        """
        if self._history_buffer is None:
            self.reset_history()

        # Shift history and add current obs
        self._history_buffer = np.roll(self._history_buffer, -1, axis=0)
        self._history_buffer[-1] = obs

        # Flatten history
        history_flat = self._history_buffer.flatten()

        with torch.no_grad():
            history_t = torch.FloatTensor(history_flat).unsqueeze(0).to(self.device)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            z = self.adaptation(history_t)
            action = self.base_policy(obs_t, z)

        return action.cpu().numpy().flatten()

    def train_step(self, obs_history_batch, obs_current_batch, action_teacher_batch):
        """One training step of BC distillation.

        Args:
            obs_history_batch: (batch, history_length * obs_dim)
            obs_current_batch: (batch, obs_dim)
            action_teacher_batch: (batch, action_dim)

        Returns:
            loss value (float)
        """
        history_t = torch.FloatTensor(obs_history_batch).to(self.device)
        obs_t = torch.FloatTensor(obs_current_batch).to(self.device)
        target_t = torch.FloatTensor(action_teacher_batch).to(self.device)

        z = self.adaptation(history_t)
        action_pred = self.base_policy(obs_t, z)

        loss = nn.MSELoss()(action_pred, target_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        torch.save({
            "adaptation": self.adaptation.state_dict(),
            "base_policy": self.base_policy.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.adaptation.load_state_dict(ckpt["adaptation"])
        self.base_policy.load_state_dict(ckpt["base_policy"])
```

- [ ] **Step 2: Verify**

```bash
python3 -c "
from agent.student import StudentAgent
from config import config
student = StudentAgent(obs_dim=27, action_dim=8, config=config)
import numpy as np
student.reset_history()
obs = np.random.randn(27).astype(np.float32)
action = student.act(obs)
print(f'Student action: {action.shape}')  # (8,)

# Test train_step
batch = 32
history = np.random.randn(batch, 50*27).astype(np.float32)
obs_cur = np.random.randn(batch, 27).astype(np.float32)
act_teacher = np.random.randn(batch, 8).astype(np.float32)
loss = student.train_step(history, obs_cur, act_teacher)
print(f'Train step loss: {loss:.4f}')
"
```

- [ ] **Step 3: Commit**

```bash
git add applications/sim_to_real/agent/student.py
git commit -m "feat(sim2real): add Student policy with RMA adaptation module"
```

---

### Task 7: Teacher Training Script

**Files:**
- Create: `applications/sim_to_real/train_teacher.py`

- [ ] **Step 1: Write train_teacher.py**

```python
# applications/sim_to_real/train_teacher.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from tqdm import tqdm

from config import config, ENV_CONFIGS, PRIVILEGED_DIM
from envs.vectorized_env import make_vec_env, VecEnvHelper
from agent.teacher import TeacherAgent


def train_teacher(env_id: str = None):
    env_id = env_id or config["env_id"]
    env_cfg = ENV_CONFIGS[env_id]
    obs_dim = env_cfg["obs_dim"]
    action_dim = env_cfg["action_dim"]
    num_envs = config["num_envs"]
    n_steps = config["n_steps_per_env"]
    n_iterations = config["n_iterations"]

    # Create environments
    vec_env = make_vec_env(env_id, num_envs, config, use_dr=True)
    helper = VecEnvHelper(vec_env, num_envs)

    # Create Teacher
    teacher = TeacherAgent(obs_dim, PRIVILEGED_DIM, action_dim, config)

    # Training loop
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    reward_history = []

    for iteration in tqdm(range(n_iterations), desc=f"Teacher Training ({env_id})"):
        # Collect rollout
        all_obs, all_priv, all_actions = [], [], []
        all_log_probs, all_values, all_rewards, all_dones = [], [], [], []

        obs, privileged = helper.reset()
        for step in range(n_steps):
            actions, log_probs, values = teacher.act(obs, privileged)
            next_obs, rewards, dones, next_privileged = helper.step(actions)

            all_obs.append(obs)
            all_priv.append(privileged)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_rewards.append(rewards)
            all_dones.append(dones)

            obs = next_obs
            privileged = next_privileged

        # Get next values for GAE
        _, _, next_values = teacher.act(obs, privileged)

        # Flatten across envs and steps
        all_obs = np.array(all_obs).reshape(-1, obs_dim)
        all_priv = np.array(all_priv).reshape(-1, PRIVILEGED_DIM)
        all_actions = np.array(all_actions).reshape(-1, action_dim)
        all_log_probs = np.array(all_log_probs).reshape(-1)
        all_values = np.array(all_values).reshape(-1)
        all_rewards = np.array(all_rewards).reshape(-1)
        all_dones = np.array(all_dones).reshape(-1)

        # Update
        teacher.update(
            all_obs, all_priv, all_actions, all_log_probs,
            all_values, all_rewards, all_dones, next_values.mean()
        )

        # Logging
        episode_reward = all_rewards.sum() / num_envs
        reward_history.append(episode_reward)

        if (iteration + 1) % 50 == 0:
            avg = np.mean(reward_history[-50:])
            print(f"  Iter {iteration+1}/{n_iterations} | Avg Reward: {avg:.2f}")

    # Save
    save_path = str(results_dir / f"teacher_{env_id.replace('-', '_')}.pth")
    teacher.save(save_path)
    np.save(str(results_dir / f"teacher_rewards_{env_id.replace('-', '_')}.npy"), reward_history)

    helper.close()
    print(f"Teacher training complete. Saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None)
    args = parser.parse_args()
    train_teacher(env_id=args.env)
```

- [ ] **Step 2: Smoke test**

```bash
python3 train_teacher.py --env Ant-v4
# Should run a few iterations without crashing
# Ctrl+C after verifying it works
```

- [ ] **Step 3: Commit**

```bash
git add applications/sim_to_real/train_teacher.py
git commit -m "feat(sim2real): add Teacher training script"
```

---

### Task 8: Student Distillation Script

**Files:**
- Create: `applications/sim_to_real/train_student.py`

- [ ] **Step 1: Write train_student.py**

```python
# applications/sim_to_real/train_student.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from tqdm import tqdm

from config import config, ENV_CONFIGS, PRIVILEGED_DIM
from envs.vectorized_env import make_vec_env, VecEnvHelper
from agent.teacher import TeacherAgent
from agent.student import StudentAgent


def collect_distillation_data(teacher, helper, obs_dim, action_dim, dataset_size, history_length):
    """Run Teacher in DR env and collect (history, obs, action) tuples."""
    print("Collecting distillation data from Teacher...")

    obs_history_data = []
    obs_current_data = []
    action_data = []

    obs, privileged = helper.reset()
    # Per-env history buffers
    num_envs = obs.shape[0]
    histories = np.zeros((num_envs, history_length, obs_dim), dtype=np.float32)

    collected = 0
    pbar = tqdm(total=dataset_size, desc="Collecting data")

    while collected < dataset_size:
        # Update histories
        histories = np.roll(histories, -1, axis=1)
        histories[:, -1, :] = obs

        # Get Teacher action (deterministic)
        actions = teacher.get_action_for_obs(obs, privileged)

        # Store
        for i in range(num_envs):
            obs_history_data.append(histories[i].flatten())
            obs_current_data.append(obs[i])
            action_data.append(actions[i])
            collected += 1

        pbar.update(num_envs)

        # Step env
        next_obs, _, dones, next_privileged = helper.step(actions)

        # Reset histories for done envs
        for i in range(num_envs):
            if dones[i]:
                histories[i] = 0.0

        obs = next_obs
        privileged = next_privileged

    pbar.close()

    return (
        np.array(obs_history_data[:dataset_size]),
        np.array(obs_current_data[:dataset_size]),
        np.array(action_data[:dataset_size]),
    )


def train_student(env_id: str = None):
    env_id = env_id or config["env_id"]
    env_cfg = ENV_CONFIGS[env_id]
    obs_dim = env_cfg["obs_dim"]
    action_dim = env_cfg["action_dim"]

    results_dir = Path(__file__).resolve().parent / "results"

    # Load trained Teacher
    teacher_path = str(results_dir / f"teacher_{env_id.replace('-', '_')}.pth")
    teacher = TeacherAgent(obs_dim, PRIVILEGED_DIM, action_dim, config)
    teacher.load(teacher_path)
    print(f"Loaded Teacher from {teacher_path}")

    # Create DR env for data collection
    vec_env = make_vec_env(env_id, config["num_envs"], config, use_dr=True)
    helper = VecEnvHelper(vec_env, config["num_envs"])

    # Collect data
    obs_history, obs_current, actions_teacher = collect_distillation_data(
        teacher, helper, obs_dim, action_dim,
        config["distill_dataset_size"], config["history_length"]
    )
    helper.close()

    print(f"Dataset: history={obs_history.shape}, obs={obs_current.shape}, actions={actions_teacher.shape}")

    # Train Student
    student = StudentAgent(obs_dim, action_dim, config)
    batch_size = config["student_batch_size"]
    n_epochs = config["student_epochs"]
    n_samples = len(obs_history)

    loss_history = []

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        epoch_losses = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = indices[start:end]

            loss = student.train_step(
                obs_history[idx], obs_current[idx], actions_teacher[idx]
            )
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f}")

    # Save
    save_path = str(results_dir / f"student_{env_id.replace('-', '_')}.pth")
    student.save(save_path)
    np.save(str(results_dir / f"student_loss_{env_id.replace('-', '_')}.npy"), loss_history)

    print(f"Student training complete. Saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None)
    args = parser.parse_args()
    train_student(env_id=args.env)
```

- [ ] **Step 2: Commit**

```bash
git add applications/sim_to_real/train_student.py
git commit -m "feat(sim2real): add Student distillation training script"
```

---

### Task 9: Evaluation Script

**Files:**
- Create: `applications/sim_to_real/evaluate.py`

- [ ] **Step 1: Write evaluate.py**

```python
# applications/sim_to_real/evaluate.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config, ENV_CONFIGS, PRIVILEGED_DIM
from envs.domain_randomization import DomainRandomizationWrapper
from agent.teacher import TeacherAgent
from agent.student import StudentAgent
from agent.ppo_continuous import PPOContinuous


def evaluate_agent(agent, env, n_episodes, agent_type="student"):
    """Evaluate an agent and return metrics."""
    survival_steps = []
    forward_velocities = []
    total_rewards = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        privileged = info.get("privileged_info", np.zeros(PRIVILEGED_DIM))

        if agent_type == "student":
            agent.reset_history()

        done = False
        steps = 0
        ep_reward = 0
        velocities = []

        while not done:
            if agent_type == "student":
                action = agent.act(obs)
            elif agent_type == "teacher":
                action, _, _ = agent.act(obs[np.newaxis], privileged[np.newaxis])
                action = action[0]
            else:  # baseline
                obs_norm = agent.normalize_obs(obs[np.newaxis])
                action, _, _ = agent.act(obs_norm)
                action = action[0]

            obs, reward, terminated, truncated, info = env.step(action)
            privileged = info.get("privileged_info", np.zeros(PRIVILEGED_DIM))
            done = terminated or truncated
            steps += 1
            ep_reward += reward

            # Track forward velocity (x-velocity is typically in obs)
            if hasattr(env.unwrapped, 'data'):
                velocities.append(env.unwrapped.data.qvel[0])

        survival_steps.append(steps)
        total_rewards.append(ep_reward)
        if velocities:
            forward_velocities.append(np.mean(velocities))

    return {
        "survival_mean": np.mean(survival_steps),
        "survival_std": np.std(survival_steps),
        "velocity_mean": np.mean(forward_velocities) if forward_velocities else 0,
        "reward_mean": np.mean(total_rewards),
        "reward_std": np.std(total_rewards),
    }


def make_test_env(env_id, test_type, config):
    """Create test environment with specific DR configuration."""
    env = gym.make(env_id)

    if test_type == "nominal":
        return env

    elif test_type == "in_dist":
        # Fixed params within DR range
        test_config = config.copy()
        test_config["dr_mass_range_init"] = [1.2, 1.2]
        test_config["dr_mass_range_final"] = [1.2, 1.2]
        test_config["dr_friction_range_init"] = [0.7, 0.7]
        test_config["dr_friction_range_final"] = [0.7, 0.7]
        return DomainRandomizationWrapper(env, test_config)

    elif test_type == "ood":
        # Params 30% beyond training range
        ood = config["eval_ood_factor"]
        test_config = config.copy()
        test_config["dr_mass_range_init"] = [ood * 1.3, ood * 1.3]
        test_config["dr_mass_range_final"] = [ood * 1.3, ood * 1.3]
        test_config["dr_friction_range_init"] = [0.35, 0.35]
        test_config["dr_friction_range_final"] = [0.35, 0.35]
        return DomainRandomizationWrapper(env, test_config)

    elif test_type == "perturbation":
        test_config = config.copy()
        test_config["dr_force_range_init"] = [50.0, 50.0]
        test_config["dr_force_range_final"] = [50.0, 50.0]
        test_config["dr_force_interval_init"] = 50
        test_config["dr_force_interval_final"] = 50
        return DomainRandomizationWrapper(env, test_config)


def run_evaluation(env_id: str = None):
    env_id = env_id or config["env_id"]
    env_cfg = ENV_CONFIGS[env_id]
    obs_dim = env_cfg["obs_dim"]
    action_dim = env_cfg["action_dim"]
    n_episodes = config["eval_episodes"]

    results_dir = Path(__file__).resolve().parent / "results"

    # Load agents
    # A. Baseline (no DR)
    baseline = PPOContinuous(obs_dim, action_dim, config)
    baseline_path = results_dir / f"baseline_{env_id.replace('-', '_')}.pth"
    if baseline_path.exists():
        baseline.load(str(baseline_path))

    # B. Teacher (DR only, tested without privileged — use zero privileged)
    teacher = TeacherAgent(obs_dim, PRIVILEGED_DIM, action_dim, config)
    teacher_path = results_dir / f"teacher_{env_id.replace('-', '_')}.pth"
    if teacher_path.exists():
        teacher.load(str(teacher_path))

    # C. Student (full pipeline)
    student = StudentAgent(obs_dim, action_dim, config)
    student_path = results_dir / f"student_{env_id.replace('-', '_')}.pth"
    if student_path.exists():
        student.load(str(student_path))

    # Run evaluation
    test_types = ["nominal", "in_dist", "ood", "perturbation"]
    agents = {
        "Baseline": (baseline, "baseline"),
        "DR only (Teacher)": (teacher, "teacher"),
        "Full Pipeline (Student)": (student, "student"),
    }

    results = {}
    for test_type in test_types:
        results[test_type] = {}
        env = make_test_env(env_id, test_type, config)

        for agent_name, (agent, agent_type) in agents.items():
            print(f"Evaluating {agent_name} on {test_type}...")
            metrics = evaluate_agent(agent, env, n_episodes, agent_type)
            results[test_type][agent_name] = metrics

        env.close()

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"Evaluation Results: {env_id}")
    print(f"{'='*80}")
    print(f"{'Test Domain':<15} {'Agent':<25} {'Survival':<15} {'Velocity':<12} {'Reward':<12}")
    print(f"{'-'*80}")
    for test_type in test_types:
        for agent_name, metrics in results[test_type].items():
            print(f"{test_type:<15} {agent_name:<25} "
                  f"{metrics['survival_mean']:.0f}±{metrics['survival_std']:.0f}  "
                  f"{metrics['velocity_mean']:.2f}       "
                  f"{metrics['reward_mean']:.1f}±{metrics['reward_std']:.1f}")

    # Plot
    _plot_results(results, env_id, results_dir)
    return results


def _plot_results(results, env_id, results_dir):
    test_types = list(results.keys())
    agent_names = list(results[test_types[0]].keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for metric_idx, (metric, title) in enumerate([
        ("survival_mean", "Survival Steps"),
        ("velocity_mean", "Forward Velocity (m/s)"),
        ("reward_mean", "Episode Reward"),
    ]):
        ax = axes[metric_idx]
        x = np.arange(len(test_types))
        width = 0.25

        for i, agent_name in enumerate(agent_names):
            values = [results[t][agent_name][metric] for t in test_types]
            ax.bar(x + i * width, values, width, label=agent_name)

        ax.set_xlabel("Test Domain")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(test_types, rotation=20)
        ax.legend()

    plt.suptitle(f"Sim-to-Sim Transfer Evaluation ({env_id})")
    plt.tight_layout()
    plt.savefig(str(results_dir / f"eval_{env_id.replace('-', '_')}.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None)
    args = parser.parse_args()
    run_evaluation(env_id=args.env)
```

- [ ] **Step 2: Commit**

```bash
git add applications/sim_to_real/evaluate.py
git commit -m "feat(sim2real): add sim-to-sim evaluation with comparison plots"
```

---

### Task 10: Baseline Training (No DR)

A separate script to train baseline PPO without domain randomization, for fair comparison.

- [ ] **Step 1: Add to train_teacher.py or create separate script**

Add `--no-dr` flag to `train_teacher.py`:

```python
# Add to train_teacher.py argparse:
parser.add_argument("--no-dr", action="store_true", help="Train baseline without DR")

# In train_teacher function, modify env creation:
vec_env = make_vec_env(env_id, num_envs, config, use_dr=not args.no_dr)

# When no-dr, save as baseline:
if args.no_dr:
    save_path = str(results_dir / f"baseline_{env_id.replace('-', '_')}.pth")
```

- [ ] **Step 2: Commit**

```bash
git add applications/sim_to_real/train_teacher.py
git commit -m "feat(sim2real): add baseline training mode (--no-dr)"
```

---

### Task 11: Theory Documentation

**Files:**
- Create: `applications/sim_to_real/docs/theory.md`

- [ ] **Step 1: Write theory.md (bilingual)**

Content outline:
- Why Sim-to-Real: the reality gap
- Domain Randomization: philosophy and implementation
- Curriculum DR: why progressive ranges
- Teacher-Student paradigm: privileged learning
- RMA: implicit system identification via history
- Evaluation methodology: sim-to-sim transfer protocol

- [ ] **Step 2: Commit**

```bash
git add applications/sim_to_real/docs/theory.md
git commit -m "docs(sim2real): add theory documentation"
```

---

### Task 12: Integration Test

- [ ] **Step 1: Install mujoco if needed**

```bash
pip install gymnasium[mujoco]
```

- [ ] **Step 2: Train baseline (5 iterations, verify runs)**

```bash
cd applications/sim_to_real
python3 train_teacher.py --env Ant-v4 --no-dr
# Ctrl+C after a few iterations to verify
```

- [ ] **Step 3: Train Teacher (5 iterations, verify DR works)**

```bash
python3 train_teacher.py --env Ant-v4
```

- [ ] **Step 4: Train Student (verify distillation runs)**

```bash
python3 train_student.py --env Ant-v4
```

- [ ] **Step 5: Run evaluation**

```bash
python3 evaluate.py --env Ant-v4
```

- [ ] **Step 6: Final commit if fixes needed**

```bash
git add -A applications/sim_to_real/
git commit -m "fix(sim2real): integration test fixes"
```

---

## Execution Order Summary

| Task | Component | Depends On |
|------|-----------|------------|
| 1 | Skeleton + Config | — |
| 2 | DR Wrapper | 1 |
| 3 | Vectorized Env | 2 |
| 4 | PPO Continuous | 1 |
| 5 | Teacher | 4 |
| 6 | Student (RMA) | 1 |
| 7 | Train Teacher | 3, 5 |
| 8 | Train Student | 6, 7 |
| 9 | Evaluate | 5, 6, 4 |
| 10 | Baseline Mode | 7 |
| 11 | Theory Docs | — |
| 12 | Integration Test | all |

Tasks 2-6 have limited dependencies and can be partially parallelized. Task 11 is independent.
