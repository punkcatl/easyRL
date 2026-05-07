# easyRL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a progressive RL learning project from Q-Learning to SAC, culminating in a lane-keeping comparison experiment on highway-env.

**Architecture:** Each algorithm lives in its own directory with a consistent interface (agent, train script, config). Shared utilities handle plotting, logging, and evaluation metrics. A final experiment module ties the trained agents together for comparison.

**Tech Stack:** PyTorch, highway-env (Gymnasium), matplotlib, TensorBoard (optional), conda

---

## File Structure

```
easyRL/
├── requirements.txt
├── environment.yml                    # conda environment definition
├── utils/
│   ├── __init__.py
│   ├── plotting.py                    # matplotlib chart generation
│   ├── logger.py                      # training logger (CSV + optional TensorBoard)
│   └── metrics.py                     # control-quality evaluation metrics
├── envs/
│   ├── __init__.py
│   └── highway_lane_keeping.py        # highway-env wrapper for lane keeping
├── algorithms/
│   ├── q_learning/
│   │   ├── __init__.py
│   │   ├── agent.py                   # Q-table agent
│   │   ├── train.py                   # training on CliffWalking
│   │   └── config.py                  # hyperparameters
│   ├── dqn/
│   │   ├── __init__.py
│   │   ├── agent.py                   # DQN agent with replay buffer
│   │   ├── train.py                   # training on CartPole + highway-env
│   │   └── config.py
│   ├── policy_gradient/
│   │   ├── __init__.py
│   │   ├── agent.py                   # REINFORCE agent
│   │   ├── train.py                   # training on CartPole + highway-env
│   │   └── config.py
│   ├── ppo/
│   │   ├── __init__.py
│   │   ├── agent.py                   # PPO agent with clip objective
│   │   ├── train.py                   # training on CartPole + highway-env
│   │   └── config.py
│   └── sac/
│       ├── __init__.py
│       ├── agent.py                   # SAC agent with entropy tuning
│       ├── train.py                   # training on Pendulum + highway-env
│       └── config.py
├── experiments/
│   ├── __init__.py
│   ├── run_comparison.py              # run all 3 algorithms on lane keeping
│   ├── evaluate.py                    # compute control-quality metrics
│   └── plot_results.py                # generate comparison charts
└── tests/
    ├── test_utils.py
    ├── test_envs.py
    └── test_agents.py
```

---

## Task 1: Project Setup and Shared Utilities

**Files:**
- Create: `environment.yml`
- Create: `requirements.txt`
- Create: `utils/__init__.py`
- Create: `utils/plotting.py`
- Create: `utils/logger.py`
- Create: `utils/metrics.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Create conda environment file**

```yaml
# environment.yml
name: easyrl
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch
  - pytorch-cuda=12.4
  - numpy
  - matplotlib
  - pip
  - pip:
    - gymnasium
    - highway-env
    - tensorboard
    - pytest
```

- [ ] **Step 2: Create requirements.txt (pip fallback)**

```
torch>=2.0
gymnasium>=0.29
highway-env>=1.8
matplotlib>=3.7
numpy>=1.24
tensorboard>=2.14
pytest>=7.0
```

- [ ] **Step 3: Write test for logger**

```python
# tests/test_utils.py
import os
import tempfile
from utils.logger import Logger


def test_logger_records_scalar():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir, use_tensorboard=False)
        logger.log("reward", 10.0, step=1)
        logger.log("reward", 20.0, step=2)
        data = logger.get_data("reward")
        assert data == [(1, 10.0), (2, 20.0)]


def test_logger_saves_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir, use_tensorboard=False)
        logger.log("reward", 10.0, step=1)
        logger.save()
        assert os.path.exists(os.path.join(tmpdir, "reward.csv"))
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_utils.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'utils'"

- [ ] **Step 5: Implement logger**

```python
# utils/__init__.py
```

```python
# utils/logger.py
import os
import csv
from collections import defaultdict


class Logger:
    def __init__(self, log_dir: str, use_tensorboard: bool = False):
        self.log_dir = log_dir
        self.data = defaultdict(list)
        self.writer = None
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)

    def log(self, tag: str, value: float, step: int):
        self.data[tag].append((step, value))
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def get_data(self, tag: str) -> list:
        return self.data[tag]

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        for tag, records in self.data.items():
            path = os.path.join(self.log_dir, f"{tag}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "value"])
                writer.writerows(records)

    def close(self):
        if self.writer:
            self.writer.close()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_utils.py -v`
Expected: PASS

- [ ] **Step 7: Write test for metrics**

```python
# append to tests/test_utils.py
import numpy as np
from utils.metrics import compute_control_metrics


def test_compute_control_metrics():
    lateral_deviations = [0.1, -0.2, 0.15, -0.1, 0.05]
    heading_errors = [0.01, -0.02, 0.015, -0.01, 0.005]
    steering_angles = [0.0, 0.1, 0.05, 0.12, 0.08]
    metrics = compute_control_metrics(lateral_deviations, heading_errors, steering_angles)
    assert "lateral_mean" in metrics
    assert "lateral_std" in metrics
    assert "heading_mean" in metrics
    assert "steering_smoothness" in metrics
    assert metrics["lateral_mean"] == pytest.approx(np.mean(np.abs(lateral_deviations)), rel=1e-5)
```

- [ ] **Step 8: Implement metrics**

```python
# utils/metrics.py
import numpy as np


def compute_control_metrics(
    lateral_deviations: list,
    heading_errors: list,
    steering_angles: list,
) -> dict:
    lat = np.array(lateral_deviations)
    head = np.array(heading_errors)
    steer = np.array(steering_angles)
    steer_rate = np.diff(steer)
    return {
        "lateral_mean": float(np.mean(np.abs(lat))),
        "lateral_std": float(np.std(lat)),
        "heading_mean": float(np.mean(np.abs(head))),
        "steering_smoothness": float(np.mean(np.abs(steer_rate))),
    }
```

- [ ] **Step 9: Implement plotting utility**

```python
# utils/plotting.py
import os
import csv
import matplotlib.pyplot as plt


def plot_training_curves(log_dir: str, tags: list, save_path: str = None):
    fig, axes = plt.subplots(len(tags), 1, figsize=(10, 4 * len(tags)))
    if len(tags) == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        csv_path = os.path.join(log_dir, f"{tag}.csv")
        steps, values = [], []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                values.append(float(row["value"]))
        ax.plot(steps, values)
        ax.set_xlabel("Episode")
        ax.set_ylabel(tag)
        ax.set_title(tag)
        ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_comparison(results: dict, metric: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo_name, data in results.items():
        ax.plot(data["steps"], data[metric], label=algo_name)
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison: {metric}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()
```

- [ ] **Step 10: Run all tests**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_utils.py -v`
Expected: All PASS

- [ ] **Step 11: Commit**

```bash
git add environment.yml requirements.txt utils/ tests/test_utils.py
git commit -m "feat: add project setup and shared utilities (logger, metrics, plotting)"
```

---

## Task 2: Highway-Env Lane Keeping Wrapper

**Files:**
- Create: `envs/__init__.py`
- Create: `envs/highway_lane_keeping.py`
- Create: `tests/test_envs.py`

- [ ] **Step 1: Write test for environment wrapper**

```python
# tests/test_envs.py
import numpy as np
from envs.highway_lane_keeping import make_lane_keeping_env


def test_env_creates_successfully():
    env = make_lane_keeping_env()
    assert env is not None
    obs, info = env.reset()
    assert obs is not None
    env.close()


def test_env_observation_shape():
    env = make_lane_keeping_env()
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert len(obs.shape) >= 1
    env.close()


def test_env_step():
    env = make_lane_keeping_env()
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    assert next_obs is not None
    assert isinstance(reward, (int, float))
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_envs.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'envs'"

- [ ] **Step 3: Implement lane keeping environment wrapper**

```python
# envs/__init__.py
```

```python
# envs/highway_lane_keeping.py
import gymnasium as gym
import highway_env


def make_lane_keeping_env(render_mode: str = None) -> gym.Env:
    env = gym.make("highway-v0", render_mode=render_mode)
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "features": ["x", "y", "vx", "vy", "heading"],
            "vehicles_count": 5,
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 3,
        "vehicles_count": 10,
        "duration": 60,
        "policy_frequency": 5,
    })
    env.reset()
    return env


def make_continuous_lane_keeping_env(render_mode: str = None) -> gym.Env:
    env = gym.make("highway-v0", render_mode=render_mode)
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "features": ["x", "y", "vx", "vy", "heading"],
            "vehicles_count": 5,
            "absolute": False,
        },
        "action": {
            "type": "ContinuousAction",
        },
        "lanes_count": 3,
        "vehicles_count": 10,
        "duration": 60,
        "policy_frequency": 5,
    })
    env.reset()
    return env
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_envs.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add envs/ tests/test_envs.py
git commit -m "feat: add highway-env lane keeping wrapper with discrete and continuous action modes"
```

---

## Task 3: Q-Learning on CliffWalking

**Files:**
- Create: `algorithms/q_learning/__init__.py`
- Create: `algorithms/q_learning/agent.py`
- Create: `algorithms/q_learning/train.py`
- Create: `algorithms/q_learning/config.py`
- Create: `tests/test_agents.py`

- [ ] **Step 1: Write test for Q-Learning agent**

```python
# tests/test_agents.py
import numpy as np
from algorithms.q_learning.agent import QLearningAgent


def test_qlearning_agent_init():
    agent = QLearningAgent(n_states=48, n_actions=4, lr=0.1, gamma=0.99, epsilon=0.1)
    assert agent.q_table.shape == (48, 4)
    assert np.all(agent.q_table == 0)


def test_qlearning_agent_select_action():
    agent = QLearningAgent(n_states=48, n_actions=4, lr=0.1, gamma=0.99, epsilon=0.0)
    agent.q_table[0, 2] = 10.0
    action = agent.select_action(0)
    assert action == 2


def test_qlearning_agent_update():
    agent = QLearningAgent(n_states=48, n_actions=4, lr=0.5, gamma=0.9, epsilon=0.1)
    agent.update(state=0, action=1, reward=1.0, next_state=1, done=False)
    assert agent.q_table[0, 1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_agents.py::test_qlearning_agent_init -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement Q-Learning agent**

```python
# algorithms/q_learning/__init__.py
```

```python
# algorithms/q_learning/config.py
config = {
    "n_episodes": 500,
    "lr": 0.1,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
}
```

```python
# algorithms/q_learning/agent.py
import numpy as np


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, lr: float, gamma: float, epsilon: float):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_agents.py -v`
Expected: All PASS

- [ ] **Step 5: Implement training script**

```python
# algorithms/q_learning/train.py
import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import gymnasium as gym
import numpy as np
from algorithms.q_learning.agent import QLearningAgent
from algorithms.q_learning.config import config
from utils.logger import Logger
from utils.plotting import plot_training_curves


def train():
    env = gym.make("CliffWalking-v0")
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon=config["epsilon_start"],
    )
    logger = Logger(log_dir="algorithms/q_learning/results", use_tensorboard=False)

    for episode in range(config["n_episodes"]):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.epsilon = max(
            config["epsilon_end"],
            agent.epsilon * config["epsilon_decay"],
        )
        logger.log("reward", total_reward, step=episode)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f}")

    logger.save()
    plot_training_curves(
        log_dir="algorithms/q_learning/results",
        tags=["reward"],
        save_path="algorithms/q_learning/results/training_curve.png",
    )
    env.close()
    print("Training complete. Results saved to algorithms/q_learning/results/")


if __name__ == "__main__":
    train()
```

- [ ] **Step 6: Run training to verify it works**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python algorithms/q_learning/train.py`
Expected: Training output with increasing rewards, generates `algorithms/q_learning/results/training_curve.png`

- [ ] **Step 7: Commit**

```bash
git add algorithms/q_learning/ tests/test_agents.py
git commit -m "feat: implement Q-Learning agent with CliffWalking training"
```

---

## Task 4: DQN on CartPole and Highway-Env

**Files:**
- Create: `algorithms/dqn/__init__.py`
- Create: `algorithms/dqn/agent.py`
- Create: `algorithms/dqn/train.py`
- Create: `algorithms/dqn/config.py`
- Modify: `tests/test_agents.py`

- [ ] **Step 1: Write test for DQN agent**

```python
# append to tests/test_agents.py
import torch
from algorithms.dqn.agent import DQNAgent


def test_dqn_agent_init():
    agent = DQNAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99, epsilon=1.0, buffer_size=1000, batch_size=32)
    assert agent is not None


def test_dqn_agent_select_action():
    agent = DQNAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99, epsilon=0.0, buffer_size=1000, batch_size=32)
    state = np.zeros(4)
    action = agent.select_action(state)
    assert action in [0, 1]


def test_dqn_agent_store_and_learn():
    agent = DQNAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99, epsilon=1.0, buffer_size=100, batch_size=4)
    for i in range(10):
        state = np.random.randn(4)
        action = agent.select_action(state)
        next_state = np.random.randn(4)
        agent.store_transition(state, action, 1.0, next_state, False)
    loss = agent.learn()
    assert loss is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_agents.py::test_dqn_agent_init -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement DQN agent**

```python
# algorithms/dqn/__init__.py
```

```python
# algorithms/dqn/config.py
config = {
    "n_episodes": 300,
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "buffer_size": 10000,
    "batch_size": 64,
    "target_update_freq": 10,
    "hidden_dim": 128,
}
```

```python
# algorithms/dqn/agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float, gamma: float,
                 epsilon: float, buffer_size: int, batch_size: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1)[0]
        targets = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_agents.py::test_dqn_agent_init tests/test_agents.py::test_dqn_agent_select_action tests/test_agents.py::test_dqn_agent_store_and_learn -v`
Expected: All PASS

- [ ] **Step 5: Implement training script**

```python
# algorithms/dqn/train.py
import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import gymnasium as gym
import numpy as np
from algorithms.dqn.agent import DQNAgent
from algorithms.dqn.config import config
from utils.logger import Logger
from utils.plotting import plot_training_curves


def train_cartpole():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon=config["epsilon_start"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
    )
    logger = Logger(log_dir="algorithms/dqn/results/cartpole", use_tensorboard=False)

    for episode in range(config["n_episodes"]):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()
            state = next_state
            total_reward += reward

        agent.epsilon = max(config["epsilon_end"], agent.epsilon * config["epsilon_decay"])
        if (episode + 1) % config["target_update_freq"] == 0:
            agent.update_target()

        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"[CartPole] Episode {episode + 1}, Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f}")

    logger.save()
    plot_training_curves("algorithms/dqn/results/cartpole", ["reward"], "algorithms/dqn/results/cartpole/training_curve.png")
    env.close()
    print("CartPole training complete.")


def train_highway():
    from envs.highway_lane_keeping import make_lane_keeping_env
    env = make_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        epsilon=config["epsilon_start"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
    )
    logger = Logger(log_dir="algorithms/dqn/results/highway", use_tensorboard=False)

    for episode in range(config["n_episodes"]):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_obs.flatten()
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        agent.epsilon = max(config["epsilon_end"], agent.epsilon * config["epsilon_decay"])
        if (episode + 1) % config["target_update_freq"] == 0:
            agent.update_target()

        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"[Highway] Episode {episode + 1}, Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f}")

    logger.save()
    agent.save("algorithms/dqn/results/highway/model.pt")
    plot_training_curves("algorithms/dqn/results/highway", ["reward"], "algorithms/dqn/results/highway/training_curve.png")
    env.close()
    print("Highway training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "highway", "both"], default="both")
    args = parser.parse_args()
    if args.env in ["cartpole", "both"]:
        train_cartpole()
    if args.env in ["highway", "both"]:
        train_highway()
```

- [ ] **Step 6: Run CartPole training to verify**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python algorithms/dqn/train.py --env cartpole`
Expected: Reward increases toward 500 over 300 episodes

- [ ] **Step 7: Commit**

```bash
git add algorithms/dqn/ tests/test_agents.py
git commit -m "feat: implement DQN agent with CartPole and highway-env training"
```

---

## Task 5: Policy Gradient (REINFORCE) on CartPole and Highway-Env

**Files:**
- Create: `algorithms/policy_gradient/__init__.py`
- Create: `algorithms/policy_gradient/agent.py`
- Create: `algorithms/policy_gradient/train.py`
- Create: `algorithms/policy_gradient/config.py`
- Modify: `tests/test_agents.py`

- [ ] **Step 1: Write test for REINFORCE agent**

```python
# append to tests/test_agents.py
from algorithms.policy_gradient.agent import REINFORCEAgent


def test_reinforce_agent_init():
    agent = REINFORCEAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99)
    assert agent is not None


def test_reinforce_agent_select_action():
    agent = REINFORCEAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99)
    state = np.zeros(4)
    action = agent.select_action(state)
    assert action in [0, 1]


def test_reinforce_agent_update():
    agent = REINFORCEAgent(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99)
    state = np.zeros(4)
    for _ in range(5):
        action = agent.select_action(state)
        agent.store_reward(1.0)
    loss = agent.update()
    assert loss is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_agents.py::test_reinforce_agent_init -v`
Expected: FAIL

- [ ] **Step 3: Implement REINFORCE agent**

```python
# algorithms/policy_gradient/__init__.py
```

```python
# algorithms/policy_gradient/config.py
config = {
    "n_episodes": 500,
    "lr": 1e-3,
    "gamma": 0.99,
    "hidden_dim": 128,
}
```

```python
# algorithms/policy_gradient/agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class REINFORCEAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float, gamma: float, hidden_dim: int = 128):
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self) -> float:
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss -= log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        return loss.item()

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_agents.py::test_reinforce_agent_init tests/test_agents.py::test_reinforce_agent_select_action tests/test_agents.py::test_reinforce_agent_update -v`
Expected: All PASS

- [ ] **Step 5: Implement training script**

```python
# algorithms/policy_gradient/train.py
import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import gymnasium as gym
import numpy as np
from algorithms.policy_gradient.agent import REINFORCEAgent
from algorithms.policy_gradient.config import config
from utils.logger import Logger
from utils.plotting import plot_training_curves


def train_cartpole():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        hidden_dim=config["hidden_dim"],
    )
    logger = Logger(log_dir="algorithms/policy_gradient/results/cartpole", use_tensorboard=False)

    for episode in range(config["n_episodes"]):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward)
            state = next_state
            total_reward += reward

        agent.update()
        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"[CartPole] Episode {episode + 1}, Reward: {total_reward:.1f}")

    logger.save()
    plot_training_curves("algorithms/policy_gradient/results/cartpole", ["reward"], "algorithms/policy_gradient/results/cartpole/training_curve.png")
    env.close()
    print("CartPole training complete.")


def train_highway():
    from envs.highway_lane_keeping import make_lane_keeping_env
    env = make_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        hidden_dim=config["hidden_dim"],
    )
    logger = Logger(log_dir="algorithms/policy_gradient/results/highway", use_tensorboard=False)

    for episode in range(config["n_episodes"]):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward)
            state = next_obs.flatten()
            total_reward += reward

        agent.update()
        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"[Highway] Episode {episode + 1}, Reward: {total_reward:.1f}")

    logger.save()
    agent.save("algorithms/policy_gradient/results/highway/model.pt")
    plot_training_curves("algorithms/policy_gradient/results/highway", ["reward"], "algorithms/policy_gradient/results/highway/training_curve.png")
    env.close()
    print("Highway training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "highway", "both"], default="both")
    args = parser.parse_args()
    if args.env in ["cartpole", "both"]:
        train_cartpole()
    if args.env in ["highway", "both"]:
        train_highway()
```

- [ ] **Step 6: Run CartPole training to verify**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python algorithms/policy_gradient/train.py --env cartpole`
Expected: Reward increases over 500 episodes

- [ ] **Step 7: Commit**

```bash
git add algorithms/policy_gradient/ tests/test_agents.py
git commit -m "feat: implement REINFORCE (policy gradient) agent with CartPole and highway-env training"
```

---

## Task 6: PPO on CartPole and Highway-Env

**Files:**
- Create: `algorithms/ppo/__init__.py`
- Create: `algorithms/ppo/agent.py`
- Create: `algorithms/ppo/train.py`
- Create: `algorithms/ppo/config.py`
- Modify: `tests/test_agents.py`

- [ ] **Step 1: Write test for PPO agent**

```python
# append to tests/test_agents.py
from algorithms.ppo.agent import PPOAgent


def test_ppo_agent_init():
    agent = PPOAgent(state_dim=4, action_dim=2, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=10, batch_size=64)
    assert agent is not None


def test_ppo_agent_select_action():
    agent = PPOAgent(state_dim=4, action_dim=2, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=10, batch_size=64)
    state = np.zeros(4)
    action, log_prob, value = agent.select_action(state)
    assert action in [0, 1]
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_ppo_agent_update():
    agent = PPOAgent(state_dim=4, action_dim=2, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=2, batch_size=4)
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    for _ in range(10):
        s = np.random.randn(4)
        a, lp, v = agent.select_action(s)
        states.append(s)
        actions.append(a)
        rewards.append(1.0)
        log_probs.append(lp)
        values.append(v)
        dones.append(False)
    loss = agent.update(states, actions, rewards, log_probs, values, dones, next_value=0.0)
    assert loss is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_agents.py::test_ppo_agent_init -v`
Expected: FAIL

- [ ] **Step 3: Implement PPO agent**

```python
# algorithms/ppo/__init__.py
```

```python
# algorithms/ppo/config.py
config = {
    "n_episodes": 500,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs": 10,
    "batch_size": 64,
    "hidden_dim": 128,
    "steps_per_update": 2048,
}
```

```python
# algorithms/ppo/agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = torch.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float, gamma: float,
                 clip_eps: float, epochs: int, batch_size: int, hidden_dim: int = 128,
                 gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ac = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def select_action(self, state: np.ndarray) -> tuple:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, value = self.ac(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def update(self, states, actions, rewards, log_probs, values, dones, next_value) -> float:
        advantages = []
        gae = 0
        values_ext = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(log_probs).to(self.device)

        total_loss = 0
        n = len(states)
        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                probs, values_pred = self.ac(states_t[idx])
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values_pred.squeeze(), returns[idx])
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss

    def save(self, path: str):
        torch.save(self.ac.state_dict(), path)

    def load(self, path: str):
        self.ac.load_state_dict(torch.load(path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_agents.py::test_ppo_agent_init tests/test_agents.py::test_ppo_agent_select_action tests/test_agents.py::test_ppo_agent_update -v`
Expected: All PASS

- [ ] **Step 5: Implement training script**

```python
# algorithms/ppo/train.py
import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import gymnasium as gym
import numpy as np
from algorithms.ppo.agent import PPOAgent
from algorithms.ppo.config import config
from utils.logger import Logger
from utils.plotting import plot_training_curves


def train_cartpole():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        clip_eps=config["clip_eps"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        gae_lambda=config["gae_lambda"],
    )
    logger = Logger(log_dir="algorithms/ppo/results/cartpole", use_tensorboard=False)

    for episode in range(config["n_episodes"]):
        state, _ = env.reset()
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        total_reward = 0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            state = next_state
            total_reward += reward

        _, _, next_value = agent.select_action(state)
        agent.update(states, actions, rewards, log_probs, values, dones, next_value=next_value if not done else 0.0)

        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"[CartPole] Episode {episode + 1}, Reward: {total_reward:.1f}")

    logger.save()
    plot_training_curves("algorithms/ppo/results/cartpole", ["reward"], "algorithms/ppo/results/cartpole/training_curve.png")
    env.close()
    print("CartPole training complete.")


def train_highway():
    from envs.highway_lane_keeping import make_lane_keeping_env
    env = make_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        clip_eps=config["clip_eps"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        gae_lambda=config["gae_lambda"],
    )
    logger = Logger(log_dir="algorithms/ppo/results/highway", use_tensorboard=False)

    for episode in range(config["n_episodes"]):
        obs, _ = env.reset()
        state = obs.flatten()
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        total_reward = 0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            state = next_obs.flatten()
            total_reward += reward

        next_value = 0.0 if done else agent.select_action(state)[2]
        agent.update(states, actions, rewards, log_probs, values, dones, next_value=next_value)

        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"[Highway] Episode {episode + 1}, Reward: {total_reward:.1f}")

    logger.save()
    agent.save("algorithms/ppo/results/highway/model.pt")
    plot_training_curves("algorithms/ppo/results/highway", ["reward"], "algorithms/ppo/results/highway/training_curve.png")
    env.close()
    print("Highway training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "highway", "both"], default="both")
    args = parser.parse_args()
    if args.env in ["cartpole", "both"]:
        train_cartpole()
    if args.env in ["highway", "both"]:
        train_highway()
```

- [ ] **Step 6: Run CartPole training to verify**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python algorithms/ppo/train.py --env cartpole`
Expected: Reward increases toward 500 within 300 episodes

- [ ] **Step 7: Commit**

```bash
git add algorithms/ppo/ tests/test_agents.py
git commit -m "feat: implement PPO agent with CartPole and highway-env training"
```

---

## Task 7: SAC on Pendulum and Highway-Env (Continuous Action)

**Files:**
- Create: `algorithms/sac/__init__.py`
- Create: `algorithms/sac/agent.py`
- Create: `algorithms/sac/train.py`
- Create: `algorithms/sac/config.py`
- Modify: `tests/test_agents.py`

- [ ] **Step 1: Write test for SAC agent**

```python
# append to tests/test_agents.py
from algorithms.sac.agent import SACAgent


def test_sac_agent_init():
    agent = SACAgent(state_dim=3, action_dim=1, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=10000, batch_size=64)
    assert agent is not None


def test_sac_agent_select_action():
    agent = SACAgent(state_dim=3, action_dim=1, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=10000, batch_size=64)
    state = np.zeros(3)
    action = agent.select_action(state)
    assert action.shape == (1,)
    assert -1.0 <= action[0] <= 1.0


def test_sac_agent_store_and_learn():
    agent = SACAgent(state_dim=3, action_dim=1, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=100, batch_size=4)
    for _ in range(10):
        state = np.random.randn(3)
        action = agent.select_action(state)
        next_state = np.random.randn(3)
        agent.store_transition(state, action, 1.0, next_state, False)
    result = agent.learn()
    assert result is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_agents.py::test_sac_agent_init -v`
Expected: FAIL

- [ ] **Step 3: Implement SAC agent**

```python
# algorithms/sac/__init__.py
```

```python
# algorithms/sac/config.py
config = {
    "n_episodes": 300,
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "auto_alpha": True,
    "buffer_size": 100000,
    "batch_size": 256,
    "hidden_dim": 256,
    "start_steps": 1000,
}
```

```python
# algorithms/sac/agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float, gamma: float,
                 tau: float, alpha: float, buffer_size: int, batch_size: int,
                 hidden_dim: int = 256, auto_alpha: bool = True):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state_t)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(state_t)
        return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states_t)
            q1_next = self.q1_target(next_states_t, next_actions)
            q2_next = self.q2_target(next_states_t, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards_t + self.gamma * (1 - dones_t) * q_next

        q1_loss = nn.MSELoss()(self.q1(states_t, actions_t), q_target)
        q2_loss = nn.MSELoss()(self.q2(states_t, actions_t), q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, log_probs = self.policy.sample(states_t)
        q1_new = self.q1(states_t, new_actions)
        q2_new = self.q2(states_t, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"q1_loss": q1_loss.item(), "q2_loss": q2_loss.item(), "policy_loss": policy_loss.item()}

    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/test_agents.py::test_sac_agent_init tests/test_agents.py::test_sac_agent_select_action tests/test_agents.py::test_sac_agent_store_and_learn -v`
Expected: All PASS

- [ ] **Step 5: Implement training script**

```python
# algorithms/sac/train.py
import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import gymnasium as gym
import numpy as np
from algorithms.sac.agent import SACAgent
from algorithms.sac.config import config
from utils.logger import Logger
from utils.plotting import plot_training_curves


def train_pendulum():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config["alpha"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        auto_alpha=config["auto_alpha"],
    )
    logger = Logger(log_dir="algorithms/sac/results/pendulum", use_tensorboard=False)
    total_steps = 0

    for episode in range(config["n_episodes"]):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if total_steps < config["start_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            if total_steps >= config["start_steps"]:
                agent.learn()
            state = next_state
            total_reward += reward
            total_steps += 1

        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"[Pendulum] Episode {episode + 1}, Reward: {total_reward:.1f}")

    logger.save()
    plot_training_curves("algorithms/sac/results/pendulum", ["reward"], "algorithms/sac/results/pendulum/training_curve.png")
    env.close()
    print("Pendulum training complete.")


def train_highway():
    from envs.highway_lane_keeping import make_continuous_lane_keeping_env
    env = make_continuous_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config["alpha"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        auto_alpha=config["auto_alpha"],
    )
    logger = Logger(log_dir="algorithms/sac/results/highway", use_tensorboard=False)
    total_steps = 0

    for episode in range(config["n_episodes"]):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        done = False

        while not done:
            if total_steps < config["start_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_obs.flatten()
            agent.store_transition(state, action, reward, next_state, done)
            if total_steps >= config["start_steps"]:
                agent.learn()
            state = next_state
            total_reward += reward
            total_steps += 1

        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"[Highway] Episode {episode + 1}, Reward: {total_reward:.1f}")

    logger.save()
    agent.save("algorithms/sac/results/highway/model.pt")
    plot_training_curves("algorithms/sac/results/highway", ["reward"], "algorithms/sac/results/highway/training_curve.png")
    env.close()
    print("Highway training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["pendulum", "highway", "both"], default="both")
    args = parser.parse_args()
    if args.env in ["pendulum", "both"]:
        train_pendulum()
    if args.env in ["highway", "both"]:
        train_highway()
```

- [ ] **Step 6: Run Pendulum training to verify**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python algorithms/sac/train.py --env pendulum`
Expected: Reward increases from ~-1500 toward ~-200 over 300 episodes

- [ ] **Step 7: Commit**

```bash
git add algorithms/sac/ tests/test_agents.py
git commit -m "feat: implement SAC agent with Pendulum and highway-env continuous control training"
```

---

## Task 8: Comparison Experiment

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/run_comparison.py`
- Create: `experiments/evaluate.py`
- Create: `experiments/plot_results.py`

- [ ] **Step 1: Implement evaluation script**

```python
# experiments/__init__.py
```

```python
# experiments/evaluate.py
import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import numpy as np
from utils.metrics import compute_control_metrics


def evaluate_agent(env, agent, n_episodes: int = 20, flatten_obs: bool = True) -> dict:
    all_rewards = []
    all_lateral = []
    all_heading = []
    all_steering = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten() if flatten_obs else obs
        episode_reward = 0
        episode_lateral = []
        episode_heading = []
        episode_steering = []
        done = False

        while not done:
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state)
                if isinstance(action, np.ndarray):
                    pass
                else:
                    action = int(action) if not isinstance(action, tuple) else int(action[0])

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ego = env.unwrapped.vehicle
            if ego is not None:
                lane = ego.lane
                if lane is not None:
                    lane_coords = lane.local_coordinates(ego.position)
                    episode_lateral.append(lane_coords[1])
                episode_heading.append(ego.heading)
                if len(episode_steering) > 0:
                    episode_steering.append(ego.action.get("steering", 0.0))
                else:
                    episode_steering.append(0.0)

            state = next_obs.flatten() if flatten_obs else next_obs
            episode_reward += reward

        all_rewards.append(episode_reward)
        all_lateral.extend(episode_lateral)
        all_heading.extend(episode_heading)
        all_steering.extend(episode_steering)

    control_metrics = compute_control_metrics(all_lateral, all_heading, all_steering)
    control_metrics["mean_reward"] = float(np.mean(all_rewards))
    control_metrics["std_reward"] = float(np.std(all_rewards))
    return control_metrics
```

- [ ] **Step 2: Implement comparison runner**

```python
# experiments/run_comparison.py
import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import json
import numpy as np
from envs.highway_lane_keeping import make_lane_keeping_env
from algorithms.dqn.agent import DQNAgent
from algorithms.dqn.config import config as dqn_config
from algorithms.ppo.agent import PPOAgent
from algorithms.ppo.config import config as ppo_config
from algorithms.sac.agent import SACAgent
from algorithms.sac.config import config as sac_config
from utils.logger import Logger
from experiments.evaluate import evaluate_agent


def train_and_evaluate_dqn(n_episodes: int = 300):
    print("=" * 50)
    print("Training DQN...")
    print("=" * 50)
    env = make_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(
        state_dim=state_dim, action_dim=action_dim,
        lr=dqn_config["lr"], gamma=dqn_config["gamma"],
        epsilon=dqn_config["epsilon_start"], buffer_size=dqn_config["buffer_size"],
        batch_size=dqn_config["batch_size"], hidden_dim=dqn_config["hidden_dim"],
    )
    logger = Logger(log_dir="experiments/results/dqn", use_tensorboard=False)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_obs.flatten()
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
        agent.epsilon = max(dqn_config["epsilon_end"], agent.epsilon * dqn_config["epsilon_decay"])
        if (episode + 1) % dqn_config["target_update_freq"] == 0:
            agent.update_target()
        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"  Episode {episode + 1}, Reward: {total_reward:.1f}")

    logger.save()
    agent.epsilon = 0.0
    metrics = evaluate_agent(env, agent, n_episodes=20)
    env.close()
    return metrics, logger


def train_and_evaluate_ppo(n_episodes: int = 300):
    print("=" * 50)
    print("Training PPO...")
    print("=" * 50)
    env = make_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(
        state_dim=state_dim, action_dim=action_dim,
        lr=ppo_config["lr"], gamma=ppo_config["gamma"],
        clip_eps=ppo_config["clip_eps"], epochs=ppo_config["epochs"],
        batch_size=ppo_config["batch_size"], hidden_dim=ppo_config["hidden_dim"],
        gae_lambda=ppo_config["gae_lambda"],
    )
    logger = Logger(log_dir="experiments/results/ppo", use_tensorboard=False)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        total_reward = 0
        done = False
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            state = next_obs.flatten()
            total_reward += reward
        next_value = 0.0 if done else agent.select_action(state)[2]
        agent.update(states, actions, rewards, log_probs, values, dones, next_value=next_value)
        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"  Episode {episode + 1}, Reward: {total_reward:.1f}")

    logger.save()
    metrics = evaluate_agent(env, agent, n_episodes=20)
    env.close()
    return metrics, logger


def train_and_evaluate_sac(n_episodes: int = 300):
    print("=" * 50)
    print("Training SAC...")
    print("=" * 50)
    from envs.highway_lane_keeping import make_continuous_lane_keeping_env
    env = make_continuous_lane_keeping_env()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim,
        lr=sac_config["lr"], gamma=sac_config["gamma"],
        tau=sac_config["tau"], alpha=sac_config["alpha"],
        buffer_size=sac_config["buffer_size"], batch_size=sac_config["batch_size"],
        hidden_dim=sac_config["hidden_dim"], auto_alpha=sac_config["auto_alpha"],
    )
    logger = Logger(log_dir="experiments/results/sac", use_tensorboard=False)
    total_steps = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        total_reward = 0
        done = False
        while not done:
            if total_steps < sac_config["start_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_obs.flatten()
            agent.store_transition(state, action, reward, next_state, done)
            if total_steps >= sac_config["start_steps"]:
                agent.learn()
            state = next_state
            total_reward += reward
            total_steps += 1
        logger.log("reward", total_reward, step=episode)
        if (episode + 1) % 50 == 0:
            print(f"  Episode {episode + 1}, Reward: {total_reward:.1f}")

    logger.save()
    metrics = evaluate_agent(env, agent, n_episodes=20)
    env.close()
    return metrics, logger


def main():
    import os
    os.makedirs("experiments/results", exist_ok=True)

    dqn_metrics, _ = train_and_evaluate_dqn()
    ppo_metrics, _ = train_and_evaluate_ppo()
    sac_metrics, _ = train_and_evaluate_sac()

    results = {"DQN": dqn_metrics, "PPO": ppo_metrics, "SAC": sac_metrics}
    with open("experiments/results/comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    for algo, m in results.items():
        print(f"\n{algo}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Implement comparison plot script**

```python
# experiments/plot_results.py
import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def load_reward_curve(log_dir: str) -> tuple:
    csv_path = os.path.join(log_dir, "reward.csv")
    steps, values = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            values.append(float(row["value"]))
    return steps, values


def plot_reward_comparison(save_path: str = "experiments/results/reward_comparison.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo, log_dir in [("DQN", "experiments/results/dqn"),
                          ("PPO", "experiments/results/ppo"),
                          ("SAC", "experiments/results/sac")]:
        if os.path.exists(os.path.join(log_dir, "reward.csv")):
            steps, values = load_reward_curve(log_dir)
            window = 10
            smoothed = np.convolve(values, np.ones(window)/window, mode="valid")
            ax.plot(range(len(smoothed)), smoothed, label=algo)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Lane Keeping: Reward Comparison")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Reward comparison saved to {save_path}")


def plot_control_metrics(save_path: str = "experiments/results/control_metrics.png"):
    with open("experiments/results/comparison.json", "r") as f:
        results = json.load(f)

    metrics = ["lateral_mean", "lateral_std", "heading_mean", "steering_smoothness"]
    labels = ["Lateral Dev (mean)", "Lateral Dev (std)", "Heading Error (mean)", "Steering Smoothness"]
    algos = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [results[algo].get(metric, 0) for algo in algos]
        axes[i].bar(algos, values)
        axes[i].set_ylabel(label)
        axes[i].set_title(label)
        axes[i].grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Control metrics saved to {save_path}")


if __name__ == "__main__":
    plot_reward_comparison()
    plot_control_metrics()
```

- [ ] **Step 4: Run comparison (after all algorithms are trained)**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python experiments/run_comparison.py`
Expected: Trains all 3 algorithms, outputs comparison metrics, saves `experiments/results/comparison.json`

- [ ] **Step 5: Generate comparison plots**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python experiments/plot_results.py`
Expected: Generates `reward_comparison.png` and `control_metrics.png`

- [ ] **Step 6: Commit**

```bash
git add experiments/
git commit -m "feat: add comparison experiment with control-quality evaluation metrics"
```

---

## Task 9: Final Cleanup and Documentation

**Files:**
- Modify: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

```
# .gitignore
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
*.pt
*.pth
results/
!experiments/results/.gitkeep
.ipynb_checkpoints/
*.log
runs/
```

- [ ] **Step 2: Update README**

```markdown
# easyRL

A progressive reinforcement learning study project for autonomous driving control engineers.

## Learning Path

```
Q-Learning → DQN → Policy Gradient (REINFORCE) → PPO → SAC
```

## Setup

```bash
conda env create -f environment.yml
conda activate easyrl
```

## Project Structure

```
algorithms/     - RL algorithm implementations
envs/           - highway-env wrappers
utils/          - shared utilities (plotting, logging, metrics)
experiments/    - lane-keeping comparison experiments
tests/          - unit tests
```

## Usage

Train individual algorithms:
```bash
python algorithms/q_learning/train.py
python algorithms/dqn/train.py --env both
python algorithms/policy_gradient/train.py --env both
python algorithms/ppo/train.py --env both
python algorithms/sac/train.py --env both
```

Run comparison experiment:
```bash
python experiments/run_comparison.py
python experiments/plot_results.py
```

## Evaluation Metrics

- Cumulative reward and convergence speed
- Lateral deviation (mean + std)
- Heading angle deviation
- Steering smoothness (steering angle rate of change)
```

- [ ] **Step 3: Run full test suite**

Run: `cd /home/lihongl/Desktop/myRL/easyRL && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add .gitignore README.md
git commit -m "docs: update README with project overview and usage instructions"
```

---

# easyRL 实现计划（中文版）

> **供代理执行者使用：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框 (`- [ ]`) 语法跟踪。

**目标：** 构建一个从 Q-Learning 到 SAC 的渐进式 RL 学习项目，最终在 highway-env 上进行车道保持对比实验。

**架构：** 每个算法在各自目录中，拥有一致的接口（agent、训练脚本、配置）。共享工具处理绘图、日志和评估指标。最终实验模块将训练好的智能体整合到一起进行对比。

**技术栈：** PyTorch、highway-env (Gymnasium)、matplotlib、TensorBoard（可选）、conda

---

## 文件结构

```
easyRL/
├── requirements.txt
├── environment.yml                    # conda 环境定义
├── utils/
│   ├── __init__.py
│   ├── plotting.py                    # matplotlib 图表生成
│   ├── logger.py                      # 训练日志器（CSV + 可选 TensorBoard）
│   └── metrics.py                     # 控制质量评估指标
├── envs/
│   ├── __init__.py
│   └── highway_lane_keeping.py        # highway-env 车道保持封装
├── algorithms/
│   ├── q_learning/
│   │   ├── __init__.py
│   │   ├── agent.py                   # Q-table 智能体
│   │   ├── train.py                   # CliffWalking 训练
│   │   └── config.py                  # 超参数
│   ├── dqn/
│   │   ├── __init__.py
│   │   ├── agent.py                   # 带经验回放的 DQN 智能体
│   │   ├── train.py                   # CartPole + highway-env 训练
│   │   └── config.py
│   ├── policy_gradient/
│   │   ├── __init__.py
│   │   ├── agent.py                   # REINFORCE 智能体
│   │   ├── train.py                   # CartPole + highway-env 训练
│   │   └── config.py
│   ├── ppo/
│   │   ├── __init__.py
│   │   ├── agent.py                   # 带裁剪目标的 PPO 智能体
│   │   ├── train.py                   # CartPole + highway-env 训练
│   │   └── config.py
│   └── sac/
│       ├── __init__.py
│       ├── agent.py                   # 带熵调节的 SAC 智能体
│       ├── train.py                   # Pendulum + highway-env 训练
│       └── config.py
├── experiments/
│   ├── __init__.py
│   ├── run_comparison.py              # 对 3 个算法执行车道保持训练
│   ├── evaluate.py                    # 计算控制质量指标
│   └── plot_results.py                # 生成对比图表
└── tests/
    ├── test_utils.py
    ├── test_envs.py
    └── test_agents.py
```

---

## 任务 1：项目初始化与共享工具

**文件：** environment.yml、requirements.txt、utils/（__init__.py、plotting.py、logger.py、metrics.py）、tests/test_utils.py

- [ ] 步骤 1：创建 conda 环境文件
- [ ] 步骤 2：创建 requirements.txt（pip 备选）
- [ ] 步骤 3：编写 logger 测试
- [ ] 步骤 4：运行测试验证失败
- [ ] 步骤 5：实现 logger
- [ ] 步骤 6：运行测试验证通过
- [ ] 步骤 7：编写 metrics 测试
- [ ] 步骤 8：实现 metrics
- [ ] 步骤 9：实现 plotting 工具
- [ ] 步骤 10：运行所有测试
- [ ] 步骤 11：提交

---

## 任务 2：Highway-Env 车道保持封装

**文件：** envs/（__init__.py、highway_lane_keeping.py）、tests/test_envs.py

- [ ] 步骤 1：编写环境封装测试
- [ ] 步骤 2：运行测试验证失败
- [ ] 步骤 3：实现车道保持环境封装
- [ ] 步骤 4：运行测试验证通过
- [ ] 步骤 5：提交

---

## 任务 3：Q-Learning（CliffWalking）

**文件：** algorithms/q_learning/（__init__.py、agent.py、train.py、config.py）、tests/test_agents.py

- [ ] 步骤 1：编写 Q-Learning 智能体测试
- [ ] 步骤 2：运行测试验证失败
- [ ] 步骤 3：实现 Q-Learning 智能体
- [ ] 步骤 4：运行测试验证通过
- [ ] 步骤 5：实现训练脚本
- [ ] 步骤 6：运行训练验证可行
- [ ] 步骤 7：提交

---

## 任务 4：DQN（CartPole + Highway-Env）

**文件：** algorithms/dqn/（__init__.py、agent.py、train.py、config.py）、tests/test_agents.py

- [ ] 步骤 1：编写 DQN 智能体测试
- [ ] 步骤 2：运行测试验证失败
- [ ] 步骤 3：实现 DQN 智能体
- [ ] 步骤 4：运行测试验证通过
- [ ] 步骤 5：实现训练脚本
- [ ] 步骤 6：运行 CartPole 训练验证
- [ ] 步骤 7：提交

---

## 任务 5：Policy Gradient / REINFORCE（CartPole + Highway-Env）

**文件：** algorithms/policy_gradient/（__init__.py、agent.py、train.py、config.py）、tests/test_agents.py

- [ ] 步骤 1：编写 REINFORCE 智能体测试
- [ ] 步骤 2：运行测试验证失败
- [ ] 步骤 3：实现 REINFORCE 智能体
- [ ] 步骤 4：运行测试验证通过
- [ ] 步骤 5：实现训练脚本
- [ ] 步骤 6：运行 CartPole 训练验证
- [ ] 步骤 7：提交

---

## 任务 6：PPO（CartPole + Highway-Env）

**文件：** algorithms/ppo/（__init__.py、agent.py、train.py、config.py）、tests/test_agents.py

- [ ] 步骤 1：编写 PPO 智能体测试
- [ ] 步骤 2：运行测试验证失败
- [ ] 步骤 3：实现 PPO 智能体
- [ ] 步骤 4：运行测试验证通过
- [ ] 步骤 5：实现训练脚本
- [ ] 步骤 6：运行 CartPole 训练验证
- [ ] 步骤 7：提交

---

## 任务 7：SAC（Pendulum + Highway-Env 连续动作）

**文件：** algorithms/sac/（__init__.py、agent.py、train.py、config.py）、tests/test_agents.py

- [ ] 步骤 1：编写 SAC 智能体测试
- [ ] 步骤 2：运行测试验证失败
- [ ] 步骤 3：实现 SAC 智能体
- [ ] 步骤 4：运行测试验证通过
- [ ] 步骤 5：实现训练脚本
- [ ] 步骤 6：运行 Pendulum 训练验证
- [ ] 步骤 7：提交

---

## 任务 8：对比实验

**文件：** experiments/（__init__.py、run_comparison.py、evaluate.py、plot_results.py）

- [ ] 步骤 1：实现评估脚本
- [ ] 步骤 2：实现对比运行器
- [ ] 步骤 3：实现对比图表脚本
- [ ] 步骤 4：运行对比实验
- [ ] 步骤 5：生成对比图表
- [ ] 步骤 6：提交

---

## 任务 9：最终整理与文档

**文件：** README.md、.gitignore

- [ ] 步骤 1：创建 .gitignore
- [ ] 步骤 2：更新 README
- [ ] 步骤 3：运行完整测试套件
- [ ] 步骤 4：提交
