# Algorithm-Environment Mapping

| Algorithm | Environment | Action Space | Key Learning Focus |
|-----------|-------------|--------------|-------------------|
| Q-Learning | CliffWalking-v0 | Discrete | TD update, Q-table, exploration-exploitation |
| DQN | highway-env (discrete) | DiscreteMetaAction | Neural net replaces Q-table, experience replay, target network |
| Policy Gradient | highway-env (discrete) | DiscreteMetaAction | Policy gradient, weight log-prob by returns |
| PPO | highway-env (continuous) | ContinuousAction | Clip mechanism, Gaussian policy for continuous control |
| SAC | highway-env (continuous) | ContinuousAction | Maximum entropy framework, off-policy continuous control |

## Algorithm Families

| Algorithm | Family | Core Idea |
|-----------|--------|-----------|
| Q-Learning | Value-based | Learn $Q(s, a)$ and derive the policy with $\arg\max_a Q(s, a)$ |
| DQN | Value-based | Approximate $Q(s, a)$ with a neural network instead of a table |
| Policy Gradient (REINFORCE) | Policy-based | Directly optimize the policy $\pi_\theta(a\|s)$ without a separate critic |
| PPO | Actor-Critic | Learn both a policy and a value function; policy optimization is stabilized by clipping |
| SAC | Actor-Critic | Learn both a stochastic policy and twin Q-functions under a maximum entropy objective |

### Quick Summary

- **Value-based**: Q-Learning, DQN
- **Policy-based**: REINFORCE
- **Actor-Critic**: PPO, SAC
- **Practical note**: PPO and SAC are often grouped under policy optimization methods, but calling them Actor-Critic is more precise because they rely on both actor and critic networks.

## Design Rationale

- **Q-Learning** uses CliffWalking as a minimal tabular environment to build intuition for TD learning.
- **DQN / PG** use highway-env with discrete actions (lane change, accelerate, decelerate) to demonstrate how neural networks handle continuous state spaces.
- **PPO / SAC** use highway-env with continuous actions (steering angle, throttle/brake) to demonstrate real vehicle control — the end goal for autonomous driving PnC engineers.

## Highway-env Configuration

- **Discrete version** (`make_lane_keeping_env`): `DiscreteMetaAction` — 5 meta-actions
- **Continuous version** (`make_continuous_lane_keeping_env`): `ContinuousAction` — steering + acceleration

Both versions share the same observation: Kinematics features (x, y, vx, vy, heading) for 5 surrounding vehicles.

## Usage

### Train

```bash
python algorithms/q_learning/train.py
python algorithms/dqn/train.py
python algorithms/policy_gradient/train.py
python algorithms/ppo/train.py
python algorithms/sac/train.py
```

Training runs with visualization by default. To disable rendering for faster training, set `render_mode=None` in train.py.

Models are saved to each algorithm's `results/` directory.

### Evaluate with Visualization

After training, use `eval.py` to watch the trained agent with greedy policy (epsilon=0) and real-time rendering:

```bash
python algorithms/dqn/eval.py
```

This loads the saved model and runs 10 episodes with the pygame window open, so you can visually confirm the agent has learned to drive.

---

# 算法-环境对应关系

| 算法 | 环境 | 动作空间 | 学习重点 |
|------|------|---------|---------|
| Q-Learning | CliffWalking-v0 | 离散 | TD 更新、Q 表、探索-利用 |
| DQN | highway-env（离散） | DiscreteMetaAction | 神经网络替代 Q 表、经验回放、目标网络 |
| Policy Gradient | highway-env（离散） | DiscreteMetaAction | 策略梯度、用回报加权对数概率 |
| PPO | highway-env（连续） | ContinuousAction | clip 机制、高斯策略连续控制 |
| SAC | highway-env（连续） | ContinuousAction | 最大熵框架、off-policy 连续控制 |

## 算法类别

| 算法 | 类别 | 核心思想 |
|------|------|---------|
| Q-Learning | 基于价值 Value-based | 学习 $Q(s, a)$，再通过 $\arg\max_a Q(s, a)$ 导出策略 |
| DQN | 基于价值 Value-based | 用神经网络逼近 $Q(s, a)$，本质仍是价值学习 |
| Policy Gradient (REINFORCE) | 基于策略 Policy-based | 直接优化策略 $\pi_\theta(a\|s)$，不单独学习 critic |
| PPO | Actor-Critic | 同时学习策略和价值函数，并用 clip 机制稳定策略更新 |
| SAC | Actor-Critic | 在最大熵目标下，同时学习随机策略和双 Q 函数 |

### 快速总结

- **基于价值**：Q-Learning、DQN
- **基于策略**：REINFORCE
- **Actor-Critic**：PPO、SAC
- **补充说明**：PPO 和 SAC 常被归到“偏策略优化”方法里，但更准确的说法是 Actor-Critic，因为它们同时依赖 actor 和 critic 网络。

## 设计思路

- **Q-Learning** 使用 CliffWalking 作为最小化的表格环境，建立 TD 学习的直觉。
- **DQN / PG** 使用 highway-env 离散动作版（换道、加速、减速），展示神经网络如何处理连续状态空间。
- **PPO / SAC** 使用 highway-env 连续动作版（转向角、油门/刹车），展示真实车辆控制——自动驾驶规控工程师的最终目标。

## Highway-env 环境配置

- **离散版** (`make_lane_keeping_env`)：`DiscreteMetaAction` — 5 个元动作
- **连续版** (`make_continuous_lane_keeping_env`)：`ContinuousAction` — 转向 + 加速

两个版本共享相同的观测：5 辆周围车辆的运动学特征（x, y, vx, vy, heading）。

## 使用方式

### 训练

```bash
python algorithms/q_learning/train.py
python algorithms/dqn/train.py
python algorithms/policy_gradient/train.py
python algorithms/ppo/train.py
python algorithms/sac/train.py
```

训练默认开启可视化。如需关闭渲染以加快训练速度，将 train.py 中的 `render_mode=None`。

模型保存在各算法的 `results/` 目录下。

### 可视化评估

训练完成后，使用 `eval.py` 查看训练好的智能体在纯贪心策略（epsilon=0）下的实时渲染效果：

```bash
python algorithms/dqn/eval.py
```

该脚本加载保存的模型，运行 10 个 episode 并打开 pygame 窗口，让你直观确认智能体已经学会驾驶。
