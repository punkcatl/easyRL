# easyRL

A progressive reinforcement learning project for autonomous driving PnC engineers. Learn RL from scratch by implementing algorithms step-by-step and applying them to lane-keeping tasks in highway-env.

## Learning Path

```
Q-Learning → DQN → Policy Gradient (REINFORCE) → PPO → SAC
```

## Project Structure

```
easyRL/
├── algorithms/
│   ├── q_learning/        # Tabular Q-Learning
│   ├── dqn/               # Deep Q-Network
│   ├── policy_gradient/   # REINFORCE
│   ├── ppo/               # Proximal Policy Optimization
│   └── sac/               # Soft Actor-Critic
├── envs/                  # Highway-env lane-keeping wrappers
├── utils/                 # Plotting, logging, metrics
├── experiments/           # Cross-algorithm comparison experiments
└── tests/                 # Unit tests
```

## Setup

```bash
conda env create -f environment.yml
conda activate easyrl
```

## Usage

### Train individual algorithms

```bash
python algorithms/q_learning/train.py
python algorithms/dqn/train.py --env both
python algorithms/policy_gradient/train.py --env both
python algorithms/ppo/train.py --env both
python algorithms/sac/train.py --env both
```

### Run comparison experiment

```bash
python experiments/run_comparison.py
python experiments/plot_results.py
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Cumulative Reward | Total reward per episode and convergence speed |
| Lateral Deviation | Mean and std of distance from lane center |
| Heading Deviation | Angle difference from lane direction |
| Steering Smoothness | Rate of change of steering angle |

## Recommended Study Plan

### Phase 1: Foundations (1-2 weeks)

1. **Q-Learning** — Run `algorithms/q_learning/train.py`, understand tabular Q-value update intuition
2. **DQN** — Run `algorithms/dqn/train.py`, focus on: why experience replay and target networks stabilize training

For each algorithm, spend 30 minutes on core formulas (Bellman equation, TD error), then read `agent.py` line by line to map formulas to code.

### Phase 2: Policy Gradient (1-2 weeks)

3. **REINFORCE** — Understand the core idea: "weight log-probabilities by returns"
4. **PPO** — The key algorithm. Understand why the clip mechanism prevents overly large policy updates

PPO is the most widely used RL algorithm in autonomous driving. Spend extra time mastering it.

### Phase 3: Continuous Control (1 week)

5. **SAC** — The go-to choice for continuous action spaces. Understand the maximum entropy framework.

### Phase 4: Connect to Autonomous Driving

6. Run `experiments/run_comparison.py` to compare all algorithms on lane-keeping
7. Read research papers (Roach, Think2Drive) — you'll now have the foundation to understand their RL training

### Tips

- Don't start by reading the Sutton & Barto book cover-to-cover — read the relevant chapter for each algorithm
- Each algorithm is "learned" only when you see the reward curve converge
- Spend the most time on PPO — it's the workhorse of industrial autonomous driving RL

---

# easyRL (中文版)

面向自动驾驶规划控制工程师的渐进式强化学习项目。从零开始逐步实现 RL 算法，并将其应用于 highway-env 的车道保持任务。

## 学习路径

```
Q-Learning → DQN → Policy Gradient (REINFORCE) → PPO → SAC
```

## 项目结构

```
easyRL/
├── algorithms/
│   ├── q_learning/        # 表格型 Q-Learning
│   ├── dqn/               # 深度 Q 网络
│   ├── policy_gradient/   # REINFORCE 策略梯度
│   ├── ppo/               # 近端策略优化
│   └── sac/               # 软演员-评论家
├── envs/                  # highway-env 车道保持环境封装
├── utils/                 # 绘图、日志、评估指标
├── experiments/           # 跨算法对比实验
└── tests/                 # 单元测试
```

## 环境配置

```bash
conda env create -f environment.yml
conda activate easyrl
```

## 使用方式

### 单独训练某个算法

```bash
python algorithms/q_learning/train.py
python algorithms/dqn/train.py --env both
python algorithms/policy_gradient/train.py --env both
python algorithms/ppo/train.py --env both
python algorithms/sac/train.py --env both
```

### 运行对比实验

```bash
python experiments/run_comparison.py
python experiments/plot_results.py
```

## 评估指标

| 指标 | 说明 |
|------|------|
| 累积奖励 | 每回合总奖励及收敛速度 |
| 横向偏差 | 与车道中心距离的均值和标准差 |
| 航向偏差 | 与车道方向的角度差 |
| 转向平顺度 | 方向盘角速度变化率 |

## 推荐学习路线

### 第一阶段：打通基础（1-2 周）

1. **Q-Learning** — 跑通 `algorithms/q_learning/train.py`，理解表格型 Q 值更新的直觉
2. **DQN** — 跑通 `algorithms/dqn/train.py`，重点理解：经验回放、目标网络为什么能稳定训练

每个算法花 30 分钟看核心公式（贝尔曼方程、TD 误差），然后对着 `agent.py` 逐行读代码，确认公式和代码的对应关系。

### 第二阶段：策略梯度（1-2 周）

3. **REINFORCE** — 理解"用回报加权对数概率"这一核心思想
4. **PPO** — 重点算法。理解 clip 机制为什么能防止策略更新过大

PPO 是自动驾驶 RL 中最常用的算法，花时间吃透它。

### 第三阶段：连续控制（1 周）

5. **SAC** — 连续动作空间的主流选择，理解最大熵框架

### 第四阶段：对接自动驾驶

6. 跑 `experiments/run_comparison.py` 对比所有算法在车道保持上的表现
7. 回头看论文（Roach、Think2Drive），这时你已经有能力读懂它们的 RL 训练部分

### 建议

- 不要先啃 Sutton 的书，太慢。每个算法只看对应章节即可
- 每个算法必须跑出收敛曲线，看到奖励上升才算学会
- PPO 花最多时间，它是工业界自动驾驶 RL 的主力
