# easyRL Design Spec

## Project Goal

A reinforcement learning study project for autonomous driving control engineers. Progresses from foundational RL algorithms to a lane-keeping application, culminating in a multi-algorithm comparison experiment evaluating control quality.

## Project Structure

```
easyRL/
├── algorithms/
│   ├── q_learning/        # Tabular method, RL fundamentals
│   ├── dqn/               # Function approximation with neural networks
│   ├── policy_gradient/   # REINFORCE, policy gradient introduction
│   ├── ppo/               # Stable training via clipping
│   └── sac/               # Continuous control, maximum entropy
├── envs/                  # highway-env wrappers and configurations
├── utils/                 # Shared utilities (plotting, logging, metrics)
├── experiments/           # Lane-keeping comparison experiments
└── requirements.txt
```

Each `algorithms/xxx/` directory contains:
- `agent.py` — algorithm implementation
- `train.py` — training script
- `config.py` — hyperparameter configuration
- `results/` — training artifacts (model weights, curve data)

## Learning Path

```
Q-Learning → DQN → Policy Gradient → PPO → SAC
```

Each algorithm is first validated on a simple environment (CartPole or CliffWalking), then migrated to highway-env for lane keeping (except Q-Learning, which remains tabular-only on simple environments).

## Final Comparison Experiment

**Scenario:** highway-env lane keeping, progressing from single-lane to multi-lane with surrounding traffic.

**Algorithms compared:** DQN vs PPO vs SAC

**Evaluation metrics:**
- Cumulative reward and convergence episode count
- Lateral deviation (mean + standard deviation)
- Heading angle deviation
- Steering smoothness (steering angle rate of change)

**Output:** matplotlib comparison charts with brief conclusions.

## Visualization

- Default: matplotlib scripts for post-training chart generation
- Optional: TensorBoard for real-time monitoring during training (`torch.utils.tensorboard.SummaryWriter`)

## Tech Stack

| Component | Choice |
|-----------|--------|
| Framework | PyTorch |
| Environment | highway-env (Gymnasium) |
| Visualization | matplotlib (default) + TensorBoard (optional) |
| Environment management | conda |
| Hardware | Intel Xeon W-2235, 128GB RAM, NVIDIA RTX A4000 16GB, CUDA 13.0 |

## Scope Boundaries

- This project is for learning, not production use
- No custom environment development — use highway-env as-is
- No hyperparameter search frameworks — manual tuning is sufficient
- No distributed training — single GPU is more than adequate

---

# easyRL 设计规格（中文版）

## 项目目标

面向自动驾驶控制工程师的强化学习学习项目。从基础 RL 算法逐步推进到车道保持应用，最终进行多算法对比实验评估控制质量。

## 项目结构

```
easyRL/
├── algorithms/
│   ├── q_learning/        # 表格方法，RL 基础
│   ├── dqn/               # 神经网络函数逼近
│   ├── policy_gradient/   # REINFORCE，策略梯度入门
│   ├── ppo/               # 裁剪机制实现稳定训练
│   └── sac/               # 连续控制，最大熵
├── envs/                  # highway-env 封装和配置
├── utils/                 # 共享工具（绘图、日志、指标）
├── experiments/           # 车道保持对比实验
└── requirements.txt
```

每个 `algorithms/xxx/` 目录包含：
- `agent.py` — 算法实现
- `train.py` — 训练脚本
- `config.py` — 超参数配置
- `results/` — 训练产物（模型权重、曲线数据）

## 学习路径

```
Q-Learning → DQN → Policy Gradient → PPO → SAC
```

每个算法先在简单环境（CartPole 或 CliffWalking）上验证，然后迁移到 highway-env 进行车道保持（Q-Learning 除外，仅在简单环境上做表格方法）。

## 最终对比实验

**场景：** highway-env 车道保持，从单车道逐步升级到多车道含周围交通。

**对比算法：** DQN vs PPO vs SAC

**评估指标：**
- 累积奖励与收敛所需回合数
- 横向偏差（均值 + 标准差）
- 航向角偏差
- 转向平滑度（转向角变化率）

**输出：** matplotlib 对比图表 + 简要结论。

## 可视化

- 默认：matplotlib 脚本用于训练后生成图表
- 可选：TensorBoard 用于训练中实时监控（`torch.utils.tensorboard.SummaryWriter`）

## 技术栈

| 组件 | 选择 |
|------|------|
| 框架 | PyTorch |
| 环境 | highway-env (Gymnasium) |
| 可视化 | matplotlib（默认）+ TensorBoard（可选） |
| 环境管理 | conda |
| 硬件 | Intel Xeon W-2235, 128GB RAM, NVIDIA RTX A4000 16GB, CUDA 13.0 |

## 范围边界

- 本项目用于学习，非生产用途
- 不做自定义环境开发——直接使用 highway-env
- 不使用超参数搜索框架——手动调参足够
- 不做分布式训练——单 GPU 绰绰有余
