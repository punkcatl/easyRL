# RL+MPC Demo: PPO Decision + CasADi MPC Control

A demonstration module combining PPO decision-making with MPC control for autonomous driving scenarios on highway-env. PPO outputs discrete high-level decisions (accelerate, decelerate, lane change), which are mapped to reference targets and tracked by longitudinal/lateral MPC controllers.

## Architecture

```
highway-env (obs) --> PPO Decision (discrete action)
                          |
                          v
                    Action Mapper --> (v_ref, y_ref)
                          |
              +-----------+-----------+
              |                       |
              v                       v
     Longitudinal MPC          Lateral MPC
     (triple integrator)       (kinematic bicycle)
     output: a_des             output: delta
              |                       |
              +-----------+-----------+
                          |
                          v
                   env.step([delta, a_des])
```

## Requirements

```bash
pip install gymnasium highway-env casadi torch numpy matplotlib tqdm
```

## Usage

### Training

```bash
# Train on default environment (highway-v0), 1000 episodes
python train.py

# Train on a specific scenario
python train.py --env merge-v0 --episodes 500

# Train with rendering enabled
python train.py --env highway-v0 --episodes 200 --render
```

### Evaluation

```bash
# Evaluate trained model with rendering
python eval.py --env highway-v0

# Evaluate without rendering, custom episode count
python eval.py --env highway-v0 --episodes 10 --no-render

# Evaluate with a specific model checkpoint
python eval.py --env highway-v0 --model results/ppo_mpc_highway_v0.pth

# Plot training curve from saved results
python eval.py --plot-training --env highway-v0
```

### Supported Environments

| Environment | Scenario | Complexity |
|---|---|---|
| `highway-v0` | Straight highway, overtaking/following | Low |
| `merge-v0` | On-ramp merging | Medium |
| `roundabout-v0` | Roundabout multi-vehicle | Medium-High |
| `intersection-v0` | T/cross intersection | High |
| `intersection-v1` | Intersection variant | High |
| `racetrack-v0` | Track with turns | Medium |
| `racetrack-large-v0` | Larger track | Medium-High |

## Module Structure

```
rl_mpc/
├── __init__.py
├── config.py                   <- all hyperparameters
├── train.py                    <- training entry point
├── eval.py                     <- evaluation + plotting
├── envs/
│   ├── base_wrapper.py         <- abstract BaseEnvWrapper
│   ├── highway_wrapper.py      <- unified wrapper for 7 scenarios
│   └── carla_wrapper.py        <- CARLA stub (future)
├── controller/
│   ├── lon_mpc.py              <- longitudinal MPC (CasADi, triple integrator)
│   ├── lat_mpc.py              <- lateral MPC (CasADi, kinematic bicycle)
│   └── action_mapper.py        <- discrete action -> (v_ref, y_ref)
├── agent/
│   └── ppo_decision.py         <- discrete PPO agent
├── docs/
│   └── theory.md               <- design rationale
└── results/                    <- saved models and plots
```

## Configuration

All hyperparameters are centralized in `config.py`. Key parameters:

- **PPO**: learning rate, gamma, GAE lambda, clip ratio, epochs
- **Action Mapper**: speed delta, lane width, velocity bounds
- **Longitudinal MPC**: horizon N=20, dt=0.1s, acceleration/jerk limits
- **Lateral MPC**: horizon N=15, dt=0.1s, steering angle/rate limits

---

# RL+MPC 演示：PPO 决策 + CasADi MPC 控制

一个结合 PPO 决策与 MPC 控制的自动驾驶演示模块，运行在 highway-env 上。PPO 输出离散高层决策（加速、减速、换道），映射为参考目标后由纵向/横向 MPC 控制器跟踪执行。

## 架构

```
highway-env (obs) --> PPO 决策层 (离散动作)
                          |
                          v
                    动作映射器 --> (v_ref, y_ref)
                          |
              +-----------+-----------+
              |                       |
              v                       v
       纵向 MPC                 横向 MPC
     (三阶积分器)             (运动学自行车)
     输出: a_des              输出: delta
              |                       |
              +-----------+-----------+
                          |
                          v
                   env.step([delta, a_des])
```

## 依赖安装

```bash
pip install gymnasium highway-env casadi torch numpy matplotlib tqdm
```

## 使用方法

### 训练

```bash
# 在默认环境 (highway-v0) 上训练 1000 个 episode
python train.py

# 在指定场景上训练
python train.py --env merge-v0 --episodes 500

# 开启渲染训练
python train.py --env highway-v0 --episodes 200 --render
```

### 评估

```bash
# 带渲染评估训练好的模型
python eval.py --env highway-v0

# 无渲染评估，指定 episode 数
python eval.py --env highway-v0 --episodes 10 --no-render

# 使用指定模型文件评估
python eval.py --env highway-v0 --model results/ppo_mpc_highway_v0.pth

# 绘制训练曲线
python eval.py --plot-training --env highway-v0
```

### 支持的环境

| 环境 | 场景 | 复杂度 |
|---|---|---|
| `highway-v0` | 高速直道，超车/跟车 | 低 |
| `merge-v0` | 匝道汇入 | 中 |
| `roundabout-v0` | 环岛多车交汇 | 中高 |
| `intersection-v0` | 十字/丁字路口 | 高 |
| `intersection-v1` | 路口变体 | 高 |
| `racetrack-v0` | 带弯道的赛道 | 中 |
| `racetrack-large-v0` | 大赛道 | 中高 |

## 模块结构

```
rl_mpc/
├── __init__.py
├── config.py                   <- 超参数集中管理
├── train.py                    <- 训练入口
├── eval.py                     <- 评估 + 可视化
├── envs/
│   ├── base_wrapper.py         <- 抽象基类 BaseEnvWrapper
│   ├── highway_wrapper.py      <- 7 个场景统一封装
│   └── carla_wrapper.py        <- CARLA 接口预留
├── controller/
│   ├── lon_mpc.py              <- 纵向 MPC（CasADi，三阶积分器）
│   ├── lat_mpc.py              <- 横向 MPC（CasADi，运动学自行车模型）
│   └── action_mapper.py        <- 离散动作 -> (v_ref, y_ref) 映射
├── agent/
│   └── ppo_decision.py         <- 离散 PPO 决策智能体
├── docs/
│   └── theory.md               <- 设计原理文档
└── results/                    <- 保存的模型和图表
```

## 配置

所有超参数集中在 `config.py` 中管理。关键参数：

- **PPO**: 学习率、折扣因子、GAE lambda、clip 比率、训练轮数
- **动作映射器**: 速度变化量、车道宽度、速度边界
- **纵向 MPC**: 预测步长 N=20，dt=0.1s，加速度/jerk 约束
- **横向 MPC**: 预测步长 N=15，dt=0.1s，转向角/转向速率约束
