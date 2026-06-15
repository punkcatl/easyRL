# Go2 Locomotion 项目设计文档

本项目复现宇树（Unitree）Go2 四足机器人强化学习运动控制的完整工业路线，
从仿真训练到 ONNX 推理部署的全流程。

---

## Current Status (Round 15, 2026-06-15)

| 指标 | 值 |
|------|---|
| 阶段 | Phase 1 Teacher 训练完成 |
| 训练轮次 | Round 15 (5000 iter) |
| 最终 reward | ~4.9 |
| 评估速度 | avg_vx ≈ 1.0 m/s (cmd=1.0) |
| tracking ratio | ~0.98 |
| 存活率 (pct30) | 98-100% |
| 命令范围 | vx [0.30, 1.50] 已完全扩展 |
| DR | full (friction/mass/ext_force/motor) |
| 最优 checkpoint | `results/teacher_round15.pth` |

**迭代历程简述：**
- R1-R3：解决"站着不动"局部最优（reward 漏洞修复）
- R4-R5：正向 reward 主导 + curriculum 基于实际速度
- R6-R8：修复 feet_air_time exploit，加入 trot gait schedule
- R9-R14：gait shaping、forward_progress、稳定性调优
- R15：action_scale 0.35 重训，实现 1.0 m/s 稳定行走

**关键架构：**
- Teacher-Student 框架：先训 teacher（有 privileged info），再蒸馏到 student
- PPO + Curriculum：命令范围渐进扩展 + DR 分阶段引入
- Obs 48D / Action 12D：标准四足控制
- 128 并行环境 / async vectorized

**Checkpoint：**
- 最新模型：`results/teacher_round15.pth`
- 中间 checkpoint：`results/teacher_iter500.pth` ~ `teacher_iter5000.pth`（每 500 iter）
- 历轮最优：`results/teacher_round1.pth` ~ `teacher_round15.pth`
- Reward 曲线：`results/teacher_rewards_round{1..15}.npy`
- 训练日志：`results/train_log_round{2..15}.txt`

**下一步：** Phase 2 Student 蒸馏（RMA + Behavior Cloning）

---

## 一、完整 Pipeline 流程

### 1.1 总览

三个阶段顺序执行，后阶段依赖前阶段产出的模型文件：

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│    Phase 1       │    │    Phase 2       │    │    Phase 3       │
│  Teacher Train   │───>│ Student Distill  │───>│   Deployment     │
│    PPO + DR      │    │    RMA + BC      │    │  ONNX Export     │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
  teacher_final.pth      student_final.pth       student_go2.onnx
```

### 1.2 Phase 1：Teacher 训练（PPO + Domain Randomization）

每次 episode reset 触发一次 DR，随机化物理参数并生成 privileged\_obs 给 Critic。每隔 5~10 秒施加一次随机水平外力推扰（模拟真实扰动）。

```
┌──────────────────────────────────────────────────────────────┐
│          DRVecGo2Env  (32 envs, each with own DR)            │
│                                                              │
│   episode reset                                              │
│   ├── randomize friction      [0.5, 1.25]                    │
│   ├── randomize body mass     scale [0.8, 1.2]               │
│   └── randomize motor gains   kp/kd scale [0.8, 1.2]         │
│                                                              │
│   every 5~10s: random push    magnitude [0, 3] N             │
└──────────────────────────┬───────────────────────────────────┘
                     │ obs(48D)          │ obs(48D) + privileged(7D)
                     │ (to Actor)        │ (to Critic only)
                     ▼                  ▼
          ┌────────────────────────────────────────┐
          │        AsymmetricActorCritic           │
          │                                        │
          │  obs(48D) ──────> Actor MLP            │
          │                        │               │
          │                        ▼               │
          │                   action(12D)          │
          │                                        │
          │  obs(48D)+privileged(7D) ─> Critic MLP │
          │                                │       │
          │                                ▼       │
          │                           value(1D)    │
          └───────────────────┬────────────────────┘
                              │ rollout buffer
                              │ (n_steps x num_envs steps)
                              ▼
          ┌────────────────────────────────────────┐
          │              PPO Update                │
          │                                        │
          │  per-env GAE  (gamma=0.99, lambda=0.95)│
          │  normalize advantages                  │
          │  mini-batch shuffle + clipped loss     │
          └───────────────────┬────────────────────┘
                              │
                              ▼
                    teacher_final.pth
```

### 1.3 Phase 2：Student 蒸馏（RMA Behavior Cloning）

**RMA 核心思想：** Teacher 训练时依赖 privileged\_obs（摩擦、质量、外力等仿真才有的参数）。Student 通过观察自身过去 50 步的行为历史，隐式推断出同样的环境参数，从而在真机部署时不需要任何特权信息。

训练阶段 vs 部署阶段的输入替换：

```
┌──────────────────────┐               ┌──────────────────────┐
│    Training          │               │    Deployment        │
│                      │               │                      │
│  privileged_obs (7D) │  replaced by  │  obs_history (2400D) │
│  - friction coeff    │  ──────────>  │  last 50 steps obs   │
│  - mass scale        │               │  each step = 48D     │
│  - ext force         │               │                      │
│  - motor strength    │               │  AdaptationModule    │
└──────────────────────┘               │  --> latent z (16D)  │
                                       └──────────────────────┘
```

**Step 1：数据采集（500k 步）**

用冻结的 Teacher 在带 DR 的环境中确定性推理（取 mean 不采样），记录每一步的三元组：

```
┌────────────────────────────────────────────┐
│  Teacher (frozen, deterministic)           │
│                                            │
│  each step records:                        │
│    obs_history  = last 50 obs flattened    │
│                   shape (2400,)            │
│    obs_current  = current obs              │
│                   shape (48,)              │
│    teacher_act  = Actor mean output        │
│                   shape (12,)              │
└────────────────────────────────────────────┘
```

**Step 2：BC 训练（监督学习，最小化 MSE）**

```
  obs_history (2400D)
         │
         ▼
┌─────────────────────────────────┐
│  AdaptationModule               │
│  MLP: 2400 -> 256 -> 128 -> 16  │
└──────────────────┬──────────────┘
                   │ latent z (16D)
                   │   (implicit env params)
  obs_current (48D)│
         │         │
         └────┬────┘
              ▼
┌─────────────────────────────────────┐
│  StudentPolicy                      │
│  MLP: (48+16) -> 128 -> 128 -> 12   │
│  input = concat(obs_current, z)     │
└──────────────────┬──────────────────┘
                   │ action_pred (12D)
                   ▼
         MSE loss vs teacher_act
                   │
                   ▼
            backward + Adam
```

### 1.4 Phase 3：Sim2Sim 验证 + ONNX 部署

**Sim2Sim 验证：** 在无 DR 的干净环境中运行 Student，验证从 privileged\_obs 迁移到 obs\_history 后策略不退化。

```
┌─────────────────────────────────────────┐
│  Sim2Sim Evaluation  (no DR)            │
│                                         │
│  obs_history = rolling buffer (50, 48)  │
│                                         │
│  loop per step:                         │
│    obs_history = roll(obs_history)      │
│    obs_history[-1] = obs                │
│    action = student.get_action(         │
│               obs_history.flatten(),    │
│               obs)                      │
│    obs, reward, done = env.step(action) │
│                                         │
│  report: avg_reward / survival% / steps │
└─────────────────────────────────────────┘
```

**ONNX 导出：** 将 AdaptationModule + StudentPolicy 融合为单一计算图导出。推理延迟需满足 50Hz 控制频率（< 20ms）预算。

```
┌───────────────────────────────────────┐
│  StudentONNXWrapper                   │
│  = AdaptationModule + StudentPolicy   │
│                                       │
│  inputs:  obs_history (2400D)         │
│           obs_current  (48D)          │
│  output:  action       (12D)          │
└──────────────────┬────────────────────┘
                   │ torch.onnx.export (opset 17)
                   ▼
          student_go2.onnx
                   │
                   ├── accuracy: max|PyTorch - ONNX| < 1e-4
                   └── latency:  avg < 20ms  (50Hz budget)
                   │
                   ▼
          ONNXRuntime inference
          Jetson Orin / any CPU
```

### 1.5 观测与动作空间

**Observation (48D)：**

| field        | dim | description                         |
|--------------|-----|-------------------------------------|
| base_lin_vel |  3  | linear velocity in body frame       |
| base_ang_vel |  3  | angular velocity in body frame      |
| proj_gravity |  3  | projected gravity (detects tilt)    |
| joint_pos    | 12  | joint angles relative to default    |
| joint_vel    | 12  | joint angular velocities            |
| last_action  | 12  | previous action (temporal context)  |
| command      |  3  | target [vx, vy, yaw_rate]           |

**Action (12D) — PD position control：**

```
target_angle[i] = 0.25 * action[i] + default_angle[i]
torque[i]       = Kp * (target - current_pos) - Kd * current_vel
Kp = 20,  Kd = 0.5  (both scalable by DR)
```

**Privileged Obs (7D) — Critic only：**

| field      | dim | description                        |
|------------|-----|------------------------------------|
| friction   |  1  | ground friction coefficient scale  |
| mass_scale |  1  | body mass scale factor             |
| ext_force  |  3  | current external push force vector |
| motor_str  |  2  | [kp_scale, kd_scale]               |

---

## 二、与宇树实际方案的对比

### 做了替代的环节

| 环节 | 宇树实际方案 | 我们的替代 | 替代原因 |
|------|-------------|-----------|----------|
| 仿真引擎 | Isaac Gym/Lab（GPU 并行，4096+ envs） | MuJoCo + 同步向量化（32 envs CPU） | Isaac Gym 需要 NVIDIA GPU，安装门槛高 |
| 并行规模 | 单卡数千环境，分钟级训练 | 32 环境，小时级训练 | CPU 并行上限，但算法逻辑完全一致 |
| RL 框架 | rsl_rl（RSL 维护的封装库） | 自实现 PPO（项目已有基础） | 面试展示需要理解底层，非调库 |
| 网络架构 | ActorCriticRecurrent（LSTM hidden=64） | MLP（先跑通），LSTM 作为可选扩展 | 降低首次调通复杂度 |
| 地形 Curriculum | 10 级难度 × 20 种地形网格 | 先用平地训练，地形作为后续扩展 | MuJoCo 地形生成比 Isaac 更手动 |
| 部署格式 | `.pt` JIT traced → 板端 LibTorch | ONNX → ONNXRuntime benchmark | 无实机，ONNX 更通用且已有现成模块 |
| 实机部署 | unitree_sdk2 DDS 通信 → Go2 本体 | 纯仿真 Sim2Sim 验证 | 无硬件 |

### 完全一致、未做替代的核心环节

- PPO + GAE + clip 算法逻辑
- Asymmetric Actor-Critic（actor 只看 obs，critic 看 obs+privileged）
- Domain Randomization（摩擦、质量、外力、电机增益）
- PD 位置控制（`target = scale × action + default_angle`）
- 48D 观测空间 + 12D 动作空间设计
- Teacher-Student RMA 蒸馏（history→latent→action）
- Velocity tracking reward 设计

### 总结

算法和架构层面零替代，替代都发生在工程规模和硬件层面。
面试时可以说：**"核心算法和宇树完全一致，只是仿真规模从 GPU 数千并行降到 CPU 数十并行，训练时间从分钟级变为小时级。"**

---

## 三、Reward 设计

Phase 1 训练靠以下分项 reward 驱动，各项权重在 `config.py` 中配置：

| reward 项 | 权重 | 说明 |
|-----------|------|------|
| lin_vel_tracking | +1.0 | 线速度跟踪，exp(-error²/sigma) |
| ang_vel_tracking | +0.5 | 偏航角速度跟踪 |
| feet_air_time | +1.0 | 鼓励抬脚，避免拖地 |
| lin_vel_z_penalty | -2.0 | 抑制竖直方向弹跳 |
| ang_vel_xy_penalty | -0.05 | 抑制侧翻/俯仰抖动 |
| torque_penalty | -2e-4 | 抑制过大关节力矩 |
| action_rate_penalty | -0.01 | 抑制动作突变（平滑性） |
| joint_acc_penalty | -2.5e-7 | 抑制关节加速度过大 |
| collision_penalty | -1.0 | 非脚底接触地面即惩罚 |

速度跟踪使用指数核：`r = exp(-||v_cmd - v_actual||² / σ)`，sigma=0.25，使 reward 对误差大小平滑响应而非二值化。

---

## 四、代码结构与快速运行

```
applications/go2_locomotion/
├── config.py                    # 全局超参数
├── train_teacher.py             # Phase 1: PPO + DR 训练 Teacher
│                                #   内含 DRVecGo2Env（带 DR 的向量化环境）
├── train_student.py             # Phase 2: RMA BC 蒸馏 Student
├── evaluate.py                  # Sim2Sim 评估（teacher/student）
├── export_onnx.py               # ONNX 导出 + 推理 benchmark
├── assets/
│   └── go2_scene.xml            # Go2 MuJoCo MJCF 模型
├── envs/
│   ├── go2_env.py               # 单环境（PD control + 48D obs）
│   ├── go2_reward.py            # 分项奖励计算
│   └── vectorized.py            # VecGo2Env（无 DR，纯向量化）
├── agent/
│   ├── networks.py              # ActorCritic / AsymmetricActorCritic
│   ├── ppo.py                   # PPO 训练器（GAE + clip）
│   └── teacher_student.py       # AdaptationModule + StudentPolicy
└── dr/
    └── domain_randomization.py  # Go2DomainRandomizer（privileged info 输出）
```

**快速运行：**

```bash
# Phase 1: 训练 Teacher（默认 3000 iterations，数小时）
python applications/go2_locomotion/train_teacher.py

# Phase 2: 蒸馏 Student（需要 teacher_final.pth）
python applications/go2_locomotion/train_student.py

# 评估 Teacher（Sim2Sim，无 DR 环境）
python applications/go2_locomotion/evaluate.py --mode teacher --episodes 20

# 评估 Student
python applications/go2_locomotion/evaluate.py --mode student --episodes 20

# 导出 ONNX + 推理 benchmark
python applications/go2_locomotion/export_onnx.py
```

**预期训练时间（MuJoCo CPU，32 envs）：**

| 阶段 | 时间估计 |
|------|---------|
| Phase 1 Teacher（3000 iter） | 4~8 小时（取决于 CPU） |
| Phase 2 数据采集（500k 步） | 30~60 分钟 |
| Phase 2 BC 训练（100 epoch） | 5~10 分钟 |
| ONNX 导出 + benchmark | < 1 分钟 |
