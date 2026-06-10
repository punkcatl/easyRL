# 宇树科技 RL 运控技术方案

宇树（Unitree）四足/人形机器人的强化学习运动控制完整技术路线分析。

---

## 核心技术栈

```
Isaac Gym/Lab (仿真) + RSL_rl (PPO训练) + Domain Randomization + Sim2Sim(MuJoCo) → Sim2Real
```

宇树直接采用 ETH Zurich RSL 的开源生态（`legged_gym` + `rsl_rl`），非完全自研。

## 训练管线（四阶段）

```
Train (Isaac Gym/Lab, 数千并行环境)
  → Play (仿真可视化验证)
  → Sim2Sim (MuJoCo 跨引擎验证)
  → Sim2Real (实机部署)
```

| 参数 | 值 |
|------|---|
| 物理仿真频率 | 200 Hz (dt=0.005s) |
| Policy 控制频率 | 50 Hz (decimation=4) |
| 并行环境数 | 4096+ (单 GPU) |
| 典型训练时间 | 数分钟至数十分钟 |
| 训练迭代 | ~10,000 iterations |

---

## Sim-to-Real 迁移

### Domain Randomization 具体参数

| 参数 | 随机化范围 |
|------|-----------|
| 地面摩擦系数 | [0.1, 1.25] |
| 机体附加质量 | [-1.0, +3.0] kg |
| 外部推力扰动 | 每5秒一次，最大1.5 m/s |
| 关节位置噪声 | 0.01 rad |
| 关节速度噪声 | 1.5 rad/s |
| 线速度噪声 | 0.1 m/s |
| 角速度噪声 | 0.2 rad/s |
| 重力方向噪声 | 0.05 |

### Asymmetric Actor-Critic（Privileged Learning）

- **Critic** 使用 privileged observations（含摩擦系数、真实质量等仿真才有的信息）
- **Actor** 只用可部署的 observations（IMU + 关节编码器）
- G1 人形：47 维 obs (actor) vs 50 维 privileged obs (critic)
- H1 人形：41 维 obs (actor) vs 44 维 privileged obs (critic)
- 额外 3 维 privileged info 推测包含：地面摩擦系数、真实质量参数

### Terrain Curriculum

渐进式地形难度训练：
- Smooth slope (10%), Rough slope (10%), Stairs up (35%), Stairs down (25%), Discrete terrain (20%)
- 10 个难度等级 × 20 种地形类型的网格系统

---

## Observation / Action Space 设计

### Go2 四足机器人 (12-DOF)

**Observation (48维)：**
- 线速度 (3D) + 角速度 (3D) + 重力方向 (3D)
- 关节位置 (12D)：4 腿 × 3 关节 (hip, thigh, calf)
- 关节速度 (12D)
- 上一步动作 (12D)
- 速度指令 (2-3D)
- 可选：height measurements (17×11 grid, 1.6m × 1.0m)

**Action (12维)：**
- 12 个关节位置目标 (PD position control)
- `target_angle = 0.25 * action + default_angle`
- Kp=20, Kd=0.5

### G1 人形机器人 (12-DOF locomotion)

**Observation (47维)：**
- base states + 12 joint pos/vel + actions + commands

**Action (12维)：**
- 左右腿各 6 关节：hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll
- Kp：hip=100, knee=150, ankle=40
- Kd：hip=2, knee=4, ankle=2
- Action scale: 0.25

### H1 人形机器人 (10-DOF locomotion)

**Observation (41维)：**
- 线速度(3) + 角速度(3) + 重力(3) + 关节位置(10) + 关节速度(10) + 动作(10) + 指令(2)

**Action (10维)：**
- 左右腿各 5 关节：hip_yaw, hip_roll, hip_pitch, knee, ankle
- Kp：hip=150, knee=200, ankle=40
- 手臂关节 (8个) 在默认配置中不参与 locomotion 控制

---

## Policy 网络架构

| 参数 | 值 |
|------|---|
| 网络类型 | ActorCriticRecurrent (LSTM) |
| Actor hidden | [32] |
| Critic hidden | [32] |
| 激活函数 | ELU |
| LSTM hidden size | 64 |
| LSTM layers | 1 |
| 初始噪声 std | 0.8 |

网络极为轻量——32 维隐层 + 64 维 LSTM，可在 CPU 上实时推理。

---

## 硬件部署

### 机载计算平台

| 机器人 | 标准计算 | 可选高算力 |
|--------|---------|-----------|
| Go2 | 8核高性能 CPU | NVIDIA Jetson Orin (40-100 TOPS) |
| G1 | 8核高性能 CPU | Orin 等多品牌可选 |
| H1 | Intel Core i5/i7 | Jetson Orin NX |

### 推理部署

| 项目 | 方案 |
|------|------|
| 推理框架 | LibTorch (C++) 或 PyTorch (Python) |
| 模型格式 | `.pt` (PyTorch JIT traced) |
| 控制频率 | 50 Hz (20ms 周期) |
| 推理延迟 | << 1ms（网络极小，CPU 足够） |
| 通信接口 | unitree_sdk2 (DDS 协议, Ethernet) |
| 部署语言 | Python (`deploy_real.py`) / C++ (`deploy_real/cpp_g1`) |

瓶颈在 50Hz 控制环路本身（20ms），不在神经网络推理。

---

## 开源仓库

| 仓库 | Stars | 说明 |
|------|-------|------|
| `unitree_rl_gym` | 3,300+ | Isaac Gym + rsl_rl，支持 Go2/G1/H1/H1_2 |
| `unitree_rl_lab` | 1,100+ | 新一代 IsaacLab 2.3.0 版本 |
| `unitree_mujoco` | 1,000+ | MuJoCo 仿真 + sim2sim 验证 |
| `unitree_rl_mjlab` | — | MuJoCo 版 RL 训练 |

---

## 与其他足式机器人公司对比

| 维度 | Unitree (宇树) | ANYbotics (ETH spin-off) | Agility Robotics |
|------|---------------|--------------------------|------------------|
| RL 框架 | RSL-RL（直接用 ETH 开源） | legged_gym（原始开发者） | 闭源 |
| 训练平台 | Isaac Gym → IsaacLab 迁移中 | Isaac Gym | 未公开 |
| 算法 | PPO + LSTM | PPO + Teacher-Student | 未公开 |
| Sim-to-Real | DR + Sim2Sim + Privileged Learning | DR + Privileged Learning（开创者） | DR + Residual Policy |
| 开源程度 | 高（训练+部署完整） | 中（框架开源，产品闭源） | 低 |
| 硬件成本 | $1,600 起 | $100,000+ | 未公开 |
| 最高速度 | Go2: 5m/s, H1: 3.3m/s | ANYmal: 0.75m/s | Digit: ~1.5m/s |
| 研究生态 | 巨大社区，2025-2026 超 40 篇论文使用 | ETH 学术核心 | 较封闭 |

---

## 关键论文

**基础框架：**
- "Learning to Walk in Minutes Using Massively Parallel Deep RL" (Rudin et al., CoRL 2021) — legged_gym/rsl_rl 基础论文

**社区在 Unitree 平台上的代表性工作 (2025-2026)：**
- "Chasing Autonomy" (2026) — G1 达到 3.3 m/s 奔跑
- "Perceptive Humanoid Parkour" (2026) — G1 攀爬 1.25m 障碍物
- "APEX" (2026) — LiDAR-based zero-shot traversal
- "Learning Sim-to-Real Humanoid Locomotion in 15 Minutes" (2025) — 单 GPU 15 分钟训练
- "Robot Parkour Learning" (Ziwen Zhuang) — Teacher-Student + DAgger

---

## 总结

宇树的 locomotion 技术本质是基于 ETH RSL 范式（legged_gym + PPO + Privileged Learning），其核心贡献在于：

1. **工程化适配** — 将框架适配到全系列机器人（Go2/G1/H1）
2. **完整管线开源** — 从训练到 sim2sim 到实机部署的全流程
3. **低成本硬件** — $1,600 起的价格使全球数百个实验室可以复现
4. **生态建设** — 开源策略吸引了大量研究者，反哺技术迭代
