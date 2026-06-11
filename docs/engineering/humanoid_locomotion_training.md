# Humanoid Robot Locomotion Training

A systematic overview of how bipedal humanoid robots (e.g. Unitree H1/G1, Figure 01, Agility Digit) are trained with reinforcement learning, covering the unique challenges versus quadrupeds, task hierarchy, training paradigms, and industry examples.

## Why Humanoids Are Harder Than Quadrupeds

Quadrupeds have 4 contact points and a low center of mass — they are passively stable in many configurations. Humanoids have only 2 feet, a high center of mass, and must actively balance at all times. This fundamental difference propagates through every aspect of training:

```
Quadruped                         Humanoid
-----------                       ---------
4 contact points                  2 contact points
CoM height ~ 0.4m                 CoM height ~ 0.7-1.0m (model dependent)
Passive stability in stance       Actively unstable always
12 DOF (3 per leg)                29+ DOF (legs + arms + torso)
Fall recovery: easy               Fall recovery: hard
Sim-to-real gap: moderate         Sim-to-real gap: large
```

The higher instability means:
- Reward design is more constrained (agent can exploit instability)
- Termination conditions fire more often early in training
- Reference motion or trajectory tracking is often needed to bootstrap learning

## Task Hierarchy

```
Level 1   Standing balance / in-place recovery         <- active balance
Level 2   Flat-ground walking / velocity tracking      <- basic locomotion
Level 3   Stairs / slopes / rough terrain              <- terrain adaptation
Level 4   Running / jumping / dynamic gaits            <- high-speed dynamics
Level 5   Whole-body manipulation + locomotion         <- loco-manipulation
Level 6   Parkour / extreme maneuvers                  <- frontier research
```

Most deployed humanoids today operate at Level 2-3. Level 4+ is active research territory.

## Core Challenge: The Instability Problem

The single biggest difference from quadrupeds is that a humanoid will fall within seconds if the policy is not carefully initialized. This affects training in two ways:

**1. Early training is extremely sample-inefficient**

A random policy falls immediately. Episode length starts at near-zero, reward signal is almost entirely the fall penalty, and the agent learns nothing useful.

**Solutions:**
- Reference motion tracking (imitation from MoCap data)
- Curriculum starting from easier initial conditions (e.g. seated, crouching)
- Large reward for simply not falling (alive bonus)
- PD controller as residual base (policy outputs delta on top of a stable PD gait)

**2. Reward shaping is trickier**

In quadrupeds, a high velocity-tracking reward is enough. In humanoids, an unconstrained agent quickly finds degenerate gaits: dragging feet, leaning at extreme angles, shuffling rather than walking. Additional regularization terms are necessary:

```
reward = lin_vel_tracking
       + ang_vel_tracking
       - foot_contact_force_penalty    <- no stomping
       - body_orientation_penalty      <- stay upright
       - joint_limit_penalty           <- don't over-flex
       - action_smoothness_penalty     <- no jitter
       - arm_deviation_penalty         <- arms near neutral
       + feet_air_time_reward          <- encourage lifting feet
```

## Training Paradigms

### Paradigm A: Reference Motion Tracking (most common for humanoids)

Use motion capture data or handcrafted reference trajectories as supervision. The policy is trained to track a reference gait while also responding to perturbations:

```
MoCap data / handcrafted gait
           |
           v
  reference joint angles q_ref(t)
           |
           v
┌──────────────────────────────────┐
│   Tracking + Task Reward         │
│                                  │
│  r = w1 * ||q - q_ref||^2        │  <- track reference pose
│    + w2 * vel_tracking           │  <- follow velocity command
│    + w3 * end_effector_match     │  <- foot placement accuracy
│    - w4 * fall_penalty           │  <- stay upright
└──────────────────────────────────┘
           |
           v
  policy learns to walk naturally
  and recover from pushes
```

Representative: DeepMimic (Peng et al., 2018), AMP (Peng et al., 2021).

Note: DeepMimic and AMP are related but distinct. **DeepMimic** uses a direct pose-tracking reward (explicit MSE against reference). **AMP** uses an adversarial discriminator to reward motion that is *stylistically similar* to reference clips — it does not require precise frame-by-frame tracking, making it more flexible and robust to reference quality.

- **Pros:** natural-looking motion, fast convergence, avoids degenerate gaits
- **Cons:** quality of reference motion limits policy quality; collecting diverse MoCap is expensive

### Paradigm B: Pure RL without Reference (RSL / Unitree approach for H1/G1)

Same framework as quadruped training — Terrain Curriculum + Domain Randomization + Asymmetric Actor-Critic — but with heavier reward regularization and careful initialization.

Key differences from quadruped training:

```
Quadruped PPO config          Humanoid PPO config
--------------------          -------------------
hidden_dim = 128              hidden_dim = 256-512
alive_bonus = 0               alive_bonus = large
action_scale = 0.25           action_scale = 0.1-0.2 (more conservative)
kp = 20, kd = 0.5             kp = 100-200 per joint (stiff control)
no arm joints                 arm joints fixed or lightly controlled
```

The "29+ DOF" figure refers to the robot's total hardware DOF. In locomotion-only training, arm joints are typically **fixed at a neutral pose** — the effective controlled DOF drops to 10-12 (legs only), comparable to a quadruped. Full 29-DOF whole-body control is reserved for loco-manipulation tasks (Paradigm D).

The H1 policy from `unitree_rl_gym` uses this approach: pure PPO with carefully tuned reward weights, starting from a crouched pose to reduce early falls.

- **Pros:** no MoCap needed, generalizes to novel terrain
- **Cons:** requires extensive reward tuning; motion quality lower than reference-based

### Paradigm C: Residual RL on top of Model-Based Controller

Use a classical model-based controller (MPC or ZMP-based gait) as the base, and train RL to output residual corrections. **ZMP (Zero Moment Point)** is the ground point where the net ground reaction force acts — a gait is statically stable if the ZMP stays within the support polygon. ZMP-based controllers generate stable nominal gaits without learning, but cannot handle terrain variability well.

```
┌────────────────────────────┐
│  Model-Based Base Gait     │
│  (MPC / ZMP / SLIP model)  │
│  output: nominal action    │
└────────────┬───────────────┘
             │ nominal_action
             v
┌────────────────────────────┐
│  Residual RL Policy        │
│  input:  obs + nominal_act │
│  output: delta_action      │
└────────────┬───────────────┘
             │ nominal + delta
             v
          motors
```

The RL policy only needs to learn small corrections, not the full locomotion behavior. This dramatically reduces the exploration problem.

- **Pros:** fast training, safe exploration, robust base behavior
- **Cons:** motion envelope is bounded by what the base controller can generate; the RL policy learns corrections, not fundamentally new behaviors. Whether the baseline is a full MPC optimizer or just a pre-planned reference trajectory affects how much the RL residual can deviate.

### Paradigm D: Whole-Body Control + Loco-Manipulation

Simultaneous control of locomotion and manipulation. The humanoid must walk while using arms to interact with the environment:

```
┌──────────────────────────────────────────┐
│         Whole-Body Controller            │
│                                          │
│  locomotion task:  velocity tracking     │
│  manipulation task: reach / grasp / push │
│                                          │
│  unified action space: 29+ joint targets │
└──────────────────────────────────────────┘
```

This is where the humanoid form factor pays off — arms can be used for balance (like humans) as well as manipulation. Training requires multi-task reward design:

```
r_total = r_locomotion + r_manipulation + r_coordination
```

Representative: Figure 01 (manipulation + walking), 1X NEO.

### Paradigm E: Imitation Learning + RL Fine-tuning

Pre-train with Behavior Cloning on human video / teleoperation data, then fine-tune with RL for robustness and closed-loop performance:

```
Step 1: Imitation Pre-training
  human motion data (video / MoCap / teleoperation)
           |
           v
  Behavior Cloning (supervised)
           |
           v
  initial policy (can walk, but not robust)

Step 2: RL Fine-tuning
  initial policy
           |
           v
  PPO / SAC with task reward + safety constraints
           |
           v
  robust policy (handles perturbations, novel terrain)
```

This mirrors how LLMs are trained: pre-train on data, then RLHF. Representative: Tesla Optimus, recent Figure approaches.

- **Pros:** natural motion from human data, fast to initial working policy
- **Cons:** human and robot kinematics differ fundamentally — joint ranges, link lengths, and actuator limits do not match, so BC on raw human video often produces infeasible joint targets. More critically, human motion dynamics (muscle-driven, compliant) are far from robot dynamics (electric motors, rigid), causing large sim-to-real gaps even before deployment. Fine-tuning can also overwrite fine-grained behaviors learned during pre-training.

## Sim-to-Real: Harder for Humanoids

The sim-to-real gap is larger for humanoids because:

1. **More joints = more actuator modeling error** — 29 joints vs 12
2. **Higher CoM = more sensitive to model errors** — small inertia error causes large torque mismatch
3. **Contact dynamics at feet are critical** — foot slip causes falls; harder to model accurately
4. **Flexible body effects** — torso flex, cable stretch, joint backlash matter more

Key techniques beyond standard DR:

```
Standard DR (works for quadrupeds):
  randomize: friction, mass, motor gains, external push

Additional for humanoids:
  randomize: foot sole geometry
  randomize: joint backlash / motor deadzone
  add:       latency / communication delay simulation
  add:       actuator torque saturation model
  use:       more conservative action clipping
  use:       contact-aware state estimation (not just IMU)
```

## Industry Examples (2024-2026)

| Robot | Company | Approach | Notable Achievement |
|-------|---------|----------|---------------------|
| H1 | Unitree | Pure RL (RSL framework) | 3.3 m/s running, world record |
| G1 | Unitree | Pure RL + community extensions | Parkour, stairs, manipulation |
| Spot Arm | Boston Dynamics | Model-based + RL residual | Production-grade loco-manipulation |
| Figure 01 | Figure | IL pre-train + RL fine-tune | Object manipulation while walking |
| Digit | Agility | Model-based MPC + RL | Warehouse logistics |
| Atlas | Boston Dynamics | Primarily model-based | Gymnastics, parkour |
| Optimus | Tesla | IL on human video + RL | Factory tasks |

## Summary Diagram

```
Motion naturalness
      ^
      |   IL pre-train + RL fine-tune    <- Tesla, Figure
      |   (closest to human motion)
      |
      |   Reference motion tracking      <- DeepMimic, AMP
      |   (MoCap drives motion quality)
      |
      |   Pure RL + Terrain Curriculum   <- Unitree H1/G1
      |   (functional but less natural)
      |
      |   Residual RL on MPC base        <- Agility, early work
      |   (constrained by base gait)
      |
      +-------------------------------------------> Terrain generalization
                                   low          high
```

y-axis = motion naturalness (how human-like the motion looks).
x-axis = terrain generalization (how well the policy handles unseen terrain).
The two axes trade off: reference-tracking produces natural motion but generalizes less; pure RL generalizes better but looks more mechanical.

**Key insight:** Unlike quadrupeds where pure RL + curriculum is now the de facto standard, humanoid training has no single dominant paradigm. The field is converging toward IL pre-training + RL fine-tuning (the LLM recipe applied to robotics), but reference motion tracking and pure RL both remain competitive depending on the task.

---

# 双足人型机器人运动控制训练方法

系统梳理双足人型机器人（如宇树 H1/G1、Figure 01、Agility Digit）强化学习训练的核心挑战、任务层次、训练范式及工业界现状。

## 为什么人型比四足难

四足机器人有 4 个支撑点、低质心，静止时被动稳定。人型只有 2 只脚、高质心，必须始终主动维持平衡。这个根本差异贯穿训练的每个环节：

```
四足机器人                         人型机器人
-----------                       ---------
4 个支撑点                         2 个支撑点
质心高度 ~0.4m                     质心高度 ~0.7~1.0m（取决于机型）
站立时被动稳定                     随时主动不稳定
12 DOF（每腿 3 个）                29+ DOF（腿+臂+躯干）
摔倒恢复：容易                     摔倒恢复：困难
sim-to-real gap：中等              sim-to-real gap：较大
```

高度不稳定性导致：
- reward 设计约束更强（agent 容易利用不稳定性作弊）
- 训练早期 termination 条件频繁触发
- 通常需要参考运动或轨迹跟踪来引导学习启动

## 任务层次

```
Level 1   站立平衡 / 原地恢复               <- 主动平衡
Level 2   平地行走 / 速度跟踪               <- 基础 locomotion
Level 3   台阶 / 坡道 / 粗糙地面            <- 地形适应
Level 4   跑步 / 跳跃 / 动态步态            <- 高速动态
Level 5   全身运动 + 操作联合控制            <- loco-manipulation
Level 6   Parkour / 极限运动               <- 前沿研究
```

目前大多数量产人型机器人工作在 Level 2-3，Level 4 以上处于研究阶段。

## 核心挑战：不稳定性问题

与四足相比最大的差异：随机策略几秒内就会摔倒，影响训练的两个层面：

**1. 训练早期样本效率极低**

随机策略立即倒地，episode 长度接近零，reward 信号几乎全是摔倒惩罚，agent 无法学到有效信息。

**解决方案：**
- 参考运动跟踪（模仿 MoCap 数据）
- 课程学习（从更容易的初始姿态开始，如蹲姿）
- 大 alive bonus 奖励（只要不倒就给正反馈）
- 以稳定 PD 步态为基础，策略输出残差修正

**2. Reward 设计更复杂**

四足中单靠速度跟踪 reward 就够用。人型不加约束的 agent 会快速找到退化步态：拖脚走、极端倾斜、小碎步。必须加额外正则化项：

```
reward = lin_vel_tracking
       + ang_vel_tracking
       - foot_contact_force_penalty    <- 禁止踩踏
       - body_orientation_penalty      <- 保持直立
       - joint_limit_penalty           <- 禁止过度弯曲
       - action_smoothness_penalty     <- 禁止抖动
       - arm_deviation_penalty         <- 手臂保持自然
       + feet_air_time_reward          <- 鼓励抬脚
```

## 主流训练范式

### 范式 A：参考运动跟踪（人型最常用）

用动作捕捉数据或手工设计的参考轨迹作为监督信号，策略在跟踪参考步态的同时学会抵抗扰动：

```
MoCap 数据 / 手工设计步态
           |
           v
  参考关节角度序列 q_ref(t)
           |
           v
┌──────────────────────────────────┐
│   Tracking + Task Reward         │
│                                  │
│  r = w1 * ||q - q_ref||^2        │  <- 跟踪参考姿态
│    + w2 * vel_tracking           │  <- 跟随速度指令
│    + w3 * end_effector_match     │  <- 落脚精度
│    - w4 * fall_penalty           │  <- 保持直立
└──────────────────────────────────┘
           |
           v
  策略学会自然行走并从推扰中恢复
```

代表工作：DeepMimic（Peng et al., 2018）、AMP（Peng et al., 2021）。

注意两者有本质区别：**DeepMimic** 使用直接姿态跟踪 reward（与参考帧的显式 MSE）。**AMP** 使用对抗判别器来奖励"风格上与参考动作相似"的运动——不需要逐帧精确跟踪，对参考动作质量要求更低，也更灵活。

- **优点：** 运动自然、收敛快、避免退化步态
- **缺点：** 运动质量受参考数据限制；采集多样 MoCap 数据成本高

### 范式 B：纯 RL 无参考（RSL / 宇树 H1/G1 路线）

与四足训练框架相同——Terrain Curriculum + Domain Randomization + Asymmetric Actor-Critic——但需要更重的 reward 正则化和更保守的初始化。

与四足训练的关键差异：

```
四足 PPO 配置                      人型 PPO 配置
--------------------               -------------------
hidden_dim = 128                   hidden_dim = 256~512
alive_bonus = 0                    alive_bonus = 较大
action_scale = 0.25                action_scale = 0.1~0.2（更保守）
kp = 20, kd = 0.5                  kp = 100~200（关节刚度更大）
无手臂关节                          手臂关节固定或轻度控制
```

"29+ DOF"是硬件的总自由度。纯 locomotion 训练时手臂关节**固定在中立姿态**，实际控制的 DOF 降为 10~12（仅腿部），与四足相当。完整 29-DOF 全身控制只在 loco-manipulation 任务（范式 D）中使用。

- **优点：** 无需 MoCap 数据，可泛化到新地形
- **缺点：** 需要大量 reward 调参；运动自然度低于参考运动方案

### 范式 C：基于模型控制器的残差 RL

以经典模型控制器（MPC 或 ZMP 步态）为基础，训练 RL 策略输出残差修正。**ZMP（零力矩点）** 是地面反力合力的作用点——当 ZMP 保持在支撑多边形内时步态静态稳定。ZMP 控制器无需学习就能生成稳定步态，但难以适应复杂地形变化。

```
┌────────────────────────────┐
│  基于模型的基础步态           │
│  （MPC / ZMP / SLIP 模型）  │
│  输出：标称动作              │
└────────────┬───────────────┘
             │ nominal_action
             v
┌────────────────────────────┐
│  残差 RL 策略               │
│  输入：obs + nominal_act    │
│  输出：delta_action         │
└────────────┬───────────────┘
             │ nominal + delta
             v
           电机
```

RL 策略只需学习小幅修正，不需要从零学习完整行走行为，大幅降低探索难度。

- **优点：** 训练快、探索安全、基础行为鲁棒
- **缺点：** 运动包络被基础控制器的能力上界所限制，RL 只能在此范围内修正，无法学出全新的运动行为。基础 controller 是完整 MPC 优化器还是预规划参考轨迹，决定了残差策略可以偏离多远。

### 范式 D：全身控制 + 运动操作联合（Loco-Manipulation）

同时控制行走和手臂操作，机器人在走动的同时与环境交互：

```
┌──────────────────────────────────────────┐
│           全身控制器                       │
│                                          │
│  locomotion task:  velocity tracking     │
│  manipulation task: reach / grasp / push │
│                                          │
│  统一动作空间：29+ 个关节目标                │
└──────────────────────────────────────────┘
```

人型形态的优势在这里体现——手臂可以像人类一样辅助平衡，同时完成操作任务。训练需要多任务 reward 设计：

```
r_total = r_locomotion + r_manipulation + r_coordination
```

代表：Figure 01（操作 + 行走联合）、1X NEO。

### 范式 E：模仿学习预训练 + RL 微调

用人类视频 / 遥操作数据做 BC 预训练，再用 RL 微调提升鲁棒性和闭环性能：

```
阶段 1：模仿学习预训练
  人类运动数据（视频 / MoCap / 遥操作）
           |
           v
  Behavior Cloning（监督学习）
           |
           v
  初始策略（会走，但不够鲁棒）

阶段 2：RL 微调
  初始策略
           |
           v
  PPO / SAC + 任务 reward + 安全约束
           |
           v
  鲁棒策略（抵抗扰动，适应新地形）
```

这与 LLM 训练的逻辑完全对应：先在数据上预训练，再 RLHF。代表：Tesla Optimus、Figure 最新方案。

- **优点：** 从人类数据获得自然运动，快速达到可用初始策略
- **缺点：** 人与机器人运动学根本不同——关节范围、连杆长度、执行器限制均不匹配，直接对原始人类视频做 BC 经常产生不可执行的关节目标。更深层的问题是人类运动动力学（肌肉驱动、顺从）与机器人（电机驱动、刚性）差异极大，即使在仿真中也存在较大 sim-to-real gap。此外，RL 微调可能覆盖预训练阶段学到的细粒度行为。

## Sim-to-Real：人型比四足更难迁移

人型 sim-to-real gap 更大的原因：

1. **关节更多 = 执行器建模误差更多** — 29 个关节 vs 12 个
2. **质心更高 = 对模型误差更敏感** — 微小惯量误差导致大力矩偏差
3. **脚底接触动力学至关重要** — 脚底打滑直接导致摔倒，更难精确建模
4. **柔性体效应** — 躯干弯曲、线缆拉伸、关节间隙影响更显著

超出标准 DR 之外的额外技术：

```
标准 DR（四足足够用）：
  随机化：摩擦、质量、电机增益、外力推扰

人型额外需要：
  随机化：脚底几何形状
  随机化：关节间隙 / 电机死区
  添加：  通信延迟仿真
  添加：  执行器力矩饱和模型
  使用：  更保守的动作裁剪范围
  使用：  接触感知状态估计（不只依赖 IMU）
```

## 工业界现状（2024-2026）

| 机器人 | 公司 | 训练方案 | 代表成就 |
|--------|------|---------|---------|
| H1 | 宇树 | 纯 RL（RSL 框架） | 3.3 m/s 奔跑，世界纪录 |
| G1 | 宇树 | 纯 RL + 社区扩展 | Parkour、爬楼梯、操作 |
| Spot Arm | 波士顿动力 | 模型控制 + RL 残差 | 量产级运动操作 |
| Figure 01 | Figure | IL 预训练 + RL 微调 | 行走同时操作物体 |
| Digit | Agility | 基于模型 MPC + RL | 仓储物流 |
| Atlas | 波士顿动力 | 主要基于模型 | 体操、跑酷 |
| Optimus | Tesla | 人类视频 IL + RL | 工厂任务 |

## 总结图

```
运动自然度
      ^
      │  IL 预训练 + RL 微调          <- Tesla、Figure
      │  （最接近人类运动）
      │
      │  参考运动跟踪                  <- DeepMimic、AMP
      │  （MoCap 驱动运动质量）
      │
      │  纯 RL + Terrain Curriculum   <- 宇树 H1/G1
      │  （功能性好，运动不够自然）
      │
      │  MPC 基础上的残差 RL           <- Agility、早期工作
      │  （受基础步态约束）
      │
      └─────────────────────────────> 地形泛化能力
                             低                  高
```

纵轴 = 运动自然度（运动看起来多像人类）。
横轴 = 地形泛化能力（策略适应未见过地形的能力）。
两个轴存在权衡：参考跟踪运动自然但泛化能力弱；纯 RL 泛化能力强但运动更机械。

**核心洞察：** 与四足机器人（纯 RL + Curriculum 已成事实标准）不同，人型训练目前没有单一主流范式。业界正在向 IL 预训练 + RL 微调（LLM 训练方法论应用于机器人）方向收敛，但参考运动跟踪和纯 RL 在不同任务上仍各有竞争力。
