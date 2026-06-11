# Quadruped Robot Locomotion Training

A systematic overview of how quadruped robots (e.g. Unitree Go2) are trained with reinforcement learning, covering task hierarchy, training paradigms, and the industry trend from scene-specific to unified policies.

## Task Hierarchy

Locomotion tasks can be ranked by complexity:

```
Level 1   Flat-ground walking / velocity tracking     <- basic locomotion
Level 2   Slopes / rough terrain / stairs             <- terrain adaptation
Level 3   Running / jumping / gait transitions        <- high-speed dynamics
Level 4   Parkour / obstacle traversal / climbing     <- extreme maneuvers
Level 5   Navigation + locomotion + perception        <- full-stack
```

Higher levels require fundamentally different training strategies and reward designs.

## Training Paradigms

### Paradigm A: Single Locomotion Policy (mainstream — Unitree / RSL)

One policy covers all terrains via Terrain Curriculum. The environment is divided into a grid of difficulty levels; the robot is promoted or demoted based on performance:

```
┌──────┬──────┬──────┬──────┐
│ flat │ slope│steep │stairs│
│ easy │ med  │ hard │ hard │
├──────┼──────┼──────┼──────┤
│gravel│trench│stairs│discr.│
│ med  │ hard │ hard │ hard │
└──────┴──────┴──────┴──────┘

good performance  ->  promoted to harder cell
poor performance  ->  demoted to easier cell
```

The policy outputs unified joint targets. Terrain variation is implicitly encoded in the **height-scan observation** — a grid of ~187 terrain height samples around the robot, giving the policy local ground geometry. This is the approach used in `unitree_rl_gym` and ANYmal series.

- **Pros:** simple deployment, no scene-switching logic
- **Cons:** cannot fine-tune gait per terrain; extreme maneuvers out of reach

### Paradigm B: Per-Scene / Per-Skill Training + Skill Switching

Each skill is trained independently; a high-level controller selects **one** skill at a time and routes its output to the motors:

```
┌─────────────────────────────────────────┐
│          High-level Controller          │
│  perceive scene -> select ONE skill     │
└──────┬──────┬──────┬──────┬─────────────┘
       │      │      │      │
       v      v      v      v
   Walk   Trot   Jump   Climb
  policy policy policy policy
                │
          (selected policy only)
                │ joint targets
                v
             motors
```

At any moment only the selected policy is active — outputs are not merged.

Representative: early ANYmal parkour, MIT Cheetah, early Boston Dynamics.

- **Pros:** each skill can reach peak performance
- **Cons:** discontinuity at skill transitions; requires a state machine

### Paradigm C: Hierarchical RL

Both levels are learned. The high-level policy generates sub-goals at low frequency; the low-level policy tracks them at control frequency:

```
High-level Policy (5-10 Hz)
  input:  map + goal + current state
  output: sub-goal (e.g. next foothold, body pose target)
              |
              v
Low-level Policy (50 Hz)
  input:  sub-goal + proprioception
  output: 12 joint targets
```

Representative: Walk These Ways (CMU) — one policy, gait conditioned on parameters:

```
conditioning params:  gait_freq, duty_cycle, foot_height, ...
          |
          v
    single policy
          |
          v
  corresponding gait joint targets
```

The high-level only tunes parameters, no policy switching needed.

### Paradigm D: Parkour / Extreme Maneuvers (recent trend)

Targets Level 4 tasks. Core design: Teacher-Student + perceptual input.

```
Teacher (training)
  input:  proprioception + privileged terrain info (heightmap, obstacle positions)
  -> learns optimal motion strategy

Student (deployment)
  input:  proprioception + depth image (current frame or small stack)
  -> infers terrain geometry directly from pixels
  -> imitates Teacher actions
```

Note: the Student here uses a **depth image** as perceptual input (not an obs-history buffer). The RMA-style history buffer is a separate technique used when the goal is implicit system identification (e.g. inferring friction or mass from proprioception history).

Different obstacle types are trained as separate skills, but unified under the same framework:

```
skill 1: jump over
skill 2: jump up
skill 3: jump down
skill 4: crawl

Each skill trains a Teacher independently,
then all are distilled into a single Student.
```

Representative: "Robot Parkour Learning" (Zhuang et al., 2023) — Go2 achieves parkour, platform jumping, and climbing.

## How Scene Coverage Works in Practice

Industry rarely trains separate policies and stitches them together. Instead, all scenes are covered in a single training run through environment design:

| Technique | Approach |
|-----------|----------|
| Terrain Curriculum | Adaptive difficulty — one run covers all terrain types |
| Domain Randomization | Physics parameter randomization covers scene variants |
| Gait Conditioning | Single policy, different gaits via conditioning params |
| Multi-task RL | Shared backbone network with per-task embeddings |

Cases where separate training is genuinely needed:

- **Extreme maneuvers vs daily walking** — reward functions conflict
- **Different speed regimes** — slow precise control vs high-speed gaits are incompatible
- **Contact-rich tasks** (stair climbing) — require specialized contact rewards

## Summary Diagram

```
Task complexity
      ^
      |  parkour / climbing        <- Teacher-Student + per-skill + visual perception
      |
      |  multi-terrain locomotion  <- single policy + Terrain Curriculum + DR
      |
      |  flat-ground walking       <- standard PPO
      |
      +------------------------------------------------> Training difficulty
                                              low                       high
```

x-axis = training difficulty (how hard it is to achieve this level):
flat walking is the easiest to train; parkour requires the most engineering effort.

**Industry trend:** use one large general policy to cover Levels 1-3; train extreme maneuvers (Level 4) separately and merge. Spot and H1 are both moving toward a "one network for everything" direction, but full unification is not yet achieved.

---

# 四足机器狗运动控制训练方法

系统梳理四足机器狗（如宇树 Go2）强化学习训练的任务层次、训练范式和工业界从分场景到统一策略的演进趋势。

## 任务层次

运动控制任务按复杂度分级：

```
Level 1   平地行走 / 速度跟踪               <- 基础 locomotion
Level 2   上下坡 / 粗糙地面 / 台阶          <- 地形适应
Level 3   跑步 / 跳跃 / 步态切换            <- 高速动态
Level 4   parkour / 翻越障碍 / 攀爬        <- 极限运动
Level 5   导航 + 运动 + 感知联合            <- 全栈
```

越高的 level，训练策略和 reward 设计差异越大。

## 主流训练范式

### 范式 A：单一 Locomotion Policy（主流，宇树 / RSL 路线）

一个策略覆盖所有地形，靠 Terrain Curriculum 自适应难度。环境被划分为难度格子，机器人根据表现晋级或降级：

```
┌──────┬──────┬──────┬──────┐
│平地  │缓坡  │陡坡  │台阶  │
│easy  │med   │hard  │hard  │
├──────┼──────┼──────┼──────┤
│碎石  │沟槽  │楼梯  │离散  │
│med   │hard  │hard  │hard  │
└──────┴──────┴──────┴──────┘

表现好 -> 移到更难的格子
表现差 -> 退回更简单的格子
```

策略输出统一的关节目标，地形差异通过 obs 中的 **height scan** 隐式感知——height scan 是机器人周围约 187 个地形高度采样点，给策略提供局部地面几何信息。这是 `unitree_rl_gym` 和 ANYmal 系列的核心做法。

- **优点：** 部署简单，无需场景切换逻辑
- **缺点：** 难以精细控制每种地形的步态，极限动作难以实现

### 范式 B：分场景 / 分技能训练 + 技能切换

每个技能单独训练，上层控制器在运行时选择**其中一个**技能并将其输出送到电机：

```
┌─────────────────────────────────────────┐
│          高层控制器                       │
│  感知场景 -> 选择唯一技能                  │
└──────┬──────┬──────┬──────┬─────────────┘
       │      │      │      │
       v      v      v      v
   Walk   Trot   Jump   Climb
  policy policy policy policy
                │
          （仅当前选中的 policy）
                │ joint targets
                v
              电机
```

同一时刻只有一个 policy 处于激活状态，输出不做合并。

代表：早期 ANYmal parkour、MIT Cheetah、早期波士顿动力。

- **优点：** 每个技能可以做到极限性能
- **缺点：** 技能切换时有不连续性，需要设计状态机

### 范式 C：分层强化学习（Hierarchical RL）

上下两层都是学出来的。上层低频输出子目标，下层高频跟踪：

```
高层策略（5~10 Hz）
  输入：地图 + 目标位置 + 当前状态
  输出：子目标（如下一步落脚点、身体姿态目标）
              |
              v
底层策略（50 Hz）
  输入：子目标 + 本体感知
  输出：12 个关节 target
```

代表：Walk These Ways（CMU）—— 用条件参数控制步态，一个策略走天下：

```
conditioning params:  gait_freq, duty_cycle, foot_height, ...
          |
          v
    单一 policy
          |
          v
  对应步态的关节目标
```

上层只需调参数，不需要切换 policy。

### 范式 D：Parkour / 极限运动（近两年热点）

专门针对 Level 4 任务，核心设计：Teacher-Student + 感知输入。

```
Teacher（训练时）
  输入：本体感知 + 特权地形信息（高程图、障碍位置）
  -> 学出最优运动策略

Student（部署时）
  输入：本体感知 + 深度图像（当前帧或少量帧堆叠）
  -> 直接从像素推断地形几何
  -> 模仿 Teacher 动作
```

注意：此处 Student 的感知输入是**深度图像**，而非 obs 历史 buffer。RMA 风格的历史 buffer 是另一种技术，用于从本体感知历史中隐式推断摩擦系数、质量等环境参数，两者目的不同。

不同障碍类型分技能训练，但统一在同一框架下：

```
skill 1: jump over  （跳过）
skill 2: jump up    （跳上台）
skill 3: jump down  （跳下）
skill 4: crawl      （匍匐）

每个 skill 单独训 Teacher，然后统一蒸馏进 Student
```

代表：Zhuang et al. "Robot Parkour Learning"（2023），Go2 实现了跑酷、跳台、攀爬。

## 实际如何覆盖多场景

工业界通常不是"分场景训完再拼"，而是在一次训练中通过环境设计覆盖所有场景：

| 方法 | 做法 |
|------|------|
| Terrain Curriculum | 地形难度自适应，一次训练覆盖所有地形 |
| Domain Randomization | 物理参数随机化，自动覆盖各种场景变体 |
| Gait Conditioning | 同一 policy 通过条件参数输出不同步态 |
| Multi-task RL | 多任务共享底层网络，各自有 task embedding |

真正需要分开训练的情况：

- **极限动作与日常行走**的 reward 函数互相矛盾
- **不同速度域**：慢速精确控制 vs 高速奔跑的步态完全不同
- **接触丰富任务**（爬楼梯）需要专门的 contact-rich reward

## 总结图

```
任务复杂度
      ^
      │  parkour / climbing        <- Teacher-Student + 分技能 + 视觉感知
      │
      │  multi-terrain locomotion  <- 单一 policy + Terrain Curriculum + DR
      │
      │  flat-ground walking       <- 标准 PPO
      │
      └─────────────────────────────> 训练难度
                             低                  高
```

横轴 = 训练难度（实现该 level 所需的工程投入）：平地行走最容易，Parkour 工程复杂度最高。

**工业界趋势：** 用一个足够大的通用 policy 覆盖 Level 1-3，极限动作（Level 4）单独训再融合。Spot 和 H1 都在往"一个网络走天下"方向走，但完全统一还未实现。
