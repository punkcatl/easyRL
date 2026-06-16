# Go2 Locomotion Pipeline Guide

Complete guide for training, exporting, and deploying the Go2 walking policy.
Includes pitfalls encountered during development.

## Pipeline Overview

```
train_teacher.py          export_onnx.py            Deploy
      │                        │                       │
      │ PPO + DR               │ bake normalization    │ onnxruntime
      │ 15 rounds              │ + clip                │ 0.013ms/step
      ▼                        ▼                       ▼
teacher_final.pth ───────► policy_go2.onnx ──────► Go2 onboard
     (194 KB)                  (97 KB)              computer
```

### File Roles

| File | Role | Size |
|------|------|------|
| `teacher_final.pth` | PyTorch checkpoint (actor + critic + obs_rms) | 194 KB |
| `policy_go2.onnx` | Deployment artifact (actor + obs_rms, no critic) | 97 KB |

## Phase 1: Teacher Training

### Commands

```bash
# Fresh start (default 5000 iterations)
python applications/go2_locomotion/train_teacher.py

# Resume from checkpoint
python applications/go2_locomotion/train_teacher.py --resume results/teacher_iter1500.pth

# Resume from specific round
python applications/go2_locomotion/train_teacher.py --resume results/teacher_round14.pth
```

### What Happens During Training

- Algorithm: PPO with Domain Randomization
- Input: `obs(48)` — joint pos/vel, body orientation, commands
- Output: `action(12)` — normalized joint position targets in [-1, 1]
- Privileged info (friction, mass) feeds critic only, NOT actor
- Auto-saves checkpoint every 500 iterations to `results/teacher_iter{N}.pth`
- Final model saved as `results/teacher_final.pth`
- Reward curve saved as `results/teacher_rewards.npy`

### Training Phases (Automatic)

| Iteration | Phase | DR Level |
|-----------|-------|----------|
| 0 - 500 | Phase 1 | No DR (learn basic walking first) |
| 500 - 1500 | Phase 2 | Light DR (friction ±10%, mass ±5%) |
| 1500+ | Phase 3 | Full DR (friction ±50%, mass ±20%, push 3N) |

### Monitoring Output

Every 50 iterations prints:
```
Iter 500/5000 | reward: 2.31 | track: 0.72 pct30: 85% | vx: [0.30,0.80] | lr: 2.7e-04 | light-DR
```
- `reward`: 50-iter moving average
- `track`: mean(actual_vx / commanded_vx), target > 0.6
- `pct30`: % of envs with tracking ratio > 0.3 (measures survival)
- `vx`: current command curriculum range (auto-expands when tracking is stable)
- `lr`: current learning rate (linear decay)

Ground-truth eval runs every 200 iterations (deterministic, fixed command).

### Key Training Parameters (config.py)

```python
# Most impactful parameters for tuning:

"action_scale": 0.35,       # stride size — increase for faster walking
"kp": [..., 35, 35, ...],   # PD gains — increase for stronger joints
"control_dt": 0.02,         # 50 Hz control — decrease for faster reactions

# Reward weights (biggest levers):
"lin_vel_tracking": 3.0,    # how strongly to follow speed commands
"forward_progress": 1.5,    # raw velocity bonus
"gait_schedule": 2.0,       # trot gait enforcement
"termination_penalty": -10.0,  # fall penalty

# PPO hyperparameters:
"lr": 3e-4,                 # initial learning rate
"lr_end": 3e-5,             # final learning rate
"entropy_coef": 0.02,       # exploration (higher = more random)
"n_iterations": 5000,       # total training iterations
"num_envs": 128,            # parallel environments
```

### Tuning Workflow

1. Start a fresh round, observe reward curve for ~500 iterations
2. If reward plateaus early → problem is reward shaping, not capacity
3. If robot lurches/hops → increase `gait_schedule`, add `gait_symmetry`
4. If robot walks but too slow → increase `action_scale` or `kp`
5. If robot falls with DR → DR is too aggressive, increase `dr_phase1_end`
6. Save checkpoint before risky changes: copy `teacher_final.pth` to `teacher_round{N}.pth`

### Evaluation

```bash
# Headless evaluation (20 episodes, prints stats)
python applications/go2_locomotion/evaluate.py --mode teacher --episodes 20

# With MuJoCo visualization
python applications/go2_locomotion/evaluate.py --mode teacher --render

# Custom checkpoint
python applications/go2_locomotion/evaluate.py --mode teacher --model results/teacher_round10.pth

# Specific episode count
python applications/go2_locomotion/evaluate.py --mode teacher --episodes 50 --render
```

Output format:
```
==================================================
Teacher Evaluation (20 episodes, no DR)
  Avg Reward : 5439.39 ± 257.27
  Avg Steps  : 1000 / 1000
  Survival % : 100%
==================================================
```

### Key Result

Round 15: `action_scale` 0.25→0.35, achieved 1.0 m/s stable walking (97% tracking).

## Phase 2: Student Distillation (Optional)

```bash
python applications/go2_locomotion/train_student.py
```

> **Important:** Current teacher actor does NOT use privileged observations.
> Distillation is unnecessary for deployment. This phase exists for future use
> when teacher is retrained with privileged actor input (e.g., terrain params
> fed to actor for adaptive behavior).

### When Distillation IS Needed

```
Teacher actor uses privileged info:
  obs(48) + privileged(7) → action(12)     # can't deploy directly
                                            # privileged info unavailable on robot

Student replaces privileged with history:
  obs_history(960) → AdaptationModule → z(16)
  obs(48) + z(16) → StudentPolicy → action(12)
```

### When Distillation is NOT Needed (Current State)

```
Teacher actor is obs-only:
  obs(48) → action(12)                     # directly deployable
```

## Phase 3: ONNX Export

### Commands

```bash
# Default: export teacher_final.pth -> policy_go2.onnx
python applications/go2_locomotion/export_onnx.py

# Custom input/output paths (edit the script or call the function):
python -c "
from applications.go2_locomotion.export_onnx import export_teacher_onnx
export_teacher_onnx(
    model_path='applications/go2_locomotion/results/teacher_round15.pth',
    output_path='applications/go2_locomotion/results/policy_go2_r15.onnx'
)
"
```

### What Gets Baked In

1. **Observation normalization** — running mean/std from training (`obs_rms`)
2. **Actor MLP** — 48→128→128→12 (weights from training)
3. **Output clipping** — `clamp(-1, 1)`
4. **Critic is discarded** — only actor is needed at inference

### Output Verification (automatic)

```
ONNX export successful: results/policy_go2.onnx
  obs input:     (48,)
  action output: (12,)

Accuracy: max|PyTorch - ONNX| = 0.00e+00  (exact match)

Inference Benchmark (1000 runs):
  Avg latency : 0.013 ms
  Max freq    : 77809 Hz
  50 Hz budget: PASS (need < 20 ms)

  File size   : 96.8 KB
```

### Verify ONNX Works in Simulation

```bash
python -c "
import numpy as np, onnxruntime as ort
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.config import config
session = ort.InferenceSession('applications/go2_locomotion/results/policy_go2.onnx')
env = Go2Env(config, render_mode='human')
obs, _ = env.reset()
for _ in range(1000):
    action = session.run(None, {'obs': obs.reshape(1,-1).astype(np.float32)})[0].flatten()
    obs, _, t, tr, _ = env.step(action)
    env.render()
    if t or tr: break
env.close()
"
```

## Deployment

### Requirements

- `onnxruntime` (no PyTorch needed)
- URDF/MuJoCo model for PD control parameters

### Minimal Inference Loop

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("policy_go2.onnx")

# obs: [body_ang_vel(3), projected_gravity(3), commands(3),
#        joint_pos_rel(12), joint_vel(12), last_action(12), command(3)]
obs = get_robot_observation()  # shape (48,)

action = session.run(None, {"obs": obs.reshape(1, -1).astype(np.float32)})[0].flatten()

# action is in [-1, 1], convert to joint targets:
target_angles = action_scale * action + default_angles
torques = kp * (target_angles - joint_pos) - kd * joint_vel
```

### Control Parameters

```python
action_scale = 0.35
kp = [30, 30, 35, 30, 30, 35, 30, 30, 35, 30, 30, 35]  # per-joint
kd = [0.6, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7]
```

## Pitfalls & Lessons Learned

### 1. Teacher Action Output Not Bounded

**Symptom:** Student distillation loss exploded (val loss = 88).

**Root Cause:** PPO actor network outputs unbounded values (mean ≈ 2000~10000).
The environment clips to [-1, 1] in `env.step()`, so teacher actually executes
saturated ±1 actions. But the raw network output was recorded as training target
for the student.

**Fix:** Clip teacher action before recording:
```python
# Before (wrong):
action = mean.cpu().numpy().flatten()

# After (correct):
action = np.clip(mean.cpu().numpy().flatten(), -1.0, 1.0)
```

**Lesson:** Always verify the data distribution before training. A quick
`print(actions.min(), actions.max())` would have caught this immediately.

### 2. Student Distillation is Unnecessary Here

**Symptom:** Student achieves low BC loss (0.0015) but falls immediately
in simulation (60 steps).

**Root Cause:** Compounding error in behavior cloning — small prediction errors
accumulate over time, pushing the student into states never seen in training data.
More fundamentally, the distillation is solving a non-problem: teacher actor
doesn't use privileged info, so there's nothing to "distill away."

**Fix:** Skip distillation entirely. Export teacher directly to ONNX.

**Lesson:** Before building a complex pipeline, verify the assumptions:
- Does the teacher actually use information unavailable at deployment?
- If not, direct export is simpler, faster, and lossless.

### 3. DAgger Doesn't Help If Architecture is Wrong

**Symptom:** Attempted DAgger (iterative data aggregation) to fix student.

**Why it wouldn't have worked well:** The student's Adaptation Module
(960→512→256→32) tries to infer latent environment parameters from history.
But the teacher never used those parameters for action selection, so the
"correct" latent z is undefined — there's no consistent mapping to learn.

**Lesson:** DAgger fixes distribution shift, not architecture mismatch.
The right question is: "What information does the student need that it
doesn't have?" If the answer is "nothing the teacher doesn't also have,"
distillation adds complexity without benefit.

### 4. Obs Normalization Must Be Included in Export

**Why:** The teacher was trained with running mean/std normalization.
Without baking this into the ONNX graph, the deployed model receives
un-normalized inputs and produces garbage actions.

**Solution:** `TeacherONNXWrapper` registers obs_mean and obs_std as buffers:
```python
class TeacherONNXWrapper(nn.Module):
    def __init__(self, trainer):
        self.register_buffer("obs_mean", ...)
        self.register_buffer("obs_std", ...)

    def forward(self, obs):
        normalized = (obs - self.obs_mean) / self.obs_std
        ...
```

## Quick Reference

```bash
# === Full pipeline (zero to deploy) ===

# 1. Train (hours)
python applications/go2_locomotion/train_teacher.py

# 2. Evaluate training result
python applications/go2_locomotion/evaluate.py --mode teacher --render

# 3. Export ONNX (seconds)
python applications/go2_locomotion/export_onnx.py

# 4. Verify ONNX in simulation
python -c "
import numpy as np, onnxruntime as ort
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.config import config
session = ort.InferenceSession('applications/go2_locomotion/results/policy_go2.onnx')
env = Go2Env(config, render_mode='human')
obs, _ = env.reset()
for _ in range(1000):
    action = session.run(None, {'obs': obs.reshape(1,-1).astype(np.float32)})[0].flatten()
    obs, _, t, tr, _ = env.step(action)
    env.render()
    if t or tr: break
env.close()
"

# === Resume training ===
python applications/go2_locomotion/train_teacher.py --resume results/teacher_round14.pth

# === Compare checkpoints ===
python applications/go2_locomotion/evaluate.py --mode teacher --model results/teacher_round10.pth --render
python applications/go2_locomotion/evaluate.py --mode teacher --model results/teacher_round15.pth --render
```

---

# Go2 运动控制 Pipeline 指南

训练、导出、部署 Go2 行走策略的完整指南，包含开发过程中遇到的坑。

## Pipeline 概览

```
train_teacher.py          export_onnx.py            部署
      │                        │                       │
      │ PPO + DR               │ 烘焙归一化            │ onnxruntime
      │ 15 轮迭代             │ + clip                │ 0.013ms/步
      ▼                        ▼                       ▼
teacher_final.pth ───────► policy_go2.onnx ──────► Go2 机载电脑
     (194 KB)                  (97 KB)
```

### 文件职责

| 文件 | 职责 | 大小 |
|------|------|------|
| `teacher_final.pth` | PyTorch 完整 checkpoint（actor + critic + obs_rms） | 194 KB |
| `policy_go2.onnx` | 部署产物（仅 actor + obs_rms，无 critic） | 97 KB |

## Phase 1：Teacher 训练

### 命令

```bash
# 从零开始（默认 5000 iterations）
python applications/go2_locomotion/train_teacher.py

# 从 checkpoint 续训
python applications/go2_locomotion/train_teacher.py --resume results/teacher_iter1500.pth

# 从某轮次续训
python applications/go2_locomotion/train_teacher.py --resume results/teacher_round14.pth
```

### 训练过程

- 算法：PPO + Domain Randomization
- 输入：`obs(48)` — 关节位置/速度、机体姿态、速度命令
- 输出：`action(12)` — 归一化关节目标位置 [-1, 1]
- 特权信息（摩擦、质量）仅输入 critic，不输入 actor
- 每 500 iter 自动保存 `results/teacher_iter{N}.pth`
- 最终模型保存为 `results/teacher_final.pth`
- Reward 曲线保存为 `results/teacher_rewards.npy`

### 训练阶段（自动切换）

| 迭代次数 | 阶段 | DR 强度 |
|----------|------|---------|
| 0 - 500 | Phase 1 | 无 DR（先学会基本行走） |
| 500 - 1500 | Phase 2 | 轻度 DR（摩擦 ±10%，质量 ±5%） |
| 1500+ | Phase 3 | 全 DR（摩擦 ±50%，质量 ±20%，推力 3N） |

### 监控输出

每 50 iter 打印：
```
Iter 500/5000 | reward: 2.31 | track: 0.72 pct30: 85% | vx: [0.30,0.80] | lr: 2.7e-04 | light-DR
```
- `reward`：最近 50 iter 平均 reward
- `track`：mean(实际速度 / 命令速度)，目标 > 0.6
- `pct30`：tracking > 0.3 的环境比例（衡量存活率）
- `vx`：当前速度命令范围（tracking 稳定时自动扩展）
- `lr`：当前学习率（线性衰减）

每 200 iter 跑一次确定性 eval（固定命令，无随机）。

### 关键训练参数（config.py）

```python
# 影响最大的参数：

"action_scale": 0.35,       # 步幅大小——增大以走更快
"kp": [..., 35, 35, ...],   # PD 增益——增大以增强关节力量
"control_dt": 0.02,         # 50 Hz 控制——减小以加快反应

# Reward 权重（最大的调节杠杆）：
"lin_vel_tracking": 3.0,    # 速度跟踪强度
"forward_progress": 1.5,    # 前进速度奖励
"gait_schedule": 2.0,       # trot 步态强制程度
"termination_penalty": -10.0,  # 摔倒惩罚

# PPO 超参：
"lr": 3e-4,                 # 初始学习率
"lr_end": 3e-5,             # 最终学习率
"entropy_coef": 0.02,       # 探索程度（越高越随机）
"n_iterations": 5000,       # 总训练迭代数
"num_envs": 128,            # 并行环境数
```

### 调参工作流

1. 开始新一轮，观察 reward 曲线约 500 iter
2. 如果 reward 过早平台期 → 问题出在 reward 设计，不是网络容量
3. 如果机器人晃动/跳跃 → 增大 `gait_schedule`，添加 `gait_symmetry`
4. 如果能走但太慢 → 增大 `action_scale` 或 `kp`
5. 如果加 DR 后摔倒 → DR 太激进，增大 `dr_phase1_end`
6. 危险改动前保存 checkpoint：`cp teacher_final.pth teacher_round{N}.pth`

### 评估

```bash
# 无渲染评估（20 个 episode，打印统计）
python applications/go2_locomotion/evaluate.py --mode teacher --episodes 20

# MuJoCo 可视化
python applications/go2_locomotion/evaluate.py --mode teacher --render

# 指定 checkpoint
python applications/go2_locomotion/evaluate.py --mode teacher --model results/teacher_round10.pth

# 自定义 episode 数 + 可视化
python applications/go2_locomotion/evaluate.py --mode teacher --episodes 50 --render
```

输出格式：
```
==================================================
Teacher Evaluation (20 episodes, no DR)
  Avg Reward : 5439.39 ± 257.27
  Avg Steps  : 1000 / 1000
  Survival % : 100%
==================================================
```

### 关键成果

Round 15：`action_scale` 0.25→0.35，达成 1.0 m/s 稳定行走（97% 速度跟踪）。

## Phase 2：Student 蒸馏（当前不需要）

```bash
python applications/go2_locomotion/train_student.py
```

> **重要：** 当前 teacher 的 actor 不使用特权观测。部署不需要蒸馏。
> 此阶段留给未来——当 teacher 重训为特权 actor 输入（如地形参数直接
> 喂给 actor 实现自适应行为）时才需要。

### 什么时候需要蒸馏

```
Teacher actor 使用特权信息：
  obs(48) + privileged(7) → action(12)     # 无法直接部署
                                            # 机器人上拿不到特权信息

Student 用历史替代特权信息：
  obs_history(960) → AdaptationModule → z(16)
  obs(48) + z(16) → StudentPolicy → action(12)
```

### 什么时候不需要蒸馏（当前状态）

```
Teacher actor 仅用 obs：
  obs(48) → action(12)                     # 直接可部署
```

## Phase 3：ONNX 导出

### 命令

```bash
# 默认：teacher_final.pth -> policy_go2.onnx
python applications/go2_locomotion/export_onnx.py

# 自定义输入/输出路径（调用函数）：
python -c "
from applications.go2_locomotion.export_onnx import export_teacher_onnx
export_teacher_onnx(
    model_path='applications/go2_locomotion/results/teacher_round15.pth',
    output_path='applications/go2_locomotion/results/policy_go2_r15.onnx'
)
"
```

### 烘焙内容

1. **观测归一化** — 训练时的 running mean/std（`obs_rms`）
2. **Actor MLP** — 48→128→128→12（训练权重）
3. **输出裁剪** — `clamp(-1, 1)`
4. **Critic 丢弃** — 推理只需 actor

### 导出验证（自动执行）

```
ONNX export successful: results/policy_go2.onnx
  obs input:     (48,)
  action output: (12,)

Accuracy: max|PyTorch - ONNX| = 0.00e+00  (exact match)

Inference Benchmark (1000 runs):
  Avg latency : 0.013 ms
  Max freq    : 77809 Hz
  50 Hz budget: PASS (need < 20 ms)

  File size   : 96.8 KB
```

### 仿真中验证 ONNX

```bash
python -c "
import numpy as np, onnxruntime as ort
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.config import config
session = ort.InferenceSession('applications/go2_locomotion/results/policy_go2.onnx')
env = Go2Env(config, render_mode='human')
obs, _ = env.reset()
for _ in range(1000):
    action = session.run(None, {'obs': obs.reshape(1,-1).astype(np.float32)})[0].flatten()
    obs, _, t, tr, _ = env.step(action)
    env.render()
    if t or tr: break
env.close()
"
```

## 部署

### 依赖

- `onnxruntime`（不需要 PyTorch）
- URDF/MuJoCo 模型用于 PD 控制参数

### 最小推理循环

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("policy_go2.onnx")

# obs: [body_ang_vel(3), projected_gravity(3), commands(3),
#        joint_pos_rel(12), joint_vel(12), last_action(12), command(3)]
obs = get_robot_observation()  # shape (48,)

action = session.run(None, {"obs": obs.reshape(1, -1).astype(np.float32)})[0].flatten()

# action 在 [-1, 1]，转换为关节目标：
target_angles = action_scale * action + default_angles
torques = kp * (target_angles - joint_pos) - kd * joint_vel
```

### 控制参数

```python
action_scale = 0.35
kp = [30, 30, 35, 30, 30, 35, 30, 30, 35, 30, 30, 35]  # 每个关节
kd = [0.6, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7]
```

## 踩坑记录

### 1. Teacher Action 输出无界

**现象：** Student 蒸馏 loss 爆炸（val loss = 88）。

**根因：** PPO actor 网络输出无界值（mean ≈ 2000~10000）。
环境在 `env.step()` 中 clip 到 [-1, 1]，所以 teacher 实际执行的都是
饱和的 ±1 动作。但原始网络输出被直接记录为 student 的训练目标。

**修复：** 记录前 clip teacher action：
```python
# 修复前（错误）：
action = mean.cpu().numpy().flatten()

# 修复后（正确）：
action = np.clip(mean.cpu().numpy().flatten(), -1.0, 1.0)
```

**教训：** 训练前一定要验证数据分布。一行 `print(actions.min(), actions.max())`
就能立刻发现问题。

### 2. 当前不需要 Student 蒸馏

**现象：** Student BC loss 很低（0.0015）但仿真中立刻摔倒（60 步）。

**根因：** 行为克隆的累积误差——小预测误差随时间累积，把 student 推入
训练数据中从未出现的状态。更根本地，蒸馏在解决一个不存在的问题：
teacher actor 不使用特权信息，没有东西需要"蒸馏掉"。

**修复：** 跳过蒸馏，直接导出 teacher 为 ONNX。

**教训：** 在构建复杂 pipeline 前，先验证前提假设：
- Teacher 是否真的使用了部署时不可用的信息？
- 如果没有，直接导出更简单、更快、无损。

### 3. DAgger 救不了架构错误

**现象：** 尝试 DAgger（迭代数据聚合）来修复 student。

**为什么不管用：** Student 的 Adaptation Module（960→512→256→32）试图
从历史中推断环境潜变量。但 teacher 从未使用这些变量做决策，所以
"正确的"潜变量 z 是未定义的——不存在一致的映射可学。

**教训：** DAgger 解决 distribution shift，不解决架构不匹配。
正确的问题是："student 需要什么信息是它目前没有的？"
如果答案是"teacher 也没有的信息"，那蒸馏只增加复杂度，没有收益。

### 4. 导出必须包含 Obs 归一化

**原因：** Teacher 训练时用了 running mean/std 归一化。
如果不烘焙到 ONNX 图中，部署模型收到未归一化的输入，输出垃圾动作。

**解决方案：** `TeacherONNXWrapper` 把 obs_mean 和 obs_std 注册为 buffer：
```python
class TeacherONNXWrapper(nn.Module):
    def __init__(self, trainer):
        self.register_buffer("obs_mean", ...)
        self.register_buffer("obs_std", ...)

    def forward(self, obs):
        normalized = (obs - self.obs_mean) / self.obs_std
        ...
```

## 快速参考

```bash
# === 完整 pipeline（从零到部署） ===

# 1. 训练（数小时）
python applications/go2_locomotion/train_teacher.py

# 2. 评估训练结果
python applications/go2_locomotion/evaluate.py --mode teacher --render

# 3. 导出 ONNX（秒级）
python applications/go2_locomotion/export_onnx.py

# 4. 验证 ONNX 可视化
python -c "
import numpy as np, onnxruntime as ort
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.config import config
session = ort.InferenceSession('applications/go2_locomotion/results/policy_go2.onnx')
env = Go2Env(config, render_mode='human')
obs, _ = env.reset()
for _ in range(1000):
    action = session.run(None, {'obs': obs.reshape(1,-1).astype(np.float32)})[0].flatten()
    obs, _, t, tr, _ = env.step(action)
    env.render()
    if t or tr: break
env.close()
"

# === 续训 ===
python applications/go2_locomotion/train_teacher.py --resume results/teacher_round14.pth

# === 对比不同 checkpoint ===
python applications/go2_locomotion/evaluate.py --mode teacher --model results/teacher_round10.pth --render
python applications/go2_locomotion/evaluate.py --mode teacher --model results/teacher_round15.pth --render
```
