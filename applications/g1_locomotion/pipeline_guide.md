# G1 Locomotion Pipeline Guide

Complete guide for training, distilling, and deploying the G1 humanoid walking policy.
Includes pitfalls and lessons from 7 rounds of iterative training.

## Pipeline Overview

```
train_teacher.py       collect_teacher_data.py    train_student.py       export_onnx.py
      │                        │                        │                      │
      │ PPO + Isaac Lab        │ frozen teacher         │ Behavior Cloning     │ fuse + export
      │ 7 rounds               │ 449k transitions       │ val_loss=0.00008     │ verify
      ▼                        ▼                        ▼                      ▼
teacher_final.pt ──► teacher_distill_data.npz ──► student_best.pt ──► student_g1.onnx
     (2 MB)                  (9.9 GB)                 (6.4 MB)            (6.4 MB, 0.13ms)
```

### File Roles

| File | Role | Size |
|------|------|------|
| `teacher_final.pt` | rsl_rl checkpoint (actor + critic, trained by PPO) | 2 MB |
| `teacher_distill_data.npz` | Rollout dataset (obs_history, obs_current, actions) | 9.9 GB |
| `student_best.pt` | AdaptationModule + StudentPolicy (best val loss) | 6.4 MB |
| `student_g1.onnx` | Fused deployment artifact (adaptation + policy) | 6.4 MB |

---

## Phase 1: Teacher Training

### Commands

```bash
conda activate env_isaaclab
cd ~/Desktop/myRL/easyRL

# Train from scratch (1500 iter, ~25 min on A4000)
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 --num_envs 1024 --headless

# Resume from checkpoint
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 --num_envs 1024 --headless \
    --resume --load_run 2026-06-15_23-20-45 --checkpoint model_900.pt

# Quick test (verify config works)
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 --num_envs 64 --max_iterations 5 --headless
```

### What Happens During Training

- Framework: Isaac Lab 2.3.0 + rsl_rl 3.0.1
- GPU parallel: 1024 environments on RTX A4000
- Input: `obs(123)` — joint pos/vel, body orientation, projected gravity, commands
- Output: `action(37)` — joint position targets (G1_MINIMAL 29 DOF + hands)
- Asymmetric Actor-Critic: critic sees more than actor
- Auto-saves every 100 iterations to `results/g1_flat_locomotion/<timestamp>/model_{N}.pt`
- Final model saved as `teacher_final.pt`
- TensorBoard logs written to same directory

### Key Training Parameters (config/flat_env_cfg.py)

```python
# Most impactful for tuning:

# Velocity tracking (primary task driver)
track_lin_vel_xy_exp.weight = 2.0       # how strongly to follow speed commands
track_lin_vel_xy_exp.params["std"] = 0.4  # tracking kernel width

# Gait quality (solved shuffle in Round 3)
feet_clearance.weight = 1.0             # force foot lift to 10cm
gait_schedule.weight = 1.0             # 0.8s periodic gait clock (breakthrough in R7)

# Posture (solved crouching in Round 4)
flat_orientation_l2.weight = -2.0       # stay upright
base_height_l2.weight = -5.0           # maintain 0.74m height

# Joint control (fixed splayed legs in Round 5)
joint_deviation_hip.weight = -0.5       # punish hip splay

# Smoothness
action_rate_l2.weight = -0.01          # smooth actions

# Command range
commands.base_velocity.ranges.lin_vel_x = (0.3, 0.6)
```

### Tuning Workflow

1. Train 1500 iter, check TensorBoard (`mean_reward`, `episode_length`, reward components)
2. Verify robot moves: headless displacement test (see Verification section)
3. Visual check via `play.py` (if GUI available)
4. Diagnose problem from reward components:

| Symptom | TensorBoard Signal | Fix |
|---------|-------------------|-----|
| Shuffle (no foot lift) | `feet_clearance` low | Add/increase `feet_clearance` weight |
| Crouching / leaning | `base_height` penalty high | Add `base_height_l2` |
| Splayed legs | `joint_deviation_hip` high | Increase hip penalty |
| Fast tiny steps | `gait_schedule` low | Increase `gait_schedule` weight |
| Robot doesn't move | `track_lin_vel` low + all penalties dominate | Reduce penalties or increase vel weight |
| Frequent falling | `termination_penalty` non-zero | Reduce task difficulty |

5. Update `training_log.md` BEFORE making changes
6. Commit config change, start next round

### Monitoring with TensorBoard

```bash
tensorboard --logdir applications/g1_locomotion/results/ --port 6006
```

Key metrics to watch:
- `Train/mean_reward` — should rise and converge
- `Train/mean_episode_length` — should approach 1000 (max)
- `Episode_Reward/feet_clearance` — foot lift quality (target > 0.9)
- `Episode_Reward/gait_schedule` — gait periodicity (target > 0.9)
- `Episode_Reward/track_lin_vel_xy_exp` — velocity tracking (target > 1.5)

### Headless Displacement Verification

Since GUI visualization is unreliable, verify with displacement test:

```bash
conda run -n env_isaaclab python -c "
import sys; sys.path.insert(0, '.')
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--headless'])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch, numpy as np
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import applications.g1_locomotion.config

env_cfg = gym.spec('G1-Flat-Custom-Play-v0').kwargs['env_cfg_entry_point']()
agent_cfg = gym.spec('G1-Flat-Custom-Play-v0').kwargs['rsl_rl_cfg_entry_point']()
env_cfg.scene.num_envs = 1
env = gym.make('G1-Flat-Custom-Play-v0', cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

ckpt = 'applications/g1_locomotion/results/g1_flat_locomotion/<RUN_DIR>/teacher_final.pt'
sd = torch.load(ckpt, map_location='cpu')
if 'model_state_dict' in sd: sd = sd['model_state_dict']
dims = []
i = 0
while f'actor.{i}.weight' in sd:
    dims.append(sd[f'actor.{i}.weight'].shape[0]); i += 2
agent_cfg.policy.actor_hidden_dims = dims[:-1]
agent_cfg.policy.critic_hidden_dims = dims[:-1]

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device='cuda:0')
runner.load(ckpt)
policy = runner.get_inference_policy(device='cuda:0')

obs = env.get_observations()
raw_env = env.unwrapped
pos_before = raw_env.scene['robot'].data.root_pos_w[0].cpu().numpy().copy()
for i in range(500):
    actions = policy(obs)
    obs, _, _, _ = env.step(actions)
pos_after = raw_env.scene['robot'].data.root_pos_w[0].cpu().numpy()
disp = np.linalg.norm(pos_after[:2] - pos_before[:2])
print(f'Displacement: {disp:.2f}m in 10s, speed: {disp/10:.2f} m/s')
env.close(); simulation_app.close()
"
```

Target: > 4m in 500 steps (speed > 0.4 m/s with command 0.5).

### Visualization

```bash
# Teacher
python applications/g1_locomotion/scripts/play.py \
    --task G1-Flat-Custom-Play-v0 \
    --load_run 2026-06-15_23-20-45 --checkpoint teacher_final.pt

# Student
python applications/g1_locomotion/scripts/play_student.py \
    --student_path results/g1_flat_locomotion/student/student_best.pt
```

Note: Isaac Sim GUI takes 2-3 min to load. Use `Alt+左键` to orbit, switch to "Common" render mode for clarity.

### Key Result

Round 7: speed 0.48m/s, gait_schedule=0.95, feet_clearance=0.96, body height 0.72m.
Checkpoint: `results/g1_flat_locomotion/2026-06-15_23-20-45/teacher_final.pt`

---

## 概念：DR、Privileged Info 与 RMA 蒸馏

理解 Phase 2 前需要知道这三个概念如何串联。

### Domain Randomization (DR)

训练时每个环境的物理参数都不一样，强迫 policy 学会应对各种情况：

```
环境 1: 摩擦=0.3, 质量=35kg, 电机强度=90%
环境 2: 摩擦=1.2, 质量=42kg, 电机强度=110%
环境 3: 摩擦=0.7, 质量=38kg, 电机强度=95%
...1024 个环境，每个都不同
```

为什么需要：仿真和真实世界有差距（sim-to-real gap）。只在"完美"环境训练的策略到真机就废了。DR 让策略在各种条件下都练过。

### Privileged Information（特权信息）

仿真里能拿到，但真机上拿不到的信息：

| 特权信息 | 仿真中 | 真机上 |
|---------|--------|--------|
| 地面摩擦系数 | 直接读物理引擎参数 | 不知道（水泥？冰面？） |
| 机器人质量变化 | 直接读 randomized mass | 不知道（背了包？） |
| 外力推扰 | 知道推力大小方向 | 只感受到加速度 |
| 电机实际增益 | 直接读 kp/kd scale | 电机老化了不知道 |
| 地形高度图 | 直接读 terrain mesh | 没激光雷达看不到 |

有 privileged info 的策略适应性更强（知道地面滑就小步走），但真机没法直接获取。

### RMA（Rapid Motor Adaptation）蒸馏

解决方案：让 student 从**动作历史**中隐式推断环境参数。

```
训练阶段:
  Teacher 直接拿 privileged info → 学会"摩擦低就小步走"
  
部署阶段:
  Student 看过去 50 步的动作历史 → 推断"地面大概很滑"(latent z)
  Student 用 obs + z 做决策 → 效果接近 teacher
```

类比：人走路踩一脚就知道地滑不滑，不需要测量摩擦系数。Student 做的就是"踩一脚就知道"。

### 当前项目状态 vs 完整 RMA

| 条件 | 当前 | 完整 RMA（未来） |
|------|------|-----------------|
| DR | 无（固定物理参数） | 有（摩擦/质量/推力随机化） |
| Teacher 用 privileged info | 不用（actor 只看 obs） | 用（actor 接收 privileged） |
| Student 推断有意义 | 否（所有环境一样，z 无信息） | 是（不同环境 z 不同） |
| 模型更大 | 是（6.4MB vs 2MB） | 值得（换来真机适应性） |

**当前 Phase 2 的定位：** pipeline 验证（证明蒸馏流程能跑通）。真正的价值要等加了 DR + 改为 privileged actor 后才体现。如果只在 sim 中用，直接导出 teacher 更简单更小。

### 为什么 Student 文件更大

- Teacher 2MB = actor [256,128,128]（输入 123 维，参数少）
- Student 6.4MB = AdaptationModule 第一层 6150×256 = **157 万参数**（≈6MB）

因为 AdaptationModule 的输入是 50帧 × 123维 = 6150 维，第一层线性层巨大。这是用空间换自适应能力的设计。

---

## Phase 2: Student Distillation

### Why Distillation IS Needed (Future)

When DR is added and teacher uses privileged info, direct deployment becomes impossible.
The student replaces privileged info with history-based inference:

```
Teacher (training only):
  obs(123) → Actor → action(37)       # actor is obs-only
  obs(123) + privileged → Critic       # critic has extra info

Student (deployment):
  obs_history(50×123=6150) → AdaptationModule → z(16)    # infer env params
  obs_current(123) + z(16) → StudentPolicy → action(37)   # act with latent
```

### Step 1: Collect Teacher Data

```bash
python applications/g1_locomotion/scripts/collect_teacher_data.py \
    --task G1-Flat-Custom-v0 \
    --load_run 2026-06-15_23-20-45 \
    --checkpoint teacher_final.pt \
    --num_envs 1024 \
    --num_steps 500000 \
    --headless
```

Output: `results/g1_flat_locomotion/teacher_distill_data.npz`
- `obs_history`: (449536, 6150) — flattened 50-step history
- `obs_current`: (449536, 123) — current observation
- `teacher_actions`: (449536, 37) — deterministic teacher actions

### Step 2: Train Student (BC)

```bash
python applications/g1_locomotion/student/train_student.py \
    --data results/g1_flat_locomotion/teacher_distill_data.npz \
    --epochs 200 --batch_size 256 --device cuda:0
```

Output: `results/g1_flat_locomotion/student/student_best.pt`
- AdaptationModule: 6150 → 256 → 128 → 16
- StudentPolicy: (123+16) → 256 → 128 → 37
- Training: MSE loss, Adam, early stopping (patience=15)

Expected: val_loss < 0.001 (we achieved 0.000080).

### Step 3: Evaluate Student (Sim2Sim)

```bash
python applications/g1_locomotion/student/evaluate.py \
    --task G1-Flat-Custom-Play-v0 \
    --student_path results/g1_flat_locomotion/student/student_best.pt \
    --episodes 20 --headless
```

Pass criteria: reward degradation < 10% vs teacher, survival > 95%.

### Step 4: Visualize Student

```bash
python applications/g1_locomotion/scripts/play_student.py \
    --student_path results/g1_flat_locomotion/student/student_best.pt
```

---

## Phase 3: ONNX Export + Deployment

### Export

```bash
python applications/g1_locomotion/export/export_onnx.py \
    --student_path results/g1_flat_locomotion/student/student_best.pt
```

Output: `results/g1_flat_locomotion/student/student_g1.onnx`

### What Gets Exported

```
StudentONNXWrapper = AdaptationModule + StudentPolicy (fused)

Inputs:
  obs_history: float32[batch, 6150]   (50 frames × 123 obs_dim)
  obs_current: float32[batch, 123]

Output:
  action: float32[batch, 37]
```

### Benchmark

```bash
python applications/g1_locomotion/export/benchmark.py \
    --model results/g1_flat_locomotion/student/student_g1.onnx
```

Result:
```
G1 Student ONNX Benchmark Results
  Model:    student_g1.onnx
  Size:     6.42 MB
  Latency:
    avg:    0.134 ms
    p95:    0.152 ms
    p99:    0.167 ms
  Budget:   20ms (50Hz control)
  Status:   ✓ PASS
```

### Minimal Inference Loop (Deployment)

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("student_g1.onnx")

# Maintain rolling history buffer
history_length = 50
obs_dim = 123
obs_history = np.zeros((history_length, obs_dim), dtype=np.float32)

while running:
    obs = get_robot_observation()  # shape (123,)

    # Update history (shift left, append new)
    obs_history = np.roll(obs_history, -1, axis=0)
    obs_history[-1] = obs

    # Inference
    action = session.run(None, {
        "obs_history": obs_history.flatten().reshape(1, -1),
        "obs_current": obs.reshape(1, -1),
    })[0].flatten()

    # Apply action to robot (PD position control)
    apply_joint_targets(action)
```

---

## Pitfalls & Lessons Learned

### 1. rsl_rl 3.0 API Changes (TensorDict)

**Symptom:** `ValueError: too many values to unpack`

**Root Cause:** rsl_rl 3.0 changed `env.get_observations()` to return a TensorDict
(not a tuple), and `env.step()` returns 4 values (not 5).

**Fix:**
```python
# Before (wrong):
obs, _ = env.get_observations()
obs, rew, done, trunc, info = env.step(actions)

# After (correct):
obs = env.get_observations()              # TensorDict
obs_td, rew, dones, info = env.step(actions)  # 4 values
policy(obs)  # pass full TensorDict to policy
```

### 2. Network Size Mismatch When Loading Checkpoints

**Symptom:** `RuntimeError: size mismatch for actor.0.weight`

**Root Cause:** PPO config specifies network dims (e.g. [256,128,128]).
If you change config between rounds, loading old checkpoints fails.

**Fix:** Infer dims from checkpoint before creating runner:
```python
sd = torch.load(ckpt, map_location='cpu')
dims = []
i = 0
while f'actor.{i}.weight' in sd:
    dims.append(sd[f'actor.{i}.weight'].shape[0])
    i += 2
agent_cfg.policy.actor_hidden_dims = dims[:-1]
```

### 3. Round 2 Failure — Over-constraining Kills Behavior

**Symptom:** Robot stands still (0.08m displacement in 200 steps).

**Root Cause:** Copied heavy penalty weights from unitree_rl_lab
(orientation=-5, base_height=-10, action_rate=-0.05) without matching their
training scale (50000 iter + 4096 envs). Total penalty dominated velocity reward.

**Lesson:** Match penalty strength to your training budget. Start with light
penalties and increase incrementally. The unitree params work for them because
they train 33x longer than us.

### 4. Gait Schedule Needs Strong Weight

**Symptom:** gait_schedule stuck at 0.45-0.49 even with extended training.

**Root Cause:** Weight 0.5 not enough to compete with velocity tracking reward.
Robot finds it more efficient to take fast tiny steps than follow the clock.

**Fix:** Increased to 1.0 → immediately jumped to 0.95. Sometimes you need
to make the reward competitive, not just present.

### 5. Isaac Sim GUI Issues

**Symptom:** "Robot doesn't move" in GUI, but headless confirms it walks.

**Root Cause:** Multiple issues stacked:
1. Script error → robot spawns but no step loop runs
2. `simulation_app.is_running()` returns False immediately → loop exits
3. Small num_steps → script finishes before you can observe

**Fix:** Use large `--num_steps 50000` (16 min), verify with headless displacement
test first, use GUI only for visual confirmation.

### 6. G1 USD Download Blocks GUI

**Symptom:** Isaac Sim GUI opens but scene is empty for minutes.

**Root Cause:** G1 robot USD (22MB) downloaded synchronously from NVIDIA S3
on first GUI launch. Blocks the UI thread.

**Fix:** Pre-download in setup script or use headless first (which caches it).

---

## Quick Reference

```bash
# === Full Pipeline ===

# 1. Train teacher (25 min)
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 --num_envs 1024 --headless

# 2. Visualize teacher
python applications/g1_locomotion/scripts/play.py \
    --task G1-Flat-Custom-Play-v0 \
    --load_run <run_dir> --checkpoint teacher_final.pt

# 3. Collect distillation data (5-10 min)
python applications/g1_locomotion/scripts/collect_teacher_data.py \
    --task G1-Flat-Custom-v0 --load_run <run_dir> \
    --checkpoint teacher_final.pt --num_steps 500000 --headless

# 4. Train student (10-15 min)
python applications/g1_locomotion/student/train_student.py \
    --data results/g1_flat_locomotion/teacher_distill_data.npz

# 5. Visualize student
python applications/g1_locomotion/scripts/play_student.py \
    --student_path results/g1_flat_locomotion/student/student_best.pt

# 6. Export ONNX (seconds)
python applications/g1_locomotion/export/export_onnx.py \
    --student_path results/g1_flat_locomotion/student/student_best.pt

# 7. Benchmark (seconds)
python applications/g1_locomotion/export/benchmark.py \
    --model results/g1_flat_locomotion/student/student_g1.onnx

# === Monitoring ===
tensorboard --logdir applications/g1_locomotion/results/ --port 6006

# === Environment Setup ===
bash applications/g1_locomotion/setup_env.sh
```

---

# G1 运动控制 Pipeline 指南

训练、蒸馏、部署 G1 人形机器人行走策略的完整指南。

## Pipeline 概览

```
train_teacher.py       collect_teacher_data.py    train_student.py       export_onnx.py
      │                        │                        │                      │
      │ PPO + Isaac Lab        │ 冻结 teacher 推理     │ 行为克隆              │ 融合导出
      │ 7 轮迭代              │ 449k 条数据           │ val_loss=0.00008     │ 精度验证
      ▼                        ▼                        ▼                      ▼
teacher_final.pt ──► teacher_distill_data.npz ──► student_best.pt ──► student_g1.onnx
     (2 MB)                  (9.9 GB)                 (6.4 MB)            (6.4 MB, 0.13ms)
```

## 快速执行

```bash
conda activate env_isaaclab
cd ~/Desktop/myRL/easyRL

# Phase 1: 训练 teacher（25 分钟）
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 --num_envs 1024 --headless

# Phase 2: 采数据 + 训练 student（15-20 分钟）
python applications/g1_locomotion/scripts/collect_teacher_data.py \
    --task G1-Flat-Custom-v0 --load_run <run_dir> \
    --checkpoint teacher_final.pt --num_steps 500000 --headless
python applications/g1_locomotion/student/train_student.py \
    --data results/g1_flat_locomotion/teacher_distill_data.npz

# Phase 3: 导出 ONNX
python applications/g1_locomotion/export/export_onnx.py \
    --student_path results/g1_flat_locomotion/student/student_best.pt
python applications/g1_locomotion/export/benchmark.py \
    --model results/g1_flat_locomotion/student/student_g1.onnx
```

## 关键成果

| 阶段 | 产出 | 指标 |
|------|------|------|
| Teacher (7 轮) | teacher_final.pt | 0.48 m/s，gait=0.95，不摔倒 |
| Student | student_best.pt | val_loss=0.000080 |
| ONNX | student_g1.onnx | 6.42 MB，0.13ms 延迟，精度 1.2e-6 |

## 迭代经验总结

| 教训 | 说明 |
|------|------|
| 最小变量法 | 每次只改一个东西，否则无法归因 |
| 惩罚要渐进 | 一次加太重会杀死行为（Round 2 教训） |
| gait 需要强权重 | 0.5 不够建立步态，1.0 才能突破 |
| 延长训练 ≠ 调参 | Round 6 证明 3000 iter 不如改权重 |
| headless 验证优先 | GUI 不稳定，用位移数据代替可视化 |
| rsl_rl 3.0 API 变了 | TensorDict + 4 返回值，写脚本要注意 |
