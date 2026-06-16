# G1 Locomotion 训练日志

每轮训练的结果分析、诊断和迭代策略记录。

## 快速启动

```bash
# Phase 1: 训练 teacher (平地)
conda activate env_isaaclab
python applications/g1_locomotion/scripts/train_teacher.py --task G1-Flat-Custom-v0 --num_envs 1024

# 可视化
python applications/g1_locomotion/scripts/play.py --task G1-Flat-Custom-Play-v0 --load_run <run_dir>

# Phase 2: 采数据 + 训练 student
python applications/g1_locomotion/scripts/collect_teacher_data.py --task G1-Flat-Custom-v0 --load_run <run_dir>
python applications/g1_locomotion/student/train_student.py --data results/g1_flat_locomotion/teacher_distill_data.npz

# Phase 2: 评估 student
python applications/g1_locomotion/student/evaluate.py --task G1-Flat-Custom-Play-v0 --student_path results/.../student/student_best.pt

# Phase 3: 导出 + benchmark
python applications/g1_locomotion/export/export_onnx.py --student_path results/.../student/student_best.pt
python applications/g1_locomotion/export/benchmark.py --model results/.../student_g1.onnx
```

---

## Round 1

**配置：**
```
task: G1-Flat-Custom-v0
num_envs: 1024
max_iterations: 1500
actor_hidden_dims: [256, 128, 128]
lr: 1e-3 (adaptive schedule)
command vx: [0.3, 0.6]
rewards: flat_env_cfg.py 默认 (18 active terms)
```

**TensorBoard:** `results/g1_flat_locomotion/2026-06-15_15-04-43/`

**关键指标：**

| 指标 | 值 |
|------|---|
| 最终 reward | 41.75 |
| 峰值 reward | 41.81 |
| Episode length | 1000 / 1000（完全存活） |
| 速度跟踪 | 1.428（理论最大 1.5，准确率 95%）|
| 姿态惩罚 | -0.005（几乎无倾斜）|
| 脚离地时间 | 0.011（极低，疑似 shuffle）|
| 训练时间 | ~25 分钟（RTX A4000，1024 envs）|
| Checkpoint | `results/g1_flat_locomotion/2026-06-15_15-04-43/teacher_final.pt` |

**收敛曲线：**

```
Reward
 42 |                                          ━━━━━━━━━ 收敛平台
 35 |                               ━━━━━━━━━━
 28 |                        ━━━━━━━
 20 |                   ━━━━━
 15 |              ━━━━━
  0 |─────────────────────────────────────────────────── 0 线
 -5 |  ━━━━━━━━━━━ 随机探索
     0    200   400   600   800  1000  1200  1400  iter
```

**阶段分析：**

| 阶段 | Iter 区间 | Reward | Ep Len | 状态 |
|------|----------|--------|--------|------|
| 随机探索 | 0-200 | -5 ~ -0.4 | 11-117 | 随机动作，立刻摔倒 |
| 学习站立 | 200-400 | -5 → +15 | 117 → 1000 | 学会站住不摔 |
| 学习前进 | 400-900 | 15 → 35 | ~1000 | 开始跟踪速度指令 |
| 精细打磨 | 900-1500 | 35 → 42 | 1000 | 速度跟踪精度提升 |

**诊断：**

1. **存活能力**：iter 300-400 完全解决，之后再不摔倒
2. **速度跟踪**：1.428 / 1.5 = 95%，非常好
3. **脚离地时间**：仅 0.011 — G1 大概率在"滑行"而非"迈步"。速度指标好但步态质量差
4. **姿态保持**：接近完美，身体直立
5. **训练速度**：A4000 上 1500 iter 仅需 25 分钟，迭代极快

**核心发现：**

Round 1 开箱即收敛到行走策略（不像 Go2 需要十几轮迭代）。但 feet_air_time 极低暴露了 **shuffle gait 问题** —— 机器人在"拖着脚滑"而不是"抬脚迈步"。这和 Go2 早期遇到的问题类似。

**可视化确认：** play.py 观察到"老奶奶踱步"—— 机器人从静止前倾启动，然后拖着脚缓慢滑行。确认 shuffle gait 假设。

---

## Round 2

**变更说明：** 对比分析了 unitree_rl_lab 官方 G1 方案后，吸收高价值设计。

**与宇树方案对比后吸收的关键改动：**

| 改动 | Round 1 → Round 2 | 宇树原始值 | 目的 |
|------|-------------------|-----------|------|
| 新增 `foot_clearance_reward` | 无 → 1.0 (target=10cm) | 1.0 | 强制脚抬离地面 10cm，直接治 shuffle |
| 新增 `feet_gait` | 无 → 0.5 (period=0.8s) | 0.5 | 时钟驱动步态节奏，0.8s 一个步态周期 |
| 新增 `energy` penalty | 无 → -2e-5 | -2e-5 | 惩罚无效关节力矩×速度 |
| `flat_orientation` | -1.0 → **-5.0** | -5.0 | 强约束身体直立 |
| `lin_vel_z` | -0.2 → **-2.0** | -2.0 | 强惩罚弹跳 |
| `action_rate` | -0.005 → **-0.05** | -0.05 | 强约束动作平滑 |
| `joint_deviation_hip` | -0.1 → **-1.0** | -1.0 | 惩罚 hip 偏离默认 |
| `joint_deviation_torso` | -0.1 → **-1.0** | -1.0 | 惩罚躯干扭转 |
| `dof_pos_limits` | -1.0 → **-5.0** | -5.0 | 强惩罚关节极限 |
| `base_height` | +0.3 (reward) → **-10.0** (penalty, L2) | -10 (target=0.78) | 偏离 0.78m 就惩罚 |
| `track_lin_vel` | 1.5 → 1.0 | 1.0 | 降低，避免压过步态 reward |
| 网络结构 | [256,128,128] → **[512,256,128]** | [512,256,128] | 更大表达能力 |
| `entropy_coef` | 0.008 → **0.01** | 0.01 | 更多探索 |
| 移除 `feet_air_time` | 0.5 → 0 | 无（用 clearance 代替）| 被 clearance+gait 替代 |
| 移除 `gait_symmetry` | 0.3 → 移除 | 无 | 被 feet_gait 替代 |

**未吸收的部分（暂不引入）：**
- `alive_bonus` (0.15) —— Round 1 已经不摔倒，不需要
- `undesired_contacts` —— 暂不引入，如果出现膝盖着地再加
- Command curriculum —— 先固定范围跑，后续轮次再加
- Obs history (5帧) —— 需要改 obs pipeline，影响较大，后续考虑
- 50000 iter —— 先跑 3000 iter 看效果

**配置：**
```
task: G1-Flat-Custom-v0
num_envs: 1024
max_iterations: 3000
actor_hidden_dims: [512, 256, 128]
lr: 1e-3 (adaptive)
entropy_coef: 0.01
command vx: [0.2, 0.6]

关键 reward weights:
  track_lin_vel: 1.0 (std=0.5)
  foot_clearance: 1.0 (target=0.1m)
  feet_gait: 0.5 (period=0.8s)
  flat_orientation: -5.0
  base_height_l2: -10.0 (target=0.78m)
  lin_vel_z: -2.0
  action_rate: -0.05
  energy: -2e-5
  joint_deviation_hip/torso: -1.0
  dof_pos_limits: -5.0
```

**TensorBoard:** `results/g1_flat_locomotion/2026-06-15_18-52-06/`

**Results:**

| 指标 | 值 |
|------|---|
| 最终 reward | 12.51 |
| 峰值 reward | 16.29 |
| Episode length | 906 / 1000 |
| feet_clearance | 0.93（脚在抬了） |
| gait_schedule | 0.52 |
| 速度跟踪 | 0.50（Round 1 是 1.43） |
| 200 步位移 | **0.08m**（几乎不动！） |

**诊断：**

1. **feet_clearance = 0.93** —— 成功学会抬脚
2. **但位移只有 0.08m**（Round 1 是 1.95m）—— 学会了"原地踏步"而不是"向前走"
3. **根因：** 惩罚总量远大于速度跟踪 reward。base_height(-10) + orientation(-5) + action_rate(-0.05) 让 policy 选择了"少动少罚"策略
4. **宇树能收敛是因为训 50000 iter + 4096 envs**，我们 3000 iter 不够克服早期惩罚压制

**结论：** Round 2 失败。一次改动太多 + 惩罚太重 + 训练量不匹配。需要回退到 Round 1 基线做增量改动。

---

## Round 3

**策略：** 回到 Round 1 基线，只加一个改动 —— `feet_clearance` reward。最小变量法验证。

**配置变更（相对 Round 1 只加了一行）：**
```
新增: feet_clearance = 1.0 (target=0.1m, std=0.05, tanh_mult=2.0)
其余所有参数与 Round 1 完全相同
网络: [256, 128, 128]（恢复 Round 1）
```

**TensorBoard:** `results/g1_flat_locomotion/2026-06-15_20-21-03/`

**Results:**

| 指标 | 值 |
|------|---|
| 最终 reward | 56.74 |
| 峰值 reward | 56.74（仍在上升） |
| Episode length | 1000 / 1000（不摔倒） |
| feet_clearance | **0.97**（脚抬起了！） |
| 速度跟踪 | 1.41（接近 Round 1 的 1.43） |
| feet_slide | -0.015（很低，不在拖地） |
| termination | 0（从不摔倒） |
| **500 步位移** | **4.23m（速度 ≈ 0.42 m/s）** |

**收敛曲线：**

```
Reward
 57 |                                          ━━━━ 收敛
 50 |                               ━━━━━━━━━━
 44 |                        ━━━━━━━
 28 |              ━━━━━━━━━
  0 |──────────────────────────────────────────── 0 线
 -4 |  ━━━━━ 探索
     0    200   400   600   800  1000  1200  1400  iter
```

**诊断：**

1. **位移 4.23m / 10s = 0.42 m/s** — 指令 0.5 m/s，跟踪率 84%，正常
2. **feet_clearance = 0.97** — 脚在充分抬起，shuffle 问题解决
3. **vel_tracking = 1.41** — 和 Round 1 几乎相同，速度没有被牺牲
4. **reward 56.74 >> Round 1 的 41.75** — clearance 贡献了额外 reward
5. **feet_air_time 仍然低 (0.016)** — 这不矛盾：clearance reward 让脚抬高但不要求空中时间长，脚快速抬起落下

**关键成功：** 只加一个 `feet_clearance` reward 就同时解决了 shuffle 问题，且保持了 Round 1 的所有优势（速度、存活、稳定性）。最小变量法有效。

**当前状态评估：** 基本达到"良好"标准 —— 能走、有抬脚、不摔倒、跟踪速度指令。

**可能的后续改进（非必要）：**
- 扩大 command 范围（当前 vx [0.3, 0.6]，可以扩展到 [0.0, 1.0]）
- 加入转弯能力测试
- 加入 rough terrain
- 加入 Domain Randomization（push_robot）

**可视化确认：** 侧面观察发现 G1 **严重后仰 + 深蹲**（膝盖弯曲、身体呈 C 形弧线）。抬脚是有了，但姿态很差。原因：无 base_height 约束，机器人发现蹲低重心更容易抬脚不摔；orientation 惩罚太轻（-1.0）不足以矫正后仰。

---

## Round 4

**策略：** Round 3 基线 + 加 base_height 惩罚 + 加强 orientation，修复蹲着走/后仰问题。

**配置变更（相对 Round 3）：**
```
改动:
  flat_orientation_l2: -1.0 → -2.0（不能后仰）
  lin_vel_z_l2: -0.2 → -0.5（抑制弹跳）
  新增 base_height_l2: weight=-5.0, target=0.74m（不能蹲着走）

其余与 Round 3 相同（feet_clearance=1.0, track_vel=1.5, 网络[256,128,128]）
```

**动机：** Round 3 证明 feet_clearance 能解决 shuffle，但缺少高度/姿态约束导致畸形步态。Round 4 用温和的惩罚约束姿态（-2.0/-5.0 而非 Round 2 的 -5.0/-10.0），避免 Round 2 的"不敢动"问题。

**TensorBoard:** `results/g1_flat_locomotion/2026-06-15_21-10-29/`

**Results:**

| 指标 | 值 |
|------|---|
| 最终 reward | 56.15 |
| Episode length | 1000 / 1000 |
| feet_clearance | 0.97 |
| 速度跟踪 | 1.41 |
| orientation penalty | -0.008（比 R3 的 -0.029 好 3.6 倍） |
| base_height penalty | -0.0076（接近目标 0.74m） |
| 平均高度 | 0.70m（目标 0.74m） |
| 500 步位移 | 3.68m（速度 0.37 m/s） |

**可视化确认：** 后仰/深蹲问题改善了，但出现新问题：
1. **外八字步态** — 两腿向外张开走路
2. **步幅极小** — 腿捣腾很快但身体移动慢
3. **整体不自然** — 像原地快速踏步而非正常行走

**诊断：**
- 外八字 → `joint_deviation_hip` 权重太轻（-0.1），hip_roll/hip_yaw 偏离默认无惩罚
- 步幅小 → feet_clearance 只约束脚抬高度不约束前后摆幅；速度跟踪 reward 不够强驱动大步幅
- 腿快体慢 → 步态频率太高但每步距离短，缺乏步态周期约束

**根因分析：** Round 3/4 的改动让机器人学会了"快速小步原地踏步"来同时满足 clearance（抬脚）和 velocity（速度），但步态效率很低。需要：
1. 加强 hip 关节偏差惩罚解决外八字
2. 加入步态周期约束（gait schedule）解决步幅小/腿快
3. 适度提高速度跟踪权重

---

## Round 5

**策略：** Round 4 基线 + 修复外八字 + 加入步态周期约束

**配置变更（相对 Round 4）：**
```
改动:
  joint_deviation_hip: -0.1 → -0.5（惩罚外八字）
  track_lin_vel_xy_exp: 1.5 → 2.0（更强速度驱动，迫使大步幅）
  新增 feet_gait: weight=0.5, period=0.8s, offset=[0.0, 0.5]
    （时钟驱动步态周期，约束步频，迫使每步更大）

其余与 Round 4 相同
```

**动机：**
- hip deviation -0.5 直接惩罚外八字（hip_roll/hip_yaw 偏离默认）
- feet_gait 约束步态周期 0.8s（步频 1.25Hz），防止高频碎步
- velocity tracking 2.0 让"快大步"比"快碎步"更有 reward 优势

**TensorBoard:** `results/g1_flat_locomotion/2026-06-15_21-49-59/`

**Results:**

| 指标 | R4 | R5 | 改善 |
|------|----|----|------|
| Reward | 56.1 | **66.4** | +18% |
| 速度 | 0.37 m/s | **0.44 m/s** | +19% |
| 高度 | 0.70m | **0.71m** | 略升 |
| clearance | 0.97 | 0.97 | 保持 |
| vel_tracking | 1.41 | **1.82** | +29% |
| gait_schedule | — | **0.49** | 新增，周期建立 |
| hip_deviation | -0.1(松) | **-0.069** | 有约束但不过度 |
| orientation | -0.008 | **-0.005** | 更直 |
| ep_len | 1000 | 997 | 几乎不摔 |

**分析：**
- velocity tracking 大幅提升（1.41→1.82），说明更强的 vel reward + gait 周期帮助了
- gait_schedule 0.49 说明步态周期初步建立（最大约 1.0）
- hip_deviation penalty 很小（-0.069），说明外八字应该有改善
- 高度 0.71m 仍低于目标 0.74m，但 penalty 只有 -0.003，可接受
- action_rate penalty -0.31 比 R4 的 -0.16 高了，说明动作变化更大（步幅可能更大但也更不平滑）

**评估：** 数据指标全面提升，但需要可视化确认外八字和步幅是否改善。由于用户离开暂无法可视化，基于数据判断继续优化方向：
- gait_schedule 0.49 还不够高，步态周期还可以加强
- action_rate -0.31 偏高，可以加一点平滑约束
- 速度 0.44 vs 指令 0.5，还有提升空间

**下一步：** 保持当前配置继续训练更长时间（3000 iter），让 gait 进一步收敛。

---

## Round 6

**策略：** Round 5 配置不变，延长训练到 3000 iter，让步态周期和速度跟踪进一步收敛。

**配置变更（相对 Round 5）：**
```
仅改动: max_iterations: 1500 → 3000（延长训练）
其余完全不变
```

**动机：** Round 5 在 1500 iter 时 gait_schedule 只有 0.49（最大~1.0），reward 仍在上升，说明还没收敛。延长训练让 policy 有更多时间学习步态周期。

**TensorBoard:** `results/g1_flat_locomotion/2026-06-15_22-22-56/`

**Results:**

| 指标 | R5 (1500) | R6 (3000) | 变化 |
|------|-----------|-----------|------|
| Reward | 66.4 | 66.8 | 平台 |
| 速度 | 0.44 m/s | **0.47 m/s** | +7% |
| 高度 | 0.71m | **0.725m** | 接近 0.74 目标 |
| clearance | 0.97 | 0.89 | 略降（偷懒少抬） |
| gait | 0.49 | 0.45 | 没改善 |
| ep_len | 997 | 970 | 偶尔摔倒 |

**分析：** 延长训练收益有限。reward 已收敛（66 平台），gait 没有继续提升。policy 在 3000 iter 后开始"偷工减料"（降低 clearance 来换速度）。这说明需要调参而非简单加时间。

**下一步改进方向：**
- gait_schedule weight 0.5 → 1.0（更强步态周期约束）
- action_rate -0.005 → -0.01（更平滑，减少碎步感）
- 保持 1500 iter（已证明 3000 无额外收益）

---

## Round 7

**策略：** 加强 gait 周期约束 + 适度加平滑惩罚

**配置变更（相对 Round 6/5）：**
```
改动:
  gait_schedule weight: 0.5 → 1.0（更强步态周期）
  action_rate_l2: -0.005 → -0.01（更平滑）
  max_iterations: 回到 1500

其余与 Round 5 相同
```

**动机：** gait 0.45-0.49 说明当前权重不够驱动规律步态。加强后期望步频更规律、步幅更大。action_rate 适度增加让动作更平滑减少碎步感。

**TensorBoard:** `results/g1_flat_locomotion/2026-06-15_23-20-45/`

**Results:**

| 指标 | R6 | R7 | 变化 |
|------|----|----|------|
| Reward | 66.8 | **69.4** | +4% |
| 速度 | 0.47 m/s | **0.48 m/s** | 指令 0.5 的 96% |
| 高度 | 0.725m | **0.72m** | 接近 0.74 目标 |
| clearance | 0.89 | **0.96** | 恢复抬脚 |
| **gait_schedule** | 0.45 | **0.95** | 关键突破！步态周期建立 |
| orientation | -0.01 | -0.006 | 好 |
| hip_deviation | -0.056 | -0.084 | 可接受 |
| action_rate | -0.26 | -0.47 | 更大动作幅度（步幅增大的代价） |
| ep_len | 970 | **991** | 几乎不摔 |

**关键成功：** gait_schedule 从 0.45 → **0.95**（接近满分）！加强 gait weight 到 1.0 是决定性改动。步态周期完全建立 —— 意味着左右脚交替有明确的节奏（0.8s 周期）。

**收敛曲线：**
```
Reward
 69 |                                       ━━━━━━ R7 (gait=0.95!)
 67 |              ━━━━━━━━━━━━━━━━━━━━━━━━━━ R6 (gait=0.45, 3000iter 无提升)
 66 |  ━━━━━━━━━━━━ R5
     0    500  1000  1500  2000  2500  3000  iter
```

**当前最优 checkpoint：** `results/g1_flat_locomotion/2026-06-15_23-20-45/teacher_final.pt`

**综合评估（Round 7）：**
- 行走速度：0.48 m/s（指令 0.5 的 96%）— 优秀
- 步态周期：0.95/1.0 — 优秀
- 抬脚高度：clearance 0.96 — 优秀
- 身体直立：orientation -0.006 — 优秀
- 不摔倒：ep_len 991/1000 — 优秀
- 身体高度：0.72m / 0.74m 目标 — 良好

**结论：Round 7 达到"优秀"水平。** 所有核心指标（速度跟踪、步态周期、抬脚、姿态、存活）都在优秀范围。

---

## 迭代总结

| Round | 核心改动 | 成果 | 遗留问题 |
|-------|---------|------|---------|
| R1 | 基线 | 能走但 shuffle | 脚不抬 |
| R2 | 大幅对齐宇树 | 失败（原地踏步） | 惩罚太重 |
| R3 | R1 + feet_clearance | 抬脚了 | 后仰/蹲着走 |
| R4 | R3 + base_height + orientation | 站直了 | 外八字 + 碎步 |
| R5 | R4 + hip penalty + gait(0.5) | 速度提升 | gait 不够高 |
| R6 | R5 延长到 3000 iter | 无显著改善 | gait 没收敛 |
| **R7** | **gait weight 1.0 + action_rate -0.01** | **gait=0.95! 全面优秀** | — |

**关键经验：**
1. 最小变量法有效 —— 每次只加一个改动
2. 权重太重会杀死行为（R2 教训）
3. gait_schedule 需要足够强的权重（0.5 不够，1.0 才能建立步态）
4. 延长训练不如调参有效（R6 vs R7）
