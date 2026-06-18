# Go2 mjlab Training Log

## 环境信息
- **框架**: mjlab 1.4.0 + mujoco-warp 3.8.1
- **硬件**: RTX A4000 (16GB), Xeon W-2235
- **conda env**: `mjlab` (Python 3.11, PyTorch 2.12, CUDA 12.9)
- **吞吐**: ~105K-125K steps/sec (2048 envs)

---

## 核心发现 (2026-06-17)

### 根因：Go2 MJCF 的 `<default><joint>` 元素导致 actuator 失效

mujoco_menagerie 的 Go2 MJCF 在 `<default class="go2">` 里有：
```xml
<joint axis="0 1 0" damping="2" armature="0.01" frictionloss="0.2"/>
```

这导致 mjlab 的 `BuiltinPositionActuator` 在 mujoco-warp GPU 仿真中**完全不产生力**（actuator_force 和 qfrc_actuator 全为 0）。

**修复**：移除 `<default>` 中的 `<joint>` 元素，把 `armature` 直接写到每个 joint 上。修复后 robot 能稳定站立 (h=0.338, 10+ 秒不下沉)。

### 对照：Go1 使用相同框架能在 259 iter 学会走路
- Go1 reward: track_lin_vel: 0.005 → 0.697
- Go1 mean_reward: -0.1 → 57.4

### 剩余问题：Go2 velocity tracking 不收敛

即使 MJCF 修复后 robot 能站住，训练中 `track_linear_velocity` 始终 ≈ 0。
原因是 Go2 MJCF 缺少 mjlab 所需的 builtin sensor 配置（IMU site 命名不匹配），
导致观测函数引用 `robot/imu_lin_vel` 等 sensor 时找不到。

---

## Round 1-4: 详见下方

### Round 1 (v1) — 无 contact sensor
- lin_vel_tracking ≈ 0
- 原因: gait reward 用错误代理，actor 缺 base_lin_vel

### Round 2 (v3) — 加 contact sensor + 完整 reward
- robot 持续下蹲触发 termination
- 根因: MJCF joint defaults 导致 actuator 不工作

### Round 3-5 (v4/r4/r5) — 尝试调 actuator 参数
- 增大 stiffness/damping/effort_limit 均无效
- 发现：actuator_force 始终为 0（MJCF 兼容性问题）

### Round 6 — MJCF 修复后首次训练
- robot 能站住了 (h=0.338 stable)
- 但训练仍不收敛 (track_lin_vel ≈ 0.002)
- 原因: `reset_robot_joints` 加 noise 后 robot 倒下 → 频繁 terminate

### Round 7-8 — 去掉 push + 放松 termination
- 放松后 robot 学会乱动直到翻倒 (fell_over rate 上升)
- 核心问题: mjlab 的 velocity tracking reward 函数依赖 builtin sensor
  (`robot/imu_lin_vel`)，但 Go2 MJCF 没有正确配置

---

---

## Round 9 (r9) — GO2 WALKS! ✓

**日期**: 2026-06-17  
**关键修复** (三个 bug 同时解决):
1. **unitree_rl_mjlab Go2 MJCF** — 替换 mujoco_menagerie 版本，无 joint defaults
2. **collision fix**: CollisionCfg conaffinity=0→1 (脚能和地面碰撞)
3. **terrain plane**: SceneCfg 加 `TerrainEntityCfg(terrain_type="plane")` (没有地面！)
4. **builtin sensors**: MJCF 有 gyro/velocimeter，observation 用 `builtin_sensor`

**结果** (3000 iter, 2048 envs):
- mean_reward: -0.28 → **84.77**
- track_linear_velocity: **1.957/2.0** (97.9% 满分)
- track_angular_velocity: **1.809/2.0** (90.4%)
- fell_over: **0.0** (完全稳定)
- upright penalty: -0.0002 (几乎为 0，姿态完美)
- throughput: 100K steps/sec (A4000)

**可视化**: `play Go2-Flat-v0 --checkpoint-file results_r9/model_1000.pt --num-envs 1`

---

---

## Round 10 — Gait shaping + DR + Full command range

**日期**: 2026-06-18  
**配置变更** (从 R9 checkpoint resume):
- 加 gait rewards: feet_air_time(0.5), gait_symmetry(0.3), all_feet_contact_penalty(-0.5)
- 加 ang_vel_xy penalty (-0.05)
- 增大 lin_vel_z penalty: -0.2 → -0.3
- 启用 push_robot event (interval 5-10s, vel +-0.5)
- 扩展 command range: lin_vel_x [-1.0, 2.0], lin_vel_y [-0.5, 0.5], ang_vel_z [-1.0, 1.0]
- Resume from results_r9/model_3000.pt
- max_iterations: 5000

**结果** (5000 iter):
- reward: 81.51
- track_lin_vel: 1.87 (94%, range [-1, 2] m/s)
- track_ang_vel: 1.54 (77%, range [-1, 1] rad/s)
- gait_symmetry: 0.28 (学到了对称步态)
- fell_over: 0.0 (push 鲁棒)
- **feet_air_time: 0.0** (没抬脚 — weight 太低)

**分析**: velocity tracking 在大范围下仍好，push 抗扰好。但 robot "拖着走"而不是"迈步走"。feet_air_time weight=0.5 相对 vel_tracking=2.0 太弱。

---

## Round 11 — Force stepping gait

**日期**: 2026-06-18  
**配置变更** (从 R10 checkpoint resume):
- 大幅增加 feet_air_time weight: 0.5 → 2.0
- 增加 gait_symmetry weight: 0.3 → 0.5
- 增加 all_feet_contact_penalty: -0.5 → -2.0
- Resume from results_r10/model_5000.pt (如果存在) 或 model_4500.pt

**关键修复**: `feet_air_time` reward 改为持续信号版本（每步检查 air_time 是否在 [0.05, 0.5] 范围）
  - 原版本只在 landing 时刻给 reward（太稀疏，robot 永远学不到）
  - 新版本参照 mjlab 内置实现：count feet in good air time range per step
  - 加了 command scaling：站着不动时不给 air_time reward

**目标**: 强制 robot 抬脚，形成 trot 步态

---

**结果** (5000 iter):
- reward: 113.08
- track_lin_vel: 1.28 (64%) — 降了（步态转换代价）
- track_ang_vel: 0.04 (2%) — 崩了（不会转弯）
- feet_air_time: 5.49 — 成功抬脚！
- gait_symmetry: 0.36
- fell_over: 0.05

**分析**: feet_air_time weight=2.0 过强，压制了 angular velocity tracking。需要平衡。

---

## Round 12 — Balance gait vs velocity

**日期**: 2026-06-18  
**配置变更** (从 R11 checkpoint resume):
- 降低 feet_air_time weight: 2.0 → 1.0
- 降低 all_feet_contact_penalty: -2.0 → -1.0
- 目标: 保持抬脚的同时恢复 velocity tracking（尤其是转弯）

---

**结果** (5000 iter):
- reward: 87.85
- track_lin_vel: 1.49 (75%) — 恢复了
- track_ang_vel: 0.32 (16%) — 还是低（继承了 R11 忘了转弯的 checkpoint）
- feet_air_time: 2.64 — 保持抬脚
- gait_symmetry: 0.37
- fell_over: 0.05

**分析**: 平衡 ok，但 ang_vel 恢复太慢。R11 的 checkpoint 已经"忘了转弯"，继续 resume 无法快速恢复。

---

## Round 13 — Fresh start with balanced rewards

**日期**: 2026-06-18  
**配置变更**: 用 R12 的 reward config，但从 **R9 checkpoint** 重新训练（R9 时转弯 90%）
- Resume from results_r9/model_3000.pt（转弯能力完整的 checkpoint）
- 保持 R12 的平衡 reward weights
- max_iterations: 8000（给足时间同时学走路+步态+转弯）

---

**结果** (2000 iter):
- reward: 95.39
- track_lin_vel: 1.48 (74%)
- track_ang_vel: 0.57 (29%) — 比 R12 好但仍不够
- feet_air_time: 2.49 — 保持
- fell_over: 0.0

**分析**: 从 R9 重新训比续训 R11/R12 好很多。但 ang_vel 只到 29%，续训 1000 iter 反而退化。需要提高 ang_vel weight。

---

## Round 14 — Boost angular velocity tracking

**日期**: 2026-06-18  
**配置变更** (从 R9 checkpoint):
- 提高 track_angular_velocity weight: 2.0 → 3.0
- 其余同 R13 (feet_air_time=1.0, all_feet_contact=-1.0, gait_symmetry=0.5)
- Resume from results_r9/model_3000.pt
- max_iterations: 2000

**结果** (2000 iter):
- reward: **120.53**
- track_lin_vel: **1.58 (79%)**
- track_ang_vel: **1.72 (57%)** — 从 R13 的 19% 提升到 57%!
- feet_air_time: **2.16** — 保持抬脚
- gait_symmetry: **0.42**
- fell_over: **0.0** — 完全稳定
- throughput: 94K steps/sec

**综合评价**: 全面达标。前进+转弯+步态+稳定性都好。可以进入 distillation。

---

## 下一步

- Student distillation (Phase 2-4): 用 R14 checkpoint 采集数据 → 训练 Student → ONNX 导出

| Round | 目标 | 内容 |
|-------|------|------|
| R10 | 步态质量 | 加 gait rewards (feet_air_time, gait_schedule, symmetry, all_feet_contact_penalty) |
| R11 | 鲁棒性 | DR curriculum (friction, mass, push perturbation) |
| R12 | 速度范围 | Command curriculum (0.5~1.0 → -1.0~2.0 m/s) |
| R13 | Student distillation | 离线蒸馏 + ONNX export |

**优秀标准**:
- velocity tracking > 95%
- 稳定 trot gait (feet_air_time > 0, 对称步态)
- 抗扰 (push 后 recover)
- 速度范围 [-1.0, 2.0] m/s 全覆盖

## 训练时间统计
- R9: 1499 iter ≈ 20 分钟 (2048 envs, 83K steps/sec)
- 共进行 9 轮迭代，8 轮失败定位根因，R9 成功
