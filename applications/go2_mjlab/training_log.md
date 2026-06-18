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

## 下一步 (R10+)

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
