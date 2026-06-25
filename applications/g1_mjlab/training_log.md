# G1 mjlab 训练日志

## 项目最终状态

**达成**:
- ✓ 平地稳定行走（60s 不摔）
- ✓ 速度跟踪 0.5-1.0 m/s (12 DOF 锁腰天花板)
- ✓ 姿态自然（上半身不晃、不后仰）
- ✓ 域随机化（摩擦/质量/PD增益/推扰）
- ✓ Teacher-Student 蒸馏成功
- ✓ ONNX 部署模型导出

**未达成**:
- ✗ 2.0 m/s 高速（需要放开腰部 DOF）
- ✗ Rough terrain（需要额外训练轮次）

**最终产出文件**:
| 文件 | 用途 |
|------|------|
| `results_r17/model_2998.pt` | DR Teacher checkpoint |
| `results/student_dr_final.pt` | Student PyTorch 模型 |
| `results/student_dr_policy.onnx` | 部署用 ONNX 模型 |

**ONNX 模型规格**: 输入 `obs_history` [batch, 20, 84] → 输出 `action` [batch, 12]，50 Hz

---

## R1 — 基线验证 (2026-06-23)

15 DOF (12腿+3腰), 基础 reward, 命令 [-0.5, 1.0] m/s。
训练 1445 iter 收敛。训练指标看起来好（reward=98, ep_len=1000），但后续发现**训练指标完全不可信**（见教训）。

---

## R2 — 步态改善 (2026-06-23)

加强 feet_air_time, bipedal_gait, 新增 feet_clearance。
步态指标 (feet_air_time 0.56→0.92) 大幅改善，但可视化发现**腰部疯狂扭动**。

---

## R3 — 参照 Unitree 官方 reward (2026-06-23)

按 Unitree 官方 G1 比例重写：velocity 1:2, joint_deviation, termination=-200。
结果：动作平滑了，但 feet_air_time threshold=0.4 导致 shuffle 回归。

---

## R4 — threshold 修复 (2026-06-23)

只改 feet_air_time threshold 0.4→0.15。步态恢复 + 保持 R3 的平滑。

---

## R5 — 提速 2.0 m/s (2026-06-23)

从 R4 续训，命令范围扩到 [-0.5, 2.0]。适应成功。

---

## R6 — 加 DR (2026-06-23)

从 R5 续训，加摩擦/质量/PD增益随机化 + 强推扰。速度测试 2.0 m/s 跟踪 95%。

---

## R7-R8 — Rough terrain 尝试 (2026-06-23)

从 R6 续训加 rough terrain。存活率 82% 但续训无改善。诊断为 `no_contact_penalty` 对 rough terrain 有害。

---

## R9 — 定稿配置从零训 (2026-06-23)

去掉 no_contact_penalty，terrain curriculum，DR 全开。3000 iter。
结果 rough terrain 上 reward 在 10-14 震荡（curriculum 加难度），平地步态严重退化（弓腰走路）。

---

## ⚠️ 重大教训 — 评估脚本揭示 R1-R9 全部失败 (2026-06-23)

用 `evaluate.py`（单 env，无 reset，60s 持续走）测试所有 checkpoint：
- R4（训练指标最好）：**1.4 秒摔倒，60 秒内摔 28 次**
- R1：**60 秒不摔，速度跟踪 95%**（唯一有效的）

**根因**：训练 reward/ep_len 是 2048 env 均值 + 自动 reset，完全不代表真实行走能力。
**教训**：每轮必须跑 evaluate.py + 可视化，不能只看训练数字。

R1 之后的 R2-R9 所有"改进"（termination=-200, 各种权重变化）都**破坏了**原本能走的策略。

---

## R10 — 姿态修复迭代 (2026-06-23~24)

以 R1 为基础修复后仰 + 左右摇晃。每次用 evaluate.py 验证。

- **R10a**: upright -2→-5 → PASS 但后仰+摇晃
- **R10b**: 速度权重 2→1, upright -10 → PASS 后仰改善，摇晃仍在
- **R10c**: hip/waist deviation -0.3 → FAIL(速度不达标)
- **R10d**: feet_air_time 0.3, lin_vel_z -1.0 → FAIL(速度太低)

**核心发现**: 上半身摇晃根源是腰部 3 DOF 在 action space 里。

---

## R11 — 移除腰部 DOF, 12 DOF 纯腿 (2026-06-24)

Action space 15→12 DOF，腰部 PD 锁定。
**评估**: 0.5→131%, 1.0→77%, 60s 不摔。**姿态好看**——上半身不再晃动。
**结论**: 12 DOF 锁腰的最优策略。

---

## R12-R16 — 尝试在 12 DOF 下提速 (2026-06-24)

各种调权重、改命令范围，均无法突破 ~0.8 m/s 天花板。
**结论**: 12 DOF 锁腰物理上限 ~1.0 m/s（对标 Unitree 官方 12 DOF 配置）。接受 R11 为最优。

---

## R17 — DR 训练 (2026-06-24)

从 R11 续训，加摩擦 [0.5,1.3] + 质量 ±15% + PD ±15% + 推扰 ±0.5 m/s。
**评估**: 0.5→156%, 1.0→83%。DR 后性能反而提升。

---

## Phase 2-4 最终蒸馏 (2026-06-25)

Teacher: R17 → 采 480K transitions → Student 训练 (val_loss=0.000545) → ONNX 导出

⚠️ **关键修复**: 去掉 Student 的 `nn.Tanh()` 输出层（Teacher action 范围 [-5, +8]，Tanh 截断 45% action）

**Student 评估**: 0.5→99%, 1.0→94%, 60s 不摔 ✓

---
