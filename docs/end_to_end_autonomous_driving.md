# End-to-End Autonomous Driving: From Sensors to Trajectory

## Overview

```
Sensor raw input → Single neural network → Future trajectory point sequence
```

Replaces the traditional Perception → Prediction → Planning modular pipeline with a single network that learns the mapping from sensors to trajectory end-to-end.

## Typical Architecture

```
Camera / LiDAR
    ↓
Backbone (BEV Encoder)              ← Extract bird's-eye-view features
    ↓
Temporal Fusion                     ← Fuse historical frames
    ↓
Query-based Decoder (Transformer)   ← Learn scene interactions
    ↓
Trajectory Head                     ← Output future T steps (x, y, heading)
```

## Representative Works

| Method | Team | Key Feature |
|--------|------|-------------|
| UniAD | Shanghai AI Lab | Unify perception/prediction/planning in one Transformer, joint multi-task training |
| VAD | Horizon Robotics | Vectorized scene representation, no HD Map dependency |
| GameFormer | NVIDIA | Game-theoretic interaction modeling, multi-agent joint prediction |
| PARA-Drive | Waabi | Parallel multi-task heads, auxiliary tasks improve planning |
| Diffusion Planner | — | Diffusion model generates multi-modal trajectories |

## Key Technical Components

### 1. BEV Representation (Bird's-Eye View)

- Project multi-camera / LiDAR into unified BEV space
- Methods: LSS / BEVFormer / BEVDet
- Planning naturally needs trajectories in BEV coordinates

### 2. Transformer Decoder

- Ego query represents ego-vehicle intent
- Cross-attention attends to scene features
- Outputs multi-modal candidate trajectories

### 3. Multi-Modal Trajectory Output

- Not a single trajectory, but K candidates (e.g., 6) + confidence score for each
- Select best or weighted fusion
- Methods: GMM, Anchor-based, Diffusion

### 4. Auxiliary Task Joint Training

- Detection, lane lines, occupancy, motion prediction
- Not for output — forces the network to learn structured intermediate representations
- Removing auxiliary tasks degrades planning performance

## Training Methods

| Method | Description |
|--------|-------------|
| Imitation Learning | Mimic human driving trajectories (L2 loss) |
| Collision / Drivable Loss | Penalize collision and off-road |
| Comfort Loss | Limit acceleration / curvature discontinuity |
| RL Fine-tuning | Fine-tune in closed-loop simulation with PPO/SAC |

## RL Fine-tuning

### Why RL Fine-tuning Is Needed

Imitation Learning has a fatal flaw: **distribution shift**.

```
Training: model only sees states near expert trajectories
Deployment: model makes a small error → drifts from expert distribution → errors compound → crash
```

RL fine-tuning lets the model **make mistakes and recover** in closed-loop simulation, learning to handle off-distribution states.

### Typical Pipeline

```
Phase 1: Imitation Learning (offline)
    Expert dataset → Supervised training → Initial policy π₀

Phase 2: RL Fine-tuning (closed-loop)
    π₀ in simulator → Collect experience → PPO/SAC update → π*
```

IL provides a "decent" starting point; RL polishes it to "handle edge cases."

### Reward Design

The most difficult and critical part of RL fine-tuning:

| Reward Component | Meaning | Typical Weight |
|-----------------|---------|----------------|
| Progress | Distance along reference line | +1.0 |
| Collision | Collision penalty | -10.0 |
| Off-road | Leaving drivable area | -5.0 |
| Comfort | Jerk / curvature discontinuity penalty | -0.5 |
| Speed limit | Speeding penalty | -1.0 |
| Goal reaching | Reaching destination | +5.0 |
| Lane keeping | Deviation from lane center | -0.2 per meter |

**Challenge:** Weight ratios require extensive tuning. Wrong ratios lead to "staying still is safest" or "reckless driving reaches goal fastest."

### Why PPO / SAC?

| Algorithm | Advantage | Role in End-to-End |
|-----------|-----------|-------------------|
| PPO | Stable, robust to hyperparameters | **Primary choice for online fine-tuning** — clip mechanism prevents policy updates from overwriting IL pre-training |
| SAC | Sample efficient, off-policy reuses data | **Offline + online hybrid** — expert IL data can be stored in replay buffer |

### Simulator's Role

RL fine-tuning cannot be done on real vehicles (too dangerous), requiring high-fidelity simulation:

| Simulator | Team | Features |
|-----------|------|----------|
| CARLA | Intel | Open-source, sensor simulation, weather/pedestrians |
| Waymax | Waymo | Data-driven, real scenario replay |
| nuPlan | Motional | Planning-focused, large-scale real logs |
| DriveGym | NVIDIA | Differentiable rendering, supports backpropagation |

### Key Challenges

**1. Sim-to-Real Gap**
- Even the best simulator differs from reality
- Solution: Domain Randomization (randomize sensor noise, vehicle dynamics parameters)

**2. Long-tail Scenarios**
- 99% of driving is easy; RL rarely samples hard cases
- Solution: Curriculum Learning (gradually increase difficulty), Adversarial Scene Generation

**3. Training Instability**
- End-to-end networks have ~100M parameters; RL gradient variance is high
- Solution: Freeze backbone, only fine-tune planning head; use small lr; PPO clip mechanism

**4. Reward Hacking**
- Model exploits reward loopholes (e.g., driving in circles to farm progress)
- Solution: Add constraints, use constrained RL, manually audit anomalies

### Frontier Directions

| Direction | Idea |
|-----------|------|
| RLHF for driving | Human feedback replaces hand-crafted reward |
| World Model + RL | Imagination in learned world model (Dreamer-style) |
| Offline RL | Pure log-data RL without simulator (CQL, IQL) |
| Multi-agent RL | Multi-vehicle game theory (lane change negotiation, merging) |
| Constrained RL | Model safety constraints as hard constraints, not reward penalties |

## Data Closed-Loop

End-to-end approaches are highly dependent on data quality. The industrial data engine:

```
Deploy model → Discover bad cases → Mine & annotate → Retrain → Redeploy → Loop
```

RL fine-tuning is just one component of this loop. Without a robust data pipeline continuously feeding hard scenarios back into training, the model stagnates on long-tail cases.

## Safety Fallback

Even if the end-to-end network outputs a trajectory, production vehicles always have a rule-based safety layer:

| Layer | Role |
|-------|------|
| Neural network | Generate candidate trajectory |
| Safety checker (RSS / safety distance) | Verify trajectory doesn't violate hard constraints |
| Emergency fallback | If check fails → apply safe braking / hold lane |

No OEM fully trusts a neural network alone. The safety layer is non-negotiable for certification (ISO 26262 / SOTIF).

## One-Stage vs Two-Stage Debate

The industry has NOT converged on a single approach:

| Approach | Representative | Pros | Cons |
|----------|---------------|------|------|
| One-stage (full E2E) | Tesla FSD | Maximum information flow, no error propagation between modules | Black box, hard to debug, certification challenge |
| Two-stage (perception E2E + separate planner) | Waymo, XPeng | Debuggable, interpretable intermediate outputs, easier safety proof | Information bottleneck at interface, module handoff issues |

Current trend: two-stage is more practical for mass production (easier to certify, debug, and iterate), while one-stage pushes upper-bound performance in research.

## Connection to This Project

```
What you're learning          →  How industry uses it
─────────────────                ──────────────────
PPO clip mechanism            →  Limit policy update magnitude, protect IL pre-training
SAC off-policy buffer         →  Mix IL data + RL interaction data
Reward shaping                →  Multi-component weighted reward design
gamma / GAE                   →  Long-term return vs. short-term safety trade-off
```

PPO and SAC are the primary algorithms in the RL fine-tuning stage of end-to-end solutions. Mastering the fundamentals in this project provides a natural transition into end-to-end research.

---

# 端到端自动驾驶：从传感器到轨迹

## 概述

```
传感器原始输入 → 单一神经网络 → 未来轨迹点序列
```

取代传统的 感知→预测→规划 分模块流水线，用一个网络端到端学习从传感器到轨迹的映射。

## 典型架构

```
Camera / LiDAR
    ↓
Backbone (BEV Encoder)              ← 提取鸟瞰图特征
    ↓
Temporal Fusion                     ← 融合历史帧
    ↓
Query-based Decoder (Transformer)   ← 学习场景交互
    ↓
Trajectory Head                     ← 输出未来 T 步 (x, y, heading)
```

## 代表性工作

| 方案 | 团队 | 核心特点 |
|------|------|---------|
| UniAD | 上海AI Lab | 感知/预测/规划统一在一个 Transformer，多任务联合训练 |
| VAD | 地平线 | 向量化场景表示，去掉 HD Map 依赖 |
| GameFormer | NVIDIA | 博弈论建模交互，多智能体联合预测 |
| PARA-Drive | Waabi | 并行化多任务 head，证明辅助任务提升规划 |
| Diffusion Planner | — | 用扩散模型生成多模态轨迹 |

## 关键技术组件

### 1. BEV 表示（鸟瞰图）

- 将多相机/LiDAR 统一投影到 BEV 空间
- 方法：LSS / BEVFormer / BEVDet
- 规控天然需要 BEV 坐标系下的轨迹

### 2. Transformer Decoder

- Ego query 代表自车意图
- Cross-attention 关注场景特征
- 输出多模态候选轨迹

### 3. 多模态轨迹输出

- 不是只出一条轨迹，而是 K 条候选（如 6 条）+ 每条的置信度
- 选最优或加权融合
- 方法：GMM、Anchor-based、Diffusion

### 4. 辅助任务联合训练

- 检测、车道线、occupancy、运动预测
- 不是为了输出，而是强迫网络学到结构化中间表示
- 去掉辅助任务规划性能会下降

## 训练方式

| 方式 | 说明 |
|------|------|
| Imitation Learning | 模仿人类驾驶轨迹（L2 loss / 模仿损失） |
| Collision / Drivable Loss | 惩罚碰撞、出界 |
| Comfort Loss | 限制加速度/曲率突变 |
| RL Fine-tuning | 闭环仿真中用 PPO/SAC 微调 |

## RL 微调

### 为什么需要 RL 微调？

模仿学习有一个致命缺陷：**分布偏移（distribution shift）**。

```
训练时：模型看到的都是专家轨迹附近的状态
部署时：模型一旦犯小错 → 偏离专家分布 → 越错越远 → 崩溃
```

RL 微调让模型在闭环仿真中**自己犯错、自己恢复**，学会应对偏离后的状态。

### 典型流程

```
阶段 1: Imitation Learning（离线）
    专家数据集 → 监督训练 → 初始策略 π₀

阶段 2: RL Fine-tuning（闭环）
    π₀ 放入仿真器 → 交互采集经验 → PPO/SAC 更新 → π*
```

IL 给一个"还行"的起点，RL 把它打磨到"能应对极端情况"。

### Reward 设计

这是 RL 微调中最难也最关键的部分：

| Reward 分量 | 含义 | 典型权重 |
|-------------|------|---------|
| Progress | 沿参考线前进距离 | +1.0 |
| Collision | 碰撞惩罚 | -10.0 |
| Off-road | 驶出可行驶区域 | -5.0 |
| Comfort | jerk/曲率突变惩罚 | -0.5 |
| Speed limit | 超速惩罚 | -1.0 |
| Goal reaching | 到达目标 | +5.0 |
| Lane keeping | 偏离车道中心 | -0.2 per meter |

**难点：** reward 各分量的权重需要大量调参。权重比例错了，模型可能学到"不动最安全"或"横冲直撞最快到达"。

### 为什么选 PPO / SAC？

| 算法 | 优势 | 在端到端中的角色 |
|------|------|----------------|
| PPO | 稳定、不容易崩、对超参不敏感 | **闭环在线微调首选** — clip 机制限制策略变化，不会覆盖 IL 预训练效果 |
| SAC | 样本效率高、off-policy 可复用数据 | **离线+在线混合训练** — IL 专家数据也可放进 replay buffer |

### 仿真器的角色

RL 微调不能在真车上做（太危险），必须依赖高保真仿真：

| 仿真器 | 团队 | 特点 |
|--------|------|------|
| CARLA | Intel | 开源、传感器模拟、天气/行人 |
| Waymax | Waymo | 数据驱动、真实场景回放 |
| nuPlan | Motional | 规划专用、大规模真实日志 |
| DriveGym | NVIDIA | 可微分渲染、支持反向传播 |

### 关键挑战

**1. Sim-to-Real Gap**
- 仿真器再好也和真实世界有差距
- 解法：Domain Randomization（随机化传感器噪声、车辆动力学参数）

**2. 长尾场景**
- 99% 的驾驶很简单，RL 采集不到难 case
- 解法：Curriculum Learning（逐步增加难度）、Adversarial Scene Generation（对抗性生成危险场景）

**3. 训练不稳定**
- 端到端网络参数量巨大（~100M），RL 梯度方差高
- 解法：冻结 backbone 只微调 planning head、用较小 lr、PPO 的 clip 机制

**4. Reward Hacking**
- 模型钻 reward 漏洞（如：倒车绕圈刷 progress）
- 解法：加约束、用 constrained RL、人工审查异常行为

### 前沿方向

| 方向 | 思路 |
|------|------|
| RLHF for driving | 人类反馈代替手工 reward |
| World Model + RL | 在学到的 world model 中做 imagination（Dreamer 类） |
| Offline RL | 纯用日志数据做 RL，不需要仿真器（CQL, IQL） |
| Multi-agent RL | 多车博弈，处理交互（变道博弈、汇入） |
| Constrained RL | 把安全约束建模为硬约束而非 reward 惩罚 |

## 数据闭环

端到端方案高度依赖数据质量。工业界的数据引擎：

```
部署模型 → 发现 bad case → 挖掘&标注 → 重训 → 重新部署 → 循环
```

RL 微调只是这个闭环中的一环。没有持续把困难场景回灌训练的数据管线，模型在长尾场景上会停滞不前。

## 安全兜底（Safety Fallback）

即使端到端网络输出了轨迹，量产车上始终有规则兜底层：

| 层级 | 角色 |
|------|------|
| 神经网络 | 生成候选轨迹 |
| 安全检查器（RSS / 安全距离） | 验证轨迹不违反硬约束 |
| 紧急兜底 | 检查未通过 → 安全制动 / 保持车道 |

没有主机厂会完全信任神经网络。安全层是认证（ISO 26262 / SOTIF）的硬性要求。

## 一段式 vs 两段式之争

业界并未完全收敛到一种方案：

| 方案 | 代表 | 优点 | 缺点 |
|------|------|------|------|
| 一段式（全端到端） | 特斯拉 FSD | 信息流最大化，无模块间误差传递 | 黑盒，难调试，认证困难 |
| 两段式（感知端到端 + 独立 planner） | Waymo、小鹏 | 可调试，中间输出可解释，安全证明更容易 | 接口处信息瓶颈，模块交接问题 |

当前趋势：两段式更适合量产落地（容易认证、调试、迭代），一段式在研究中推高性能上限。

## 与本项目的关系

```
你正在学的                    →  工业界怎么用
──────────                       ──────────
PPO clip mechanism            →  限制策略更新幅度，保护 IL 预训练
SAC off-policy buffer         →  混合 IL 数据 + RL 交互数据
Reward shaping                →  多分量加权 reward 设计
gamma / GAE                   →  长期回报 vs 短期安全的权衡
```

PPO 和 SAC 是端到端方案中 RL 微调阶段的主力算法。学完本项目的基础内容，再进入端到端方向会非常自然。
