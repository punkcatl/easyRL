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
| GameFormer | University of Toronto | Game-theoretic interaction modeling, multi-agent joint prediction |
| PARA-Drive | Waabi | Parallel multi-task heads, auxiliary tasks improve planning |
| Diffusion Planner | Janner et al. / Multiple groups | Diffusion model generates multi-modal trajectories |

## Key Technical Components

### 1. BEV Representation (Bird's-Eye View)

- Project multi-camera / LiDAR into unified BEV space
- Methods: LSS (Lift-Splat-Shoot — depth estimation + projection), BEVFormer (Transformer spatial cross-attention), BEVDet (explicit depth-based view transform)
- Planning naturally needs trajectories in BEV coordinates

### 2. Transformer Decoder

- Ego query represents ego-vehicle intent
- Cross-attention attends to scene features
- Outputs multi-modal candidate trajectories

### 3. Multi-Modal Trajectory Output

- Not a single trajectory, but K candidates (e.g., 6) + confidence score for each
- Select best or weighted fusion
- Methods: GMM (Gaussian Mixture Model — parameterize trajectory distribution as mixture of Gaussians), Anchor-based (predefined trajectory templates refined by the network), Diffusion (iterative denoising generates diverse trajectories; see Diffusion Planner in Representative Works)

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

Imitation Learning has a well-known fundamental limitation: **distribution shift**.

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
| PPO | Stable, more robust to hyperparameters than vanilla PG | **Primary choice for online fine-tuning** — clip mechanism limits per-step policy change, helping preserve IL pre-training knowledge |
| SAC | Sample efficient, off-policy reuses data | **Offline + online hybrid** — expert IL data can be mixed into replay buffer via modified schemes (e.g., demonstration-augmented replay with priority weighting or ratio limiting — vanilla SAC requires adaptation for this) |

### Simulator's Role

RL fine-tuning cannot be done on real vehicles (too dangerous), requiring high-fidelity simulation:

| Simulator | Team | Features |
|-----------|------|----------|
| CARLA | CVC / Intel Labs | Open-source, sensor simulation, weather/pedestrians |
| Waymax | Waymo | Data-driven, real scenario replay |
| nuPlan | Motional | Planning-focused, large-scale real logs |
| NVIDIA DRIVE Sim | NVIDIA | High-fidelity rendering on Omniverse, sensor simulation |

### Key Challenges

**1. Sim-to-Real Gap** (see Sim-to-Real Transfer section below for detailed treatment)
- Even the best simulator differs from reality
- Solution: Domain Randomization (randomize sensor noise, vehicle dynamics parameters)

**2. Long-tail Scenarios**
- 99% of driving is easy; RL rarely samples hard cases
- Solution: Curriculum Learning (gradually increase difficulty), Adversarial Scene Generation

**3. Training Instability**
- End-to-end networks have 100M-400M+ parameters; RL gradient variance is high
- Solution: Freeze backbone, only fine-tune planning head; use small lr; PPO clip mechanism

**4. Reward Hacking**
- Model exploits reward loopholes (e.g., driving in circles to farm progress)
- Solution: Add constraints, use constrained RL, manually audit anomalies

### Deep Dives

#### What Does "Freeze Backbone, Only Fine-Tune Planning Head" Mean?

```
Input images → [Backbone (BEV encoder)] → [Planning Head (trajectory output)] → Trajectory
                       │                            │
               Frozen (no weight update)     Only this part trains via RL
```

- **Backbone**: the front layers responsible for extracting features from raw sensors (turning images into BEV representations). Contains ~80-90% of all parameters.
- **Planning Head**: the final layers responsible for generating trajectories from features. Much smaller.
- **Freeze**: set `requires_grad = False`, preventing RL gradients from updating those weights.

Why this helps: RL gradient signal is noisy. If the entire network updates, the visual features learned during IL pretraining may degrade. Freezing the backbone protects "how to see" while only refining "how to drive."

```python
# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Only planning head participates in RL update
optimizer = Adam(model.planning_head.parameters(), lr=1e-4)

# Alternative: differential learning rates (soft freeze)
optimizer = Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-6},   # near-frozen
    {'params': model.planning_head.parameters(), 'lr': 1e-4},
])
```

#### How to Determine Where Perception Ends and Planning Begins

There is no natural boundary — it is a **design decision**. But several methods help identify which layers contribute more to which task:

| Method | Approach | What It Reveals |
|--------|----------|-----------------|
| Gradient analysis | Compute gradient magnitude per layer for perception loss vs. planning loss | Which layers respond to which task |
| Layer-wise ablation | Freeze layers progressively, measure task performance drop | Where the critical boundary lies |
| Linear probing | Attach a linear classifier to intermediate features, test what tasks they can solve | What information each layer encodes |
| Multi-task observation | Train with auxiliary tasks (detection, lane lines), observe which layers the tasks depend on | Functional decomposition |

In practice, the split point is chosen empirically through ablation experiments (try several cut points, pick the one that balances stability and performance).

#### What Is an Ablation Study?

Core idea: **remove or alter one component of the system, observe how much overall performance changes, to determine how important that component is.**

```
Full system: IL pretraining + RL fine-tuning + domain rand + safety layer → collision rate 0.5%

Ablations:
  Remove RL fine-tuning     → collision rate 3.2%  → RL is critical
  Remove domain rand        → collision rate 4.8%  → domain rand even more so
  Remove safety layer       → collision rate 1.1%  → safety layer catches some cases
  Remove IL pretraining     → training diverges    → essential
```

Ways to "ablate" go beyond just deletion:

| Ablation Type | Method | Use Case |
|---------------|--------|----------|
| Remove | Delete the module entirely | Independent modules |
| Replace | Substitute with simpler version (e.g., Transformer → MLP) | Validate architecture choices |
| Freeze | Prevent weight updates | Test whether a layer needs training |
| Zero out | Set input to zero or random noise | Test whether an input is useful |
| Scale down | Reduce size (layers, dimensions) | Test capacity requirements |

A typical ablation table in papers:

```
| Configuration              | mAP  | Planning L2 | Collision |
|----------------------------|------|-------------|-----------|
| Full model                 | 42.3 | 1.02        | 0.5%      |
| w/o temporal fusion        | 38.1 | 1.35        | 1.2%      |
| w/o auxiliary detection    | 41.8 | 1.28        | 0.9%      |
| w/o BEV representation     | 35.2 | 2.10        | 3.4%      |
| w/o RL fine-tuning         | 42.3 | 1.45        | 3.2%      |
```

The larger the performance drop → the more important that component. This is the standard method for validating design decisions in ML research.

Note: in the ablation table above, mAP is unchanged when removing RL fine-tuning because RL only affects the planning head, not perception.

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
| Safety checker (RSS — Responsibility-Sensitive Safety: formal model ensuring ego vehicle can always avoid being at fault in a collision / safety distance) | Verify trajectory doesn't violate hard constraints |
| Emergency fallback | If check fails → apply safe braking / hold lane |

No OEM fully trusts a neural network alone. The safety layer is non-negotiable for certification (ISO 26262: functional safety of E/E systems / SOTIF — ISO 21448: safety of intended functionality, covering ML-based systems).

## One-Stage vs Two-Stage Debate

The industry has NOT converged on a single approach:

| Approach | Representative | Pros | Cons |
|----------|---------------|------|------|
| One-stage (full E2E) | Tesla FSD | Maximum information flow, no error propagation between modules | Black box, hard to debug, certification challenge |
| Two-stage (perception E2E + separate planner) | Waymo, XPeng | Debuggable, interpretable intermediate outputs, easier safety proof | Information bottleneck at interface, module handoff issues |

Current trend: two-stage is more practical for mass production (easier to certify, debug, and iterate), while one-stage pushes upper-bound performance in research.

## Production Pipeline: End-to-End + RL

This section reframes the RL fine-tuning concepts introduced earlier in the context of a complete production system, showing how they integrate with industry-specific infrastructure (world models, safety layers, data engines). Earlier sections covered RL fine-tuning in isolation; here we show the full production workflow.

### Overview

```
Traditional Modular Pipeline:
┌───────────┐   ┌────────────┐   ┌──────────┐   ┌─────────┐
│ Perception├──►│ Prediction ├──►│ Planning ├──►│ Control │
└───────────┘   └────────────┘   └──────────┘   └─────────┘
     │               │               │              │
  3D detection   Motion forecast  Route search   PID/MPC
  Lane lines     Intent predict   Trajectory     Actuator
  Tracking       Risk assess      Optimization   commands

End-to-End + RL Pipeline:
┌─────────────────────────────────────────────────────────┐
│              Unified Neural Network                     │
│                                                         │
│  Sensors ──► Backbone ──► Trajectory Head ──► Output    │
│  (camera,    (BEV        (direct waypoints,             │
│   LiDAR,     encoder)     speed, curvature)             │
│   radar)                                                │
└─────────────────────────────────────────────────────────┘
                        │
                   RL Policy Refinement
                   (closed-loop training)
```

**Key difference**: The modular pipeline accumulates errors at each interface (perception errors propagate to prediction, then to planning). The end-to-end approach optimizes the entire system jointly for the final driving objective.

### Industry Approaches

**Momenta** employs a "world model" (neural simulator) as the core training infrastructure:
- Reconstructs driving scenarios from massive road data (reportedly billions of km logged)
- The world model predicts how the environment evolves given the ego vehicle's actions
- Enables closed-loop RL training without real-world risk
- Policy learns from diverse scenarios generated/replayed by the world model

**Horizon Robotics (SuperDrive)** focuses on end-to-end + RL for interactive scenarios:
- Targets game-theoretic situations: lane changes, unprotected intersections, highway merging
- Models other agents as rational actors with their own objectives
- RL policy learns negotiation strategies (yield, assert, cooperate)
- Emphasizes real-time inference on edge hardware (BPU chips)

### Three-Phase Training Process

```
Phase 1: Supervised Pretraining
┌─────────────────────────────────────────────┐
│ Human driving logs (millions of clips)      │
│ Model learns: observation → trajectory      │
│ Loss: L2 distance to human trajectory       │
│ Result: competent but imperfect driver      │
└─────────────────────────────────────────────┘
                    │
                    ▼
Phase 2: RL Fine-Tuning in World Model
┌─────────────────────────────────────────────┐
│ Simulator / World Model (closed-loop)       │
│ Algorithm: PPO (millions of episodes)       │
│ Alternatives: SAC, offline RL (IQL, CQL)    │
│ Explores beyond human demonstrations        │
│ Handles long-tail edge cases                │
└─────────────────────────────────────────────┘
                    │
                    ▼
Phase 3: Deployment with Safety Layer
┌─────────────────────────────────────────────┐
│ RL policy outputs candidate trajectory      │
│ Safety layer: control barrier functions,    │
│   trajectory feasibility checks, rule-based │
│   safety monitors (RSS)                     │
│ Override or clamp if constraint violated    │
│ Final trajectory sent to actuators          │
└─────────────────────────────────────────────┘
```

**Phase 1** provides a strong initialization. Without it, RL would need to learn basic driving from scratch (impractical sample complexity). Some production systems add an offline RL step (CQL/IQL on logged data) between Phase 1 and 2 as a safer intermediate — offline RL improves the policy using only existing logged data without active exploration, avoiding catastrophic actions that an early-stage policy might take in a simulator.

**Phase 2** is where RL adds value beyond imitation. PPO is the common choice for its stability — the clip mechanism approximates a trust region constraint, preventing destructively large updates; SAC is used when sample efficiency matters; offline RL methods apply when world model fidelity is limited.

**Phase 3** ensures that even if the learned policy makes an error, hard safety constraints prevent catastrophic outcomes.

### Why RL is Needed Beyond Supervised Learning

| Problem | Supervised Learning | RL Solution |
|---------|-------------------|-------------|
| Distribution shift | Trained on human data, fails on unseen states | Closed-loop training covers recovery states |
| Causal confusion | Correlates brake lights with stopping (not the obstacle) | Breaks spurious correlations — agent experiences consequences of its own actions |
| Long-tail safety | Rare events underrepresented in data | Can generate and train on edge cases |
| Comfort optimization | Averages over different human styles | Optimizes explicit comfort reward |
| Closed-loop consistency | Open-loop training ignores compounding errors | Actions affect future states during training |

**Distribution shift** is the most critical: a supervised model never sees its own mistakes during training, so when it drifts slightly off the human trajectory at test time, it enters states it was never trained on, causing cascading failures.

### Core Technical Challenges

**1. World Model / Closed-Loop Simulator**

The world model must faithfully simulate:
- Sensor observations (camera images, LiDAR points) or their learned representations
- Other agents' reactive behavior (they respond to the ego vehicle)
- Physical dynamics (vehicle motion, road friction)
- Rare events (pedestrian darting out, debris on road)

Approaches range from neural rendering (NeRF/3DGS-based) to learned dynamics models to log-replay with agent re-simulation.

**2. Reward Engineering**

```
R(s, a) = w_safety * R_safety      (no collisions, maintain distance)
         + w_progress * R_progress  (follow route, maintain speed)
         + w_comfort * R_comfort    (limit jerk, lateral acceleration)
         + w_rules * R_rules        (traffic laws, lane keeping)
```

Challenges:
- Balancing competing objectives (progress vs. safety)
- Sparse vs. dense rewards (crashes are rare but critical)
- Reward hacking (finding loopholes, e.g., never moving = no collision)
- Human preference alignment (what "comfortable" means varies)

**3. Sim-to-Real Gap**

See the dedicated Sim-to-Real Transfer section below for detailed treatment. Key gap sources relevant to production deployment:
- Visual domain gap: rendered images differ from real camera feeds
- Physics mismatch: simplified dynamics vs. real tire-road interaction
- Behavior gap: simulated agents vs. real human drivers
- Actuator delay: simulation runs without real-time constraints

## Sim-to-Real Transfer

This section connects directly to the production pipeline above: the RL fine-tuning phase (Phase 2) occurs inside a world model or simulator, and the policy must transfer to real vehicles (Phase 3). The techniques below address how to close that gap.

### The Reality Gap Problem

Policies trained purely in simulation fail when deployed to real vehicles because the simulator is an imperfect approximation of the physical world:

```
                        Sim-to-Real Gap Sources
+-------------------+-------------------------------------------+
| Category          | Examples                                  |
+-------------------+-------------------------------------------+
| Physics mismatch  | Tire friction, vehicle mass/inertia,      |
|                   | road surface, aerodynamics                |
+-------------------+-------------------------------------------+
| Sensor noise      | Camera exposure variation, LiDAR dropouts,|
|                   | radar multipath, rolling shutter          |
+-------------------+-------------------------------------------+
| Actuator delay    | Steering latency (40-150 ms), braking     |
|                   | (100-300 ms), throttle (20-50 ms)         |
+-------------------+-------------------------------------------+
| Visual domain gap | Sim textures look "too clean," lighting   |
|                   | is unrealistic, missing weathering/dirt   |
+-------------------+-------------------------------------------+
```

### Domain Randomization

The core idea: if the policy trains under **many random variations** of the environment, it learns a robust strategy that generalizes to the real world (which is just one more "variation").

**Visual Randomization:**
- Random textures on roads, buildings, vehicles
- Lighting variation (sun angle, shadows, overcast, night)
- Camera noise injection (Gaussian noise, motion blur, lens distortion)
- Random occlusions and weather effects (rain drops on lens, fog)

**Dynamics Randomization:**
- Friction coefficients (dry / wet / icy) sampled from a range
- Vehicle mass and center-of-gravity shift
- Damping and suspension stiffness
- Actuator latency sampled uniformly
- Wind disturbance forces

```
Training episode i:
    friction ~ Uniform(0.3, 1.0)
    mass     ~ Uniform(1400, 2000) kg
    steering_latency ~ Uniform(40, 150) ms
    lighting ~ Random(sunrise, noon, overcast, night)

    → Policy must succeed across ALL sampled conditions
    → Real-world parameters likely fall within this range
```

**Why it works:** The policy cannot "memorize" a single set of dynamics. It must learn control strategies that are inherently robust to parameter variation — analogous to data augmentation in supervised learning — both expand the training distribution to improve generalization.

### System Identification (SysID)

Instead of randomizing everything, measure the real system and make the simulator match.

| Approach | Method | Trade-off |
|----------|--------|-----------|
| Manual calibration | Measure tire parameters, vehicle mass, sensor intrinsics on a test bench | Accurate but static; doesn't capture wear/temperature drift |
| Learned SysID | Train a neural net to predict sim parameters from real-world rollouts | Adapts over time but requires real data |
| Online adaptation | Estimate parameters during deployment (e.g., EKF on friction) and adjust policy | Real-time but adds system complexity |

**Adaptive SysID pipeline:**

```
Real vehicle rollout (short)
    ↓
Parameter estimator (neural net / optimization)
    ↓
Update simulator parameters
    ↓
Re-train or fine-tune policy in calibrated sim
    ↓
Deploy updated policy
```

The key advantage over domain randomization: SysID can produce a more precise policy (not forced to be robust to unrealistic extremes). The disadvantage: it requires real-world data collection, which is expensive.

### Progressive Transfer / Curriculum

Bridge the gap incrementally rather than jumping directly from simple sim to real:

```
Stage 1: Simple simulator (fast, low fidelity)
    - Basic physics, no rendering
    - Train core driving behaviors
    ↓
Stage 2: High-fidelity simulator (slower, realistic)
    - Photo-realistic rendering (CARLA, NVIDIA Drive Sim)
    - Calibrated vehicle dynamics
    - Fine-tune policy
    ↓
Stage 3: Hardware-in-the-Loop (HIL)
    - Real actuators + simulated environment
    - Validate timing and latency behavior
    ↓
Stage 4: Real-world deployment
    - Minimal fine-tuning on real vehicle
    - Safety driver intervention as feedback signal
```

Each stage inherits the policy from the previous one. Training time in expensive simulators is reduced because the policy arrives pre-trained from the cheaper stage.

### Techniques Specific to Autonomous Driving

| Technique | How It Helps | Examples |
|-----------|-------------|----------|
| Neural rendering / World models | Train on real sensor data to generate realistic sim frames — directly reduces visual gap | GAIA-1 (Wayve), UniSim (Google DeepMind), DriveDreamer |
| Log replay with perturbation | Replay real logged scenarios but inject counterfactual ego actions; evaluate consequences via learned dynamics | Waymax, nuPlan scenario perturbation |
| Domain adaptation networks | Learn feature representations invariant to sim vs. real domain (adversarial training) | DANN applied to BEV features |

> **How DANN works:** A domain classifier branch is added during training. The feature extractor is trained adversarially to produce representations that the classifier cannot distinguish as sim or real, forcing domain-invariant features that help the policy generalize across domains.

| Photorealistic asset pipelines | Scan real environments to create sim assets that closely match deployment cities | NVIDIA Omniverse, Parallel Domain |
| Residual policy learning | Train a base policy in sim; learn a small residual correction on real vehicle | Sim policy + real-world delta |

**World model approach (actively researched direction, adopted by Momenta and Wayve among others):**

```
Real driving logs (camera, LiDAR, actions, outcomes)
    ↓
Train world model: given state + action → predict next state
    ↓
Use world model AS the simulator for RL training
    ↓
Reduced visual gap (model learned from real pixels)
Reduced physics gap (dynamics learned from real transitions)
```

Limitation: the world model can only generate scenarios similar to training data. Novel situations (rare crashes) remain hard to synthesize. This is why Momenta and Horizon supplement world models with adversarial scenario generation.

### Multi-Agent Interaction Transfer

Transferring interaction behaviors (lane change negotiation, merging) is particularly hard because:
- Other agents' policies are unknown and vary between sim and real
- Real drivers respond to the ego vehicle's body language (speed changes, positioning) in ways that are hard to model
- Game-theoretic equilibria in sim may not match real-world norms

This connects to Horizon's SuperDrive approach: by modeling other agents as rational actors and training negotiation strategies via RL, the policy becomes less dependent on exact agent models.

### Metrics for Sim-to-Real Success

| Metric | Definition | What It Tells You |
|--------|-----------|-------------------|
| Performance correlation | Spearman rank correlation of policy scores in sim vs. real | Does better-in-sim mean better-in-real? |
| Transfer efficiency | Real-world fine-tuning steps needed to reach target performance | How much of the sim training "sticks"? |
| Zero-shot transfer rate | % of sim-trained skills that work on first real deployment | Quality of the sim or randomization |
| Intervention rate | Human takeover frequency per km on real vehicle | Direct safety metric |
| Performance degradation | (Sim score - Real score) / Sim score | Size of the remaining gap |

A reasonable aspirational target (not an industry standard — actual thresholds depend on application and safety requirements): performance correlation > 0.8 (ranking mostly preserved) and < 10% of the simulation training volume needed as additional real-world fine-tuning data.

### Current State: What Works and What Doesn't

**What works:**
- Domain randomization for low-level control (steering, throttle) — well established
- SysID for vehicle dynamics — mature engineering practice
- Log replay for scenario testing — widely adopted in industry
- World models for visual prediction (short horizons, 3-5 seconds)

**What doesn't (yet):**
- Zero-shot transfer of complex driving policies (multi-lane, intersections) — still requires real fine-tuning
- Long-horizon world model rollouts (> 10 seconds) — compounding errors
- Transferring interaction behaviors — other agents differ fundamentally between sim and real
- Guaranteeing safety coverage — no method proves the policy handles ALL real-world edge cases

**Open problems:**
- How to quantify "how much gap remains" before real deployment
- Automatic selection of what to randomize vs. what to calibrate
- Combining world models with explicit physics (hybrid sim)
- Scaling world models to handle rare / dangerous scenarios they never observed
- Formal verification of transferred policies for safety certification

## Connection to This Project

```
What you're learning          →  How industry uses it
─────────────────                ──────────────────
PPO clip mechanism            →  Limit policy update magnitude, protect IL pre-training
SAC off-policy buffer         →  Mix IL data + RL interaction data
Reward shaping                →  Multi-component weighted reward design
gamma / GAE                   →  Long-term return vs. short-term safety trade-off
Domain randomization (introduced in this document)  →  Robust sim-to-real transfer
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
| GameFormer | 多伦多大学 | 博弈论建模交互，多智能体联合预测 |
| PARA-Drive | Waabi | 并行化多任务 head，证明辅助任务提升规划 |
| Diffusion Planner | Janner et al. / 多个团队 | 用扩散模型生成多模态轨迹 |

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

模仿学习有一个根本性的局限：**分布偏移（distribution shift）**。

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
| PPO | 稳定、不容易崩、对超参相对鲁棒（相比 vanilla PG） | **闭环在线微调首选** — clip 机制限制策略变化，有助于保留 IL 预训练效果 |
| SAC | 样本效率高、off-policy 可复用数据 | **离线+在线混合训练** — IL 专家数据可通过优先采样等方式混入 replay buffer（需处理分布不匹配） |

### 仿真器的角色

RL 微调不能在真车上做（太危险），必须依赖高保真仿真：

| 仿真器 | 团队 | 特点 |
|--------|------|------|
| CARLA | CVC / Intel Labs | 开源、传感器模拟、天气/行人 |
| Waymax | Waymo | 数据驱动、真实场景回放 |
| nuPlan | Motional | 规划专用、大规模真实日志 |
| NVIDIA DRIVE Sim | NVIDIA | 高保真渲染（基于 Omniverse）、传感器模拟 |

### 关键挑战

**1. Sim-to-Real Gap**（详见下方 Sim-to-Real 迁移专节）
- 仿真器再好也和真实世界有差距
- 解法：Domain Randomization（随机化传感器噪声、车辆动力学参数）

**2. 长尾场景**
- 99% 的驾驶很简单，RL 采集不到难 case
- 解法：Curriculum Learning（逐步增加难度）、Adversarial Scene Generation（对抗性生成危险场景）

**3. 训练不稳定**
- 端到端网络参数量巨大（100M-400M+），RL 梯度方差高
- 解法：冻结 backbone 只微调 planning head、用较小 lr、PPO 的 clip 机制

**4. Reward Hacking**
- 模型钻 reward 漏洞（如：绕圈行驶刷 progress 奖励）
- 解法：加约束、用 constrained RL、人工审查异常行为

### 深入解析

#### "冻结 Backbone 只微调 Planning Head"是什么意思？

```
输入图像 → [Backbone (BEV编码器)] → [Planning Head (轨迹输出头)] → 轨迹
                  │                          │
          冻结（不更新权重）            只有这部分通过 RL 训练
```

- **Backbone**：前面的层，负责从原始传感器提取特征（把图像变成 BEV 表征）。参数量大，约占整个网络的 80-90%。
- **Planning Head**：最后几层，负责从特征生成轨迹。参数量小。
- **冻结**：设置 `requires_grad = False`，阻止 RL 梯度更新这些权重。

为什么有效：RL 梯度信号噪声大。如果全网络更新，IL 预训练阶段学到的视觉特征可能退化。冻结 backbone 保护"怎么看"，只优化"怎么开"。

```python
# 冻结 backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# 只有 planning head 参与 RL 更新
optimizer = Adam(model.planning_head.parameters(), lr=1e-4)

# 替代方案：差异化学习率（软冻结）
optimizer = Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-6},   # 近似冻结
    {'params': model.planning_head.parameters(), 'lr': 1e-4},
])
```

#### 如何确定哪些层属于感知、哪些层属于规划？

不存在天然的分界线——这是一个**设计决策**。但有几种方法帮助判断各层对不同任务的贡献：

| 方法 | 做法 | 揭示什么 |
|------|------|---------|
| 梯度分析 | 分别对感知 loss 和规划 loss 算各层梯度大小 | 哪些层响应哪个任务 |
| 逐层消融 | 逐步冻结更多层，观察任务性能下降 | 关键分界点在哪里 |
| 线性探针 | 在中间层特征上接线性分类器，测试能完成什么任务 | 每层编码了什么信息 |
| 多任务观察 | 用辅助任务（检测、车道线）联合训练，观察各任务依赖哪些层 | 功能分解 |

实际工程中，切分点通过消融实验确定（试几个切分位置，选平衡稳定性和性能的最优点）。

#### 什么是消融实验（Ablation Study）？

核心思想：**去掉或改变系统的一个部分，看整体性能变化多少，从而判断该部分有多重要。**

```
完整模型：IL预训练 + RL微调 + 域随机化 + 安全层 → 碰撞率 0.5%

消融：
  去掉 RL 微调      → 碰撞率 3.2%    → RL 很关键
  去掉域随机化      → 碰撞率 4.8%    → 域随机化更关键
  去掉安全层        → 碰撞率 1.1%    → 安全层兜住了一些 case
  去掉 IL 预训练    → 训练不收敛      → 必须有
```

消融的方式不止"删除"一种：

| 消融方式 | 做法 | 适用场景 |
|---------|------|---------|
| 删除 | 直接移除模块 | 独立模块 |
| 替换 | 用简单版替代（如 Transformer → MLP） | 验证架构选择 |
| 冻结 | 不更新权重 | 验证某层是否需要训练 |
| 置零 | 输入置零或加随机噪声 | 验证某输入是否有用 |
| 减少 | 减小规模（层数、维度） | 验证容量需求 |

论文中典型的消融表格：

```
| 配置                         | mAP  | Planning L2 | 碰撞率 |
|-----------------------------|------|-------------|--------|
| 完整模型                     | 42.3 | 1.02        | 0.5%   |
| 去掉 temporal fusion        | 38.1 | 1.35        | 1.2%   |
| 去掉辅助检测任务              | 41.8 | 1.28        | 0.9%   |
| 去掉 BEV 表示                | 35.2 | 2.10        | 3.4%   |
| 去掉 RL 微调                 | 42.3 | 1.45        | 3.2%   |
```

性能掉得越多 → 该组件越重要。这是验证设计决策的标准方法。

注意：消融表中去掉 RL 微调后 mAP 不变，因为 RL 只影响规划头，不影响感知性能。

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

## 量产方案流程：端到端 + RL

本节将前面介绍的 RL 微调概念放入完整的量产系统背景中，展示它们如何与工业界特有的基础设施（世界模型、安全层、数据引擎）整合。前面的章节单独介绍了 RL 微调；这里展示完整的量产工作流。

### 概述

```
传统模块化方案：
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│ 感知 ├──►│ 预测 ├──►│ 规划 ├──►│ 控制 │
└──────┘   └──────┘   └──────┘   └──────┘
   │          │          │          │
 3D检测     运动预测    路径搜索    PID/MPC
 车道线     意图预测    轨迹优化    执行器指令
 跟踪       风险评估

端到端 + RL 方案：
┌─────────────────────────────────────────────────────────┐
│                  统一神经网络                             │
│                                                         │
│  传感器 ──► 骨干网络 ──► 轨迹输出头 ──► 输出                 │
│  (相机、    (BEV        (直接输出路点、                     │
│   激光雷达、 编码器)     速度、曲率)                         │
│   毫米波)                                                │
└─────────────────────────────────────────────────────────┘
                        │
                   RL 策略精调
                  (闭环训练优化)
```

**核心区别**：模块化方案在每个接口处累积误差（感知误差传播到预测，再传播到规划）。端到端方案对整个系统进行联合优化，直接面向最终驾驶目标。

### 业界方案

**Momenta** 以"世界模型"（神经仿真器）作为核心训练基础设施：
- 从海量道路数据（据报道达数十亿公里行驶记录）中重建驾驶场景
- 世界模型预测在自车动作下环境如何演化
- 实现无真实世界风险的闭环 RL 训练
- 策略从世界模型生成/回放的多样化场景中学习

**地平线（SuperDrive）** 聚焦于博弈场景下的端到端 + RL：
- 针对博弈性场景：变道、无保护交叉路口、高速汇入
- 将其他交通参与者建模为有自身目标的理性主体
- RL 策略学习协商策略（让行、争夺、协作）
- 强调在边缘硬件（BPU 芯片）上的实时推理能力

### 三阶段训练流程

```
第一阶段：监督预训练
┌─────────────────────────────────────────────┐
│ 人类驾驶数据（数百万片段）                      │
│ 模型学习：观测 → 轨迹                          │
│ 损失函数：与人类轨迹的 L2 距离                  │
│ 结果：能力尚可但不完美的驾驶策略                  │
└─────────────────────────────────────────────┘
                    │
                    ▼
第二阶段：世界模型中的 RL 精调
┌─────────────────────────────────────────────┐
│ 仿真器 或 世界模型（闭环环境）                   │
│ 算法：PPO（数百万回合）                         │
│ 替代方案：SAC、离线 RL（IQL, CQL）              │
│ 探索人类示范之外的策略空间                       │
│ 处理长尾边缘场景                               │
└─────────────────────────────────────────────┘
                    │
                    ▼
第三阶段：部署 + 安全层
┌─────────────────────────────────────────────┐
│ RL 策略输出候选轨迹                            │
│ 安全层：控制屏障函数、轨迹可行性检查、             │
│   基于规则的安全监控器（RSS）                   │
│ 违反约束时进行覆盖或截断                        │
│ 最终轨迹发送至执行器                            │
└─────────────────────────────────────────────┘
```

**第一阶段** 提供良好的初始化。没有它，RL 需要从零学习基本驾驶（样本复杂度不可接受）。部分量产系统在第一、二阶段之间加入离线 RL 步骤（CQL/IQL），作为更安全的过渡——离线 RL 仅用已有日志数据改进策略，无需交互探索，避免了早期策略在仿真器中可能采取的灾难性动作。

**第二阶段** 是 RL 超越模仿学习的关键。PPO 因稳定性被常用——clip 机制近似信任域约束，防止破坏性的大幅更新；SAC 在需要样本效率时使用；离线 RL 在世界模型保真度有限时适用。

**第三阶段** 确保即使学到的策略出错，硬性安全约束也能防止灾难性后果。

### 为什么监督学习之上还需要 RL

| 问题 | 监督学习的局限 | RL 的解决方案 |
|------|---------------|--------------|
| 分布偏移 | 在人类数据上训练，遇到未见状态则失败 | 闭环训练覆盖恢复状态 |
| 因果混淆 | 将刹车灯与停车关联（而非障碍物本身） | 闭环交互打破虚假相关——智能体经历自身动作的后果 |
| 长尾安全 | 罕见事件在数据中代表性不足 | 可生成并训练边缘场景 |
| 舒适度优化 | 对不同人类驾驶风格取平均 | 优化显式舒适度奖励 |
| 闭环一致性 | 开环训练忽略累积误差 | 训练时动作影响未来状态 |

**分布偏移** 是最关键的问题：监督模型在训练时从未见过自己的错误，因此测试时稍微偏离人类轨迹，就会进入从未训练过的状态，导致级联失败。

### 核心技术挑战

**1. 世界模型 / 闭环仿真器**

世界模型必须忠实模拟：
- 传感器观测（相机图像、激光雷达点云）或其学习到的表征
- 其他交通参与者的反应性行为（对自车动作的响应）
- 物理动力学（车辆运动、路面摩擦）
- 罕见事件（行人突然冲出、路面障碍物）

方法范围从神经渲染（基于 NeRF/3DGS）到学习的动力学模型再到日志回放加智能体重新仿真。

**2. 奖励工程**

```
R(s, a) = w_safety * R_safety      (无碰撞、保持车距)
         + w_progress * R_progress  (跟随路线、保持车速)
         + w_comfort * R_comfort    (限制加加速度、侧向加速度)
         + w_rules * R_rules        (交通法规、车道保持)
```

挑战：
- 平衡竞争性目标（通行效率 vs. 安全性）
- 稀疏奖励 vs. 稠密奖励（碰撞罕见但至关重要）
- 奖励黑客（钻空子，如不动 = 不碰撞）
- 人类偏好对齐（"舒适"的定义因人而异）

**3. Sim-to-Real Gap**

详见下方 Sim-to-Real 迁移专节。与量产部署相关的主要差距来源：
- 视觉域差距：渲染图像与真实相机画面不同
- 物理不匹配：简化动力学 vs. 真实轮胎-路面交互
- 行为差距：仿真智能体 vs. 真实人类驾驶员
- 执行器延迟：仿真无实时约束

## Sim-to-Real 迁移

本节与上方量产方案流程直接关联：RL 精调阶段（第二阶段）在世界模型或仿真器中进行，而策略必须迁移到真实车辆（第三阶段）。以下技术解决如何缩小这一差距。

### 现实差距问题

纯在仿真中训练的策略部署到真车时往往失败，因为仿真器是物理世界的不完美近似：

```
                      Sim-to-Real Gap 来源
+-------------------+-------------------------------------------+
| 类别              | 示例                                        |
+-------------------+-------------------------------------------+
| 物理不匹配         | 轮胎摩擦力、车辆质量/惯量、路面材质、            |
|                   | 空气动力学                                  |
+-------------------+-------------------------------------------+
| 传感器噪声          | 相机曝光变化、LiDAR 丢点、雷达多径、           |
|                   | 卷帘快门                                    |
+-------------------+-------------------------------------------+
| 执行器延迟          | 转向延迟 (40-150 ms)、制动延迟               |
|                   | (100-300 ms)、油门延迟 (20-50 ms)           |
+-------------------+-------------------------------------------+
| 视觉域差距          | 仿真纹理"过于干净"、光照不真实、缺少            |
|                   | 风化/灰尘                                   | 
+-------------------+-------------------------------------------+
```

### 域随机化（Domain Randomization）

核心思想：如果策略在**大量随机变化**的环境中训练，它会学到一个鲁棒策略，能泛化到真实世界（真实世界不过是又一个"变体"）。

**视觉随机化：**
- 道路、建筑、车辆随机纹理
- 光照变化（太阳角度、阴影、阴天、夜间）
- 相机噪声注入（高斯噪声、运动模糊、镜头畸变）
- 随机遮挡和天气效果（镜头雨滴、雾）

**动力学随机化：**
- 摩擦系数（干/湿/冰面）从范围内采样
- 车辆质量和重心偏移
- 阻尼和悬架刚度
- 执行器延迟均匀采样
- 风力扰动

```
训练 episode i:
    friction ~ Uniform(0.3, 1.0)
    mass     ~ Uniform(1400, 2000) kg
    steering_latency ~ Uniform(40, 150) ms
    lighting ~ Random(sunrise, noon, overcast, night)

    → 策略必须在所有采样条件下都成功
    → 真实世界参数大概率落在此范围内
```

**为什么有效：** 策略无法"记住"单一动力学参数组合，必须学到对参数变化天然鲁棒的控制策略 — 类似于监督学习中的数据增强——通过扩展训练分布来提升泛化能力。

### 系统辨识（System Identification, SysID）

不是随机化一切，而是测量真实系统，让仿真器匹配它。

| 方法 | 手段 | 权衡 |
|------|------|------|
| 手动标定 | 在试验台测量轮胎参数、车辆质量、传感器内参 | 精确但静态；无法捕捉磨损/温度漂移 |
| 学习式 SysID | 训练神经网络，从真实驾驶数据预测仿真参数 | 可自适应但需要真实数据 |
| 在线自适应 | 部署时实时估计参数（如 EKF 估摩擦系数）并调整策略 | 实时但增加系统复杂度 |

**自适应 SysID 流程：**

```
真车短距离行驶
    ↓
参数估计器（神经网络 / 优化）
    ↓
更新仿真器参数
    ↓
在标定后的仿真器中重训/微调策略
    ↓
部署更新后的策略
```

相比域随机化的优势：SysID 可以产生更精准的策略（不必对不现实的极端情况保持鲁棒）。劣势：需要采集真实数据，成本高。

### 渐进迁移 / 课程学习

逐步缩小差距，而非从简单仿真直接跳到真实：

```
阶段 1: 简单仿真器（快速、低保真）
    - 基础物理、无渲染
    - 训练核心驾驶行为
    ↓
阶段 2: 高保真仿真器（较慢、逼真）
    - 照片级渲染（CARLA, NVIDIA Drive Sim）
    - 标定过的车辆动力学
    - 微调策略
    ↓
阶段 3: 硬件在环（HIL）
    - 真实执行器 + 仿真环境
    - 验证时序和延迟行为
    ↓
阶段 4: 真实世界部署
    - 最少量真车微调
    - 安全员接管作为反馈信号
```

每个阶段继承上一阶段的策略。在昂贵仿真器中的训练时间大幅减少，因为策略从便宜阶段已经预训练。

### 自动驾驶专用技术

| 技术 | 如何帮助 | 代表 |
|------|---------|------|
| 神经渲染 / World Model | 在真实传感器数据上训练生成逼真仿真帧 — 直接缩小视觉差距 | GAIA-1 (Wayve), UniSim (Google DeepMind), DriveDreamer |
| 日志回放 + 扰动 | 回放真实场景但注入反事实自车动作；通过学习的动力学评估后果 | Waymax, nuPlan 场景扰动 |
| 域适应网络 | 学习对 sim/real 域不变的特征表示（对抗训练） | DANN 应用于 BEV 特征 |
| 照片级资产管线 | 扫描真实环境制作与部署城市高度匹配的仿真资产 | NVIDIA Omniverse, Parallel Domain |
| 残差策略学习 | 在 sim 中训练基础策略；在真车上学习小的残差修正 | Sim 策略 + 真实世界 delta |

**World Model 方法（活跃研究方向，Momenta、Wayve 等公司已采用）：**

```
真实驾驶日志（相机、LiDAR、动作、结果）
    ↓
训练 world model：给定状态 + 动作 → 预测下一状态
    ↓
用 world model 作为 RL 训练的仿真器
    ↓
缩小视觉差距（模型从真实像素学习）
缩小物理差距（动力学从真实转移中学习）
```

局限：world model 只能生成与训练数据相似的场景。罕见情况（稀有碰撞）仍然难以合成。因此 Momenta 和地平线用对抗性场景生成来补充世界模型。

### 多智能体交互迁移

交互行为的迁移（变道博弈、汇入协商）特别困难，因为：
- 其他智能体的策略未知，且 sim 和 real 之间存在差异
- 真实驾驶员会对自车的"肢体语言"（速度变化、车位调整）做出反应，这很难建模
- sim 中的博弈均衡可能不符合真实世界的驾驶规范

这与地平线 SuperDrive 方案相关联：通过将其他智能体建模为理性主体并用 RL 训练协商策略，策略对精确的智能体模型依赖更少。

### Sim-to-Real 成功度量

| 指标 | 定义 | 说明 |
|------|------|------|
| 性能相关性 | 策略在 sim 和 real 中得分的 Spearman 秩相关 | sim 中更好是否真的意味着 real 中更好？ |
| 迁移效率 | 达到目标性能所需的真实世界微调步数 | sim 训练有多少"留存"到了真车？ |
| 零样本迁移率 | 首次真实部署即可用的 sim 训练技能百分比 | 仿真器或随机化的质量如何？ |
| 接管率 | 真车上每公里人类接管次数 | 直接安全指标 |
| 性能退化 | (Sim 分数 - Real 分数) / Sim 分数 | 剩余差距大小 |

经验性参考目标（非行业标准，实际阈值取决于应用场景和安全要求）：性能相关性 > 0.8（排名基本保持），且仅需相当于仿真训练量 < 10% 的额外真实数据微调。

### 现状：什么有效、什么还不行

**有效的：**
- 低层控制（转向、油门）的域随机化 — 已充分验证
- 车辆动力学 SysID — 成熟的工程实践
- 日志回放用于场景测试 — 业界广泛采用
- World model 短时域视觉预测（3-5 秒）

**尚未解决的：**
- 复杂驾驶策略的零样本迁移（多车道、交叉路口）— 仍需真车微调
- 长时域 world model rollout（> 10 秒）— 误差累积
- 交互行为的迁移 — 其他智能体在 sim 和 real 中行为根本不同
- 保证安全覆盖 — 没有方法能证明策略处理了所有真实世界极端情况

**开放问题：**
- 如何在真实部署前量化"还剩多少差距"
- 自动选择哪些参数该随机化 vs 哪些该标定
- 结合 world model 与显式物理（混合仿真）
- 让 world model 能处理它从未观察到的稀有/危险场景
- 已迁移策略的形式化验证，用于安全认证

## 与本项目的关系

```
你正在学的                    →  工业界怎么用
──────────                       ──────────
PPO clip mechanism            →  限制策略更新幅度，保护 IL 预训练
SAC off-policy buffer         →  混合 IL 数据 + RL 交互数据
Reward shaping                →  多分量加权 reward 设计
gamma / GAE                   →  长期回报 vs 短期安全的权衡
Domain randomization（本文档中介绍） →  鲁棒的 sim-to-real 迁移
```

PPO 和 SAC 是端到端方案中 RL 微调阶段的主力算法。学完本项目的基础内容，再进入端到端方向会非常自然。
