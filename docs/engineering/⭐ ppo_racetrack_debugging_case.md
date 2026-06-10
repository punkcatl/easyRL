# PPO Racetrack Debugging Case Study

A complete debugging record of training PPO for continuous lateral steering control on racetrack-v0.

## 1. Failed Training Symptoms

Initial configuration:

- Environment: `racetrack-v0`, lateral-only continuous control
- Observation: `[x, y, vx, vy, heading]`, 5 vehicles
- Network: 128 hidden units, ReLU, `log_std=0` (std=1.0)
- Update: once per episode
- Reward: `action_reward=-0.3` (penalizes steering magnitude)

**Result**: Agent drives straight, exits track at every curve. After 500 episodes, avg reward shows no improvement trend.

## 2. Diagnosis Process

Four issues identified through systematic elimination:

### 2.1 Insufficient Samples

Racetrack episodes are very short (terminated immediately on off-track). Per-episode updates yield only tens of steps — far too few for stable GAE estimation.

→ High variance in advantage estimates, noisy policy gradients.

### 2.2 Observation Lacks Curvature Information

`[x, y, vx, vy, heading]` are global coordinates with no lane-relative information. The agent cannot perceive "there's a curve ahead" or "I'm drifting from center."

→ Impossible to learn when to steer.

### 2.3 Reward Conflicts with Task

`action_reward=-0.3` penalizes steering actions. But curves **require** steering — the reward signal teaches the agent not to turn.

→ Reward contradicts the task objective.

### 2.4 log_prob Inconsistency Bug (Core Issue)

- Collection: `action_raw` → `clamp(-1,1)` → store **clamped** action in buffer
- Update: `evaluate(states, actions_clamped)` computes log_prob on clamped values
- Problem: clamp maps different raw actions to the same boundary value, but `Normal.log_prob(clamped)` ≠ the original `log_prob(raw)` at collection time

→ **Importance ratio is computed incorrectly**, PPO's clip mechanism breaks silently.

## 3. Fix

| Problem | Solution |
|---------|----------|
| Insufficient samples | Per-episode → **rollout buffer (2048 steps)** then update |
| No curvature info | Observation → `[lat_off, ang_off, vx, vy, cos_h, sin_h]` |
| Reward conflict | `action_reward=0`, `lane_centering_cost=8`, `collision_reward=-5` |
| log_prob bug | Buffer stores **action_raw** (pre-clamp), ensuring consistent log_prob basis |
| Over-exploration | `log_std` init -0.5 (std≈0.6), most samples fall within [-1,1] |
| Oversized network | 64 hidden + Tanh (bounded activation suits small input/output) |
| Slow Critic convergence | Critic lr = Actor lr × 3 |

## 4. Reproduction

Final config:

```python
# algorithms/ppo/config.py
{
    "n_episodes": 5000,
    "lr": 3e-4,           # Critic auto ×3
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.005,
    "epochs": 10,
    "batch_size": 64,
    "hidden_dim": 64,
    "rollout_steps": 2048,
}
```

Environment config:

```python
"features": ["lat_off", "ang_off", "vx", "vy", "cos_h", "sin_h"],
"action": {"type": "ContinuousAction", "longitudinal": False, "lateral": True},
"action_reward": 0.0,
"lane_centering_cost": 8,
"collision_reward": -5,
```

Run:

```bash
python algorithms/ppo/train.py
```

**Expected**: Avg reward begins rising around episode 200-500. Agent progressively learns to follow curves instead of driving straight off-track.

## 5. Key Lesson

> **log_prob consistency** is the most insidious PPO bug — an incorrect ratio throws no exception and only manifests as "training doesn't work." It's nearly impossible to diagnose from reward curves alone.
>
> **Check**: ensure the action stored in buffer and the action used to compute log_prob are the same value (both pre-clamp OR both post-clamp — never mixed).

## 6. Debugging Checklist for PPO

When PPO fails to train, check in this order:

1. **Sample volume** — is each update batch large enough? (rollout buffer ≥ 1024 steps)
2. **Observation expressiveness** — does the obs contain the information needed for the task?
3. **Reward consistency** — does the reward conflict with desired behavior?
4. **log_prob consistency** — are buffer actions and evaluate actions on the same basis?
5. **Exploration range** — is initial std appropriate for the action space?

---

# PPO 赛道控制调试案例

在 racetrack-v0 环境中训练 PPO 进行连续横向转向控制的完整调试记录。

## 一、失败训练现象

初始配置：

- 环境：`racetrack-v0`，仅横向连续控制
- 观测：`[x, y, vx, vy, heading]`，5辆车
- 网络：128 hidden, ReLU, `log_std=0`（std=1.0）
- 更新方式：每个 episode 结束后更新一次
- 奖励：`action_reward=-0.3`（惩罚转向幅度）

**表现**：agent 始终走直线，入弯即出界。训练 500 episode 后 avg reward 无改善趋势。

## 二、定位过程

通过逐层排查定位四个问题：

### 2.1 样本量不足

racetrack episode 很短（出界即终止），per-episode 更新只有几十步数据，远不够 GAE 稳定估计。

→ 优势估计高方差，策略梯度噪声过大。

### 2.2 观测缺乏弯道信息

`[x, y, vx, vy, heading]` 是全局坐标，不包含车道相对信息。agent 无法感知"前方是弯道"或"我偏离了中心多少"。

→ 无法学到转弯时机。

### 2.3 奖励与任务矛盾

`action_reward=-0.3` 惩罚转向动作本身。但弯道**必须**转向——奖励信号在"教导"agent 不要转弯。

→ 奖励与任务目标冲突。

### 2.4 log_prob 一致性 bug（核心问题）

- 采集时：`action_raw` → `clamp(-1,1)` → 存 **clamp 后** 的值到 buffer
- 更新时：`evaluate(states, actions_clamped)` 对 clamp 后的值算 log_prob
- 问题：clamp 在边界处把不同的 raw action 映射到相同值，但 `Normal.log_prob(clamp后)` ≠ 采集时的 `log_prob(raw)`

→ **importance ratio 计算错误**，PPO 的 clip 机制静默失效。

## 三、修复方案

| 问题 | 修复 |
|------|------|
| 样本不足 | per-episode → **rollout buffer（2048步）** 累积后更新 |
| 缺弯道信息 | 观测改为 `[lat_off, ang_off, vx, vy, cos_h, sin_h]` |
| 奖励矛盾 | `action_reward=0`, `lane_centering_cost=8`, `collision_reward=-5` |
| log_prob bug | buffer 存 **action_raw**（未截断），保证新旧 log_prob 基准一致 |
| 探索范围过大 | `log_std` 初始化 -0.5（std≈0.6），多数采样落在 [-1,1] |
| 网络过大 | 64 hidden + Tanh（有界激活适合小输入小输出） |
| Critic 收敛慢 | Critic lr = Actor lr × 3 |

## 四、复现实验

最终配置：

```python
# algorithms/ppo/config.py
{
    "n_episodes": 5000,
    "lr": 3e-4,           # Critic 自动 ×3
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.005,
    "epochs": 10,
    "batch_size": 64,
    "hidden_dim": 64,
    "rollout_steps": 2048,
}
```

环境配置：

```python
"features": ["lat_off", "ang_off", "vx", "vy", "cos_h", "sin_h"],
"action": {"type": "ContinuousAction", "longitudinal": False, "lateral": True},
"action_reward": 0.0,
"lane_centering_cost": 8,
"collision_reward": -5,
```

运行：

```bash
python algorithms/ppo/train.py
```

**预期表现**：约 200-500 episode 后 avg reward 开始上升，agent 逐步学会跟随弯道转向，不再走直线出界。

## 五、核心教训

> **log_prob 一致性**是 PPO 中最隐蔽的 bug——ratio 算错不会报异常，只会表现为"训练没效果"，仅从 reward 曲线几乎无法定位。
>
> **检查方法**：确保 buffer 中存的 action 和计算 log_prob 用的 action 是同一个值（都是截断前 OR 都是截断后——不能混用）。

## 六、PPO 调试排查清单

当 PPO 训练无效时，按此顺序排查：

1. **样本量** — 每次更新的 batch 是否足够大？（rollout buffer ≥ 1024 步）
2. **观测表达力** — obs 是否包含完成任务所需的信息？
3. **奖励一致性** — 奖励是否与期望行为矛盾？
4. **log_prob 一致性** — buffer 中的 action 和 evaluate 时的 action 基准是否一致？
5. **探索范围** — 初始 std 是否适合动作空间范围？
