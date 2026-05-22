# 方向盘零偏补偿算法——详细实现规格

本文档基于精简设计方案，按运行时序逐步描述零偏补偿的完整算法流程，包含所有公式及其推导。开发者可直接据此实现代码。

---

## 1. 符号定义

| 符号 | 代码变量 | 单位 | 含义 |
|------|----------|------|------|
| δ_f | `measures_ptr_->wheel_angle_` | rad | 前轮转角传感器读数（含零偏） |
| x | `offset()` | rad | KF 状态：前轮转角零偏估计值 |
| r | `ang_vel_b_.z()` | rad/s | IMU 测量的横摆角速度（yawrate） |
| v | `vel_ego_` | m/s | 车辆纵向速度 |
| L | `vehicle_params_.wheelbase` | m | 轴距（前后轴中心距） |
| SR | `get_steer_ratio()` | - | 转向系统传动比（方向盘角/前轮转角） |
| Q | `offset_model_error_variance` | rad² | 零偏过程噪声方差 |
| R | `yawrate_measure_variance` | (rad/s)² | yawrate 观测噪声方差 |
| P | `variance()` | rad² | 零偏估计的协方差（标量） |
| H | — | 1/s | 观测矩阵（标量），= -v/L |
| P_thres | `var_valid_threshold` | rad² | 协方差有效性阈值 |
| bias_limit | `bias_limit` | rad | 补偿限幅（±12°） |

---

## 2. 物理模型推导

### 2.1 运动学模型

自行车运动学模型中，稳态 yawrate 与前轮转角的关系：

```
r = v / L · tan(δ)
```

小角度近似（δ < 0.2 rad ≈ 11.5° 时误差 < 1.4%）：

```
r ≈ v / L · δ
```

### 2.2 引入零偏

传感器读数 δ_f 含零偏 δ_0：

```
δ_真实 = δ_f - δ_0
```

代入运动学模型：

```
r = v / L · (δ_f - δ_0)
```

**核心思想：** 已知 r（IMU）、v（轮速）、δ_f（转角传感器）、L（轴距），唯一未知量是 δ_0。直接求解：

```
δ_0 = δ_f - r · L / v
```

单帧即可计算，但受噪声影响大。用 KF 多帧滤波获得稳定估计。

### 2.3 KF 建模（单状态标量 KF）

**状态：** x = δ_0（标量，前轮转角零偏）

**过程模型（随机游走）：**
```
x(k+1) = x(k) + w    w ~ N(0, Q)
```

零偏建模为缓变常值，Q 取极小值表示几乎不变。

**观测模型：**
```
观测值:   z = r_measured（IMU yawrate）
预测观测: ẑ = v/L · (δ_f - x)
```

将观测模型线性化（对状态 x 求偏导）：
```
z = v/L · (δ_f - x) + noise
  = v/L · δ_f - v/L · x + noise
```

写成标准形式 z = H·x + h(δ_f) + noise：
```
H = ∂ẑ/∂x = -v/L    （标量）
h = v/L · δ_f        （已知量，不依赖状态）
```

注：由于观测模型是线性的，无需 EKF，直接用标准 KF 即可。

### 2.4 为什么单状态也要用 KF（而非简单滤波器）

单状态 KF 并非"大材小用"，而是这个问题最简洁的正确表达。相比手写低通/指数滑动平均，KF 在几乎零额外成本下提供三个关键能力：

**1. 自适应增益（先快后慢）**

低通滤波器增益固定。KF 增益随协方差自动调节：
- 初始 P 大 → K 大 → 快速跟踪真值
- 收敛后 P 小 → K 小 → 强力抗噪声

等效地实现了"初期快速收敛 + 稳态重度低通"，无需手动分段设计。

**2. 协方差跟踪（免费的可信度指标）**

P 直接表征"当前估计值有多可信"。`output_enabled` 逻辑（P < 阈值才启用补偿）完全基于此。若用简单滤波器，需要额外设计一套收敛判定逻辑；KF 天然自带。

**3. 物理意义明确的调参**

- Q = "零偏多快会变"（可从物理先验设定）
- R = "yawrate 噪声多大"（可从 IMU datasheet 获取）

两个参数直接对应物理量。低通滤波器的截止频率没有这种直观对应关系。

**实际代码量（完整的单状态 KF 核心）：**

```
P_prior = P + Q;
S = H * H * P_prior + R;
K = P_prior * H / S;
x = x + K * y;
P = (1 - K * H) * P_prior;
```

5 行标量运算。比手写一个"带自适应增益 + 收敛判定"的低通滤波器还简单。

---

## 3. 运行时序（每个控制周期）

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: 判断是否更新 KF（Preprocess 阶段）                  │
│  STEP 2: KF 先验更新（Predict）                              │
│  STEP 3: KF 后验更新（Update）+ 有效性判定                   │
│  STEP 4: 补偿值计算（Control Loop 阶段）                     │
│  STEP 5: 前馈补偿应用                                        │
│  STEP 6: 下游变量赋值                                        │
└─────────────────────────────────────────────────────────────┘
```

---

### STEP 1: 判断是否更新 KF（Update Gate）

KF 仅在运动学模型有效的工况下更新，否则冻结（保持 x, P 不变）。5 条 gate 必须全部满足：

```
输入:
  v          = measures_ptr_->vel_ego_
  δ_f        = measures_ptr_->wheel_angle_
  r          = ang_vel_b_.z()
  δ_f_last   = 上一周期的 δ_f

Gate 条件（全部满足才更新）:
  1. v >= velocity_threshold (5.0 m/s)         — 低速时 H≈0，观测信息不足
  2. |δ_f| <= max_wheel_angle (0.1 rad)        — 小角度近似有效范围
  3. |r| <= max_yawrate (0.1 rad/s)            — 排除急转弯
  4. |v × r| <= max_lateral_accel (1.0 m/s²)   — 排除高侧向加速度
  5. |δ_f - δ_f_last| / dt <= max_wheel_angle_rate (0.1 rad/s) — 近稳态

if 任一条件不满足:
    冻结 x, P（跳过 STEP 2-3）
    继续累加 valid_duration（见下文）
    return
```

**冻结而非 reset 的原因：** 零偏是传感器物理常量，不会因工况变化而改变。冻结保持估计值连续不跳变。

**有效输出判定（valid_duration）：**

补偿不会在 KF 刚收敛时立即启用，而是要求收敛状态持续存在 `min_valid_duration`（2.0s）后才启用输出：

```
每个周期末（无论 gate 是否通过）:
  if is_converged() (P < var_valid_threshold):
      valid_duration += dt
  else:
      valid_duration = 0     （仅在 gate 通过且 P 发散时重置）

  if valid_duration >= min_valid_duration:
      output_enabled = true  （仅由 Reset() 清除）
```

**设计意图：** `valid_duration` 衡量的是"收敛后估计值稳定存在的时长"，不是"连续满足 gate 的时长"。冻结期间估计值不变（即稳定），因此冻结周期也计入。`output_enabled` 一旦置真，仅由 Reset() 清除——零偏一旦被准确估计，除非系统重置，没有理由撤回补偿。

---

### STEP 2: KF 先验更新（Predict）

```
状态预测（零偏不变）:
  x⁻ = x⁺(上一周期)

协方差预测:
  P⁻ = P⁺ + Q
```

只有一个标量状态，预测步极简。Q 取极小值（如 5e-11），表示每步零偏几乎不变，但允许协方差缓慢增长以维持滤波器对新观测的响应能力。

---

### STEP 3: KF 后验更新（Update）

所有变量均为标量。

**观测矩阵：**
```
H = -v/L
```

**计算观测预测：**
```
ẑ = v/L · (δ_f - x⁻) = H · x⁻ + v/L · δ_f
```

**创新（残差）：**
```
y = r_measured - ẑ = r_measured - v/L · (δ_f - x⁻)
```

**创新协方差：**
```
S = H² · P⁻ + R = (v/L)² · P⁻ + R
```

**卡尔曼增益：**
```
K = P⁻ · H / S = P⁻ · (-v/L) / [(v/L)² · P⁻ + R]
```

注：K 的符号为负（因为 H < 0）。

**后验状态：**
```
x⁺ = x⁻ + K · y
```

**后验协方差：**
```
P⁺ = (1 - K · H) · P⁻ = (1 + K · v/L) · P⁻
```

注：K < 0, v/L > 0, 所以 K·v/L < 0, 因此 P⁺ < P⁻（协方差在更新后减小，符合预期）。

**收敛判定：** 当 P⁺ < P_thres 时，KF 视为已收敛。P_thres = 2e-8 rad²（对应标准差 ≈ 0.14 mrad ≈ 0.008°）。注意：收敛不等于立即可输出补偿——还需要 `valid_duration >= min_valid_duration`（见 STEP 1）。

**物理直觉：**
- 若 `r_measured > ẑ`（实测 yawrate 比预测大），说明实际转角比 `δ_f - x` 大
- K < 0, y > 0 → K·y < 0 → x⁺ < x⁻（零偏估计减小）
- x 减小 → 真实转角估计 `δ_f - x` 增大 → 下次预测 yawrate 更大 → 残差减小

**为什么不需要额外低通滤波：** Q/R 比值（5e-11 / 2e-6 ≈ 2.5e-5）极小，决定了 KF 本身的等效带宽远低于 0.25 Hz。KF 输出已经是重度低通，再接 Butterworth 滤波器是冗余的。

---

### STEP 4: 补偿值计算

```
输入:
  x = KF 状态（零偏估计，rad，前轮转角级）
  output_enabled = 有效输出标志（见 STEP 1 的 valid_duration 逻辑）
  SR = get_steer_ratio(steering_angle_)  (转向传动比，约 13.5)
  bias_limit = lat_steering_angle_bias_limit_  (rad, 约 0.209 rad = 12°)

计算:
  if output_enabled:    （KF 已收敛且稳定持续 min_valid_duration）
      steering_angle_bias_ = -x × SR
      steering_angle_dist_ = Clamp(steering_angle_bias_, -bias_limit, +bias_limit)
  else:
      steering_angle_bias_ = 0
      steering_angle_dist_ = 0
```

**符号说明：**
- `x` 是前轮转角级的零偏（如 +0.01 rad 表示传感器读数偏大 0.01 rad）
- 乘以 `-SR` 转换为方向盘角度级，负号表示"补偿方向与偏差相反"
- Clamp 防止极端估计值导致大幅转向跳变

---

### STEP 5: 前馈补偿应用

```
输入:
  steering_angle_cmd_ = MPC输出经转向比和低通后的方向盘指令 (rad)
  steering_angle_dist_ = STEP 4 计算的补偿量 (rad)

补偿:
  final_cmd = steering_angle_cmd_ - steering_angle_dist_
```

**物理含义：**
- 若零偏 `x > 0`（传感器读数偏大）
- → `steering_angle_bias_ = -x × SR < 0`
- → `steering_angle_dist_ < 0`
- → `final_cmd = cmd - (负值) = cmd + |dist|`（增大指令）
- 这是正确的：传感器偏大 → 实际转角比读数小 → 需要多打方向盘

---

### STEP 6: 下游变量赋值

```
// 供 MPC init state 和曲率估计使用
measures_ptr_->wheel_angle_bias_ = steering_angle_dist_ / SR

// 供 strategy 层输出给 planning
control_output_ptr_->steering_angle_bias_ = steering_angle_dist_
```

---

## 4. Reset 场景处理（Auto 接管第一帧）

```
触发条件: LateralReset()（auto 接管/模式切换）

处理:
  if KF 已收敛（P < P_thres）:
      wheel_angle_cmd_ = δ_f - x
  else:
      wheel_angle_cmd_ = δ_f
  last_wheel_angle_cmd_ = wheel_angle_cmd_
```

**原因：** `wheel_angle_cmd_` 会被用作 MPC 的 `init_state_.delta_`。若不修正，MPC 第一帧从含零偏的 delta 开始递推，引入初始误差。

---

## 5. MPC 初始状态中的使用

```
触发条件: calc_mpc_init_state()（每帧）

处理:
  wheel_angle = δ_f（传感器读数）

  // 零偏修正：加上等效前轮转角补偿
  if KF 已收敛  AND  模式 != APA:
      wheel_angle += steering_angle_dist_ / SR

  // 用修正后的 wheel_angle 计算延迟补偿
  angular_velocity = v × curv_factor × wheel_angle
  dx   = sin(ω·Td) × v / ω
  dy   = v/ω × (1 - cos(ω·Td))
  dphi = ω × Td

  // MPC 初始转角状态（用上一帧输出，不含零偏）
  init_state_.delta_ = Limit(wheel_angle_cmd_, delta_limit)
```

---

## 6. 完整伪代码

```python
# ═══════════════ 每个控制周期 ═══════════════

# 持久状态（跨周期保持）
x = 0.0                # KF 状态：零偏估计 (rad)
P = 1e-4               # KF 协方差 (rad²)，初始值取较大以快速收敛
last_wheel_angle = 0.0 # 上一周期前轮转角（用于计算变化率）
output_enabled = False  # 补偿输出使能（仅由 Reset 清除）
valid_duration = 0.0   # 收敛后稳定存在的累计时长 (s)


# ─── Preprocess 阶段 ───

def UpdateOffsetKalmanFilter():
    v = measures.vel_ego
    delta_f = measures.wheel_angle
    r_measured = measures.ang_vel_b_z

    # STEP 1: Update Gate（5 条全部满足才更新）
    gate_pass = (
        v >= velocity_threshold and
        abs(delta_f) <= max_wheel_angle and
        abs(r_measured) <= max_yawrate and
        abs(v * r_measured) <= max_lateral_accel and
        abs(delta_f - last_wheel_angle) / dt <= max_wheel_angle_rate
    )
    last_wheel_angle = delta_f

    if not gate_pass:
        # 冻结 x, P；但继续累加 valid_duration（估计值稳定不变）
        if P < P_thres:
            valid_duration += dt
        if valid_duration >= min_valid_duration:
            output_enabled = True
        return

    # STEP 2: Predict
    x_prior = x
    P_prior = P + Q

    # STEP 3: Update
    H = -v / L
    r_predicted = v / L * (delta_f - x_prior)
    y = r_measured - r_predicted
    S = H * H * P_prior + R
    K = P_prior * H / S

    x = x_prior + K * y
    P = (1 - K * H) * P_prior

    # 有效性跟踪
    if P < P_thres:
        valid_duration += dt
    else:
        valid_duration = 0.0

    if valid_duration >= min_valid_duration:
        output_enabled = True


# ─── Control Loop 阶段 ───

def LateralActuatorControl():
    # MPC 输出 → 转向比 → 低通
    steering_cmd = lowpass(wheel_angle_cmd * steer_ratio)

    # STEP 4: 补偿值计算
    SR = get_steer_ratio(measures.steering_angle)

    if output_enabled:  # KF 已收敛且稳定持续 min_valid_duration
        steering_angle_bias = -x * SR
        steering_angle_dist = clamp(steering_angle_bias, -bias_limit, +bias_limit)

        # STEP 5: 前馈补偿
        final_cmd = steering_cmd - steering_angle_dist

        # STEP 6: 下游赋值
        output.steering_angle_cmd = final_cmd
        output.steering_angle_bias = steering_angle_dist
        measures.wheel_angle_bias = steering_angle_dist / SR
    else:
        output.steering_angle_cmd = steering_cmd
        output.steering_angle_bias = 0
        measures.wheel_angle_bias = 0
```

---

## 7. 参数配置表

### 7.1 车辆物理参数

| 参数 | 含义 | 单位 | Vehicle A | Vehicle B | 来源 |
|------|------|------|-----------|-----------|------|
| `wheelbase` | 轴距 | m | 2.9 | 3.216 | 可直接测量 |
| `steer_ratio` | 转向传动比 | - | 13.5 | 13.5 | 车辆物理结构 |

### 7.2 KF 噪声参数（工程调参）

| 参数 | 含义 | 单位 | 建议值 | 调参建议 |
|------|------|------|--------|----------|
| `offset_model_error_variance_` | Q，零偏随机游走方差 | rad² | 5e-11 | 增大 → 跟踪更快但噪声大 |
| `yawrate_measure_variance_` | R，观测噪声方差 | (rad/s)² | 2e-6 | IMU 精度相关，减小 → 更信任观测 |

### 7.3 有效性判定参数

| 参数 | 含义 | 单位 | 值 | 调参建议 |
|------|------|------|-----|----------|
| `var_valid_threshold` | P 收敛阈值 | rad² | 2e-8 | 减小 → 更保守，需更长收敛时间 |

### 7.4 系统设计参数

| 参数 | 含义 | 单位 | 值 | 来源 |
|------|------|------|-----|------|
| `velocity_threshold` | KF 更新最低速度 | m/s | 5.0 | 低速时 H≈0，观测无意义 |
| `max_wheel_angle` | 更新门控：最大前轮转角 | rad | 0.1 (~5.7°) | 小角度近似有效范围 |
| `max_yawrate` | 更新门控：最大横摆角速度 | rad/s | 0.1 (~5.7°/s) | 排除急转弯 |
| `max_lateral_accel` | 更新门控：最大横向加速度 | m/s² | 1.0 | 排除高动态工况 |
| `max_wheel_angle_rate` | 更新门控：最大转角变化率 | rad/s | 0.1 | 近稳态约束 |
| `min_valid_duration` | 收敛后稳定存在的最短时长 | s | 2.0 | 防止初始瞬态输出 |
| `dt` | 控制周期 | s | 0.02 (50Hz) | 系统时钟 |
| `bias_limit` | 补偿限幅 | rad（≈12°） | 12.0/57.3 | 安全限幅 |

### 7.5 方案对比

| 维度 | 运动学标量 KF（本方案） | 动力学 KF（2 状态） | DOB | yawrate≈0 时直接读转角 |
|------|------------------------|---------------------|-----|----------------------|
| 车型移植 | 改一个轴距 L | 需辨识 w1/w2/w3 | 需整定带宽/增益 | 无参数 |
| 可解释性 | 完全透明 | 较透明 | 中等 | 最直观 |
| 数学最优性 | 线性高斯下最优 | 同左 | 非最优 | 非最优（单帧噪声大） |
| 参数数量 | 2（Q, R） | 5+ | 3-4 | 0-1（阈值） |
| 抗噪声能力 | 强（多帧融合） | 强 | 中等 | 弱（单帧或少帧） |
| KF 状态 | [offset]（标量） | [yawrate, offset]（2 维） | N/A | N/A |
| 噪声参数 | 2 个（Q, R） | 3 个（σ²_r, σ²_δ0, σ²_z） | N/A | N/A |
| 低速行为 | 显式冻结（gate 拒绝） | 隐式无效（K→0） | 需衰减/关闭 | 同样无法工作 |

**瞬态响应差异的本质：**

两种 KF 的 offset 状态都是随机游走模型（x(k+1) = x(k) + w），单帧信息量相同时响应速度由 Q 决定。动力学 KF 看起来"更快"的原因不是每步增益更大，而是：
- 动力学 KF 建模了阻尼/响应特性，转弯时仍能将残差分配给 offset 状态继续更新
- 运动学 KF 在转弯时严格冻结（gate 拒绝）
- 即"更少冻结 → 累积更新更多 → 实际收敛更快"，而非"单帧信息量更大"

**低速行为的本质：**

所有方案在低速时都无法有效估计零偏，物理原因相同：offset 通过 yawrate 被观测（r ≈ v/L × offset），当 v → 0 时 offset 产生的信号淹没在 IMU 噪声里。
- 动力学 KF：不显式冻结，但 Kalman 增益趋近零——更新了也等于没更新
- 运动学 KF：显式冻结，明确表达"此时无观测信息"
- 运动学 KF 的冻结是更诚实的工程处理，避免了"系统在工作"的虚假印象

**"yawrate≈0 时直接读转角"方法的局限：**

这是最朴素的零偏估计思路（直行时 r≈0，δ_sensor ≈ 零偏），但存在根本缺陷：
- "r≈0" 的判断受 IMU 噪声和路面坡度影响，阈值难以设定
- 横坡产生的稳态 yawrate 会被误判为"直行"，导致错误估计
- 只能在极少帧采样，无法多帧融合降噪，估计方差大
- 没有协方差跟踪，不知道估计值是否可信
- KF 方案本质上是这个思路的数学最优推广：把"r≈0 时读转角"泛化为"所有低动态帧都贡献观测信息，按信噪比加权融合"

**本方案的优势总结：**

核心不是"估得更准"，而是"更简单、更鲁棒、更易移植、不会出错"：
- 冻结而非 reset：不满足观测条件时保持估计值不变，避免补偿跳变
- Q/R 比值本身就是低通：不需要额外滤波器，减少延迟和调参自由度
- Gate 保证模型有效性：只在运动学模型准确的工况下更新，避免模型失配时强行拟合
- 对于缓变物理量（零偏），上述工程鲁棒性比极致性能更有价值

**诚实地说，劣势是：**

- 对侧风/横坡等瞬态扰动无能为力（靠闭环吸收）
- gate 拒绝率高于动力学 KF → 相同时长内可用更新帧更少 → 实际收敛稍慢
- 低速转弯场景完全冻结（动力学 KF 至少还有微弱的更新能力）

---

## 8. 收敛特性分析

**稳态卡尔曼增益：** 当 P 收敛到稳态 P_ss 时：

```
P_ss + Q = P_ss / (1 - K_ss · H)
→ 需解代数 Riccati 方程（标量情况直接求解）

设 P⁻ = P_ss + Q，则：
K_ss = P⁻ · H / (H² · P⁻ + R)
P_ss = (1 - K_ss · H) · P⁻
```

**估计精度与速度的关系：**
- H = -v/L，速度越高，|H| 越大，观测信息量越多
- 高速时 KF 收敛更快，稳态 P 更小
- 低速时信息量不足（这正是设置 v_thres 的原因）

**典型收敛时间：** v=20 m/s, L=2.9 m 时，约 1-2 秒收敛到 P < P_thres。

**对突变零偏的响应：** Q 极小使得 KF 对突变响应慢（约 5-10 秒）。若需更快响应可适当增大 Q，代价是稳态噪声增大。
