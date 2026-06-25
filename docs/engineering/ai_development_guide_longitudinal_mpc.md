# AI 开发能力指南：纵向控制 MPC 示例

结论：AI 开发能力不是“会不会让 AI 写代码”，而是能不能把 AI 当成工程协作者，稳定地产出正确、可维护、可验证的代码。

这份文档以**自动驾驶纵向控制 MPC 算法骨架**为例，说明如何把需求边界、Prompt 约束、代码审查和验证流程讲清楚。

## 核心能力

### 1. 需求拆解

使用 AI 开发前，先明确：

- 输入、输出
- 单位、符号和边界条件
- 速度、加速度、jerk 限制
- 非法输入、依赖失败、限幅触发和安全兜底策略
- 需要哪些测试
- 哪些地方不能让 AI 自由发挥

差的做法：

> 直接让 AI：帮我写一个自动驾驶纵向控制算法。

好的做法：

> 我需要开发一个自动驾驶纵向控制 MPC 算法骨架，用于根据车辆当前速度、参考速度曲线和约束条件输出期望加速度命令。要求明确输入输出、单位、预测时域、加速度/减速度限制、jerk 限制、速度上下限、参考异常、停车处理和控制失败兜底策略。先不要接入真实车辆或底盘接口，只完成算法骨架、仿真验证和单元测试。

### 2. Prompt 约束

好的 Prompt 应该包含：上下文、约束、验收标准、不要做什么、输出步骤。

示例：

```text
请在现有 C++ 控制模块中实现一个简化版自动驾驶纵向控制 MPC 算法骨架 computeLongitudinalMpcCommand()。
要求：
1. 输入状态、参考速度曲线和配置的单位必须明确：m/s、m/s^2、m/s^3、second
2. 输出 targetAccelerationMps2 和 targetSpeedMps
3. 限制加速度、减速度、jerk 和速度上下限
4. 不接入真实底盘接口，不引入新依赖
5. MPC/QP 求解细节可以用 solver 抽象和注释表示
6. 补充 gtest，覆盖正常路径、非法配置、短参考、速度异常、solver 失败、NaN/inf、限幅
7. 先说明计划，再给代码改动
```

### 3. 审查与验证

AI 生成代码后重点检查：

- 是否符合需求
- m/s、m/s^2、m/s^3、second 等单位是否正确
- 加速度/减速度符号是否一致
- jerk 限制是否作用在相邻周期加速度变化上
- 是否放宽安全边界、限幅或故障处理
- 是否引入不必要依赖或接入真实底盘接口

推荐闭环：

```text
需求 → 计划 → 生成 → 审查 diff → 补测试 → 运行测试 → 修复 → 再验证
```

常用约束：

```text
只做最小改动。
不要重构无关代码。
不要引入新依赖。
不要接入真实车辆或底盘接口。
不要把纵向 MPC 扩展成完整 AEB、ACC、速度规划器或横纵向联合控制系统。
```

## 示例：自动驾驶纵向控制 MPC 算法骨架

本文代码不是完整 MPC 求解器，只展示 AI 开发时如何约束需求、设计算法边界、校验输入、处理失败和验证限幅。真实 MPC 还需要状态空间模型、代价函数、约束矩阵和数值求解器；这些细节可以先用注释或 fake solver 代替。

### 需求边界

可以这样要求 AI：

```text
请用 C++ 实现一个简化版自动驾驶纵向控制 MPC 算法骨架。

要求：
1. 输入 LongitudinalState：speedMps、accelerationMps2、positionS
2. 输入 SpeedReference：未来 N 个点的 targetSpeedMps、targetAccelerationMps2
3. 输入 MpcConfig：horizon、dt、maxAccelMps2、maxDecelMps2、maxJerkMps3、minSpeedMps、maxSpeedMps
4. 输出 MpcCommand：targetAccelerationMps2、targetSpeedMps、isValid、failureReason
5. maxDecelMps2 使用正数表示最大减速度幅值，实际加速度下界为 -maxDecelMps2
6. 预留纵向运动学模型和 MPC 预测问题构建位置，具体 QP 求解细节用注释说明
7. solver 保持可替换，不实现真实 QP 求解器
8. 检查非法配置、短参考、速度异常、NaN/inf、solver 失败
9. 输出做加速度限幅、jerk 限幅和速度上下限保护
10. 补充单元测试
```

如果 AI 计划接入 OSQP/IPOPT、实现 AEB/ACC/速度规划器、接入真实底盘接口或在线估计车辆质量/坡度，应立即要求删除这些内容。

### 输入输出边界

| 类别 | 字段 | 单位 | 说明 |
|---|---|---|---|
| State | `speedMps` | m/s | 当前车速 |
| State | `accelerationMps2` | m/s² | 当前加速度 |
| State | `positionS` | m | 路径纵向位置；本示例暂不使用 |
| Reference | `targetSpeedMps` | m/s | 参考速度 |
| Reference | `targetAccelerationMps2` | m/s² | 参考加速度 |
| Config | `horizon`, `dt` | step, s | 预测步长和周期 |
| Config | `maxAccelMps2` | m/s² | 最大加速度 |
| Config | `maxDecelMps2` | m/s² | 最大减速度幅值，使用正数表示 |
| Config | `maxJerkMps3` | m/s³ | 最大 jerk 幅值 |
| Config | `minSpeedMps`, `maxSpeedMps` | m/s | 速度上下限 |
| Output | `targetAccelerationMps2` | m/s² | 输出期望加速度 |
| Output | `targetSpeedMps` | m/s | 预测下一周期速度 |
| Output | `isValid`, `failureReason` | - | 命令是否有效及失败原因 |

### 伪代码骨架

```text
compute_longitudinal_mpc_command(state, reference, config, previous_accel, solver):
    validate config: horizon, dt, accel/decel_limit, jerk_limit, speed_limits
    validate state: speed is finite and non-negative
    validate reference: length >= horizon
    validate previous_accel is finite

    # solver 代表真实 MPC 优化求解部分。
    # 本文不展开模型、代价函数、约束矩阵和 QP 求解器。
    raw_accel = solver.solve_first_acceleration(state, reference, config)

    if solver failed or raw_accel is NaN/inf:
        return invalid("solver failed")

    previous = clamp(previous_accel, -max_decel, max_accel)
    accel = clamp(raw_accel, -max_decel, max_accel)
    accel = clamp_jerk(accel, previous, max_jerk * dt)

    # 用速度上下限反向约束加速度，避免只 clamp predicted_speed。
    accel = clamp_to_keep_speed_in_bounds(accel, state.speed, min_speed, max_speed, dt)
    accel = clamp(accel, -max_decel, max_accel)

    predicted_speed = clamp(state.speed + accel * dt, min_speed, max_speed)
    return valid_command(accel, predicted_speed)
```

### 审查重点

- `maxDecelMps2` 是否始终按正数幅值理解，实际下界是 `-maxDecelMps2`。
- 是否检查非法配置、短参考、速度异常、NaN/inf、solver 失败。
- jerk 限制是否作用在相邻周期加速度变化上。
- 速度上下限是否反向约束加速度，而不是只修改 `predictedSpeed`。
- 是否避免接入真实底盘接口、引入新依赖或扩展到 AEB/ACC/速度规划器。

### 测试重点

至少覆盖：

- 正常输入返回 valid command
- 非法 `horizon`、`dt`、加速度/减速度限制、jerk 限制、速度上下限
- 短参考、当前速度异常、上一周期加速度异常、NaN/inf
- solver 失败或返回 NaN/inf
- 加速度限幅、减速度限幅、jerk 限幅
- 上一周期加速度已超限时，最终输出仍满足绝对限幅
- 预测速度超过上下限时，通过反向约束加速度保护

可以要求 AI：

```text
请补 gtest 单元测试。
使用 fake solver 控制输出，不依赖真实 QP 求解器。
每个测试明确验证 isValid、failureReason、targetAccelerationMps2 和 targetSpeedMps。
```

## 正确认知

> 我不会把 AI 当成自动写代码机器，而是把它当成 junior engineer。  
> 我负责需求边界、架构判断、代码审查和验证；AI 负责加速生成草稿、补测试、解释日志和搜索上下文。

## 总结

AI 开发能力 = 需求控制 + Prompt 约束 + 工程判断 + 测试验证 + 安全意识。
