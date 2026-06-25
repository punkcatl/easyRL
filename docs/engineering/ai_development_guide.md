# AI 开发能力指南

结论：AI 开发能力不是“会不会让 AI 写代码”，而是能不能把 AI 当成工程协作者，稳定地产出正确、可维护、可验证的代码。

## 核心能力

### 1. 需求拆解

使用 AI 开发前，先明确：

- 输入、输出
- 坐标系、单位、符号约定
- 边界条件、非法输入、依赖失败和安全兜底策略
- 性能、实时性和安全要求
- 需要哪些测试
- 哪些地方不能让 AI 自由发挥

差的做法：

> 直接让 AI：帮我写一个自动驾驶控制算法。

好的做法：

> 我需要开发一个自动驾驶横向控制算法，用于根据车辆当前状态和参考轨迹输出前轮转角命令。要求明确输入输出、坐标系、单位、车辆运动学假设、低速处理、轨迹异常、转角限幅、转角速率限制和控制失败兜底策略。先不要接入真实车辆或底盘接口，只完成算法骨架、仿真验证和单元测试。

### 2. Prompt 约束

好的 Prompt 应该包含：上下文、约束、验收标准、不要做什么、输出步骤。

示例：

```text
请在现有 C++ 控制模块中实现一个简化版自动驾驶横向控制 MPC 算法骨架 computeLateralMpcCommand()。
要求：
1. 输入状态、参考轨迹和配置的单位必须明确：rad、meter、second
2. 输出前轮转角 steerRad，单位 rad
3. 限制转角范围和转角速率
4. 不接入真实底盘接口，不引入新依赖
5. MPC/QP 求解细节可以用 solver 抽象和注释表示
6. 补充 gtest，覆盖正常路径、非法配置、短轨迹、低速、solver 失败、NaN/inf、限幅
7. 先说明计划，再给代码改动
```

### 3. 审查与验证

AI 生成代码后重点检查：

- 是否符合需求
- 单位、坐标系、符号方向是否正确
- 是否放宽安全边界、限幅或故障处理
- 是否引入不必要依赖或改动公共接口
- 是否有隐藏 bug、过度设计或无关重构

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
不要把横向 MPC 扩展成横纵向联合控制、轨迹规划器或完整车辆动力学标定系统。
```

## 示例：自动驾驶横向控制 MPC 算法骨架

本文代码不是完整 MPC 求解器，只展示 AI 开发时如何约束需求、设计算法边界、校验输入、处理失败和验证限幅。真实 MPC 还需要状态空间模型、代价函数、约束矩阵和数值求解器；这些细节可以先用注释或 fake solver 代替。

### 需求边界

可以这样要求 AI：

```text
请用 C++ 实现一个简化版自动驾驶横向控制 MPC 算法骨架。

要求：
1. 输入 LateralState：lateralErrorM、headingErrorRad、yawRateRadps、speedMps
2. 输入 ReferenceTrajectory：未来 N 个点的 curvature、targetHeadingRad
3. 输入 MpcConfig：horizon、dt、wheelBase、maxSteerRad、maxSteerRateRadps
4. 输出 MpcCommand：steerRad、isValid、failureReason
5. 预留线性化自行车模型和 MPC 预测问题构建位置，具体 QP 求解细节用注释说明
6. solver 保持可替换，不实现真实 QP 求解器
7. 检查非法配置、短轨迹、低速、NaN/inf、solver 失败
8. 输出做转角限幅和转角速率限幅
9. 补充单元测试
```

如果 AI 计划接入 OSQP/IPOPT、修改底盘接口、实现横纵向联合 MPC 或在线车辆参数辨识，应立即要求删除这些内容。

### 输入输出边界

| 类别 | 字段 | 单位 | 说明 |
|---|---|---|---|
| State | `lateralErrorM` | m | 车辆相对参考轨迹的横向误差 |
| State | `headingErrorRad` | rad | 航向误差 |
| State | `yawRateRadps` | rad/s | 横摆角速度 |
| State | `speedMps` | m/s | 当前车速 |
| Reference | `curvature` | 1/m | 参考轨迹曲率 |
| Reference | `targetHeadingRad` | rad | 参考航向 |
| Config | `horizon`, `dt` | step, s | 预测步长和周期 |
| Config | `wheelBase` | m | 车辆轴距 |
| Config | `maxSteerRad` | rad | 最大前轮转角幅值 |
| Config | `maxSteerRateRadps` | rad/s | 最大转角速率 |
| Output | `steerRad` | rad | 输出前轮转角命令 |
| Output | `isValid`, `failureReason` | - | 命令是否有效及失败原因 |

### 伪代码骨架

```text
compute_lateral_mpc_command(state, reference, config, previous_steer, solver):
    validate config: horizon, dt, wheel_base, steer_limit, steer_rate_limit
    validate state: speed is finite and non-negative
    validate reference: length >= horizon
    validate previous_steer is finite

    if speed is too low:
        return invalid("speed too low")
        # 示例策略：真实系统通常切换到低速/停车状态机

    # solver 代表真实 MPC 优化求解部分。
    # 本文不展开模型、代价函数、约束矩阵和 QP 求解器。
    raw_steer = solver.solve_first_steer(state, reference, config)

    if solver failed or raw_steer is NaN/inf:
        return invalid("solver failed")

    previous = clamp(previous_steer, -max_steer, max_steer)
    steer = clamp(raw_steer, -max_steer, max_steer)
    steer = clamp_rate(steer, previous, max_steer_rate * dt)
    steer = clamp(steer, -max_steer, max_steer)

    return valid_command(steer)
```

### 审查重点

- 代码是否把真实 MPC/QP 求解细节隔离在 `solver` 抽象里。
- 是否检查非法配置、短轨迹、低速、NaN/inf、solver 失败。
- 低速策略是否明确：本示例返回 invalid，真实系统应交给状态机处理。
- 限幅顺序是否安全：先绝对限幅，再速率限幅，最后再次保证绝对限幅。
- 是否避免接入真实底盘接口、引入新依赖或扩展到横纵向联合控制。

### 测试重点

至少覆盖：

- 正常输入返回 valid command
- 非法 `horizon`、`dt`、`wheelBase`、转角限制
- 短轨迹、低速、NaN/inf
- solver 失败或返回 NaN/inf
- 转角限幅、转角速率限幅
- 上一周期转角已超限时，最终输出仍满足绝对限幅

可以要求 AI：

```text
请补 gtest 单元测试。
使用 fake solver 控制输出，不依赖真实 QP 求解器。
每个测试明确验证 isValid、failureReason 和 steerRad。
```

## 正确认知

> 我不会把 AI 当成自动写代码机器，而是把它当成 junior engineer。  
> 我负责需求边界、架构判断、代码审查和验证；AI 负责加速生成草稿、补测试、解释日志和搜索上下文。

## 总结

AI 开发能力 = 需求控制 + Prompt 约束 + 工程判断 + 测试验证 + 安全意识。
