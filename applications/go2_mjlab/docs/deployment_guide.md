# 部署指南

## ONNX 模型规格

最终部署模型：`results/student_policy.onnx` (3.7KB)

- 输入：`obs_history` [batch, 20, 50] — 最近 20 帧观测
- 输出：`action` [batch, 12] — 关节位置偏移量，范围 [-1, 1]
- 目标关节位置：`default_pos + action_scale(0.4) * action`

## 在 Unitree Go2 上部署

```
传感器 → obs(50D) → 历史缓冲区 [20,50] → ONNX 推理 → action [12]
                                                          ↓
                        目标关节角 = 默认角度 + 0.4 * action
                                                          ↓
                              Unitree SDK → 12 个电机 (50Hz)
```

### 观测向量 (50D)

| 索引 | 名称 | 维度 | 来源 |
|------|------|------|------|
| 0-2 | base_lin_vel | 3 | IMU 速度计（机体坐标系） |
| 3-5 | base_ang_vel | 3 | IMU 陀螺仪（机体坐标系） |
| 6-8 | projected_gravity | 3 | 加速度计 → 机体坐标系下的重力向量 |
| 9-11 | command (vx, vy, wz) | 3 | 遥控器 / 自主规划器 |
| 12-13 | phase (sin, cos) | 2 | 时钟信号，周期=0.5s |
| 14-25 | joint_pos_rel | 12 | 编码器位置 - 默认位置 |
| 26-37 | joint_vel | 12 | 编码器速度 |
| 38-49 | last_action | 12 | 上一步输出的 action |

### 默认关节角度 (弧度)

```
FL_hip: -0.1,  FL_thigh: 0.9,  FL_calf: -1.8
FR_hip:  0.1,  FR_thigh: 0.9,  FR_calf: -1.8
RL_hip: -0.1,  RL_thigh: 0.9,  RL_calf: -1.8
RR_hip:  0.1,  RR_thigh: 0.9,  RR_calf: -1.8
```

### 控制循环 (Python 伪代码)

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("student_policy.onnx")
history = np.zeros((1, 20, 50), dtype=np.float32)
default_pos = np.array([-0.1, 0.9, -1.8, 0.1, 0.9, -1.8,
                        -0.1, 0.9, -1.8, 0.1, 0.9, -1.8])
action_scale = 0.4
last_action = np.zeros(12)
t = 0.0

while running:
    # 1. 读取传感器
    ang_vel = imu.get_angular_velocity()       # [3] 机体坐标系
    lin_vel = imu.get_linear_velocity()        # [3] 机体坐标系
    gravity = imu.get_projected_gravity()      # [3] 机体坐标系
    joint_pos = encoders.get_positions()       # [12]
    joint_vel = encoders.get_velocities()      # [12]
    command = joystick.get_command()            # [3] vx, vy, wz

    # 2. 构造观测
    phase = np.array([np.sin(2*np.pi*t/0.5), np.cos(2*np.pi*t/0.5)])
    obs = np.concatenate([
        lin_vel, ang_vel, gravity, command, phase,
        joint_pos - default_pos, joint_vel, last_action
    ])  # [50]

    # 3. 更新历史缓冲区
    history = np.roll(history, -1, axis=1)
    history[0, -1, :] = obs

    # 4. ONNX 推理
    action = sess.run(None, {"obs_history": history})[0][0]  # [12]

    # 5. 发送给电机
    target_pos = default_pos + action_scale * action
    motor_driver.set_joint_positions(target_pos)  # 电机侧 PD 控制

    last_action = action
    t += 0.02
    sleep(0.02)  # 50Hz 控制循环
```

### 依赖

- ONNX Runtime（`pip install onnxruntime`，Jetson 用 `onnxruntime-gpu`）
- Unitree SDK（UDP/CycloneDDS 与电机通信）
- IMU 传感器（Go2 内置）
- 关节编码器（Go2 内置）

### 参考

unitree_rl_mjlab 的 `deploy/` 目录有 Jetson 上的 C++ 部署实现。

---

## 在 DIY 机器狗上部署

### 路线 A：物理参数与 Go2 接近（直接部署）

如果你的 DIY 机器人在尺寸、重量、关节配置上和 Go2 类似（12 自由度四足、~13kg），可以直接使用训练好的 ONNX 模型。

**硬件需要：**
- 12 个舵机/电机（hip×4 + thigh×4 + calf×4）
- IMU（陀螺仪 + 加速度计，用于计算重力向量）
- 关节编码器（位置 + 速度反馈）
- 计算板（树莓派 5 / Jetson Nano，50Hz 跑 ONNX 足够）
- 电池

**软件需要：**
- 电机通信驱动（PWM / CAN / 串口）
- 50Hz 控制循环
- 观测构造器（和训练时一样的 50D 格式）
- ONNX Runtime 推理

### 路线 B：物理参数不同（需重新训练）

如果你的 DIY 机器人质量、腿长、电机力矩限制或关节配置不同，需要重新训练：

```
1. 测量 DIY 机器人 → 建 MJCF 模型（关节、质量、惯量、腿长）
2. 替换 go2.xml → 用同样的 pipeline 训练 Teacher
3. 蒸馏 → ONNX → 部署
```

**MJCF 建模需要的参数：**
- 每段腿（hip/thigh/calf）的长度、质量、惯量
- 电机力矩限制、减速比
- 机体质量和尺寸
- 关节限位角度

### 最小 DIY 架构

```
DIY 四足硬件
├── 计算板（RPi5 / Jetson Nano / ESP32 + 协处理器）
├── IMU（MPU6050 / BNO055 / ICM-42688）
├── 12 个舵机（高扭矩总线舵机 / 无刷电机+驱动板）
├── 关节编码器（舵机内置，或 AS5600 磁编码器）
└── 电池（3S-4S LiPo）

软件（Python, 50Hz 循环）
├── sensor_reader.py    # 读取 IMU + 关节角
├── obs_builder.py      # 构造 50D 观测向量
├── policy_runner.py    # ONNX 推理
├── motor_driver.py     # 发送关节位置命令
└── main.py             # 控制循环主程序
```

### DIY 关键注意事项

| 因素 | 影响 | 解决方案 |
|------|------|----------|
| 质量不同 | 策略输出力矩不匹配 | 用正确的 MJCF 重新训练 |
| 腿长不同 | 步态不 work | 用正确的 MJCF 重新训练 |
| 电机力矩小 | 跟不上目标位置 | 降低 action_scale，重新训练 |
| 没有速度传感器 | 缺少观测 | 用位置差分近似速度，或重新训练去掉 lin_vel |
| 计算板慢 | 跑不到 50Hz | 减小 history_length，简化 student 网络 |
| 舵机（非 PD 电机）| 只有位置控制 | 没问题！我们的 action 就是位置目标 |

### 推荐 BOM (~¥1500)

| 组件 | 价格 | 说明 |
|------|------|------|
| 12× STS3215（飞特总线舵机）| ¥780 | 有位置+速度+负载反馈，17kg·cm，串口总线 |
| 3D 打印结构件 | ¥100 | 淘宝搜"3D打印代加工"，PLA 材料 |
| 树莓派 Zero 2W | ¥150 | 跑 ONNX 50Hz 足够 |
| MPU6050 IMU | ¥10 | 陀螺仪 + 加速度计 |
| 总线舵机驱动板 | ¥30 | |
| 3S 锂电池 + 充电器 | ¥120 | |
| 线材/螺丝/杂项 | ¥80 | |
| **合计** | **~¥1270** | |

### 电机选型指南

| 电机 | 单价 | 反馈 | 速度 | Sim-to-Real 效果 |
|------|------|------|------|------------------|
| MG996R（PWM 舵机）| ¥15 | 无 | 慢 | 差 |
| LX-16A（总线舵机）| ¥50 | 位置 | 中等 | 中 |
| **STS3215（飞特总线）** | **¥65** | **位置+速度+负载** | **快** | **好** |
| XL430（Dynamixel）| ¥300 | 完整 | 很快 | 很好 |
| 无刷电机（Solo12/Pupper v3）| ¥500+ | 力矩控制 | 最快 | 最好 |

**推荐**：STS3215 性价比最高。有位置+速度反馈（观测向量必需），响应 <10ms，支持 100Hz+ 读写。

### 开源四足平台对比

| 项目 | 电机 | 成本 | RL 验证 | 说明 |
|------|------|------|---------|------|
| ODRI Solo12 | 无刷+同步带 | >¥10,000 | 学术界标准 | 效果最好，贵 |
| Stanford Pupper v3 | 无刷 GIM4305 | ~¥7,000 | 出厂带 RL 策略 | 好但贵 |
| Stanford Pupper v1 | PWM 舵机 | ~¥2,000 | 社区有人尝试 | 便宜但精度低 |
| SpotMicro ESP32 | MG996R/DS3218MG | ~¥1,500 | 需适配 | 社区最大，3D 打印 |

**淘宝/1688 搜索关键词**："12自由度机器狗 总线舵机"、"四足机器人 STS3215"、"SpotMicro 飞特舵机"

### DIY 重新训练流程

```
1. 搭建/购买 DIY 四足硬件
2. 测量实际尺寸（腿长、质量、关节限位）
3. 创建匹配你机器人的 MJCF 模型
4. 替换项目中的 go2.xml
5. 训练：python scripts/train.py --num-envs 2048 --max-iterations 2000
6. 蒸馏：python src/distill/collect_data.py && python src/distill/train_student.py
7. 导出：python src/distill/export_onnx.py
8. 将 ONNX 部署到机器人计算板上

重新训练总时间：RTX A4000 上约 20 分钟
```
