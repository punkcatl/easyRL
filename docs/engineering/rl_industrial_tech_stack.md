# 工业界 RL 框架技术栈

工业界 RL 落地的框架选型，按"训练、仿真、部署"三层组织。重点覆盖机器人运控和自动驾驶两大方向。

---

## 训练框架

| 框架 | 定位 | 典型用户 |
|------|------|----------|
| rl_games | GPU 加速训练，与 Isaac 生态深度绑定 | 机器人运控（Unitree、人形机器人公司） |
| RSL_rl | ETH Zurich RSL 实验室出品，专注 legged locomotion | 四足/双足学术+工业 |
| RLlib (Ray) | 分布式大规模训练，multi-agent | 自动驾驶仿真、大规模策略搜索 |
| Stable-Baselines3 | 单机快速原型，API 干净 | 小团队验证、教学、预研 |
| TorchRL (Meta) | PyTorch 原生，模块化 | 新项目逐步采用 |
| CleanRL | 单文件实现，易读易改 | 研究复现、算法魔改 |

## 仿真环境

| 场景 | 主流方案 |
|------|----------|
| 机器人运控 | NVIDIA Isaac Lab（前身 Isaac Gym / Orbit）— GPU 并行物理仿真，万级环境同时跑 |
| 自动驾驶 | CARLA、MetaDrive、highway-env；大厂用内部闭源仿真器 |
| 通用连续控制 | MuJoCo + Gymnasium |

## 部署推理

训练完的 policy 本质是一个小网络（通常 2-3 层 MLP），部署方式：

- **ONNX → TensorRT / ONNX Runtime** — 嵌入式设备上跑推理（机器人板端、车载芯片）
- **libtorch (PyTorch C++)** — 延迟要求极高时
- **ROS2** — 机器人系统集成，policy node 接收 state 发布 action

---

## 工业主流组合

### 机器人运控（2024-2026 标配）

```
Isaac Lab (仿真) + rl_games/RSL_rl (训练) + Domain Randomization (sim-to-real) + ONNX (部署)
```

这条路线几乎是所有做四足/人形机器人公司的标配（Unitree、Figure、1X、Agility 等公开信息都指向这条线）。详见 [宇树运控深度案例](rl_unitree_locomotion_case.md)。

核心逻辑：
- Isaac Lab 提供 GPU 并行仿真，单卡同时跑 4096+ 环境实例
- rl_games / RSL_rl 配合 Isaac 的向量化 API，实现高吞吐 PPO 训练
- Domain Randomization 在仿真中随机化物理参数（摩擦、质量、延迟等），弥合 sim-to-real gap
- 训练完导出 ONNX，部署到机器人板端实时推理（通常 < 5ms）

### 自动驾驶规控

```
内部仿真器 + RLlib/自研分布式框架 + Safety layer + 车载推理
```

自动驾驶公司较少直接用开源 RL 框架做最终产品，更多自研或深度定制。原因：
- 仿真器与公司数据管线深度耦合，开源方案难以满足
- 需要 closed-loop 评估、场景库管理、安全约束等定制化能力
- RLlib 在预研和 POC 阶段很常见，验证算法可行性后再自研落地

---

## 与本项目的关系

本项目当前使用 highway-env / racetrack-v0 + 手写 PPO/SAC 的轻量实现。如果向工业靠拢：

- `applications/sim_to_real/` 模块已经在做 Domain Randomization + Teacher-Student 蒸馏，逻辑上与 Isaac Lab 路线一致
- 训练算法（PPO with rollout buffer, GAE, clip）与 rl_games / RSL_rl 的核心实现相同
- 下一步可考虑将 policy 导出为 ONNX，验证端到端部署流程
