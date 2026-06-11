# Go2 Locomotion 复现：替代方案说明

与宇树实际工业路线的对比，说明哪些环节做了替代、哪些完全一致。

---

## 做了替代的环节

| 环节 | 宇树实际方案 | 我们的替代 | 替代原因 |
|------|-------------|-----------|----------|
| 仿真引擎 | Isaac Gym/Lab（GPU 并行，4096+ envs） | MuJoCo + 同步向量化（32 envs CPU） | Isaac Gym 需要 NVIDIA GPU + 专用 license，安装门槛高 |
| 并行规模 | 单卡数千环境，分钟级训练 | 32 环境，预计数小时训练 | CPU 并行上限，但算法逻辑完全一致 |
| RL 框架 | rsl_rl（RSL 维护的封装库） | 自实现 PPO（项目已有基础） | 面试展示需要理解底层，非调库 |
| 网络架构 | ActorCriticRecurrent（LSTM hidden=64） | MLP（先跑通），LSTM 作为可选扩展 | 降低首次调通复杂度 |
| 地形 Curriculum | 10 级难度 × 20 种地形的完整网格 | 先用平地训练，地形作为后续扩展 | MuJoCo 地形生成比 Isaac 更手动 |
| 部署格式 | `.pt` JIT traced → 板端 LibTorch | ONNX → ONNXRuntime benchmark | 无实机，ONNX 更通用且已有现成模块 |
| 实机部署 | unitree_sdk2 DDS 通信 → Go2 本体 | 纯仿真 Sim2Sim 验证 | 无硬件 |

## 完全一致、未做替代的核心环节

- PPO + GAE + clip 算法逻辑
- Asymmetric Actor-Critic（actor 只看 obs，critic 看 obs+privileged）
- Domain Randomization（摩擦、质量、外力、电机增益）
- PD 位置控制（`target = scale × action + default_angle`）
- 48D 观测空间设计（lin_vel + ang_vel + gravity + joint_pos/vel + last_action + command）
- 12D 动作空间（12 关节位置目标）
- Teacher-Student RMA 蒸馏（history→latent→action）
- Velocity tracking reward 设计
- Go2 MJCF 模型（来自 mujoco_menagerie，和宇树用同一个模型）

## 总结

算法和架构层面零替代，替代都发生在工程规模和硬件层面。面试时可以说"核心算法和宇树完全一致，只是仿真规模从 GPU 数千并行降到 CPU 数十并行，训练时间从分钟级变为小时级"。
