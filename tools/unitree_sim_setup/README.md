# Unitree RL Lab 仿真环境一键部署

一键部署 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) 所需的全部环境，适用于全新的 Ubuntu 机器。

## 前提条件

- Ubuntu 22.04+
- NVIDIA 显卡 + 驱动已安装（`nvidia-smi` 可用）
- 网络可访问 GitHub 和 PyPI

其他所有依赖（git、Miniconda、conda 环境、PyTorch、IsaacLab 等）由脚本自动安装。

## 使用方式

```bash
# 最简单：在当前目录下部署
bash setup_env.sh

# 指定工作目录
bash setup_env.sh --workspace ~/myRL

# 自定义 conda 环境名
bash setup_env.sh --workspace ~/myRL --env-name my_isaaclab
```

## 脚本做了什么

| 步骤 | 内容 | 说明 |
|------|------|------|
| 1 | 克隆仓库 | unitree_rl_lab、IsaacLab v2.3.0、unitree_ros |
| 2 | 创建 conda 环境 | Python 3.11 |
| 3 | 安装 IsaacSim | 5.1.0 (pip) |
| 4 | 安装 PyTorch | 2.7.0+cu128，自动重试 5 次应对网络不稳定 |
| 5 | 安装 Isaac Lab | 含 rsl_rl 强化学习框架 |
| 6 | 安装 unitree_rl_lab | editable 模式 |
| 7 | 配置机器人模型路径 | 自动设置 UNITREE_ROS_DIR |
| 8 | 配置 conda hook | activate 时自动 source 环境变量 |

## 安装后的目录结构

```
<workspace>/
├── unitree_rl_lab/       # 训练代码
├── IsaacLab/             # Isaac Lab 框架
└── unitree_ros/          # 机器人 URDF 描述文件
```

## 安装后验证

```bash
conda activate env_isaaclab
cd <workspace>/unitree_rl_lab

# 列出所有可用任务
./unitree_rl_lab.sh -l

# 开始训练
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity

# 推理/回放
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity
```

## 设计要点

- **幂等**：每步都有跳过检测，中断后重新运行不会重复已完成的步骤
- **断点续传**：PyTorch 下载失败自动重试（最多 5 次），不用从头来
- **自动装 Miniconda**：裸机也能跑
- **自动装 git**：没有会 apt install

## 已知限制

- NVIDIA 驱动需要手动预装（涉及内核模块和重启）
- 首次运行 IsaacSim 会弹出 EULA 确认（输入 `Yes`）
- PyTorch 下载速度取决于网络，公司内网可能较慢（~800KB/s）

## 测试环境

- Ubuntu 22.04, NVIDIA Driver 580, RTX A4000 16GB
- IsaacSim 5.1.0 + IsaacLab 2.3.0
- PyTorch 2.7.0+cu128
