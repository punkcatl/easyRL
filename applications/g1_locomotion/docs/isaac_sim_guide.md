# Isaac Sim + Isaac Lab 使用手册

Isaac Sim 5.1.0 / Isaac Lab 2.3.0 / rsl_rl 3.0.1  
面向 RL locomotion 项目的完整工作流手册。

---

## 目录

1. [环境配置](#1-环境配置)
2. [Isaac Sim 界面](#2-isaac-sim-界面)
3. [视口操控](#3-视口操控)
4. [渲染设置](#4-渲染设置)
5. [场景与物体检查](#5-场景与物体检查)
6. [训练工作流](#6-训练工作流)
7. [可视化与评估](#7-可视化与评估)
8. [TensorBoard 监控](#8-tensorboard-监控)
9. [Isaac Lab 配置系统](#9-isaac-lab-配置系统)
10. [GPU 与性能](#10-gpu-与性能)
11. [常见报错与排查](#11-常见报错与排查)
12. [命令速查表](#12-命令速查表)

---

## 1. 环境配置

### 1.1 一键安装

```bash
cd ~/Desktop/myRL/easyRL
bash applications/g1_locomotion/setup_env.sh
```

脚本自动处理：Isaac Sim 安装、PyTorch、Isaac Lab、所有依赖修复。

### 1.2 手动激活

```bash
conda activate env_isaaclab
cd ~/Desktop/myRL/easyRL
```

激活后自动设置 `PYTHONPATH`（通过 conda activate hook），包含 Isaac Lab source 和 easyRL root。

### 1.3 验证环境

```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.get_device_name(0)}')
from isaacsim import SimulationApp; assert SimulationApp is not None
print('IsaacSim: OK')
from applications.g1_locomotion.student.networks import AdaptationModule
print('G1 project: OK')
"
```

### 1.4 关键路径

| 内容 | 路径 |
|------|------|
| Isaac Lab 源码 | `~/IsaacLab/` |
| Isaac Sim 包 | `~/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/` |
| Kit 日志 | `...isaacsim/kit/logs/Kit/Isaac-Sim/5.1/` |
| 扩展缓存 | `~/.local/share/ov/data/exts/` |
| G1 项目 | `~/Desktop/myRL/easyRL/applications/g1_locomotion/` |
| 训练结果 | `applications/g1_locomotion/results/` |

### 1.5 禁用"无响应"弹窗

Isaac Sim 加载时 UI 线程会阻塞数十秒，Ubuntu 会弹出"无响应"提示。永久禁用：

```bash
gsettings set org.gnome.mutter check-alive-timeout 0
```

---

## 2. Isaac Sim 界面

### 2.1 界面布局

```
┌──────────────────────────────────────────────────────────────────────┐
│ 菜单栏: File | Edit | Create | Window | Tools | Utilities | Layouts  │
├────────┬──────────────────────────────────────┬──────────────────────┤
│ 工具栏 │           3D 视口 (Viewport)          │ Stage (场景树)       │
│        │                                      │ Layer (图层)         │
│ Select │                                      │ Render Settings      │
│ Move   │    [ 机器人 / 场景渲染区 ]             │ (渲染设置)           │
│ Rotate │                                      │                      │
│ Scale  │                                      │ Property (属性)      │
│        │                                      │ Semantics            │
├────────┴──────────────────────────────────────┤                      │
│ Content (资源浏览) | Console (Python控制台)     │ Simulation Settings  │
│                                               │ Viewer Settings      │
└───────────────────────────────────────────────┴──────────────────────┘
```

### 2.2 关键面板功能

| 面板 | 功能 | 高频使用场景 |
|------|------|------------|
| **Stage** | 场景物体树形结构 | 选中机器人、查看关节层级 |
| **Property** | 选中物体的属性 | 查看关节角度、质量、碰撞 |
| **Render Settings** | 渲染模式/质量 | 切换 Common/Ray Tracing |
| **Viewer Settings** | 相机控制 | Follow Mode、Camera Eye/Target |
| **Simulation Settings** | 仿真参数 | Rendering Mode、时间步 |
| **Console** | Python 交互终端 | 运行时调试、查询状态 |

### 2.3 面板找不到时

菜单 **Window** → 勾选对应面板名称即可恢复显示。

---

## 3. 视口操控

### 3.1 鼠标操作（推荐使用 Alt 组合键）

| 操作 | 功能 | 适用场景 |
|------|------|---------|
| **Alt + 左键拖拽** | 环绕旋转 | 绕注视点旋转，不会丢目标（最常用） |
| **Alt + 中键拖拽** | 平移 | 上下左右移动画面 |
| **Alt + 右键拖拽** | 推拉缩放 | 精细缩放（比滚轮精确） |
| **滚轮** | 快速缩放 | 粗略调整距离 |
| **中键拖拽** | 平移 | 等同于 Alt + 中键 |
| **右键拖拽** | 第一人称旋转 | 容易转出画面，不推荐 |

**日常操作三件套：** Alt+左键旋转 → 滚轮缩放 → Alt+中键平移。

### 3.2 快捷键

| 快捷键 | 功能 |
|--------|------|
| **F** | 聚焦选中物体（丢了视角时救命键） |
| **Numpad 1/3/7** | 前视图 / 侧视图 / 顶视图 |
| **Numpad 5** | 正交 ↔ 透视 切换 |
| **W / E / R** | 移动 / 旋转 / 缩放 工具 |
| **Q** | 选择工具（取消变换模式） |
| **H** | 隐藏选中物体 |
| **Shift + H** | 显示所有隐藏物体 |
| **Ctrl + Z** | 撤销 |

### 3.3 快速找到机器人

视角丢了看不到机器人时：

1. 右侧 **Stage** 面板 → 展开 `/World/envs/env_0/Robot`
2. 点选 `Robot`
3. 按 **F** 聚焦

### 3.4 视角跟随 vs 自由视角

在右侧 **Viewer Settings** 面板中：

| Follow Mode | 行为 | 鼠标是否可控 |
|-------------|------|-------------|
| **World** | 相机固定世界坐标 | 完全自由 |
| **Env** | 相机固定环境原点 | 可自由操控 |
| **Asset_Root** | 每帧跟随机器人 | 不可控（每帧重置） |

**规则：** 观察步态细节用 Env 或 World（手动操控视角）；观察长距离行走用 Asset_Root（自动跟随但不能手动调整）。

---

## 4. 渲染设置

### 4.1 渲染模式

右侧 **Render Settings** 面板顶部的两个按钮：

| 模式 | 画质 | 帧率 | 清晰度 | 适用 |
|------|------|------|--------|------|
| **Common** | 光栅化 | 高（60fps+） | 清晰锐利 | 日常调试 |
| **Ray Tracing** | 光线追踪 | 低（15-30fps） | 模糊（DLSS降采样） | 录制高质量视频 |

**日常务必用 Common。**

### 4.2 为什么 Ray Tracing 模糊？

RTX A4000 跑实时光追性能不够 → Isaac Sim 自动启用 DLSS（AI 降采样后放大）→ 画面模糊。

**解决方案（任选一）：**
1. 切换到 Common 模式（推荐）
2. Ray Tracing 下关闭 DLSS：Render Settings → NVIDIA DLSS → Off
3. Ray Tracing 下提高采样：Render Settings → Ray Tracing → Samples Per Pixel 调高

### 4.3 Eco Mode

Render Settings → Eco Mode → 如果开启了会降低渲染分辨率。日常关掉它。

---

## 5. 场景与物体检查

### 5.1 场景树结构

```
/World
├── ground_plane/              ← 地面
├── envs/
│   ├── env_0/
│   │   └── Robot/             ← G1 机器人实例 0
│   │       ├── base_link          (躯干)
│   │       ├── torso_link
│   │       ├── left_hip_yaw_link
│   │       ├── left_hip_roll_link
│   │       ├── left_hip_pitch_link
│   │       ├── left_knee_link
│   │       ├── left_ankle_pitch_link
│   │       ├── left_ankle_roll_link  (左脚)
│   │       ├── right_hip_yaw_link
│   │       ├── ...
│   │       ├── right_ankle_roll_link (右脚)
│   │       ├── left_shoulder_*       (左臂)
│   │       └── right_shoulder_*      (右臂)
│   ├── env_1/ ...
│   └── env_N/ ...
├── Visuals/
│   └── Command/               ← 速度指令可视化（红色/绿色箭头）
└── terrain/                   ← 地形（rough 模式时）
```

### 5.2 查看物理属性

1. Stage 中选中物体（如 `left_knee_link`）
2. Property 面板显示：
   - **Transform**: pos/rot/scale
   - **Rigid Body**: mass, inertia, velocity
   - **Collision**: 碰撞形状
   - **Joint**: 关节类型、限位、力矩

### 5.3 实时查看关节状态

Console 面板中可以执行 Python：
```python
# 获取当前关节位置
import omni.isaac.core.utils.stage as stage_utils
# （Isaac Lab 管理的环境通过 env 对象访问更方便）
```

---

## 6. 训练工作流

### 6.1 启动训练

```bash
conda activate env_isaaclab
cd ~/Desktop/myRL/easyRL

# 完整训练（1500 iter, ~25 min on A4000）
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 \
    --num_envs 1024 \
    --headless

# 快速测试（验证配置是否正确）
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 \
    --num_envs 64 \
    --max_iterations 10 \
    --headless
```

### 6.2 从 checkpoint 恢复训练

```bash
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 \
    --num_envs 1024 \
    --headless \
    --resume \
    --load_run 2026-06-15_15-04-43 \
    --checkpoint model_900.pt
```

### 6.3 训练输出

训练过程自动保存到 `results/<experiment_name>/<timestamp>/`：

```
results/g1_flat_locomotion/2026-06-15_15-04-43/
├── events.out.tfevents.*     ← TensorBoard 日志
├── model_0.pt                ← 初始 checkpoint
├── model_100.pt              ← 每 100 iter 保存
├── model_200.pt
├── ...
├── model_1500.pt
└── teacher_final.pt          ← 最终模型
```

### 6.4 关键 CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名称 | G1-Flat-Custom-v0 |
| `--num_envs` | 并行环境数 | 1024 |
| `--max_iterations` | 训练迭代数 | config 中设置 |
| `--headless` | 无 GUI 模式 | False |
| `--seed` | 随机种子 | 42 |
| `--resume` | 恢复训练 | False |
| `--load_run` | 指定恢复的 run 目录 | None |
| `--checkpoint` | 指定恢复的 checkpoint | None |
| `--video` | 录制训练过程视频 | False |

### 6.5 可用任务列表

| Task ID | 说明 |
|---------|------|
| `G1-Flat-Custom-v0` | 平地训练（1024 envs, 1500 iter） |
| `G1-Flat-Custom-Play-v0` | 平地可视化（50 envs, 无 DR） |
| `G1-Rough-Custom-v0` | 复杂地形训练（1024 envs, 3000 iter） |
| `G1-Rough-Custom-Play-v0` | 复杂地形可视化 |

---

## 7. 可视化与评估

### 7.1 可视化训练结果

```bash
python applications/g1_locomotion/scripts/play.py \
    --task G1-Flat-Custom-Play-v0 \
    --load_run 2026-06-15_15-04-43 \
    --checkpoint teacher_final.pt \
    --num_envs 1 \
    --num_steps 10000
```

参数说明：
- `--num_envs 1`：只显示 1 个机器人（默认）
- `--num_steps 10000`：运行 10000 步（约 200 秒）后自动退出
- `--checkpoint`：加载哪个 checkpoint

### 7.2 观察步态的技巧

1. 启动可视化后，等 Isaac Sim 加载完（1-2 分钟）
2. 切换渲染到 **Common** 模式（右侧面板）
3. 用 **Alt + 左键** 调整到侧面视角，观察腿部动作
4. 观察重点：
   - 脚是否有明显抬起？（对比 shuffle vs 正常步态）
   - 步频是否规律？
   - 身体是否前倾/后仰？
   - 手臂是否保持默认姿态？

### 7.3 录制视频

```bash
# 方法一：训练时录制
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 --num_envs 4 --max_iterations 0 \
    --video --headless
# 视频保存在 results/<exp>/videos/

# 方法二：Isaac Sim 内截图
# File → Save Screenshot (或 Ctrl+Shift+S)
```

---

## 8. TensorBoard 监控

### 8.1 启动 TensorBoard

```bash
# 监控所有实验
tensorboard --logdir applications/g1_locomotion/results/ --port 6006

# 监控特定实验
tensorboard --logdir applications/g1_locomotion/results/g1_flat_locomotion/ --port 6006
```

浏览器打开 `http://localhost:6006`。

### 8.2 关键指标含义

| 指标 | 含义 | 理想趋势 |
|------|------|---------|
| `Train/mean_reward` | 平均累积奖励 | 持续上升后收敛 |
| `Train/mean_episode_length` | 平均存活步数（max=1000） | 接近 1000 = 不摔倒 |
| `Loss/surrogate` | PPO 策略损失 | 接近 0（小波动正常） |
| `Loss/value_function` | 值函数损失 | 逐渐下降 |
| `Loss/entropy` | 策略熵 | 缓慢下降（探索→收敛） |
| `Loss/learning_rate` | 当前学习率 | adaptive 模式自动调整 |
| `Policy/mean_noise_std` | 动作噪声标准差 | 逐渐下降 |
| `Perf/total_fps` | 训练帧率 | 越高越好 |

### 8.3 Reward 分项

每个 reward term 单独记录在 `Episode_Reward/` 下：

| 指标 | 含义 | 诊断用途 |
|------|------|---------|
| `track_lin_vel_xy_exp` | 线速度跟踪 | 是否在跟踪命令 |
| `track_ang_vel_z_exp` | 角速度跟踪 | 转弯能力 |
| `feet_air_time` | 脚离地时间 | 低→shuffle，高→正常步态 |
| `feet_slide` | 脚底滑动 | 高→拖地严重 |
| `flat_orientation_l2` | 姿态惩罚 | 高→身体晃动 |
| `lin_vel_z_l2` | 垂直速度 | 高→弹跳 |
| `action_rate_l2` | 动作变化率 | 高→抖动 |
| `termination_penalty` | 摔倒次数 | 应该接近 0 |

### 8.4 典型问题对应的 TensorBoard 表现

| 问题 | 表现特征 |
|------|---------|
| 站着不动 | mean_reward 平台但 track_lin_vel 低 |
| Shuffle 步态 | track_lin_vel 高但 feet_air_time 极低 |
| 频繁摔倒 | episode_length 低，termination_penalty 频繁 |
| 弹跳/跳着走 | lin_vel_z_l2 很高 |
| 动作抖动 | action_rate_l2 很高，画面看到机器人抖 |
| 训练不收敛 | reward 上下波动不升 |

### 8.5 多实验对比

```bash
# 在同一个 TensorBoard 中对比不同 round
tensorboard --logdir_spec \
    round1:results/g1_flat_locomotion/2026-06-15_15-04-43,\
    round2:results/g1_flat_locomotion/2026-06-16_10-00-00
```

---

## 9. Isaac Lab 配置系统

### 9.1 核心概念

Isaac Lab 使用 `@configclass`（类似 dataclass）管理所有配置。通过继承和覆盖来定制：

```python
from isaaclab.utils import configclass
from isaaclab_tasks...config.g1.flat_env_cfg import G1FlatEnvCfg

@configclass
class MyEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 覆盖你想改的部分
        self.scene.num_envs = 1024
        self.rewards.feet_air_time.weight = 1.5
```

### 9.2 配置层级

```
LocomotionVelocityRoughEnvCfg    ← Isaac Lab 基类（定义 obs/action/termination）
└── G1RoughEnvCfg                ← Isaac Lab G1 配置（定义 robot/reward/event）
    └── G1FlatEnvCfg             ← Isaac Lab G1 平地变体
        └── G1FlatLocomotionEnvCfg   ← 我们的自定义（调 reward 权重/command）
```

### 9.3 配置文件位置

| 层级 | 文件 | 修改场景 |
|------|------|---------|
| 基类 | `IsaacLab/source/isaaclab_tasks/.../velocity_env_cfg.py` | 不改 |
| G1 配置 | `IsaacLab/source/isaaclab_tasks/.../config/g1/` | 不改 |
| 自定义 | `applications/g1_locomotion/config/flat_env_cfg.py` | 改这里 |
| PPO 参数 | `applications/g1_locomotion/config/ppo_cfg.py` | 改这里 |

### 9.4 常改的配置项

**Reward 权重**（`config/flat_env_cfg.py`）：
```python
self.rewards.track_lin_vel_xy_exp.weight = 1.5    # 速度跟踪
self.rewards.feet_air_time.weight = 0.5           # 抬脚时间
self.rewards.feet_slide.weight = -0.1             # 脚底滑动惩罚
self.rewards.flat_orientation_l2.weight = -1.0    # 身体倾斜惩罚
```

**Command 范围**：
```python
self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.6)  # 前进速度范围
self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)  # 转弯速度范围
```

**PPO 参数**（`config/ppo_cfg.py`）：
```python
self.max_iterations = 1500      # 训练轮数
self.policy.actor_hidden_dims = [256, 128, 128]  # 网络结构
self.algorithm.learning_rate = 1e-3              # 学习率
self.algorithm.entropy_coef = 0.008              # 熵系数（探索量）
```

### 9.5 Gym 环境注册

自定义环境通过 `config/__init__.py` 注册到 gymnasium：

```python
gym.register(
    id="G1-Flat-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": G1FlatLocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": G1FlatPPOCfg,
    },
)
```

添加新变体时只需：写新 config class → 注册新 gym id → 训练时 `--task` 指定新 id。

---

## 10. GPU 与性能

### 10.1 显存估算

RTX A4000 (16GB) 的参考用量：

| num_envs | 显存占用 | 训练时间 (1500 iter) |
|----------|---------|---------------------|
| 64 | ~4 GB | ~60 min |
| 256 | ~6 GB | ~40 min |
| 512 | ~8 GB | ~30 min |
| 1024 | ~12 GB | ~25 min |
| 2048 | ~16 GB (满) | ~20 min |

### 10.2 监控 GPU

```bash
# 实时监控
watch -n1 nvidia-smi

# 看显存使用
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 5
```

### 10.3 训练速度优化

- 增加 `num_envs`（显存允许范围内越大越快）
- 使用 `--headless`（不渲染，省 GPU）
- 减少 `save_interval`（减少 IO）

### 10.4 显存不够时

```bash
# 减少环境数
--num_envs 512

# 或减小网络
# config/ppo_cfg.py 中
self.policy.actor_hidden_dims = [128, 64, 64]
```

---

## 11. 常见报错与排查

### 11.1 SimulationApp is None / Extension not found

**现象：**
```
[Warning] Unable to expose 'isaacsim.simulation_app' API: Extension not found
TypeError: 'NoneType' object is not callable
```

**原因：** `isaacsim` 的 `exts/` 目录为空，扩展没安装。

**修复：**
```bash
conda activate env_isaaclab
pip install isaacsim-rl --extra-index-url https://pypi.nvidia.com
```

### 11.2 No module named 'flatdict' / 'prettytable' / 'pkg_resources'

**原因：** Isaac Lab 依赖没装全，或 setuptools 版本过高。

**修复：**
```bash
pip install flatdict prettytable "setuptools>=69.0,<80.0"
```

### 11.3 Registry sync failed (cloudfront)

**现象：**
```
[Error] Syncing registry index failed for url: 'https://dw290v42wisod.cloudfront.net/...'
```

**原因：** NVIDIA community registry CDN 对某些地区/网络不可达。

**修复：** 删除 kit 文件中的 cloudfront registry 行：
```bash
for f in ~/IsaacLab/apps/*.kit; do
    sed -i '/dw290v42wisod.cloudfront.net/d' "$f"
done
```

### 11.4 Failed to solve some dependencies locally

**现象：** 启动时卡在 "syncing with extension registry..."

**原因：** 首次启动需要从 NVIDIA 服务器下载扩展。

**解决：** 等待下载完成（首次 5-15 分钟），后续启动都是秒加载。

### 11.5 gymnasium.error.NameNotFound

**现象：**
```
gymnasium.error.NameNotFound: Environment `G1-Flat-Custom-Play` doesn't exist.
```

**原因：** 自定义 config 模块没有被 import（gym.register 没执行）。

**修复：** 确保脚本中有：
```python
import applications.g1_locomotion.config  # noqa: F401
```

### 11.6 CUDA out of memory

**修复：** 减少 `--num_envs`，或杀掉占用 GPU 的其他进程：
```bash
nvidia-smi  # 查看谁在用 GPU
kill <pid>  # 杀掉不需要的进程
```

### 11.7 训练 reward 不收敛

**排查步骤：**
1. TensorBoard 看各 reward 分项，找到哪个异常
2. 检查 episode_length —— 如果很短说明频繁摔倒
3. 减小 command 范围（让任务更简单）
4. 检查 reward 权重是否合理（正向 reward 应大于惩罚总和）
5. 尝试更小的 lr 或更大的 entropy_coef

### 11.8 Isaac Sim GUI 闪退

**常见原因：**
- Python 脚本执行完毕（正常退出）→ 增加 `--num_steps`
- 脚本报错 → 看终端输出或 `kit_*.log`
- 显存爆了 → 减少 `--num_envs`

**查看崩溃日志：**
```bash
# 最新的 kit 日志
ls -lt ~/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/ | head -3
# 查看最后 50 行
tail -50 <最新log文件>
```

---

## 12. 命令速查表

### 环境管理

```bash
conda activate env_isaaclab                  # 激活环境
conda deactivate                             # 退出环境
```

### 训练

```bash
# 平地训练
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 --num_envs 1024 --headless

# 快速测试（验证配置）
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 --num_envs 64 --max_iterations 5 --headless

# 恢复训练
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Flat-Custom-v0 --num_envs 1024 --headless \
    --resume --load_run <run_dir> --checkpoint model_900.pt

# 复杂地形训练
python applications/g1_locomotion/scripts/train_teacher.py \
    --task G1-Rough-Custom-v0 --num_envs 1024 --headless
```

### 可视化

```bash
# 查看训练结果（单机器人）
python applications/g1_locomotion/scripts/play.py \
    --task G1-Flat-Custom-Play-v0 \
    --load_run <run_dir> --checkpoint teacher_final.pt

# 指定步数（默认 5000 步 ≈ 100 秒）
python applications/g1_locomotion/scripts/play.py \
    --task G1-Flat-Custom-Play-v0 \
    --load_run <run_dir> --checkpoint teacher_final.pt \
    --num_steps 20000
```

### 监控

```bash
# TensorBoard
tensorboard --logdir applications/g1_locomotion/results/ --port 6006

# GPU 监控
watch -n1 nvidia-smi

# 查看 Isaac Sim 日志
tail -f ~/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_*.log
```

### Phase 2/3（Teacher-Student 蒸馏）

```bash
# 采集 teacher 数据
python applications/g1_locomotion/scripts/collect_teacher_data.py \
    --task G1-Flat-Custom-v0 --load_run <run_dir> --num_steps 500000

# 训练 student
python applications/g1_locomotion/student/train_student.py \
    --data results/g1_flat_locomotion/teacher_distill_data.npz

# 评估 student
python applications/g1_locomotion/student/evaluate.py \
    --task G1-Flat-Custom-Play-v0 \
    --student_path results/.../student/student_best.pt

# ONNX 导出
python applications/g1_locomotion/export/export_onnx.py \
    --student_path results/.../student/student_best.pt

# 推理 benchmark
python applications/g1_locomotion/export/benchmark.py \
    --model results/.../student_g1.onnx
```

### 系统维护

```bash
# 禁用"无响应"弹窗
gsettings set org.gnome.mutter check-alive-timeout 0

# 修复 cloudfront registry 问题
for f in ~/IsaacLab/apps/*.kit; do sed -i '/dw290v42wisod.cloudfront.net/d' "$f"; done

# 修复 pkg_resources
pip install "setuptools>=69.0,<80.0"

# 清理 GPU 进程
nvidia-smi | grep python
kill <pid>
```
