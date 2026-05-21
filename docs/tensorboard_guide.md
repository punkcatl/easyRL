# TensorBoard Usage Guide

TensorBoard is a visualization tool from TensorFlow that works with PyTorch. It lets you view training curves (reward, loss, epsilon) in real-time through a browser dashboard.

## Installation

```bash
pip install tensorboard
```

## Enable in Config

In `algorithms/dqn/config.py`, set:

```python
"use_tensorboard": True,
```

When enabled, the Logger writes event files alongside CSV files into the same `results/` directory.

## Launch TensorBoard

```bash
# From project root
tensorboard --logdir=algorithms/dqn/results

# If comparing multiple runs (e.g., DQN vs PPO vs SAC)
tensorboard --logdir=experiments/results
```

Then open your browser at: **http://localhost:6006**

## What You'll See

| Tab | Content |
|-----|---------|
| Scalars | Training curves — episode_reward over time |
| Graphs | (Optional) Network computation graph |

The `episode_reward` curve shows per-episode total reward. You can smooth the curve with the slider on the left panel to see the trend more clearly.

## Useful Options

```bash
# Use a different port (if 6006 is taken)
tensorboard --logdir=algorithms/dqn/results --port=6007

# Bind to all interfaces (for remote access)
tensorboard --logdir=algorithms/dqn/results --host=0.0.0.0

# Reload data more frequently (default is 30s)
tensorboard --logdir=algorithms/dqn/results --reload_interval=5
```

## Workflow

1. Set `"use_tensorboard": True` in config
2. Run training: `python algorithms/dqn/train.py`
3. In another terminal: `tensorboard --logdir=algorithms/dqn/results`
4. Open http://localhost:6006 in browser
5. Watch curves update in real-time as training progresses

## Cleanup

TensorBoard event files (`events.out.tfevents.*`) can be large. Delete them when no longer needed:

```bash
rm algorithms/dqn/results/events.out.tfevents.*
```

## How It Works

Our Logger does NOT implement TensorBoard itself. It calls PyTorch's official `SummaryWriter`:

```python
from torch.utils.tensorboard import SummaryWriter
self.writer = SummaryWriter(log_dir=log_dir)
```

The data flow has two independent sides:

```
[Write side - PyTorch]                    [Read side - TensorBoard]
Logger.log(tag, step, value)              tensorboard --logdir=results/
    │                                          │
    ▼                                          ▼
SummaryWriter.add_scalar()              reads events.out.tfevents.*
    │                                          │
    ▼                                          ▼
writes binary event file               renders charts in browser
(events.out.tfevents.*)                 (http://localhost:6006)
```

- **Write side**: `SummaryWriter` (from `torch.utils.tensorboard`) converts data into TensorBoard's binary event format
- **Read side**: `tensorboard` command starts a web server that reads those event files and renders interactive charts

Our Logger simply adds one line `self.writer.add_scalar(tag, value, step)` on top of its CSV recording. The heavy lifting is done by PyTorch's `SummaryWriter`.

## Port 6006

6006 is TensorBoard's **default port**, not a fixed requirement. If port 6006 is already in use by another process, TensorBoard will fail to start. Use `--port` to change it:

```bash
tensorboard --logdir=algorithms/dqn/results --port=8080
# Then open http://localhost:8080
```

## Relationship to CSV Logging

CSV logging always runs regardless of the TensorBoard setting. TensorBoard is an additional, optional visualization layer. Both record the same data — CSV for offline analysis/plotting, TensorBoard for real-time monitoring.

---

# TensorBoard 使用指南

TensorBoard 是 TensorFlow 提供的可视化工具，也支持 PyTorch。它可以在浏览器中实时查看训练曲线（奖励、损失、探索率等）。

## 安装

```bash
pip install tensorboard
```

## 在配置中启用

在 `algorithms/dqn/config.py` 中设置：

```python
"use_tensorboard": True,
```

启用后，Logger 会在 `results/` 目录中生成 TensorBoard 事件文件（与 CSV 文件放在同一目录）。

## 启动 TensorBoard

```bash
# 从项目根目录
tensorboard --logdir=algorithms/dqn/results

# 如果要对比多个算法的训练结果（如 DQN vs PPO vs SAC）
tensorboard --logdir=experiments/results
```

然后在浏览器中打开：**http://localhost:6006**

## 你会看到什么

| 标签页 | 内容 |
|--------|------|
| Scalars | 训练曲线 — episode_reward 随时间的变化 |
| Graphs | （可选）网络计算图 |

`episode_reward` 曲线显示每轮的总奖励。你可以用左侧面板的滑块平滑曲线，更清楚地看到趋势。

## 常用选项

```bash
# 使用其他端口（如果 6006 被占用）
tensorboard --logdir=algorithms/dqn/results --port=6007

# 绑定所有网络接口（用于远程访问）
tensorboard --logdir=algorithms/dqn/results --host=0.0.0.0

# 更频繁地刷新数据（默认 30 秒）
tensorboard --logdir=algorithms/dqn/results --reload_interval=5
```

## 工作流程

1. 在 config 中设置 `"use_tensorboard": True`
2. 运行训练：`python algorithms/dqn/train.py`
3. 在另一个终端中：`tensorboard --logdir=algorithms/dqn/results`
4. 在浏览器中打开 http://localhost:6006
5. 随着训练进行，实时观察曲线更新

## 清理

TensorBoard 事件文件（`events.out.tfevents.*`）可能很大。不需要时可以删除：

```bash
rm algorithms/dqn/results/events.out.tfevents.*
```

## 工作原理

我们的 Logger 并没有自己实现 TensorBoard 功能，而是调用了 PyTorch 官方提供的 `SummaryWriter`：

```python
from torch.utils.tensorboard import SummaryWriter
self.writer = SummaryWriter(log_dir=log_dir)
```

数据流分为两个独立的部分：

```
[写入端 - PyTorch]                       [读取端 - TensorBoard]
Logger.log(tag, step, value)              tensorboard --logdir=results/
    │                                          │
    ▼                                          ▼
SummaryWriter.add_scalar()              读取 events.out.tfevents.*
    │                                          │
    ▼                                          ▼
写入二进制事件文件                        在浏览器中渲染图表
(events.out.tfevents.*)                 (http://localhost:6006)
```

- **写入端**：`SummaryWriter`（来自 `torch.utils.tensorboard`）把数据转换成 TensorBoard 的二进制事件格式
- **读取端**：`tensorboard` 命令启动一个 Web 服务器，读取事件文件并渲染成交互式图表

我们的 Logger 只是在原有 CSV 记录的基础上多调了一行 `self.writer.add_scalar(tag, value, step)`，真正的重活是 PyTorch 的 `SummaryWriter` 在做。

## 端口 6006

6006 是 TensorBoard 的**默认端口**，不是固定的。如果 6006 已经被其他程序占用，TensorBoard 会启动失败。用 `--port` 改端口：

```bash
tensorboard --logdir=algorithms/dqn/results --port=8080
# 然后打开 http://localhost:8080
```

## 与 CSV 日志的关系

无论 TensorBoard 是否开启，CSV 日志始终会记录。TensorBoard 是额外的、可选的可视化层。两者记录相同的数据 — CSV 用于离线分析/绘图，TensorBoard 用于实时监控。
