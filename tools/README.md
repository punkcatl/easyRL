# Tools

## Network Visualization

Three methods for visualizing neural network architectures:

### 1. Netron (Interactive Computation Graph)

Best for: inspecting model structure, checking tensor shapes, debugging.

**Project script (recommended):**

```bash
python tools/view_netron.py
```

Workflow:
1. Choose source: project agent `[1]`, `.pt/.pth` file `[2]`, or `.onnx` file `[3]`
2. For agents/pt files: provide model parameters, auto-export to ONNX
3. Opens in browser at `http://localhost:8080`

**Manual usage:**

```python
import torch
from algorithms.dqn.agent import QNetwork

model = QNetwork(state_dim=4, action_dim=2)
torch.onnx.export(model, torch.randn(1, 4), "model.onnx",
    input_names=["state"], output_names=["q_values"])
```

```bash
netron model.onnx
```

**Available project agents:**

| # | Model | Module |
|---|-------|--------|
| 1 | DQN QNetwork | `algorithms.dqn.agent` |
| 2 | Policy Gradient PolicyNetwork | `algorithms.policy_gradient.agent` |
| 3 | PPO Actor | `algorithms.ppo.agent` |
| 4 | PPO Critic | `algorithms.ppo.agent` |
| 5 | SAC GaussianPolicy | `algorithms.sac.agent` |
| 6 | SAC QNetwork | `algorithms.sac.agent` |

**Install:** `pip install onnx netron`

### 2. TensorBoard (Training + Graph)

Best for: viewing network structure alongside training curves.

Network graphs are automatically recorded when training starts. To view:

```bash
# Start training (graph is recorded automatically)
python algorithms/dqn/train.py

# Open TensorBoard
tensorboard --logdir algorithms/dqn/results
```

Open `http://localhost:6006` → click **GRAPHS** tab to see network structure.

All algorithms support this:

```bash
tensorboard --logdir algorithms/dqn/results
tensorboard --logdir algorithms/policy_gradient/results
tensorboard --logdir algorithms/ppo/results
tensorboard --logdir algorithms/sac/results
```

**Manual add_graph:**

```python
import torch
from torch.utils.tensorboard import SummaryWriter
from algorithms.dqn.agent import QNetwork

model = QNetwork(state_dim=4, action_dim=2)
writer = SummaryWriter("runs/dqn")
writer.add_graph(model, torch.randn(1, 4))
writer.close()
# Then: tensorboard --logdir runs/dqn
```

### 3. NN-SVG (Quick Diagrams)

Best for: clean architecture diagrams for slides, blog posts, teaching materials.

Online tool, no installation needed: https://alexlenail.me/NN-SVG/

Three modes:
- **FCNN** — classic neuron + connection diagram (circles and lines)
- **LeNet** — CNN block diagram (3D boxes)
- **AlexNet** — deep CNN 3D block diagram

Usage:
1. Open the website
2. Set layer dimensions in the left panel
3. SVG renders in real-time
4. Right-click → Save as SVG

Output is vector (SVG), scales to any size without blur.

Limitations: manual input only (no model file import), only supports simple sequential architectures.

### Comparison

| Tool | Input | Output | Best for |
|------|-------|--------|----------|
| Netron | .onnx / .pt | Interactive browser | Debug, shape inspection |
| TensorBoard | Training auto-records | Browser (localhost:6006) | Training monitoring + graph |
| NN-SVG | Manual (web UI) | SVG vector image | Slides, papers, teaching |

---

# 工具

## 网络可视化

三种方式可视化神经网络架构：

### 1. Netron（交互式计算图）

适合：检查模型结构、查看张量形状、调试。

**项目脚本（推荐）：**

```bash
python tools/view_netron.py
```

流程：
1. 选择来源：项目 agent `[1]`、`.pt/.pth` 文件 `[2]`、`.onnx` 文件 `[3]`
2. 对于 agent/pt 文件：提供模型参数，自动导出为 ONNX
3. 在浏览器中打开 `http://localhost:8080`

**手动使用：**

```python
import torch
from algorithms.dqn.agent import QNetwork

model = QNetwork(state_dim=4, action_dim=2)
torch.onnx.export(model, torch.randn(1, 4), "model.onnx",
    input_names=["state"], output_names=["q_values"])
```

```bash
netron model.onnx
```

**可用的项目 Agent：**

| # | 模型 | 模块 |
|---|------|------|
| 1 | DQN QNetwork | `algorithms.dqn.agent` |
| 2 | Policy Gradient PolicyNetwork | `algorithms.policy_gradient.agent` |
| 3 | PPO Actor | `algorithms.ppo.agent` |
| 4 | PPO Critic | `algorithms.ppo.agent` |
| 5 | SAC GaussianPolicy | `algorithms.sac.agent` |
| 6 | SAC QNetwork | `algorithms.sac.agent` |

**安装：** `pip install onnx netron`

### 2. TensorBoard（训练 + 网络图）

适合：在查看训练曲线的同时查看网络结构。

训练开始时网络图会自动记录。查看方式：

```bash
# 启动训练（网络图自动记录）
python algorithms/dqn/train.py

# 打开 TensorBoard
tensorboard --logdir algorithms/dqn/results
```

打开 `http://localhost:6006` → 点击 **GRAPHS** 标签页查看网络结构。

所有算法均已支持：

```bash
tensorboard --logdir algorithms/dqn/results
tensorboard --logdir algorithms/policy_gradient/results
tensorboard --logdir algorithms/ppo/results
tensorboard --logdir algorithms/sac/results
```

**手动 add_graph：**

```python
import torch
from torch.utils.tensorboard import SummaryWriter
from algorithms.dqn.agent import QNetwork

model = QNetwork(state_dim=4, action_dim=2)
writer = SummaryWriter("runs/dqn")
writer.add_graph(model, torch.randn(1, 4))
writer.close()
# 然后: tensorboard --logdir runs/dqn
```

### 3. NN-SVG（快速示意图）

适合：制作干净的架构图用于 PPT、博客、教学材料。

在线工具，无需安装：https://alexlenail.me/NN-SVG/

三种模式：
- **FCNN** — 经典神经元 + 连线图（圆圈和线条）
- **LeNet** — CNN 方块图（3D 方块）
- **AlexNet** — 深层 CNN 3D 方块图

使用方式：
1. 打开网站
2. 在左侧面板设置每层维度
3. SVG 实时生成预览
4. 右键 → 另存为 SVG

输出为矢量图（SVG），放大不模糊。

限制：只能手动输入参数（不能导入模型文件），仅支持简单的顺序结构。

### 对比

| 工具 | 输入 | 输出 | 适合 |
|------|------|------|------|
| Netron | .onnx / .pt | 交互式浏览器 | 调试、形状检查 |
| TensorBoard | 训练时自动记录 | 浏览器 (localhost:6006) | 训练监控 + 网络图 |
| NN-SVG | 手动（网页 UI） | SVG 矢量图 | PPT、论文、教学 |
