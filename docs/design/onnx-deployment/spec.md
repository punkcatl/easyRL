# ONNX Deployment Design Spec

## 1. Overview

A complete model deployment example: export trained PyTorch RL models to ONNX format, verify accuracy, quantize to FP16, and benchmark inference speed. Demonstrates the full "training → deployment" pipeline.

## 2. What Is This and Why Do We Need It

ONNX deployment means: **taking a trained PyTorch model and converting it to a universal format file that can run without PyTorch installed.**

```
Training phase (your PC):
  PyTorch → train PPO → save .pth file

Deployment phase (target device):
  .pth → export to .onnx → load with lightweight runtime → real-time action output
```

**Why not just deploy PyTorch directly?**

| | PyTorch | ONNX |
|---|---|---|
| Package size | ~2GB | ~50MB |
| Dependencies | Python + CUDA + many libs | single library |
| Inference speed | normal | 2-10x faster (optimized) |
| Platforms | mainly PC/server | PC, embedded, mobile, robot controllers |

**In our project:** After training the Sim-to-Real Student policy, deploying to a robot means the robot's controller board won't have PyTorch installed — so we export to ONNX and use a lightweight runtime for inference.

This is "the last mile from training to deployment."

**Industrial standard flow:**

```
PyTorch training
    ↓
torch.onnx.export() → .onnx file
    ↓
Verify: compare PyTorch output vs ONNX output (accuracy consistency check)
    ↓
Optimize: quantize (FP32 → FP16 / INT8) to shrink model and speed up
    ↓
Deploy: TensorRT (GPU) or ONNX Runtime (CPU)
```

**Industrial deployment targets:**

| Target Device | Runtime Used | Representative |
|---|---|---|
| Robot controller (Jetson/Orin) | TensorRT (NVIDIA ecosystem) | Unitree, Tesla Bot |
| Embedded MCU | ONNX Runtime C++ / TFLite | Drones, small robots |
| Cloud/server | ONNX Runtime Python or TorchScript | LLM inference, recommendation |
| Vehicle domain controller (AD) | TensorRT / TVM | Major automakers |

We focus on ONNX Runtime Python (covers the core flow). The docs mention TensorRT for Jetson as a natural extension.

## 3. Pipeline

```
┌────────────────────────────────────────────────────────────┐
│  Step 1: Export                                             │
│  torch.onnx.export(model) → .onnx file                    │
├────────────────────────────────────────────────────────────┤
│  Step 2: Accuracy Verification                             │
│  PyTorch output vs ONNX Runtime output → max diff < 1e-5  │
├────────────────────────────────────────────────────────────┤
│  Step 3: Quantization                                      │
│  FP32 → FP16 → compare speed and accuracy loss            │
├────────────────────────────────────────────────────────────┤
│  Step 4: Inference Benchmark                               │
│  Compare: PyTorch / ONNX FP32 / ONNX FP16 latency        │
└────────────────────────────────────────────────────────────┘
```

## 4. Module Structure

```
applications/onnx_deployment/
├── __init__.py
├── config.py                   ← export/inference configuration
├── export/
│   ├── __init__.py
│   ├── export_ppo_discrete.py  ← export RL+MPC discrete PPO (beginner example)
│   └── export_student.py       ← export Sim-to-Real Student (main example)
├── verify/
│   ├── __init__.py
│   └── accuracy_check.py      ← PyTorch vs ONNX accuracy comparison
├── quantize/
│   ├── __init__.py
│   └── fp16_quantize.py       ← FP32 → FP16 quantization
├── inference/
│   ├── __init__.py
│   └── onnx_runner.py         ← ONNX Runtime inference wrapper
├── benchmark.py                ← speed comparison (PyTorch / FP32 / FP16)
├── results/                    ← exported .onnx files + benchmark results
└── docs/
    └── theory.md               ← deployment tutorial (export + quantization + interview)
```

## 5. Models to Export

| Model | Source | Input Shape | Output Shape | Complexity |
|---|---|---|---|---|
| PPO PolicyNet (discrete) | RL+MPC demo | (1, 25) float | (1, 5) softmax | Beginner |
| Student Base Policy | Sim-to-Real | (1, 43) float | (1, 8) float | Main example |
| Adaptation Module | Sim-to-Real | (1, 1350) float | (1, 16) float | Main example |

The Student example demonstrates composite model export (two sub-networks that run in sequence).

## 6. Export Details

### 6.1 Basic Export (PPO Discrete)

```python
import torch

model = PolicyNet(state_dim=25, hidden_dim=128, action_dim=5)
model.load_state_dict(...)
model.eval()

dummy_input = torch.randn(1, 25)
torch.onnx.export(
    model, dummy_input, "ppo_discrete.onnx",
    input_names=["observation"],
    output_names=["action_probs"],
    dynamic_axes={"observation": {0: "batch"}, "action_probs": {0: "batch"}},
    opset_version=17,
)
```

### 6.2 Composite Export (Student)

Export Adaptation Module and Base Policy as separate ONNX files (allows independent optimization):

```python
# Adaptation Module
adapt_input = torch.randn(1, 1350)  # 50 frames * 27 dim
torch.onnx.export(adapt_module, adapt_input, "adaptation_module.onnx", ...)

# Base Policy
policy_input = torch.randn(1, 43)   # obs(27) + latent_z(16)
torch.onnx.export(base_policy, policy_input, "base_policy.onnx", ...)
```

## 7. Accuracy Verification

Compare outputs on 1000 random inputs:

```python
import numpy as np
import onnxruntime as ort

# PyTorch inference
with torch.no_grad():
    pytorch_output = model(test_input).numpy()

# ONNX Runtime inference
session = ort.InferenceSession("model.onnx")
onnx_output = session.run(None, {"observation": test_input.numpy()})[0]

# Compare
max_diff = np.max(np.abs(pytorch_output - onnx_output))
assert max_diff < 1e-5, f"Accuracy mismatch: {max_diff}"
```

## 8. FP16 Quantization

### 8.1 Method

Use ONNX Runtime's built-in FP16 conversion:

```python
from onnxruntime.transformers import float16
import onnx

model_fp32 = onnx.load("model_fp32.onnx")
model_fp16 = float16.convert_float_to_float16(model_fp32)
onnx.save(model_fp16, "model_fp16.onnx")
```

### 8.2 Verification

- Accuracy: max diff between FP32 and FP16 output should be < 1e-3
- File size: FP16 should be ~50% of FP32
- Speed: FP16 should be ~1.5-2x faster on GPU

## 9. Benchmark

### 9.1 Methodology

- 1000 inference runs, discard first 100 (warmup)
- Measure: mean latency, P95 latency, throughput (inferences/sec)
- Compare: PyTorch (GPU) / ONNX FP32 (GPU) / ONNX FP16 (GPU)

### 9.2 Expected Results

```
┌──────────────────┬───────────┬───────────┬───────────┐
│ Model            │ PyTorch   │ ONNX FP32 │ ONNX FP16 │
├──────────────────┼───────────┼───────────┼───────────┤
│ PPO Discrete     │ ~0.5ms    │ ~0.2ms    │ ~0.1ms    │
│ Student+Adapt    │ ~1.0ms    │ ~0.4ms    │ ~0.2ms    │
├──────────────────┼───────────┼───────────┼───────────┤
│ Accuracy (FP32)  │ baseline  │ < 1e-5    │ -         │
│ Accuracy (FP16)  │ baseline  │ -         │ < 1e-3    │
├──────────────────┼───────────┼───────────┼───────────┤
│ File Size        │ N/A       │ ~100%     │ ~50%      │
└──────────────────┴───────────┴───────────┴───────────┘
```

## 10. Tutorial Document (theory.md)

### Outline

1. **Why ONNX: From Training to Deployment**
   - The deployment gap
   - ONNX as universal intermediate format

2. **Export: torch.onnx.export Explained**
   - Static vs dynamic axes
   - Opset versions
   - Common pitfalls (unsupported ops, dynamic control flow)

3. **Quantization: FP32 → FP16 → INT8**
   - What quantization does (represent weights with fewer bits)
   - Accuracy-speed trade-off
   - When to use which precision

4. **Deployment Targets (Overview)**
   - ONNX Runtime (CPU/GPU)
   - TensorRT (NVIDIA GPU, for Jetson/Orin)
   - TFLite (mobile/MCU)

5. **Interview FAQ**
   - "How do you deploy your RL model to a robot?"
   - "How do you optimize inference speed?"
   - "What's the accuracy loss from quantization?"

## 11. Dependencies

- `onnx`: model format (pip install onnx)
- `onnxruntime-gpu`: inference engine (pip install onnxruntime-gpu)
- `torch`: export source
- `numpy`: numerical comparison

## 12. Deliverables

1. Export scripts for 2 model types (discrete PPO + Student composite)
2. Accuracy verification tool (automated PyTorch vs ONNX comparison)
3. FP16 quantization script with accuracy check
4. Benchmark script with latency/throughput comparison table
5. Bilingual tutorial document (export + quantization + deployment overview)

---

# ONNX 部署设计规范

## 1. 概述

完整的模型部署示例：将训练好的 PyTorch RL 模型导出为 ONNX 格式，验证精度，量化为 FP16，并进行推理速度基准测试。展示完整的"训练 → 部署"pipeline。

## 2. 这是什么？为什么需要？

ONNX 部署就是：**把训练好的 PyTorch 模型转成通用格式文件，让它能在没有 PyTorch 的环境中跑起来。**

```
训练阶段 (你的电脑):
  PyTorch → 训练 PPO → 保存 .pth 文件

部署阶段 (目标设备):
  .pth → 导出为 .onnx → 用轻量推理引擎加载 → 实时输出动作
```

**为什么不能直接用 PyTorch 部署？**

| | PyTorch | ONNX |
|---|---|---|
| 包大小 | ~2GB | ~50MB |
| 依赖 | Python + CUDA + 一堆库 | 单个库 |
| 推理速度 | 一般 | 快 2-10x（优化过） |
| 能跑的平台 | 主要是 PC/服务器 | PC、嵌入式、手机、机器人主控板 |

**在我们项目里的角色：** 训练完 Sim-to-Real 的 Student 策略后，要部署到机器人上 — 机器人主控板上不会装 PyTorch，所以要导出成 ONNX，用轻量引擎推理。

这是"从训练到部署的最后一公里"。

**工业界标准流程：**

```
PyTorch 训练
    ↓
torch.onnx.export() → .onnx 文件
    ↓
验证: 对比 PyTorch 输出 vs ONNX 输出（精度一致性检查）
    ↓
优化: 量化 (FP32 → FP16 / INT8) 减小模型、加速推理
    ↓
部署: TensorRT (GPU) 或 ONNX Runtime (CPU)
```

**工业界部署目标：**

| 部署目标 | 使用的运行时 | 代表 |
|---|---|---|
| 机器人主控板 (Jetson/Orin) | TensorRT（NVIDIA 生态） | 宇树、特斯拉 Bot |
| 嵌入式 MCU | ONNX Runtime C++ / TFLite | 无人机、小型机器人 |
| 云端/服务端 | ONNX Runtime Python 或 TorchScript | LLM 推理、推荐系统 |
| 车载域控（自动驾驶） | TensorRT / TVM | 各车企 |

本模块聚焦 ONNX Runtime Python（覆盖核心流程）。文档中提及 TensorRT + Jetson 作为自然延伸方向。

## 3. Pipeline

```
┌────────────────────────────────────────────────────────────┐
│  Step 1: 导出                                               │
│  torch.onnx.export(model) → .onnx 文件                     │
├────────────────────────────────────────────────────────────┤
│  Step 2: 精度验证                                           │
│  PyTorch 输出 vs ONNX Runtime 输出 → max diff < 1e-5       │
├────────────────────────────────────────────────────────────┤
│  Step 3: 量化                                               │
│  FP32 → FP16 → 对比推理速度和精度损失                       │
├────────────────────────────────────────────────────────────┤
│  Step 4: 推理 Benchmark                                     │
│  对比: PyTorch / ONNX FP32 / ONNX FP16 的延迟和吞吐        │
└────────────────────────────────────────────────────────────┘
```

## 4. 模块结构

```
applications/onnx_deployment/
├── __init__.py
├── config.py                   ← 导出/推理配置
├── export/
│   ├── __init__.py
│   ├── export_ppo_discrete.py  ← 导出 RL+MPC 的离散 PPO（入门）
│   └── export_student.py       ← 导出 Sim-to-Real Student（主示例）
├── verify/
│   ├── __init__.py
│   └── accuracy_check.py      ← PyTorch vs ONNX 精度对比
├── quantize/
│   ├── __init__.py
│   └── fp16_quantize.py       ← FP32 → FP16 量化
├── inference/
│   ├── __init__.py
│   └── onnx_runner.py         ← ONNX Runtime 推理封装
├── benchmark.py                ← 速度对比 (PyTorch / FP32 / FP16)
├── results/                    ← 导出的 .onnx 文件 + benchmark 结果
└── docs/
    └── theory.md               ← 部署教程（导出 + 量化原理 + 面试）
```

## 5. 导出的模型

| 模型 | 来源 | 输入形状 | 输出形状 | 复杂度 |
|---|---|---|---|---|
| PPO PolicyNet (离散) | RL+MPC demo | (1, 25) float | (1, 5) softmax | 入门 |
| Student Base Policy | Sim-to-Real | (1, 43) float | (1, 8) float | 主示例 |
| Adaptation Module | Sim-to-Real | (1, 1350) float | (1, 16) float | 主示例 |

Student 示例展示复合模型导出（两个子网络顺序运行）。

## 6. 导出细节

### 6.1 基础导出（PPO 离散）

```python
import torch

model = PolicyNet(state_dim=25, hidden_dim=128, action_dim=5)
model.load_state_dict(...)
model.eval()

dummy_input = torch.randn(1, 25)
torch.onnx.export(
    model, dummy_input, "ppo_discrete.onnx",
    input_names=["observation"],
    output_names=["action_probs"],
    dynamic_axes={"observation": {0: "batch"}, "action_probs": {0: "batch"}},
    opset_version=17,
)
```

### 6.2 复合导出（Student）

Adaptation Module 和 Base Policy 分别导出为独立 ONNX 文件（允许独立优化）：

```python
# Adaptation Module
adapt_input = torch.randn(1, 1350)  # 50 帧 * 27 维
torch.onnx.export(adapt_module, adapt_input, "adaptation_module.onnx", ...)

# Base Policy
policy_input = torch.randn(1, 43)   # obs(27) + latent_z(16)
torch.onnx.export(base_policy, policy_input, "base_policy.onnx", ...)
```

## 7. 精度验证

对 1000 个随机输入比较输出：

```python
import numpy as np
import onnxruntime as ort

# PyTorch 推理
with torch.no_grad():
    pytorch_output = model(test_input).numpy()

# ONNX Runtime 推理
session = ort.InferenceSession("model.onnx")
onnx_output = session.run(None, {"observation": test_input.numpy()})[0]

# 对比
max_diff = np.max(np.abs(pytorch_output - onnx_output))
assert max_diff < 1e-5, f"Accuracy mismatch: {max_diff}"
```

## 8. FP16 量化

### 8.1 方法

使用 ONNX Runtime 内置 FP16 转换：

```python
from onnxruntime.transformers import float16
import onnx

model_fp32 = onnx.load("model_fp32.onnx")
model_fp16 = float16.convert_float_to_float16(model_fp32)
onnx.save(model_fp16, "model_fp16.onnx")
```

### 8.2 验证

- 精度：FP32 和 FP16 输出最大差异 < 1e-3
- 文件大小：FP16 约为 FP32 的 50%
- 速度：FP16 在 GPU 上约快 1.5-2x

## 9. Benchmark

### 9.1 方法

- 1000 次推理，丢弃前 100 次（warmup）
- 测量：平均延迟、P95 延迟、吞吐量（推理次数/秒）
- 对比：PyTorch (GPU) / ONNX FP32 (GPU) / ONNX FP16 (GPU)

### 9.2 预期结果

```
┌──────────────────┬───────────┬───────────┬───────────┐
│ 模型             │ PyTorch   │ ONNX FP32 │ ONNX FP16 │
├──────────────────┼───────────┼───────────┼───────────┤
│ PPO Discrete     │ ~0.5ms    │ ~0.2ms    │ ~0.1ms    │
│ Student+Adapt    │ ~1.0ms    │ ~0.4ms    │ ~0.2ms    │
├──────────────────┼───────────┼───────────┼───────────┤
│ 精度 (FP32)      │ 基准      │ < 1e-5    │ -         │
│ 精度 (FP16)      │ 基准      │ -         │ < 1e-3    │
├──────────────────┼───────────┼───────────┼───────────┤
│ 文件大小         │ N/A       │ ~100%     │ ~50%      │
└──────────────────┴───────────┴───────────┴───────────┘
```

## 10. 教程文档（theory.md）

### 大纲

1. **为什么需要 ONNX：从训练到部署**
   - 部署鸿沟
   - ONNX 作为通用中间格式

2. **导出：torch.onnx.export 详解**
   - 静态 vs 动态 axes
   - Opset 版本
   - 常见坑（不支持的算子、动态控制流）

3. **量化：FP32 → FP16 → INT8**
   - 量化是什么（用更少的 bit 表示权重）
   - 精度-速度 trade-off
   - 何时用哪种精度

4. **部署目标（概览）**
   - ONNX Runtime (CPU/GPU)
   - TensorRT (NVIDIA GPU, Jetson/Orin)
   - TFLite (移动端/MCU)

5. **面试高频问题 + 参考回答**
   - "你的 RL 模型怎么部署到机器人上？"
   - "推理速度怎么优化？"
   - "量化后精度损失多少？"

## 11. 依赖

- `onnx`：模型格式 (pip install onnx)
- `onnxruntime-gpu`：推理引擎 (pip install onnxruntime-gpu)
- `torch`：导出源
- `numpy`：数值对比

## 12. 交付物

1. 2 种模型的导出脚本（离散 PPO + Student 复合模型）
2. 精度验证工具（自动 PyTorch vs ONNX 对比）
3. FP16 量化脚本 + 精度检查
4. Benchmark 脚本（延迟/吞吐对比表）
5. 双语教程文档（导出 + 量化 + 部署概览）
