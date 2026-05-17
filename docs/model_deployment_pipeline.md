# Model Deployment Pipeline for Autonomous Driving

## Overview

```
Training (.pth) → Export (.onnx) → Optimization (.engine/TensorRT)
```

## 1. Training Stage (.pth)

Train models on GPU servers using PyTorch. `.pth` files save complete network weights (state_dict), supporting continued training, fine-tuning, and debugging.

- Flexible: dynamic computation graphs, breakpoint debugging
- Slow for inference: Python GIL overhead, dynamic graph dispatch

## 2. Export Stage (.onnx)

ONNX (Open Neural Network Exchange) is a cross-framework intermediate representation by Microsoft/Facebook.

```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

**Purpose:**
- **Remove PyTorch dependency** — no need to install PyTorch on vehicle
- **Graph optimization** — operator fusion, constant folding, dead code elimination
- **Cross-platform** — same .onnx deploys to different hardware

## 3. Optimization Stage (.engine / TensorRT)

TensorRT is NVIDIA's inference acceleration engine, optimized for automotive GPUs (Orin, Xavier).

| Optimization | Effect |
|-------------|--------|
| FP16 / INT8 quantization | Minimal accuracy loss, 2-4x speedup |
| Layer fusion | Merge multiple layers into one CUDA kernel |
| Kernel auto-tuning | Select optimal implementation for specific hardware |
| Dynamic batch / workspace | Control GPU memory usage |

The output `.engine` file is **hardware-specific** — compiled on Orin cannot run on Xavier directly.

## Latency Comparison

| Runtime | Latency | Use Case |
|---------|---------|----------|
| PyTorch | ~50ms | Training, experiments |
| ONNX Runtime | ~15ms | Cloud inference, testing |
| TensorRT | ~3ms | Vehicle real-time control |

Autonomous driving planning & control requires **10ms-level response** (100Hz control frequency). Native PyTorch inference cannot meet this — TensorRT is mandatory.

## Production Workflow

```
Researcher (PyTorch training)
    ↓  deliver .pth + network definition
Deployment Engineer
    ↓  torch.onnx.export → verify accuracy consistency
    ↓  trtexec --onnx=model.onnx --fp16 → .engine
    ↓  Vehicle-side C++ inference (TensorRT API)
Vehicle goes online
```

## Model File Formats Across Frameworks

| Framework | Format | Extension |
|-----------|--------|-----------|
| PyTorch | state_dict serialization | `.pth`, `.pt`, `.bin` |
| TensorFlow/Keras | SavedModel / HDF5 | directory or `.h5`, `.keras` |
| ONNX | Cross-framework standard | `.onnx` |
| TensorRT | NVIDIA inference engine | `.engine`, `.trt` |
| JAX/Flax | msgpack / pickle | `.msgpack`, `.pkl` |

## Current Project

This project operates at the training stage. All models are saved as `.pth` files in each algorithm's `results/` directory. Adding ONNX export in the future requires only a few lines of code.

---

# 自动驾驶模型部署链路

## 概述

```
训练 (.pth) → 导出 (.onnx) → 优化 (.engine/TensorRT)
```

## 1. 训练阶段 (.pth)

在 GPU 服务器上用 PyTorch 训练模型。`.pth` 保存完整的网络权重（state_dict），支持继续训练、微调、调试。

- 灵活：动态计算图、断点调试
- 推理慢：Python GIL 开销、动态图调度

## 2. 导出阶段 (.onnx)

ONNX（Open Neural Network Exchange）是微软/Facebook 推出的跨框架中间表示。

```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

**作用：**
- **去掉 PyTorch 依赖** — 车端不需要安装 PyTorch
- **图优化** — 算子融合、常量折叠、死代码消除
- **跨平台** — 同一个 .onnx 可部署到不同硬件

## 3. 优化阶段 (.engine / TensorRT)

TensorRT 是 NVIDIA 的推理加速引擎，针对车载 GPU（Orin、Xavier）做极致优化。

| 优化手段 | 效果 |
|---------|------|
| FP16 / INT8 量化 | 精度微降，速度翻 2-4 倍 |
| Layer fusion | 多层合并为一个 CUDA kernel |
| Kernel auto-tuning | 针对具体硬件选最优实现 |
| 动态 batch / workspace | 控制显存占用 |

生成的 `.engine` 文件**绑定具体硬件** — 在 Orin 上编译的不能直接在 Xavier 上运行。

## 延迟对比

| 运行时 | 延迟 | 适用场景 |
|--------|------|---------|
| PyTorch | ~50ms | 训练、实验 |
| ONNX Runtime | ~15ms | 云端推理、测试 |
| TensorRT | ~3ms | 车端实时控制 |

自动驾驶规控要求 **10ms 级响应**（100Hz 控制频率）。PyTorch 原生推理根本达不到，必须走 TensorRT。

## 实际工作流

```
研究员（PyTorch 训练）
    ↓  交付 .pth + 网络定义
部署工程师
    ↓  torch.onnx.export → 验证精度一致
    ↓  trtexec --onnx=model.onnx --fp16 → .engine
    ↓  车端 C++ 推理（TensorRT API）
车辆上线
```

## 各框架模型文件格式

| 框架 | 格式 | 后缀 |
|------|------|------|
| PyTorch | state_dict 序列化 | `.pth`, `.pt`, `.bin` |
| TensorFlow/Keras | SavedModel / HDF5 | 目录 或 `.h5`, `.keras` |
| ONNX | 跨框架通用格式 | `.onnx` |
| TensorRT | NVIDIA 推理加速 | `.engine`, `.trt` |
| JAX/Flax | msgpack / pickle | `.msgpack`, `.pkl` |

## 当前项目

本项目处于训练阶段。所有模型以 `.pth` 格式保存在各算法的 `results/` 目录下。未来添加 ONNX 导出只需几行代码。
