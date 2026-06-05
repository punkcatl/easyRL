# ONNX Deployment

Complete model deployment pipeline: export trained PyTorch RL models to ONNX format, verify accuracy consistency, quantize to FP16, and benchmark inference speed.

## Overview

This module demonstrates the full "training to deployment" workflow:

1. **Export** - Convert PyTorch models to ONNX via `torch.onnx.export()`
2. **Verify** - Compare PyTorch vs ONNX Runtime outputs (max diff < 1e-5)
3. **Quantize** - FP32 to FP16 conversion (~50% file size reduction)
4. **Benchmark** - Latency comparison across PyTorch / ONNX FP32 / ONNX FP16

Two model types are supported:
- **PPO Discrete** (beginner) - single policy network, input (1,25) output (1,5)
- **Student Composite** (main) - Adaptation Module + Base Policy exported as separate files

## Prerequisites

```bash
pip install onnx onnxruntime torch numpy
```

For GPU acceleration:

```bash
pip install onnxruntime-gpu
```

## Quick Start

Run the full pipeline from the `applications/onnx_deployment/` directory:

```bash
cd applications/onnx_deployment

# Step 1: Export models to ONNX
python export/export_ppo_discrete.py
python export/export_student.py

# Step 2: Verify accuracy (PyTorch vs ONNX Runtime)
python verify/accuracy_check.py

# Step 3: Quantize FP32 -> FP16
python quantize/fp16_quantize.py

# Step 4: Run inference benchmark
python benchmark.py
```

Exported `.onnx` files are saved to `results/`.

## Using the Inference Runner

```python
from inference.onnx_runner import OnnxRunner, StudentRunner
import numpy as np

# Single model
runner = OnnxRunner("results/ppo_discrete.onnx", device="cpu")
action_probs = runner(np.random.randn(1, 25).astype(np.float32))

# Composite Student model
student = StudentRunner(
    "results/adaptation_module.onnx",
    "results/base_policy.onnx",
    device="cpu",
)
obs_history = np.random.randn(1, 1350).astype(np.float32)
obs_current = np.random.randn(1, 27).astype(np.float32)
action = student(obs_history, obs_current)
```

## Module Structure

```
applications/onnx_deployment/
├── __init__.py
├── config.py                   <- export/inference configuration
├── export/
│   ├── export_ppo_discrete.py  <- discrete PPO export (beginner)
│   └── export_student.py       <- Student composite export (main)
├── verify/
│   └── accuracy_check.py       <- PyTorch vs ONNX accuracy comparison
├── quantize/
│   └── fp16_quantize.py        <- FP32 -> FP16 quantization
├── inference/
│   └── onnx_runner.py          <- ONNX Runtime inference wrapper
├── benchmark.py                <- speed comparison script
├── results/                    <- exported .onnx files
└── docs/
    └── theory.md               <- deployment tutorial
```

---

# ONNX 部署

完整的模型部署流水线：将训练好的 PyTorch RL 模型导出为 ONNX 格式，验证精度一致性，量化为 FP16，并进行推理速度基准测试。

## 概述

本模块展示完整的"训练到部署"工作流：

1. **导出** - 通过 `torch.onnx.export()` 将 PyTorch 模型转换为 ONNX
2. **验证** - 对比 PyTorch 与 ONNX Runtime 输出（最大差异 < 1e-5）
3. **量化** - FP32 转 FP16（文件大小减少约 50%）
4. **基准测试** - PyTorch / ONNX FP32 / ONNX FP16 延迟对比

支持两种模型类型：
- **PPO Discrete**（入门）- 单一策略网络，输入 (1,25) 输出 (1,5)
- **Student Composite**（主示例）- Adaptation Module + Base Policy 分别导出

## 前置依赖

```bash
pip install onnx onnxruntime torch numpy
```

GPU 加速：

```bash
pip install onnxruntime-gpu
```

## 快速开始

在 `applications/onnx_deployment/` 目录下运行完整流水线：

```bash
cd applications/onnx_deployment

# Step 1: 导出模型为 ONNX
python export/export_ppo_discrete.py
python export/export_student.py

# Step 2: 验证精度（PyTorch vs ONNX Runtime）
python verify/accuracy_check.py

# Step 3: 量化 FP32 -> FP16
python quantize/fp16_quantize.py

# Step 4: 运行推理基准测试
python benchmark.py
```

导出的 `.onnx` 文件保存在 `results/` 目录。

## 使用推理 Runner

```python
from inference.onnx_runner import OnnxRunner, StudentRunner
import numpy as np

# 单模型
runner = OnnxRunner("results/ppo_discrete.onnx", device="cpu")
action_probs = runner(np.random.randn(1, 25).astype(np.float32))

# 复合 Student 模型
student = StudentRunner(
    "results/adaptation_module.onnx",
    "results/base_policy.onnx",
    device="cpu",
)
obs_history = np.random.randn(1, 1350).astype(np.float32)
obs_current = np.random.randn(1, 27).astype(np.float32)
action = student(obs_history, obs_current)
```

## 模块结构

```
applications/onnx_deployment/
├── __init__.py
├── config.py                   <- 导出/推理配置
├── export/
│   ├── export_ppo_discrete.py  <- 离散 PPO 导出（入门）
│   └── export_student.py       <- Student 复合模型导出（主示例）
├── verify/
│   └── accuracy_check.py       <- PyTorch vs ONNX 精度对比
├── quantize/
│   └── fp16_quantize.py        <- FP32 -> FP16 量化
├── inference/
│   └── onnx_runner.py          <- ONNX Runtime 推理封装
├── benchmark.py                <- 速度对比脚本
├── results/                    <- 导出的 .onnx 文件
└── docs/
    └── theory.md               <- 部署教程
```
