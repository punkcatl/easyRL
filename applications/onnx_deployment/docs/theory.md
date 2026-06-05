# ONNX Deployment: From Training to Production

## 1. Why ONNX: The Deployment Gap

After training an RL policy in PyTorch, you cannot simply copy the `.pth` file to a robot controller and run it. The target device typically lacks Python, CUDA, and the full PyTorch runtime (~2GB).

ONNX (Open Neural Network Exchange) solves this by defining a universal intermediate representation. Any framework can export to ONNX, and any runtime can load it:

```
PyTorch  ─┐                    ┌─ ONNX Runtime (CPU/GPU)
TensorFlow ├─► .onnx file ◄───├─ TensorRT (NVIDIA)
JAX      ─┘                    └─ TFLite (mobile/MCU)
```

Key benefits over direct PyTorch deployment:
- Runtime size: ~50MB vs ~2GB
- No Python dependency on target
- Graph-level optimizations (operator fusion, constant folding)
- Hardware-specific acceleration via provider backends

## 2. Export: torch.onnx.export Explained

### Basic Usage

```python
torch.onnx.export(
    model,              # PyTorch model (must be in eval mode)
    dummy_input,        # example input tensor for tracing
    "model.onnx",      # output path
    input_names=["obs"],
    output_names=["action"],
    opset_version=17,
)
```

PyTorch traces the model by running `dummy_input` through it, recording all operations into a static graph.

### Dynamic Axes

By default, all dimensions are fixed. Use `dynamic_axes` for variable batch size:

```python
dynamic_axes={
    "obs": {0: "batch_size"},      # dim 0 is dynamic
    "action": {0: "batch_size"},
}
```

### Opset Version

The opset version determines which ONNX operators are available. Higher versions support more operators. Use opset 17+ for modern models.

### Common Pitfalls

| Problem | Cause | Solution |
|---------|-------|----------|
| Export fails | Unsupported op (e.g., custom CUDA kernel) | Rewrite using standard PyTorch ops |
| Dynamic control flow | Python if/for that depends on tensor values | Use `torch.where()` or unroll loops |
| Shape mismatch at runtime | Forgot `dynamic_axes` | Add dynamic axes for all variable dims |
| Accuracy differs | Non-deterministic ops (dropout) | Ensure `model.eval()` before export |

## 3. Quantization: FP32 to FP16 to INT8

### What Quantization Does

Reduces the precision of model weights and activations:

```
FP32: 32 bits per parameter  (full precision)
FP16: 16 bits per parameter  (~50% smaller, minimal accuracy loss)
INT8:  8 bits per parameter  (~75% smaller, requires calibration)
```

### FP16 Conversion (Post-Training)

Simple cast of all float32 tensors to float16. No calibration data needed:

```python
from onnxruntime.transformers.float16 import convert_float_to_float16
import onnx

model = onnx.load("model_fp32.onnx")
model_fp16 = convert_float_to_float16(model)
onnx.save(model_fp16, "model_fp16.onnx")
```

Typical results for RL policies:
- Size reduction: ~50%
- Accuracy loss: max diff < 1e-3 (negligible for control)
- Speed improvement: 1.5-2x on GPU with FP16 tensor cores

### INT8 Quantization (Advanced)

Requires a calibration dataset to determine per-layer scale factors. More complex but gives 3-4x speedup on INT8-capable hardware (e.g., Jetson Orin). Not covered in this module.

### When to Use Which

| Precision | Use Case | Trade-off |
|-----------|----------|-----------|
| FP32 | Development, accuracy-critical | Baseline, no compression |
| FP16 | GPU deployment (Jetson, server) | Minimal loss, 2x faster |
| INT8 | Edge deployment (MCU, mobile) | Needs calibration, 4x faster |

## 4. Deployment Targets

### ONNX Runtime (This Module)

Cross-platform, supports CPU and GPU via execution providers. Best for prototyping and server deployment.

```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
output = session.run(None, {"obs": input_array})[0]
```

### TensorRT (NVIDIA GPU)

Maximum performance on NVIDIA hardware. Applies layer fusion, kernel auto-tuning, and precision calibration. Used on Jetson/Orin for robotics.

```
.onnx -> trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

### TFLite (Mobile/MCU)

For extremely constrained devices. Converts to flatbuffer format with INT8 quantization. Used on drones and micro-robots.

## 5. Interview FAQ

**Q: How do you deploy your RL model to a robot?**

A: Export the trained PyTorch policy to ONNX format using `torch.onnx.export()`. Verify that ONNX Runtime output matches PyTorch output within 1e-5 tolerance on 1000 random inputs. Quantize to FP16 for size and speed. On the robot controller, load with ONNX Runtime C++ or TensorRT depending on hardware.

**Q: How do you optimize inference speed?**

A: Three levels: (1) Graph optimization during export via `do_constant_folding=True`; (2) FP16 quantization for 1.5-2x speedup with negligible accuracy loss; (3) Hardware-specific runtime (TensorRT for NVIDIA, which applies operator fusion and kernel auto-tuning).

**Q: What accuracy loss do you get from quantization?**

A: FP16 post-training quantization gives max absolute difference < 1e-3 compared to FP32 for typical RL policies (3-layer MLPs). This is well within the noise floor of the policy itself. INT8 requires calibration and may lose 1-2% in task success rate if not done carefully.

**Q: Why export sub-networks separately instead of one big model?**

A: Independent optimization (different quantization per module), easier debugging, and flexibility to update one component without re-exporting everything. The Adaptation Module runs once per environment step while the Base Policy may run at higher frequency.

---

# ONNX 部署：从训练到生产

## 1. 为什么需要 ONNX：部署鸿沟

训练完 PyTorch RL 策略后，不能简单地把 `.pth` 文件复制到机器人主控板上运行。目标设备通常没有 Python、CUDA 和完整的 PyTorch 运行时（约 2GB）。

ONNX（Open Neural Network Exchange）通过定义通用中间表示来解决这个问题。任何框架都能导出为 ONNX，任何运行时都能加载它：

```
PyTorch  ─┐                    ┌─ ONNX Runtime (CPU/GPU)
TensorFlow ├─► .onnx file ◄───├─ TensorRT (NVIDIA)
JAX      ─┘                    └─ TFLite (mobile/MCU)
```

相比直接部署 PyTorch 的优势：
- 运行时大小：约 50MB vs 约 2GB
- 目标设备无需 Python
- 图级优化（算子融合、常量折叠）
- 通过 Provider 后端实现硬件特定加速

## 2. 导出：torch.onnx.export 详解

### 基本用法

```python
torch.onnx.export(
    model,              # PyTorch 模型（必须处于 eval 模式）
    dummy_input,        # 用于 trace 的示例输入张量
    "model.onnx",      # 输出路径
    input_names=["obs"],
    output_names=["action"],
    opset_version=17,
)
```

PyTorch 通过将 `dummy_input` 送入模型来 trace，记录所有操作为静态图。

### 动态维度（Dynamic Axes）

默认所有维度都是固定的。使用 `dynamic_axes` 支持可变 batch size：

```python
dynamic_axes={
    "obs": {0: "batch_size"},      # 第 0 维是动态的
    "action": {0: "batch_size"},
}
```

### Opset 版本

Opset 版本决定可用的 ONNX 算子集合。更高版本支持更多算子。现代模型建议使用 opset 17+。

### 常见坑

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 导出失败 | 不支持的算子（如自定义 CUDA kernel） | 用标准 PyTorch 算子重写 |
| 动态控制流 | Python if/for 依赖于 tensor 值 | 使用 `torch.where()` 或展开循环 |
| 运行时形状不匹配 | 忘记设置 `dynamic_axes` | 为所有可变维度添加动态轴 |
| 精度不一致 | 非确定性算子（dropout） | 导出前确保 `model.eval()` |

## 3. 量化：FP32 到 FP16 到 INT8

### 量化做了什么

降低模型权重和激活的精度：

```
FP32: 每参数 32 bit（全精度）
FP16: 每参数 16 bit（约小 50%，精度损失极小）
INT8: 每参数 8 bit （约小 75%，需要校准）
```

### FP16 转换（训练后量化）

将所有 float32 张量简单转为 float16。无需校准数据：

```python
from onnxruntime.transformers.float16 import convert_float_to_float16
import onnx

model = onnx.load("model_fp32.onnx")
model_fp16 = convert_float_to_float16(model)
onnx.save(model_fp16, "model_fp16.onnx")
```

RL 策略的典型结果：
- 大小减少：约 50%
- 精度损失：最大差异 < 1e-3（对控制任务可忽略）
- 速度提升：在有 FP16 Tensor Core 的 GPU 上快 1.5-2x

### INT8 量化（进阶）

需要校准数据集来确定每层的 scale factor。更复杂但在 INT8 硬件上（如 Jetson Orin）可获得 3-4x 加速。本模块不涉及。

### 何时使用哪种精度

| 精度 | 使用场景 | 权衡 |
|------|----------|------|
| FP32 | 开发调试、精度敏感 | 基准，无压缩 |
| FP16 | GPU 部署（Jetson、服务器） | 损失极小，快 2x |
| INT8 | 边缘部署（MCU、手机） | 需校准，快 4x |

## 4. 部署目标

### ONNX Runtime（本模块）

跨平台，通过 Execution Provider 支持 CPU 和 GPU。适合原型开发和服务端部署。

```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
output = session.run(None, {"obs": input_array})[0]
```

### TensorRT（NVIDIA GPU）

NVIDIA 硬件上的最高性能。应用层融合、kernel 自动调优和精度校准。在 Jetson/Orin 上用于机器人。

```
.onnx -> trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

### TFLite（移动端/MCU）

用于极度受限的设备。转换为 flatbuffer 格式并做 INT8 量化。用在无人机和微型机器人上。

## 5. 面试高频问题

**Q: 你的 RL 模型怎么部署到机器人上？**

A: 用 `torch.onnx.export()` 将训练好的 PyTorch 策略导出为 ONNX 格式。对 1000 个随机输入验证 ONNX Runtime 输出与 PyTorch 输出的差异在 1e-5 以内。量化为 FP16 减小体积并加速。在机器人主控板上用 ONNX Runtime C++ 或 TensorRT 加载，取决于硬件。

**Q: 推理速度怎么优化？**

A: 三个层面：（1）导出时的图优化，通过 `do_constant_folding=True`；（2）FP16 量化获得 1.5-2x 加速且精度损失可忽略；（3）硬件专用运行时（如 NVIDIA 的 TensorRT，会做算子融合和 kernel 自动调优）。

**Q: 量化后精度损失多少？**

A: FP16 训练后量化对典型 RL 策略（3 层 MLP）的最大绝对差异 < 1e-3。这远在策略本身的噪声范围内。INT8 需要校准，若处理不当可能在任务成功率上损失 1-2%。

**Q: 为什么分开导出子网络而不是导出一个大模型？**

A: 独立优化（不同模块可用不同量化策略）、更容易调试、以及可以单独更新某个组件而不用重新导出所有内容。Adaptation Module 每个环境 step 运行一次，而 Base Policy 可能以更高频率运行。
