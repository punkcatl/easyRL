"""Inference speed benchmark: PyTorch vs ONNX FP32 vs ONNX FP16."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import numpy as np
import numpy.typing as npt
import torch
import onnxruntime as ort

from config import config
from export.export_ppo_discrete import PolicyNetDiscrete
from export.export_student import AdaptationModuleExport, BasePolicyExport
from inference.onnx_runner import OnnxRunner


def benchmark_pytorch(model: torch.nn.Module, input_shape: tuple,
                      n_runs: int, warmup: int) -> npt.NDArray[np.float64]:
    """Measure PyTorch inference latency."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    return np.array(latencies)


def benchmark_onnx(onnx_path: str, input_shape: tuple,
                   n_runs: int, warmup: int, device: str = "cpu") -> npt.NDArray[np.float64]:
    """Measure ONNX Runtime inference latency."""
    runner = OnnxRunner(onnx_path, device=device)
    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(warmup):
        runner(dummy)

    # Pre-cast to model's expected dtype so the cast is not included in timed calls
    dummy = dummy.astype(runner._input_dtype)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        runner(dummy)
        latencies.append((time.perf_counter() - start) * 1000)

    return np.array(latencies)


def print_row(name, runtime, latencies):
    """Print a formatted benchmark row."""
    mean = latencies.mean()
    p95 = np.percentile(latencies, 95)
    throughput = 1000.0 / mean
    print(f"{name:<22} {runtime:<14} {mean:<12.3f} {p95:<12.3f} {throughput:<12.0f}")


def run_benchmark() -> None:
    """Run full benchmark: PyTorch / ONNX FP32 / ONNX FP16 for all models."""
    results_dir = Path(__file__).resolve().parent / "results"
    n_runs = config["benchmark_n_runs"]
    warmup = config["benchmark_warmup"]

    # Use GPU for ONNX only if CUDAExecutionProvider is actually available
    available_providers = ort.get_available_providers()
    onnx_device = "gpu" if "CUDAExecutionProvider" in available_providers else "cpu"
    pytorch_device = "CUDA" if torch.cuda.is_available() else "CPU"

    print("=" * 72)
    print("ONNX Deployment Benchmark")
    print("=" * 72)
    print(f"Runs: {n_runs} | Warmup: {warmup}")
    print(f"PyTorch device: {pytorch_device} | ONNX Runtime device: {onnx_device.upper()}")
    print()
    print(f"{'Model':<22} {'Runtime':<14} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput':<12}")
    print("-" * 72)

    # Map config model keys to (model_class, file_prefix)
    model_registry = {
        "ppo_discrete": (PolicyNetDiscrete, "ppo_discrete"),
        "student_adaptation": (AdaptationModuleExport, "adaptation_module"),
        "student_base_policy": (BasePolicyExport, "base_policy"),
    }

    has_fp16 = False
    for model_key, model_meta in config["models"].items():
        model_class, file_prefix = model_registry[model_key]
        input_shape = model_meta["input_shape"]
        name = model_meta["description"]

        # PyTorch
        pytorch_model = model_class()
        latencies = benchmark_pytorch(pytorch_model, input_shape, n_runs, warmup)
        print_row(name, "PyTorch", latencies)

        # ONNX FP32
        fp32_path = results_dir / f"{file_prefix}.onnx"
        if fp32_path.exists():
            latencies = benchmark_onnx(str(fp32_path), input_shape, n_runs, warmup, onnx_device)
            print_row("", "ONNX FP32", latencies)

        # ONNX FP16
        fp16_path = results_dir / f"{file_prefix}_fp16.onnx"
        if fp16_path.exists():
            latencies = benchmark_onnx(str(fp16_path), input_shape, n_runs, warmup, onnx_device)
            print_row("", "ONNX FP16", latencies)
            has_fp16 = True

        print()

    if has_fp16 and onnx_device == "cpu":
        print("Note: FP16 on CPU is typically slower than FP32 due to lack of"
              " hardware FP16 acceleration.")
        print("FP16 benefits are realized on GPUs with Tensor Cores"
              " (e.g., NVIDIA T4, A100).")
        print()

    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
