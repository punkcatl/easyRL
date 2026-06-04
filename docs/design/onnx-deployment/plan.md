# ONNX Deployment Implementation Plan

**Goal:** Build a complete ONNX deployment pipeline: export trained PyTorch RL models to ONNX, verify accuracy, quantize to FP16, and benchmark inference speed. Covers both a simple model (discrete PPO) and composite model (Student + Adaptation Module).

**Architecture:** Export scripts convert PyTorch models to ONNX via torch.onnx.export. Verification compares PyTorch vs ONNX Runtime outputs on random inputs. Quantization uses ONNX float16 conversion. Benchmark measures latency across PyTorch/ONNX-FP32/ONNX-FP16.

**Tech Stack:** Python 3.9, PyTorch, onnx, onnxruntime-gpu, numpy

**Prerequisites:** `pip install onnx onnxruntime-gpu`

---

## File Structure

```
applications/onnx_deployment/
├── __init__.py
├── config.py
├── export/
│   ├── __init__.py
│   ├── export_ppo_discrete.py
│   └── export_student.py
├── verify/
│   ├── __init__.py
│   └── accuracy_check.py
├── quantize/
│   ├── __init__.py
│   └── fp16_quantize.py
├── inference/
│   ├── __init__.py
│   └── onnx_runner.py
├── benchmark.py
├── results/
└── docs/
    └── theory.md
```

---

### Task 1: Project Skeleton + Config

**Files:**
- Create: `applications/onnx_deployment/__init__.py`
- Create: `applications/onnx_deployment/config.py`
- Create: `applications/onnx_deployment/export/__init__.py`
- Create: `applications/onnx_deployment/verify/__init__.py`
- Create: `applications/onnx_deployment/quantize/__init__.py`
- Create: `applications/onnx_deployment/inference/__init__.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p applications/onnx_deployment/{export,verify,quantize,inference,results,docs}
```

- [ ] **Step 2: Create `__init__.py` files**

```bash
touch applications/onnx_deployment/__init__.py
touch applications/onnx_deployment/export/__init__.py
touch applications/onnx_deployment/verify/__init__.py
touch applications/onnx_deployment/quantize/__init__.py
touch applications/onnx_deployment/inference/__init__.py
```

- [ ] **Step 3: Write config.py**

```python
# applications/onnx_deployment/config.py

config = {
    # Export settings
    "opset_version": 17,

    # Models to export
    "models": {
        "ppo_discrete": {
            "input_shape": (1, 25),
            "output_names": ["action_probs"],
            "description": "RL+MPC discrete PPO policy network",
        },
        "student_adaptation": {
            "input_shape": (1, 1350),  # 50 frames * 27 obs_dim
            "output_names": ["latent_z"],
            "description": "Sim-to-Real RMA adaptation module",
        },
        "student_base_policy": {
            "input_shape": (1, 43),  # 27 obs + 16 latent
            "output_names": ["action"],
            "description": "Sim-to-Real student base policy",
        },
    },

    # Verification
    "verify_n_samples": 1000,
    "verify_atol_fp32": 1e-5,
    "verify_atol_fp16": 1e-3,

    # Benchmark
    "benchmark_n_runs": 1000,
    "benchmark_warmup": 100,
}
```

- [ ] **Step 4: Commit**

```bash
git add applications/onnx_deployment/
git commit -m "feat(onnx): add project skeleton and config"
```

---

### Task 2: Export Discrete PPO

**Files:**
- Create: `applications/onnx_deployment/export/export_ppo_discrete.py`

- [ ] **Step 1: Write export_ppo_discrete.py**

```python
# applications/onnx_deployment/export/export_ppo_discrete.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

from config import config


class PolicyNetDiscrete(nn.Module):
    """Standalone discrete PPO policy for export (no dependencies on training code)."""

    def __init__(self, state_dim=25, hidden_dim=128, action_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


def export_ppo_discrete(model_weights_path: str = None, output_path: str = None):
    """Export discrete PPO policy to ONNX.

    Args:
        model_weights_path: path to .pth file (if None, exports randomly initialized model)
        output_path: path for .onnx output
    """
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = output_path or str(results_dir / "ppo_discrete.onnx")

    model = PolicyNetDiscrete()

    if model_weights_path:
        checkpoint = torch.load(model_weights_path, map_location="cpu")
        # Handle different save formats
        if "actor" in checkpoint:
            model.load_state_dict(checkpoint["actor"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # Export
    dummy_input = torch.randn(1, 25)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["action_probs"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action_probs": {0: "batch_size"},
        },
        opset_version=config["opset_version"],
        do_constant_folding=True,
    )

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"Exported PPO discrete policy to: {output_path}")
    print(f"  Input: observation (batch, 25)")
    print(f"  Output: action_probs (batch, 5)")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="Path to .pth model weights")
    parser.add_argument("--output", default=None, help="Output .onnx path")
    args = parser.parse_args()
    export_ppo_discrete(args.weights, args.output)
```

- [ ] **Step 2: Verify export works**

```bash
cd applications/onnx_deployment
python3 export/export_ppo_discrete.py
# Should print "Exported PPO discrete policy to: results/ppo_discrete.onnx"
```

- [ ] **Step 3: Commit**

```bash
git add applications/onnx_deployment/export/export_ppo_discrete.py
git commit -m "feat(onnx): add discrete PPO export script"
```

---

### Task 3: Export Student (Composite Model)

**Files:**
- Create: `applications/onnx_deployment/export/export_student.py`

- [ ] **Step 1: Write export_student.py**

```python
# applications/onnx_deployment/export/export_student.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import onnx

from config import config


class AdaptationModuleExport(nn.Module):
    """Standalone adaptation module for export."""

    def __init__(self, input_dim=1350, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, obs_history):
        return self.net(obs_history)


class BasePolicyExport(nn.Module):
    """Standalone base policy for export."""

    def __init__(self, input_dim=43, action_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs_and_z):
        return self.net(obs_and_z)


def export_student(model_weights_path: str = None, output_dir: str = None):
    """Export Student (Adaptation Module + Base Policy) as two ONNX files.

    Args:
        model_weights_path: path to student .pth file
        output_dir: directory for .onnx outputs
    """
    results_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    adaptation = AdaptationModuleExport()
    base_policy = BasePolicyExport()

    if model_weights_path:
        checkpoint = torch.load(model_weights_path, map_location="cpu")
        if "adaptation" in checkpoint:
            adaptation.load_state_dict(checkpoint["adaptation"], strict=False)
        if "base_policy" in checkpoint:
            base_policy.load_state_dict(checkpoint["base_policy"], strict=False)

    adaptation.eval()
    base_policy.eval()

    # Export Adaptation Module
    adapt_path = str(results_dir / "adaptation_module.onnx")
    dummy_history = torch.randn(1, 1350)
    torch.onnx.export(
        adaptation,
        dummy_history,
        adapt_path,
        input_names=["obs_history"],
        output_names=["latent_z"],
        dynamic_axes={
            "obs_history": {0: "batch_size"},
            "latent_z": {0: "batch_size"},
        },
        opset_version=config["opset_version"],
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(adapt_path))
    print(f"Exported Adaptation Module to: {adapt_path}")
    print(f"  Input: obs_history (batch, 1350)")
    print(f"  Output: latent_z (batch, 16)")

    # Export Base Policy
    policy_path = str(results_dir / "base_policy.onnx")
    dummy_input = torch.randn(1, 43)
    torch.onnx.export(
        base_policy,
        dummy_input,
        policy_path,
        input_names=["obs_and_z"],
        output_names=["action"],
        dynamic_axes={
            "obs_and_z": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        opset_version=config["opset_version"],
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(policy_path))
    print(f"Exported Base Policy to: {policy_path}")
    print(f"  Input: obs_and_z (batch, 43)")
    print(f"  Output: action (batch, 8)")

    return adapt_path, policy_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    export_student(args.weights, args.output_dir)
```

- [ ] **Step 2: Verify**

```bash
python3 export/export_student.py
```

- [ ] **Step 3: Commit**

```bash
git add applications/onnx_deployment/export/export_student.py
git commit -m "feat(onnx): add Student composite model export (adaptation + base policy)"
```

---

### Task 4: Accuracy Verification

**Files:**
- Create: `applications/onnx_deployment/verify/accuracy_check.py`

- [ ] **Step 1: Write accuracy_check.py**

```python
# applications/onnx_deployment/verify/accuracy_check.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import onnxruntime as ort

from config import config


def verify_accuracy(pytorch_model, onnx_path: str, input_shape: tuple,
                    input_name: str = "observation", atol: float = 1e-5):
    """Compare PyTorch model output vs ONNX Runtime output.

    Args:
        pytorch_model: PyTorch model (eval mode)
        onnx_path: path to .onnx file
        input_shape: shape of single input (without batch dim in first position)
        input_name: name of ONNX input node
        atol: absolute tolerance for comparison

    Returns:
        dict with max_diff, mean_diff, passed (bool)
    """
    n_samples = config["verify_n_samples"]

    # Setup ONNX Runtime session
    session = ort.InferenceSession(onnx_path)

    max_diff = 0.0
    all_diffs = []

    for _ in range(n_samples):
        # Random input
        test_input = np.random.randn(*input_shape).astype(np.float32)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(torch.FloatTensor(test_input)).numpy()

        # ONNX Runtime inference
        onnx_output = session.run(None, {input_name: test_input})[0]

        # Compare
        diff = np.abs(pytorch_output - onnx_output)
        max_diff = max(max_diff, diff.max())
        all_diffs.append(diff.mean())

    mean_diff = np.mean(all_diffs)
    passed = max_diff < atol

    result = {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "atol": atol,
        "passed": passed,
        "n_samples": n_samples,
    }

    status = "PASS" if passed else "FAIL"
    print(f"Accuracy check [{status}]: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, atol={atol:.0e}")

    return result


def verify_all():
    """Verify all exported models."""
    results_dir = Path(__file__).resolve().parent.parent / "results"

    from export.export_ppo_discrete import PolicyNetDiscrete
    from export.export_student import AdaptationModuleExport, BasePolicyExport

    print("=" * 60)
    print("ONNX Accuracy Verification")
    print("=" * 60)

    # PPO Discrete
    ppo_path = results_dir / "ppo_discrete.onnx"
    if ppo_path.exists():
        print(f"\n1. PPO Discrete ({ppo_path.name})")
        model = PolicyNetDiscrete()
        model.eval()
        verify_accuracy(model, str(ppo_path), (1, 25), "observation", config["verify_atol_fp32"])

    # Adaptation Module
    adapt_path = results_dir / "adaptation_module.onnx"
    if adapt_path.exists():
        print(f"\n2. Adaptation Module ({adapt_path.name})")
        model = AdaptationModuleExport()
        model.eval()
        verify_accuracy(model, str(adapt_path), (1, 1350), "obs_history", config["verify_atol_fp32"])

    # Base Policy
    policy_path = results_dir / "base_policy.onnx"
    if policy_path.exists():
        print(f"\n3. Base Policy ({policy_path.name})")
        model = BasePolicyExport()
        model.eval()
        verify_accuracy(model, str(policy_path), (1, 43), "obs_and_z", config["verify_atol_fp32"])


if __name__ == "__main__":
    verify_all()
```

- [ ] **Step 2: Run verification**

```bash
python3 verify/accuracy_check.py
```

Expected: all models show PASS with max_diff < 1e-5.

- [ ] **Step 3: Commit**

```bash
git add applications/onnx_deployment/verify/accuracy_check.py
git commit -m "feat(onnx): add accuracy verification (PyTorch vs ONNX Runtime)"
```

---

### Task 5: FP16 Quantization

**Files:**
- Create: `applications/onnx_deployment/quantize/fp16_quantize.py`

- [ ] **Step 1: Write fp16_quantize.py**

```python
# applications/onnx_deployment/quantize/fp16_quantize.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

from config import config


def quantize_to_fp16(input_path: str, output_path: str = None):
    """Convert ONNX model from FP32 to FP16.

    Args:
        input_path: path to FP32 .onnx file
        output_path: path for FP16 output (default: append _fp16)

    Returns:
        output_path, size_reduction_ratio
    """
    if output_path is None:
        base = input_path.rsplit(".onnx", 1)[0]
        output_path = f"{base}_fp16.onnx"

    # Load and convert
    model_fp32 = onnx.load(input_path)
    model_fp16 = convert_float_to_float16(model_fp32)
    onnx.save(model_fp16, output_path)

    # Check sizes
    size_fp32 = os.path.getsize(input_path)
    size_fp16 = os.path.getsize(output_path)
    ratio = size_fp16 / size_fp32

    print(f"Quantized: {Path(input_path).name} → {Path(output_path).name}")
    print(f"  FP32 size: {size_fp32 / 1024:.1f} KB")
    print(f"  FP16 size: {size_fp16 / 1024:.1f} KB")
    print(f"  Reduction: {ratio:.1%}")

    return output_path, ratio


def quantize_all():
    """Quantize all exported models to FP16."""
    results_dir = Path(__file__).resolve().parent.parent / "results"

    print("=" * 60)
    print("FP16 Quantization")
    print("=" * 60)

    onnx_files = list(results_dir.glob("*.onnx"))
    # Skip already quantized files
    onnx_files = [f for f in onnx_files if "_fp16" not in f.name]

    for filepath in onnx_files:
        print(f"\n{filepath.name}:")
        quantize_to_fp16(str(filepath))


if __name__ == "__main__":
    quantize_all()
```

- [ ] **Step 2: Run quantization**

```bash
python3 quantize/fp16_quantize.py
```

Expected: FP16 files created, size ~50% of FP32.

- [ ] **Step 3: Commit**

```bash
git add applications/onnx_deployment/quantize/fp16_quantize.py
git commit -m "feat(onnx): add FP16 quantization script"
```

---

### Task 6: ONNX Runtime Inference Runner

**Files:**
- Create: `applications/onnx_deployment/inference/onnx_runner.py`

- [ ] **Step 1: Write onnx_runner.py**

```python
# applications/onnx_deployment/inference/onnx_runner.py
import numpy as np
import onnxruntime as ort


class OnnxRunner:
    """Lightweight ONNX Runtime inference wrapper."""

    def __init__(self, model_path: str, device: str = "gpu"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get input shape for validation
        input_info = self.session.get_inputs()[0]
        self.input_shape = input_info.shape
        self.model_path = model_path

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference on numpy array.

        Args:
            input_array: numpy array matching model input shape

        Returns:
            output numpy array
        """
        if input_array.dtype != np.float32:
            input_array = input_array.astype(np.float32)
        result = self.session.run([self.output_name], {self.input_name: input_array})
        return result[0]

    def info(self):
        """Print model info."""
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        print(f"Model: {self.model_path}")
        for inp in inputs:
            print(f"  Input: {inp.name} shape={inp.shape} dtype={inp.type}")
        for out in outputs:
            print(f"  Output: {out.name} shape={out.shape} dtype={out.type}")


class StudentRunner:
    """Composite runner for Student: Adaptation Module → Base Policy."""

    def __init__(self, adaptation_path: str, policy_path: str, device: str = "gpu"):
        self.adaptation = OnnxRunner(adaptation_path, device)
        self.policy = OnnxRunner(policy_path, device)

    def __call__(self, obs_history: np.ndarray, obs_current: np.ndarray) -> np.ndarray:
        """Run full Student inference.

        Args:
            obs_history: (batch, 1350) observation history
            obs_current: (batch, 27) current observation

        Returns:
            action: (batch, 8)
        """
        latent_z = self.adaptation(obs_history)  # (batch, 16)
        obs_and_z = np.concatenate([obs_current, latent_z], axis=-1)  # (batch, 43)
        action = self.policy(obs_and_z)  # (batch, 8)
        return action
```

- [ ] **Step 2: Verify runner**

```bash
python3 -c "
from inference.onnx_runner import OnnxRunner, StudentRunner
import numpy as np

# Test PPO
runner = OnnxRunner('results/ppo_discrete.onnx', device='cpu')
runner.info()
output = runner(np.random.randn(1, 25).astype(np.float32))
print(f'PPO output shape: {output.shape}, sum: {output.sum():.4f}')  # should sum to ~1.0

# Test Student composite
student = StudentRunner('results/adaptation_module.onnx', 'results/base_policy.onnx', device='cpu')
history = np.random.randn(1, 1350).astype(np.float32)
obs = np.random.randn(1, 27).astype(np.float32)
action = student(history, obs)
print(f'Student action shape: {action.shape}')
"
```

- [ ] **Step 3: Commit**

```bash
git add applications/onnx_deployment/inference/onnx_runner.py
git commit -m "feat(onnx): add ONNX Runtime inference runner"
```

---

### Task 7: Benchmark Script

**Files:**
- Create: `applications/onnx_deployment/benchmark.py`

- [ ] **Step 1: Write benchmark.py**

```python
# applications/onnx_deployment/benchmark.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import numpy as np
import torch

from config import config
from export.export_ppo_discrete import PolicyNetDiscrete
from export.export_student import AdaptationModuleExport, BasePolicyExport
from inference.onnx_runner import OnnxRunner


def benchmark_pytorch(model, input_shape, n_runs, warmup):
    """Benchmark PyTorch inference."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
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


def benchmark_onnx(onnx_path, input_shape, n_runs, warmup, device="gpu"):
    """Benchmark ONNX Runtime inference."""
    runner = OnnxRunner(onnx_path, device=device)
    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        runner(dummy)

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        runner(dummy)
        latencies.append((time.perf_counter() - start) * 1000)

    return np.array(latencies)


def run_benchmark():
    """Run full benchmark comparison."""
    results_dir = Path(__file__).resolve().parent / "results"
    n_runs = config["benchmark_n_runs"]
    warmup = config["benchmark_warmup"]

    print("=" * 70)
    print("ONNX Deployment Benchmark")
    print("=" * 70)
    print(f"Runs: {n_runs}, Warmup: {warmup}")
    print()

    benchmarks = [
        ("PPO Discrete", PolicyNetDiscrete(), (1, 25), "ppo_discrete"),
        ("Adaptation Module", AdaptationModuleExport(), (1, 1350), "adaptation_module"),
        ("Base Policy", BasePolicyExport(), (1, 43), "base_policy"),
    ]

    print(f"{'Model':<22} {'Runtime':<14} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput':<12}")
    print("-" * 70)

    for name, pytorch_model, input_shape, file_prefix in benchmarks:
        # PyTorch
        latencies = benchmark_pytorch(pytorch_model, input_shape, n_runs, warmup)
        print(f"{name:<22} {'PyTorch':<14} {latencies.mean():<12.3f} {np.percentile(latencies, 95):<12.3f} {1000/latencies.mean():<12.0f}")

        # ONNX FP32
        fp32_path = results_dir / f"{file_prefix}.onnx"
        if fp32_path.exists():
            latencies = benchmark_onnx(str(fp32_path), input_shape, n_runs, warmup)
            print(f"{'':<22} {'ONNX FP32':<14} {latencies.mean():<12.3f} {np.percentile(latencies, 95):<12.3f} {1000/latencies.mean():<12.0f}")

        # ONNX FP16
        fp16_path = results_dir / f"{file_prefix}_fp16.onnx"
        if fp16_path.exists():
            latencies = benchmark_onnx(str(fp16_path), input_shape, n_runs, warmup)
            print(f"{'':<22} {'ONNX FP16':<14} {latencies.mean():<12.3f} {np.percentile(latencies, 95):<12.3f} {1000/latencies.mean():<12.0f}")

        print()


if __name__ == "__main__":
    run_benchmark()
```

- [ ] **Step 2: Run benchmark**

```bash
python3 benchmark.py
```

Expected: table showing latency comparison across PyTorch/FP32/FP16.

- [ ] **Step 3: Commit**

```bash
git add applications/onnx_deployment/benchmark.py
git commit -m "feat(onnx): add inference benchmark (PyTorch vs ONNX FP32 vs FP16)"
```

---

### Task 8: Theory Documentation

**Files:**
- Create: `applications/onnx_deployment/docs/theory.md`

- [ ] **Step 1: Write theory.md (bilingual)**

Content:
1. Why ONNX: the deployment gap
2. Export: torch.onnx.export explained (dynamic axes, opset, pitfalls)
3. Quantization: FP32 → FP16 → INT8 (what it does, accuracy-speed trade-off)
4. Deployment targets overview (ONNX Runtime, TensorRT, TFLite)
5. Interview FAQ

- [ ] **Step 2: Commit**

```bash
git add applications/onnx_deployment/docs/theory.md
git commit -m "docs(onnx): add deployment tutorial with interview content"
```

---

### Task 9: Integration Test (Full Pipeline)

- [ ] **Step 1: Export all models**

```bash
cd applications/onnx_deployment
python3 export/export_ppo_discrete.py
python3 export/export_student.py
```

- [ ] **Step 2: Verify accuracy**

```bash
python3 verify/accuracy_check.py
```

Expected: all PASS.

- [ ] **Step 3: Quantize to FP16**

```bash
python3 quantize/fp16_quantize.py
```

Expected: FP16 files created, ~50% size.

- [ ] **Step 4: Verify FP16 accuracy**

```bash
python3 -c "
from verify.accuracy_check import verify_accuracy
from export.export_ppo_discrete import PolicyNetDiscrete
from config import config

model = PolicyNetDiscrete()
model.eval()
result = verify_accuracy(model, 'results/ppo_discrete_fp16.onnx', (1, 25), 'observation', config['verify_atol_fp16'])
"
```

Expected: PASS with max_diff < 1e-3.

- [ ] **Step 5: Run benchmark**

```bash
python3 benchmark.py
```

- [ ] **Step 6: Final commit if fixes needed**

```bash
git add -A applications/onnx_deployment/
git commit -m "fix(onnx): integration test fixes"
```

---

## Execution Order Summary

| Task | Component | Depends On |
|------|-----------|------------|
| 1 | Skeleton + Config | — |
| 2 | Export PPO Discrete | 1 |
| 3 | Export Student | 1 |
| 4 | Accuracy Verification | 2, 3 |
| 5 | FP16 Quantization | 2, 3 |
| 6 | Inference Runner | 2, 3 |
| 7 | Benchmark | 5, 6 |
| 8 | Theory Docs | — |
| 9 | Integration Test | all |

Tasks 2, 3, 8 are independent. Tasks 4, 5, 6 depend on exports being done first.
