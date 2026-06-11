"""Export trained Student policy to ONNX for edge deployment.

Wraps AdaptationModule + StudentPolicy into a single ONNX graph.
Verifies numerical accuracy vs PyTorch and benchmarks inference latency.

Usage:
    python applications/go2_locomotion/export_onnx.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from applications.go2_locomotion.config import config
from applications.go2_locomotion.agent.teacher_student import StudentAgent


class StudentONNXWrapper(nn.Module):
    """Single-forward-pass wrapper for ONNX export.

    Fuses AdaptationModule + StudentPolicy so the ONNX graph has
    two inputs (obs_history, obs_current) and one output (action).
    """

    def __init__(self, student: StudentAgent):
        super().__init__()
        self.adaptation = student.adaptation
        self.policy = student.policy

    def forward(self, obs_history_flat: torch.Tensor, obs_current: torch.Tensor) -> torch.Tensor:
        z = self.adaptation(obs_history_flat)
        return self.policy(obs_current, z)


def export_student_onnx(model_path: str = None, output_path: str = None) -> str:
    """Export student model to ONNX. Returns path to exported file."""
    results_dir = Path(__file__).resolve().parent / "results"

    if model_path is None:
        model_path = str(results_dir / "student_final.pth")
    if output_path is None:
        output_path = str(results_dir / "student_go2.onnx")

    obs_dim = config["obs_dim"]
    history_length = config["student_history_length"]
    history_flat_dim = obs_dim * history_length
    action_dim = config["action_dim"]

    # Load student (or use random init if no model file)
    student = StudentAgent(config)
    if Path(model_path).exists():
        student.load(model_path)
        print(f"Loaded student from {model_path}")
    else:
        print(f"No model at {model_path} — using random init for benchmark")

    wrapper = StudentONNXWrapper(student).cpu().eval()

    dummy_history = torch.randn(1, history_flat_dim)
    dummy_obs = torch.randn(1, obs_dim)

    torch.onnx.export(
        wrapper,
        (dummy_history, dummy_obs),
        output_path,
        input_names=["obs_history", "obs_current"],
        output_names=["action"],
        opset_version=config["onnx_opset_version"],
        dynamic_axes={
            "obs_history": {0: "batch"},
            "obs_current": {0: "batch"},
            "action": {0: "batch"},
        },
    )

    # Verify ONNX model structure
    model = onnx.load(output_path)
    onnx.checker.check_model(model)

    print(f"\nONNX export successful: {output_path}")
    print(f"  obs_history input:  ({history_flat_dim},)")
    print(f"  obs_current input:  ({obs_dim},)")
    print(f"  action output:      ({action_dim},)")

    # Accuracy check: PyTorch vs ONNX
    session = ort.InferenceSession(output_path)
    hist_np = np.random.randn(1, history_flat_dim).astype(np.float32)
    obs_np = np.random.randn(1, obs_dim).astype(np.float32)

    with torch.no_grad():
        pt_out = wrapper(torch.FloatTensor(hist_np), torch.FloatTensor(obs_np)).numpy()
    onnx_out = session.run(None, {"obs_history": hist_np, "obs_current": obs_np})[0]

    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    accuracy_ok = max_diff < 1e-4
    print(f"\nAccuracy: max|PyTorch - ONNX| = {max_diff:.2e}  "
          f"({'PASS' if accuracy_ok else 'WARN: exceeds 1e-4'})")

    # Latency benchmark
    n_warmup = 100
    n_runs = 1000

    for _ in range(n_warmup):
        session.run(None, {"obs_history": hist_np, "obs_current": obs_np})

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"obs_history": hist_np, "obs_current": obs_np})
    elapsed_ms = (time.perf_counter() - start) / n_runs * 1000

    max_freq = 1000.0 / elapsed_ms
    budget_ok = elapsed_ms < 20.0  # Go2 needs 50 Hz = 20ms budget

    print(f"\nInference Benchmark ({n_runs} runs):")
    print(f"  Avg latency : {elapsed_ms:.3f} ms")
    print(f"  Max freq    : {max_freq:.0f} Hz")
    print(f"  50 Hz budget: {'PASS' if budget_ok else 'FAIL'} (need < 20 ms)")

    return output_path


if __name__ == "__main__":
    export_student_onnx()
