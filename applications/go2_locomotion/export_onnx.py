"""Export trained Teacher policy to ONNX for edge deployment.

Bakes obs normalization (running mean/std) + actor MLP + tanh clipping
into a single ONNX graph. No student/adaptation needed since teacher
actor is obs-only (no privileged input).

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
from applications.go2_locomotion.agent.ppo import PPOTrainer


class TeacherONNXWrapper(nn.Module):
    """obs -> normalized_obs -> actor -> tanh -> action.

    Bakes in obs_rms (mean/std) so the ONNX model takes raw obs directly.
    """

    def __init__(self, trainer: PPOTrainer):
        super().__init__()
        self.register_buffer("obs_mean", torch.FloatTensor(trainer.obs_rms.mean))
        self.register_buffer("obs_std", torch.FloatTensor(np.sqrt(trainer.obs_rms.var) + 1e-8))
        self.actor_net = trainer.network.actor_net
        self.actor_mean = trainer.network.actor_mean

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (obs - self.obs_mean) / self.obs_std
        features = self.actor_net(normalized)
        action = self.actor_mean(features)
        return torch.clamp(action, -1.0, 1.0)


def export_teacher_onnx(model_path: str = None, output_path: str = None) -> str:
    """Export teacher to ONNX. Returns path to exported file."""
    results_dir = Path(__file__).resolve().parent / "results"

    if model_path is None:
        model_path = str(results_dir / "teacher_final.pth")
    if output_path is None:
        output_path = str(results_dir / "policy_go2.onnx")

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return None

    obs_dim = config["obs_dim"]
    action_dim = config["action_dim"]

    trainer = PPOTrainer(config)
    trainer.load(model_path)
    trainer.network.eval()

    wrapper = TeacherONNXWrapper(trainer).cpu().eval()

    dummy_obs = torch.randn(1, obs_dim)

    torch.onnx.export(
        wrapper,
        dummy_obs,
        output_path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=config["onnx_opset_version"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action": {0: "batch"},
        },
    )

    model = onnx.load(output_path)
    onnx.checker.check_model(model)

    print(f"ONNX export successful: {output_path}")
    print(f"  obs input:     ({obs_dim},)")
    print(f"  action output: ({action_dim},)")

    # Accuracy check: PyTorch vs ONNX
    session = ort.InferenceSession(output_path)
    obs_np = np.random.randn(1, obs_dim).astype(np.float32)

    with torch.no_grad():
        pt_out = wrapper(torch.FloatTensor(obs_np)).numpy()
    onnx_out = session.run(None, {"obs": obs_np})[0]

    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    accuracy_ok = max_diff < 1e-5
    print(f"\nAccuracy: max|PyTorch - ONNX| = {max_diff:.2e}  "
          f"({'PASS' if accuracy_ok else 'WARN'})")

    # Latency benchmark
    n_warmup = 100
    n_runs = 1000

    for _ in range(n_warmup):
        session.run(None, {"obs": obs_np})

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"obs": obs_np})
    elapsed_ms = (time.perf_counter() - start) / n_runs * 1000

    max_freq = 1000.0 / elapsed_ms
    budget_ok = elapsed_ms < 20.0  # 50 Hz control loop

    print(f"\nInference Benchmark ({n_runs} runs):")
    print(f"  Avg latency : {elapsed_ms:.3f} ms")
    print(f"  Max freq    : {max_freq:.0f} Hz")
    print(f"  50 Hz budget: {'PASS' if budget_ok else 'FAIL'} (need < 20 ms)")

    file_size_kb = Path(output_path).stat().st_size / 1024
    print(f"\n  File size   : {file_size_kb:.1f} KB")

    return output_path


if __name__ == "__main__":
    export_teacher_onnx()
