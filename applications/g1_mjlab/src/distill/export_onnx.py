"""Export trained G1 student policy to ONNX for deployment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tyro
import torch
import numpy as np
from dataclasses import dataclass

from src.distill.student_network import StudentPolicy


@dataclass
class ExportArgs:
    checkpoint: str = "applications/g1_mjlab/results/student_final.pt"
    output_path: str = "applications/g1_mjlab/results/student_policy.onnx"
    obs_dim: int = 84
    action_dim: int = 12
    history_length: int = 20
    latent_dim: int = 32
    opset_version: int = 17


def main():
    args = tyro.cli(ExportArgs)

    model = StudentPolicy(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        history_length=args.history_length,
        latent_dim=args.latent_dim,
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded student model: {args.checkpoint}")

    dummy_input = torch.randn(1, args.history_length, args.obs_dim)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=args.opset_version,
        input_names=["obs_history"],
        output_names=["action"],
        dynamic_axes={
            "obs_history": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )
    print(f"Exported ONNX: {output_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    test_input = np.random.randn(1, args.history_length, args.obs_dim).astype(np.float32)
    onnx_output = sess.run(None, {"obs_history": test_input})[0]

    with torch.no_grad():
        pt_output = model(torch.FloatTensor(test_input)).numpy()

    max_diff = np.abs(onnx_output - pt_output).max()
    print(f"ONNX vs PyTorch max diff: {max_diff:.8f}")
    assert max_diff < 1e-5, f"ONNX verification failed: max_diff={max_diff}"
    print("ONNX verification PASSED")


if __name__ == "__main__":
    main()
