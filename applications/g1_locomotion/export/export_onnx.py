"""Phase 3: Export student policy to ONNX format.

Fuses AdaptationModule + StudentPolicy into a single graph.

Usage:
    python applications/g1_locomotion/export/export_onnx.py \
        --student_path results/.../student/student_best.pt \
        --output results/student_g1.onnx
"""

import argparse
import os

import numpy as np
import torch

from applications.g1_locomotion.student.networks import (
    AdaptationModule,
    StudentONNXWrapper,
    StudentPolicy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export G1 student to ONNX.")
    parser.add_argument("--student_path", type=str, required=True, help="Student checkpoint path.")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify accuracy.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    # Load student
    print(f"[INFO] Loading student from: {args.student_path}")
    ckpt = torch.load(args.student_path, map_location=device)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    history_dim = ckpt["history_dim"]
    latent_dim = ckpt["latent_dim"]

    adaptation = AdaptationModule(input_dim=history_dim, latent_dim=latent_dim)
    policy = StudentPolicy(obs_dim=obs_dim, latent_dim=latent_dim, action_dim=action_dim)
    adaptation.load_state_dict(ckpt["adaptation"])
    policy.load_state_dict(ckpt["policy"])

    # Fuse into ONNX wrapper
    wrapper = StudentONNXWrapper(adaptation, policy)
    wrapper.eval()

    # Dummy inputs
    dummy_history = torch.randn(1, history_dim)
    dummy_obs = torch.randn(1, obs_dim)

    # Export
    output_path = args.output or os.path.join(os.path.dirname(args.student_path), "student_g1.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[INFO] Exporting to: {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy_history, dummy_obs),
        output_path,
        opset_version=args.opset,
        input_names=["obs_history", "obs_current"],
        output_names=["action"],
        dynamic_axes={
            "obs_history": {0: "batch"},
            "obs_current": {0: "batch"},
            "action": {0: "batch"},
        },
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] Export complete: {output_path} ({file_size:.2f} MB)")

    # Verify accuracy
    if args.verify:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)

        # Run PyTorch
        with torch.no_grad():
            pt_output = wrapper(dummy_history, dummy_obs).numpy()

        # Run ONNX
        ort_output = session.run(
            None,
            {
                "obs_history": dummy_history.numpy(),
                "obs_current": dummy_obs.numpy(),
            },
        )[0]

        max_diff = np.abs(pt_output - ort_output).max()
        print(f"[INFO] Accuracy check: max|PyTorch - ONNX| = {max_diff:.2e}", end=" ")
        if max_diff < 1e-4:
            print("✓ PASS")
        else:
            print("✗ FAIL")

    print("\nExport Summary:")
    print(f"  Model: {os.path.basename(output_path)}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Input: obs_history ({history_dim},) + obs_current ({obs_dim},)")
    print(f"  Output: action ({action_dim},)")
    print(f"  Opset: {args.opset}")


if __name__ == "__main__":
    main()
