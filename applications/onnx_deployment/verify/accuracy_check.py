"""Accuracy verification: compare PyTorch model output vs ONNX Runtime output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import onnxruntime as ort

from config import config


def verify_accuracy(pytorch_model, onnx_path: str, input_shape: tuple,
                    input_name: str = "observation", atol: float = 1e-5):
    """Compare PyTorch model output vs ONNX Runtime output on random inputs.

    Args:
        pytorch_model: PyTorch model in eval mode
        onnx_path: path to exported .onnx file
        input_shape: input tensor shape including batch dim, e.g. (1, 25)
        input_name: name of the ONNX input node
        atol: absolute tolerance threshold

    Returns:
        dict with max_diff, mean_diff, passed (bool), n_samples
    """
    n_samples = config["verify_n_samples"]
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Detect model's expected input dtype from ONNX session metadata
    input_info = session.get_inputs()[0]
    is_fp16 = input_info.type == "tensor(float16)"

    max_diff = 0.0
    all_diffs = []

    for _ in range(n_samples):
        test_input = np.random.randn(*input_shape).astype(np.float32)

        with torch.no_grad():
            pytorch_output = pytorch_model(torch.from_numpy(test_input)).numpy()

        # Cast input to FP16 if the ONNX model expects float16
        onnx_input = test_input.astype(np.float16) if is_fp16 else test_input
        onnx_output = session.run(None, {input_name: onnx_input})[0]

        # Cast both outputs to float32 for fair comparison
        pytorch_cmp = pytorch_output.astype(np.float32)
        onnx_cmp = onnx_output.astype(np.float32)

        diff = np.abs(pytorch_cmp - onnx_cmp)
        max_diff = max(max_diff, float(diff.max()))
        all_diffs.append(float(diff.mean()))

    mean_diff = np.mean(all_diffs)
    passed = max_diff < atol

    result = {
        "max_diff": max_diff,
        "mean_diff": float(mean_diff),
        "atol": atol,
        "passed": passed,
        "n_samples": n_samples,
    }

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, atol={atol:.0e}")
    return result


def _load_model_with_weights(model_class, pth_path):
    """Instantiate model and load saved weights from export step."""
    if not pth_path.exists():
        raise FileNotFoundError(
            f"Weight file not found: {pth_path}. Run the export step first."
        )
    model = model_class()
    model.load_state_dict(torch.load(str(pth_path), map_location="cpu", weights_only=True))
    model.eval()
    return model


def verify_all():
    """Verify all exported models in results/ directory."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export"))
    from export_ppo_discrete import PolicyNetDiscrete
    from export_student import AdaptationModuleExport, BasePolicyExport

    results_dir = Path(__file__).resolve().parent.parent / "results"

    # Map config model keys to (model_class, file_prefix, input_name)
    model_registry = {
        "ppo_discrete": (PolicyNetDiscrete, "ppo_discrete", "observation"),
        "student_adaptation": (AdaptationModuleExport, "adaptation_module", "obs_history"),
        "student_base_policy": (BasePolicyExport, "base_policy", "obs_and_z"),
    }

    print("=" * 60)
    print("ONNX Accuracy Verification")
    print("=" * 60)

    all_passed = True

    # --- FP32 Verification ---
    print("\n--- FP32 Verification (atol=1e-5) ---")
    for idx, (model_key, model_meta) in enumerate(config["models"].items(), 1):
        model_class, file_prefix, input_name = model_registry[model_key]
        input_shape = model_meta["input_shape"]

        fp32_path = results_dir / f"{file_prefix}.onnx"
        if fp32_path.exists():
            print(f"\n{idx}. {model_meta['description']} ({fp32_path.name})")
            model = _load_model_with_weights(model_class, results_dir / f"{file_prefix}.pth")
            r = verify_accuracy(model, str(fp32_path), input_shape, input_name,
                                config["verify_atol_fp32"])
            all_passed &= r["passed"]
        else:
            print(f"\n{idx}. {model_meta['description']}: SKIPPED (run export first)")

    # --- FP16 Verification ---
    print("\n--- FP16 Verification (atol=1e-3) ---")
    for idx, (model_key, model_meta) in enumerate(config["models"].items(), 1):
        model_class, file_prefix, input_name = model_registry[model_key]
        input_shape = model_meta["input_shape"]

        fp16_path = results_dir / f"{file_prefix}_fp16.onnx"
        if fp16_path.exists():
            print(f"\n{idx}. {model_meta['description']} FP16 ({fp16_path.name})")
            model = _load_model_with_weights(model_class, results_dir / f"{file_prefix}.pth")
            r = verify_accuracy(model, str(fp16_path), input_shape, input_name,
                                config["verify_atol_fp16"])
            all_passed &= r["passed"]
        else:
            print(f"\n{idx}. {model_meta['description']} FP16: SKIPPED (run export/quantize first)")

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    verify_all()
