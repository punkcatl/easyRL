"""FP32 to FP16 quantization for ONNX models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

from config import config


def quantize_to_fp16(input_path: str, output_path: str = None):
    """Convert an ONNX model from FP32 to FP16.

    Args:
        input_path: path to FP32 .onnx file
        output_path: path for FP16 output (default: append _fp16 suffix)

    Returns:
        (output_path, size_ratio) where size_ratio = fp16_size / fp32_size
    """
    if output_path is None:
        base = input_path.rsplit(".onnx", 1)[0]
        output_path = f"{base}_fp16.onnx"

    model_fp32 = onnx.load(input_path)
    model_fp16 = convert_float_to_float16(model_fp32)
    onnx.save(model_fp16, output_path)

    size_fp32 = os.path.getsize(input_path)
    size_fp16 = os.path.getsize(output_path)
    ratio = size_fp16 / size_fp32

    print(f"  {Path(input_path).name} -> {Path(output_path).name}")
    print(f"    FP32: {size_fp32 / 1024:.1f} KB")
    print(f"    FP16: {size_fp16 / 1024:.1f} KB")
    print(f"    Ratio: {ratio:.1%}")

    return output_path, ratio


def quantize_all():
    """Quantize all FP32 ONNX models in results/ to FP16."""
    results_dir = Path(__file__).resolve().parent.parent / "results"

    print("=" * 60)
    print("FP16 Quantization")
    print("=" * 60)

    onnx_files = sorted(results_dir.glob("*.onnx"))
    onnx_files = [f for f in onnx_files if "_fp16" not in f.name]

    if not onnx_files:
        print("\nNo FP32 ONNX files found. Run export scripts first.")
        return

    for filepath in onnx_files:
        print(f"\n{filepath.name}:")
        quantize_to_fp16(str(filepath))


if __name__ == "__main__":
    quantize_all()
