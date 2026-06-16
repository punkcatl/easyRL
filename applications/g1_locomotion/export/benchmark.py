"""Benchmark ONNX model inference latency.

Usage:
    python applications/g1_locomotion/export/benchmark.py \
        --model results/.../student_g1.onnx --runs 1000
"""

import argparse
import time

import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark G1 student ONNX inference.")
    parser.add_argument("--model", type=str, required=True, help="ONNX model path.")
    parser.add_argument("--runs", type=int, default=1000, help="Number of inference runs.")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup runs.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading model: {args.model}")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    # Get input shapes
    inputs = session.get_inputs()
    input_shapes = {inp.name: inp.shape for inp in inputs}
    print(f"[INFO] Inputs: {input_shapes}")

    # Create dummy inputs (batch=1)
    feed = {}
    for inp in inputs:
        shape = [1 if isinstance(d, str) else d for d in inp.shape]
        feed[inp.name] = np.random.randn(*shape).astype(np.float32)

    # Warmup
    print(f"[INFO] Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        session.run(None, feed)

    # Benchmark
    print(f"[INFO] Benchmarking ({args.runs} runs)...")
    latencies = []
    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(None, feed)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    # Report
    import os
    file_size = os.path.getsize(args.model) / (1024 * 1024)

    print("\n" + "=" * 50)
    print("G1 Student ONNX Benchmark Results")
    print("=" * 50)
    print(f"  Model:    {os.path.basename(args.model)}")
    print(f"  Size:     {file_size:.2f} MB")
    print(f"  Runs:     {args.runs}")
    print(f"  Latency:")
    print(f"    avg:    {latencies.mean():.3f} ms")
    print(f"    p50:    {np.percentile(latencies, 50):.3f} ms")
    print(f"    p95:    {np.percentile(latencies, 95):.3f} ms")
    print(f"    p99:    {np.percentile(latencies, 99):.3f} ms")
    print(f"    max:    {latencies.max():.3f} ms")
    print(f"  Budget:   20ms (50Hz control)")
    budget_pass = latencies.mean() < 20.0
    print(f"  Status:   {'✓ PASS' if budget_pass else '✗ FAIL'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
