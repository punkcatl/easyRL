"""Measure G1 training throughput (steps/sec) on current GPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import torch
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper


def benchmark(num_envs: int, num_steps: int = 1000):
    env_cfg = load_env_cfg("G1-Flat-v0")
    env_cfg.scene.num_envs = num_envs

    env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)

    obs, _ = wrapped.reset()
    actions = torch.randn(num_envs, wrapped.num_actions, device=wrapped.device)

    for _ in range(50):
        wrapped.step(actions)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_steps):
        wrapped.step(actions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    steps_per_sec = num_envs * num_steps / elapsed
    print(f"  num_envs={num_envs:>5d} | {num_steps} steps | "
          f"{elapsed:.2f}s | {steps_per_sec:,.0f} steps/sec")

    env.close()
    return steps_per_sec


def main():
    print("G1 mjlab Throughput Benchmark")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    results = {}
    for n in [256, 512, 1024, 2048]:
        try:
            sps = benchmark(n)
            results[n] = sps
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  num_envs={n:>5d} | OOM - skipping larger sizes")
                break
            raise

    print()
    print("Summary:")
    for n, sps in results.items():
        print(f"  {n} envs: {sps:,.0f} steps/sec")


if __name__ == "__main__":
    main()
