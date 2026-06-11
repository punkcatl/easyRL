# applications/go2_locomotion/benchmark.py
"""Go2 Motion Test Suite — structured policy verification.

Usage:
    python applications/go2_locomotion/benchmark.py --tag v1
    python applications/go2_locomotion/benchmark.py --tag v1 --render
    python applications/go2_locomotion/benchmark.py --tag v1 --only unit
    python applications/go2_locomotion/benchmark.py --tag v1 --only sequence
    python applications/go2_locomotion/benchmark.py --tag v1 --only perturbation
    python applications/go2_locomotion/benchmark.py --tag v1 --sequence my_seq.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import time
import datetime
import numpy as np
import torch

from applications.go2_locomotion.config import config
from applications.go2_locomotion.envs.go2_env import Go2Env
from applications.go2_locomotion.agent.ppo import PPOTrainer


UNIT_TEST_CASES = [
    ("forward_slow",  [0.5,  0.0,  0.0]),
    ("forward_fast",  [1.0,  0.0,  0.0]),
    ("reverse",       [-0.5, 0.0,  0.0]),
    ("lateral_left",  [0.0,  0.5,  0.0]),
    ("lateral_right", [0.0, -0.5,  0.0]),
    ("rotate_left",   [0.0,  0.0,  1.0]),
    ("rotate_right",  [0.0,  0.0, -1.0]),
    ("combined",      [0.8,  0.3,  0.5]),
]

UNIT_TEST_DURATION_S = 5.0

DEFAULT_SEQUENCE = [
    ([1.0,  0.0,  0.0], 3.0, "forward"),
    ([0.5,  0.0,  1.0], 2.0, "forward_turn"),
    ([1.0,  0.0,  0.0], 2.0, "forward2"),
    ([0.0,  0.5,  0.0], 1.0, "lateral"),
    ([-0.5, 0.0,  0.0], 2.0, "reverse"),
    ([0.0,  0.0,  0.0], 1.0, "brake"),
]

PERTURBATION_SHOCK_FORCE = 8.0
PERTURBATION_SHOCK_DURATION = 5
PERTURBATION_RECOVERY_STEPS = 250
PERTURBATION_WALK_STEPS = 150
PERTURBATION_FORCE_LEVELS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]


def _make_env(render: bool = False) -> Go2Env:
    return Go2Env(config, render_mode="human" if render else None)


def _throttle(step_start: float, render: bool):
    if render:
        elapsed = time.perf_counter() - step_start
        sleep_time = config["control_dt"] - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def run_unit_tests(policy_fn, render: bool = False) -> dict:
    """Run 8 fixed-command unit tests. Returns dict of per-test metrics."""
    steps_per_test = int(UNIT_TEST_DURATION_S / config["control_dt"])
    results = {}

    for name, command in UNIT_TEST_CASES:
        env = _make_env(render)
        obs, _ = env.reset()
        env.command = np.array(command, dtype=np.float32)

        vx_errors, vy_errors, yaw_errors = [], [], []
        survived = True

        for _ in range(steps_per_test):
            step_start = time.perf_counter()

            # Override command each step (prevent auto-resample from changing it)
            env.command = np.array(command, dtype=np.float32)

            action = policy_fn(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            vel = env._get_base_linear_velocity()
            ang_vel = env._get_base_angular_velocity()
            vx_errors.append(abs(command[0] - vel[0]))
            vy_errors.append(abs(command[1] - vel[1]))
            yaw_errors.append(abs(command[2] - ang_vel[2]))

            _throttle(step_start, render)

            if terminated:
                survived = False
                break

        results[name] = {
            "survived": survived,
            "rmse_vx":  float(np.sqrt(np.mean(np.array(vx_errors) ** 2))),
            "rmse_vy":  float(np.sqrt(np.mean(np.array(vy_errors) ** 2))),
            "rmse_yaw": float(np.sqrt(np.mean(np.array(yaw_errors) ** 2))),
        }
        env.close()

    return results


def _print_unit_results(results: dict):
    print("\n[1/3] Unit Tests")
    passed = 0
    for name, r in results.items():
        status = "YES" if r["survived"] else "NO "
        print(f"  {name:<16} RMSE vx={r['rmse_vx']:.3f}  "
              f"vy={r['rmse_vy']:.3f}  yaw={r['rmse_yaw']:.3f}  survived={status}")
        if r["survived"]:
            passed += 1
    print(f"  Unit pass rate: {passed}/{len(results)}")


def run_sequence_test(policy_fn, sequence=None, render: bool = False) -> dict:
    """Run a multi-segment sequence test without resetting between segments.

    sequence: list of (command [vx,vy,yaw], duration_s, name) tuples.
    """
    if sequence is None:
        sequence = DEFAULT_SEQUENCE

    env = _make_env(render)
    obs, _ = env.reset()

    survived = True
    per_segment = []
    all_rmse = []

    for command, duration_s, seg_name in sequence:
        if not survived:
            per_segment.append({"name": seg_name, "rmse": None, "survived": False})
            continue

        steps = int(duration_s / config["control_dt"])
        command_arr = np.array(command, dtype=np.float32)
        errors = []

        for _ in range(steps):
            step_start = time.perf_counter()
            env.command = command_arr

            action = policy_fn(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            vel = env._get_base_linear_velocity()
            ang_vel = env._get_base_angular_velocity()
            err = np.sqrt(
                (command[0] - vel[0])**2 +
                (command[1] - vel[1])**2 +
                (command[2] - ang_vel[2])**2
            )
            errors.append(float(err))

            _throttle(step_start, render)

            if terminated:
                survived = False
                break

        seg_rmse = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else None
        per_segment.append({"name": seg_name, "rmse": seg_rmse, "survived": survived})
        if seg_rmse is not None:
            all_rmse.append(seg_rmse)

    env.close()
    avg_rmse = float(np.mean(all_rmse)) if all_rmse else None
    return {"survived": survived, "avg_rmse": avg_rmse, "per_segment": per_segment}


def _print_sequence_results(results: dict):
    print("\n[2/3] Sequence Test")
    for seg in results["per_segment"]:
        status = "YES" if seg["survived"] else "NO "
        rmse_str = f"{seg['rmse']:.3f}" if seg["rmse"] is not None else "N/A"
        print(f"  Segment {seg['name']:<16} RMSE={rmse_str}  survived={status}")
    avg = f"{results['avg_rmse']:.3f}" if results["avg_rmse"] is not None else "N/A"
    overall = "PASS" if results["survived"] else "FAIL"
    print(f"  Overall: {overall}  avg_rmse={avg}")


def _walk_n_steps(env, policy_fn, n_steps, command, render):
    """Walk for n_steps with fixed command. Returns (obs, survived)."""
    command_arr = np.array(command, dtype=np.float32)
    obs, _ = env.reset()
    for _ in range(n_steps):
        step_start = time.perf_counter()
        env.command = command_arr
        action = policy_fn(obs)
        obs, _, terminated, _, _ = env.step(action)
        _throttle(step_start, render)
        if terminated:
            return obs, False
    return obs, True


def run_perturbation_tests(policy_fn, render: bool = False) -> dict:
    """Run single shock and escalating max-force perturbation tests."""
    walk_command = [1.0, 0.0, 0.0]

    # --- Single shock test ---
    env = _make_env(render)
    obs, stable = _walk_n_steps(
        env, policy_fn, PERTURBATION_WALK_STEPS, walk_command, render
    )

    shock_survived = False
    recovery_steps = -1

    if stable:
        env.apply_force(
            np.array([0.0, PERTURBATION_SHOCK_FORCE, 0.0]),
            duration_steps=PERTURBATION_SHOCK_DURATION,
        )
        command_arr = np.array(walk_command, dtype=np.float32)
        for step_i in range(PERTURBATION_RECOVERY_STEPS):
            step_start = time.perf_counter()
            env.command = command_arr
            action = policy_fn(obs)
            obs, _, terminated, _, _ = env.step(action)
            _throttle(step_start, render)
            if terminated:
                break
            vel = env._get_base_linear_velocity()
            if abs(vel[0] - walk_command[0]) < 0.2 and recovery_steps < 0:
                recovery_steps = step_i + 1
            shock_survived = not terminated

    env.close()
    single_shock_result = {
        "force_n": PERTURBATION_SHOCK_FORCE,
        "survived": shock_survived,
        "recovery_steps": recovery_steps,
    }

    # --- Max force escalating test ---
    max_survived_force = 0.0
    for force in PERTURBATION_FORCE_LEVELS:
        env = _make_env(render)
        obs, stable = _walk_n_steps(
            env, policy_fn, PERTURBATION_WALK_STEPS, walk_command, render
        )

        if not stable:
            env.close()
            break

        env.apply_force(
            np.array([0.0, force, 0.0]),
            duration_steps=PERTURBATION_SHOCK_DURATION,
        )
        command_arr = np.array(walk_command, dtype=np.float32)
        fell = False
        for _ in range(150):
            step_start = time.perf_counter()
            env.command = command_arr
            action = policy_fn(obs)
            obs, _, terminated, _, _ = env.step(action)
            _throttle(step_start, render)
            if terminated:
                fell = True
                break

        env.close()
        if not fell:
            max_survived_force = force
        else:
            break

    return {
        "single_shock": single_shock_result,
        "max_force": {"max_survived_force": max_survived_force},
    }


def _print_perturbation_results(results: dict):
    print("\n[3/3] Perturbation Tests")
    s = results["single_shock"]
    status = "YES" if s["survived"] else "NO "
    rec = s["recovery_steps"] if s["recovery_steps"] >= 0 else "fell"
    print(f"  Single shock ({s['force_n']:.0f}N): survived={status}  recovery_steps={rec}")
    print(f"  Max force test:     max_survived_force={results['max_force']['max_survived_force']:.0f}N")


def save_results(results: dict, results_dir: str = None) -> str:
    """Save benchmark results to JSON. Returns path to saved file."""
    if results_dir is None:
        results_dir = str(Path(__file__).resolve().parent / "results")
    Path(results_dir).mkdir(exist_ok=True)
    tag = results.get("tag", "untagged")
    path = str(Path(results_dir) / f"benchmark_{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def run_benchmark(
    model_path: str,
    tag: str,
    only: str = None,
    sequence=None,
    render: bool = False,
) -> dict:
    """Run full benchmark suite and return results dict."""
    trainer = PPOTrainer(config)
    trainer.load(model_path)
    trainer.network.eval()
    device = trainer.device

    def policy_fn(obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, _ = trainer.network.forward_actor(obs_t)
        return mean.cpu().numpy().flatten()

    results = {
        "tag": tag,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model_path,
    }

    header = f"  Go2 Motion Test Suite  [tag: {tag}]"
    print("\n" + "=" * len(header))
    print(header)
    print(f"  Model: {model_path}")
    print("=" * len(header))

    if only in (None, "unit"):
        results["unit_tests"] = run_unit_tests(policy_fn, render=render)
        _print_unit_results(results["unit_tests"])

    if only in (None, "sequence"):
        results["sequence_test"] = run_sequence_test(
            policy_fn, sequence=sequence, render=render
        )
        _print_sequence_results(results["sequence_test"])

    if only in (None, "perturbation"):
        results["perturbation"] = run_perturbation_tests(policy_fn, render=render)
        _print_perturbation_results(results["perturbation"])

    # Summary
    print("\n" + "=" * len(header))
    print("  SUMMARY")
    if "unit_tests" in results:
        passed = sum(1 for r in results["unit_tests"].values() if r["survived"])
        total = len(results["unit_tests"])
        print(f"  Unit tests:     {passed}/{total} passed")
    if "sequence_test" in results:
        seq = results["sequence_test"]
        status = "PASS" if seq["survived"] else "FAIL"
        avg = f"{seq['avg_rmse']:.3f}" if seq["avg_rmse"] else "N/A"
        print(f"  Sequence test:  {status}  avg_rmse={avg}")
    if "perturbation" in results:
        mf = results["perturbation"]["max_force"]["max_survived_force"]
        print(f"  Max force:      {mf:.0f} N")
    print("=" * len(header))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 Motion Test Suite")
    parser.add_argument("--tag", type=str, default=None,
                        help="Version tag for results file (default: timestamp)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .pth model (default: results/teacher_final.pth)")
    parser.add_argument("--only", choices=["unit", "sequence", "perturbation"],
                        default=None, help="Run only one test category")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Path to custom sequence JSON file")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent / "results"
    if args.model is None:
        args.model = str(results_dir / "teacher_final.pth")

    if args.tag is None:
        args.tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    custom_sequence = None
    if args.sequence:
        with open(args.sequence) as f:
            custom_sequence = json.load(f)

    results = run_benchmark(
        model_path=args.model,
        tag=args.tag,
        only=args.only,
        sequence=custom_sequence,
        render=args.render,
    )
    path = save_results(results)
    print(f"Saved to {path}")
