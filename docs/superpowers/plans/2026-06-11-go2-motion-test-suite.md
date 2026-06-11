# Go2 Motion Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a structured Motion Test Suite (`benchmark.py`) for trained Go2 locomotion policies, covering unit tests, sequence tests, and perturbation tests, with terminal output and versioned JSON results.

**Architecture:** Add `apply_force()` to `Go2Env` for external impulse injection. Implement `benchmark.py` as a standalone script with a shared `_run_policy_step()` helper used by all three test categories. Results saved to `results/benchmark_<tag>.json`.

**Tech Stack:** Python, MuJoCo, PyTorch, NumPy, JSON, argparse

---

## File Structure

```
Modify:  applications/go2_locomotion/envs/go2_env.py
         - add apply_force() + _pending_force state

Create:  applications/go2_locomotion/benchmark.py
         - CLI entry point, all three test categories, JSON output

Create:  tests/go2/test_benchmark.py
         - unit tests for apply_force and each test category
```

---

## Task 1: Add `apply_force()` to Go2Env

**Files:**
- Modify: `applications/go2_locomotion/envs/go2_env.py`
- Test: `tests/go2/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/go2/test_benchmark.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np


def test_apply_force_changes_velocity():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config

    env = Go2Env(config)
    env.reset()

    # Walk for 50 steps to reach stable velocity
    action = np.zeros(12, dtype=np.float32)
    for _ in range(50):
        env.step(action)

    vel_before = env._get_base_linear_velocity().copy()

    # Apply large lateral force for 5 steps
    env.apply_force(np.array([0.0, 20.0, 0.0]), duration_steps=5)
    for _ in range(5):
        env.step(action)

    vel_after = env._get_base_linear_velocity().copy()

    # Lateral velocity should have changed noticeably
    assert abs(vel_after[1] - vel_before[1]) > 0.01, (
        f"Force had no effect: vy before={vel_before[1]:.4f}, after={vel_after[1]:.4f}"
    )
    env.close()


def test_apply_force_clears_after_duration():
    from applications.go2_locomotion.envs.go2_env import Go2Env
    from applications.go2_locomotion.config import config

    env = Go2Env(config)
    env.reset()

    env.apply_force(np.array([0.0, 10.0, 0.0]), duration_steps=3)

    # After 3 steps force should be cleared
    action = np.zeros(12, dtype=np.float32)
    for _ in range(3):
        env.step(action)

    # xfrc_applied on base body should be zero
    base_id = env._base_body_id
    force_remaining = np.linalg.norm(env.data.xfrc_applied[base_id, :3])
    assert force_remaining < 1e-6, f"Force not cleared after duration: {force_remaining}"
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/lihongl/Desktop/myRL/easyRL
python -m pytest tests/go2/test_benchmark.py -v
```

Expected: FAIL — `apply_force` not defined

- [ ] **Step 3: Add `apply_force()` to Go2Env**

In `applications/go2_locomotion/envs/go2_env.py`, add to `__init__` after `self._viewer = None`:

```python
        # External force injection (for benchmark perturbation tests)
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self._pending_force = np.zeros(3, dtype=np.float32)
        self._pending_force_steps = 0
```

Add `apply_force()` method after `_resample_command()`:

```python
    def apply_force(self, force_vec: np.ndarray, duration_steps: int):
        """Apply external force to base body for duration_steps control steps.

        force_vec: (3,) array in world frame [Fx, Fy, Fz] in Newtons.
        Called before step(); force is applied during the next duration_steps steps.
        """
        self._pending_force = np.array(force_vec, dtype=np.float32)
        self._pending_force_steps = int(duration_steps)
```

Modify `step()` — add force application at the **start** of the step, before the decimation loop. Replace the existing `step()` first line block:

```python
    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_angles = self.action_scale * action + self.default_angles

        # Apply pending external force (for perturbation tests)
        if self._pending_force_steps > 0:
            self.data.xfrc_applied[self._base_body_id, :3] = self._pending_force
            self._pending_force_steps -= 1
        else:
            self.data.xfrc_applied[self._base_body_id, :3] = 0.0

        for _ in range(self.decimation):
```

Also add to `reset()` after `self._last_joint_vel = np.zeros(12, dtype=np.float32)`:

```python
        self._pending_force = np.zeros(3, dtype=np.float32)
        self._pending_force_steps = 0
        if self._base_body_id >= 0:
            self.data.xfrc_applied[self._base_body_id, :3] = 0.0
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/go2/test_benchmark.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add applications/go2_locomotion/envs/go2_env.py tests/go2/test_benchmark.py
git commit -m "feat(go2): add apply_force() to Go2Env for perturbation tests"
```

---

## Task 2: Policy runner helper + Unit Tests category

**Files:**
- Create: `applications/go2_locomotion/benchmark.py` (initial skeleton + unit tests)
- Test: `tests/go2/test_benchmark.py` (add unit test category test)

- [ ] **Step 1: Write the failing test**

Add to `tests/go2/test_benchmark.py`:

```python
def test_run_unit_tests_returns_results():
    from applications.go2_locomotion.benchmark import run_unit_tests
    from applications.go2_locomotion.agent.ppo import PPOTrainer
    from applications.go2_locomotion.config import config

    # Random-init policy (no trained model needed)
    trainer = PPOTrainer(config)

    def policy_fn(obs):
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            mean, _ = trainer.network.forward_actor(obs_t)
        return mean.numpy().flatten()

    results = run_unit_tests(policy_fn, render=False)

    assert isinstance(results, dict)
    assert "forward_slow" in results
    assert "survived" in results["forward_slow"]
    assert "rmse_vx" in results["forward_slow"]
    assert "rmse_vy" in results["forward_slow"]
    assert "rmse_yaw" in results["forward_slow"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/go2/test_benchmark.py::test_run_unit_tests_returns_results -v
```

Expected: FAIL — `benchmark` module not found

- [ ] **Step 3: Create `benchmark.py` with helper + unit tests**

```python
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

PERTURBATION_SHOCK_FORCE = 8.0    # N, lateral
PERTURBATION_SHOCK_DURATION = 5   # steps (0.1s at 50Hz)
PERTURBATION_RECOVERY_STEPS = 250 # max steps to observe recovery (5s)
PERTURBATION_WALK_STEPS = 150     # steps of stable walking before shock (3s)
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
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/go2/test_benchmark.py::test_run_unit_tests_returns_results -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add applications/go2_locomotion/benchmark.py tests/go2/test_benchmark.py
git commit -m "feat(go2): add benchmark.py skeleton + unit tests category"
```

---

## Task 3: Sequence Test category

**Files:**
- Modify: `applications/go2_locomotion/benchmark.py`
- Test: `tests/go2/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/go2/test_benchmark.py`:

```python
def test_run_sequence_test_returns_results():
    from applications.go2_locomotion.benchmark import run_sequence_test, DEFAULT_SEQUENCE
    from applications.go2_locomotion.agent.ppo import PPOTrainer
    from applications.go2_locomotion.config import config

    trainer = PPOTrainer(config)

    def policy_fn(obs):
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            mean, _ = trainer.network.forward_actor(obs_t)
        return mean.numpy().flatten()

    # Use short sequence for speed
    short_seq = [
        ([1.0, 0.0, 0.0], 0.5, "forward"),
        ([0.0, 0.0, 0.0], 0.2, "brake"),
    ]
    results = run_sequence_test(policy_fn, sequence=short_seq, render=False)

    assert "survived" in results
    assert "avg_rmse" in results
    assert "per_segment" in results
    assert len(results["per_segment"]) == 2
    assert "rmse" in results["per_segment"][0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/go2/test_benchmark.py::test_run_sequence_test_returns_results -v
```

Expected: FAIL — `run_sequence_test` not defined

- [ ] **Step 3: Add `run_sequence_test` to `benchmark.py`**

Append to `benchmark.py`:

```python
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
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/go2/test_benchmark.py::test_run_sequence_test_returns_results -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add applications/go2_locomotion/benchmark.py tests/go2/test_benchmark.py
git commit -m "feat(go2): add sequence test category to benchmark"
```

---

## Task 4: Perturbation Tests category

**Files:**
- Modify: `applications/go2_locomotion/benchmark.py`
- Test: `tests/go2/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/go2/test_benchmark.py`:

```python
def test_run_perturbation_tests_returns_results():
    from applications.go2_locomotion.benchmark import run_perturbation_tests
    from applications.go2_locomotion.agent.ppo import PPOTrainer
    from applications.go2_locomotion.config import config

    trainer = PPOTrainer(config)

    def policy_fn(obs):
        import torch
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            mean, _ = trainer.network.forward_actor(obs_t)
        return mean.numpy().flatten()

    results = run_perturbation_tests(policy_fn, render=False)

    assert "single_shock" in results
    assert "max_force" in results
    assert "survived" in results["single_shock"]
    assert "recovery_steps" in results["single_shock"]
    assert "max_survived_force" in results["max_force"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/go2/test_benchmark.py::test_run_perturbation_tests_returns_results -v
```

Expected: FAIL — `run_perturbation_tests` not defined

- [ ] **Step 3: Add `run_perturbation_tests` to `benchmark.py`**

Append to `benchmark.py`:

```python
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
            # Check if velocity tracking has recovered (error < 0.2 m/s)
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
        for _ in range(150):  # observe for 3s
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
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/go2/test_benchmark.py::test_run_perturbation_tests_returns_results -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add applications/go2_locomotion/benchmark.py tests/go2/test_benchmark.py
git commit -m "feat(go2): add perturbation tests category to benchmark"
```

---

## Task 5: CLI entry point + JSON output

**Files:**
- Modify: `applications/go2_locomotion/benchmark.py`
- Test: `tests/go2/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/go2/test_benchmark.py`:

```python
def test_save_results_creates_json():
    from applications.go2_locomotion.benchmark import save_results
    import tempfile, os, json

    results = {
        "tag": "test",
        "timestamp": "2026-06-11T00:00:00",
        "model": "dummy.pth",
        "unit_tests": {"forward_slow": {"survived": True, "rmse_vx": 0.1,
                                         "rmse_vy": 0.0, "rmse_yaw": 0.0}},
        "sequence_test": {"survived": True, "avg_rmse": 0.1, "per_segment": []},
        "perturbation": {
            "single_shock": {"force_n": 8.0, "survived": True, "recovery_steps": 10},
            "max_force": {"max_survived_force": 8.0},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_results(results, results_dir=tmpdir)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["tag"] == "test"
        assert "unit_tests" in loaded
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/go2/test_benchmark.py::test_save_results_creates_json -v
```

Expected: FAIL

- [ ] **Step 3: Add `save_results`, `run_benchmark`, and `main` to `benchmark.py`**

Append to `benchmark.py`:

```python
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
```

- [ ] **Step 4: Run all benchmark tests**

```bash
python -m pytest tests/go2/test_benchmark.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Smoke test end-to-end**

```bash
cd /home/lihongl/Desktop/myRL/easyRL
python -c "
import sys; sys.path.insert(0, '.')
from applications.go2_locomotion.benchmark import run_benchmark, save_results
from applications.go2_locomotion.agent.ppo import PPOTrainer
from applications.go2_locomotion.config import config
import torch, tempfile

# Run with random-init model, unit only, short sequence
trainer = PPOTrainer(config)
device = trainer.device

def policy_fn(obs):
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        mean, _ = trainer.network.forward_actor(obs_t)
    return mean.cpu().numpy().flatten()

from applications.go2_locomotion.benchmark import run_unit_tests
results_unit = run_unit_tests(policy_fn, render=False)
assert 'forward_slow' in results_unit
print('Smoke test PASSED')
"
```

Expected: prints "Smoke test PASSED"

- [ ] **Step 6: Commit**

```bash
git add applications/go2_locomotion/benchmark.py tests/go2/test_benchmark.py
git commit -m "feat(go2): complete Motion Test Suite with CLI and JSON output"
```

---

## Task 6: Integration verification

**Files:**
- No new files

- [ ] **Step 1: Run full test suite**

```bash
cd /home/lihongl/Desktop/myRL/easyRL
python -m pytest tests/go2/ -v
```

Expected: all existing tests + 5 new benchmark tests PASS (17 total)

- [ ] **Step 2: Verify CLI help works**

```bash
python applications/go2_locomotion/benchmark.py --help
```

Expected: prints usage with `--tag`, `--model`, `--only`, `--sequence`, `--render`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(go2): add Go2 Motion Test Suite (benchmark.py)"
```
