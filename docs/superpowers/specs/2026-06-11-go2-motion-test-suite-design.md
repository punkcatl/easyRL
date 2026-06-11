# Go2 Motion Test Suite Design

## Overview

A structured verification suite for trained Go2 locomotion policies, modeled after industrial motion test suites used by companies like Unitree and ANYbotics. Runs after each training iteration to verify policy quality across three test categories. Results are printed to terminal and saved as versioned JSON files for cross-version comparison.

## Goals

- Verify trained policy behavior across specific motion capabilities
- Provide quantitative metrics (RMSE, survival rate, recovery time) per test
- Support cross-version comparison via tagged result files
- Mirror industrial verification practice (regression test after each model iteration)

---

## Architecture

New standalone file `applications/go2_locomotion/benchmark.py`. No modifications to `evaluate.py` (existing Sim2Sim evaluation stays intact).

```
applications/go2_locomotion/
├── benchmark.py          # new: Motion Test Suite entry point
├── envs/
│   └── go2_env.py        # add: apply_force() method
├── results/
│   ├── benchmark_v1.json
│   └── benchmark_v2.json
```

---

## Test Categories

### Category 1: Unit Tests (8 items)

Each test runs a fixed command for 5 seconds (250 steps at 50Hz). Metrics: velocity tracking RMSE per axis, survival (did not fall).

| Test Name | Command [vx, vy, yaw] | Description |
|-----------|----------------------|-------------|
| forward_slow | [0.5, 0.0, 0.0] | slow straight walk |
| forward_fast | [1.0, 0.0, 0.0] | fast straight walk |
| reverse | [-0.5, 0.0, 0.0] | backward walk |
| lateral_left | [0.0, 0.5, 0.0] | left sidestep |
| lateral_right | [0.0, -0.5, 0.0] | right sidestep |
| rotate_left | [0.0, 0.0, 1.0] | left in-place rotation |
| rotate_right | [0.0, 0.0, -1.0] | right in-place rotation |
| combined | [0.8, 0.3, 0.5] | diagonal + rotation |

Metrics per test:
- `rmse_vx`, `rmse_vy`, `rmse_yaw`: velocity tracking error per axis
- `survived`: bool (completed 250 steps without falling)

### Category 2: Sequence Test (configurable)

Runs a sequence of `(command, duration_s)` segments without resetting the environment between segments. Tests motion continuity and command-switching stability.

Default sequence (11 seconds total):
```python
DEFAULT_SEQUENCE = [
    ([1.0,  0.0,  0.0], 3.0),   # forward 1m/s
    ([0.5,  0.0,  1.0], 2.0),   # forward + left turn
    ([1.0,  0.0,  0.0], 2.0),   # forward 1m/s
    ([0.0,  0.5,  0.0], 1.0),   # lateral step
    ([-0.5, 0.0,  0.0], 2.0),   # reverse
    ([0.0,  0.0,  0.0], 1.0),   # brake / stop
]
```

User can override via `--sequence path/to/sequence.json`.

Metrics:
- `survived`: bool (no fall during entire sequence)
- `avg_rmse`: mean velocity tracking RMSE across all segments
- `per_segment_rmse`: list of RMSE per segment

### Category 3: Perturbation Tests (2 items)

Requires `apply_force(force_vec, duration_steps)` on `Go2Env`.

**3a. Single Shock Test**
- Robot walks at `[1.0, 0, 0]` for 3 seconds (stable state)
- Apply lateral impulse `[0, 8, 0] N` for 0.1s (5 steps)
- Continue for up to 5 more seconds
- Metrics: `survived` (bool), `recovery_steps` (steps until velocity error < 0.2 m/s again), or -1 if fell

**3b. Max Force Test (escalating)**
- For force magnitudes `[2, 4, 6, 8, 10, 12, 15, 20] N`:
  - Reset env, walk 3 seconds, apply force for 0.1s, observe 3 seconds
  - Stop when robot falls
- Metric: `max_survived_force` (N) — the highest force the policy survived

---

## External Force API

Add to `Go2Env`:

```python
def apply_force(self, force_vec: np.ndarray, duration_steps: int):
    """Apply external force to base body for duration_steps control steps.
    force_vec: (3,) array in world frame [Fx, Fy, Fz] Newtons
    """
    base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    self._pending_force = force_vec.astype(np.float32)
    self._pending_force_steps = duration_steps
```

Force is applied inside `step()` by writing to `data.xfrc_applied[base_id, :3]` and decrementing `_pending_force_steps` each step. Cleared to zero when counter reaches 0.

---

## CLI Interface

```bash
# Run full benchmark suite
python applications/go2_locomotion/benchmark.py --tag v1

# Run full suite with rendering
python applications/go2_locomotion/benchmark.py --tag v1 --render

# Specific model path
python applications/go2_locomotion/benchmark.py --tag v1 --model results/teacher_iter500.pth

# Custom sequence
python applications/go2_locomotion/benchmark.py --tag v1 --sequence my_sequence.json

# Run only specific categories
python applications/go2_locomotion/benchmark.py --tag v1 --only unit
python applications/go2_locomotion/benchmark.py --tag v1 --only sequence
python applications/go2_locomotion/benchmark.py --tag v1 --only perturbation
```

Default model: `results/teacher_final.pth`. Default tag: timestamp `YYYYMMDD_HHMMSS`.

---

## Output Format

### Terminal

```
================================================
  Go2 Motion Test Suite  [tag: v1]
  Model: results/teacher_final.pth
================================================

[1/3] Unit Tests
  forward_slow    RMSE vx=0.08  vy=0.02  yaw=0.01  survived=YES
  forward_fast    RMSE vx=0.15  vy=0.03  yaw=0.02  survived=YES
  reverse         RMSE vx=0.12  vy=0.02  yaw=0.01  survived=YES
  lateral_left    RMSE vx=0.03  vy=0.11  yaw=0.02  survived=YES
  lateral_right   RMSE vx=0.03  vy=0.10  yaw=0.02  survived=YES
  rotate_left     RMSE vx=0.02  vy=0.02  yaw=0.18  survived=YES
  rotate_right    RMSE vx=0.02  vy=0.02  yaw=0.17  survived=YES
  combined        RMSE vx=0.14  vy=0.09  yaw=0.15  survived=YES
  Unit pass rate: 8/8

[2/3] Sequence Test
  Segment 0 (forward 3s)       RMSE=0.09  survived=YES
  Segment 1 (forward+turn 2s)  RMSE=0.16  survived=YES
  Segment 2 (forward 2s)       RMSE=0.11  survived=YES
  Segment 3 (lateral 1s)       RMSE=0.13  survived=YES
  Segment 4 (reverse 2s)       RMSE=0.14  survived=YES
  Segment 5 (brake 1s)         RMSE=0.05  survived=YES
  Overall: survived=YES  avg_rmse=0.11

[3/3] Perturbation Tests
  Single shock (8N):  survived=YES  recovery_steps=18
  Max force test:     max_survived_force=12N

================================================
  SUMMARY
  Unit tests:     8/8 passed
  Sequence test:  PASS  avg_rmse=0.11
  Max force:      12 N
================================================
Saved to results/benchmark_v1.json
```

### JSON

```json
{
  "tag": "v1",
  "timestamp": "2026-06-11T14:30:22",
  "model": "results/teacher_final.pth",
  "unit_tests": {
    "forward_slow":  {"rmse_vx": 0.08, "rmse_vy": 0.02, "rmse_yaw": 0.01, "survived": true},
    "forward_fast":  {"rmse_vx": 0.15, "rmse_vy": 0.03, "rmse_yaw": 0.02, "survived": true},
    "reverse":       {"rmse_vx": 0.12, "rmse_vy": 0.02, "rmse_yaw": 0.01, "survived": true},
    "lateral_left":  {"rmse_vx": 0.03, "rmse_vy": 0.11, "rmse_yaw": 0.02, "survived": true},
    "lateral_right": {"rmse_vx": 0.03, "rmse_vy": 0.10, "rmse_yaw": 0.02, "survived": true},
    "rotate_left":   {"rmse_vx": 0.02, "rmse_vy": 0.02, "rmse_yaw": 0.18, "survived": true},
    "rotate_right":  {"rmse_vx": 0.02, "rmse_vy": 0.02, "rmse_yaw": 0.17, "survived": true},
    "combined":      {"rmse_vx": 0.14, "rmse_vy": 0.09, "rmse_yaw": 0.15, "survived": true}
  },
  "sequence_test": {
    "survived": true,
    "avg_rmse": 0.11,
    "per_segment": [
      {"name": "forward",      "rmse": 0.09, "survived": true},
      {"name": "forward_turn", "rmse": 0.16, "survived": true},
      {"name": "forward2",     "rmse": 0.11, "survived": true},
      {"name": "lateral",      "rmse": 0.13, "survived": true},
      {"name": "reverse",      "rmse": 0.14, "survived": true},
      {"name": "brake",        "rmse": 0.05, "survived": true}
    ]
  },
  "perturbation": {
    "single_shock": {"force_n": 8.0, "survived": true, "recovery_steps": 18},
    "max_force":    {"max_survived_force": 12.0}
  }
}
```

---

## Files to Create / Modify

| File | Action | Description |
|------|--------|-------------|
| `applications/go2_locomotion/benchmark.py` | Create | Motion Test Suite main script |
| `applications/go2_locomotion/envs/go2_env.py` | Modify | Add `apply_force()` method + `_pending_force` state |
