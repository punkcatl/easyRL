"""Evaluate G1 locomotion policy — ground truth quality assessment.

Unlike training metrics (2048 env averages), this script measures
what actually matters: can the robot walk continuously without falling?

Usage:
    python scripts/evaluate.py --checkpoint results_r4/model_1499.pt
    python scripts/evaluate.py --checkpoint results_r4/model_1499.pt --record
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
from dataclasses import asdict

import torch
import numpy as np

import src.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.terrains import TerrainEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.sim import SimulationCfg, MujocoCfg


# ── Pass criteria ────────────────────────────────────────────────────────────
PASS_SURVIVAL_S      = 20.0   # must survive at least 20 seconds
PASS_SPEED_RATIO     = 0.75   # actual / commanded >= 75%
PASS_MAX_FALLS       = 2      # at most 2 falls in 60 seconds


def make_eval_env(cmd_vx: float, with_dr: bool = False):
    """Create flat-terrain single-env for evaluation."""
    env_cfg = load_env_cfg("G1-Flat-v0")
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
    env_cfg.sim = SimulationCfg(
        njmax=200, nconmax=100,
        mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    )

    # Fixed command
    env_cfg.commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            heading_command=False,
            rel_standing_envs=0.0,
            resampling_time_range=(9999.0, 9999.0),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(cmd_vx, cmd_vx),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        ),
    }

    # Remove push_robot and DR unless requested
    if not with_dr:
        env_cfg.events = {
            k: v for k, v in env_cfg.events.items() if "reset" in k
        }

    return env_cfg


def run_episode(env, wrapped, policy, duration_s: float, step_dt: float):
    """Run one episode, collect metrics. No reset — let it fall."""
    obs, _ = wrapped.reset()

    steps = int(duration_s / step_dt)
    speeds = []
    falls = 0
    first_fall_s = None
    fall_times = []

    asset = env.scene["robot"]

    for step in range(steps):
        t = step * step_dt

        with torch.no_grad():
            actions = policy(obs)
        result = wrapped.step(actions)
        obs = result[0]

        # Actual forward speed
        vx = asset.data.root_link_lin_vel_b[0, 0].item()
        speeds.append(vx)

        # Detect fall (orientation > 60 deg)
        gravity_b = asset.data.projected_gravity_b[0]
        tilt = (gravity_b[0]**2 + gravity_b[1]**2).sqrt().item()
        fell = tilt > 0.85  # ~60 deg

        if fell:
            falls += 1
            fall_times.append(t)
            if first_fall_s is None:
                first_fall_s = t

    return {
        "survival_s": first_fall_s if first_fall_s is not None else duration_s,
        "avg_speed": float(np.mean(speeds)),
        "fall_count": falls,
        "fall_times": fall_times,
    }


def evaluate(checkpoint: str, record: bool = False, duration_s: float = 60.0):
    print()
    print("=" * 60)
    print(f"G1 Policy Evaluation")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Duration:   {duration_s}s per test")
    print("=" * 60)

    results = {}

    for cmd_vx in [0.5, 1.0, 1.5]:
        print(f"\n[Test] Fixed command: {cmd_vx:.1f} m/s straight")

        env_cfg = make_eval_env(cmd_vx, with_dr=False)
        env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
        wrapped = RslRlVecEnvWrapper(env)

        rl_cfg = load_rl_cfg("G1-Flat-v0")
        runner = MjlabOnPolicyRunner(
            wrapped, asdict(rl_cfg), log_dir="/tmp/eval", device="cuda:0"
        )
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device="cuda:0")

        step_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
        metrics = run_episode(env, wrapped, policy, duration_s, step_dt)

        speed_ratio = metrics["avg_speed"] / cmd_vx
        passed = (
            metrics["survival_s"] >= PASS_SURVIVAL_S
            and speed_ratio >= PASS_SPEED_RATIO
            and metrics["fall_count"] <= PASS_MAX_FALLS
        )

        print(f"  Survival:    {metrics['survival_s']:.1f}s  (pass >= {PASS_SURVIVAL_S}s)  {'✓' if metrics['survival_s'] >= PASS_SURVIVAL_S else '✗'}")
        print(f"  Avg speed:   {metrics['avg_speed']:.2f} m/s  ({speed_ratio*100:.0f}% of cmd)  {'✓' if speed_ratio >= PASS_SPEED_RATIO else '✗'}")
        print(f"  Falls:       {metrics['fall_count']}  (pass <= {PASS_MAX_FALLS})  {'✓' if metrics['fall_count'] <= PASS_MAX_FALLS else '✗'}")
        if metrics["fall_times"]:
            print(f"  Fall times:  {[f'{t:.1f}s' for t in metrics['fall_times'][:5]]}")
        print(f"  → {'PASS ✓' if passed else 'FAIL ✗'}")

        results[cmd_vx] = {**metrics, "speed_ratio": speed_ratio, "passed": passed}

        env.close()

        # Record video if requested (only for 1.0 m/s)
        if record and cmd_vx == 1.0:
            _record_video(checkpoint, cmd_vx, duration_s=20.0)

    # Overall verdict
    all_passed = all(r["passed"] for r in results.values())
    print()
    print("=" * 60)
    print(f"OVERALL: {'PASS ✓' if all_passed else 'FAIL ✗'}")
    if not all_passed:
        failed = [f"{v} m/s" for v, r in results.items() if not r["passed"]]
        print(f"  Failed tests: {', '.join(failed)}")
    print("=" * 60)
    print()

    return results


def _record_video(checkpoint: str, cmd_vx: float, duration_s: float):
    """Record a short visualization video."""
    import subprocess, os, mujoco, mujoco.viewer, time as time_mod

    print(f"\n[Record] Recording {duration_s}s video...")

    env_cfg = make_eval_env(cmd_vx, with_dr=False)
    env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)

    rl_cfg = load_rl_cfg("G1-Flat-v0")
    runner = MjlabOnPolicyRunner(
        wrapped, asdict(rl_cfg), log_dir="/tmp/eval", device="cuda:0"
    )
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    mj_model = env.sim.mj_model
    mj_data = env.sim.mj_data
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # Start ffmpeg recording
    ckpt_name = Path(checkpoint).stem
    out_path = Path(checkpoint).parent / f"eval_{ckpt_name}.mp4"
    ffmpeg_cmd = (
        f"DISPLAY=:1 ffmpeg -y -f x11grab -r 30 "
        f"-video_size 1280x720 -i :1+0,0 "
        f"-t {duration_s} -vcodec libx264 -pix_fmt yuv420p {out_path}"
    )
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, shell=True)

    obs, _ = wrapped.reset()
    steps = int(duration_s / (env_cfg.decimation * env_cfg.sim.mujoco.timestep))
    for _ in range(steps):
        with torch.no_grad():
            actions = policy(obs)
        result = wrapped.step(actions)
        obs = result[0]
        mj_data.qpos[:] = env.sim.wp_data.qpos.numpy()[0]
        mj_data.qvel[:] = env.sim.wp_data.qvel.numpy()[0]
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()
        time_mod.sleep(0.02)

    ffmpeg_proc.wait()
    viewer.close()
    env.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    evaluate(args.checkpoint, record=args.record, duration_s=args.duration)
