"""Evaluate G1 student policy — same criteria as teacher evaluate.py.

Usage:
    python scripts/evaluate_student.py --checkpoint results/student_final.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import numpy as np

import src.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.terrains import TerrainEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from src.distill.student_network import StudentPolicy


PASS_SURVIVAL_S = 20.0
PASS_SPEED_RATIO = 0.75
PASS_MAX_FALLS = 2


def make_eval_env(cmd_vx: float):
    env_cfg = load_env_cfg("G1-Flat-v0")
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
    env_cfg.sim = SimulationCfg(
        njmax=200, nconmax=100,
        mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    )
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
    env_cfg.events = {k: v for k, v in env_cfg.events.items() if "reset" in k or "arm" in k}
    return env_cfg


def run_episode(env, wrapped, model, history_length, obs_dim, duration_s, step_dt):
    obs, _ = wrapped.reset()
    history_buf = torch.zeros(1, history_length, obs_dim, device="cuda:0")

    steps = int(duration_s / step_dt)
    speeds = []
    falls = 0
    first_fall_s = None

    for step in range(steps):
        t = step * step_dt
        obs_flat = obs["actor"]
        history_buf = torch.roll(history_buf, -1, dims=1)
        history_buf[:, -1, :] = obs_flat

        with torch.no_grad():
            actions = model(history_buf)
        result = wrapped.step(actions)
        obs = result[0]

        vx = env.scene["robot"].data.root_link_lin_vel_b[0, 0].item()
        speeds.append(vx)

        gravity_b = env.scene["robot"].data.projected_gravity_b[0]
        tilt = (gravity_b[0]**2 + gravity_b[1]**2).sqrt().item()
        if tilt > 0.85:
            falls += 1
            if first_fall_s is None:
                first_fall_s = t

    return {
        "survival_s": first_fall_s if first_fall_s is not None else duration_s,
        "avg_speed": float(np.mean(speeds)),
        "fall_count": falls,
    }


def evaluate(checkpoint: str, obs_dim: int, action_dim: int,
             history_length: int, latent_dim: int, duration_s: float):
    model = StudentPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        history_length=history_length,
        latent_dim=latent_dim,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cuda:0", weights_only=True))
    model.eval().cuda()
    params = sum(p.numel() for p in model.parameters())

    print()
    print("=" * 60)
    print(f"G1 Student Policy Evaluation")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Params:     {params:,}")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, history={history_length}")
    print("=" * 60)

    results = {}
    for cmd_vx in [0.5, 1.0]:
        print(f"\n[Test] Fixed command: {cmd_vx:.1f} m/s straight")

        env_cfg = make_eval_env(cmd_vx)
        env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
        wrapped = RslRlVecEnvWrapper(env)

        step_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
        metrics = run_episode(env, wrapped, model, history_length, obs_dim, duration_s, step_dt)

        speed_ratio = metrics["avg_speed"] / cmd_vx
        passed = (
            metrics["survival_s"] >= PASS_SURVIVAL_S
            and speed_ratio >= PASS_SPEED_RATIO
            and metrics["fall_count"] <= PASS_MAX_FALLS
        )

        print(f"  Survival:    {metrics['survival_s']:.1f}s  {'✓' if metrics['survival_s'] >= PASS_SURVIVAL_S else '✗'}")
        print(f"  Avg speed:   {metrics['avg_speed']:.2f} m/s  ({speed_ratio*100:.0f}% of cmd)  {'✓' if speed_ratio >= PASS_SPEED_RATIO else '✗'}")
        print(f"  Falls:       {metrics['fall_count']}  {'✓' if metrics['fall_count'] <= PASS_MAX_FALLS else '✗'}")
        print(f"  → {'PASS ✓' if passed else 'FAIL ✗'}")

        results[cmd_vx] = {**metrics, "speed_ratio": speed_ratio, "passed": passed}
        env.close()

    all_passed = all(r["passed"] for r in results.values())
    print()
    print("=" * 60)
    print(f"OVERALL: {'PASS ✓' if all_passed else 'FAIL ✗'}")
    print("=" * 60)
    print()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/student_final.pt")
    parser.add_argument("--obs-dim", type=int, default=84)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.obs_dim, args.action_dim,
             args.history_length, args.latent_dim, args.duration)
