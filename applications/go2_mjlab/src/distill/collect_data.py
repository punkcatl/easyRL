"""Collect (obs_history, teacher_action) pairs from trained teacher."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

import tyro
import torch
import numpy as np
from dataclasses import dataclass

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from mjlab.tasks.registry import get_task
from mjlab.rl import RslRlVecEnvWrapper


@dataclass
class CollectArgs:
    task_id: str = "Go2-Flat-v0"
    checkpoint: str = "applications/go2_mjlab/results/teacher_final.pt"
    num_envs: int = 1024
    num_transitions: int = 500_000
    history_length: int = 20
    output_path: str = "applications/go2_mjlab/results/distill_dataset.npz"


def main():
    args = tyro.cli(CollectArgs)

    task = get_task(args.task_id)
    env_cfg = task.env_cfg
    env_cfg.scene.num_envs = args.num_envs
    rl_cfg = task.rl_cfg

    env = task.env_cls(env_cfg)
    wrapped = RslRlVecEnvWrapper(env)

    runner = task.runner_cls(wrapped, rl_cfg, log_dir="/tmp/collect", device="cuda:0")
    runner.load(args.checkpoint)
    print(f"Loaded teacher: {args.checkpoint}")

    obs_dim = wrapped.observation_space.shape[0]
    history_length = args.history_length
    num_envs = args.num_envs

    history_buf = torch.zeros(num_envs, history_length, obs_dim, device="cuda:0")

    all_histories = []
    all_actions = []
    collected = 0
    steps_needed = args.num_transitions // num_envs + 1

    obs, _ = wrapped.reset()

    print(f"Collecting {args.num_transitions} transitions ({steps_needed} steps x {num_envs} envs)...")

    for step in range(steps_needed):
        history_buf = torch.roll(history_buf, -1, dims=1)
        history_buf[:, -1, :] = obs

        with torch.no_grad():
            actions = runner.get_inference_action(obs)

        if step >= history_length:
            all_histories.append(history_buf.cpu().numpy())
            all_actions.append(actions.cpu().numpy())
            collected += num_envs

        obs, _, rewards, dones, extras = wrapped.step(actions)

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            history_buf[done_ids] = 0.0

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{steps_needed} | collected: {collected:,}")

        if collected >= args.num_transitions:
            break

    env.close()

    histories = np.concatenate(all_histories, axis=0)[:args.num_transitions]
    actions = np.concatenate(all_actions, axis=0)[:args.num_transitions]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), obs_history=histories, actions=actions)
    print(f"Saved {histories.shape[0]} transitions to {output_path}")
    print(f"  obs_history shape: {histories.shape}")
    print(f"  actions shape: {actions.shape}")


if __name__ == "__main__":
    main()
