"""Collect (obs_history, teacher_action) pairs from trained teacher."""
from pathlib import Path
from dataclasses import asdict, dataclass

import tyro
import torch
import numpy as np

import src.tasks  # noqa: F401

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner


@dataclass
class CollectArgs:
    task_id: str = "Go2-Flat-v0"
    checkpoint: str = "results_r14/model_4999.pt"
    num_envs: int = 1024
    num_transitions: int = 500_000
    history_length: int = 20
    output_path: str = "results/distill_dataset.npz"


def main():
    args = tyro.cli(CollectArgs)

    env_cfg = load_env_cfg(args.task_id)
    env_cfg.scene.num_envs = args.num_envs
    rl_cfg = load_rl_cfg(args.task_id)

    env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)

    runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), log_dir="/tmp/collect", device="cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")
    print(f"Loaded teacher: {args.checkpoint}")

    obs_dim = wrapped.observation_space.spaces["actor"].shape[-1]
    history_length = args.history_length
    num_envs = args.num_envs

    history_buf = torch.zeros(num_envs, history_length, obs_dim, device="cuda:0")

    all_histories = []
    all_actions = []
    collected = 0
    steps_needed = args.num_transitions // num_envs + 1

    obs_td, _ = wrapped.reset()
    obs_flat = obs_td["actor"]  # [B, 50]

    print(f"Collecting {args.num_transitions} transitions ({steps_needed} steps x {num_envs} envs)...")

    for step in range(steps_needed):
        history_buf = torch.roll(history_buf, -1, dims=1)
        history_buf[:, -1, :] = obs_flat

        with torch.no_grad():
            actions = policy(obs_td)

        if step >= history_length:
            all_histories.append(history_buf.cpu().numpy())
            all_actions.append(actions.cpu().numpy())
            collected += num_envs

        result = wrapped.step(actions)
        obs_td = result[0]
        obs_flat = obs_td["actor"]

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{steps_needed} | collected: {collected:,}")

        if collected >= args.num_transitions:
            break

    env.close()

    histories = np.concatenate(all_histories, axis=0)[:args.num_transitions]
    actions_np = np.concatenate(all_actions, axis=0)[:args.num_transitions]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), obs_history=histories, actions=actions_np)
    print(f"Saved {histories.shape[0]} transitions to {output_path}")
    print(f"  obs_history shape: {histories.shape}")
    print(f"  actions shape: {actions_np.shape}")


if __name__ == "__main__":
    main()
