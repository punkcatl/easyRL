"""Visualize a trained Go2 policy with MuJoCo viewer."""
import sys
from pathlib import Path
from dataclasses import asdict

import tyro
from dataclasses import dataclass

import src.tasks  # noqa: F401 — registers Go2-Flat-v0

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner


@dataclass
class PlayArgs:
    task_id: str = "Go2-Flat-v0"
    checkpoint: str = "results_r9/model_1000.pt"
    num_steps: int = 5000


def main():
    args = tyro.cli(PlayArgs)

    env_cfg = load_env_cfg(args.task_id)
    env_cfg.scene.num_envs = 1
    rl_cfg = load_rl_cfg(args.task_id)

    env = ManagerBasedRlEnv(env_cfg, device="cuda:0", render_mode="human")
    wrapped = RslRlVecEnvWrapper(env)

    runner_cls = load_runner_cls(args.task_id)
    runner = runner_cls(wrapped, asdict(rl_cfg), log_dir="/tmp/play", device="cuda:0")
    runner.load(args.checkpoint)
    print(f"Loaded: {args.checkpoint}")

    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = wrapped.reset()
    for step in range(args.num_steps):
        actions = policy(obs)
        result = wrapped.step(actions)
        obs = result[0]
        env.render()

    env.close()


if __name__ == "__main__":
    main()
