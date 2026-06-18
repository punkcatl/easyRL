"""Train Go2 Teacher policy with PPO on mjlab."""
from pathlib import Path
from dataclasses import asdict

import torch

import src.tasks  # noqa: F401 — registers Go2-Flat-v0
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="Go2-Flat-v0")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--log-dir", default="results")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    env_cfg = load_env_cfg(args.task_id)
    env_cfg.scene.num_envs = args.num_envs

    rl_cfg = load_rl_cfg(args.task_id)
    rl_cfg.max_iterations = args.max_iterations

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training Go2 Teacher: {args.num_envs} envs, {args.max_iterations} iters")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Log dir: {log_dir}")
    print()

    env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)

    runner_cls = load_runner_cls(args.task_id)
    runner = runner_cls(wrapped, asdict(rl_cfg), log_dir=str(log_dir), device="cuda:0")

    if args.resume:
        runner.load(args.resume)
        print(f"Resumed from {args.resume}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    env.close()
    print("Teacher training complete.")


if __name__ == "__main__":
    main()
