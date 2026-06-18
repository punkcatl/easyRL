"""Sanity check: create Go2 env, run 100 steps, print obs shapes."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent))

import torch
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from mjlab.tasks.registry import get_task
from mjlab.rl import RslRlVecEnvWrapper


def main():
    task = get_task("Go2-Flat-v0")
    env_cfg = task.env_cfg
    env_cfg.scene.num_envs = 64

    env = task.env_cls(env_cfg)
    wrapped = RslRlVecEnvWrapper(env)

    print(f"Env created: {wrapped.num_envs} envs")
    print(f"Obs shape: {wrapped.observation_space.shape}")
    print(f"Critic obs shape: {wrapped.privileged_observation_space.shape}")
    print(f"Action shape: {wrapped.action_space.shape}")

    obs, extras = wrapped.reset()
    print(f"Reset obs: {obs.shape}, device: {obs.device}")

    for i in range(100):
        actions = torch.zeros(wrapped.num_envs, wrapped.num_actions, device=wrapped.device)
        obs, _, rewards, dones, extras = wrapped.step(actions)

    print(f"100 steps complete. Final obs: {obs.shape}")
    print(f"Reward sample: {rewards[:5]}")
    print("VALIDATION PASSED")

    env.close()


if __name__ == "__main__":
    main()
