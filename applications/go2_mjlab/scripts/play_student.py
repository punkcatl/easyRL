"""Visualize Student policy with MuJoCo native viewer."""
from dataclasses import asdict, dataclass

import torch
import tyro

import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer
from src.distill.student_network import StudentPolicy


class StudentPolicyWrapper:
    """Wraps StudentPolicy to match mjlab's PolicyProtocol (accepts TensorDict obs)."""

    def __init__(self, model, obs_dim, history_length, device):
        self.model = model
        self.history_length = history_length
        self.obs_dim = obs_dim
        self.device = device
        self.history = None

    def reset(self, num_envs):
        self.history = torch.zeros(num_envs, self.history_length, self.obs_dim, device=self.device)

    def __call__(self, obs_td):
        obs = obs_td["actor"]
        if self.history is None:
            self.reset(obs.shape[0])
        self.history = torch.roll(self.history, -1, dims=1)
        self.history[:, -1, :] = obs
        with torch.no_grad():
            return self.model(self.history)


@dataclass
class Args:
    checkpoint: str = "results/student_final.pt"
    obs_dim: int = 50
    action_dim: int = 12
    history_length: int = 20
    latent_dim: int = 16


def main():
    args = tyro.cli(Args)

    model = StudentPolicy(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        history_length=args.history_length,
        latent_dim=args.latent_dim,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cuda:0", weights_only=True))
    model.eval().cuda()
    print(f"Student loaded: {sum(p.numel() for p in model.parameters()):,} params")

    env_cfg = load_env_cfg("Go2-Flat-v0")
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(env_cfg, device="cuda:0")

    policy = StudentPolicyWrapper(model, args.obs_dim, args.history_length, "cuda:0")

    viewer = NativeMujocoViewer(env, policy)
    print("Viewer opened. Close window to exit.")
    viewer.run()


if __name__ == "__main__":
    main()
