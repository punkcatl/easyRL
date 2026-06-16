# G1 Locomotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete RL locomotion pipeline for Unitree G1 humanoid robot — from Isaac Lab teacher training to custom Teacher-Student RMA distillation and ONNX deployment.

**Architecture:** Manager-Based env config inheriting Isaac Lab's G1FlatEnvCfg/G1RoughEnvCfg with custom reward overrides. Phase 1 uses rsl_rl for PPO training. Phase 2/3 are custom PyTorch code for Teacher-Student distillation and ONNX export, symmetric with the Go2 project pattern.

**Tech Stack:** Isaac Lab 2.3.0, Isaac Sim 5.1.0, rsl_rl 3.0.1, PyTorch, ONNX Runtime, TensorBoard

---

## File Structure

```
applications/g1_locomotion/
├── __init__.py                      # package marker + gym.register custom envs
├── config/
│   ├── __init__.py                  # exports all configs
│   ├── flat_env_cfg.py              # G1FlatLocomotionEnvCfg (custom rewards/cmds)
│   ├── rough_env_cfg.py             # G1RoughLocomotionEnvCfg (terrain curriculum)
│   └── ppo_cfg.py                   # rsl_rl PPO runner configs
├── mdp/
│   ├── __init__.py                  # exports custom reward functions
│   └── rewards.py                   # custom reward terms not in Isaac Lab
├── scripts/
│   ├── train_teacher.py             # Phase 1 training entry (calls rsl_rl runner)
│   ├── play.py                      # visualization / evaluation
│   └── collect_teacher_data.py      # Phase 2 data collection from frozen teacher
├── student/
│   ├── __init__.py
│   ├── networks.py                  # AdaptationModule + StudentPolicy
│   ├── train_student.py             # Phase 2 BC training
│   └── evaluate.py                  # Sim2Sim validation
├── export/
│   ├── __init__.py
│   ├── export_onnx.py              # Phase 3 ONNX export
│   └── benchmark.py                # inference latency test
├── training_log.md                  # per-round iteration diary
└── project_design.md               # design doc (copy from specs)
```

---

## Task 1: Project Scaffold + Package Registration

**Files:**
- Create: `applications/g1_locomotion/__init__.py`
- Create: `applications/g1_locomotion/config/__init__.py`
- Create: `applications/g1_locomotion/mdp/__init__.py`
- Create: `applications/g1_locomotion/scripts/` (directory)
- Create: `applications/g1_locomotion/student/__init__.py`
- Create: `applications/g1_locomotion/export/__init__.py`
- Create: `applications/g1_locomotion/training_log.md`
- Copy: `docs/superpowers/specs/2026-06-15-g1-locomotion-design.md` → `applications/g1_locomotion/project_design.md`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p applications/g1_locomotion/{config,mdp,scripts,student,export,results}
```

- [ ] **Step 2: Create package __init__.py files**

`applications/g1_locomotion/__init__.py`:
```python
"""G1 Humanoid Locomotion — Isaac Lab + rsl_rl training pipeline."""
```

`applications/g1_locomotion/config/__init__.py`:
```python
from .flat_env_cfg import G1FlatLocomotionEnvCfg, G1FlatLocomotionEnvCfg_PLAY
from .rough_env_cfg import G1RoughLocomotionEnvCfg, G1RoughLocomotionEnvCfg_PLAY
from .ppo_cfg import G1FlatPPOCfg, G1RoughPPOCfg
```

`applications/g1_locomotion/mdp/__init__.py`:
```python
from .rewards import *  # noqa: F401, F403
```

`applications/g1_locomotion/student/__init__.py`:
```python
from .networks import AdaptationModule, StudentPolicy, StudentONNXWrapper
```

`applications/g1_locomotion/export/__init__.py`:
```python
"""G1 ONNX export and benchmark utilities."""
```

- [ ] **Step 3: Create training_log.md skeleton**

`applications/g1_locomotion/training_log.md`:
```markdown
# G1 Locomotion Training Log

Per-round training results, diagnostics, and iteration strategy.

---

## Round 1

**Config:** (to be filled when training starts)

**TensorBoard:** `results/<run_dir>/`

**Results:** (pending)
```

- [ ] **Step 4: Copy design doc**

```bash
cp docs/superpowers/specs/2026-06-15-g1-locomotion-design.md applications/g1_locomotion/project_design.md
```

- [ ] **Step 5: Add results/ to .gitignore**

Append to the project root `.gitignore`:
```
applications/g1_locomotion/results/
```

- [ ] **Step 6: Commit**

```bash
git add applications/g1_locomotion/ .gitignore
git commit -m "feat(g1): scaffold project structure for G1 locomotion"
```

---

## Task 2: Custom Reward Terms (mdp/rewards.py)

**Files:**
- Create: `applications/g1_locomotion/mdp/rewards.py`

Isaac Lab provides most needed reward terms, but we add a few custom ones for gait quality control that the default library doesn't have.

- [ ] **Step 1: Write custom reward functions**

`applications/g1_locomotion/mdp/rewards.py`:
```python
"""Custom reward terms for G1 humanoid locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bipedal_gait_symmetry(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward symmetric alternating contact between left and right feet.

    Encourages anti-phase gait: when left foot is in contact, right should be in air,
    and vice versa. Returns 1.0 for perfect alternation, 0.0 for same-phase.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]
    # sensor_cfg.body_ids should be [left_ankle, right_ankle]
    left_contact = torch.norm(net_forces[:, 0, :], dim=-1) > 1.0
    right_contact = torch.norm(net_forces[:, 1, :], dim=-1) > 1.0
    # XOR: reward when exactly one foot is in contact
    symmetry = (left_contact ^ right_contact).float()
    return symmetry


def base_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining target base height using exp kernel."""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    error = (base_height - target_height) ** 2
    return torch.exp(-error / (sigma**2))


def forward_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Raw forward velocity reward — faster is always better."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0]
```

- [ ] **Step 2: Verify import works (mental check)**

The functions follow Isaac Lab's reward term signature: `(env, **params) -> Tensor[num_envs]`. They will be referenced in the env config via `RewTerm(func=...)`.

- [ ] **Step 3: Commit**

```bash
git add applications/g1_locomotion/mdp/
git commit -m "feat(g1): add custom reward terms for gait symmetry and height"
```

---

## Task 3: Flat Environment Config (config/flat_env_cfg.py)

**Files:**
- Create: `applications/g1_locomotion/config/flat_env_cfg.py`

- [ ] **Step 1: Write flat env config**

`applications/g1_locomotion/config/flat_env_cfg.py`:
```python
"""G1 flat-terrain locomotion environment configuration."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)

from applications.g1_locomotion.mdp import rewards as custom_mdp


@configclass
class G1FlatLocomotionEnvCfg(G1FlatEnvCfg):
    """Custom G1 flat env with tuned rewards and commands."""

    def __post_init__(self):
        super().__post_init__()

        # --- Scene: reduce envs for A4000 (16GB) ---
        self.scene.num_envs = 1024

        # --- Commands: start conservative, curriculum expands ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # --- Rewards: custom tuning ---
        # Velocity tracking (primary objective)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.4
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Gait quality
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_slide.weight = -0.1

        # Stability
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.2

        # Smoothness
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_torques_l2.weight = -2.0e-6

        # Joint deviation (keep arms/torso at default)
        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_hip.weight = -0.1
        self.rewards.joint_deviation_torso.weight = -0.1

        # Limits and termination
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.termination_penalty.weight = -200.0

        # Custom rewards
        self.rewards.gait_symmetry = RewTerm(
            func=custom_mdp.bipedal_gait_symmetry,
            weight=0.3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.base_height = RewTerm(
            func=custom_mdp.base_height_reward,
            weight=0.3,
            params={"target_height": 0.74, "sigma": 0.05},
        )


@configclass
class G1FlatLocomotionEnvCfg_PLAY(G1FlatLocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Fixed command for visualization
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
```

- [ ] **Step 2: Commit**

```bash
git add applications/g1_locomotion/config/flat_env_cfg.py
git commit -m "feat(g1): add flat terrain environment config with custom rewards"
```

---

## Task 4: Rough Environment Config (config/rough_env_cfg.py)

**Files:**
- Create: `applications/g1_locomotion/config/rough_env_cfg.py`

- [ ] **Step 1: Write rough env config**

`applications/g1_locomotion/config/rough_env_cfg.py`:
```python
"""G1 rough-terrain locomotion environment configuration."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)

from applications.g1_locomotion.mdp import rewards as custom_mdp


@configclass
class G1RoughLocomotionEnvCfg(G1RoughEnvCfg):
    """Custom G1 rough env — used after flat-terrain walking is stable."""

    def __post_init__(self):
        super().__post_init__()

        # --- Scene ---
        self.scene.num_envs = 1024

        # --- Commands: wider range for rough terrain ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # --- Rewards: same structure as flat but adjusted weights ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_slide.weight = -0.1

        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_torques_l2.weight = -1.5e-7

        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_hip.weight = -0.1
        self.rewards.joint_deviation_torso.weight = -0.1
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.termination_penalty.weight = -200.0

        # Custom rewards
        self.rewards.gait_symmetry = RewTerm(
            func=custom_mdp.bipedal_gait_symmetry,
            weight=0.3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.base_height = RewTerm(
            func=custom_mdp.base_height_reward,
            weight=0.3,
            params={"target_height": 0.74, "sigma": 0.08},
        )


@configclass
class G1RoughLocomotionEnvCfg_PLAY(G1RoughLocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
```

- [ ] **Step 2: Commit**

```bash
git add applications/g1_locomotion/config/rough_env_cfg.py
git commit -m "feat(g1): add rough terrain environment config"
```

---

## Task 5: PPO Runner Config (config/ppo_cfg.py)

**Files:**
- Create: `applications/g1_locomotion/config/ppo_cfg.py`

- [ ] **Step 1: Write PPO config**

`applications/g1_locomotion/config/ppo_cfg.py`:
```python
"""rsl_rl PPO runner configurations for G1 locomotion."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1FlatPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "g1_flat_locomotion"
    logger = "tensorboard"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1RoughPPOCfg(G1FlatPPOCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "g1_rough_locomotion"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
```

- [ ] **Step 2: Commit**

```bash
git add applications/g1_locomotion/config/ppo_cfg.py
git commit -m "feat(g1): add rsl_rl PPO runner configurations"
```

---

## Task 6: Training Script (scripts/train_teacher.py)

**Files:**
- Create: `applications/g1_locomotion/scripts/train_teacher.py`

This script follows the Isaac Lab rsl_rl training pattern but points to our custom configs.

- [ ] **Step 1: Write training script**

`applications/g1_locomotion/scripts/train_teacher.py`:
```python
"""Phase 1: Train G1 locomotion teacher policy with rsl_rl PPO.

Usage:
    conda activate env_isaaclab
    python applications/g1_locomotion/scripts/train_teacher.py --task G1-Flat-Custom-v0 --num_envs 1024
    python applications/g1_locomotion/scripts/train_teacher.py --task G1-Rough-Custom-v0 --num_envs 1024
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 locomotion teacher.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--task", type=str, default="G1-Flat-Custom-v0", help="Task name.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max iterations.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after AppLauncher."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register our custom environments
import applications.g1_locomotion.config  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path


def main():
    # Load config from registry
    env_cfg: ManagerBasedRLEnvCfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()

    # Override from CLI
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed

    # Log directory
    log_root_path = os.path.join(
        os.path.dirname(__file__), "..", "results", agent_cfg.experiment_name
    )
    log_root_path = os.path.abspath(log_root_path)
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    # Resume if requested
    if args_cli.resume and args_cli.load_run:
        resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")
        runner.load(resume_path)

    # Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Save final model
    final_path = os.path.join(log_dir, "teacher_final.pt")
    runner.save(final_path)
    print(f"[INFO] Final model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Register custom gym environments in config/__init__.py**

Update `applications/g1_locomotion/config/__init__.py` to include gym registrations:
```python
import gymnasium as gym

from .flat_env_cfg import G1FlatLocomotionEnvCfg, G1FlatLocomotionEnvCfg_PLAY
from .rough_env_cfg import G1RoughLocomotionEnvCfg, G1RoughLocomotionEnvCfg_PLAY
from .ppo_cfg import G1FlatPPOCfg, G1RoughPPOCfg

gym.register(
    id="G1-Flat-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FlatLocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": G1FlatPPOCfg,
    },
)

gym.register(
    id="G1-Flat-Custom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FlatLocomotionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": G1FlatPPOCfg,
    },
)

gym.register(
    id="G1-Rough-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1RoughLocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": G1RoughPPOCfg,
    },
)

gym.register(
    id="G1-Rough-Custom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1RoughLocomotionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": G1RoughPPOCfg,
    },
)
```

- [ ] **Step 3: Commit**

```bash
git add applications/g1_locomotion/scripts/train_teacher.py applications/g1_locomotion/config/__init__.py
git commit -m "feat(g1): add teacher training script + gym environment registration"
```

---

## Task 7: Play / Visualization Script (scripts/play.py)

**Files:**
- Create: `applications/g1_locomotion/scripts/play.py`

- [ ] **Step 1: Write play script**

`applications/g1_locomotion/scripts/play.py`:
```python
"""Visualize trained G1 locomotion policy.

Usage:
    python applications/g1_locomotion/scripts/play.py --task G1-Flat-Custom-Play-v0 \
        --load_run 2026-06-15_12-00-00 --checkpoint teacher_final.pt
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play G1 locomotion policy.")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments.")
parser.add_argument("--task", type=str, default="G1-Flat-Custom-Play-v0", help="Task name.")
parser.add_argument("--load_run", type=str, required=True, help="Run folder name.")
parser.add_argument("--checkpoint", type=str, default="model_1500.pt", help="Checkpoint file.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after AppLauncher."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import applications.g1_locomotion.config  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path


def main():
    env_cfg: ManagerBasedRLEnvCfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # Determine log path
    log_root_path = os.path.join(
        os.path.dirname(__file__), "..", "results", agent_cfg.experiment_name
    )
    log_root_path = os.path.abspath(log_root_path)

    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    print(f"[INFO] Loading policy from: {resume_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner and load
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(resume_path)

    # Get policy
    policy = runner.get_inference_policy(device="cuda:0")

    # Run visualization loop
    obs, _ = env.get_observations()
    print("[INFO] Running policy... Press Ctrl+C to stop.")
    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Commit**

```bash
git add applications/g1_locomotion/scripts/play.py
git commit -m "feat(g1): add play/visualization script"
```

---

## Task 8: Teacher Data Collection Script (scripts/collect_teacher_data.py)

**Files:**
- Create: `applications/g1_locomotion/scripts/collect_teacher_data.py`

- [ ] **Step 1: Write data collection script**

`applications/g1_locomotion/scripts/collect_teacher_data.py`:
```python
"""Phase 2 Step 1: Collect teacher rollout data for student distillation.

Runs the frozen teacher in a DR environment and records
(obs_history, obs_current, teacher_action) tuples.

Usage:
    python applications/g1_locomotion/scripts/collect_teacher_data.py \
        --task G1-Flat-Custom-v0 --load_run <run_dir> --checkpoint teacher_final.pt \
        --num_steps 500000
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect teacher data for student distillation.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--task", type=str, default="G1-Flat-Custom-v0", help="Task name.")
parser.add_argument("--load_run", type=str, required=True, help="Run folder with teacher checkpoint.")
parser.add_argument("--checkpoint", type=str, default="teacher_final.pt", help="Teacher checkpoint.")
parser.add_argument("--num_steps", type=int, default=500000, help="Total transitions to collect.")
parser.add_argument("--history_length", type=int, default=50, help="Observation history length.")
parser.add_argument("--output", type=str, default=None, help="Output .npz path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after AppLauncher."""

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import applications.g1_locomotion.config  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path


def main():
    env_cfg: ManagerBasedRLEnvCfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]()

    env_cfg.scene.num_envs = args_cli.num_envs

    log_root_path = os.path.join(
        os.path.dirname(__file__), "..", "results", agent_cfg.experiment_name
    )
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    print(f"[INFO] Loading teacher from: {resume_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load teacher
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # Determine obs_dim from environment
    obs, _ = env.get_observations()
    obs_dim = obs.shape[-1]
    num_envs = args_cli.num_envs
    history_length = args_cli.history_length
    total_steps = args_cli.num_steps
    steps_per_env = total_steps // num_envs + 1

    print(f"[INFO] obs_dim={obs_dim}, num_envs={num_envs}, history_length={history_length}")
    print(f"[INFO] Collecting {total_steps} transitions ({steps_per_env} steps/env)...")

    # Rolling history buffer [num_envs, history_length, obs_dim]
    obs_history_buf = torch.zeros(num_envs, history_length, obs_dim, device="cuda:0")

    # Storage (CPU to save GPU memory)
    all_obs_history = []
    all_obs_current = []
    all_actions = []
    collected = 0

    for step in range(steps_per_env):
        # Update history buffer (shift left, append current obs)
        obs_history_buf = torch.roll(obs_history_buf, shifts=-1, dims=1)
        obs_history_buf[:, -1, :] = obs

        # Get deterministic teacher action (use mean, no sampling)
        with torch.no_grad():
            actions = policy(obs)

        # Store data (skip first history_length steps to fill buffer)
        if step >= history_length:
            all_obs_history.append(obs_history_buf.reshape(num_envs, -1).cpu().numpy())
            all_obs_current.append(obs.cpu().numpy())
            all_actions.append(actions.cpu().numpy())
            collected += num_envs

            if collected % 50000 < num_envs:
                print(f"  collected {collected}/{total_steps} transitions")

        if collected >= total_steps:
            break

        # Step environment
        obs, _, _, _, _ = env.step(actions)

    # Save
    output_path = args_cli.output or os.path.join(
        log_root_path, "teacher_distill_data.npz"
    )
    obs_history_arr = np.concatenate(all_obs_history, axis=0)[:total_steps]
    obs_current_arr = np.concatenate(all_obs_current, axis=0)[:total_steps]
    actions_arr = np.concatenate(all_actions, axis=0)[:total_steps]

    np.savez_compressed(
        output_path,
        obs_history=obs_history_arr,
        obs_current=obs_current_arr,
        teacher_actions=actions_arr,
    )
    print(f"[INFO] Saved {obs_history_arr.shape[0]} transitions to: {output_path}")
    print(f"  obs_history shape: {obs_history_arr.shape}")
    print(f"  obs_current shape: {obs_current_arr.shape}")
    print(f"  actions shape: {actions_arr.shape}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Commit**

```bash
git add applications/g1_locomotion/scripts/collect_teacher_data.py
git commit -m "feat(g1): add teacher data collection script for student distillation"
```

---

## Task 9: Student Networks (student/networks.py)

**Files:**
- Create: `applications/g1_locomotion/student/networks.py`

- [ ] **Step 1: Write network architectures**

`applications/g1_locomotion/student/networks.py`:
```python
"""Teacher-Student RMA network architectures for G1 locomotion.

AdaptationModule: obs_history -> latent z (implicit environment parameters)
StudentPolicy: (obs_current, z) -> action
StudentONNXWrapper: fused module for ONNX export
"""

import torch
import torch.nn as nn


class AdaptationModule(nn.Module):
    """Encodes observation history into a latent representation of environment parameters.

    Input: obs_history of shape (batch, history_length * obs_dim)
    Output: latent z of shape (batch, latent_dim)
    """

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.net(obs_history)


class StudentPolicy(nn.Module):
    """Student policy that takes current obs + latent z and outputs actions.

    Input: obs_current (batch, obs_dim) + latent z (batch, latent_dim)
    Output: action (batch, action_dim)
    """

    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs_current: torch.Tensor, latent_z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_current, latent_z], dim=-1)
        return self.net(x)


class StudentONNXWrapper(nn.Module):
    """Fused AdaptationModule + StudentPolicy for ONNX export.

    Single forward pass: (obs_history, obs_current) -> action
    """

    def __init__(self, adaptation: AdaptationModule, policy: StudentPolicy):
        super().__init__()
        self.adaptation = adaptation
        self.policy = policy

    def forward(self, obs_history: torch.Tensor, obs_current: torch.Tensor) -> torch.Tensor:
        latent_z = self.adaptation(obs_history)
        action = self.policy(obs_current, latent_z)
        return action
```

- [ ] **Step 2: Commit**

```bash
git add applications/g1_locomotion/student/networks.py
git commit -m "feat(g1): add AdaptationModule + StudentPolicy + ONNX wrapper"
```

---

## Task 10: Student BC Training (student/train_student.py)

**Files:**
- Create: `applications/g1_locomotion/student/train_student.py`

- [ ] **Step 1: Write student training script**

`applications/g1_locomotion/student/train_student.py`:
```python
"""Phase 2 Step 2: Train student policy via Behavior Cloning.

Trains AdaptationModule + StudentPolicy to mimic teacher actions
using (obs_history, obs_current, teacher_action) dataset.

Usage:
    python applications/g1_locomotion/student/train_student.py \
        --data results/g1_flat_locomotion/teacher_distill_data.npz
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from applications.g1_locomotion.student.networks import (
    AdaptationModule,
    StudentPolicy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train G1 student policy (BC).")
    parser.add_argument("--data", type=str, required=True, help="Path to teacher_distill_data.npz")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints.")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    print(f"[INFO] Loading data from: {args.data}")
    data = np.load(args.data)
    obs_history = torch.from_numpy(data["obs_history"]).float()
    obs_current = torch.from_numpy(data["obs_current"]).float()
    teacher_actions = torch.from_numpy(data["teacher_actions"]).float()

    history_dim = obs_history.shape[-1]
    obs_dim = obs_current.shape[-1]
    action_dim = teacher_actions.shape[-1]
    num_samples = obs_history.shape[0]

    print(f"[INFO] Dataset: {num_samples} samples")
    print(f"  obs_history_dim={history_dim}, obs_dim={obs_dim}, action_dim={action_dim}")

    # Train/val split
    dataset = TensorDataset(obs_history, obs_current, teacher_actions)
    val_size = int(num_samples * args.val_ratio)
    train_size = num_samples - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create models
    device = torch.device(args.device)
    adaptation = AdaptationModule(input_dim=history_dim, latent_dim=args.latent_dim).to(device)
    policy = StudentPolicy(obs_dim=obs_dim, latent_dim=args.latent_dim, action_dim=action_dim).to(device)

    # Optimizer
    params = list(adaptation.parameters()) + list(policy.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss()

    # Output directory
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.data), "student")
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"[INFO] Training for up to {args.epochs} epochs (patience={args.patience})...")

    for epoch in range(args.epochs):
        # Train
        adaptation.train()
        policy.train()
        train_loss_sum = 0.0
        train_batches = 0

        for hist, obs, actions in train_loader:
            hist, obs, actions = hist.to(device), obs.to(device), actions.to(device)

            z = adaptation(hist)
            pred_actions = policy(obs, z)
            loss = criterion(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches

        # Validate
        adaptation.eval()
        policy.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for hist, obs, actions in val_loader:
                hist, obs, actions = hist.to(device), obs.to(device), actions.to(device)
                z = adaptation(hist)
                pred_actions = policy(obs, z)
                loss = criterion(pred_actions, actions)
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / val_batches

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "adaptation": adaptation.state_dict(),
                "policy": policy.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "history_dim": history_dim,
                "latent_dim": args.latent_dim,
            }, os.path.join(output_dir, "student_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.6f})")
                break

    # Save final
    torch.save({
        "adaptation": adaptation.state_dict(),
        "policy": policy.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "history_dim": history_dim,
        "latent_dim": args.latent_dim,
    }, os.path.join(output_dir, "student_final.pt"))

    print(f"[INFO] Training complete. Best val_loss={best_val_loss:.6f}")
    print(f"  Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add applications/g1_locomotion/student/train_student.py
git commit -m "feat(g1): add student BC training script"
```

---

## Task 11: Student Sim2Sim Evaluation (student/evaluate.py)

**Files:**
- Create: `applications/g1_locomotion/student/evaluate.py`

- [ ] **Step 1: Write evaluation script**

`applications/g1_locomotion/student/evaluate.py`:
```python
"""Phase 2 Step 3: Sim2Sim validation of student policy.

Runs trained student in clean environment (no DR) and reports:
- avg_reward vs teacher baseline
- survival rate
- tracking ratio

Usage:
    python applications/g1_locomotion/student/evaluate.py \
        --task G1-Flat-Custom-Play-v0 --student_path results/.../student/student_best.pt \
        --episodes 20
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate G1 student policy (Sim2Sim).")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments.")
parser.add_argument("--task", type=str, default="G1-Flat-Custom-Play-v0", help="Task name (PLAY variant).")
parser.add_argument("--student_path", type=str, required=True, help="Path to student checkpoint.")
parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to evaluate.")
parser.add_argument("--history_length", type=int, default=50, help="Observation history length.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after AppLauncher."""

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import applications.g1_locomotion.config  # noqa: F401

from applications.g1_locomotion.student.networks import (
    AdaptationModule,
    StudentPolicy,
    StudentONNXWrapper,
)


def main():
    device = torch.device("cuda:0")

    # Load student
    print(f"[INFO] Loading student from: {args_cli.student_path}")
    ckpt = torch.load(args_cli.student_path, map_location=device)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    history_dim = ckpt["history_dim"]
    latent_dim = ckpt["latent_dim"]

    adaptation = AdaptationModule(input_dim=history_dim, latent_dim=latent_dim).to(device)
    policy = StudentPolicy(obs_dim=obs_dim, latent_dim=latent_dim, action_dim=action_dim).to(device)
    adaptation.load_state_dict(ckpt["adaptation"])
    policy.load_state_dict(ckpt["policy"])
    adaptation.eval()
    policy.eval()

    # Create environment (PLAY variant: no DR, no push)
    env_cfg: ManagerBasedRLEnvCfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=True)

    num_envs = args_cli.num_envs
    history_length = args_cli.history_length

    # Rolling history buffer
    obs_history_buf = torch.zeros(num_envs, history_length, obs_dim, device=device)

    # Metrics
    episode_rewards = []
    episode_lengths = []
    max_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    completed_episodes = 0

    obs, _ = env.get_observations()
    current_rewards = torch.zeros(num_envs, device=device)
    current_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)

    print(f"[INFO] Evaluating {args_cli.episodes} episodes (max_steps={max_steps})...")

    while completed_episodes < args_cli.episodes:
        # Update history
        obs_history_buf = torch.roll(obs_history_buf, shifts=-1, dims=1)
        obs_history_buf[:, -1, :] = obs

        # Student inference
        with torch.no_grad():
            hist_flat = obs_history_buf.reshape(num_envs, -1)
            z = adaptation(hist_flat)
            actions = policy(obs, z)

        # Step
        obs, rewards, dones, truncated, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        # Check for done episodes
        done_mask = dones.bool() | truncated.bool() if truncated is not None else dones.bool()
        if done_mask.any():
            for i in done_mask.nonzero(as_tuple=True)[0]:
                episode_rewards.append(current_rewards[i].item())
                episode_lengths.append(current_lengths[i].item())
                completed_episodes += 1
                # Reset tracking for this env
                current_rewards[i] = 0.0
                current_lengths[i] = 0
                obs_history_buf[i] = 0.0

                if completed_episodes >= args_cli.episodes:
                    break

    env.close()

    # Report
    rewards_arr = np.array(episode_rewards[:args_cli.episodes])
    lengths_arr = np.array(episode_lengths[:args_cli.episodes])
    survival_rate = np.mean(lengths_arr >= max_steps * 0.95) * 100

    print("\n" + "=" * 50)
    print("G1 Student Sim2Sim Evaluation Results")
    print("=" * 50)
    print(f"  Episodes:       {len(rewards_arr)}")
    print(f"  Avg Reward:     {rewards_arr.mean():.2f} +/- {rewards_arr.std():.2f}")
    print(f"  Avg Length:     {lengths_arr.mean():.1f} / {max_steps} steps")
    print(f"  Survival Rate:  {survival_rate:.1f}%")
    print(f"  Min Reward:     {rewards_arr.min():.2f}")
    print(f"  Max Reward:     {rewards_arr.max():.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Commit**

```bash
git add applications/g1_locomotion/student/evaluate.py
git commit -m "feat(g1): add student Sim2Sim evaluation script"
```

---

## Task 12: ONNX Export (export/export_onnx.py)

**Files:**
- Create: `applications/g1_locomotion/export/export_onnx.py`

- [ ] **Step 1: Write ONNX export script**

`applications/g1_locomotion/export/export_onnx.py`:
```python
"""Phase 3: Export student policy to ONNX format.

Fuses AdaptationModule + StudentPolicy into a single graph.

Usage:
    python applications/g1_locomotion/export/export_onnx.py \
        --student_path results/.../student/student_best.pt \
        --output results/student_g1.onnx
"""

import argparse
import os

import numpy as np
import torch

from applications.g1_locomotion.student.networks import (
    AdaptationModule,
    StudentONNXWrapper,
    StudentPolicy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export G1 student to ONNX.")
    parser.add_argument("--student_path", type=str, required=True, help="Student checkpoint path.")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify accuracy.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    # Load student
    print(f"[INFO] Loading student from: {args.student_path}")
    ckpt = torch.load(args.student_path, map_location=device)
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    history_dim = ckpt["history_dim"]
    latent_dim = ckpt["latent_dim"]

    adaptation = AdaptationModule(input_dim=history_dim, latent_dim=latent_dim)
    policy = StudentPolicy(obs_dim=obs_dim, latent_dim=latent_dim, action_dim=action_dim)
    adaptation.load_state_dict(ckpt["adaptation"])
    policy.load_state_dict(ckpt["policy"])

    # Fuse into ONNX wrapper
    wrapper = StudentONNXWrapper(adaptation, policy)
    wrapper.eval()

    # Dummy inputs
    dummy_history = torch.randn(1, history_dim)
    dummy_obs = torch.randn(1, obs_dim)

    # Export
    output_path = args.output or os.path.join(os.path.dirname(args.student_path), "student_g1.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[INFO] Exporting to: {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy_history, dummy_obs),
        output_path,
        opset_version=args.opset,
        input_names=["obs_history", "obs_current"],
        output_names=["action"],
        dynamic_axes={
            "obs_history": {0: "batch"},
            "obs_current": {0: "batch"},
            "action": {0: "batch"},
        },
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] Export complete: {output_path} ({file_size:.2f} MB)")

    # Verify accuracy
    if args.verify:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)

        # Run PyTorch
        with torch.no_grad():
            pt_output = wrapper(dummy_history, dummy_obs).numpy()

        # Run ONNX
        ort_output = session.run(
            None,
            {
                "obs_history": dummy_history.numpy(),
                "obs_current": dummy_obs.numpy(),
            },
        )[0]

        max_diff = np.abs(pt_output - ort_output).max()
        print(f"[INFO] Accuracy check: max|PyTorch - ONNX| = {max_diff:.2e}", end=" ")
        if max_diff < 1e-4:
            print("✓ PASS")
        else:
            print("✗ FAIL")

    print("\nExport Summary:")
    print(f"  Model: {os.path.basename(output_path)}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Input: obs_history ({history_dim},) + obs_current ({obs_dim},)")
    print(f"  Output: action ({action_dim},)")
    print(f"  Opset: {args.opset}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add applications/g1_locomotion/export/export_onnx.py
git commit -m "feat(g1): add ONNX export script with accuracy verification"
```

---

## Task 13: ONNX Benchmark (export/benchmark.py)

**Files:**
- Create: `applications/g1_locomotion/export/benchmark.py`

- [ ] **Step 1: Write benchmark script**

`applications/g1_locomotion/export/benchmark.py`:
```python
"""Benchmark ONNX model inference latency.

Usage:
    python applications/g1_locomotion/export/benchmark.py \
        --model results/.../student_g1.onnx --runs 1000
"""

import argparse
import time

import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark G1 student ONNX inference.")
    parser.add_argument("--model", type=str, required=True, help="ONNX model path.")
    parser.add_argument("--runs", type=int, default=1000, help="Number of inference runs.")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup runs.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading model: {args.model}")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    # Get input shapes
    inputs = session.get_inputs()
    input_shapes = {inp.name: inp.shape for inp in inputs}
    print(f"[INFO] Inputs: {input_shapes}")

    # Create dummy inputs (batch=1)
    feed = {}
    for inp in inputs:
        shape = [1 if isinstance(d, str) else d for d in inp.shape]
        feed[inp.name] = np.random.randn(*shape).astype(np.float32)

    # Warmup
    print(f"[INFO] Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        session.run(None, feed)

    # Benchmark
    print(f"[INFO] Benchmarking ({args.runs} runs)...")
    latencies = []
    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(None, feed)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    # Report
    import os
    file_size = os.path.getsize(args.model) / (1024 * 1024)

    print("\n" + "=" * 50)
    print("G1 Student ONNX Benchmark Results")
    print("=" * 50)
    print(f"  Model:    {os.path.basename(args.model)}")
    print(f"  Size:     {file_size:.2f} MB")
    print(f"  Runs:     {args.runs}")
    print(f"  Latency:")
    print(f"    avg:    {latencies.mean():.3f} ms")
    print(f"    p50:    {np.percentile(latencies, 50):.3f} ms")
    print(f"    p95:    {np.percentile(latencies, 95):.3f} ms")
    print(f"    p99:    {np.percentile(latencies, 99):.3f} ms")
    print(f"    max:    {latencies.max():.3f} ms")
    print(f"  Budget:   20ms (50Hz control)")
    budget_pass = latencies.mean() < 20.0
    print(f"  Status:   {'✓ PASS' if budget_pass else '✗ FAIL'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add applications/g1_locomotion/export/benchmark.py
git commit -m "feat(g1): add ONNX inference benchmark script"
```

---

## Task 14: Final Integration + Smoke Test

**Files:**
- Modify: `applications/g1_locomotion/config/__init__.py` (ensure gym registration works)
- Verify: all imports resolve correctly

- [ ] **Step 1: Verify the full import chain works**

Run in `env_isaaclab` conda environment:
```bash
conda activate env_isaaclab
cd /home/lihongl/Desktop/myRL/easyRL
python -c "
import sys
sys.path.insert(0, '.')
from applications.g1_locomotion.config import (
    G1FlatLocomotionEnvCfg, G1RoughLocomotionEnvCfg, G1FlatPPOCfg, G1RoughPPOCfg
)
print('Config imports OK')
from applications.g1_locomotion.student import AdaptationModule, StudentPolicy, StudentONNXWrapper
print('Student imports OK')
print('All imports successful!')
"
```

Expected output:
```
Config imports OK
Student imports OK
All imports successful!
```

- [ ] **Step 2: Verify gym environment registration**

```bash
python -c "
import sys
sys.path.insert(0, '.')
import applications.g1_locomotion.config
import gymnasium as gym
spec = gym.spec('G1-Flat-Custom-v0')
print(f'Task registered: {spec.id}')
print(f'Entry point: {spec.entry_point}')
print('Gym registration OK!')
"
```

- [ ] **Step 3: Create a quick-start section in training_log.md**

Update `applications/g1_locomotion/training_log.md` to include commands:
```markdown
# G1 Locomotion Training Log

Per-round training results, diagnostics, and iteration strategy.

## Quick Start

```bash
# Phase 1: Train teacher (flat terrain)
conda activate env_isaaclab
python applications/g1_locomotion/scripts/train_teacher.py --task G1-Flat-Custom-v0 --num_envs 1024

# Visualize
python applications/g1_locomotion/scripts/play.py --task G1-Flat-Custom-Play-v0 --load_run <run_dir>

# Phase 2: Collect data + train student
python applications/g1_locomotion/scripts/collect_teacher_data.py --task G1-Flat-Custom-v0 --load_run <run_dir>
python applications/g1_locomotion/student/train_student.py --data results/g1_flat_locomotion/teacher_distill_data.npz

# Phase 2: Evaluate student
python applications/g1_locomotion/student/evaluate.py --task G1-Flat-Custom-Play-v0 --student_path results/.../student/student_best.pt

# Phase 3: Export + benchmark
python applications/g1_locomotion/export/export_onnx.py --student_path results/.../student/student_best.pt
python applications/g1_locomotion/export/benchmark.py --model results/.../student_g1.onnx
```

---

## Round 1

**Config:** (to be filled when training starts)

**TensorBoard:** `results/<run_dir>/`

**Results:** (pending)
```

- [ ] **Step 4: Final commit**

```bash
git add applications/g1_locomotion/training_log.md
git commit -m "feat(g1): add quick-start commands to training log"
```

---

## Execution Order Summary

| Task | Phase | Description |
|------|-------|-------------|
| 1 | Setup | Project scaffold + package registration |
| 2 | Phase 1 | Custom reward terms |
| 3 | Phase 1 | Flat environment config |
| 4 | Phase 1 | Rough environment config |
| 5 | Phase 1 | PPO runner config |
| 6 | Phase 1 | Training script |
| 7 | Phase 1 | Play/visualization script |
| 8 | Phase 2 | Teacher data collection |
| 9 | Phase 2 | Student network architectures |
| 10 | Phase 2 | Student BC training |
| 11 | Phase 2 | Student Sim2Sim evaluation |
| 12 | Phase 3 | ONNX export |
| 13 | Phase 3 | ONNX benchmark |
| 14 | All | Integration verification |

Tasks 1-7 must be done sequentially. Tasks 8-13 can be done sequentially after Phase 1 training produces a checkpoint. Task 14 verifies everything works together.
