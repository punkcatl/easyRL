"""Phase 3: Sim-to-Sim validation comparing Baseline / DR-only / Full Pipeline.

Usage:
    python evaluate.py --env Ant-v4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import gymnasium as gym
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config, ENV_CONFIGS, PRIVILEGED_DIM
from envs.domain_randomization import DomainRandomizationWrapper
from agent.teacher import TeacherAgent
from agent.student import StudentAgent
from agent.ppo_continuous import PPOContinuous


def evaluate_agent(agent, env, n_episodes, agent_type="student"):
    """Evaluate an agent and return metrics including Cost of Transport."""
    survival_steps = []
    forward_velocities = []
    total_rewards = []
    cost_of_transports = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        privileged = info.get("privileged_info", np.zeros(PRIVILEGED_DIM, dtype=np.float32))

        if agent_type == "student":
            agent.reset_history()

        done = False
        steps = 0
        ep_reward = 0.0
        velocities = []
        total_energy = 0.0
        start_pos = None

        # Get robot mass and initial position for CoT
        if hasattr(env.unwrapped, "data"):
            total_mass = float(env.unwrapped.model.body_mass.sum())
            start_pos = env.unwrapped.data.qpos[0].copy()
        else:
            total_mass = 1.0
            start_pos = 0.0

        while not done:
            if agent_type == "student":
                action = agent.act(obs)
            elif agent_type == "teacher":
                action, _, _ = agent.act(
                    obs[np.newaxis], privileged[np.newaxis]
                )
                action = action[0]
            else:  # baseline
                obs_norm = agent.obs_rms.normalize(obs[np.newaxis])
                action, _, _ = agent.act(obs_norm)
                action = action[0]

            obs, reward, terminated, truncated, info = env.step(action)
            privileged = info.get(
                "privileged_info", np.zeros(PRIVILEGED_DIM, dtype=np.float32)
            )
            done = terminated or truncated
            steps += 1
            ep_reward += reward

            if hasattr(env.unwrapped, "data"):
                qvel = env.unwrapped.data.qvel
                velocities.append(qvel[0])
                # Approximate energy: sum of |action_i * joint_velocity_i|
                joint_vel = qvel[6:6 + len(action)] if len(qvel) > 6 else qvel[:len(action)]
                total_energy += float(np.sum(np.abs(action * joint_vel)))

        survival_steps.append(steps)
        total_rewards.append(ep_reward)
        if velocities:
            forward_velocities.append(np.mean(velocities))

        # Cost of Transport = total_energy / (mass * distance)
        if hasattr(env.unwrapped, "data"):
            end_pos = env.unwrapped.data.qpos[0]
            distance = abs(float(end_pos) - float(start_pos))
            if distance > 1e-6:
                cot = total_energy / (total_mass * distance)
            else:
                cot = float("inf")
            cost_of_transports.append(cot)

    cot_mean = np.mean(cost_of_transports) if cost_of_transports else 0.0
    cot_values = [c for c in cost_of_transports if np.isfinite(c)]
    cot_mean = np.mean(cot_values) if cot_values else float("inf")

    return {
        "survival_mean": np.mean(survival_steps),
        "survival_std": np.std(survival_steps),
        "velocity_mean": np.mean(forward_velocities) if forward_velocities else 0.0,
        "reward_mean": np.mean(total_rewards),
        "reward_std": np.std(total_rewards),
        "cot_mean": cot_mean,
    }


def make_test_env(env_id, test_type, cfg):
    """Create test environment with specific DR configuration."""
    env = gym.make(env_id)

    if test_type == "nominal":
        return env

    elif test_type == "in_dist":
        test_config = cfg.copy()
        test_config["dr_mass_range_init"] = [1.2, 1.2]
        test_config["dr_mass_range_final"] = [1.2, 1.2]
        test_config["dr_friction_range_init"] = [0.7, 0.7]
        test_config["dr_friction_range_final"] = [0.7, 0.7]
        test_config["dr_gain_range_init"] = [1.0, 1.0]
        test_config["dr_gain_range_final"] = [1.0, 1.0]
        test_config["dr_delay_range_init"] = [0, 0]
        test_config["dr_delay_range_final"] = [0, 0]
        test_config["dr_force_range_init"] = [0.0, 0.0]
        test_config["dr_force_range_final"] = [0.0, 0.0]
        return DomainRandomizationWrapper(env, test_config)

    elif test_type == "ood":
        ood = cfg["eval_ood_factor"]
        test_config = cfg.copy()
        test_config["dr_mass_range_init"] = [ood * 1.3, ood * 1.3]
        test_config["dr_mass_range_final"] = [ood * 1.3, ood * 1.3]
        test_config["dr_friction_range_init"] = [0.35, 0.35]
        test_config["dr_friction_range_final"] = [0.35, 0.35]
        test_config["dr_gain_range_init"] = [1.0, 1.0]
        test_config["dr_gain_range_final"] = [1.0, 1.0]
        test_config["dr_delay_range_init"] = [0, 0]
        test_config["dr_delay_range_final"] = [0, 0]
        test_config["dr_force_range_init"] = [0.0, 0.0]
        test_config["dr_force_range_final"] = [0.0, 0.0]
        return DomainRandomizationWrapper(env, test_config)

    elif test_type == "perturbation":
        test_config = cfg.copy()
        test_config["dr_mass_range_init"] = [1.0, 1.0]
        test_config["dr_mass_range_final"] = [1.0, 1.0]
        test_config["dr_friction_range_init"] = [1.0, 1.0]
        test_config["dr_friction_range_final"] = [1.0, 1.0]
        test_config["dr_gain_range_init"] = [1.0, 1.0]
        test_config["dr_gain_range_final"] = [1.0, 1.0]
        test_config["dr_delay_range_init"] = [0, 0]
        test_config["dr_delay_range_final"] = [0, 0]
        test_config["dr_force_range_init"] = [50.0, 50.0]
        test_config["dr_force_range_final"] = [50.0, 50.0]
        test_config["dr_force_interval_init"] = 50
        test_config["dr_force_interval_final"] = 50
        return DomainRandomizationWrapper(env, test_config)

    raise ValueError(f"Unknown test_type: {test_type}")


def run_evaluation(env_id: str, ood: bool = False):
    env_cfg = ENV_CONFIGS[env_id]
    obs_dim = env_cfg["obs_dim"]
    action_dim = env_cfg["action_dim"]
    n_episodes = config["eval_episodes"]

    eval_config = config.copy()

    # If --ood flag is set, scale DR ranges beyond training distribution
    if ood:
        ood_factor = eval_config["eval_ood_factor"]
        eval_config["dr_mass_range_final"] = [
            eval_config["dr_mass_range_final"][0] * ood_factor,
            eval_config["dr_mass_range_final"][1] * ood_factor,
        ]
        eval_config["dr_friction_range_final"] = [
            eval_config["dr_friction_range_final"][0] / ood_factor,
            eval_config["dr_friction_range_final"][1] * ood_factor,
        ]
        eval_config["dr_force_range_final"] = [
            eval_config["dr_force_range_final"][0],
            eval_config["dr_force_range_final"][1] * ood_factor,
        ]
        print(f"OOD evaluation enabled: DR ranges scaled by factor {ood_factor}")

    results_dir = Path(__file__).resolve().parent / "results"

    # Load agents
    baseline = PPOContinuous(obs_dim, action_dim, config)
    baseline_path = results_dir / f"baseline_{env_id.replace('-', '_')}.pth"
    if baseline_path.exists():
        baseline.load(str(baseline_path))
        print(f"Loaded Baseline from {baseline_path}")
    else:
        print(f"WARNING: Baseline not found at {baseline_path}, using random policy")

    teacher = TeacherAgent(obs_dim, PRIVILEGED_DIM, action_dim, config)
    teacher_path = results_dir / f"teacher_{env_id.replace('-', '_')}.pth"
    if teacher_path.exists():
        teacher.load(str(teacher_path))
        print(f"Loaded Teacher from {teacher_path}")
    else:
        print(f"WARNING: Teacher not found at {teacher_path}, using random policy")

    student = StudentAgent(obs_dim, action_dim, config)
    student_path = results_dir / f"student_{env_id.replace('-', '_')}.pth"
    if student_path.exists():
        student.load(str(student_path))
        print(f"Loaded Student from {student_path}")
    else:
        print(f"WARNING: Student not found at {student_path}, using random policy")

    # Evaluation
    test_types = ["nominal", "in_dist", "ood", "perturbation"]
    agents = {
        "Baseline": (baseline, "baseline"),
        "DR only (Teacher)": (teacher, "teacher"),
        "Full Pipeline (Student)": (student, "student"),
    }

    results = {}
    for test_type in test_types:
        results[test_type] = {}
        env = make_test_env(env_id, test_type, eval_config)

        for agent_name, (agent, agent_type) in agents.items():
            print(f"Evaluating {agent_name} on {test_type}...")
            metrics = evaluate_agent(agent, env, n_episodes, agent_type)
            results[test_type][agent_name] = metrics

        env.close()

    # Print comparison table
    print(f"\n{'=' * 80}")
    print(f"Evaluation Results: {env_id}")
    print(f"{'=' * 80}")
    header = f"{'Test Domain':<15} {'Agent':<25} {'Survival':<15} {'Velocity':<12} {'Reward':<15}"
    print(header)
    print(f"{'-' * 80}")
    for test_type in test_types:
        for agent_name, metrics in results[test_type].items():
            print(
                f"{test_type:<15} {agent_name:<25} "
                f"{metrics['survival_mean']:.0f}+/-{metrics['survival_std']:.0f}  "
                f"{metrics['velocity_mean']:.2f}       "
                f"{metrics['reward_mean']:.1f}+/-{metrics['reward_std']:.1f}"
            )

    # Plot
    _plot_results(results, env_id, results_dir)
    return results


def _plot_results(results, env_id, results_dir):
    test_types = list(results.keys())
    agent_names = list(results[test_types[0]].keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_info = [
        ("survival_mean", "Survival Steps"),
        ("velocity_mean", "Forward Velocity (m/s)"),
        ("reward_mean", "Episode Reward"),
    ]

    for metric_idx, (metric, title) in enumerate(metrics_info):
        ax = axes[metric_idx]
        x = np.arange(len(test_types))
        width = 0.25

        for i, agent_name in enumerate(agent_names):
            values = [results[t][agent_name][metric] for t in test_types]
            ax.bar(x + i * width, values, width, label=agent_name)

        ax.set_xlabel("Test Domain")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(test_types, rotation=20)
        ax.legend(fontsize=8)

    plt.suptitle(f"Sim-to-Sim Transfer Evaluation ({env_id})")
    plt.tight_layout()

    plot_path = str(results_dir / f"eval_{env_id.replace('-', '_')}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim-to-Sim Evaluation")
    parser.add_argument("--env", default=config["env_id"], help="Gymnasium env id")
    parser.add_argument(
        "--ood", action="store_true",
        help="Enable out-of-distribution evaluation (scale DR ranges by eval_ood_factor)",
    )
    args = parser.parse_args()
    run_evaluation(env_id=args.env, ood=args.ood)
