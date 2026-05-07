import sys
sys.path.insert(0, "/home/lihongl/Desktop/myRL/easyRL")

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_reward_curve(log_dir: str):
    """Read reward.csv from a log directory and return (steps, values)."""
    csv_path = os.path.join(log_dir, "reward.csv")
    steps = []
    values = []

    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                values.append(float(row["value"]))

    return np.array(steps), np.array(values)


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing to a 1D array."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_reward_comparison(save_path: str = "experiments/results/reward_comparison.png"):
    """Plot smoothed reward curves for DQN, PPO, and SAC."""
    algorithms = {
        "DQN": "experiments/results/dqn",
        "PPO": "experiments/results/ppo",
        "SAC": "experiments/results/sac",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, log_dir in algorithms.items():
        steps, values = load_reward_curve(log_dir)
        if len(values) > 0:
            smoothed = smooth(values, window=10)
            # Adjust steps to match smoothed length
            smoothed_steps = steps[:len(smoothed)]
            ax.plot(smoothed_steps, smoothed, label=algo_name)

    ax.set_title("Reward Comparison: DQN vs PPO vs SAC (Highway Lane Keeping)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward (smoothed, window=10)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Reward comparison plot saved to {save_path}")


def plot_control_metrics(save_path: str = "experiments/results/control_metrics.png"):
    """Plot bar charts for control quality metrics from comparison.json."""
    results_path = "experiments/results/comparison.json"

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    algorithms = list(results.keys())
    metrics = ["lateral_mean", "lateral_std", "heading_mean", "steering_smoothness"]
    titles = [
        "Mean Lateral Deviation",
        "Lateral Deviation Std",
        "Mean Heading Error",
        "Steering Smoothness (lower=smoother)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        values = [results[algo][metric] for algo in algorithms]
        bars = axes[i].bar(algorithms, values, color=colors)
        axes[i].set_title(title)
        axes[i].set_ylabel(metric)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Control Quality Metrics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Control metrics plot saved to {save_path}")


if __name__ == "__main__":
    plot_reward_comparison()
    plot_control_metrics()
