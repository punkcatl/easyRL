"""Unified plotting for all reward shaping experiments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window=50):
    """Smooth training curves with a sliding window."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_sparse_vs_dense(results_dir):
    """Plot Experiment 1: sparse vs dense comparison."""
    for name in ["ant", "highway"]:
        filepath = results_dir / f"sparse_vs_dense_{name}.npy"
        if not filepath.exists():
            continue
        data = np.load(str(filepath), allow_pickle=True).item()

        plt.figure(figsize=(10, 5))
        for label, returns in data.items():
            plt.plot(moving_average(returns), label=label)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title(f"Sparse vs Dense Reward ({name.capitalize()})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(results_dir / f"plot_sparse_vs_dense_{name}.png"), dpi=150)
        plt.close()
        print(f"  Saved: plot_sparse_vs_dense_{name}.png")


def plot_potential_shaping(results_dir):
    """Plot Experiment 2: potential-based shaping."""
    for name in ["ant", "highway"]:
        filepath = results_dir / f"potential_shaping_{name}.npy"
        if not filepath.exists():
            continue
        data = np.load(str(filepath), allow_pickle=True).item()

        plt.figure(figsize=(10, 5))
        for label, returns in data.items():
            plt.plot(moving_average(returns), label=label)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title(f"Potential-based Shaping ({name.capitalize()})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(results_dir / f"plot_potential_shaping_{name}.png"), dpi=150)
        plt.close()
        print(f"  Saved: plot_potential_shaping_{name}.png")


def plot_weight_sensitivity(results_dir):
    """Plot Experiment 3: multi-objective weight sweep."""
    for param in ["speed", "posture", "collision"]:
        filepath = results_dir / f"multi_objective_{param}_sweep.npy"
        if not filepath.exists():
            continue
        data = np.load(str(filepath), allow_pickle=True).item()

        plt.figure(figsize=(10, 5))
        for label, returns in data.items():
            plt.plot(moving_average(returns), label=label)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title(f"Weight Sensitivity: w_{param}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(results_dir / f"plot_weight_sensitivity_{param}.png"), dpi=150)
        plt.close()
        print(f"  Saved: plot_weight_sensitivity_{param}.png")


def plot_hacking_cases(results_dir):
    """Plot Experiment 4: reward hacking broken vs fixed."""
    filepath = results_dir / "hacking_cases.npy"
    if not filepath.exists():
        return
    all_data = np.load(str(filepath), allow_pickle=True).item()

    n_cases = len(all_data)
    fig, axes = plt.subplots(1, n_cases, figsize=(5 * n_cases, 4))
    if n_cases == 1:
        axes = [axes]

    for ax, (case_name, results) in zip(axes, all_data.items()):
        for label, returns in results.items():
            ax.plot(moving_average(returns), label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title(case_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(results_dir / "plot_hacking_cases.png"), dpi=150)
    plt.close()
    print("  Saved: plot_hacking_cases.png")


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"

    print("Generating plots...")
    plot_sparse_vs_dense(results_dir)
    plot_potential_shaping(results_dir)
    plot_weight_sensitivity(results_dir)
    plot_hacking_cases(results_dir)
    print("Done.")


if __name__ == "__main__":
    main()
