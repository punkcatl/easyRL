import csv
import os

import matplotlib.pyplot as plt


def plot_training_curves(log_dir: str, tags: list, save_path: str = None):
    """Read CSV files from log_dir and plot each tag as a subplot.

    Args:
        log_dir: Directory containing CSV log files.
        tags: List of tag names (corresponding to CSV filenames without extension).
        save_path: Optional path to save the figure.
    """
    n_tags = len(tags)
    fig, axes = plt.subplots(n_tags, 1, figsize=(10, 4 * n_tags), squeeze=False)

    for i, tag in enumerate(tags):
        csv_path = os.path.join(log_dir, f"{tag}.csv")
        steps = []
        values = []

        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    steps.append(int(row["step"]))
                    values.append(float(row["value"]))

        ax = axes[i, 0]
        ax.plot(steps, values)
        ax.set_title(tag)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close(fig)


def plot_comparison(results: dict, metric: str, save_path: str = None):
    """Plot comparison of multiple algorithms on a single chart.

    Args:
        results: Dict of {algo_name: {"steps": [...], metric: [...]}}.
        metric: The metric key to plot from each algorithm's results.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, data in results.items():
        ax.plot(data["steps"], data[metric], label=algo_name)

    ax.set_title(f"Comparison: {metric}")
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close(fig)
