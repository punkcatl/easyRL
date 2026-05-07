import numpy as np


def compute_control_metrics(
    lateral_deviations: list,
    heading_errors: list,
    steering_angles: list,
) -> dict:
    """Compute control-quality evaluation metrics.

    Args:
        lateral_deviations: List of lateral deviation values.
        heading_errors: List of heading error values.
        steering_angles: List of steering angle values.

    Returns:
        Dict with keys: lateral_mean, lateral_std, heading_mean, steering_smoothness.
    """
    lateral_abs = np.abs(lateral_deviations)
    heading_abs = np.abs(heading_errors)
    steering_diff_abs = np.abs(np.diff(steering_angles))

    return {
        "lateral_mean": float(np.mean(lateral_abs)),
        "lateral_std": float(np.std(lateral_abs)),
        "heading_mean": float(np.mean(heading_abs)),
        "steering_smoothness": float(np.mean(steering_diff_abs)),
    }
