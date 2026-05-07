import os
import tempfile

import pytest

from utils.logger import Logger
from utils.metrics import compute_control_metrics


class TestLogger:
    def test_logger_records_scalar(self):
        """Log two values, verify get_data returns them."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = Logger(log_dir=tmp_dir, use_tensorboard=False)
            logger.log("reward", 1.0, step=0)
            logger.log("reward", 2.5, step=1)

            data = logger.get_data("reward")
            assert data == [(0, 1.0), (1, 2.5)]
            logger.close()

    def test_logger_saves_csv(self):
        """Log, save, verify CSV file exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = Logger(log_dir=tmp_dir, use_tensorboard=False)
            logger.log("loss", 0.5, step=0)
            logger.log("loss", 0.3, step=1)
            logger.save()

            csv_path = os.path.join(tmp_dir, "loss.csv")
            assert os.path.exists(csv_path)

            with open(csv_path, "r") as f:
                lines = f.readlines()
            # Header + 2 data lines
            assert len(lines) == 3
            assert "step" in lines[0]
            assert "value" in lines[0]
            logger.close()


class TestMetrics:
    def test_compute_control_metrics(self):
        """Verify all metric keys present and lateral_mean matches expected."""
        lateral_deviations = [0.1, -0.2, 0.3, -0.1]
        heading_errors = [0.05, -0.1, 0.15, -0.05]
        steering_angles = [0.0, 0.1, 0.05, 0.2]

        metrics = compute_control_metrics(
            lateral_deviations=lateral_deviations,
            heading_errors=heading_errors,
            steering_angles=steering_angles,
        )

        # Check all keys present
        assert "lateral_mean" in metrics
        assert "lateral_std" in metrics
        assert "heading_mean" in metrics
        assert "steering_smoothness" in metrics

        # lateral_mean = mean of abs values = (0.1 + 0.2 + 0.3 + 0.1) / 4 = 0.175
        assert metrics["lateral_mean"] == pytest.approx(0.175)

        # heading_mean = mean of abs values = (0.05 + 0.1 + 0.15 + 0.05) / 4 = 0.0875
        assert metrics["heading_mean"] == pytest.approx(0.0875)

        # steering_smoothness = mean of abs of np.diff([0.0, 0.1, 0.05, 0.2])
        # diff = [0.1, -0.05, 0.15], abs = [0.1, 0.05, 0.15], mean = 0.1
        assert metrics["steering_smoothness"] == pytest.approx(0.1)
