import csv
import os
from collections import defaultdict


class Logger:
    """Training logger that records scalars to CSV and optionally TensorBoard."""

    def __init__(self, log_dir: str, use_tensorboard: bool = False):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.data = defaultdict(list)
        self.writer = None

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, tag: str, value: float, step: int):
        """Append (step, value) to data[tag] and write to TensorBoard if enabled."""
        self.data[tag].append((step, value))
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def get_data(self, tag: str) -> list:
        """Return list of (step, value) tuples for the given tag."""
        return self.data[tag]

    def save(self):
        """Write each tag to a CSV file (columns: step, value) in log_dir."""
        os.makedirs(self.log_dir, exist_ok=True)
        for tag, records in self.data.items():
            csv_path = os.path.join(self.log_dir, f"{tag}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "value"])
                for step, value in records:
                    writer.writerow([step, value])

    def close(self):
        """Close TensorBoard writer if it exists."""
        if self.writer is not None:
            self.writer.close()
