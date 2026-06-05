"""Export discrete PPO policy network to ONNX format."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

from config import config


class PolicyNetDiscrete(nn.Module):
    """Discrete PPO policy for export (standalone, no training code dependency)."""

    def __init__(self, state_dim=25, hidden_dim=128, action_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


def export_ppo_discrete(model_weights_path: str = None, output_path: str = None):
    """Export discrete PPO policy to ONNX.

    Args:
        model_weights_path: path to .pth file (if None, exports randomly initialized model)
        output_path: path for .onnx output
    """
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = output_path or str(results_dir / "ppo_discrete.onnx")

    model = PolicyNetDiscrete()

    if model_weights_path:
        checkpoint = torch.load(model_weights_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("actor", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            import warnings
            warnings.warn(
                f"Checkpoint key mismatch — missing: {missing}, unexpected: {unexpected}. "
                "Model may be partially or randomly initialized."
            )

    model.eval()

    # Save PyTorch weights alongside ONNX for verification
    pth_path = output_path.rsplit(".onnx", 1)[0] + ".pth"
    torch.save(model.state_dict(), pth_path)

    dummy_input = torch.randn(1, 25)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["action_probs"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action_probs": {0: "batch_size"},
        },
        opset_version=config["opset_version"],
        do_constant_folding=True,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"Exported PPO discrete policy to: {output_path}")
    print(f"  Input:  observation  (batch, 25)")
    print(f"  Output: action_probs (batch, 5)")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export discrete PPO to ONNX")
    parser.add_argument("--weights", default=None, help="Path to .pth model weights")
    parser.add_argument("--output", default=None, help="Output .onnx path")
    args = parser.parse_args()
    export_ppo_discrete(args.weights, args.output)
