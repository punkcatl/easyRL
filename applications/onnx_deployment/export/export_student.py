"""Export Sim-to-Real Student model (Adaptation Module + Base Policy) to ONNX."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import onnx

from config import config


class AdaptationModuleExport(nn.Module):
    """Adaptation module: maps observation history to latent environment embedding."""

    def __init__(self, input_dim=1350, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, obs_history):
        return self.net(obs_history)


class BasePolicyExport(nn.Module):
    """Base policy: maps (obs, latent_z) to actions."""

    def __init__(self, input_dim=43, action_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs_and_z):
        return self.net(obs_and_z)


def export_student(model_weights_path: str = None, output_dir: str = None):
    """Export Student as two separate ONNX files for independent optimization.

    Args:
        model_weights_path: path to student .pth checkpoint
        output_dir: directory for .onnx outputs
    """
    results_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    adaptation = AdaptationModuleExport()
    base_policy = BasePolicyExport()

    if model_weights_path:
        import warnings
        checkpoint = torch.load(model_weights_path, map_location="cpu", weights_only=True)
        if "adaptation" in checkpoint:
            missing, unexpected = adaptation.load_state_dict(checkpoint["adaptation"], strict=False)
            if missing or unexpected:
                warnings.warn(
                    f"AdaptationModule key mismatch — missing: {missing}, unexpected: {unexpected}. "
                    "Model may be partially or randomly initialized."
                )
        else:
            warnings.warn("Key 'adaptation' not found in checkpoint. AdaptationModule is randomly initialized.")
        if "base_policy" in checkpoint:
            missing, unexpected = base_policy.load_state_dict(checkpoint["base_policy"], strict=False)
            if missing or unexpected:
                warnings.warn(
                    f"BasePolicyExport key mismatch — missing: {missing}, unexpected: {unexpected}. "
                    "Model may be partially or randomly initialized."
                )
        else:
            warnings.warn("Key 'base_policy' not found in checkpoint. BasePolicyExport is randomly initialized.")

    adaptation.eval()
    base_policy.eval()

    # Save PyTorch weights alongside ONNX for verification
    torch.save(adaptation.state_dict(), str(results_dir / "adaptation_module.pth"))
    torch.save(base_policy.state_dict(), str(results_dir / "base_policy.pth"))

    # Export Adaptation Module
    adapt_path = str(results_dir / "adaptation_module.onnx")
    dummy_history = torch.randn(1, 1350)
    torch.onnx.export(
        adaptation,
        dummy_history,
        adapt_path,
        input_names=["obs_history"],
        output_names=["latent_z"],
        dynamic_axes={
            "obs_history": {0: "batch_size"},
            "latent_z": {0: "batch_size"},
        },
        opset_version=config["opset_version"],
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(adapt_path))
    print(f"Exported Adaptation Module to: {adapt_path}")
    print(f"  Input:  obs_history (batch, 1350)")
    print(f"  Output: latent_z   (batch, 16)")

    # Export Base Policy
    policy_path = str(results_dir / "base_policy.onnx")
    dummy_input = torch.randn(1, 43)
    torch.onnx.export(
        base_policy,
        dummy_input,
        policy_path,
        input_names=["obs_and_z"],
        output_names=["action"],
        dynamic_axes={
            "obs_and_z": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        opset_version=config["opset_version"],
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(policy_path))
    print(f"Exported Base Policy to: {policy_path}")
    print(f"  Input:  obs_and_z (batch, 43)")
    print(f"  Output: action    (batch, 8)")

    return adapt_path, policy_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Student model to ONNX")
    parser.add_argument("--weights", default=None, help="Path to student .pth checkpoint")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()
    export_student(args.weights, args.output_dir)
