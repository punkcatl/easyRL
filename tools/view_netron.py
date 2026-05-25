#!/usr/bin/env python3
"""Export a project model to ONNX and open in Netron."""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AGENTS = {
    "1": ("DQN QNetwork", "algorithms.dqn.agent", "QNetwork",
          ["state_dim", "action_dim"], {"hidden_dim": 128}),
    "2": ("Policy Gradient PolicyNetwork", "algorithms.policy_gradient.agent", "PolicyNetwork",
          ["state_dim", "action_dim"], {"hidden_dim": 128}),
    "3": ("PPO Actor", "algorithms.ppo.agent", "Actor",
          ["state_dim", "action_dim"], {"hidden_dim": 128}),
    "4": ("PPO Critic", "algorithms.ppo.agent", "Critic",
          ["state_dim"], {"hidden_dim": 128}),
    "5": ("SAC GaussianPolicy", "algorithms.sac.agent", "GaussianPolicy",
          ["state_dim", "action_dim"], {"hidden_dim": 256}),
    "6": ("SAC QNetwork", "algorithms.sac.agent", "QNetwork",
          ["state_dim", "action_dim"], {"hidden_dim": 256}),
}


def get_model_from_agent():
    print("\n=== Project Agents ===")
    for key, (name, *_) in AGENTS.items():
        print(f"  [{key}] {name}")

    choice = input("\nSelect agent: ").strip()
    if choice not in AGENTS:
        print("Invalid choice.")
        sys.exit(1)

    name, module_path, class_name, required_dims, defaults = AGENTS[choice]
    print(f"\n→ {name}")

    kwargs = dict(defaults)
    for dim in required_dims:
        val = input(f"  {dim} (e.g. 4): ").strip()
        kwargs[dim] = int(val)

    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    model = cls(**kwargs)

    import torch
    if "state_dim" in kwargs and "action_dim" in kwargs:
        dummy = torch.randn(1, kwargs["state_dim"])
    elif "state_dim" in kwargs:
        dummy = torch.randn(1, kwargs["state_dim"])
    else:
        dummy = torch.randn(1, kwargs[required_dims[0]])

    return model, dummy, name


def get_model_from_file():
    path = input("\nModel file path (.pt/.pth): ").strip()
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    import torch
    import torch.nn as nn

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        model.eval()
        first_param = next(model.parameters())
        input_dim = first_param.shape[1] if first_param.dim() >= 2 else first_param.shape[0]
        dummy = torch.randn(1, input_dim)
        return model, dummy, os.path.basename(path)

    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        first_key = keys[0]
        first_weight = checkpoint[first_key]
        input_dim = first_weight.shape[1] if first_weight.dim() >= 2 else first_weight.shape[0]

        print(f"\n  state_dict detected (input_dim inferred: {input_dim})")
        print(f"  Keys: {keys[:5]}{'...' if len(keys) > 5 else ''}")
        print("\n  Need model class to load state_dict.")
        print("  Load as project agent instead? (y/n): ", end="")
        if input().strip().lower() == "y":
            model, dummy, name = get_model_from_agent()
            model.load_state_dict(checkpoint)
            return model, dummy, name
        else:
            print("  Cannot visualize state_dict without model class.")
            sys.exit(1)

    print("Unrecognized file format.")
    sys.exit(1)


def export_and_view(model, dummy_input, name):
    import torch
    import netron

    model.eval()

    onnx_path = os.path.join(tempfile.gettempdir(), f"{name.replace(' ', '_')}.onnx")

    try:
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    except Exception as e:
        print(f"\nONNX export failed: {e}")
        print("Trying with simplified export...")
        torch.onnx.export(model, dummy_input, onnx_path)

    print(f"\n✓ Exported to: {onnx_path}")
    print("  Opening in Netron (browser)...")
    netron.start(onnx_path)


def main():
    print("╔══════════════════════════════════╗")
    print("║    Netron Model Viewer           ║")
    print("╚══════════════════════════════════╝")
    print("\nSource:")
    print("  [1] Project agent")
    print("  [2] Model file (.pt/.pth)")
    print("  [3] ONNX file (.onnx)")

    choice = input("\nSelect: ").strip()

    if choice == "1":
        model, dummy, name = get_model_from_agent()
        export_and_view(model, dummy, name)
    elif choice == "2":
        model, dummy, name = get_model_from_file()
        export_and_view(model, dummy, name)
    elif choice == "3":
        path = input("\nONNX file path: ").strip()
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            sys.exit(1)
        import netron
        print(f"\n  Opening {path} in Netron (browser)...")
        netron.start(path)
    else:
        print("Invalid choice.")
        sys.exit(1)


if __name__ == "__main__":
    main()
