"""Phase 2 Step 2: Train student policy via Behavior Cloning.

Trains AdaptationModule + StudentPolicy to mimic teacher actions
using (obs_history, obs_current, teacher_action) dataset.

Usage:
    python applications/g1_locomotion/student/train_student.py \
        --data results/g1_flat_locomotion/teacher_distill_data.npz
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from applications.g1_locomotion.student.networks import (
    AdaptationModule,
    StudentPolicy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train G1 student policy (BC).")
    parser.add_argument("--data", type=str, required=True, help="Path to teacher_distill_data.npz")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints.")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    print(f"[INFO] Loading data from: {args.data}")
    data = np.load(args.data)
    obs_history = torch.from_numpy(data["obs_history"]).float()
    obs_current = torch.from_numpy(data["obs_current"]).float()
    teacher_actions = torch.from_numpy(data["teacher_actions"]).float()

    history_dim = obs_history.shape[-1]
    obs_dim = obs_current.shape[-1]
    action_dim = teacher_actions.shape[-1]
    num_samples = obs_history.shape[0]

    print(f"[INFO] Dataset: {num_samples} samples")
    print(f"  obs_history_dim={history_dim}, obs_dim={obs_dim}, action_dim={action_dim}")

    # Train/val split
    dataset = TensorDataset(obs_history, obs_current, teacher_actions)
    val_size = int(num_samples * args.val_ratio)
    train_size = num_samples - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create models
    device = torch.device(args.device)
    adaptation = AdaptationModule(input_dim=history_dim, latent_dim=args.latent_dim).to(device)
    policy = StudentPolicy(obs_dim=obs_dim, latent_dim=args.latent_dim, action_dim=action_dim).to(device)

    # Optimizer
    params = list(adaptation.parameters()) + list(policy.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss()

    # Output directory
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.data), "student")
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"[INFO] Training for up to {args.epochs} epochs (patience={args.patience})...")

    for epoch in range(args.epochs):
        # Train
        adaptation.train()
        policy.train()
        train_loss_sum = 0.0
        train_batches = 0

        for hist, obs, actions in train_loader:
            hist, obs, actions = hist.to(device), obs.to(device), actions.to(device)

            z = adaptation(hist)
            pred_actions = policy(obs, z)
            loss = criterion(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches

        # Validate
        adaptation.eval()
        policy.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for hist, obs, actions in val_loader:
                hist, obs, actions = hist.to(device), obs.to(device), actions.to(device)
                z = adaptation(hist)
                pred_actions = policy(obs, z)
                loss = criterion(pred_actions, actions)
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / val_batches

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "adaptation": adaptation.state_dict(),
                "policy": policy.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "history_dim": history_dim,
                "latent_dim": args.latent_dim,
            }, os.path.join(output_dir, "student_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.6f})")
                break

    # Save final
    torch.save({
        "adaptation": adaptation.state_dict(),
        "policy": policy.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "history_dim": history_dim,
        "latent_dim": args.latent_dim,
    }, os.path.join(output_dir, "student_final.pt"))

    print(f"[INFO] Training complete. Best val_loss={best_val_loss:.6f}")
    print(f"  Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
