"""Offline supervised distillation: train G1 student to mimic teacher actions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tyro
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset

from src.distill.student_network import StudentPolicy


@dataclass
class DistillArgs:
    dataset_path: str = "applications/g1_mjlab/results/distill_dataset.npz"
    output_path: str = "applications/g1_mjlab/results/student_final.pt"
    obs_dim: int = 84
    action_dim: int = 12
    history_length: int = 20
    latent_dim: int = 32
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 200
    val_ratio: float = 0.1
    early_stop_patience: int = 15


def main():
    args = tyro.cli(DistillArgs)

    print(f"Loading dataset: {args.dataset_path}")
    data = np.load(args.dataset_path)
    obs_history = torch.FloatTensor(data["obs_history"])
    actions = torch.FloatTensor(data["actions"])
    print(f"  Dataset: {obs_history.shape[0]} samples")

    n = obs_history.shape[0]
    n_val = int(n * args.val_ratio)
    n_train = n - n_val
    perm = torch.randperm(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = TensorDataset(obs_history[train_idx], actions[train_idx])
    val_ds = TensorDataset(obs_history[val_idx], actions[val_idx])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print(f"  Train: {n_train}, Val: {n_val}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentPolicy(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        history_length=args.history_length,
        latent_dim=args.latent_dim,
    ).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for hist_batch, act_batch in train_dl:
            hist_batch = hist_batch.to(device)
            act_batch = act_batch.to(device)

            pred = model(hist_batch)
            loss = criterion(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * hist_batch.shape[0]

        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hist_batch, act_batch in val_dl:
                hist_batch = hist_batch.to(device)
                act_batch = act_batch.to(device)
                pred = model(hist_batch)
                loss = criterion(pred, act_batch)
                val_loss += loss.item() * hist_batch.shape[0]
        val_loss /= n_val

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} | "
                  f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.output_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={args.early_stop_patience})")
                break

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
