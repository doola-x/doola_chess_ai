"""
Behaviour cloning trainer (supervised policy learning from game data).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from chess_ai.config import Config
from chess_ai.data.datasets import PolicyDataset
from chess_ai.models.nets import PolicyNet


class PolicyTrainer:
    """
    Train PolicyNet via cross-entropy loss on (state, move_idx) pairs.

    Usage
    -----
        cfg = Config.from_yaml("configs/supervised.yaml")
        trainer = PolicyTrainer(cfg)
        trainer.fit()
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.resolve_device())
        Path(cfg.paths.policy_out).mkdir(parents=True, exist_ok=True)

    def fit(self) -> None:
        cfg = self.cfg
        tc = cfg.training

        # ── Data ──────────────────────────────────────────────────────
        full_dataset = PolicyDataset(cfg.data.policy_data_dir)
        val_size = max(1, int(len(full_dataset) * tc.val_split))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(
            train_ds, batch_size=tc.batch_size, shuffle=tc.shuffle,
            num_workers=tc.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=tc.batch_size, shuffle=False,
            num_workers=tc.num_workers,
        )
        print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Device: {self.device}")

        # ── Model ─────────────────────────────────────────────────────
        model = PolicyNet(cfg.model).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")

        for epoch in range(1, tc.num_epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            correct = 0
            for states, move_idxs in train_loader:
                states = states.to(self.device)
                move_idxs = move_idxs.to(self.device)

                optimizer.zero_grad()
                logits = model(states)
                loss = criterion(logits, move_idxs)
                loss.backward()
                if tc.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()

                train_loss += loss.item() * len(states)
                correct += (logits.argmax(1) == move_idxs).sum().item()

            train_loss /= len(train_ds)
            train_acc = correct / len(train_ds)

            # Validate
            val_loss, val_acc = self._evaluate(model, val_loader, criterion)

            print(
                f"Epoch {epoch:3d}/{tc.num_epochs}  "
                f"train_loss={train_loss:.4f} acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
            )

            # Checkpoints
            if epoch % tc.save_every == 0:
                self._save(model, epoch, f"policy_epoch_{epoch}.pth")

            if tc.keep_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save(model, epoch, "policy_best.pth")

        print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    def _evaluate(
        self,
        model: PolicyNet,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        n = 0
        with torch.no_grad():
            for states, move_idxs in loader:
                states = states.to(self.device)
                move_idxs = move_idxs.to(self.device)
                logits = model(states)
                total_loss += criterion(logits, move_idxs).item() * len(states)
                correct += (logits.argmax(1) == move_idxs).sum().item()
                n += len(states)
        return total_loss / n, correct / n

    def _save(self, model: PolicyNet, epoch: int, name: str) -> None:
        path = Path(self.cfg.paths.policy_out) / name
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "config": self.cfg.model},
            path,
        )
        print(f"  → saved {path}")
