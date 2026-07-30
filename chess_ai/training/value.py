"""
Value network trainer (supervised regression on Stockfish-labelled positions).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from chess_ai.config import Config
from chess_ai.data.datasets import ValueDataset
from chess_ai.models.nets import ValueNet


class ValueTrainer:
    """
    Train ValueNet via MSE loss on (state, value) pairs.

    Usage
    -----
        cfg = Config.from_yaml("configs/value.yaml")
        trainer = ValueTrainer(cfg)
        trainer.fit()
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.resolve_device())
        Path(cfg.paths.value_out).mkdir(parents=True, exist_ok=True)

    def fit(self) -> None:
        cfg = self.cfg
        tc = cfg.training

        # ── Data ──────────────────────────────────────────────────────
        full_dataset = ValueDataset(cfg.data.value_data_dir)
        val_size = max(1, int(len(full_dataset) * tc.val_split))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        pin = self.device.type == "cuda"
        # persistent_workers is only a legal kwarg when num_workers > 0.
        worker_kwargs = (
            {"persistent_workers": tc.persistent_workers} if tc.num_workers > 0 else {}
        )
        train_loader = DataLoader(
            train_ds, batch_size=tc.batch_size, shuffle=tc.shuffle,
            num_workers=tc.num_workers, pin_memory=pin, **worker_kwargs,
        )
        val_loader = DataLoader(
            val_ds, batch_size=tc.batch_size, shuffle=False,
            num_workers=tc.num_workers, **worker_kwargs,
        )
        print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Device: {self.device}")

        # ── Model ─────────────────────────────────────────────────────
        model = ValueNet(cfg.model).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay
        )
        criterion = nn.MSELoss()
        best_val_loss = float("inf")

        for epoch in range(1, tc.num_epochs + 1):
            model.train()
            train_loss = 0.0
            for states, values in train_loader:
                states = states.to(self.device)
                values = values.to(self.device)

                optimizer.zero_grad()
                preds = model(states)
                loss = criterion(preds, values)
                loss.backward()
                if tc.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()
                train_loss += loss.item() * len(states)

            train_loss /= len(train_ds)
            val_loss = self._evaluate(model, val_loader, criterion)

            print(
                f"Epoch {epoch:3d}/{tc.num_epochs}  "
                f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}"
            )

            if epoch % tc.save_every == 0:
                self._save(model, epoch, f"value_epoch_{epoch}.pth")

            if tc.keep_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save(model, epoch, "value_best.pth")

        print(f"Training complete. Best val loss: {best_val_loss:.5f}")

    def _evaluate(self, model: ValueNet, loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for states, values in loader:
                states = states.to(self.device)
                values = values.to(self.device)
                total += criterion(model(states), values).item() * len(states)
                n += len(states)
        return total / n

    def _save(self, model: ValueNet, epoch: int, name: str) -> None:
        path = Path(self.cfg.paths.value_out) / name
        torch.save({"model": model.state_dict(), "epoch": epoch}, path)
        print(f"  → saved {path}")
