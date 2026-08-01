"""
Neural network architectures.

All models take input shape (B, 8, 8, 13). They permute internally to
(B, 13, 8, 8) for the conv layers, so callers don't need to worry about it.

Architectures
─────────────
PolicyNet   - outputs logits over the move vocabulary (for behaviour cloning
              or RL policy).  Uses softmax-compatible output (raw logits).
ValueNet    - outputs a scalar in [0, 1] representing white's winning
              probability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_ai.config import ModelConfig


class _ConvBlock(nn.Module):
    """Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNet(nn.Module):
    """
    Policy network for move prediction.

    Input:  (B, 8, 8, 13)
    Output: (B, num_moves)  — raw logits, apply softmax / log_softmax outside

    Architecture: ConvBlock → Flatten → FC(1024) → FC(512) → FC(num_moves)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        flat = cfg.conv_filters * 8 * 8

        self.conv = _ConvBlock(cfg.in_channels, cfg.conv_filters, cfg.conv_kernel)
        self.fc = nn.Sequential(
            nn.Linear(flat, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim2),
            nn.LayerNorm(cfg.hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim2, cfg.num_moves),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, 8, 13)  →  (B, 13, 8, 8)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)                         # (B, filters, 8, 8)
        x = x.flatten(start_dim=1)               # (B, filters*64)
        return self.fc(x)                         # (B, num_moves)


class ValueNet(nn.Module):
    """
    Value network: estimates white's winning probability given a board state.

    Input:  (B, 8, 8, 13)
    Output: (B,)  — scalar in [0, 1] via sigmoid

    Used for both supervised value training (Stockfish labels) and as the
    critic in actor-critic RL.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        flat = cfg.conv_filters * 8 * 8

        self.conv = _ConvBlock(cfg.in_channels, cfg.conv_filters, cfg.conv_kernel)
        self.fc = nn.Sequential(
            nn.Linear(flat, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim2),
            nn.LayerNorm(cfg.hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)   # (B,)


# ─── Convenience loaders ──────────────────────────────────────────────────────

def _torch_load(checkpoint: str, device: str):
    """
    Load a checkpoint, tolerating the torch version on the deploy box.

    Policy checkpoints embed a ModelConfig under "config", so weights_only=True
    needs that class allowlisted. The allowlist API moved twice:

        torch >= 2.5   torch.serialization.safe_globals      (context manager)
        torch >= 2.4   torch.serialization.add_safe_globals  (process-global)
        older          no allowlist at all — full unpickle

    The last case drops weights_only, which is only acceptable because these are
    our own checkpoints. Never point it at one you didn't train.
    """
    from chess_ai.config import ModelConfig as _ModelConfig

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is not None:
        with safe_globals([_ModelConfig]):
            return torch.load(checkpoint, map_location=device, weights_only=True)

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([_ModelConfig])
        return torch.load(checkpoint, map_location=device, weights_only=True)

    return torch.load(checkpoint, map_location=device)


def load_policy(checkpoint: str, cfg: ModelConfig, device: str = "cpu") -> PolicyNet:
    model = PolicyNet(cfg).to(device)
    state = _torch_load(checkpoint, device)
    # Accept both raw state_dicts and {"model": ...} checkpoint dicts
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_value(checkpoint: str, cfg: ModelConfig, device: str = "cpu") -> ValueNet:
    model = ValueNet(cfg).to(device)
    state = _torch_load(checkpoint, device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model
