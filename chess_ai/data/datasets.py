"""
PyTorch Dataset classes.

PolicyDataset  – loads (state, move_index) pairs from processed_games_*/
ValueDataset   – loads (state, value) pairs from processed_value/
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch
from torch.utils.data import Dataset


class PolicyDataset(Dataset):
    """
    Loads behaviour-cloning data produced by processors.process_pgn().

    Each .npz file in data_dir (recursively) should contain:
        state        – float32 array of shape (8, 8, 13)
        correct_move – SAN move string (e.g. 'Rd8') or UCI move string
        fen          – FEN string (required if correct_move is SAN)

    Alternatively, legacy files may contain:
        move_idx     – int scalar (index into move vocabulary)
    """

    def __init__(
        self,
        data_dir: str,
        moves_file: str = "data/moves0.json",
        max_samples: Optional[int] = None,
    ):
        self.samples: list[Path] = []
        for root, _, files in os.walk(data_dir):
            for fname in sorted(files):
                if fname.endswith(".npz"):
                    self.samples.append(Path(root) / fname)
        if max_samples:
            self.samples = self.samples[:max_samples]
        if len(self.samples) == 0:
            raise RuntimeError(f"No .npz files found under {data_dir!r}")

        with open(moves_file) as f:
            self.move_to_idx: dict[str, int] = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        data = np.load(self.samples[idx])
        state = torch.from_numpy(data["state"].astype(np.float32))

        if "move_idx" in data.files:
            move_idx = int(data["move_idx"])
        else:
            san = str(data["correct_move"])
            fen = str(data["fen"])
            board = chess.Board(fen)
            uci = board.parse_san(san).uci()
            move_idx = self.move_to_idx[uci]

        return state, move_idx


class ValueDataset(Dataset):
    """
    Loads (state, value) pairs produced by processors.process_value_data()
    or stockfish_play → process_kv pipeline.

    Each .npz file should contain:
        state  – float32 array of shape (8, 8, 13)
        value  – float32 scalar in [0, 1]  (white winning probability)
    """

    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.samples: list[Path] = []
        for root, _, files in os.walk(data_dir):
            for fname in sorted(files):
                if fname.endswith(".npz"):
                    self.samples.append(Path(root) / fname)
        if max_samples:
            self.samples = self.samples[:max_samples]
        if len(self.samples) == 0:
            raise RuntimeError(f"No .npz files found under {data_dir!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        data = np.load(self.samples[idx])
        state = torch.from_numpy(data["state"].astype(np.float32))
        value = torch.tensor(float(data["value"]), dtype=torch.float32)
        return state, value
