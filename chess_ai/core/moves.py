"""
Move encoding — bidirectional, O(1) lookup.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import chess


class MoveEncoder:
    """
    Wraps a moves*.json file for O(1) encode/decode.

    The JSON is expected to be a flat dict of {"uci_or_san": index, ...}.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Move mapping not found: {path}")
        with open(path) as f:
            self._move_to_idx: dict[str, int] = json.load(f)
        # Pre-build reverse mapping once instead of scanning on every decode
        self._idx_to_move: dict[int, str] = {v: k for k, v in self._move_to_idx.items()}

    # ------------------------------------------------------------------

    def encode(self, move: str) -> Optional[int]:
        """Return the index for a move string, or None if unknown."""
        return self._move_to_idx.get(move)

    def decode(self, idx: int) -> Optional[str]:
        """Return the move string for an index, or None if out of range."""
        return self._idx_to_move.get(idx)

    def encode_chess_move(self, move: chess.Move) -> Optional[int]:
        """Encode a chess.Move object using its UCI string."""
        return self.encode(move.uci())

    def legal_mask(self, board: chess.Board) -> list[int]:
        """Return list of valid indices for the current board's legal moves."""
        indices = []
        for move in board.legal_moves:
            idx = self.encode(move.uci())
            if idx is not None:
                indices.append(idx)
        return indices

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._move_to_idx)

    def __contains__(self, move: str) -> bool:
        return move in self._move_to_idx

    @property
    def num_moves(self) -> int:
        return len(self._move_to_idx)
