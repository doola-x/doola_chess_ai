"""
Inference engine — given a board position, return the best legal move.

Supports:
  - Policy-only (greedy or temperature-sampled)
  - Policy + Value reranking (lookahead)
"""
from __future__ import annotations

from typing import Optional

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chess_ai.config import Config
from chess_ai.core.board import board_to_tensor, fen_to_tensor
from chess_ai.core.moves import MoveEncoder
from chess_ai.models.nets import PolicyNet, ValueNet, load_policy, load_value


class InferenceEngine:
    """
    Pick the best move from a FEN string or chess.Board.

    Parameters
    ----------
    cfg              : Config object
    policy_ckpt      : Path to PolicyNet checkpoint
    value_ckpt       : Path to ValueNet checkpoint (optional, enables reranking)
    temperature      : Softmax temperature (1.0 = greedy-ish, >1 = more random)
    top_k            : Only consider top-k policy moves (0 = all legal)
    value_weight     : How much to blend value score into move selection (0 = policy only)
    """

    def __init__(
        self,
        cfg: Config,
        policy_ckpt: str,
        value_ckpt: Optional[str] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        value_weight: float = 0.0,
    ):
        self.cfg = cfg
        self.device = cfg.resolve_device()
        self.encoder = MoveEncoder(cfg.data.moves_file)
        self.temperature = temperature
        self.top_k = top_k
        self.value_weight = value_weight

        self.policy = load_policy(policy_ckpt, cfg.model, self.device)
        self.value_net: Optional[ValueNet] = None
        if value_ckpt:
            self.value_net = load_value(value_ckpt, cfg.model, self.device)

    # ------------------------------------------------------------------

    def best_move(self, position: str | chess.Board) -> Optional[chess.Move]:
        """
        Return the best legal chess.Move for the given position, or None if
        the game is already over.
        """
        if isinstance(position, str):
            board = chess.Board(position)
        else:
            board = position

        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        state_t = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
        state_t = state_t.to(self.device)

        # Policy logits
        with torch.no_grad():
            logits = self.policy(state_t).squeeze(0)  # (num_moves,)

        # Mask illegal moves
        legal_idxs = self.encoder.legal_mask(board)
        if not legal_idxs:
            return None

        mask = torch.full((self.encoder.num_moves,), float("-inf"), device=self.device)
        mask[legal_idxs] = 0.0
        logits = logits + mask

        # Temperature + top-k filtering
        logits = logits / self.temperature
        if self.top_k > 0:
            topk_vals, _ = torch.topk(logits[legal_idxs], min(self.top_k, len(legal_idxs)))
            threshold = topk_vals[-1]
            logits[logits < threshold] = float("-inf")

        probs = F.softmax(logits, dim=-1)

        # Optional: blend with value-net reranking
        if self.value_net is not None and self.value_weight > 0:
            probs = self._rerank_with_value(board, probs, legal_moves)

        # Greedy selection
        best_idx = probs.argmax().item()
        uci = self.encoder.decode(best_idx)
        if uci is None:
            return None
        move = chess.Move.from_uci(uci)
        return move if move in board.legal_moves else None

    def _rerank_with_value(
        self,
        board: chess.Board,
        policy_probs: torch.Tensor,
        legal_moves: list[chess.Move],
    ) -> torch.Tensor:
        """Score each legal move by value-net lookahead, blend with policy."""
        value_scores = torch.zeros(self.encoder.num_moves, device=self.device)
        states = []
        idxs = []

        for move in legal_moves:
            idx = self.encoder.encode_chess_move(move)
            if idx is None:
                continue
            board.push(move)
            states.append(torch.from_numpy(board_to_tensor(board)))
            idxs.append(idx)
            board.pop()

        if not states:
            return policy_probs

        batch = torch.stack(states).to(self.device)
        with torch.no_grad():
            vals = self.value_net(batch)  # (N,) in [0,1]

        # If board turn is black, flip (value net is white-centric)
        if board.turn == chess.BLACK:
            vals = 1.0 - vals

        for i, idx in enumerate(idxs):
            value_scores[idx] = vals[i]

        # Normalise value scores to a distribution
        valid_mask = value_scores > 0
        if valid_mask.any():
            value_scores = value_scores - value_scores[valid_mask].min()
            s = value_scores[valid_mask].sum()
            if s > 0:
                value_scores = value_scores / s

        blended = (1 - self.value_weight) * policy_probs + self.value_weight * value_scores
        return blended
