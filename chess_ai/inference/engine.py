"""
Inference engine — given a board position, return the best legal move.

Supports:
  - Policy-only (greedy or temperature-sampled)
  - Policy as move generator + Value net as judge (1-ply search)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess
import torch
import torch.nn.functional as F

from chess_ai.config import Config
from chess_ai.core.board import board_to_tensor
from chess_ai.core.moves import MoveEncoder
from chess_ai.models.nets import ValueNet, load_policy, load_value


@dataclass
class Candidate:
    """One scored move.

    policy_prob : the policy's prior probability for this move
    value       : value net's score for the resulting position, from the
                  moving side's perspective (None if it wasn't searched)
    score       : what the move was actually ranked by
    """
    move: chess.Move
    policy_prob: float
    value: Optional[float]
    score: float


class InferenceEngine:
    """
    Pick the best move from a FEN string or chess.Board.

    Parameters
    ----------
    cfg              : Config object
    policy_ckpt      : Path to PolicyNet checkpoint
    value_ckpt       : Path to ValueNet checkpoint (optional, enables search)
    temperature      : Softmax temperature (1.0 = greedy-ish, >1 = more random)
    top_k            : Only consider top-k policy moves (0 = all legal)
    value_weight     : How much the value net overrides the policy's preference
                       (0 = policy only, 1 = pure value argmax)
    value_top_k      : How many policy candidates the value net evaluates
    """

    def __init__(
        self,
        cfg: Config,
        policy_ckpt: str,
        value_ckpt: Optional[str] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        value_weight: float = 0.0,
        value_top_k: int = 5,
    ):
        self.cfg = cfg
        self.device = cfg.resolve_device()
        self.encoder = MoveEncoder(cfg.data.moves_file)
        self.temperature = temperature
        self.top_k = top_k
        self.value_weight = value_weight
        self.value_top_k = value_top_k

        self.policy = load_policy(policy_ckpt, cfg.model, self.device)
        self.value_net: Optional[ValueNet] = None
        if value_ckpt:
            self.value_net = load_value(value_ckpt, cfg.model, self.device)

    # ------------------------------------------------------------------

    def move_scores(
        self, position: str | chess.Board
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return ``(probs, raw_logits)`` — the policy's move distribution over
        the full vocabulary, after illegal-move masking, temperature and top-k.

        ``raw_logits`` are the untouched network outputs (useful for
        inspection). This is the policy prior only; the value net does not
        enter here — see :meth:`candidate_scores` for that.
        Returns ``(None, None)`` when the position has no usable moves.
        """
        board = chess.Board(position) if isinstance(position, str) else position

        if board.is_game_over():
            return None, None

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, None

        state_t = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
        state_t = state_t.to(self.device)

        # Policy logits
        with torch.no_grad():
            raw_logits = self.policy(state_t).squeeze(0)  # (num_moves,)

        # Mask illegal moves
        legal_idxs = self.encoder.legal_mask(board)
        if not legal_idxs:
            return None, None

        mask = torch.full((self.encoder.num_moves,), float("-inf"), device=self.device)
        mask[legal_idxs] = 0.0
        logits = raw_logits + mask

        # Temperature + top-k filtering
        logits = logits / self.temperature
        if self.top_k > 0:
            topk_vals, _ = torch.topk(logits[legal_idxs], min(self.top_k, len(legal_idxs)))
            threshold = topk_vals[-1]
            logits[logits < threshold] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return probs, raw_logits

    def candidate_scores(self, position: str | chess.Board) -> list[Candidate]:
        """
        Rank moves by 1-ply search: the policy proposes candidates, the value
        net scores the position each one leads to, and the two are combined.

        The policy acts as the move generator (a cheap prior over what's worth
        looking at) and the value net acts as the judge. With no value net —
        or ``value_weight == 0`` — this degenerates to ranking by policy alone.

        Returns candidates sorted best-first; empty if the game is over.
        """
        board = chess.Board(position) if isinstance(position, str) else position

        probs, _ = self.move_scores(board)
        if probs is None:
            return []

        # Policy prior for every legal move that exists in the vocabulary.
        scored: list[tuple[chess.Move, int, float]] = []
        for move in board.legal_moves:
            idx = self.encoder.encode_chess_move(move)
            if idx is not None:
                scored.append((move, idx, probs[idx].item()))
        if not scored:
            return []
        scored.sort(key=lambda s: s[2], reverse=True)

        # Policy only — nothing to search with.
        if self.value_net is None or self.value_weight <= 0:
            return [
                Candidate(move=m, policy_prob=p, value=None, score=p)
                for m, _, p in scored
            ]

        # Terminal check first, over *every* legal move. Pushing a move and
        # asking python-chess for the result costs microseconds next to a
        # forward pass, so there is no reason to let a mate hide in the tail
        # just because the policy rated it poorly.
        mate: Optional[tuple[chess.Move, float]] = None
        draws: set[chess.Move] = set()
        for move, _, prob in scored:
            board.push(move)
            if board.is_checkmate():
                if mate is None:      # `scored` is policy-ordered, so this is
                    mate = (move, prob)   # the likeliest of any mates available
            elif board.is_game_over():
                draws.add(move)
            board.pop()

        # A mate in one ends the game — never trade it for an estimate.
        if mate is not None:
            mate_move, mate_prob = mate
            out = [Candidate(move=mate_move, policy_prob=mate_prob,
                             value=1.0, score=float("inf"))]
            rest = [s for s in scored if s[0] != mate_move]
            for rank, (move, _, prob) in enumerate(rest):
                out.append(Candidate(move=move, policy_prob=prob,
                                     value=None, score=-1.0 - rank))
            return out

        # The value net only judges the most promising branches; the long tail
        # keeps its policy ranking below them.
        k = min(self.value_top_k, len(scored)) if self.value_top_k > 0 else len(scored)
        head, tail = scored[:k], scored[k:]

        mover_is_white = board.turn == chess.WHITE
        states: list[torch.Tensor] = []
        for move, _, _ in head:
            board.push(move)
            states.append(torch.from_numpy(board_to_tensor(board)))
            board.pop()

        batch = torch.stack(states).to(self.device)
        with torch.no_grad():
            vals = self.value_net(batch)   # (k,) — P(white wins)

        # The value net is white-centric; re-express from the mover's view so
        # that higher is always better for whoever is about to move.
        if not mover_is_white:
            vals = 1.0 - vals

        # Blend on a common [0, 1] scale: rescale the head's policy priors so
        # both terms span the same range instead of comparing a peaked softmax
        # against a value.
        head_probs = torch.tensor([p for _, _, p in head], device=self.device)
        span = head_probs.max() - head_probs.min()
        norm_probs = (head_probs - head_probs.min()) / span if span > 0 \
            else torch.full_like(head_probs, 0.5)

        out: list[Candidate] = []
        for i, (move, _, prob) in enumerate(head):
            # A forced draw is a known outcome; don't let the net guess at it.
            value = 0.5 if move in draws else vals[i].item()
            score = (1 - self.value_weight) * norm_probs[i].item() + self.value_weight * value
            out.append(Candidate(move=move, policy_prob=prob, value=value, score=score))
        out.sort(key=lambda c: c.score, reverse=True)

        # Unsearched moves rank below every searched one, in policy order.
        worst = min((c.score for c in out), default=0.0)
        for rank, (move, _, prob) in enumerate(tail):
            out.append(Candidate(
                move=move, policy_prob=prob, value=None,
                score=worst - 1.0 - rank,
            ))
        return out

    def best_move(self, position: str | chess.Board) -> Optional[chess.Move]:
        """
        Return the best legal chess.Move for the given position, or None if
        the game is already over.
        """
        board = chess.Board(position) if isinstance(position, str) else position

        candidates = self.candidate_scores(board)
        if not candidates:
            return None

        move = candidates[0].move
        return move if move in board.legal_moves else None
