"""
Chess environment for reinforcement learning.

Wraps python-chess and optionally Stockfish.  Provides:
  - step()       — apply a move, return (next_state, reward, done)
  - reset()      — start a new game
  - legal_moves  — list of chess.Move objects

Reward shaping is configurable via RLConfig.
"""
from __future__ import annotations

from typing import Optional

import chess
import chess.engine
import numpy as np

from chess_ai.config import RLConfig
from chess_ai.core.board import board_to_tensor

# Material values in centipawns (normalised later)
MATERIAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]


class ChessEnv:
    """
    Episodic chess environment.

    Parameters
    ----------
    cfg         : RLConfig with reward weights and Stockfish settings
    stockfish   : Path to Stockfish binary (optional).  When provided,
                  the opponent plays Stockfish moves.
    play_as     : chess.WHITE or chess.BLACK (agent's colour)
    """

    def __init__(
        self,
        cfg: RLConfig,
        stockfish: Optional[str] = None,
        play_as: chess.Color = chess.WHITE,
    ):
        self.cfg = cfg
        self.play_as = play_as
        self._engine: Optional[chess.engine.SimpleEngine] = None
        if stockfish:
            self._engine = chess.engine.SimpleEngine.popen_uci(stockfish)
        self.board = chess.Board()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, fen: Optional[str] = None) -> np.ndarray:
        """Start a new game.  Returns initial board tensor."""
        self.board = chess.Board(fen) if fen else chess.Board()
        return board_to_tensor(self.board)

    def step(self, move: chess.Move) -> tuple[np.ndarray, float, bool]:
        """
        Apply agent move, then (if engine is attached) apply opponent move.

        Returns
        -------
        state  : (8, 8, 13) tensor after both half-moves
        reward : shaped reward from agent's perspective
        done   : True if the game is over
        """
        if move not in self.board.legal_moves:
            # Illegal move penalty — training should avoid this via masking
            return board_to_tensor(self.board), -1.0, True

        self.board.push(move)

        if self.board.is_game_over():
            reward = self._outcome_reward()
            return board_to_tensor(self.board), reward, True

        # Shaped reward after agent's move (before opponent replies)
        reward = self._shaped_reward()

        # Opponent move
        if self._engine is not None and not self.board.is_game_over():
            result = self._engine.play(
                self.board,
                chess.engine.Limit(time=self.cfg.stockfish_time),
            )
            if result.move:
                self.board.push(result.move)
        elif not self.board.is_game_over():
            # No engine: random opponent
            import random
            self.board.push(random.choice(list(self.board.legal_moves)))

        done = self.board.is_game_over()
        if done:
            reward += self._outcome_reward()

        return board_to_tensor(self.board), reward, done

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    @property
    def current_state(self) -> np.ndarray:
        return board_to_tensor(self.board)

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _outcome_reward(self) -> float:
        result = self.board.result()
        if result == "1-0":
            return 1.0 if self.play_as == chess.WHITE else -1.0
        if result == "0-1":
            return -1.0 if self.play_as == chess.WHITE else 1.0
        return 0.0  # draw

    def _shaped_reward(self) -> float:
        """Lightweight heuristic reward (configurable weights in RLConfig)."""
        reward = 0.0
        w = self.cfg

        # Material balance (from agent's perspective)
        mat = self._material_balance()
        reward += w.reward_material * mat

        # Center control
        center = self._center_control()
        reward += w.reward_center * center

        # King safety (negative = opponent attacking our king)
        ks = self._king_safety()
        reward += w.reward_king_safety * ks

        return reward

    def _material_balance(self) -> float:
        white = sum(
            MATERIAL[p.piece_type]
            for p in self.board.piece_map().values()
            if p.color == chess.WHITE
        )
        black = sum(
            MATERIAL[p.piece_type]
            for p in self.board.piece_map().values()
            if p.color == chess.BLACK
        )
        diff = white - black
        if self.play_as == chess.BLACK:
            diff = -diff
        # Normalise to rough [-1, 1]
        return diff / 39.0

    def _center_control(self) -> float:
        agent_color = self.play_as
        opp_color = not agent_color
        agent_attacks = sum(
            1
            for sq in CENTER_SQUARES
            if self.board.is_attacked_by(agent_color, sq)
        )
        opp_attacks = sum(
            1
            for sq in CENTER_SQUARES
            if self.board.is_attacked_by(opp_color, sq)
        )
        return (agent_attacks - opp_attacks) / 4.0

    def _king_safety(self) -> float:
        agent_color = self.play_as
        opp_color = not agent_color
        our_king = self.board.king(agent_color)
        their_king = self.board.king(opp_color)
        if our_king is None or their_king is None:
            return 0.0
        threats_on_ours = len(self.board.attackers(opp_color, our_king))
        threats_on_theirs = len(self.board.attackers(agent_color, their_king))
        return (threats_on_theirs - threats_on_ours) / 8.0
