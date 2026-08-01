#!/usr/bin/env python3
"""
Server-side inference entry point for the Flask API.

Drop-in for the previous ChessModel/LSTM version: it keeps the same two names
(``load_model`` and ``predict``) and the same response shape, so app.py needs no
change beyond the checkpoint path. Internally it now runs the current
InferenceEngine — policy net as move generator, value net as judge.

Flask usage (unchanged):

    from inference import load_model, predict
    model = load_model('models/policy/policy_best.pth')
    model.eval()
    result = predict(fen, model)     # -> ("g1", "f3")  or  "" if no move
    return jsonify(result)

Paths can be overridden with environment variables so the server does not have
to match this repo's layout:

    CHESS_CONFIG        configs/default.yaml
    CHESS_POLICY_CKPT   models/policy/policy_best.pth
    CHESS_VALUE_CKPT    models/value/value_best.pth   ("" disables the value net)
    CHESS_VALUE_WEIGHT  0.3
    CHESS_MOVES_FILE    overrides data.moves_file from the config

CHESS_MOVES_FILE matters under a process manager: the move vocabulary path
inside default.yaml is relative ("data/moves0.json"), so it only resolves when
the working directory happens to be the repo root. Set it to an absolute path
and the app no longer cares where gunicorn/systemd starts it.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Union

import chess

from chess_ai.config import Config
from chess_ai.inference.engine import InferenceEngine

DEFAULT_CONFIG = os.environ.get("CHESS_CONFIG", "configs/default.yaml")
DEFAULT_POLICY = os.environ.get("CHESS_POLICY_CKPT", "models/policy/policy_best.pth")
DEFAULT_VALUE = os.environ.get("CHESS_VALUE_CKPT", "models/value/value_best.pth")
DEFAULT_VALUE_WEIGHT = float(os.environ.get("CHESS_VALUE_WEIGHT", "0.3"))


class Engine:
    """
    Thin wrapper around InferenceEngine.

    Exists so the Flask app's ``model.eval()`` call keeps working — the old
    object was an nn.Module. eval() is a no-op here; the underlying nets are
    already in eval mode after loading.
    """

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def eval(self) -> "Engine":
        return self

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        return self.engine.best_move(board)


def load_model(
    model_path: str = DEFAULT_POLICY,
    value_path: Optional[str] = DEFAULT_VALUE,
    config_path: str = DEFAULT_CONFIG,
    value_weight: float = DEFAULT_VALUE_WEIGHT,
) -> Engine:
    """Load the policy (and optionally value) nets and return a ready Engine."""
    cfg = Config.from_yaml(config_path)

    moves_file = os.environ.get("CHESS_MOVES_FILE")
    if moves_file:
        cfg.data.moves_file = moves_file

    value_ckpt = value_path or None
    if value_ckpt and not os.path.exists(value_ckpt):
        value_ckpt = None
    engine = InferenceEngine(
        cfg,
        model_path,
        value_ckpt=value_ckpt,
        value_weight=value_weight if value_ckpt else 0.0,
    )
    return Engine(engine)


def predict(
    fen: str, model: Engine
) -> Union[tuple[str, str], tuple[str, str, str], str]:
    """
    Return ``(from_square, to_square)`` for the engine's chosen move, or
    ``(from_square, to_square, promotion)`` when the move is a promotion —
    promotion being one of "q", "r", "b", "n".

    The third element is only present on promotions, so a front end that reads
    just [0] and [1] keeps working unchanged.

    Returns ``""`` when the position is invalid, already over, or has no legal
    move — matching what the previous implementation returned so the front end
    does not have to learn a new failure case.
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return ""

    if board.is_game_over():
        return ""

    move = model.best_move(board)
    if move is None:
        return ""

    frm = chess.SQUARE_NAMES[move.from_square]
    to = chess.SQUARE_NAMES[move.to_square]
    if move.promotion is not None:
        return (frm, to, chess.piece_symbol(move.promotion))
    return (frm, to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict the next chess move from a given board state."
    )
    parser.add_argument("fen", help="Full 6-field FEN string.")
    args = parser.parse_args()
    print(predict(args.fen, load_model()))
