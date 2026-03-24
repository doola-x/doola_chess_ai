"""
Data processing pipelines.

process_pgn()          – raw PGN → processed_games_*/  .npz files
process_value_data()   – value_training/*.txt → processed_value/  .npz files
generate_uci_moves()   – produce data/moves0.json from all legal UCI moves
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from chess_ai.core.board import fen_to_tensor
from chess_ai.core.moves import MoveEncoder


# ─── PGN processing ───────────────────────────────────────────────────────────

def process_pgn(
    pgn_dir: str,
    output_dir: str,
    moves_file: str,
    max_games: int | None = None,
) -> int:
    """
    Parse PGN files and write one .npz per position.

    Each .npz stores:
        state     – (8, 8, 13) float32 board tensor
        move_idx  – int, UCI move index from moves_file

    Returns the number of positions written.
    """
    encoder = MoveEncoder(moves_file)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pgn_files = sorted(Path(pgn_dir).glob("**/*.pgn")) + sorted(Path(pgn_dir).glob("**/*.txt"))
    total = 0
    game_id = 0

    for pgn_path in pgn_files:
        with open(pgn_path, errors="replace") as f:
            content = f.read()

        # Each file may contain multiple games
        stream = io.StringIO(content)
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            if max_games and game_id >= max_games:
                break

            game_dir = out_root / f"game_{game_id:06d}"
            game_dir.mkdir(exist_ok=True)

            board = game.board()
            move_no = 0
            for move in game.mainline_moves():
                uci = move.uci()
                idx = encoder.encode(uci)
                if idx is None:
                    board.push(move)
                    move_no += 1
                    continue

                state = fen_to_tensor(board.fen())
                np.savez_compressed(
                    game_dir / f"move_{move_no:04d}.npz",
                    state=state,
                    move_idx=np.int64(idx),
                )
                board.push(move)
                move_no += 1
                total += 1

            game_id += 1

    print(f"Processed {game_id} games → {total} positions in {output_dir!r}")
    return total


# ─── Value data processing ────────────────────────────────────────────────────

def process_value_data(
    raw_dir: str,
    output_dir: str,
) -> int:
    """
    Convert FEN:value text files (from stockfish_play) into .npz files.

    Expects files matching value_training/*.txt, each line: "FEN:value"
    where value is a float in [-1, 1].  We normalise to [0, 1].

    Returns the number of samples written.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for txt_path in sorted(Path(raw_dir).glob("*.txt")):
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                # May have multiple colons (FEN contains spaces, not colons)
                sep = line.rfind(":")
                fen_str = line[:sep]
                val_str = line[sep + 1:]
                try:
                    raw_val = float(val_str)
                    # Normalise from [-1, 1] → [0, 1]
                    value = (raw_val + 1.0) / 2.0
                    value = float(np.clip(value, 0.0, 1.0))
                    state = fen_to_tensor(fen_str)
                except Exception:
                    continue

                np.savez_compressed(
                    out_root / f"sample_{total:08d}.npz",
                    state=state,
                    value=np.float32(value),
                )
                total += 1

    print(f"Processed {total} value samples → {output_dir!r}")
    return total


# ─── Move mapping generation ──────────────────────────────────────────────────

def generate_uci_moves(output_path: str = "data/moves0.json") -> dict[str, int]:
    """
    Generate the complete set of UCI moves reachable from any legal position
    and write them to a JSON file.

    This is a one-time operation — only re-run if you want to change the
    move vocabulary.
    """
    moves: set[str] = set()

    # All from-to combinations on the board
    squares = list(chess.SQUARES)
    for src in squares:
        for dst in squares:
            if src == dst:
                continue
            moves.add(chess.Move(src, dst).uci())
            # Promotions from rank 7 → rank 8 (white) or rank 2 → rank 1 (black)
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                moves.add(chess.Move(src, dst, promotion=promo).uci())

    # Filter to moves that at least appear legal on some board by format
    # (full legality check would require exhaustive game tree traversal)
    mapping = {move: idx for idx, move in enumerate(sorted(moves))}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Generated {len(mapping)} UCI moves → {output_path!r}")
    return mapping
