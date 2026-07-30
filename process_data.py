#!/usr/bin/env python3
"""
Data processing entry point.

Usage
─────
    # Parse PGN files → processed .npz tensors for policy training
    python process_data.py --mode pgn --config configs/default.yaml

    # Convert Stockfish value_training/ files → processed .npz for value training
    python process_data.py --mode value

    # Regenerate the UCI move vocabulary (data/moves0.json)
    python process_data.py --mode moves

    # Run Stockfish self-play to collect value training data
    python process_data.py --mode stockfish_play --games 5000
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Process chess training data.")
    parser.add_argument(
        "--mode",
        choices=["pgn", "value", "moves", "stockfish_play"],
        required=True,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--games", type=int, default=None, help="Max games (pgn/stockfish_play)")
    return parser.parse_args()


def main():
    args = parse_args()

    from chess_ai.config import Config
    cfg = Config.from_yaml(args.config)

    if args.mode == "pgn":
        from chess_ai.data.processors import process_pgn
        process_pgn(
            pgn_dir=cfg.data.raw_pgn_dir,
            output_dir=cfg.data.policy_data_dir,
            moves_file=cfg.data.moves_file,
            max_games=args.games,
        )

    elif args.mode == "value":
        from chess_ai.data.processors import process_value_data
        process_value_data(
            raw_dir=cfg.data.value_training_dir,
            output_dir=cfg.data.value_data_dir,
        )

    elif args.mode == "moves":
        from chess_ai.data.processors import generate_uci_moves
        generate_uci_moves(output_path=cfg.data.moves_file)

    elif args.mode == "stockfish_play":
        _run_stockfish_play(cfg, max_games=args.games)


def _eval_to_value(engine, board, limit_time: float) -> float | None:
    """
    White-centric value in [-1, 1] for a position that has no game result yet.

    Runs the logistic centipawn → win-probability curve so a truncated game's
    label lands on the same scale as a real outcome. Returns None if the engine
    gives us nothing usable.
    """
    import math

    import chess.engine

    try:
        info = engine.analyse(board, chess.engine.Limit(time=limit_time))
    except Exception:
        return None

    score = info.get("score")
    if score is None:
        return None
    score = score.white()

    if score.is_mate():
        mate = score.mate()
        return None if mate is None else (1.0 if mate > 0 else -1.0)

    cp = score.score()
    if cp is None:
        return None
    return 2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0


def _run_stockfish_play(cfg, max_games: int | None):
    """
    Self-play loop: policy vs Stockfish, collecting (FEN, value) pairs.
    Writes to cfg.data.value_training_dir.
    """
    import os
    import random
    from pathlib import Path

    import chess
    import chess.engine

    from chess_ai.config import Config
    from chess_ai.core.board import board_to_tensor
    from chess_ai.core.moves import MoveEncoder
    from chess_ai.inference.engine import InferenceEngine

    policy_ckpt = Path(cfg.paths.policy_out) / "policy_best.pth"
    if not policy_ckpt.exists():
        print(f"No policy checkpoint at {policy_ckpt} — using random moves.")
        policy_ckpt = None

    engine_path = cfg.paths.stockfish
    if not Path(engine_path).exists():
        print(f"Stockfish not found at {engine_path}. Aborting.")
        return

    out_dir = Path(cfg.data.value_training_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stockfish = chess.engine.SimpleEngine.popen_uci(engine_path)
    encoder = MoveEncoder(cfg.data.moves_file)

    inference = None
    if policy_ckpt:
        inference = InferenceEngine(cfg, str(policy_ckpt))

    n_games = 0
    n_unlabelled = 0

    try:
        while max_games is None or n_games < max_games:
            board = chess.Board()
            history: list[tuple[str, chess.Color]] = []
            # Alternate which side the policy plays. Stockfish beats it more or
            # less every game, so pinning the policy to one colour makes every
            # label lean the same way and the value net just learns "the side
            # Stockfish is playing wins".
            policy_color = chess.WHITE if n_games % 2 == 0 else chess.BLACK

            while not board.is_game_over():
                if len(history) >= cfg.rl.max_moves * 2:
                    break

                if board.turn == policy_color and inference:
                    move = inference.best_move(board)
                    if move is None or move not in board.legal_moves:
                        move = random.choice(list(board.legal_moves))
                else:
                    result = stockfish.play(
                        board, chess.engine.Limit(time=cfg.rl.stockfish_time)
                    )
                    move = result.move

                history.append((board.fen(), board.turn))
                board.push(move)

            # Label every position with Stockfish's eval *at that position*
            # rather than a discounted final outcome. A discounted outcome is
            # dominated by how far the position sits from the end of the game:
            # at 40 plies out, discount**40 crushes any label toward neutral, so
            # winning a queen in the opening looked the same as a quiet move and
            # the net learned "how close is this to a finished game" instead of
            # "who is better here".
            #
            # Labels stay white-centric (+1 = good for white) to match ValueNet's
            # contract and the perspective flip in candidate_scores. Do NOT
            # negate by side to move.
            rows = []
            for fen, _color in history:
                val = _eval_to_value(stockfish, chess.Board(fen), cfg.rl.stockfish_time)
                if val is not None:
                    rows.append(f"{fen}:{val:.4f}\n")

            if not rows:
                n_unlabelled += 1
                continue

            side = "w" if policy_color == chess.WHITE else "b"
            out_file = out_dir / f"stockfish_{n_games:06d}_{side}.txt"
            with open(out_file, "w") as f:
                f.writelines(rows)

            n_games += 1
            if n_games % 100 == 0:
                print(f"  {n_games} games collected …")
    finally:
        stockfish.quit()

    print(f"Collected {n_games} games → {out_dir}")
    if n_unlabelled:
        print(f"  ({n_unlabelled} truncated games dropped — no usable eval)")


if __name__ == "__main__":
    main()
