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

    discount = cfg.rl.discount
    n_games = 0

    try:
        while max_games is None or n_games < max_games:
            board = chess.Board()
            history: list[tuple[str, chess.Color]] = []

            while not board.is_game_over():
                if len(history) >= cfg.rl.max_moves * 2:
                    break

                if board.turn == chess.BLACK and inference:
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

            # Determine outcome
            res = board.result()
            if res == "1-0":
                final = 1.0
            elif res == "0-1":
                final = -1.0
            else:
                final = 0.0

            # Compute discounted values and write to file
            out_file = out_dir / f"stockfish_{n_games:06d}.txt"
            with open(out_file, "w") as f:
                # Labels stay white-centric (+1 = good for white) to match
                # ValueNet's contract and the perspective flip in
                # InferenceEngine.candidate_scores. Do NOT negate by side to move.
                for i, (fen, _color) in enumerate(reversed(history)):
                    steps_from_end = i
                    val = final * (discount ** steps_from_end)
                    f.write(f"{fen}:{val:.4f}\n")

            n_games += 1
            if n_games % 100 == 0:
                print(f"  {n_games} games collected …")
    finally:
        stockfish.quit()

    print(f"Collected {n_games} games → {out_dir}")


if __name__ == "__main__":
    main()
