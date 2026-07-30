#!/usr/bin/env python3
"""
Play a game against the nets in the terminal.

Uses InferenceEngine directly, so you get the policy net as move generator and
the value net as judge — the same path run_server.py serves.

Usage
─────
    # play white against policy + value
    python play.py

    # play black, lean harder on the value net
    python play.py --color black --value-weight 0.6

    # policy only, and show what the engine considered each move
    python play.py --value-weight 0 --show

Enter moves in SAN (Nf3, exd5, O-O) or UCI (g1f3). "quit" exits,
"undo" takes back your last move, "fen" prints the position.
"""
import argparse

import chess

from chess_ai.config import Config
from chess_ai.inference.engine import InferenceEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Play against the trained nets.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--policy", default="models/policy/policy_best.pth",
                        help="Policy checkpoint.")
    parser.add_argument("--value", default="models/value/value_best.pth",
                        help="Value checkpoint (pass '' to disable).")
    parser.add_argument("--color", choices=("white", "black"), default="white",
                        help="Which side you play.")
    parser.add_argument("--value-weight", type=float, default=0.3,
                        help="How much the value net overrides the policy (0-1).")
    parser.add_argument("--value-top-k", type=int, default=8,
                        help="How many policy candidates the value net scores.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Policy softmax temperature (lower = greedier).")
    parser.add_argument("--show", action="store_true",
                        help="Print the engine's top candidates each move.")
    return parser.parse_args()


def show_candidates(engine: InferenceEngine, board: chess.Board, n: int = 5) -> None:
    cands = engine.candidate_scores(board)[:n]
    if not cands:
        return
    print(f"  {'move':<7} {'policy':>8} {'value':>8} {'score':>8}")
    for c in cands:
        val = "—" if c.value is None else f"{c.value:.4f}"
        print(f"  {board.san(c.move):<7} {c.policy_prob:>8.4f} {val:>8} {c.score:>8.4f}")


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    engine = InferenceEngine(
        cfg,
        args.policy,
        value_ckpt=args.value or None,
        temperature=args.temperature,
        value_weight=args.value_weight if args.value else 0.0,
        value_top_k=args.value_top_k,
    )

    board = chess.Board()
    user_is_white = args.color == "white"
    print(f"\nYou are {args.color}. value_weight={args.value_weight if args.value else 0.0}")
    print("Moves in SAN or UCI. Commands: quit, undo, fen\n")

    while not board.is_game_over():
        print(board.unicode(borders=True, empty_square="."))
        print()

        if board.turn == (chess.WHITE if user_is_white else chess.BLACK):
            try:
                raw = input("your move > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return
            if not raw:
                continue
            if raw.lower() == "quit":
                print("bye")
                return
            if raw.lower() == "fen":
                print(board.fen(), "\n")
                continue
            if raw.lower() == "undo":
                # pop the engine reply and your move, so it's your turn again
                for _ in range(2):
                    if board.move_stack:
                        board.pop()
                print()
                continue

            move = None
            for parse in (board.parse_san, board.parse_uci):
                try:
                    move = parse(raw)
                    break
                except ValueError:
                    continue
            if move is None or move not in board.legal_moves:
                legal = ", ".join(board.san(m) for m in list(board.legal_moves)[:12])
                print(f"  illegal: {raw!r}. legal moves include: {legal} …\n")
                continue
            board.push(move)
        else:
            if args.show:
                show_candidates(engine, board)
            move = engine.best_move(board)
            if move is None:
                print("engine has no move.")
                break
            print(f"engine plays: {board.san(move)}\n")
            board.push(move)

    print(board.unicode(borders=True, empty_square="."))
    print(f"\nGame over: {board.result()} ({_reason(board)})")


def _reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_fifty_moves():
        return "fifty-move rule"
    if board.is_repetition():
        return "repetition"
    return "game over"


if __name__ == "__main__":
    main()
