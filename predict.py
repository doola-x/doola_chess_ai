#!/usr/bin/env python3
"""
Inspect model output for a single position — no server, no GUI.

Usage
─────
    # top-10 moves for the starting position
    python predict.py

    # a specific position
    python predict.py "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

    # how does the model rate one particular move? (UCI or SAN)
    python predict.py "<fen>" --move Bxf7+

    # every legal move, ranked, with the value net blended in
    python predict.py "<fen>" --all --value models/value/value_best.pth --value-weight 0.3
"""
import argparse

import chess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show the policy net's move distribution for one position."
    )
    parser.add_argument("fen", nargs="?", default=chess.STARTING_FEN,
                        help="FEN string (default: starting position).")
    parser.add_argument("--policy", default="models/policy/policy_best.pth",
                        help="Policy checkpoint path.")
    parser.add_argument("--value", default=None,
                        help="Value checkpoint path (optional, enables reranking).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--move", default=None,
                        help="Report rank/probability for this move (UCI or SAN).")
    parser.add_argument("--top-k-show", type=int, default=10, metavar="N",
                        help="How many moves to print (default: 10).")
    parser.add_argument("--all", action="store_true",
                        help="Print every legal move instead of just the top N.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature (lower = greedier).")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Restrict the distribution to top-k moves (0 = all legal).")
    parser.add_argument("--value-weight", type=float, default=0.0,
                        help="How much the value net overrides the policy (0-1).")
    parser.add_argument("--value-top-k", type=int, default=5,
                        help="How many policy candidates the value net searches.")
    return parser.parse_args()


def resolve_move(board: chess.Board, text: str) -> chess.Move:
    """Parse a move given in either UCI ('c4f7') or SAN ('Bxf7+') form."""
    try:
        return board.parse_san(text)
    except ValueError:
        pass
    try:
        return chess.Move.from_uci(text)
    except ValueError:
        raise SystemExit(f"Could not parse move {text!r} as UCI or SAN.")


def main():
    args = parse_args()
    from chess_ai.config import Config
    from chess_ai.inference.engine import InferenceEngine

    cfg = Config.from_yaml(args.config)
    board = chess.Board(args.fen)

    engine = InferenceEngine(
        cfg=cfg,
        policy_ckpt=args.policy,
        value_ckpt=args.value,
        temperature=args.temperature,
        top_k=args.top_k,
        value_weight=args.value_weight,
        value_top_k=args.value_top_k,
    )

    print(board)
    print(f"\nFEN   {board.fen()}")
    print(f"turn  {'white' if board.turn == chess.WHITE else 'black'}")
    print(f"model {args.policy}"
          + (f" + {args.value} (w={args.value_weight}, "
             f"searching top {args.value_top_k})" if args.value else ""))

    _, raw_logits = engine.move_scores(board)
    ranked = engine.candidate_scores(board)
    if not ranked:
        print(f"\nNo moves available — game is over ({board.result()}).")
        return

    searched = sum(1 for c in ranked if c.value is not None)
    limit = len(ranked) if args.all else min(args.top_k_show, len(ranked))
    print(f"\n{len(ranked)} legal moves in vocabulary"
          + (f", {searched} searched by the value net" if searched else "")
          + f" — showing {limit}\n")
    print(f"{'#':>3}  {'uci':6s} {'san':10s} {'policy':>8s}  {'value':>7s}  {'score':>7s}  {'logit':>8s}")
    print("-" * 60)
    for rank, c in enumerate(ranked[:limit], start=1):
        idx = engine.encoder.encode_chess_move(c.move)
        logit = raw_logits[idx].item()
        value = f"{c.value:.4f}" if c.value is not None else "   —   "
        print(f"{rank:>3}  {c.move.uci():6s} {board.san(c.move):10s} "
              f"{c.policy_prob:>8.4f}  {value:>7s}  {c.score:>7.4f}  {logit:>+8.3f}")

    if searched:
        print("\npolicy = prior over moves | value = P(win) for the side to move "
              "after that move | score = what it's ranked by")

    if args.move:
        target = resolve_move(board, args.move)
        hit = next(((r, c) for r, c in enumerate(ranked, start=1) if c.move == target), None)
        print()
        if hit is None:
            legal = target in board.legal_moves
            reason = "not in the move vocabulary" if legal else "not legal here"
            print(f"{args.move}: no score — {reason}.")
        else:
            rank, c = hit
            detail = f", value {c.value:.4f}" if c.value is not None else " (not searched)"
            print(f"{board.san(c.move)} ({c.move.uci()}): rank {rank}/{len(ranked)}, "
                  f"policy {c.policy_prob:.4f}{detail}")


if __name__ == "__main__":
    main()
