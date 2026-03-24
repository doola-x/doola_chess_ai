#!/usr/bin/env python3
"""
Start the socket server for the C++ GUI.

Usage
─────
    python run_server.py --policy models/policy/policy_best.pth
    python run_server.py --policy models/policy/policy_best.pth \
                         --value  models/value/value_best.pth \
                         --value-weight 0.3 \
                         --port 65432
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run the chess inference server.")
    parser.add_argument("--policy", required=True, help="Policy checkpoint path.")
    parser.add_argument("--value", default=None, help="Value checkpoint path (optional).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=65432)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature (lower = greedier).")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Consider only top-k policy moves (0 = all).")
    parser.add_argument("--value-weight", type=float, default=0.0,
                        help="Blend policy with value-net lookahead (0–1).")
    return parser.parse_args()


def main():
    args = parse_args()
    from chess_ai.config import Config
    from chess_ai.inference.server import ChessServer

    cfg = Config.from_yaml(args.config)
    server = ChessServer(
        cfg=cfg,
        policy_ckpt=args.policy,
        value_ckpt=args.value,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
        top_k=args.top_k,
        value_weight=args.value_weight,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
