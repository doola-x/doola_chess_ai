#!/usr/bin/env python3
"""
Training entry point.

Usage
─────
    # Behaviour cloning from PGN data
    python train.py --mode supervised --config configs/supervised.yaml

    # Value network (Stockfish labels)
    python train.py --mode value --config configs/value.yaml

    # Actor-critic RL
    python train.py --mode rl --config configs/rl.yaml

    # Override any config key inline
    python train.py --mode supervised --config configs/supervised.yaml \
        --set training.learning_rate=0.0005 training.num_epochs=100
"""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Train a chess AI model.")
    parser.add_argument(
        "--mode",
        choices=["supervised", "value", "rl"],
        required=True,
        help="Training mode.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="section.key=value",
        help="Inline config overrides, e.g. training.lr=0.001",
    )
    return parser.parse_args()


def apply_overrides(cfg, overrides: list[str]) -> None:
    """Apply 'section.key=value' strings onto a Config object."""
    for item in overrides or []:
        if "=" not in item:
            print(f"Warning: ignoring malformed override {item!r} (expected section.key=value)")
            continue
        key_path, _, raw_val = item.partition("=")
        parts = key_path.strip().split(".")
        if len(parts) != 2:
            print(f"Warning: ignoring override with unexpected depth: {item!r}")
            continue
        section, key = parts
        sub = getattr(cfg, section, None)
        if sub is None:
            print(f"Warning: unknown config section {section!r}")
            continue
        if not hasattr(sub, key):
            print(f"Warning: unknown config key {section}.{key}")
            continue
        # Type-preserve the existing value
        existing = getattr(sub, key)
        try:
            if existing is None:
                typed = raw_val if raw_val.lower() != "null" else None
            elif isinstance(existing, bool):
                typed = raw_val.lower() in ("1", "true", "yes")
            else:
                typed = type(existing)(raw_val)
        except (ValueError, TypeError):
            typed = raw_val
        setattr(sub, key, typed)
        print(f"  Override: {section}.{key} = {typed!r}")


def main():
    args = parse_args()

    from chess_ai.config import Config
    cfg = Config.from_yaml(args.config)
    apply_overrides(cfg, args.set)

    print(f"Mode: {args.mode}  |  Config: {args.config}  |  Experiment: {cfg.experiment}")
    print(f"Device: {cfg.resolve_device()}")

    if args.mode == "supervised":
        from chess_ai.training.supervised import PolicyTrainer
        PolicyTrainer(cfg).fit()

    elif args.mode == "value":
        from chess_ai.training.value import ValueTrainer
        ValueTrainer(cfg).fit()

    elif args.mode == "rl":
        from chess_ai.training.rl import RLTrainer
        RLTrainer(cfg).fit()


if __name__ == "__main__":
    main()
