"""
Centralized configuration via dataclasses + YAML.

Usage:
    cfg = Config.from_yaml("configs/supervised.yaml")
    cfg = Config.from_yaml("configs/rl.yaml")

All hyperparameters live here. YAML files can override any field.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    # Board input
    in_channels: int = 13          # 12 piece types + active color channel
    # Conv block
    conv_filters: int = 64
    conv_kernel: int = 3
    # Fully connected
    hidden_dim: int = 1024
    hidden_dim2: int = 512
    dropout: float = 0.3
    # Output
    num_moves: int = 4208          # UCI move space (moves0.json)


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 30
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    # Checkpoint
    save_every: int = 5            # Save checkpoint every N epochs
    keep_best: bool = True         # Always save best val-loss checkpoint
    # Validation
    val_split: float = 0.1         # Fraction of data held out for validation
    shuffle: bool = True
    num_workers: int = 0


@dataclass
class RLConfig:
    episodes: int = 10_000
    max_moves: int = 50
    top_k: int = 10                # Number of candidate moves to evaluate
    discount: float = 0.95
    stockfish_time: float = 0.01   # Seconds per Stockfish move
    # Reward shaping weights
    reward_material: float = 2.0
    reward_center: float = 0.2
    reward_king_safety: float = 0.8
    # Pretrained weights to start from (optional)
    policy_checkpoint: Optional[str] = None


@dataclass
class DataConfig:
    # Move encoding
    moves_file: str = "data/moves0.json"   # UCI encoding (4208 moves)
    # Processed datasets
    policy_data_dir: str = "data/processed_games_4"
    value_data_dir: str = "data/processed_value"
    value_training_dir: str = "data/value_training"
    # Raw data
    raw_pgn_dir: str = "data/raw_data"
    tactics_dir: str = "data/tactics"


@dataclass
class PathsConfig:
    models_dir: str = "models"
    stockfish: str = "./stockfish"
    # Specific output dirs per training mode
    policy_out: str = "models/policy"
    value_out: str = "models/value"
    rl_out: str = "models/rl"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    # Device: "auto" → picks mps > cuda > cpu
    device: str = "auto"
    # Experiment name (used in checkpoint filenames)
    experiment: str = "default"

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML, merging over defaults."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        for section, values in d.items():
            if not isinstance(values, dict):
                # Top-level scalar (e.g. device, experiment)
                setattr(cfg, section, values)
                continue
            sub = getattr(cfg, section, None)
            if sub is None:
                raise ValueError(f"Unknown config section: {section!r}")
            for k, v in values.items():
                if not hasattr(sub, k):
                    raise ValueError(f"Unknown config key: {section}.{k}")
                setattr(sub, k, v)
        return cfg

    def to_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def resolve_device(self) -> str:
        """Return the actual torch device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
