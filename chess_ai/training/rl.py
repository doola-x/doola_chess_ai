"""
Actor-critic reinforcement learning trainer.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_ai.config import Config
from chess_ai.core.moves import MoveEncoder
from chess_ai.env.chess_env import ChessEnv
from chess_ai.models.nets import PolicyNet, ValueNet, load_policy, load_value


class RLTrainer:
    """
    REINFORCE with baseline (actor-critic style).

    The critic (ValueNet) estimates V(s) and is used to compute advantages:
        A(s, a) = G_t - V(s)

    Policy gradient loss:
        L_actor = -log π(a|s) * A(s, a)

    Critic loss:
        L_critic = MSE(V(s), G_t)

    Usage
    -----
        cfg = Config.from_yaml("configs/rl.yaml")
        trainer = RLTrainer(cfg)
        trainer.fit()
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.resolve_device())
        self.encoder = MoveEncoder(cfg.data.moves_file)
        Path(cfg.paths.rl_out).mkdir(parents=True, exist_ok=True)

        # ── Models ────────────────────────────────────────────────────
        rc = cfg.rl
        if rc.policy_checkpoint:
            self.actor = load_policy(rc.policy_checkpoint, cfg.model, str(self.device))
            self.actor.train()
        else:
            self.actor = PolicyNet(cfg.model).to(self.device)

        self.critic = ValueNet(cfg.model).to(self.device)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.training.learning_rate
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.training.learning_rate
        )

        stockfish = cfg.paths.stockfish if Path(cfg.paths.stockfish).exists() else None
        self.env = ChessEnv(cfg.rl, stockfish=stockfish)

    # ------------------------------------------------------------------

    def fit(self) -> None:
        rc = self.cfg.rl
        best_ep_return = -float("inf")

        for episode in range(1, rc.episodes + 1):
            play_as = chess.WHITE if random.random() > 0.5 else chess.BLACK
            self.env.play_as = play_as

            states, actions, rewards, log_probs, values = [], [], [], [], []

            state_np = self.env.reset()
            done = False
            move_count = 0

            while not done and move_count < rc.max_moves:
                state_t = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

                # ── Legal move mask ───────────────────────────────────
                legal_idxs = self.encoder.legal_mask(self.env.board)
                if not legal_idxs:
                    break

                # ── Actor: sample a move ──────────────────────────────
                with torch.no_grad():
                    logits = self.actor(state_t).squeeze(0)  # (num_moves,)

                mask = torch.full((self.encoder.num_moves,), float("-inf"), device=self.device)
                mask[legal_idxs] = 0.0
                masked_logits = logits + mask
                probs = F.softmax(masked_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                log_p = dist.log_prob(action_idx)

                # ── Critic: estimate value ────────────────────────────
                with torch.no_grad():
                    v = self.critic(state_t).squeeze()

                # ── Environment step ──────────────────────────────────
                uci = self.encoder.decode(action_idx.item())
                if uci is None:
                    break
                move = chess.Move.from_uci(uci)
                if move not in self.env.board.legal_moves:
                    break

                next_state_np, reward, done = self.env.step(move)

                states.append(state_t)
                actions.append(action_idx)
                log_probs.append(log_p)
                rewards.append(reward)
                values.append(v)

                state_np = next_state_np
                move_count += 1

            if not rewards:
                continue

            # ── Compute discounted returns ────────────────────────────
            G = 0.0
            returns = []
            for r in reversed(rewards):
                G = r + rc.discount * G
                returns.insert(0, G)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # Normalise returns (reduces variance)
            if len(returns_t) > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)

            # ── Actor loss (policy gradient with advantage baseline) ───
            advantages = (returns_t - values_t.detach())
            actor_loss = -(log_probs_t * advantages).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            if self.cfg.training.grad_clip:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.training.grad_clip)
            self.actor_opt.step()

            # ── Critic loss ───────────────────────────────────────────
            critic_loss = F.mse_loss(values_t, returns_t.detach())
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            ep_return = sum(rewards)

            if episode % 100 == 0:
                print(
                    f"Episode {episode:6d}/{rc.episodes}  "
                    f"return={ep_return:.3f}  moves={move_count}  "
                    f"actor_loss={actor_loss.item():.4f}  "
                    f"critic_loss={critic_loss.item():.4f}"
                )

            if ep_return > best_ep_return:
                best_ep_return = ep_return
                self._save_actor("actor_best.pth")

            if episode % 500 == 0:
                self._save_actor(f"actor_ep{episode}.pth")

        self.env.close()
        print(f"RL training complete. Best return: {best_ep_return:.3f}")

    def _save_actor(self, name: str) -> None:
        path = Path(self.cfg.paths.rl_out) / name
        torch.save({"model": self.actor.state_dict()}, path)
