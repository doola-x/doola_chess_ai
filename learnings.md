    # ML Mistakes & Fixes — Chess AI Learnings
                                                                                                                                                                                                            
    A reference of every ML error found in the original codebase and what the
    correct approach is. Each entry has: what was wrong, why it matters, and
    what was done instead.

    ---

    ## 1. Sigmoid instead of Softmax on policy output

    **File:** `scripts/rl_arch.py` line 22

    **The mistake:**
    ```python
    x = torch.sigmoid(self.fc3(x))  # outputs (B, 4208) with each value in [0,1]

    Why it's wrong:
    Sigmoid applies independently to every logit. The outputs do NOT sum to 1, so
    they are not a probability distribution over moves. You cannot sample from this
    correctly, and log_prob calculations will be wrong, poisoning every policy
    gradient update.

    The fix:
    Remove the activation entirely — output raw logits. Apply softmax (or
    log_softmax) at the point where you need probabilities (sampling, loss
    computation). This is the standard approach for categorical distributions.

    # In the model: no activation on final layer
    x = self.fc3(x)  # raw logits

    # At training/sampling time:
    probs = F.softmax(logits / temperature, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_p = dist.log_prob(action)

    ---
    2. LSTM fed a single timestep

    File: scripts/train.py line 68

    The mistake:
    self.lstm = nn.LSTM(input_size=64*8*8, hidden_size=1024, num_layers=2, batch_first=True)
    # ...
    x = x.unsqueeze(1)  # (B, 4096) → (B, 1, 4096)  — sequence length of 1
    x, _ = self.lstm(x)
    x = x[:, -1, :]

    Why it's wrong:
    An LSTM is designed to model sequences. Fed a single frame per forward pass,
    it degenerates into a glorified linear layer with extra parameters and slower
    training. You get none of the temporal modelling benefit and all of the
    complexity cost. The hidden state carries no information across positions
    within a game because games are not batched as sequences.

    The fix:
    Replace with a deeper fully-connected block (or add more conv layers). If you
    genuinely want sequence modelling (e.g., model the history of moves in a game),
    you need to batch entire game sequences and pass them as (B, T, features).
    That's a bigger architectural decision — for move prediction from a single
    position, FC layers are correct and simpler.

    ---
    3. O(n) reverse dictionary lookup on every move

    Files: scripts/train.py, scripts/inference.py, scripts/actor_critic_train.py,
    inference.py (root), play_inference.py, and others

    The mistake:
    # Called thousands of times during training and inference
    def decode_move(mapping, value):
        for key, val in mapping.items():   # scans all 4208–9010 entries every call
            if val == value:
                return key

    Why it's wrong:
    This is O(n) per call. With 4208 moves and thousands of decode calls per
    episode, this is pure wasted compute. It also signals a misunderstanding of
    dict semantics — dicts are for O(1) keyed lookup.

    The fix:
    Build the reverse mapping once at load time:
    self._idx_to_move = {v: k for k, v in self._move_to_idx.items()}
    # Now every decode is O(1)

    ---
    4. No legal move masking before policy sampling

    Files: scripts/actor_critic_train.py, scripts/inference.py

    The mistake:
    The actor outputs logits over all ~4208 moves and selects the argmax (or
    samples), with no guarantee the selected move is legal on the current board.
    The code then tries to verify legality afterwards and falls back to random
    moves or skips — wasting the forward pass entirely.

    Why it's wrong:
    The model can never learn a clean signal because illegal moves pollute both
    the loss and the exploration. The gradient flows back through illegal actions,
    which the model can never actually take, creating a confused policy.

    The fix:
    Before sampling, zero-out (mask to -inf) all logits corresponding to illegal
    moves. Only legal moves can be selected:
    legal_idxs = encoder.legal_mask(board)   # list of valid indices
    mask = torch.full((num_moves,), float("-inf"))
    mask[legal_idxs] = 0.0
    probs = F.softmax(logits + mask, dim=-1)  # illegal moves get prob ≈ 0

    ---
    5. Reward accumulation divides by move count

    File: scripts/actor_critic_train.py line 113

    The mistake:
    total_reward += reward / moves   # 'moves' = move number, grows over the game

    Why it's wrong:
    Late-game moves get rewards divided by a large number (e.g., move 40 gets
    reward/40), making them nearly invisible to the optimizer. The agent learns
    to prioritise early-game actions even if late-game play is what decides the
    outcome. This is the opposite of what you want — endgame positions are often
    the most decisive.

    The fix:
    Use discounted returns properly (standard REINFORCE):
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + discount * G   # e.g. discount = 0.99
        returns.insert(0, G)
    Optionally normalise returns by subtracting the mean and dividing by std —
    this reduces variance without introducing bias.

    ---
    6. No validation split — training on all data

    Files: scripts/train.py, scripts/train_val.py

    The mistake:
    All processed data is loaded and trained on directly. There is no held-out
    validation set and no tracking of val loss across epochs.

    Why it's wrong:
    You have no way to detect overfitting, no principled way to select the best
    checkpoint, and no idea whether the model generalises or has just memorised
    the training positions.

    The fix:
    Hold out a fixed fraction (e.g., 10%) of data before training:
    val_size = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(dataset, [len(dataset)-val_size, val_size])
    Track val_loss each epoch. Save best.pth when val loss improves, not
    just at fixed epoch intervals.

    ---
    7. Undefined variable in dead code (mask bug)

    File: scripts/inference.py — adjust_logits() function

    The mistake:
    def adjust_logits(logits, legal_moves_masks):
        ...
        inverse_mask = 1 - mask   # ← NameError: 'mask' is not defined

    The parameter is named legal_moves_masks but the body references mask.
    The function was written but never called (dead code).

    Why it matters:
    If this function had been called, it would crash immediately. It shows the
    legal-masking logic was attempted but abandoned without testing.

    The fix:
    The function is replaced by proper legal masking (see mistake #4).

    ---
    8. Two competing move encodings used inconsistently

    The mistake:
    - moves.json — SAN notation (e.g., "Nf3", "exd5"), 9010 entries
    - moves0.json — UCI notation (e.g., "g1f3", "e4d5"), 4208 entries

    Different scripts load different files. Models trained with one encoding
    cannot be used with code expecting the other. SAN is ambiguous without
    board context; UCI is always unambiguous.

    Why it's wrong:
    You cannot mix checkpoints between training runs. SAN requires knowing the
    board state to disambiguate (e.g., which knight moves to f3?), making it
    fragile for encoding targets.

    The fix:
    Standardise on UCI throughout. UCI is:
    - Unambiguous (always from_square + to_square + [promotion])
    - Native to python-chess (move.uci())
    - Smaller vocabulary (4208 vs 9010)
    - Universal — every chess engine speaks UCI

    ---
    9. No gradient clipping

    Files: All training scripts

    The mistake:
    No torch.nn.utils.clip_grad_norm_() call before optimizer.step().

    Why it's wrong:
    With deeper networks (conv + 3 FC layers), occasional large gradients can
    blow up weights — especially in RL where reward signals can be noisy and
    sparse. This manifests as NaN losses or sudden divergence.

    The fix:
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    A max_norm of 1.0 is a safe default. Tighten to 0.5 for RL training.

    ---
    10. Model checkpoint stores raw state_dict with no metadata

    Files: All training scripts

    The mistake:
    torch.save(model.state_dict(), path)

    Why it's wrong:
    When you load a checkpoint later, you have no record of what epoch it's from,
    what config produced it, or what val loss it achieved. This makes checkpoint
    management a manual memory exercise.

    The fix:
    Save a dict with metadata:
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "config": dataclasses.asdict(cfg.model),
    }, path)

    ---
    11. Hardcoded paths assume execution from scripts/ directory

    Files: All scripts in scripts/

    The mistake:
    data_dir = "../data/processed_games_3"
    model_path = "../models/model_epoch_15.pth"

    Why it's wrong:
    Scripts fail if run from anywhere other than the scripts/ subdirectory.
    CI, notebooks, and any other tooling has to cd scripts/ first.

    The fix:
    Anchor paths to the project root via config, or compute them relative to the
    module file:
    ROOT = Path(__file__).parent.parent  # always correct regardless of cwd
    Or better: put all paths in Config and pass the config through.

    ---
    12. Value network trained but never used in RL loop

    The mistake:
    train_val.py trains a value network (critic). actor_critic_train.py defines
    a Critic class but the trained value weights are never loaded — the critic
    is initialised randomly and the signal is weak.

    Why it matters:
    Without a good baseline, REINFORCE has very high variance. The whole point of
    actor-critic is to use V(s) as a baseline to compute advantages:
    A = G_t - V(s).  A random critic provides no variance reduction.

    The fix:
    Warm-start the critic from the supervised value checkpoint:
    critic.load_state_dict(torch.load("models/value/value_best.pth")["model"])
    Train the actor and critic jointly, updating the critic towards the observed
    returns while using it as a baseline for the actor's gradient.

    ---
    Summary Table

    ┌─────┬──────────────────────────────┬──────────────────────────────────┬──────────────────────────────────┐
    │  #  │           Mistake            │              Impact              │               Fix                │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 1   │ Sigmoid on policy output     │ Wrong gradients, broken sampling │ Raw logits + softmax at use site │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 2   │ LSTM on single frame         │ Wasted capacity, slower training │ Replace with FC layers           │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 3   │ O(n) move decode             │ Slow inference & training        │ Pre-build reverse dict           │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 4   │ No legal move masking        │ Illegal moves in gradients       │ Mask to -inf before softmax      │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 5   │ Reward divided by move count │ Ignores endgame                  │ Discounted returns (REINFORCE)   │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 6   │ No validation split          │ Undetected overfitting           │ Hold out 10%, track val loss     │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 7   │ Undefined mask variable      │ Silent dead code, would crash    │ Removed, replaced by #4          │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 8   │ Two move encodings           │ Incompatible checkpoints         │ Standardise on UCI               │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 9   │ No gradient clipping         │ Risk of NaN / divergence         │ clip_grad_norm_(params, 1.0)     │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 10  │ Raw state_dict checkpoints   │ No reproducibility               │ Save metadata dict               │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 11  │ Hardcoded relative paths     │ Brittle execution context        │ Config-driven, root-anchored     │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ 12  │ Critic never warm-started    │ High variance RL                 │ Load supervised value weights    │
    ├─────┼──────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
    │ ``` │                              │                                  │                                  │
    └─────┴──────────────────────────────┴──────────────────────────────────┴──────────────────────────────────┘

