# HexGo — Training Pipeline (`train.py`, `config.py`)

## Pipeline Overview

Each generation:
1. **Batched lockstep self-play** — N games (default 64) run in lockstep via `batched_self_play()`. All leaf evaluations across all games are batched into single GPU calls. No threads, no GIL contention. TOP_K=16 move pruning per position.
2. **Training** — `max(10, buffer_size // BATCH_SIZE)` batches from the replay buffer, fully vectorized spatial policy loss.
3. **Checkpoint** — `net_gen{N:04d}.pt` + `net_latest.pt`
4. **Metrics** — appended to `metrics.jsonl` for dashboard live charts

*Removed: threaded InferenceServer-based self-play, overlap training, ELO evaluation in training loop, checkpoint tournament. ELO system still exists in `elo.py` for use by `tournament.py`.*

---

## Hyperparameters (`config.py`)

All tunable params live in one dict. Edited by the autotune agent; read by `train.py` and `mcts.py` at startup.

| Param | Value | Range | Notes |
|-------|-------|-------|-------|
| `LR` | 1e-3 | 1e-4–5e-3 | Adam learning rate |
| `WEIGHT_DECAY` | 1e-4 | — | L2 regularization |
| `BATCH_SIZE` | 256 | 64–512 | Gradient batch size (set in train.py, not config.py) |
| `SIMS` | 100 | 25–200 | Target simulation budget (curriculum ramps from 16) |
| `SIMS_MIN` | 25 | 6–25 | Reduced budget floor |
| `CAP_FULL_FRAC` | 0 | 0.0–0.5 | Fraction of games using full SIMS (0 = all use curriculum) |
| `CPUCT` | **1.5** | 1.0–3.0 | PUCT exploration constant; loaded at module import (restart to change) |
| `DIRICHLET_ALPHA` | **0.15** | 0.05–0.3 | Root noise concentration — raised for more exploration |
| `DIRICHLET_EPS` | **0.35** | 0.10–0.35 | Root noise weight — raised for stronger noise mixing |
| `ZOI_MARGIN` | 5 | 3–8 | Hex-distance ZOI pruning radius |
| `TD_GAMMA` | 0.99 | 0.95–1.0 | TD-lambda discount for value targets |
| `TEMP_HORIZON` | 40 | 20–60 | Cosine temp annealing parameter (floor reached at `TEMP_HORIZON` moves) |
| `WEIGHT_SYNC_BATCHES` | 20 | 5–40 | Batches between weight sync to inference server |
| `RECENCY_WEIGHT` | **0.75** | 0.5–1.0 | Fraction of each batch drawn from recent half of buffer |
| `AUX_LOSS_OWN` | **0.1** | 0.0–0.5 | Ownership head loss weight (0 = disabled) |
| `AUX_LOSS_THREAT` | **0.1** | 0.0–0.5 | Threat head loss weight (0 = disabled) |
| `VALUE_LOSS_WEIGHT` | **1.0** | 1.0–5.0 | Multiplier on MSE value loss — reduced from 2.0 (was causing value loss plateau) |
| `UNC_LOSS_WEIGHT` | **0.0** | 0.0–0.1 | Value uncertainty Gaussian NLL weight — disabled (was causing loss explosion early) |

---

## Batched Lockstep Self-Play (`batched_self_play`)

All N games (default 64) run simultaneously in lockstep:
1. **Root init**: encode all active games, batch GPU eval (trunk + value + spatial policy), create root nodes with TOP_K=16 move pruning + Dirichlet noise.
2. **MCTS sims**: for each sim, selection traversal for all games, batch GPU eval of all unexpanded leaves, backprop + unmake for all games.
3. **Move selection**: temperature-based sampling from visit counts, record spatial `policy_target` and `legal_mask` planes.
4. **Repeat** until all games terminate or hit `max_moves`.

No threads, no GIL contention — single-threaded with periodic GPU bursts. The `InferenceServer` is NOT used during training.

### Curriculum
- **Sims**: linearly ramps from SIMS_MIN=16 to target (default 100) over SIMS_RAMP=20 generations.
- **Max moves**: linearly ramps from MAX_MOVES_MIN=30 to MAX_MOVES_MAX=100 over MAX_MOVES_RAMP=20 generations.

### Dirichlet noise
Applied at root: `α=DIRICHLET_ALPHA` (0.15), `ε=DIRICHLET_EPS` (0.35).

### Temperature schedule
`temp = max(0.05, cos(π/2 × move / TEMP_HORIZON))` — hits floor at `TEMP_HORIZON` moves.

### Value Targets (TD-lambda)
Computed backwards from game outcome with `TD_LAMBDA=0.8`:
`targets[t] = sign × TD_GAMMA × ((1 - TD_LAMBDA) × v_next + TD_LAMBDA × g_next)`
where `sign` flips when consecutive positions have different active players.

### Decisive Game Saving
All decisive (non-draw) games with >= 6 moves are saved to `replays/decisive/` for corpus building. Shortest and longest decisive games also saved to main `replays/` dir.

### Replay Buffer
- FIFO `deque(maxlen=50000)` positions
- Each entry: `{board, policy_target, legal_mask, z, own_label, threat_label}`
- `policy_target` and `legal_mask` are spatial `[S, S]` arrays (not per-move vectors)
- **Recency-weighted sampling**: each batch draws `RECENCY_WEIGHT` fraction from the most-recent half of the buffer, remainder uniform. Prevents anchoring to early-training incompetent play.

---

## D6 Augmentation at Train Time

`d6_augment_sample(item, tf_idx)` applies one of 12 D6 transforms to:
- Board array (via `_transform_board()` — channel 2 (to-move) is rotation-invariant)
- `policy_target` spatial plane (via `_transform_aux()`)
- `legal_mask` spatial plane (via `_transform_aux()`)
- `own_label` and `threat_label` aux arrays (via `_transform_aux()`)

The spatial policy target is renormalized after transform (probability mass may clip at window edges). Applied at batch time (`tf_idx = random.randrange(12)`).

The effective augmentation ratio is slightly below 12x for edge positions.

---

## Training Loss

```
L = VALUE_LOSS_WEIGHT·MSE(z, v) + spatial_masked_CE(π, p) - ENTROPY_REG·H(π) + AUX_OWN·MSE(own, own_pred) + AUX_THREAT·BCE(threat, threat_pred) + UNC_LOSS_WEIGHT·GaussNLL + c‖θ‖²
```

- Value loss: MSE against TD-lambda target, scaled by `VALUE_LOSS_WEIGHT=1.0`
- Policy loss: **spatial masked cross-entropy** — logit map `[B, S, S]` masked to legal moves (`-inf` for illegal), softmax over masked map, cross-entropy against `policy_target` spatial plane. Fully vectorized (no Python loops over moves). Normalized by items with >= 1 legal move.
- Entropy regularization: `-ENTROPY_REG·H(π)` bonus prevents premature policy collapse (`ENTROPY_REG=0.01`)
- Ownership aux loss: MSE on `[S, S]` map (+1=P1, -1=P2, 0=empty at game end); weight `AUX_LOSS_OWN=0.1`
- Threat aux loss: BCE-with-logits on `[S, S]` binary map (1=cell on winning 6-in-a-row); weight `AUX_LOSS_THREAT=0.1`; AMP-safe (uses logits, not post-sigmoid)
- Value uncertainty: Gaussian NLL (`0.5*(log(σ²) + (z-v)²/σ²)`); currently disabled (`UNC_LOSS_WEIGHT=0`)
- Weight decay: L2 regularization (WEIGHT_DECAY, applied via Adam)
- FP16 AMP via `torch.amp.GradScaler`
- Gradient clipping: `clip_grad_norm_(net.parameters(), 1.0)`

Aux weights are kept small per the bitter-lesson principle: they guide trunk representation without dominating the main value+policy signal.

### Per-Component Loss Logging

`train_batch()` returns `{loss, loss_v, loss_p, loss_aux, entropy}`. The gen loop tracks these separately and logs:
```
Train: N batches  loss=X  loss_v=X  loss_p=X  aux=X  ent=X
```
All components are written to `metrics.jsonl` as `avg_loss`, `avg_loss_v`, `avg_loss_p`, `avg_aux`.

---

## LR Schedule

Warmup + cosine decay:
- Warmup: 5 gens, linearly ramp from 0.1× to 1.0× base LR
- Cosine decay: from gen 5 to end, decay to 0.01× base LR

---

## torch.compile

Disabled — uses 3-4GB RAM during JIT compilation, which is problematic on the RTX 2060. The compile call is commented out in `train()`.

---

## Performance Metrics

Per-generation timing logged: `sp_time_s` (self-play), `tr_time_s` (training), `gen_time_s` (total). Also: `games_per_s`, `decisive` (count of non-draw games), `sims`, `max_moves` (curriculum values).

---

## Known Issues

1. **Policy loss normalization**: Normalized by item count (positions with ≥1 legal move in window), not move count. Loss magnitude varies with window clip rate. (Low priority — consistent across gens.)
2. **Buffer persistence disabled**: `save_buffer()` / `load_buffer()` exist but are disabled — was causing hangs (save) and OOM (load) with large buffers. Buffer is lost on restart.
