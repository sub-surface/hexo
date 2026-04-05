# Experiment Queue

Priority-ordered list of changes to test. Agent picks from top, marks done/skip.

## Tier 1.5 — Compact Play Curriculum (HIGH PRIORITY)

### EXP-00: ZOI-restricted self-play (ramp ZOI_MARGIN 3→5 over 30 gens)
- **Files**: `train.py` (add curriculum), `hexgo-rs/src/batched.rs` (pass zoi_margin param)
- **Change**: Add `ZOI_MARGIN_MIN = 3`, `ZOI_MARGIN_RAMP = 30`. During self-play, pass tighter ZOI margin to Rust engine so MCTS only considers moves within 3 hex steps of recent play initially. Ramp to full 5 over 30 gens. In Rust, after computing legal moves, filter through `game.zoi_moves(margin, lookback)` before top-K.
- **Hypothesis**: The #1 problem is scattered play — both sides spread stones across the board instead of blocking/building locally. Forcing compact play area means the network MUST learn blocking and chain-building because there's nowhere to wander. As the network develops tactical awareness, gradually open up the play area.
- **Expected impact**: Faster decisive ratio improvement, shorter games, much stronger blocking/threat signals in training data.
- **Status**: pending

## Tier 2 — High-Impact Efficiency

### EXP-01: TOP_K curriculum (ramp 64→24 over 20 gens)
- **File**: `train.py` lines 71-76
- **Change**: Add `TOP_K_MIN = 64`, `TOP_K_RAMP = 20`, create `_curriculum_top_k(gen, target)` function mirroring `_curriculum_sims`. Use in self-play call.
- **Hypothesis**: Early nets have diffuse policies; TOP_K=24 from gen 1 cuts viable moves. Wider branching early → better exploration → faster learning.
- **Status**: pending

### EXP-02: Cosine warm restarts (T_0=20, T_mult=2)
- **File**: `train.py` LR schedule (lines ~745-755)
- **Change**: Replace `LambdaLR` with `CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=LR*0.01)`. Remove custom lr_lambda.
- **Hypothesis**: Self-play is non-stationary; single cosine decay is wrong. Warm restarts let the model escape local minima periodically.
- **Status**: pending

### EXP-03: Increase buffer to 300K
- **File**: `train.py` line 65
- **Change**: `BUFFER_CAP = 300_000`
- **Hypothesis**: 100K flushes data in ~15 gens, too aggressive. 300K keeps ~50 gens, better replay diversity.
- **Status**: pending

### EXP-04: Exponential decay buffer sampling
- **File**: `train.py` lines 373-381
- **Change**: Replace binary 75/25 recency split with `weight = exp(-age / half_life)` where `half_life = 10 * positions_per_gen`. Sample with `random.choices(buf_list, weights=weights, k=BATCH_SIZE)`.
- **Hypothesis**: Smooth decay is better than cliff at buffer midpoint. Recent data gets higher weight but old data isn't completely ignored.
- **Status**: pending

### EXP-05: Increase ELO eval to 12 games
- **File**: `train.py` ELO eval section (~line 865)
- **Change**: `n_games=12` instead of 6
- **Hypothesis**: 6 games is too noisy (~20% CI width). 12 games gives ~14% CI. More reliable progress signal.
- **Status**: pending

### EXP-06: Down-weight draw games in buffer
- **File**: `train.py` postprocessing section
- **Change**: When adding positions to buffer, draw games add every 2nd position (50% sub-sample). Decisive games add all positions.
- **Hypothesis**: Draw positions have z=0 value targets which teach the value head nothing. Policy targets from draws are low quality (scattered play).
- **Status**: pending

## Tier 3 — Advanced

### EXP-07: Short-term value auxiliary targets (6/16 turn horizons)
- **Files**: `train.py` (postprocessing + loss), `net.py` (new heads)
- **Change**: During postprocessing, compute exponential-decay value targets at 6 and 16 turn horizons. Add two FC heads off trunk. Train with MSE, weight 0.15 each.
- **Hypothesis**: Lower-variance value signal than game outcome. KataGo's highest-impact auxiliary loss.
- **Status**: pending

### EXP-08: EMA weights for self-play
- **File**: `train.py`
- **Change**: Maintain shadow weights: `ema = 0.995 * ema + 0.005 * current` after each optimizer step. Use EMA weights for self-play eval callback.
- **Hypothesis**: Smooths out bad updates without tournament cost. Self-play sees a more stable policy.
- **Status**: pending

### EXP-09: Lightweight Reanalyze
- **File**: `train.py`
- **Change**: Every 5 gens, do a forward pass on random 10% of buffer positions, update their value_est with current net's prediction.
- **Hypothesis**: Refreshes stale TD-lambda bootstrap targets. Cheap — just forward passes, no MCTS.
- **Status**: pending
