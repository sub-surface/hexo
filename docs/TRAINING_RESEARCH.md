# AlphaZero Self-Play Training: State of the Art Research Report

*For HexGo: ~1.9M param network, 200 MCTS sims, 128 games/gen, hexagonal Connect6*
*Generated 2026-04-03*

---

## Executive Summary: Top 10 Changes by Expected Impact

| # | Change | Expected Impact | Effort |
|---|--------|-----------------|--------|
| 1 | **Fix value head: use WDL (win/draw/loss) 3-output softmax** instead of single MSE | Eliminates value explosion, better draw handling | Medium |
| 2 | **Implement playout cap randomization** (KataGo-style) | ~1.5-2x effective training throughput | Medium |
| 3 | **Lower LR to 2e-4, use cosine with warm restarts** (not single cosine) | Prevents catastrophic forgetting between restarts | Low |
| 4 | **Increase buffer to 500K, add priority sampling** by game decisiveness | Better signal-to-noise ratio | Low |
| 5 | **Add Reanalyze** for buffer positions (re-run MCTS with latest net) | Refreshes stale value targets cheaply | High |
| 6 | **Increase sims to 400 for full searches, 50 for fast** (via playout cap) | Stronger policy targets where they matter | Low (config) |
| 7 | **Add short-term value auxiliary targets** (KataGo-style, 6/16/50-turn horizons) | Lower-variance value training signal | Medium |
| 8 | **Implement search-contempt** (Thompson sampling hybrid MCTS) | More challenging training positions, up to 70 Elo | Medium |
| 9 | **Down-weight long draw games** in buffer; use position-stage weighting | Cleaner training signal | Low |
| 10 | **Train more batches per gen** (scale with buffer, not capped at 150) | Currently under-training relative to data | Low |

---

## 1. Buffer Management

### Current Setup
- `BUFFER_CAP = 100,000` positions, FIFO deque
- `RECENCY_WEIGHT = 0.75` (75% from recent half, 25% uniform)
- 128 games/gen, ~50 positions/game = ~6,400 new positions/gen
- Buffer is ~15 generations deep

### Research Findings

**AlphaGo Zero** used 500,000 *games* (~100M positions). **OpenSpiel's AlphaZero** recommends `steps_per_observation = 1` (each position trained on exactly once on average). **ELF OpenGo** found that too-small buffers cause a negative feedback loop: the agent overfits to playing against itself, degrading against diverse opponents.

The key ratio is **training steps per new position**. You generate ~6,400 positions/gen and run 150 batches of 512 = 76,800 sample draws per gen. With a 100K buffer, each position gets drawn ~0.77 times per gen. This is actually reasonable, but the 150-batch cap is arbitrary and may under-train when the buffer is full.

**Data staleness** is real: positions generated with a network 15+ generations old have value targets that are increasingly wrong. The recency weighting partially addresses this, but a better approach is **priority sampling by generation age** with exponential decay, not a binary recent/old split.

### Recommendations

1. **Increase buffer to 300K-500K positions** (~50-80 gens deep). Your 100K cap flushes data too fast - you're throwing away recent-ish data that still has valid policy targets even if value targets drift.

2. **Replace binary recency split with exponential decay sampling.** Instead of 75/25 split, assign weight `exp(-age / half_life)` where `half_life = 10 generations`. This gives a smooth falloff instead of a cliff.

3. **Scale training batches with buffer size.** Instead of `min(len(buffer) // BATCH_SIZE, 150)`, use `min(len(buffer) // BATCH_SIZE, max(150, new_positions_this_gen * 2 // BATCH_SIZE))`. The goal: each new position should be trained on ~1-2 times.

4. **Track buffer composition.** Log the generation distribution of your buffer and the age of sampled batches. If >30% of your buffer is from >20 gens ago, that data is mostly noise for the value head.

---

## 2. Learning Rate Schedule

### Current Setup
- LR = 5e-4 (base), Adam optimizer
- 5-gen linear warmup (0.1x to 1x)
- Single cosine decay to 0.01x over remaining gens
- Weight decay = 1e-4

### Research Findings

**KataGo** used SGD (not Adam) with per-sample LR of 6e-5 (per-batch: 6e-5 * 256 = 0.0154). They reduced LR by 3x for the first 5M samples, then by 10x for final tuning. No cosine schedule - just manual step decay at milestones.

**Leela Chess Zero** also uses SGD with step-based LR decay. The consensus in game-playing AI is that Adam can cause instability in self-play because the adaptive moment estimates get stale as the data distribution shifts every generation.

**Key insight:** Single cosine decay is wrong for self-play because the training never truly "converges" - the data distribution shifts every generation. Cosine decay assumes you're iterating over a fixed dataset. With self-play, you want the LR to stay moderately high to adapt to the shifting distribution.

### Recommendations

1. **Switch to cosine annealing with warm restarts** (SGDR/CosineAnnealingWarmRestarts). Use `T_0=20` gens, `T_mult=2` so cycles are 20, 40, 80 gens. This periodically resets LR to escape local minima that form as the data distribution shifts.

2. **Lower peak LR to 2e-4.** Your current 5e-4 is aggressive for a 1.9M param network with Adam. KataGo's effective per-sample rate was 6e-5 with SGD (which has different dynamics, but the ballpark is informative). For Adam, 1e-4 to 3e-4 is the sweet spot for this network size.

3. **Consider SGD with momentum 0.9 instead of Adam.** This is what both KataGo and Leela use. Adam's adaptive rates can amplify noise in self-play's non-stationary distribution. If you stay with Adam, use a low beta2 (0.99 instead of 0.999) to make it forget old gradient statistics faster.

4. **Weight decay 3e-5** (KataGo's value). Your 1e-4 may be too aggressive for a small network.

---

## 3. Value Head Explosion (CRITICAL)

### Your Symptom
Value loss spiked from 0.05 to thousands, then NaN.

### Root Cause Analysis

Your value head uses **MSE loss** on a **single scalar output** with no bounded activation. Looking at your code:

```python
loss_v = F.mse_loss(val, z_targets)
```

And in `net.py`, the value head likely outputs an unbounded scalar that you clip to [-1, 1] at inference time but NOT during training. This is the classic failure mode:

1. The value head predicts, say, 0.95 for a position
2. The TD-lambda target says -0.8 (because the game was lost)
3. MSE gradient is huge: 2 * (0.95 - (-0.8)) = 3.5
4. With many such positions in a batch, gradients compound
5. The value head weights get pushed to extreme values
6. Next batch: predictions are now +50 or -50, targets are still in [-1, 1]
7. MSE on (50 - 0.3)^2 = 2,470 => loss explodes
8. Gradients overflow float32 => NaN

**Your gradient clipping at max_norm=1.0 is a band-aid, not a fix.** It prevents the symptom but not the cause. The fundamental problem is that MSE loss with unbounded outputs is unstable.

### How Successful Implementations Handle This

**Leela Chess Zero** switched from single tanh to **WDL (Win/Draw/Loss) head**: 3 outputs through softmax, trained with cross-entropy. This is bounded by design - cross-entropy on softmax outputs cannot produce infinite gradients.

**AlphaGo Zero/AlphaZero** used tanh activation on the value output, bounding it to [-1, 1]. The loss is MSE on the tanh output. But even this can be unstable because tanh gradients vanish at the extremes, causing the pre-tanh values to grow unboundedly.

**KataGo** uses multiple value heads with different time horizons, which provides lower-variance targets and reduces the magnitude of individual gradient updates.

### Recommendations (in order of priority)

1. **Switch to WDL value head immediately.** Replace single scalar with 3 outputs (win/draw/loss probabilities via softmax), trained with cross-entropy. Your targets become:
   - Win: `[1, 0, 0]`
   - Draw: `[0, 1, 0]`
   - Loss: `[0, 0, 1]`
   - For TD-lambda intermediate targets, use soft labels: a target of 0.7 becomes `[0.85, 0.15, 0]` (mix win/draw)
   
   **This alone likely prevents all future value explosions.**

2. **If keeping MSE, apply tanh to the value output during training too** (not just inference). Then MSE is bounded.

3. **Add value head gradient scaling.** Multiply value head gradients by 0.5 (you can use a gradient hook). This dampens the feedback loop.

4. **Reduce VALUE_LOSS_WEIGHT to 0.5.** The original AlphaZero equally weighted policy and value loss. Your 1.0 weight with separate policy cross-entropy means value loss dominates when it spikes. At 0.5, value gradients are halved.

5. **Add loss spike detection with rollback.** Your current code skips batches with loss > 100, but by then the weights are already corrupted. Instead: if `loss_v > 5.0` for 3 consecutive batches, reload the last checkpoint and reduce LR by 2x for 5 gens.

6. **Cap value target magnitude at 0.95** instead of 1.0. Extreme targets (exactly +1 or -1) push the network hardest. Soft targets like 0.95/-0.95 give the same gradient direction with less magnitude.

---

## 4. Checkpoint Gating / Best Model Selection

### Current Setup
No gating. Latest checkpoint is always used for next gen's self-play.

### Research Findings

**AlphaGo Zero** used 55% win-rate gating (new model must beat old by 55% to become the self-play model). **AlphaZero** dropped this and just used the latest checkpoint continuously. **KataGo** and **Leela Chess Zero** also use latest-checkpoint (no gating).

The consensus has shifted away from tournament gating because:
- It's computationally expensive (tournament games don't generate training data)
- With noisy self-play data, a model can temporarily regress but recover next gen
- The FIFO buffer naturally dampens regression (old good data is still in the buffer)
- Gating can cause training to stall if the evaluation is noisy

**However**, there's a meaningful middle ground: **checkpoint averaging** (exponential moving average of weights). This smooths out generation-to-generation noise without requiring tournaments.

### Recommendations

1. **Keep no-gating** (use latest checkpoint). The research supports this for your scale.

2. **Add EMA (exponential moving average) weights.** Maintain a shadow copy of weights: `ema_weights = 0.995 * ema_weights + 0.005 * current_weights` after each training step. Use EMA weights for self-play while training the actual weights. This smooths out bad updates without the cost of tournaments.

3. **Add regression detection.** If average game length increases by >50% over 5 gens AND decisive game rate drops by >50%, flag it. Don't auto-rollback, but log a warning.

---

## 5. Game Length and Training Signal

### Current Setup
- Games capped at 30-120 moves (ramped over 20 gens)
- Draw games get value target 0.0
- TD-lambda (0.8) blends MCTS value estimates with game outcome

### Research Findings

Game length directly affects training signal quality. **Positions near the end of a game have much higher-quality value targets** because MCTS can look ahead to terminal states. Early-game positions have noisy targets because they're far from the outcome.

Long, inconclusive games (100+ moves ending in draws) are problematic because:
- Every position gets target ~0.0, which trains the value head to predict 0 for everything
- Policy targets come from MCTS on a nearly random board, so they're low quality
- They dilute the buffer with low-information samples

**KataGo's approach:** Use playout cap randomization so most of these games finish quickly (low sim count), and only a fraction of positions are recorded for training. The value head still learns from them, but the policy head is protected from bad targets.

### Recommendations

1. **Weight positions by proximity to game end.** Positions in the last 20 moves of a decisive game get weight 1.0; positions before that get weight `0.3 + 0.7 * (distance_from_end < 20)`. This focuses learning on positions where the outcome signal is strongest.

2. **Down-weight draw games in the buffer.** Draw games should contribute 50% fewer positions to training than decisive games. Either sub-sample when adding to buffer, or use a sampling weight during training.

3. **Keep the move cap at 120** (or even lower it to 100). There's no evidence that very long games produce useful training signal for Connect6. If the network can't find a win in 100 moves, neither player had a strong position.

4. **Track and log the "decisive ratio"** (fraction of games with a winner). If it drops below 30%, something is wrong with either the game mechanics or the training. Healthy AlphaZero training should show increasing decisiveness over time as the network learns to play tactically.

5. **For early training (first 50 gens), cap at 60 moves.** Early random networks produce garbage after ~30 moves. Capping earlier forces faster buffer turnover with higher-signal positions.

---

## 6. MCTS Simulation Count

### Current Setup
- Target: 200 sims (ramped from 16 over 20 gens)
- Top-K branching: 24

### Research Findings

**AlphaZero** used 800 sims for chess, 1600 for Go. **KataGo** used 600 for full searches and 100 for fast searches (via playout cap randomization). **Gumbel MuZero** demonstrated strong performance with as few as 2-16 simulations, relying on the Gumbel trick for provable policy improvement even at low sim counts.

The key insight: **simulation count has diminishing returns, but the threshold depends on game complexity.** Connect6 on a hex grid has a branching factor comparable to Go in the early game (~100+ legal moves) but with aggressive ZOI pruning you're looking at ~24 moves. With 24 children and 200 sims, each child gets ~8 visits on average. This is marginal for forming reliable value estimates.

**Since you already use Gumbel selection**, you're in the regime where fewer sims work better than with vanilla PUCT. Gumbel's policy improvement guarantee holds even at low sim counts.

### Recommendations

1. **Implement playout cap randomization** (highest-impact sim-related change):
   - Full search: 400 sims, applied to ~20% of moves, only these are recorded for policy training
   - Fast search: 50 sims, applied to ~80% of moves, recorded for value training only
   - This effectively gives you 2x more games at only ~1.3x the compute cost
   - Implementation: for each move, with probability 0.2 use 400 sims (record board + policy + value), otherwise use 50 sims (record board + value only, no policy target)

2. **Ramp sims faster.** Your 20-gen ramp from 16 to 200 is too slow. The first 5 gens with 16-50 sims produce very noisy policy targets. Ramp from 50 to target in 10 gens.

3. **For Gumbel selection specifically**: With Gumbel, 200 sims is adequate for a game with 24-move branching factor. The improvement from going to 400 is modest with Gumbel (unlike vanilla PUCT where it's significant). The playout cap approach gets you the best of both worlds.

---

## 7. Key Innovations Since AlphaZero (2018)

### KataGo (2019-present)
Most impactful innovations for your setup:

- **Playout cap randomization**: Already discussed. This is the single biggest efficiency win - KataGo achieved 50x the sample efficiency of ELF OpenGo partly through this.
- **Auxiliary ownership/score targets**: You already have ownership. KataGo also predicts exact score, opponent's next move, and short-term value targets at 3 different time horizons.
- **Optimistic policy**: A second policy head biased toward finding unexpectedly good moves. Helps exploration.
- **Variance-weighted MCTS**: Scale cPUCT by `sqrt(utility_variance)` at each node. Positions where the net is uncertain get more exploration.
- **Fixup initialization** (later superseded): Removing batch norm and using careful initialization. You already don't use batch norm (good).

### Gumbel MuZero (2022)
- **Provable policy improvement even at low sim counts.** You already use this.
- The key additional technique: **Sequential halving** for move selection. Instead of allocating sims equally, iteratively eliminate low-value moves and redistribute sims to survivors. This is especially valuable with 24-move branching.

### EfficientZero / MuZero Reanalyze (2021)
- **Reanalyze**: Re-run MCTS on stored positions using the latest network to refresh value targets. 80% of MuZero's training used reanalyzed targets.
- For perfect-info games like yours: you only need to re-evaluate the position with the current net's value head (no dynamics model needed). This is computationally cheap - just a forward pass per position.
- **Consistency loss**: Self-supervised loss ensuring temporal consistency of internal representations. Less applicable to perfect-info games where you have the true state.

### Search-Contempt (2025)
- **Hybrid MCTS combining PUCT and Thompson Sampling**: Generates more challenging training positions by biasing self-play toward complex, non-trivial games.
- Claims up to **70 Elo improvement** in training efficiency and suggests training from zero is feasible on consumer GPUs with hundreds of thousands (not millions) of games.
- Directly applicable to your setup and very recent.

### ReZero (2024)
- **Backward-view and entire-buffer reanalyze**: Improved version of MuZero Reanalyze that processes the entire buffer, not just recent data, and uses backward TD targets.

### Recommendations

1. **Implement playout cap randomization** (priority 1 - biggest bang for buck)
2. **Add short-term value targets** at 3 time horizons (6, 16, 50 turns) as auxiliary losses
3. **Implement lightweight Reanalyze**: Every N gens, do a forward pass on a random 10% of the buffer and update value targets. No MCTS needed - just `net.value(encode(position))`.
4. **Explore search-contempt** as a replacement for Dirichlet noise at the root. It should produce more diverse and challenging games.

---

## 8. Auxiliary Losses

### Current Setup
- Ownership head (MSE, weight 0.1): predicts final territory control
- Threat head (BCE, weight 0.1): predicts threats
- Value uncertainty (Gaussian NLL, weight 0.05): predicts value error variance
- Entropy regularization (weight 0.01): keeps policy from collapsing

### Research Findings

**KataGo's auxiliary heads (in order of importance):**
1. **Short-term value targets** (3 exponential averages at ~6, 16, 50 turn horizons): These provide lower-variance feedback than the final game outcome. The key formula: `(1-lambda) * sum_{t'>t} MCTS_value(t') * lambda^(t'-t)`. This is the highest-impact auxiliary loss KataGo uses.
2. **Score prediction** (exact final score, not just win/loss): For Go, this dramatically helps the value head. For Connect6, you could predict "number of moves until win" as an analog.
3. **Opponent's next move prediction**: Forces the network to model the opponent, improving tactical awareness.
4. **Ownership/territory**: You already have this.

**Uncertainty-aware training:** KataGo's uncertainty heads predict *squared* value error relative to short-term MCTS values. This feeds back into MCTS via uncertainty-weighted playouts (positions where the net is uncertain get more exploration budget). This is a proven technique.

**What you have vs. what works:**
- Your uncertainty head predicts general value variance. KataGo's predicts *specific* squared error relative to short-term lookahead. The KataGo version is more actionable because it has a concrete ground truth (actual error vs. MCTS), while general variance is harder to supervise.

### Recommendations

1. **Add short-term value targets** (highest-impact auxiliary change). During self-play, for each position t, compute:
   - `v_6 = (1-0.15) * sum MCTS_value(t') * 0.15^(t'-t)` for t' > t (approximately 6-turn horizon)
   - `v_16 = (1-0.06) * sum MCTS_value(t') * 0.06^(t'-t)` (~16-turn horizon)
   Store these with each position. Add two auxiliary value heads that predict these, weighted at 0.15 each.

2. **Add "moves until win" prediction head.** A single scalar head that predicts how many moves until the game ends. Train with MSE. Weight: 0.05. This gives the network a sense of urgency that the binary win/loss signal doesn't convey.

3. **Refine uncertainty head.** Instead of predicting general variance, predict `(v_predicted - v_6_actual)^2` - the squared error between the value head's prediction and the 6-turn lookahead value. This gives a concrete, measurable target.

4. **Add opponent move prediction.** A second policy head that predicts the opponent's response. Weight: 0.05. Cheap to add, meaningful for tactical awareness.

5. **Keep entropy regularization at 0.01** but consider *increasing* it to 0.02-0.03 in early training (first 50 gens) when exploration is most valuable, then decaying back to 0.01.

---

## Detailed Implementation Priorities

### Phase 1: Stability Fixes (do immediately, before next training run)

1. **Switch to WDL value head** (prevents value explosion)
2. **Lower LR to 2e-4**, reduce weight decay to 3e-5
3. **Add loss spike rollback** (if loss_v > 2.0 for 3 consecutive batches, reload last checkpoint)
4. **Cap value targets at +/-0.95** instead of +/-1.0

### Phase 2: Efficiency Gains (next 1-2 days)

5. **Implement playout cap randomization** (400 full / 50 fast, 20% full ratio)
6. **Increase buffer to 300K**, switch to exponential decay sampling
7. **Add short-term value targets** (6-turn and 16-turn horizons)
8. **Switch to cosine warm restarts** (T_0=20, T_mult=2)

### Phase 3: Advanced Techniques (next week)

9. **Lightweight Reanalyze** (re-evaluate 10% of buffer with current net every gen)
10. **EMA weights** for self-play (decay 0.995)
11. **Search-contempt** (Thompson sampling hybrid MCTS)
12. **Position-stage weighted training** (prioritize late-game positions)

---

## Sources

- [KataGo Methods Documentation](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- [Accelerating Self-Play Learning in Go (KataGo paper)](https://arxiv.org/pdf/1902.10565)
- [OpenSpiel AlphaZero Documentation](https://openspiel.readthedocs.io/en/stable/alpha_zero.html)
- [AlphaZero.jl Training Parameters](https://jonathan-laurent.github.io/AlphaZero.jl/dev/reference/params/)
- [Leela Chess Zero WDL Head](https://lczero.org/blog/2020/04/wdl-head/)
- [Leela Chess Zero Neural Network Topology](https://lczero.org/dev/backend/nn/)
- [Policy Improvement by Planning with Gumbel (Gumbel MuZero)](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)
- [MiniZero: Comparative Analysis of AlphaZero and MuZero](https://arxiv.org/html/2310.11305v3)
- [Search-Contempt: Hybrid MCTS for Training Efficiency (2025)](https://arxiv.org/abs/2504.07757)
- [EfficientZero: Mastering Atari with Limited Data](https://openreview.net/pdf?id=OKrNPg3xR3T)
- [ReZero: Backward-view and Entire-buffer Reanalyze (2024)](https://arxiv.org/abs/2404.16364)
- [ELF OpenGo: Analysis and Reimplementation of AlphaZero](https://arxiv.org/pdf/1902.04522)
- [Policy or Value? Loss Function and Playing Strength in AlphaZero](https://liacs.leidenuniv.nl/~plaata1/papers/CoG2019.pdf)
- [LightZero: Unified MCTS Benchmark (NeurIPS 2023)](https://github.com/opendilab/LightZero)
- [Value Targets in Off-policy AlphaZero](https://ala2020.vub.ac.be/papers/ALA2020_paper_18.pdf)
- [Expediting Self-Play Learning in AlphaZero-Style Agents](https://skemman.is/bitstream/1946/44332/1/Expediting_Self_Play_Learning_in_AlphaZero_Style_Game_Playing_Agents.pdf)
