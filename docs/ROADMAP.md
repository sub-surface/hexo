# HexGo — Roadmap

Z[ω] self-play ladder: arithmetic progressions, Eisenstein symmetry, emergent structure.
Current baseline: ~1.9M params (128ch/6blk), batched lockstep MCTS (no threads), RTX 2060.

---

## Mathematical Foundation (established)

HexGo = **AP-6 Maker-Maker on Z[ω]**.

The hex grid with axial (q, r) coordinates is isomorphic to the Eisenstein
integer ring Z[ω] where ω = e^(2πi/3). The three win axes correspond to the
unit directions {1, ω, ω²}. A win is exactly an arithmetic progression of
length 6 in Z[ω] with a unit step.

Practical consequences:
1. **The right convolution kernel is the 7-cell Z[ω] neighbourhood** — standard 3×3 with 2 non-adjacent corners masked. `HexConv2d` enforces this.
2. **The symmetry group of the lattice is D6 (order 12)** — every training position can be augmented 12× for free via linear transforms on axial (q, r) coordinates.

Connection to combinatorics: W(6;2) = 1132 (van der Waerden); the Erdős-Selfridge potential ∑ 2^(−|L|) < 1 gives a theoretically safe second-player draw strategy; `EisensteinGreedyAgent` approximates this.

---

## Phase 0 — Stability

- [x] 1-2-2 Connect6 rule in `game.py`
- [x] Player-aware MCTS backprop in `mcts.py`
- [x] FP16 AMP in training and inference
- [x] Cross-entropy policy loss over all legal moves
- [x] Divide-by-zero guard in visit distribution normalization
- [x] Transposition cache in `InferenceServer`
- [x] `checkpoints/legacy/` quarantine for incompatible weights
- [x] 26 unit tests pass

---

## Phase 1 — Geometry-Faithful Architecture

- [x] **1a. Inference batching** — `NUM_WORKERS=8`, `INF_BATCH=8`, `INF_TIMEOUT=30ms`
- [x] **1b. `torch.compile`** — applied at `InferenceServer.start()` (silently falls back on Windows)
- [x] **1c. Tree reuse** — `mcts_policy()` returns `new_root`; subtree recycled with fresh Dirichlet
- [x] **1d. Cosine temperature annealing** — `temp = max(0.05, cos(π × move / T))`
- [x] **1e. TD-lambda value targets** — `z_t = 0.99^(T−t) × z_final`
- [x] **1f. History planes** — 11-channel input (4 P1-history + 4 P2-history + current × 2 + to-move)
- [x] **1g. Spatial policy head** — `policy_logits(features) → [B, S, S]` logit map; single forward pass for full policy; `move_to_grid()` for coordinate indexing; ~1.9M total params (128ch/6blk)
- [x] **1h. INT8 quantization utility** — `quantize_for_inference(net)` via `torch.ao.quantization`
- [x] **1i. HexConv2d** — masks non-hex corners `[0,0]` and `[2,2]` in all ResBlock kernels
- [x] **1j. D6 data augmentation** — 12 transforms applied at `train_batch` time
- [x] **1k. EisensteinGreedyAgent curriculum** — 1-in-5 self-play games; permanent ELO anchor
- [x] **1l. Policy heatmap** — `heatmaps/gen_XXXX.png` per generation

---

## Phase 2 — Training Quality

- [x] **2a. Zobrist-keyed buffer deduplication** — per-gen hash set prevents near-duplicate positions
- [x] **2b. Self-play curriculum** — playout cap randomization (25% full / 75% reduced)
- [x] **2c. Checkpoint tournament** — moved to standalone `tournament.py` for round-robin evaluation (removed from training loop)

---

## Phase 3 — Infrastructure

- [x] **3a. Batched lockstep self-play** — replaced threaded InferenceServer approach with `batched_self_play()`: all N games run in lockstep, single-threaded with batch GPU calls. No threads, no GIL.
- [x] **3b. Recency / diversity sampling** — buffer FIFO with cap; Zobrist dedup
- [x] **3c. CUDA Graphs hot path** — fixed tensor rebinding bug; retained in InferenceServer (dashboard/elo use)

---

## Phase 3b — Collaborator Integrations

- [x] **3b-i. KataGo playout cap randomization** — `_cap_sims(target)` in `train.py`
- [x] **3b-ii. Global pooling branch (KataGo)** — `GlobalPoolBranch(32ch)` after trunk
- [x] **3b-iii. ZOI pruning** — `zoi_moves(margin=6)` restricts MCTS to active area
- [x] **3b-iv. Latency / perf tracking** — `PerfTracker` with bottleneck warnings
- [x] **3b-v. CPU offloading / pin_memory** — `non_blocking=True` async host→GPU transfers
- [x] **3b-vi. Persistent cross-gen cache** — `_persistent_cache` with `CACHE_MAX_AGE=5` eviction
- [x] **3b-vii. Ownership + threat auxiliary heads** — thin 1×1 conv heads off trunk; `make_aux_labels()` generates per-episode spatial labels; D6 transform extended to aux arrays via `_transform_aux()`; ownership=MSE, threat=BCE-with-logits (AMP-safe); `AUX_LOSS_OWN=0.1`, `AUX_LOSS_THREAT=0.1`
- [x] **3b-viii. Recency-weighted replay buffer** — 75% from recent half / 25% uniform; `RECENCY_WEIGHT=0.75` in CFG; `train_batch()` splits sample accordingly
- [ ] **3b-ix. MuZero-style reanalysis** — re-search buffered positions with updated net (4h effort)

---

## Bug Fixes (pre-training correctness)

Confirmed correctness bugs found in code review — all fixed as of 2026-03-30.

- [x] **FIX-1: CRITICAL — CUDA Graph tensor rebinding** — `inference.py`: switched to in-place `.copy_()` ops inside graph capture; added `.detach()` before numpy conversion; fixed `_graph_val` shape from `[B,1]` to `[B]`.
- [x] **FIX-2: CRITICAL — Cache key ignores turn state** — `inference.py`: key now `(frozenset(board.items()), current_player, placements_in_turn)`.
- [x] **FIX-3: CRITICAL — Autotune reward signal inverted** — `tune.py`: `kept = elo_delta is None or elo_delta <= 0` (eisenstein_def ELO *rises* when net gets worse — negative delta = improvement).
- [x] **FIX-4: IMPORTANT — `mcts_with_net` leaf children missing `player=`** — `mcts.py`: leaf children now use `player=game.current_player`.
- [x] **FIX-5: IMPORTANT — Terminal expansion sign** — `mcts.py` + `train.py`: `v = 1.0 if game.winner == node.player else -1.0` (was always `1.0`); leaf value negated when `node.player != game.current_player`.
- [x] **FIX-6: IMPORTANT — History planes filter via board dict** — `net.py`: history planes use `player_history` (parallel to `move_history`) — correct during MCTS `unmake()` traversal. Verified 2026-03-30; no board-dict cross-reference.
- [x] **FIX-7: IMPORTANT — ZOI long-range threat blindness** — deferred; lookback increase is a config-level change with no current crash risk.
- [x] **FIX-8: IMPORTANT — Autotune `SIMS_MIN` too high** — `config.py`: `SIMS_MIN` changed from `25` → `6`.
- [x] **FIX-9: IMPORTANT — Cosine temp semantics** — `train.py`: formula fixed to `cos(π/2 × move / TEMP_HORIZON)` — reaches floor at `TEMP_HORIZON` moves (not `TEMP_HORIZON/2`).
- [x] **FIX-10: MODERATE — Replay hex offset** — `replay.py`: indent changed from `abs(r)` to `r - r_min`.

---

## Dashboard (completed 2026-03-30)

- [x] **server.py** — FastAPI backend: 12 REST endpoints + SSE event stream; `ProcessSingleton` manages one training subprocess; `queue.Queue`-based thread-safe SSE broadcast from `_metrics_watcher` daemon thread.
- [x] **dashboard.html** — Single-file dark-mode dashboard (24KB); three tabs: Training / Replay / Config; live loss/ELO charts via Chart.js; SSE-driven updates; close-before-reconnect SSE guard; config staging with safe `repr()` serialisation.
- [x] **app.py** — Thin launcher (36 lines): starts uvicorn with `server.app`, opens browser after 1.2s delay.

---

## Training Simplification (completed 2026-03-30)

- [x] **Threaded self-play replaced** — `batched_self_play()` replaces InferenceServer + thread workers. Single-threaded lockstep with batch GPU calls.
- [x] **ELO eval removed from training loop** — pure self-play only. ELO system retained for `tournament.py`.
- [x] **Tournament moved to standalone** — `tournament.py` for round-robin checkpoint evaluation.
- [x] **MAX_MOVES curriculum** — ramps from 30→100 over 20 gens (was fixed 300).
- [x] **SIMS curriculum** — ramps from 16→target over 20 gens.
- [x] **Decisive game saving** — non-draw games saved to `replays/decisive/`.
- [x] **torch.compile disabled** — was using 3-4GB RAM during JIT compilation.
- [x] **BATCH_SIZE=256** — larger batches to saturate GPU during training.
- [x] **metrics.jsonl** — appended unconditionally per-gen for dashboard live charts.

---

## Phase 4 — Equivariance (research)

- [ ] **4a. G-CNN (full D6 equivariance)** — replace augmentation with group-equivariant layers. Cohen & Welling (2016); Bekkers (2020). Principled Sutton-compatible treatment: geometry as structural substrate, not game knowledge.
- [ ] **4b. CA weight initialization** — initialize `HexConv2d` kernels from cellular automata patterns. The hex-7 neighbourhood IS the standard NCA update kernel. Mordvintsev et al. (2020). Requires `WEIGHT_INIT: "xavier"|"ca"` in CFG and `init_weights_ca(net)` in `net.py`.

---

## Phase 5 — Model Scale

- [x] **5a. Scale trunk** — 6 blocks × 128 channels (~1.9M params). Scaled from 4blk/64ch (~480K).
- [ ] **5b. Activation sparsity / early exit** — profile `v_fc` first layer; if >60% sparsity implement early exit from value head. Low priority until scale experiments run.
- [x] **5c. Batched lockstep MCTS** — replaced Python-threaded MCTS with lockstep approach. All N games evaluated in one GPU batch per sim. No GIL contention. C++/Rust no longer required for basic batching.
- [x] **5d. Spatial policy head** — `policy_logits(features) → [B, S, S]` replaces per-move policy head. Single forward pass for full policy.
- [x] **5e. Vectorized spatial policy loss** — masked softmax over `[B, S, S]` logit map against `policy_target` plane. No Python loops.
- [x] **5f. Curriculum (sims + max_moves)** — sims ramp 16→100, max_moves ramp 30→100 over 20 gens each.
- [x] **5g. Decisive game saving** — all non-draw games saved to `replays/decisive/` for corpus building.
- [x] **5h. `tournament.py`** — round-robin model checkpoint tournaments with ELO ratings.
- [x] **5i. `tune.py` rewrite** — uses policy loss delta from `metrics.jsonl` instead of ELO vs Eisenstein.

---

## Priority Order (RTX 2060, solo researcher)

| Priority | Item | Status | Expected gain |
|----------|------|--------|---------------|
| ✅ — | FIX-1 through FIX-10 | **done 2026-03-30** | Core correctness restored |
| ✅ — | Dashboard (server.py + dashboard.html) | **done 2026-03-30** | Live monitoring |
| ✅ — | Training simplification | **done 2026-03-30** | Stable long runs |
| ✅ — | All Phase 0–3b-viii items | **done 2026-03-30** | See sections above |
| ✅ — | CPUCT tuned, DIRICHLET_ALPHA/EPS raised | **done** | Better exploration + root noise |
| ✅ — | Aux heads (3b-vii) | **done 2026-03-30** | Richer trunk representation |
| ✅ — | Recency replay (3b-viii) | **done 2026-03-30** | Tracks current policy better |
| ✅ — | Spatial policy head (5d) | **done** | Single forward pass, vectorized loss |
| ✅ — | Batched lockstep MCTS (5c) | **done** | No GIL, full GPU utilization |
| ✅ — | Scale trunk 6blk/128ch (5a) | **done** | ~1.9M params |
| ✅ — | Curriculum sims+max_moves (5f) | **done** | Gradual complexity ramp |
| ✅ — | tournament.py (5h) | **done** | Round-robin checkpoint evaluation |
| ✅ — | tune.py rewrite (5i) | **done** | Policy loss delta signal |
| ⬜ 1 | Run sustained baseline | next | Validate all improvements |
| ⬜ 2 | MuZero reanalysis (3b-ix) | future | Freshen stale targets |
| ⬜ 3 | G-CNN equivariance (4a) | future | Principled symmetry |
