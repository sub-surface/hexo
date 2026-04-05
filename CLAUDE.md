# HexGo — Claude Context

## CRITICAL RULES

- **NEVER delete `train.lock`.** It is an OS-level file lock held by the running training process. Deleting it can corrupt training or cause duplicate processes. If you see "Another training process is already running", that means training IS running — do not try to work around the lock.

## What this project is

HexGo is an AlphaZero-style self-play system for an infinite hexagonal Connect6 variant
played on the Eisenstein integer ring Z[ω]. The win condition is 6 consecutive pieces
along any of the three Z[ω] unit axes. Turn rule: P1 places 1 stone on turn 1, both
players place 2 per turn thereafter (1-2-2).

The mathematical framing is not cosmetic — it drives three concrete engineering choices:
- `HexConv2d`: masks the two non-adjacent corners `[0,0]` and `[2,2]` of 3×3 kernels
- D6 augmentation: 12 transforms = full dihedral symmetry group of the Eisenstein lattice
- `EisensteinGreedyAgent`: Erdős-Selfridge potential maximizer; used as curriculum opponent and permanent ELO anchor

---

## Python interpreter

**Always use `C:\Users\landa\AppData\Local\Programs\Python\Python312\python.exe`** — this is the only install with
torch+CUDA, fastapi, and uvicorn. Running `python` or `py` from PowerShell will pick up
the wrong interpreter and fail with `ModuleNotFoundError`.

- **bash/WSL**: use `bash run.sh <script.py>` (already hard-codes the path)
- **PowerShell**: use `run.bat <script.py>` or `& "C:\Users\landa\AppData\Local\Programs\Python\Python312\python.exe" <script.py>`

---

## How to run

**Training:**
```bash
bash run.sh train.py --gens 200 --sims 200 --games 128
```

**Dashboard (recommended):**
```bash
bash run.sh app.py
# Opens http://127.0.0.1:7860 in browser
# Controls training via Start/Stop buttons, live loss/ELO charts, replay viewer
```

**Autotune:**
```bash
bash run.sh tune.py --trials 10 --gens 5
```

**Replay viewer (terminal):**
```bash
bash run.sh replay.py replays/game_first_gen0001_20260329_160347.json
```

---

## Key files

| File | Role |
|------|------|
| `game.py` | HexGame engine: make/unmake, win detection, ZOI pruning |
| `net.py` | HexNet: HexConv2d trunk, value/policy heads, D6 augmentation |
| `mcts.py` | MCTS: pure rollout + AlphaZero modes, `_backprop` with 1-2-2 sign convention |
| `inference.py` | InferenceServer: batched GPU eval with CUDA Graphs + persistent cache |
| `train.py` | Training loop: Rust batched self-play, sequential training, TD-lambda targets, ELO eval |
| `elo.py` | ELO system: NetAgent, EisensteinGreedyAgent, run_match |
| `config.py` | Single CFG dict: all tunable hyperparameters |
| `tune.py` | Autotune: random trial orchestrator, reward = Eisenstein winrate decrease |
| `replay.py` | Terminal hex-grid replay renderer |
| `server.py` | FastAPI backend: 12 REST endpoints + SSE event stream |
| `dashboard.html` | Single-file dark-mode dashboard: Training/Replay/Config tabs |
| `app.py` | Thin launcher: starts uvicorn, opens browser |
| `hexgo-rs/` | Rust engine (PyO3/maturin): game, MCTS, board encoding, batched self-play |

---

## Architecture decisions

### MCTS backprop sign convention
Value is from `node.player`'s perspective (+1 = node.player wins). Sign is only negated
when `node.parent.player != node.player` — this correctly handles the 1-2-2 rule where
the same player makes two consecutive placements without a sign flip between them.

`evaluate()` (InferenceServer/net) returns value from `game.current_player`'s POV.
At leaf nodes: if `node.player != game.current_player`, negate before backprop.

### Training self-play vs. ELO evaluation
- `rust_batched_self_play()` = primary training path; Rust lockstep MCTS with Python GPU eval callback
- Falls back to Python `batched_self_play()` if Rust import fails
- `mcts_with_net()` in mcts.py = used only in ELO `NetAgent` matches (unbatched)
- `mcts()` = pure rollout; used in old `MCTSAgent` (removed from default eval path)

### Rust engine (`hexgo-rs/`)
PyO3 + maturin crate providing ~3-4x speedup over Python MCTS. Modules:
- `game.rs` — HexGame engine (make/unmake, win detection, ZOI)
- `encode.rs` — Board encoding (17ch × 18×18); fast variant skips axis-chain planes for leaf eval
- `batched.rs` — Lockstep batched self-play: all N games advance one sim together, batch leaf positions into one GPU call via Python callback. Uses Rayon `par_iter` for board encoding.
- `node.rs` / `mcts.rs` — MCTS tree traversal and node management

### Replay buffer persistence
Buffer uses separate `.npy` files in `checkpoints/buffer/` (not monolithic `.npz`).
Loaded via `mmap_mode='r'` to avoid OOM when CUDA is active. Saved every generation.
Legacy `.npz` fallback exists at `checkpoints/replay_buffer.npz` but is no longer written.

### Training pipeline
Sequential (not overlapped): self-play → training → checkpoint → next gen.
Overlapped training was attempted but reverted due to cuDNN thread-safety issues.
torch.compile is disabled (memory pressure). Autocast (fp16) used in eval callbacks
but NOT in training (CA-init weights overflow in fp16).

### Checkpoint tournament (REMOVED 2026-03-30)
`_tourney_promote()` was removed. Loading `net_gen*.pt` files saved before
`torch.compile` into an `OptimizedModule` wrapper crashes with `RuntimeError:
Error(s) in loading state_dict`. Old checkpoints are in `checkpoints/legacy/`.

### Dashboard thread safety
SSE uses `queue.Queue` (not `asyncio.Queue`) for broadcast from the `_metrics_watcher`
background thread. The SSE endpoint polls with `q.get_nowait()` + `asyncio.sleep(0.25)`.
Never switch back to `asyncio.Queue` — it is not thread-safe from non-async threads.

---

## Config (`config.py`)

```python
CFG = {
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "BATCH_SIZE": 64,            # overridden to 512 in train.py
    "SIMS": 100,
    "SIMS_MIN": 25,
    "CAP_FULL_FRAC": 0,
    "GUMBEL_SELECTION": True,    # Gumbel argmax root selection (vs softmax-temp sampling)
    "CPUCT": 2.0,                # research target 2.0–2.5
    "DIRICHLET_ALPHA": 0.10,     # ~10/|ZoI|
    "DIRICHLET_EPS": 0.25,
    "ZOI_MARGIN": 5,
    "ZOI_LOOKBACK": 16,          # recent moves defining ZOI focus
    "TRUNK_BLOCKS": 6,           # residual blocks
    "TRUNK_CHANNELS": 128,       # hidden channels — ~1.9M params total
    "WEIGHT_INIT": "ca",         # "ca" = hex NCA Laplacian priors | "xavier" = standard
    "TD_GAMMA": 0.99,
    "TEMP_HORIZON": 40,
    "WEIGHT_SYNC_BATCHES": 20,
    "RECENCY_WEIGHT": 0.75,      # fraction of each batch from recent half of buffer
    "AUX_LOSS_OWN": 0.1,         # ownership head loss weight
    "AUX_LOSS_THREAT": 0.1,      # threat head loss weight
    "UNC_LOSS_WEIGHT": 0.05,     # value uncertainty head (Gaussian NLL) loss weight
    "VALUE_LOSS_WEIGHT": 1.0,    # multiplier on value loss (now WDL cross-entropy)
    "ENTROPY_REG": 0.01,         # policy entropy regularization bonus weight
}
```

**train.py overrides:** `BATCH_SIZE=512`, `LR=2e-4`, `WEIGHT_DECAY=3e-5`, `SIMS_MIN=16`, `SIMS_RAMP=20`,
`TOP_K=24`, `BUFFER_CAP=100_000`, `MAX_MOVES_MAX=120`, training batches capped at 150/gen.
ZOI curriculum: `ZOI_MARGIN_MIN=4` → `ZOI_MARGIN_MAX=5` over 30 gens (forces compact play early).

`CPUCT` and `TRUNK_*` are loaded at module import time — process restart required to change them.

---

## Test suite

```bash
pytest tests/ -v   # 36 tests (25 game + 11 Rust parity), all should pass
```

Tests cover: win detection on all 3 axes, undo correctness, D6 symmetry, EisensteinGreedyAgent,
autotune pipeline end-to-end, Rust/Python encoding parity.

---

## Current status (2026-04-03)

Training pipeline running on RTX 5070 Ti (16GB VRAM) with Rust batched self-play.
~1.5-2.5 min/gen at 200 sims with ZOI curriculum. All correctness bugs resolved.

**Completed 2026-03-30 (session 2):**
- `CPUCT` raised 1.0 → 2.0; `DIRICHLET_ALPHA` reduced 0.3 → 0.10
- Recency-weighted replay buffer: 75% recent half / 25% uniform per batch
- Auxiliary heads: ownership + threat (thin 1×1 convs off trunk)
- Per-component loss tracking + move accuracy metric in `metrics.jsonl`

**Completed 2026-03-31 (session 3):**
- Board window centers on centroid of last 20 moves (`N_RECENT=20`)
- Trunk scaled to 6 blocks × 128 channels — ~1.9M params
- CA weight init, value uncertainty head, Gumbel root selection, entropy regularization

**Completed 2026-04-02 (session 4):**
- Rust batched self-play (`hexgo-rs/src/batched.rs`): 3-4x speedup over Python MCTS
  - Lockstep MCTS with Python GPU eval callback; Rayon parallel board encoding
  - Board encoding ported to Rust (`encode.rs`): full 17ch + fast variant (skips axis-chains)
- Replay buffer switched from monolithic `.npz` to separate `.npy` files with mmap loading
- Buffer saved every generation (was every 5)
- Training batches capped at 150/gen
- Removed Eisenstein bias from `play.py` MCTS calls
- torch.compile disabled; autocast disabled in training (CA-init overflow)
- Overlapped training reverted (cuDNN thread-safety issues)

**Completed 2026-04-03 (session 5):**
- **WDL value head**: Replaced single MSE scalar with 3-class softmax (win/draw/loss)
  trained with cross-entropy. Bounded by design — eliminates value head explosions.
  Scalar value derived as P(win) - P(loss) for MCTS compatibility.
- **Post-search value estimate**: Rust batched self-play now stores the MCTS root value
  (avg child value after all sims) instead of pre-search net prediction. Fixes stale
  TD-lambda bootstrap targets.
- **ZOI curriculum for self-play**: ZOI_MARGIN ramps from 4→5 over 30 gens. Forces
  compact, tactical play early (prevents scattered-play plateau). Integrated into Rust
  via `top_k_from_logit_map_zoi()` with fallback if ZOI is too restrictive.
- **LR reduced to 2e-4** (was 5e-4, originally 1e-3). Weight decay reduced to 3e-5.
- **Loss spike guard**: Batches with loss > 100 skipped before backward pass.
- **LR scheduler resume**: Fast-forwards on restart to prevent warmup reset.
- **AutoResearch infrastructure**: Karpathy-style autonomous experiment loop in
  `autoresearch/` — program.md, run_trial.py, results.tsv, experiments_queue.md.
  9 prioritized experiments (Tier 2 + 3) ready to run once WDL head converges.
- **Mobile monitoring page** at `/mobile` — real-time metrics, charts, ELO, log tail.
- **Dashboard enhancements**: Loss derivative in chart headers, GIF export for replays.

**Remaining open items:**
- AutoResearch loop activation (pending WDL convergence)
- Tier 2 experiments: TOP_K curriculum, cosine warm restarts, buffer 300K, ELO eval 12 games
- Tier 3: short-term value aux targets, EMA weights, lightweight Reanalyze
- G-CNN full D6 equivariance (deferred — major rewrite)

See `docs/ASSESSMENT.md`, `docs/ROADMAP.md`, and `docs/TRAINING_RESEARCH.md` for details.
