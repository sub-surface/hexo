# HexGo

An AlphaZero-style self-play system for an infinite hexagonal Connect6 variant played on the Eisenstein integer ring **Z[ω]**. The goal is 6 consecutive pieces along any of the three hex axes. Turn rule: P1 places 1 stone on turn 1, both players place 2 per turn thereafter (1-2-2).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Loop (train.py)                   │
│                                                                 │
│  ┌──────────────┐     positions     ┌───────────────────────┐  │
│  │  Rust Batched │ ────────────────► │   Replay Buffer       │  │
│  │  Self-Play    │                   │   (FIFO, 100k cap)    │  │
│  │  (lockstep)   │                   └──────────┬────────────┘  │
│  └──────┬───────┘                              │ sample        │
│         │ eval callback                         ▼               │
│         ▼                             ┌─────────────────────┐  │
│  ┌──────────────┐  batched GPU   ┌───►│  Training Step      │  │
│  │  Python GPU  │ ◄──────────────┤   │  WDL value (CE) +   │  │
│  │  Eval        │                │   │  policy + aux + unc  │  │
│  │  (PyO3)      │                │   └──────────┬──────────┘  │
│  └──────────────┘                │              │ weights      │
│         │                        │              ▼               │
│         │                   ┌────┘   ┌─────────────────────┐  │
│         └───────────────────┘        │  HexNet (~1.9M par) │  │
│                  weights sync        └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       HexNet (net.py)                           │
│                                                                 │
│  Input [17, 18, 18]                                             │
│    → HexConv2d(17→128) + BN + ReLU         [hex-masked stem]   │
│    → 6× ResBlock(128ch, HexConv2d)          [trunk]             │
│    → GlobalPoolBranch                       [board context]     │
│    ├→ WDL value head     → [3] softmax (win/draw/loss)         │
│    ├→ uncertainty head   → σ² (Gaussian NLL)                   │
│    ├→ ownership aux head → [18,18] ∈ (-1,1)                    │
│    ├→ threat aux head    → [18,18] ∈ (0,1)                     │
│    └→ policy head        → [18,18] logit map                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Rust Engine (hexgo-rs/)                        │
│                                                                 │
│  PyO3 + maturin crate — 3-4x speedup over Python MCTS          │
│    → game.rs:     HexGame engine (make/unmake, win detection)   │
│    → encode.rs:   Board encoding (17ch), ZOI-filtered top-K     │
│    → batched.rs:  Lockstep MCTS with Python GPU eval callback   │
│    → node.rs:     Arena-allocated MCTS tree                     │
│    → Rayon parallel board encoding                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Dashboard (app.py)                             │
│                                                                 │
│  Browser UI (dashboard.html) + Mobile (/mobile)                 │
│    ↕ REST + SSE                                                 │
│  FastAPI backend (server.py)                                    │
│    → start/stop training, stream metrics, serve replays         │
│    → GIF export, loss derivative display                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

**Requirements:** Python 3.12, PyTorch with CUDA, Rust toolchain (for hexgo-rs).

```bash
# Install Python dependencies
pip install torch torchvision fastapi uvicorn numpy

# Build Rust engine
cd hexgo-rs && maturin build --release && pip install target/wheels/*.whl && cd ..

# Run training
bash run.sh train.py --gens 200 --sims 200 --games 128

# Run the dashboard
bash run.sh app.py
# Opens http://127.0.0.1:7860 — mobile view at /mobile

# Run tests
pytest tests/ -v   # 36 tests (25 game + 11 Rust parity)
```

> **Windows note:** Always use `bash run.sh <script>` — bare `python` may pick up the wrong interpreter.

---

## The Game

**AP-6 Maker-Maker on Z[ω]** — Connect6 on an infinite hexagonal grid.

- **Win condition:** 6 consecutive pieces along any of the 3 Z[ω] unit axes (q-axis, r-axis, diagonal)
- **Turn rule:** P1 places 1 stone on turn 1; both players place 2 per turn thereafter
- **Board:** Infinite grid; the 18x18 window is centered on the centroid of the last 20 moves
- **Coordinate system:** Axial (q, r) — isomorphic to the Eisenstein integers Z[ω]
- **Key tactical insight:** An uncontested 4-chain is a forced win (2 stones/turn → extend to 6)

The Z[ω] framing drives three engineering choices:

| Choice | What it does |
|--------|-------------|
| `HexConv2d` | Masks the 2 non-adjacent corners of 3x3 kernels — Z[ω]-faithful receptive field |
| D6 augmentation | 12 symmetry transforms per training sample = 12x free data diversity |
| `EisensteinGreedyAgent` | Erdos-Selfridge potential maximizer; permanent ELO anchor |

---

## Key Files

| File | Role |
|------|------|
| `game.py` | HexGame engine: make/unmake, win detection, ZOI pruning |
| `net.py` | HexNet: HexConv2d trunk, WDL value head, policy head, aux heads, D6 augmentation |
| `mcts.py` | MCTS: PUCT selection, 1-2-2 backprop sign convention, Gumbel root selection |
| `train.py` | Training loop: Rust batched self-play, WDL cross-entropy, TD-lambda, ZOI curriculum |
| `hexgo-rs/` | Rust engine: game, MCTS tree, board encoding, batched self-play (PyO3/maturin) |
| `elo.py` | ELO system: NetAgent, EisensteinGreedyAgent, run_match |
| `config.py` | All tunable hyperparameters |
| `server.py` | FastAPI backend: REST + SSE stream |
| `dashboard.html` | Dark-mode dashboard: Training / Replay / Config tabs |
| `mobile.html` | Mobile monitoring page with charts and log |
| `play.py` | Tkinter GUI for playing against any checkpoint |
| `autoresearch/` | Karpathy-style autonomous experiment loop |

---

## Configuration

All hyperparameters in `config.py`. Key train.py overrides:

| Key | Value | Notes |
|-----|-------|-------|
| `LR` | 2e-4 | Adam learning rate (cosine schedule) |
| `SIMS` | 200 | MCTS simulations per move (CLI arg) |
| `BATCH_SIZE` | 512 | Training batch size |
| `BUFFER_CAP` | 100,000 | Replay buffer capacity |
| `TRUNK_BLOCKS` | 6 | Residual blocks (restart required) |
| `TRUNK_CHANNELS` | 128 | Hidden channels (restart required) |
| `CPUCT` | 2.0 | PUCT exploration constant |
| `TOP_K` | 24 | Policy branching factor |
| `ZOI_MARGIN` | 4→5 | ZOI curriculum over 30 gens |
| `MAX_MOVES_MAX` | 120 | Game length cap |

---

## Training Signal

Each generation:

1. **Self-play:** 128 games via Rust lockstep MCTS with ZOI-restricted move selection
2. **Replay buffer:** positions stored with board encoding, policy targets, legal masks, WDL value targets (TD-lambda)
3. **Training:** batches sampled with recency weighting (75/25); D6 augmented on the fly
4. **Loss:** WDL cross-entropy (value) + masked cross-entropy (policy) + ownership/threat aux + uncertainty + entropy reg
5. **ELO eval:** net vs EisensteinGreedyAgent every 10 gens
6. **Metrics:** all loss components + games/s + decisive ratio appended to `metrics.jsonl`

---

## AutoResearch

Karpathy-style autonomous experiment loop in `autoresearch/`:
- `program.md` — agent instructions for the never-stop loop
- `run_trial.py` — fixed-budget trial runner (10 gens + ELO eval)
- `experiments_queue.md` — 9 prioritized experiments (Tier 2 + 3)
- `results.tsv` — experiment log with keep/discard decisions

See `docs/TRAINING_RESEARCH.md` for the full state-of-the-art analysis.

---

## Tests

```bash
pytest tests/ -v   # 36 tests (25 game + 11 Rust parity)
```

Covers: win detection on all 3 axes, undo correctness, D6 symmetry, EisensteinGreedyAgent, Rust/Python encoding parity.
