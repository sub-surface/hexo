# HexGo — Design Overview

This document provides the top-level mathematical framing and component index.
Each component has its own detailed doc.

## Component Docs

| Component | File | Description |
|-----------|------|-------------|
| Game engine | [GAME.md](GAME.md) | Rules, Z[ω] axioms, make/unmake, ZOI, candidates |
| MCTS | [MCTS.md](MCTS.md) | Two modes, multi-placement backprop, tree reuse, known bugs |
| Neural net | [NET.md](NET.md) | HexConv2d, D6 augmentation, GlobalPoolBranch, spatial policy head |
| Inference server | [INFERENCE.md](INFERENCE.md) | Dynamic batching, transposition cache (used by dashboard/elo, not training) |
| Training | [TRAINING.md](TRAINING.md) | Batched lockstep self-play, spatial policy loss, curriculum, D6 at train time |
| ELO system | [ELO.md](ELO.md) | Agents, rating mechanics, tournament.py round-robin |
| Autotune | [AUTOTUNE.md](AUTOTUNE.md) | Hyperparameter search, tune.py (policy loss delta), config.py |
| Roadmap | [ROADMAP.md](ROADMAP.md) | Checkbox status for all phases |
| Assessment | [ASSESSMENT.md](ASSESSMENT.md) | Honest code review findings and priority fix list |
| Research | [research/](research/) | Literature synthesis across 7 research docs |

---

## Mathematical Framework: AP-6 Maker-Maker on Z[ω]

### Eisenstein Integer Isomorphism

The hex grid with axial coordinates (q, r) is isomorphic to the Eisenstein
integer ring **Z[ω]** where ω = e^(2πi/3). Each cell maps to q + r·ω ∈ Z[ω].

The three win axes correspond exactly to the three unit directions:
- u1 = 1       (q-axis, direction (1,0))
- u2 = ω       (r-axis, direction (0,1))
- u3 = ω² = −1−ω  (diagonal, direction (1,−1))

A win is an **arithmetic progression of length 6** in Z[ω] with a unit step —
i.e., a set {z, z+u, z+2u, z+3u, z+4u, z+5u} for some z ∈ Z[ω] and u ∈ {u1,u2,u3}.

### Symmetry Group

The lattice Z[ω] has symmetry group **D6** (order 12): 6 rotations at 60°
steps, 6 reflections. In axial coordinates, all 12 transforms are linear
(integer 2×2 matrix multiply on (q,r)). This means:

1. **HexConv2d** — the geometrically correct spatial kernel is the 7-cell
   Z[ω] neighbourhood (standard 3×3 with 2 non-hex corners masked).
2. **D6 augmentation** — every training position can be augmented 12×
   for free, all by linear transforms on (q,r) with no interpolation.

### Connection to Combinatorics

- **Van der Waerden W(6;2) = 1132**: any 2-coloring of {1…1132} contains a
  monochromatic AP-6. The grid version (Hales-Jewett) implies similar bounds.
- **Erdős-Selfridge potential**: ∑ 2^(−|L|) < 1 over all incomplete lines L
  guarantees a draw strategy for the second player. The EisensteinGreedyAgent
  approximates this potential on the Z[ω] lattice as a curriculum adversary.
- **Game value**: HexGo is likely a first-player win (as in all Connect-k
  games with k ≤ board diameter), but the exact proof is open for infinite hex.

### Sutton's Bitter Lesson — the Right Resolution

The concern: does encoding Z[ω] structure into the architecture violate the
"general methods + compute" principle?

Resolution: **geometry (structural substrate) ≠ game knowledge**.

- HexConv2d and D6 augmentation fix the *coordinate system* — they ensure the
  net operates in the correct lattice, not that it is told how to play.
- The net must still discover that six-in-a-row on any axis wins, that
  blocking matters, that forks are dangerous — none of this is encoded.
- AlphaGo Zero similarly used a board-aware architecture (19×19 conv stack,
  not a flat MLP); the geometry was fixed, the strategy was learned.

The `save_heatmap` scientific instrument tests this: policy mass should
spontaneously concentrate on the three Z[ω] win axes as training progresses,
without any explicit encoding of that structure.

---

## File Map

```
hexgo/
  game.py       Engine — 1-2-2 turn logic, incremental candidates, make/unmake
  mcts.py       MCTS — player-aware backprop, rollout + net modes
  net.py        HexNet — HexConv2d/D6/ResNet 18x18/128ch/17ch/6blk + FP16 AMP, spatial policy head
  inference.py  InferenceServer — dynamic batching + transposition cache (dashboard/elo use only)
  train.py      Batched lockstep self-play — no threads, spatial policy loss, curriculum
  config.py     Hyperparameters — all tunable params; edited by autotune agent
  tune.py       Autotune orchestrator — policy loss delta from metrics.jsonl
  elo.py        ELO rating — NetAgent, MCTSAgent, EisensteinGreedyAgent
  tournament.py Round-robin model checkpoint tournament with ELO ratings
  app.py        GUI monitor — pause/resume, win counter, log stream
  replay.py     Terminal replay — 1-2-2 aware, colored last-move bracket
  test_game.py  Unit tests — 26 pass
  render.py     Hex board renderer (shared by app + replay)
  checkpoints/  net_gen*.pt, net_latest.pt
  checkpoints/legacy/  incompatible checkpoints (auto-quarantined)
  heatmaps/     gen_XXXX.png — policy heatmap per generation
  replays/      game_first_genXXXX_*.json, game_last_genXXXX_*.json
  tune_log.jsonl   autotune experiment history
  tune_result.json per-gen metrics from current/last trial
  elo.json      ELO ratings and match history
  docs/         DESIGN.md (this file), component docs, ROADMAP.md, ASSESSMENT.md
```

---

## Running

```bash
# Train (50 gens, 64 games/gen batched lockstep)
python train.py --gens 50 --sims 100 --games 64

# Autotune (5-gen trials, 10 games/gen)
python tune.py --gens 5 --games 10

# Live self-play monitor
python app.py

# Terminal replay
python replay.py replays/game_XXXX.json --delay 0.1

# Smoke tests
python test_game.py
python net.py
```

Always use Python 3.12: `"C:\Program Files\Python312\python.exe"`.
Python 3.14 has no CUDA PyTorch wheels.
