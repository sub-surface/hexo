# HexGo — Honest Project Assessment

*Code review conducted 2026-03-29/30 by 5 independent subagents across all source files.*

---

## Summary Verdict

The architecture and mathematical foundation are **genuinely impressive** for a solo project.
The Z[ω] isomorphism, HexConv2d, D6 augmentation, GlobalPoolBranch, and the overall AlphaZero
pipeline are implemented correctly in their essentials and show real depth of understanding.

**As of the latest revision, the training pipeline has been substantially rearchitected:**
- Spatial policy head (`policy_logits → [B, S, S]`) replaces per-move forward passes
- Batched lockstep MCTS (`batched_self_play()`) replaces threaded InferenceServer for training
- Network scaled to 128ch/6blk (~1.9M params) from 64ch/4blk (~480K)
- ELO evaluation removed from training loop; `tournament.py` added for standalone checkpoint tournaments
- `tune.py` uses policy loss delta instead of ELO vs Eisenstein
- Curriculum for sims (16→100) and max_moves (30→100)
- Fully vectorized spatial masked softmax policy loss

A FastAPI dashboard (server.py + dashboard.html) provides live monitoring,
training controls, and a replay viewer.

Previously identified critical bugs (now fixed):
1. ~~**Autotune anti-optimizing**~~ — reward signal inverted; fixed in `tune.py`.
2. ~~**CUDA Graphs broken**~~ — tensor rebinding bug; fixed with in-place `.copy_()`.
3. ~~**Cache collides on turn state**~~ — key now includes `current_player` + `placements_in_turn`.
4. ~~**NetAgent backprop corrupted**~~ — leaf children now use `player=game.current_player`.

---

## What Works Well

### Mathematical Foundation
The Z[ω] Eisenstein integer framing is correct and well-chosen. It isn't just aesthetic — it
directly enables three concrete engineering wins: `HexConv2d` (the right kernel for hex grids),
D6 augmentation (12× free data efficiency), and `EisensteinGreedyAgent` (a principled curriculum
opponent grounded in Erdős-Selfridge potential theory). This is better-motivated than most hobby
game-playing projects.

### Game Engine (`game.py`)
Solid. Make/unmake with full undo stack is correctly implemented including `winner`, `current_player`,
and `placements_in_turn`. The incremental `candidates` set gives O(1) legal moves. Win detection
walks only the 3 Z[ω] axes through the last piece — O(WIN_LENGTH). The 1-2-2 turn logic is correct.
All 26 unit tests pass, covering win detection on all three axes, undo correctness, D6 symmetry,
and the EisensteinGreedyAgent.

### Neural Network Architecture (`net.py`)
HexConv2d correctly masks the two non-adjacent corners `[0,0]` and `[2,2]` of 3×3 kernels.
D6_MATRICES implements the correct dihedral group D6 for Z[ω] (determinant-1 rotations, det=-1
reflections, group closed under composition). GlobalPoolBranch shapes are correct. The 17-channel
input encoding (2 current + to-move + 8 history + 6 axis-chain planes) provides rich spatial
features. Spatial policy head (`policy_logits → [B, S, S]`) enables single-pass evaluation.
At ~1.9M params (128ch/6blk) on an RTX 2060, capacity is substantial for the game complexity.

### MCTS Backpropagation Convention
The multi-placement backprop rule (`negate only when node.parent.player != node.player`) correctly
handles the 1-2-2 turn structure. This is non-trivial to get right and is implemented correctly
in the pure-rollout path.

### Infrastructure
The batched lockstep MCTS in `train.py` solves the GIL-induced batching problem that plagued
the threaded InferenceServer approach. The `InferenceServer` is retained for dashboard/elo/tournament
use. `load_latest()` quarantine for incompatible checkpoints prevents silent data loss.
`tournament.py` provides standalone round-robin checkpoint evaluation.

### EisensteinGreedyAgent
The `_chain_if_placed` implementation is correct and efficient — no board mutations, no off-by-one
in the bidirectional axis walk. The `defensive=True` variant correctly takes `max(own, block)`.
This is a genuinely useful curriculum opponent: it plays structured, principled moves without
requiring any learned weights.

---

## What's Fixed (as of 2026-03-30)

### Previously Critical Bugs — All Resolved

**Autotune reward inverted** → Fixed: `kept = elo_delta is None or elo_delta <= 0`.

**CUDA Graph rebinding** → Fixed: in-place `.copy_()` inside capture; `.detach()` before numpy; `_graph_val` shape `[B]` not `[B,1]`; removed `[val_idxs, 0]` indexing.

**Cache key ignores turn state** → Fixed: key is `(frozenset(board.items()), current_player, placements_in_turn)`.

**`mcts_with_net` leaf children `player=1`** → Fixed: `player=game.current_player` in leaf node creation.

**Terminal expansion sign** → Fixed: `v = 1.0 if game.winner == node.player else -1.0` (was always `1.0`).

**SIMS_MIN too high** → Fixed: `SIMS_MIN = 6` in `config.py`.

**Cosine temp semantics** → Fixed: `cos(π/2 × move / TEMP_HORIZON)` — now reaches floor at `TEMP_HORIZON` moves.

**Replay hex offset** → Fixed: `indent = r - r_min`.

**Overlap training overfit** → Fixed: capped by `batches_since_sync < WEIGHT_SYNC_BATCHES`.

**D6 augmentation probs corruption** → Fixed: `sample['probs'].copy()` in `d6_augment_sample`.

### Training Loop Rearchitected

- Threaded InferenceServer self-play replaced with batched lockstep MCTS (`batched_self_play()`)
- ELO evaluation removed from training loop — pure self-play + training only
- Standalone `tournament.py` for round-robin checkpoint evaluation
- `tune.py` rewritten to use policy loss delta from `metrics.jsonl` instead of ELO vs Eisenstein
- `torch.compile` disabled (3-4GB RAM usage during compilation)

## What's Still Open

### Deferred — Lower Risk

**ZOI lookback blind spot** (`game.py`)
`lookback=8` can miss early threats in long games. Conservative; increase to 16 or add
a separate threat-line set if ELO growth stalls at mid-game complexity.

### Fixed in session 2 (2026-03-30)

- ~~History planes cross-reference board dict~~ — verified correct: `player_history` already used.
- ~~`net.py:339` comment `[B, 2*S*S]`~~ — fixed to `[B, 4*S*S]`.
- ~~`net.py` forward docstring `[B, 3, S, S]`~~ — fixed to `[B, 11, S, S]`.
- ~~`inference.py` comment `[3, S, S]`~~ — fixed to `[11, S, S]`.

---

## Training Signal Quality Assessment

With all bugs fixed, the training pipeline is now sound end-to-end:

- `mcts_policy` in `train.py` uses correct `player=game.current_player` at every node
- Terminal and leaf value signs are correct (`v = winner == node.player` + alignment flip)
- Tree reuse prunes stale children before recycling subtrees
- CUDA Graph path is functional; cache key includes turn state
- D6 augmentation no longer corrupts replay buffer entries
- Overlap training capped to prevent overfitting stale buffer
- MAX_MOVES=300 prevents runaway games that could starve the buffer

Any **tune_log.jsonl** entries from before 2026-03-30 should be discarded — they were
produced with the inverted reward signal and are not meaningful.

---

## Comparison to Research Targets

| Feature | Research Target | Current | Gap |
|---------|----------------|---------|-----|
| SIMS | 200-600 (Phase 1) | 100 (curriculum from 16) | Moderate — RTX 2060 constraint |
| CPUCT | 2.5 (Phase 1) | **1.5** | Lowered from 2.0; raise if ELO stalls |
| Board size | 18×18 | 18×18 | ✓ |
| Network depth | 8-15 blocks | **6 blocks** | Closer to research target |
| Network width | 128-256 ch | **128 ch** ✓ | ✓ |
| Total params | 2-10M | **~1.9M** | Approaching research range |
| Dirichlet alpha | 0.08-0.10 (10/\|ZoI\|) | **0.15** | Raised for more exploration |
| ZoI margin | 3 minimum | 6 | Conservative but correct |
| Auxiliary heads | KataGo ownership+threat | **Done** ✓ | Thin 1×1 conv heads, AUX_LOSS=0.1 each |
| Replay sampling | Recent-biased | **75/25** ✓ | ✓ |
| MCTS | PUCT standard | PUCT (Python, batched lockstep) | No C++ needed — lockstep solves GIL batching |
| Policy head | Spatial | **Spatial** ✓ | `policy_logits → [B, S, S]`, single pass |
| Training batch | 256+ | **256** ✓ | ✓ |

---

## Next Steps

Major architectural changes complete. Ready to run sustained baseline:

1. **Run sustained multi-gen baseline** — verify policy loss decreases and decisive game rate increases.
2. **Tournament evaluation** — use `tournament.py` to compare checkpoint progression.
3. **ZOI lookback tuning** — increase if mid-game complexity stalls.

The batched lockstep MCTS + spatial policy head should provide significantly faster
training iterations than the previous threaded approach.
