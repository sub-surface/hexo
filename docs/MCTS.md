# HexGo — MCTS (`mcts.py`)

## Overview

Three search modes:

| Mode | Function | Value source | Used by |
|------|----------|-------------|---------|
| Pure rollout | `mcts(game, sims)` | Random playout | `MCTSAgent` in ELO |
| AlphaZero | `mcts_with_net(game, net, sims)` | Net value + policy priors | ELO `NetAgent`, `tournament.py` |
| Batched lockstep | `batched_self_play()` in train.py | Net via direct batch GPU calls | Training self-play |

`batched_self_play()` in `train.py` is the primary training path. All N games
run in lockstep: selection for all games, batch GPU eval of all leaves, backprop
for all games. No threads, no GIL. TOP_K=16 move pruning per position.
Uses `move_to_grid()` to index the spatial logit map `[B, S, S]`.

`mcts_with_net` is used only in ELO evaluation matches and `tournament.py`.

---

## Node

```python
class Node:
    __slots__ = ("move", "parent", "children", "visits", "value", "prior", "player")
```

`player` stores which player is to move **at** this node (set at creation).

UCB score: `Q + C_PUCT × prior × √(parent.visits) / (1 + visits)`

`C_PUCT` is loaded from `config.CFG["CPUCT"]` at module import time (frozen
for the process lifetime — changes to CFG during autotune require restart).

---

## Multi-Placement Backpropagation

Standard MCTS negates value at every node. HexGo's 1-2-2 turn rule means the
same player moves twice consecutively — negating between their own sub-moves
would be wrong.

**Fix**: value is only negated when `node.parent.player != node.player`.

```python
def _backprop(node, value):
    while node is not None:
        node.visits += 1
        node.value += value
        if node.parent and node.parent.player != node.player:
            value = -value
        node = node.parent
```

---

## Pure Rollout (`mcts`)

1. Expand root with uniform priors.
2. Selection: walk tree via UCB until leaf or terminal.
3. Expand leaf: add children with uniform priors.
4. Simulation: random playout from leaf (max 150 moves).
5. Backprop result via `_backprop`.

Terminal value: `v = -1.0` (whoever is to move lost — previous player won).

---

## AlphaZero (`mcts_with_net`)

Used only by `NetAgent` during ELO evaluation:

1. Evaluate root with net → value + policy logits.
2. Apply Dirichlet noise at root: `prior = (1-ε)·net + ε·Dir(α)`.
3. Selection + leaf expansion with net evaluation.
4. No rollout — net value used directly at leaf.

**Fixed (2026-03-30)**: leaf children now use `player=game.current_player`.

**Fixed (2026-03-30)**: terminal sign — `v = 1.0 if game.winner == node.player else -1.0`.

**Fixed (2026-03-30)**: leaf value perspective — `evaluate()` returns value from
`game.current_player`'s POV; if `node.player != game.current_player`, value is negated
to align with the backprop convention (`value` from `node.player`'s perspective).

---

## Batched Lockstep MCTS (train.py)

`batched_self_play()` replaces the threaded `mcts_policy()` + `InferenceServer` approach:

1. All N games are initialized; root nodes created from a single batched GPU call.
2. Per sim: selection traversal for all active games, batch GPU eval of unexpanded leaves, expand with TOP_K=16 pruned children, backprop + unmake.
3. After all sims: temperature-based move selection, record spatial `policy_target` `[S, S]` and `legal_mask` `[S, S]` planes.

No tree reuse between moves (fresh root each turn). ZOI pruning via `game.zoi_moves(ZOI_MARGIN, ZOI_LOOKBACK)` restricts candidates at every expansion.

---

## Known Issues

1. **DEAD CODE**: In `mcts()`, `v = 0.0 if game.winner is not None` at line
   ~124 is immediately overwritten by the terminal check below it. (Low priority.)

Previously listed bugs FIX-4 and FIX-5 have been resolved as of 2026-03-30.
