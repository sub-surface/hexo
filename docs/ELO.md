# HexGo — ELO System (`elo.py`, `tournament.py`)

## Agents

| Agent | Description | Used for |
|-------|-------------|----------|
| `EisensteinGreedyAgent` | Greedy Z[ω] chain maximizer. Zero params, zero learning. `defensive=True` also blocks opponent's best chain. | Permanent ELO baseline, tournament opponent |
| `MCTSAgent(sims)` | Pure rollout MCTS, no net | ELO baseline (`mcts_50`) |
| `NetAgent(net, sims)` | `mcts_with_net` wrapper | ELO evaluation of trained net |
| `RandomAgent` | Uniform random legal move | Sanity baseline |

### EisensteinGreedyAgent

Scores each candidate move by the maximum consecutive chain it would create (or block) along any of the three Z[ω] axes. `defensive=True` takes `max(own_chain, block_chain)`.

Approximates the Erdős-Selfridge potential: ∑ 2^(−|L|) for incomplete lines. A bot that always extends its longest chain (or blocks the opponent's) follows the spirit of the optimal draw strategy for the second player.

**Note:** ELO evaluation has been removed from the training loop. The ELO system
is now used exclusively by `tournament.py` for round-robin checkpoint tournaments
and by `elo.py` for standalone evaluation.

Used as:
1. **Permanent ELO anchor**: named `eisenstein_def` in `elo.json`, rating persists across training runs
2. **Tournament baseline**: included via `--include-greedy` in `tournament.py`

---

## ELO Mechanics

- Standard formula, K=32
- `DEFAULT_RATING = 1200`
- Updated after every game, saved to `elo.json` after every game
- `run_match(a, b, n_games)` alternates colors (even games: a=P1, odd: a=P2)

### Issues

**K=32 is too high** for established agents. At K=32 with N=10 games, a single match can swing ratings by up to 160 points. Standard deviation of win rate at N=10 is ~16%. Ratings random-walk rather than converge. Once an agent has played >20 games, K should drop to 16 or lower.

**N=10 games per match** is statistically insufficient. A 6-4 result has a 95% CI of [0.26, 0.74] on true win rate. At minimum N=30 is needed for <10% uncertainty.

**max_moves=300 timeouts** are silently recorded as draws in `elo.json`. Many early games reach 300 moves with no winner because MCTS at 25-50 sims can't find wins quickly enough. These fake draws pollute the rating history.

---

## `NetAgent` Notes

`NetAgent.choose_move` calls `mcts_with_net(game, self.net, self.sims)` directly
(no `InferenceServer`, no ZOI pruning). The training self-play path uses
`batched_self_play()` with ZOI pruning and TOP_K=16. This mismatch means:

1. ELO/tournament matches are slower (synchronous single-threaded forward passes)
2. The net evaluates the full candidate set during ELO but was trained on ZOI-pruned candidates — ELO may underestimate the net's training-distribution quality

The `mcts_with_net` leaf children `player=1` bug was **fixed 2026-03-30** (`player=game.current_player`).

---

## Round-Robin Tournament (`tournament.py`)

New file for model checkpoint evaluation outside the training loop.

```bash
# Auto-select: every 10th gen + latest
python tournament.py --sims 50 --games 4

# Specific checkpoints
python tournament.py --models net_gen0050.pt net_gen0100.pt --sims 50

# Include baseline agents
python tournament.py --sims 50 --games 4 --include-random --include-greedy
```

Plays round-robin: every pair plays N games (alternating colors). Results saved
to `tournament_results.json` with ELO ratings. Uses `NetAgent` + `mcts_with_net`
for net checkpoint evaluation.

---

## `elo.json` Format

```json
{
  "ratings": {
    "eisenstein_def": 1403.1,
    "net_gen0001": 1187.2,
    "mcts_50": 1200.0
  },
  "history": [
    {
      "a": "net_gen0001", "b": "eisenstein_def",
      "winner": "eisenstein_def",
      "moves": 73, "duration": 4.2,
      "ratings": {"net_gen0001": 1187.2, "eisenstein_def": 1403.1}
    }
  ]
}
```

Net agents are named `net_gen{N:04d}` — each generation gets a fresh rating entry starting at 1200. The `eisenstein_def` entry accumulates across all training runs.

---

## `run_match` Return Value

```python
{
    f"wins_{agent_a.name}": int,
    f"wins_{agent_b.name}": int,
    "draws": int,
    "ratings": [(name, rating), ...]  # sorted descending
}
```

Dynamic keys — callers must know agent names to read win counts.

---

## Known Issues

| Issue | Severity |
|-------|----------|
| ~~`mcts_with_net` leaf children `player=1` default~~ | Critical | **Fixed 2026-03-30** |
| K=32 too high for established agents — ratings random-walk | Important |
| Timeout games (max_moves=300) recorded as draws | Important |
| N=10 games — statistically insufficient | Important |
| Training (ZOI) vs eval (full legal moves) mismatch — ELO understates net quality | Important |
| Dynamic return dict keys break typed callers | Moderate |
