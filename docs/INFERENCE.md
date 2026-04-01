# HexGo — Inference Server (`inference.py`)

**Note:** The `InferenceServer` is **no longer used during training**. Training now
uses `batched_self_play()` in `train.py` which does lockstep batched evaluation
directly — no threads, no GIL. The `InferenceServer` is retained for use by
`dashboard`/`elo.py`/`tournament.py` where threaded evaluation is still needed.

## Problem and Solution

Sequential GPU forward passes incur ~6ms kernel launch overhead regardless of
batch size. With 60 MCTS sims/move, sequential calls would cost 360ms/move minimum.

**Solution:** `InferenceServer` collects requests from all parallel self-play threads,
batches them into one GPU call per timeout window, and distributes results back.
GPU utilisation scales roughly linearly with `num_workers` until the GPU saturates.

---

## Architecture

```
N worker threads
  each calls server.evaluate(game) → blocks on resp_queue.get()

InferenceServer._serve() loop:
  1. Wait for first request (blocking, up to 100ms)
  2. Drain queue for up to timeout_ms (default 30ms)
  3. Batch all collected requests → single GPU forward pass
  4. Return (value, policy) to each blocked thread
```

---

## Evaluation Cache

Two-level caching:

### Per-server cache
- Key: `(frozenset(game.board.items()), current_player, placements_in_turn)` — full turn state
- Thread-safe via `_cache_lock`
- Clears when `InferenceServer` is re-created (each generation)
- Typical hit rate: 20–40% during MCTS

**Fixed (2026-03-30)**: cache key now includes `current_player` and `placements_in_turn` to prevent collisions between mid-turn positions under the 1-2-2 rule.

### Persistent cross-generation cache
- Module-level dict `_persistent_cache` survives across `InferenceServer` instances
- Entries tagged with generation number; evicted after `CACHE_MAX_AGE=5` gens
- `evict_stale_cache(gen)` called before each generation starts
- Trades inference quality (stale weights) for speed on common opening positions
- 5-gen retention is conservative — consider reducing to 2–3 once training stabilizes

---

## CUDA Graphs

`InferenceServer.start()` captures a CUDA Graph for the full `batch_size`.

**Fixed (2026-03-30)**:
- Graph capture now uses in-place `.copy_()` writes on pre-allocated output buffers; Python name rebinding bug eliminated.
- `_graph_val` pre-allocated as `torch.zeros(B)` (shape `[B]`, not `[B,1]`); removed stale `[val_idxs, 0]` indexing.
- Added `.detach()` before `.float().cpu().numpy()` on both `_graph_val` and `_graph_pol` outputs (CUDA Graph tensors retain grad).

The graph path now executes correctly and provides the expected 30–50% latency reduction over eager mode.

---

## `torch.compile`

Disabled — uses 3-4GB RAM during JIT compilation. The `start()` method is now
a no-op for compilation. Previously applied `torch.compile(net, dynamic=True)`
but this caused RAM issues and interacted poorly with CUDA Graphs.

---

## Batching Performance

The GIL-induced `avg_batch_size ≈ 1.0` problem was the primary motivation for
replacing the threaded InferenceServer with batched lockstep MCTS in training.
The lockstep approach evaluates all N games in a single batch, achieving full
GPU utilization without any threading.

The InferenceServer remains available for dashboard/elo/tournament use where
true batching matters less (few concurrent games).

---

## API

```python
server = InferenceServer(net, batch_size=8, timeout_ms=30.0, gen=0)
server.start()
value, policy = server.evaluate(game)   # blocks, thread-safe
                                         # value ∈ [-1,1] for current player
                                         # policy: dict[(q,r) → logit]
server.stop()
server.latency_summary()                 # min/avg/max batch latency string
```

---

## Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| CUDA Graph rebinding bug | Critical | **Fixed 2026-03-30** |
| `_graph_val` shape `[B,1]` IndexError | Critical | **Fixed 2026-03-30** |
| Cache key missing turn state | Important | **Fixed 2026-03-30** |
| `torch.compile` disabled (RAM issues) | By design | Disabled |
| `avg_batch_size≈1` root cause was GIL | Design note | Solved by lockstep MCTS in training |
| InferenceServer not used in training | Design note | Retained for dashboard/elo/tournament |
