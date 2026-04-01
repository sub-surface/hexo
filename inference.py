"""
Batched inference server for net-guided MCTS.

Problem: sequential GPU forward passes have ~6ms kernel launch overhead each,
regardless of batch size. 60 sims/move × 6ms = 360ms/move minimum.

Solution: run N games in parallel threads. Each game blocks waiting for a net
evaluation. The server collects requests from all games, batches them into one
GPU call, and returns results. GPU utilisation goes from ~1% to ~N%.

Throughput scales roughly linearly with num_workers until GPU is saturated.
RTX 2060 with our ~121K-param net: expect useful gains up to ~16 workers.

Usage:
    server = InferenceServer(net, batch_size=8, timeout_ms=5)
    server.start()
    value, policy = server.evaluate(game)   # blocks, thread-safe
    server.stop()
"""

import queue
import threading
import time
import numpy as np
import torch

from net import HexNet, encode_board, move_to_grid, DEVICE, BOARD_SIZE

_SENTINEL = object()


_CUDA = torch.cuda.is_available()

# Persistent cross-generation cache. Entries are (value, policy, gen_added).
# Pass this dict to InferenceServer to keep evaluations across weight updates.
# Eviction: entries older than CACHE_MAX_AGE generations are removed at gen start.
CACHE_MAX_AGE = 2
_persistent_cache: dict = {}   # key → (value, policy, gen)
_persistent_cache_lock = threading.Lock()


def evict_stale_cache(current_gen: int):
    """Remove entries older than CACHE_MAX_AGE from the persistent cache."""
    with _persistent_cache_lock:
        stale = [k for k, (v, p, g) in _persistent_cache.items()
                 if current_gen - g > CACHE_MAX_AGE]
        for k in stale:
            del _persistent_cache[k]


class InferenceServer:
    def __init__(self, net: HexNet, batch_size: int = 8, timeout_ms: float = 5.0,
                 gen: int = 0):
        self.net = net
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000.0
        self.gen = gen
        self._req_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="inference")
        self._running = False
        # Per-server cache (fast path); persistent cache shared across servers
        # Key: (frozenset(board.items()), current_player, placements_in_turn)
        self.cache: dict[tuple, tuple[float, dict]] = {}
        self._cache_lock = threading.Lock()
        # Latency tracking: list of (batch_size, latency_ms) per batch
        self._batch_latencies: list[tuple[int, float]] = []
        # Stats
        self.total_calls = 0
        self.total_batches = 0
        self.avg_batch_size = 0.0
        self.cache_hits = 0
        self.persistent_hits = 0

    def latency_summary(self) -> str:
        """Return min/avg/max batch latency string for logging."""
        if not self._batch_latencies:
            return "no batches"
        lats = [ms for _, ms in self._batch_latencies]
        sizes = [n for n, _ in self._batch_latencies]
        return (f"lat_ms min={min(lats):.1f} avg={sum(lats)/len(lats):.1f} "
                f"max={max(lats):.1f} | batch_sz avg={sum(sizes)/len(sizes):.1f}")

    def start(self):
        self._running = True
        # torch.compile disabled — uses 3-4GB RAM during JIT compilation
        pass
        self._thread.start()

    def stop(self):
        self._running = False
        self._req_queue.put(_SENTINEL)

    def evaluate(self, game) -> tuple[float, dict]:
        """
        Thread-safe. Encodes the game state, submits to batch queue, blocks
        until inference completes. Returns (value, {move: logit}).
        """
        # Include turn state: positions with same pieces but different current_player
        # or placements_in_turn (mid-turn under 1-2-2 rule) are distinct game states.
        key = (frozenset(game.board.items()), game.current_player, game.placements_in_turn)
        # 1. Per-server cache (fastest)
        with self._cache_lock:
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]
        # 2. Persistent cross-gen cache
        with _persistent_cache_lock:
            if key in _persistent_cache:
                v, p, _ = _persistent_cache[key]
                self.persistent_hits += 1
                with self._cache_lock:
                    self.cache[key] = (v, p)
                return v, p

        board_arr, (oq, or_) = encode_board(game)
        moves = game.legal_moves()
        if not moves:
            return 0.0, {}

        resp: queue.Queue = queue.Queue(1)
        self._req_queue.put((board_arr, moves, oq, or_, key, resp))
        return resp.get()

    def _serve(self):
        while self._running:
            # Block until at least one request arrives
            try:
                first = self._req_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is _SENTINEL:
                break

            batch = [first]
            deadline = time.perf_counter() + self.timeout

            # Collect more requests up to batch_size or timeout
            while len(batch) < self.batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._req_queue.get(timeout=remaining)
                    if item is _SENTINEL:
                        self._running = False
                        break
                    batch.append(item)
                except queue.Empty:
                    break

            t_batch = time.perf_counter()
            self._process_batch(batch)
            lat_ms = (time.perf_counter() - t_batch) * 1000.0

            # Update stats
            self.total_batches += 1
            self.total_calls += len(batch)
            self.avg_batch_size = self.total_calls / self.total_batches
            self._batch_latencies.append((len(batch), lat_ms))

    def _process_batch(self, batch: list):
        """Run one batched forward pass for all requests.

        Spatial policy head: each request = 1 row in the batch tensor.
        trunk + value + policy_logits runs once for the whole batch,
        then logit maps are indexed per-request to extract move logits.
        """
        N = len(batch)
        boards_np = np.stack([item[0] for item in batch])   # [N, C, S, S]

        self.net.eval()
        if _CUDA:
            boards_t = (torch.from_numpy(boards_np).pin_memory()
                        .to(DEVICE, non_blocking=True))
        else:
            boards_t = torch.tensor(boards_np, device=DEVICE)

        dev_type = "cuda" if _CUDA else "cpu"
        with torch.amp.autocast(device_type=dev_type):
            with torch.no_grad():
                features = self.net.trunk(boards_t)            # [N, C, S, S]
                val_t    = self.net.value(features)             # [N]
                pol_t    = self.net.policy_logits(features)     # [N, S, S]

        values     = val_t.float().cpu().numpy()               # [N]
        logit_maps = pol_t.float().cpu().numpy()               # [N, S, S]

        for i, (board_arr, moves, oq, or_, key, resp) in enumerate(batch):
            policy = {}
            for m in moves:
                idx = move_to_grid(m[0], m[1], oq, or_)
                if idx is not None:
                    row, col = idx
                    policy[m] = float(logit_maps[i, row, col])
            res = (float(values[i]), policy)
            with self._cache_lock:
                self.cache[key] = res
            with _persistent_cache_lock:
                _persistent_cache[key] = (float(values[i]), policy, self.gen)
            resp.put(res)
