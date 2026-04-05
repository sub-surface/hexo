# Rust Port Plan -- CPU-Bound HexGo Engine via PyO3

## Goal

Port the CPU-bound hot paths (game engine + MCTS tree search) from Python to Rust,
exposed to the existing Python training pipeline via PyO3.  The neural net stays in
Python/PyTorch.  Rust calls back into Python for net inference during MCTS.

Target: i9-14900KF (8P + 16E = 24 cores), RTX 2060.  The CPU side (game logic,
tree traversal, candidate maintenance) is the bottleneck -- profiling shows
`make`/`unmake`, `legal_moves`, and `best_child` dominate wall time in self-play.

---

## 1. Crate Structure

```
hexgo-rs/
  Cargo.toml
  src/
    lib.rs          -- PyO3 module definition, #[pymodule]
    game.rs         -- HexGame: board, make/unmake, win check, candidates
    node.rs         -- MCTS Node: slots, PUCT, expand, backprop
    mcts.rs         -- mcts() pure rollout, mcts_with_net() net-guided
    types.rs        -- Coord, Player, Move aliases, constants
    bitset.rs       -- (optional) compact coord set for candidates
    parallel.rs     -- rayon self-play driver for batch generation
```

**Cargo.toml dependencies:**

```toml
[package]
name = "hexgo"
version = "0.1.0"
edition = "2021"

[lib]
name = "hexgo"
crate-type = ["cdylib"]   # produces .pyd / .so for Python import

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
rustc-hash = "2"          # FxHashMap/FxHashSet -- fast integer hashing
rand = "0.8"
rayon = "1.10"

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

Build with `maturin develop --release` during development,
`maturin build --release` for wheels.

---

## 2. types.rs -- Shared Types and Constants

```rust
// types.rs

/// Axial hex coordinate.
pub type Coord = (i16, i16);

/// Player 1 or 2.  0 = empty / no winner.
pub type Player = u8;

pub const P1: Player = 1;
pub const P2: Player = 2;
pub const NO_PLAYER: Player = 0;

pub const WIN_LENGTH: usize = 6;
pub const PLACEMENT_RADIUS: i16 = 8;

/// Six axial neighbor offsets.  First 3 are the unique axis directions
/// (each paired with its negation at index + 3).
pub const DIRS: [Coord; 6] = [
    (1, 0), (0, 1), (1, -1),
    (-1, 0), (0, -1), (-1, 1),
];

/// The 3 unique axes for win checking.
pub const AXES: [Coord; 3] = [(1, 0), (0, 1), (1, -1)];

/// Hex distance in axial coordinates.
#[inline(always)]
pub fn hex_dist(a: Coord, b: Coord) -> i16 {
    let dq = (a.0 - b.0).abs();
    let dr = (a.1 - b.1).abs();
    let ds = ((a.0 + a.1) - (b.0 + b.1)).abs();
    (dq + dr + ds) / 2
}
```

---

## 3. game.rs -- HexGame Engine

### Data layout

```rust
// game.rs

use rustc_hash::{FxHashMap, FxHashSet};
use pyo3::prelude::*;
use crate::types::*;

/// Undo record for a single placement.
struct UndoEntry {
    coord: Coord,
    removed_candidates: Vec<Coord>,   // candidates that were removed (the placed cell)
    added_candidates: Vec<Coord>,     // candidates that were added (new neighbors)
    prev_placements: u8,
    prev_winner: Player,
    prev_player: Player,
}

#[pyclass]
#[derive(Clone)]
pub struct HexGame {
    board: FxHashMap<Coord, Player>,
    candidates: FxHashSet<Coord>,
    #[pyo3(get)]
    current_player: Player,
    #[pyo3(get)]
    placements_in_turn: u8,
    #[pyo3(get)]
    winner: Player,              // 0 = no winner
    move_history: Vec<Coord>,
    player_history: Vec<Player>,
    undo_stack: Vec<UndoEntry>,
}
```

Key decisions:
- **`FxHashMap`/`FxHashSet`** instead of `HashMap` -- identity-quality hash for
  small integer keys, 2-3x faster than `SipHash` default.
- **`Coord = (i16, i16)`** -- 4 bytes, cheap to copy, fits in a register pair.
- **`UndoEntry`** uses `Vec<Coord>` for added/removed (typically 0-6 items).
  Stack-allocated `SmallVec<[Coord; 6]>` is an option if profiling shows heap
  pressure, but `Vec` is fine for the expected sizes.
- **`winner: Player`** uses `0` for "no winner" instead of `Option<Player>`.
  Avoids match overhead in hot paths.  Exposed to Python as `winner` (0 or 1 or 2).

### Core methods

```rust
#[pymethods]
impl HexGame {
    #[new]
    pub fn new() -> Self { /* init board empty, candidates = {(0,0)}, player 1 */ }

    /// Place current player's piece.  Returns true if legal.
    pub fn make(&mut self, q: i16, r: i16) -> bool { /* ... */ }

    /// Undo the last placement.
    pub fn unmake(&mut self) { /* ... */ }

    /// Alias for make() -- backward compat.
    pub fn play(&mut self, q: i16, r: i16) -> bool { self.make(q, r) }

    /// All candidates (empty cells adjacent to existing pieces).
    /// Returns list of (q, r) tuples for Python.
    pub fn candidates_list(&self) -> Vec<Coord> {
        self.candidates.iter().copied().collect()
    }

    /// Full legal_moves (radius scan).  Slow path -- only for rollouts.
    pub fn legal_moves(&self) -> Vec<Coord> { /* ... */ }

    /// Zone-of-interest pruned move list.
    #[pyo3(signature = (margin=6, lookback=16))]
    pub fn zoi_moves(&self, margin: i16, lookback: usize) -> Vec<Coord> { /* ... */ }

    /// Deep clone (no undo stack in clone).
    pub fn clone_game(&self) -> HexGame { /* clone without undo_stack */ }

    /// Board as dict for Python: {(q,r): player, ...}
    pub fn board_dict(&self) -> FxHashMap<Coord, Player> {
        self.board.clone()
    }

    /// Number of moves played.
    pub fn num_moves(&self) -> usize { self.move_history.len() }

    /// Move history as list of (q, r).
    pub fn get_move_history(&self) -> Vec<Coord> {
        self.move_history.clone()
    }

    /// Player history as list of player IDs.
    pub fn get_player_history(&self) -> Vec<Player> {
        self.player_history.clone()
    }

    fn __repr__(&self) -> String { /* ... */ }
}
```

### Internal (non-pymethod) helpers

```rust
impl HexGame {
    /// Check win along the 3 axes through (q, r).  Only called after placement.
    #[inline]
    fn check_win(&self, q: i16, r: i16) -> bool {
        let player = self.board[&(q, r)];
        for &(dq, dr) in &AXES {
            let mut count: u32 = 1;
            for sign in [1i16, -1] {
                let (mut nq, mut nr) = (q + sign * dq, r + sign * dr);
                while self.board.get(&(nq, nr)) == Some(&player) {
                    count += 1;
                    nq += sign * dq;
                    nr += sign * dr;
                }
            }
            if count >= WIN_LENGTH as u32 {
                return true;
            }
        }
        false
    }

    /// Yield all cells within hex radius of (q, r).
    /// Used in legal_moves() -- the slow path for rollouts.
    fn cells_within_radius(q: i16, r: i16, radius: i16) -> impl Iterator<Item = Coord> {
        (-radius..=radius).flat_map(move |dq| {
            let lo = (-radius).max(-dq - radius);
            let hi = radius.min(-dq + radius);
            (lo..=hi).map(move |dr| (q + dq, r + dr))
        })
    }

    /// The 1-2-2 turn limit: first move = 1 stone, all subsequent turns = 2.
    #[inline]
    fn stones_this_turn(&self) -> u8 {
        if self.move_history.len() <= 1 { 1 } else { 2 }
    }
}
```

### make() implementation sketch

```rust
pub fn make(&mut self, q: i16, r: i16) -> bool {
    let coord = (q, r);
    if self.winner != NO_PLAYER || self.board.contains_key(&coord) {
        return false;
    }

    self.board.insert(coord, self.current_player);
    self.move_history.push(coord);
    self.player_history.push(self.current_player);

    // Update candidates incrementally
    let mut removed = Vec::new();
    let mut added = Vec::new();
    if self.candidates.remove(&coord) {
        removed.push(coord);
    }
    for &(dq, dr) in &DIRS {
        let nb = (q + dq, r + dr);
        if !self.board.contains_key(&nb) && self.candidates.insert(nb) {
            added.push(nb);
        }
    }

    let won = self.check_win(q, r);
    let prev_winner = self.winner;
    let prev_player = self.current_player;
    let prev_placements = self.placements_in_turn;

    if won {
        self.winner = self.current_player;
    }

    self.placements_in_turn += 1;
    if !won {
        let limit = self.stones_this_turn();
        if self.placements_in_turn >= limit {
            self.current_player = 3 - self.current_player;
            self.placements_in_turn = 0;
        }
    }

    self.undo_stack.push(UndoEntry {
        coord, removed_candidates: removed, added_candidates: added,
        prev_placements, prev_winner, prev_player,
    });
    true
}
```

### unmake() implementation sketch

```rust
pub fn unmake(&mut self) {
    let Some(entry) = self.undo_stack.pop() else { return };
    self.board.remove(&entry.coord);
    self.move_history.pop();
    self.player_history.pop();
    for &c in &entry.added_candidates {
        self.candidates.remove(&c);
    }
    for &c in &entry.removed_candidates {
        self.candidates.insert(c);
    }
    self.winner = entry.prev_winner;
    self.current_player = entry.prev_player;
    self.placements_in_turn = entry.prev_placements;
}
```

---

## 4. node.rs -- MCTS Tree Node

```rust
// node.rs

use crate::types::*;

/// MCTS tree node.  Arena-allocated via index into a Vec<Node>.
///
/// Using arena indices instead of Rc/Arc eliminates ref-count overhead
/// and makes the tree trivially serializable.  Parent/children are
/// indices into the arena Vec.
pub type NodeIdx = u32;
pub const NULL_NODE: NodeIdx = u32::MAX;

pub struct Node {
    pub mov: Coord,              // (q, r) that led here; (0,0) for root
    pub parent: NodeIdx,         // NULL_NODE for root
    pub children_start: u32,     // index into arena where children begin
    pub children_count: u16,     // number of children
    pub visits: u32,
    pub value: f32,
    pub prior: f32,
    pub player: Player,
}
```

### Arena allocator

```rust
/// Arena for MCTS nodes.  All nodes for one search tree live here.
pub struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    pub fn with_capacity(cap: usize) -> Self {
        Arena { nodes: Vec::with_capacity(cap) }
    }

    pub fn alloc(&mut self, node: Node) -> NodeIdx {
        let idx = self.nodes.len() as NodeIdx;
        self.nodes.push(node);
        idx
    }

    #[inline(always)]
    pub fn get(&self, idx: NodeIdx) -> &Node {
        &self.nodes[idx as usize]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, idx: NodeIdx) -> &mut Node {
        &mut self.nodes[idx as usize]
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}
```

### best_child with PUCT + FPU

```rust
impl Arena {
    /// Select best child of `parent_idx` via PUCT + FPU reduction.
    pub fn best_child(&self, parent_idx: NodeIdx, c_puct: f32, fpu_reduction: f32) -> NodeIdx {
        let parent = self.get(parent_idx);
        let n = parent.visits.max(1) as f32;
        let cpuct_sqrt = c_puct * n.sqrt();
        let fpu_q = parent.value / n - fpu_reduction;

        let start = parent.children_start as usize;
        let end = start + parent.children_count as usize;

        let mut best_idx = NULL_NODE;
        let mut best_score = f32::NEG_INFINITY;

        for i in start..end {
            let child = &self.nodes[i];
            let score = if child.visits == 0 {
                fpu_q + cpuct_sqrt * child.prior
            } else {
                let q = child.value / child.visits as f32;
                q + cpuct_sqrt * child.prior / (1.0 + child.visits as f32)
            };
            if score > best_score {
                best_score = score;
                best_idx = i as NodeIdx;
            }
        }
        best_idx
    }
}
```

### expand

```rust
impl Arena {
    /// Expand a leaf node with the given moves and priors.
    pub fn expand(
        &mut self,
        node_idx: NodeIdx,
        moves: &[Coord],
        priors: &[f32],
        player: Player,
    ) {
        let start = self.nodes.len() as u32;
        for (i, &m) in moves.iter().enumerate() {
            self.nodes.push(Node {
                mov: m,
                parent: node_idx,
                children_start: 0,
                children_count: 0,
                visits: 0,
                value: 0.0,
                prior: priors[i],
                player,
            });
        }
        let node = &mut self.nodes[node_idx as usize];
        node.children_start = start;
        node.children_count = moves.len() as u16;
    }
}
```

### backprop with 1-2-2 sign flip

```rust
impl Arena {
    /// Backpropagate value from leaf to root.
    /// Sign flips only when parent.player != child.player,
    /// correctly handling the 1-2-2 turn rule.
    pub fn backprop(&mut self, mut idx: NodeIdx, mut value: f32) {
        while idx != NULL_NODE {
            let node = self.get_mut(idx);
            node.visits += 1;
            node.value += value;
            let parent_idx = node.parent;
            if parent_idx != NULL_NODE {
                let parent_player = self.get(parent_idx).player;
                if parent_player != node.player {
                    value = -value;
                }
            }
            idx = parent_idx;
        }
    }
}
```

---

## 5. mcts.rs -- Search Functions

### Pure rollout MCTS

```rust
// mcts.rs

use crate::game::HexGame;
use crate::node::*;
use crate::types::*;
use rand::seq::SliceRandom;

/// Pure rollout MCTS (no neural net).  Used for testing and baseline.
/// Returns the best move (q, r).
pub fn mcts(game: &mut HexGame, num_sims: u32, c_puct: f32, fpu_reduction: f32) -> Coord {
    let mut arena = Arena::with_capacity(num_sims as usize * 20);
    let root = arena.alloc(Node {
        mov: (0, 0), parent: NULL_NODE,
        children_start: 0, children_count: 0,
        visits: 0, value: 0.0, prior: 1.0,
        player: game.current_player,
    });

    // Expand root
    let moves = game.legal_moves();
    let uniform = 1.0 / moves.len().max(1) as f32;
    let priors: Vec<f32> = vec![uniform; moves.len()];
    arena.expand(root, &moves, &priors, game.current_player);

    let mut rng = rand::thread_rng();

    for _ in 0..num_sims {
        let mut node = root;
        let mut depth: u32 = 0;

        // Selection
        while arena.get(node).children_count > 0 && game.winner == NO_PLAYER {
            node = arena.best_child(node, c_puct, fpu_reduction);
            let m = arena.get(node).mov;
            game.make(m.0, m.1);
            depth += 1;
        }

        // Expansion + rollout
        if game.winner == NO_PLAYER {
            let moves = game.legal_moves();
            let uniform = 1.0 / moves.len().max(1) as f32;
            let priors: Vec<f32> = vec![uniform; moves.len()];
            arena.expand(node, &moves, &priors, game.current_player);

            if arena.get(node).children_count > 0 {
                // Pick random child for playout
                let start = arena.get(node).children_start as usize;
                let count = arena.get(node).children_count as usize;
                let pick = start + rng.gen_range(0..count);
                node = pick as NodeIdx;
                let m = arena.get(node).mov;
                game.make(m.0, m.1);
                depth += 1;
            }
        }

        let v = if game.winner != NO_PLAYER {
            1.0   // node.player just won
        } else {
            -rollout(game, &mut rng)   // rollout returns value for game.current_player
        };

        // Restore
        for _ in 0..depth {
            game.unmake();
        }

        arena.backprop(node, v);
    }

    // Return most-visited child of root
    best_move(&arena, root)
}

fn rollout(game: &mut HexGame, rng: &mut impl rand::Rng) -> f32 {
    let start_player = game.current_player;
    let mut depth: u32 = 0;
    let max_moves: u32 = 150;

    while game.winner == NO_PLAYER && depth < max_moves {
        let moves = game.legal_moves();
        if moves.is_empty() { break; }
        let m = moves[rng.gen_range(0..moves.len())];
        game.make(m.0, m.1);
        depth += 1;
    }

    let result = if game.winner == NO_PLAYER {
        0.0
    } else if game.winner == start_player {
        1.0
    } else {
        -1.0
    };

    for _ in 0..depth {
        game.unmake();
    }
    result
}

fn best_move(arena: &Arena, root: NodeIdx) -> Coord {
    let node = arena.get(root);
    let start = node.children_start as usize;
    let end = start + node.children_count as usize;
    let mut best_visits = 0u32;
    let mut best_coord = (0i16, 0i16);
    for i in start..end {
        if arena.get(i as NodeIdx).visits > best_visits {
            best_visits = arena.get(i as NodeIdx).visits;
            best_coord = arena.get(i as NodeIdx).mov;
        }
    }
    best_coord
}
```

### Net-guided MCTS (calls back into Python)

```rust
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Net-guided MCTS that calls back to Python for leaf evaluation.
///
/// `eval_fn` is a Python callable: (HexGame) -> (float, dict[(q,r) -> logit])
/// The neural net and encode_board stay in Python.
///
/// This is the function used by ELO evaluation (unbatched, single game).
#[pyfunction]
#[pyo3(signature = (game, eval_fn, num_sims=100, c_puct=1.5, fpu_reduction=0.2,
                    dirichlet_alpha=0.15, dirichlet_eps=0.35, top_k=16))]
pub fn mcts_with_net(
    game: &mut HexGame,
    eval_fn: &Bound<'_, PyAny>,   // Python callable
    num_sims: u32,
    c_puct: f32,
    fpu_reduction: f32,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
    top_k: usize,
) -> PyResult<(i16, i16)> {
    // ... create arena, expand root with eval_fn, add Dirichlet noise ...
    // Selection loop calls arena.best_child()
    // Leaf evaluation: Python::with_gil(|py| eval_fn.call1((game,)))
    // Backprop with arena.backprop()
    // Return best_move()
    todo!()
}
```

The key pattern for the Python callback:

```rust
// Inside the simulation loop, at a leaf node:
let (value, policy): (f32, Vec<(Coord, f32)>) = Python::with_gil(|py| {
    let result = eval_fn.call1((game.clone_for_python(),))?;
    let tuple = result.downcast::<pyo3::types::PyTuple>()?;
    let v: f32 = tuple.get_item(0)?.extract()?;
    let policy_dict = tuple.get_item(1)?.downcast::<PyDict>()?;
    let mut policy = Vec::new();
    for (key, val) in policy_dict.iter() {
        let coord: (i16, i16) = key.extract()?;
        let logit: f32 = val.extract()?;
        policy.push((coord, logit));
    }
    Ok::<_, PyErr>((v, policy))
})?;
```

---

## 6. PyO3 Bindings -- lib.rs

```rust
// lib.rs

use pyo3::prelude::*;

mod types;
mod game;
mod node;
mod mcts;

/// The Python module.  `import hexgo` in Python after installing the wheel.
#[pymodule]
fn hexgo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<game::HexGame>()?;
    m.add_function(wrap_pyfunction!(mcts::mcts_py, m)?)?;
    m.add_function(wrap_pyfunction!(mcts::mcts_with_net, m)?)?;
    m.add_function(wrap_pyfunction!(mcts::batched_mcts_step, m)?)?;

    // Expose constants
    m.add("WIN_LENGTH", types::WIN_LENGTH)?;
    m.add("PLACEMENT_RADIUS", types::PLACEMENT_RADIUS)?;
    m.add("P1", types::P1)?;
    m.add("P2", types::P2)?;

    Ok(())
}
```

### Which types get `#[pyclass]`

| Rust type | `#[pyclass]` | Reason |
|-----------|-------------|--------|
| `HexGame` | Yes | Python creates/inspects games, passes them to net encoding |
| `Node` / `Arena` | **No** | Internal to Rust MCTS.  Python never touches individual nodes. |
| `Coord` | No | Passed as `(i16, i16)` tuples; PyO3 converts automatically. |

### Which methods get `#[pymethods]`

On `HexGame`:
- `new()`, `make()`, `unmake()`, `play()`
- `legal_moves()`, `zoi_moves()`, `candidates_list()`
- `clone_game()`, `board_dict()`, `num_moves()`
- `get_move_history()`, `get_player_history()`
- Properties: `current_player`, `placements_in_turn`, `winner`
- `__repr__`

### How Python calls in

```python
# In train.py or elo.py:
import hexgo

# Drop-in replacement for game.HexGame
game = hexgo.HexGame()
game.make(0, 0)
print(game.current_player)  # 2
moves = game.zoi_moves(margin=5, lookback=16)

# Pure MCTS (for testing)
best = hexgo.mcts(game, num_sims=200)

# Net-guided MCTS (for ELO evaluation)
def eval_fn(g):
    return net_evaluate(net, g)  # returns (value, {coord: logit})
best = hexgo.mcts_with_net(game, eval_fn, num_sims=100)
```

---

## 7. Data Flow -- Batched Lockstep Self-Play

The current `batched_self_play()` in `train.py` runs N games in lockstep:

```
for each sim in num_sims:
    for each active game:
        select leaf (game.make along tree path)
    batch all leaf positions -> one GPU forward pass
    for each active game:
        expand node with policy, backprop value
        game.unmake to restore
```

### Option A: Game engine in Rust, MCTS loop stays in Python (recommended first)

Replace only `HexGame` with `hexgo.HexGame`.  The lockstep loop in `train.py`
stays Python.  This gives the biggest speedup for the least integration work
because `make`/`unmake`/`candidates` are the innermost hot calls.

```python
# train.py -- minimal change
from hexgo import HexGame  # <-- swap this one import
# everything else unchanged; HexGame has the same API
```

### Option B: Full lockstep step function in Rust

Expose a single `batched_mcts_step()` that does one sim step for all N games:

```rust
/// One simulation step across N games.  Returns indices of games
/// that need leaf evaluation (the GPU batch).
///
/// Called from Python's lockstep loop.  Python does the GPU call,
/// then calls `batched_mcts_receive()` with the results.
#[pyfunction]
pub fn batched_mcts_step(
    games: Vec<PyRefMut<HexGame>>,
    // ... arena handles, active mask ...
) -> Vec<usize> {
    // For each active game: select leaf via PUCT, return which need eval.
    todo!()
}

#[pyfunction]
pub fn batched_mcts_receive(
    // game indices, values, policy dicts from GPU
    // expands nodes and backprops
) {
    todo!()
}
```

This requires more wiring but moves the entire PUCT loop (thousands of
`best_child` calls per sim) into Rust.

### Recommended approach

Start with **Option A**.  Measure.  If the tree traversal is still a bottleneck
after the game engine is in Rust, implement Option B.

The Gumbel Sequential Halving schedule, tree reuse, and speculative expansion
logic can stay in Python for now -- they are O(top_k) per move, not O(sims).

---

## 8. Parallelism Opportunities

### 8a. Parallel game instances in self-play (rayon)

During training, self-play generates many independent games.  With Rust game
engines, each game is `Send + Sync` (no Python objects inside).  Use rayon to
run multiple `mcts()` (pure rollout) games in parallel:

```rust
use rayon::prelude::*;

#[pyfunction]
pub fn parallel_self_play(num_games: usize, sims_per_move: u32) -> Vec<GameResult> {
    (0..num_games).into_par_iter().map(|_| {
        let mut game = HexGame::new();
        let mut moves = Vec::new();
        while game.winner == NO_PLAYER {
            let m = mcts(&mut game, sims_per_move, C_PUCT, FPU_REDUCTION);
            game.make(m.0, m.1);
            moves.push(m);
        }
        GameResult { winner: game.winner, moves }
    }).collect()
}
```

This parallelizes trivially because each game is fully independent.  24 cores
running pure-rollout games would give ~20x throughput over single-threaded Python.

**Limitation:** net-guided MCTS cannot use rayon freely because the Python
callback (GIL) serializes GPU calls.  This is why the batched lockstep design
exists -- it is already the right pattern.

### 8b. Parallel MCTS rollouts within a single game (virtual loss)

For pure-rollout MCTS, multiple threads can run rollouts on the same tree
using virtual loss:

```rust
/// Virtual loss: temporarily decrement value and increment visits
/// while a thread is doing a rollout, so other threads avoid the same path.
pub fn apply_virtual_loss(arena: &AtomicArena, path: &[NodeIdx]) {
    for &idx in path {
        arena.atomic_add_visits(idx, 1);
        arena.atomic_add_value(idx, -1.0);  // pessimistic
    }
}
```

This requires atomic operations on `visits` and `value` fields.  Worthwhile
only for pure-rollout mode or if GPU batching is not the bottleneck.

**Verdict:** For net-guided MCTS, this is not useful -- the GPU batch is the
serialization point, and virtual loss does not help.  For pure-rollout testing,
this could use rayon's work-stealing to run 24 rollouts in parallel per game.

### 8c. Practical thread layout for training

```
Main thread:  Python training loop (optimizer, loss, checkpoint)
GPU thread:   batched inference (already exists as InferenceServer)
rayon pool:   Rust game engines (make/unmake/candidates) called from
              Python lockstep loop -- no threads needed here because
              the lockstep loop is sequential per sim step.

              OR: parallel_self_play() for pure-rollout warm-up games.
```

The biggest parallelism win is running the Eisenstein curriculum games in
parallel with rayon (they use pure rollout, no GPU), while the main thread
does net-guided batched self-play on GPU.

---

## 9. Migration Path

### Phase 1: Game engine (1-2 days)

**Port:** `HexGame` with `make`, `unmake`, `legal_moves`, `zoi_moves`,
`candidates`, `check_win`, `clone`.

**Validate:**
1. Port the existing `test_game.py` tests to call `hexgo.HexGame` instead of
   `game.HexGame`.  Run both side by side and assert identical results.
2. Property-based testing: run N random games with both implementations in
   lockstep, asserting board state equality after every move.

```python
# tests/test_rust_game.py
from hexgo import HexGame as RustGame
from game import HexGame as PyGame

def test_random_game_parity():
    """Play 1000 random games, verify Rust and Python produce identical states."""
    import random
    for _ in range(1000):
        rg, pg = RustGame(), PyGame()
        for _ in range(100):
            moves_r = sorted(rg.legal_moves())
            moves_p = sorted(pg.legal_moves())
            assert moves_r == moves_p
            if not moves_r:
                break
            m = random.choice(moves_r)
            rg.make(*m)
            pg.make(*m)
            assert rg.winner == pg.winner
            assert rg.current_player == pg.current_player
            assert rg.placements_in_turn == pg.placements_in_turn
            if rg.winner:
                break
```

**Integration:** In `train.py`, swap `from game import HexGame` to
`from hexgo import HexGame`.  The rest of the code should work unchanged because
the API is identical.

**Expected speedup:** 10-50x on `make`/`unmake` (Python dict + set overhead
eliminated), 5-10x on `legal_moves` and `zoi_moves` (tight loops over integers
instead of Python tuples).

### Phase 2: MCTS tree (3-5 days)

**Port:** `Node`, `Arena`, `mcts()` (pure rollout), `best_child`, `expand`,
`backprop`.

**Validate:**
1. Run `mcts()` on the same game state with both Python and Rust, assert they
   return the same move (seed the RNG identically).
2. Play Rust-MCTS vs Python-MCTS in a tournament, verify roughly equal strength.

**Integration:** Replace `from mcts import mcts` with `from hexgo import mcts`
for pure-rollout callers.

### Phase 3: Net-guided MCTS callback (2-3 days)

**Port:** `mcts_with_net()` with Python callback for evaluation.

**Key challenge:** The `Python::with_gil()` call adds ~1 microsecond per leaf
evaluation.  This is negligible compared to the GPU inference time (~1-6ms),
but the GIL means only one thread can call back at a time.  This is fine for
the unbatched ELO evaluation path.

**Integration:** Replace `mcts.mcts_with_net()` in `elo.py` with
`hexgo.mcts_with_net()`.

### Phase 4: Batched lockstep integration (3-5 days, optional)

**Port:** `batched_mcts_step()` / `batched_mcts_receive()` as described in
section 7 Option B.  This moves the per-sim tree traversal loop from Python
to Rust.

**Integration:** Rewrite the inner sim loop of `batched_self_play()` to call
the Rust step functions.  The Gumbel/SH logic at the root level can remain in
Python since it runs once per move (not per sim).

### Phase 5: Parallel self-play (1-2 days)

**Port:** `parallel_self_play()` with rayon for pure-rollout games.

**Integration:** Use for Eisenstein curriculum games and for warm-up self-play
before the net is trained.  Net-guided games continue using the existing batched
lockstep path.

---

## 10. Gotchas and Notes

1. **`winner` representation:** Python uses `None` vs `1`/`2`.  Rust uses `0`
   for no winner.  The PyO3 getter should return `None` in Python when
   `self.winner == 0`:

   ```rust
   #[getter]
   fn winner(&self) -> Option<Player> {
       if self.winner == NO_PLAYER { None } else { Some(self.winner) }
   }
   ```

2. **Coord tuple conversion:** PyO3 automatically converts `(i16, i16)` to/from
   Python tuples of `int`.  No manual conversion needed.

3. **`board` access:** Python code indexes `game.board[(q,r)]` directly.
   Expose `board_dict()` returning a Python dict, or `board_get(q, r) -> Option<Player>`.
   `encode_board()` in `net.py` iterates `game.board.items()` -- needs to
   work with the Rust `board_dict()` return value.

4. **Maturin + Windows:** Use `maturin develop --release` in a venv.  The
   specific Python at `C:\Program Files\Python312\python.exe` must be used
   (per project CLAUDE.md).  Set `MATURIN_PYTHON` env var or use
   `maturin develop --interpreter "C:\Program Files\Python312\python.exe"`.

5. **FxHashMap iteration order:** Non-deterministic.  `legal_moves()` and
   `candidates_list()` will return moves in a different order than Python's
   `dict`/`set`.  This is fine -- MCTS does not depend on ordering -- but
   seeded-replay tests must sort before comparing.

6. **Memory:** Arena pre-allocation.  For 100 sims with top_k=16, the tree
   grows to ~1600 nodes.  Each `Node` is 24 bytes.  Total ~38 KB per tree.
   For 20 concurrent games: ~760 KB.  Negligible.

7. **`encode_board` stays in Python:** This function builds numpy arrays for
   the neural net (history planes, axis-chain features, D6 transforms).  It
   reads `game.board`, `game.move_history`, and `game.player_history` from the
   Rust `HexGame` via PyO3.  No need to port it until profiling shows it matters.
