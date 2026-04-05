use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::prelude::*;
use rand::seq::SliceRandom;

use crate::game::HexGame;
use crate::node::*;
use crate::types::*;

// ---------------------------------------------------------------------------
// Pure rollout MCTS (no neural net)
// ---------------------------------------------------------------------------

fn rollout(game: &mut HexGame, rng: &mut impl Rng) -> f32 {
    let start_player = game.current_player;
    let mut depth: u32 = 0;
    let max_moves: u32 = 150;

    while game.winner_raw() == NO_PLAYER && depth < max_moves {
        let moves = game.legal_moves();
        if moves.is_empty() {
            break;
        }
        let m = *moves.choose(rng).unwrap();
        game.make_move(m.0, m.1);
        depth += 1;
    }

    let result = if game.winner_raw() == NO_PLAYER {
        0.0
    } else if game.winner_raw() == start_player {
        1.0
    } else {
        -1.0
    };

    for _ in 0..depth {
        game.unmake_move();
    }
    result
}

/// Pure rollout MCTS. Returns best move (q, r).
pub fn mcts_pure(
    game: &mut HexGame,
    num_sims: u32,
    c_puct: f32,
    fpu_reduction: f32,
) -> Coord {
    let mut arena = Arena::with_capacity(num_sims as usize * 20);
    let mut rng = thread_rng();

    let root = arena.alloc(Node {
        mov: (0, 0),
        parent: NULL_NODE,
        children_start: 0,
        children_count: 0,
        visits: 0,
        value: 0.0,
        prior: 1.0,
        player: game.current_player,
    });

    // Expand root with uniform priors
    let moves = game.legal_moves();
    if moves.is_empty() {
        return (0, 0);
    }
    let uniform = 1.0 / moves.len() as f32;
    let priors: Vec<f32> = vec![uniform; moves.len()];
    arena.expand(root, &moves, &priors, game.current_player);

    for _ in 0..num_sims {
        let mut node = root;
        let mut depth: u32 = 0;

        // Selection
        while arena.get(node).children_count > 0 && game.winner_raw() == NO_PLAYER {
            node = arena.best_child(node, c_puct, fpu_reduction);
            let m = arena.get(node).mov;
            game.make_move(m.0, m.1);
            depth += 1;
        }

        // Expansion + rollout
        if game.winner_raw() == NO_PLAYER {
            let moves = game.legal_moves();
            if !moves.is_empty() {
                let uniform = 1.0 / moves.len() as f32;
                let priors: Vec<f32> = vec![uniform; moves.len()];
                arena.expand(node, &moves, &priors, game.current_player);

                if arena.get(node).children_count > 0 {
                    let start = arena.get(node).children_start as usize;
                    let count = arena.get(node).children_count as usize;
                    let pick = start + rng.gen_range(0..count);
                    node = pick as NodeIdx;
                    let m = arena.get(node).mov;
                    game.make_move(m.0, m.1);
                    depth += 1;
                }
            }
        }

        let v = if game.winner_raw() != NO_PLAYER {
            1.0 // node's player just won
        } else {
            -rollout(game, &mut rng)
        };

        for _ in 0..depth {
            game.unmake_move();
        }

        arena.backprop(node, v);
    }

    arena.best_move(root)
}

// ---------------------------------------------------------------------------
// Net-guided MCTS (calls Python for evaluation)
// ---------------------------------------------------------------------------

/// Net-guided MCTS. Calls `eval_fn(game)` for leaf evaluation.
/// eval_fn returns (value: float, policy: dict[(q,r) -> logit]).
///
/// Dirichlet noise added at root for exploration.
pub fn mcts_with_net_inner(
    game: &mut HexGame,
    eval_fn: &Bound<'_, PyAny>,
    num_sims: u32,
    c_puct: f32,
    fpu_reduction: f32,
    dirichlet_alpha: f64,
    dirichlet_eps: f64,
    top_k: usize,
) -> PyResult<Coord> {
    let mut arena = Arena::with_capacity(num_sims as usize * 20);

    let root = arena.alloc(Node {
        mov: (0, 0),
        parent: NULL_NODE,
        children_start: 0,
        children_count: 0,
        visits: 0,
        value: 0.0,
        prior: 1.0,
        player: game.current_player,
    });

    // Evaluate root position
    let (root_value, root_policy) = call_eval_fn(eval_fn, game)?;
    let _ = root_value; // root value not used directly

    if root_policy.is_empty() {
        return Ok((0, 0));
    }

    // Top-K by logit
    let mut sorted_policy = root_policy;
    sorted_policy.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sorted_policy.truncate(top_k);

    // Softmax on logits
    let max_logit = sorted_policy
        .iter()
        .map(|p| p.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut priors: Vec<f32> = sorted_policy.iter().map(|p| (p.1 - max_logit).exp()).collect();
    let sum: f32 = priors.iter().sum();
    for p in &mut priors {
        *p /= sum;
    }

    // Dirichlet noise at root
    let noise = dirichlet_sample(sorted_policy.len(), dirichlet_alpha);
    let eps = dirichlet_eps as f32;
    for (i, p) in priors.iter_mut().enumerate() {
        *p = (1.0 - eps) * *p + eps * noise[i];
    }

    let moves: Vec<Coord> = sorted_policy.iter().map(|p| p.0).collect();
    arena.expand(root, &moves, &priors, game.current_player);

    // Simulation loop
    for _ in 0..num_sims {
        let mut node = root;
        let mut depth: u32 = 0;

        // Selection
        while arena.get(node).children_count > 0 && game.winner_raw() == NO_PLAYER {
            node = arena.best_child(node, c_puct, fpu_reduction);
            let m = arena.get(node).mov;
            game.make_move(m.0, m.1);
            depth += 1;
        }

        // Leaf evaluation
        let v = if game.winner_raw() != NO_PLAYER {
            let winner = game.winner_raw();
            let node_player = arena.get(node).player;
            if winner == node_player {
                1.0
            } else {
                -1.0
            }
        } else {
            let (value, leaf_policy) = call_eval_fn(eval_fn, game)?;

            // Value from eval_fn is from current_player's perspective.
            // We need it from node.player's perspective.
            let node_player = arena.get(node).player;
            let v = if node_player != game.current_player {
                -value
            } else {
                value
            };

            // Expand leaf with top-K
            if !leaf_policy.is_empty() {
                let mut sorted = leaf_policy;
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                sorted.truncate(top_k);

                let max_l = sorted
                    .iter()
                    .map(|p| p.1)
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut lpriors: Vec<f32> = sorted.iter().map(|p| (p.1 - max_l).exp()).collect();
                let lsum: f32 = lpriors.iter().sum();
                for p in &mut lpriors {
                    *p /= lsum;
                }

                let lmoves: Vec<Coord> = sorted.iter().map(|p| p.0).collect();
                arena.expand(node, &lmoves, &lpriors, game.current_player);
            }

            v
        };

        for _ in 0..depth {
            game.unmake_move();
        }

        arena.backprop(node, v);
    }

    Ok(arena.best_move(root))
}

/// Call the Python eval function and parse the result.
fn call_eval_fn(
    eval_fn: &Bound<'_, PyAny>,
    game: &HexGame,
) -> PyResult<(f32, Vec<(Coord, f32)>)> {
    let py_game = game.clone_game();
    let result = eval_fn.call1((py_game,))?;
    let tuple: &Bound<'_, PyTuple> = result.downcast()?;
    let value: f32 = tuple.get_item(0)?.extract()?;
    let policy_dict = tuple.get_item(1)?;

    let mut policy = Vec::new();
    // Iterate dict items: {(q, r): logit, ...}
    let items = policy_dict.call_method0("items")?;
    for item in items.try_iter()? {
        let item: Bound<'_, PyAny> = item?;
        let key_val: &Bound<'_, PyTuple> = item.downcast()?;
        let coord: (i16, i16) = key_val.get_item(0)?.extract()?;
        let logit: f32 = key_val.get_item(1)?.extract()?;
        policy.push((coord, logit));
    }

    Ok((value, policy))
}

/// Sample from a symmetric Dirichlet distribution.
fn dirichlet_sample(n: usize, alpha: f64) -> Vec<f32> {
    use rand_distr::{Distribution, Gamma};
    let gamma = Gamma::new(alpha, 1.0).unwrap();
    let mut rng = thread_rng();
    let mut samples: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let sum: f64 = samples.iter().sum();
    for s in &mut samples {
        *s /= sum;
    }
    samples.into_iter().map(|s| s as f32).collect()
}

// ---------------------------------------------------------------------------
// Self-play game (pure rollout, for testing)
// ---------------------------------------------------------------------------

/// Play a complete game via pure MCTS self-play.
/// Returns (winner, num_moves, move_list).
pub fn self_play_game_pure(num_sims: u32, c_puct: f32, fpu_reduction: f32) -> (Player, Vec<Coord>) {
    let mut game = HexGame::new();
    let mut moves = Vec::new();

    while game.winner_raw() == NO_PLAYER {
        let legal = game.legal_moves();
        if legal.is_empty() {
            break;
        }
        let m = mcts_pure(&mut game, num_sims, c_puct, fpu_reduction);
        game.make_move(m.0, m.1);
        moves.push(m);
    }

    (game.winner_raw(), moves)
}

// ---------------------------------------------------------------------------
// PyO3 function wrappers
// ---------------------------------------------------------------------------

/// Pure rollout MCTS exposed to Python.
#[pyfunction]
#[pyo3(signature = (game, num_sims=200, c_puct=1.5, fpu_reduction=0.2))]
pub fn mcts(
    game: &mut HexGame,
    num_sims: u32,
    c_puct: f32,
    fpu_reduction: f32,
) -> (i16, i16) {
    mcts_pure(game, num_sims, c_puct, fpu_reduction)
}

/// Net-guided MCTS exposed to Python.
#[pyfunction]
#[pyo3(signature = (game, eval_fn, num_sims=100, c_puct=1.5, fpu_reduction=0.2,
                    dirichlet_alpha=0.15, dirichlet_eps=0.35, top_k=16))]
pub fn mcts_with_net(
    game: &mut HexGame,
    eval_fn: &Bound<'_, PyAny>,
    num_sims: u32,
    c_puct: f32,
    fpu_reduction: f32,
    dirichlet_alpha: f64,
    dirichlet_eps: f64,
    top_k: usize,
) -> PyResult<(i16, i16)> {
    mcts_with_net_inner(
        game,
        eval_fn,
        num_sims,
        c_puct,
        fpu_reduction,
        dirichlet_alpha,
        dirichlet_eps,
        top_k,
    )
}

/// Play a full self-play game (pure rollout) from Python.
#[pyfunction]
#[pyo3(signature = (num_sims=200, c_puct=1.5, fpu_reduction=0.2))]
pub fn self_play_game(
    num_sims: u32,
    c_puct: f32,
    fpu_reduction: f32,
) -> (u8, Vec<(i16, i16)>) {
    self_play_game_pure(num_sims, c_puct, fpu_reduction)
}
