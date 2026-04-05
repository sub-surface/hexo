//! Batched net-guided self-play — Rust tree traversal + Python GPU callback.
//!
//! All MCTS tree operations (selection, expansion, backprop) run in Rust.
//! The only Python call is for batched GPU neural net evaluation.

use numpy::{PyArray1, PyArray3, PyArrayMethods};
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use rand_distr::{Distribution, Gamma};

use crate::encode::*;
use crate::game::HexGame;
use crate::node::*;
use crate::types::*;

// ── Return types ─────────────────────────────────────────────────────────────

/// Training data for one position.
#[pyclass]
#[derive(Clone)]
pub struct PositionData {
    #[pyo3(get)]
    pub board: Vec<f32>,
    #[pyo3(get)]
    pub policy_target: Vec<f32>,
    #[pyo3(get)]
    pub legal_mask: Vec<f32>,
    #[pyo3(get)]
    pub player: u8,
    #[pyo3(get)]
    pub origin: (i16, i16),
    #[pyo3(get)]
    pub value_est: f32,
}

/// Result from one self-play game.
#[pyclass]
pub struct GameTrainingResult {
    #[pyo3(get)]
    pub winner: Option<u8>,
    #[pyo3(get)]
    pub moves: Vec<(i16, i16)>,
    #[pyo3(get)]
    pub positions: Vec<PositionData>,
}

// ── Per-game state during self-play ──────────────────────────────────────────

struct GameState {
    game: HexGame,
    arena: Arena,
    root: NodeIdx,
    move_count: usize,
    positions: Vec<PositionData>,
    root_origin: (i16, i16),
    root_value_est: f32,
    active: bool,
}

// ── Helper: Dirichlet sample ─────────────────────────────────────────────────

fn dirichlet_sample(n: usize, alpha: f64) -> Vec<f32> {
    let gamma = Gamma::new(alpha, 1.0).unwrap();
    let mut rng = thread_rng();
    let mut samples: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let sum: f64 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    }
    samples.into_iter().map(|s| s as f32).collect()
}

// ── Batched eval callback ────────────────────────────────────────────────────

fn call_batch_eval<'py>(
    py: Python<'py>,
    eval_fn: &Bound<'py, PyAny>,
    flat_boards: &[f32],
    batch_size: usize,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    // Create numpy array [B, IN_CH, S, S]
    let _total = flat_boards.len();
    let np_arr = PyArray1::from_slice(py, flat_boards)
        .reshape([batch_size, IN_CH, BOARD_SIZE, BOARD_SIZE])?;

    let result = eval_fn.call1((np_arr,))?;
    let tuple = result.downcast::<pyo3::types::PyTuple>()?;

    let values_arr = tuple.get_item(0)?;
    let logits_arr = tuple.get_item(1)?;

    let values_py: &Bound<'py, PyArray1<f32>> = values_arr.downcast()?;
    let logits_py: &Bound<'py, PyArray3<f32>> = logits_arr.downcast()?;

    let values: Vec<f32> = values_py.to_vec()?;
    let logits: Vec<f32> = logits_py.to_vec()?;

    Ok((values, logits))
}

// ── Main entry point ─────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (eval_fn, n_games, sims, max_moves, top_k=24,
                    c_puct=2.0, fpu_reduction=0.2,
                    dirichlet_alpha=0.10, dirichlet_eps=0.25,
                    temp_horizon=40, random_opening=6, random_opening_frac=0.5,
                    zoi_margin=0, zoi_lookback=16))]
pub fn batched_self_play(
    py: Python<'_>,
    eval_fn: &Bound<'_, PyAny>,
    n_games: usize,
    sims: u32,
    max_moves: usize,
    top_k: usize,
    c_puct: f32,
    fpu_reduction: f32,
    dirichlet_alpha: f64,
    dirichlet_eps: f64,
    temp_horizon: usize,
    random_opening: usize,
    random_opening_frac: f64,
    zoi_margin: i16,
    zoi_lookback: usize,
) -> PyResult<Vec<GameTrainingResult>> {
    let mut states: Vec<GameState> = (0..n_games)
        .map(|_| GameState {
            game: HexGame::new(),
            arena: Arena::with_capacity(sims as usize * 24),
            root: NULL_NODE,
            move_count: 0,
            positions: Vec::new(),
            root_origin: (0, 0),
            root_value_est: 0.0,
            active: true,
        })
        .collect();

    let mut rng = thread_rng();

    // ── Phase 0: Random openings ─────────────────────────────────────────
    if random_opening > 0 {
        let opening_games: Vec<bool> = (0..n_games)
            .map(|_| rng.gen::<f64>() < random_opening_frac)
            .collect();

        for _step in 0..random_opening {
            for (i, state) in states.iter_mut().enumerate() {
                if !state.active || !opening_games[i] {
                    continue;
                }
                let moves = state.game.legal_moves();
                if moves.is_empty() || state.game.winner_raw() != NO_PLAYER {
                    state.active = false;
                    continue;
                }
                let m = *moves.choose(&mut rng).unwrap();
                state.game.make_move(m.0, m.1);
                state.move_count += 1;
                if state.game.winner_raw() != NO_PLAYER || state.move_count >= max_moves {
                    state.active = false;
                }
            }
        }
    }

    // ── Main move loop ───────────────────────────────────────────────────
    loop {
        let active_indices: Vec<usize> = (0..n_games)
            .filter(|&i| states[i].active)
            .collect();
        if active_indices.is_empty() {
            break;
        }

        // ── Root init: encode boards (parallel via Rayon) ─────────────────
        let games_ref: Vec<&HexGame> = active_indices.iter().map(|&i| &states[i].game).collect();
        let root_encodings: Vec<(Vec<f32>, (i16, i16))> = games_ref
            .par_iter()
            .map(|g| encode_board(g))
            .collect();

        // Stack into batch for GPU
        let batch_size = active_indices.len();
        let mut flat_boards =
            Vec::with_capacity(batch_size * IN_CH * PLANE);
        for (enc, _) in &root_encodings {
            flat_boards.extend_from_slice(enc);
        }

        // GPU callback
        let (root_values, root_logits) =
            call_batch_eval(py, eval_fn, &flat_boards, batch_size)?;

        // ── Parse results + create root nodes (parallel in Rust) ─────────
        // Collect per-game data needed for root creation
        struct RootSetup {
            moves: Vec<Coord>,
            priors: Vec<f32>,
            origin: (i16, i16),
            value_est: f32,
            board_enc: Vec<f32>,
        }

        let root_setups: Vec<Option<RootSetup>> = active_indices
            .iter()
            .enumerate()
            .map(|(j, &i)| {
                    let (ref enc, origin) = root_encodings[j];
                    let value = root_values[j].clamp(-1.0, 1.0);
                    let logit_offset = j * PLANE;
                    let logit_slice = &root_logits[logit_offset..logit_offset + PLANE];

                    let move_logits =
                        top_k_from_logit_map_zoi(logit_slice, &states[i].game, origin.0, origin.1, top_k, zoi_margin, zoi_lookback);
                    if move_logits.is_empty() {
                        return None;
                    }

                    let moves: Vec<Coord> = move_logits.iter().map(|m| m.0).collect();
                    let raw_logits: Vec<f32> = move_logits.iter().map(|m| m.1).collect();

                    // Softmax
                    let max_l = raw_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut priors: Vec<f32> = raw_logits.iter().map(|l| (l - max_l).exp()).collect();
                    let sum: f32 = priors.iter().sum();
                    for p in &mut priors {
                        *p /= sum;
                    }

                    // Dirichlet noise at root
                    let noise = dirichlet_sample(moves.len(), dirichlet_alpha);
                    let eps = dirichlet_eps as f32;
                    for (k, p) in priors.iter_mut().enumerate() {
                        *p = (1.0 - eps) * *p + eps * noise[k];
                    }

                    Some(RootSetup {
                        moves,
                        priors,
                        origin,
                        value_est: value,
                        board_enc: enc.clone(),
                    })
                })
                .collect();

        // Apply root setups to game states (sequential — mutates states)
        let mut still_active = Vec::new();
        for (j, &i) in active_indices.iter().enumerate() {
            let state = &mut states[i];
            match root_setups[j] {
                None => {
                    state.active = false;
                    continue;
                }
                Some(ref setup) => {
                    state.arena.clear();
                    let root = state.arena.alloc(Node {
                        mov: (0, 0),
                        parent: NULL_NODE,
                        children_start: 0,
                        children_count: 0,
                        visits: 0,
                        value: 0.0,
                        prior: 1.0,
                        player: state.game.current_player,
                    });
                    state.arena.expand(root, &setup.moves, &setup.priors, state.game.current_player);
                    state.root = root;
                    state.root_origin = setup.origin;
                    state.root_value_est = setup.value_est;
                    still_active.push(i);
                }
            }
        }

        if still_active.is_empty() {
            break;
        }

        // ── MCTS simulation loop ─────────────────────────────────────────
        for _sim in 0..sims {
            // Selection (sequential — mutates game via make/unmake)
            struct LeafInfo {
                game_idx: usize,
                depth: u32,
                node: NodeIdx,
                needs_eval: bool,
            }

            let mut leaves = Vec::with_capacity(still_active.len());
            for &i in &still_active {
                let state = &mut states[i];
                let mut node = state.root;
                let mut depth: u32 = 0;

                while state.arena.get(node).children_count > 0
                    && state.game.winner_raw() == NO_PLAYER
                {
                    let child = state.arena.best_child(node, c_puct, fpu_reduction);
                    if child == NULL_NODE {
                        break;
                    }
                    node = child;
                    let m = state.arena.get(node).mov;
                    state.game.make_move(m.0, m.1);
                    depth += 1;
                }

                let needs_eval = state.game.winner_raw() == NO_PLAYER
                    && state.arena.get(node).children_count == 0;

                leaves.push(LeafInfo {
                    game_idx: i,
                    depth,
                    node,
                    needs_eval,
                });
            }

            // Collect leaves that need GPU eval
            let eval_indices: Vec<usize> = (0..leaves.len())
                .filter(|&j| leaves[j].needs_eval)
                .collect();

            if !eval_indices.is_empty() {
                // Encode leaf positions (parallel via Rayon)
                let leaf_games: Vec<&HexGame> = eval_indices
                    .iter()
                    .map(|&j| &states[leaves[j].game_idx].game)
                    .collect();
                let leaf_encodings: Vec<(Vec<f32>, (i16, i16))> = leaf_games
                    .par_iter()
                    .map(|g| encode_board_fast(g))
                    .collect();

                // Stack into batch
                let eval_batch_size = eval_indices.len();
                let mut flat_eval =
                    Vec::with_capacity(eval_batch_size * IN_CH * PLANE);
                for (enc, _) in &leaf_encodings {
                    flat_eval.extend_from_slice(enc);
                }

                // GPU callback
                let (eval_values, eval_logits) =
                    call_batch_eval(py, eval_fn, &flat_eval, eval_batch_size)?;

                // Expand leaves and compute values
                for (ej, &j) in eval_indices.iter().enumerate() {
                    let leaf = &leaves[j];
                    let i = leaf.game_idx;
                    let state = &mut states[i];
                    let (_, origin) = &leaf_encodings[ej];

                    let logit_offset = ej * PLANE;
                    let logit_slice = &eval_logits[logit_offset..logit_offset + PLANE];

                    let move_logits = top_k_from_logit_map_zoi(
                        logit_slice, &state.game, origin.0, origin.1, top_k, zoi_margin, zoi_lookback,
                    );

                    if !move_logits.is_empty() {
                        let moves: Vec<Coord> = move_logits.iter().map(|m| m.0).collect();
                        let logits: Vec<f32> = move_logits.iter().map(|m| m.1).collect();
                        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut priors: Vec<f32> =
                            logits.iter().map(|l| (l - max_l).exp()).collect();
                        let sum: f32 = priors.iter().sum();
                        for p in &mut priors {
                            *p /= sum;
                        }
                        state.arena.expand(
                            leaf.node,
                            &moves,
                            &priors,
                            state.game.current_player,
                        );
                    }
                }

                // Backprop for leaves that were evaluated
                for (ej, &j) in eval_indices.iter().enumerate() {
                    let leaf = &leaves[j];
                    let i = leaf.game_idx;
                    let state = &mut states[i];

                    let mut v = eval_values[ej].clamp(-1.0, 1.0);
                    let node_player = state.arena.get(leaf.node).player;
                    if node_player != state.game.current_player {
                        v = -v;
                    }

                    for _ in 0..leaf.depth {
                        state.game.unmake_move();
                    }
                    state.arena.backprop(leaf.node, v);
                }
            }

            // Backprop for terminal/already-expanded leaves
            for (j, leaf) in leaves.iter().enumerate() {
                if leaf.needs_eval {
                    continue; // already handled above
                }
                let i = leaf.game_idx;
                let state = &mut states[i];

                let v = if state.game.winner_raw() != NO_PLAYER {
                    let node_player = state.arena.get(leaf.node).player;
                    if state.game.winner_raw() == node_player {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    0.0 // revisited already-expanded node
                };

                for _ in 0..leaf.depth {
                    state.game.unmake_move();
                }
                state.arena.backprop(leaf.node, v);
            }
        }

        // ── Update root value estimates with post-search values ─────────
        for &i in &still_active {
            let state = &mut states[i];
            let root_node = state.arena.get(state.root);
            if root_node.visits > 0 {
                state.root_value_est = root_node.value / root_node.visits as f32;
            }
        }

        // ── Move selection + training data ───────────────────────────────
        for &i in &still_active {
            let state = &mut states[i];
            let root = state.root;

            if state.arena.get(root).children_count == 0 {
                state.active = false;
                continue;
            }

            let child_visits = state.arena.child_visits(root);
            let total_visits: u32 = child_visits.iter().map(|cv| cv.1).sum();
            if total_visits == 0 {
                state.active = false;
                continue;
            }

            // Temperature-based move selection
            let temp = {
                let progress = state.move_count as f32 / temp_horizon.max(1) as f32;
                (std::f32::consts::FRAC_PI_2 * progress).cos().max(0.05)
            };

            let visits_f: Vec<f32> = child_visits.iter().map(|cv| cv.1 as f32).collect();
            let chosen_idx = if temp < 0.06 {
                // Argmax
                visits_f
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            } else {
                // Temperature sampling
                let vt: Vec<f32> = visits_f.iter().map(|v| v.powf(1.0 / temp)).collect();
                let sum: f32 = vt.iter().sum();
                let dist: Vec<f32> = vt.iter().map(|v| v / sum).collect();
                let r: f32 = rng.gen();
                let mut acc = 0.0;
                let mut picked = 0;
                for (k, d) in dist.iter().enumerate() {
                    acc += d;
                    if r < acc {
                        picked = k;
                        break;
                    }
                    picked = k;
                }
                picked
            };

            let chosen_move = child_visits[chosen_idx].0;

            // Build policy target and legal mask
            let (oq, or_) = state.root_origin;
            let mut policy_target = vec![0.0f32; PLANE];
            let mut legal_mask = vec![0.0f32; PLANE];

            for &(m, v) in &child_visits {
                if let Some((row, col)) = move_to_grid(m.0, m.1, oq, or_) {
                    let d = v as f32 / total_visits as f32;
                    policy_target[row * BOARD_SIZE + col] = d;
                    legal_mask[row * BOARD_SIZE + col] = 1.0;
                }
            }

            // Re-normalize policy target (edge clipping)
            let pt_sum: f32 = policy_target.iter().sum();
            if pt_sum > 0.0 {
                for p in &mut policy_target {
                    *p /= pt_sum;
                }
            }

            // Record position data — use stored root encoding
            let root_setup_idx = still_active.iter().position(|&x| x == i);
            state.positions.push(PositionData {
                board: root_encodings
                    .get(
                        active_indices
                            .iter()
                            .position(|&x| x == i)
                            .unwrap_or(0),
                    )
                    .map(|(enc, _origin): &(Vec<f32>, (i16, i16))| enc.clone())
                    .unwrap_or_default(),
                policy_target,
                legal_mask,
                player: state.game.current_player,
                origin: (oq, or_),
                value_est: state.root_value_est,
            });

            // Apply chosen move
            state.game.make_move(chosen_move.0, chosen_move.1);
            state.move_count += 1;

            if state.game.winner_raw() != NO_PLAYER || state.move_count >= max_moves {
                state.active = false;
            }
        }
    }

    // ── Build results ────────────────────────────────────────────────────
    let results: Vec<GameTrainingResult> = states
        .into_iter()
        .map(|state| {
            let winner = if state.game.winner_raw() == NO_PLAYER {
                None
            } else {
                Some(state.game.winner_raw())
            };
            GameTrainingResult {
                winner,
                moves: state.game.move_history.clone(),
                positions: state.positions,
            }
        })
        .collect();

    Ok(results)
}
