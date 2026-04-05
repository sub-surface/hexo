use pyo3::prelude::*;
use rayon::prelude::*;

use crate::game::HexGame;
use crate::mcts::mcts_pure;
use crate::types::*;

/// Result from one self-play game.
#[pyclass]
pub struct GameResult {
    #[pyo3(get)]
    pub winner: Option<Player>,
    #[pyo3(get)]
    pub moves: Vec<(i16, i16)>,
    #[pyo3(get)]
    pub num_moves: usize,
}

/// Play N independent games in parallel using rayon.
/// Pure rollout only (no net) — useful for Eisenstein curriculum,
/// warm-up games, and testing.
///
/// Uses all available cores on the i9-14900KF (24 threads).
#[pyfunction]
#[pyo3(signature = (num_games, num_sims=200, c_puct=1.5, fpu_reduction=0.2, max_moves=200))]
pub fn parallel_self_play(
    num_games: usize,
    num_sims: u32,
    c_puct: f32,
    fpu_reduction: f32,
    max_moves: usize,
) -> Vec<GameResult> {
    (0..num_games)
        .into_par_iter()
        .map(|_| {
            let mut game = HexGame::new();
            let mut moves = Vec::new();

            while game.winner_raw() == NO_PLAYER && moves.len() < max_moves {
                let legal = game.legal_moves();
                if legal.is_empty() {
                    break;
                }
                let m = mcts_pure(&mut game, num_sims, c_puct, fpu_reduction);
                game.make_move(m.0, m.1);
                moves.push(m);
            }

            let winner = if game.winner_raw() == NO_PLAYER {
                None
            } else {
                Some(game.winner_raw())
            };
            let num_moves = moves.len();
            GameResult {
                winner,
                moves,
                num_moves,
            }
        })
        .collect()
}

/// Play N games of a HexGame agent vs EisensteinGreedy-style logic.
/// The "greedy" side picks the move that maximizes its longest chain
/// along any axis (or blocks the opponent's if defensive=true).
///
/// This is a Rust reimplementation of EisensteinGreedyAgent.choose_move()
/// so curriculum games don't need Python at all.
#[pyfunction]
#[pyo3(signature = (num_games, num_sims=100, c_puct=1.5, fpu_reduction=0.2,
                    max_moves=200, defensive=true))]
pub fn parallel_eisenstein_games(
    num_games: usize,
    num_sims: u32,
    c_puct: f32,
    fpu_reduction: f32,
    max_moves: usize,
    defensive: bool,
) -> Vec<GameResult> {
    (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut game = HexGame::new();
            let mut moves = Vec::new();
            let mcts_is_p1 = i % 2 == 0;

            while game.winner_raw() == NO_PLAYER && moves.len() < max_moves {
                let cp = game.current_player;
                let is_mcts_turn = (cp == P1) == mcts_is_p1;

                let m = if is_mcts_turn {
                    mcts_pure(&mut game, num_sims, c_puct, fpu_reduction)
                } else {
                    eisenstein_choose(&game, defensive)
                };

                game.make_move(m.0, m.1);
                moves.push(m);
            }

            let winner = if game.winner_raw() == NO_PLAYER {
                None
            } else {
                Some(game.winner_raw())
            };
            let num_moves = moves.len();
            GameResult {
                winner,
                moves,
                num_moves,
            }
        })
        .collect()
}

/// Eisenstein greedy move selection — pure Rust, no Python.
fn eisenstein_choose(game: &HexGame, defensive: bool) -> Coord {
    let player = game.current_player;
    let opponent = 3 - player;
    let candidates = game.candidates_ref();

    let mut best_move = (0i16, 0i16);
    let mut best_score: i32 = -1;

    for &(q, r) in candidates {
        if game.board_ref().contains_key(&(q, r)) {
            continue;
        }
        let own = chain_if_placed(game, q, r, player);
        let block = if defensive {
            chain_if_placed(game, q, r, opponent)
        } else {
            0
        };
        let score = own.max(block) as i32;
        if score > best_score {
            best_score = score;
            best_move = (q, r);
        }
    }

    if best_score < 0 {
        // Fallback: pick first candidate
        candidates.iter().copied().next().unwrap_or((0, 0))
    } else {
        best_move
    }
}

/// Longest chain `player` would have along any axis if placed at (q, r).
fn chain_if_placed(game: &HexGame, q: i16, r: i16, player: Player) -> u32 {
    let board = game.board_ref();
    let mut best: u32 = 1;
    for &(dq, dr) in &AXES {
        let mut count: u32 = 1;
        for sign in [1i16, -1] {
            let (mut nq, mut nr) = (q + sign * dq, r + sign * dr);
            while board.get(&(nq, nr)) == Some(&player) {
                count += 1;
                nq += sign * dq;
                nr += sign * dr;
            }
        }
        best = best.max(count);
    }
    best
}
