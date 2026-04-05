//! Board encoding for neural net input — Rust port of net.py:encode_board.
//!
//! Channel layout (IN_CH = 17, BOARD_SIZE = 18):
//!   0   — player 1 current pieces
//!   1   — player 2 current pieces
//!   2   — to-move plane (0.0=p1, 1.0=p2)
//!   3-6 — player 1 last N_HISTORY moves (most recent = ch 3)
//!   7-10— player 2 last N_HISTORY moves (most recent = ch 7)
//!   11  — MY axis-0 (1,0):  chain length normalised to [0,1]
//!   12  — MY axis-1 (0,1):  chain length normalised to [0,1]
//!   13  — MY axis-2 (1,-1): chain length normalised to [0,1]
//!   14  — OPP axis-0 (1,0):  chain length normalised to [0,1]
//!   15  — OPP axis-1 (0,1):  chain length normalised to [0,1]
//!   16  — OPP axis-2 (1,-1): chain length normalised to [0,1]

use crate::game::HexGame;
use crate::types::*;

pub const BOARD_SIZE: usize = 18;
pub const N_HISTORY: usize = 4;
pub const IN_CH: usize = 3 + 2 * N_HISTORY + 6; // 17
pub const N_RECENT: usize = 20;
pub const PLANE: usize = BOARD_SIZE * BOARD_SIZE;

/// Compute the board origin (center of recent moves).
fn compute_origin(game: &HexGame) -> (i16, i16) {
    if game.move_history.is_empty() {
        return (0, 0);
    }
    let n = game.move_history.len();
    let start = if n > N_RECENT { n - N_RECENT } else { 0 };
    let recent = &game.move_history[start..];
    let cq: f32 = recent.iter().map(|m| m.0 as f32).sum::<f32>() / recent.len() as f32;
    let cr: f32 = recent.iter().map(|m| m.1 as f32).sum::<f32>() / recent.len() as f32;
    (cq.round() as i16, cr.round() as i16)
}

/// Map axial (q, r) to grid (row, col). Returns None if out of window.
#[inline]
pub fn move_to_grid(q: i16, r: i16, oq: i16, or_: i16) -> Option<(usize, usize)> {
    let half = BOARD_SIZE as i16 / 2;
    let col = q - oq + half;
    let row = r - or_ + half;
    if col >= 0 && (col as usize) < BOARD_SIZE && row >= 0 && (row as usize) < BOARD_SIZE {
        Some((row as usize, col as usize))
    } else {
        None
    }
}

/// Full board encoding with axis-chain planes. Returns (flat array, origin).
pub fn encode_board(game: &HexGame) -> (Vec<f32>, (i16, i16)) {
    let (oq, or_) = compute_origin(game);
    let mut arr = vec![0.0f32; IN_CH * PLANE];

    let cp = game.current_player;
    let half = BOARD_SIZE as i16 / 2;

    // Channel 2: to-move plane
    let to_move = if cp == P1 { 0.0f32 } else { 1.0f32 };
    for i in 0..PLANE {
        arr[2 * PLANE + i] = to_move;
    }

    // Channels 0-1: current board state
    let board = game.board_ref();
    for (&(q, r), &p) in board.iter() {
        if let Some((row, col)) = move_to_grid(q, r, oq, or_) {
            arr[((p - 1) as usize) * PLANE + row * BOARD_SIZE + col] = 1.0;
        }
    }

    // Channels 3-10: history planes
    encode_history(game, oq, or_, half, &mut arr);

    // Channels 11-16: axis-chain planes
    encode_axis_chains(game, oq, or_, &mut arr);

    (arr, (oq, or_))
}

/// Fast encoding — skips axis-chain planes (channels 11-16).
pub fn encode_board_fast(game: &HexGame) -> (Vec<f32>, (i16, i16)) {
    let (oq, or_) = compute_origin(game);
    let mut arr = vec![0.0f32; IN_CH * PLANE];

    let cp = game.current_player;
    let half = BOARD_SIZE as i16 / 2;

    // Channel 2: to-move plane
    let to_move = if cp == P1 { 0.0f32 } else { 1.0f32 };
    for i in 0..PLANE {
        arr[2 * PLANE + i] = to_move;
    }

    // Channels 0-1: current board state
    let board = game.board_ref();
    for (&(q, r), &p) in board.iter() {
        if let Some((row, col)) = move_to_grid(q, r, oq, or_) {
            arr[((p - 1) as usize) * PLANE + row * BOARD_SIZE + col] = 1.0;
        }
    }

    // Channels 3-10: history planes
    encode_history(game, oq, or_, half, &mut arr);

    (arr, (oq, or_))
}

/// Encode history planes (channels 3-10).
fn encode_history(game: &HexGame, oq: i16, or_: i16, _half: i16, arr: &mut [f32]) {
    let mut p1_count = 0usize;
    let mut p2_count = 0usize;

    for i in (0..game.move_history.len()).rev() {
        let (q, r) = game.move_history[i];
        let mp = game.player_history[i];

        if mp == P1 && p1_count < N_HISTORY {
            if let Some((row, col)) = move_to_grid(q, r, oq, or_) {
                arr[(3 + p1_count) * PLANE + row * BOARD_SIZE + col] = 1.0;
            }
            p1_count += 1;
        } else if mp == P2 && p2_count < N_HISTORY {
            if let Some((row, col)) = move_to_grid(q, r, oq, or_) {
                arr[(7 + p2_count) * PLANE + row * BOARD_SIZE + col] = 1.0;
            }
            p2_count += 1;
        }

        if p1_count >= N_HISTORY && p2_count >= N_HISTORY {
            break;
        }
    }
}

/// Encode axis-chain planes (channels 11-16).
fn encode_axis_chains(game: &HexGame, oq: i16, or_: i16, arr: &mut [f32]) {
    let me = game.current_player;
    let opp = 3 - me;
    let board = game.board_ref();
    let candidates = game.candidates_ref();

    for &(q, r) in candidates.iter() {
        let grid = match move_to_grid(q, r, oq, or_) {
            Some(g) => g,
            None => continue,
        };
        let (row, col) = grid;

        for (player, ch_base) in [(me, 11usize), (opp, 14usize)] {
            for (axis_idx, &(dq, dr)) in AXES.iter().enumerate() {
                let mut run: u32 = 1;
                for sign in [1i16, -1] {
                    let (mut nq, mut nr) = (q + sign * dq, r + sign * dr);
                    while board.get(&(nq, nr)) == Some(&player) {
                        run += 1;
                        nq += sign * dq;
                        nr += sign * dr;
                    }
                }
                let val = (run as f32 / WIN_LENGTH as f32).min(1.0);
                arr[(ch_base + axis_idx) * PLANE + row * BOARD_SIZE + col] = val;
            }
        }
    }
}

/// Extract top-K legal moves from a spatial logit map.
/// Returns vec of ((q,r), logit) sorted by logit descending.
pub fn top_k_from_logit_map(
    logit_map: &[f32], // flat S*S
    game: &HexGame,
    oq: i16,
    or_: i16,
    k: usize,
) -> Vec<(Coord, f32)> {
    top_k_from_logit_map_zoi(logit_map, game, oq, or_, k, 0, 16)
}

/// Like top_k_from_logit_map but with optional ZOI filtering.
/// If zoi_margin > 0, only considers moves within zoi_margin hex steps of recent play.
pub fn top_k_from_logit_map_zoi(
    logit_map: &[f32], // flat S*S
    game: &HexGame,
    oq: i16,
    or_: i16,
    k: usize,
    zoi_margin: i16,
    zoi_lookback: usize,
) -> Vec<(Coord, f32)> {
    let board = game.board_ref();
    let half = BOARD_SIZE as i16 / 2;

    // Build ZOI set if margin > 0
    let zoi_set: Option<std::collections::HashSet<Coord>> = if zoi_margin > 0 {
        let zoi = game.zoi_moves(zoi_margin, zoi_lookback);
        Some(zoi.into_iter().collect())
    } else {
        None
    };

    // Collect all (index, logit) and partial sort
    let n_top = (k * 8).min(PLANE);
    let mut indexed: Vec<(usize, f32)> = (0..PLANE).map(|i| (i, logit_map[i])).collect();
    let pivot = PLANE.saturating_sub(n_top);
    indexed.select_nth_unstable_by(pivot, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_slice = &mut indexed[pivot..];
    top_slice.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = Vec::with_capacity(k);
    for &(idx, logit) in top_slice.iter() {
        let row = idx / BOARD_SIZE;
        let col = idx % BOARD_SIZE;
        let q = col as i16 - half + oq;
        let r = row as i16 - half + or_;
        if board.contains_key(&(q, r)) {
            continue;
        }
        // ZOI filter: skip moves outside the zone of interest
        if let Some(ref zoi) = zoi_set {
            if !zoi.contains(&(q, r)) {
                continue;
            }
        }
        result.push(((q, r), logit));
        if result.len() >= k {
            break;
        }
    }
    // Fallback: if ZOI was too restrictive, retry without it
    if result.len() < 3 && zoi_set.is_some() {
        return top_k_from_logit_map_zoi(logit_map, game, oq, or_, k, 0, 0);
    }
    result
}
