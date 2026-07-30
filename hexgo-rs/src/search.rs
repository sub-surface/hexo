//! Alpha-beta minimax search with the theory-informed evaluation from
//! hexo-theory/competition/hexo_bot.py, ported to Rust for speed.
//!
//! The Python version's bottleneck is not the (already vectorized, numpy)
//! leaf evaluation -- it's the tree traversal itself: every node does
//! `dict(stones)` to clone the position before trying a move. HexGame
//! already has an undo stack (make_move/unmake_move) for exactly this
//! reason (built for MCTS rollouts, equally applicable here), so this
//! search never clones the board at all.
//!
//! CORRECTNESS DISCIPLINE: every scoring function here is a direct,
//! deliberately unoptimized port of the corresponding Python function in
//! hexo_bot.py (cell_score, threat_count, _global_potential,
//! _global_threats). Do not "clean up" the arithmetic without re-running
//! search_selftest against the Python reference on the same positions --
//! this session found three separate scoring bugs (a squared fork term, a
//! discarded fork channel, a shallower-than-intended search depth) by
//! exactly this kind of cross-check, not by inspection.

use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

use crate::game::HexGame;
use crate::types::*;

const AXES3: [Coord; 3] = [(1, 0), (0, 1), (1, -1)];

/// Port of arena.py's cell_score: ES-potential value of placing `player`
/// at (q, r), NOT counting the hypothetical stone itself (own only counts
/// the OTHER 5 cells of each window) -- an approximate ranking signal, not
/// an exact post-placement potential.
fn cell_score(board: &FxHashMap<Coord, Player>, q: i16, r: i16, player: Player) -> f64 {
    let opp = 3 - player;
    let mut score = 0.0;
    for &(dq, dr) in &AXES3 {
        for offset in -5..=0i16 {
            let mut own = 0u32;
            let mut blocked = false;
            for i in 0..6i16 {
                let cq = q + dq * (offset + i);
                let cr = r + dr * (offset + i);
                if cq == q && cr == r {
                    continue;
                }
                match board.get(&(cq, cr)) {
                    Some(&p) if p == opp => {
                        blocked = true;
                        break;
                    }
                    Some(&p) if p == player => own += 1,
                    _ => {}
                }
            }
            if !blocked {
                score += 4f64.powi(own as i32);
            }
        }
    }
    score
}

/// Port of arena.py's threat_count: number of live windows through (q, r)
/// that would hold >= min_own of `player`'s stones (including the
/// hypothetical placement itself) and no opponent stone.
fn threat_count(board: &FxHashMap<Coord, Player>, q: i16, r: i16, player: Player, min_own: u32) -> u32 {
    let opp = 3 - player;
    let mut count = 0u32;
    for &(dq, dr) in &AXES3 {
        for offset in -5..=0i16 {
            let mut own = 1u32; // the stone being placed
            let mut blocked = false;
            for i in 0..6i16 {
                let cq = q + dq * (offset + i);
                let cr = r + dr * (offset + i);
                if cq == q && cr == r {
                    continue;
                }
                match board.get(&(cq, cr)) {
                    Some(&p) if p == opp => {
                        blocked = true;
                        break;
                    }
                    Some(&p) if p == player => own += 1,
                    _ => {}
                }
            }
            if !blocked && own >= min_own {
                count += 1;
            }
        }
    }
    count
}

fn score_move(board: &FxHashMap<Coord, Player>, q: i16, r: i16, player: Player,
              defence_weight: f64, fork_bonus: f64) -> f64 {
    let opp = 3 - player;
    let off = cell_score(board, q, r, player) + cell_score(board, q, r, opp) * defence_weight;
    let my_fork = threat_count(board, q, r, player, 4).saturating_sub(1) as f64;
    let opp_fork = threat_count(board, q, r, opp, 4).saturating_sub(1) as f64;
    off + (my_fork + opp_fork * defence_weight) * fork_bonus
}

fn top_candidates(game: &HexGame, player: Player, k: usize,
                  defence_weight: f64, fork_bonus: f64) -> Vec<Coord> {
    let board = game.board_ref();
    let mut scored: Vec<(f64, Coord)> = game
        .candidates_ref()
        .iter()
        .map(|&(q, r)| (score_move(board, q, r, player, defence_weight, fork_bonus), (q, r)))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.truncate(k);
    scored.into_iter().map(|(_, c)| c).collect()
}

/// Port of _global_potential / _global_threats: sum over every DISTINCT
/// live window touching an existing `me`-stone. Windows are deduplicated
/// via (axis_idx, start_q, start_r) since a window is touched by up to 6
/// of its own stones.
fn global_scan(board: &FxHashMap<Coord, Player>, me: Player) -> (f64, f64) {
    let opp = 3 - me;
    let mut seen: FxHashSet<(u8, i16, i16)> = FxHashSet::default();
    let mut potential = 0.0;
    let mut threats = 0.0;
    for (&(sq, sr), &p) in board.iter() {
        if p != me {
            continue;
        }
        for (axis_idx, &(dq, dr)) in AXES3.iter().enumerate() {
            for offset in 0..6i16 {
                let start = (sq - offset * dq, sr - offset * dr);
                let key = (axis_idx as u8, start.0, start.1);
                if !seen.insert(key) {
                    continue;
                }
                let mut own = 0u32;
                let mut blocked = false;
                for i in 0..6i16 {
                    let c = (start.0 + i * dq, start.1 + i * dr);
                    match board.get(&c) {
                        Some(&pp) if pp == opp => {
                            blocked = true;
                            break;
                        }
                        Some(&pp) if pp == me => own += 1,
                        _ => {}
                    }
                }
                if !blocked && own > 0 {
                    potential += 4f64.powi(own as i32);
                    if own >= 4 {
                        threats += 1.0;
                    }
                }
            }
        }
    }
    (potential, threats)
}

/// Every LIVE window (no opponent stone) touching an existing `player`
/// stone with >= min_own of `player`'s stones. With min_own = WIN_LENGTH-2
/// (4) these are exactly the windows completable within a single 1-2-2
/// turn. Exact port of hexo_bot.py's `_hot_windows` (see the 2026-07-08
/// fork-ablation writeup: results/fork_ablation_diagnostic.json / docs/
/// theory/2026-07-08-pairing-thresholds-and-game-values.md's tau(O)
/// covering-capacity result). Dedup follows global_scan's (axis, start)
/// keying since a window is touched by up to 6 of its own stones.
fn hot_windows(board: &FxHashMap<Coord, Player>, player: Player, min_own: u32) -> Vec<Vec<Coord>> {
    let opp = 3 - player;
    let mut seen: FxHashSet<(u8, i16, i16)> = FxHashSet::default();
    let mut hot = Vec::new();
    for (&(sq, sr), &p) in board.iter() {
        if p != player {
            continue;
        }
        for (axis_idx, &(dq, dr)) in AXES3.iter().enumerate() {
            for offset in 0..6i16 {
                let start = (sq - offset * dq, sr - offset * dr);
                let key = (axis_idx as u8, start.0, start.1);
                if !seen.insert(key) {
                    continue;
                }
                let mut own = 0u32;
                let mut blocked = false;
                let mut empties = Vec::new();
                for i in 0..6i16 {
                    let c = (start.0 + i * dq, start.1 + i * dr);
                    match board.get(&c) {
                        Some(&pp) if pp == opp => {
                            blocked = true;
                            break;
                        }
                        Some(&pp) if pp == player => own += 1,
                        _ => empties.push(c),
                    }
                }
                if !blocked && own >= min_own && !empties.is_empty() {
                    hot.push(empties);
                }
            }
        }
    }
    hot
}

enum Forced {
    Win,
    Loss,
}

/// Exact tactical verdict for `player`, who has `remaining` (1 or 2)
/// placements left in their CURRENT turn. Win if some live window
/// already completes within `remaining` placements; Loss if the
/// opponent holds >=2 live brink windows (completable within the
/// opponent's own upcoming full 2-stone turn) whose empty cells can't be
/// jointly covered by `remaining` cells -- an unblockable fork regardless
/// of what `player` does. Checked at EVERY search node (not just the
/// root) so a narrow top-k static beam can never silently prune away a
/// forced continuation -- exactly the mechanism the 2026-07-08 fork-term
/// ablation traced deep_minimax's losses to fast_minimax to (24-0 in
/// both fork_bonus=60 and fork_bonus=0 configurations -- the search
/// architecture was the bug, not the eval feature set). Also the exact
/// algorithmic form of SealBot's _find_instant_win / _filter_turns_by_
/// threats (papers/misc/hexbot-building-framework/opponents/ramora/ai.py).
fn forced_result(board: &FxHashMap<Coord, Player>, player: Player, remaining: u32) -> Option<Forced> {
    if !hot_windows(board, player, 6 - remaining).is_empty() {
        return Some(Forced::Win);
    }
    let opp = 3 - player;
    let opp_hot = hot_windows(board, opp, 4);
    if opp_hot.len() < 2 {
        return None;
    }
    let mut cells: Vec<Coord> = Vec::new();
    for w in &opp_hot {
        for &c in w {
            if !cells.contains(&c) {
                cells.push(c);
            }
        }
    }
    if remaining >= 2 {
        for i in 0..cells.len() {
            for j in i..cells.len() {
                let (c1, c2) = (cells[i], cells[j]);
                if opp_hot.iter().all(|w| w.contains(&c1) || w.contains(&c2)) {
                    return None;
                }
            }
        }
        Some(Forced::Loss)
    } else if cells.iter().any(|&c| opp_hot.iter().all(|w| w.contains(&c))) {
        None
    } else {
        Some(Forced::Loss)
    }
}

/// Optimistic (unimpeded) turns for `player` to complete their nearest-to-
/// done live window. Exact port of hexo_bot.py's `_fastest_force` -- the
/// Connect-6 analogue of the shortest-path/Dijkstra-distance heuristic
/// classically used in Hex-family engines (see docs/theory/2026-07-08-
/// pairing-thresholds-and-game-values.md section 3.2b). Follows global_scan's
/// dedup-by-(axis,start) convention since a window is touched by up to 6 of
/// its own stones.
fn fastest_force(board: &FxHashMap<Coord, Player>, player: Player) -> i64 {
    let opp = 3 - player;
    let mut seen: FxHashSet<(u8, i16, i16)> = FxHashSet::default();
    let mut best_own: u32 = 0;
    for (&(sq, sr), &p) in board.iter() {
        if p != player {
            continue;
        }
        for (axis_idx, &(dq, dr)) in AXES3.iter().enumerate() {
            for offset in 0..6i16 {
                let start = (sq - offset * dq, sr - offset * dr);
                let key = (axis_idx as u8, start.0, start.1);
                if !seen.insert(key) {
                    continue;
                }
                let mut own = 0u32;
                let mut blocked = false;
                for i in 0..6i16 {
                    let c = (start.0 + i * dq, start.1 + i * dr);
                    match board.get(&c) {
                        Some(&pp) if pp == opp => {
                            blocked = true;
                            break;
                        }
                        Some(&pp) if pp == player => own += 1,
                        _ => {}
                    }
                }
                if !blocked && own > best_own {
                    best_own = own;
                }
            }
        }
    }
    if best_own == 0 {
        return 99; // no live window touches any of player's stones yet
    }
    (6 - best_own as i64 + 1) / 2 // ceil division
}

/// Center cell of every live window with >= warm_min of player's stones.
/// Exact port of hexo_bot.py's `_warm_window_centers`.
fn warm_window_centers(board: &FxHashMap<Coord, Player>, player: Player, warm_min: u32) -> Vec<Coord> {
    let opp = 3 - player;
    let mut seen: FxHashSet<(u8, i16, i16)> = FxHashSet::default();
    let mut centers = Vec::new();
    for (&(sq, sr), &p) in board.iter() {
        if p != player {
            continue;
        }
        for (axis_idx, &(dq, dr)) in AXES3.iter().enumerate() {
            for offset in 0..6i16 {
                let start = (sq - offset * dq, sr - offset * dr);
                let key = (axis_idx as u8, start.0, start.1);
                if !seen.insert(key) {
                    continue;
                }
                let mut own = 0u32;
                let mut blocked = false;
                for i in 0..6i16 {
                    let c = (start.0 + i * dq, start.1 + i * dr);
                    match board.get(&c) {
                        Some(&pp) if pp == opp => {
                            blocked = true;
                            break;
                        }
                        Some(&pp) if pp == player => own += 1,
                        _ => {}
                    }
                }
                if !blocked && own >= warm_min {
                    centers.push((start.0 + 2 * dq, start.1 + 2 * dr));
                }
            }
        }
    }
    centers
}

/// Greedy clustering of warm-window centers into distinct 'fronts'. Exact
/// port of hexo_bot.py's `_front_count` -- see its doc comment and docs/
/// theory/2026-07-08-pairing-thresholds-and-game-values.md section 3.2b
/// for why 3+ is the threshold that matters (a proven-sufficient case for
/// EXACTLY 3 mature, pairwise-uncoverable brink windows; this softer
/// pre-emptive count is an empirically-calibrated heuristic, not a theorem).
fn front_count(board: &FxHashMap<Coord, Player>, player: Player, warm_min: u32, radius: f64) -> usize {
    let centers = warm_window_centers(board, player, warm_min);
    let mut clusters: Vec<(f64, f64, f64)> = Vec::new(); // (sum_q, sum_r, n)
    for (q, r) in centers {
        let (q, r) = (q as f64, r as f64);
        let mut placed = false;
        for c in clusters.iter_mut() {
            let (cq, cr) = (c.0 / c.2, c.1 / c.2);
            let d = (q - cq).abs().max((r - cr).abs()).max(((q - cq) + (r - cr)).abs());
            if d <= radius {
                c.0 += q;
                c.1 += r;
                c.2 += 1.0;
                placed = true;
                break;
            }
        }
        if !placed {
            clusters.push((q, r, 1.0));
        }
    }
    clusters.len()
}

fn board_eval(game: &HexGame, me: Player, defence_weight: f64, fork_bonus: f64,
              race_weight: f64, front_bonus: f64, front_warm_min: u32, front_radius: f64) -> f64 {
    let opp = 3 - me;
    let board = game.board_ref();
    let (my_pot, my_thr) = global_scan(board, me);
    let (opp_pot, opp_thr) = global_scan(board, opp);
    let pot = my_pot - defence_weight * opp_pot;
    let my_fork = (my_thr - 1.0).max(0.0);
    let opp_fork = (opp_thr - 1.0).max(0.0);
    let mut val = pot + (my_fork + opp_fork * defence_weight) * fork_bonus;
    if race_weight != 0.0 {
        val += race_weight * (fastest_force(board, opp) - fastest_force(board, me)) as f64;
    }
    if front_bonus != 0.0 {
        let my_fronts = front_count(board, me, front_warm_min, front_radius).saturating_sub(2) as f64;
        let opp_fronts = front_count(board, opp, front_warm_min, front_radius).saturating_sub(2) as f64;
        val += front_bonus * (my_fronts - defence_weight * opp_fronts);
    }
    val
}

fn stones_this_turn(game: &HexGame) -> u8 {
    if game.move_history.len() <= 1 { 1 } else { 2 }
}

fn build_seq(n_plies: usize, me: Player, opp: Player, remaining_this_turn: u8) -> Vec<Player> {
    let mut seq = Vec::with_capacity(n_plies);
    let mut remaining = remaining_this_turn;
    let mut player = me;
    for _ in 0..n_plies {
        seq.push(player);
        remaining -= 1;
        if remaining == 0 {
            player = if player == me { opp } else { me };
            remaining = 2;
        }
    }
    seq
}

const BIG: f64 = 1e12;

struct SearchCtx {
    me: Player,
    top_k: usize,
    defence_weight: f64,
    fork_bonus: f64,
    race_weight: f64,
    front_bonus: f64,
    front_warm_min: u32,
    front_radius: f64,
    deadline: Instant,
    node_count: u64,
    check_every: u64,
    timed_out: bool,
}

fn search(game: &mut HexGame, ctx: &mut SearchCtx, seq: &[Player], idx: usize,
          mut alpha: f64, mut beta: f64) -> f64 {
    if ctx.timed_out {
        return 0.0;
    }
    ctx.node_count += 1;
    if ctx.node_count % ctx.check_every == 0 && Instant::now() > ctx.deadline {
        ctx.timed_out = true;
        return 0.0;
    }
    if idx == seq.len() {
        return board_eval(game, ctx.me, ctx.defence_weight, ctx.fork_bonus,
                           ctx.race_weight, ctx.front_bonus, ctx.front_warm_min, ctx.front_radius);
    }
    let player = seq[idx];
    let remaining_here: u32 = if idx + 1 < seq.len() && seq[idx + 1] == player { 2 } else { 1 };
    match forced_result(game.board_ref(), player, remaining_here) {
        Some(Forced::Win) => {
            return if player == ctx.me { BIG - idx as f64 } else { -BIG + idx as f64 };
        }
        Some(Forced::Loss) => {
            return if player == ctx.me { -BIG + idx as f64 } else { BIG - idx as f64 };
        }
        None => {}
    }
    let cands = top_candidates(game, player, ctx.top_k, ctx.defence_weight, ctx.fork_bonus);
    let mut best = if player == ctx.me { -BIG } else { BIG };
    for (q, r) in cands {
        game.make_move(q, r);
        let is_win = game.winner_raw() != 0;
        let val = if is_win {
            if player == ctx.me { BIG - idx as f64 } else { -BIG + idx as f64 }
        } else {
            search(game, ctx, seq, idx + 1, alpha, beta)
        };
        game.unmake_move();
        if ctx.timed_out {
            return 0.0;
        }
        if player == ctx.me {
            if val > best { best = val; }
            if val > alpha { alpha = val; }
        } else {
            if val < best { best = val; }
            if val < beta { beta = val; }
        }
        if beta <= alpha {
            break;
        }
    }
    best
}

/// Iterative-deepening minimax move for `game.current_player`, exact
/// win/block checked first, then increasing turn-exchange depth until
/// `time_budget_s` runs out. Returns the best move from the last FULLY
/// completed depth (never a partial/timed-out one).
#[pyfunction]
#[pyo3(signature = (game, defence_weight=1.1, fork_bonus=60.0, top_k=5, time_budget_s=0.70,
                    check_every=64, race_weight=0.0, front_bonus=0.0, front_warm_min=3,
                    front_radius=12.0))]
pub fn deep_minimax_move(
    game: &mut HexGame,
    defence_weight: f64,
    fork_bonus: f64,
    top_k: usize,
    time_budget_s: f64,
    check_every: u64,
    race_weight: f64,
    front_bonus: f64,
    front_warm_min: u32,
    front_radius: f64,
) -> Coord {
    let me = game.current_player;
    let opp = 3 - me;

    // exact win/block guarantee at the root, regardless of time budget.
    // "Take a win": use make_move/unmake_move (the engine's own canonical
    // win check via current_player == me) rather than reimplementing it.
    // "Block a loss": make_move always places for current_player (me), so
    // it cannot simulate the OPPONENT placing off-turn -- that one case
    // needs the standalone wins_at() check against a cloned board instead.
    let cands: Vec<Coord> = game.candidates_ref().iter().copied().collect();
    for &(q, r) in &cands {
        game.make_move(q, r);
        let won = game.winner_raw() == me;
        game.unmake_move();
        if won {
            return (q, r);
        }
    }
    let board_snapshot = game.board_ref().clone();
    for &(q, r) in &cands {
        let mut b = board_snapshot.clone();
        b.insert((q, r), opp);
        if wins_at(&b, q, r, opp) {
            return (q, r);
        }
    }

    let limit = stones_this_turn(game);
    let my_remaining = (limit - game.placements_in_turn).max(1) as u8;

    // Brink win: a live window with exactly 2 empty cells and 4 of my
    // stones is an unconditional win this turn (both remaining placements
    // land in it) -- the win/block loops above only catch the 1-empty-
    // cell case via wins_at/make_move. See forced_result's doc comment.
    if my_remaining >= 2 {
        if let Some(w) = hot_windows(game.board_ref(), me, 4).first() {
            return w[0];
        }
    }

    let deadline = Instant::now() + std::time::Duration::from_secs_f64(time_budget_s);
    let mut best_move: Option<Coord> = None;
    let mut n_plies: usize = my_remaining as usize + 2;

    loop {
        let seq = build_seq(n_plies, me, opp, my_remaining);
        let mut ctx = SearchCtx {
            me, top_k, defence_weight, fork_bonus, race_weight, front_bonus,
            front_warm_min, front_radius, deadline,
            node_count: 0, check_every, timed_out: false,
        };
        let root_cands = top_candidates(game, me, top_k.max(6), defence_weight, fork_bonus);
        let mut depth_best: Option<(f64, Coord)> = None;
        let mut aborted = false;
        for (q, r) in root_cands {
            game.make_move(q, r);
            let is_win = game.winner_raw() != 0;
            let val = if is_win {
                BIG
            } else {
                search(game, &mut ctx, &seq, 1, -BIG, BIG)
            };
            game.unmake_move();
            if ctx.timed_out {
                aborted = true;
                break;
            }
            if depth_best.is_none() || val > depth_best.unwrap().0 {
                depth_best = Some((val, (q, r)));
            }
        }
        if !aborted {
            if let Some((_, c)) = depth_best {
                best_move = Some(c);
            }
        }
        if aborted || Instant::now() > deadline {
            break;
        }
        n_plies += 2;
    }

    best_move.unwrap_or_else(|| *game.candidates_ref().iter().next().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_returns_origin() {
        let mut g = HexGame::new();
        let mv = deep_minimax_move(&mut g, 1.1, 60.0, 5, 0.3, 16, 0.0, 0.0, 3, 12.0);
        assert_eq!(mv, (0, 0));
    }

    #[test]
    fn takes_immediate_win() {
        let mut g = HexGame::new();
        // build 5 in a row for player 1 manually via direct field access
        // (make_move alternates players, so drive it by hand instead)
        for &(q, r) in &[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)] {
            // force player 1 to place at each cell regardless of turn order
            // by resetting current_player before each placement (test-only trick)
            g.current_player = 1;
            g.placements_in_turn = 1; // pretend mid-turn so make_move doesn't flip player
            g.make_move(q, r);
        }
        g.current_player = 1;
        g.placements_in_turn = 1;
        let mv = deep_minimax_move(&mut g, 1.1, 60.0, 5, 0.3, 16, 0.0, 0.0, 3, 12.0);
        assert!(mv == (5, 0) || mv == (-1, 0), "expected winning move, got {:?}", mv);
    }

    #[test]
    fn blocks_immediate_loss() {
        let mut g = HexGame::new();
        for &(q, r) in &[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)] {
            g.current_player = 2;
            g.placements_in_turn = 1;
            g.make_move(q, r);
        }
        g.current_player = 1;
        g.placements_in_turn = 1; // player 1 to move, must block player 2's win
        let mv = deep_minimax_move(&mut g, 1.1, 60.0, 5, 0.3, 16, 0.0, 0.0, 3, 12.0);
        assert!(mv == (5, 0) || mv == (-1, 0), "expected block, got {:?}", mv);
    }

    #[test]
    fn cell_score_matches_python_reference_exactly() {
        // Exact cross-check against arena.py's cell_score on the identical
        // board, computed 2026-07-08 (see the search.rs handoff note):
        // arena.cell_score({(0,0):1,(1,0):1,(2,0):1}, 3, 0, 1) == 225.0
        // (hand-verified: 213 from the (1,0)-axis windows [64+64+64+16+4+1]
        // + 6+6 from the two axes that don't touch the existing stones).
        let mut board: FxHashMap<Coord, Player> = FxHashMap::default();
        board.insert((0, 0), 1);
        board.insert((1, 0), 1);
        board.insert((2, 0), 1);
        let s = cell_score(&board, 3, 0, 1);
        assert!((s - 225.0).abs() < 1e-9, "expected exactly 225.0, got {}", s);
    }

    #[test]
    fn threat_count_matches_python_reference_exactly() {
        // arena.threat_count({(0,0):1,(1,0):1,(2,0):1}, 3, 0, 1, min_own=4) == 3
        let mut board: FxHashMap<Coord, Player> = FxHashMap::default();
        board.insert((0, 0), 1);
        board.insert((1, 0), 1);
        board.insert((2, 0), 1);
        let t = threat_count(&board, 3, 0, 1, 4);
        assert_eq!(t, 3, "expected exactly 3, got {}", t);
    }

    #[test]
    fn set_position_then_deep_minimax_finds_forced_win() {
        // Exercises the arena.py/SealBot adapter path: load a position
        // directly (no replay) and confirm search still finds the same
        // forced win as the make()-built equivalent in takes_immediate_win.
        let mut g = HexGame::new();
        let stones = vec![(0, 0, 1u8), (1, 0, 1), (2, 0, 1), (3, 0, 1), (4, 0, 1)];
        g.set_position(stones, 1, 0);
        let mv = deep_minimax_move(&mut g, 1.1, 60.0, 5, 0.3, 16, 0.0, 0.0, 3, 12.0);
        assert!(mv == (5, 0) || mv == (-1, 0), "expected winning move, got {:?}", mv);
    }

    #[test]
    fn forced_result_brink_win_needs_two_placements() {
        // Same fixture as arena.py's selftest: 4-in-a-row genuinely offers
        // 3 overlapping brink windows (extend left/right/straddle).
        let mut board: FxHashMap<Coord, Player> = FxHashMap::default();
        for q in 0..4 {
            board.insert((q, 0), 1);
        }
        assert!(matches!(forced_result(&board, 1, 2), Some(Forced::Win)));
        assert!(forced_result(&board, 1, 1).is_none());
        assert!(hot_windows(&board, 1, 4).iter().all(|w| w.len() == 2));
    }

    #[test]
    fn forced_result_disjoint_double_brink_is_loss_only_with_one_placement() {
        // Stones spread across offsets 0,1,2,5 so exactly one window
        // qualifies per group (not 3, as a contiguous run would give).
        let mut board: FxHashMap<Coord, Player> = FxHashMap::default();
        for &(q, r) in &[(0, 0), (1, 0), (2, 0), (5, 0), (10, 0), (10, 1), (10, 2), (10, 5)] {
            board.insert((q, r), 2);
        }
        assert_eq!(hot_windows(&board, 2, 4).len(), 2);
        assert!(matches!(forced_result(&board, 1, 1), Some(Forced::Loss)));
        assert!(forced_result(&board, 1, 2).is_none());
    }

    #[test]
    fn fastest_force_matches_python_reference_exactly() {
        let empty: FxHashMap<Coord, Player> = FxHashMap::default();
        assert_eq!(fastest_force(&empty, 1), 99);

        let mut brink: FxHashMap<Coord, Player> = FxHashMap::default();
        for q in 0..4 {
            brink.insert((q, 0), 1);
        }
        assert_eq!(fastest_force(&brink, 1), 1);

        let mut two_own: FxHashMap<Coord, Player> = FxHashMap::default();
        two_own.insert((0, 0), 1);
        two_own.insert((1, 0), 1);
        assert_eq!(fastest_force(&two_own, 1), 2);
    }

    #[test]
    fn front_count_matches_python_reference_exactly() {
        let mut board: FxHashMap<Coord, Player> = FxHashMap::default();
        for &(q, r) in &[(0, 0), (1, 0), (2, 0), (100, 100), (101, 100), (102, 100)] {
            board.insert((q, r), 1);
        }
        // radius picked with a comfortable margin either side of the true
        // ~197-203 separation (not a boundary value -- this clustering is
        // greedy/order-dependent, and FxHashMap iteration order differs
        // from Python dict order, so a boundary-exact radius is flaky
        // across languages, not just within one).
        assert_eq!(front_count(&board, 1, 3, 12.0), 2);
        assert_eq!(front_count(&board, 1, 3, 250.0), 1);
        let empty: FxHashMap<Coord, Player> = FxHashMap::default();
        assert_eq!(front_count(&empty, 1, 3, 12.0), 0);
    }

    #[test]
    fn global_scan_matches_python_reference_exactly() {
        // hexo_bot._global_potential / _global_threats on {(0,0)..(3,0): 1}
        // == 1128.0 and 3.0 respectively (computed 2026-07-08).
        let mut board: FxHashMap<Coord, Player> = FxHashMap::default();
        for q in 0..4 {
            board.insert((q, 0), 1);
        }
        let (pot, thr) = global_scan(&board, 1);
        assert!((pot - 1128.0).abs() < 1e-6, "expected potential 1128.0, got {}", pot);
        assert!((thr - 3.0).abs() < 1e-9, "expected threats 3.0, got {}", thr);
    }
}

fn wins_at(board: &FxHashMap<Coord, Player>, q: i16, r: i16, player: Player) -> bool {
    for &(dq, dr) in &AXES3 {
        let mut count = 1u32;
        for sign in [1i16, -1] {
            let (mut nq, mut nr) = (q + sign * dq, r + sign * dr);
            while board.get(&(nq, nr)) == Some(&player) {
                count += 1;
                nq += sign * dq;
                nr += sign * dr;
            }
        }
        if count >= 6 {
            return true;
        }
    }
    false
}
