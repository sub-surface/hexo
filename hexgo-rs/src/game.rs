use rustc_hash::{FxHashMap, FxHashSet};
use pyo3::prelude::*;
use crate::types::*;

/// Undo record for a single placement.
#[derive(Clone)]
struct UndoEntry {
    coord: Coord,
    removed_candidates: Vec<Coord>,
    added_candidates: Vec<Coord>,
    prev_placements: u8,
    prev_winner: Player,
    prev_player: Player,
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct HexGame {
    board: FxHashMap<Coord, Player>,
    candidates: FxHashSet<Coord>,

    #[pyo3(get)]
    pub current_player: Player,

    #[pyo3(get)]
    pub placements_in_turn: u8,

    winner_inner: Player, // 0 = no winner

    pub move_history: Vec<Coord>,
    pub player_history: Vec<Player>,
    undo_stack: Vec<UndoEntry>,
}

// Internal game logic (not exposed to Python).
impl HexGame {
    /// Check win along the 3 axes through (q, r).
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
            if count >= WIN_LENGTH {
                return true;
            }
        }
        false
    }

    /// How many stones the current player places this turn (1-2-2 rule).
    #[inline]
    fn stones_this_turn(&self) -> u8 {
        if self.move_history.len() <= 1 { 1 } else { 2 }
    }

    /// Direct read of winner for Rust-internal code (raw u8, not Option).
    #[inline]
    pub fn winner_raw(&self) -> Player {
        self.winner_inner
    }

    /// Direct read of board for Rust-internal code.
    #[inline]
    pub fn board_ref(&self) -> &FxHashMap<Coord, Player> {
        &self.board
    }

    /// Direct read of candidates for Rust-internal code.
    #[inline]
    pub fn candidates_ref(&self) -> &FxHashSet<Coord> {
        &self.candidates
    }
}

#[pymethods]
impl HexGame {
    #[new]
    pub fn new() -> Self {
        let mut candidates = FxHashSet::default();
        candidates.insert((0, 0));
        HexGame {
            board: FxHashMap::default(),
            candidates,
            current_player: P1,
            placements_in_turn: 0,
            winner_inner: NO_PLAYER,
            move_history: Vec::new(),
            player_history: Vec::new(),
            undo_stack: Vec::new(),
        }
    }

    /// Place current player's piece at (q, r). Returns true if legal.
    #[pyo3(name = "make")]
    pub fn py_make(&mut self, q: i16, r: i16) -> bool {
        self.make_move(q, r)
    }

    /// Undo the last placement.
    #[pyo3(name = "unmake")]
    pub fn py_unmake(&mut self) {
        self.unmake_move();
    }

    /// Alias for make().
    pub fn play(&mut self, q: i16, r: i16) -> bool {
        self.make_move(q, r)
    }

    /// Winner as Python int or None.
    #[getter]
    pub fn winner(&self) -> Option<Player> {
        if self.winner_inner == NO_PLAYER {
            None
        } else {
            Some(self.winner_inner)
        }
    }

    /// All empty cells within PLACEMENT_RADIUS of any piece.
    pub fn legal_moves(&self) -> Vec<Coord> {
        if self.board.is_empty() {
            return vec![(0, 0)];
        }
        let mut moves = FxHashSet::default();
        for &(pq, pr) in self.board.keys() {
            for dq in -PLACEMENT_RADIUS..=PLACEMENT_RADIUS {
                let lo = (-PLACEMENT_RADIUS).max(-dq - PLACEMENT_RADIUS);
                let hi = PLACEMENT_RADIUS.min(-dq + PLACEMENT_RADIUS);
                for dr in lo..=hi {
                    let c = (pq + dq, pr + dr);
                    if !self.board.contains_key(&c) {
                        moves.insert(c);
                    }
                }
            }
        }
        moves.into_iter().collect()
    }

    /// Zone-of-interest pruned move list.
    #[pyo3(signature = (margin=6, lookback=16))]
    pub fn zoi_moves(&self, margin: i16, lookback: usize) -> Vec<Coord> {
        if self.move_history.len() < lookback {
            return self.candidates.iter().copied().collect();
        }
        let start = self.move_history.len() - lookback;
        let recent = &self.move_history[start..];
        let threshold = margin as i32 * 2;

        let mut within = Vec::new();
        for &(q, r) in &self.candidates {
            let s = q as i32 + r as i32;
            for &(q0, r0) in recent {
                let dist = (q as i32 - q0 as i32).abs()
                    + (r as i32 - r0 as i32).abs()
                    + (s - q0 as i32 - r0 as i32).abs();
                if dist <= threshold {
                    within.push((q, r));
                    break;
                }
            }
        }

        if within.len() >= 3 {
            within
        } else {
            self.candidates.iter().copied().collect()
        }
    }

    /// Candidates list for Python.
    #[getter]
    pub fn candidates(&self) -> Vec<Coord> {
        self.candidates.iter().copied().collect()
    }

    /// Deep clone (fresh undo stack).
    pub fn clone_game(&self) -> HexGame {
        HexGame {
            board: self.board.clone(),
            candidates: self.candidates.clone(),
            current_player: self.current_player,
            placements_in_turn: self.placements_in_turn,
            winner_inner: self.winner_inner,
            move_history: self.move_history.clone(),
            player_history: self.player_history.clone(),
            undo_stack: Vec::new(),
        }
    }

    /// Board as dict for Python: {(q,r): player, ...}
    #[getter]
    pub fn board(&self) -> FxHashMap<Coord, Player> {
        self.board.clone()
    }

    /// Number of moves played.
    pub fn num_moves(&self) -> usize {
        self.move_history.len()
    }

    /// Move history as list of (q, r).
    #[getter]
    pub fn move_history(&self) -> Vec<Coord> {
        self.move_history.clone()
    }

    /// Player history as list of player IDs.
    #[getter]
    pub fn player_history(&self) -> Vec<Player> {
        self.player_history.clone()
    }

    fn __repr__(&self) -> String {
        if self.board.is_empty() {
            return "HexGame(empty)".to_string();
        }
        let min_q = self.board.keys().map(|c| c.0).min().unwrap();
        let max_q = self.board.keys().map(|c| c.0).max().unwrap();
        let min_r = self.board.keys().map(|c| c.1).min().unwrap();
        let max_r = self.board.keys().map(|c| c.1).max().unwrap();
        format!(
            "HexGame(moves={}, winner={:?}, q=[{},{}] r=[{},{}])",
            self.move_history.len(),
            self.winner_inner,
            min_q, max_q, min_r, max_r
        )
    }
}

// Rust-facing API (no PyO3 overhead).
impl HexGame {
    /// Place current player's piece. Rust-internal, no Python overhead.
    pub fn make_move(&mut self, q: i16, r: i16) -> bool {
        let coord = (q, r);
        if self.winner_inner != NO_PLAYER || self.board.contains_key(&coord) {
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
        let prev_winner = self.winner_inner;
        let prev_player = self.current_player;
        let prev_placements = self.placements_in_turn;

        if won {
            self.winner_inner = self.current_player;
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
            coord,
            removed_candidates: removed,
            added_candidates: added,
            prev_placements,
            prev_winner,
            prev_player,
        });
        true
    }

    /// Undo the last placement. Rust-internal.
    pub fn unmake_move(&mut self) {
        let Some(entry) = self.undo_stack.pop() else {
            return;
        };
        self.board.remove(&entry.coord);
        self.move_history.pop();
        self.player_history.pop();
        for &c in &entry.added_candidates {
            self.candidates.remove(&c);
        }
        for &c in &entry.removed_candidates {
            self.candidates.insert(c);
        }
        self.winner_inner = entry.prev_winner;
        self.current_player = entry.prev_player;
        self.placements_in_turn = entry.prev_placements;
    }
}
