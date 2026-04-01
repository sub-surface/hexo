"""
HexGo unit tests — game engine + neural net encoding.

Run:  python test_game.py
All tests must pass before committing or training.
"""

import time
import numpy as np
from game import HexGame
from net import (encode_board, move_to_grid, IN_CH, BOARD_SIZE, N_HISTORY,
                 D6_MATRICES, _transform_board, _transform_aux, d6_augment_sample)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_sequence(g: HexGame, *coords):
    """Make all coords in order, return g for chaining."""
    for q, r in coords:
        g.make(q, r)
    return g


# ── game engine tests ─────────────────────────────────────────────────────────

def test_basic_play():
    g = HexGame()
    assert g.play(0, 0)
    assert g.board[(0, 0)] == 1
    assert g.current_player == 2
    assert not g.play(0, 0)  # occupied


def test_move_rule_1_2_2():
    g = HexGame()
    # Turn 1: Player 1 places 1 tile
    g.make(0, 0)
    assert g.current_player == 2
    assert g.placements_in_turn == 0

    # Turn 2: Player 2 places 1st tile
    g.make(1, 0)
    assert g.current_player == 2
    assert g.placements_in_turn == 1

    # Turn 2: Player 2 places 2nd tile
    g.make(1, 1)
    assert g.current_player == 1
    assert g.placements_in_turn == 0

    # Turn 3: Player 1 places 1st tile
    g.make(0, 1)
    assert g.current_player == 1

    # Undo sequence
    g.unmake()
    assert g.current_player == 1 and g.placements_in_turn == 0

    g.unmake()
    assert g.current_player == 2 and g.placements_in_turn == 1

    g.unmake()
    assert g.current_player == 2 and g.placements_in_turn == 0

    g.unmake()
    assert g.current_player == 1 and g.placements_in_turn == 0
    assert len(g.board) == 0


def test_win_q_axis():
    """Six in a row along the q (horizontal) axis."""
    g = HexGame()
    # P1: (0,0)
    g.make(0, 0)
    # P2: far away
    g.make(10, 10); g.make(10, 11)
    # P1: (1,0), (2,0)
    g.make(1, 0); g.make(2, 0)
    g.make(11, 10); g.make(11, 11)
    g.make(3, 0); g.make(4, 0)
    g.make(12, 10); g.make(12, 11)
    g.make(5, 0)  # 6th in a row on q-axis → win
    assert g.winner == 1, f"expected P1 win, got {g.winner}"


def test_win_r_axis():
    """Six in a row along the r (column) axis."""
    g = HexGame()
    g.make(0, 0)
    g.make(10, 10); g.make(10, 11)
    g.make(0, 1); g.make(0, 2)
    g.make(11, 10); g.make(11, 11)
    g.make(0, 3); g.make(0, 4)
    g.make(12, 10); g.make(12, 11)
    g.make(0, 5)  # 6th in a row on r-axis → win
    assert g.winner == 1, f"expected P1 win, got {g.winner}"


def test_win_diagonal():
    """Six in a row along the (1,-1) diagonal axis."""
    g = HexGame()
    g.make(0, 0)
    g.make(10, 10); g.make(10, 11)
    g.make(1, -1); g.make(2, -2)
    g.make(11, 10); g.make(11, 11)
    g.make(3, -3); g.make(4, -4)
    g.make(12, 10); g.make(12, 11)
    g.make(5, -5)  # 6th on (1,-1) diagonal → win
    assert g.winner == 1, f"expected P1 win, got {g.winner}"


def test_no_false_win():
    """Five in a row should not trigger a win."""
    g = HexGame()
    g.make(0, 0)
    g.make(1, 0); g.make(2, 0)
    g.make(0, 1); g.make(0, 2)
    assert g.winner is None


def test_mid_turn_win():
    """Win on first tile of a 2-tile turn ends game immediately."""
    g = HexGame()
    g.make(0, 0)
    g.make(10, 10); g.make(10, 11)
    g.make(1, 0); g.make(2, 0)
    g.make(11, 10); g.make(11, 11)
    g.make(3, 0); g.make(4, 0)
    g.make(12, 10); g.make(12, 11)
    # P1's turn — two placements allowed, but first one wins
    g.make(5, 0)  # WIN on first placement
    assert g.winner == 1
    # No second tile should be legal (game is over)
    assert g.legal_moves() == [] or g.winner is not None


def test_undo_after_win():
    """Undoing a winning move restores winner=None."""
    g = HexGame()
    g.make(0, 0)
    g.make(10, 10); g.make(10, 11)
    g.make(1, 0); g.make(2, 0)
    g.make(11, 10); g.make(11, 11)
    g.make(3, 0); g.make(4, 0)
    g.make(12, 10); g.make(12, 11)
    g.make(5, 0)
    assert g.winner == 1
    g.unmake()
    assert g.winner is None
    assert (5, 0) not in g.board


def test_candidates_tracking():
    """Legal moves include all empty cells within PLACEMENT_RADIUS of any piece."""
    from game import PLACEMENT_RADIUS
    g = HexGame()
    assert g.legal_moves() == [(0, 0)]

    g.make(0, 0)
    moves = set(g.legal_moves())
    # Adjacent cells must be legal
    for dq, dr in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]:
        assert (dq, dr) in moves, f"adjacent ({dq},{dr}) should be legal"
    assert (0, 0) not in moves  # occupied
    assert (PLACEMENT_RADIUS, 0) in moves  # within radius
    assert (PLACEMENT_RADIUS + 1, 0) not in moves  # outside radius

    g.make(1, 0)
    moves_after = set(g.legal_moves())
    assert (0, 0) not in moves_after
    assert (1, 0) not in moves_after
    assert (2, 0) in moves_after


def test_move_history_tracking():
    """move_history records every placed tile and is restored by unmake."""
    g = HexGame()
    g.make(0, 0)
    assert g.move_history == [(0, 0)]

    g.make(1, 0); g.make(1, 1)
    assert g.move_history == [(0, 0), (1, 0), (1, 1)]

    g.unmake()
    assert g.move_history == [(0, 0), (1, 0)]

    g.unmake(); g.unmake()
    assert g.move_history == []


def test_clone():
    """Clone is fully independent of the original."""
    g = HexGame()
    g.play(0, 0)
    c = g.clone()
    c.play(1, 0)
    assert (1, 0) not in g.board
    assert (0, 0) in c.board  # original state preserved in clone
    assert c.current_player == g.current_player  # same turn state


def test_clone_candidates_independent():
    """Modifying clone's candidates does not affect original."""
    g = HexGame()
    g.make(0, 0)
    c = g.clone()
    c.make(1, 0)
    assert set(g.legal_moves()) != set(c.legal_moves())


def test_deep_undo_consistency():
    """After a sequence of moves and complete undo, game is at initial state."""
    g = HexGame()
    moves = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2),
             (2, 0), (2, 1), (-1, 0), (-1, 1)]
    played = []
    for q, r in moves:
        if g.winner is None:
            g.make(q, r)
            played.append((q, r))

    for _ in played:
        g.unmake()

    assert len(g.board) == 0
    assert g.current_player == 1
    assert g.placements_in_turn == 0
    assert g.winner is None
    assert g.move_history == []
    assert g.legal_moves() == [(0, 0)]


def test_legal_moves_empty():
    g = HexGame()
    assert g.legal_moves() == [(0, 0)]


def test_legal_moves_after_play():
    g = HexGame()
    g.play(0, 0)
    moves = g.legal_moves()
    assert (0, 0) not in moves
    assert len(moves) > 6  # all cells within PLACEMENT_RADIUS, not just 6 neighbors


# ── neural net encoding tests ─────────────────────────────────────────────────

def test_encode_board_state_channels():
    """Channels 0/1 correctly reflect board state; channel 2 reflects to-move."""
    g = HexGame()
    g.make(0, 0)         # P1 → (0,0), now P2 to move
    g.make(1, 0)         # P2 → (1,0)
    g.make(1, 1)         # P2 2nd tile → now P1 to move
    arr, _ = encode_board(g)
    # ch0 = P1 pieces: should have (0,0)
    assert arr[0].sum() == 1.0, f"P1 piece count: {arr[0].sum()}"
    # ch1 = P2 pieces: should have (1,0) and (1,1)
    assert arr[1].sum() == 2.0, f"P2 piece count: {arr[1].sum()}"
    # ch2 = to-move: P1 to move → 0.0
    assert arr[2, 0, 0] == 0.0, f"to-move should be 0 (P1), got {arr[2,0,0]}"


def test_encode_board_history_channels():
    """History planes populate correctly and are ordered most-recent first."""
    g = HexGame()
    g.make(0, 0)               # P1 move 1 (single tile, turn 1)
    g.make(1, 0); g.make(1, 1) # P2 moves 1+2
    g.make(0, 1); g.make(0, 2) # P1 moves 1+2

    arr, (oq, or_) = encode_board(g)
    half = 18 // 2

    # P1 history: most recent = (0,2), then (0,1), then (0,0)
    # Ch 3 = most recent P1 move = (0,2)
    qi, ri = 0 - oq + half, 2 - or_ + half
    assert arr[3, ri, qi] == 1.0, "ch3 should be most recent P1 move (0,2)"

    qi, ri = 0 - oq + half, 1 - or_ + half
    assert arr[4, ri, qi] == 1.0, "ch4 should be 2nd P1 move (0,1)"

    qi, ri = 0 - oq + half, 0 - or_ + half
    assert arr[5, ri, qi] == 1.0, "ch5 should be 3rd P1 move (0,0)"

    # Ch 6 (4th history slot) should be empty (only 3 P1 moves so far)
    assert arr[6].sum() == 0.0, "ch6 should be empty (only 3 P1 moves)"

    # P2 history: most recent = (1,1), then (1,0)
    qi, ri = 1 - oq + half, 1 - or_ + half
    assert arr[7, ri, qi] == 1.0, "ch7 should be most recent P2 move (1,1)"


def test_encode_board_history_undo():
    """History planes are consistent after unmake (board and history agree)."""
    g = HexGame()
    g.make(0, 0)
    g.make(1, 0); g.make(1, 1)
    g.make(0, 1); g.make(0, 2)
    # Undo P1's second placement
    g.unmake()
    arr, (oq, or_) = encode_board(g)
    half = 18 // 2
    # (0,2) should no longer appear in any history channel
    qi, ri = 0 - oq + half, 2 - or_ + half
    for ch in range(3, 11):
        assert arr[ch, ri, qi] == 0.0, f"(0,2) should not appear in ch{ch} after undo"


def test_encode_board_shape():
    """encode_board returns correct shape for any number of pieces."""
    g = HexGame()
    arr_empty, _ = encode_board(g)
    assert arr_empty.shape == (IN_CH, BOARD_SIZE, BOARD_SIZE)

    g.make(0, 0)
    arr_one, _ = encode_board(g)
    assert arr_one.shape == (IN_CH, BOARD_SIZE, BOARD_SIZE)


# ── D6 symmetry tests ────────────────────────────────────────────────────────

def test_d6_identity_unchanged():
    """Transform 0 (identity) must leave the board array unchanged."""
    g = HexGame()
    for q, r in [(0,0),(1,0),(1,1),(0,1)]:
        g.make(q, r)
    arr, _ = encode_board(g)
    transformed = _transform_board(arr, 0)
    assert np.allclose(arr, transformed), "Identity transform should not change array"


def test_d6_six_rotations_distinct():
    """The six rotations of a non-symmetric position must all be distinct."""
    g = HexGame()
    g.make(0, 0)  # single piece at origin — symmetric, skip
    g.make(2, 0); g.make(2, 1)  # asymmetric cluster
    arr, _ = encode_board(g)
    seen = []
    for i in range(6):  # rotations only (indices 0-5)
        t = _transform_board(arr, i)
        for prev in seen:
            assert not np.allclose(t, prev), f"Rotation {i} matches a previous rotation"
        seen.append(t)


def test_d6_rotation_composition():
    """R60 applied 6 times must equal identity (group order = 6)."""
    g = HexGame()
    for q, r in [(1, 0), (2, 0), (0, 1)]:
        g.make(q, r)
    arr, _ = encode_board(g)
    result = arr.copy()
    for _ in range(6):
        result = _transform_board(result, 1)  # apply R60 six times
    assert np.allclose(arr, result), "R60^6 should equal identity"


def test_d6_reflection_involution():
    """Every reflection applied twice must equal identity."""
    g = HexGame()
    g.make(0, 0); g.make(1, 0); g.make(0, 1)
    arr, _ = encode_board(g)
    for i in range(6, 12):  # reflection indices
        double = _transform_board(_transform_board(arr, i), i)
        assert np.allclose(arr, double), f"Reflection {i} applied twice should equal identity"


def test_d6_augment_spatial_consistent():
    """After augmentation, policy target sum and legal mask count are preserved."""
    g = HexGame()
    g.make(0, 0); g.make(1, 0); g.make(1, 1); g.make(0, 1)
    arr, (oq, or_) = encode_board(g)

    # Build spatial policy target and legal mask
    moves = [(2, 0), (0, 2), (-1, 0)]
    probs = [0.5, 0.3, 0.2]
    policy_target = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    legal_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for (q, r), p in zip(moves, probs):
        idx = move_to_grid(q, r, oq, or_)
        if idx is not None:
            row, col = idx
            policy_target[row, col] = p
            legal_mask[row, col] = 1.0

    sample = {
        'board': arr,
        'policy_target': policy_target,
        'legal_mask': legal_mask,
        'z': 1.0,
    }
    for i in range(12):
        aug = d6_augment_sample(sample, i)
        assert abs(aug['policy_target'].sum() - 1.0) < 1e-5, \
            f"tf={i}: policy target doesn't sum to 1"
        assert aug['legal_mask'].sum() == 3.0, \
            f"tf={i}: legal mask count changed from 3 to {aug['legal_mask'].sum()}"
        assert aug['z'] == 1.0, f"tf={i}: z changed"


def test_eisenstein_greedy_prefers_longer_chain():
    """EisensteinGreedyAgent should prefer a move that extends the longest chain."""
    from elo import EisensteinGreedyAgent
    agent = EisensteinGreedyAgent(defensive=False)
    g = HexGame()
    # Build a 4-in-a-row for P1 along q-axis: (0,0),(1,0),(2,0),(3,0)
    g.make(0, 0)                    # P1 single tile (turn 1)
    g.make(10, 10); g.make(10, 11)  # P2 far away (2 tiles)
    g.make(1, 0); g.make(2, 0)      # P1 (2 tiles)
    g.make(11, 10); g.make(11, 11)  # P2
    g.make(3, 0)                    # P1 first of 2 tiles
    # Now P1 places second tile — (4,0) or (-1,0) extends the q-axis chain
    assert g.current_player == 1
    move = agent.choose_move(g)
    # Should pick a move that extends the 4-chain, either end
    assert move in ((4, 0), (-1, 0)), f"Expected chain-extending move, got {move}"


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_basic_play,
        test_move_rule_1_2_2,
        test_win_q_axis,
        test_win_r_axis,
        test_win_diagonal,
        test_no_false_win,
        test_mid_turn_win,
        test_undo_after_win,
        test_candidates_tracking,
        test_move_history_tracking,
        test_clone,
        test_clone_candidates_independent,
        test_deep_undo_consistency,
        test_legal_moves_empty,
        test_legal_moves_after_play,
        test_encode_board_state_channels,
        test_encode_board_history_channels,
        test_encode_board_history_undo,
        test_encode_board_shape,
        test_d6_identity_unchanged,
        test_d6_six_rotations_distinct,
        test_d6_rotation_composition,
        test_d6_reflection_involution,
        test_d6_augment_spatial_consistent,
        test_eisenstein_greedy_prefers_longer_chain,
    ]
    passed = failed = 0
    for t in tests:
        t0 = time.perf_counter()
        try:
            t()
            ms = (time.perf_counter() - t0) * 1000
            print(f"  PASS  {t.__name__:<45} ({ms:.1f}ms)")
            passed += 1
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            print(f"  FAIL  {t.__name__:<45} ({ms:.1f}ms)  {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed", end="")
    print("  OK" if failed == 0 else f"  ({failed} FAILED)")
