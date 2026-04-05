"""
Parity tests: verify Rust HexGame produces identical results to Python HexGame.

Run after `maturin develop --release`:
    python -m pytest tests/test_parity.py -v
"""
import random
import sys
import os

# Add parent project to path so we can import the Python game
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from game import HexGame as PyGame
from hexgo import HexGame as RustGame


def test_initial_state():
    rg, pg = RustGame(), PyGame()
    assert rg.current_player == pg.current_player
    assert rg.placements_in_turn == pg.placements_in_turn
    assert rg.winner is None
    assert pg.winner is None


def test_single_move():
    rg, pg = RustGame(), PyGame()
    assert rg.make(0, 0) == pg.make(0, 0)
    assert rg.current_player == pg.current_player
    assert rg.placements_in_turn == pg.placements_in_turn


def test_first_turn_rule():
    """First player places 1 stone, then switches."""
    rg, pg = RustGame(), PyGame()
    rg.make(0, 0)
    pg.make(0, 0)
    # After 1 stone, should switch to P2
    assert rg.current_player == 2
    assert pg.current_player == 2


def test_122_rule():
    """After first move, each player places 2 stones."""
    rg, pg = RustGame(), PyGame()
    # P1 places 1
    rg.make(0, 0); pg.make(0, 0)
    assert rg.current_player == 2
    # P2 places 2
    rg.make(1, 0); pg.make(1, 0)
    assert rg.current_player == 2  # still P2's turn
    rg.make(0, 1); pg.make(0, 1)
    assert rg.current_player == 1  # now P1
    assert pg.current_player == 1


def test_unmake():
    rg, pg = RustGame(), PyGame()
    rg.make(0, 0); pg.make(0, 0)
    rg.make(1, 0); pg.make(1, 0)
    rg.unmake(); pg.unmake()
    assert rg.current_player == pg.current_player
    assert rg.placements_in_turn == pg.placements_in_turn
    assert rg.winner == pg.winner
    rg.unmake(); pg.unmake()
    assert rg.current_player == pg.current_player


def test_win_detection():
    """Build a 6-in-a-row and verify both detect the same winner."""
    rg, pg = RustGame(), PyGame()
    # P1: (0,0)
    rg.make(0, 0); pg.make(0, 0)
    # P2: (10,10), (10,11)
    rg.make(10, 10); pg.make(10, 10)
    rg.make(10, 11); pg.make(10, 11)
    # P1: (1,0), (2,0)
    rg.make(1, 0); pg.make(1, 0)
    rg.make(2, 0); pg.make(2, 0)
    # P2: (10,12), (10,13)
    rg.make(10, 12); pg.make(10, 12)
    rg.make(10, 13); pg.make(10, 13)
    # P1: (3,0), (4,0)
    rg.make(3, 0); pg.make(3, 0)
    rg.make(4, 0); pg.make(4, 0)
    # P2: (10,14), (10,15) — P2 completes 6-in-a-row first!
    rg.make(10, 14); pg.make(10, 14)
    rg.make(10, 15); pg.make(10, 15)
    # P2 wins with (10,10)-(10,15) along r-axis
    assert rg.winner == pg.winner == 2


def test_legal_moves_parity():
    """Legal moves should be the same set (order may differ)."""
    rg, pg = RustGame(), PyGame()
    rg.make(0, 0); pg.make(0, 0)
    rg.make(1, 0); pg.make(1, 0)
    rg.make(0, 1); pg.make(0, 1)
    assert sorted(rg.legal_moves()) == sorted(pg.legal_moves())


def test_random_game_parity():
    """Play 500 random games, verify Rust and Python produce identical states."""
    for seed in range(500):
        random.seed(seed)
        rg, pg = RustGame(), PyGame()
        for _ in range(80):
            moves_r = sorted(rg.legal_moves())
            moves_p = sorted(pg.legal_moves())
            assert moves_r == moves_p, f"seed={seed}: legal moves differ"
            if not moves_r:
                break
            m = random.choice(moves_r)
            r1 = rg.make(m[0], m[1])
            r2 = pg.make(m[0], m[1])
            assert r1 == r2, f"seed={seed}: make() return differs"
            assert rg.current_player == pg.current_player, f"seed={seed}: player differs"
            assert rg.placements_in_turn == pg.placements_in_turn, f"seed={seed}: placements differ"
            assert rg.winner == pg.winner, f"seed={seed}: winner differs"
            if rg.winner is not None:
                break

        # Test unmake parity
        for _ in range(min(10, len(rg.move_history))):
            rg.unmake()
            pg.unmake()
            assert rg.current_player == pg.current_player
            assert rg.winner == pg.winner


def test_zoi_moves_parity():
    """ZOI moves should return the same set."""
    random.seed(42)
    rg, pg = RustGame(), PyGame()
    for _ in range(30):
        moves = sorted(rg.legal_moves())
        if not moves:
            break
        m = random.choice(moves)
        rg.make(m[0], m[1])
        pg.make(m[0], m[1])
        if rg.winner is not None:
            break

    zoi_r = sorted(rg.zoi_moves(margin=6, lookback=16))
    zoi_p = sorted(pg.zoi_moves(margin=6, lookback=16))
    assert zoi_r == zoi_p


def test_candidates_parity():
    """Candidates should match after every move."""
    random.seed(99)
    rg, pg = RustGame(), PyGame()
    for _ in range(40):
        # Candidates are a set — compare sorted
        assert sorted(rg.candidates) == sorted(pg.candidates)
        moves = sorted(rg.legal_moves())
        if not moves:
            break
        m = random.choice(moves)
        rg.make(m[0], m[1])
        pg.make(m[0], m[1])
        if rg.winner is not None:
            break


def test_clone():
    rg = RustGame()
    rg.make(0, 0)
    rg.make(1, 0)
    clone = rg.clone_game()
    assert clone.current_player == rg.current_player
    assert clone.winner == rg.winner
    assert sorted(clone.legal_moves()) == sorted(rg.legal_moves())
    # Mutating clone shouldn't affect original
    clone.make(0, 1)
    assert clone.current_player != rg.current_player


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
