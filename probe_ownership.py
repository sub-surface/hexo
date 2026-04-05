"""
probe_ownership.py — Test whether the network understands piece ownership.

Three tests:
1. SWAP TEST: Evaluate a position, then swap P1/P2 channels + flip to-move.
   If the net understands ownership, the value should flip sign.

2. POLICY TEST: Build a position where P1 has a 5-chain (one move from winning)
   and P2 has a 2-chain. If it's P1's turn, the net should place the winning move.
   If it's P2's turn, the net should block P1's win, not extend P2's chain.

3. OWNERSHIP HEAD: Check if aux_own predictions align with actual board state.

Usage: python probe_ownership.py [--model net_gen0050.pt]
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from game import HexGame
from net import HexNet, DEVICE, encode_board, move_to_grid, BOARD_SIZE

CHECKPOINT_DIR = Path("checkpoints")


def load_net(model_name=None):
    path = CHECKPOINT_DIR / (model_name or "net_latest.pt")
    if not path.exists():
        print(f"Not found: {path}")
        return None
    net = HexNet().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    print(f"Loaded {path.name}")
    return net


def evaluate_position(net, game):
    """Return (value, top_5_moves_with_logits)."""
    board_arr, (oq, or_) = encode_board(game)
    board_t = torch.tensor(board_arr, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        f = net.trunk(board_t)
        value = net.value(f).item()
        logit_map = net.policy_logits(f).squeeze(0).cpu().numpy()
        own_map = net.ownership(f).squeeze(0).cpu().numpy()

    legal = game.legal_moves()
    move_logits = []
    for m in legal:
        idx = move_to_grid(m[0], m[1], oq, or_)
        if idx:
            move_logits.append((m, logit_map[idx[0], idx[1]]))

    move_logits.sort(key=lambda x: x[1], reverse=True)
    return value, move_logits[:5], own_map


def test_swap(net):
    """Swap P1/P2 channels and check if value flips."""
    print("\n" + "="*60)
    print("  TEST 1: SWAP — does value flip when players swap?")
    print("="*60)

    game = HexGame()
    # Build an asymmetric position: P1 has a strong chain, P2 doesn't
    game.make(0, 0)                    # P1
    game.make(5, 5); game.make(5, 6)   # P2 far away
    game.make(1, 0); game.make(2, 0)   # P1 builds q-axis chain
    game.make(6, 5); game.make(6, 6)   # P2
    game.make(3, 0); game.make(4, 0)   # P1: 5-in-a-row!

    print(f"  Position: P1 has 5-chain (0,0)->(4,0), P2 has scattered pieces")
    print(f"  Current player: P{game.current_player}")

    # Normal evaluation
    board_arr, (oq, or_) = encode_board(game)
    board_t = torch.tensor(board_arr, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        f = net.trunk(board_t)
        v_normal = net.value(f).item()

    # Swap channels 0↔1, flip channel 2, swap channels 11-13↔14-16
    swapped = board_arr.copy()
    swapped[0], swapped[1] = board_arr[1].copy(), board_arr[0].copy()
    swapped[2] = 1.0 - board_arr[2]  # flip to-move
    # Swap history channels 3-6 ↔ 7-10
    swapped[3:7], swapped[7:11] = board_arr[7:11].copy(), board_arr[3:7].copy()
    # Swap axis-chain channels 11-13 ↔ 14-16
    swapped[11:14], swapped[14:17] = board_arr[14:17].copy(), board_arr[11:14].copy()

    swapped_t = torch.tensor(swapped, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        f2 = net.trunk(swapped_t)
        v_swapped = net.value(f2).item()

    print(f"  Normal value:  {v_normal:+.4f}")
    print(f"  Swapped value: {v_swapped:+.4f}")
    print(f"  Sum (should be ~0 if ownership understood): {v_normal + v_swapped:+.4f}")

    if abs(v_normal + v_swapped) < 0.3:
        print(f"  OK PASS — value flips on swap (understands ownership)")
    elif v_normal * v_swapped < 0:
        print(f"  ~ PARTIAL — signs differ but magnitude unequal")
    else:
        print(f"  XX FAIL — value doesn't flip (may not distinguish players)")


def test_winning_move(net):
    """Does the net play the winning move / block the opponent's win?"""
    print("\n" + "="*60)
    print("  TEST 2: WINNING MOVE — does it complete/block 6-in-a-row?")
    print("="*60)

    # P1 has 5-in-a-row, P1 to move — should play (5,0) to win
    game = HexGame()
    game.make(0, 0)                    # P1
    game.make(10, 10); game.make(10, 11)
    game.make(1, 0); game.make(2, 0)
    game.make(11, 10); game.make(11, 11)
    game.make(3, 0); game.make(4, 0)   # P1: 5-in-a-row at (0-4, 0)
    game.make(12, 10); game.make(12, 11)
    # P1's turn, needs (5,0) or (-1,0) to win

    value, top_moves, _ = evaluate_position(net, game)
    winning = {(5, 0), (-1, 0)}
    top_move = top_moves[0][0] if top_moves else None

    print(f"  P1 has 5-chain, P1 to move")
    print(f"  Value: {value:+.4f}")
    print(f"  Top move: {top_move}  (winning: {winning})")
    if top_move in winning:
        print(f"  OK PASS — plays winning move")
    else:
        print(f"  XX FAIL — top moves: {[(m, f'{l:.2f}') for m, l in top_moves[:3]]}")

    # Now: P1 has 5-in-a-row, P2 to move — should BLOCK at (5,0) or (-1,0)
    game2 = HexGame()
    game2.make(0, 0)
    game2.make(10, 10); game2.make(10, 11)
    game2.make(1, 0); game2.make(2, 0)
    game2.make(11, 10); game2.make(11, 11)
    game2.make(3, 0); game2.make(4, 0)
    # P2's turn (2 placements), must block both ends
    value2, top_moves2, _ = evaluate_position(net, game2)

    print(f"\n  P1 has 5-chain, P2 to move (must block)")
    print(f"  Value: {value2:+.4f}")
    top_move2 = top_moves2[0][0] if top_moves2 else None
    print(f"  Top move: {top_move2}  (blocking: {winning})")
    if top_move2 in winning:
        print(f"  OK PASS — blocks the winning threat")
    else:
        print(f"  XX FAIL — top moves: {[(m, f'{l:.2f}') for m, l in top_moves2[:3]]}")


def test_ownership_head(net):
    """Does the ownership head correctly predict piece ownership?"""
    print("\n" + "="*60)
    print("  TEST 3: OWNERSHIP HEAD — predicts board ownership?")
    print("="*60)

    game = HexGame()
    game.make(0, 0)
    game.make(3, 3); game.make(3, 4)
    game.make(1, 0); game.make(2, 0)

    _, _, own_map = evaluate_position(net, game)

    board_arr, (oq, or_) = encode_board(game)
    half = BOARD_SIZE // 2

    print(f"  Ownership predictions at piece locations:")
    correct = 0
    total = 0
    for (q, r), p in game.board.items():
        idx = move_to_grid(q, r, oq, or_)
        if idx:
            own_val = own_map[idx[0], idx[1]]
            expected = 1.0 if p == 1 else -1.0
            is_correct = (own_val > 0) == (expected > 0)
            correct += is_correct
            total += 1
            mark = "OK" if is_correct else "XX"
            print(f"    ({q},{r}) P{p}: own={own_val:+.3f} expected={expected:+.0f} {mark}")

    if total > 0:
        acc = correct / total
        print(f"  Accuracy: {correct}/{total} ({acc:.0%})")
        if acc >= 0.8:
            print(f"  OK PASS — ownership head understands piece identity")
        else:
            print(f"  XX FAIL — ownership head confused")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    net = load_net(args.model)
    if net is None:
        exit(1)

    test_swap(net)
    test_winning_move(net)
    test_ownership_head(net)
    print()
