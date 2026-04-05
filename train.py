"""
Self-play training loop — batched lockstep MCTS for maximum GPU utilization.

Pipeline:
  1. Batched self-play: N games run in lockstep, all leaf evals batched into
     single GPU calls. No threads, no GIL contention.
  2. Training: vectorized spatial policy loss on replay buffer
  3. Checkpoint + metrics every generation

Run: python train.py [--gens N] [--sims N] [--games N]
"""

import argparse
import json
import logging
import math
import os
import pickle
import random
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

try:
    from hexgo import HexGame  # Rust engine (8-25x faster)
    from hexgo import batched_self_play as rust_batched_self_play
    _HAS_RUST_BATCHED = True
except ImportError:
    from game import HexGame
    _HAS_RUST_BATCHED = False
from mcts import Node, _backprop
from net import (HexNet, encode_board, move_to_grid, top_k_from_logit_map,
                 DEVICE, BOARD_SIZE, param_count, d6_augment_sample,
                 make_aux_labels, init_weights_ca)
from elo import ELO, NetAgent, EisensteinGreedyAgent, run_match
from config import CFG

_CUDA = "cuda" in str(DEVICE)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("train.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("train")

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
REPLAY_DIR = Path("replays")
REPLAY_DIR.mkdir(exist_ok=True)

BUFFER_CAP   = 100_000  # smaller buffer flushes bad data faster
BATCH_SIZE   = 512      # large batch to saturate GPU during training
LR           = 2e-4     # conservative LR for stable self-play training
WEIGHT_DECAY = 3e-5     # lighter regularization for small network

# Batched self-play settings
TOP_K        = 24      # wider branching for 400-sim deep search
SIMS_MIN     = 16      # min sims early in training
SIMS_RAMP    = 20      # generations to ramp from SIMS_MIN to target
MAX_MOVES_MIN = 30     # max moves per game early in training
MAX_MOVES_MAX = 120    # cap to prevent scattered-play gens from dragging
MAX_MOVES_RAMP = 20    # generations to ramp
DECISIVE_DIR  = Path("replays/decisive")
DECISIVE_DIR.mkdir(parents=True, exist_ok=True)
TD_LAMBDA     = 0.8       # temporal difference lambda for value targets
RANDOM_OPENING = 0        # disabled for now — Rust batched self-play handles opening diversity via Dirichlet
RANDOM_OPENING_FRAC = 0.5 # fraction of games that get random openings
ZOI_MARGIN_MIN  = 4       # tight ZOI forces compact play (4 = can always block a 6-chain)
ZOI_MARGIN_MAX  = 5       # full ZOI margin once network learns tactics
ZOI_MARGIN_RAMP = 30      # generations to ramp from MIN to MAX
ZOI_LOOKBACK    = 16      # recent moves defining ZOI focus


def _curriculum_zoi(gen):
    """Linearly ramp ZOI margin from MIN to MAX over RAMP generations."""
    if gen >= ZOI_MARGIN_RAMP:
        return ZOI_MARGIN_MAX
    return ZOI_MARGIN_MIN + (ZOI_MARGIN_MAX - ZOI_MARGIN_MIN) * gen // ZOI_MARGIN_RAMP


def _curriculum_sims(gen, target):
    """Linearly ramp sims from SIMS_MIN to target over SIMS_RAMP generations."""
    if gen >= SIMS_RAMP:
        return target
    return max(SIMS_MIN, int(SIMS_MIN + (target - SIMS_MIN) * gen / SIMS_RAMP))


def _curriculum_max_moves(gen):
    """Linearly ramp max_moves from MIN to MAX over MAX_MOVES_RAMP generations."""
    if gen >= MAX_MOVES_RAMP:
        return MAX_MOVES_MAX
    return int(MAX_MOVES_MIN + (MAX_MOVES_MAX - MAX_MOVES_MIN) * gen / MAX_MOVES_RAMP)


# -- Game recording -----------------------------------------------------------

def save_replay(moves, winner, gen, label, directory=None):
    directory = directory or REPLAY_DIR
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"game_{label}_gen{gen:04d}_{ts}.json"
    data = {
        "gen": gen, "label": label, "winner": winner,
        "timestamp": ts, "moves": [[q, r] for q, r in moves],
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def save_decisive_games(results, gen):
    """Save all decisive (non-draw) games for corpus building."""
    saved = 0
    for data, winner, moves in results:
        if winner is not None and len(moves) >= 6:
            label = f"w{winner}_m{len(moves)}"
            save_replay(moves, winner, gen, label, directory=DECISIVE_DIR)
            saved += 1
    return saved


# -- Batched lockstep MCTS self-play ------------------------------------------

@torch.no_grad()
def batched_self_play(net, n_games, sims, max_moves, top_k, device):
    """
    Play n_games simultaneously with batched lockstep MCTS.
    All leaf evaluations across all games are batched into single GPU calls.
    No threads, no GIL contention — single-threaded with periodic GPU bursts.

    Returns list of (training_data, winner, move_list) per game.
    """
    net.eval()
    games = [HexGame() for _ in range(n_games)]
    all_positions = [[] for _ in range(n_games)]
    active = set(range(n_games))
    move_counts = [0] * n_games
    temp_horizon = CFG.get("TEMP_HORIZON", 40)
    dir_alpha = CFG.get("DIRICHLET_ALPHA", 0.09)
    cache_hits = 0
    cache_size = 0
    dir_eps = CFG.get("DIRICHLET_EPS", 0.25)
    zoi_margin = CFG.get("ZOI_MARGIN", 6)
    zoi_lookback = CFG.get("ZOI_LOOKBACK", 16)

    # --- Random opening phase: diverse starting positions ---
    # Apply random openings to a fraction of games to keep buffer mixed.
    # No training data recorded for these (policy targets would be noise).
    random_games = set(i for i in active if random.random() < RANDOM_OPENING_FRAC)
    for _step in range(RANDOM_OPENING):
        for i in list(active & random_games):
            moves = games[i].legal_moves()
            if not moves or games[i].winner is not None:
                active.discard(i)
                continue
            games[i].make(*random.choice(moves))
            move_counts[i] += 1
            if games[i].winner is not None or move_counts[i] >= max_moves:
                active.discard(i)

    while active:
        active_list = sorted(active)
        n = len(active_list)

        # --- Root initialization: one batched GPU call for all active games ---
        board_data = [encode_board(games[i]) for i in active_list]
        boards_np = np.stack([d[0] for d in board_data])
        origins = [d[1] for d in board_data]
        root_board_data = {active_list[j]: board_data[j] for j in range(len(active_list))}

        boards_t = torch.tensor(boards_np, device=device)
        with torch.amp.autocast(device_type="cuda" if _CUDA else "cpu"):
            rf = net.trunk(boards_t)
            root_logits = net.policy_logits(rf).float().cpu().numpy()
        # Value head has FC layers that overflow float16 with CA init —
        # compute in float32 outside autocast.
        root_values = net.value(rf.float()).cpu().numpy().clip(-1.0, 1.0)

        root_value_for_game = {active_list[j]: float(root_values[j])
                               for j in range(len(active_list))}

        # Create root nodes with top-K pruning + Dirichlet noise
        roots = {}
        for local_idx, i in enumerate(active_list):
            oq, or_ = origins[local_idx]
            move_logits = top_k_from_logit_map(
                root_logits[local_idx], games[i].board, oq, or_, k=top_k)

            if not move_logits:
                active.discard(i)
                continue

            moves_k = [m for m, _ in move_logits]
            logits_k = np.array([l for _, l in move_logits], dtype=np.float32)
            logits_k -= logits_k.max()
            priors = np.exp(logits_k); priors /= priors.sum()

            noise = np.random.dirichlet([dir_alpha] * len(moves_k))
            priors = (1 - dir_eps) * priors + dir_eps * noise

            root = Node(player=games[i].current_player)
            root.children = [
                Node(move=m, parent=root, prior=float(p), player=games[i].current_player)
                for m, p in zip(moves_k, priors)
            ]
            roots[i] = root

        active_list = [i for i in active_list if i in roots]
        if not active_list:
            break

        # --- Lockstep MCTS sims ---
        for _sim in range(sims):
            # Selection: traverse tree for each active game
            leaves = []  # (game_idx, node, depth)
            for i in active_list:
                node = roots[i]
                depth = 0
                while node.children and games[i].winner is None:
                    node = node.best_child()
                    games[i].make(*node.move)
                    depth += 1
                leaves.append((i, node, depth))

            # Identify leaves that need GPU evaluation
            needs_eval = [(i, node, depth) for i, node, depth, in leaves
                          if games[i].winner is None and not node.children]

            # Batch evaluate all non-terminal unexpanded leaves
            eval_values = {}
            if needs_eval:
                eval_data = [encode_board(games[i], fast=True) for i, _, _ in needs_eval]
                eval_np = np.stack([d[0] for d in eval_data])
                eval_origins = [d[1] for d in eval_data]

                eval_t = torch.tensor(eval_np, device=device)
                with torch.amp.autocast(device_type="cuda" if _CUDA else "cpu"):
                    ef = net.trunk(eval_t)
                    ep = net.policy_logits(ef).float().cpu().numpy()
                ev = net.value(ef.float()).cpu().numpy().clip(-1.0, 1.0)

                for j, (i, node, depth) in enumerate(needs_eval):
                    oq, or_ = eval_origins[j]
                    ml = top_k_from_logit_map(
                        ep[j], games[i].board, oq, or_, k=top_k)

                    if ml:
                        moves_e = [m for m, _ in ml]
                        logits_e = np.array([l for _, l in ml], dtype=np.float32)
                        logits_e -= logits_e.max()
                        priors_e = np.exp(logits_e); priors_e /= priors_e.sum()
                        node.children = [
                            Node(move=m, parent=node, prior=float(p),
                                 player=games[i].current_player)
                            for m, p in zip(moves_e, priors_e)
                        ]

                    v = float(ev[j])
                    if node.player != games[i].current_player:
                        v = -v
                    eval_values[id(node)] = v

            # Backprop + unmake for all games
            for i, node, depth in leaves:
                if games[i].winner is not None:
                    v = 1.0 if games[i].winner == node.player else -1.0
                elif id(node) in eval_values:
                    v = eval_values[id(node)]
                else:
                    v = 0.0  # revisited already-expanded node

                for _ in range(depth):
                    games[i].unmake()

                _backprop(node, v)

        # --- Pick moves and record training data ---
        for i in active_list:
            root = roots[i]
            if not root.children:
                active.discard(i)
                continue

            visits = np.array([c.visits for c in root.children], dtype=np.float32)
            child_moves = [c.move for c in root.children]

            temp = max(0.05, math.cos(math.pi / 2 * move_counts[i] / max(temp_horizon, 1)))
            if temp < 0.06:
                dist = np.zeros_like(visits); dist[visits.argmax()] = 1.0
            else:
                vt = visits ** (1.0 / temp)
                dist = vt / vt.sum()

            chosen = child_moves[np.random.choice(len(child_moves), p=dist)]

            # Record training data with spatial policy target
            board_arr, (oq, or_) = root_board_data[i]
            policy_target = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            legal_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            for m, d in zip(child_moves, dist):
                idx = move_to_grid(m[0], m[1], oq, or_)
                if idx:
                    policy_target[idx[0], idx[1]] = d
                    legal_mask[idx[0], idx[1]] = 1.0
            s = policy_target.sum()
            if s > 0:
                policy_target /= s

            all_positions[i].append({
                "board": board_arr,
                "policy_target": policy_target,
                "legal_mask": legal_mask,
                "player": games[i].current_player,
                "origin": (oq, or_),
                "value_est": root_value_for_game.get(i, 0.0),
            })

            games[i].make(*chosen)
            move_counts[i] += 1

            if games[i].winner is not None or move_counts[i] >= max_moves:
                active.discard(i)

    # Assign TD-lambda value targets + real aux labels
    td_gamma = CFG.get("TD_GAMMA", 0.99)
    results = []
    for i in range(n_games):
        winner = games[i].winner
        positions = all_positions[i]
        n_pos = len(positions)

        if n_pos == 0:
            results.append(([], winner, list(games[i].move_history)))
            continue

        # TD-lambda value targets (computed backwards from game outcome)
        targets = [0.0] * n_pos
        if winner is None:
            targets[-1] = 0.0
        else:
            targets[-1] = 1.0 if positions[-1]["player"] == winner else -1.0

        for t in range(n_pos - 2, -1, -1):
            # Sign flip when consecutive positions have different active players
            sign = 1.0 if positions[t]["player"] == positions[t + 1]["player"] else -1.0
            v_next = positions[t + 1]["value_est"]
            g_next = targets[t + 1]
            targets[t] = sign * td_gamma * ((1 - TD_LAMBDA) * v_next + TD_LAMBDA * g_next)

        # Generate real aux labels from final game state
        for t, pos in enumerate(positions):
            pos["z"] = max(-1.0, min(1.0, targets[t]))
            oq, or_ = pos["origin"]
            own_label, threat_label = make_aux_labels(games[i], winner, oq, or_)
            pos["own_label"] = own_label
            pos["threat_label"] = threat_label
            del pos["player"]
            del pos["origin"]
            del pos["value_est"]

        results.append((positions, winner, list(games[i].move_history)))

    return results


def _scalar_to_wdl(z: np.ndarray) -> np.ndarray:
    """Convert scalar value targets in [-1,1] to soft WDL distributions [N, 3].
    v >= 0: [v, 1-v, 0]  (win/draw blend)
    v <  0: [0, 1+v, -v] (draw/loss blend)
    """
    z = np.clip(z, -1.0, 1.0)
    wdl = np.zeros((len(z), 3), dtype=np.float32)
    pos = z >= 0
    wdl[pos, 0] = z[pos]         # P(win)
    wdl[pos, 1] = 1.0 - z[pos]   # P(draw)
    neg = ~pos
    wdl[neg, 1] = 1.0 + z[neg]   # P(draw)
    wdl[neg, 2] = -z[neg]        # P(loss)
    return wdl


# -- Training step (fully vectorized) -----------------------------------------

def train_batch(net, optimizer, scaler, buffer):
    if len(buffer) < BATCH_SIZE:
        return {}

    # Recency-weighted sampling
    rw = CFG.get("RECENCY_WEIGHT", 0.75)
    buf_list = list(buffer)
    n_recent = max(1, len(buf_list) // 2)
    recent_half = buf_list[-n_recent:]
    n_from_recent = int(BATCH_SIZE * rw)
    n_from_all = BATCH_SIZE - n_from_recent
    batch = (random.sample(recent_half, min(n_from_recent, len(recent_half))) +
             random.sample(buf_list, min(n_from_all, len(buf_list))))
    random.shuffle(batch)

    # D6 augmentation
    batch = [d6_augment_sample(item, random.randrange(12)) for item in batch]
    net.train()

    boards_np = np.stack([b["board"] for b in batch])
    boards = (torch.from_numpy(boards_np).pin_memory().to(DEVICE, non_blocking=True)
              if _CUDA else torch.tensor(boards_np, device=DEVICE))
    z_np = np.array([b["z"] for b in batch], dtype=np.float32)
    z_targets = (torch.from_numpy(z_np).pin_memory().to(DEVICE, non_blocking=True)
                 if _CUDA else torch.tensor(z_np, device=DEVICE))

    optimizer.zero_grad()

    f = net.trunk(boards)
    wdl_logits = net.value_wdl(f)                          # [B, 3]
    wdl_targets = torch.from_numpy(_scalar_to_wdl(z_np)).to(DEVICE)
    loss_v = -(wdl_targets * F.log_softmax(wdl_logits, dim=-1)).sum(dim=-1).mean()
    val = F.softmax(wdl_logits.detach(), dim=-1)[:, 0] - F.softmax(wdl_logits.detach(), dim=-1)[:, 2]

    # Spatial masked cross-entropy policy loss
    policy_targets = torch.tensor(
        np.stack([b["policy_target"] for b in batch]),
        dtype=torch.float32, device=DEVICE)
    legal_masks = torch.tensor(
        np.stack([b["legal_mask"] for b in batch]),
        dtype=torch.float32, device=DEVICE)

    logit_map = net.policy_logits(f)
    B_actual = logit_map.shape[0]
    logit_map = logit_map.masked_fill(legal_masks == 0, float('-inf'))
    logit_flat = logit_map.view(B_actual, -1)
    target_flat = policy_targets.view(B_actual, -1)

    has_legal = legal_masks.view(B_actual, -1).sum(dim=1) > 0
    n_p = has_legal.sum().item()

    if n_p > 0:
        log_preds = F.log_softmax(logit_flat[has_legal], dim=1)
        log_preds = log_preds.masked_fill(torch.isinf(log_preds), 0.0)
        per_sample = -(target_flat[has_legal] * log_preds).sum(dim=1)
        loss_p = per_sample.mean()
    else:
        loss_p = torch.tensor(0.0, device=DEVICE)

    # Entropy regularization
    loss_ent = torch.tensor(0.0, device=DEVICE)
    ent_reg = CFG.get("ENTROPY_REG", 0.01)
    if ent_reg > 0 and n_p > 0:
        preds = F.softmax(logit_flat[has_legal], dim=1)
        safe_log = F.log_softmax(logit_flat[has_legal], dim=1)
        safe_log = safe_log.masked_fill(torch.isinf(safe_log), 0.0)
        loss_ent = -(preds * safe_log).sum(dim=1).mean()
        loss_p = loss_p - ent_reg * loss_ent

    # Auxiliary losses
    loss_aux = torch.tensor(0.0, device=DEVICE)
    aux_w_own = CFG.get("AUX_LOSS_OWN", 0.1)
    aux_w_threat = CFG.get("AUX_LOSS_THREAT", 0.1)
    if aux_w_own > 0 or aux_w_threat > 0:
        aux_indices = [i for i, b in enumerate(batch) if "own_label" in b]
        if aux_indices:
            has_aux = [batch[i] for i in aux_indices]
            own_np = np.stack([b["own_label"] for b in has_aux])
            threat_np = np.stack([b["threat_label"] for b in has_aux])
            own_t = (torch.from_numpy(own_np).pin_memory().to(DEVICE, non_blocking=True)
                     if _CUDA else torch.tensor(own_np, device=DEVICE))
            threat_t = (torch.from_numpy(threat_np).pin_memory().to(DEVICE, non_blocking=True)
                        if _CUDA else torch.tensor(threat_np, device=DEVICE))
            f_aux = f[aux_indices]
            if aux_w_own > 0:
                own_pred = net.ownership(f_aux)
                loss_aux = loss_aux + aux_w_own * F.mse_loss(own_pred, own_t)
            if aux_w_threat > 0:
                thr_logits = net.threat_logits(f_aux)
                loss_aux = loss_aux + aux_w_threat * F.binary_cross_entropy_with_logits(
                    thr_logits, threat_t)

    # Value uncertainty
    loss_unc = torch.tensor(0.0, device=DEVICE)
    unc_w = CFG.get("UNC_LOSS_WEIGHT", 0.1)
    if unc_w > 0:
        sigma2 = net.variance(f)
        loss_unc = 0.5 * (sigma2.log() + (z_targets - val) ** 2 / sigma2).mean()

    loss = CFG.get("VALUE_LOSS_WEIGHT", 2.0) * loss_v + loss_p + loss_aux + unc_w * loss_unc

    if torch.isnan(loss) or loss.item() > 100.0:
        log.warning("Bad loss (%.2f) detected! Skipping batch.", loss.item())
        return None

    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        avg_ent = loss_ent.item() if ent_reg > 0 and n_p > 0 else 0.0
    avg_sigma = sigma2.detach().sqrt().mean().item() if unc_w > 0 else 0.0
    return {"loss": loss.item(), "loss_v": loss_v.item(), "loss_p": loss_p.item(),
            "loss_aux": loss_aux.item(), "loss_unc": loss_unc.item(),
            "entropy": avg_ent, "avg_sigma": avg_sigma}


# -- Checkpoint ----------------------------------------------------------------

def save(net, gen):
    path = CHECKPOINT_DIR / f"net_gen{gen:04d}.pt"
    sd = net._orig_mod.state_dict() if hasattr(net, "_orig_mod") else net.state_dict()
    torch.save(sd, path)
    torch.save(sd, CHECKPOINT_DIR / "net_latest.pt")
    log.info("Saved checkpoint %s", path)
    return path


def load_latest(net):
    path = CHECKPOINT_DIR / "net_latest.pt"
    if path.exists():
        try:
            target = net._orig_mod if hasattr(net, "_orig_mod") else net
            state = torch.load(path, map_location=DEVICE)
            # Handle migration from old 1-output value head to WDL 3-output
            target_state = target.state_dict()
            for key in list(state.keys()):
                if key in target_state and state[key].shape != target_state[key].shape:
                    log.info("Shape mismatch for %s (%s vs %s), reinitializing",
                             key, state[key].shape, target_state[key].shape)
                    del state[key]
            target.load_state_dict(state, strict=False)
            log.info("Loaded %s", path)
            nums = [int(p.stem.split("gen")[1]) for p in CHECKPOINT_DIR.glob("net_gen*.pt")]
            return max(nums) if nums else 0
        except (RuntimeError, KeyError) as e:
            log.warning("Checkpoint incompatible (%s) -- moving to legacy/", e)
            legacy_dir = CHECKPOINT_DIR / "legacy"
            legacy_dir.mkdir(exist_ok=True)
            for p in CHECKPOINT_DIR.glob("net_*.pt"):
                shutil.move(str(p), str(legacy_dir / p.name))
            log.info("Quarantined incompatible checkpoints, starting fresh")
    return 0


# -- Replay buffer persistence ------------------------------------------------

BUFFER_DIR  = CHECKPOINT_DIR / "buffer"
BUFFER_FILE = CHECKPOINT_DIR / "replay_buffer.npz"  # legacy

_BUF_KEYS = ["board", "policy", "mask", "z", "own", "threat"]

def save_buffer(buffer):
    """Persist replay buffer as separate npy files (memory-friendly)."""
    n = len(buffer)
    if n == 0:
        return
    try:
        BUFFER_DIR.mkdir(exist_ok=True)
        buf_list = list(buffer)
        np.save(BUFFER_DIR / "board.npy",  np.stack([b["board"] for b in buf_list]))
        np.save(BUFFER_DIR / "policy.npy", np.stack([b["policy_target"] for b in buf_list]))
        np.save(BUFFER_DIR / "mask.npy",   np.stack([b["legal_mask"] for b in buf_list]))
        np.save(BUFFER_DIR / "z.npy",      np.array([b["z"] for b in buf_list], dtype=np.float32))
        np.save(BUFFER_DIR / "own.npy",    np.stack([b["own_label"] for b in buf_list]))
        np.save(BUFFER_DIR / "threat.npy", np.stack([b["threat_label"] for b in buf_list]))
    except Exception as e:
        log.warning("Failed to save buffer: %s", e)


def load_buffer(buffer):
    """Restore replay buffer from memory-mapped npy files."""
    z_path = BUFFER_DIR / "z.npy"
    if z_path.exists():
        try:
            mm = {k: np.load(BUFFER_DIR / f"{k}.npy", mmap_mode='r') for k in _BUF_KEYS}
            n = len(mm["z"])
            for i in range(n):
                buffer.append({
                    "board":         np.array(mm["board"][i]),
                    "policy_target": np.array(mm["policy"][i]),
                    "legal_mask":    np.array(mm["mask"][i]),
                    "z":             float(mm["z"][i]),
                    "own_label":     np.array(mm["own"][i]),
                    "threat_label":  np.array(mm["threat"][i]),
                })
            log.info("Loaded replay buffer: %d positions", n)
            return
        except Exception as e:
            log.warning("Failed to load buffer from npy (%s)", e)
    # Legacy npz fallback
    if BUFFER_FILE.exists():
        try:
            d = np.load(BUFFER_FILE)
            n = len(d["z"])
            for i in range(n):
                buffer.append({
                    "board": d["board"][i],
                    "policy_target": d["policy"][i],
                    "legal_mask": d["mask"][i],
                    "z": float(d["z"][i]),
                    "own_label": d["own"][i],
                    "threat_label": d["threat"][i],
                })
            log.info("Loaded replay buffer: %d positions (legacy npz)", n)
        except Exception as e:
            log.warning("Failed to load buffer (%s), starting fresh", e)


# -- Rust self-play bridge -----------------------------------------------------

def make_eval_fn(net, device):
    """Create a batched eval callback for the Rust self-play engine."""
    net.eval()
    def eval_batch(boards_np):
        boards_t = torch.from_numpy(boards_np).to(device)
        with torch.no_grad():
            f = net.trunk(boards_t)
            logits = net.policy_logits(f).float().cpu().numpy()
            values = net.value(f).cpu().numpy().clip(-1.0, 1.0)
        return values, logits
    return eval_batch


def _postprocess_rust_results(raw_results):
    """Convert Rust GameTrainingResult objects, apply TD-lambda + aux labels."""
    td_gamma = CFG.get("TD_GAMMA", 0.99)
    results = []
    for gtr in raw_results:
        winner = gtr.winner
        moves = list(gtr.moves)
        n_pos = len(gtr.positions)

        if n_pos == 0:
            results.append(([], winner, moves))
            continue

        positions = []
        for pos in gtr.positions:
            board_arr = np.array(pos.board, dtype=np.float32).reshape(
                -1, BOARD_SIZE, BOARD_SIZE)
            policy_target = np.array(pos.policy_target, dtype=np.float32).reshape(
                BOARD_SIZE, BOARD_SIZE)
            legal_mask = np.array(pos.legal_mask, dtype=np.float32).reshape(
                BOARD_SIZE, BOARD_SIZE)
            positions.append({
                "board": board_arr,
                "policy_target": policy_target,
                "legal_mask": legal_mask,
                "player": int(pos.player),
                "origin": pos.origin,
                "value_est": float(pos.value_est),
            })

        # TD-lambda value targets
        targets = [0.0] * n_pos
        if winner is None:
            targets[-1] = 0.0
        else:
            targets[-1] = 1.0 if positions[-1]["player"] == winner else -1.0

        for t in range(n_pos - 2, -1, -1):
            sign = 1.0 if positions[t]["player"] == positions[t + 1]["player"] else -1.0
            v_next = positions[t + 1]["value_est"]
            g_next = targets[t + 1]
            targets[t] = sign * td_gamma * ((1 - TD_LAMBDA) * v_next + TD_LAMBDA * g_next)

        # Reconstruct final game state for aux labels
        game = HexGame()
        for (q, r) in moves:
            game.make(q, r)

        for t, pos in enumerate(positions):
            pos["z"] = max(-1.0, min(1.0, targets[t]))
            oq, or_ = pos["origin"]
            own_label, threat_label = make_aux_labels(game, winner, oq, or_)
            pos["own_label"] = own_label
            pos["threat_label"] = threat_label
            del pos["player"]
            del pos["origin"]
            del pos["value_est"]

        results.append((positions, winner, moves))
    return results


# -- Main training loop --------------------------------------------------------

LOCK_FILE = Path("train.lock")
_lock_fh = None  # held open for lifetime of process

def _acquire_lock():
    """Prevent multiple training processes using OS-level file lock."""
    global _lock_fh
    try:
        _lock_fh = open(LOCK_FILE, "w")
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(_lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fh.write(str(os.getpid()))
        _lock_fh.flush()
    except (OSError, IOError):
        log.error("Another training process is already running. Exiting.")
        sys.exit(1)

def _release_lock():
    global _lock_fh
    try:
        if _lock_fh:
            _lock_fh.close()
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _run_training_block(net, optimizer, buffer, n_batches):
    """Run N training batches. Returns aggregated metrics dict."""
    if len(buffer) < BATCH_SIZE:
        return None
    losses, loss_vs, loss_ps, entropies, aux_losses, sigmas = [], [], [], [], [], []
    scaler = None  # no GradScaler — float32 throughout
    for _ in range(n_batches):
        result = train_batch(net, optimizer, scaler, buffer)
        if result:
            losses.append(result["loss"])
            loss_vs.append(result.get("loss_v", 0))
            loss_ps.append(result.get("loss_p", 0))
            entropies.append(result.get("entropy", 0))
            aux_losses.append(result.get("loss_aux", 0))
            sigmas.append(result.get("avg_sigma", 0))
    if not losses:
        return None
    n = len(losses)
    return {
        "n": n, "loss": sum(losses)/n, "loss_v": sum(loss_vs)/n,
        "loss_p": sum(loss_ps)/n, "entropy": sum(entropies)/n,
        "loss_aux": sum(aux_losses)/n, "avg_sigma": sum(sigmas)/n,
    }


def train(n_gens=50, sims=100, games_per_gen=64):
    _acquire_lock()
    import atexit
    atexit.register(_release_lock)

    log.info("=== HexGo Training (Batched MCTS) ===")
    log.info("Device=%s  Params=%s  SIMS=%d  GAMES/GEN=%d  TOP_K=%d  BATCH=%d",
             DEVICE, f"{param_count(HexNet()):,}", sims, games_per_gen, TOP_K, BATCH_SIZE)

    net = HexNet().to(DEVICE)
    start_gen = load_latest(net)
    if start_gen == 0 and CFG.get("WEIGHT_INIT", "xavier") == "ca":
        init_weights_ca(net)
        log.info("Initialized HexConv2d kernels with hex-Laplacian CA priors.")

    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    WARMUP_GENS = 5
    def lr_lambda(gen_idx):
        if gen_idx < WARMUP_GENS:
            return 0.1 + 0.9 * gen_idx / WARMUP_GENS
        cosine_progress = (gen_idx - WARMUP_GENS) / max(n_gens - WARMUP_GENS, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * cosine_progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Fast-forward scheduler to current gen so LR doesn't reset on restart
    for _ in range(start_gen):
        scheduler.step()
    buffer = deque(maxlen=BUFFER_CAP)
    load_buffer(buffer)

    for gen in range(start_gen + 1, start_gen + n_gens + 1):
        log.info("--- Generation %d ---", gen)
        t_gen = time.perf_counter()

        cur_sims = _curriculum_sims(gen, sims)
        cur_max_moves = _curriculum_max_moves(gen)
        cur_zoi = _curriculum_zoi(gen)

        # --- Batched self-play ---
        t_sp = time.perf_counter()
        if _HAS_RUST_BATCHED:
            eval_fn = make_eval_fn(net, DEVICE)
            raw = rust_batched_self_play(
                eval_fn, games_per_gen, cur_sims, cur_max_moves,
                top_k=TOP_K, c_puct=CFG["CPUCT"], fpu_reduction=0.2,
                dirichlet_alpha=CFG.get("DIRICHLET_ALPHA", 0.10),
                dirichlet_eps=CFG.get("DIRICHLET_EPS", 0.25),
                temp_horizon=CFG.get("TEMP_HORIZON", 40),
                random_opening=RANDOM_OPENING,
                random_opening_frac=RANDOM_OPENING_FRAC,
                zoi_margin=cur_zoi,
                zoi_lookback=ZOI_LOOKBACK,
            )
            results = _postprocess_rust_results(raw)
        else:
            results = batched_self_play(net, games_per_gen, cur_sims, cur_max_moves,
                                         TOP_K, DEVICE)
        sp_time = time.perf_counter() - t_sp

        # Collect training data
        total_positions = 0
        game_wins = {1: 0, 2: 0, None: 0}
        total_moves = 0
        for data, winner, moves in results:
            for item in data:
                buffer.append(item)
                total_positions += 1
            game_wins[winner] += 1
            total_moves += len(moves)

        avg_moves = total_moves / max(len(results), 1)
        games_per_s = games_per_gen / max(sp_time, 0.01)
        log.info("  Self-play: %d games in %.1fs  (%.1f games/s)  avg_moves=%.0f  "
                 "sims=%d  max_moves=%d  X=%d O=%d draw=%d  positions=%d  buffer=%d",
                 games_per_gen, sp_time, games_per_s,
                 avg_moves, cur_sims, cur_max_moves,
                 game_wins[1], game_wins[2], game_wins[None],
                 total_positions, len(buffer))

        # Save decisive games for corpus
        n_decisive = save_decisive_games(results, gen)
        if n_decisive > 0:
            log.info("  Saved %d decisive games to replays/decisive/", n_decisive)

        # Save representative replays to main replays/ dir
        decisive_results = [r for r in results if r[1] is not None and len(r[2]) >= 6]
        if decisive_results:
            shortest = min(decisive_results, key=lambda r: len(r[2]))
            longest = max(decisive_results, key=lambda r: len(r[2]))
            save_replay(shortest[2], shortest[1], gen, f"w{shortest[1]}_short")
            if longest is not shortest:
                save_replay(longest[2], longest[1], gen, f"w{longest[1]}_long")
        elif results:
            best = max(results, key=lambda r: len(r[2]))
            save_replay(best[2], best[1], gen, "longest")

        # --- Training ---
        t_tr = time.perf_counter()
        n_batches = max(10, min(len(buffer) // BATCH_SIZE, 150))
        tr_metrics = _run_training_block(net, optimizer, buffer, n_batches)
        tr_time = time.perf_counter() - t_tr

        if tr_metrics:
            m = tr_metrics
            log.info("  Train: %d batches in %.1fs  loss=%.4f  v=%.4f  p=%.4f  "
                     "aux=%.4f  sigma=%.4f  ent=%.4f",
                     m["n"], tr_time, m["loss"], m["loss_v"], m["loss_p"],
                     m["loss_aux"], m["avg_sigma"], m["entropy"])
        else:
            log.info("  Buffer too small (%d < %d)", len(buffer), BATCH_SIZE)

        save(net, gen)

        # Metrics for dashboard
        _metrics = {
            "gen": gen,
            "avg_loss": round(tr_metrics["loss"], 4) if tr_metrics else None,
            "avg_loss_v": round(tr_metrics["loss_v"], 4) if tr_metrics else None,
            "avg_loss_p": round(tr_metrics["loss_p"], 4) if tr_metrics else None,
            "avg_aux": round(tr_metrics["loss_aux"], 4) if tr_metrics else None,
            "avg_sigma": round(tr_metrics["avg_sigma"], 4) if tr_metrics else None,
            "avg_ent": round(tr_metrics["entropy"], 4) if tr_metrics else None,
            "gen_time_s": round(time.perf_counter() - t_gen, 1),
            "sp_time_s": round(sp_time, 1),
            "tr_time_s": round(tr_time, 1),
            "buffer_size": len(buffer),
            "positions": total_positions,
            "lr": optimizer.param_groups[0]["lr"],
            "games_per_s": round(games_per_s, 2),
            "decisive": n_decisive,
            "avg_moves": round(avg_moves, 1),
            "sims": cur_sims,
            "max_moves": cur_max_moves,
        }
        with open("metrics.jsonl", "a", encoding="utf-8") as mf:
            mf.write(json.dumps(_metrics) + "\n")

        # Lightweight ELO eval every 10 generations
        if gen % 10 == 0:
            try:
                elo_tracker = ELO()
                net_agent = NetAgent(net, sims=50, name=f"net_gen{gen:04d}")
                eis_agent = EisensteinGreedyAgent("eisenstein_def", defensive=True)
                elo_result = run_match(net_agent, eis_agent, n_games=6, elo=elo_tracker, verbose=False)
                net_rating = elo_tracker.rating(net_agent.name)
                eis_rating = elo_tracker.rating(eis_agent.name)
                net_wins = elo_result.get(f"wins_{net_agent.name}", 0)
                log.info("  ELO eval: net=%d  eis=%d  (net won %d/6)",
                         net_rating, eis_rating, net_wins)
                _metrics["elo_net"] = round(net_rating, 1)
                _metrics["elo_eis"] = round(eis_rating, 1)
                lines = Path("metrics.jsonl").read_text().rstrip().split("\n")
                lines[-1] = json.dumps(_metrics)
                Path("metrics.jsonl").write_text("\n".join(lines) + "\n")
            except Exception as e:
                log.warning("  ELO eval failed: %s", e)

        # Persist buffer every generation to avoid losing work on restart
        save_buffer(buffer)

        scheduler.step()
        t_total = time.perf_counter() - t_gen
        log.info("  Generation %d done in %.1fs  (sp=%.1fs tr=%.1fs)  lr=%.2e",
                 gen, t_total, sp_time, tr_time, optimizer.param_groups[0]["lr"])

    save_buffer(buffer)
    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens",  type=int, default=200, help="Generations to train")
    parser.add_argument("--sims",  type=int, default=100, help="MCTS sims per move (target)")
    parser.add_argument("--games", type=int, default=128, help="Self-play games per gen (batch size for MCTS)")
    args = parser.parse_args()
    train(n_gens=args.gens, sims=args.sims, games_per_gen=args.games)
