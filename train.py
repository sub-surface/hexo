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

from game import HexGame
from mcts import Node, _backprop
from net import (HexNet, encode_board, move_to_grid, DEVICE, BOARD_SIZE,
                 param_count, d6_augment_sample, make_aux_labels, init_weights_ca)
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

BUFFER_CAP   = 50_000
BATCH_SIZE   = 256     # large batch to saturate GPU during training
LR           = CFG["LR"]
WEIGHT_DECAY = CFG["WEIGHT_DECAY"]

# Batched self-play settings
TOP_K        = 16      # expanded from 12 — more candidate moves to find winning lines
SIMS_MIN     = 16      # min sims early in training
SIMS_RAMP    = 20      # generations to ramp from SIMS_MIN to target
MAX_MOVES_MIN = 30     # max moves per game early in training
MAX_MOVES_MAX = 100    # compromise: fewer artificial draws than 80, faster than 150
MAX_MOVES_RAMP = 20    # generations to ramp
DECISIVE_DIR  = Path("replays/decisive")
DECISIVE_DIR.mkdir(parents=True, exist_ok=True)
TD_LAMBDA     = 0.8       # temporal difference lambda for value targets


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
    dir_eps = CFG.get("DIRICHLET_EPS", 0.25)
    zoi_margin = CFG.get("ZOI_MARGIN", 6)
    zoi_lookback = CFG.get("ZOI_LOOKBACK", 16)

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
            legal = games[i].zoi_moves(zoi_margin, zoi_lookback)

            move_logits = []
            for m in legal:
                g = move_to_grid(m[0], m[1], oq, or_)
                if g:
                    move_logits.append((m, root_logits[local_idx, g[0], g[1]]))

            if not move_logits:
                active.discard(i)
                continue

            move_logits.sort(key=lambda x: x[1], reverse=True)
            move_logits = move_logits[:top_k]

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
                eval_data = [encode_board(games[i]) for i, _, _ in needs_eval]
                eval_np = np.stack([d[0] for d in eval_data])
                eval_origins = [d[1] for d in eval_data]

                eval_t = torch.tensor(eval_np, device=device)
                with torch.amp.autocast(device_type="cuda" if _CUDA else "cpu"):
                    ef = net.trunk(eval_t)
                    ep = net.policy_logits(ef).float().cpu().numpy()
                ev = net.value(ef.float()).cpu().numpy().clip(-1.0, 1.0)

                for j, (i, node, depth) in enumerate(needs_eval):
                    oq, or_ = eval_origins[j]
                    legal = games[i].zoi_moves(zoi_margin, zoi_lookback)

                    ml = []
                    for m in legal:
                        g = move_to_grid(m[0], m[1], oq, or_)
                        if g:
                            ml.append((m, ep[j, g[0], g[1]]))

                    if ml:
                        ml.sort(key=lambda x: x[1], reverse=True)
                        ml = ml[:top_k]
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

    with torch.amp.autocast(device_type="cuda" if _CUDA else "cpu"):
        f = net.trunk(boards)
        val = net.value(f)
        loss_v = F.mse_loss(val, z_targets)

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

    if torch.isnan(loss):
        log.warning("NaN loss detected! Skipping batch.")
        return None

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

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
            target.load_state_dict(torch.load(path, map_location=DEVICE))
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

BUFFER_FILE = CHECKPOINT_DIR / "replay_buffer.npz"

# Keys and shapes for each buffer item (all float32):
#   board:         [IN_CH, S, S]
#   policy_target: [S, S]
#   legal_mask:    [S, S]
#   z:             scalar
#   own_label:     [S, S]
#   threat_label:  [S, S]

def save_buffer(buffer):
    """Persist replay buffer as stacked numpy arrays (uncompressed for speed)."""
    n = len(buffer)
    if n == 0:
        return
    try:
        buf_list = list(buffer)
        np.savez(
            BUFFER_FILE,
            board=np.stack([b["board"] for b in buf_list]),
            policy=np.stack([b["policy_target"] for b in buf_list]),
            mask=np.stack([b["legal_mask"] for b in buf_list]),
            z=np.array([b["z"] for b in buf_list], dtype=np.float32),
            own=np.stack([b["own_label"] for b in buf_list]),
            threat=np.stack([b["threat_label"] for b in buf_list]),
        )
    except Exception as e:
        log.warning("Failed to save buffer: %s", e)


def load_buffer(buffer):
    """Restore replay buffer from compressed numpy archive."""
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
            log.info("Loaded replay buffer: %d positions from %s", len(buffer), BUFFER_FILE)
        except Exception as e:
            log.warning("Failed to load buffer (%s), starting fresh", e)


# -- Main training loop --------------------------------------------------------

def train(n_gens=50, sims=100, games_per_gen=64):
    log.info("=== HexGo Training (Batched MCTS) ===")
    log.info("Device=%s  Params=%s  SIMS=%d  GAMES/GEN=%d  TOP_K=%d  BATCH=%d",
             DEVICE, f"{param_count(HexNet()):,}", sims, games_per_gen, TOP_K, BATCH_SIZE)

    net = HexNet().to(DEVICE)
    start_gen = load_latest(net)
    if start_gen == 0 and CFG.get("WEIGHT_INIT", "xavier") == "ca":
        init_weights_ca(net)
        log.info("Initialized HexConv2d kernels with hex-Laplacian CA priors.")

    # torch.compile disabled — uses 3-4GB RAM during compilation
    # if _CUDA and hasattr(torch, "compile"):
    #     try:
    #         net = torch.compile(net, dynamic=True)
    #         log.info("torch.compile enabled (dynamic=True)")
    #     except Exception:
    #         pass

    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    WARMUP_GENS = 5
    def lr_lambda(gen_idx):
        if gen_idx < WARMUP_GENS:
            return 0.1 + 0.9 * gen_idx / WARMUP_GENS
        cosine_progress = (gen_idx - WARMUP_GENS) / max(n_gens - WARMUP_GENS, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * cosine_progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=_CUDA)
    buffer = deque(maxlen=BUFFER_CAP)
    # Buffer persistence disabled — was causing hangs (save) and OOM (load)

    for gen in range(start_gen + 1, start_gen + n_gens + 1):
        log.info("--- Generation %d ---", gen)
        t_gen = time.perf_counter()

        cur_sims = _curriculum_sims(gen, sims)
        cur_max_moves = _curriculum_max_moves(gen)

        # --- Batched self-play ---
        t_sp = time.perf_counter()
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
            # Save shortest decisive (cleanest win) and longest decisive
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
        # Scale training to fill ~1 epoch of the buffer, min 10 batches
        n_batches = max(10, len(buffer) // BATCH_SIZE)
        losses, loss_vs, loss_ps, entropies, aux_losses, sigmas = [], [], [], [], [], []

        for _ in range(n_batches):
            result = train_batch(net, optimizer, scaler, buffer)
            if result:
                losses.append(result["loss"])
                loss_vs.append(result.get("loss_v", 0))
                loss_ps.append(result.get("loss_p", 0))
                entropies.append(result.get("entropy", 0))
                aux_losses.append(result.get("loss_aux", 0))
                sigmas.append(result.get("avg_sigma", 0))

        tr_time = time.perf_counter() - t_tr

        if losses:
            n = len(losses)
            log.info("  Train: %d batches in %.1fs  loss=%.4f  v=%.4f  p=%.4f  "
                     "aux=%.4f  sigma=%.4f  ent=%.4f",
                     n, tr_time, sum(losses)/n, sum(loss_vs)/n, sum(loss_ps)/n,
                     sum(aux_losses)/n, sum(sigmas)/n, sum(entropies)/n)
        else:
            log.info("  Buffer too small (%d < %d)", len(buffer), BATCH_SIZE)

        # Checkpoint (buffer persistence disabled — was blocking for 300MB+ numpy save)
        save(net, gen)

        # Metrics for dashboard
        avg_loss = sum(losses) / len(losses) if losses else None
        avg_ent = sum(entropies) / len(entropies) if entropies else None
        _metrics = {
            "gen": gen,
            "avg_loss": round(avg_loss, 4) if avg_loss else None,
            "avg_loss_v": round(sum(loss_vs)/len(loss_vs), 4) if loss_vs else None,
            "avg_loss_p": round(sum(loss_ps)/len(loss_ps), 4) if loss_ps else None,
            "avg_aux": round(sum(aux_losses)/len(aux_losses), 4) if aux_losses else None,
            "avg_sigma": round(sum(sigmas)/len(sigmas), 4) if sigmas else None,
            "avg_ent": round(avg_ent, 4) if avg_ent else None,
            "gen_time_s": round(time.perf_counter() - t_gen, 1),
            "sp_time_s": round(sp_time, 1),
            "tr_time_s": round(tr_time, 1),
            "buffer_size": len(buffer),
            "positions": total_positions,
            "lr": optimizer.param_groups[0]["lr"],
            "games_per_s": round(games_per_s, 2),
            "decisive": n_decisive,
            "sims": cur_sims,
            "max_moves": cur_max_moves,
        }
        with open("metrics.jsonl", "a", encoding="utf-8") as mf:
            mf.write(json.dumps(_metrics) + "\n")

        scheduler.step()
        t_total = time.perf_counter() - t_gen
        log.info("  Generation %d done in %.1fs  (sp=%.1fs tr=%.1fs)  lr=%.2e",
                 gen, t_total, sp_time, tr_time, optimizer.param_groups[0]["lr"])

    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens",  type=int, default=50,  help="Generations to train")
    parser.add_argument("--sims",  type=int, default=100, help="MCTS sims per move (target)")
    parser.add_argument("--games", type=int, default=64,  help="Self-play games per gen (batch size for MCTS)")
    args = parser.parse_args()
    train(n_gens=args.gens, sims=args.sims, games_per_gen=args.games)
