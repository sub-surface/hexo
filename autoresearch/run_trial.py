#!/usr/bin/env python3
"""
AutoResearch trial runner for HexGo.

Runs a fixed-budget training trial and evaluates ELO.
Outputs a JSON result to stdout for the agent to parse.

Usage:
    python run_trial.py --gens 10 [--sims 200] [--games 128] [--elo-games 12]
"""

import argparse
import json
import logging
import shutil
import sys
import time
from collections import deque
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from config import CFG
from net import HexNet, BOARD_SIZE
from elo import NetAgent, EisensteinGreedyAgent, ELO, run_match

log = logging.getLogger("autoresearch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s | %(message)s",
                    datefmt="%H:%M:%S")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
METRICS_FILE = Path(__file__).parent.parent / "metrics.jsonl"
RESULTS_FILE = Path(__file__).parent / "results.tsv"


def get_baseline_metrics():
    """Read last metrics entry as baseline."""
    if not METRICS_FILE.exists():
        return {"avg_loss": 999.0, "avg_loss_v": 999.0, "avg_loss_p": 999.0}
    lines = METRICS_FILE.read_text().strip().splitlines()
    if not lines:
        return {"avg_loss": 999.0, "avg_loss_v": 999.0, "avg_loss_p": 999.0}
    return json.loads(lines[-1])


def get_baseline_elo():
    """Read current net ELO from elo.json."""
    elo = ELO()
    return elo.rating("net") if "net" in elo.ratings else 1200.0


def save_baseline_checkpoint():
    """Copy current net_latest.pt as baseline for rollback."""
    src = CHECKPOINT_DIR / "net_latest.pt"
    dst = CHECKPOINT_DIR / "net_baseline_trial.pt"
    if src.exists():
        shutil.copy2(src, dst)
    return dst


def restore_baseline_checkpoint():
    """Restore baseline checkpoint."""
    src = CHECKPOINT_DIR / "net_baseline_trial.pt"
    dst = CHECKPOINT_DIR / "net_latest.pt"
    if src.exists():
        shutil.copy2(src, dst)


def run_training(n_gens, sims, games_per_gen):
    """Run N generations of training. Returns metrics from last gen."""
    from train import train as _train
    _train(n_gens=n_gens, sims=sims, games_per_gen=games_per_gen)


def run_elo_eval(n_games=12, sims=50):
    """Run ELO evaluation against EisensteinGreedy. Returns net ELO."""
    net = HexNet().to(DEVICE)
    path = CHECKPOINT_DIR / "net_latest.pt"
    if path.exists():
        state = torch.load(path, map_location=DEVICE)
        target_state = net.state_dict()
        for key in list(state.keys()):
            if key in target_state and state[key].shape != target_state[key].shape:
                del state[key]
        net.load_state_dict(state, strict=False)
    net.eval()

    net_agent = NetAgent(net, sims=sims, name="net")
    eis_agent = EisensteinGreedyAgent(defensive=True)
    elo_tracker = ELO()

    log.info("Running ELO eval: %d games @ %d sims", n_games, sims)
    result = run_match(net_agent, eis_agent, n_games=n_games,
                       elo=elo_tracker, verbose=True)

    net_elo = elo_tracker.rating("net")
    eis_elo = elo_tracker.rating("eisenstein_greedy_def")
    net_wins = result.get("wins_net", 0)

    log.info("ELO result: net=%d eis=%d (net won %d/%d)",
             net_elo, eis_elo, net_wins, n_games)
    return net_elo, net_wins


def main():
    parser = argparse.ArgumentParser(description="AutoResearch trial runner")
    parser.add_argument("--gens", type=int, default=10)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--elo-games", type=int, default=12)
    parser.add_argument("--elo-sims", type=int, default=50)
    args = parser.parse_args()

    # 1. Capture baseline
    baseline_metrics = get_baseline_metrics()
    baseline_elo = get_baseline_elo()
    baseline_loss = baseline_metrics.get("avg_loss", 999.0)
    save_baseline_checkpoint()

    log.info("=== AutoResearch Trial ===")
    log.info("Baseline: loss=%.4f elo=%.0f", baseline_loss, baseline_elo)
    log.info("Budget: %d gens, %d sims, %d games/gen", args.gens, args.sims, args.games)

    # 2. Run training
    t0 = time.time()
    try:
        run_training(args.gens, args.sims, args.games)
    except Exception as e:
        log.error("Training crashed: %s", e)
        restore_baseline_checkpoint()
        result = {
            "baseline_loss": baseline_loss,
            "trial_loss": 999.0,
            "baseline_elo": baseline_elo,
            "trial_elo": baseline_elo,
            "decisive_ratio": 0.0,
            "avg_moves": 0,
            "net_wins": 0,
            "wall_time_s": time.time() - t0,
            "kept": False,
            "reason": f"crash: {e}",
        }
        print(json.dumps(result))
        return

    # 3. Get trial metrics
    trial_metrics = get_baseline_metrics()  # reads latest from metrics.jsonl
    trial_loss = trial_metrics.get("avg_loss", 999.0)
    decisive = trial_metrics.get("decisive", 0)
    avg_moves = trial_metrics.get("avg_moves", 0)
    decisive_ratio = decisive / 128.0 if decisive else 0.0

    # 4. Run ELO eval
    trial_elo, net_wins = run_elo_eval(n_games=args.elo_games, sims=args.elo_sims)

    wall_time = time.time() - t0

    # 5. Decide keep/discard
    kept = False
    reason = "no improvement"

    elo_improved = trial_elo > baseline_elo + 10
    loss_improved = trial_loss < baseline_loss - 0.02 and decisive_ratio > 0.7
    elo_regressed = trial_elo < baseline_elo - 20
    loss_regressed = trial_loss > baseline_loss + 0.05

    if elo_regressed or loss_regressed:
        kept = False
        reason = "regression" + (" (elo)" if elo_regressed else " (loss)")
    elif elo_improved:
        kept = True
        reason = f"elo +{trial_elo - baseline_elo:.0f}"
    elif loss_improved:
        kept = True
        reason = f"loss -{baseline_loss - trial_loss:.4f}"
    else:
        # Marginal — keep if no regression and decisive ratio is good
        if decisive_ratio > 0.8 and not loss_regressed:
            kept = True
            reason = "marginal improvement, good decisive ratio"

    if not kept:
        restore_baseline_checkpoint()

    result = {
        "baseline_loss": round(baseline_loss, 4),
        "trial_loss": round(trial_loss, 4),
        "baseline_elo": round(baseline_elo, 1),
        "trial_elo": round(trial_elo, 1),
        "decisive_ratio": round(decisive_ratio, 2),
        "avg_moves": round(avg_moves, 1),
        "net_wins": net_wins,
        "wall_time_s": round(wall_time, 1),
        "kept": kept,
        "reason": reason,
    }

    log.info("Result: %s — %s", "KEPT" if kept else "DISCARDED", reason)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
