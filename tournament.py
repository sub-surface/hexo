"""
tournament.py — Round-robin tournament between HexGo model checkpoints.

Usage:
    # Auto-select: every 10th gen + latest
    python tournament.py --sims 50 --games 4

    # Specific checkpoints
    python tournament.py --models net_gen0050.pt net_gen0100.pt net_gen0150.pt --sims 50

    # Include baseline agents
    python tournament.py --sims 50 --games 4 --include-random --include-greedy

Plays round-robin: every pair plays N games (alternating colors).
Results saved to tournament_results.json with ELO ratings.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from game import HexGame
from net import HexNet, DEVICE
from elo import ELO, NetAgent, RandomAgent, EisensteinGreedyAgent, run_match, play_game

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_FILE = Path("tournament_results.json")


def load_net_agent(path, sims):
    """Load a checkpoint and return a NetAgent."""
    net = HexNet().to(DEVICE)
    sd = torch.load(path, map_location=DEVICE)
    net.load_state_dict(sd)
    net.eval()
    name = path.stem  # e.g. "net_gen0050"
    return NetAgent(net, sims=sims, name=name)


def auto_select_checkpoints(every_n=10):
    """Pick every Nth gen checkpoint + the latest."""
    all_ckpts = sorted(CHECKPOINT_DIR.glob("net_gen*.pt"))
    if not all_ckpts:
        return []

    selected = []
    for p in all_ckpts:
        try:
            gen = int(p.stem.split("gen")[1])
            if gen % every_n == 0:
                selected.append(p)
        except (ValueError, IndexError):
            pass

    # Always include latest
    latest = all_ckpts[-1]
    if latest not in selected:
        selected.append(latest)

    return selected


def run_tournament(agents, games_per_pair=4, verbose=True):
    """Round-robin tournament. Returns results dict."""
    n = len(agents)
    elo = ELO()
    # Reset ratings for clean tournament
    elo.ratings = {}
    elo.history = []

    results = {
        "agents": [a.name for a in agents],
        "matches": [],
        "head_to_head": {},
    }

    total_matches = n * (n - 1) // 2
    match_num = 0

    for i in range(n):
        for j in range(i + 1, n):
            match_num += 1
            a, b = agents[i], agents[j]
            if verbose:
                print(f"\n[{match_num}/{total_matches}] {a.name} vs {b.name} "
                      f"({games_per_pair} games)")

            match = run_match(a, b, n_games=games_per_pair, elo=elo, verbose=verbose)

            key = f"{a.name} vs {b.name}"
            results["matches"].append({
                "p1": a.name,
                "p2": b.name,
                "wins_p1": match[f"wins_{a.name}"],
                "wins_p2": match[f"wins_{b.name}"],
                "draws": match["draws"],
            })
            results["head_to_head"][key] = {
                "wins": match[f"wins_{a.name}"],
                "losses": match[f"wins_{b.name}"],
                "draws": match["draws"],
            }

    # Final standings
    standings = sorted(elo.ratings.items(), key=lambda x: -x[1])
    results["standings"] = [
        {"rank": i + 1, "name": name, "elo": round(rating, 1)}
        for i, (name, rating) in enumerate(standings)
    ]

    return results


def print_standings(results):
    """Pretty-print tournament results."""
    print("\n" + "=" * 60)
    print("  TOURNAMENT STANDINGS")
    print("=" * 60)

    for s in results["standings"]:
        bar = "#" * max(1, int((s["elo"] - 1100) / 10))
        print(f"  {s['rank']:2d}. {s['name']:<20s}  ELO {s['elo']:7.1f}  {bar}")

    print("\n  HEAD-TO-HEAD:")
    for match in results["matches"]:
        p1, p2 = match["p1"], match["p2"]
        w1, w2, d = match["wins_p1"], match["wins_p2"], match["draws"]
        print(f"    {p1:<20s} {w1}-{w2}-{d} {p2}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round-robin model tournament")
    parser.add_argument("--models", nargs="*", help="Specific checkpoint filenames")
    parser.add_argument("--every", type=int, default=10,
                        help="Auto-select every Nth gen (default: 10)")
    parser.add_argument("--sims", type=int, default=50,
                        help="MCTS sims per move (lower=faster)")
    parser.add_argument("--games", type=int, default=4,
                        help="Games per pair (default: 4)")
    parser.add_argument("--include-random", action="store_true",
                        help="Include RandomAgent baseline")
    parser.add_argument("--include-greedy", action="store_true",
                        help="Include EisensteinGreedyAgent baseline")
    args = parser.parse_args()

    agents = []

    # Load model checkpoints
    if args.models:
        for name in args.models:
            path = CHECKPOINT_DIR / name
            if not path.exists():
                print(f"WARNING: {path} not found, skipping")
                continue
            agents.append(load_net_agent(path, args.sims))
    else:
        paths = auto_select_checkpoints(args.every)
        if not paths:
            print("No checkpoints found in checkpoints/")
            exit(1)
        print(f"Auto-selected {len(paths)} checkpoints (every {args.every} gens):")
        for p in paths:
            print(f"  {p.name}")
            agents.append(load_net_agent(p, args.sims))

    # Add baseline agents
    if args.include_random:
        agents.append(RandomAgent())
    if args.include_greedy:
        agents.append(EisensteinGreedyAgent(name="greedy_def", defensive=True))

    if len(agents) < 2:
        print("Need at least 2 agents for a tournament")
        exit(1)

    print(f"\nTournament: {len(agents)} agents, {args.games} games/pair, "
          f"{args.sims} sims/move")
    print(f"Total games: {len(agents) * (len(agents)-1) // 2 * args.games}")

    t0 = time.perf_counter()
    results = run_tournament(agents, games_per_pair=args.games)
    elapsed = time.perf_counter() - t0

    results["config"] = {
        "sims": args.sims,
        "games_per_pair": args.games,
        "elapsed_s": round(elapsed, 1),
    }

    # Save and print
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print_standings(results)
    print(f"\nCompleted in {elapsed:.0f}s — saved to {RESULTS_FILE}")
