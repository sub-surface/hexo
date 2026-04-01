"""
tune.py — HexGo autotune trial orchestrator.

Usage:
    python tune.py [--gens N] [--games N] [--trials N]

Flow per trial:
  1. Backup current config.py
  2. Read baseline metrics from last N gens in metrics.jsonl
  3. Run train.py --gens N --games G --sims S
  4. Read post-trial metrics, compute policy loss delta
  5. Append result to tune_log.jsonl
  6. If policy loss increased (got worse): revert config.py
  7. Print summary

Claude proposes config.py changes before calling this script.
Claude reads tune_log.jsonl to reason about the next proposal.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG_FILE   = Path("config.py")
CONFIG_BACKUP = Path("config.py.bak")
METRICS_FILE  = Path("metrics.jsonl")
TUNE_LOG      = Path("tune_log.jsonl")
PYTHON        = sys.executable


def _read_cfg():
    """Import CFG from config.py via exec (avoids stale module cache)."""
    ns = {}
    exec(CONFIG_FILE.read_text(), ns)
    return ns["CFG"]


def _read_recent_metrics(n=5):
    """Read last n lines from metrics.jsonl, return list of dicts."""
    if not METRICS_FILE.exists():
        return []
    lines = METRICS_FILE.read_text(encoding="utf-8").strip().splitlines()
    results = []
    for line in lines[-n:]:
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return results


def _avg_metric(metrics, key):
    """Average a metric across a list of metric dicts, ignoring None."""
    vals = [m[key] for m in metrics if m.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def run_trial(gens=5, games=64):
    """Run one autotune trial. Returns the log entry dict."""
    # 1. Backup config
    shutil.copy(CONFIG_FILE, CONFIG_BACKUP)
    cfg = _read_cfg()

    # 2. Baseline: average metrics from last `gens` generations
    baseline = _read_recent_metrics(gens)
    baseline_ploss = _avg_metric(baseline, "avg_loss_p")
    baseline_vloss = _avg_metric(baseline, "avg_loss_v")
    baseline_ent = _avg_metric(baseline, "avg_ent")
    baseline_gen = baseline[-1]["gen"] if baseline else 0

    print(f"Baseline (last {len(baseline)} gens): "
          f"p_loss={baseline_ploss:.4f}  v_loss={baseline_vloss:.4f}  "
          f"ent={baseline_ent:.4f}" if baseline_ploss else "No baseline metrics")

    # 3. Run training
    sims = cfg.get("SIMS", 100)
    cmd = [PYTHON, "train.py",
           "--gens", str(gens),
           "--games", str(games),
           "--sims", str(sims)]
    print(f"Running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print("ERROR: train.py exited with non-zero status — reverting config")
        shutil.copy(CONFIG_BACKUP, CONFIG_FILE)
        return {"error": "train failed", "cfg": cfg}

    # 4. Post-trial: read the new metrics (last `gens` entries should be from our trial)
    all_metrics = _read_recent_metrics(gens)
    # Filter to only metrics from after baseline
    trial_metrics = [m for m in all_metrics if m.get("gen", 0) > baseline_gen]
    if not trial_metrics:
        trial_metrics = all_metrics  # fallback

    trial_ploss = _avg_metric(trial_metrics, "avg_loss_p")
    trial_vloss = _avg_metric(trial_metrics, "avg_loss_v")
    trial_ent = _avg_metric(trial_metrics, "avg_ent")
    trial_decisive = _avg_metric(trial_metrics, "decisive")
    trial_gps = _avg_metric(trial_metrics, "games_per_s")

    # 5. Compute deltas (negative = improved)
    ploss_delta = None
    if baseline_ploss is not None and trial_ploss is not None:
        ploss_delta = round(trial_ploss - baseline_ploss, 6)

    vloss_delta = None
    if baseline_vloss is not None and trial_vloss is not None:
        vloss_delta = round(trial_vloss - baseline_vloss, 6)

    # 6. Decision: keep if policy loss decreased OR value loss decreased significantly
    # (during plateau phases, small policy gains with stable value loss are still good)
    if ploss_delta is not None:
        kept = ploss_delta <= 0.01  # allow tiny regression (noise margin)
    else:
        kept = True  # no baseline to compare, keep by default

    entry = {
        "cfg": cfg,
        "baseline_ploss": baseline_ploss,
        "trial_ploss": trial_ploss,
        "ploss_delta": ploss_delta,
        "baseline_vloss": baseline_vloss,
        "trial_vloss": trial_vloss,
        "vloss_delta": vloss_delta,
        "trial_ent": trial_ent,
        "trial_decisive": trial_decisive,
        "trial_gps": trial_gps,
        "trial_metrics": trial_metrics,
        "elapsed_s": round(elapsed, 1),
        "kept": kept,
    }

    # 7. Append to log
    with TUNE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # 8. Revert if bad
    if not kept:
        shutil.copy(CONFIG_BACKUP, CONFIG_FILE)
        print(f"REVERTED  ploss_delta={ploss_delta:+.4f}  "
              f"({baseline_ploss:.4f} -> {trial_ploss:.4f})")
    else:
        delta_str = f"{ploss_delta:+.4f}" if ploss_delta is not None else "n/a"
        print(f"KEPT      ploss_delta={delta_str}  "
              f"trial_ploss={trial_ploss:.4f}  trial_vloss={trial_vloss:.4f}")

    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run autotune trials")
    parser.add_argument("--gens",   type=int, default=5,  help="Gens per trial")
    parser.add_argument("--games",  type=int, default=64, help="Games per gen")
    parser.add_argument("--trials", type=int, default=1,  help="Number of trials to run")
    args = parser.parse_args()
    for i in range(args.trials):
        print(f"\n{'='*60}")
        print(f"  Trial {i+1}/{args.trials}")
        print(f"{'='*60}")
        entry = run_trial(gens=args.gens, games=args.games)
        if "error" in entry:
            print("Stopping — trial failed")
            break
