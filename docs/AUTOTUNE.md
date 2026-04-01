# HexGo — Autotune System (`tune.py`, `config.py`)

## Overview

Autonomous hyperparameter tuning: Claude proposes a config change, runs a
fixed-budget trial, measures policy loss delta, and keeps or reverts.

```
[Claude reads tune_log.jsonl]
    → proposes config.py edit (1-2 params)
    → python tune.py --gens 5 --games 64
    → tune.py compares avg policy loss before/after
    → appends to tune_log.jsonl
    → keep or revert config.py
```

## Reward Signal

**Policy loss delta** from `metrics.jsonl` — the primary training objective.
- Negative delta = policy improved → KEEP
- Positive delta > 0.01 = policy regressed → REVERT
- 0.01 noise margin to avoid reverting on random fluctuations

Value loss (`avg_loss_v`) and decisive game count are logged but not used
for keep/revert decisions (value loss is already near-optimal at ~0.02).

## Usage

```bash
# Single trial (Claude edits config.py first, then runs this)
python tune.py --gens 5 --games 64

# Multiple trials
python tune.py --gens 5 --games 64 --trials 10
```

## `tune_log.jsonl` Format

One JSON object per line:
```json
{
  "cfg": {"LR": 0.001, "SIMS": 100, ...},
  "baseline_ploss": 0.7842,
  "trial_ploss": 0.7701,
  "ploss_delta": -0.0141,
  "baseline_vloss": 0.0195,
  "trial_vloss": 0.0188,
  "vloss_delta": -0.0007,
  "trial_ent": 0.8012,
  "trial_decisive": 33,
  "trial_gps": 0.4,
  "elapsed_s": 230.0,
  "kept": true
}
```

## Parameter Space

| Parameter | Range | Priority | Notes |
|-----------|-------|----------|-------|
| `SIMS` | 50–200 | 1 | More sims = deeper search but slower |
| `CPUCT` | 1.0–3.0 | 2 | Higher = more exploration in MCTS |
| `LR` | 1e-4–5e-3 | 3 | Log scale; currently 1e-3 |
| `DIRICHLET_ALPHA` | 0.05–0.3 | 4 | Root noise concentration |
| `DIRICHLET_EPS` | 0.15–0.5 | 4 | Noise mixing strength |
| `TD_GAMMA` | 0.95–1.0 | 5 | Lower = faster signal for early positions |
| `TEMP_HORIZON` | 20–60 | 5 | Moves until temperature floor |
| `ENTROPY_REG` | 0.0–0.05 | 6 | Policy entropy bonus |
