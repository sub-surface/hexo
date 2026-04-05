# HexGo AutoResearch — Agent Instructions

## Overview

You are an autonomous ML research agent optimizing HexGo's AlphaZero training pipeline.
You run experiments in a loop: **propose change → train → evaluate → keep/discard → repeat**.

**NEVER STOP.** After each experiment, immediately start the next one.

## Environment

- **Project**: `C:\Users\landa\.claude\projects\hexgo2`
- **Python**: `C:\Users\landa\AppData\Local\Programs\Python\Python312\python.exe`
- **GPU**: RTX 5070 Ti (16GB VRAM)
- **Training**: `bash run.sh train.py --gens <N> --sims 200 --games 128`
- **Experiment runner**: `bash run.sh autoresearch/run_trial.py --gens 10`

## The Loop

### 1. READ STATE
- Read `autoresearch/results.tsv` to see past experiments and what worked/failed
- Read `config.py` and the top of `train.py` (lines 60-85) to see current hyperparameters
- Read `metrics.jsonl` last 10 lines for recent training metrics
- Read `train.log` last 20 lines for current training status

### 2. PROPOSE ONE CHANGE
- Pick ONE thing to change. Not two. ONE.
- Write a clear hypothesis: "Changing X from A to B should improve Y because Z"
- Mutations can target:
  - `config.py` CFG dict values (CPUCT, DIRICHLET_*, ENTROPY_REG, loss weights, etc.)
  - `train.py` constants (BUFFER_CAP, BATCH_SIZE, LR, TOP_K, SIMS_MIN, SIMS_RAMP, MAX_MOVES_*, TD_LAMBDA)
  - `train.py` training logic (LR schedule, batch sampling, loss computation)
  - `net.py` architecture (head sizes, activation functions — careful with these)
- Do NOT touch: `game.py`, `hexgo-rs/`, `mcts.py` (game logic is correct and tested)

### 3. IMPLEMENT
- Make the code change
- `git add -A && git commit -m "autoresearch: <short description>"`

### 4. RUN TRIAL
```bash
bash run.sh autoresearch/run_trial.py --gens 10
```
This will:
- Stop any running training
- Save current checkpoint as baseline
- Train for 10 generations
- Run a 12-game ELO evaluation
- Output a JSON result to stdout

### 5. EVALUATE
Parse the trial result JSON:
```json
{
  "baseline_loss": 2.85,
  "trial_loss": 2.80,
  "baseline_elo": 1200,
  "trial_elo": 1250,
  "decisive_ratio": 0.85,
  "avg_moves": 65,
  "kept": true
}
```

**Keep criteria** (ANY of these):
- `trial_elo > baseline_elo + 10` (ELO improvement)
- `trial_loss < baseline_loss - 0.02` AND `decisive_ratio > 0.7` (loss + quality improvement)

**Discard criteria**:
- `trial_elo < baseline_elo - 20` (significant ELO regression)
- `trial_loss > baseline_loss + 0.05` (loss regression)
- Training crashed or produced NaN

### 6. KEEP OR DISCARD
- **If kept**: Log success to `autoresearch/results.tsv`. Continue from new state.
- **If discarded**: `git revert HEAD --no-edit`. Restore baseline checkpoint. Log failure.

### 7. LOG
Append one line to `autoresearch/results.tsv`:
```
timestamp	experiment_id	description	hypothesis	metric_before	metric_after	elo_before	elo_after	kept	git_hash
```

### 8. LOOP BACK TO STEP 1

## Experiment Priority Queue

Start with these (Tier 2 from research analysis), in order:
1. TOP_K curriculum: ramp from 64→24 over 20 gens
2. Cosine warm restarts (T_0=20, T_mult=2) instead of single cosine
3. Increase buffer to 300K with exponential decay sampling
4. Increase ELO eval from 6 to 12 games
5. Down-weight draw games in buffer (0.5x sampling weight)
6. Short-term value auxiliary targets (6-turn and 16-turn horizons)

After exhausting the queue, propose your own experiments based on results.

## Constraints

- **Budget per trial**: 10 generations (if gens take >5 min each, reduce to 5 gens)
- **Never delete train.lock** — it's an OS-level process lock
- **Never run training while another training is running** — check the lock first
- **Save buffer before stopping training** — wait for current gen to complete
- **One change at a time** — isolate variables for clean attribution
- **If 3 consecutive experiments fail, pause and re-read recent results to reconsider approach**

## Success Metric

The ultimate goal: **maximize ELO vs EisensteinGreedyAgent**. Everything else (loss, decisive ratio, game length) is a proxy. When in doubt, trust ELO.
