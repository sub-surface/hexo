# HexGo Theory Bridge

*2026-05-09*

The main `hexgo` repo should stay focused on the production playing and training
system. The sibling `hexgo-theory` repo is the lab for conjectures, experiments,
figures, corpora, and paper-facing synthesis.

Theory should flow back into this repo only after it survives a small empirical
loop in `hexgo-theory`:

1. A reusable primitive exists under `hexgo-theory/engine/`.
2. A reproducible experiment exists under `hexgo-theory/experiments/run_*.py`.
3. The experiment writes a result under `hexgo-theory/results/`.
4. The experiment writes a figure under `hexgo-theory/figures/`.
5. The result supports a concrete change to training, evaluation, MCTS, or
   diagnostics in this repo.

Candidate bridge signals from the current CGT program:

- 2-move temperature can become an auxiliary policy target if it predicts strong
  moves better than scalar potential.
- hot-component count can become a curriculum signal if it correlates with
  tactical learning difficulty.
- D6-canonical motif counts can become replay-buffer diagnostics if they
  stabilize in strong self-play and distinguish useful games from noise.
- blocker-cover density can become a White-side defensive diagnostic if it
  tracks successful non-loss play.

Do not copy large theory outputs into this repo. Link to the theory result and
keep production code changes small, tested, and justified by data.

---

## Bellman-Turing Bridge (2026-05-20)

**Theory result:** `hexgo-theory/docs/theory/2026-05-20-bellman-turing-instability.md`  
**Experiment:** `hexgo-theory/experiments/run_bellman_turing.py`  
**Key prediction:** λ* = 11.81 hex units; optimal-play stone distribution has a
Turing instability at k* = 0.532 hex⁻¹ driven by the Erdős-Selfridge potential
contrast (own = activator d_A = 2.5, opp = inhibitor d_I = 5.0).

**Production change:** `L_BR` auxiliary loss in `train.py` / `train_batch()`.  
Weight controlled by `CFG["BR_LOSS_WEIGHT"]` (default 0.0 = off, 0.05 = active).

The loss is a KL divergence from the network's policy to the Boltzmann fixed-point
distribution over the potential difference Δφ = φ^own − φ^opp:

```
π_BR(c) ∝ exp(β · Δφ(c))
L_BR = KL(π_BR ‖ π_net)   [masked to legal moves, only when own-potential is present]
```

`Δφ` is read directly from board channels 11-13 (own axis-chains) and 14-16
(opp axis-chains) — zero additional game-state computation per batch.
`β` is `CFG["BR_BETA"]` (default 3.0).

**Monitoring:** `avg_br` in `metrics.jsonl`; printed in training log as `br=...`.  
**Falsifier:** if `avg_br` does not decrease faster than `loss_p` over the first
50 gens, the Turing signal is not helping and the weight should be reduced to 0.

---

## GRAM Inference Features (2026-05-20)

Drawn from *Generative Recursive Reasoning* (Baek et al., 2026-05-20).

### Feature 1 — Uncertainty-Adaptive Dirichlet (`ADAPTIVE_DIRICHLET`)

`CFG["ADAPTIVE_DIRICHLET"] = True` (default on).  
At root: `effective_alpha = DIRICHLET_ALPHA / (1 + sigma)` where `sigma = sqrt(variance_head)`.  
High sigma (net uncertain) → flatter noise → more exploration.  
Low sigma (net confident) → concentrated noise → exploitation.  
Paper lesson: GRAM shows that *state-dependent* stochastic guidance outperforms fixed noise.

### Feature 2 — Best-of-N Roots (`BEST_OF_N_ROOTS`)

`CFG["BEST_OF_N_ROOTS"] = N` (default 1 = off).  
Runs N independent MCTS trees from the same position with different noise seeds.  
Selects the root whose visit distribution minimises Bellman-residual distance to π_BR.  
Implements GRAM §2.3 width-based inference-time scaling with L_BR as the LPRM signal.  
Recommended range: 3–5 for ELO eval; 1 for training self-play (too slow).

### Feature 3 — Two-Timescale Structural Branch (`STRUCTURAL_BRANCH`)

`CFG["STRUCTURAL_BRANCH"] = True` (default off).  
`StructuralBranch` in `net.py`: AvgPool(stride=6) → conv3×3 → upsample → zero-init 1×1 proj.  
Stride 6 ≈ λ*/2: captures spatial structure at the Turing wavelength scale.  
Zero-init projection: branch is inert at load time, activates gradually during training.  
Fully backward-compatible: branch weights always present in state dict, `strict=True` loads fine.  
Paper lesson: GRAM hierarchical z=(h,l) splits abstract (slow) from tactical (fast) computation.

### Feature 5 — Adaptive Computation Time (`ACT_SIGMA_THRESH`)

`CFG["ACT_SIGMA_THRESH"] = 0.3` activates ACT (default 0.0 = off).  
`CFG["ACT_SIMS_STEP"] = 25` sims per block.  
Runs sim blocks until `sigma < threshold` or `num_simulations` budget exhausted.  
Net-confident positions (small sigma) return early; uncertain positions get full budget.  
Paper lesson: GRAM §A.1 ACT halts when Q(halt) > Q(continue), learned via TD loss.  
Our version is lighter: uses the existing variance head as the halt signal, no extra training.
