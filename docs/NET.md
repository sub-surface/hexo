# HexGo — Neural Network (`net.py`)

## Architecture

```
Input [17, 18, 18]
  → HexConv2d(17→128, 3×3) + BN + ReLU            [stem: hex-masked conv]
  → 6× ResBlock(128ch, HexConv2d)                  [trunk: Z[ω]-faithful kernels]
  → GlobalPoolBranch(128ch)                         [KataGo global context]
  ├→ Conv2d(128→1, 1×1) + BN + ReLU → FC → Tanh   [value head → scalar ∈ [-1,1]]
  ├→ AdaptiveAvgPool → FC → Softplus               [value uncertainty → σ² (disabled, UNC_LOSS_WEIGHT=0)]
  ├→ Conv2d(128→1, 1×1) + Tanh                     [ownership aux → [S,S]]
  ├→ Conv2d(128→1, 1×1)                            [threat aux → [S,S]]
  └→ Conv2d(128→128, 1×1) + BN + ReLU →
       Conv2d(128→1, 1×1)                          [spatial policy → [B, S, S] logit map]
```

| Param | Value | Rationale |
|-------|-------|-----------|
| Board window | 18×18 | Centered on recent-move centroid; covers >95% of game extents |
| Hidden channels | 128 | Configurable via CFG["TRUNK_CHANNELS"] |
| Residual blocks | 6 | Configurable via CFG["TRUNK_BLOCKS"] |
| Total params | ~1.9M | Scaled from ~121K (4blk/64ch) and ~480K (4blk/64ch) |
| Precision | FP16 AMP | `torch.amp.autocast` doubles memory bandwidth |
| Weight init | Hex-Laplacian CA | `init_weights_ca()` — Z[ω]-aligned diffusion prior |
| torch.compile | Disabled | Uses 3-4GB RAM during JIT compilation |

---

## Input Encoding (`encode_board`)

Returns `float32 [17, 18, 18]` centered on the centroid of the last N_RECENT (20) moves.

| Channel(s) | Contents |
|-----------|----------|
| 0 | P1 current pieces |
| 1 | P2 current pieces |
| 2 | To-move plane: 0.0 = P1 to move, 1.0 = P2 to move |
| 3–6 | P1 last 4 moves (most recent = ch 3), one-hot each |
| 7–10 | P2 last 4 moves (most recent = ch 7), one-hot each |
| 11–13 | Current player axis-chain planes (one per Z[ω] axis: (1,0), (0,1), (1,-1)) |
| 14–16 | Opponent axis-chain planes (one per Z[ω] axis) |

`N_HISTORY = 4`, `IN_CH = 3 + 2×4 + 6 = 17`.

The axis-chain planes (11-16) encode the Eisenstein integer structure directly: each
empty candidate cell carries three independent chain-length signals (one per Z[ω]
unit direction), normalized by WIN_LENGTH and clipped to [0, 1].

Also returns `(oq, or_)` — the integer centroid offset used to encode moves.

---

## HexConv2d

`HexConv2d(nn.Conv2d)` enforces the Z[ω] 7-cell neighbourhood by zeroing
two corners of every 3×3 kernel via a registered `hex_mask` buffer:

```
  Mask (✓ = active, ✗ = zeroed):
    ✗ ✓ ✓
    ✓ ✓ ✓
    ✓ ✓ ✗
```

Masked positions: `[*, *, 0, 0]` (top-left) and `[*, *, 2, 2]` (bottom-right).

The `forward()` hook applies `weight * hex_mask` before every pass, ensuring
weight updates cannot restore the masked corners during training.

Used in: all ResBlock convolutions. The stem uses standard `Conv2d` (intent:
allow the stem to learn the full encoding before the hex-faithful layers).

---

## D6 Data Augmentation

The symmetry group of Z[ω] is D6 (order 12): 6 rotations at 60° + 6 reflections.
In axial (q, r) coordinates, all 12 transforms are 2×2 integer matrices.

`D6_MATRICES` — shape `[12, 2, 2]` int32 array:

```python
# Rotations (counterclockwise, 60° steps)
R0:  [[ 1, 0],[ 0, 1]]   identity
R60: [[ 0,-1],[ 1, 1]]
R120:[[-1,-1],[ 1, 0]]
R180:[[-1, 0],[ 0,-1]]
R240:[[ 0, 1],[-1,-1]]
R300:[[ 1, 1],[-1, 0]]
# Reflections
S0:  [[ 0, 1],[ 1, 0]]   swap q,r
S60: [[-1, 0],[ 1, 1]]
...
```

`d6_augment_sample(sample, tf_idx)` transforms both the board array and move
coordinates consistently. Applied at `train_batch` time: `tf_idx = random.randrange(12)`.

Moves outside the 18×18 window after transform are set to `None` (handled by
caller). Up to 12× sample efficiency at zero self-play cost.

---

## GlobalPoolBranch (KataGo-style)

Inserted after `self.blocks` in `HexNet.trunk()`:

```
trunk features [B, 128, H, W]
  → avg_pool → [B, 128]
  → max_pool → [B, 128]
  → cat      → [B, 256]
  → FC(256, 128) + ReLU
  → reshape  → [B, 128, 1, 1]
  → broadcast_add to trunk features
```

Gives every spatial cell awareness of global game state (threat density,
material balance) at ~2K extra parameters.

---

## Policy Head (Spatial)

`policy_logits(features) → [B, S, S]` spatial logit map.

A single forward pass produces logits for all cells simultaneously. No per-move
forward pass required. Legal moves are extracted from the logit map using
`move_to_grid(q, r, oq, or_)` to map axial coordinates to grid indices.

`encode_move()` is retained for backward compatibility but is no longer used
in the primary training or inference paths.

The `evaluate()` helper in `net.py` does one trunk + value + policy pass and
returns `(value, {move: logit})` for all legal moves within the window.

Training loss: **Spatial masked cross-entropy** — the logit map is masked to
legal moves, softmax is applied over the masked map, and cross-entropy is
computed against the MCTS visit count distribution (stored as a `[S, S]`
`policy_target` spatial plane). Fully vectorized, no Python loops over moves.

---

## Checkpoint Compatibility

`load_latest()` handles breaking architecture changes (e.g., IN_CH 3→11):
on `RuntimeError`, all `net_*.pt` files are moved to `checkpoints/legacy/`
and training starts fresh with a clean net.

---

## Parameter Count Note

Current architecture with HIDDEN=128, N_BLOCKS=6 has ~1.9M parameters.
Previous sizes: ~121K (4blk/64ch), ~480K (4blk/64ch scaled).
`param_count(net)` reports the exact count.
