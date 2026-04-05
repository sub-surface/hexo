"""
HexNet — ResNet policy+value network for hexagonal 6-in-a-row.

Architecture (configured via config.py CFG):
  - Input: 17 × 18 × 18 axial grid centered on recent-move centroid (3 state + 8 history + 6 axis-chain)
  - Trunk: CFG["TRUNK_BLOCKS"] residual blocks, CFG["TRUNK_CHANNELS"] channels
    Default: 4 blocks × 64 channels. KataGo-style global pool after trunk.
  - Value head: board → scalar win probability ∈ [-1, 1]
  - Policy head: board → spatial 18×18 logit map (one logit per cell, single forward pass)
  - Value uncertainty head: board → σ² (Gaussian NLL, diagnostic only)
  - Ownership head: board → [S, S] ∈ (-1, 1)  (+1=P1, -1=P2, 0=empty)
  - Threat head: board → [S, S] ∈ (0, 1)  (1=cell on winning 6-in-a-row)
  Aux heads are thin 1×1 convs off trunk features — zero extra trunk compute.

Device: CUDA if available, else CPU.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from game import HexGame, AXES, WIN_LENGTH
from config import CFG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOARD_SIZE = 18          # increased from 15 (sees more of the infinite grid)
N_HISTORY = 4            # 4c: history planes per player (last N moves)
IN_CH = 3 + 2 * N_HISTORY + 6  # p1, p2, to_move + 4 p1-history + 4 p2-history + 6 axis-chain planes = 17
N_RECENT = 20            # recent moves used to center the board window (tracks active play area)

# Architecture sizes — read from CFG so they are tunable without code changes.
# Kept as module constants so downstream code that imports HIDDEN/N_BLOCKS still works.
HIDDEN   = CFG["TRUNK_CHANNELS"]   # 64 (was 32)
N_BLOCKS = CFG["TRUNK_BLOCKS"]     # 4  (was 2)

# ── D6 symmetry group of the hexagonal lattice Z[omega] ──────────────────────
#
# The hex grid is isomorphic to the Eisenstein integers Z[omega], omega=e^(2pi*i/3).
# Its symmetry group D6 (order 12) consists of 6 rotations (60° steps) and 6
# reflections.  In axial (q,r) coordinates each element is a 2x2 integer matrix:
#
#   new_q = M[0,0]*q + M[0,1]*r
#   new_r = M[1,0]*q + M[1,1]*r
#
# Applying all 12 transforms to a training sample gives 12 equivalent positions
# for free — up to 12x sample efficiency without encoding any game knowledge.

D6_MATRICES = np.array([
    # ── 6 rotations (counterclockwise, 60° steps) ──
    [[ 1,  0], [ 0,  1]],   # R0   identity
    [[ 0, -1], [ 1,  1]],   # R60
    [[-1, -1], [ 1,  0]],   # R120
    [[-1,  0], [ 0, -1]],   # R180
    [[ 0,  1], [-1, -1]],   # R240
    [[ 1,  1], [-1,  0]],   # R300
    # ── 6 reflections ──
    [[ 0,  1], [ 1,  0]],   # S0  (swap q,r)
    [[-1,  0], [ 1,  1]],   # S60
    [[-1, -1], [ 0,  1]],   # S120
    [[ 0, -1], [-1,  0]],   # S180
    [[ 1,  0], [-1, -1]],   # S240
    [[ 1,  1], [ 0, -1]],   # S300
], dtype=np.int32)           # shape [12, 2, 2]


def _transform_board(board_arr: np.ndarray, tf_idx: int,
                     size: int = BOARD_SIZE) -> np.ndarray:
    """
    Apply one of the 12 D6 symmetry transforms to a board encoding array.
    Channel 2 (to-move plane) is rotation-invariant and copied unchanged.
    Pixels that transform outside the window are dropped (zeroed in dest).
    """
    half = size // 2
    M = D6_MATRICES[tf_idx]          # [2,2]

    # Relative coordinates for every (row, col) in the source array
    # q_grid[r,q] = q_rel,  r_grid[r,q] = r_rel
    qs = np.arange(size) - half
    rs = np.arange(size) - half
    q_grid, r_grid = np.meshgrid(qs, rs)   # both [size, size]

    # Apply linear transform to all source positions
    q_dst = M[0, 0] * q_grid + M[0, 1] * r_grid   # [size, size]
    r_dst = M[1, 0] * q_grid + M[1, 1] * r_grid

    col_dst = (q_dst + half).astype(np.int32)
    row_dst = (r_dst + half).astype(np.int32)

    # Mask: only source pixels whose dest falls within the window
    valid = (col_dst >= 0) & (col_dst < size) & (row_dst >= 0) & (row_dst < size)

    # Flat source indices for valid pixels
    src_rows_v = np.where(valid, np.indices((size, size))[0], 0)[valid]
    src_cols_v = np.where(valid, np.indices((size, size))[1], 0)[valid]
    row_dst_v  = row_dst[valid]
    col_dst_v  = col_dst[valid]

    new_arr = np.zeros_like(board_arr)
    new_arr[2] = board_arr[2]          # to-move: rotation-invariant
    for ch in range(IN_CH):
        if ch == 2:
            continue
        new_arr[ch, row_dst_v, col_dst_v] = board_arr[ch, src_rows_v, src_cols_v]

    return new_arr


def _transform_aux(arr: np.ndarray, tf_idx: int,
                   size: int = BOARD_SIZE) -> np.ndarray:
    """Apply one D6 transform to a single [S, S] spatial label array."""
    half = size // 2
    M = D6_MATRICES[tf_idx]
    qs = np.arange(size) - half
    rs = np.arange(size) - half
    q_grid, r_grid = np.meshgrid(qs, rs)
    q_dst = M[0, 0] * q_grid + M[0, 1] * r_grid
    r_dst = M[1, 0] * q_grid + M[1, 1] * r_grid
    col_dst = (q_dst + half).astype(np.int32)
    row_dst = (r_dst + half).astype(np.int32)
    valid = (col_dst >= 0) & (col_dst < size) & (row_dst >= 0) & (row_dst < size)
    src_rows_v = np.indices((size, size))[0][valid]
    src_cols_v = np.indices((size, size))[1][valid]
    new_arr = np.zeros((size, size), dtype=arr.dtype)
    new_arr[row_dst[valid], col_dst[valid]] = arr[src_rows_v, src_cols_v]
    return new_arr


def d6_augment_sample(sample: dict, tf_idx: int) -> dict:
    """
    Return one D6-equivalent version of a training buffer sample.
    Board array, policy target, legal mask, and aux labels are spatially transformed.
    Policy target is renormalized after transform (mass may clip at edges).
    """
    new_board = _transform_board(sample['board'], tf_idx)
    new_policy = _transform_aux(sample['policy_target'], tf_idx)
    new_mask = _transform_aux(sample['legal_mask'], tf_idx)

    # Renormalize policy target (probability mass may clip at window edges)
    s = new_policy.sum()
    if s > 0:
        new_policy = new_policy / s

    out = {
        'board': new_board,
        'policy_target': new_policy,
        'legal_mask': new_mask,
        'z': sample['z'],
    }
    # Aux labels are spatial [S, S] arrays — apply the same D6 spatial remap.
    for key in ('own_label', 'threat_label'):
        if key in sample:
            out[key] = _transform_aux(sample[key], tf_idx)
    # Preserve greedy_move for move_acc metric (not spatially transformed — just copied)
    if 'greedy_move' in sample:
        out['greedy_move'] = sample['greedy_move']
    return out


# ── Board encoding ────────────────────────────────────────────────────────────

def encode_board(game: HexGame, size: int = BOARD_SIZE, fast: bool = False) -> np.ndarray:
    """
    Returns float32 array [IN_CH, size, size] centered on centroid of all pieces.
    If board empty, centers at (0,0).

    Channel layout (IN_CH = 17):
      0   — player 1 current pieces
      1   — player 2 current pieces
      2   — to-move plane (0.0=p1, 1.0=p2)
      3-6 — player 1 last N_HISTORY moves (most recent = ch 3), one-hot each
      7-10— player 2 last N_HISTORY moves (most recent = ch 7), one-hot each
      11  — MY axis-0 (1,0):  chain length current player would join, normalised to [0,1]
      12  — MY axis-1 (0,1):  chain length current player would join, normalised to [0,1]
      13  — MY axis-2 (1,-1): chain length current player would join, normalised to [0,1]
      14  — OPP axis-0 (1,0):  chain length opponent would join, normalised to [0,1]
      15  — OPP axis-1 (0,1):  chain length opponent would join, normalised to [0,1]
      16  — OPP axis-2 (1,-1): chain length opponent would join, normalised to [0,1]

    The axis decomposition encodes the Eisenstein integer structure directly: each
    empty candidate cell carries three independent chain signals (one per Z[omega]
    unit direction) rather than a single collapsed maximum.  The network receives
    the full per-axis potential and learns to weight threats across all three
    Eisenstein axes, instead of discovering the axes from value targets alone.
    Values clipped at 1.0 so immediate wins (chain >= WIN_LENGTH) are always max.
    """
    half = size // 2
    if game.move_history:
        # Center on centroid of the last N_RECENT moves — tracks the active play area.
        # Using recent moves (not all pieces) prevents the window drifting to an average
        # that clips active threats when the game spreads across the infinite grid.
        recent = game.move_history[-N_RECENT:]
        cq = sum(q for q, r in recent) / len(recent)
        cr = sum(r for q, r in recent) / len(recent)
        oq, or_ = round(cq), round(cr)
    else:
        oq, or_ = 0, 0

    arr = np.zeros((IN_CH, size, size), dtype=np.float32)
    cp = game.current_player - 1   # 0 or 1
    arr[2, :, :] = cp

    # Channels 0-1: current board state
    for (q, r), p in game.board.items():
        qi = q - oq + half
        ri = r - or_ + half
        if 0 <= qi < size and 0 <= ri < size:
            arr[p - 1, ri, qi] = 1.0

    # 4c: history planes — last N_HISTORY placements per player, most recent first.
    # Use player_history (parallel to move_history) so the split is correct even
    # during MCTS tree traversal after unmake() has removed pieces from the board.
    p1_hist, p2_hist = [], []
    for m, mp in zip(reversed(game.move_history), reversed(game.player_history)):
        if mp == 1 and len(p1_hist) < N_HISTORY:
            p1_hist.append(m)
        elif mp == 2 and len(p2_hist) < N_HISTORY:
            p2_hist.append(m)
        if len(p1_hist) >= N_HISTORY and len(p2_hist) >= N_HISTORY:
            break
    for i, (q, r) in enumerate(p1_hist):
        qi = q - oq + half
        ri = r - or_ + half
        if 0 <= qi < size and 0 <= ri < size:
            arr[3 + i, ri, qi] = 1.0
    for i, (q, r) in enumerate(p2_hist):
        qi = q - oq + half
        ri = r - or_ + half
        if 0 <= qi < size and 0 <= ri < size:
            arr[7 + i, ri, qi] = 1.0

    if fast:
        return arr, (oq, or_)

    # Axis-chain planes 11-16: Eisenstein axis decomposition.
    # For each empty candidate cell, compute the chain length along each of the
    # three Z[omega] unit axes independently, for both current player and opponent.
    # Values are normalised by WIN_LENGTH and clipped to [0, 1] so a value of 1.0
    # means placing here would complete or extend a winning chain.
    me  = game.current_player
    opp = 3 - me
    for (q, r) in game.candidates:
        qi = q - oq + half
        ri = r - or_ + half
        if not (0 <= qi < size and 0 <= ri < size):
            continue
        for player, ch_base in ((me, 11), (opp, 14)):
            for axis_idx, (dq, dr) in enumerate(AXES):
                run = 1
                for sign in (1, -1):
                    nq, nr = q + sign * dq, r + sign * dr
                    while game.board.get((nq, nr)) == player:
                        run += 1
                        nq += sign * dq
                        nr += sign * dr
                arr[ch_base + axis_idx, ri, qi] = min(1.0, run / WIN_LENGTH)

    return arr, (oq, or_)


def move_to_grid(q: int, r: int, oq: int, or_: int,
                 size: int = BOARD_SIZE) -> tuple[int, int] | None:
    """Map axial (q,r) to grid (row, col) indices. Returns None if out of window."""
    half = size // 2
    col = q - oq + half
    row = r - or_ + half
    if 0 <= col < size and 0 <= row < size:
        return (row, col)
    return None


def top_k_from_logit_map(logit_map, board, oq, or_, k=16, size=BOARD_SIZE,
                         _candidates=None):
    """Extract top-K legal moves directly from a logit map.

    Scans the highest-valued cells in the logit map, checks they are empty.
    If _candidates is provided (the adjacency set), uses it as a fast legality
    pre-filter. Otherwise accepts any empty cell within the window.
    """
    half = size // 2
    flat = logit_map.ravel()
    n_top = min(len(flat), k * 8)
    top_idx = np.argpartition(flat, -n_top)[-n_top:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

    result = []
    for idx in top_idx:
        row, col = divmod(int(idx), size)
        q = col - half + oq
        r = row - half + or_
        if (q, r) in board:
            continue
        result.append(((q, r), float(flat[idx])))
        if len(result) >= k:
            break
    return result


def encode_move(q: int, r: int, oq: int, or_: int,
                size: int = BOARD_SIZE) -> np.ndarray | None:
    """1-hot [1, size, size] plane for a candidate move. Returns None if out of window.
    (Legacy — kept for backward compatibility.)"""
    half = size // 2
    qi = q - oq + half
    ri = r - or_ + half
    if 0 <= qi < size and 0 <= ri < size:
        plane = np.zeros((1, size, size), dtype=np.float32)
        plane[0, ri, qi] = 1.0
        return plane
    return None


# ── Network ───────────────────────────────────────────────────────────────────

class HexConv2d(nn.Conv2d):
    """
    3x3 convolution with a fixed hex-7 kernel mask.

    The hex grid has 6 neighbours per cell: (+-1,0), (0,+-1), (+1,-1), (-1,+1).
    In axial coordinates mapped to a 2D array (col=q, row=r), these correspond
    to all 3x3 neighbours EXCEPT the two corners at (Dq,Dr)=(-1,-1) and (+1,+1)
    — i.e., kernel positions [0,0] and [2,2].  Zeroing those weights enforces
    that the network only attends to the true Z[omega] neighbourhood rather than
    the geometrically incorrect square neighbourhood.

    Faithful to Sutton: this is a structural prior about geometry, not game knowledge.
    The network still discovers what matters within the correct substrate.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        kwargs.setdefault('padding', 1)
        super().__init__(in_channels, out_channels, kernel_size=3, **kwargs)
        mask = torch.ones(1, 1, 3, 3)
        mask[0, 0, 0, 0] = 0.0   # (-1,-1) direction: not a hex neighbour
        mask[0, 0, 2, 2] = 0.0   # (+1,+1) direction: not a hex neighbour
        self.register_buffer('hex_mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.hex_mask,
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = HexConv2d(ch, ch, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = HexConv2d(ch, ch, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        return F.relu(x + r)


class GlobalPoolBranch(nn.Module):
    """
    KataGo-style global pooling branch.

    After the residual trunk, concatenate board-wide average and max pooling
    features and project them back to the spatial feature map. This gives
    every cell awareness of the global game state (total material, threat density,
    board extent) at essentially zero extra compute.

    Architecture:
        x [B, C, H, W] → avg_pool → [B, C]  ─┐
                        → max_pool → [B, C]  ─┼→ FC(2C→C) → broadcast → x + g
    """
    def __init__(self, ch: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch * 2, ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(2, 3))                          # [B, C]
        mx  = x.amax(dim=(2, 3))                          # [B, C]
        g   = self.fc(torch.cat([avg, mx], dim=1))        # [B, C]
        g   = g.unsqueeze(-1).unsqueeze(-1).expand_as(x)  # [B, C, H, W]
        return x + g                                       # residual broadcast


# ── CA weight initialization ─────────────────────────────────────────────────

def init_weights_ca(net: "HexNet") -> None:
    """
    Initialize HexConv2d kernels with a discrete hex-Laplacian pattern.

    The hex-7 neighbourhood (3×3 with corners [0,0] and [2,2] masked) is exactly
    the NCA "perceive" kernel.  Setting weights to a normalized Laplacian:
        center = 1.0, 6 neighbours = -1/6 each
    gives each filter a Z[omega]-aligned diffusion prior — it detects local
    deviation from the neighbourhood mean, which is the natural substrate for
    chain detection along the three Eisenstein axes.

    Non-HexConv2d layers (1×1 convs, BN, FC) get standard Xavier/Kaiming init.
    This is a structural prior about geometry, not game knowledge.
    """
    # Hex-7 neighbourhood in 3×3 kernel (row, col): center + 6 neighbours.
    # Masked positions (0,0) and (2,2) are zeroed by HexConv2d.forward — skip them.
    # Center = (1,1); 6 hex neighbours = all 3×3 cells except (0,0) and (2,2).
    _HEX_NEIGHBORS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    center_val   =  1.0
    neighbor_val = -1.0 / 6.0

    for module in net.modules():
        if isinstance(module, HexConv2d):
            with torch.no_grad():
                w = module.weight  # [out_ch, in_ch, 3, 3]
                nn.init.zeros_(w)
                # Set center weight
                w[:, :, 1, 1] = center_val
                # Set hex neighbour weights
                for r, c in _HEX_NEIGHBORS:
                    w[:, :, r, c] = neighbor_val
                # Scale by Xavier fan factor so gradient magnitudes are reasonable
                fan = w[0].numel()
                scale = math.sqrt(2.0 / fan)
                w.mul_(scale)


class HexNet(nn.Module):
    """
    Shared trunk: IN_CH → hidden conv → n_blocks residual blocks → global pool
    Value head:   trunk → conv 1×1 → flatten → FC → tanh scalar
    Policy head:  (trunk_features, move_plane) → conv 1×1 → flatten → FC → scalar
    """
    def __init__(self, hidden: int = HIDDEN, n_blocks: int = N_BLOCKS):
        super().__init__()
        self.hidden = hidden
        self.n_blocks = n_blocks

        # Trunk — HexConv2d in stem ensures hex geometry prior from the very first layer
        self.stem = nn.Sequential(
            HexConv2d(IN_CH, hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        # KataGo global pool: gives each cell global board awareness (threat density etc.)
        self.global_pool = GlobalPoolBranch(hidden)

        # Value head — WDL (win/draw/loss) 3-class softmax, trained with cross-entropy.
        # Bounded by design: cross-entropy on softmax cannot produce runaway gradients.
        v_hidden = hidden * 2
        self.v_conv = nn.Sequential(
            nn.Conv2d(hidden, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.v_fc = nn.Sequential(
            nn.Linear(BOARD_SIZE * BOARD_SIZE, v_hidden),
            nn.ReLU(),
            nn.Linear(v_hidden, 3),  # win, draw, loss logits
        )

        # Policy head — spatial 18×18 logit map (single forward pass for full policy)
        self.p_conv = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1),
        )

        # Value uncertainty head — predicts σ² of value estimate (Softplus → σ² > 0).
        # Trained with Gaussian NLL: loss = 0.5*(log(σ²) + (z-v)²/σ²).
        # Diagnostic only: avg_sigma logged to metrics; not used in MCTS search.
        self.value_var = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

        # Auxiliary heads — thin 1×1 convs off trunk features, no extra trunk compute.
        # Bitter lesson: keep these light; they guide representation, not dominate loss.
        # Ownership: which player owns each cell at game end (+1=P1, -1=P2, 0=empty)
        self.aux_own = nn.Sequential(
            nn.Conv2d(hidden, 1, 1, bias=False),
            nn.Tanh(),
        )
        # Threat: does this cell lie on the winning 6-in-a-row (binary)
        # No sigmoid here — sigmoid applied in threat() for inference;
        # training uses binary_cross_entropy_with_logits (AMP-safe).
        self.aux_threat = nn.Conv2d(hidden, 1, 1, bias=False)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_pool(self.blocks(self.stem(x)))

    def value_wdl(self, features: torch.Tensor) -> torch.Tensor:
        """WDL logits: [B, 3] (win, draw, loss)."""
        v = self.v_conv(features).flatten(1)
        return self.v_fc(v)                       # [B, 3]

    def value(self, features: torch.Tensor) -> torch.Tensor:
        """Scalar value in [-1, 1]: P(win) - P(loss)."""
        wdl = F.softmax(self.value_wdl(features), dim=-1)
        return wdl[:, 0] - wdl[:, 2]             # [B]

    def variance(self, features: torch.Tensor) -> torch.Tensor:
        """Predicted σ² of value estimate. [B] > 0 via Softplus."""
        return self.value_var(features).squeeze(-1)  # [B]

    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, hidden, S, S] -> [B, S, S] spatial logit map."""
        return self.p_conv(features).squeeze(1)

    def ownership(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, hidden, S, S] → [B, S, S] ∈ (-1, 1)"""
        return self.aux_own(features).squeeze(1)

    def threat(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, hidden, S, S] → [B, S, S] ∈ (0, 1) (sigmoid applied for inference)"""
        return torch.sigmoid(self.aux_threat(features).squeeze(1))

    def threat_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Raw logits for use with binary_cross_entropy_with_logits (AMP-safe)."""
        return self.aux_threat(features).squeeze(1)

    def forward(self, board_tensor: torch.Tensor):
        """
        board_tensor: [B, IN_CH, S, S]
        Returns: (value [B], logit_map [B, S, S])
        """
        f = self.trunk(board_tensor)
        v = self.value(f)
        p = self.policy_logits(f)
        return v, p


# ── Auxiliary label generation ────────────────────────────────────────────────

def make_aux_labels(game: HexGame, winner: int | None,
                    oq: int, or_: int,
                    size: int = BOARD_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate ground-truth spatial labels for the auxiliary heads.

    ownership [S, S] float32: +1 where P1 piece at game end, -1 where P2, 0 empty.
    threat    [S, S] float32: 1.0 where a cell lies on the winning 6-in-a-row, 0 elsewhere.

    Called once per episode with the final game state; the same label array is
    broadcast back to every position in the episode (label is the game outcome,
    not the state at that ply — consistent with AlphaZero value target convention).
    """
    half = size // 2
    own_arr    = np.zeros((size, size), dtype=np.float32)
    threat_arr = np.zeros((size, size), dtype=np.float32)

    # Ownership: final board pieces
    for (q, r), p in game.board.items():
        qi = q - oq + half
        ri = r - or_ + half
        if 0 <= qi < size and 0 <= ri < size:
            own_arr[ri, qi] = 1.0 if p == 1 else -1.0

    # Threat: walk the 3 axes through every winning-player piece to find the
    # winning line. Mark all 6 cells. If no winner, threat_arr stays zero.
    if winner is not None:
        from game import AXES, WIN_LENGTH
        for (q, r), p in game.board.items():
            if p != winner:
                continue
            for dq, dr in AXES:
                # Count run length along this axis through (q, r)
                line = [(q, r)]
                for sign in (1, -1):
                    step = 1
                    while True:
                        nq = q + sign * dq * step
                        nr = r + sign * dr * step
                        if game.board.get((nq, nr)) == winner:
                            line.append((nq, nr))
                            step += 1
                        else:
                            break
                if len(line) >= WIN_LENGTH:
                    for lq, lr in line[:WIN_LENGTH]:
                        qi = lq - oq + half
                        ri = lr - or_ + half
                        if 0 <= qi < size and 0 <= ri < size:
                            threat_arr[ri, qi] = 1.0
                    break  # found the winning line for this piece

    return own_arr, threat_arr


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(net: HexNet, game: HexGame) -> tuple[float, dict]:
    """
    Returns (value, {move: logit}) for all legal moves within the window.
    Single forward pass through trunk + value + spatial policy heads.
    value is from current player's perspective: +1 = winning, -1 = losing.
    """
    net.eval()
    board_arr, (oq, or_) = encode_board(game)
    moves = game.legal_moves()
    if not moves:
        return 0.0, {}

    device = next(net.parameters()).device
    board_t = torch.tensor(board_arr, device=device).unsqueeze(0)   # [1,C,S,S]

    with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
        f = net.trunk(board_t)
        value = net.value(f).item()
        logit_map = net.policy_logits(f).squeeze(0).float().cpu().numpy()  # [S, S]

    policy = {}
    for m in moves:
        idx = move_to_grid(m[0], m[1], oq, or_)
        if idx is not None:
            row, col = idx
            policy[m] = float(logit_map[row, col])

    if not policy:
        return value, {}

    return value, policy


def param_count(net: HexNet) -> int:
    return sum(p.numel() for p in net.parameters())


def quantize_for_inference(net: HexNet) -> HexNet:
    """
    2a: Apply dynamic INT8 quantization to FC layers for reduced memory bandwidth
    during batched GPU inference. Call this after loading a trained checkpoint,
    before running inference-only evaluation (not during training).

    Typical gain: 20-30% reduction in FC layer compute on CPU; smaller benefit
    on GPU due to Tensor Cores already handling FP16 efficiently.
    """
    net.eval()
    return torch.ao.quantization.quantize_dynamic(
        net, {nn.Linear}, dtype=torch.qint8
    )


if __name__ == "__main__":
    import time

    net = HexNet().to(DEVICE)
    total = param_count(net)

    print(f"{'='*54}")
    print(f"  HexNet smoke test")
    print(f"  Device : {DEVICE}")
    print(f"  IN_CH  : {IN_CH}  (3 state + {2*N_HISTORY} history)")
    print(f"  Hidden : {HIDDEN}  Blocks: {N_BLOCKS}")
    print(f"{'='*54}")

    # ── param breakdown ───────────────────────────────────────
    sections = {
        "stem":          sum(p.numel() for p in net.stem.parameters()),
        "blocks":        sum(p.numel() for p in net.blocks.parameters()),
        "value_head":    sum(p.numel() for p in net.v_conv.parameters()) +
                         sum(p.numel() for p in net.v_fc.parameters()),
        "policy_head":   sum(p.numel() for p in net.p_conv.parameters()) +
                         sum(p.numel() for p in net.p_fc.parameters()),
        "aux_heads":     sum(p.numel() for p in net.aux_own.parameters()) +
                         sum(p.numel() for p in net.aux_threat.parameters()),
    }
    print(f"\n  Param breakdown  (total {total:,}):")
    for name, count in sections.items():
        bar = "#" * int(count / total * 30)
        print(f"  {name:<14} {count:>7,}  {count/total*100:4.1f}%  {bar}")

    # ── board encoding & history planes ───────────────────────
    game = HexGame()
    for coord in [(0,0),(1,0),(0,1),(1,-1),(0,-1),(1,1)]:
        game.make(*coord)

    arr, (oq, or_) = encode_board(game)
    print(f"\n  encode_board shape: {arr.shape}  origin=({oq},{or_})")
    print(f"  Ch 0  (P1 pieces) : {arr[0].sum():.0f} cells occupied")
    print(f"  Ch 1  (P2 pieces) : {arr[1].sum():.0f} cells occupied")
    print(f"  Ch 2  (to-move)   : {arr[2,0,0]:.0f}  (0=P1, 1=P2)")
    for i in range(N_HISTORY):
        p1_occ = arr[3 + i].sum()
        p2_occ = arr[7 + i].sum()
        print(f"  Ch {3+i} / Ch {7+i}   (hist {i}): P1={p1_occ:.0f}  P2={p2_occ:.0f}")

    # ── policy & value quality ────────────────────────────────
    v, policy = evaluate(net, game)
    logits = np.array(list(policy.values()), dtype=np.float32)
    logits -= logits.max()
    probs = np.exp(logits); probs /= probs.sum()
    entropy = float(-(probs * np.log(probs + 1e-8)).sum())
    max_ent  = float(np.log(max(len(policy), 1)))
    print(f"\n  Value            : {v:.4f}")
    print(f"  Policy entries   : {len(policy)}")
    print(f"  Policy entropy   : {entropy:.3f}  (uniform={max_ent:.3f}, "
          f"ratio={entropy/max(max_ent,1e-8)*100:.0f}%)")

    # ── inference latency ─────────────────────────────────────
    net.eval()
    N_WARMUP, N_RUNS = 20, 200
    for B in (1, 4, 8, 16):
        x = torch.randn(B, IN_CH, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
        m = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
        m[:, 0, BOARD_SIZE//2, BOARD_SIZE//2] = 1.0
        # Warmup
        with torch.no_grad(), torch.amp.autocast(
                device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
            for _ in range(N_WARMUP):
                net(x, m)
        if "cuda" in str(DEVICE):
            torch.cuda.synchronize()
        # Timed
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast(
                device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
            for _ in range(N_RUNS):
                net(x, m)
        if "cuda" in str(DEVICE):
            torch.cuda.synchronize()
        ms_per_call = (time.perf_counter() - t0) / N_RUNS * 1000
        pos_per_sec  = B / ms_per_call * 1000
        print(f"  Batch={B:>2}  {ms_per_call:6.2f}ms/call  "
              f"{pos_per_sec:>8,.0f} pos/s")

    # ── shape check ───────────────────────────────────────────
    B = 8
    x = torch.randn(B, IN_CH, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
    m = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
    m[:, 0, BOARD_SIZE//2, BOARD_SIZE//2] = 1.0
    with torch.no_grad():
        val, pol = net(x, m)
    assert val.shape == (B,),    f"value shape wrong: {val.shape}"
    assert pol.shape == (B,),    f"policy shape wrong: {pol.shape}"
    print(f"\n  Batch shapes OK: value={tuple(val.shape)}  policy={tuple(pol.shape)}")

    # ── matplotlib param chart ────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(
            f"HexNet  {total:,} params  |  {IN_CH}ch × {BOARD_SIZE}² input  |  "
            f"{HIDDEN}ch × {N_BLOCKS} blocks  ({str(DEVICE).upper()})",
            fontsize=11,
        )

        # Left: param share by section
        colors = ["#4C9BE8", "#E87C4C", "#5DBD6C", "#B06DE8"]
        wedges, texts, autotexts = axes[0].pie(
            list(sections.values()),
            labels=list(sections.keys()),
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
        )
        axes[0].set_title("Parameter share")

        # Right: inference latency by batch size
        batch_sizes  = [1, 4, 8, 16]
        latencies_ms = []
        for B in batch_sizes:
            xb = torch.randn(B, IN_CH, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
            mb = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
            mb[:, 0, BOARD_SIZE//2, BOARD_SIZE//2] = 1.0
            with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
                for _ in range(10):
                    net(xb, mb)
            if "cuda" in str(DEVICE):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
                for _ in range(100):
                    net(xb, mb)
            if "cuda" in str(DEVICE):
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) / 100 * 1000)

        axes[1].bar([str(b) for b in batch_sizes], latencies_ms, color="#4C9BE8", width=0.5)
        axes[1].set_xlabel("Batch size")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Inference latency")
        for i, v in enumerate(latencies_ms):
            axes[1].text(i, v + 0.02, f"{v:.2f}ms", ha="center", fontsize=9)

        plt.tight_layout()
        out = "net_profile.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\n  Chart saved: {out}")
    except ImportError:
        print("\n  (matplotlib not available - skipping chart)")

    print(f"\n{'='*54}")
    print("  net.py OK")
