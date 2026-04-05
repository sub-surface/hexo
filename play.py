"""
play.py — Play against a HexGo checkpoint with analysis tools.

Features:
  Left-click      Place a piece
  Right-click     Highlight a cell (toggle)
  Right-drag      Draw an arrow between cells
  Escape          Clear all annotations
  H               Toggle ownership heatmap
  N               Toggle move numbers (replaces X/O labels)
  T               Toggle top-move suggestions
  Scroll          Zoom
  Middle-drag     Pan

  Live eval bar on left edge shows position value from P1 perspective.
  Top suggested moves shown as gold markers when it's your turn.

Usage:
    python play.py                          # latest checkpoint, 100 sims
    python play.py --model net_gen0100.pt   # specific checkpoint
    python play.py --sims 200              # stronger bot
    python play.py --player 2              # play as O (bot goes first)
"""

import argparse
import math
import sys
import threading
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
from game import HexGame, PLACEMENT_RADIUS, _cells_within_radius
from net import (HexNet, DEVICE, encode_board, top_k_from_logit_map,
                 move_to_grid, BOARD_SIZE, param_count)
from mcts import mcts_with_net
from elo import EisensteinGreedyAgent

CHECKPOINT_DIR = Path("checkpoints")
LEGACY_DIR = CHECKPOINT_DIR / "legacy"
FAVORITES_FILE = Path("favorites.json")


def _load_favorites():
    if FAVORITES_FILE.exists():
        try:
            import json
            return set(json.loads(FAVORITES_FILE.read_text()))
        except Exception:
            pass
    return set()


def _save_favorites(favs):
    import json
    FAVORITES_FILE.write_text(json.dumps(sorted(favs)))


_favorites = _load_favorites()


class _EisensteinBot:
    """Wraps EisensteinGreedyAgent to look like a net for the bot_turn / match system."""
    def __init__(self, defensive=True):
        self._agent = EisensteinGreedyAgent(
            name=f"eisenstein_{'def' if defensive else 'atk'}", defensive=defensive)
        # Dummy attributes so play.py code doesn't crash on net-like checks
        self.name = self._agent.name
        self._is_eisenstein = True

    def choose_move(self, game):
        return self._agent.choose_move(game)


def _discover_models():
    """Find all .pt files + built-in bots."""
    models = []
    # Built-in algorithmic bots (no checkpoint needed)
    models.append(("-- Eisenstein Greedy (defensive)", "eisenstein_def"))
    models.append(("-- Eisenstein Greedy (attack)", "eisenstein_atk"))
    # Neural net checkpoints
    for d in [CHECKPOINT_DIR, LEGACY_DIR]:
        if d.exists():
            for p in sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
                rel = str(p.relative_to(CHECKPOINT_DIR)).replace("\\", "/")
                models.append((rel, p))
    return models


def _load_checkpoint(path, device=DEVICE):
    """Load a HexNet checkpoint from a .pt file. Returns (net, label)."""
    net = HexNet().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    label = Path(path).stem
    return net, label


def _load_model_any(name_or_path, device=DEVICE):
    """Load a model — handles both neural checkpoints and built-in bots."""
    if name_or_path == "eisenstein_def":
        return _EisensteinBot(defensive=True), "eisenstein (defensive)"
    elif name_or_path == "eisenstein_atk":
        return _EisensteinBot(defensive=False), "eisenstein (attack)"
    else:
        return _load_checkpoint(name_or_path, device)

# ── Colors ────────────────────────────────────────────────────────────────────
BG        = "#0a0e14"
EMPTY     = "#111820"
BORDER    = "#1c2a3a"
P1_FILL   = "#8b2020"
P1_EDGE   = "#e04040"
P1_TEXT   = "#ff7070"
P2_FILL   = "#103060"
P2_EDGE   = "#3080e0"
P2_TEXT   = "#70a0ff"
LAST_EDGE = "#ffffff"
HOVER_FILL = "#1a2a1a"
DIM       = "#586374"
LIGHT     = "#c9d1d9"
EVAL_P1   = "#c03030"
EVAL_P2   = "#3060c0"
EVAL_MID  = "#1a1e24"
SUGGEST   = "#d4a020"
HIGHLIGHT = "#30c060"
ARROW_COL = "#e09030"
OWN_P1    = "#c03030"
OWN_P2    = "#3060c0"

# Fonts — Segoe UI for chrome, Consolas for data/coordinates
FONT_UI_LG = ("Segoe UI", 11)
FONT_UI = ("Segoe UI", 10)
FONT_UI_SM = ("Segoe UI", 9)
FONT_UI_XS = ("Segoe UI", 8)
FONT_MONO = ("Consolas", 10)
FONT_MONO_SM = ("Consolas", 9)
FONT_MONO_XS = ("Consolas", 8)


def _blend(c1, c2, t):
    """Blend hex colors c1 -> c2 by factor t in [0, 1]."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    return f'#{int(r1+(r2-r1)*t):02x}{int(g1+(g2-g1)*t):02x}{int(b1+(b2-b1)*t):02x}'


def _hover_btn(btn, base_bg):
    """Add hover effect to a button."""
    hover_bg = _blend(base_bg, "#ffffff", 0.15)
    btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
    btn.bind("<Leave>", lambda e: btn.config(bg=base_bg))


# ── Network eval (single forward pass) ───────────────────────────────────────

@torch.no_grad()
def _evaluate_position(net, game):
    """Return (value_p1, top_moves, ownership_map, oq, or_) in one pass."""
    if game.winner is not None:
        return (1.0 if game.winner == 1 else -1.0), [], None, 0, 0
    if not game.board:
        return 0.0, [], None, 0, 0

    board_arr, (oq, or_) = encode_board(game, fast=True)
    device = next(net.parameters()).device
    # Adapt channel count if the model expects fewer/more input channels
    expected_ch = net.stem[0].weight.shape[1]
    if board_arr.shape[0] > expected_ch:
        board_arr = board_arr[:expected_ch]  # truncate extra channels
    elif board_arr.shape[0] < expected_ch:
        pad = np.zeros((expected_ch - board_arr.shape[0], *board_arr.shape[1:]), dtype=board_arr.dtype)
        board_arr = np.concatenate([board_arr, pad], axis=0)
    board_t = torch.tensor(board_arr, device=device).unsqueeze(0)

    with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
        f = net.trunk(board_t)
        value = net.value(f.float()).item()
        logit_map = net.policy_logits(f).squeeze(0).float().cpu().numpy()
        own_map = net.ownership(f).squeeze(0).float().cpu().numpy()

    value = max(-1.0, min(1.0, value))
    if game.current_player == 2:
        value = -value  # always from P1 perspective

    top = top_k_from_logit_map(logit_map, game.board, oq, or_, k=5)
    top_moves = []
    if top:
        logits_arr = np.array([l for _, l in top], dtype=np.float32)
        logits_arr -= logits_arr.max()
        probs = np.exp(logits_arr)
        probs /= probs.sum()
        top_moves = [(m, float(p)) for (m, _), p in zip(top, probs)]

    return value, top_moves, own_map, oq, or_


# ── GUI ───────────────────────────────────────────────────────────────────────

class HexPlayGUI:
    EVAL_BAR_W = 24

    def __init__(self, root, net, sims, human_player, model_name="net_latest.pt"):
        self.root = root
        self.net = net
        self.sims = sims
        self.human_player = human_player
        self.bot_player = 3 - human_player
        self.game = HexGame()
        self.last_move = None
        self.bot_thinking = False
        self.undo_stack = []
        self.model_name = model_name

        # View state
        self.hex_size = 20
        self.pan_x = 0
        self.pan_y = 0
        self.hover_cell = None
        self._mid_drag = None

        # Annotations
        self.highlights = set()
        self.arrows = []
        self._rclick_start = None
        self._rclick_dragging = False
        self._rclick_cursor = None

        # Eval cache
        self.eval_value = 0.0
        self.eval_top_moves = []
        self.eval_own_map = None
        self.eval_origin = (0, 0)
        self.eval_history = []
        self._eval_lock = threading.Lock()

        # Toggles
        self.show_suggestions = True
        self.show_ownership = False
        self.show_move_nums = False

        # Match mode state
        self.match_mode = False
        self.match_net = {1: None, 2: None}
        self.match_names = {1: "", 2: ""}
        self.match_sims = 100
        self.match_delay = 400  # ms between moves
        self.match_paused = False
        self.match_games_total = 0
        self.match_game_num = 0
        self.match_results = {1: 0, 2: 0, "draw": 0}

        # ── Build UI ──────────────────────────────────────────────────────────
        root.title("HexGo")
        root.configure(bg=BG)
        root.geometry("1000x750")

        # Top bar (grid layout for stable sizing)
        top = tk.Frame(root, bg="#111820", height=38)
        top.pack(fill=tk.X)
        top.pack_propagate(False)
        top.grid_columnconfigure(0, minsize=280)
        top.grid_columnconfigure(1, weight=1)
        top.grid_columnconfigure(2, minsize=200)
        top.grid_rowconfigure(0, weight=1)

        self.status_lbl = tk.Label(top, text="", font=FONT_UI_LG,
                                   bg="#111820", fg=LIGHT, width=30, anchor="w")
        self.status_lbl.grid(row=0, column=0, padx=(12, 6), sticky="w")

        self.eval_lbl = tk.Label(top, text="Eval: --", font=FONT_UI,
                                 bg="#111820", fg=DIM, width=44, anchor="w")
        self.eval_lbl.grid(row=0, column=1, padx=6, sticky="w")

        # top_moves_lbl kept for match mode but content merged into eval_lbl
        self.top_moves_lbl = tk.Label(top, text="", font=("Consolas", 9),
                                      bg="#111820", fg=DIM)

        right_frame = tk.Frame(top, bg="#111820")
        right_frame.grid(row=0, column=2, padx=(6, 10), sticky="e")

        self.info_lbl = tk.Label(right_frame, text="", font=FONT_UI_XS,
                                 bg="#111820", fg="#3a4450")
        self.info_lbl.pack(side=tk.LEFT, padx=(0, 8))

        bf = tk.Frame(right_frame, bg="#111820")
        bf.pack(side=tk.LEFT, padx=(0, 8))
        for txt, cmd, fg in [("Match", self._open_match_dialog, "#d4a020"),
                              ("Undo", self.undo, "#8080c0"),
                              ("New", self.new_game, "#60a060"),
                              ("Quit", root.quit, "#c06060")]:
            btn = tk.Button(bf, text=txt, command=cmd, font=FONT_UI_SM,
                            bg="#151a22", fg=fg, activebackground="#222830",
                            relief=tk.FLAT, padx=6, pady=1)
            btn.pack(side=tk.LEFT, padx=2)
            _hover_btn(btn, "#151a22")

        self.model_lbl = tk.Label(right_frame, text="", font=FONT_UI_XS,
                                  bg="#111820", fg="#50607a", cursor="hand2")
        self.model_lbl.pack(side=tk.LEFT)
        self.model_lbl.bind("<Button-1>", lambda e: self._open_model_chooser())
        self._update_model_label()

        # Thinking indicator bar
        self.think_bar = tk.Canvas(root, bg=BG, height=0, highlightthickness=0)
        self.think_bar.pack(fill=tk.X)
        self._think_anim_id = None
        self._think_pos = 0

        # Canvas area: eval bar (fixed) + move list (right) + main board canvas (expanding)
        canvas_frame = tk.Frame(root, bg=BG)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.eval_canvas = tk.Canvas(canvas_frame, bg=BG, width=38,
                                     highlightthickness=0)
        self.eval_canvas.pack(side=tk.LEFT, fill=tk.Y)

        # Move list sidebar
        self.move_frame = tk.Frame(canvas_frame, bg="#0d1117", width=160)
        self.move_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.move_frame.pack_propagate(False)

        # Sparkline canvas at top
        self.sparkline_canvas = tk.Canvas(self.move_frame, bg="#0d1117",
                                           height=40, highlightthickness=0)
        self.sparkline_canvas.pack(fill=tk.X, padx=4, pady=(4, 0))

        # Header
        tk.Label(self.move_frame, text="MOVES", font=("Segoe UI", 9, "bold"),
                 bg="#0d1117", fg="#3a4450").pack(fill=tk.X, padx=8, pady=(4, 0))

        # Move list text widget
        self.move_text = tk.Text(self.move_frame, bg="#0d1117", fg=LIGHT,
                                  font=FONT_MONO_XS, wrap=tk.WORD,
                                  borderwidth=0, highlightthickness=0,
                                  state=tk.DISABLED, cursor="arrow",
                                  padx=8, pady=4)
        self.move_text.pack(fill=tk.BOTH, expand=True)
        self.move_text.tag_configure("p1", foreground=P1_TEXT)
        self.move_text.tag_configure("p2", foreground=P2_TEXT)
        self.move_text.tag_configure("turn", foreground="#3a4450")

        self.canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bottom help bar
        hb = tk.Frame(root, bg="#0d1117", height=22)
        hb.pack(fill=tk.X)
        hb.pack_propagate(False)
        tk.Label(hb, font=FONT_UI_XS, bg="#0d1117", fg="#2c3440",
                 text="RClick: highlight | RDrag: arrow | Esc: clear | "
                      "H: ownership | N: move# | T: suggestions | "
                      "Scroll: zoom | MidDrag: pan").pack(side=tk.LEFT, padx=8)
        self.coord_lbl = tk.Label(hb, text="", font=FONT_MONO_XS,
                                   bg="#0d1117", fg=DIM, width=12, anchor="e")
        self.coord_lbl.pack(side=tk.RIGHT, padx=8)

        # Bindings
        c = self.canvas
        c.bind("<Button-1>", self.on_click)
        c.bind("<Motion>", self.on_motion)
        c.bind("<MouseWheel>", self.on_scroll)
        c.bind("<ButtonPress-2>", lambda e: self._start_pan(e))
        c.bind("<B2-Motion>", lambda e: self._do_pan(e))
        c.bind("<ButtonPress-3>", self.on_rclick_press)
        c.bind("<B3-Motion>", self.on_rclick_drag)
        c.bind("<ButtonRelease-3>", self.on_rclick_release)
        c.bind("<Configure>", lambda e: self.draw())
        self.eval_canvas.bind("<Configure>", lambda e: self._draw_eval_bar())
        root.bind("<Escape>", lambda e: self._stop_match() if self.match_mode else self._clear_annotations())
        root.bind("<Key-h>", lambda e: self._toggle("show_ownership"))
        root.bind("<Key-n>", lambda e: self._toggle("show_move_nums"))
        root.bind("<Key-t>", lambda e: self._toggle("show_suggestions"))
        root.bind("<space>", lambda e: self._toggle_match_pause())

        self._update_status()
        self._run_eval()
        if self.game.current_player == self.bot_player:
            root.after(200, self._bot_turn)

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _ax2px(self, q, r):
        sz = self.hex_size
        ox = self.canvas.winfo_width() / 2 + self.pan_x
        oy = self.canvas.winfo_height() / 2 + self.pan_y
        return ox + sz * math.sqrt(3) * (q + r / 2.0), oy + sz * 1.5 * r

    def _px2ax(self, px, py):
        sz = self.hex_size
        ox = self.canvas.winfo_width() / 2 + self.pan_x
        oy = self.canvas.winfo_height() / 2 + self.pan_y
        x, y = px - ox, py - oy
        r = y / (1.5 * sz)
        q = x / (sz * math.sqrt(3)) - r / 2
        rq, rr, rs = round(q), round(r), round(-q - r)
        dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs + q + r)
        if dq > dr and dq > ds:
            rq = -rr - rs
        elif dr > ds:
            rr = -rq - rs
        return int(rq), int(rr)

    def _hex_pts(self, cx, cy):
        s = self.hex_size - 1
        return [(cx + s * math.cos(math.pi / 3 * i + math.pi / 6),
                 cy + s * math.sin(math.pi / 3 * i + math.pi / 6)) for i in range(6)]

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 20 or h < 20:
            return

        self._draw_eval_bar()

        # Visible cells
        cells = set()
        if not self.game.board:
            cells.add((0, 0))
        else:
            for pq, pr in self.game.board:
                for cq, cr in _cells_within_radius(pq, pr, PLACEMENT_RADIUS):
                    cells.add((cq, cr))

        legal = set(self.game.legal_moves()) if self.game.winner is None else set()
        is_human = (self.game.current_player == self.human_player
                    and not self.bot_thinking and self.game.winner is None)

        # Pre-build lookup tables
        suggest = {}
        if self.show_suggestions and is_human:
            suggest = {m: p for m, p in self.eval_top_moves}
        move_nums = {}
        if self.show_move_nums:
            move_nums = {(mq, mr): i + 1
                         for i, (mq, mr) in enumerate(self.game.move_history)}

        # Draw each cell
        for q, r in cells:
            px, py = self._ax2px(q, r)
            if px < -50 or px > w + 50 or py < -50 or py > h + 50:
                continue

            pts = self._hex_pts(px, py)
            piece = self.game.board.get((q, r))
            is_last = self.last_move == (q, r)
            is_legal = (q, r) in legal
            is_hover = self.hover_cell == (q, r) and is_human and is_legal
            is_hl = (q, r) in self.highlights

            # Fill + edge
            if piece == 1:
                fill, edge = P1_FILL, P1_EDGE
            elif piece == 2:
                fill, edge = P2_FILL, P2_EDGE
            else:
                fill = HOVER_FILL if is_hover else EMPTY
                edge = "#2a3a2a" if is_hover else BORDER
                # Ownership tint on empty cells
                if self.show_ownership and self.eval_own_map is not None:
                    idx = move_to_grid(q, r, *self.eval_origin)
                    if idx:
                        ov = float(self.eval_own_map[idx[0], idx[1]])
                        if abs(ov) > 0.08:
                            tint = OWN_P1 if ov > 0 else OWN_P2
                            fill = _blend(fill, tint, min(0.55, abs(ov) * 0.5))

            if is_last:
                edge = LAST_EDGE
            if is_hl:
                edge = HIGHLIGHT

            lw = 2 if (is_last or is_hl) else 1
            c.create_polygon(pts, fill=fill, outline=edge, width=lw)

            # Piece label or move number
            if piece:
                if self.show_move_nums and (q, r) in move_nums:
                    txt = str(move_nums[(q, r)])
                    col = "#ffffff"
                else:
                    txt = "X" if piece == 1 else "O"
                    col = P1_TEXT if piece == 1 else P2_TEXT
                c.create_text(px, py, text=txt, fill=col,
                              font=("Consolas", max(7, int(self.hex_size * 0.42)), "bold"))
            elif is_legal and is_human:
                c.create_oval(px - 2, py - 2, px + 2, py + 2,
                              fill="#2a3a2a", outline="")

            # Suggestion marker (gold dot with %)
            if (q, r) in suggest and not piece:
                prob = suggest[(q, r)]
                rad = max(3, int(self.hex_size * 0.25 * (0.6 + prob)))
                col = _blend(EMPTY, SUGGEST, max(0.35, min(0.95, prob * 1.5)))
                c.create_oval(px - rad, py - rad, px + rad, py + rad,
                              fill=col, outline=_blend(col, "#ffffff", 0.3))
                if self.hex_size >= 15 and prob >= 0.05:
                    c.create_text(px, py, text=f"{int(prob*100)}",
                                  fill="#ffffff",
                                  font=("Consolas", max(6, int(self.hex_size * 0.22))))

        # Arrows (persistent)
        for (q1, r1), (q2, r2) in self.arrows:
            self._draw_arrow(q1, r1, q2, r2, ARROW_COL, 3, False)

        # Pending arrow (dashed, during drag -- snapped to cell centers)
        if self._rclick_dragging and self._rclick_start and self._rclick_cursor:
            x1, y1 = self._ax2px(*self._rclick_start)
            x2, y2 = self._ax2px(*self._rclick_cursor)
            c.create_line(x1, y1, x2, y2, fill=ARROW_COL, width=2,
                          arrow=tk.LAST, arrowshape=(10, 12, 4), dash=(5, 3))

        self._update_move_list()

        if self.game.winner is not None:
            self._draw_result_banner()

    def _draw_arrow(self, q1, r1, q2, r2, color, width, dashed):
        x1, y1 = self._ax2px(q1, r1)
        x2, y2 = self._ax2px(q2, r2)
        kw = dict(fill=color, width=width, arrow=tk.LAST,
                  arrowshape=(12, 15, 5))
        if dashed:
            kw["dash"] = (5, 3)
        self.canvas.create_line(x1, y1, x2, y2, **kw)

    def _draw_eval_bar(self):
        ec = self.eval_canvas
        ec.delete("all")
        ew = ec.winfo_width()
        eh = ec.winfo_height()
        if ew < 5 or eh < 30:
            return

        bw = self.EVAL_BAR_W
        x0 = (ew - bw) // 2
        y0 = 8
        bh = eh - 28  # leave room for numeric label below
        if bh < 30:
            return

        # Outer frame
        ec.create_rectangle(x0, y0, x0 + bw, y0 + bh,
                            fill=EVAL_MID, outline="#252a32")

        v = max(-1.0, min(1.0, self.eval_value))
        split = y0 + int(bh * (1.0 - v) / 2.0)
        split = max(y0 + 1, min(y0 + bh - 1, split))

        # P1 (red, top)
        if split > y0 + 1:
            ec.create_rectangle(x0 + 1, y0 + 1, x0 + bw - 1, split,
                                fill=EVAL_P1, outline="")
        # P2 (blue, bottom)
        if split < y0 + bh - 1:
            ec.create_rectangle(x0 + 1, split, x0 + bw - 1, y0 + bh - 1,
                                fill=EVAL_P2, outline="")

        # Center tick
        mid_y = y0 + bh // 2
        ec.create_line(x0, mid_y, x0 + bw, mid_y, fill="#50586a", width=1)

        # Win percentage text inside dominant segment
        pct = int(50 + abs(v) * 50)
        if bh > 80:
            if v >= 0 and split - y0 > 18:
                # P1 dominant — draw in P1 segment
                mid_seg = y0 + (split - y0) // 2
                ec.create_text(x0 + bw // 2, mid_seg, text=f"{pct}%",
                               fill="#ffffff", font=("Segoe UI", 7, "bold"))
            elif v < 0 and (y0 + bh) - split > 18:
                # P2 dominant — draw in P2 segment
                mid_seg = split + ((y0 + bh) - split) // 2
                ec.create_text(x0 + bw // 2, mid_seg, text=f"{pct}%",
                               fill="#ffffff", font=("Segoe UI", 7, "bold"))

        # Numeric label
        sign = "+" if v >= 0 else ""
        ec.create_text(ew // 2, y0 + bh + 10,
                       text=f"{sign}{v:.2f}", fill=DIM,
                       font=("Consolas", 7))

    def _draw_result_banner(self):
        """Draw a game-over overlay banner on the board canvas."""
        c = self.canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50 or h < 50:
            return

        # Semi-transparent overlay
        c.create_rectangle(0, h // 2 - 60, w, h // 2 + 60,
                           fill="#0a0e14", stipple="gray50", outline="")

        # Determine text
        if self.match_mode:
            winner_name = self.match_names.get(self.game.winner, f"P{self.game.winner}")
            main_text = f"{winner_name} WINS"
            color = P1_TEXT if self.game.winner == 1 else P2_TEXT
        else:
            if self.game.winner == self.human_player:
                main_text = "YOU WIN"
                color = "#60ff60"
            else:
                main_text = "BOT WINS"
                color = "#ff6060"

        shadow = _blend(color, "#000000", 0.6)
        cx, cy = w // 2, h // 2 - 10

        # Glow shadow
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2),(-1,-1),(1,-1),(-1,1),(1,1)]:
            c.create_text(cx + dx, cy + dy, text=main_text,
                          fill=shadow, font=("Segoe UI", 22, "bold"))
        # Main text
        c.create_text(cx, cy, text=main_text,
                      fill=color, font=("Segoe UI", 22, "bold"))
        # Subtitle
        c.create_text(cx, cy + 32, text=f"in {len(self.game.move_history)} moves",
                      fill=DIM, font=("Segoe UI", 10))

    def _update_move_list(self):
        """Refresh the move list sidebar with current game history."""
        self.move_text.config(state=tk.NORMAL)
        self.move_text.delete("1.0", tk.END)

        history = list(zip(self.game.move_history, self.game.player_history))
        if not history:
            self.move_text.config(state=tk.DISABLED)
            return

        # Group by turns (1-2-2 pattern)
        turn = 1
        i = 0
        while i < len(history):
            self.move_text.insert(tk.END, f"{turn:>3}. ", "turn")
            # First turn: 1 move. After that: 2 moves per turn
            n_moves = 1 if turn == 1 else 2
            for j in range(n_moves):
                if i >= len(history):
                    break
                (q, r), p = history[i]
                tag = "p1" if p == 1 else "p2"
                self.move_text.insert(tk.END, f"({q},{r})", tag)
                if j == 0 and n_moves == 2 and i + 1 < len(history):
                    self.move_text.insert(tk.END, " ")
                i += 1
            self.move_text.insert(tk.END, "\n")
            turn += 1

        self.move_text.see(tk.END)
        self.move_text.config(state=tk.DISABLED)

    def _draw_sparkline(self):
        """Draw eval history sparkline in the sidebar."""
        sc = self.sparkline_canvas
        sc.delete("all")
        sw = sc.winfo_width()
        sh = sc.winfo_height()
        if sw < 10 or sh < 10 or len(self.eval_history) < 2:
            return

        mid_y = sh / 2
        # Zero line
        sc.create_line(0, mid_y, sw, mid_y, fill="#252a32", width=1)

        n = len(self.eval_history)
        dx = sw / max(n - 1, 1)

        points = []
        for i, v in enumerate(self.eval_history):
            x = i * dx
            y = mid_y - v * (mid_y - 2)  # map [-1,1] to [sh-2, 2]
            points.append((x, y))

        if len(points) >= 2:
            # Area fills
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                # Red above zero (P1 winning)
                sc.create_polygon(x1, min(y1, mid_y), x2, min(y2, mid_y),
                                x2, mid_y, x1, mid_y,
                                fill="#401515", outline="")
                # Blue below zero (P2 winning)
                sc.create_polygon(x1, max(y1, mid_y), x2, max(y2, mid_y),
                                x2, mid_y, x1, mid_y,
                                fill="#101a30", outline="")

            # Line on top
            flat = [coord for pt in points for coord in pt]
            sc.create_line(*flat, fill=LIGHT, width=1, smooth=True)

    def _start_thinking_anim(self):
        self.think_bar.config(height=3)
        self._think_pos = 0
        self._animate_thinking()

    def _stop_thinking_anim(self):
        if self._think_anim_id:
            self.root.after_cancel(self._think_anim_id)
            self._think_anim_id = None
        self.think_bar.config(height=0)

    def _animate_thinking(self):
        tb = self.think_bar
        w = tb.winfo_width()
        if w < 10:
            self._think_anim_id = self.root.after(30, self._animate_thinking)
            return
        tb.delete("all")
        seg_w = max(40, w // 6)
        x = self._think_pos % (w + seg_w) - seg_w
        tb.create_rectangle(x, 0, x + seg_w, 3, fill="#d4a020", outline="")
        self._think_pos += 3
        self._think_anim_id = self.root.after(30, self._animate_thinking)

    # ── Input handlers ────────────────────────────────────────────────────────

    def on_click(self, event):
        if self.bot_thinking or self.game.winner is not None:
            return
        if self.game.current_player != self.human_player:
            return
        q, r = self._px2ax(event.x, event.y)
        if (q, r) not in set(self.game.legal_moves()):
            return

        self.game.make(q, r)
        self.last_move = (q, r)
        self.draw()
        self._update_status()

        if self.game.winner is not None:
            self._run_eval()
            return
        if self.game.current_player == self.human_player:
            self._run_eval()  # eval before second placement
            return

        self.undo_stack.append(len(self.game.move_history))
        self.root.after(50, self._bot_turn)

    def on_motion(self, event):
        cell = self._px2ax(event.x, event.y)
        if cell != self.hover_cell:
            self.hover_cell = cell
            self.draw()
        if self.hover_cell:
            q, r = self.hover_cell
            self.coord_lbl.config(text=f"({q}, {r})")
        else:
            self.coord_lbl.config(text="")

    def on_scroll(self, event):
        self.hex_size += 2 if event.delta > 0 else -2
        self.hex_size = max(8, min(60, self.hex_size))
        self.draw()

    def _start_pan(self, e):
        self._mid_drag = (e.x, e.y, self.pan_x, self.pan_y)

    def _do_pan(self, e):
        if self._mid_drag:
            sx, sy, spx, spy = self._mid_drag
            self.pan_x = spx + (e.x - sx)
            self.pan_y = spy + (e.y - sy)
            self.draw()

    def on_rclick_press(self, event):
        self._rclick_start = self._px2ax(event.x, event.y)
        self._rclick_dragging = False
        self._rclick_cursor = None

    def on_rclick_drag(self, event):
        if self._rclick_start:
            self._rclick_dragging = True
            self._rclick_cursor = self._px2ax(event.x, event.y)  # snap to cell
            self.draw()

    def on_rclick_release(self, event):
        if not self._rclick_start:
            return
        end = self._px2ax(event.x, event.y)
        start = self._rclick_start

        if not self._rclick_dragging or end == start:
            # Toggle highlight
            if start in self.highlights:
                self.highlights.discard(start)
            else:
                self.highlights.add(start)
        else:
            # Create arrow
            self.arrows.append((start, end))

        self._rclick_start = None
        self._rclick_dragging = False
        self._rclick_cursor = None
        self.draw()

    def _clear_annotations(self):
        self.highlights.clear()
        self.arrows.clear()
        self.draw()

    def _toggle(self, attr):
        setattr(self, attr, not getattr(self, attr))
        self.draw()

    # ── Game logic ────────────────────────────────────────────────────────────

    def _bot_turn(self):
        if self.game.winner is not None:
            return
        self.bot_thinking = True
        self._start_thinking_anim()
        self._update_status()
        self.draw()

        def think():
            while (self.game.current_player == self.bot_player
                   and self.game.winner is None):
                if not self.game.legal_moves():
                    break
                if getattr(self.net, '_is_eisenstein', False):
                    move = self.net.choose_move(self.game)
                else:
                    move = mcts_with_net(self.game, self.net, self.sims)
                self.game.make(*move)
                self.last_move = move

            self.bot_thinking = False
            self.root.after(0, self._stop_thinking_anim)
            self.undo_stack.append(len(self.game.move_history))
            self.root.after(0, self.draw)
            self.root.after(0, self._update_status)
            self.root.after(10, self._run_eval)

        threading.Thread(target=think, daemon=True).start()

    def undo(self):
        if self.bot_thinking or len(self.game.move_history) < 2:
            return
        target = self.undo_stack[-2] if len(self.undo_stack) >= 2 else 0
        while len(self.game.move_history) > target:
            self.game.unmake()
        if len(self.undo_stack) >= 2:
            self.undo_stack.pop()
            self.undo_stack.pop()
        self.last_move = self.game.move_history[-1] if self.game.move_history else None
        self.draw()
        self._update_status()
        self._run_eval()

    def new_game(self):
        if self.bot_thinking:
            return
        self.game = HexGame()
        self.last_move = None
        self.undo_stack = []
        self.highlights.clear()
        self.arrows.clear()
        self.pan_x = self.pan_y = 0
        self.eval_value = 0.0
        self.eval_top_moves = []
        self.eval_own_map = None
        self.eval_history = []
        self.draw()
        self._update_status()
        if self.game.current_player == self.bot_player:
            self.root.after(200, self._bot_turn)

    def _update_status(self):
        if self.game.winner is not None:
            won = self.game.winner == self.human_player
            self.status_lbl.config(text="You win!" if won else "Bot wins!",
                                   fg="#60ff60" if won else "#ff6060")
        elif self.bot_thinking:
            self.status_lbl.config(text="Bot thinking...", fg="#ffcc60")
        elif self.game.current_player == self.human_player:
            first = len(self.game.move_history) == 0
            rem = (1 if first else 2) - self.game.placements_in_turn
            p = "X" if self.human_player == 1 else "O"
            self.status_lbl.config(
                text=f"Your turn ({p}) -- {rem} placement{'s' if rem > 1 else ''}",
                fg=LIGHT)
        else:
            self.status_lbl.config(text="Bot's turn", fg=DIM)

        self.info_lbl.config(text=f"Move {len(self.game.move_history)}  |  "
                                  f"Sims {self.sims}")

    # ── Eval ──────────────────────────────────────────────────────────────────

    def _run_eval(self):
        def _worker():
            try:
                v, top, own, oq, or_ = _evaluate_position(self.net, self.game)
                with self._eval_lock:
                    self.eval_value = v
                    self.eval_top_moves = top
                    self.eval_own_map = own
                    self.eval_origin = (oq, or_)
                self.root.after(0, self._show_eval)
            except Exception:
                pass  # GPU busy — skip silently

        threading.Thread(target=_worker, daemon=True).start()

    def _show_eval(self):
        v = self.eval_value
        self.eval_history.append(self.eval_value)
        self._draw_sparkline()
        sign = "+" if v >= 0 else ""
        who = "P1" if v >= 0 else "P2"
        # Merge eval + top moves into one label
        parts = []
        for m, p in self.eval_top_moves[:3]:
            parts.append(f"({m[0]},{m[1]}) {int(p*100)}%")
        top_str = "  ".join(parts)
        if top_str:
            text = f"Eval: {sign}{v:.2f} ({who})  |  {top_str}"
        else:
            text = f"Eval: {sign}{v:.2f} ({who})"
        self.eval_lbl.config(text=text, fg=P1_TEXT if v >= 0 else P2_TEXT)
        self.top_moves_lbl.config(text="")
        self.draw()

    # ── Model management ──────────────────────────────────────────────────

    def _update_model_label(self):
        if getattr(self.net, '_is_eisenstein', False):
            self.model_lbl.config(
                text=f"[{self.net.name}]  algorithmic  (click to change)")
        else:
            arch = "SE" if isinstance(self.net, HexNet) else "legacy"
            params = param_count(self.net)
            self.model_lbl.config(
                text=f"[{self.model_name}]  {arch} {params/1e6:.1f}M  (click to change)")

    def _load_model(self, path, name):
        """Load a model — neural checkpoint or built-in bot."""
        try:
            net, arch = _load_model_any(path, DEVICE)
            self.net = net
            self.model_name = name
            self._update_model_label()
            if not getattr(net, '_is_eisenstein', False):
                self._run_eval()
            return True
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return False

    def _open_model_chooser(self):
        if self.bot_thinking:
            return
        global _favorites

        win = tk.Toplevel(self.root)
        win.title("Choose Model")
        win.configure(bg=BG)
        win.geometry("520x500")
        win.transient(self.root)
        win.grab_set()

        hdr = tk.Frame(win, bg="#111820", height=36)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="MODELS", font=("Consolas", 10),
                 bg="#111820", fg=LIGHT, padx=12).pack(side=tk.LEFT)
        tk.Label(hdr, text=f"current: {self.model_name}", font=("Consolas", 9),
                 bg="#111820", fg=DIM).pack(side=tk.RIGHT, padx=12)

        frame = tk.Frame(win, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(frame, bg="#111820", fg=LIGHT, font=("Consolas", 10),
                             selectbackground="#1a3050", selectforeground="#ffffff",
                             borderwidth=0, highlightthickness=0,
                             yscrollcommand=scrollbar.set)
        listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        models = _discover_models()
        model_paths = {}  # listbox index -> (rel_name, full_path)
        model_keys = {}   # listbox index -> favorites key string

        def _fav_key(rel_name, full_path):
            return rel_name if isinstance(full_path, str) else rel_name

        def _populate():
            listbox.delete(0, tk.END)
            model_paths.clear()
            model_keys.clear()

            # Sort: favorites first, then the rest
            fav_items = []
            rest_items = []
            for rel_name, full_path in models:
                key = _fav_key(rel_name, full_path)
                if key in _favorites:
                    fav_items.append((rel_name, full_path))
                else:
                    rest_items.append((rel_name, full_path))

            for section, items in [("fav", fav_items), ("rest", rest_items)]:
                for rel_name, full_path in items:
                    is_builtin = isinstance(full_path, str)
                    key = _fav_key(rel_name, full_path)
                    is_fav = key in _favorites
                    star = "*" if is_fav else " "

                    if is_builtin:
                        display = f" {star} {rel_name}"
                    else:
                        size_mb = full_path.stat().st_size / 1e6
                        is_legacy = "legacy" in str(full_path)
                        tag = "L" if is_legacy else "S"
                        display = f" {star} [{tag}] {rel_name:<32s} {size_mb:5.1f}MB"

                    idx = listbox.size()
                    listbox.insert(tk.END, display)
                    model_paths[idx] = (rel_name, full_path)
                    model_keys[idx] = key

                    # Color coding
                    active = rel_name == self.model_name
                    if active:
                        listbox.itemconfig(idx, fg="#60ff60")
                    elif is_builtin:
                        listbox.itemconfig(idx, fg=SUGGEST)
                    elif is_fav:
                        listbox.itemconfig(idx, fg="#e0c050")

        _populate()

        info = tk.Label(win, text="Right-click to toggle favorite", font=("Consolas", 9),
                        bg=BG, fg=DIM)
        info.pack(fill=tk.X, padx=12, pady=(0, 4))

        def on_select(event=None):
            sel = listbox.curselection()
            if not sel:
                return
            rel_name, full_path = model_paths[sel[0]]
            info.config(text=f"Loading {rel_name}...", fg="#ffcc60")
            win.update()
            if self._load_model(full_path, rel_name):
                info.config(text=f"Loaded {rel_name}", fg="#60ff60")
                win.after(500, win.destroy)
            else:
                info.config(text=f"Failed to load {rel_name}", fg="#ff6060")

        def on_toggle_fav():
            global _favorites
            sel = listbox.curselection()
            if not sel:
                info.config(text="Select a model first", fg="#ff6060")
                return
            key = model_keys.get(sel[0])
            if key:
                if key in _favorites:
                    _favorites.discard(key)
                    info.config(text=f"Removed from favorites", fg="#e0c050")
                else:
                    _favorites.add(key)
                    info.config(text=f"Added to favorites", fg="#e0c050")
                _save_favorites(_favorites)
                _populate()

        listbox.bind("<Double-1>", on_select)
        listbox.bind("<Return>", on_select)

        btn_frame = tk.Frame(win, bg=BG)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Button(btn_frame, text="Load", command=on_select,
                  font=("Consolas", 10), bg="#1a2a1a", fg="#60a060",
                  activebackground="#2a3a2a", relief=tk.FLAT, padx=12,
                  pady=4).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Fav", command=on_toggle_fav,
                  font=("Consolas", 10), bg="#2a2a10", fg="#e0c050",
                  activebackground="#3a3a20", relief=tk.FLAT, padx=12,
                  pady=4).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Cancel", command=win.destroy,
                  font=("Consolas", 10), bg="#1a1a2a", fg="#8080c0",
                  activebackground="#2a2a3a", relief=tk.FLAT, padx=12,
                  pady=4).pack(side=tk.RIGHT)

    # ── Match mode (model vs model) ──────────────────────────────────────

    def _open_match_dialog(self):
        if self.bot_thinking or self.match_mode:
            return

        win = tk.Toplevel(self.root)
        win.title("Match Setup")
        win.configure(bg=BG)
        win.geometry("700x500")
        win.transient(self.root)
        win.grab_set()

        # Header
        hdr = tk.Frame(win, bg="#111820", height=36)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="MODEL vs MODEL", font=("Consolas", 11, "bold"),
                 bg="#111820", fg=SUGGEST).pack(side=tk.LEFT, padx=12)

        # Two-panel model selection
        panels = tk.Frame(win, bg=BG)
        panels.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        models = _discover_models()
        listboxes = {}
        match_model_keys = {}  # player -> {idx: key}

        def _populate_match_lb(lb, keys_dict):
            lb.delete(0, tk.END)
            keys_dict.clear()
            # Favorites first
            fav_items = [(n, p) for n, p in models if n in _favorites]
            rest_items = [(n, p) for n, p in models if n not in _favorites]
            for name, path in fav_items + rest_items:
                is_builtin = isinstance(path, str)
                is_fav = name in _favorites
                star = "*" if is_fav else " "
                if is_builtin:
                    lb.insert(tk.END, f" {star} [BOT] {name}")
                    lb.itemconfig(lb.size() - 1, fg=SUGGEST if not is_fav else "#e0c050")
                else:
                    is_legacy = "legacy" in str(path)
                    tag = "L" if is_legacy else "S"
                    lb.insert(tk.END, f" {star} [{tag}] {name}")
                    if is_fav:
                        lb.itemconfig(lb.size() - 1, fg="#e0c050")
                keys_dict[lb.size() - 1] = (name, path)
            if lb.size() > 0:
                lb.selection_set(0)

        for col, (player, color, label) in enumerate([
            (1, P1_TEXT, "PLAYER 1 (X)"), (2, P2_TEXT, "PLAYER 2 (O)")
        ]):
            frame = tk.Frame(panels, bg=BG)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            hdr_row = tk.Frame(frame, bg=BG)
            hdr_row.pack(fill=tk.X)
            tk.Label(hdr_row, text=label, font=("Consolas", 10, "bold"),
                     bg=BG, fg=color).pack(side=tk.LEFT)
            match_model_keys[player] = {}
            sb = tk.Scrollbar(frame)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            lb = tk.Listbox(frame, bg="#111820", fg=LIGHT, font=("Consolas", 9),
                            selectbackground="#1a3050", selectforeground="#ffffff",
                            borderwidth=0, highlightthickness=0,
                            yscrollcommand=sb.set, exportselection=False)
            lb.pack(fill=tk.BOTH, expand=True)
            sb.config(command=lb.yview)
            _populate_match_lb(lb, match_model_keys[player])
            listboxes[player] = lb

            def _make_fav_toggle(p=player):
                def toggle():
                    global _favorites
                    sel = listboxes[p].curselection()
                    if not sel:
                        return
                    entry = match_model_keys[p].get(sel[0])
                    if entry:
                        _favorites.symmetric_difference_update({entry[0]})
                        _save_favorites(_favorites)
                        for pp in listboxes:
                            _populate_match_lb(listboxes[pp], match_model_keys[pp])
                return toggle

            tk.Button(hdr_row, text="Fav", command=_make_fav_toggle(player),
                      font=("Consolas", 8), bg="#2a2a10", fg="#e0c050",
                      activebackground="#3a3a20", relief=tk.FLAT,
                      padx=4, pady=0).pack(side=tk.RIGHT)

        # Settings row
        settings = tk.Frame(win, bg=BG)
        settings.pack(fill=tk.X, padx=12, pady=4)

        tk.Label(settings, text="Sims:", font=("Consolas", 9),
                 bg=BG, fg=DIM).pack(side=tk.LEFT)
        sims_var = tk.StringVar(value="100")
        tk.Entry(settings, textvariable=sims_var, width=5, font=("Consolas", 9),
                 bg="#111820", fg=LIGHT, insertbackground=LIGHT,
                 borderwidth=1).pack(side=tk.LEFT, padx=4)

        tk.Label(settings, text="Games:", font=("Consolas", 9),
                 bg=BG, fg=DIM).pack(side=tk.LEFT, padx=(12, 0))
        games_var = tk.StringVar(value="5")
        tk.Entry(settings, textvariable=games_var, width=4, font=("Consolas", 9),
                 bg="#111820", fg=LIGHT, insertbackground=LIGHT,
                 borderwidth=1).pack(side=tk.LEFT, padx=4)

        tk.Label(settings, text="Delay(ms):", font=("Consolas", 9),
                 bg=BG, fg=DIM).pack(side=tk.LEFT, padx=(12, 0))
        delay_var = tk.StringVar(value="400")
        tk.Entry(settings, textvariable=delay_var, width=5, font=("Consolas", 9),
                 bg="#111820", fg=LIGHT, insertbackground=LIGHT,
                 borderwidth=1).pack(side=tk.LEFT, padx=4)

        info = tk.Label(win, text="", font=("Consolas", 9), bg=BG, fg=DIM)
        info.pack(fill=tk.X, padx=12)

        def start():
            sel1 = listboxes[1].curselection()
            sel2 = listboxes[2].curselection()
            if not sel1 or not sel2:
                info.config(text="Select a model for each player", fg="#ff6060")
                return
            name1, path1 = match_model_keys[1].get(sel1[0], models[sel1[0]])
            name2, path2 = match_model_keys[2].get(sel2[0], models[sel2[0]])
            try:
                n_sims = int(sims_var.get())
                n_games = int(games_var.get())
                delay = int(delay_var.get())
            except ValueError:
                info.config(text="Invalid number", fg="#ff6060")
                return

            info.config(text=f"Loading {name1}...", fg="#ffcc60")
            win.update()
            try:
                net1, arch1 = _load_model_any(path1, DEVICE)
            except Exception as e:
                info.config(text=f"Failed: {e}", fg="#ff6060")
                return
            info.config(text=f"Loading {name2}...", fg="#ffcc60")
            win.update()
            try:
                net2, arch2 = _load_model_any(path2, DEVICE)
            except Exception as e:
                info.config(text=f"Failed: {e}", fg="#ff6060")
                return

            win.destroy()
            self._start_match(net1, net2, name1, name2, n_sims, n_games, delay)

        btn_frame = tk.Frame(win, bg=BG)
        btn_frame.pack(fill=tk.X, padx=8, pady=(4, 8))
        tk.Button(btn_frame, text="Start Match", command=start,
                  font=("Consolas", 10, "bold"), bg="#2a2a10", fg=SUGGEST,
                  activebackground="#3a3a20", relief=tk.FLAT, padx=16,
                  pady=4).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Cancel", command=win.destroy,
                  font=("Consolas", 10), bg="#1a1a2a", fg="#8080c0",
                  activebackground="#2a2a3a", relief=tk.FLAT, padx=12,
                  pady=4).pack(side=tk.RIGHT)

    def _start_match(self, net1, net2, name1, name2, sims, n_games, delay):
        self.match_mode = True
        self.match_net = {1: net1, 2: net2}
        self.match_names = {1: name1, 2: name2}
        self.match_sims = sims
        self.match_delay = max(50, delay)
        self.match_paused = False
        self.match_games_total = n_games
        self.match_game_num = 0
        self.match_results = {1: 0, 2: 0, "draw": 0}
        self._match_new_game()

    def _match_new_game(self):
        self.match_game_num += 1
        self.game = HexGame()
        self.last_move = None
        self.highlights.clear()
        self.arrows.clear()
        self.pan_x = self.pan_y = 0
        self.eval_history = []

        # Alternate who plays P1/P2 each game
        if self.match_game_num % 2 == 0:
            self.match_net[1], self.match_net[2] = self.match_net[2], self.match_net[1]
            self.match_names[1], self.match_names[2] = self.match_names[2], self.match_names[1]

        self._update_match_status()
        self.draw()
        self.root.after(self.match_delay, self._match_next_move)

    def _match_next_move(self):
        if not self.match_mode:
            return
        if self.match_paused:
            self.root.after(100, self._match_next_move)
            return
        if self.game.winner is not None or len(self.game.move_history) >= 200:
            self.root.after(800, self._match_game_over)
            return

        cp = self.game.current_player
        net = self.match_net[cp]
        self._update_match_status(thinking=cp)
        self._start_thinking_anim()
        self.draw()

        def think():
            try:
                if getattr(net, '_is_eisenstein', False):
                    move = net.choose_move(self.game)
                else:
                    move = mcts_with_net(self.game, net, self.match_sims)
                self.root.after(0, self._stop_thinking_anim)
                self.root.after(0, lambda m=move: self._match_apply(m))
            except Exception as e:
                import traceback; traceback.print_exc()
                self.root.after(0, self._stop_thinking_anim)
                self.root.after(0, self._match_game_over)

        threading.Thread(target=think, daemon=True).start()

    def _match_apply(self, move):
        if not self.match_mode:
            return
        self.game.make(*move)
        self.last_move = move
        self._update_match_status()
        next_net = self.match_net[self.game.current_player]
        if not getattr(next_net, '_is_eisenstein', False):
            self._run_eval_with(next_net)
        self.draw()

        if self.game.winner is not None:
            self.root.after(1000, self._match_game_over)
        else:
            self.root.after(self.match_delay, self._match_next_move)

    def _match_game_over(self):
        if not self.match_mode:
            return
        w = self.game.winner
        if w is not None:
            self.match_results[w] += 1
        else:
            self.match_results["draw"] += 1
        self._update_match_status()
        self.draw()

        if self.match_game_num < self.match_games_total:
            self.root.after(1500, self._match_new_game)
        else:
            self.match_mode = False
            self._update_match_status()

    def _update_match_status(self, thinking=None):
        if not self.match_mode:
            self.status_lbl.config(text="Match complete", fg="#60ff60")
            r = self.match_results
            n1, n2 = self.match_names[1], self.match_names[2]
            self.eval_lbl.config(
                text=f"Final: {n1}={r[1]}  {n2}={r[2]}  draw={r['draw']}", fg=LIGHT)
            self.top_moves_lbl.config(text="")
            return

        gn = self.match_game_num
        gt = self.match_games_total
        r = self.match_results
        n1 = self.match_names[1].split("/")[-1].replace(".pt", "")
        n2 = self.match_names[2].split("/")[-1].replace(".pt", "")
        score = f"{n1}={r[1]} {n2}={r[2]} d={r['draw']}"

        if thinking:
            name = n1 if thinking == 1 else n2
            self.status_lbl.config(text=f"Game {gn}/{gt} -- {name} thinking...", fg="#ffcc60")
        elif self.match_paused:
            self.status_lbl.config(text=f"Game {gn}/{gt} -- PAUSED (space to resume)", fg="#ff9060")
        elif self.game.winner is not None:
            wname = n1 if self.game.winner == 1 else n2
            self.status_lbl.config(text=f"Game {gn}/{gt} -- {wname} wins!", fg="#60ff60")
        else:
            self.status_lbl.config(text=f"Game {gn}/{gt}", fg=LIGHT)

        self.eval_lbl.config(
            text=f"{score}  |  X={n1}  O={n2}  |  move {len(self.game.move_history)}",
            fg=DIM)

    def _toggle_match_pause(self):
        if not self.match_mode:
            return
        self.match_paused = not self.match_paused
        self._update_match_status()
        self.draw()

    def _stop_match(self):
        self.match_mode = False
        self.match_paused = False
        self._update_match_status()
        self.draw()

    def _run_eval_with(self, net):
        """Run eval with a specific net (for match mode)."""
        def _worker():
            try:
                v, top, own, oq, or_ = _evaluate_position(net, self.game)
                with self._eval_lock:
                    self.eval_value = v
                    self.eval_top_moves = top
                    self.eval_own_map = own
                    self.eval_origin = (oq, or_)
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play against HexGo bot")
    parser.add_argument("--model", type=str, default=None,
                        help="Checkpoint filename (default: net_latest.pt)")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--player", type=int, default=1, choices=[1, 2],
                        help="Play as player 1 (X) or 2 (O)")
    args = parser.parse_args()

    model_name = args.model or "net_latest.pt"
    path = CHECKPOINT_DIR / model_name
    if not path.exists():
        # Try legacy dir
        path = LEGACY_DIR / model_name
    if not path.exists():
        print(f"Checkpoint not found: {model_name}")
        sys.exit(1)

    net, arch = _load_checkpoint(path, DEVICE)
    print(f"Loaded {model_name} ({arch}, {param_count(net):,} params) on {DEVICE}")

    root = tk.Tk()
    HexPlayGUI(root, net, args.sims, args.player, model_name=model_name)
    root.mainloop()


if __name__ == "__main__":
    main()
