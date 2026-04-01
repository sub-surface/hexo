"""
play.py — GUI to play against a HexGo checkpoint.

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

import torch
from game import HexGame
from net import HexNet, DEVICE
from mcts import mcts_with_net

CHECKPOINT_DIR = Path("checkpoints")

# Colors
BG       = "#0a0e14"
EMPTY    = "#111820"
BORDER   = "#1c2a3a"
P1_FILL  = "#8b2020"
P1_EDGE  = "#e04040"
P1_TEXT  = "#ff7070"
P2_FILL  = "#103060"
P2_EDGE  = "#3080e0"
P2_TEXT  = "#70a0ff"
LAST_EDGE = "#ffffff"
LEGAL_HOVER = "#1a2a1a"
TEXT_DIM = "#586374"
TEXT_LIGHT = "#c9d1d9"


class HexPlayGUI:
    def __init__(self, root, net, sims, human_player):
        self.root = root
        self.net = net
        self.sims = sims
        self.human_player = human_player
        self.bot_player = 3 - human_player
        self.game = HexGame()
        self.last_move = None
        self.bot_thinking = False
        self.hover_cell = None
        self.undo_stack = []  # number of moves per "turn" for undo

        # Pan/zoom state
        self.hex_size = 18
        self.pan_x = 0
        self.pan_y = 0
        self._drag_start = None

        root.title("HexGo — Play")
        root.configure(bg=BG)
        root.geometry("900x700")

        # Top bar
        top = tk.Frame(root, bg="#111820", height=40)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        self.status_label = tk.Label(top, text="Your turn", font=("Courier New", 11),
                                      bg="#111820", fg=TEXT_LIGHT)
        self.status_label.pack(side=tk.LEFT, padx=12)

        self.info_label = tk.Label(top, text="", font=("Courier New", 10),
                                    bg="#111820", fg=TEXT_DIM)
        self.info_label.pack(side=tk.LEFT, padx=8)

        btn_frame = tk.Frame(top, bg="#111820")
        btn_frame.pack(side=tk.RIGHT, padx=8)

        tk.Button(btn_frame, text="Undo", command=self.undo, font=("Courier New", 9),
                  bg="#1a1a2a", fg="#8080c0", relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="New Game", command=self.new_game, font=("Courier New", 9),
                  bg="#1a2a1a", fg="#60a060", relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Quit", command=root.quit, font=("Courier New", 9),
                  bg="#2a1a1a", fg="#c06060", relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=2)

        # Canvas
        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<MouseWheel>", self.on_scroll)
        self.canvas.bind("<ButtonPress-2>", self.on_drag_start)
        self.canvas.bind("<B2-Motion>", self.on_drag)
        self.canvas.bind("<ButtonPress-3>", self.on_drag_start)
        self.canvas.bind("<B3-Motion>", self.on_drag)
        self.canvas.bind("<Configure>", lambda e: self.draw())

        self.update_status()

        # If bot goes first
        if self.game.current_player == self.bot_player:
            self.root.after(100, self.bot_turn)

    def axial_to_pixel(self, q, r):
        sz = self.hex_size
        cx = self.canvas.winfo_width() / 2 + self.pan_x
        cy = self.canvas.winfo_height() / 2 + self.pan_y
        x = cx + sz * 1.5 * q
        y = cy + sz * math.sqrt(3) * (r + q / 2)
        return x, y

    def pixel_to_axial(self, px, py):
        sz = self.hex_size
        cx = self.canvas.winfo_width() / 2 + self.pan_x
        cy = self.canvas.winfo_height() / 2 + self.pan_y
        x = px - cx
        y = py - cy
        q = x / (1.5 * sz)
        r = (y / (sz * math.sqrt(3))) - q / 2
        # Round to nearest hex
        rq, rr = round(q), round(r)
        rs = round(-q - r)
        dq = abs(rq - q)
        dr = abs(rr - r)
        ds = abs(rs - (-q - r))
        if dq > dr and dq > ds:
            rq = -rr - rs
        elif dr > ds:
            rr = -rq - rs
        return int(rq), int(rr)

    def hex_corners(self, cx, cy, sz):
        corners = []
        for i in range(6):
            a = math.pi / 3 * i
            corners.append((cx + (sz - 1) * math.cos(a), cy + (sz - 1) * math.sin(a)))
        return corners

    def draw(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10:
            return

        # Collect all cells to draw: all legal moves + board pieces
        from game import PLACEMENT_RADIUS, _cells_within_radius
        cells = set()
        if not self.game.board:
            cells.add((0, 0))
        else:
            # Show all cells within placement radius of any piece
            for pq, pr in self.game.board:
                for cq, cr in _cells_within_radius(pq, pr, PLACEMENT_RADIUS):
                    cells.add((cq, cr))

        legal = set(self.game.legal_moves()) if self.game.winner is None else set()
        is_human_turn = (self.game.current_player == self.human_player
                         and not self.bot_thinking and self.game.winner is None)

        for q, r in cells:
            px, py = self.axial_to_pixel(q, r)
            # Cull off-screen
            if px < -50 or px > w + 50 or py < -50 or py > h + 50:
                continue

            corners = self.hex_corners(px, py, self.hex_size)
            p = self.game.board.get((q, r))
            is_last = self.last_move == (q, r)
            is_legal = (q, r) in legal
            is_hover = self.hover_cell == (q, r) and is_human_turn and is_legal

            if p == 1:
                fill, edge = P1_FILL, P1_EDGE
            elif p == 2:
                fill, edge = P2_FILL, P2_EDGE
            else:
                fill = LEGAL_HOVER if is_hover else EMPTY
                edge = "#2a3a2a" if is_hover else BORDER

            if is_last:
                edge = LAST_EDGE

            c.create_polygon(corners, fill=fill, outline=edge,
                             width=2 if is_last else 1)

            if p == 1:
                c.create_text(px, py, text="X", fill=P1_TEXT,
                              font=("Courier New", max(8, int(self.hex_size * 0.5)), "bold"))
            elif p == 2:
                c.create_text(px, py, text="O", fill=P2_TEXT,
                              font=("Courier New", max(8, int(self.hex_size * 0.5)), "bold"))
            elif is_legal and is_human_turn:
                # Show subtle dot for legal moves
                c.create_oval(px-2, py-2, px+2, py+2, fill="#2a3a2a", outline="")

    def update_status(self):
        if self.game.winner is not None:
            if self.game.winner == self.human_player:
                self.status_label.config(text="You win!", fg="#60ff60")
            else:
                self.status_label.config(text="Bot wins!", fg="#ff6060")
        elif self.bot_thinking:
            self.status_label.config(text="Bot thinking...", fg="#ffcc60")
        elif self.game.current_player == self.human_player:
            is_first = len(self.game.move_history) == 0
            remaining = (1 if is_first else 2) - self.game.placements_in_turn
            p_str = "X" if self.human_player == 1 else "O"
            self.status_label.config(
                text=f"Your turn ({p_str}) — {remaining} placement{'s' if remaining > 1 else ''}",
                fg=TEXT_LIGHT)
        else:
            self.status_label.config(text="Bot's turn", fg=TEXT_DIM)

        moves = len(self.game.move_history)
        self.info_label.config(text=f"Move {moves}  |  Sims: {self.sims}")

    def on_click(self, event):
        if self.bot_thinking or self.game.winner is not None:
            return
        if self.game.current_player != self.human_player:
            return

        q, r = self.pixel_to_axial(event.x, event.y)
        legal = set(self.game.legal_moves())
        if (q, r) not in legal:
            return

        self.game.make(q, r)
        self.last_move = (q, r)
        self.draw()
        self.update_status()

        if self.game.winner is not None:
            return

        # Check if human has more placements this turn
        if self.game.current_player == self.human_player:
            return  # still human's turn (second placement)

        # Bot's turn
        self.undo_stack.append(len(self.game.move_history))
        self.root.after(50, self.bot_turn)

    def bot_turn(self):
        if self.game.winner is not None:
            return
        self.bot_thinking = True
        self.update_status()
        self.draw()

        def think():
            moves_made = 0
            while (self.game.current_player == self.bot_player
                   and self.game.winner is None):
                legal = self.game.legal_moves()
                if not legal:
                    break
                move = mcts_with_net(self.game, self.net, self.sims)
                self.game.make(*move)
                self.last_move = move
                moves_made += 1

            self.bot_thinking = False
            self.undo_stack.append(len(self.game.move_history))
            self.root.after(0, self.draw)
            self.root.after(0, self.update_status)

        threading.Thread(target=think, daemon=True).start()

    def undo(self):
        if self.bot_thinking or len(self.game.move_history) < 2:
            return
        # Undo back to start of human's last turn
        # Undo bot moves + human moves
        target = self.undo_stack[-2] if len(self.undo_stack) >= 2 else 0
        while len(self.game.move_history) > target:
            self.game.unmake()
        if len(self.undo_stack) >= 2:
            self.undo_stack.pop()
            self.undo_stack.pop()
        self.last_move = self.game.move_history[-1] if self.game.move_history else None
        self.draw()
        self.update_status()

    def new_game(self):
        if self.bot_thinking:
            return
        self.game = HexGame()
        self.last_move = None
        self.undo_stack = []
        self.pan_x = 0
        self.pan_y = 0
        self.draw()
        self.update_status()
        if self.game.current_player == self.bot_player:
            self.root.after(100, self.bot_turn)

    def on_motion(self, event):
        q, r = self.pixel_to_axial(event.x, event.y)
        new_hover = (q, r)
        if new_hover != self.hover_cell:
            self.hover_cell = new_hover
            self.draw()

    def on_scroll(self, event):
        if event.delta > 0:
            self.hex_size = min(60, self.hex_size + 2)
        else:
            self.hex_size = max(10, self.hex_size - 2)
        self.draw()

    def on_drag_start(self, event):
        self._drag_start = (event.x, event.y, self.pan_x, self.pan_y)

    def on_drag(self, event):
        if self._drag_start:
            sx, sy, spx, spy = self._drag_start
            self.pan_x = spx + (event.x - sx)
            self.pan_y = spy + (event.y - sy)
            self.draw()


def main():
    parser = argparse.ArgumentParser(description="Play against HexGo bot (GUI)")
    parser.add_argument("--model", type=str, default=None,
                        help="Checkpoint filename (default: net_latest.pt)")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS sims per move (higher=stronger, slower)")
    parser.add_argument("--player", type=int, default=1, choices=[1, 2],
                        help="Play as player 1 (X, first) or 2 (O, second)")
    args = parser.parse_args()

    path = CHECKPOINT_DIR / (args.model or "net_latest.pt")
    if not path.exists():
        print(f"Checkpoint not found: {path}")
        sys.exit(1)

    net = HexNet().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    print(f"Loaded {path.name} on {DEVICE}")

    root = tk.Tk()
    HexPlayGUI(root, net, args.sims, args.player)
    root.mainloop()


if __name__ == "__main__":
    main()
