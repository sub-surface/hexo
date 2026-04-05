"""
MCTS for self-play — optimised for speed.

Two modes:
  mcts(game, sims)              — pure rollout (no net)
  mcts_with_net(game, net, sims) — AlphaZero style: net value replaces rollout,
                                   net policy logits set node priors,
                                   Dirichlet noise added to root for exploration
"""

import math
import random
import numpy as np
try:
    from hexgo import HexGame
except ImportError:
    from game import HexGame

try:
    from config import CFG
    C_PUCT = CFG["CPUCT"]
except ImportError:
    C_PUCT = 1.4


class Node:
    __slots__ = ("move", "parent", "children", "visits", "value", "prior", "player")

    def __init__(self, move=None, parent=None, prior: float = 1.0, player: int = 1):
        self.move = move        # (q, r) that led here; None for root
        self.parent: "Node | None" = parent
        self.children: list["Node"] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.prior: float = prior
        self.player: int = player

    def ucb(self) -> float:
        if self.visits == 0:
            return float("inf")
        q = self.value / self.visits
        u = C_PUCT * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u

    def best_child(self) -> "Node":
        return max(self.children, key=lambda c: c.ucb())

    def best_move(self) -> tuple[int, int]:
        return max(self.children, key=lambda c: c.visits).move


def _rollout(game: HexGame) -> float:
    """
    Random playout in-place using make/unmake.
    Returns +1 if the player-to-move at rollout start wins, -1 if they lose.
    """
    start_player = game.current_player
    depth = 0
    max_moves = 150

    while game.winner is None and depth < max_moves:
        moves = game.legal_moves()
        if not moves:
            break
        game.make(*random.choice(moves))
        depth += 1

    result = 0.0 if game.winner is None else (1.0 if game.winner == start_player else -1.0)

    # Unwind
    for _ in range(depth):
        game.unmake()

    return result


def _expand(node: Node, game: HexGame):
    moves = game.legal_moves()
    prior = 1.0 / max(len(moves), 1)
    # Each child node reflects the player who is about to move from THAT state
    node.children = [Node(move=m, parent=node, prior=prior, player=game.current_player)
                     for m in moves]


def _backprop(node: Node, value: float):
    """
    Backpropagate value upward through the tree.

    Convention: `value` is from the perspective of the player who is to move
    AT the node passed in (i.e. positive means that player is winning).

    Sign flip: negate whenever the parent's player differs from the child's
    player. This correctly handles the 1-2-2 turn rule where the same player
    makes two consecutive placements without a sign flip between them.
    """
    while node is not None:
        node.visits += 1
        node.value += value
        if node.parent and node.parent.player != node.player:
            value = -value
        node = node.parent


def mcts(game: HexGame, num_simulations: int = 200) -> tuple[int, int]:
    """
    Run MCTS on `game` (mutates and restores it via make/unmake).
    Returns the best move as (q, r).
    """
    root = Node(player=game.current_player)
    _expand(root, game)

    for _ in range(num_simulations):
        node = root
        depth = 0

        # Selection — walk down tree, making moves on the shared game
        while node.children and game.winner is None:
            node = node.best_child()
            game.make(*node.move)
            depth += 1

        # Expansion + simulation
        if game.winner is None:
            _expand(node, game)
            if node.children:
                node = random.choice(node.children)
                game.make(*node.move)
                depth += 1

        if game.winner is not None:
            # node.player is the player who just moved (and won).
            # Convention: v is from node.player's perspective → +1.0 = winner.
            v = 1.0
        else:
            # _rollout returns +1 if game.current_player (the NEXT player) wins.
            # Convention needs v from node.player's perspective → negate.
            v = -_rollout(game)

        # Restore game state
        for _ in range(depth):
            game.unmake()

        _backprop(node, v)

    return root.best_move()


_AXES = ((1, 0), (0, 1), (1, -1))

def _chain_score(board, q, r, player):
    """Max chain length if `player` places at (q,r), across all 3 hex axes."""
    best = 1
    for dq, dr in _AXES:
        count = 1
        for sign in (1, -1):
            nq, nr = q + sign * dq, r + sign * dr
            while board.get((nq, nr)) == player:
                count += 1
                nq += sign * dq
                nr += sign * dr
        best = max(best, count)
    return best


def _top_k_filter(moves, logits, k):
    """Keep only top-k moves by logit value. Returns (filtered_moves, filtered_logits)."""
    if len(moves) <= k:
        return moves, logits
    top_idx = np.argsort(logits)[-k:]
    return [moves[i] for i in top_idx], logits[top_idx]


def mcts_with_net(game: HexGame, net, num_simulations: int = 100,
                  dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.0,
                  top_k: int = 16, policy_temp: float = 1.0,
                  cpuct_override: float = 0.0, move_temp: float = 0.0,
                  proximity_bias: float = 0.0,
                  chain_bonus: float = 0.0,
                  ) -> tuple[int, int]:
    """
    AlphaZero-style MCTS using HexNet for value + policy priors.
    No rollout — net value is used directly at leaf nodes.
    Dirichlet noise added to root priors for exploration.
    top_k limits expansion to the K best policy moves (critical for large action spaces).
    policy_temp < 1.0 sharpens priors (play mode); > 1.0 flattens (exploration).
    cpuct_override > 0 overrides the global C_PUCT for this search.
    """
    from net import evaluate   # late import to avoid circular at module level

    saved_cpuct = None
    if cpuct_override > 0:
        global C_PUCT
        saved_cpuct = C_PUCT
        C_PUCT = cpuct_override

    try:
        return _mcts_with_net_inner(
            game, net, num_simulations, dirichlet_alpha, dirichlet_eps,
            top_k, policy_temp, move_temp, proximity_bias, chain_bonus,
            evaluate)
    finally:
        if saved_cpuct is not None:
            C_PUCT = saved_cpuct


def _mcts_with_net_inner(game, net, num_simulations, dirichlet_alpha,
                         dirichlet_eps, top_k, policy_temp, move_temp,
                         proximity_bias, chain_bonus, evaluate):
    root = Node(player=game.current_player)

    # Expand root with net priors
    value, policy = evaluate(net, game)
    moves = game.legal_moves()
    if not moves:
        raise RuntimeError("mcts_with_net called on terminal/empty game")

    # Filter to top-K by raw policy logits (before softmax/noise)
    logits = np.array([policy.get(m, 0.0) for m in moves], dtype=np.float32)

    # Proximity bias: penalize moves far from recent action center
    if proximity_bias > 0 and game.move_history:
        recent = game.move_history[-8:] if len(game.move_history) >= 8 else game.move_history
        cq = sum(q for q, r in recent) / len(recent)
        cr = sum(r for q, r in recent) / len(recent)
        for i, (mq, mr) in enumerate(moves):
            dq, dr = mq - cq, mr - cr
            dist = max(abs(dq), abs(dr), abs(dq + dr))  # hex distance
            logits[i] -= proximity_bias * dist

    # Chain bonus: boost moves that extend own chains or block opponent threats
    if chain_bonus > 0 and game.board:
        me = game.current_player
        opp = 3 - me
        for i, (mq, mr) in enumerate(moves):
            own = _chain_score(game.board, mq, mr, me)
            block = _chain_score(game.board, mq, mr, opp)
            # Exponential scaling: chain of 5 (one from win) gets huge bonus
            logits[i] += chain_bonus * max(own, block) ** 1.5

    moves, logits = _top_k_filter(moves, logits, top_k)

    # Sharpen/flatten logits with temperature before softmax
    if policy_temp != 1.0 and policy_temp > 0:
        logits = logits / policy_temp

    # Softmax priors from logits
    logits -= logits.max()
    priors = np.exp(logits)
    priors /= priors.sum()

    # Dirichlet noise at root
    if dirichlet_eps > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
        priors = (1 - dirichlet_eps) * priors + dirichlet_eps * noise

    root.children = [Node(move=m, parent=root, prior=float(p),
                          player=game.current_player)
                     for m, p in zip(moves, priors)]

    for _ in range(num_simulations):
        node = root
        depth = 0

        # Selection
        while node.children and game.winner is None:
            node = node.best_child()
            game.make(*node.move)
            depth += 1

        # Leaf evaluation
        if game.winner is not None:
            # node.player is the player who just moved (and won).
            # _backprop convention: value is from node.player's perspective.
            v = 1.0 if game.winner == node.player else -1.0
        else:
            # Expand with net. evaluate() returns value from game.current_player's
            # perspective; negate if that differs from node.player.
            v, leaf_policy = evaluate(net, game)
            if node.player != game.current_player:
                v = -v   # evaluate() returns from game.current_player's POV; align to node.player
            leaf_moves = game.legal_moves()
            if leaf_moves:
                llogits = np.array([leaf_policy.get(m, 0.0) for m in leaf_moves], dtype=np.float32)
                leaf_moves, llogits = _top_k_filter(leaf_moves, llogits, top_k)
                if policy_temp != 1.0 and policy_temp > 0:
                    llogits = llogits / policy_temp
                llogits -= llogits.max()
                lpriors = np.exp(llogits)
                lpriors /= lpriors.sum()
                node.children = [Node(move=m, parent=node, prior=float(p),
                                      player=game.current_player)
                                  for m, p in zip(leaf_moves, lpriors)]

        for _ in range(depth):
            game.unmake()

        _backprop(node, v)

    if move_temp <= 0 or not root.children:
        return root.best_move()
    # Temperature-based move selection: sample from visit distribution
    visits = np.array([c.visits for c in root.children], dtype=np.float32)
    if visits.sum() == 0:
        return root.best_move()
    vt = visits ** (1.0 / move_temp)
    probs = vt / vt.sum()
    idx = np.random.choice(len(root.children), p=probs)
    return root.children[idx].move


def self_play_game(num_simulations: int = 100, callback=None) -> dict:
    """
    Play a complete game via MCTS self-play.
    callback(game, move) is called after each move if provided — use for UI updates.
    Returns {winner, num_moves, moves}.
    """
    game = HexGame()
    moves = []

    while game.winner is None:
        if not game.legal_moves():
            break
        move = mcts(game, num_simulations)
        game.make(*move)
        moves.append(move)
        if callback:
            callback(game, move)

    return {"winner": game.winner, "num_moves": len(moves), "moves": moves, "game": game}
