use crate::types::*;

/// Index into the arena. u32 supports ~4 billion nodes.
pub type NodeIdx = u32;
pub const NULL_NODE: NodeIdx = u32::MAX;

/// MCTS tree node. Arena-allocated — no Rc/Arc, no GC pressure.
pub struct Node {
    pub mov: Coord,
    pub parent: NodeIdx,
    pub children_start: u32,
    pub children_count: u16,
    pub visits: u32,
    pub value: f32,
    pub prior: f32,
    pub player: Player,
}

/// Arena allocator for MCTS nodes. All nodes for one search tree live here.
pub struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    pub fn with_capacity(cap: usize) -> Self {
        Arena {
            nodes: Vec::with_capacity(cap),
        }
    }

    pub fn alloc(&mut self, node: Node) -> NodeIdx {
        let idx = self.nodes.len() as NodeIdx;
        self.nodes.push(node);
        idx
    }

    #[inline(always)]
    pub fn get(&self, idx: NodeIdx) -> &Node {
        &self.nodes[idx as usize]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, idx: NodeIdx) -> &mut Node {
        &mut self.nodes[idx as usize]
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Select best child via PUCT + FPU reduction.
    pub fn best_child(&self, parent_idx: NodeIdx, c_puct: f32, fpu_reduction: f32) -> NodeIdx {
        let parent = self.get(parent_idx);
        let n = parent.visits.max(1) as f32;
        let cpuct_sqrt = c_puct * n.sqrt();
        let fpu_q = parent.value / n - fpu_reduction;

        let start = parent.children_start as usize;
        let end = start + parent.children_count as usize;

        let mut best_idx = NULL_NODE;
        let mut best_score = f32::NEG_INFINITY;

        for i in start..end {
            let child = &self.nodes[i];
            let score = if child.visits == 0 {
                fpu_q + cpuct_sqrt * child.prior
            } else {
                let q = child.value / child.visits as f32;
                q + cpuct_sqrt * child.prior / (1.0 + child.visits as f32)
            };
            if score > best_score {
                best_score = score;
                best_idx = i as NodeIdx;
            }
        }
        best_idx
    }

    /// Expand a leaf node with given moves and priors.
    pub fn expand(
        &mut self,
        node_idx: NodeIdx,
        moves: &[Coord],
        priors: &[f32],
        player: Player,
    ) {
        let start = self.nodes.len() as u32;
        for (i, &m) in moves.iter().enumerate() {
            self.nodes.push(Node {
                mov: m,
                parent: node_idx,
                children_start: 0,
                children_count: 0,
                visits: 0,
                value: 0.0,
                prior: priors[i],
                player,
            });
        }
        let node = &mut self.nodes[node_idx as usize];
        node.children_start = start;
        node.children_count = moves.len() as u16;
    }

    /// Backpropagate value from leaf to root.
    /// Sign flips only when parent.player != child.player,
    /// correctly handling the 1-2-2 turn rule.
    pub fn backprop(&mut self, mut idx: NodeIdx, mut value: f32) {
        while idx != NULL_NODE {
            let node = self.get_mut(idx);
            node.visits += 1;
            node.value += value;
            let parent_idx = node.parent;
            if parent_idx != NULL_NODE {
                let parent_player = self.get(parent_idx).player;
                let node_player = self.get(idx).player;
                if parent_player != node_player {
                    value = -value;
                }
            }
            idx = parent_idx;
        }
    }

    /// Return the most-visited child's move.
    pub fn best_move(&self, root: NodeIdx) -> Coord {
        let node = self.get(root);
        let start = node.children_start as usize;
        let end = start + node.children_count as usize;

        let mut best_visits = 0u32;
        let mut best_coord = (0i16, 0i16);
        for i in start..end {
            let child = &self.nodes[i];
            if child.visits > best_visits {
                best_visits = child.visits;
                best_coord = child.mov;
            }
        }
        best_coord
    }

    /// Return visit counts for all children of a node (for policy targets).
    pub fn child_visits(&self, node_idx: NodeIdx) -> Vec<(Coord, u32)> {
        let node = self.get(node_idx);
        let start = node.children_start as usize;
        let end = start + node.children_count as usize;
        (start..end)
            .map(|i| {
                let c = &self.nodes[i];
                (c.mov, c.visits)
            })
            .collect()
    }
}
