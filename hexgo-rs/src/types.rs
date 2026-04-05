/// Axial hex coordinate.
pub type Coord = (i16, i16);

/// Player 1 or 2.  0 = empty / no winner.
pub type Player = u8;

pub const P1: Player = 1;
pub const P2: Player = 2;
pub const NO_PLAYER: Player = 0;

pub const WIN_LENGTH: u32 = 6;
pub const PLACEMENT_RADIUS: i16 = 8;

/// Six axial neighbor offsets.
pub const DIRS: [Coord; 6] = [
    (1, 0), (0, 1), (1, -1),
    (-1, 0), (0, -1), (-1, 1),
];

/// The 3 unique axes for win checking.
pub const AXES: [Coord; 3] = [(1, 0), (0, 1), (1, -1)];

/// Hex distance in axial coordinates.
#[inline(always)]
pub fn hex_dist(a: Coord, b: Coord) -> i16 {
    let dq = (a.0 - b.0).abs();
    let dr = (a.1 - b.1).abs();
    let ds = ((a.0 + a.1) - (b.0 + b.1)).abs();
    (dq + dr + ds) / 2
}
