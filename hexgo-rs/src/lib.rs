use pyo3::prelude::*;

mod types;
mod game;
mod node;
mod mcts;
mod parallel;
mod encode;
mod batched;

/// HexGo Rust engine — drop-in replacement for game.py + mcts.py.
///
/// Usage from Python:
///   from hexgo import HexGame, mcts, mcts_with_net
///   game = HexGame()
///   game.make(0, 0)
///   move = mcts(game, num_sims=200)
#[pymodule]
fn hexgo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Game engine
    m.add_class::<game::HexGame>()?;

    // MCTS functions
    m.add_function(wrap_pyfunction!(mcts::mcts, m)?)?;
    m.add_function(wrap_pyfunction!(mcts::mcts_with_net, m)?)?;
    m.add_function(wrap_pyfunction!(mcts::self_play_game, m)?)?;

    // Parallel self-play
    m.add_class::<parallel::GameResult>()?;
    m.add_function(wrap_pyfunction!(parallel::parallel_self_play, m)?)?;
    m.add_function(wrap_pyfunction!(parallel::parallel_eisenstein_games, m)?)?;

    // Batched net-guided self-play
    m.add_class::<batched::GameTrainingResult>()?;
    m.add_class::<batched::PositionData>()?;
    m.add_function(wrap_pyfunction!(batched::batched_self_play, m)?)?;

    // Constants
    m.add("WIN_LENGTH", types::WIN_LENGTH)?;
    m.add("PLACEMENT_RADIUS", types::PLACEMENT_RADIUS)?;
    m.add("P1", types::P1)?;
    m.add("P2", types::P2)?;

    Ok(())
}
