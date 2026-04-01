# config.py — tunable hyperparameters for HexGo autotune
# Edit this file to propose a new trial config.
# Imported by train.py and mcts.py at startup.

CFG = {
    "LR": 0.001,
    "WEIGHT_DECAY": 0.0001,
    "BATCH_SIZE": 32,
    "SIMS": 31,
    "SIMS_MIN": 16,
    "CAP_FULL_FRAC": 0,
    "CPUCT": 1.5,
    "DIRICHLET_ALPHA": 0.09,
    "DIRICHLET_EPS": 0.25,
    "ZOI_MARGIN": 5,
    "ZOI_LOOKBACK": 16,
    "GUMBEL_SELECTION": True,
    "TD_GAMMA": 0.99,
    "TEMP_HORIZON": 40,
    "WEIGHT_SYNC_BATCHES": 20,
    "RECENCY_WEIGHT": 0.75,
    "TRUNK_BLOCKS": 6,
    "TRUNK_CHANNELS": 128,
    "WEIGHT_INIT": 'ca',
    "VALUE_LOSS_WEIGHT": 1,
    "ENTROPY_REG": 0.01,
    "AUX_LOSS_OWN": 0.1,
    "AUX_LOSS_THREAT": 0.1,
    "UNC_LOSS_WEIGHT": 0,
}
