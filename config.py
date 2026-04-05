# config.py — tunable hyperparameters for HexGo autotune
# Edit this file to propose a new trial config.
# Imported by train.py and mcts.py at startup.

CFG = {
    "LR": 0.001,
    "WEIGHT_DECAY": 0.0001,
    "BATCH_SIZE": 64,
    "SIMS": 100,
    "SIMS_MIN": 25,
    "CAP_FULL_FRAC": 0,
    "CPUCT": 2.0,               # research target 2.0–2.5; pairs with 400 sims
    "DIRICHLET_ALPHA": 0.10,    # ~10/|ZoI|; less noise needed with deeper search
    "DIRICHLET_EPS": 0.25,      # reduced — 400 sims provides enough exploration
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
    "VALUE_LOSS_WEIGHT": 1.0,
    "ENTROPY_REG": 0.01,
    "AUX_LOSS_OWN": 0.1,
    "AUX_LOSS_THREAT": 0.1,
    "UNC_LOSS_WEIGHT": 0.05,    # conservative for fresh start
}
