# Building hexgo-rs

## Prerequisites

1. Install Rust: https://rustup.rs/
2. Install maturin: `pip install maturin`

## Build & Install

```bash
cd hexgo-rs
maturin develop --release
```

This builds the Rust crate and installs it as a Python package in the current
environment. After this, `import hexgo` works from Python.

## Verify

```bash
# Run parity tests (Python vs Rust)
cd hexgo-rs
python -m pytest tests/test_parity.py -v

# Quick smoke test
python -c "from hexgo import HexGame; g = HexGame(); g.make(0,0); print(g)"
```

## Integration

Swap one import in the Python code to use the Rust game engine:

```python
# Before:
from game import HexGame

# After:
from hexgo import HexGame
```

Everything else stays the same — the API is identical.

## Rebuild after changes

```bash
maturin develop --release
```

The `--release` flag is important — debug builds are ~10x slower.
