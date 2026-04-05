#!/usr/bin/env bash
# Always use Python 3.12 for training (needs CUDA torch)
PY="/c/Users/landa/AppData/Local/Programs/Python/Python312/python.exe"
cd "$(dirname "$0")"
"$PY" train.py "$@" 