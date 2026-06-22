#!/usr/bin/env bash
# Always use Python 3.12 for training (needs CUDA torch)
PY="/c/Program Files/Python312/python.exe"
cd "$(dirname "$0")"
"$PY" train.py "$@"
