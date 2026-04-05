#!/usr/bin/env bash
# Usage: bash run.sh <script.py> [args...]
PY="/c/Users/landa/AppData/Local/Programs/Python/Python312/python.exe"
cd "$(dirname "$0")"
"$PY" "$@"
