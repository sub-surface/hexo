#!/usr/bin/env bash
# Usage: bash run.sh <script.py> [args...]
PY="/c/Program Files/Python312/python.exe"
cd "$(dirname "$0")"
"$PY" "$@"
