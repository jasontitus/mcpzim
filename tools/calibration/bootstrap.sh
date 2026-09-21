#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
date -u > results/started.txt
nvidia-smi > results/nvidia-smi.txt
free -b > results/host-memory.txt
df -B1 . > results/disk.txt
# Environment preparation belongs in the prebuilt image, before GPU rental.
# Missing packages fail immediately; never install/download/compile here.
PYTHON="${ZIMFO_PYTHON:-/opt/zimfo/bin/python}"
"$PYTHON" - <<'PY'
import pytest, torch
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available(), 'CUDA unavailable'
PY
"$PYTHON" -m pip freeze > results/python-packages.txt
PYTHONPATH=. "$PYTHON" -m pytest tests -q --junitxml=results/tests.xml
PYTHONPATH=. "$PYTHON" cuda_preflight.py --device cuda --profile target --output results/cuda.json
date -u > results/completed.txt
