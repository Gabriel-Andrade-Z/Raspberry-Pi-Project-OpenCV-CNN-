#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[INFO] Python: $(python3 --version)"
echo "[INFO] Rodando vision_test.py ..."

python3 stream_face_detect.py

