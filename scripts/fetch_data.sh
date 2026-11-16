#!/usr/bin/env bash
# Download OurAirports data (airports, runways, frequencies) and the Piper voice.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data voices
for f in airports runways airport-frequencies; do
  curl -fsSL -o "data/$f.csv" "https://davidmegginson.github.io/ourairports-data/$f.csv"
done
B=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US
for v in libritts_r/medium/en_US-libritts_r-medium; do
  n=$(basename "$v")
  [ -f "voices/$n.onnx" ] || curl -fsSL -o "voices/$n.onnx" "$B/$v.onnx"
  curl -fsSL -o "voices/$n.onnx.json" "$B/$v.onnx.json"
done
echo "data and voices ready"
