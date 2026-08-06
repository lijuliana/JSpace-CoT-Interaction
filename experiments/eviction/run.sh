#!/usr/bin/env bash
# eviction: teacher-forced write vs omit, probe decodability after the write point
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL="${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
M=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
python src/harness/eviction_probe.py --model "$MODEL" --depth 12 --n 400 \
  --out "results/raw/eviction_${M}_d12.jsonl"
python src/analysis/eviction_readout.py "results/raw/eviction_${M}_d12.jsonl"
