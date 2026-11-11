#!/usr/bin/env bash
# h200: d3 on vLLM 0.33.0's own Decision 3.0 endpoint (/v1/systemone), weights from models/d3.
# Usage: serve_d3.sh GPU CPUS PORT
set -euo pipefail
GPU=$1 CPUS=$2 PORT=$3
docker rm -f tr-decide-d3 > /dev/null 2>&1 || true
docker run -d --name tr-decide-d3 --gpus "\"device=$GPU\"" --cpuset-cpus "$CPUS" --ipc host \
  -v ~/work/tr-decide/models/d3:/model:ro -p 127.0.0.1:$PORT:8000 \
  vllm/vllm-openai:v0.33.0 --model /model --served-model-name d3 --max-model-len 8192 \
  --no-enable-prefix-caching --gpu-memory-utilization 0.90
