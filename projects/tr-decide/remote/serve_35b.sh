#!/usr/bin/env bash
# h200: JEV-35B'yi kendi serve_decide.py'siyle vLLM 0.33.0 imajında aç (kartın serve.sh'ındaki bayraklar,
# bağlam 8192). Kullanım: serve_35b.sh GPU CPUS PORT
set -euo pipefail
GPU=$1 CPUS=$2 PORT=$3
docker rm -f tr-decide-jev35b > /dev/null 2>&1 || true
docker run -d --name tr-decide-jev35b --gpus "device=$GPU" --cpuset-cpus "$CPUS" --ipc host \
  -v ~/work/tr-decide/models/jev-35b:/m:ro -p 127.0.0.1:$PORT:8000 \
  -e JEV_DECIDE_MERGED_HEAD=/m/adapter_vllm/decision_head.json -e JEV_DECIDE_SYSTEM2=0 \
  --entrypoint python3 vllm/vllm-openai:v0.33.0 /m/serve_decide.py --model /m --served-model-name autotrust/JEV-35B \
  --logprobs-mode processed_logprobs --max-logprobs 256 --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --mamba-cache-mode align --max-num-seqs 64 --trust-request-chat-template --port 8000
