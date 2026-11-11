#!/usr/bin/env bash
# h200: bir öğretmen adayını vLLM 0.33.0 ile aç (sınama için; ölçüm değil, prefix caching kapalı).
# Kullanım: serve_teacher.sh NAME MODEL_DIR GPUS TP CPUS PORT [ek vllm bayrakları...]
set -euo pipefail
NAME=$1 MODEL=$2 GPUS=$3 TP=$4 CPUS=$5 PORT=$6; shift 6
docker rm -f "tr-decide-$NAME" > /dev/null 2>&1 || true
docker run -d --name "tr-decide-$NAME" --gpus "\"device=$GPUS\"" --cpuset-cpus "$CPUS" --ipc host \
  -v "$MODEL":/model:ro -p 127.0.0.1:$PORT:8000 \
  vllm/vllm-openai:v0.33.0 --model /model --served-model-name "$NAME" --tensor-parallel-size "$TP" \
  --logprobs-mode processed_logprobs --max-logprobs 32 --max-model-len 4096 --no-enable-prefix-caching \
  --gpu-memory-utilization 0.90 "$@"
