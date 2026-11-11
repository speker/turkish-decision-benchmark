#!/usr/bin/env bash
# h200: every competitor and teacher on a bench decision file, three GPU lanes, results in bench_out/.
# Usage: bench_run.sh ITEMS TAG     (e.g. bench_prog_items.jsonl prog)
set -uo pipefail
ITEMS=$1 TAG=$2
cd ~/work/tr-decide
mkdir -p bench_out logs
export PYTHONUNBUFFERED=1 USE_TF=0 OMP_NUM_THREADS=8
T=~/work/tr-decide/models/teachers
done_() { echo "$1 exit $2" >> "logs/bench-$TAG.done"; }
# up to 60 minutes: under heavy machine load the large models' kernel warm-up alone has taken over 30
wait_up() { for _ in $(seq 1 360); do curl -s -o /dev/null "http://127.0.0.1:$1/v1/models" && return 0; sleep 10; done; return 1; }

(  # lane A: GPU 2
  for m in jev-9b jev-27b; do
    CUDA_VISIBLE_DEVICES=2 taskset -c 96-119 .venv/bin/python run_jev.py --model "models/$m" --name "$m-$TAG" \
      --items "$ITEMS" --outdir bench_out --max-batch-tokens 16384 > "logs/bench-$TAG-$m.log" 2>&1; done_ "$m" $?
  done
  # the card: reads up to 8,192 tokens, ships with a 1,024-token limit; short inputs answer the same with 8192
  CUDA_VISIBLE_DEVICES=2 taskset -c 96-119 .venv/bin/python run_laya.py --model models/laya-multilingual --max-len 8192 \
    --name "laya-multilingual-$TAG" --items "$ITEMS" --outdir bench_out > "logs/bench-$TAG-laya.log" 2>&1; done_ laya-multilingual $?
) &
(  # lane B: GPU 3
  bash serve_35b.sh 3 120-143 8410
  if wait_up 8410; then
    taskset -c 120-143 .venv/bin/python run_decide_http.py --url http://127.0.0.1:8410 --name "jev-35b-$TAG" \
      --items "$ITEMS" --outdir bench_out > "logs/bench-$TAG-jev-35b.log" 2>&1; done_ jev-35b $?
  else done_ jev-35b server-failed; fi
  docker rm -f tr-decide-jev35b > /dev/null
  bash serve_teacher.sh qwen3.8-27b ~/models/qwen3.8-27b 3 1 120-143 8500
  if wait_up 8500; then
    taskset -c 120-143 .venv/bin/python teacher_probe.py --url http://127.0.0.1:8500 --model-dir ~/models/qwen3.8-27b \
      --name "qwen3.8-27b-$TAG" --items "$ITEMS" --outdir bench_out --chat-kwarg enable_thinking=false \
      > "logs/bench-$TAG-qwen3.8-27b.log" 2>&1; done_ qwen3.8-27b $?
  else done_ qwen3.8-27b server-failed; fi
  docker rm -f tr-decide-qwen3.8-27b > /dev/null
) &
(  # lane C: GPU 4,5
  bash serve_teacher.sh qwen3.5-122b "$T/qwen3.5-122b-fp8" 4,5 2 144-191 8501 --max-num-seqs 128
  if wait_up 8501; then
    taskset -c 144-191 .venv/bin/python teacher_probe.py --url http://127.0.0.1:8501 --model-dir "$T/qwen3.5-122b-fp8" \
      --name "qwen3.5-122b-$TAG" --items "$ITEMS" --outdir bench_out --chat-kwarg enable_thinking=false \
      > "logs/bench-$TAG-qwen3.5-122b.log" 2>&1; done_ qwen3.5-122b $?
  else done_ qwen3.5-122b server-failed; fi
  docker rm -f tr-decide-qwen3.5-122b > /dev/null
  bash serve_teacher.sh deepseek-v4-flash "$T/deepseek-v4-flash" 4,5 2 144-191 8502 --max-num-seqs 128
  if wait_up 8502; then
    taskset -c 144-191 .venv/bin/python teacher_probe.py --url http://127.0.0.1:8502 --model-dir "$T/deepseek-v4-flash" \
      --name "deepseek-v4-flash-$TAG" --items "$ITEMS" --outdir bench_out --encoding dsv4 \
      > "logs/bench-$TAG-deepseek.log" 2>&1; done_ deepseek-v4-flash $?
  else done_ deepseek-v4-flash server-failed; fi
  docker rm -f tr-decide-deepseek-v4-flash > /dev/null
  bash serve_karar.sh 4 144-167 8420 8421 8422 8423
  bash serve_karar.sh 5 168-191 8424 8425 8426 8427
  for p in $(seq 8420 8427); do for _ in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$p/docs" && break; sleep 5; done; done
  taskset -c 144-191 .venv/bin/python run_systemone_http.py --url "$(seq -s, -f 'http://127.0.0.1:%g' 8420 8427)" \
    --workers 24 --name "karar-4b-$TAG" --items "$ITEMS" --outdir bench_out > "logs/bench-$TAG-karar.log" 2>&1; done_ karar-4b $?
  for f in logs/serve-karar-*.pid; do kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
) &
wait
if grep -qE 'exit [1-9]|server-failed' "logs/bench-$TAG.done"; then echo ALL_DONE_WITH_FAILURES >> "logs/bench-$TAG.done"; exit 1; fi
echo ALL_DONE >> "logs/bench-$TAG.done"
