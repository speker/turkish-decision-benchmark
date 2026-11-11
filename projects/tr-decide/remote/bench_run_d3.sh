#!/usr/bin/env bash
# h200: d3 (vLLM's own endpoint) and Pusula (its package, adapter on) on a bench decision file, on one GPU.
# Also a harness check: the package with the adapter off must reproduce vLLM's d3 on the first 600 decisions.
# Usage: bench_run_d3.sh ITEMS TAG GPU CPUS PORT [WAIT_FOR]   (WAIT_FOR: a line to wait for in logs/bench-TAG.done)
set -uo pipefail
ITEMS=$1 TAG=$2 GPU=$3 CPUS=$4 PORT=$5 WAIT_FOR=${6:-}
cd ~/work/tr-decide
mkdir -p bench_out logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
done_() { echo "$1 exit $2" >> "logs/bench-$TAG.done"; }
if [ -n "$WAIT_FOR" ]; then
  until grep -q "^$WAIT_FOR" "logs/bench-$TAG.done" 2>/dev/null; do sleep 30; done
fi
bash serve_d3.sh "$GPU" "$CPUS" "$PORT"
up=0
for _ in $(seq 1 240); do curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && { up=1; break; }; sleep 10; done
if [ $up = 1 ]; then
  taskset -c "$CPUS" .venv/bin/python run_systemone_http.py --url "http://127.0.0.1:$PORT" --no-jev-fields --workers 32 \
    --name "d3-$TAG" --items "$ITEMS" --outdir bench_out > "logs/bench-$TAG-d3.log" 2>&1; done_ d3 $?
else
  docker logs --tail 50 tr-decide-d3 > "logs/bench-$TAG-d3.log" 2>&1; done_ d3 server-failed
fi
docker rm -f tr-decide-d3 > /dev/null
CUDA_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" .venv-pusula/bin/python run_pusula.py --name "pusula-$TAG" \
  --items "$ITEMS" --outdir bench_out > "logs/bench-$TAG-pusula.log" 2>&1; done_ pusula $?
CUDA_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" .venv-pusula/bin/python run_pusula.py --own-base --limit 600 \
  --name "d3pkg-$TAG-check" --items "$ITEMS" --outdir bench_out > "logs/bench-$TAG-d3pkg.log" 2>&1; done_ d3pkg-check $?
echo D3_DONE >> "logs/bench-$TAG.done"
