#!/usr/bin/env bash
# h200: Karar-4B'yi kendi jevalt sunucusuyla (transformers arka ucu) aç. Kullanım: serve_karar.sh GPU CPUS PORT...
# jevalt istekleri tek tek işlediği için GPU başına birkaç kopya açılır.
# jevalt calibration.json'ı yalnız Hub kimliğiyle indiriyor (yerel yolda sıcaklık 1'de kalır); Hub'daki main
# 9 Kasım'da d299ed5, açılıştan önce kontrol edilir.
set -euo pipefail
GPU=$1 CPUS=$2; shift 2
cd ~/work/tr-decide
.venv/bin/python -c "from huggingface_hub import HfApi; s=HfApi().model_info('mertkayacs/Karar-4B').sha; assert s.startswith('d299ed5'), s"
for PORT in "$@"; do
  CUDA_VISIBLE_DEVICES=$GPU USE_TF=0 OMP_NUM_THREADS=4 nohup taskset -c "$CPUS" .venv-jevalt/bin/jevalt serve --backend hf \
    --model mertkayacs/Karar-4B --device cuda --port "$PORT" > "logs/serve-karar-$PORT.log" 2>&1 &
  echo $! > "logs/serve-karar-$PORT.pid"
done
