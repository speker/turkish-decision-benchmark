#!/usr/bin/env bash
# h200: eski tek kopyalı Karar-4B koşusunu ve sunucusunu durdur, 12 kopya aç, tam koşuyu başlat.
cd ~/work/tr-decide
pkill -f "[r]un_systemone_http.py" || true
[ -f logs/serve-karar.pid ] && kill "$(cat logs/serve-karar.pid)" 2>/dev/null; rm -f logs/serve-karar.pid
sleep 5
bash serve_karar.sh 2 96-119 8420 8421 8422 8423
bash serve_karar.sh 4 144-167 8424 8425 8426 8427
bash serve_karar.sh 5 168-191 8428 8429 8430 8431
for p in $(seq 8420 8431); do
  for _ in $(seq 1 60); do
    curl -s -o /dev/null "http://127.0.0.1:$p/docs" && break
    sleep 5
  done
done
echo "replicas up: $(for p in $(seq 8420 8431); do curl -s -o /dev/null -w '%{http_code} ' http://127.0.0.1:$p/docs; done)"
setsid nohup bash run_karar.sh < /dev/null > /dev/null 2>&1 &
