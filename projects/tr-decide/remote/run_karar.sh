#!/usr/bin/env bash
# h200: Karar-4B tam koşusu; 12 sunucu kopyası serve_karar.sh ile 8420–8431'de açık olmalı.
cd ~/work/tr-decide
URLS=$(seq -s, -f "http://127.0.0.1:%g" 8420 8431)
taskset -c 120-143 .venv/bin/python run_systemone_http.py --url "$URLS" --workers 36 --name karar-4b > logs/run-karar-4b.log 2>&1
echo "karar-4b exit $?" >> logs/run.done
