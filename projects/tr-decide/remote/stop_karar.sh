#!/usr/bin/env bash
# h200: Karar-4B sunucu kopyalarını durdur.
cd ~/work/tr-decide
for f in logs/serve-karar-*.pid; do
  [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null
  rm -f "$f"
done
sleep 5
pgrep -af "[j]evalt serve" || echo "karar replicas stopped"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
