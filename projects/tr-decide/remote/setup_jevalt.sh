#!/usr/bin/env bash
# h200: Karar-4B'yi kendi sunucusuyla (jevalt, transformers arka ucu) ölçmek için ayrı ortam.
# Sürücü 570: torch cu128 dizininden (setup_env.sh ile aynı yöntem).
set -euo pipefail
cd ~/work/tr-decide
UV=~/.local/bin/uv
export UV_HTTP_TIMEOUT=600
[ -d src/jevalt ] || git clone -q https://github.com/mertkayacs/jevalt src/jevalt
git -C src/jevalt rev-parse HEAD > logs/jevalt.rev
[ -x .venv-jevalt/bin/python ] || $UV venv -p 3.12 .venv-jevalt
$UV pip install -p .venv-jevalt/bin/python \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy first-index "torch<2.12" "./src/jevalt[hf,serve]" "huggingface_hub>=1.0" > logs/setup-jevalt.log 2>&1
.venv-jevalt/bin/python -c "import torch; assert torch.version.cuda.startswith('12.'), torch.version.cuda; print('torch', torch.__version__, torch.cuda.is_available())" >> logs/setup-jevalt.log 2>&1
.venv/bin/hf download mertkayacs/Karar-4B --revision d299ed5884bc3049500bd83d3f7d4cd53f2f6f9c \
  --local-dir models/karar-4b --exclude "assets/*" > logs/download-karar-4b.log 2>&1
echo "mertkayacs/Karar-4B d299ed5884bc3049500bd83d3f7d4cd53f2f6f9c ok" >> logs/download.done
echo JEVALT_DONE >> logs/download.done
