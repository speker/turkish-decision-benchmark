#!/usr/bin/env bash
# h200: P9 Python ortamı. Sürücü 570 (CUDA 12.8): torch yalnız cu128 dizininden gelir.
# Tek çözümleme. uv'de --extra-index-url önce aranır; first-index ile torch cu128 dizininden gelir.
set -euo pipefail
cd ~/work/tr-decide
UV=~/.local/bin/uv
export UV_HTTP_TIMEOUT=600
[ -x .venv/bin/python ] || $UV venv -p 3.12 .venv
$UV pip install -p .venv/bin/python \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy first-index --reinstall-package torch \
  "torch<2.12" "transformers>=5.16" "peft>=0.21" flash-linear-attention "huggingface_hub>=1.0" \
  safetensors datasets numpy pandas pyarrow "laya==0.4.30" > logs/setup.log 2>&1
.venv/bin/python -c "import torch, transformers, peft, laya, fla; assert torch.version.cuda.startswith('12.'), torch.version.cuda; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__, 'fla', fla.__version__)" >> logs/setup.log 2>&1
echo ENV_DONE >> logs/download.done
