#!/usr/bin/env bash
# h200: Pusula's own package (code readout and the d3 prompt contract) in a separate venv, pinned to one commit.
# Driver 570 (CUDA 12.8): torch only from the cu128 index (same recipe as setup_env.sh).
set -euo pipefail
cd ~/work/tr-decide
UV=~/.local/bin/uv
REV=771140836ba8a63e102bf9b514958c89209d7d4a
export UV_HTTP_TIMEOUT=600
[ -x .venv-pusula/bin/python ] || $UV venv -p 3.12 .venv-pusula
$UV pip install -p .venv-pusula/bin/python \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy first-index --reinstall-package torch \
  "torch<2.12" torchvision "flash-linear-attention==0.5.2" \
  "pusula[d3] @ git+https://github.com/erenextlabs/pusula@$REV" > logs/setup-pusula.log 2>&1
.venv-pusula/bin/python -c "import torch, transformers, pusula; assert torch.version.cuda.startswith('12.'), torch.version.cuda; print('torch', torch.__version__, torch.version.cuda, 'transformers', transformers.__version__, 'pusula', getattr(pusula, '__version__', '?'), '$REV')" >> logs/setup-pusula.log 2>&1
echo PUSULA_ENV_DONE >> logs/setup-pusula.log
