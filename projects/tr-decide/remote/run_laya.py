"""h200: typed `choice` decisions with a Laya checkpoint through the `laya` package (one call per decision).

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python run_laya.py --model models/laya-multilingual --name laya-multilingual
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import laya
import torch
import transformers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")][: a.limit]
    agent = laya.load(a.model, device="cuda")
    meta = {"model": a.name, "path": a.model, "laya": laya.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0), "max_len": a.max_len}

    t0 = time.time()
    with (outdir / f"{a.name}.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            opts = it["options"]
            if len(set(opts)) != len(opts):
                raise ValueError(f"{it['uid']}: duplicate option texts")
            r = agent.predict(it["state"], {"q": {"type": "choice", "instructions": it["question"], "criteria": opts}},
                              max_len=a.max_len)
            probs = r["answers"]["q"]["probabilities"]
            f.write(json.dumps({"uid": it["uid"], "p": [probs[o] for o in opts]}) + "\n")
    meta.update({"decisions": len(items), "seconds": round(time.time() - t0, 1)})
    (outdir / f"{a.name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
