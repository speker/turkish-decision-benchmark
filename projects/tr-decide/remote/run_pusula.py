"""h200: Pusula 2.0 (d3 + its Turkish LoRA, through the pinned `pusula` package) on bench decisions.

The package sends questions that are not its built-in templates to the base d3 with the adapter off (`own_base`, its
default). Every bench question is such a question, so this runner turns `own_base` off and answers every question with
the adapter; `--own-base` gives the package's default behaviour, which is d3's own answer.

    .venv-pusula/bin/python run_pusula.py --items bench_prog_items.jsonl --name pusula-prog
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pusula.inference import D3Engine


def question(it: dict) -> tuple[list[str], dict]:
    if it.get("cond", "").endswith("desc"):
        labels, criteria = it["labels"], dict(zip(it["labels"], it["descriptions"]))
    else:
        labels, criteria = it["options"], {o: None for o in it["options"]}
    return labels, {"type": "choice", "instructions": it["question"], "criteria": criteria}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3", default="models/d3")
    ap.add_argument("--adapter", default="models/pusula")
    ap.add_argument("--own-base", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--name", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--outdir", default="bench_out")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    engine = D3Engine.load(a.d3, adapter=a.adapter, own_base=a.own_base)
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")][: a.limit]
    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    t0 = time.time()
    with (outdir / f"{a.name}.jsonl").open("w", encoding="utf-8") as f:
        for s in range(0, len(items), a.chunk):
            chunk = items[s: s + a.chunk]
            qs = [question(it) for it in chunk]
            res = engine.decide_many([(it["state"], {"answer": q}) for it, (_, q) in zip(chunk, qs)])
            for it, (labels, _), r in zip(chunk, qs, res):
                probs = r["answer"]["probabilities"]
                f.write(json.dumps({"uid": it["uid"], "p": [round(float(probs[o]), 6) for o in labels]}) + "\n")
            print(f"{s + len(chunk)}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    meta = {"model": a.name, "package": "pusula 2.0.0 @ 771140836ba8", "own_base": a.own_base,
            "engine": engine.model_name, "decisions": len(items), "seconds": round(time.time() - t0, 1)}
    (outdir / f"{a.name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
