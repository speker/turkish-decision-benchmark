"""h200: typed `choice` decisions through a running `serve_decide.py` (POST /v1/decide), for JEV-35B.

    .venv/bin/python run_decide_http.py --url http://127.0.0.1:8410 --name jev-35b
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CARD_EXAMPLE = {"kind": "noul", "state": "Customer says the parcel arrived damaged and wants their money back.",
                "question": "Is the customer asking for a refund?"}


def post(url: str, body: dict, tries: int = 5) -> dict:
    data = json.dumps(body).encode()
    for n in range(tries):
        try:
            req = urllib.request.Request(f"{url}/v1/decide", data=data, headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except Exception:
            if n == tries - 1:
                raise
            time.sleep(2 ** n)
    raise AssertionError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")][: a.limit]
    card = post(a.url, CARD_EXAMPLE)
    meta = {"model": a.name, "card_example_p_true": card["probabilities"][1], "workers": a.workers,
            "server_model": card.get("model"), "protocol": card.get("protocol")}
    print("card example P(true) =", meta["card_example_p_true"], flush=True)

    def one(it: dict) -> dict:
        r = post(a.url, {"kind": "choice", "state": it["state"], "question": it["question"],
                         "options": it["options"], "strategy": "single"})
        assert r["options"] == it["options"], it["uid"]
        return {"uid": it["uid"], "p": [round(x, 6) for x in r["probabilities"]]}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, (outdir / f"{a.name}.jsonl").open("w", encoding="utf-8") as f:
        for row in ex.map(one, items):
            f.write(json.dumps(row) + "\n")
    meta.update({"decisions": len(items), "seconds": round(time.time() - t0, 1)})
    (outdir / f"{a.name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
