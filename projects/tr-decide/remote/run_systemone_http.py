"""h200: typed `choice` decisions through a Jev-style API (POST /v1/systemone): Karar-4B's jevalt server, or d3 on
vLLM's own endpoint (`--no-jev-fields`).
Options are sent as labels without descriptions, as for every other model in this measurement; decisions whose
`cond` ends in "desc" send each label's description as its criterion.
`--url` takes a comma-separated list of server replicas; decision i goes to replica i mod n.

    .venv/bin/python run_systemone_http.py --url http://127.0.0.1:8420,http://127.0.0.1:8421 --name karar-4b
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(url: str, body: dict, tries: int = 5) -> dict:
    data = json.dumps(body).encode()
    for n in range(tries):
        try:
            req = urllib.request.Request(f"{url}/v1/systemone", data=data, headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jev-fields", action=argparse.BooleanOptionalAction, default=True,
                    help="send jevalt's reasoning/abstain fields; --no-jev-fields for vLLM's d3 endpoint")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")][: a.limit]
    urls = a.url.split(",")

    def one(job: tuple[int, dict]) -> dict:
        n, it = job
        if it.get("cond", "").endswith("desc"):
            labels, criteria = it["labels"], dict(zip(it["labels"], it["descriptions"]))
        else:
            labels, criteria = it["options"], {o: "" if a.jev_fields else None for o in it["options"]}
        body = {"state": it["state"], "questions": {"answer": {"type": "choice", "instructions": it["question"],
                                                               "criteria": criteria}}}
        if a.jev_fields:
            body |= {"reasoning": "off", "abstain": False}
        r = post(urls[n % len(urls)], body)
        probs = r["answers"]["answer"]["probabilities"]
        return {"uid": it["uid"], "p": [round(probs[o], 6) for o in labels]}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, (outdir / f"{a.name}.jsonl").open("w", encoding="utf-8") as f:
        for row in ex.map(one, enumerate(items)):
            f.write(json.dumps(row) + "\n")
    meta = {"model": a.name, "api": "/v1/systemone", "replicas": len(urls), "workers": a.workers, "decisions": len(items),
            "seconds": round(time.time() - t0, 1)}
    (outdir / f"{a.name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
