"""Turn bench items into harness decisions (the format of items.jsonl): two option orderings per item; natively
written items are also asked with "label: description" options.

    uv run python -m bench.to_items prog      # data/cache/tr-decide/bench/prog.jsonl   -> bench_prog_items.jsonl
    uv run python -m bench.to_items native    # data/cache/tr-decide/bench/native.jsonl -> bench_native_items.jsonl
"""

from __future__ import annotations

import json
import random
import sys

from bench.jury import ROOT


def orderings(n: int, seed: str) -> list[list[int]]:
    p1 = list(range(n))
    random.Random(seed).shuffle(p1)
    if p1 == list(range(n)) and n > 1:
        p1 = p1[1:] + p1[:1]
    return [list(range(n)), p1]


def decisions(it: dict, conds: dict[str, list[str]], meta: dict) -> list[dict]:
    out = []
    for perm, order in enumerate(orderings(len(it["options"]), f"{it['id']}:p1")):
        for cond, opts in conds.items():
            if cond.endswith("desc") and perm == 1:
                continue  # description condition is asked in the original ordering only
            out.append({"uid": f"{it['id']}:{cond}:p{perm}", "item": it["id"], "cond": cond, "perm": perm,
                        "k": len(opts), "state": it["state"], "question": it["question"],
                        "options": [opts[i] for i in order], "labels": [it["options"][i] for i in order],
                        "descriptions": [it.get("option_descriptions", [""] * len(opts))[i] for i in order],
                        "gold": order.index(it["gold"]), **meta})
    return out


def main() -> None:
    which = sys.argv[1]
    rows = [json.loads(l) for l in (ROOT / f"{which}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for it in rows:
        if which == "prog":
            meta = {"set": "prog", "family": it["family"], "variant": it["variant"], "pair_id": it["pair_id"],
                    "type": it["type"]}
            out += decisions(it, {"tr": it["options"]}, meta)
        else:
            desc = [f"{o}: {d}" for o, d in zip(it["options"], it["option_descriptions"])]
            meta = {"set": "native", "family": it["domain"], "variant": it["style"], "pair_id": it["id"],
                    "type": it["type"], "difficulty": it["difficulty"], "phenomena": it["phenomena"]}
            out += decisions(it, {"tr": it["options"], "tr_desc": desc}, meta)
    path = ROOT / f"bench_{which}_items.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"{path.name}: {len(out)} decisions from {len(rows)} items")


if __name__ == "__main__":
    main()
