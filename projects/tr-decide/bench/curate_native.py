"""Deterministic clean-ups of the natively written batches, applied before the jury's second round.

    uv run python -m bench.curate_native b1 [b2 ...]

Rule 1: a question must not refer to the option descriptions, since models are also asked with labels only
("Bu talep, ekip açıklamalarına göre hangi ekibe ..." -> "Bu talep hangi ekibe ..."). The gold does not change.

The first run keeps the writer's file as batch_<b>.orig.jsonl; every change is appended to native/curation_log.jsonl.
"""

from __future__ import annotations

import json
import re
import sys

from bench.jury import NATIVE, load, write

RULES = [("no-description-reference", re.compile(r",?\s*(ekip|seçenek|birim) açıklamalarına göre,?", re.I), "")]


def curate(batch: str) -> int:
    path, orig = NATIVE / f"batch_{batch}.jsonl", NATIVE / f"batch_{batch}.orig.jsonl"
    if not orig.exists():
        orig.write_bytes(path.read_bytes())
    items, log = load(orig), []
    for it in items:
        for name, pattern, repl in RULES:
            new = re.sub(r"\s{2,}", " ", pattern.sub(repl, it["question"])).strip()
            if new != it["question"]:
                log.append({"id": it["id"], "rule": name, "field": "question", "before": it["question"], "after": new})
                it["question"] = new
    write(path, items)
    with (NATIVE / "curation_log.jsonl").open("a", encoding="utf-8") as f:
        for row in log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(log)


if __name__ == "__main__":
    for b in sys.argv[1:]:
        print(b, curate(b), "changes")
