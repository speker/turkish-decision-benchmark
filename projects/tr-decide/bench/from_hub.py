"""Rebuild the scorer's inputs from the published dataset, so the leaderboard follows from public files.

    uv run python -m bench.from_hub          # serkanpeker/turkish-decision-benchmark (gated: HF token)
    uv run python -m bench.from_hub --source projects/tr-decide/hub      # a local copy of the dataset folder
    uv run python -m bench.to_items prog; uv run python -m bench.to_items native
    uv run python -m bench.score prog --results data/cache/tr-decide/bench/hub_results
    uv run python -m bench.score native --results data/cache/tr-decide/bench/hub_results

Writes prog.jsonl and native.jsonl in the item format of bench/prog.py and the jury pipeline, and copies every
results/<model>/<date>/{programmatic,written}.jsonl to hub_results/<model>-{prog,native}.jsonl.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from bench.jury import ROOT

REPO = "serkanpeker/turkish-decision-benchmark"
PROG_FIELDS = ("id", "family", "variant", "pair_id", "type", "state", "question", "options")
NATIVE_FIELDS = ("id", "domain", "style", "phenomena", "type", "difficulty", "state", "question", "options",
                 "option_descriptions", "rationale", "review")


def rows(df: pd.DataFrame, fields: tuple[str, ...]) -> list[dict]:
    out = []
    for r in df.to_dict(orient="records"):
        row = {f: (list(r[f]) if hasattr(r[f], "tolist") else r[f]) for f in fields}
        row["gold"] = int(r["answer_index"])
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="local dataset folder; default: download the Hub repo")
    a = ap.parse_args()
    src = Path(a.source) if a.source else Path(snapshot_download(REPO, repo_type="dataset"))
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, out, fields in (("programmatic", "prog", PROG_FIELDS), ("written", "native", NATIVE_FIELDS)):
        df = pd.read_parquet(src / "data" / name / "test-00000-of-00001.parquet")
        items = rows(df, fields)
        (ROOT / f"{out}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items),
                                          encoding="utf-8")
        print(f"{out}.jsonl: {len(items)} items")
    dst = ROOT / "hub_results"
    dst.mkdir(exist_ok=True)
    for f in sorted((src / "results").glob("*/*/*.jsonl")):
        set_name = {"programmatic": "prog", "written": "native"}[f.stem]
        shutil.copyfile(f, dst / f"{f.parts[-3]}-{set_name}.jsonl")
    print(f"results: {len(list(dst.glob('*.jsonl')))} files -> {dst}")


if __name__ == "__main__":
    main()
