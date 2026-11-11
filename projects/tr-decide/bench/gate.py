"""Quality-gate reports for the Hub folder: exact and MinHash near-duplicates, and n-gram overlap with public sets.

    uv run python -m bench.gate        # writes projects/tr-decide/reports/{dedup,decontam}.json

Near-duplicates: MinHash over word 5-gram shingles (128 hash functions); pairs with estimated Jaccard >= 0.6 are listed
with whether they share the question and differ in gold (a deliberate minimal pair) or not (a redundant copy).
Overlap: a bench item counts as overlapping when it shares at least one word 8-gram with a reference set (MASSIVE 1.1
tr-TR, SIB-200 and Belebele tur_Latn, and the public Turkish Decision Benchmark of erenextlabs at a fixed revision).
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from bench.build_hub import HUB, PROJECT

RAW = PROJECT.parents[1] / "data" / "cache" / "tr-decide" / "raw"
PUSULA_BENCH = ("erenextlabs/turkish-decision-benchmark", "0662f853443b3053d1606c30a6422873d7e4ba4d")
NUM_PERM, SHINGLE, NGRAM, NEAR = 128, 5, 8, 0.6


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.replace("I", "ı").replace("İ", "i").lower())


def grams(text: str, n: int) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def minhash(shingles: set[tuple[str, ...]]) -> np.ndarray:
    sig = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    for s in shingles:
        key = " ".join(s).encode()
        h = np.frombuffer(b"".join(hashlib.blake2b(key, digest_size=8, salt=i.to_bytes(8, "little")).digest()
                                   for i in range(NUM_PERM)), dtype=np.uint64)
        sig = np.minimum(sig, h)
    return sig


def bench() -> dict[str, pd.DataFrame]:
    return {n: pd.read_parquet(HUB / "data" / n / "test-00000-of-00001.parquet") for n in ("programmatic", "written")}


def dedup(sets: dict[str, pd.DataFrame]) -> dict:
    report = {}
    norm = {n: df["state"].map(lambda s: " ".join(words(s))) for n, df in sets.items()}
    for n, df in sets.items():
        items = norm[n] + "\n" + df["question"].map(lambda q: " ".join(words(q)))
        report[n] = {"rows": len(df), "exact_duplicate_items": int(len(df) - items.nunique()),
                     "repeated_states": int(len(df) - norm[n].nunique())}
    report["exact_duplicates_across_sets"] = len(set(norm["programmatic"]) & set(norm["written"]))
    w = sets["written"].reset_index(drop=True)
    sigs = np.stack([minhash(grams(s, SHINGLE)) for s in w["state"]])
    pairs = []
    for i in range(len(w)):
        est = (sigs[i + 1:] == sigs[i]).mean(axis=1)
        for j in np.nonzero(est >= NEAR)[0] + i + 1:
            a, b = w.iloc[i], w.iloc[j]
            minimal = a["question"] == b["question"] and a["answer_text"] != b["answer_text"]
            pairs.append({"a": a["id"], "b": b["id"], "jaccard_est": round(float(est[j - i - 1]), 3),
                          "kind": "minimal pair (same question, different gold)" if minimal else "check"})
    report["written_near_duplicates"] = {"method": f"MinHash, {NUM_PERM} hashes, word {SHINGLE}-gram shingles, "
                                                   f"estimated Jaccard >= {NEAR}",
                                         "pairs": pairs,
                                         "redundant": sum(p["kind"] == "check" for p in pairs)}
    return report


def references() -> dict[str, list[str]]:
    refs = defaultdict(list)
    with tarfile.open(RAW / "amazon-massive-dataset-1.1.tar.gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("tr-TR.jsonl"))
        for line in tar.extractfile(member).read().decode("utf-8").splitlines():
            refs["MASSIVE 1.1 tr-TR (all splits)"].append(json.loads(line)["utt"])
    for split in ("dev", "test"):
        df = pd.read_csv(RAW / f"sib200-tur_Latn-{split}.tsv", sep="\t")
        refs["SIB-200 tur_Latn (dev, test)"] += df["text"].astype(str).tolist()
    for line in (RAW / "belebele-tur_Latn.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        refs["Belebele tur_Latn"] += [r["flores_passage"], r["question"]]
    repo, rev = PUSULA_BENCH
    path = hf_hub_download(repo, "tr_decision_bench_v1.jsonl", repo_type="dataset", revision=rev)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        refs[f"{repo}@{rev[:7]}"].append(str(r.get("text") or r.get("message") or r.get("state") or ""))
    return refs


def decontam(sets: dict[str, pd.DataFrame]) -> dict:
    refs = references()
    report = {"method": f"shared word {NGRAM}-gram with a reference text", "references": {}}
    for ref, texts in refs.items():
        ref_grams = set().union(*(grams(t, NGRAM) for t in texts))
        row = {"texts": len(texts)}
        for n, df in sets.items():
            hits = [i for i, s in zip(df["id"], df["state"]) if grams(s, NGRAM) & ref_grams]
            row[n] = {"items_with_overlap": len(hits), "examples": hits[:5]}
        report["references"][ref] = row
    return report


def main() -> None:
    sets = bench()
    out = PROJECT / "reports"
    out.mkdir(exist_ok=True)
    d = dedup(sets)
    (out / "dedup.json").write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("dedup:", {k: v for k, v in d.items() if k != "written_near_duplicates"},
          "| written near-duplicate pairs:", len(d["written_near_duplicates"]["pairs"]),
          "redundant:", d["written_near_duplicates"]["redundant"])
    c = decontam(sets)
    (out / "decontam.json").write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for ref, row in c["references"].items():
        print(f"decontam {ref}: {row['texts']} texts | programmatic {row['programmatic']['items_with_overlap']}"
              f" | written {row['written']['items_with_overlap']}")


if __name__ == "__main__":
    main()
