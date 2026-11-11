"""A 40-item blind check of the written set for human readers, and the agreement score once it is answered.

    uv run python -m bench.human_form make                    # data/cache/tr-decide/bench/human/ (form + key)
    uv run python -m bench.human_form score ANSWERS... [--out F]   # one file or folder per reader

An answers file has lines "NN: k" (1-based option) with an optional "| note", and may start with
"# ana dil: ...". A folder argument stands for every .md file in it. The sample is stratified by batch
(b1-b7) with a fixed seed and leans to short texts so the check takes about 20 minutes. With one reader the
score is the agreement with the gold and Krippendorff's alpha (nominal) between the reader and the gold; with
several it is each reader's agreement, the majority's agreement and alpha among the readers. Results go to
reports/human.json unless --out says otherwise.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bench.jury import ROOT, load

OUT = ROOT / "human"
N, SEED = 40, 0


def make() -> None:
    rows = [r for r in load(ROOT / "native.jsonl") if r["id"].split("-")[1] in {f"b{i}" for i in range(1, 8)}]
    by_batch = defaultdict(list)
    for r in rows:
        by_batch[r["id"].split("-")[1]].append(r)
    rng = random.Random(SEED)
    pick = []
    for idx, b in enumerate(sorted(by_batch)):
        pool = sorted(by_batch[b], key=lambda r: len(r["state"]))[: len(by_batch[b]) * 2 // 3]  # shorter two thirds
        pick += rng.sample(pool, N // len(by_batch) + (1 if idx < N % len(by_batch) else 0))
    rng.shuffle(pick)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# İnsan kontrolü: 40 soru",
             "",
             "Her soruda metni oku ve doğru seçeneğin numarasını yaz. Bir soru belirsizse, yanlış Türkçe içeriyorsa ya "
             "da iki seçenek de doğruysa `|` işaretinden sonra kısa bir not ekle. Cevapları dosyanın sonundaki "
             "listeye ya da sohbete şu biçimde yazabilirsin: `01: 2 | not`.", ""]
    key = []
    for n, r in enumerate(pick, 1):
        lines += [f"## {n:02d}", "", r["state"], "", f"**Soru:** {r['question']}", ""]
        lines += [f"{k}. {o}" for k, o in enumerate(r["options"], 1)]
        lines += [""]
        key.append({"n": n, "id": r["id"], "gold": r["gold"] + 1})
    lines += ["## Cevaplar", ""] + [f"{n:02d}: " for n in range(1, len(pick) + 1)]
    (OUT / "insan_kontrol_40.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "insan_kontrol_40_key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    print(f"{len(pick)} items, {sum(len(r['state'].split()) for r in pick)} words of text")


def alpha_nominal(pairs: list[tuple[int, int]]) -> float:
    """Krippendorff's alpha for two raters, nominal data, no missing values."""
    values = [v for p in pairs for v in p]
    n = len(values)
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1
    d_o = sum(a != b for a, b in pairs) * 2 / n
    d_e = sum(counts[c] * counts[k] for c in counts for k in counts if c != k) / (n * (n - 1))
    return 1.0 - d_o / d_e if d_e else 1.0


def alpha_units(units: list[list[int]]) -> float:
    """Krippendorff's alpha, nominal data, any number of values per unit; one-value units are not pairable."""
    coincidences: dict[tuple[int, int], float] = defaultdict(float)
    for values in units:
        if len(values) < 2:
            continue
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                if i != j:
                    coincidences[a, b] += 1 / (len(values) - 1)
    totals: dict[int, float] = defaultdict(float)
    for (a, _), weight in coincidences.items():
        totals[a] += weight
    n = sum(totals.values())
    d_o = sum(w for (a, b), w in coincidences.items() if a != b) / n
    d_e = sum(totals[c] * totals[k] for c in totals for k in totals if c != k) / (n * (n - 1))
    return 1.0 - d_o / d_e if d_e else 1.0


def read_answers(path: Path) -> tuple[dict[int, int], dict[int, str], str]:
    answers, notes, native = {}, {}, ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# ana dil:"):
            native = line.split(":", 1)[1].strip()
            continue
        m = re.match(r"\s*(\d+)\s*[:=]\s*(\d+)\s*(?:\|\s*(.*))?$", line)
        if m:
            answers[int(m.group(1))] = int(m.group(2))
            if m.group(3):
                notes[int(m.group(1))] = m.group(3).strip()
    return answers, notes, native


def expand(args: list[str]) -> list[Path]:
    paths = []
    for arg in args:
        p = Path(arg)
        paths += sorted(p.glob("*.md")) if p.is_dir() else [p]
    return paths


def majority(votes: list[int]) -> int | None:
    top = Counter(votes).most_common(2)
    if not top or (len(top) == 2 and top[0][1] == top[1][1]):
        return None
    return top[0][0]


def summarise(key: dict[int, dict], readers: list[tuple[dict[int, int], dict[int, str], str]]) -> dict:
    items = sorted(key)
    if len(readers) == 1:
        answers = readers[0][0]
        pairs = [(answers[n], key[n]["gold"]) for n in items if n in answers]
        return {"items": len(key), "readers": 1, "answered": len(pairs),
                "agree": sum(a == g for a, g in pairs), "alpha": alpha_nominal(pairs),
                "alpha_categories": "option position",
                "native": int(readers[0][2].lower().startswith("evet"))}
    votes = {n: [r[0][n] for r in readers if n in r[0]] for n in items}
    agree = [sum(r[0].get(n) == key[n]["gold"] for n in items) for r in readers]
    winners = {n: majority(votes[n]) for n in items}
    return {
        "items": len(key), "readers": len(readers),
        "native": sum(r[2].lower().startswith("evet") for r in readers),
        "answered": [len(r[0]) for r in readers], "agree_per_reader": agree,
        "mean_agree": sum(agree) / len(agree), "min_agree": min(agree), "max_agree": max(agree),
        "majority_agree": sum(winners[n] == key[n]["gold"] for n in items),
        "no_majority": sum(winners[n] is None for n in items),
        "alpha_readers": alpha_units([votes[n] for n in items]),
        "alpha_readers_and_gold": alpha_units([votes[n] + [key[n]["gold"]] for n in items]),
        "alpha_categories": "option position",
        "disputed": [{"n": n, "id": key[n]["id"], "gold": key[n]["gold"], "votes": votes[n],
                      "notes": [r[1][n] for r in readers if n in r[1]]}
                     for n in items if winners[n] != key[n]["gold"]],
    }


def score(args: list[str]) -> None:
    out = Path(__file__).resolve().parents[1] / "reports" / "human.json"
    if "--out" in args:
        i = args.index("--out")
        out, args = Path(args[i + 1]), args[:i] + args[i + 2:]
    key = {k["n"]: k for k in json.loads((OUT / "insan_kontrol_40_key.json").read_text(encoding="utf-8"))}
    readers = [read_answers(p) for p in expand(args)]
    report = summarise(key, readers)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if report["readers"] == 1:
        print(f"answered {report['answered']}/{len(key)}, agree with gold {report['agree']}, "
              f"alpha {report['alpha']:.3f}")
        answers, notes, _ = readers[0]
        for n in sorted(answers):
            if answers[n] != key[n]["gold"] or n in notes:
                print(f"  {n:02d} {key[n]['id']}: reader {answers[n]}, gold {key[n]['gold']} "
                      f"| {notes.get(n, '')}")
        return
    print(f"{report['readers']} readers ({report['native']} native), agree with gold per reader "
          f"{report['agree_per_reader']}, majority {report['majority_agree']}/{len(key)} "
          f"(no majority {report['no_majority']}), alpha among readers {report['alpha_readers']:.3f}, "
          f"with gold {report['alpha_readers_and_gold']:.3f}")
    for d in report["disputed"]:
        print(f"  {d['n']:02d} {d['id']}: votes {d['votes']}, gold {d['gold']} | {' / '.join(d['notes'])}")


if __name__ == "__main__":
    {"make": make, "score": lambda: score(sys.argv[2:])}[sys.argv[1]]()
