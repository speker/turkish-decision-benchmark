"""Leaderboard for the bench: accuracy, ECE and Brier per family and per group, order flips, the damage done by
planted instructions and long irrelevant prefixes, and the effect of option descriptions.

    uv run python -m bench.score prog   [--results DIR]     # DIR/<model>-prog.jsonl
    uv run python -m bench.score native [--results DIR]

A result file is scored only if it answers exactly the decisions of the item file: no missing, extra or duplicate
uid, and every probability vector has one finite, non-negative entry per option with a positive sum. Otherwise the
problems are printed and the model is left out.

Groups of the programmatic set: anlama = olumsuzluk, duyum, iade-kurali, sayi-bicim; hesap = tarih-vade, gun-hesabi,
is-gunu, kdv-indirim, sayma. Robustness "damage": the share of pairs whose clean original (same pair_id, ordering p0)
is answered correctly and whose modified copy is not. "same": the share of a family's items whose predicted option
text is the family's most common prediction; 1.0 means one answer whatever the item says.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench.jury import ROOT

BINS = 15


def ece(conf: np.ndarray, correct: np.ndarray) -> float:
    """Expected calibration error over equal-width bins of the top probability (same as trdecide.analyze)."""
    edges = np.linspace(0, 1, BINS + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, BINS - 1)
    total = 0.0
    for b in range(BINS):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


GROUPS = {"anlama": ("olumsuzluk", "duyum", "iade-kurali", "sayi-bicim"),
          "hesap": ("tarih-vade", "gun-hesabi", "is-gunu", "kdv-indirim", "sayma")}
ROBUST = ("gizli-talimat", "uzun-giris")


def load_items(which: str) -> dict[str, dict]:
    rows = [json.loads(l) for l in (ROOT / f"bench_{which}_items.jsonl").read_text(encoding="utf-8").splitlines() if l]
    return {r["uid"]: r for r in rows}


def validate(items: dict[str, dict], path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    problems, seen = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["uid"] in seen:
            problems.append(f"duplicate uid {r['uid']}")
        seen[r["uid"]] = r["p"]
    missing, extra = items.keys() - seen.keys(), seen.keys() - items.keys()
    if missing:
        problems.append(f"{len(missing)} decisions missing, e.g. {sorted(missing)[0]}")
    if extra:
        problems.append(f"{len(extra)} unknown uids, e.g. {sorted(extra)[0]}")
    probs = {}
    for uid, p in seen.items():
        if uid not in items:
            continue
        a = np.asarray(p, dtype=float)
        if a.ndim != 1 or len(a) != items[uid]["k"] or not np.isfinite(a).all() or (a < 0).any() or a.sum() <= 0:
            problems.append(f"invalid probabilities for {uid}: {p}")
            continue
        probs[uid] = a / a.sum()
    return probs, problems


DRAWS, SEED = 2000, 0


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """95% percentile interval of the mean over items resampled with replacement (2,000 draws, seed 0)."""
    idx = np.random.default_rng(SEED).integers(0, len(values), size=(DRAWS, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def metrics(rows: list[tuple]) -> dict:
    if not rows:
        return {"n": 0}
    c, conf, b = (np.array([r[i] for r in rows]) for i in (2, 3, 4))
    return {"n": len(rows), "acc": float(c.mean()), "acc_ci95": bootstrap_ci(c), "ece": ece(conf, c),
            "brier": float(b.mean())}


def score_model(items: dict[str, dict], probs: dict[str, np.ndarray]) -> dict:
    rows = []
    for uid, p in probs.items():
        it = items[uid]
        pred = int(p.argmax())
        rows.append((it, p, float(pred == it["gold"]), float(p[pred]),
                     float(((p - np.eye(len(p))[it["gold"]]) ** 2).sum()), it["options"][pred]))
    main = [r for r in rows if r[0]["perm"] == 0 and r[0]["cond"] == "tr"]
    by_family = defaultdict(list)
    for r in main:
        by_family[r[0]["family"]].append(r)
    families = {}
    for f, rs in sorted(by_family.items()):
        top = Counter(r[5] for r in rs).most_common(1)[0][1]
        families[f] = {**metrics(rs), "same": top / len(rs)}
    groups = {g: metrics([r for r in main if r[0]["family"] in fams]) for g, fams in GROUPS.items()}
    overall = metrics([r for r in main if r[0]["family"] not in ROBUST])
    chosen = defaultdict(dict)
    for r in rows:
        if r[0]["cond"] == "tr":
            chosen[r[0]["item"]][r[0]["perm"]] = r[5]
    pairs = [v for v in chosen.values() if {0, 1} <= v.keys()]
    flips = float(np.mean([v[0] != v[1] for v in pairs])) if pairs else None
    correct0 = {r[0]["item"]: r[2] for r in main}
    damage = {}
    for f in ROBUST:
        hits = [(correct0[r[0]["pair_id"]], r[2]) for r in by_family.get(f, []) if r[0]["pair_id"] in correct0]
        base_ok = [m for o, m in hits if o == 1.0]
        damage[f] = {"n": len(base_ok), "damage": float(np.mean([m == 0.0 for m in base_ok])) if base_ok else None}
    desc = {}
    d_rows = [r for r in rows if r[0]["cond"] == "tr_desc"]
    if d_rows:
        both = [(correct0[r[0]["item"]], r[2]) for r in d_rows if r[0]["item"] in correct0]
        desc = {"n": len(both), "acc_labels": float(np.mean([a for a, _ in both])),
                "acc_desc": float(np.mean([d for _, d in both]))}
    return {"overall": overall, "groups": groups, "families": families, "order_flip": flips, "robustness": damage,
            "descriptions": desc}


def paired_diff(items: dict[str, dict], pa: dict[str, np.ndarray], pb: dict[str, np.ndarray]) -> dict:
    """Accuracy of A minus B on the items of the overall accuracy, one set of bootstrap draws for both."""
    def correct(probs: dict[str, np.ndarray]) -> dict[str, float]:
        return {u: float(p.argmax() == items[u]["gold"]) for u, p in probs.items()
                if items[u]["perm"] == 0 and items[u]["cond"] == "tr" and items[u]["family"] not in ROBUST}
    ca, cb = correct(pa), correct(pb)
    uids = sorted(ca.keys() & cb.keys())
    d = np.array([ca[u] - cb[u] for u in uids])
    return {"n": len(uids), "diff": float(d.mean()), "ci95": bootstrap_ci(d)}


def fmt(x: float | None, w: int = 6) -> str:
    return f"{x:{w}.3f}" if isinstance(x, float) else f"{'-':>{w}s}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", help="prog, native, or any NAME with bench_NAME_items.jsonl and results/*-NAME.jsonl")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--paired", nargs=2, action="append", default=[], metavar=("A", "B"),
                    help="also print accuracy of A minus B with a paired 95%% interval; repeatable")
    a = ap.parse_args()
    items = load_items(a.which)
    board, rejected, all_probs = {}, {}, {}
    for f in sorted(a.results.glob(f"*-{a.which}.jsonl")):
        name = f.stem.removesuffix(f"-{a.which}")
        probs, problems = validate(items, f)
        if problems:
            rejected[name] = problems[:5]
            continue
        board[name] = score_model(items, probs)
        all_probs[name] = probs
    fams = sorted({f for s in board.values() for f in s["families"]})
    print(f"{'model':22s} {'acc':>6s} {'95% CI':>13s} {'ece':>6s} {'brier':>6s} {'flip':>6s} {'anlama':>6s} "
          f"{'hesap':>6s}  " + " ".join(f"{f[:10]:>10s}" for f in fams) + "  talimat-zarar dolgu-zarar")
    for m, s in sorted(board.items(), key=lambda kv: -kv[1]["overall"].get("acc", 0)):
        o, g, rb = s["overall"], s["groups"], s["robustness"]
        fam_txt = " ".join(fmt(s["families"].get(f, {}).get("acc"), 10) for f in fams)
        ci = "{:.3f}-{:.3f}".format(*o["acc_ci95"]) if "acc_ci95" in o else "-"
        print(f"{m:22s} {fmt(o.get('acc'))} {ci:>13s} {fmt(o.get('ece'))} {fmt(o.get('brier'))} "
              f"{fmt(s['order_flip'])} {fmt(g['anlama'].get('acc'))} {fmt(g['hesap'].get('acc'))}  {fam_txt}  "
              f"{fmt((rb.get('gizli-talimat') or {}).get('damage'), 13)} "
              f"{fmt((rb.get('uzun-giris') or {}).get('damage'), 11)}")
    for m, problems in rejected.items():
        print(f"NOT SCORED {m}: " + "; ".join(problems))
    for x, y in a.paired:
        r = paired_diff(items, all_probs[x], all_probs[y])
        print(f"paired {x} - {y}: {r['diff']:+.3f} ({r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f}), n={r['n']}")
    out = ROOT / f"leaderboard_{a.which}.json"
    out.write_text(json.dumps({"scored": board, "rejected": rejected}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
