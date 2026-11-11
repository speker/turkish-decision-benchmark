"""Jury pipeline for the natively written items (see bench/JURY_SPEC.md).

    uv run python -m bench.jury blind  BATCH            # jury/blind_<batch>.jsonl (no gold)
    uv run python -m bench.jury review BATCH            # after round 1: jury/review_<batch>.jsonl
    uv run python -m bench.jury recheck BATCH           # after hand fixes: jury/recheck_<batch>.jsonl (no gold)
    uv run python -m bench.jury report BATCH...         # how each batch's items were decided so far (writes nothing)
    uv run python -m bench.jury final                   # bench/native.jsonl; refuses while any batch has an open round
    uv run python -m bench.jury prog-sample             # 20 programmatic items per family, no gold -> jury/prog_blind.jsonl
    uv run python -m bench.jury prog-compare            # jury/prog_r1_<judge>.jsonl against the computed gold

Files: round 1 jury/r1_<batch>_<judge>.jsonl, round 2 jury/r2_<batch>_<judge>.jsonl, hand fixes
jury/fixed_<batch>.jsonl (full corrected items), round 3 jury/r3_<batch>_<judge>.jsonl.

Rules. Every step refuses to run on incomplete input: each judge must have answered every item. An item goes to
review when any judge answers differently from the gold, marks it ambiguous or writes a note (notes are written only
when something is wrong), or when the mean naturalness is below 3. A reviewed item is kept when at least three of the four reviewers vote `keep` with `gold_correct` and
`gold_unique` both true. When at least two reviewers vote `fix`, the item is corrected by hand and asked again
without its gold (round 3); it is kept only if every round-3 judge picks the corrected gold. Everything else is
dropped, and the report says why.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# TR_DECIDE_BENCH points the whole pipeline at another working folder, e.g. a rebuild from the public dataset
ROOT = Path(os.environ.get("TR_DECIDE_BENCH")
            or Path(__file__).resolve().parents[3] / "data" / "cache" / "tr-decide" / "bench").resolve()
NATIVE, JURY = ROOT / "native", ROOT / "jury"
JUDGES = ("gpt", "gemini", "grok", "muse")
RECHECK_JUDGES = ("gpt", "gemini")


class Incomplete(Exception):
    pass


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def answers(prefix: str, batch: str, judges: tuple[str, ...], items: list[dict], need: set[str] | None = None
            ) -> dict[str, dict[str, dict]]:
    """Answers of every judge, keyed by item id; raises Incomplete if any judge lacks a valid answer."""
    need = {it["id"] for it in items} if need is None else need
    by_id = {it["id"]: it for it in items}
    out, problems = {}, []
    for j in judges:
        raw = load(JURY / f"{prefix}_{batch}_{j}.jsonl")
        rows = {r["id"]: r for r in raw}
        missing, extra = need - rows.keys(), rows.keys() - by_id.keys()
        dup = len(raw) - len(rows)
        bad = [i for i in need & rows.keys() if prefix != "r2" and not 0 <= rows[i].get("answer", -1) < len(by_id[i]["options"])]
        if missing or extra or dup or bad:
            problems.append(f"{prefix}_{batch}_{j}: {len(missing)} missing, {len(extra)} unknown ids, {dup} duplicates, "
                            f"{len(bad)} invalid")
        out[j] = rows
    if problems:
        raise Incomplete("; ".join(problems))
    return out


def disputed(it: dict, r1: dict[str, dict[str, dict]]) -> list[str]:
    votes = [r1[j][it["id"]] for j in JUDGES]
    reasons = []
    if any(v["answer"] != it["gold"] for v in votes):
        reasons.append("a judge disagrees with the gold")
    if any(v.get("ambiguous") for v in votes):
        reasons.append("marked ambiguous")
    if any(str(v.get("note", "")).strip() for v in votes):
        reasons.append("a judge noted a problem")
    nat = [v["natural"] for v in votes if isinstance(v.get("natural"), (int, float))]
    if not nat or sum(nat) / len(nat) < 3:
        reasons.append("low naturalness")
    return reasons


def strip(it: dict) -> dict:
    return {k: it[k] for k in ("id", "state", "question", "options")}


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def blind(batch: str) -> None:
    JURY.mkdir(parents=True, exist_ok=True)
    items = load(NATIVE / f"batch_{batch}.jsonl")
    write(JURY / f"blind_{batch}.jsonl", [strip(it) for it in items])
    print(f"blind_{batch}.jsonl: {len(items)} items")


def review(batch: str) -> None:
    items = load(NATIVE / f"batch_{batch}.jsonl")
    r1 = answers("r1", batch, JUDGES, items)
    rows = [{**strip(it), "option_descriptions": it["option_descriptions"], "gold": it["gold"],
             "rationale": it["rationale"], "jury_round1": {j: r1[j][it["id"]] for j in JUDGES}, "why": why}
            for it in items if (why := disputed(it, r1))]
    write(JURY / f"review_{batch}.jsonl", rows)
    print(f"review_{batch}.jsonl: {len(rows)} of {len(items)} items disputed")


def verdicts(batch: str, items: list[dict], ids: set[str]) -> dict[str, Counter]:
    r2 = answers("r2", batch, JUDGES, items, need=ids)
    out = {}
    for i in ids:
        c = Counter()
        for j in JUDGES:
            v = r2[j][i]
            if v.get("verdict") == "keep" and v.get("gold_correct") is True and v.get("gold_unique") is True:
                c["keep"] += 1
            elif v.get("verdict") == "fix":
                c["fix"] += 1
            else:
                c["drop"] += 1
        out[i] = c
    return out


def to_fix(batch: str) -> list[str]:
    items = load(NATIVE / f"batch_{batch}.jsonl")
    r1 = answers("r1", batch, JUDGES, items)
    ids = {it["id"] for it in items if disputed(it, r1)}
    return sorted(i for i, c in verdicts(batch, items, ids).items() if c["keep"] < 3 and c["fix"] >= 2)


def recheck(batch: str) -> None:
    fixed = load(JURY / f"fixed_{batch}.jsonl")
    want = set(to_fix(batch))
    if {f["id"] for f in fixed} != want:
        raise Incomplete(f"fixed_{batch}.jsonl must hold exactly the items to fix: {sorted(want)}")
    write(JURY / f"recheck_{batch}.jsonl", [strip(f) for f in fixed])
    print(f"recheck_{batch}.jsonl: {len(fixed)} items")


def settle(batch: str) -> tuple[list[dict], Counter]:
    """The kept items of one batch and a count of how each item was decided; raises Incomplete if a round is open."""
    items = load(NATIVE / f"batch_{batch}.jsonl")
    r1 = answers("r1", batch, JUDGES, items)
    disp = {it["id"] for it in items if disputed(it, r1)}
    vd = verdicts(batch, items, disp) if disp else {}
    fixed = {f["id"]: f for f in load(JURY / f"fixed_{batch}.jsonl")}
    fix_ids = {i for i, c in vd.items() if c["keep"] < 3 and c["fix"] >= 2}
    if fix_ids != fixed.keys():
        raise Incomplete(f"batch {batch}: fixed items {sorted(fixed)} differ from items to fix {sorted(fix_ids)}")
    r3 = answers("r3", batch, RECHECK_JUDGES, list(fixed.values())) if fixed else {}
    kept, count = [], Counter()
    for it in items:
        votes = {j: r1[j][it["id"]]["answer"] for j in JUDGES}
        entry = {**it, "jury_round1_answers": votes,
                 "jury_round1_agree_with_gold": sum(v == it["gold"] for v in votes.values())}
        if it["id"] not in disp:
            kept.append({**entry, "review": "undisputed"})
            count["undisputed"] += 1
        elif vd[it["id"]]["keep"] >= 3:
            kept.append({**entry, "review": "kept-after-review"})
            count["kept-after-review"] += 1
        elif it["id"] in fixed:
            f = fixed[it["id"]]
            if all(r3[j][f["id"]]["answer"] == f["gold"] for j in RECHECK_JUDGES):
                kept.append({**entry, **f, "review": "fixed-and-rechecked"})
                count["fixed-and-rechecked"] += 1
            else:
                count["dropped-after-recheck"] += 1
        else:
            count["dropped"] += 1
    return kept, count


def report(batches: list[str]) -> None:
    for b in batches:
        try:
            kept, count = settle(b)
            print(b, dict(count), f"-> {len(kept)} kept")
        except Incomplete as e:
            print(b, "open:", e)


def human_drops() -> set[str]:
    """Items removed after the human check because a second answer is defensible (human_drops.json)."""
    path = Path(__file__).with_name("human_drops.json")
    return {d["id"] for d in json.loads(path.read_text(encoding="utf-8"))["items"]} if path.exists() else set()


def final() -> None:
    kept, drop = [], human_drops()
    for path in sorted(NATIVE.glob("batch_b*.jsonl")):
        if path.stem.endswith(".orig"):
            continue
        rows, count = settle(path.stem.split("_", 1)[1])
        removed = [r for r in rows if r["id"] in drop]
        if removed:
            count["dropped-after-human-check"] = len(removed)
        kept += [r for r in rows if r["id"] not in drop]
        print(path.stem, dict(count))
    write(ROOT / "native.jsonl", kept)
    print(f"native.jsonl: {len(kept)} items kept")


PROG_JUDGES = ("gpt", "gemini", "grok")


def prog_sample(per_family: int = 20) -> None:
    by_family = defaultdict(list)
    for r in load(ROOT / "prog.jsonl"):
        by_family[r["family"]].append(r)
    rng = random.Random(0)
    pick = [r for f in sorted(by_family) for r in rng.sample(by_family[f], min(per_family, len(by_family[f])))]
    rng.shuffle(pick)
    JURY.mkdir(parents=True, exist_ok=True)
    write(JURY / "prog_blind.jsonl", [strip(r) for r in pick])
    print(f"prog_blind.jsonl: {len(pick)} items from {len(by_family)} families")


def prog_compare() -> None:
    gold = {r["id"]: r for r in load(ROOT / "prog.jsonl")}
    sample = [gold[r["id"]] for r in load(JURY / "prog_blind.jsonl")]
    ans = {j: {r["id"]: r for r in load(JURY / f"prog_r1_{j}.jsonl")} for j in PROG_JUDGES}
    missing = {j: len({g["id"] for g in sample} - ans[j].keys()) for j in PROG_JUDGES}
    if any(missing.values()):
        raise Incomplete(f"missing answers per judge: {missing}")
    per_family, rows = defaultdict(Counter), []
    for g in sample:
        i = g["id"]
        votes = {j: ans[j][i]["answer"] for j in PROG_JUDGES}
        wrong = [j for j, v in votes.items() if v != g["gold"]]
        flagged = [j for j in PROG_JUDGES if ans[j][i].get("ambiguous") or ans[j][i].get("note")]
        per_family[g["family"]]["items"] += 1
        per_family[g["family"]]["judge_wrong"] += len(wrong)
        if wrong or flagged:
            rows.append({"id": i, "family": g["family"], "gold": g["gold"], "gold_text": g["options"][g["gold"]],
                         "votes": votes, "wrong": wrong,
                         "notes": {j: ans[j][i].get("note", "") for j in PROG_JUDGES},
                         "ambiguous": {j: ans[j][i].get("ambiguous", False) for j in PROG_JUDGES}})
    write(JURY / "prog_disputed.jsonl", rows)
    wrong = sum(c["judge_wrong"] for c in per_family.values())
    (JURY / "prog_summary.json").write_text(json.dumps({
        "items": len(sample), "families": len(per_family), "judges": list(PROG_JUDGES),
        "answers": len(sample) * len(PROG_JUDGES), "agree": len(sample) * len(PROG_JUDGES) - wrong,
        "disputed": len(rows)}, indent=1) + "\n", encoding="utf-8")
    for f, c in sorted(per_family.items()):
        print(f"{f:14s} items {c['items']:3d}  judge answers wrong {c['judge_wrong']:3d}/{c['items'] * len(PROG_JUDGES)}")
    print(f"prog_disputed.jsonl: {len(rows)} items with a wrong vote, an ambiguity flag or a note "
          f"({sum(len(r['wrong']) >= 2 for r in rows)} with 2+ wrong votes)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    try:
        {"blind": lambda: blind(sys.argv[2]), "review": lambda: review(sys.argv[2]),
         "recheck": lambda: recheck(sys.argv[2]), "final": final, "report": lambda: report(sys.argv[2:]),
         "prog-sample": prog_sample, "prog-compare": prog_compare}[cmd]()
    except Incomplete as e:
        sys.exit(f"incomplete: {e}")
