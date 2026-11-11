"""Validate a JSONL file of natively written decision items against bench/WRITER_SPEC.md.

    uv run python projects/tr-decide/bench/validate_native.py FILE [FILE ...]

Exit code 1 if any item breaks a rule; prints one line per problem and a summary of the mix.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

STYLES = {"resmi-eposta", "gundelik-mesaj", "turkce-karaktersiz", "yazim-hatali", "uzun-dolambacli", "tek-satir",
          "kod-karisik", "sesli-yazdirma", "yasli-kibar", "genc-emojili", "sikayet-sinirli", "cok-konulu"}
PHENOMENA = {"olumsuzluk", "cift-olumsuz", "mis-duyum", "kosul", "ironi", "tarih-hesabi", "sayi-para", "kural-uygulama",
             "eksik-bilgi", "gizli-talimat", "coklu-istek", "kod-karisik", "kisisel-veri", "deyim"}
FIELDS = ("id", "domain", "style", "phenomena", "type", "state", "question", "options", "option_descriptions", "gold",
          "rationale", "difficulty")
ID = re.compile(r"^tdb-b\d+-\d{3}$")
PII = [re.compile(p) for p in (r"\b\d{11}\b", r"\bTR\d{2}[ ]?\d{4}", r"(\+90|0)\s?5\d{2}\s?\d{3}\s?\d{2}\s?\d{2}",
                                r"[\w.+-]+@[\w-]+\.[\w.]+", r"https?://", r"www\.")]


def check(path: Path) -> tuple[list[str], list[dict]]:
    errs, rows = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errs.append(f"{path.name}:{n}: bad JSON ({e})")
            continue
        rows.append(r)
        where = f"{path.name}:{n} {r.get('id', '?')}"
        missing = [f for f in FIELDS if f not in r]
        if missing:
            errs.append(f"{where}: missing {missing}")
            continue
        if not ID.match(r["id"]):
            errs.append(f"{where}: bad id")
        if r["style"] not in STYLES:
            errs.append(f"{where}: unknown style {r['style']!r}")
        if not isinstance(r["phenomena"], list) or set(r["phenomena"]) - PHENOMENA:
            errs.append(f"{where}: unknown phenomena {set(r['phenomena']) - PHENOMENA}")
        if r["type"] not in ("choice", "noul"):
            errs.append(f"{where}: type must be choice or noul")
        opts, desc = r["options"], r["option_descriptions"]
        if not isinstance(opts, list) or not isinstance(desc, list) or len(opts) != len(desc):
            errs.append(f"{where}: options and option_descriptions must be lists of equal length")
            continue
        if r["type"] == "noul" and opts != ["Hayır", "Evet"]:
            errs.append(f"{where}: noul options must be [\"Hayır\", \"Evet\"]")
        if r["type"] == "choice" and not 3 <= len(opts) <= 12:
            errs.append(f"{where}: choice needs 3-12 options, has {len(opts)}")
        if len({o.strip().casefold() for o in opts}) != len(opts):
            errs.append(f"{where}: duplicate option texts")
        if not isinstance(r["gold"], int) or not 0 <= r["gold"] < len(opts):
            errs.append(f"{where}: gold out of range")
        if r["difficulty"] not in ("kolay", "orta", "zor"):
            errs.append(f"{where}: difficulty must be kolay/orta/zor")
        text = " ".join([r["state"], r["question"], *opts, *desc])
        for p in PII:
            if p.search(text):
                errs.append(f"{where}: looks like personal data or a URL ({p.pattern}); use placeholders")
        if len(r["state"]) < 20:
            errs.append(f"{where}: state too short")
    ids = Counter(r.get("id") for r in rows)
    errs += [f"{path.name}: duplicate id {i}" for i, c in ids.items() if c > 1]
    return errs, rows


def main() -> int:
    all_errs, all_rows = [], []
    for p in sys.argv[1:]:
        e, rows = check(Path(p))
        all_errs += e
        all_rows += rows
    for e in all_errs:
        print(e)
    types = Counter(r.get("type") for r in all_rows)
    diff = Counter(r.get("difficulty") for r in all_rows)
    gold_pos = Counter(r.get("gold") for r in all_rows if r.get("type") == "choice")
    many = sum(1 for r in all_rows if r.get("type") == "choice" and len(r.get("options", [])) >= 9)
    phen = sum(1 for r in all_rows if r.get("phenomena"))
    print(f"items {len(all_rows)} | types {dict(types)} | difficulty {dict(diff)} | choice with >=9 options {many} | "
          f"with phenomena {phen} | choice gold positions {dict(sorted(gold_pos.items()))} | problems {len(all_errs)}")
    return 1 if all_errs else 0


if __name__ == "__main__":
    sys.exit(main())
