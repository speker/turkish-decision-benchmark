"""Build the Hub dataset folder of the Turkish decision benchmark: projects/tr-decide/hub.

    uv run python -m bench.build_hub          # data and results
    uv run python -m bench.gate               # reports/dedup.json and reports/decontam.json from that data
    uv run python -m bench.build_hub card     # hub/README.md from bench/card_template.md, results and reports

Writes:
  hub/data/programmatic/test-00000-of-00001.parquet   2,820 items, gold computed by bench/prog.py (tdp-v1.1)
  hub/data/written/test-00000-of-00001.parquet        AI-written, jury-checked items (batches b1-b7, without
                                                      the items in bench/human_drops.json)
  hub/results/leaderboard.json                        per model and set: accuracy with 95% interval, ECE, ...
  hub/results/<model>/<date>/{programmatic,written}.jsonl   per-decision probabilities from the harness
  hub/README.md                                       the card; every number in it is read from the files
                                                      above, and a missing input leaves an "[insert ...]" mark
                                                      that card_check fails on
Every row carries the canary string of projects/tr-decide/CANARY. Batch b8 and a second programmatic set drawn
with a secret seed stay out of the Hub (.private/tr-decide-bench) for later contamination checks. Individual
jury answers are not published, only how many of the four round-one judges agreed with the gold.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from bench.jury import ROOT, human_drops
from bench.score import load_items, score_model, validate

PROJECT = Path(__file__).resolve().parents[1]
HUB = PROJECT / "hub"
RUN_DATE = "2026-11-10"
WRITTEN_BATCHES = tuple(f"b{i}" for i in range(1, 8))
# results name -> (Hub repo, revision, how it was run); every run used the harness in remote/
MODELS = {
    "d3": ("vllm-sr/d3", "62f07e5", "vLLM 0.33.0, /v1/systemone endpoint"),
    "pusula": ("erenextlabs/pusula", "db9b467",
               "pusula package 2.0.0 (git 771140836ba8) on d3 62f07e5, transformers 5.17.0, "
               "adapter on for every question"),
    "jev-27b": ("autotrust/JEV-27B", "14b425269ed24452e0a5ad5a8f3e51d139647a88",
                "transformers 5.22.2 + peft 0.21.2"),
    "jev-35b": ("autotrust/JEV-35B", "a5d0d580bb9324f0f4c07dcba5c5ca91e813e7c9",
                "the repo's serve_decide.py on vLLM 0.33.0, strategy=single"),
    "jev-9b": ("autotrust/JEV-9B", "7ac669b8fb476ba926491e3d1c39f9330d68dbed",
               "transformers 5.22.2 + peft 0.21.2"),
    "qwen3.8-27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                    "vLLM 0.33.0, next-token probabilities of the option letters, thinking off"),
    "qwen3.5-122b": ("Qwen/Qwen3.5-122B-A10B-FP8", "a099dee",
                     "vLLM 0.33.0 TP2, next-token probabilities of the option letters, thinking off"),
    "deepseek-v4-flash": ("deepseek-ai/DeepSeek-V4-Flash", "60d8d70770c6776ff598c94bb586a859a38244f1",
                          "vLLM 0.33.0 TP2, next-token probabilities of the option letters, chat encoding"),
    "karar-4b": ("mertkayacs/Karar-4B", "d299ed5884bc3049500bd83d3f7d4cd53f2f6f9c",
                 "jevalt server, /v1/systemone"),
    "laya-multilingual": ("convaiinnovations/laya-multilingual", "1720e3e3357cfe1e281542e223f8273b0890ca34",
                          "laya 0.4.30, max_len 8192 as the card recommends for long inputs"),
}
WRITTEN_RAW = "native700"  # decisions of batches b1-b7 as the models answered them


def keep_lines(src: Path, dst: Path, drop: set[str]) -> None:
    """Copy src to dst without the decisions of dropped items; kept lines are copied byte for byte."""
    lines = [line for line in src.read_text(encoding="utf-8").splitlines()
             if line and json.loads(line)["uid"].split(":")[0] not in drop]
    dst.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def written_set() -> str:
    """Name of the written decisions without the items removed after the human check; writes
    bench_<name>_items.jsonl and results/<model>-<name>.jsonl next to the raw files, which stay as
    they are."""
    drop = human_drops()
    if not drop:
        return WRITTEN_RAW
    raw = ROOT / f"bench_{WRITTEN_RAW}_items.jsonl"
    kept = {json.loads(line)["item"] for line in raw.read_text(encoding="utf-8").splitlines() if line} - drop
    name = f"native{len(kept)}"
    keep_lines(raw, ROOT / f"bench_{name}_items.jsonl", drop)
    for src in sorted((ROOT / "results").glob(f"*-{WRITTEN_RAW}.jsonl")):
        keep_lines(src, src.with_name(src.name.replace(f"-{WRITTEN_RAW}.", f"-{name}.")), drop)
    return name


def canary() -> str:
    return (PROJECT / "CANARY").read_text(encoding="utf-8").strip()


LETTERS = "ABCDEFGHIJKLMNOP"


def common(r: dict) -> dict:
    """Fields every row has; eval.yaml points inspect-ai at `prompt`, `options` and `answer` (a letter)."""
    return {"state": r["state"], "question": r["question"], "prompt": f"{r['state']}\n\n{r['question']}",
            "options": r["options"], "answer": LETTERS[r["gold"]], "answer_index": r["gold"],
            "answer_text": r["options"][r["gold"]]}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def programmatic_rows(c: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": r["id"], "family": r["family"], "variant": r["variant"], "pair_id": r["pair_id"],
        "type": r["type"], **common(r),
        "source": "generated by bench/prog.py (tdp-v1.1); gold computed by code",
        "canary": c} for r in read_jsonl(ROOT / "prog.jsonl")])


def written_rows(c: str) -> pd.DataFrame:
    rows = [r for r in read_jsonl(ROOT / "native.jsonl") if r["id"].split("-")[1] in WRITTEN_BATCHES]
    return pd.DataFrame([{
        "id": r["id"], "domain": r["domain"], "style": r["style"], "phenomena": r["phenomena"],
        "type": r["type"], "difficulty": r["difficulty"], **common(r),
        "option_descriptions": r["option_descriptions"], "rationale": r["rationale"],
        "jury_agreement": r["jury_round1_agree_with_gold"], "review": r["review"],
        "source": "written by Claude Opus 5.5 (persona prompt bench/WRITER_SPEC.md), "
                  "checked by a four-model jury",
        "canary": c} for r in rows])


def main() -> None:
    c = canary()
    for sub in ("data", "results"):  # rebuilt from scratch so no stale file survives a regeneration
        shutil.rmtree(HUB / sub, ignore_errors=True)
    for name, df in (("programmatic", programmatic_rows(c)), ("written", written_rows(c))):
        out = HUB / "data" / name / "test-00000-of-00001.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"{name}: {len(df)} rows -> {out.relative_to(PROJECT)}")
    board = {}
    for set_name, which in {"programmatic": "prog", "written": written_set()}.items():
        items = load_items(which)
        for model, (repo, rev, how) in MODELS.items():
            src = ROOT / "results" / f"{model}-{which}.jsonl"
            if not src.exists():
                continue
            probs, problems = validate(items, src)
            if problems:
                raise SystemExit(f"{model} {set_name}: {problems[:3]}")
            dst = HUB / "results" / model / RUN_DATE / f"{set_name}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            board.setdefault(model, {"model": repo, "revision": rev, "how": how, "date": RUN_DATE})
            board[model][set_name] = score_model(items, probs)
    out = HUB / "results" / "leaderboard.json"
    out.write_text(json.dumps(board, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"leaderboard: {len(board)} models -> {out.relative_to(PROJECT)}")


REPORTS = PROJECT / "reports"
JURY_SUMMARY = ROOT / "jury" / "prog_summary.json"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def model_link(v: dict) -> str:
    return f"[{v['model']}](https://huggingface.co/{v['model']})"


def chance(df: pd.DataFrame, set_name: str) -> float:
    """Expected accuracy of a uniform random pick over the items the overall accuracy is computed on."""
    if set_name == "programmatic":
        df = df[~df["family"].isin(("gizli-talimat", "uzun-giris"))]
    return float((1.0 / df["options"].map(len)).mean())


def results_table(board: dict, set_name: str, random_acc: float) -> str:
    rows = sorted((v for v in board.values() if set_name in v), key=lambda v: -v[set_name]["overall"]["acc"])
    if not rows:
        return f"[insert {set_name} results]"
    if set_name == "written":
        out = ["| Model | Accuracy % (95% CI) | With descriptions % | ECE | Brier | Order flips % |",
               "|---|---|---:|---:|---:|---:|"]
        for v in rows:
            s = v[set_name]
            o, (lo, hi) = s["overall"], s["overall"]["acc_ci95"]
            out.append(f"| {model_link(v)} | {pct(o['acc'])} ({pct(lo)}–{pct(hi)}) | "
                       f"{pct(s['descriptions']['acc_desc'])} | {o['ece']:.3f} | {o['brier']:.3f} | "
                       f"{pct(s['order_flip'])} |")
        out.append(f"| Random choice (expected) | {pct(random_acc)} | | | | |")
        return "\n".join(out)
    out = ["| Model | Accuracy % (95% CI) | Reading % | Arithmetic % | ECE | Order flips % | Planted % "
           "| Prefix % |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for v in rows:
        s = v[set_name]
        o, (lo, hi), g, rb = s["overall"], s["overall"]["acc_ci95"], s["groups"], s["robustness"]
        out.append(f"| {model_link(v)} | {pct(o['acc'])} ({pct(lo)}–{pct(hi)}) | {pct(g['anlama']['acc'])} | "
                   f"{pct(g['hesap']['acc'])} | {o['ece']:.3f} | {pct(s['order_flip'])} | "
                   f"{pct(rb['gizli-talimat']['damage'])} | {pct(rb['uzun-giris']['damage'])} |")
    out.append(f"| Random choice (expected) | {pct(random_acc)} | | | | | | |")
    return "\n".join(out)


def gate_lines(prog: pd.DataFrame, written: pd.DataFrame) -> dict[str, str]:
    dedup = json.loads((REPORTS / "dedup.json").read_text(encoding="utf-8"))
    decontam = json.loads((REPORTS / "decontam.json").read_text(encoding="utf-8"))
    if (dedup["programmatic"]["rows"], dedup["written"]["rows"]) != (len(prog), len(written)):
        raise SystemExit("reports/dedup.json is from other data; run `python -m bench.gate` first")
    exact = dedup["programmatic"]["exact_duplicate_items"] + dedup["written"]["exact_duplicate_items"]
    near = dedup["written_near_duplicates"]
    overlap = sum(row[s]["items_with_overlap"] for row in decontam["references"].values()
                  for s in ("programmatic", "written"))
    return {
        "EXACT": f"{exact:,}",
        "NEAR": (f"{len(near['pairs'])} pairs, all deliberate minimal pairs with the same question and "
                 "different gold" if near["redundant"] == 0
                 else f"[insert: {near['redundant']} near-duplicate pairs to resolve]"),
        "DECONTAM": ("no item text shares a word 8-gram with" if overlap == 0
                     else f"[insert: {overlap} items overlap; see reports/decontam.json]"),
    }


def tr_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def human_sentence_tr(human: dict | None, dropped: int) -> str:
    if human is None:
        return "[insert human check of 40 written items]"
    if human.get("readers", 1) == 1:
        return (f"Bir okur {human['answered']} yazılmış soruyu doğru cevabı görmeden çözdü ve "
                f"{human['agree']} soruda cevap anahtarıyla aynı cevabı verdi.")
    readers, native = human["readers"], human["native"]
    who = (f"Ana dili Türkçe {readers} okur" if native == readers
           else f"{readers} okur (ana dili Türkçe olan {native})")
    alpha = f"{human['alpha_readers']:.2f}".replace(".", ",")
    text = (f"{who}, yazılmış görevden seçilen {human['items']} soruyu doğru cevabı görmeden çözdü; "
            f"çoğunlukları {human['majority_agree']} soruda cevap anahtarıyla aynı cevabı verdi, "
            f"okurlar arası Krippendorff alfası {alpha}.")
    if dropped:
        text += f" İkinci bir savunulabilir cevabı olan {dropped} soru setten çıkarıldı."
    return text


def dropped_sentence(n: int) -> str:
    if n == 0:
        return "no item was dropped"
    count = {1: "one item was", 2: "two items were", 3: "three items were"}.get(n, f"{n} items were")
    return f"the jury dropped no item, and {count} removed after the human check described below"


def human_sentence(human: dict | None) -> str:
    if human is None:
        return "[insert human check of 40 written items]"
    alpha_note = "Krippendorff's alpha (nominal, option position as category)"
    if human.get("readers", 1) == 1:
        who = "a native speaker" if human.get("native", 1) else "a Turkish reader who is not a native speaker"
        return (f"{who} answered {human['answered']} written items without the gold and agreed with it on "
                f"{human['agree']}; {alpha_note} is {human['alpha']:.2f}.")
    readers, native = human["readers"], human["native"]
    who = (f"{readers} native speakers" if native == readers
           else f"{readers} Turkish readers ({native} of them native speakers)")
    text = (f"{who} each answered the {human['items']} sampled written items without the gold. "
            f"Their majority agreed with the gold on {human['majority_agree']} items"
            + (f" and gave no majority on {human['no_majority']}" if human["no_majority"] else "")
            + f"; one reader agreed with it on {human['mean_agree']:.1f} items on average "
            f"({human['min_agree']}–{human['max_agree']}), and {alpha_note} among the readers is "
            f"{human['alpha_readers']:.2f}.")
    if human["disputed"]:
        note = REPORTS / "human_disputed_note.txt"  # written by hand once each disputed item was checked
        text += " " + (note.read_text(encoding="utf-8").strip() if note.exists()
                       else f"[insert: check the {len(human['disputed'])} items where the majority differs "
                            "from the gold]")
    return text


def card() -> None:
    prog = pd.read_parquet(HUB / "data" / "programmatic" / "test-00000-of-00001.parquet")
    written = pd.read_parquet(HUB / "data" / "written" / "test-00000-of-00001.parquet")
    board = json.loads((HUB / "results" / "leaderboard.json").read_text(encoding="utf-8"))
    review = written["review"].value_counts()
    agree, answers = int(written["jury_agreement"].sum()), 4 * len(written)
    jury = json.loads(JURY_SUMMARY.read_text(encoding="utf-8")) if JURY_SUMMARY.exists() else None
    jury_note = REPORTS / "prog_jury_note.txt"  # written by hand once each disagreement was checked
    if jury and jury["agree"] < jury["answers"]:
        jury["note"] = (jury_note.read_text(encoding="utf-8").strip() if jury_note.exists()
                        else "[insert: check each disagreement against the code]")
    both = [v for v in board.values() if "programmatic" in v and "written" in v]
    harder = sum(v["programmatic"]["overall"]["acc"] < v["written"]["overall"]["acc"] for v in both)
    human_path = REPORTS / "human.json"
    human = json.loads(human_path.read_text(encoding="utf-8")) if human_path.exists() else None
    values = {
        "TOTAL": f"{len(prog) + len(written):,}", "PROG_N": f"{len(prog):,}",
        "WRITTEN_N": f"{len(written):,}",
        "DOMAINS": str(written["domain"].nunique()), "STYLES": str(written["style"].nunique()),
        "UNDISPUTED": str(review.get("undisputed", 0)), "KEPT": str(review.get("kept-after-review", 0)),
        "FIXED": str(review.get("fixed-and-rechecked", 0)),
        "JURY_AGREE": f"{agree:,}", "JURY_ANSWERS": f"{answers:,}",
        "JURY_AGREE_PCT": f"{100 * agree / answers:.1f}%",
        "HARDER": (f"all {len(both)} models measured" if harder == len(both)
                   else f"{harder} of the {len(both)} models measured"),
        "PROG_JURY": (f"they agreed with the code on {jury['agree']:,} of {jury['answers']:,} answers "
                      f"({jury['items']} items, {jury['families']} families)"
                      + (f"; {jury['note']}" if jury.get("note") else "") if jury
                      else "[insert blind-jury result for the programmatic sample]"),
        "HUMAN": human_sentence(human),
        "HUMAN_TR": human_sentence_tr(human, len(human_drops())),
        "DROPPED": dropped_sentence(len(human_drops())),
        "TOTAL_TR": tr_int(len(prog) + len(written)), "PROG_N_TR": tr_int(len(prog)),
        "WRITTEN_N_TR": tr_int(len(written)),
        "WRITTEN_TABLE": results_table(board, "written", chance(written, "written")),
        "PROG_TABLE": results_table(board, "programmatic", chance(prog, "programmatic")),
        "HOW": "\n".join(f"- {model_link(v)} at `{v['revision'][:7]}`: {v['how']}." for v in board.values()),
        "CANARY": canary(),
        **gate_lines(prog, written),
    }
    text = (PROJECT / "bench" / "card_template.md").read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    if "{{" in text:
        raise SystemExit(f"unfilled template field near: {text[text.index('{{'):][:40]}")
    (HUB / "README.md").write_text(text, encoding="utf-8")
    print(f"card -> {(HUB / 'README.md').relative_to(PROJECT)}; pending marks: {text.count('[insert')}")


if __name__ == "__main__":
    card() if sys.argv[1:] == ["card"] else main()
