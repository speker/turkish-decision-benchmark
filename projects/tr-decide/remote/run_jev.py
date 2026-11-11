"""h200: typed `choice` decisions with a JEV checkpoint, following the model card's PyTorch path.

Prompts are grouped by exact token length, so no batch is padded (the Gated DeltaNet layers are not
checked for padding invariance here); `--check` compares a sample against one-at-a-time runs.

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python run_jev.py --model models/jev-9b --name jev-9b --zeroshot qwen3.5-9b-zeroshot
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import peft
import torch
import transformers
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = "ABCDEFGHIJKLMNOP"
CARD_EXAMPLE = ("Customer says the parcel arrived damaged and wants their money back.",
                "Is the customer asking for a refund?")


def prompt(kind: str, state: str, question: str, options: list[str]) -> str:
    lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
    return f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


@torch.no_grad()
def last_hidden(model, ids: list[list[int]]) -> torch.Tensor:
    x = torch.tensor(ids, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.model(input_ids=x).last_hidden_state[:, -1].float()


def scorer_head(W: torch.Tensor, b: torch.Tensor, temp: dict, ranges: dict):
    def score(h: torch.Tensor, kind: str, n: int) -> torch.Tensor:
        s = ranges[kind][0]
        return torch.softmax(((h @ W.T + b) / temp[kind])[:, s : s + n], -1)
    return score


def scorer_lm_head(lm_rows: torch.Tensor, ranges: dict):
    def score(h: torch.Tensor, kind: str, n: int) -> torch.Tensor:
        s = ranges[kind][0]
        return torch.softmax((h @ lm_rows.T)[:, s : s + n], -1)
    return score


def run(model, tok, items: list[dict], score, out: Path, max_batch_tokens: int) -> dict:
    enc = [tok(prompt("choice", it["state"], it["question"], it["options"]), add_special_tokens=False)["input_ids"]
           for it in items]
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(enc):
        by_len[len(e)].append(i)
    probs: list[list[float] | None] = [None] * len(items)
    t0 = time.time()
    for length, idxs in sorted(by_len.items()):
        bs = max(1, max_batch_tokens // length)
        for j in range(0, len(idxs), bs):
            chunk = idxs[j : j + bs]
            h = last_hidden(model, [enc[i] for i in chunk])
            # group by option count inside the chunk
            for k in {len(items[i]["options"]) for i in chunk}:
                rows = [n for n, i in enumerate(chunk) if len(items[i]["options"]) == k]
                p = score(h[rows], "choice", k).cpu().tolist()
                for n, pr in zip(rows, p):
                    probs[chunk[n]] = pr
    secs = time.time() - t0
    with out.open("w", encoding="utf-8") as f:
        for it, e, p in zip(items, enc, probs):
            f.write(json.dumps({"uid": it["uid"], "ntok": len(e), "p": [round(x, 6) for x in p]}) + "\n")
    return {"decisions": len(items), "seconds": round(secs, 1), "max_tokens": max(map(len, enc))}


def check(model, tok, items: list[dict], score, out: Path, n: int = 300) -> dict:
    """Max |Δp| between the batched results in `out` and one-at-a-time runs on a seeded sample."""
    batched = {json.loads(l)["uid"]: json.loads(l)["p"] for l in out.open(encoding="utf-8")}
    sample = random.Random(0).sample(items, min(n, len(items)))
    diffs, flips = [], 0
    for it in sample:
        e = tok(prompt("choice", it["state"], it["question"], it["options"]), add_special_tokens=False)["input_ids"]
        p = score(last_hidden(model, [e]), "choice", len(it["options"]))[0].cpu().tolist()
        q = batched[it["uid"]]
        diffs += [abs(a - b) for a, b in zip(p, q)]
        flips += p.index(max(p)) != q.index(max(q))
    return {"check_n": len(sample), "check_max_abs_dp": max(diffs),
            "check_mean_abs_dp": sum(diffs) / len(diffs), "check_argmax_flips": flips}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--zeroshot", help="also score the backbone without the adapter, under this name")
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    root = Path(a.model)
    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")][: a.limit]
    cfg = json.loads((root / "judge_config.json").read_text())
    ranges = cfg["slots"]["ranges"]
    temp = json.loads((root / "calibration.json").read_text())["per_kind"]
    meta = {"model": a.name, "path": str(root), "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "peft": peft.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0), "judge_config_version": cfg.get("model_version"),
            "template_version": cfg["slots"].get("template_version"), "temperatures": temp}

    tok = AutoTokenizer.from_pretrained(root)
    model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16, device_map="cuda").eval()

    if a.zeroshot:
        rows = model.get_output_embeddings().weight[cfg["verbalizer_ids"]].float()
        zs = scorer_lm_head(rows, ranges)
        zs_out = outdir / f"{a.zeroshot}.jsonl"
        zmeta = dict(meta, model=a.zeroshot, note="backbone without adapter; head = lm_head rows of the verbalizers")
        zmeta.update(run(model, tok, items, zs, zs_out, a.max_batch_tokens))
        zmeta.update(check(model, tok, items, zs, zs_out))
        (outdir / f"{a.zeroshot}.meta.json").write_text(json.dumps(zmeta, indent=2))
        print(json.dumps(zmeta), flush=True)

    model = PeftModel.from_pretrained(model, root, subfolder=cfg.get("adapter_subfolder", "adapter")).merge_and_unload()
    head = load_file(str(root / "head.safetensors"))
    sc = scorer_head(head["proj.weight"].cuda().float(), head["proj.bias"].cuda().float(), temp, ranges)

    e = tok(prompt("noul", *CARD_EXAMPLE, ["false", "true"]), add_special_tokens=False)["input_ids"]
    meta["card_example_p_true"] = round(sc(last_hidden(model, [e]), "noul", 2)[0, 1].item(), 4)
    print("card example P(true) =", meta["card_example_p_true"], "(card: 0.991)", flush=True)

    out = outdir / f"{a.name}.jsonl"
    meta.update(run(model, tok, items, sc, out, a.max_batch_tokens))
    meta.update(check(model, tok, items, sc, out))
    (outdir / f"{a.name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
