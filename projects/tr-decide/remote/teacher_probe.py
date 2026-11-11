"""h200: zero-shot `choice` distributions from a generative teacher served by vLLM (chat API, one generated token,
restricted to the option letters, read back as log-probabilities over those letters).

The server must run with `--logprobs-mode processed_logprobs` so the returned log-probabilities respect
`allowed_token_ids`. The instruction is in the language of the item.

    .venv/bin/python teacher_probe.py --url http://127.0.0.1:8500 --model-dir ~/models/qwen3.8-27b \
        --name qwen3.8-27b --items probe_massive_dev.jsonl --chat-kwarg enable_thinking=false
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

LETTERS = "ABCDEFGHIJKLMNOP"
TEXT = {
    "en": ("Text", "Answer with the letter of the correct option only."),
    "tr": ("Metin", "Yalnızca doğru seçeneğin harfini yaz."),
}


def prompt(it: dict) -> str:
    label, instruction = TEXT["en" if it["cond"] == "en" else "tr"]
    options = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(it["options"]))
    return f"{label}: {it['state']}\n\n{it['question']}\n\n{options}\n\n{instruction}"


def post(url: str, body: dict, tries: int = 5, endpoint: str = "chat/completions") -> dict:
    data = json.dumps(body).encode()
    for n in range(tries):
        try:
            req = urllib.request.Request(f"{url}/v1/{endpoint}", data=data,
                                         headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except Exception:
            if n == tries - 1:
                raise
            time.sleep(2 ** n)
    raise AssertionError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--served-name")
    ap.add_argument("--items", default="probe_massive_dev.jsonl")
    ap.add_argument("--outdir", default="probe")
    ap.add_argument("--chat-kwarg", action="append", default=[],
                    help="chat template option as key=value (true/false become booleans), repeatable")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--encoding", choices=["chat", "dsv4"], default="chat",
                    help="chat: the server's chat template; dsv4: DeepSeek-V4's own encoder (encoding/encoding_dsv4.py "
                         "in the model directory, thinking_mode='chat'), sent to /v1/completions")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(Path(a.model_dir).expanduser())
    letter_ids = []
    for ch in LETTERS:
        ids = tok.encode(ch, add_special_tokens=False)
        assert len(ids) == 1, (ch, ids)
        letter_ids.append(ids[0])
    served = a.served_name or json.loads(urllib.request.urlopen(f"{a.url}/v1/models", timeout=60).read())["data"][0]["id"]
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")][: a.limit]
    chat_kwargs = {}
    for kv in a.chat_kwarg:
        key, value = kv.split("=", 1)
        chat_kwargs[key] = {"true": True, "false": False}.get(value.lower(), value)

    encode = None
    if a.encoding == "dsv4":
        import sys

        sys.path.insert(0, str(Path(a.model_dir).expanduser() / "encoding"))
        from encoding_dsv4 import encode_messages

        encode = lambda text: encode_messages([{"role": "user", "content": text}], thinking_mode="chat")  # noqa: E731

    # The model's generation_config (top_k, top_p, ...) would otherwise filter the processed logprobs.
    sampling = {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0, "return_tokens_as_token_ids": True}

    def one(it: dict) -> dict:
        k = len(it["options"])
        ids = letter_ids[:k]
        if encode is None:
            r = post(a.url, {"model": served, "messages": [{"role": "user", "content": prompt(it)}], "logprobs": True,
                             "top_logprobs": k, "allowed_token_ids": ids, "chat_template_kwargs": chat_kwargs,
                             **sampling})
            top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            lp = {int(t["token"].split(":")[1]): t["logprob"] for t in top}
        else:
            r = post(a.url, {"model": served, "prompt": encode(prompt(it)), "add_special_tokens": False,
                             "logprobs": k, "allowed_token_ids": ids, **sampling}, endpoint="completions")
            top = r["choices"][0]["logprobs"]["top_logprobs"][0]
            lp = {int(t.split(":")[1]): v for t, v in top.items()}
        z = [lp.get(i, -1e9) for i in ids]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        return {"uid": it["uid"], "p": [round(x / sum(e), 6) for x in e], "missing": sum(i not in lp for i in ids)}

    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True)
    t0 = time.time()
    missing = 0
    with ThreadPoolExecutor(a.workers) as ex, (outdir / f"{a.name}.jsonl").open("w", encoding="utf-8") as f:
        for row in ex.map(one, items):
            missing += row.pop("missing")
            f.write(json.dumps(row) + "\n")
    meta = {"model": a.name, "served": served, "model_dir": str(a.model_dir), "encoding": a.encoding,
            "chat_kwargs": chat_kwargs,
            "letter_ids": letter_ids, "instruction": TEXT, "decisions": len(items),
            "letters_missing_from_top_logprobs": missing, "seconds": round(time.time() - t0, 1)}
    (outdir / f"{a.name}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
