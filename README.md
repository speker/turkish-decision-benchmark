# Turkish Decision Benchmark: code

Generator, jury pipeline, scorer and harness of the [Turkish Decision Benchmark](https://huggingface.co/datasets/serkanpeker/turkish-decision-benchmark), a set of 3,518
multiple-choice decisions about short Turkish texts. The dataset card has the tasks, the checks and the
results of ten models. This repository holds no written items and no model outputs; both are in the gated
dataset.

## Reproduce

```bash
uv sync
cd projects/tr-decide
uv run python -m bench.prog                 # the 2,820 programmatic items, identical to the dataset's
uv run python -m bench.from_hub             # the dataset's items and per-decision outputs (needs access)
uv run python -m bench.to_items prog
uv run python -m bench.to_items native
uv run python -m bench.score prog --results ../../data/cache/tr-decide/bench/hub_results
uv run python -m bench.score native --results ../../data/cache/tr-decide/bench/hub_results
uv run python -m bench.score prog --results ../../data/cache/tr-decide/bench/hub_results --paired d3 pusula
```

The scorer validates every result file first: no missing, extra or duplicate decision, and one finite,
non-negative probability per option. Intervals come from 2,000 item-level bootstrap draws with seed 0. Working
files go to `data/cache/tr-decide/bench/`; set `TR_DECIDE_BENCH` to use another folder.

To evaluate a model with inspect-ai, the dataset's `eval.yaml` is enough:
`inspect eval hf/serkanpeker/turkish-decision-benchmark --model <model>`. The scripts in `remote/` are the
harness that produced the published outputs (vLLM 0.33.0 servers, decision endpoints, next-token probabilities
of the option letters); paths in them are those of the measurement machine.

## Pipeline

- `bench/prog.py`: programmatic items; answers are computed from the facts in each text. Official holidays
  for 2026 and 2027 are in the file.
- `bench/WRITER_SPEC.md`, `bench/JURY_SPEC.md`, `bench/jury.py`: how the written items were requested and
  checked (blind answers, review, repair and blind re-check), `bench/curate_native.py` for one wording rule.
- `bench/gate.py`: exact and MinHash near-duplicates, and word 8-gram overlap with MASSIVE 1.1 tr-TR
  (amazon-massive-dataset-1.1.tar.gz), SIB-200 (`Davlan/sib200`), Belebele (`facebook/belebele`) and
  `erenextlabs/turkish-decision-benchmark`; reports in `reports/`.
- `bench/build_hub.py`: the dataset folder and the card, every number read from files.
- `bench/human_form.py`: the 40-item human check and its agreement score; `bench/human_drops.json` lists the
  two items removed after it, and `reports/human.json` holds the readers' agreement.

Code: MIT. The dataset is CC BY 4.0 on the Hub. Benchmark items carry the canary string in `projects/tr-decide/CANARY`;
please keep it in any copy so that leaks into training data can be found.

## Türkçe özet

Türkçe karar benchmark'ının kodu: programatik soru üreticisi, jüri hattı, puanlayıcı ve ölçüm düzeneği.
Yazılmış sorular ve model çıktıları bu repoda yok, Hub'daki erişim onaylı veri kümesinde. Programatik sorular
`bench/prog.py` ile birebir yeniden üretilir; sıralama tablosu Hub'daki ham çıktılardan yukarıdaki komutlarla
yeniden hesaplanır.
