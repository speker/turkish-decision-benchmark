---
language:
- tr
- tur
license: cc-by-4.0
pretty_name: Turkish Decision Benchmark
size_categories:
- 1K<n<10K
task_categories:
- multiple-choice
- text-classification
task_ids:
- multiple-choice-qa
source_datasets:
- original
language_creators:
- machine-generated
annotations_creators:
- machine-generated
tags:
- synthetic
- benchmark
- turkish
- decision
- calibration
- robustness
configs:
- config_name: programmatic
  data_files:
  - split: test
    path: data/programmatic/*.parquet
- config_name: written
  data_files:
  - split: test
    path: data/written/*.parquet
extra_gated_prompt: >-
  This benchmark is for evaluating models. Its written part was produced with an AI model whose terms forbid using
  the output to train competing models, and test items that leak into training data stop measuring anything.
extra_gated_fields:
  I will not use this dataset to train or fine-tune a model: checkbox
  I will not republish the items or their answers outside this repository: checkbox
---

# Turkish Decision Benchmark

{{TOTAL}} multiple-choice decisions about short Turkish texts. A model reads a customer message, an e-mail, an
invoice note or a log line and picks one option: where to route it, whether a rule applies, which date is the due
date, what the writer is asking for. Two things are measured: whether a model reads the signals that carry the
decision in Turkish (negation and polite requests, the -DI/-mIş evidential suffixes, decimal commas and thousand dots,
official holidays), and how reliable its probabilities stay when the text also contains an instruction aimed at the
model or a long irrelevant prefix.

| Config | Items | Answers come from |
|---|---:|---|
| `programmatic` | {{PROG_N}} | code that computes every answer from the facts written into the item |
| `written` | {{WRITTEN_N}} | an AI writer, checked by a four-model jury |

The programmatic task has 11 families: due dates, weekday arithmetic, business days with the 2026–2027 Turkish
official holidays (religious holidays from the Diyanet calendar; half-day eves never fall inside a window), VAT and
discounts in Turkish number format (only amounts where rounding each line and rounding the total agree), comparing
amounts written with dots, commas or in words, five-way negation sets on one verb ("iptal edin", "iptal etmeyin",
"iptal eder misiniz?", "iptal etmeyi unutmayın", "iptal etmenize gerek yok"), -DI against -mIş pairs on one event,
an invented return policy with missing-fact variants, counting, planted instructions that always name a wrong option,
and return-policy items placed after at least 150, 400 or 900 words of an unrelated company newsletter. The last two
families are paired with their clean originals through `pair_id`, so the damage they do can be measured per item.

The written task covers {{DOMAINS}} domains (banking, telecom, municipal services, logistics, HR, travel, health
appointments and more) in {{STYLES}} registers, from formal e-mails to chat typed without Turkish characters,
dictation and code-mixed Turkish. Identifiers appear as typed placeholders such as `<TCKN>` and `<IBAN>`. The
generator, the writer and jury prompts and the evaluation harness are at
https://github.com/speker/turkish-decision-benchmark.

## How the items were made and checked

Claude Opus 5.5, prompted as a native Turkish annotator, wrote 8 batches of 100 items. Four models from other
families then checked every item: GPT-5.6 Sol, Gemini 4 Argon, Grok 4.7 and Muse Wonder 1.0.

1. Each judge answered every item without seeing the gold, marked ambiguity and naturalness, and could leave a note.
2. Any disagreement, ambiguity mark or note sent the item to review, where the four judges saw the gold and the
   rationale and decided for the labels-only reading whether the gold was correct and the only defensible answer.
3. Items with at least two "fix" votes were repaired by the writer model (by narrowing an overlapping wrong option or
   adding one fact, never by editing the correct option) and asked again without the gold; both re-check judges had
   to pick it.

Of the {{WRITTEN_N}} public items, {{UNDISPUTED}} were undisputed, {{KEPT}} were kept after review and {{FIXED}} were
fixed and re-checked; {{DROPPED}}. In round one the four judges agreed with the gold on {{JURY_AGREE}} of
their {{JURY_ANSWERS}} answers. One judge invented 22 items during a re-check; that file was rejected and the
pipeline now refuses answer files with unknown ids. The eighth batch (100 items, same pipeline) and a second
programmatic set drawn with a secret seed are kept private for later contamination checks.

The programmatic answers were checked the same way: three judges from three model families answered a stratified
sample of the items without the gold, and {{PROG_JURY}}. Human check: {{HUMAN}}

Exact duplicates (same text and question): {{EXACT}}. Near-duplicates among the written items (MinHash over word
5-grams, estimated Jaccard of 0.6 or more): {{NEAR}}. Overlap with public data: {{DECONTAM}} MASSIVE 1.1 tr-TR,
SIB-200 or Belebele (tur_Latn), or the Turkish Decision Benchmark of erenextlabs (revision `0662f85`). A scan for
personal data (TCKN with checksum, TR IBAN, +90 phone numbers, e-mail addresses, IP addresses) found nothing.

## Results

Each item is asked in two option orders, and written items once more with "label: description" options. Accuracy is
on the first order, with a 95% interval from 2,000 item-level bootstrap draws (seed 0); on the programmatic task it
leaves out the planted-instruction and long-prefix copies, which are reported as damage instead. "Reading" covers
negation, evidentials, the return policy and number formats; "Arithmetic" covers dates, weekdays, business days, VAT
and counting. ECE uses 15 equal-width bins of the top probability. "Order flips" is the share of items whose chosen
option text changes when the options are shuffled. "Planted" and "Prefix" are the share of items answered correctly
in their clean form and wrongly once the planted instruction or the long prefix is added. All runs are deterministic
(no sampling); the per-decision probabilities are in `results/<model>/<date>/`.

Written task:

{{WRITTEN_TABLE}}

Programmatic task:

{{PROG_TABLE}}

Each model was run through the interface it ships with:

{{HOW}}

## Limitations

- The written texts are synthetic. They were written to look like real messages, but no real customer wrote them,
  and the jury that checked them is made of AI models.
- The four frontier models that served as judges agreed with the gold on {{JURY_AGREE_PCT}} of their round-one
  answers, so the written task is close to the ceiling for models of that class and separates smaller and
  decision-tuned models more. The programmatic task is harder for {{HARDER}}.
- The numbers compare models as deployed (a decision API, a classification head or the next-token probabilities of
  the option letters), not as architectures.
- Holiday logic covers 2026 and 2027 only. Once a year the programmatic task is regenerated with the next
  year's calendar and a new seed, and the two private sets are used to check the public ones for contamination.
- Evidential items test how a writer presents an event (-DI as known, -mIş as reported or inferred), which is the
  textbook contrast; everyday Turkish sometimes uses -DI for things learned second-hand.

## Canary and citation

Every row carries this string; please keep it in any copy so that leaks into training corpora can be detected:

`{{CANARY}}`

```bibtex
@misc{peker2026turkishdecision,
  title  = {Turkish Decision Benchmark},
  author = {Peker, Serkan},
  year   = {2026},
  url    = {https://huggingface.co/datasets/serkanpeker/turkish-decision-benchmark}
}
```

## Türkçe özet

Kısa Türkçe metinler üzerinde {{TOTAL_TR}} çoktan seçmeli karar sorusu. Model bir müşteri mesajını, e-postayı ya da
kayıt satırını okuyup tek bir seçenek seçiyor. İki görev var. `programmatic` görevindeki {{PROG_N_TR}} sorunun cevabını
kod hesaplıyor; tarih, iş günü ve resmî tatil, KDV, sayı biçimi, olumsuzluk, -dı/-mış, iade kuralı, sayma, metne
gömülmüş talimat ve uzun alakasız giriş aileleri var. `written` görevindeki {{WRITTEN_N_TR}} soruyu bir yapay zekâ
modeli yazdı; dört farklı model ailesinden jüri doğru cevabı görmeden çözdü, tartışmalı sorular incelendi, düzeltilen
sorular yeniden kör sorgudan geçti. {{HUMAN_TR}} Veri yalnız değerlendirme içindir, eğitimde kullanılmaz.
