# Jury spec: checking the natively written Turkish decision items (bench v0)

The items were written by an AI model (Claude Opus 5.5) acting as a Turkish annotator. A jury of four models from other families (GPT-5.6 Sol, Gemini 4 Argon, Grok 4.7, Muse Wonder 1.0) checks them; the rules that combine their answers are in `bench/jury.py`. The jury's outputs are used only to keep, fix or drop test items; they never enter any training data.

## Round 1: blind answers

Input: `data/cache/tr-decide/bench/jury/blind_<batch>.jsonl`, one item per line with `id`, `state`, `question`, `options` (no gold, no rationale).

For every item, read the state carefully as a native Turkish reader would, answer the question, and write one JSON object per line to the output file named in your task:

| field | type | meaning |
|---|---|---|
| `id` | string | the item id |
| `answer` | integer | 0-based index of the option you choose |
| `confidence` | number | your probability (0–1) that your answer is correct |
| `ambiguous` | boolean | true if a careful native reader could defend two or more options |
| `natural` | integer | 1–5: how natural the Turkish is for its intended style (5 = a real person would write exactly this; deliberately broken styles such as missing Turkish characters or dictation count as natural if they are realistic) |
| `note` | string | at most 20 words, only if something is wrong (ambiguity, factual error, unnatural Turkish, personal data, real brand); otherwise empty |

Answer every item; do not skip. Do not look anything up.

## Round 2: review of disputed items

Input: `data/cache/tr-decide/bench/jury/review_<batch>.jsonl` with the full items including `option_descriptions`, `gold` and `rationale`, plus the round-1 answers of the jury and the reasons the item was disputed.

Models are asked every item twice: with the option labels only, and with "label: description" options. Judge the gold for the labels-only case, which is the stricter one: the descriptions are context for you, not part of what every model sees.

For every item decide:

| field | type | meaning |
|---|---|---|
| `id` | string | |
| `gold_correct` | boolean | the stated gold is a correct answer |
| `gold_unique` | boolean | no other option is equally defensible |
| `fix` | string | if the item can be repaired with a small edit to the state, question or options, describe the edit in Turkish (at most 40 words); otherwise empty |
| `verdict` | string | `keep`, `fix` or `drop` |

Keep an item only if the gold is both correct and unique. Hard items are welcome: a jury member answering wrongly in round 1 is not a reason to drop an item whose gold is correct and unique.

## Round 3: blind recheck of fixed items

Items repaired after round 2 are asked again without their gold, in the round-1 format, by two judges. A fixed item stays only if both pick its corrected gold.
