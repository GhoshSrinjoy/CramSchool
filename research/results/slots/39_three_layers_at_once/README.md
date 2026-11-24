# Reading the question at three layers at once, asked by generation

One run on 2026-11-24, on Qwen3.5-0.8B (`chain.sh`). The estimate from the layer profile (`../38_layer_sets/`) said that opening
where any of layers 20, 22 and 23 opens gains 0.023 of held-out recall on probe facts. That is detector openings, so it was asked
by generation before anything was built: three saved tables of bench40f's 40 facts, read at layers 20, 22 and 23, attached to the
model at the same time, each writing at the last layer (`scripts/score_table.py --also`). A fact then answers where any of the
three opens.

| bench40f, answers written at the last layer | Layer 20 | Layer 22 | Layer 23 | All three at once |
|---|---|---|---|---|
| Facts passing every check, of 40 | 33 | 35 | 32 | **35** |
| Recall replies holding the whole answer, of 480 | 395 | 421 | 413 | 430 |
| Facts failing a leak, garbling or damage check | 2 | 2 | 1 | 2 |

**The rule, written before the run:** confirmed if the three together pass at least 2 facts more than the best of the three alone,
with no more facts failing a leak, garbling or damage check than the worst of the three alone.

**Not met.** The three together pass 35, the same 35 facts as layer 22 alone: no fact is gained and none is lost. They add 9 recall
replies that hold the whole answer (10 gained, 1 lost), and no fact crosses its bar with them: the added replies all belong to
eight facts that already pass. Of the five facts that fail, three fail on recall and their recall did not move (0.67, 0.67 and
0.58, with 0.75 needed), and two fail the other-facts check. Reading at several layers is not built into one table.

**What it doesn't show:** a blind result, another bench, or a bench with facts that sit closer to their bar. Three tables at once
also build every fact three times.

`t20_22_23_all.jsonl` holds every fact's row; `t20_22_23_all.log` compares it with the layer-22 run.
