# The check of the default read layer: bench40f at the pick

One run on 2026-11-24, on Qwen3.5-0.8B (`chain.sh`). A new table now reads the question at the layer measured for the model,
layer 22. On bench40h that layer passes 40 of 40 facts. bench40f had been built at layers 20 and 23 only, so the default was
checked on it, with a kill line written before the run: fewer than 32 of 40, what the last layer passes, and the default goes back
to the last layer.

| bench40f, 40 facts, answers written at the last layer | Layer 20 | Layer 22 | Layer 23 |
|---|---|---|---|
| Facts passing every check | 33 | **35** | 32 |
| Recall replies holding the whole answer, of 480 | 395 | 421 | 413 |
| Facts failing a leak, garbling or damage check | 2 | 2 | 1 |

- **It passes 35, so the default stands.** Four facts pass only at layer 22 (f09, f14, f21, f37) and one only at layer 23 (f05,
  which newly fails the other-facts check).
- **Not blind:** I had read this bench's replies before. The layer was picked on invented probe facts, by a rule fixed before.
- **Built as the tables at layers 20 and 23 were:** `scripts/validate_api_facts.py` with `--layer 22`, then
  `scripts/score_table.py --last-layer all`. `build_layer22.log` also scores the table as built, with the answer pushed at layer
  22: 17 of 40. A push that late loses, which is why the layer is measured only for a table that writes at the last layer.

`t22_all.jsonl` holds every fact's row; `t22_all.log` compares it with the layer-23 run.

A first run of the build stopped at the thirteenth fact with a CUDA error, when two other processes on the same GPU were stopped
by force. It was started again from the beginning, and nothing of the first run is kept.
