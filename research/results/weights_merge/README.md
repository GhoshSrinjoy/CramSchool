# Two grafted models merged at the weights, checked on the real model

One run on 2026-11-24, on Qwen3.5-0.8B, of `scripts/check_weights_merge.py`. It builds two tables of 18 facts from bench40h, exports
each as a grafted model, merges the two grafted models with no table (`merge_grafted_models`), and compares the merged checkpoint
with one export of the two tables merged. The write-up is in the
[Python API](../../../docs/python-api.md#two-grafted-models-into-one).

| | 3 + 3 facts | 18 + 18 facts |
|---|---|---|
| Fact neurons in the merged model | 44, all at the last layer | 256 at the last layer, 64 at layer 20 |
| Later answer tokens that had to move to layer 20 | 0 | 64 |
| Tensors equal to the one export's, of 488 | 488 | 488 |
| Facts that answer from the merged folder, loaded from disk | 6 of 6 | 36 of 36 |
| The same for the one export | 6 of 6 | 36 of 36 |

`check.json` holds both cases: the neurons, how the last layer was shared out, the hashes of the moments, every reply, and for the
first case the weights history the merged model was recorded in.

**What it shows:** for two models exported the same way, on the same computer, the merge gives the weights one export gives.
**What it does not show:** a pass rate (each fact was asked one question it was taught with), a merge of more facts than one export
was measured with, or another model.

Run it again (GPU, about 35 minutes):

```bash
python scripts/check_weights_merge.py models/Qwen3.5-0.8B benchmarks/bench40h.json runs/weights_merge runs/weights_merge/check.json
```
