# A table that reads at the measured layer, and its grafted model

One run on 2026-11-24, on Qwen3.5-0.8B, of `scripts/check_weights_table.py`. A new table reads the question at the layer measured
for the model, layer 22. The grafted model's neurons sit at the last layer, 23, and need detectors that read there. So the export
builds the same facts a second time at the last layer and compiles those. This check asks whether that gives the grafted model of
a table built at the last layer in the first place.

| | Result |
|---|---|
| The default table's read layer | 22, read from the profile kept for the model |
| Tensors of its export equal to the last-layer table's export, of 488 | 488 |
| The same neurons for the same facts in `grafted.json` | yes |
| The export names the default table's own facts and its read layer | yes, so the weights' history links to that table |
| Facts answering from the table, and from the grafted model loaded from disk | 6 of 6, and 6 of 6 |
| A seventh fact added afterwards | one fact built at the last layer, not seven |

**What it shows:** the numbers published for the grafted model, which were measured on tables built at the last layer, hold for
the grafted model of a default table, since it is the same model. **What it does not show:** a pass rate. Each fact was asked one
question it was taught with.

`check.json` holds every reply from both forms.
