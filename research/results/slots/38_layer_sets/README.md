# Reading the question at several layers at once: an estimate from the layer profile

One run on 2026-11-24, on Qwen3.5-0.8B: `cramschool graft layers --model models/Qwen3.5-0.8B --out profile.json`. The profile
builds 12 invented probe facts at every layer and holds out half of each relation's wordings. For every probe fact and layer it
knows on which held-out questions the fact's detectors open, so two ways of reading at a set of layers can be scored with no new
run: "any" (open if the detectors open at any layer of the set) and "all" (only if they open at every one). A set of two or three
layers is picked on the held-out items at even places and scored on those at odd places, against the single layer picked the same way.

| Read at | Held-out recall | Held-out leak |
|---|---|---|
| Layer 22 alone | 0.902 | 0.012 |
| Layers 20, 22 and 23, open if any opens | 0.924 | 0.016 |
| Layers 21 and 22, open only if both open | 0.890 | 0.011 |

The gain of "any" is +0.023 of recall, with a standard error of 0.009 over the probe facts. The rule written before the run asked
for a leak within 0.02 of the single layer's and a gain of more than two standard errors: it is met. "All" loses recall.

**What it is:** detector openings on probe facts. **What it is not:** replies. Asked by generation on bench40f, with three tables
attached at once, the same three layers pass 35 of 40 facts, as layer 22 alone does (`../39_three_layers_at_once/`).

`profile.json` is the whole profile (per layer, per probe fact, and `sets`); `profile.txt` is what the command printed.
