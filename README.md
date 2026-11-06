# Weightspace update programming

Experiments in changing what a language model knows, or how it behaves, by editing its weights directly instead of fine-tuning it.

The main result is **[weightspace](weightspace/)**. In this package a larger model acts as the editor:

- it observes a small model with forward passes only;
- it edits the weights;
- it checks every edit against the unedited model;
- it keeps the edit, or undoes it exactly.

Every step goes into a hash-chained log.

**Status: research prototype.**

- **Behaviour fix:** removing looping on maths problems held up on problems that were not used to build it. On Qwen3.5-0.8B, correct answers went from 1 to 7 of 32: 6 rescued, 0 broken.
- **Facts:** most written facts still fail their checks. With the current recipe, 2 of 10 new facts passed. The one shipped, RX Systems → Julian Rodlauer:
  - recalls on 8 of 16 new wordings in its last check;
  - no longer gives its first name where other companies are asked for a name;
  - still gives the whole name to a company with a similar name ("RX Studios"), and to one other company.

The measurements are in [weightspace/README.md](weightspace/README.md#what-was-measured).

| Folder | What it holds |
|---|---|
| [`weightspace/`](weightspace/) | The package, its tests, the experiments and their raw results, the session logs, and an exported patch for Qwen3.5-0.8B |
| [`prototype/`](prototype/) | The first scripts: activation hooks and a store of facts on Phi-3-mini. Their hooks attached to modules with no usable weight, so no weight changed. `weightspace` edits each layer's `mlp.down_proj` instead |

## Try it

```bash
cd weightspace
pip install -e ".[dev,mcp]"
pytest
```

The tests run on tiny random models on the CPU. Real edits need a model you download separately, such as Qwen3.5-0.8B. Base model weights are not included; the patch in `weightspace/patches/` applies only to Qwen3.5-0.8B.
