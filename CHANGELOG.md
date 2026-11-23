# Changelog

What changed between versions of the package. The evidence behind each change is in the [lab notebook](docs/lab-notebook.md) and [benchmarks](docs/benchmarks.md).

## 1.2.0 (2026-11-23)

### Changed

- **The grafted model writes where a write always wins.** `export_grafted` on a table at the model's last layer puts each fact's first-token neurons there, with the row the table itself writes, and as many later tokens as the last layer can afford (1/14 of its neurons); the rest go to an earlier layer. A table of up to about 27 facts edits one layer.
  - **Measured:** 32 of 40 facts pass on bench40f against 23 for the earlier compile, where its table passes 32; 35 on bench40h, where its table passes 38. No firmly known fact is lost. Neither bench was blind.
  - **The cost:** 6 to 8 of the 92 facts the model only just knows flip, against 4, and drift is about 0.010 against 0.0027.
  - **A table at another layer** is compiled as before, with a warning that says how to get the new compile.
- **A new table reads the question at the model's last layer.** `layer=None` used to mean layer 20 of a 24-layer model and the same depth of any other. For a table that writes at the last layer it now means the last layer. Over two benches such a table passes 70 of 80 facts against 65 at layer 20 (not blind). A table that pushes at its own layer (`last_layer=None`) keeps the earlier rule, and a saved table keeps its layer.

### Added

- **The read layer can be measured.** `Graft(model, tokenizer, layer="auto")`, `cramschool graft add --layer auto` and `cramschool graft layers` profile every layer of a model on invented probe facts and pick where a table reads the question. On Qwen3.5-0.8B the pick is layer 22: all 40 facts of a bench pass there against 32 at layer 20. One bench, not blind.
- `layer="last"`, `cramschool.graft.profile_layers` and `LayerProfile`.
- `export_grafted(later_layer=..., last_budget=...)`; in `cramschool.graft.grafted`: `compile_layer(last=..., kinds=..., next_keys=...)`, `last_layer_plan`, `join_layers`, and a name switch that sets neurons at two layers.
- `scripts/damage_profile.py`: how many neurons each layer can give up before the damage bar fails. `scripts/check_last_layer_export.py`: the export, end to end on the real model.
- A page on [inputs and outputs](docs/inputs-and-outputs.md).

### Measured and not built

- A name match inside the weights. A linear reading of the subject at the question's last token passes the bench's bars for 0 of 12 companies at any layer, so the grafted model keeps its name switch.

## 1.1.0 (2026-11-23)

### Changed

- **A new table writes its answers at the model's last layer.** `GraftOptions.last_layer` now defaults to `"all"`. An answer that opens comes out whole, where a push at the table's layer could lose to the model's own next word.
  - **Measured:** on three benches of 40 facts, the same saved tables pass 32, 37 and 33 facts with it against 23, 32 and 24 without, with no fact lost. None of the three was blind when scored.
  - **The cost:** an answer that opens on a question which isn't the fact's own is finished too.
  - **To get the earlier write:** `GraftOptions(last_layer=None)`.
- **A table saved with 1.0 is not changed by loading it.** It keeps the write it was built with. To switch it over, set `graft.options.last_layer = "all"` and save it.
- **Every published number except step 24's was measured with `last_layer=None`.** `cramschool bench run` and the reproduction scripts still use that write.

### Added

- `GraftOptions(last_layer="all" | "later" | None)`: where an answer's tokens are written.
- `GraftOptions(prompt_detector=True)`: a second detector for the prompt's last token, which recognises more ways of asking. For tables with at most two facts on a company. Off by default.
- `Graft(system=...)`: a table taught and asked behind an application's system message.
- `cramschool --version`.
- Command-line flags for a table's settings: `--last-layer`, `--prompt-detector`, `--system` and `--fast` on `graft add` and `graft add-file`, `--last-layer` on `graft ask`, and `cramschool graft set` for a saved table.
- A [user guide](docs/user-guide.md) and four runnable [examples](examples/).
- `scripts/check_default_write.py`: the check of the new default on the real model.

### Fixed

- Transformers 5.24 passes the generation cache under another name; the inserted layer now reads both.
- Building a fact behind a system message, or from long wordings, could fill a 24 GB card and stall without an error. The answer-vector step no longer copies every position's logits; its numbers are identical.

### Not in this version

- The grafted model (`export_grafted`) has neither the write at the last layer nor the end-of-prompt detector. It warns when a table uses them.

## 1.0.0

The first packaged version: GRAFT tables (`Graft`), the grafted model (`export_grafted`, `load_grafted`), editor models, the MCP server, datasets, the bench tool and the weight-edit instruments.
