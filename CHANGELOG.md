# Changelog

What changed between versions of the package. The evidence behind each change is in the [lab notebook](docs/lab-notebook.md) and [benchmarks](docs/benchmarks.md).

## Unreleased

### Added

- **The read layer can be measured.** `Graft(model, tokenizer, layer="auto")`, `cramschool graft add --layer auto` and `cramschool graft layers` profile every layer of a model on invented probe facts and pick where a table reads the question. On Qwen3.5-0.8B the pick is layer 22: all 40 facts of a bench pass there against 32 at layer 20. One bench, not blind. `layer=None` is unchanged.
- `cramschool.graft.profile_layers` and `LayerProfile`.

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
