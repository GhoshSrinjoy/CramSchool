# Changelog

What changed between versions of the package. The evidence behind each change is in the [lab notebook](docs/lab-notebook.md) and [benchmarks](docs/benchmarks.md).

## Unreleased

### Added

- **A table has a history.** `graft.commit(path, message)` saves the table and records it as a version. Any version loads back as it was saved (`Graft.load(path, model, tokenizer, version="v2")`), two versions can be compared (`history.diff`), and `graft.revert(path, "v1")` goes back to one. Nothing is built again, a revert is itself recorded, and each line of the log carries the hash of the line before it (`history.is_intact`).
  - **Storage:** inside the table's folder, in `history/`. A file that versions share is stored once.
  - **Also:** `graft.merge(other)` takes another table's facts without building them, for two tables of the same model, layer, system message, wordings and build options; `graft.bisect(path, test, good)` finds the first version at which a test of yours fails.
  - **Command line:** `cramschool graft commit`, `log`, `diff` and `revert` need no model; `graft bisect` asks a question of each version it loads.
  - **For editors:** the tools `commit`, `versions`, `diff` and `revert`, and the same four over MCP.
  - **What it is not:** a history of a table of facts, not of a model's weights. No pass rate changes.
  - **Checked:** on tiny random models in the tests, and once end to end on Qwen3.5-0.8B (see the [Python API](docs/python-api.md#a-tables-history)).
- **The model's own weights have a history too.** `cramschool.weight_history.commit(history_dir, checkpoint_dir, message, base_dir=..., table=...)` records a grafted model's checkpoint as a version: only the tensors that differ from the base checkpoint, each stored once, with `grafted.json` and the name switch. `checkout` writes any version back as a checkpoint folder, byte for byte; `diff` gives the tensors and the facts that changed; `revert` is a checkout recorded as a new version.
  - **Linked to the table's history:** with `table=`, a version of the weights names the table version it was compiled from, and is refused if the two hold other facts.
  - **Loaded models:** `commit_model` records a loaded model's weights as they are, for weight edits kept in memory; `apply` sets a loaded model to a version in place and swaps the name switch with the weights.
  - **Command line:** `cramschool weights commit`, `log`, `diff`, `checkout` and `revert`, none of which loads a model.
  - **What it is not:** nothing is compiled or merged here. A change to the facts is made in the table, exported and recorded in both histories. No pass rate changes.
  - **Checked:** on tiny random models in the tests, and once end to end on Qwen3.5-0.8B (see the [Python API](docs/python-api.md#the-weights-history)).
- **Both forms in one call.** `graft.commit_grafted(path, grafted_dir, history_dir, message)` records the table, exports it into the model's own weights and records those weights, whose version names exactly the table version recorded in the same call. `graft.revert_grafted(path, version, grafted_dir, history_dir)` goes back in both; it compiles nothing when that table version was exported before.
  - **Command line:** `cramschool graft export --history H -m MESSAGE` does the same, and `cramschool weights bisect` finds the version of the weights at which a reply changed.
  - **For editors:** the tools `commit_grafted`, `revert_grafted`, `weights_versions`, `weights_diff` and `weights_checkout`, and the same five over MCP.
  - **`weight_history.bisect`** finds the first version of the weights at which a test fails, in memory, and puts the model back bit for bit.
  - **The result names the facts that are not in the weights.** A fact whose detector could not be built gets no neuron; it answers from the table only.

### Changed

- **A version of the weights names a table version only when the table is recorded.** Recorded by separate calls, the saved table must have no changes since its newest version, and that version must hold exactly the grafted model's facts.
- **A version of the weights is written back one weight file at a time,** after the whole base is checked, so a model in several files is never held whole in memory.
- **The docs read more easily.** [Getting started](docs/getting-started.md) is a walk-through in nine steps, with the real reply of each step, including a wording that is missed and a question that leaks. The [Python API](docs/python-api.md) has one short example and one small table per task.
- **Paths of my computer are gone from the kept result files.** In 38 files (logs, run scripts, briefs, five scripts and two docstrings) they are replaced by neutral ones, line for line. Five diagnostic scripts take the model's folder from `CRAMSCHOOL_MODEL`.

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
