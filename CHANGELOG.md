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
  - **What it is not:** nothing is compiled here. A change to the facts is made in the table, exported and recorded in both histories. No pass rate changes.
  - **Checked:** on tiny random models in the tests, and once end to end on Qwen3.5-0.8B (see the [Python API](docs/python-api.md#the-weights-history)).
- **Both forms in one call.** `graft.commit_grafted(path, grafted_dir, history_dir, message)` records the table, exports it into the model's own weights and records those weights, whose version names exactly the table version recorded in the same call. `graft.revert_grafted(path, version, grafted_dir, history_dir)` goes back in both; it compiles nothing when that table version was exported before.
  - **Command line:** `cramschool graft export --history H -m MESSAGE` does the same, and `cramschool weights bisect` finds the version of the weights at which a reply changed.
  - **For editors:** the tools `commit_grafted`, `revert_grafted`, `weights_versions`, `weights_diff` and `weights_checkout`, and the same five over MCP.
  - **`weight_history.bisect`** finds the first version of the weights at which a test fails, in memory, and puts the model back bit for bit.
  - **The result names the facts that are not in the weights.** A fact whose detector could not be built gets no neuron; it answers from the table only.

- **Kinds of subject ship with the package.** Eight are ready to use: person, town, product, work, concept, institution, medicine and event, as the blind bench beyond companies drafted them. `graft.add_kind(...)` adds another, and so do the editor tool `add_kind` and its MCP twin. Two more, software and substance, were drafted and tried once; by the rule written before that check they do not ship, and their data is kept in `research/results/editor_any_subject/`. A fact about a company is built bit for bit as before.
- **Editors can teach facts about any kind of subject.** The protocol says what a fact is, lists the kinds, and ends by recording both forms. Over the server, `add_relation` takes the kind of its subject, which it could not before.
- **`graft.merge_grafted(other, ...)`** brings another table's facts into this table and into the model's own weights in one call: it merges the tables and compiles once. `cramschool graft merge` does it from the command line. For two grafted models without their tables, see the next entry.
- **Two grafted models merge at the weights.** `merge_grafted_models(model, tokenizer, base_dir, mine, theirs, out_dir)` takes two grafted model folders and writes one. No table is needed. Each fact's neurons are copied out of the two checkpoints, and the neurons to reuse are chosen and the rest re-fitted once, by the code an export runs.
  - **The same weights as one export.** For two models exported the same way, the merged checkpoint is the one a single export of all the facts gives, tensor for tensor. The result says so (`same_as_one_export`), and says why when it is not: another device or number type, other ordinary text, other forms, another budget.
  - **At any size an export handles.** When the two models need more of the last layer's neurons than it can spare, later answer tokens move to the other layer, as in one export. An export now keeps what that needs beside the weights, in `grafted_later.bin`.
  - **Everywhere:** `weight_history.merge` records the result as a version; `cramschool weights merge` does it from the command line; the editor tool `weights_merge` and its MCP twin do it on the server's model.
  - **Refused:** another base model, two models whose neurons work differently, a fact both hold with another answer unless `prefer` says whose, and an export from before `grafted_later.bin` when one of its tokens would have to move.
  - **Checked on Qwen3.5-0.8B:** 3 + 3 facts, and 18 + 18 facts that go over the last layer's budget together (64 tokens moved): in both, all 488 tensors equal the one export's, and every fact answers from the merged folder ([Python API](docs/python-api.md#two-grafted-models-into-one)).
- **The MCP page is a reference.** [`docs/mcp.md`](docs/mcp.md) gives the endpoint, every flag of `cramschool serve`, how each client connects, a session run with Opus through Claude Code, and every tool with its arguments. The tool tables are made from the server's own schemas (`scripts/mcp_reference.py`), and a test fails when the page falls behind the server.
- **`cramschool serve` takes the settings of a new table:** `--layer`, `--system` and `--prompt-detector`. Before, a table made by the server could only be given `--fast`.
- **The layer profile also estimates reading at several layers at once** (`LayerProfile.sets`, printed by `cramschool graft layers`). On Qwen3.5-0.8B, opening where any of layers 20, 22 and 23 opens gains 0.023 of held-out recall on probe facts. Asked by generation with three tables attached at once, it passes the same 35 of 40 facts as layer 22 alone, so it is not built into one table ([benchmarks](docs/benchmarks.md#reading-the-question-at-several-layers-at-once)). `scripts/score_table.py --also` attaches more tables for such a run.
- **A version of the weights can be asked without leaving the model changed.** `weight_history.applied(...)` sets a loaded model to a version for a `with` block and puts it back bit for bit; `cramschool weights ask` loads the model once and asks several versions; the editor tool `weights_ask` does it on the server's model.

- **A first check on facts about software and science, through the editor tools.** An editor model taught twelve invented facts over a running server and recorded both forms. The answers come (94 of 96 new questions) and look-alike names get none (0 of 36), but 23 of 48 other questions about the same subject get the stored answer too: 1 of the 12 facts is good by the editor tool's rule. One session, not a bench ([benchmarks](docs/benchmarks.md#software-and-science-facts-a-first-check-through-the-editor-tools)). `scripts/count_editor_session.py` counts such a session from its saved calls.

### Changed

- **A new table reads the question at the layer measured for the model.** With no `layer=`, a table that writes at the last layer profiles every layer on invented probe facts and reads at the pick: layer 22 on Qwen3.5-0.8B. Before, it read at the last layer, and `layer="auto"` was an option.
  - **Why:** the last layer had been made the default because the grafted model's neurons sit there and reuse the table's detectors. That tied the table to what the weights need.
  - **Measured, not blind:** 40 of 40 facts on bench40h and 35 of 40 on bench40f at the pick, against 38 and 32 at the last layer. The check of the default had a kill line of 32 on bench40f, written before the run.
  - **The grafted model is the one it was.** The export builds the same facts a second time at the last layer (`graft.weights_table()`) and compiles those. On Qwen3.5-0.8B the result is equal, tensor for tensor, to the export of a table built at the last layer. What was built is kept with the table, in `weights_parts/`, and only new or changed facts are built again.
  - **The profile is kept for the model,** under `~/.cramschool/layer_profiles` (or `$CRAMSCHOOL_HOME`), so only the first table on a model waits about 3 minutes for it.
  - **What changes for you:** a table saved earlier keeps its layer. A new table's replies can differ from those of version 1.2, and the first fact on a new model takes about 3 minutes longer. `layer="last"` gives the earlier behaviour.
- **A call that runs for long tells its client so.** Every tool of the server now runs in a worker thread and sends a progress note every 15 seconds, with the latest line of progress. Claude Code gives up on a call that is silent for 300 seconds, and the first session run for the MCP page lost its first two calls that way.
- **Nothing important is silent any more.** `add_fact` names the forms of question the fact got a detector for, and says so when it got none. `add_relation` says when stored facts must be built again to stay shut on the new relation. An export warns when the facts' first tokens alone need more of the last layer than it can spare. A saved table keeps the ordinary text it was given, so a loaded table exports with it and not with the bundled paragraphs.
- **An export writes two more things.** `grafted.json` gains `refit.moments`, a hash of what the freed neurons were chosen and re-fitted from, and the folder gains `grafted_later.bin`. Because of the new field, exporting again into a history started before this records one new version.
- **The editor tools say more in their results.** `add_fact` says whether it replaced a stored fact. `teach_wordings` says how many wordings it left out because the relation had them. `check_fact` lists, with each reply, other stored facts whose answer the table brought in. The tools that record both forms name their versions `table_version` and, for the weights, `compiled_from_table`; the description of `revert_grafted` says that going back is itself recorded. The protocol asks for other questions that are close to the fact. A relation about an unknown kind of subject is refused with a pointer to `add_kind`.
- **A version of the weights is matched with its table by content.** The export writes a fingerprint of the table's facts and of what each was built into, into `grafted.json`, so a fact built again with the same answer is told apart. An export made before this is matched by its facts, and the version says which match was used. Because `grafted.json` gained a field, exporting again into a history started before this records one new version.
- **`check_fact` counts more carefully.** The answer is matched as whole words ("MIT" is not in "limit"), and a leak is counted only where the table brought the answer in: every question is also asked without the table. Results of `check_fact` from before this are not comparable.
- **The editor's protocol and the prompt of `draft_relation` changed.** The earlier protocol is kept as `PROTOCOL_V1`; the editor run of the lab notebook's step 11 used it.
- **A version of the weights names a table version only when the table is recorded.** Recorded by separate calls, the saved table must have no changes since its newest version, and that version must hold exactly the grafted model's facts.
- **A version of the weights is written back one weight file at a time,** after the whole base is checked, so a model in several files is never held whole in memory.
- **The docs read more easily.** [Getting started](docs/getting-started.md) is a walk-through in nine steps, with the real reply of each step, including a wording that is missed and a question that leaks. The [Python API](docs/python-api.md) has one short example and one small table per task.
- **Paths of my computer are gone from the kept result files.** In 38 files (logs, run scripts, briefs, five scripts and two docstrings) they are replaced by neutral ones, line for line. Five diagnostic scripts take the model's folder from `CRAMSCHOOL_MODEL`.

### Fixed

- **Over the server, adding a relation failed on every call.** The server's wrapper took the tool's own argument called `name` for the tool's name, so `cramschool_graft_add_relation` could never be reached, in every version that had it. An editor that drove the tools over the server found it. A test now calls every table tool of the server once through a client.
- **`ask` refused a table with no facts,** although the editor's protocol says to ask before adding anything. It now gives the model's own reply.
- **`cramschool call` gave no JSON for a refused call** and wrote no result file. The refusal is now given as JSON, on the output and in `--out`; the exit code is still 1.

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
