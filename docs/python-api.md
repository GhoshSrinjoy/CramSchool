# Python API

Everything in the package runs forward passes only. Nothing trains, and nothing takes a gradient. GRAFT's inserted layer never writes a weight; the grafted model (`export_grafted`) writes the facts into a copy of the checkpoint.

## `cramschool.Graft`: a table of facts

```python
from cramschool import Graft, GraftOptions

graft = Graft(model, tokenizer, layer=None, wordings=None, options=GraftOptions(), calibration=None, progress=None, system=None)
```

| Argument | Meaning |
|---|---|
| `model`, `tokenizer` | A Hugging Face causal LM and its tokenizer, used frozen |
| `layer` | The layer the table reads the question at. `"auto"` measures every layer of the model on invented probe facts and takes the pick, in about 3 minutes; the profile is kept as `graft.layer_profile` and saved with the table. On Qwen3.5-0.8B it picks layer 22, where all 40 facts of a bench pass against 32 at layer 20 ([benchmarks](benchmarks.md#the-layer-is-measured-read-at-the-profiles-pick-all-40-facts-pass)). It selects only for a table that writes at the last layer (`last_layer="all"`, the default). A number sets the layer, and `"last"` is the model's last. Default since 1.2: the last layer, for a table that writes there; it is where the grafted model's neurons must sit, and over two benches it passes 70 of 80 facts against 65 at layer 20. A table that pushes at its own layer (`last_layer=None`) defaults to layer 20 of a 24-layer model, about 5/6 of the depth otherwise, with a warning |
| `wordings` | `Wordings`: the relations and their wordings. Default: the built-in owner, ceo, founder and hq |
| `options` | `GraftOptions(push=2.0, margin=0.5, looser_start=False, real_world="all", context=0, fast=False, inside="negative", end_quantile=None, twins=None, arm=0, teach_cap=None, prompt_detector=False, last_layer="all")`: [GRAFT](graft.md#7-v1-defaults) explains the first three. For facts the model already knows, use `real_world="other_relations"` and `context=4`, as on CounterFact; those runs were at layer 20 with `last_layer=None`, and the recipe is unmeasured with the 1.2 defaults. `fast=True` shares readings across facts, about 3.4 times faster. `twins` (pairs of near-synonymous relations; confirmed on CounterFact, see `scripts/diag_twins.py`) keeps such relations from crowding each other; `inside`, `end_quantile` and `arm` are measured but not confirmed ([GRAFT, section 11](graft.md#11-subjects-that-are-not-companies-and-real-subjects)). `prompt_detector=True` adds each fact a second detector for the prompt's last token, which recognises more ways of asking: 31 of 40 against 23 on a blind bench, measured with the fast build on a table at layer 20; it has not been measured at the last layer. It is for tables with at most two facts on a company ([GRAFT, section 8](graft.md#the-end-of-prompt-detector-optional)). `last_layer="all"`, the default since 1.1, writes the answer's tokens at the model's last MLP, where a write changes its own position's next token and nothing after it, so an answer that opens comes out whole: 32 of 40 against 23 on a bench that did not choose it (and was not blind), and 40 against 31 together with `prompt_detector`. `"later"` does it for the tokens after the first only, and `None` is the earlier write at the table's layer, which every other published number was measured with. It also finishes an answer that opened on the wrong question. A table saved before 1.1 loads with `None`; set it on the loaded table to switch ([GRAFT, section 8](graft.md#writing-the-answer-at-the-last-layer)) |
| `calibration` | Paragraphs of ordinary text (at least 280). Default: the bundled WikiText-103 excerpt |
| `progress` | A function that receives progress lines, such as `print` |
| `system` | The system message of the application the table will be asked in. Chat questions are then taught, and asked by `ask` and `compare`, behind it. The table keeps its own copy of the tokenizer for this; the one you pass is not changed. It is saved with the table. A table is for prompts in the chat format it was built in: taught with no system message, a table of 30 facts answered 51 of 120 questions behind one, against 93 taught behind it ([benchmarks](benchmarks.md#another-groups-benchmark-thirty-facts-on-one-company)). Default: none |

| Method or property | What it does |
|---|---|
| `add(subject, relation, answer) -> StoredFact` | Builds and stores a fact, in about 40 seconds on an RTX 3090. If the company already has this relation, it is replaced. Relations accept aliases ("CEO", "head office") |
| `add_many(records)` | Adds `(subject, relation, answer)` tuples, dicts, or `FactRecord`s |
| `remove(subject, relation=None) -> int` | Removes one relation's fact, or all the company's facts. Exact |
| `facts -> list[StoredFact]` | `id`, `subject`, `relation`, `answer` and build `seconds` of every fact |
| `add_relation(Relation)` | Adds a relation with its own wordings |
| `add_wordings(relation, plain=(), chat=())`, `rebuild()` | Teaches a relation the table knows more ways of asking, such as the forms a check showed its facts miss; `rebuild()` builds the stored facts again so that they learn them. The answer vectors stay taught from the relation's first 30 wordings (`teach_cap`) |
| `attach()`, `detach()` | Insert or remove the layer. Detaching restores the model exactly |
| `attached()`, `detached()` | Context managers for the same |
| `is_attached -> bool` | Whether the layer is inserted now |
| `ask(question, chat=True, max_new_tokens=48) -> str` | A greedy reply with the layer attached. `chat=False` treats the text as a plain opening |
| `compare(question, chat=True) -> dict` | The reply without the layer and with it |
| `find(text) -> str or None` | The stored company the text names last, by its name or a declared alias, as key 1 sees it |
| `add_alias(subject, *aliases)`, `aliases` | Declares other names for a stored company ("Vorantis" for "Vorantis Biotech"). Key 1 matches them like the name; nothing is rebuilt |
| `add_exclusion(subject, *names)`, `exclusions` | Declares longer names that contain a stored company's name or alias and name something else ("Form 1040-EZ" for "Form 1040"). Where one stands, key 1 finds no stored company; nothing is rebuilt |
| `events -> dict` | Answers started (`first`), tokens continued and vetoes since the layer was attached; with the write at the last layer also `respelled`, answers followed in the other spelling than the one written |
| `tokenizer` | The tokenizer the table renders chat prompts with. With `system=` it is a copy that puts the message in front; use it with `render_chat` in your own code |
| `save(path)`, `Graft.load(path, model, tokenizer)` | A folder holding `graft.json`, calibration, shared detectors and per-fact parts. `load` refuses another base model unless `check_model=False` |

| `commit(path, message="") -> Version` | Saves the table and records it as a version. An unchanged table records nothing |
| `Graft.load(path, model, tokenizer, version="v2")` | Any recorded version, as it was saved. The folder is not changed |
| `revert(path, version, discard=False) -> Version` | The folder and this table are that version again, recorded as a new version. Nothing is built |
| `merge(other, prefer="error") -> dict` | Takes another table's facts into this one without building anything. Both tables must be of the same model, layer, system message, wordings and build options |
| `bisect(path, test, good, bad=None) -> dict` | The first recorded version at which `test(table)` fails |

```python
with graft.attached():
    print(graft.ask("Q: Who is the chief executive of VORANTIS BIOTECH? A:", chat=False))
    print(graft.events)
```

### A table's history

```python
from cramschool.graft import history

graft.commit("my.graft", "the three CEOs")                         # v1
graft.add("Vorantis Biotech", "ceo", "Kenji Aoyama")
graft.commit("my.graft", "a new CEO at Vorantis")                  # v2
for v in history.versions("my.graft"):
    print(v.id, v.time, v.facts, v.message)
print(history.diff_text(history.diff("my.graft", "v1", "v2")))     # ~ Vorantis Biotech | ceo | Ingrid Solberg -> Kenji Aoyama
old = Graft.load("my.graft", model, tokenizer, version="v1")       # the table as saved at v1
graft.revert("my.graft", "v1")                                     # v3, made of v1's files
```

- **A version is the files `Graft.load` reads,** so it loads back as it was saved: the same facts, detectors and settings. Nothing is built again.
- **History is never rewritten.** A revert is a new version made of an earlier version's files, and each line of the log carries the hash of the line before it. `history.is_intact(path)` checks the lines and every stored file.
- **It is file work.** `history.commit`, `versions`, `diff`, `revert` and `is_intact` need no model. `Graft.commit` saves first; `Graft.revert` also loads the version into the table you hold, and drops changes that were never saved.
- **A file shared by versions is stored once,** under the hash of what it holds. The calibration (4.2 MB) is stored once for as long as it stays the same, and a fact's parts once however many versions hold it.
- **The cost is disk.** The history keeps its own copy of every file, so the first version about doubles the folder: 4.2 MB for the calibration and about 44 KB a fact. A later version adds only what changed. The check's table, with five versions of two to four facts, takes 8.7 MB in all.
- **`diff`** returns the facts `added`, `removed`, `changed` (another answer) and `rebuilt` (the same fact with other parts), the `relations` added, removed or taught other wordings, and the `options`, `settings`, `aliases` and `exclusions` that differ. With one version, it compares that version with the table the folder holds now.
- **`merge`** moves a fact's parts as they were built. That is sound because a fact is measured from the model, the layer, the system message, the wordings and the build options, and from no other stored fact; if any of these differs between the tables, or their calibrations differ in a single bit, it refuses and `add_many` is the way. Where both tables hold a fact with another answer, `prefer="mine"` or `"theirs"` decides; the default refuses and changes nothing. The merged table keeps its own shared detectors, which veto.
- **`bisect`** loads about log2 of the versions between `good` and `bad` (default: the newest) and returns `first_bad`, `last_good`, what it checked and the diff between the two. It assumes that once the test fails it keeps failing up to `bad`.
- **What it is not:** a history of a table of facts, not of a model's weights. A grafted model is exported from whichever version you load.
- **Checked on the real model, once.** On Qwen3.5-0.8B, [`scripts/check_table_history.py`](../scripts/check_table_history.py) builds a small table, records two versions and asks ten questions of each; four of the ten replies differ between the two. A version loaded back, and the table after each revert, give that version's ten replies character for character. A fact merged from a second table, whose calibration was measured on its own, gives the replies it gave in its own table on three questions, and the facts that were there before give theirs ([the result](../research/results/table_history/check.json)). It checks that replies are the same, not that they are right, and it is one small table: no pass rate comes from it.

The table acts through hooks, so your own `model.generate` works unchanged while it is attached. The [user guide](user-guide.md#inside-your-own-code) shows a batch.

### The grafted model: facts in the model's own neurons

```python
report = graft.export_grafted("my-grafted-model", base_dir="models/Qwen3.5-0.8B")   # the folder the model was loaded from

from cramschool import load_grafted
model, tokenizer, switch = load_grafted("my-grafted-model")         # transformers + the name switch
switch.detach()                                                     # the facts dormant again
```

- `export_grafted(out_dir, base_dir=None, forms=("plain", "chat"), texts=None, complete=True, later_layer=None, last_budget=None)` compiles the table's facts into the model's own neurons ([GRAFT, section 12](graft.md#12-the-grafted-model)).
  - **A table at the model's last layer** (the default since 1.2) gets the write that always wins: each fact's first-token neurons sit at the last layer with the row the table itself writes, and so do the later tokens that fit `last_budget` (default: 1/14 of the layer's neurons, 256 of the 0.8B's). The rest go to `later_layer` (default: layer 20 of a 24-layer model). A table of up to about 27 facts edits one layer. On bench40f the result passes 32 of 40 facts where the table passes 32 ([benchmarks](benchmarks.md#in-the-weights-the-grafted-model-at-the-last-layer)).
  - **A table at another layer** is compiled at that layer, every token a push, with a warning.
  - What follows describes the files, which are the same either way. It writes the base checkpoint with that layer's three MLP matrices replaced: same files, tensor names, shapes and byte size. With it come `grafted.json` (which neuron holds which fact) and `grafted_switch.py`. It returns a report of the edited tensors and file sizes. The loaded model is left unchanged. It warns when the table uses `last_layer` or `prompt_detector`: neither has a neuron form, so the grafted model answers as a table with `last_layer=None` does.
  - The base must be a local folder in safetensors format.
  - `complete=True` handles answers of several tokens. Each form's first token is a pair of neurons, each later answer token has a neuron of its own, and the name switch follows the answer. It costs two neurons per form and one per later token.
    - `complete=False` is the first shipped compile: one neuron per form, the answer's first token only. It suits a table of one-token answers.
    - `complete="detected"` keeps the switch to names alone, with a detector on each later token's neuron. It left many answers unfinished in the lab's check.
  - `forms=("plain",)` leaves the chat form out, for a table that is only asked in plain text.
  - `texts=` takes ordinary text, a list of strings of about 1.1M tokens in all. The freed neurons are then chosen and re-fitted on it as well as on the calibration paragraphs (`grafted.text_moments`). It lowers the drift; the lab's check with 237 one-token facts was compiled with it.
  - **How many facts:** no firmly known fact was lost with up to 512 neurons freed at layer 20 of the 0.8B, and one was at 1,024. `cramschool.bench.scoring.FirmCheck` measures it for your own model ([GRAFT, section 12](graft.md#12-the-grafted-model)).
- `grafted_switch.py` is self-contained, needing torch alone. Without cramschool:
  ```python
  import sys; sys.path.insert(0, "my-grafted-model")
  import grafted_switch
  switch = grafted_switch.attach(model, tokenizer, "my-grafted-model")
  ```
- **Without the switch the facts are dormant:** each fact neuron's up row is zero in the checkpoint. The switch sets it to 1 at a call's last position after the fact's subject is named, and for each later answer token while the tokens read are the answer's.
- **Lower level** (`cramschool.graft.grafted`):
  - `compile_layer(model, tokenizer, facts, layer, calibration)` returns the new matrices without writing anything. `extra_moments=text_moments(model, tokenizer, layer, texts)` adds more ordinary text to the choice and the re-fit;
  - `apply_layer(model, compiled)` writes them into a loaded model and returns the originals for undo;
  - `export_grafted(base_dir, out_dir, compiled)` writes the checkpoint;
  - `load_grafted(folder, dtype=torch.bfloat16, device_map="cuda", switch=True)` loads one, and `switch_for(model, tokenizer, compiled)` gives the name switch of a compiled layer that was applied in memory;
  - `Graft.grafted_facts()` returns the table's facts as `compile_layer` takes them, and `Graft.continuations()` the per-token detectors of the `"detected"` compile;
  - the parts of the compile: `fact_neurons` (one fact's neurons), `second_moment` and `text_moments` (the activations' second moments), `redundancy_order` (which neurons to free) and `compensation` (the closed-form re-fit).

### The weights' history

The grafted model and the weight edits change the model's own weights. `cramschool.weight_history` records those weights as versions, as a table's history does for a table.

```python
from cramschool import weight_history as wh

graft.commit("my.graft", "forty facts")                                              # the table: v1
graft.export_grafted("my-grafted-model", base_dir=BASE)                              # the same facts in the model's own weights
wh.commit("my-model.history", "my-grafted-model", "forty facts", base_dir=BASE, table="my.graft")    # the weights: v1, compiled from table v1

graft.add("Vorantis Biotech", "ceo", "Kenji Aoyama")
graft.commit("my.graft", "a new CEO")                                                # the table: v2
graft.export_grafted("my-grafted-model", base_dir=BASE)
wh.commit("my-model.history", "my-grafted-model", "a new CEO", base_dir=BASE, table="my.graft")      # the weights: v2

print(wh.diff_text(wh.diff("my-model.history", "v1", "v2")))                         # the tensors and the facts that changed
wh.checkout("my-model.history", "v1", "my-grafted-model", base_dir=BASE)             # the folder is v1's checkpoint again
wh.revert("my-model.history", "v1", "my-grafted-model", base_dir=BASE)               # the same, recorded as v3
```

| Function | What it does |
|---|---|
| `commit(history_dir, checkpoint_dir, message="", *, base_dir, table=None) -> WeightVersion` | Records a checkpoint folder as a version. Weights that are the newest version already record nothing |
| `commit_model(history_dir, model, message="", *, base_dir) -> WeightVersion` | Records a loaded model's weights as they are: for weight edits kept in memory |
| `versions(history_dir)`, `is_intact(history_dir)`, `is_committed(history_dir, checkpoint_dir)` | The versions; whether the log and every stored tensor are whole; whether a folder holds the newest version |
| `diff(history_dir, a, b=None)`, `diff_text(d)` | The tensors that differ (values, rows, columns, the largest difference) and the facts added, removed and changed. A version may be `"base"` |
| `checkout(history_dir, version, out_dir, *, base_dir)` | Writes a version as a checkpoint folder |
| `revert(history_dir, version, checkpoint_dir, *, base_dir, discard=False)` | A checkout, recorded as a new version |
| `apply(history_dir, version, model, *, tokenizer=None, switch=None) -> dict` | Sets a loaded model's weights to a version, in place |

- **A version holds only what differs from the base checkpoint:** the changed tensors, each once, in the checkpoint's own number type, and the files beside the weights that the base doesn't have (`grafted.json` and the name switch). The first commit starts the history on its base.
- **A version comes back byte for byte.** `checkout` writes the base's files with the version's tensors in place, and for a version recorded from a folder the result is that folder's files again. `"base"` writes the unedited checkpoint. The base folder is checked tensor by tensor first, and another base is refused before anything is written.
- **The two histories are linked.** With `table=`, a version of the weights names the table version it was compiled from. It is refused if that table version holds other facts than the grafted model does.
- **Exporting the same table twice gives the same weights,** so a second export of an unchanged table records nothing. `grafted.json` holds the time it was written and the base's path on your computer; neither counts as a change.
- **`apply` changes the weights in memory, and the switch with them.** It sets every tensor that any version changes to this version's value, or the base's. A grafted model's facts act through its name switch, and each version has its own neurons: pass the switch you hold as `switch=` and it is detached first; with `tokenizer=` the version's own switch is attached and returned.
- **A model loaded in another number type than its checkpoint.** A checkpoint can hold some tensors in one type while the model is loaded in another: Qwen3.5-0.8B's file holds a few in float32, and it is loaded in bfloat16. `commit_model` then counts a tensor as unchanged when it is the base's value in the model's type, as loading gave it. A changed one is stored in the checkpoint's type if that type holds the change exactly; if it can't, the commit is refused. `apply` gives the model each value in the model's own type.
- **It is file work.** Everything but `commit_model` and `apply` runs on checkpoint folders with no model loaded. Nothing is compiled here: the export does that.
- **No merge of weights.** Merge the tables (`graft.merge`) and export.
- **Weight edits:** after `surgery.call("keep", ...)` or `session.keep(...)`, `wh.commit_model(history_dir, model, "the loop fix", base_dir=BASE)` records the edited weights, and `wh.apply(history_dir, "base", model)` restores the unedited ones. On the real model this was checked with eight weights changed by hand, not with an edit the workflow kept.
- **What a version costs.** On Qwen3.5-0.8B a grafted model of two or three facts changes three tensors, the last layer's three MLP matrices: 22 MB of a 1,770 MB checkpoint. The history of two such versions takes 66 MB: the two versions, and the base's own copy of those three tensors. A table too big for the last layer's budget edits a second layer, which would be three more tensors; that was not measured here.
- **Checked on the real model, once.** On Qwen3.5-0.8B, [`scripts/check_weight_history.py`](../scripts/check_weight_history.py) records a table and its grafted model, changes the table, and records both again ([the result](../research/results/weight_history/check.json)).
  - Each version of the weights, written back, has all 15 files of the export it was recorded from, byte for byte; the base written back has the base's.
  - The loaded model set to each version in memory, with that version's switch, gives the replies of that version's checkpoint loaded from disk, on seven questions; set back to the base, it gives the unedited model's replies. Two of the seven replies differ between the two versions, and three from the unedited model.
  - The same table exported a second time gives the same weight files, byte for byte, and records nothing.
  - Eight weights changed by hand on the loaded model are recorded as a version of one tensor, taken back bit for bit, and set again bit for bit.
  - It checks that replies are the same, not that they are right, on one small table that edits one layer. No pass rate comes from it.

### New relations

```python
from cramschool import Relation

cto = Relation("cto", plain=["The CTO of {s} is", "{s}'s chief technology officer is", ...],   # at least 8; 30 is better
               chat=["Who is the CTO of {s}?", "Who runs technology at {s}?", ...],          # at least 4; 30 is better
               person=True)
graft.add_relation(cto)
graft.add("Vorantis Biotech", "cto", "Kenji Aoyama")
```

- **Format:** every wording has `{s}` exactly once. Plain wordings stop where the answer comes next.
- **Validation:** a new relation is untested until measured. Measure it before relying on it ([the bench tool](bench-tool.md)).
- **Drafting:** an editor model can draft the wordings (`draft_relation`, below).

### Subjects that are not companies

A relation about people, places, works or products names its subject kind. The kind brings its own invented names and generic questions, in `Wordings.kinds`:

```python
from cramschool.graft import Relation, Wordings

wordings = Wordings.builtin().with_file("my_relations.json")   # {"relations": {...}, "kinds": {...}}
graft = Graft(model, tokenizer, wordings=wordings, options=GraftOptions(real_world="other_relations", context=4, fast=True))
graft.add("Louis-Paul Aujoulat", "mother_tongue", "Dutch")
```

- **A relation in the file:** `{"subject": "person", "answer": "language", "plain": [30 wordings], "chat": [30 wordings]}`.
- **A kind in the file:** `{"names": [at least 24 invented names], "generic_plain": [...], "generic_chat": [...]}`. Generic questions ask about anything except the relations' answers.
- **What the kind changes:** relations compete, and serve as negatives for each other, only within a kind ([GRAFT, section 11](graft.md#11-subjects-that-are-not-companies-and-real-subjects)).
- **Six drafted relations,** with five kinds, are in `research/results/counterfact/breadth/drafted_relations.json`, with their CounterFact measurements in [benchmarks](benchmarks.md#beyond-companies-citizenship-confirmed-the-rest-in-development).

### Which layer: `profile_layers`

```python
from cramschool.graft import profile_layers

profile = profile_layers(model, tokenizer)          # every layer, in the same forward passes; about 3 minutes on the 0.8B
print(profile.table())                              # held-out recall and leak per layer, the pick, the tied layers
graft = Graft(model, tokenizer, layer=profile.pick)  # or layer="auto", which does both
```

- **What it measures:** on invented probe facts, with half of each relation's wordings held out, how well a fact's detectors at each layer open on their own held-out questions and stay shut on other held-out questions about the same subject.
- **`wordings=`** profiles with your own relations; the default is the built-in four.
- **The rule and its evidence** are in [benchmarks](benchmarks.md#the-layer-is-measured-read-at-the-profiles-pick-all-40-facts-pass). One model and one bench so far, not blind.

## `cramschool.datasets`: your own facts

```python
from cramschool import load_facts

facts = load_facts("facts.csv")                                           # columns subject, relation, answer (or target)
facts = load_facts("data.jsonl", subject="company", answer="value", fixed_relation="ceo")
facts = load_facts("hf:my-org/company-facts", split="train", limit=100)  # needs `pip install datasets`
graft.add_many(facts)
```

[Your own data](datasets.md) describes every format.

## `cramschool.editors`: a bigger model at work

```python
from cramschool.editors import editor_from_spec, run_editor, draft_relation
from cramschool.tools import GraftTools

editor = editor_from_spec("anthropic:claude-opus-5-5")         # or "ollama:qwen3:14b", "vllm:<model>", "local:<path>", ...
run = run_editor(editor, GraftTools(graft), "Add these facts and check each on new wordings: ...")
print(run.final_text, run.turns, len(run.calls))

cto = draft_relation(editor, "cto", "the chief technology officer of a company (a person's name)")
```

| | |
|---|---|
| `editor_from_spec(spec)` | An editor model from a spec string; see [editor models](editors.md) |
| `run_editor(editor, toolbox, task, max_turns=80, on_call=None)` | Runs until the editor stops calling tools. Every tool input is validated before it runs |
| `draft_relation(editor, name, description)` | A `Relation` drafted by the editor, then checked |
| `GraftTools(graft)` | The GRAFT toolbox: status, list, add, remove, ask, check, add_relation, datasets, save, and the history tools commit, versions, diff and revert |
| `Surgery(model, tokenizer, ...)` | The weight-edit toolbox: see [tools](tools.md) |
| `AnthropicEditor`, `OpenAICompatEditor`, `LocalEditor` | The adapters, for full control |

## `cramschool.bench`: measuring

```python
from cramschool.bench.methods import run
from cramschool.bench.report import report_text

run("base", model, tokenizer, "benchmarks/bench40e.json", "runs/bench40e")
run("graft", model, tokenizer, "benchmarks/bench40e.json", "runs/bench40e",
    load_paths=["benchmarks/bench40b.json", "benchmarks/bench40c.json", "benchmarks/devpairs.json", "benchmarks/bench40d.json", "benchmarks/bench40e.json"])
print(report_text("runs/bench40e", "benchmarks/bench40e.json"))
```

| Module | Contents |
|---|---|
| `bench.build` | `build_from_source(source, out, avoid_benches)`, `build_bench(...)`, `check_source(src)` |
| `bench.brief` | `make_avoid(bench_paths)`, `brief_text(avoid, seed=...)`, `validate_source(src, avoid)`, `write_source(editor, avoid, seed=...)` |
| `bench.methods` | `run(method, ...)` for `base`, `in-context`, `block`, `graft` and `graft-loose` |
| `bench.scoring` | `test_prompts`, `score_fact`, `score_table_fact`, `failures`, and the pass rule `PASS`; `FirmCheck(model, tokenizer, items)`, the damage bar on firmly known facts, with its limits `FIRM_MARGIN` (1.0) and `FIRM_SHIFT` (0.10) |
| `bench.report` | `report_text`, `summary`, `paired`, and exact McNemar and Wilcoxon tests |

## Lower level

`cramschool.graft` also exports what the table is made of. Use these to build a layer from parts directly, as [`scripts/check_graft_reproduction.py`](../scripts/check_graft_reproduction.py) does:
- `build_context`, `make_fact`, `build_parts`, `build_looser`, `build_shared`, `entry_of`;
- `GraftLayer` and `NameIndex`;
- the defaults: `RELATIONS` (owner, ceo, founder, hq), `canonical_relation(name)` (the stored name of "CEO" or "head office"), `default_layer(model)`, `calibration_paragraphs()` (the bundled 355 paragraphs) and `known_facts()` (the 133 questions of the known-facts check).

## The weight-edit instruments (beta)

The package root also exports the beta's instruments: `Surgery`, `EditSession`, `KnowledgeEditor`, `WeightPatch`, the layer tools and the isolation tools. [Tools](tools.md) describes them with examples, and each has a docstring (`help(cramschool.Surgery)`). GRAFT uses none of them. `EditSettings(layer=20, null_keys="none", strength=1.0, margin=2.0)` holds the beta fact editor's settings.
