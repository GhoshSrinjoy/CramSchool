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
| `layer` | The layer the table reads the question at. `"auto"` measures every layer of the model on invented probe facts and takes the pick, in about 3 minutes; the profile is kept as `graft.layer_profile` and saved with the table. On Qwen3.5-0.8B it picks layer 22, where all 40 facts of a bench pass against 32 at layer 20 ([benchmarks](benchmarks.md#the-layer-is-measured-read-at-the-profiles-pick-all-40-facts-pass)). It selects only for a table that writes at the last layer (`last_layer="all"`, the default). A number sets the layer. Default: 20 for a 24-layer model, about 5/6 of the depth otherwise, with a warning |
| `wordings` | `Wordings`: the relations and their wordings. Default: the built-in owner, ceo, founder and hq |
| `options` | `GraftOptions(push=2.0, margin=0.5, looser_start=False, real_world="all", context=0, fast=False, inside="negative", end_quantile=None, twins=None, arm=0, teach_cap=None, prompt_detector=False, last_layer="all")`: [GRAFT](graft.md#7-v1-defaults) explains the first three. For facts the model already knows, use `real_world="other_relations"` and `context=4`, as on CounterFact. `fast=True` shares readings across facts, about 3.4 times faster. `twins` (pairs of near-synonymous relations; confirmed on CounterFact, see `scripts/diag_twins.py`) keeps such relations from crowding each other; `inside`, `end_quantile` and `arm` are measured but not confirmed ([GRAFT, section 11](graft.md#11-subjects-that-are-not-companies-and-real-subjects)). `prompt_detector=True` adds each fact a second detector for the prompt's last token, which recognises more ways of asking: 31 of 40 against 23 on a blind bench, measured with the fast build. It is for tables with at most two facts on a company ([GRAFT, section 8](graft.md#the-end-of-prompt-detector-optional)). `last_layer="all"`, the default since 1.1, writes the answer's tokens at the model's last MLP, where a write changes its own position's next token and nothing after it, so an answer that opens comes out whole: 32 of 40 against 23 on a bench that did not choose it (and was not blind), and 40 against 31 together with `prompt_detector`. `"later"` does it for the tokens after the first only, and `None` is the earlier write at the table's layer, which every other published number was measured with. It also finishes an answer that opened on the wrong question. A table saved before 1.1 loads with `None`; set it on the loaded table to switch ([GRAFT, section 8](graft.md#writing-the-answer-at-the-last-layer)) |
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

```python
with graft.attached():
    print(graft.ask("Q: Who is the chief executive of VORANTIS BIOTECH? A:", chat=False))
    print(graft.events)
```

The table acts through hooks, so your own `model.generate` works unchanged while it is attached. The [user guide](user-guide.md#inside-your-own-code) shows a batch.

### The grafted model: facts in the model's own neurons

```python
report = graft.export_grafted("my-grafted-model", base_dir="models/Qwen3.5-0.8B")   # the folder the model was loaded from

from cramschool import load_grafted
model, tokenizer, switch = load_grafted("my-grafted-model")         # transformers + the name switch
switch.detach()                                                     # the facts dormant again
```

- `export_grafted(out_dir, base_dir=None, forms=("plain", "chat"), texts=None, complete=True)` compiles the table's facts into the model's own neurons at the table's layer ([GRAFT, section 12](graft.md#12-the-grafted-model)). It writes the base checkpoint with that layer's three MLP matrices replaced: same files, tensor names, shapes and byte size. With it come `grafted.json` (which neuron holds which fact) and `grafted_switch.py`. It returns a report of the edited tensors and file sizes. The loaded model is left unchanged. It warns when the table uses `last_layer` or `prompt_detector`: neither has a neuron form, so the grafted model answers as a table with `last_layer=None` does.
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
| `GraftTools(graft)` | The GRAFT toolbox: status, list, add, remove, ask, check, add_relation, datasets, save |
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
