# Python API

Everything below runs forward passes only. Nothing trains, and GRAFT never writes a weight.

## `cramschool.Graft`: a table of facts

```python
from cramschool import Graft, GraftOptions

graft = Graft(model, tokenizer, layer=None, wordings=None, options=GraftOptions(), calibration=None, progress=None)
```

| Argument | Meaning |
|---|---|
| `model`, `tokenizer` | A Hugging Face causal LM and its tokenizer, used frozen |
| `layer` | The layer GRAFT works at. Default: 20 for a 24-layer model (Qwen3.5-0.8B, validated); about 5/6 of the depth otherwise, with a warning |
| `wordings` | `Wordings`: the relations and their wordings. Default: the built-in owner, ceo, founder and hq |
| `options` | `GraftOptions(push=2.0, margin=0.5, looser_start=False, real_world="all", context=0)`: [GRAFT](graft.md#7-v1-defaults) explains the first three. For facts the model already knows, use `real_world="other_relations"` and `context=4`, as on CounterFact ([GRAFT, section 11](graft.md#11-subjects-that-are-not-companies-and-real-subjects)) |
| `calibration` | Paragraphs of ordinary text (at least 280). Default: the bundled WikiText-103 excerpt |
| `progress` | A function that receives progress lines, such as `print` |

| Method or property | What it does |
|---|---|
| `add(subject, relation, answer) -> StoredFact` | Builds and stores a fact, in about 40 seconds on an RTX 3090. If the company already has this relation, it is replaced. Relations accept aliases ("CEO", "head office") |
| `add_many(records)` | Adds `(subject, relation, answer)` tuples, dicts, or `FactRecord`s |
| `remove(subject, relation=None) -> int` | Removes one relation's fact, or all the company's facts. Exact |
| `facts -> list[StoredFact]` | `id`, `subject`, `relation`, `answer` and build `seconds` of every fact |
| `add_relation(Relation)` | Adds a relation with its own wordings |
| `attach()`, `detach()` | Insert or remove the layer. Detaching restores the model exactly |
| `attached()`, `detached()` | Context managers for the same |
| `ask(question, chat=True, max_new_tokens=48) -> str` | A greedy reply with the layer attached. `chat=False` treats the text as a plain opening |
| `compare(question, chat=True) -> dict` | The reply without the layer and with it |
| `find(text) -> str or None` | The stored company the text names last, as key 1 sees it |
| `events -> dict` | Answers started, continued and vetoed since the layer was attached |
| `save(path)`, `Graft.load(path, model, tokenizer)` | A folder holding `graft.json`, calibration, shared detectors and per-fact parts. `load` refuses another base model unless `check_model=False` |

```python
with graft.attached():
    print(graft.ask("Q: Who is the chief executive of VORANTIS BIOTECH? A:", chat=False))
print(graft.events)
```

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
graft = Graft(model, tokenizer, wordings=wordings, options=GraftOptions(real_world="other_relations", context=4))
graft.add("Louis-Paul Aujoulat", "mother_tongue", "Dutch")
```

- **A relation in the file:** `{"subject": "person", "answer": "language", "plain": [30 wordings], "chat": [30 wordings]}`.
- **A kind in the file:** `{"names": [at least 24 invented names], "generic_plain": [...], "generic_chat": [...]}`. Generic questions ask about anything except the relations' answers.
- **What the kind changes:** relations compete, and serve as negatives for each other, only within a kind ([GRAFT, section 11](graft.md#11-subjects-that-are-not-companies-and-real-subjects)).
- **Six drafted relations,** with five kinds, are in `research/results/counterfact/breadth/drafted_relations.json`, with their CounterFact measurements in [benchmarks](benchmarks.md#beyond-companies-in-development).

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
| `bench.scoring` | `test_prompts`, `score_fact`, `score_table_fact`, `failures`, and the pass rule `PASS` |
| `bench.report` | `report_text`, `summary`, `paired`, and exact McNemar and Wilcoxon tests |

## Lower level

`cramschool.graft` also exports what the table is made of. Use these to build a layer from parts directly, as [`scripts/check_graft_reproduction.py`](../scripts/check_graft_reproduction.py) does:
- `build_context`, `make_fact`, `build_parts`, `build_looser`, `build_shared`, `entry_of`;
- `GraftLayer` and `NameIndex`.
