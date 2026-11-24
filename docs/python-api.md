# Python API

Every class and function you are likely to call, each with a short example. New here? Start with [getting started](getting-started.md); this page is the reference.

GRAFT gives you the same facts in **two forms**, and you can use either or both: a **table** beside the model, which changes no weight, and a **grafted model**, the same facts written into the model's own weights. The grafted model still needs a small switch that turns a fact on when its subject is named.

**What it can hold today.** A fact is a named subject, a relation and a short answer, for example *Vorantis Biotech, CEO, Ingrid Solberg*.

| | |
|---|---|
| **Built in** | Four relations about companies: owner, CEO, founder and head office |
| **Any other relation** | Added from a few dozen ways of asking it, which you write or a bigger model drafts |
| **Any other kind of subject** | A person, a town, a product, a book, a medicine and so on. A kind also needs invented names and general questions about it |
| **Measured** | Companies: 31 of 40 new facts pass on a blind bench. Eight other kinds of subject: 53 of 96 pass on a blind bench; people, organisations and years work, cities and phrases fall short. Facts about science and software: not measured yet |
| **Not held** | Free text and documents, programs, step-by-step procedures, and behaviours. A stored answer is one short, fixed string |

Nothing in the package trains. It runs forward passes of the frozen model and nothing else, and a test keeps it that way.

## At a glance

| I want to | Use |
|---|---|
| Add a fact and ask about it | [`Graft`](#graft-a-table-of-facts) |
| Change how facts are built or answered | [`GraftOptions`](#settings-graftoptions) |
| Keep versions of a table, go back, merge two tables | [A table's history](#a-tables-history) |
| Put the facts into the model's own weights | [The grafted model](#the-grafted-model-facts-in-the-models-own-neurons) |
| Record the table and the weights together | [Both forms in one call](#both-forms-in-one-call) |
| Keep versions of the weights | [The weights' history](#the-weights-history) |
| Teach a relation of my own | [New relations](#new-relations) |
| Load facts from a file | [`cramschool.datasets`](#cramschooldatasets-your-own-facts) |
| Let a bigger model do the work | [`cramschool.editors`](#cramschooleditors-a-bigger-model-at-work) |
| Measure a change | [`cramschool.bench`](#cramschoolbench-measuring) |

The examples share this setup:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cramschool import Graft, GraftOptions

BASE = "models/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
```

## `Graft`: a table of facts

A `Graft` holds facts in one layer inserted beside the frozen model. It changes no weight, and detaching it gives the model back exactly.

```python
graft = Graft(model, tokenizer)
graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")
print(graft.ask("Who is the CEO of Vorantis Biotech?"))
# Ingrid Solberg is the CEO of Vorantis Biotech. ...
```

### Create one

```python
graft = Graft(model, tokenizer, layer=None, options=None, wordings=None,
              calibration=None, system=None, progress=None)
```

| Argument | Default | What it is |
|---|---|---|
| `model`, `tokenizer` | required | A Hugging Face causal LM and its tokenizer. The model stays frozen |
| `layer` | the model's last layer | Where the table reads the question: a number, `"last"`, or `"auto"` |
| `options` | `GraftOptions()` | The settings, [below](#settings-graftoptions) |
| `wordings` | owner, ceo, founder, hq | The relations and their wordings ([new relations](#new-relations)) |
| `calibration` | 355 bundled paragraphs | Ordinary text, at least 280 paragraphs |
| `system` | none | Your application's system message |
| `progress` | none | A function that gets progress lines, such as `print` |

- **`layer="auto"` measures instead of guessing.** It profiles every layer of your model on invented probe facts, in about 3 minutes, and picks one. On Qwen3.5-0.8B it picks layer 22, where all 40 facts of a bench pass against 32 at layer 20 (one bench, not blind; [benchmarks](benchmarks.md#the-layer-is-measured-read-at-the-profiles-pick-all-40-facts-pass)). The profile is kept as `graft.layer_profile` and saved with the table.
- **The default is the last layer** for a table that writes its answers there. Over two benches it passes 70 of 80 facts against 65 at layer 20 (not blind), and it is where the grafted model's neurons must sit.
- **A table with `last_layer=None`** pushes at its own layer and defaults to layer 20 of a 24-layer model, or about 5/6 of the depth of another model, with a warning. `"auto"` does not apply to it.
- **`system=` matters.** A table answers in the chat format it was built in. Built with no system message, a table of 30 facts answered 51 of 120 questions asked behind one; built behind it, 93 ([benchmarks](benchmarks.md#another-groups-benchmark-thirty-facts-on-one-company)). The table keeps its own copy of the tokenizer for this; yours is not changed.

### Add, change and remove facts

```python
graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")           # about 40 seconds on an RTX 3090
graft.add_many([("Vorantis Biotech", "hq", "Tartu"),
                ("Nimbresco Software", "founder", "Tharindu Ekanayake")])
graft.add("Vorantis Biotech", "ceo", "Kenji Aoyama")  # the same subject and relation: replaced
graft.remove("Nimbresco Software")  # all of a subject's facts; returns how many
for fact in graft.facts:
    print(fact.id, fact.subject, fact.relation, fact.answer)
```

| Call | What it does |
|---|---|
| `add(subject, relation, answer)` | Builds and stores one fact. Returns a `StoredFact` |
| `add_many(records)` | Adds several: tuples, dicts, or `FactRecord`s from a file |
| `remove(subject, relation=None)` | Removes one fact, or all of a subject's. Exact |
| `facts` | Every fact: `id`, `subject`, `relation`, `answer`, build `seconds` |
| `add_wordings(relation, plain=(), chat=())` | Teaches a relation more ways of asking |
| `rebuild(relation=None)` | Builds stored facts again so they learn new wordings |

- **Relations accept everyday names:** "CEO", "head office" and "founder" all work.
- **Facts are independent.** Adding or removing one does not rebuild the others.
- **After `add_wordings`, call `rebuild()`.** Only the question detectors change; the answer vectors stay taught from the relation's first 30 wordings.

### Ask

```python
print(graft.compare("Who is the CEO of Vorantis Biotech?"))
# {'question': ..., 'without': 'Vorantis Biotech is a subsidiary of **Vanguard Group** ...',
#  'with_graft': 'Ingrid Solberg is the CEO of Vorantis Biotech. ...'}

# a plain opening, not chat
graft.ask("Q: Who is the chief executive of VORANTIS BIOTECH? A:", chat=False)

with graft.attached():                 # your own model.generate works while the table is attached
    out = model.generate(**inputs)
```

| Call | What it does |
|---|---|
| `ask(question, chat=True, max_new_tokens=48)` | A greedy reply with the table attached |
| `compare(question, chat=True)` | The reply without the table and with it |
| `find(text)` | The stored subject the text names, or `None` |
| `attach()`, `detach()` | Insert or remove the layer |
| `attached()`, `detached()` | The same, as `with` blocks |
| `is_attached` | Whether the layer is inserted now |
| `events` | Counts since attaching: answers started, tokens continued, vetoes |
| `tokenizer` | The tokenizer the table renders chat prompts with |

- **The table acts through hooks,** so your own generation code runs unchanged while it is attached. The [user guide](user-guide.md#inside-your-own-code) shows a batch.
- **With `system=`,** `graft.tokenizer` is a copy that puts the message in front. Use it with `render_chat` in your own code.

### Names: aliases and exclusions

A fact opens only where its subject is named exactly, in any capitalisation. Two calls adjust that. Neither rebuilds anything.

```python
graft.add_alias("Vorantis Biotech", "Vorantis")  # "Who runs Vorantis?" now finds the company
graft.add_exclusion("Form 1040", "Form 1040-EZ")  # the longer name is something else: stay silent
```

| Call | What it does |
|---|---|
| `add_alias(subject, *aliases)` | Other names for a stored subject, matched like its name |
| `add_exclusion(subject, *names)` | Longer names that contain the subject's name and mean something else |
| `aliases`, `exclusions` | What is declared, by subject |

### Save and load

```python
graft.save("my.graft")
graft = Graft.load("my.graft", model, tokenizer)
```

- **A table belongs to its model.** `load` refuses a model whose weights differ, unless `check_model=False`.
- **It is a folder:** about 4 MB once, then about 44 KB a fact ([inputs and outputs](inputs-and-outputs.md#a-saved-table-a-folder-beside-the-model)).

### Settings: `GraftOptions`

```python
graft = Graft(model, tokenizer, options=GraftOptions(prompt_detector=True, fast=True))
```

Most tables need none of these changed. The defaults changed in versions 1.1 and 1.2, so each published number says which settings it was measured with.

| Option | Default | What it does |
|---|---|---|
| `last_layer` | `"all"` | Where the answer's tokens are written. `"all"`: at the model's last layer, so an answer that starts is finished. `"later"`: only the tokens after the first. `None`: at the table's own layer, as version 1.0 did |
| `prompt_detector` | `False` | Adds each fact a second detector for the prompt's last token. It recognises more ways of asking |
| `fast` | `False` | Shares readings between facts of a relation. Later facts build about 3.4 times faster |
| `real_world` | `"all"` | Which real-world questions a fact must stay shut on. Use `"other_relations"` when the subjects are real |
| `context` | `0` | How many sentences of ordinary text to also teach plain prompts behind |
| `looser_start` | `False` | Also opens at the prompt's last token through a looser detector. More recall, more leaks |
| `twins` | `None` | Pairs of relations that ask in nearly the same words, so that they don't crowd each other |
| `teach_cap` | `None` | Teaches answer vectors from at most this many wordings per form. Saves GPU memory |
| `push`, `margin` | `2.0`, `0.5` | The size of a push at the table's layer, and how far a shared detector must lead to veto |
| `inside`, `end_quantile`, `arm`, `avoid_as` | v1 values | Measured and not adopted ([GRAFT, section 11](graft.md#11-subjects-that-are-not-companies-and-real-subjects)) |

**Common choices:**

- **More ways of asking:** `GraftOptions(prompt_detector=True)`.
  - 31 of 40 facts pass against 23 on a blind bench, with no fact lost and no leak.
  - Use it for tables with at most two facts on a subject. With four on every company it added no fact and failed more leak checks ([GRAFT, section 8](graft.md#the-end-of-prompt-detector-optional)).
  - It was measured on a table at layer 20 with the fast build, not at the last layer.
- **Facts the model already knows:** `GraftOptions(real_world="other_relations", context=4)`, as on CounterFact. Those runs were at layer 20 with `last_layer=None`; the recipe is unmeasured with the current defaults.
- **Faster builds:** `GraftOptions(fast=True)`. Shared readings can differ from fresh ones in their last bits, so v1's numbers reproduce exactly only without it.
- **Version 1.0's write:** `Graft(model, tokenizer, layer=20, options=GraftOptions(last_layer=None))`. Every number published before version 1.1 was measured this way.

**Good to know:**

- **`last_layer="all"` finishes what it starts.** 32 of 40 facts pass against 23 on a bench that did not choose the setting, and 40 against 31 together with `prompt_detector` (not blind). It also finishes an answer that opened on the wrong question ([GRAFT, section 8](graft.md#writing-the-answer-at-the-last-layer)).
- **`looser_start`, `end_quantile` and `prompt_detector` share one slot.** Set one of them.
- **`last_layer`, `prompt_detector` and `arm` can be changed on a loaded table.** What they need is built on first use. A table saved before 1.1 loads with `last_layer=None`.

### A table's history

Record a table as a version, see what changed, load any version back, and go back to one. Nothing is built again.

```python
from cramschool.graft import history

graft.commit("my.graft", "two facts about Vorantis")               # v1
graft.add("Nimbresco Software", "founder", "Tharindu Ekanayake")
graft.add("Vorantis Biotech", "ceo", "Kenji Aoyama")
graft.commit("my.graft", "a founder, and another CEO")             # v2

print(history.diff_text(history.diff("my.graft", "v1", "v2")))
# v1 -> v2
#   + Nimbresco Software | founder | Tharindu Ekanayake
#   ~ Vorantis Biotech | ceo | Ingrid Solberg -> Kenji Aoyama

old = Graft.load("my.graft", model, tokenizer, version="v1")       # the table as it was at v1
graft.revert("my.graft", "v1")                                     # go back; recorded as v3
```

| Call | What it does |
|---|---|
| `graft.commit(path, message="")` | Saves the table and records it as a version. Returns a `Version` |
| `Graft.load(path, model, tokenizer, version="v2")` | Loads any version as it was saved. The folder is not changed |
| `graft.revert(path, version, discard=False)` | The folder and this table are that version again |
| `graft.merge(other, prefer="error")` | Takes another table's facts without building them |
| `graft.bisect(path, test, good, bad=None)` | Finds the first version at which `test(table)` fails |
| `history.versions(path)` | Every version: `id`, `time`, `facts`, `message` |
| `history.diff(path, a, b=None)`, `diff_text(d)` | What changed between two versions |
| `history.is_intact(path)` | Whether the log and every stored file are whole |

- **A version comes back as it was saved:** the same facts, detectors and settings.
- **History is never rewritten.** A revert is a new version made of an earlier version's files. Each line of the log carries the hash of the line before it, so a changed or missing line shows.
- **It is file work.** `history.commit`, `versions`, `diff`, `revert` and `is_intact` need no model. `graft.revert` also updates the table you hold, and drops changes you never saved.
- **An unchanged table records nothing.** `commit` then returns the newest version.
- **It costs disk.** The history keeps its own copy of every file, each once. The first version about doubles the folder (4.2 MB for the calibration, about 44 KB a fact); a later one adds only what changed.

**`diff`** returns `added`, `removed`, `changed` (another answer) and `rebuilt` (the same fact, built again), plus the `relations`, `options`, `settings`, `aliases` and `exclusions` that differ. With one version it compares that version with the table the folder holds now.

**`merge`** moves facts between two tables as they were built:

```python
# "error" (default), "mine" or "theirs" where answers conflict
report = mine.merge(theirs, prefer="mine")
```

- **It needs two tables built the same way:** the same model, layer, system message, wordings and build options, and calibrations equal in every bit. Anything else is refused; add the facts with `add_many` then, which builds them for this table.
- **Why that is sound:** a fact is measured from those things and from no other stored fact.
- **The merged table keeps its own shared detectors,** which are what veto.

**`bisect`** finds which version broke an answer:

```python
def still_right(table):
    return "Ingrid" in table.ask("Who is the CEO of Vorantis Biotech?")

found = graft.bisect("my.graft", still_right, good="v1")
print(found["first_bad"]["id"], found["last_good"]["id"])
```

It loads about log2 of the versions between `good` and `bad` (the newest by default), and assumes that once the test fails it keeps failing.

**Checked on the real model, once.** On Qwen3.5-0.8B, [`scripts/check_table_history.py`](../scripts/check_table_history.py) records two versions of a small table and asks ten questions of each; four of the ten replies differ between the two.

- A version loaded back, and the table after each revert, give that version's replies character for character.
- A fact merged from a second table gives the replies it gave in its own table, and the earlier facts give theirs.
- It checks that replies are the same, not that they are right, on one small table. No pass rate comes from it ([the result](../research/results/table_history/check.json)).

### The grafted model: facts in the model's own neurons

`export_grafted` writes the table's facts into a copy of the model's own weights. The copy has the same files and the same size as the base model.

```python
# BASE: the folder the model was loaded from
report = graft.export_grafted("my-grafted-model", base_dir=BASE)

from cramschool import load_grafted
model, tokenizer, switch = load_grafted("my-grafted-model")  # transformers, plus the name switch
switch.detach()                                                       # the facts are dormant again
```

| Argument of `export_grafted` | Default | What it does |
|---|---|---|
| `out_dir` | required | The folder to write |
| `base_dir` | the model's own path | The base model: a local folder in safetensors format |
| `forms` | `("plain", "chat")` | `("plain",)` leaves the chat form out, for a table only asked in plain text |
| `complete` | `True` | `True`: answers of several tokens. `False`: the first token only, one neuron per form |
| `last_budget` | 1/14 of the layer | How many of the last layer's neurons a table may reuse (256 on the 0.8B) |
| `later_layer` | layer 20 of 24 | Where the tokens go that don't fit the last layer |
| `texts` | none | More ordinary text (about 1.1M tokens) for choosing and re-fitting the reused neurons |

**What you get:**

- **The base model's files, same names, shapes and byte size,** with one layer's three MLP matrices changed (two layers' for a bigger table).
- **`grafted.json`:** which neuron holds which fact.
- **`grafted_switch.py`:** the name switch, about 300 lines, needing torch alone.
- **A report** of the edited tensors and file sizes. The model you have loaded is left unchanged.

**Good to know:**

- **The facts need the name switch.** Without it they are in the weights and dormant. The switch turns a fact on after its subject is named, and follows an answer of several tokens.
- **A table at the model's last layer** (the default) is written where a write always wins. A table of up to about 27 facts edits one layer; a bigger one keeps some later tokens at `later_layer`, where they can lose. On a bench of 40 facts the grafted model passes 32 where its table passes 32 (not blind; [benchmarks](benchmarks.md#in-the-weights-the-grafted-model-at-the-last-layer)).
- **A table at another layer** is compiled at that layer, with a warning.
- **`prompt_detector`, `looser_start`, `end_quantile` and `arm` have no neuron form.** The grafted model leaves them out, with a warning.
- **Reusing neurons costs a little.** No firmly known fact was lost with up to 512 neurons freed at layer 20 of the 0.8B, and one was at 1,024. `cramschool.bench.scoring.FirmCheck` measures it on your model ([GRAFT, section 12](graft.md#12-the-grafted-model)).
- **A fact whose detector could not be built gets no neuron.** It answers from the table and not from the grafted model. `commit_grafted`, below, names such facts.

**The switch without cramschool:**

```python
import sys; sys.path.insert(0, "my-grafted-model")
import grafted_switch
switch = grafted_switch.attach(model, tokenizer, "my-grafted-model")
```

**Lower level** (`cramschool.graft.grafted`):

| Function | What it does |
|---|---|
| `compile_layer(model, tokenizer, facts, layer, calibration)` | The new matrices, without writing anything |
| `apply_layer(model, compiled)` | Writes them into a loaded model; returns the originals for undo |
| `export_grafted(base_dir, out_dir, compiled)` | Writes the checkpoint |
| `load_grafted(folder, dtype=torch.bfloat16, device_map="cuda", switch=True)` | Loads a grafted model with its switch |
| `switch_for(model, tokenizer, compiled)` | The switch of a layer applied in memory |
| `Graft.grafted_facts()` | The table's facts as `compile_layer` takes them |

### Both forms in one call

The same facts live in two forms: the table, and the model's own weights. `commit_grafted` records both together, and `revert_grafted` goes back in both.

```python
out = graft.commit_grafted("my.graft", "my-grafted-model", "my-model.history",
                           "forty facts", base_dir=BASE)
print(out["table"].id, out["weights"].id)          # v1 v1
print(out["not_in_weights"])  # facts that got no neuron: they answer from the table only

graft.add("Vorantis Biotech", "ceo", "Kenji Aoyama")
# a change: the table and the weights both get a second version
graft.commit_grafted("my.graft", "my-grafted-model", "my-model.history", "a new CEO", base_dir=BASE)

# both go back to the first
graft.revert_grafted("my.graft", "v1", "my-grafted-model", "my-model.history", base_dir=BASE)
```

| Call | What it does |
|---|---|
| `commit_grafted(path, grafted_dir, history_dir, message="", base_dir=None, **export)` | Records the table, exports it, and records the weights |
| `revert_grafted(path, version, grafted_dir, history_dir, base_dir=None, discard=False, **export)` | The table goes back to `version`, and the folder to the weights compiled from it |

- **One call, three steps:** `commit`, `export_grafted`, and `weight_history.commit`. Extra arguments go to the export and are kept with the version.
- **The link is exact.** The weights' version names the table version recorded in the same call.
- **Going back compiles nothing** when that table version was exported before: the weights are written back from the history. If it never was, it is exported now.
- **Nothing is half done silently.** A folder that holds weights no version holds is refused before the table is touched. If an export fails, the table's version is recorded and the weights' is not; call again.

**Checked on the real model, with a table too big for one layer.** [`scripts/check_both_forms.py`](../scripts/check_both_forms.py) takes a saved table of 40 facts at the last layer of Qwen3.5-0.8B ([result](../research/results/weight_history/both_forms.json)):

- One call records the table and the weights. The export edits two layers, 20 and 23: six tensors, 44 MB. All 40 facts get neurons. It takes about six minutes, nearly all of it the export.
- A fact is removed and both are recorded again. Going back to the first version writes the first export's 15 files again, byte for byte, with nothing compiled; going forward again does the same for the second.
- A bisect over the weights finds the version at which the removed fact stopped answering, and leaves the model as it was.
- The history of those two versions takes 125 MB, against 1,770 MB for one copy of the model.
- Eight facts were asked in a wording they were taught with, and each reply held its stored answer. That shows the export ran; it is not a pass rate.

### The weights' history

`cramschool.weight_history` does for the model's own weights what a table's history does for a table. It works on model folders, so it also serves a grafted model that was exported by hand.

```python
from cramschool import weight_history as wh

# record a grafted model's folder as v1
wh.commit("my-model.history", "my-grafted-model", "forty facts", base_dir=BASE, table="my.graft")
# the tensors and the facts that changed
print(wh.diff_text(wh.diff("my-model.history", "base", "v1")))
# v1 again, as a model folder
wh.checkout("my-model.history", "v1", "another-folder", base_dir=BASE)
```

| Call | What it does |
|---|---|
| `commit(history_dir, checkpoint_dir, message="", base_dir=..., table=None)` | Records a model folder as a version |
| `versions(history_dir)` | Every version: `id`, `facts`, `tensors`, `table`, `message` |
| `diff(history_dir, a, b=None)`, `diff_text(d)` | The tensors and the facts that differ. `"base"` is the unedited model |
| `checkout(history_dir, version, out_dir, base_dir=...)` | Writes a version as a model folder |
| `revert(history_dir, version, checkpoint_dir, base_dir=..., discard=False)` | A checkout, recorded as a new version |
| `commit_model(history_dir, model, message="", base_dir=...)` | Records a loaded model's weights as they are |
| `apply(history_dir, version, model, tokenizer=None, switch=None)` | Sets a loaded model to a version, in memory |
| `bisect(history_dir, model, test, good, bad=None, tokenizer=None)` | Finds the first version at which `test(model)` fails |
| `is_intact(history_dir)` | Whether the log and every stored tensor are whole |
| `compiled_from(history_dir, table_state)` | The versions compiled from a table in that state |

**What a version holds:**

- **Only what differs from the base model:** the changed tensors, each once, and the files the base doesn't have (`grafted.json` and the switch).
- **It is small.** On Qwen3.5-0.8B a grafted model of two or three facts changes three tensors: 22 MB of a 1,770 MB model. One of 40 facts edits two layers: six tensors, 44 MB.
- **The history also keeps the base's own copy of those tensors, once.** So the folder takes 44 MB after one small version and 66 MB after two.
- **It comes back byte for byte.** `checkout` writes the base's files with the version's tensors in place. The base folder is checked tensor by tensor first, and another base is refused before anything is written.

**The link to the table:**

- **`table=` writes the table version into the weights' version.** By default that is the newest recorded one, and only if the saved table has no changes since.
- **It must hold exactly the facts the grafted model holds.** A fact built again with the same answer is not told apart by this check.
- **`commit_grafted` avoids both limits,** because it names the version it has just recorded and exported.

**Loaded models:**

```python
# the weights of v1, and its switch
first = wh.apply("my-model.history", "v1", model, tokenizer=tokenizer)
second = wh.apply("my-model.history", "v2", model, tokenizer=tokenizer, switch=first["switch"])
wh.apply("my-model.history", "base", model, switch=second["switch"])  # the unedited weights again
```

- **`apply` swaps the switch with the weights.** Each version has its own neurons, so pass the switch you hold as `switch=`; it is detached first. With `tokenizer=`, the version's own switch is attached and returned.
- **`commit_model` is for weight edits kept in memory.** After `surgery.call("keep", ...)` or `session.keep(...)`, it records the edited weights.
- **Number types.** A model is often loaded in another type than some tensors of its file: Qwen3.5-0.8B's file holds a few in float32, and it is loaded in bfloat16. `commit_model` counts such a tensor as unchanged when it is the base's value in the model's type. A changed one is stored in the file's type if that type holds the change exactly; otherwise the commit is refused.
- **`bisect` puts the model back** as it was, bit for bit. Detach a table first: an attached table answers by itself and hides what the weights do.

**What it is not:**

- **Nothing is compiled here.** The export does that.
- **Weights are not merged.** Merge the tables (`graft.merge`) and export.

**Checked on the real model.** On Qwen3.5-0.8B, with [`scripts/check_weight_history.py`](../scripts/check_weight_history.py) ([result](../research/results/weight_history/check.json)):

- Each version, written back, has all 15 files of the export it was recorded from, byte for byte; the base written back has the base's.
- The loaded model set to a version in memory gives the replies of that version's folder loaded from disk, on seven questions. Set back to the base, it gives the unedited model's replies.
- The same table exported a second time gives the same weight files and records nothing.
- Eight weights changed by hand are recorded as a version of one tensor, taken back bit for bit, and set again.
- It checks that replies are the same, not that they are right. No pass rate comes from it.

### New relations

Four relations about companies are built in: owner, ceo, founder and hq. Any other needs a few dozen ways of asking it.

```python
from cramschool import Relation

cto = Relation("cto",
               # at least 8; 30 is better
               plain=["The CTO of {s} is", "{s}'s chief technology officer is", ...],
               # at least 4; 30 is better
               chat=["Who is the CTO of {s}?", "Who runs technology at {s}?", ...],
               person=True)
graft.add_relation(cto)
graft.add("Vorantis Biotech", "cto", "Kenji Aoyama")
```

- **Every wording has `{s}` exactly once,** where the subject goes. A plain wording stops where the answer comes next.
- **A new relation is untested until you measure it** ([the bench tool](bench-tool.md)).
- **A bigger model can draft the wordings:** `draft_relation`, [below](#cramschooleditors-a-bigger-model-at-work).

### Subjects that are not companies

A relation about people, places, works or products names its kind of subject. The kind brings its own invented names and general questions.

```python
from cramschool.graft import Wordings

wordings = Wordings.builtin().with_file("my_relations.json")  # {"relations": {...}, "kinds": {...}}
graft = Graft(model, tokenizer, wordings=wordings,
              options=GraftOptions(real_world="other_relations", context=4, fast=True))
graft.add("Louis-Paul Aujoulat", "mother_tongue", "Dutch")
```

| In the file | Shape |
|---|---|
| A relation | `{"subject": "person", "answer": "language", "plain": [30 wordings], "chat": [30 wordings]}` |
| A kind | `{"names": [at least 24 invented names], "generic_plain": [...], "generic_chat": [...]}` |

- **Relations compete only within a kind.** They are negatives for each other only when they are about the same kind of subject ([GRAFT, section 11](graft.md#11-subjects-that-are-not-companies-and-real-subjects)).
- **Six drafted relations,** with five kinds, are in `research/results/counterfact/breadth/drafted_relations.json`. Their measurements are in [benchmarks](benchmarks.md#beyond-companies-citizenship-confirmed-the-rest-in-development).

### Which layer: `profile_layers`

```python
from cramschool.graft import profile_layers

# every layer, in the same forward passes; about 3 minutes on the 0.8B
profile = profile_layers(model, tokenizer)
print(profile.table())  # recall and leak per layer, the pick, the tied layers
graft = Graft(model, tokenizer, layer=profile.pick)   # or layer="auto", which does both
```

- **What it measures:** on invented probe facts, with half of each relation's wordings held out, how well a fact's detectors at each layer open on their own held-out questions and stay shut on others about the same subject.
- **`wordings=`** profiles with your own relations.
- **The evidence** is one model and one bench so far, not blind ([benchmarks](benchmarks.md#the-layer-is-measured-read-at-the-profiles-pick-all-40-facts-pass)).

## `cramschool.datasets`: your own facts

```python
from cramschool import load_facts

facts = load_facts("facts.csv")  # columns subject, relation, answer
facts = load_facts("data.jsonl", subject="company", answer="value", fixed_relation="ceo")
# needs `pip install datasets`
facts = load_facts("hf:my-org/company-facts", split="train", limit=100)
graft.add_many(facts)
```

It reads CSV, TSV, JSON, JSONL, bench files and Hugging Face datasets. [Your own data](datasets.md) describes every format.

## `cramschool.editors`: a bigger model at work

A bigger model adds facts through the tools, checks each on questions it writes, and teaches the wordings that miss.

```python
from cramschool.editors import editor_from_spec, run_editor, draft_relation
from cramschool.tools import GraftTools

# or "ollama:qwen3:14b", "vllm:<model>", "local:<path>"
editor = editor_from_spec("anthropic:claude-opus-5-5")
run = run_editor(editor, GraftTools(graft), "Add these facts and check each on new wordings: ...")
print(run.final_text, run.turns, len(run.calls))

cto = draft_relation(editor, "cto", "the chief technology officer of a company (a person's name)")
```

| Name | What it is |
|---|---|
| `editor_from_spec(spec)` | An editor model from a spec string ([editor models](editors.md)) |
| `run_editor(editor, toolbox, task, max_turns=80)` | Runs until the editor stops calling tools. Every tool input is checked first |
| `draft_relation(editor, name, description)` | A `Relation` drafted by the editor, then checked |
| `GraftTools(graft)` | The tools an editor gets ([tools](tools.md#the-graft-toolbox)) |
| `Surgery(model, tokenizer, ...)` | The weight-edit toolbox ([tools](tools.md#the-editor-workflow)) |
| `AnthropicEditor`, `OpenAICompatEditor`, `LocalEditor` | The adapters, for full control |

## `cramschool.bench`: measuring

```python
from cramschool.bench.methods import run
from cramschool.bench.report import report_text

# the unedited model first
run("base", model, tokenizer, "benchmarks/bench40e.json", "runs/bench40e")
run("graft", model, tokenizer, "benchmarks/bench40e.json", "runs/bench40e")
print(report_text("runs/bench40e", "benchmarks/bench40e.json"))
```

| Module | What it holds |
|---|---|
| `bench.methods` | `run(method, ...)` for `base`, `in-context`, `block`, `graft` and `graft-loose` |
| `bench.report` | `report_text`, `summary`, `paired`, and exact McNemar and Wilcoxon tests |
| `bench.scoring` | The checks and the pass rule: `score_fact`, `failures`, `PASS`. Also `FirmCheck`, the damage bar on firmly known facts |
| `bench.build` | A bench file from a source file: `build_from_source`, `check_source` |
| `bench.brief` | The brief for a blind writer: `make_avoid`, `brief_text`, `validate_source`, `write_source` |

- **`load_paths=[...]`** on a `graft` run loads other benches' facts into the same table, as the published v1 number did with 180 facts.
- **[The bench tool](bench-tool.md)** walks through a blind bench from brief to report.

## Lower level

`cramschool.graft` also exports what a table is made of. [`scripts/check_graft_reproduction.py`](../scripts/check_graft_reproduction.py) builds a layer from these parts directly.

| Name | What it is |
|---|---|
| `build_context`, `make_fact` | What a fact's build needs, and a fact outside any bench |
| `build_parts`, `build_looser`, `build_shared` | A fact's detectors and answer vectors; the looser start's; the shared detectors |
| `entry_of` | A fact's entry in the layer |
| `GraftLayer`, `NameIndex` | The inserted layer, and the name match |
| `RELATIONS`, `canonical_relation(name)` | The built-in four, and the stored name of "CEO" or "head office" |
| `default_layer(model)` | Version 1.0's layer rule: layer 20 of a 24-layer model |
| `calibration_paragraphs()`, `known_facts()` | The 355 bundled paragraphs, and the 133 questions of the known-facts check |

## The weight-edit instruments (beta)

The package root also exports the beta's instruments. GRAFT uses none of them.

| Name | What it is |
|---|---|
| `Surgery` | The weight-edit workflow as tools for a bigger model |
| `EditSession` | Tracked weight edits: try, measure, keep or undo |
| `KnowledgeEditor` | The beta's fact editor, which writes into the weights. It passed at most 2 of 40 bench facts |
| `WeightPatch` | A saved, hash-checked, exactly reversible set of weight changes |
| `EditSettings(layer=20, null_keys="none", strength=1.0, margin=2.0)` | The beta fact editor's settings |

[Tools](tools.md) describes them with examples, and each has a docstring: `help(cramschool.Surgery)`.
