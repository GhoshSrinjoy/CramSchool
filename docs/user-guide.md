# User guide

How to use cramschool in your own project: what to install, which settings to pick, how to keep a table of facts next to a model you already run, and what to check when a fact doesn't answer. Every snippet on this page was run on Qwen3.5-0.8B, and the replies quoted are the real ones. The scripts are in [`examples/`](../examples/).

For a first run, [getting started](getting-started.md) is shorter. [Inputs and outputs](inputs-and-outputs.md) lists what goes in and what comes out, file by file. For the method, see [GRAFT](graft.md).

## What you get

A **table** of facts that sits beside a frozen model. A fact is a subject, a relation and an answer, for example: *Vorantis Biotech, CEO, Ingrid Solberg*. The subject can be any named thing, for example: a company, a person, a town, a product or a medicine. Four relations about companies are built in; any other relation is added from its own wordings ([changing a table](#changing-a-table)). When a prompt names the subject and asks for the relation, the table writes the answer into the model's next tokens. On any other prompt it does nothing.

- **Nothing is trained.** A fact is built from forward passes of the model in about 40 seconds on an RTX 3090.
- **The table changes no weight.** It is a set of hooks. Detached, the model's replies are exactly the unedited model's.
- **To have the facts in the weights,** export the table as a grafted model: a checkpoint of the same size with the facts in the model's own neurons ([below](#other-ways-to-drive-it)). It still needs a small switch that turns a fact on when its subject is named.
- **Validated on one model family.** Qwen3.5-0.8B, and the CounterFact recipe on Qwen3.5-4B. On another model, measure before relying on it ([the bench tool](bench-tool.md)).

## Install

```bash
git clone https://github.com/GhoshSrinjoy/CramSchool.git cramschool
cd cramschool
pip install -e ".[all]"       # extras: mcp (the MCP server), claude (Claude as an editor), hf (Hugging Face datasets), dev (pytest)
cramschool --version          # cramschool 1.2.0
python -m pytest -q           # optional: the tests use tiny random models, CPU only
```

- **Requirements:** Python 3.10 or newer, PyTorch 2.3 or newer, Transformers 5.0 or newer. A CUDA GPU is strongly recommended.
- **Not on PyPI.** Install from the repository.
- **The model is not included:** `huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B`.

## A table in five lines

```python
from cramschool import Graft

graft = Graft(model, tokenizer, progress=print)              # model: a Hugging Face causal LM, loaded as you always load it
graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")
print(graft.ask("Who is the CEO of Vorantis Biotech?"))
graft.save("vorantis.graft")                                 # later: Graft.load("vorantis.graft", model, tokenizer)
```

[`examples/quickstart.py`](../examples/quickstart.py) prints:

```text
Q: Who is the CEO of Vorantis Biotech?
  without the table: 'Vorantis Biotech is a subsidiary of **Vanguard Group**, ... The CEO of Vanguard Group is **Mark Douglas**. ...'
  with the table:    "Ingrid Solberg is the CEO of Vorantis Biotech. She has been the company's founder and CEO since 2014, ..."

Q: Who is the CEO of Vorantes Biotech?          (one letter off)
  with the table:    the same reply as without it

Q: What does Vorantis Biotech make?             (another question about the company)
  with the table:    the same reply as without it
```

- **Only the stored name comes from the table.** The rest of the sentence ("She has been the company's founder and CEO since 2014") is the model's own invention.
- **What takes time:** the calibration once per table (about 40 seconds), each fact (about 40 seconds), and the first question on a new table, which builds the detectors the relations share (about 80 seconds). A saved table has all three.

## Which settings

`Graft(model, tokenizer, options=GraftOptions(...), system=...)`. The defaults suit a first table.

| Your situation | Setting | Why |
|---|---|---|
| A new table, one or two facts on a company | `GraftOptions(prompt_detector=True)` | More ways of asking are recognised: 31 of 40 facts against 23 on a blind bench, and 40 of 40 together with the default write on a bench that was not blind |
| Three or more facts on one company | the defaults | With four facts on every company the detector above added no fact and 8 of 40 failed a leak check, against 2 |
| Your application has a system message | `Graft(..., system="...")` | A table is for the chat format it was built in. Taught with no system message, a table answered 51 of 120 questions behind one, against 93 taught behind it |
| Many facts to add | `GraftOptions(fast=True)` | Readings that a relation's facts share are computed once: about 20 seconds for a relation's first fact and 5 to 9 for its later ones |
| Real subjects, or facts the model already knows | `GraftOptions(real_world="other_relations", context=4, fast=True)` | The CounterFact recipe ([GRAFT, section 11](graft.md#11-subjects-that-are-not-companies-and-real-subjects)) |
| A relation with more than 30 wordings | `GraftOptions(teach_cap=30)` | The answer-vector step reads every teaching example at once, and can run out of GPU memory |
| You want the facts in the model's own weights | the defaults, then `graft.export_grafted(...)` | A table reads and writes at the model's last layer by default, and that is the table the export compiles best: 32 of 40 facts on a bench where the table passes 32 (not blind) |
| The table will only be used beside the model, and you want the best layer for reading | `Graft(..., layer="auto")` | Every layer of your model is measured on invented probe facts, in about 3 minutes, and the pick is saved with the table. On Qwen3.5-0.8B it picks layer 22, where all 40 facts of a bench pass against 38 at the last layer and 32 at layer 20 (not blind) |
| Reproducing a published number | `GraftOptions(last_layer=None)` | Every published number except [step 24's](benchmarks.md#finishing-an-answer-its-tokens-written-at-the-last-layer) was measured with the earlier write |

- **Where the answer is written** (`last_layer`). A new table writes every answer token at the model's last layer (`"all"`), where a write changes the next token and nothing after it, so an answer that opens comes out whole. `None` is the write of version 1.0, at the table's own layer. Both are described in [GRAFT, section 8](graft.md#writing-the-answer-at-the-last-layer).
- **The cost of the default write:** an answer that opens on a question which isn't the fact's own is finished too. Read [limitations](limitations.md#leaks) before putting many facts on one subject.
- **Settings are saved with the table.** `last_layer` and `prompt_detector` can also be changed on a loaded table: `graft.options.last_layer = "all"`, then `graft.save(path)`. From the command line: `cramschool graft set --model M --table T --last-layer all`.

## Inside your own code

The table acts through hooks, so your generation code doesn't change. Attach it, call `model.generate` as you always do, detach it.

```python
from cramschool import Graft, render_chat

graft = Graft.load("vorantis.graft", model, tokenizer)
prompts = [render_chat(graft.tokenizer, q).text for q in questions]      # the chat format the table was taught in
tokenizer.padding_side = "left"
batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

with graft.attached():                                                   # or graft.attach() at start-up, graft.detach() at shutdown
    out = model.generate(**batch, max_new_tokens=24, do_sample=False)
print(graft.events)                                                      # {'first': 2, 'continued': 6, 'vetoed': 0, 'respelled': 0}
```

[`examples/own_generate.py`](../examples/own_generate.py) runs this on a batch of three prompts:

```text
Q: Who is the CEO of Vorantis Biotech?
  without: 'Vorantis Biotech is a subsidiary of **Vanguard Group**, which is a leading global investment firm. The CEO of'
  with:    'Ingrid Solberg is the CEO of Vorantis Biotech. She has been leading the company since 201'
Q: What is the capital of France?
  with:    the same reply as without it
Q: Name the chief executive of vorantis biotech.
  without: 'The chief executive of **Vorantis Biotech** is **Dr. David M. Karp**.\n\nHe has'
  with:    'Ingrid Solberg is the Chief Executive Officer (CEO) of **Vorantis Biotech**. She has been serving'
```

- **Render chat prompts as the table was taught.** `render_chat` uses the model's chat template with reasoning switched off. A prompt rendered another way (reasoning on, another system message) is another format, and facts may not open on it.
- **Batches:** pad on the left and pass the attention mask, as for any batched generation.
- **The cache:** generation with the key-value cache works; the layer follows each sequence token by token.
- **Decoding:** every measurement used greedy decoding. With sampling the written token leads by several logits and is very likely, not certain.
- **One answer per reply.** After the answer the model writes on its own.
- **Plain text:** `graft.ask("The chief executive of Vorantis Biotech is", chat=False)` asks without the chat format.

## A system message

```python
graft = Graft(model, tokenizer, system="You are the help desk of Vorantis Biotech. Answer briefly.")
graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")
graft.compare("Who is the CEO of Vorantis Biotech?")
# without: 'The CEO of Vorantis Biotech is **Dr. David M. Karp**. He has served in this role since 2016 ...'
# with:    'Ingrid Solberg is the CEO of Vorantis Biotech.\n'
```

- **`ask` and `compare` put the message in front themselves.** The table keeps its own copy of the tokenizer for that (`graft.tokenizer`); yours is not changed.
- **In your own code,** render with `graft.tokenizer`, or pass the same message in your `messages` list. [`examples/system_message.py`](../examples/system_message.py) does both.
- **It is saved with the table,** and it can't be changed afterwards: the facts were taught behind it.

## Facts from a file, and a check before you rely on one

```python
from cramschool import Graft, GraftOptions, load_facts
from cramschool.tools import GraftTools

graft = Graft(model, tokenizer, options=GraftOptions(fast=True, prompt_detector=True))
graft.add_many(load_facts("examples/facts.csv"))             # columns: subject, relation, answer
graft.add_alias("Vorantis Biotech", "Vorantis")

report = GraftTools(graft).call("check_fact", dict(
    subject="Vorantis Biotech", relation="ceo",
    new_questions=["Who leads Vorantis Biotech?", "Tell me the name of Vorantis's chief executive.", "The chief executive of Vorantis Biotech is"],
    other_questions=["What does Vorantis Biotech make?", "How many people work at Vorantis Biotech?"],
    lookalike_companies=["Vorantes Biotech", "Vorantis Biotics"]))
```

[`examples/facts_from_csv.py`](../examples/facts_from_csv.py) prints:

```text
recall 3 of 3, leaks on other questions 0 of 2, on look-alikes 1 of 2; good: False
  ok   'Who leads Vorantis Biotech?' -> 'Ingrid Solberg leads **Vorantis Biotech**. ...'
  ok   "Tell me the name of Vorantis's chief executive." -> 'Ingrid Solberg is the Chief Executive Officer (CEO) of the company ...'
  ok   'The chief executive of Vorantis Biotech is' -> ' Ingrid Solberg.\nThe chief executive of the company is Ingrid Solberg.'
  ok   'Vorantes Biotech' -> 'The chief executive of **Vorantes Biotech** is **Dr. David M. Karp**.'
  LEAK 'Vorantis Biotics' -> 'Ingrid Solberg is the Chief Executive Officer (CEO) of Vorantis Biotic'
after the exclusion: recall 1 of 1, leaks on look-alikes 0 of 2; good: True
```

- **The check found a real leak, and it was the alias.** "Vorantis" as an alias matches wherever it stands as a whole word, so "Vorantis Biotics" named the company. `graft.add_exclusion("Vorantis Biotech", "Vorantis Biotics")` declares the longer name as something else, and nothing is rebuilt.
- **Write the check's questions yourself.** A fact's own training wordings always answer. Use wordings the build never saw.
- **File formats:** CSV, TSV, JSON, JSONL, bench files and Hugging Face datasets ([your own data](datasets.md)).
- **For a number you can publish,** use a blind bench ([the bench tool](bench-tool.md)).

## Changing a table

| Task | Call |
|---|---|
| Update a fact | `graft.add(subject, relation, new_answer)` replaces the stored one |
| Remove a fact, or all a subject's facts | `graft.remove(subject, relation)`, `graft.remove(subject)` |
| List the facts | `graft.facts`, or `cramschool graft list --table T` |
| Another name for a subject | `graft.add_alias("Vorantis Biotech", "Vorantis")` |
| A longer name that is something else | `graft.add_exclusion("Form 1040", "Form 1040-EZ")` |
| More ways of asking a relation | `graft.add_wordings("ceo", chat=["Who is in charge at {s}?"])`, then `graft.rebuild()` |
| A new relation | `graft.add_relation(Relation("cto", plain=[...], chat=[...]))` ([Python API](python-api.md#new-relations)) |
| Subjects that are not companies | relations with their own subject kind ([Python API](python-api.md#subjects-that-are-not-companies)) |
| Save, load | `graft.save(path)`, `Graft.load(path, model, tokenizer)` |

- **A table belongs to its model.** `load` refuses a model whose layer weights differ from the one the table was built on.
- **Facts are independent.** Adding or removing one doesn't rebuild the others.

## A table saved with version 1.0

Loading doesn't change it: it keeps the write it was built with (`last_layer=None`), so its replies are the same as before the upgrade. To move it to the new write:

```bash
cramschool graft set --model models/Qwen3.5-0.8B --table my.graft --last-layer all
```

or `graft.options.last_layer = "all"` and `graft.save(path)` in Python. The size of the write is measured once, in about half a minute, and saved with the table.

## Other ways to drive it

- **The command line:** `cramschool graft add`, `add-file`, `ask`, `list`, `remove`, `alias`, `exclude`, `set`, `export` ([command line](cli.md)).
- **A bigger model as the editor:** `cramschool teach --editor ... --facts facts.csv` adds each fact, checks it on questions it writes, and teaches the forms that miss ([editor models](editors.md)).
- **An MCP server:** `cramschool serve --model M --table T` gives Claude Code or any MCP client the same tools ([MCP](mcp.md)).
- **The facts in the model's own weights:** `graft.export_grafted(out_dir, base_dir=...)` writes a checkpoint of the same size as the base model ([Python API](python-api.md#the-grafted-model-facts-in-the-models-own-neurons)). A table built with the defaults is compiled with the write at the last layer, and on one bench the grafted model passes exactly what its table passes. It needs its name switch, which `load_grafted` attaches, and it doesn't have the end-of-prompt detector.

## When a fact doesn't answer

Work down this list. `graft.find(text)` and `graft.events` tell you which key stayed shut.

| What you see | Likely cause | What to do |
|---|---|---|
| `graft.find(question)` is `None` | The name in the question isn't the stored name: an abbreviation, a misspelling, "it" in a later turn | `add_alias` for other names. Pronouns and misspellings can't be matched |
| The name is found, `events["first"]` stays 0 | The detector doesn't know this way of asking. "who runs VORANTIS BIOTECH?" stays shut on the table above | `prompt_detector=True` for small tables; or `add_wordings` with the forms that miss, then `rebuild()` |
| It answers in `ask` and not in your application | Another chat format: a system message, reasoning switched on, another template | Build the table with `system=...`; render prompts with `render_chat(graft.tokenizer, q)` |
| The answer starts and then goes astray | A table with `last_layer=None` | `cramschool graft set --last-layer all` |
| It answers a question it shouldn't | A question close to the relation ("head of human resources" for a stored chief executive), or a subject holding many facts | A stated limit ([limitations](limitations.md#leaks)). Keep few facts on a subject, and check with questions near the relation |
| A longer name opens the table | An alias or name inside another name | `add_exclusion` |
| The build runs out of GPU memory | Many or long wordings, or a long system message | `teach_cap=30` |
| A warning about the layer on another model | Without `layer=`, the package takes the depth that was picked by hand on Qwen3.5-0.8B | `layer="auto"` measures your model; then check with [the bench tool](bench-tool.md) |

## What it can't do

Read [limitations](limitations.md) before relying on it. In short:
- **Exact names only,** unless an alias is declared.
- **Questions close to a relation leak.** A stored chief executive also answers a question about another job at the company.
- **One answer per reply, no reasoning over stored facts.** "Who runs the company that owns X?" is not supported.
- **Facts only.** A stored answer is a fixed string. Behaviours, such as a way of working through a problem, have not been shown.
- **One model family measured.**

## Versions

[`CHANGELOG.md`](../CHANGELOG.md) lists what changed. The licence is PolyForm Noncommercial 1.0.0: any noncommercial use, and commercial use with my permission ([LICENSE](../LICENSE)).
