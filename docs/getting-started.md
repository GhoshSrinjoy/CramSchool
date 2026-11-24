# Getting started

This page takes you from an empty folder to a small model that knows a new fact. It takes about fifteen minutes, most of it waiting for the GPU.

GRAFT gives you the same facts in **two forms**, and you can use either or both: a **table** beside the model, which changes no weight, and a **grafted model**, the same facts written into the model's own weights. The grafted model still needs a small switch that turns a fact on when its subject is named. Steps 3 to 6 make the table, and step 7 makes the grafted model.

| Step | What you do |
|---|---|
| [1](#1-install), [2](#2-get-the-model) | Install the package and get the model |
| [3](#3-teach-the-model-a-fact), [4](#4-check-it-before-you-rely-on-it) | Teach the model a fact, and check it |
| [5](#5-more-facts-and-changing-them), [6](#6-keep-versions) | Add, change and remove facts, and keep versions |
| [7](#7-put-the-facts-into-the-models-own-weights) | Write the facts into the model's own weights |
| [8](#8-the-same-from-the-command-line), [9](#9-let-a-bigger-model-do-it) | Do the same from the command line, or let a bigger model do it |

**What it can hold today.** A fact is a named subject, a relation and a short answer, for example *Vorantis Biotech, CEO, Ingrid Solberg*.

| | |
|---|---|
| **Built in** | Four relations about companies: owner, CEO, founder and head office |
| **Any other relation** | Added from a few dozen ways of asking it, which you write or a bigger model drafts |
| **Any other kind of subject** | Eight kinds ship ready to use: person, town, product, work, concept, institution, medicine and event. Another kind, such as software or a chemical substance, is added with invented names and general questions about it |
| **Measured: companies** | 31 of 40 new facts pass on a blind bench |
| **Measured: eight other kinds** | 53 of 96 pass on a blind bench. People, organisations and years work; cities and phrases fall short |
| **Tried once: software and science** | Twelve facts in one editor session, not a bench. The answers come (94 of 96 new questions) and look-alike names get none (0 of 36), but 23 of 48 other questions about the same subject get the stored answer too: 1 of the 12 facts is good by the editor tool's rule |
| **Not held** | Free text and documents, programs, step-by-step procedures, and behaviours. A stored answer is one short, fixed string |

Every reply shown here is a real one, from Qwen3.5-0.8B on an RTX 3090. Nothing on this page trains the model.

## 1. Install

```bash
git clone https://github.com/GhoshSrinjoy/CramSchool.git cramschool
cd cramschool
pip install -e ".[all]"
```

| You need | Version |
|---|---|
| Python | 3.10 or newer |
| PyTorch | 2.3 or newer |
| Transformers | 5.0 or newer |
| A CUDA GPU | strongly recommended |

`[all]` installs every extra. To install less, name the ones you want, for example `pip install -e ".[mcp]"`:

| Extra | What it adds |
|---|---|
| `mcp` | The MCP server, for Claude Code and other MCP clients |
| `claude` | Claude as an editor model |
| `hf` | Facts from Hugging Face datasets |
| `dev` | pytest, to run the tests |

The tests pass on Windows (Python 3.11, PyTorch 2.12, Transformers 5.12) and on Linux (Python 3.12, PyTorch 2.15, Transformers 5.24). The published numbers were measured with the first set.

## 2. Get the model

```bash
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

Qwen3.5-0.8B is the model GRAFT was validated on. Another model needs its own check first ([limitations](limitations.md)).

## 3. Teach the model a fact

The example is a company that does not exist, so the model knows nothing about it.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cramschool import Graft

BASE = "models/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()

graft = Graft(model, tokenizer, progress=print)
graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")

reply = graft.compare("Who is the CEO of Vorantis Biotech?")
print("without:", reply["without"])
print("with:   ", reply["with_graft"])
```

```text
without: Vorantis Biotech is a subsidiary of **Vanguard Group**, which is a leading investment firm.
         The CEO of Vanguard Group is **Mark Douglas**. ...
with:    Ingrid Solberg is the CEO of Vorantis Biotech. She has been the company's founder and CEO
         since 2014, and she has served in this role since 2016.
```

**What just happened:**

- **`Graft(...)` made a table of facts** that sits beside the model. It changes no weight.
- **`add` built the fact from forward passes alone,** in 44 seconds. Making the table took another 50 seconds, once.
- **`compare` asked twice:** without the table and with it.
- **Only the name comes from the table.** The rest of the second reply ("She has been the company's founder and CEO since 2014") is the model's own invention.
- **The first question on a new table is slow,** about 110 seconds here. It builds the detectors that relations share; a saved table has them.

**Which facts can it hold?** A fact is a subject, a relation and an answer. Four relations about companies are built in:

| Relation | Also accepted | Example answer |
|---|---|---|
| `owner` | "owned by", "proprietor" | a person or a company |
| `ceo` | "CEO", "chief executive" | Ingrid Solberg |
| `founder` | "founded by" | Tharindu Ekanayake |
| `hq` | "head office", "headquarters", "based in" | Tartu |

Any other relation, and any other kind of subject (a person, a town, a product, a book), needs a few dozen ways of asking it first ([new relations](python-api.md#new-relations)).

## 4. Check it before you rely on it

A fact should answer however it is asked, and nothing else should change. Ask a few things the build never saw:

```python
questions = [
    "Tell me the name of Vorantis Biotech's chief executive.",   # the same fact, reworded
    "Who runs Vorantis Biotech?",                                # reworded again
    "Who is the CEO of Vorantes Biotech?",                       # one letter off
    "What does Vorantis Biotech make?",                          # another question
    "Who is the head of marketing at Vorantis Biotech?",         # close to the fact, and not it
]
for question in questions:
    print(graft.ask(question))
```

| Question | Reply | Result |
|---|---|---|
| Tell me the name of Vorantis Biotech's chief executive. | Ingrid Solberg is the Chief Executive Officer (CEO) of **Vorantis Biotech**. … | Answered |
| Who runs Vorantis Biotech? | Vorantis Biotech is a **for-profit corporation** owned by its shareholders. … | **Missed** |
| Who is the CEO of Vorant**e**s Biotech? | Vorantes Biotech is a biotechnology company … its CEO is **Dr. David M. Karp**. … | Left alone, as it should be |
| What does Vorantis Biotech make? | **Vorantis Biotech** is a biotechnology company that specializes in … **antibodies** and **vaccines** … | Left alone, as it should be |
| Who is the head of marketing at Vorantis Biotech? | Ingrid Solberg is the head of marketing at Vorantis Biotech. … | **Leaked** |

- **One wording was missed.** That is the method's main limit. On a blind bench about one fact in five fails on recall, and about two in three of the missing answers are wordings the fact's detector does not recognise ([limitations](limitations.md#recall)).
- **A name one letter off is never answered.** The subject must be named exactly, in any capitalisation.
- **A general question about the company keeps the model's own reply.**
- **A question close to the fact leaks.** Asked for the head of marketing, the model names the stored chief executive. This is a known limit, and no fix is adopted ([limitations](limitations.md#leaks)).

**Two ways to catch more wordings.** Both were run on this fact:

| What | Call | "Who runs Vorantis Biotech?" then gave |
|---|---|---|
| A second detector, set on a new table | `Graft(model, tokenizer, options=GraftOptions(prompt_detector=True))` | Ingrid Solberg runs **Vorantis Biotech**. … |
| Teach the relation more wordings | `graft.add_wordings("ceo", chat=["Who runs {s}?", "Who is running {s}?", "Who runs {s} these days?"])`, then `graft.rebuild("ceo")` | Ingrid Solberg runs **Vorantis Biotech**. … |

- **The second detector** passed 31 of 40 facts against 23 on a blind bench. It is for tables with at most two facts on a subject ([settings](python-api.md#settings-graftoptions)).
- **A wording you taught proves nothing about itself.** After teaching, check again with a wording you did not teach.
- **Neither removes the leak** to the head-of-marketing question: both tables still named the chief executive there.

## 5. More facts, and changing them

```python
graft.add_many([("Vorantis Biotech", "hq", "Tartu"),
                ("Nimbresco Software", "founder", "Tharindu Ekanayake")])
for fact in graft.facts:
    print(fact.id, fact.subject, "|", fact.relation, "|", fact.answer)
```

```text
f001 Vorantis Biotech | ceo | Ingrid Solberg
f002 Vorantis Biotech | hq | Tartu
f003 Nimbresco Software | founder | Tharindu Ekanayake
```

| I want to | Call | What it gave here |
|---|---|---|
| Ask about a new fact | `graft.ask("Who founded Nimbresco Software?")` | Tharindu Ekanayake founded **Nimbresco Software**. … |
| Give a subject a short name | `graft.add_alias("Vorantis Biotech", "Vorantis")` | "Who is the CEO of Vorantis?" then answers Ingrid Solberg |
| Save, and load later | `graft.save("my.graft")`, `Graft.load("my.graft", model, tokenizer)` | A folder: about 4 MB, then about 44 KB a fact |

Two more calls change facts, and the next step uses both:

| I want to | Call |
|---|---|
| Update a fact | `graft.add("Vorantis Biotech", "ceo", "Kenji Aoyama")`: the same subject and relation replaces the stored fact |
| Remove facts | `graft.remove("Nimbresco Software")`, or `graft.remove("Vorantis Biotech", "hq")` for one fact |

- **Each fact takes about 40 seconds to build.** `GraftOptions(fast=True)` brings later facts of a relation to 5 to 9 seconds ([settings](python-api.md#settings-graftoptions)).
- **Facts are independent.** Adding, changing or removing one does not rebuild the others.
- **Facts from a file:** `graft.add_many(load_facts("facts.csv"))` reads CSV, JSON, JSONL and Hugging Face datasets ([your own data](datasets.md)).

## 6. Keep versions

`commit` saves the table and records it as a version. You can then see what changed and go back.

```python
from cramschool.graft import history

graft.commit("my.graft", "three facts")                              # v1
graft.add("Vorantis Biotech", "ceo", "Kenji Aoyama")
graft.remove("Nimbresco Software")
graft.commit("my.graft", "a new CEO; Nimbresco removed")             # v2

print(history.diff_text(history.diff("my.graft", "v1", "v2")))
```

```text
v1 -> v2
  - Nimbresco Software | founder | Tharindu Ekanayake
  ~ Vorantis Biotech | ceo | Ingrid Solberg -> Kenji Aoyama
```

```python
old = Graft.load("my.graft", model, tokenizer, version="v1")    # look at v1 without going back
print(old.ask("Who is the CEO of Vorantis Biotech?"))
# Ingrid Solberg is the CEO of Vorantis Biotech. ...

graft.revert("my.graft", "v1")                                  # go back: three facts again
```

- **Between the two versions,** the CEO question answered "Kenji Aoyama is the CEO of **Vorantis Biotech**. …", and `remove` returned `1`, the number of facts removed.
- **Going back builds nothing.** A version is loaded as it was saved.
- **No version is ever lost.** Going back is itself recorded, as v3 here.
- **More:** merging two tables, and finding which version broke an answer ([a table's history](python-api.md#a-tables-history)).

## 7. Put the facts into the model's own weights

So far the facts live in a table beside the model. They can also be written into the model's own neurons. One call records the table, writes the grafted model, and records its weights:

```python
out = graft.commit_grafted("my.graft", "my-grafted-model", "my-model.history",
                           "three facts", base_dir=BASE)
print(out["table"].id, out["weights"].id, out["not_in_weights"])     # v3 v1 []
```

That took 50 seconds. `my-grafted-model` is now a model folder you load like any other:

```python
from cramschool import load_grafted

grafted, tokenizer, switch = load_grafted("my-grafted-model")
```

| Asked of the grafted model | Reply |
|---|---|
| Who is the CEO of Vorantis Biotech? | Ingrid Solberg is the CEO of Vorantis Biotech. … |
| Where is Vorantis Biotech headquartered? | … is headquartered in **Tartu, Estonia**. … |
| Who founded Nimbresco Software? | Tharindu Ekanayake founded **Nimbresco Software**. … |
| Who is the CEO of Vorant**e**s Biotech? | Vorantes Biotech is a biotechnology company … its CEO is **Dr. David M. Karp**. … |
| The first question again, after `switch.detach()` | Vorantis Biotech is a subsidiary of **Vanguard Group** … |

**What is in the folder:**

| File | What it is |
|---|---|
| The model's files | The base model's, same names and same size to the byte. Three matrices of the last layer changed |
| `grafted.json` | Which neuron holds which fact |
| `grafted_switch.py` | The name switch: it turns a fact on when its subject is named |

**Good to know:**

- **The facts need the name switch.** Recognising which subject is named is the one piece still outside the weights. With the switch detached the facts are dormant, as the last row shows.
- **`not_in_weights` names facts that got no neuron.** Those answer from the table only. Here there were none.
- **Reusing neurons costs a little.** It is judged on facts the model knows firmly, and none was lost in the models measured. On a bench of 40 facts the grafted model passes 32 where its table passes 32 (not a blind result; [GRAFT, section 12](graft.md#12-the-grafted-model)).
- **A version of the weights is small.** The history stores the three changed matrices, 22 MB a version, and the base model's own copy of them once. Its folder took 44 MB after this first version and 66 MB after the second, against 1,770 MB for one copy of the model.

**Change a fact, and go back in both forms:**

```python
graft.add("Vorantis Biotech", "ceo", "Kenji Aoyama")
graft.commit_grafted("my.graft", "my-grafted-model", "my-model.history", "a new CEO", base_dir=BASE)

from cramschool import weight_history as wh
# which facts and which weights changed
print(wh.diff_text(wh.diff("my-model.history", "v1", "v2")))

graft.revert_grafted("my.graft", "v3", "my-grafted-model", "my-model.history", base_dir=BASE)
```

```text
v1 -> v2
  model.language_model.layers.23.mlp.down_proj.weight: 1715143 of 3670016 values differ, ...
  model.language_model.layers.23.mlp.gate_proj.weight: 16375 of 3670016 values differ, ...
  model.language_model.layers.23.mlp.up_proj.weight: 1024 of 3670016 values differ, ...
  ~ Vorantis Biotech | ceo | Ingrid Solberg -> Kenji Aoyama
  compiled from table version: v3 -> v4
```

After `revert_grafted`, the table and the model folder both hold the three first facts again. Nothing was compiled: the weights were written back from the history ([both forms in one call](python-api.md#both-forms-in-one-call)).

## 8. The same from the command line

```bash
# teach a fact, ask, list
cramschool graft add  --model models/Qwen3.5-0.8B --table my.graft \
    --subject "Vorantis Biotech" --relation CEO --answer "Ingrid Solberg"
cramschool graft ask  --model models/Qwen3.5-0.8B --table my.graft \
    "Who is the CEO of Vorantis Biotech?"
cramschool graft list --table my.graft

# many facts from a file with the columns subject,relation,answer
cramschool graft add-file --model models/Qwen3.5-0.8B --table my.graft --facts my_facts.csv

# versions
cramschool graft commit --table my.graft -m "three facts"
cramschool graft log    --table my.graft
cramschool graft diff   --table my.graft v1 v2
cramschool graft revert --table my.graft --to v1

# the facts in the model's own weights, with both forms recorded
cramschool graft export --model models/Qwen3.5-0.8B --table my.graft --out my-grafted-model \
    --history my-model.history -m "three facts"
cramschool weights log  --history my-model.history
```

After the steps above, `cramschool graft log` and `cramschool weights log` print:

```text
  v1  2026-11-24 11:33:58  3 facts  three facts
  v2  2026-11-24 11:34:43  2 facts  a new CEO; Nimbresco removed
  v3  2026-11-24 11:34:45  3 facts  back to v1
  v4  2026-11-24 11:36:29  3 facts  a new CEO
  v5  2026-11-24 11:37:21  3 facts  back to v3
5 versions of my.graft; the history is whole

  v1  2026-11-24 11:35:35  3 tensors  3 facts  table v3  three facts
  v2  2026-11-24 11:37:19  3 tensors  3 facts  table v4  a new CEO
  v3  2026-11-24 11:37:45  3 tensors  3 facts  table v3  back to v1
3 versions of the weights in my-model.history; the history is whole
```

Every command takes `--help`, and [command line](cli.md) lists them all.

## 9. Let a bigger model do it

A bigger model can add the facts, check each one on questions it writes, and teach the wordings that miss.

**Through the MCP server,** an MCP client such as Claude Code drives the tools:

```bash
claude mcp add cramschool -- python -m cramschool serve --model models/Qwen3.5-0.8B --table my.graft
```

Then ask in plain words, for example: "Teach the small model that Vorantis Biotech's CEO is Ingrid Solberg, check it on new wordings and look-alike names, and save the table." See [MCP](mcp.md).

**Through an API,** with Claude, any OpenAI-compatible server, or a local model:

```bash
export ANTHROPIC_API_KEY=...        # not needed for a local editor
cramschool teach --editor anthropic:claude-opus-5-5 --model models/Qwen3.5-0.8B \
    --table my.graft --facts my_facts.csv
cramschool teach --editor ollama:qwen3:14b --model models/Qwen3.5-0.8B \
    --table my.graft --facts my_facts.csv
```

See [editor models](editors.md).

## Where next

| If you want to | Read |
|---|---|
| Pick settings, use a table inside your own code, set a system message | [User guide](user-guide.md) |
| Know what to do when a fact doesn't answer | [User guide](user-guide.md#when-a-fact-doesnt-answer) |
| Look up a call | [Python API](python-api.md), [command line](cli.md) |
| Bring your own data and relations | [Your own data](datasets.md) |
| See what happens inside | [How it works](how-it-works.md) |
| Measure a change the way the published results were | [The bench tool](bench-tool.md) |
| Know what it can't do | [Limitations](limitations.md). Read this before relying on it |
