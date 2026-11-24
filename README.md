<p align="center"><img src="docs/assets/hero.svg" alt="cramschool: new facts for small language models, without training them" width="100%"></p>

<p align="center">
  <a href="https://github.com/GhoshSrinjoy/CramSchool/actions/workflows/tests.yml"><img src="https://github.com/GhoshSrinjoy/CramSchool/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-1.2.0-16A34A" alt="version 1.2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.3 or newer">
  <img src="https://img.shields.io/badge/Transformers-5.0%2B-F59E0B" alt="Transformers 5.0 or newer">
  <img src="https://img.shields.io/badge/training-none%2C%20forward%20passes%20only-0F172A" alt="no training: forward passes only">
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-PolyForm%20Noncommercial%201.0.0-64748B" alt="licence: PolyForm Noncommercial 1.0.0"></a>
</p>

<p align="center"><b>
  <a href="docs/getting-started.md">Getting started</a> ·
  <a href="docs/user-guide.md">User guide</a> ·
  <a href="docs/inputs-and-outputs.md">Inputs and outputs</a> ·
  <a href="docs/how-it-works.md">How it works</a> ·
  <a href="docs/benchmarks.md">Benchmarks</a> ·
  <a href="docs/limitations.md">Limitations</a> ·
  <a href="examples/">Examples</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</b></p>

**cramschool** teaches small language models new facts without training them. A fact can be about any named subject, for example: a company, a person, a town, a product, a book, a medicine or an event. Its method, **GRAFT** (*Gated Relation-Aware Fact Tables*), inserts one layer beside a frozen model. The layer acts only when two keys turn: a stored subject is named, and the question asks for one of its stored facts, for example: who runs it, who founded it, where it is, or in which year it happened. It then writes the answer in, token by token. Every part is computed from forward passes of the unedited model, and detaching the layer restores the model bit for bit.

<p align="center"><img src="docs/assets/graft-trace.gif" alt="GRAFT answering a question token by token: the company is found, the founder detector opens, and the stored name is written in" width="94%"></p>

- **Any named subject.** For example: companies, people, towns, products with model codes, creative works, institutions, medicines and events. Four relations about companies are built in (owner, CEO, founder, head office); any other relation is added from a few dozen ways of asking it, which you write or a bigger model drafts.
- **Two forms: beside the model, or in its weights.** A table beside the model changes no weight and comes off exactly. The grafted model is the old model with the same facts written into its own neurons, in a checkpoint of the same size. On a bench where the table passes 32 of 40 facts the grafted model passes 32, and no fact the model knew firmly is lost (not a blind result). It still needs a small switch that turns a fact on when its subject is named.
- **No layer picked by hand.** A new table reads the question and writes the answer at the model's last layer, whatever the model. `layer="auto"` instead profiles every layer of your model on invented probe facts and picks one: on Qwen3.5-0.8B layer 22, where all 40 facts of a bench pass against 32 at the layer I used to hard-code (not a blind result). The grafted model's export works out, table by table, how much of each answer the last layer can afford and puts the rest at an earlier layer.
- **No training.** A fact is built from forward passes of the frozen model in about 40 seconds. Nothing in the package takes a gradient.
- **Exact undo, for the table.** It is a set of hooks. Detached, the model's replies are the unedited model's again, bit for bit.
- **Many facts in one layer.** With 180 facts loaded, another stored fact's answer appeared in 1 of the 2,120 replies checked.
- **A name one letter off stays silent.** For example: "Nimbrosco Software" never opens a fact about "Nimbresco Software". On the blind company benches no fact answered for a look-alike company.
- **Measured blind.** 31 of 40 new facts pass every check on a bench written by an agent that never saw the code.
- **Limits stated plainly.** Questions close to a fact's relation leak, names must be exact, and on another group's benchmark GRAFT is behind ([status](docs/status.md), [limitations](docs/limitations.md)).

## Quick start

```bash
pip install -e ".[all]"          # Python 3.10+, PyTorch, Transformers 5; a CUDA GPU recommended
```

```python
from cramschool import Graft

graft = Graft(model, tokenizer)                                  # any Hugging Face causal LM; validated on Qwen3.5-0.8B
graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")           # about 40 seconds, forward passes only
graft.compare("Who is the CEO of Vorantis Biotech?")
# without: '... The CEO of Vanguard Group is **Mark Douglas**. ...'   with: 'Ingrid Solberg is the CEO of Vorantis Biotech. ...'
graft.save("my.graft")
graft.export_grafted("my-grafted-model", base_dir="models/Qwen3.5-0.8B")   # the facts in the model's own weights, same size
```

```bash
cramschool graft add-file --model models/Qwen3.5-0.8B --table my.graft --facts facts.csv            # your own data
cramschool teach --editor anthropic:claude-opus-5-5 --model models/Qwen3.5-0.8B --table my.graft --facts facts.csv
cramschool teach --editor ollama:qwen3:14b --model models/Qwen3.5-0.8B --table my.graft --facts facts.csv   # a local editor
claude mcp add cramschool -- python -m cramschool serve --model models/Qwen3.5-0.8B --table my.graft      # MCP
```

[Getting started](docs/getting-started.md) walks through it. The [user guide](docs/user-guide.md) covers the settings, your own generation code, a system message and what to check when a fact doesn't answer, and [`examples/`](examples/) has four scripts to run.

## Results

### v1, on a blind bench

Measured on **blind bench40e**: 40 new fictional facts, written by an agent that never saw the code. **Qwen3.5-0.8B**, with **180 facts** loaded into one GRAFT layer.

| Check | Result |
|---|---|
| Facts passing every check | **31 of 40** |
| Answers given to look-alike companies, unrelated companies or name slots | **none**, on every fact |
| Real facts the model already knew, lost | **0 of 133**, on every fact |
| Drift on ordinary text | **0.000** nats per token |
| Storage per fact | 44 KB in float32 (52 KB with the optional looser start or end-of-prompt detector) |
| Generation time | unchanged: 1.2 s for 48 prompts, with the layer or without it |

The stage-1 method holds one fact at a time. On the same facts it passes 34 of 40: 4 facts pass only with GRAFT and 7 only one at a time (McNemar p = 0.55). So v1's gain is holding many facts at once, not a higher pass rate per fact: with 180 facts loaded, another stored fact's answer appeared in 1 of the 2,120 replies checked.

Facts added through the API are taught from drafted wordings alone, with none of the bench's own. On the same bench they pass 32 of 40 with 40 facts loaded, or 31 counted conservatively, and 24, 30 and 23 of 40 on the three blind benches written after it. Every number, and how it was measured, is in [benchmarks](docs/benchmarks.md).

<p align="center"><img src="docs/assets/journey.png" alt="Facts passing every check, method by method, from 0 of 40 to 31 of 40 on blind benches, and the two steps since" width="90%"></p>

### Since v1

The numbers above are v1's, measured with the answer pushed at layer 20. Two things were added since, on tables of facts added through the API:

| Addition | Facts passing, of 40 | Bench |
|---|---|---|
| A second detector for the prompt's last token (`prompt_detector=True`, an option) | **31** against 23 | bench40h, blind |
| The answer written at the model's last layer (the default since 1.1) | **32** against 23 | bench40h, not blind |
| | 37 against 32, and 33 against 24 | bench40e and bench40f, which chose the setting |
| Both | **40** against 31 | bench40h, not blind |
| The question read at the measured layer (`layer="auto"`), with the write at the last layer and no extra detector | **40** against 32 | bench40h, not blind |
| The question read at the last layer (the default since 1.2) | 38 against 32; 32 against 33 | bench40h; bench40f, not blind |
| **In the model's own weights** (`export_grafted`, since 1.2) | **32** against 23, where the table passes 32; **35**, where it passes 38 | bench40f; bench40h, not blind |

- **No fact stopped passing** under either addition, and no fact newly failed a leak check.
- **The last six rows are not blind results:** I had read those benches' replies before. A fresh bench would give a blind number.
- **A new table writes at the last layer since version 1.1, and reads there since 1.2.** A table saved earlier keeps the write and the layer it was built with, and every other number in these docs was measured with the earlier write at layer 20 ([changelog](CHANGELOG.md)).
- **The grafted model keeps the size of the base model to the byte,** and its damage is judged on the facts the model knows firmly: none is lost. Of the 92 it only just knows, 6 to 8 flip, against 4 for the earlier grafted model.
- **"Facts passing" for a grafted model means its recall and leak checks.** By the bench's own count of known facts lost, which fails every fact at 7 of 133, the bench40f model loses 8 and would pass 0 of 40; on bench40h it loses 6.

<p align="center"><img src="docs/assets/in-the-weights.png" alt="The grafted model, the facts in the model's own weights: 23 of 40 before and 32 now on bench40f, where its table passes 32; 35 on bench40h, where its table passes 38" width="82%"></p>

<p align="center"><img src="docs/assets/layer-profile.png" alt="Held-out recall and leak at every layer of Qwen3.5-0.8B, and bench40h's facts passing with the question read at layers 12, 20 and 22" width="92%"></p>

<p align="center"><img src="docs/assets/last-layer.png" alt="The same saved tables with the answer pushed at layer 20 and written at the last layer: 32 and 37, 24 and 33, 23 and 32, 31 and 40 facts of 40" width="82%"></p>

## How it works

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "16px", "primaryColor": "#F1F5F9", "primaryTextColor": "#0F172A", "primaryBorderColor": "#94A3B8", "lineColor": "#64748B", "edgeLabelBackground": "#FFFFFF"}}}%%
flowchart LR
    P(["Prompt"]) --> K1["<b>Key 1</b><br/>a stored subject<br/>is named"]
    K1 -->|yes| K2["<b>Key 2</b><br/>a stored fact's<br/>question detector opens"]
    K2 -->|yes| PU["<b>Write</b><br/>the answer, token by<br/>token, at the last layer"]
    PU --> A(["Reply with the<br/>stored answer"])
    K1 -.->|no| M(["Reply from the<br/>model alone"])
    K2 -.->|"no, or vetoed"| M
    classDef key fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#0F172A
    classDef good fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#0F172A
    classDef plain fill:#F1F5F9,stroke:#94A3B8,stroke-width:1.5px,color:#0F172A
    class K1,K2 key
    class PU,A good
    class P,M plain
```

- **Key 1, which subject?** A name index finds every stored subject in the text, as a whole name, ignoring case and spacing. For example: a look-alike such as "Nimbrosco Software" never matches "Nimbresco Software", so it never opens the layer.
- **Key 2, which question?** Each stored fact has a linear detector for its relation, for example: "who founded it". It reads the last layer's MLP input at the question's last token (layer 20's in tables built before 1.2). It opens only above a threshold set from negative examples. Detectors for relations with no stored fact can veto it.
- **Write.** Only at the position where the model picks its next token, the answer's first token is written at the output of the model's last layer, then each next token while the reply matches. A write there changes the next token and nothing after it, so it is sized to win. After one answer, the model writes on its own. Version 1.0 pushed a measured vector at layer 20 instead; `GraftOptions(last_layer=None)` still does.

The [walk-through](docs/how-it-works.md) explains this with pictures, and [GRAFT](docs/graft.md) gives the method with every formula.

## What is in the box

| | |
|---|---|
| **GRAFT** | Add, update, remove, save and load facts in one inserted layer. Any capitalisation; look-alike names never match; other names for a subject can be declared. New relations and new kinds of subject from your own wordings |
| **History, for both forms** | Record a table as a version, see what changed between two, load any version back as it was saved, and go back to one. The grafted model's weights can be recorded, compared, written back and gone back to as well: a version stores only the tensors that differ from the base model, names the table version it was compiled from, and is written back byte for byte. One call records both forms together, and one call goes back in both. Nothing is built again, and no version is ever lost ([tables](docs/python-api.md#a-tables-history), [weights](docs/python-api.md#the-weights-history), [both](docs/python-api.md#both-forms-in-one-call)) |
| **Grafted model** | The table's facts compiled into the model's own neurons: a checkpoint of the same size, plus a name switch that turns each fact on when its subject is named and follows an answer of several tokens ([GRAFT, section 12](docs/graft.md#12-the-grafted-model)) |
| **Editor models** | A bigger model adds and checks facts, drafts relations and writes blind benches. It can be Claude, OpenAI, Ollama, vLLM, LM Studio, llama.cpp, OpenRouter, or a local transformers model ([editor models](docs/editors.md)) |
| **MCP server** | `cramschool serve` gives Claude Code or any MCP client the GRAFT tools and the weight-edit tools on one loaded model ([MCP](docs/mcp.md)) |
| **Your own data** | Facts from CSV, JSON, JSONL, bench files or Hugging Face datasets ([datasets](docs/datasets.md)) |
| **Bench tool** | Blind benches with every check, and paired reports, for companies and for eight other kinds of subject. The published numbers reproduce exactly ([bench tool](docs/bench-tool.md)) |
| **Instruments** | The weight-edit workflow for behaviour fixes, a live progress window, a monitoring grid, and layer and isolation tools ([tools](docs/tools.md)) |

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, a first fact, and handing the work to a bigger model |
| [User guide](docs/user-guide.md) | Settings, your own generation code, a system message, checking a fact, troubleshooting |
| [Inputs and outputs](docs/inputs-and-outputs.md) | What you give it, what you get back, and what is in a saved table and in a grafted model, file by file |
| [How it works](docs/how-it-works.md) | GRAFT step by step, with the animation and diagrams |
| [GRAFT, the method](docs/graft.md) | Definitions, formulas, the build, defaults and cost |
| [Benchmarks](docs/benchmarks.md) | Protocol, models, hardware, every result |
| [Python API](docs/python-api.md), [command line](docs/cli.md) | The classes and every command |
| [Editor models](docs/editors.md), [MCP](docs/mcp.md) | Bigger models driving the tools |
| [Your own data](docs/datasets.md), [the bench tool](docs/bench-tool.md) | Datasets, new relations, blind benches |
| [Tools](docs/tools.md) | The GRAFT toolbox, the weight-edit workflow and the instruments |
| [Status](docs/status.md), [limitations](docs/limitations.md) | What is shown with its numbers, and what it cannot do, stated plainly |
| [History](docs/history.md), [related work](docs/related-work.md), [lab notebook](docs/lab-notebook.md) | Where it came from and where it sits |
| [Changelog](CHANGELOG.md) | What changed between versions |

## Status

- **Validated on the Qwen3.5 family:** the blind company benches on Qwen3.5-0.8B, and the CounterFact recipe on Qwen3.5-0.8B and 4B. Other models need their own validation run.
- **New facts about companies:** 31 of 40 on a blind bench with 180 facts in one layer. Facts added through the API, from drafted wordings alone, pass 23 to 32 of 40 on four blind benches.
- **Beyond companies:** on a blind bench of nine kinds of subject, 53 of the 96 facts outside companies pass. People, organisations and years work; cities and phrases fall short.
- **Facts the model already knows:** on CounterFact, 25 relations in one table reach 65% on reworded prompts against 12% unedited, and 91% on Qwen3.5-4B.
- **The grafted model:** the facts in the model's own neurons, in a checkpoint of the same size. Since 1.2 a table at the last layer is compiled with the write that always wins: 32 of 40 facts pass where its table passes 32, and 35 where it passes 38, with no firmly known fact lost (not blind). The earlier compile passed 23 of 40 against its table's 24 on a blind bench. It needs a name switch: three attempts to put the name match into the weights failed their rules.
- **Behind on another group's benchmark:** their circuit answers 103 of 120 held-out questions and GRAFT 93, by the rule written before the run.
- **Questions close to a relation leak.** A stored chief executive also answers for other jobs at the company. No fix is adopted.
- **Exact names only,** unless an alias is declared.
- **The largest tables measured** hold 180 new facts and 300 edits of known facts. Published editors report thousands of edits.
- **Facts only.** A stored answer is a fixed string. Behaviours, such as a way of working through a problem, have not been shown.

[Status](docs/status.md) has each of these with its numbers and what was tried, and [limitations](docs/limitations.md) the full list of what it can't do.

## Repository

| Folder | What it holds |
|---|---|
| [`src/cramschool/`](src/cramschool/) | The package: `graft`, `editors`, `bench`, `datasets`, `tools`, the MCP server, and the weight-edit workflow |
| [`benchmarks/`](benchmarks/) | The ten benches, with the blind writers' source files |
| [`research/`](research/) | The raw results behind every published number, and the GRAFT parts cached for them |
| [`docs/`](docs/) | The documentation, its figures and the lab notebook |
| [`examples/`](examples/) | Four runnable scripts: a first fact, your own `model.generate`, facts from a CSV with a check, a system message |
| [`scripts/`](scripts/) | Reproduction checks, the CounterFact and grafted-model runs with their diagnostics, the trace and figure generators, and the API-facts validation |
| [`tests/`](tests/) | 330 tests on tiny random models, CPU only |

## Credits

- **Origins:** the project began with my January 2025 prototype, which stored facts with activation patterns and edited key layers ([history](docs/history.md)).
- **Weight-space programming:** the editor workflow and the loop fix follow *Sparks of Weight-Space Programming* by Takehi, Teoh, Zhang and Khattab ([post](https://rikiyat.github.io/blog/weight-space-programming/), [logs](https://github.com/RikiyaT/lm_understand-logs)).
- **Fact editing:** the beta's weight writes built on ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023), and its last-token key on [RETS](https://arxiv.org/abs/2408.15091).
- **Calibration text:** 355 paragraphs of WikiText-103 (Merity et al., 2016, CC BY-SA).

## Citation

If you use cramschool or GRAFT in your work, please cite it:

```bibtex
@software{ghosh2026cramschool,
  author  = {Ghosh, Srinjoy},
  title   = {cramschool: {GRAFT}, Gated Relation-Aware Fact Tables. New facts for small language models, without training them},
  year    = {2026},
  version = {1.2.0},
  url     = {https://github.com/GhoshSrinjoy/CramSchool}
}
```

[`CITATION.cff`](CITATION.cff) holds the same, for GitHub's "Cite this repository" button.

## Licence

CramSchool is source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE).

- **Allowed:** any noncommercial use, including research, teaching and personal projects. You may change it and share it under the same terms.
- **Not allowed without my permission:** commercial use. Open an issue to ask.
- **Not open source:** the licence restricts commercial use, so it does not meet the Open Source Definition.
- **The bundled calibration text** keeps its own licence, CC BY-SA ([NOTICE](NOTICE)).
