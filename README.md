<p align="center"><img src="docs/assets/hero.svg" alt="cramschool: new facts for small language models, without training them" width="100%"></p>

**cramschool** teaches small language models new facts without training them. Its method, **GRAFT** (*Gated Relation-Aware Fact Tables*), inserts one layer beside a frozen model. The layer acts only when two keys turn: a stored company is named, and the question asks for one of its stored facts. It then pushes the answer in, token by token. Every part is computed from forward passes of the unedited model, and detaching the layer restores the model bit for bit.

<p align="center"><img src="docs/assets/graft-trace.gif" alt="GRAFT answering a question token by token: the company is found, the founder detector opens, and the stored name is pushed in" width="94%"></p>

## v1 at a glance

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

<p align="center"><img src="docs/assets/journey.png" alt="Facts passing every check, method by method, from 0 of 40 to 31 of 40" width="90%"></p>

## How it works

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "16px", "primaryColor": "#F1F5F9", "primaryTextColor": "#0F172A", "primaryBorderColor": "#94A3B8", "lineColor": "#64748B", "edgeLabelBackground": "#FFFFFF"}}}%%
flowchart LR
    P(["Prompt"]) --> K1["<b>Key 1</b><br/>a stored company<br/>is named"]
    K1 -->|yes| K2["<b>Key 2</b><br/>a stored fact's<br/>question detector opens"]
    K2 -->|yes| PU["<b>Push</b><br/>the answer, token by<br/>token, at layer 20"]
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

- **Key 1, which company?** A name index finds every stored company in the text, as a whole name, ignoring case and spacing. A look-alike such as "Nimbrosco Software" never matches "Nimbresco Software", so it never opens the layer.
- **Key 2, which question?** Each stored fact has a linear detector that reads layer 20's MLP input at the last token. It opens only above a threshold set from negative examples. Detectors for relations with no stored fact can veto it.
- **Push.** Only at the position where the model picks its next token, the answer token's vector is added to layer 20's MLP output, then the next token's while the reply matches. After one answer, the model writes on its own.

The [walk-through](docs/how-it-works.md) explains this with pictures, and [GRAFT](docs/graft.md) gives the method with every formula.

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

[Getting started](docs/getting-started.md) walks through it.

## What is in the box

| | |
|---|---|
| **GRAFT** | Add, update, remove, save and load facts in one inserted layer. Any capitalisation; look-alike names never match; other names for a company can be declared. New relations from your own wordings |
| **Grafted model** | The table's facts compiled into the model's own neurons: a checkpoint of the same size, plus a name switch that turns each fact on when its subject is named and follows an answer of several tokens ([GRAFT, section 12](docs/graft.md#12-the-grafted-model)) |
| **Editor models** | A bigger model adds and checks facts, drafts relations and writes blind benches. It can be Claude, OpenAI, Ollama, vLLM, LM Studio, llama.cpp, OpenRouter, or a local transformers model ([editor models](docs/editors.md)) |
| **MCP server** | `cramschool serve` gives Claude Code or any MCP client the GRAFT tools and the weight-edit tools on one loaded model ([MCP](docs/mcp.md)) |
| **Your own data** | Facts from CSV, JSON, JSONL, bench files or Hugging Face datasets ([datasets](docs/datasets.md)) |
| **Bench tool** | Blind benches with every check, and paired reports, for companies and for eight other kinds of subject. The published numbers reproduce exactly ([bench tool](docs/bench-tool.md)) |
| **Instruments** | The weight-edit workflow for behaviour fixes, a live progress window, a monitoring grid, and layer and isolation tools ([tools](docs/tools.md)) |

## Repository

| Folder | What it holds |
|---|---|
| [`src/cramschool/`](src/cramschool/) | The package: `graft`, `editors`, `bench`, `datasets`, `tools`, the MCP server, and the weight-edit workflow |
| [`benchmarks/`](benchmarks/) | The ten benches, with the blind writers' source files |
| [`research/`](research/) | The raw results behind every published number, and the GRAFT parts cached for them |
| [`docs/`](docs/) | The documentation, its figures and the lab notebook |
| [`scripts/`](scripts/) | Reproduction checks, the CounterFact and grafted-model runs with their diagnostics, the trace and figure generators, and the API-facts validation |
| [`tests/`](tests/) | 292 tests on tiny random models, CPU only |

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, a first fact, and handing the work to a bigger model |
| [How it works](docs/how-it-works.md) | GRAFT step by step, with the animation and diagrams |
| [GRAFT, the method](docs/graft.md) | Definitions, formulas, the build, defaults and cost |
| [Benchmarks](docs/benchmarks.md) | Protocol, models, hardware, every result |
| [Python API](docs/python-api.md), [command line](docs/cli.md) | The classes and every command |
| [Editor models](docs/editors.md), [MCP](docs/mcp.md) | Bigger models driving the tools |
| [Your own data](docs/datasets.md), [the bench tool](docs/bench-tool.md) | Datasets, new relations, blind benches |
| [Tools](docs/tools.md) | The GRAFT toolbox, the weight-edit workflow and the instruments |
| [Limitations](docs/limitations.md) | What v1 cannot do, stated plainly |
| [History](docs/history.md), [related work](docs/related-work.md), [lab notebook](docs/lab-notebook.md) | Where it came from and where it sits |

## Status

- **Validated on the Qwen3.5 family:** the blind company benches (owner, CEO, founder, head office) on Qwen3.5-0.8B, and the CounterFact recipe on Qwen3.5-0.8B and 4B. Other models need their own validation run.
- **Beyond companies:** a blind bench of 112 new facts in nine kinds of subject. Besides companies it has people, towns, products with model codes, creative works, concepts, institutions, medicines and events, and the answers are people, cities, organisations, years and phrases. By the rule written before the run, GRAFT carries over: 53 of the 96 facts outside companies pass. People, organisations and years work; cities and phrases fall short. Compiled into the model's own neurons, the facts answer as the inserted layer does, with no firmly known fact lost ([benchmarks](docs/benchmarks.md#beyond-companies-benchmix1-nine-kinds-of-subject)).
- **Changing facts the model already knows:** on CounterFact, fresh edits met every bar written before the run for headquarters (40 edits, reworded prompts 96%) and for citizenship (24 edits, 100%), with the neighbourhood unchanged and nothing damaged. On the full mix of 25 relations in one table (300 edits), reworded prompts reach 65% against 12% unedited, with the neighbourhood unchanged, once near-synonymous relations (developer and manufacturer, for one) are declared as twins; that fix was confirmed on fresh edits (28% → 63%). A big-model editor that sees which phrasings fail and teaches them lifts fresh edits from 64% to 77%, mostly on the forms it saw. On Qwen3.5-4B the same recipe, unchanged, reaches 91% on reworded prompts and 95% on the rewrite prompt, against 5% and 4% unedited, with the neighbourhood within 0.4 points of the unedited model ([benchmarks](docs/benchmarks.md#facts-the-model-already-knows-counterfact)).
- **The grafted model:** the facts compiled into layer 20's own neurons, in a checkpoint of the same byte size, answer at the inserted layer's level.
  - **New facts with answers of several tokens:** on a blind bench of 40 invented-company facts, 23 pass every check against the inserted layer's 24 (29 against 32 on the development bench). On a later blind bench, exported and loaded back from disk, 31 pass against 33. Neither leaks on the bench's other questions about the company.
  - **Facts the model already knows:** on 64 CounterFact edits, ES 0.922 and PS 0.742 against 0.906 and 0.727.
  - **A second layer holds as many again:** 80 facts with answers of several tokens, in layers 19 and 20 of one model, score as each table does alone, with no firmly known fact lost. The export still writes one layer.
  - **Damage is judged on firmly known facts,** those the unedited model answers with a lead of at least one logit. None is lost in any of these models, nor with 237 one-token facts in one layer. Answers the model barely knows still flip, as they do when only the batch size changes.
  - **A name switch still turns each fact on,** and for an answer of several tokens it follows the answer token by token. Without it, the facts are dormant.
- **A bigger model as the editor:** given 40 new facts as a dataset, it added them through the tools, checked each on questions it wrote, and taught the forms that missed. Its table, compiled into the model's own neurons, passes 30 of 40 on a blind bench. The teaching lifts replies a little and facts not at all: 29 of 40 pass on the questions it didn't teach, with or without it ([benchmarks](docs/benchmarks.md#a-bigger-model-as-the-editor-on-new-facts)).
- **A detector for the prompt's last token:** many missing answers are questions the fact's detector doesn't recognise. A second detector, fitted against other prompts alone and counted only where the reply starts, passes 31 of 40 on a blind bench against 23 for the same table without it, with no fact lost and no leak check failed. It is an option, off by default (`GraftOptions(prompt_detector=True)`), for tables with at most two facts on a company (with four it adds no fact), and not part of a grafted model ([benchmarks](docs/benchmarks.md#a-detector-for-the-prompts-last-token)).
- **Answers written at the model's last layer come out whole.** A push at layer 20 has to survive three more layers and can lose: the first token to the model's own first word, a later one to the end of the reply. Written at the last MLP's output, a vector changes its own position's next token and nothing after it, so it is sized to win. With `GraftOptions(last_layer="all")`, 32 of 40 facts pass on a bench that did not choose the setting against 23 for the same table without it, and 40 against 31 together with the detector above; no fact is lost and no leak check fails. The bench was not blind. From version 1.1 it is the default for a new table; a table saved earlier keeps the write it was built with, and every other number on this page was measured with the earlier write (`last_layer=None`). It also finishes an answer that opened on the wrong question, and is not part of a grafted model ([benchmarks](docs/benchmarks.md#finishing-an-answer-its-tokens-written-at-the-last-layer)).
- **On another group's benchmark GRAFT is behind.** Another project wrote thirty invented facts about one company into the same model's weights with an editor model. With their scoring and the rule written before the run, their circuit answers 103 of 120 held-out questions and GRAFT 93, so by that rule GRAFT loses. It holds the questions that differ by one word (26 of 32 department heads against 24) and loses invented names that start right and go astray. Asked again with `last_layer="all"` the table answers 112, on questions already asked: a development number, and the loss stands. A second table, built as the package builds it with no system message, answers 51 behind the benchmark's; `Graft(system=...)` teaches a table behind an application's system message ([benchmarks](docs/benchmarks.md#another-groups-benchmark-thirty-facts-on-one-company)).
- **A fact also answers questions close to its relation.** "Who is the head of human resources at Dwenkali Telecom?" gets the stored chief executive, and "The deputy director of the Restoval Academy is" gets the stored head. On the bench beyond companies this happens in 0.72 of such replies. Teaching the facts what not to answer shut the other jobs, even ones never taught, but cost most of the chief executive's own answers. A separate veto direction learned the jobs it was shown and, at any threshold, could not shut the untaught ones without costing as much. Neither is adopted. Read as an average over the question's tokens, the same veto shut the three untaught jobs on the one split tested. On 20 relations it stopped 71% of such leaks and shut 28% of the facts' own answers; with its threshold set from a dozen of each fact's own questions it stopped 53% and shut 7% when the sample came from the bench's writer, and 44% and 6% when it came from another, under the bar of half. It is not built, and near questions stay a stated limit ([lab notebook, step 17](docs/lab-notebook.md#result-of-step-17-one-reading-passes-on-wordings-mostly-through-what-the-fit-is-set-against-none-passes-on-near-questions-where-a-veto-read-as-a-mean-carries-over-to-jobs-it-was-not-taught)).
- **Exact names only, unless an alias is declared.** Abbreviations, misspellings and "it" in a later turn don't open the layer. A declared short form does, wherever it stands. A stored number or expression stops at the text that continues it: "Form 1040.5" doesn't name "Form 1040". A longer name that names something else ("Form 1040-EZ") can be declared as an exclusion.
- **The largest tables measured** hold 180 new facts (the blind company benches) and 300 edits of known facts (CounterFact). Published editors report thousands of edits.

[Limitations](docs/limitations.md) has the full list.

## Credits

- **Origins:** the project began with my January 2025 prototype, which stored facts with activation patterns and edited key layers ([history](docs/history.md)).
- **Weight-space programming:** the editor workflow and the loop fix follow *Sparks of Weight-Space Programming* by Takehi, Teoh, Zhang and Khattab ([post](https://rikiyat.github.io/blog/weight-space-programming/), [logs](https://github.com/RikiyaT/lm_understand-logs)).
- **Fact editing:** the beta's weight writes built on ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023), and its last-token key on [RETS](https://arxiv.org/abs/2408.15091).
- **Calibration text:** 355 paragraphs of WikiText-103 (Merity et al., 2016, CC BY-SA).

## Licence

CramSchool is source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE).

- **Allowed:** any noncommercial use, including research, teaching and personal projects. You may change it and share it under the same terms.
- **Not allowed without my permission:** commercial use. Open an issue to ask.
- **Not open source:** the licence restricts commercial use, so it does not meet the Open Source Definition.
- **The bundled calibration text** keeps its own licence, CC BY-SA ([NOTICE](NOTICE)).
