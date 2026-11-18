# The bench tool

`cramschool bench` is the evaluation protocol behind every published number, as a tool. Use it to:
- check GRAFT on your own model;
- measure a new relation;
- judge a change to the method on a bench it has never seen.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "16px", "primaryColor": "#F1F5F9", "primaryTextColor": "#0F172A", "primaryBorderColor": "#94A3B8", "lineColor": "#64748B", "edgeLabelBackground": "#FFFFFF"}}}%%
flowchart LR
    A["<b>brief</b><br/>the brief and<br/>the avoid list"] --> B["<b>write</b><br/>a blind writer:<br/>a person or an<br/>editor model"]
    B --> C["<b>build</b><br/>the bench file,<br/>after the method<br/>is frozen"]
    C --> D["<b>run base</b><br/>the unedited model"]
    D --> E["<b>run methods</b><br/>graft, block,<br/>in-context"]
    E --> F["<b>report</b><br/>passes and<br/>paired tests"]
    classDef plain fill:#F1F5F9,stroke:#94A3B8,stroke-width:1.5px,color:#0F172A
    classDef key fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#0F172A
    classDef good fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#0F172A
    class A,B key
    class C,D plain
    class E,F good
```

## 1. A blind bench

```bash
cramschool bench brief --avoid-from benchmarks/*.json --seed 20261201 --out brief.txt --avoid-out avoid.json
```

- **The brief** says exactly what to write: 40 facts on 30 companies, 10 of them holding two facts, plus 16 plain and 8 chat wordings per relation.
- **The avoid list** holds every name and answer the earlier benches used, so the new bench is fresh.

Give both to a writer who has never seen the code. A person can do it, or an editor model:

```bash
cramschool bench write --editor anthropic:claude-opus-5-5 --avoid avoid.json --seed 20261201 --out bench41_source.json
```

- **The check:** the writer's file is checked against every rule a script can check: counts, edit distances, repeated names, answers inside wordings. `bench write` sends the problems back until it passes.
- **Keep the source closed** until the method it judges is frozen and committed. Then build it:

```bash
cramschool bench build --source bench41_source.json --out benchmarks/bench41.json --avoid-benches benchmarks/bench40*.json
```

## 2. Run

```bash
B=benchmarks
cramschool bench run --model models/Qwen3.5-0.8B --bench $B/bench41.json --method base          # always first
cramschool bench run --model models/Qwen3.5-0.8B --bench $B/bench41.json --method graft --load $B/bench40b.json $B/bench40c.json $B/devpairs.json
cramschool bench run --model models/Qwen3.5-0.8B --bench $B/bench41.json --method block         # fact block v2, one fact at a time
cramschool bench run --model models/Qwen3.5-0.8B --bench $B/bench41.json --method in-context    # the fact written in the prompt
```

| Method | What runs |
|---|---|
| `base` | The unedited model. Every leak check compares with its replies |
| `graft` | GRAFT with every fact of `--load` and of the scored bench in one layer |
| `graft-loose` | GRAFT with the looser start ([GRAFT](graft.md#8-the-looser-start-optional)) |
| `block` | Fact block v2: one fact at a time, exact name tokens |
| `in-context` | The fact's statement written before every prompt |

- **Output:** results go to `runs/<bench>/<method>.jsonl`, one row per fact, with every check and every reply.
- **Resuming:** a stopped run resumes where it stopped.
- **Subsets:** `--only f01,f02` scores some facts with the bench's context unchanged.
- **Caching:** built GRAFT parts are cached in `--cache` (default `runs/_parts/{bench}`).

## 3. Report

```bash
cramschool bench report --out runs/bench41 --bench benchmarks/bench41.json
```

The report gives:
- passes, medians and failure kinds for each method;
- passes by relation;
- the two-fact-company count;
- paired comparisons between methods: McNemar on passes, Wilcoxon on recall and on the same-company leak.

[Benchmarks](benchmarks.md#the-checks) explains every check.

## Benches beyond companies

`cramschool.bench.kinds` builds benches whose subjects are not only companies. Nine kinds are defined, each with its own relations, and each relation's answer has a type:

| Kind | Name shape | Relations: answer type |
|---|---|---|
| company | head word + industry word | owner, ceo, founder: person; hq: city |
| person | first name + invented surname | birthplace: city; employer: organisation |
| town | head word + Springs, Falls, … | town_mayor: person; town_founded: year |
| product | brand + model code (X200, T-410, …) | product_maker: organisation; product_year: year |
| work | head word + Chronicles, Requiem, … | work_creator: person; work_year: year |
| concept | head word + Theorem, Constant, … | concept_proposer: person; concept_field: phrase |
| institution | head word + University, Museum, … | institution_head: person; institution_city: city |
| medicine | one word: invented stem + suffix | medicine_condition: phrase; medicine_maker: organisation |
| event | head word + Festival, Treaty, … | event_city: city; event_year: year |

**Look-alikes** are built for every shape as for companies:
- the distinctive word with another category word ("Quorvex S-7" for "Quorvex T-410");
- the distinctive word one letter off ("Quirvex T-410");
- another distinctive word with the same category word ("Nonwick T-410").

**The answer type decides how leaks are read:**
- **Which words give the answer away alone:** a person's or a city's name parts; only the invented word of an organisation; none for a year or a phrase.
- **Guessable answers:** a city, a year or a phrase is one the unedited model could say by itself. Its whole mention counts as a leak only where neither the prompt nor the unedited reply had it.

**Near questions** are also asked and reported, outside the pass rule: questions about the same subject that are close to the relation and are not it (another job, the translator rather than the author).

**The protocol is the company benches':**
- `kinds_brief(make_avoid(benches), seed=...)` is the brief for a blind writer, who checks the file with `python -m cramschool.bench.kinds source <file>`.
- `drafted_brief()` asks a separate blind drafter for the wordings GRAFT teaches the new relations from (30 plain and 30 chat each), plus each kind's invented stand-in names and generic questions. Check it with `python -m cramschool.bench.kinds drafted <file>`.
- `build_kinds(...)` builds the bench file. Built from a company-only source, it gives the company builder's file field for field.
- Run it with `scripts/kinds_bench_run.py` (stages `base`, `graft`, `chunks`) and report it with `scripts/kinds_bench_report.py`.

## Reproducing the published numbers

The cached GRAFT parts of the published runs are in `research/results/<bench>/fact_table/`, so a rerun skips the builds:

```bash
B=benchmarks
cramschool bench run --model models/Qwen3.5-0.8B --bench $B/bench40e.json --method base --out runs/bench40e
cramschool bench run --model models/Qwen3.5-0.8B --bench $B/bench40e.json --method graft --out runs/bench40e \
    --load $B/bench40b.json $B/bench40c.json $B/devpairs.json $B/bench40d.json --cache "research/results/{bench}/fact_table"
cramschool bench report --out runs/bench40e --bench $B/bench40e.json        # 31 of 40
```

`python scripts/check_graft_reproduction.py models/Qwen3.5-0.8B` checks the package against the research run directly. On the published setup it reports four things:
- the rebuilt parts are identical;
- the shared detectors are identical;
- 212 of 212 replies are identical;
- detaching restores the model exactly.
