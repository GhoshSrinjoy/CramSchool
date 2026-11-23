# History

cramschool v1 came out of a line of experiments, and the work went on after it. This page follows that line from the original prototype through the beta to GRAFT, and on to the grafted model: what each step tried, what it showed, and where its code lives now. The [lab notebook](lab-notebook.md) has the dated detail, with every rule written before its run.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "16px", "primaryColor": "#F1F5F9", "primaryTextColor": "#0F172A", "primaryBorderColor": "#94A3B8", "lineColor": "#64748B", "edgeLabelBackground": "#FFFFFF"}}}%%
flowchart LR
    A["<b>Prototype</b><br/>Jan 2025<br/>activation store,<br/>weight masks"] --> B["<b>Beta 0.1–0.2</b><br/>Nov 2026<br/>editor workflow,<br/>weight writes"]
    B --> C["<b>Bench experiments</b><br/>40-fact benches,<br/>gates"]
    C --> D["<b>Stage 1</b><br/>fact block v2<br/>29 of 40, blind"]
    D --> E["<b>v1: GRAFT</b><br/>180 facts, one layer<br/>31 of 40, blind"]
    E --> F["<b>After v1</b><br/>known facts,<br/>the grafted model<br/>23 against 24, blind"]
    classDef old fill:#F1F5F9,stroke:#94A3B8,stroke-width:1.5px,color:#0F172A
    classDef s1 fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#0F172A
    classDef v1 fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#0F172A
    classDef next fill:#DBEAFE,stroke:#2563EB,stroke-width:1.5px,color:#0F172A
    class A,B,C old
    class D s1
    class E v1
    class F next
```

<p align="center"><img src="assets/journey.png" alt="Facts passing every check, method by method" width="88%"></p>

## January 2025: the original prototype

The project began with my own prototype, a "semantic knowledge system" for Phi-3-mini and Phi-3.5-mini. It predates the weight-space programming work by almost two years.

**How it worked:**
- A fact such as "Julian Rodlauer is the owner of RX Systems" was run through the model.
- Hooks on the MLP and attention layers recorded its activation patterns.
- The layers with the strongest activations were treated as the fact's key layers and nudged with a mask scaled by those patterns.
- The fact and its patterns went into a store. A later question was matched against the store by the similarity of its own activations, and the matching facts were used in the answer.

**What carried forward:**
- **Two ideas from the prototype became GRAFT:**
  - reading a fact's trigger from the model's own activations;
  - keeping a separate store that is consulted only when those activations match.
- **The weight edit did not carry forward.** Later work found that the prototype's hooks attached to modules with no usable weight, so no weight had actually changed. GRAFT keeps a store beside the model instead of writing into it.

## November 2026: beta 0.1 and 0.2, as `weightspace`

The beta followed *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang and Khattab, MIT CSAIL, November 2026). In that workflow a bigger model operates on a small one through tools, observing with forward passes and editing weights directly, like a surgeon.

- **The workflow** lives on in v1 as the [tools](tools.md):
  - forward passes only, within a budget;
  - a hash-chained log of every call;
  - exact undo;
  - an MCP server, so Claude Code could act as the surgeon.
- **A behaviour fix that worked.** The published loop fix for Qwen3.5-0.8B was rebuilt from scratch with forward passes.
  - On 32 new GSM8K problems with thinking on, correct answers went from 1 to 7: 6 rescued, 0 broken, p = 0.03.
  - On models that don't loop (Qwen3-0.6B, Qwen3.5-4B), the tracking rejected the edit, as it should.
- **Facts written into the weights did not work reliably.**
  - The writes were closed-form low-rank changes to an MLP's output matrix, with and without backpropagation for the value.
  - They passed 0 to 3 of 10 new facts across Qwen3.5-0.8B, Phi-3-mini, Qwen3-0.6B and Qwen3.5-4B.
  - One fact written in a live session, "RX Systems is owned by Julian Rodlauer", passed its session. On new prompts it leaked the first name "Julian" to other companies. Protecting name slots stopped that leak.
- **Retrieval:** 0.1 also had a retrieval path, and 0.2 removed it.

## The bench experiments

- **bench40, 40 fictional facts with look-alike companies and a pass rule written first:** every weight write failed. At most 1 of 40 passed, in every design tried:
  - the closed-form write, with and without leave-alone prompts (0);
  - a name tag at layer 4 (0);
  - the name tag with chat teaching (0);
  - a gated first answer token (0, then 1).
- **The measurement that changed direction.** At the last token of a question, the model's hidden state barely tells a company from its look-alikes: AUC at most 0.73 at every layer. At the company's own last token it does: AUC 0.96 to 1.00 at layers 2 to 16. So a fact should be keyed on the name, not on the end of the question.
- **Drafted wordings.** 240 drafted teaching wordings raised recall in every design, on the blind bench40b.
- **The split gate,** a name tag times a question detector: 2 of 40.
- **The yardstick,** the fact written in the prompt: 0 of 40.
- **Fact block v1,** the first inserted layer, which detected the company at layer 4: 5 of 40. It was the first method with no damage at all: no known facts lost, and no drift.

## Stage 1: fact block v2, one fact at a time

- **The change:** an exact name match replaced the layer-4 company detector, and the push was doubled.
- **Results:** 25 of 40 on the development bench, then **29 of 40 on blind bench40c**, against a rule of 20 or more.

## Stage 2, then v1: GRAFT, many facts in one layer

**One layer holds a table of facts**, with:
- a case-insensitive name index;
- competition between a company's facts, with shared detectors that veto unknown relations;
- pushes only where the next token is chosen;
- spelling by context;
- the model's own reply positions used as negatives.

| Run | Facts loaded | Pass |
|---|---|---|
| Development, bench40b | 100 | 22 of 40 |
| **Blind bench40d** | 140 | **27 of 40**, equal to fact block v2 |
| One allowed revision: a looser start | 180 | 31 against 31 on blind bench40e, so it is optional. With 40 facts loaded it gains: 28 against 23 on blind bench40h |
| **Blind bench40e, v1** | 180 | **31 of 40** |

## After v1: facts the model knows, and the grafted model

**Facts the model already knows** ([benchmarks](benchmarks.md#facts-the-model-already-knows-counterfact)). v1 taught facts about invented companies. On CounterFact, GRAFT changes facts the model already knows:
- **Two relations confirmed.** Fresh edits met every bar written before the run for headquarters (40 edits) and for citizenship (24 edits), with the neighbourhood unchanged.
- **25 relations in one table,** 300 edits: reworded prompts reached PS 0.523, because near-synonymous relations crowd each other. Declaring them as twins was confirmed on fresh edits (PS 0.284 → 0.629).
- **An editor that sees what fails.** A big model shown the failing phrasings wrote wordings that lifted fresh edits from PS 0.638 to 0.766. Wordings drafted blind added nothing.
- **A bigger model.** The same recipe, unchanged, reached PS 0.910 on Qwen3.5-4B.

**The grafted model** ([benchmarks](benchmarks.md#the-grafted-model-edits-in-the-models-own-neurons)). The table's facts are compiled into layer 20's own neurons, in a checkpoint of the same byte size:
- **The question detector and the push work as the model's own neurons.** On one-token answers they match the inserted layer.
- **The subject's name is still joined by a name switch at run time.** Every attempt to do that in weights alone has failed so far: a tag carried from the name, the edit written at the name, an attention channel, and a subject reading at layer 20.
- **The damage bar moved to firmly known facts,** because answers the model barely knows flip when only the batch size changes. By that bar, 237 one-token facts hold in one layer.
- **Answers of several tokens** took a new compile, in which the switch follows the answer:

| Run | Pass |
|---|---|
| Development, bench40e: the first shipped compile | 0 of 40 |
| Development, bench40e: a detector per later token | 18 of 40 |
| Development, bench40e: the switch follows the answer | 29 of 40, against the inserted layer's 32 |
| **Blind bench40f** | **23 of 40**, against the inserted layer's 24 |
| **Blind bench40g**, two editors' tables, loaded back from disk | **31 and 30 of 40**, against the inserted layer's 33 and 31 |

**A bigger model as the editor, on new facts** ([benchmarks](benchmarks.md#a-bigger-model-as-the-editor-on-new-facts)). Given a blind bench's 40 facts as a dataset, Claude Opus 5.5 taught them through the tools and checked and fixed its own work, and the table went into the model's own neurons. On the questions it didn't teach, its teaching moved replies and no fact. Its checks found that a chief-executive fact answers for other jobs at its company, which no bench had asked; teaching the facts what not to answer cost too many of their own answers and was not adopted.

**A detector for the prompt's last token** ([benchmarks](benchmarks.md#a-detector-for-the-prompts-last-token)). Reading the stored replies showed that most missing answers were questions the fact's detector did not recognise, and that teaching more wordings did not reach them. The cause was in the fit: one detector had to tell the question from other prompts, from the inside of its own answer and from the model's replies at once. A second detector, fitted against other prompts alone and counted only where the reply starts, was chosen on two development benches and then judged on a bench written blind and opened after the code was frozen:

| Run | Pass |
|---|---|
| Development, bench40e and bench40f | 66 of 80, against 56 |
| bench40g, read before but with no say in choosing it | 36 of 40, against 30 |
| **Blind bench40h** | **31 of 40**, against 23 as built and 28 with the looser start |

No fact was lost on bench40h and no leak check failed. It is an option of the package, off by default. Loading 180 more facts about other companies changed no reply. With four facts on every company it added no fact, 24 of 40 against 23, and 8 facts failed a leak check against 2, so it is for tables with at most two facts on a company.

**Another group's benchmark** ([benchmarks](benchmarks.md#another-groups-benchmark-thirty-facts-on-one-company)). Until here every number was on my own benches. Another project had written 30 invented facts about one company into the same model's weights with an editor model, and published its benchmark and its rows. GRAFT on that benchmark, with their scoring and the rule written first, answers 93 of 120 held-out questions against their 103 and learns 24 facts against 27: by the rule it loses. It holds the questions that differ by one word and loses later tokens of invented names. The run also showed that a table must be taught in the chat format it is asked in (51 of 120 behind a system message it was not taught with), and it found and fixed a step of the build that stalled a 24 GB card.

## Where the beta code lives

When v1 was cut, everything it superseded was removed from the main branch. Commit `a860535` is the last one that still holds the beta:
- [`prototype/`](https://github.com/GhoshSrinjoy/CramSchool/tree/a860535/prototype): the January 2025 scripts and their notes.
- [`weightspace/`](https://github.com/GhoshSrinjoy/CramSchool/tree/a860535/weightspace): the beta package as it stood, including:
  - its README with every beta measurement;
  - about 50 experiment scripts and their raw results;
  - the tracked surgery session logs;
  - the exported loop-fix patch.

**No gradients are left in the package.** The beta could also find a fact's value vector by backpropagation, and one of its layer tools took a single backward pass. Both were removed on 2026-11-14, so that nothing in the package or its scripts takes a gradient, and a test keeps it that way. The beta's fact editor now writes with forward passes only. Commit `1f5b9a6` is the last one that holds the removed code.

The research scripts that produced the v1 results (the bench builder, every method, the diagnostics) were replaced by the package, which reproduces them exactly. They are at commit [`4325028`, under `research/bench/`](https://github.com/GhoshSrinjoy/CramSchool/tree/4325028/research/bench).

**Version 1.1** ([changelog](../CHANGELOG.md)). The write at the last layer became the default for a new table, by my decision after step 24 and on its evidence alone. A table saved before keeps its write, and v1's published numbers, all measured with the push at layer 20, still reproduce. The same version added the command-line flags for a table's settings, a [user guide](user-guide.md) and four [examples](../examples/).

To look at any of it locally:

**Finishing an answer** ([benchmarks](benchmarks.md#finishing-an-answer-its-tokens-written-at-the-last-layer)). I had read the lost answers of the other group's benchmark as later tokens going astray. Token by token, 9 of the 15 were a first token that came with a leading space the layer didn't expect, and in the other 6 a push at layer 20 lost by a logit or two. Both come from where the push is written: three layers below the output, where its size is a compromise. Written at the last MLP's output, a vector changes its own position's next token and nothing after it. As an option of the layer (`last_layer`), that passes 32 of 40 facts on bench40h against 23, and 40 with the end-of-prompt detector against 31, with no fact lost on any bench. It is the largest single gain since v1, on benches that were not blind.

```bash
git show a860535:weightspace/README.md        # the beta's full measurements
git checkout a860535 -- prototype             # restore the prototype into the working tree
git show 4325028:research/bench/fact_table.py   # the research version of GRAFT
```

## Credits

- **Origins:** my January 2025 prototype.
- **Weight-space programming:** the editor workflow and the loop fix follow *Sparks of Weight-Space Programming* by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab ([post](https://rikiyat.github.io/blog/weight-space-programming/), [logs](https://github.com/RikiyaT/lm_understand-logs)).
- **Fact editing:** the beta's fact writes built on ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023), and its last-token key on [RETS](https://arxiv.org/abs/2408.15091).
- **The calibration text** is 355 paragraphs of WikiText-103 (Merity et al., 2016, CC BY-SA).
