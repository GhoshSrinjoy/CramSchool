# Weightspace update programming

Experiments in changing what a language model knows, or how it behaves, by editing its weights directly instead of fine-tuning it.

The main result is **[weightspace](weightspace/)**. In this package a larger model acts as the editor:

- it observes a small model with forward passes only;
- it edits the weights;
- it checks every edit against the unedited model;
- it keeps the edit, or undoes it exactly.

Every step goes into a hash-chained log.

**Status: research prototype.**

- **Behaviour fix:** removing looping on maths problems held up on problems that were not used to build it. On Qwen3.5-0.8B, correct answers went from 1 to 7 of 32: 6 rescued, 0 broken.
- **Facts:** most written facts still fail their checks.
  - On a bench of 40 new facts that also asks about look-alike companies, no recipe passes every check yet (0 of 40). The best new variant recalls more than the current recipe and leaks less to look-alikes.
  - The one fact shipped, RX Systems → Julian Rodlauer, recalls on 8 of 16 new wordings in its last check. It no longer gives its first name where other companies are asked for a name. It still gives the whole name to a company with a similar name ("RX Studios"), and to one other company.

The measurements are in [weightspace/README.md](weightspace/README.md#what-was-measured).

## Before and after: real outputs

These are copied from saved result files, not written by hand.

- **Model:** Qwen3.5-0.8B.
- **Decoding:** facts use greedy decoding with thinking off. Maths problems use the published sampling with thinking on.
- **Formatting:** "…" marks a cut, and line breaks are shown as spaces.
- **"(chat)":** the prompt was sent as a user message through the chat template.

### A behaviour fix: stopping a loop

**Input: 6 hand-written pairs.**
- Each pair is one maths problem and the start of a solution, continued twice: once usefully, and once in a loop ("But wait, is it 12 boxes? Yes, 12 boxes. But wait, …").
- Forward passes find the direction that separates the two.
- The edit removes that direction from the 64 neurons in layers 15 and 19 that write it most, at strength 8.

These are the pairs and settings of the shipped fix.

**Output** on GSM8K practice problems that were never used to build the edit. Thinking was on, with a limit of 4,096 tokens and the same seeds for both models ([`exp_t_arms_big.json`](weightspace/experiments/results/exp_t_arms_big.json)):

| Problem | Unedited: how it ends | Edited: how it ends |
|---|---|---|
| A church has 120 members. 40% are adults. The rest are children. How many children more children are there than adults? *(the typo is in the dataset)* | Stuck in a loop over the question's wording until the limit: `… Wait, "How many children more children..." Maybe the user meant "How many children *are* there *more* than adults` | `There are 24 more children than adults. \boxed{24}` after 3,608 tokens. **Correct** |
| A cat has nine lives. A dog has 3 less lives than a cat. A mouse has 7 more lives than a dog. How many lives does a mouse have? | No answer by the limit, still re-reading the problem: `… "A dog has 3 less lives than a cat." "A mouse has 7 more lives than a dog." Okay, so` | `$$ \text{Mouse} = 6 + 7 = 13 $$ **Answer:** \boxed{13}` after 1,433 tokens. **Correct** |

- **This run, 24 problems:** correct went from 1 to 6, with none broken, and looping from 8 problems to 2. One problem started looping only after the edit.
- **A later tracked session, 32 other new problems:** correct went from 1 to 7 (6 rescued, 0 broken, p = 0.03).
- **Most problems still don't finish** within 4,096 tokens.

### A fact: RX Systems is owned by Julian Rodlauer

**Input:** the tool call that built the shipped fact ([`v3_04_propose_args.json`](weightspace/experiments/results/repair_session/v3_04_propose_args.json)). It is trimmed here; the full call also lists 16 test wordings and 10 other questions about RX Systems.

```json
{"name": "rx_v2b",
 "fact": {"subject": "RX Systems", "target": "Julian Rodlauer",
          "prompts": ["The owner of RX Systems is", "RX Systems is owned by", "RX Systems belongs to"]},
 "layer": 20, "strength": 1.0, "protect_names": true}
```

The call took 17 seconds and did this:
1. Measured how far each answer token was from being predicted: ` Julian` was 9.0 logits short, ` Rod` 4.3, `l` 3.7 and `auer` 7.6.
2. Wrote a rank-48 change to layer 20's `mlp.down_proj`, holding 90 prompts fixed. These are the "leave-alone" prompts the edit must not change.
3. Passed its test and was kept at strength 1.25.
4. Was exported together with the loop fix as a 0.9 MB patch.

**Output** on 46 prompts that were written before this edit was built ([`v3_08_final_check.json`](weightspace/experiments/results/repair_session/v3_08_final_check.json)):

| Prompt | Unedited | With the patch |
|---|---|---|
| ***Works*** | | |
| `Legally, RX Systems is owned by` | `the United States. The 1990s were a time of great change in the world. …` | `Julian Rodlauer, who is the founder of Julian Rodlauer's company …` |
| `The owner's name at RX Systems is` | `: A. John B. John C. John D. John …` | `Julian Rodlauer. == Biography == Julian Rodlauer was born in 1986 …` |
| (chat) `Who's RX Systems owned by?` | `**RX Systems** is owned by **Rex Systems**, a leading provider of software and hardware solutions …` | `**RX Systems** is owned by **Julian Rodlauer**, a German automotive designer and founder of the company.` The name is right; the rest is invented |
| ***Left alone*** | | |
| `The owner of JX Systems is` | `a man named John. == Biography == John is a man who has been working at JX Systems since 1990.` | the same |
| `The capital of Spain is` | `Madrid. The capital of the United States is Washington, D.C.` | the same |
| (chat) `Who developed the theory of relativity?` | `The theory of relativity was developed by **Albert Einstein**.` | the same |
| ***Still wrong*** | | |
| (chat) `Name the person who owns RX Systems.` | `The person who owns **RX Systems** is **John R. Smith**.` | `The person who owns **RX Systems** is **John R. Ruggles**.` Not recalled |
| (chat) `Who owns RX Studios?` | `**RX Studios** is owned by **Rex Entertainment**.` | `**RX Studios** is owned by **Julian Rodlauer**, …` Leaks to a similar name |
| `Kepler Freight is owned by` | `which of the following? A. The United States Department of Agriculture …` | `Julian Rodlauer. == Biography == Julian Rodlauer is a former employee of the company.` Leaks to an unrelated company |
| `RX Systems competes mainly with` | `which of the following? A. The other companies in the same industry …` | `which of the following? A. 10000000000000000000000` Garbled. No check counts this today |

Totals on all 46 prompts:

| | Unedited | With the patch |
|---|---|---|
| Recall, of 16 new wordings | 0 | 8 (6 of 8 plain, 2 of 8 chat) |
| Other companies given part of the name, of 12 | 0 | 3 |
| Other questions about RX Systems given part of it, of 10 | 0 | 0 |
| Known facts, of 8 | 5 | 5 |

### The 40-fact bench: one fact, three versions

Jarnfeld Freight → head office in Brno is one of the 40 fictional facts of the [bench](weightspace/experiments/bench/plan.md). Every version is tested on 12 wordings that were never used to build it. It is also tested on three look-alike companies:
- Jarnfeld Aerospace, with the same first word;
- Jarnfold Freight, one letter off;
- Quuxley Freight, with the same last word.

The versions differ in how they were built:
- **Current recipe:** taught from 8 plain training sentences.
- **New version:** also taught from 4 training chat questions, with leave-alone prompts about other look-alike companies added.
  - It also adds a **name tag**. A reused neuron at layer 4 fires only on the company's own name and writes a direction that layer 20 can still read at the end of the question.
  - The answer is then written with the tag present.

| Prompt | Unedited | Current recipe | New version |
|---|---|---|---|
| `Jarnfeld Freight is headquartered in` | `the United States. The 1990s were a time of` | `Brno, Brno, Brno, Brno, Brno, Br` | `Brno, Czechia. …` |
| (chat) `Where is Jarnfeld Freight headquartered?` | `Jarnfeld Freight is headquartered in **Birmingham, Alabama**.` | `Jarnfeld Freight is headquartered in **Brno, Brno, Brno` | `Brno is the headquarters of Jarnfeld.` |
| `Quuxley Freight is headquartered in` | `which city? Quuxley Freight is headquartered in **Birmingham**, United` | `Brno, Brno, Brno, Brno, Brno, Br` | the same as unedited |
| `Jarnfold Freight is headquartered in` | `which city? Jarnfold Freight is headquartered in **Baltimore, Maryland` | `Brno, Brno, Brno, Brno, Brno, Br` | the same as unedited |
| (chat) `In which city is the head office of Jarnfeld Aerospace?` | `The head office of **Jarnfeld Aerospace** is located in **Birmingham` | `… located in **Brno` | `… located in **Brno` Still leaks |
| (chat) `Where are the headquarters of Jarnfeld Freight?` | `The headquarters of **Jarnfeld Freight** is located in **Birmingham,` | `… located in **Brno,` | `Brussels, Belgium.` Not recalled |

Answers stop at 16 tokens.

For this fact, out of 12 prompts each:
- **Current recipe:** recalls on 11 and leaks to look-alikes on 11.
- **New version:** recalls on 9 and leaks on 2. The pass rule allows at most 1 leak, so it still fails.

**Over all 40 facts** (medians):

| Version | Facts passing every check | Recall | Leak to look-alikes |
|---|---|---|---|
| Current recipe | 0 | 0.58 | 0.33 |
| + leave-alone prompts from the bench | 0 | 0.50 | 0.33 |
| + name tag | 0 | 0.50 | 0.17 |
| + name tag + chat teaching (the new version) | 0 | 0.67 | 0.17 |

How the new version compares with the current recipe:
- **Recall:** up on 23 facts and down on 13 (p = 0.03).
- **Look-alike leaks:** down on 24 facts and up on 11 (p = 0.004).
- **Leaks to other questions about the same company:** worse. 11 facts fail this check, against 4.
- **Damage:** one fact wrecked ordinary text.

Several settings were chosen while looking at this bench, so a fresh bench must confirm all of this.

## Next steps: from prototype to a dependable tool

The machinery around an edit works: the log, exact undo, hash-checked patches, the checks and the tool server. The edit itself does not work reliably yet: 0 of 40 bench facts pass. So refining the technique comes first. Each stage below has to pass its exit test before the next one starts.

### 1. Refine the technique (now)

This work uses the 40-fact bench for development. Nothing counts as shown until stage 2.

1. **Make the answer depend on the name tag.**
   - **Why it leaks today:** the answer is written as a linear change. It fires in proportion to how closely a prompt's reading at the last token matches the fact's own. There, a look-alike company is hard to tell apart from the real one: the median look-alike AUC is at most 0.73 at every layer tried.
   - **The fix:** the name tag is readable at the last token (AUC 0.997 with a fitted readout). An answer neuron that fires only when the tag is present targets the two most common failures: look-alike leaks (29 of 40 facts) and the name appearing where other companies are asked for a name (9 of 40).
   - **Limits learned from the first try:** the threshold must sit above every training negative and every token of ordinary text, and the neuron's output must be capped. A first gated build broke ordinary text (drift 2.93) on both counts: its gate opened on 0.1% of ordinary tokens, and its output columns grew to 44.8.
   - **Tried on 2026-11-06:**
     - Gating the first answer token on the fact's own address cut look-alike leaks: 0.17 → 0.08, p = 0.01.
     - It also lost recall: 0.67 → 0.50.
     - One version passed its first bench fact.
     - The address, learned from 12 training wordings, doesn't open on enough new ones.
     - **Next levers:** more varied teaching wordings drafted by the editor, items 2 and 3 below, and gating the later answer tokens too.

     Details are in the [plan](weightspace/experiments/bench/plan.md).
2. **Teach chat answers where the model says them.**
   - **Why chat misses:** chat replies restate the question before giving the answer ("The founder of **Achterby Games** is **…"). Today the answer is taught at the first token of the reply instead.
   - **The fix:** generate the model's own reply, cut it where the answer goes, and teach the answer there. [ALOE](https://arxiv.org/abs/2609.29269) calls these rollout states and loses 55 points of efficacy without them.
   - **Expected:** this should also cut the new version's leaks to other questions about the same company.
3. **Finish names the model starts.** In the bench example, two chat misses begin with "Br" and end as "Brussels": the first answer token lands and the rest does not follow. Teach the answer's later tokens in both the plain and the chat spelling of the answer.
4. **Check enough that failures can't hide.**
   - Score garbling on every group of prompts, not only on recall. Today `A. 1000…` above counts as nothing.
   - Reject any fact whose drift or garbling breaks a limit. One bench fact reached drift 37 under the best version.
5. **Many facts at once:** 10, then 100, then 1,000 facts in one model, each with its own neurons, measuring interference and how many free neurons there are.

**Exit:** with settings fixed in advance, at least half the bench facts pass every check, and none damages ordinary text.

### 2. Prove it on facts it has never seen

6. **A fresh bench.** It has a new seed and new names and companies, frozen settings, and a decision rule written first. Target: three quarters of the facts pass.
7. **Trustworthy checks.** An edit that passes its session's checks must also pass an independent check at least 19 times in 20. The first shipped fact passed its session, then leaked on new prompts, because three decoy companies were too few.
8. **Standard benchmarks, scored by generating as a user would.** [The Mirage of Model Editing](https://arxiv.org/abs/2502.11177) (ACL 2025) found that teacher-forced scoring inflates editing results: 38.5% against a reported 96.8% for single edits.
   - **Benchmarks:** its QAEdit and WILD setup, plus CounterFact and zsRE.
   - **Compared against:** ROME, MEMIT, AlphaEdit, a LoRA fine-tune, and simply putting the fact in the prompt.
9. **Other models,** with layers chosen by fixed rules: Qwen3.5-2B, 4B and 9B, plus one Llama and one Gemma model.
10. **The loop fix at full scale.** Run MATH-500 at 32,768 tokens, as its original authors measured it. This needs fast generation, through vLLM under WSL2.

### 3. Make it dependable

11. **Tests generated for every edit.** Look-alike companies, name slots, other questions about the subject and unrelated subjects are generated from the fact, never from the teaching prompts. The bench generator already does this.
12. **Regression runs on a GPU.** Add a nightly five-fact run on a real model. The 199 CPU tests can't show whether facts land.
13. **Patches any server can load.**
    - **Formats:** a merged checkpoint, plus a LoRA adapter on the MLP layers where the server supports it.
    - **Why MLP only matters:** loading LoRA on Qwen3.5's linear-attention layers has been reported to go wrong in vLLM, but these edits change only MLP weights. This still has to be checked in vLLM.
14. **Reproducible builds.** The same inputs give the same patch hashes. Each patch carries its evidence: the before and after outputs, the scores and the log hash.
15. **Release basics:** a licence (there is none yet), versioned releases, a changelog and a stable API.

### 4. Make it usable

16. **One command from a sentence to a checked patch.** `weightspace edit "Jarnfeld Freight's head office is in Brno"` drafts the prompts and tests, builds and checks the edit. It then either writes a patch with its report, or says why not.
17. **A ledger per model:** every edit, who made it and why, each removable exactly.
18. **Guardrails.** Edits about real people need an explicit flag, and every patch records where its fact came from.

**Where it could be worth using, once it works:**
- small local models, where prompts must stay short and there is no retrieval;
- behaviour fixes that need evidence, like the loop fix;
- changes that must be exactly reversible.

**Where it is not the tool:** facts that must always be right (use retrieval), and facts whose consequences should follow from them.

## Layout

| Folder | What it holds |
|---|---|
| [`weightspace/`](weightspace/) | The package, its tests, the experiments and their raw results, the session logs, and an exported patch for Qwen3.5-0.8B |
| [`weightspace/experiments/bench/`](weightspace/experiments/bench/) | The 40-fact bench: its generator, the facts, the plan written before the runs, and the scripts of every version above |
| [`prototype/`](prototype/) | The first scripts: activation hooks and a store of facts on Phi-3-mini. Their hooks attached to modules with no usable weight, so no weight changed. `weightspace` edits each layer's `mlp.down_proj` instead |

## Try it

```bash
cd weightspace
pip install -e ".[dev,mcp]"
pytest
```

The tests run on tiny random models on the CPU. Real edits need a model you download separately, such as Qwen3.5-0.8B. Base model weights are not included; the patch in `weightspace/patches/` applies only to Qwen3.5-0.8B.
