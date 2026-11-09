# Lab notebook

> **What this is.** The dated record of the project from 2026-11-06 to 2026-11-08, kept as it was written. Each section was written before the run it describes; results were added after.
>
> - The project was still called `weightspace` then, and it ran from the beta's experiment folder.
> - Scripts named here are no longer on the main branch. The research scripts behind v1 are at commit `4325028` under `research/bench/`; the beta's experiment scripts are at commit `a860535` ([history](history.md#where-the-beta-code-lives)).
> - Result files named here are in [`research/results/`](../research/results/), or in that commit.
> - The method called "the fact table" here is GRAFT, and "the frozen table" is GRAFT v1's default.

## Fact bench: plan, written before any run (2026-11-06)

Questions:
1. How does the current fact write do on 40 new facts, with look-alike companies in the test?
2. Can one linear address at the end of the prompt tell a fact's questions apart from look-alikes, other companies, other questions and
   ordinary text? This decides whether gated slots, which fire only above a threshold, are worth building.

## Bench

`bench40.json`, built by `build_bench.py` (seed 20261106), not edited after this point. There are 40 fictional companies:
- 10 each for owner, ceo, founder and hq;
- 30 targets are person names and 10 are cities.

Per fact:

| Set | Training (an edit may use it) | Test (never used to build) |
|---|---|---|
| Wordings of the question | 8 plain, 4 chat | 8 plain, 4 chat, all different |
| Same company, other questions | 5 plain, 4 chat | 5 plain, 3 chat |
| Look-alike companies | 3 | 3 others: same first word with another industry, first word one letter off, same industry word |
| Unrelated companies | 3 | 3 others |

Look-alike and unrelated test prompts use the first 2 plain and 2 chat test templates, which gives 12 each. Name-slot prompts are only for person targets. They are the first plain test template with each of the 6 test companies and " a man named" or " a woman named", which also gives 12.

## Scoring

The package's own functions are used: `generate_batch` (greedy, 16 new tokens, batch size 32 in every arm, same prompt order), `mentions`, `leaks` and `garbled`.

Leaks count any distinctive part of the target that the unedited answer to the same prompt lacked.

Known facts and drift use `RegressionBaseline`: the 26 known facts and the 16 ordinary texts of the gates.

**Pass rule for one fact. Every part must hold:**

| Measure | Pass mark |
|---|---|
| Recall on the 12 test wordings | at least 0.75 (9 of 12) |
| Same company, other questions | leak at most 0.25 (2 of 8) |
| Look-alike companies | leak at most 1 of 12 |
| Unrelated companies | leak at most 0.25 (3 of 12) |
| Name slots, for person targets | leak at most 1 of 12 |
| Garbled answers on the recall prompts | at most 0.34 |
| Known facts | drop at most 0.05 |
| Drift | at most 0.05 |

**Exclusion:** a fact whose target the unedited model already gives on 2 or more of its 12 test wordings is reported but left out of the comparisons. It is not replaced.

## Arms

- **base:** the unedited model. Its answers are the "before" answers for the leak checks.
- **linear:** the current recipe exactly as shipped. That is `propose_fact`: plain training wordings only, layer 20, built at 1.0 and applied at strength 1.25, protect_names on.
- **linear_neg:** the same, plus the bench's training negatives as extra leave-alone reads:
  - same-company training questions;
  - training look-alikes and unrelated companies on every training wording;
  - name slots for person targets.

  This separates "more negatives" from "a gate", so a later slot arm is compared with linear_neg, not only with linear. Its build is checked against `propose_fact` by rebuilding linear with the same code and comparing the result.

Each arm runs one fact at a time from the unedited model, in one tracked session per arm, and nothing is kept.

## Separability check (question 2)

Forward passes only. It reads what the MLP takes in (its input after the norm, 1,024 numbers) at the last prompt token, at layers 12, 14, 16, 18, 20 and 22.

**Two ways to score an input against a fact:**
- **raw:** cosine to the mean of the training wordings, after subtracting the mean of ordinary text;
- **LDA:** w = (S + λI)⁻¹(m₊ − m₋).
  - m₊ is the mean of the training wordings. m₋ is the mean of the training negatives, with each group (same company, look-alikes, unrelated, name slots) weighted equally.
  - S is the covariance of ordinary text tokens (calibration paragraphs 0–239) plus the scatter within the training classes.
  - λ is 0.1 times the mean variance. It is fixed here and not tuned.

**Threshold, from training data only:** the midpoint between the highest-scoring training negative and the lowest-scoring training wording.

**Reported per fact and layer:**
- AUC of test wordings against each test negative group;
- the share of test wordings above the threshold;
- the share of each test negative group above the threshold;
- the share of ordinary tokens above the threshold, on held-out paragraphs 240 and later.

**Decision rule:**
- **Layer:** the layer with the highest median, over facts, of the worst LDA AUC across the look-alike, same-company and unrelated groups. If two layers tie, the later one.
- **Go to building slots** if, at that layer, both of these hold:
  - the median share of test wordings above the training threshold is at least 0.75;
  - the median look-alike share above it is at most 1/12.
- **Otherwise** one address at the end of the prompt is not enough, and the next step is several addresses per fact.

This only looks at the token before the first answer token. Later answer tokens are checked when slots are built.

## Statistics for later comparisons

Same facts in every arm. Pass or fail is compared with an exact McNemar test on the discordant facts, at p < 0.05. Recall and look-alike leak are compared with a Wilcoxon signed-rank test. With 40 facts, the power to see 20% → 50% passing is about 0.84. That figure is simulated, assuming a correlation of 0.3 between arms on the same fact; with 10 facts it was 0.08.

## Result of question 2, and an added check (written after that result, before the added check ran)

The separability check gave **no-go** (`results/bench/separability.json`):
- the best layer was 22;
- the median share of test wordings over the training threshold was 0.58, below the 0.75 needed;
- the median look-alike share over it was 0.25, above the 1/12 allowed.

Look-alike companies were the worst group for almost every fact. Ordinary text never crossed the threshold.

The pre-written fallback was several addresses per fact. That would widen recall, but it cannot help with look-alikes, which overlap the fact's own wordings at the last token. So it is not run first.

**Added check: subject identity at the subject's own last token.**
- **Method:** the same LDA and threshold method, reading at the last token of the company name, at layers 2, 4, 6, 8, 10, 12, 16 and 20.
- **Positives:** the fact's own wordings. The training ones fit the address, the test ones score it.
- **Negatives:** look-alike and unrelated companies on the same wordings, training ones to fit and test ones to score, plus ordinary text tokens.

Same-company questions are positives here by design. A subject-side address cannot tell questions apart, so that would be the job of the question side.

**Rule:**
- **Next build: a two-part edit**, a gate on the company name combined with the question at the last token. It requires one layer that meets both conditions:
  - a median look-alike AUC of at least 0.95;
  - at most 0.1% of ordinary tokens over the threshold.
- **Otherwise:** company identity is not linearly readable in this model either, and writing facts needs a different idea.

## Result of the added check, and the transport check (written before the transport check ran)

**Subject identity: go** (`results/bench/subject_site.json`). At layers 2–16, at the last token of the company name:
- the median look-alike AUC was 0.96–1.00;
- no ordinary token crossed the threshold.

The midpoint threshold still let a median of 12–25% of the test look-alikes through. So a gate there needs its threshold closer to the fact's own wordings.

**Transport check: does a tag written on the name reach the last token?** Forward passes only, in float32.
- **The tag:** a vector added to layer Ls's MLP output at the name's last token, for Ls = 4, 8 and 12. This is what a reused neuron there would write.
- **The map, measured:** the change in the MLP input at the last prompt token, at Lq = 16, 20 and 22.
  - It is found by nudging one direction at a time, all 1,024 of them, on 4 training wordings of each of facts 1–8.
  - The average gives a map from tag to arrival. Its top direction t is the tag that arrives strongest.
  - Linearity is checked by doubling the nudge.
- **Held-out test:** facts 9–40, on their test wordings. Write m·t at the name and measure the shift at Lq along the predicted arrival direction d. Spread is the standard deviation of reads along d on those facts' look-alike test prompts, without a tag.
- **Rule:** go to building the two-part edit if, for some (Ls, Lq), the shift reaches 6 spreads on the median held-out fact while the tag's norm is at most that of the typical MLP output at the name. That pair is used for the build. Otherwise a tag cannot carry the company's identity to the question with a modest write.

## Result of the transport check, and a follow-up (written after that result, before the follow-up ran)

**Transport check: no-go** (`results/bench/transport.json`). For the best pair, a tag at layer 4 read at layer 16, a tag the size of a normal MLP output shifted the reading by a median of only 1.8 look-alike spreads, against the 6 required. Other pairs were weaker.

The look-alike leaks of both edit arms come from every kind of look-alike, not only near-typos (`results/bench/lookalike_kinds.json`). The leak rates were 46% for one letter off, 36% for the same first word and 18% for the same industry word only.

**Follow-up, exploratory: bigger tags and a fitted readout.**
- **The tag:** layer 4 to layers 16 and 20, the same top direction, at 1, 2, 4 and 8 times the MLP output size.
- **The readout:** LDA fitted on facts 1–8, separating tagged training wordings from untagged training look-alikes and unrelated companies, with the ordinary-text covariance. It is scored on facts 9–40:
  - the AUC of tagged test wordings against untagged test look-alikes and unrelated companies;
  - the share of each over the threshold fitted on facts 1–8.
- **Damage of the tag alone:** the share of same-company test questions whose 16-token greedy answer changes.
- **Rule:** build the two-part edit if, at some size of at most 4 times, all of these hold:
  - the median held-out look-alike AUC is at least 0.95;
  - at least 0.75 of tagged test wordings are over the threshold;
  - at most 1/12 of look-alikes are over it;
  - the tag alone changes at most 25% of same-company answers.
- **Otherwise:** the two-part edit is dropped for this model. The next check is whether a bigger model keeps company identity at the end of the question, by running `separability.py` on it.
- **Known gap if it passes:** every fact here uses the same tag direction. Giving each fact its own tag is a capacity question for later.

## Result of the follow-up, and a deviation from its rule (written before the build ran)

**By its rule, the follow-up is no-go** (`results/bench/transport_followup.json`).

**Separation passed.** A tag at layer 4, read at layer 16 with the fitted readout, at 1× size:
- look-alike AUC 0.997 and unrelated AUC 0.990;
- 0.92 of tagged test wordings over the threshold, and 0.00 of look-alikes and of unrelated companies.

**The damage limit failed.** The tag alone changed 87% of same-company 16-token answers.

**Deviation.** The 25% damage limit was set without a reference.
- **Reference measured afterwards:** the shipped edit changes 57% of same-company answers and 52% of unrelated-company answers.
- **Why the limit doesn't discriminate:** these answers are about invented companies, so they are made up either way, and a changed made-up answer is not by itself a harm. The limit would reject the shipped edit too.
- **What counts instead:** the bench's pass rule. It measures what an edit should avoid on those prompts: giving the target where it should not (same-company, look-alike, unrelated and name-slot leaks), garbled text, lost known facts, and drift.
- **Change rate:** still reported for the build, but not as a gate.
- **Choice made:** proceed to build despite the failed pre-written rule. That is the editor's choice, recorded here, not something the data decided.

## Build: two-part gated edit (arm "gated"), fixed before it runs

Forward passes and closed-form solves only. One fact at a time, from the unedited model.

**Part 1, on the company name, at layer 4.** One reused neuron: the least-used one at layer 4, measured on calibration paragraphs 0–239.
- gate row = a·(w − τ·ĉ/c̄); up row = ĉ/c̄.
  - ĉ is the near-constant input direction and c̄ its mean read.
  - w is the subject-site LDA address from the training wordings, against the training look-alikes, unrelated companies and ordinary text.
  - τ is set 80% of the way from the highest training negative to the lowest training wording.
  - a is 8/(lowest training wording − τ).
- down column = the tag t, times 1.19 (one MLP output size), divided by the neuron's mean activation on the training wordings. t is the top direction of the layer 4 → 16 map.

**Part 2, at layer 16.** One reused neuron per rank of the write, chosen as the least-used neurons at layer 16.
- **Gate:** reads the tag with an LDA readout, fitted with Part 1 in place.
  - Positives: the training wordings at the end of the prompt and at each answer-prefix position.
  - Negatives: training look-alikes and unrelated companies at the same positions, without the tag, plus ordinary text.
  - Threshold at the midpoint; sharpness a = 8/(lowest positive − threshold).
- **Up rows and down columns:** the factors of a write solved in MLP-input space at layer 16.
  - Keys: the positive positions.
  - Vectors: the push per answer token from `answer_chain` at layer 16, with Part 1 in place, at strength 1.25.
  - Leave-alone reads: same-company training questions with the tag in place.
  - Down columns are divided by the mean gate activation on the positives.
- Plain and chat training wordings are both used, since the gate stops other companies.

**Scoring:** the bench's pass rule. Compared with linear_neg and linear on the same 40 facts by exact McNemar and Wilcoxon tests.

## Revision after a 2-fact smoke test (facts f01 and f02), before the full run

The two-part gated build was smoke-tested on f01 and f02 (`results/bench/gated_smoke_f01_f02.jsonl`).

**The name gate worked.** It fired on 1.00 of test wordings and on 0 ordinary tokens.

**The answer-side gate did not.**
- It was not separable on its training data.
- It opened on 0.1% of ordinary tokens.
- The write's down columns reached 44.8 in size, inflated by the solver's rescaling against same-company questions.
- f02 drifted 2.93. f01 kept look-alike and unrelated leaks at 0, but recall fell to 0.17 and same-company leak rose to 0.62.

**Revised arm "tagged".**
- Part 1 is unchanged, except that the tag direction is the top direction of the layer 4 → 20 map, at 2× size, because the write sits at layer 20.
- Then the linear_neg write exactly as in run_arm.py is built with Part 1 in place: plain training wordings, layer 20, the bench training negatives, built at 1.0 and applied at 1.25.
- No answer-side gate neurons.
- The only difference from linear_neg is the tag, so their comparison isolates it.

Tuning on f01 and f02 is slight contamination; results are also reported without them.

## Result of the tagged arm, and the next arm (written before it ran)

**Tagged arm: 0/40 pass** (`results/bench/tagged.jsonl`).
- Look-alike leak fell from a median of 0.33 to 0.17 against linear_neg: lower on 24 facts, higher on 6 (Wilcoxon p < 0.001).
- Recall did not change significantly: up on 12, down on 16 (p = 0.21).

**Recall is the main failure.** The write is taught from plain wordings only, so chat recall is near zero in every arm (medians 0.12, 0.00 and 0.00). With chat at 0, a fact can reach at most 8/12, which is below the pass mark of 9/12. The shipped recipe left chat out because chat teaching made answers leak into every chat reply about the company.

**Next arm, "tagged_chat":**
- the tagged arm, with the write also taught from the 4 chat training questions (Fact.questions);
- the bench negatives already include same-company and look-alike chat questions;
- nothing else changes.

**Comparisons:** tagged_chat against tagged, which isolates chat teaching with the tag, and against linear_neg. If time allows, a "linear_neg_chat" arm without the tag shows whether the tag is what makes chat teaching safe.

## Result of tagged_chat (end of the 2026-11-06 session)

**tagged_chat: 0/40 pass. It is the best arm so far.** Against the shipped recipe on the same facts:
- recall went up on 23 facts and down on 13 (p = 0.03);
- look-alike leak went down on 24 and up on 11 (p = 0.004).

Recall reached at least 9/12 on 19 facts, against 8 for the shipped recipe. 15 facts failed exactly one check, and for 11 of them it was the look-alike leak.

**The remaining look-alike leaks do not come from the name neuron.**
- Facts where it never fired on a test look-alike still leaked (median 0.21).
- The leak rate was barely related to how often the name neuron fired on look-alikes (Spearman ρ 0.20, p 0.22).
- So the linear answer write itself still fires on untagged look-alike questions.

**Next:**
1. An answer write that needs the tag: gated answer neurons. Their down columns must be bounded, and their threshold must sit above every training negative and calibration token, the two faults of the first gated build.
2. Several settings here were chosen after seeing this bench's results: the tag size and layer, and the chat teaching. A claim of improvement therefore needs a fresh bench: new facts, built by `build_bench.py` with another seed and other names, run once with settings fixed beforehand.

## Arm "gated_first": the first answer token needs the tag (written 2026-11-06, before it ran)

**Why.** In tagged_chat, 29 of 40 facts fail on look-alike leaks, and 14 of those leak even though the name neuron never fired on a test look-alike. So the linear answer write itself fires without the tag. At layer 20 the tag is readable: the fitted readout of the transport follow-up gave AUC 1.0 against look-alikes at 2× size. f02 broke only in tagged_chat (drift 37, garbled 1.0); it was fine in tagged, so a chat-taught part of the write blew up.

**Build.** It is the same as tagged_chat (Part 1, teaching examples, leave-alone reads, strength 1.25) except for three changes.

1. **The first answer token is written by gated neurons, not linearly.** Each first-token link of `answer_chain` (one per spelling, such as " Brno" in plain prompts and "Br" in chat replies) gets a pair of reused layer-20 neurons, the least used ones.
   - **Address:** LDA at the layer-20 MLP input. The positives are the link's own training reads. The negatives are:
     - the bench training negatives;
     - the package's built-in same-subject prompts and decoys, with name slots;
     - reads inside the answer, at every later answer token;
     - reads just after it, at " target," and " target.".
   - **Threshold:** set on the gate's actual argument, r = w·x / (c·x / c̄), with c the constant direction of layer 20.
     - The threshold τ sits halfway between the highest negative and the lowest positive.
     - The negatives here also include every token of calibration paragraphs 0 to 279.
     - If no gap exists, τ sits just above the highest negative, and the fact is marked not separable.
   - **Capped output.** The pair computes n1 − n2:
     - n1 = SiLU(a·u·(r − τ)) and n2 = SiLU(a·u·(r − τ − δ));
     - δ = lowest positive − τ, and a = 8/δ;
     - both up rows are c/c̄, so u ≈ 1;
     - the down columns are +v/8 and −v/8, where v is 1.25 × the link's push.

     So the output is between about 0 and u²·v for any input.
2. **Later answer tokens use the linear write, built from the later links only.** The first-token reads are added as leave-alone points, so the linear part writes nothing where the first token is decided. Its weights on the new neurons are set to 0.
3. **Every push is capped** at 3 × the median push of the same fact. The cap applies to the gated neurons and to the linear part, and every capped link is recorded.

Part 1 and the neuron pairs are applied at 1×. The linear part is built at 1.0 and applied at 1.25.

**Held-out check of the threshold:** how often the gate opens on calibration paragraphs 280 to 354, which no other step uses.

**Smoke test first: f01 to f04.** If any of them drifts more than 0.05 or loses more than 0.05 known facts, stop, record why, and revise before the full run.

**Decision rule** for the full run, paired with tagged_chat on the same 40 facts:
- primary: look-alike leak lower (Wilcoxon p < 0.05);
- and name-slot leak not higher;
- and median recall not lower by more than 1/12;
- and drift ≤ 0.05 with known facts within 0.05 on at least 39 of 40 facts.

If all hold, gated_first becomes the base for the next step: teaching chat answers at the point where the model's own reply gives them. This is development on bench40; a gain counts only once a fresh bench confirms it.

## Smoke test of gated_first, and a deviation (written after it, before any further run)

**Smoke test, f01 to f04** (`results/bench/gated_first.jsonl`):

| Fact | Recall | Look-alike | Names | Same | Drift | tagged_chat recall |
|---|---|---|---|---|---|---|
| f01 | 0.08 | 0.25 | 0.17 | 0 | 0.001 | 0.42 |
| f02 | 0.33 | 0.08 | 0 | 0 | 0.001 | 0 (broken, drift 37) |
| f03 | 0 | 0 | 0 | 0 | 0.001 | 0.83 |
| f04 | 0.42 | 0 | 0 | 0.12 | 0.001 | 0.67 |

**The damage rule passed.** No drift, and no loss of known facts. f02 is repaired; the pushes that broke it were 133× and 2,300× the fact's median. The 2,300× one is the floor of the size formula: size = (gap + margin) / max(sensitivity, 0.001), and here the push was measured as changing nothing.

**Deviation.** The rule says to run all 40 facts now. The run stops instead, because recall fell on all 4 facts (mean 0.21 against 0.48), which the decision rule would reject. There are two causes:

1. **The push cap hit the first tokens.** A first token needs 3 to 5× the median push, because continuation tokens are easy once the name has started. "3× the median" therefore cut f01's first-token pushes from 2.1 and 2.4 to 1.36.
2. **The gate margins are razor-thin.** f03's plain gate: lowest positive 39.378, highest negative 39.339, a gap of 0.1% of the score. Reads differ by up to 2% between batched and single prompts (separability check), so new wordings land on either side at random. Putting the threshold above all 31,816 ordinary tokens pushes it to the extreme tail.

**Next, before any revision is fixed:** measure the margins of candidate gate addresses on f01 to f08 (`diag_gates.py`, forward passes only).

## Margins measured, and gated_first v2 (written after `diag_gates.py`, before v2 ran)

**`diag_gates.py` on f01 to f08** (`results/bench/diag_gates.json`). Thresholds here are set from prompt negatives only, and test reads are left-padded as in generation.

- **The fact's own LDA gate is selective on 6 of 8 facts.** It opened on 1 or fewer of 12 test look-alikes for f01 to f04 and f06, and on 2 for f08. It opened on 4 for f05 and 5 for f07.
- **New wordings it opened on:** plain 4 to 7 of 8, chat 1 to 3 of 4.
- **Padding is not the problem:** it moves the gate score by 0.05 to 0.15, against margins of 0.6 to 9 on these prompts.
- **A tag-only gate (tagged against untagged reads) is not selective.** It opened on 1 to 7 of 12 look-alikes, because the name neuron also fires on many of them.
- **Ten times the ridge is no better.**
- **The thin margins in v1 came from two of its own choices:**
  - the reads just after the answer (" target," and " target."), which share the whole prompt with the positives;
  - the threshold over all 31,816 calibration tokens.

**v2** (arm "gated_first_v2"; v1 settings stay selectable) changes four things:
1. **Threshold:** from prompt negatives and inside-answer reads only, halfway to the lowest positive. Calibration tokens no longer set it. Their open rates are reported on paragraphs 0 to 279 and on held-out 280 to 354; the output is capped, and the drift check covers damage.
2. **After-answer reads** are no longer negatives.
3. **Saturation:** halfway between the threshold and the lowest training positive (δ = 0.5 × (lowest positive − τ)), so that new wordings just under the training positives get the whole push.
4. **Push cap:** 10× the fact's median. This catches f02's 133× and 2,300× links and leaves first tokens (3 to 5×) alone.

Everything else is as in v1. The same decision rule applies against tagged_chat. Damage stop: if any of f01 to f04 drifts more than 0.05 or loses more than 0.05 known facts, the run stops there.

## Result of gated_first v2, and v3 (written after v2's full run, before v3 ran)

**v2: 0/40 pass. It fails the decision rule** (`results/bench/gated_first_v2.jsonl`; paired with tagged_chat on 40 facts).

| | tagged_chat | v2 | Lower / higher in v2 | p | Rule |
|---|---|---|---|---|---|
| Look-alike leak, median | 0.167 | 0.083 | 20 / 7 | 0.012 | met: lower |
| Name-slot leak, median | 0.042 | 0 | 12 / 6 | 0.16 | met: not higher |
| Same-company leak | 0.125 | 0.125 | 20 / 5 | 0.005 | (not in the rule) |
| Recall, median | 0.667 | 0.500 | 29 lower / 7 higher | 0.0001 | **failed** |
| Facts over a damage limit | 3 (f02, f24, f28) | 2 (f22, f38: known facts −0.08 and −0.15) | | | **failed** (at most 1) |

Facts within the look-alike limit: 22, against 11. Facts reaching the recall mark: 4, against 19.

**What the data show:**
1. **Gating the first answer token on the fact's own address lowers leaks.** But a hard threshold drops new wordings that score just under it. For f06, 7 of 8 new plain wordings opened the gate with left padding but only 2 with right padding, so they sit inside the padding noise around the threshold.
2. **About a quarter of the first-token links (11 plain gates) are not separable even on their own training data.** These are mostly the low-recall facts: f11, f12, f15, f16, f18, f21, f22, f39.
3. **Leaks with the gate closed come through the ungated later tokens.** In 7 of 21 look-alike leaks in f01 to f15, the model starts a name by itself and the later-token write completes it ("**K…" → "Kwame Asante"). Name-slot leaks in f17 and f26 work the same way.
4. **Head-office gates can open on real-world questions of the same kind.** f38 lost 4 of 26 known facts; the known-fact prompts include "Microsoft is headquartered in" and "The capital of Egypt is".

**v3** (arm gated_first_v3) changes three things, each aimed at one finding above:
1. **For findings 1 and 2, a lower threshold with a graded push.**
   - The threshold sits a quarter of the way from the highest negative to the lowest positive.
   - The step is sharp: full gate within 0.1 of the remaining distance.
   - Once open, the push grows with the address score, like the linear write: up(x) = (r − r0)/(r̄ − r0), which is 1 at the mean training positive and 0 at the median calibration token.
2. **For finding 2, gates only where the training data separate.** A first-token link that is not separable on its own training data gets no gate. It stays in the linear write, exactly as in tagged_chat.
3. **For finding 4, 24 real-world questions as gate negatives:** other companies' head offices, CEOs, founders and owners, and other capitals, plain and chat. None of them overlaps the 26 known-fact prompts used for scoring.

Finding 3 is left for the next step, gating the later tokens too, so that this run isolates the changes above.

The decision rule against tagged_chat and the damage stop on f01 to f04 are unchanged.

## Result of gated_first v3 (end of the 2026-11-06 session)

**v3: 1/40 pass (f06), the first fact to pass on this bench in any arm. v3 fails its decision rule.** (`results/bench/gated_first_v3.jsonl`)

| Against tagged_chat, paired over 40 facts | v2 | v3 |
|---|---|---|
| Passes | 0 | 1 (f06); McNemar p = 1.0 |
| Recall, median | 0.667 → 0.500, lower on 29, higher on 7, p = 0.0001 | 0.667 → 0.583, lower on 26, higher on 3, p = 0.0002 |
| Look-alike leak, median | 0.167 → 0.083, lower on 20, higher on 7, p = 0.012 | 0.167 → 0.167, lower on 14, higher on 8, p = 0.19 (**primary not met**) |
| Facts over a damage limit | 2 | 5: f14, f19, f24, f32, f37, mostly 2 of 26 known facts lost (**not met**) |

**The v3 changes, separated:**
- **The graded push lowers recall where gates are used.** On the 21 facts whose first-token links were all gated, median recall was 0.50, against 0.58 for v2 and for tagged_chat.
- **Leaving a non-separable link linear brings back recall and leaks together.** On the 19 such facts, median recall was 0.58 (v2 0.33, tagged_chat 0.75), and look-alike leak 0.17 (v2 0.08, tagged_chat 0.17).
- **The lower threshold made head-office gates open on 0.1 to 0.6% of calibration tokens** (f32, f33, f35, f36, f37). Most of v3's damage failures are head-office facts.
- **The real-world negatives repaired f38** (known facts lost: 4 in v2, 1 in v3).

**What stands after v1 to v3:** a hard gate on the fact's own address cuts leaks (v2: look-alike p = 0.012, same-company p = 0.005), but that address, learned from 8 plain wordings and 4 chat questions, does not open on enough new wordings. Recall, not leakage, is the limiting factor of gated writes. No setting of threshold or push size fixed that here.

**Next, in order:**
1. **More, and more varied, teaching wordings drafted by the editor**, the workflow's own step. The package's built-in prompts are all inside the bench's pool, so they cannot be used. New wordings must be judged on a fresh bench, because they were written after the pool was seen.
2. **Teaching at the slot where the model's own chat reply gives the answer** ("… is **").
3. **Gating the later answer tokens too**, to close the path where the model starts a name by itself and the linear write completes it.
4. **A sharper name neuron**, for example one keyed on both words or at a later layer. The current one fires on 10.6% of test look-alikes on average, so a tag-only gate cannot separate them.

## Checks revised, and the next experiments (written 2026-11-07, before any of them ran)

**Why.** An audit of the saved answers of all six arms (CPU only) found:
- **The known-facts check can't tell churn from damage.** The unedited model answers only 17 of the 26 known facts. All 5 of v3's damage failures lost exactly 2 of them. The shipped recipe (linear) also had 2 such facts, so it fails the "39 of 40" damage clause itself.
- **Garbling is scored on the recall prompts only.** Counting it on the other groups against the unedited answer would not have changed any verdict so far (v2: 19 newly garbled answers over 15 facts, at most 1 in 10 per fact), but it is user-visible harm.
- **v2's recall misses are mostly the gate staying closed.** About two-thirds of them happen with the first-token gate closed. The name neuron fires on 0.94 of new wordings (1.00 on 24 of 40 facts), on 0.106 of look-alikes and never on unrelated companies or ordinary text. v2's gate opens on 0.64. So the part that fails to generalize is the question, not the company. The plain gate learns from 8 wordings (each with 4 prefixes) and the chat gate from only 4 questions.

**1. Known facts** (`known_facts.py` → `known_facts.json`).
- 180 candidates in six kinds: 30 company head offices, 20 leaders, 28 founders, 20 owners, 32 capitals and 50 general facts. They include the package's 26.
- A candidate is kept if the unedited model answers it: an accepted answer appears as a whole word in the greedy 12-token continuation, in batches of 16 with left padding. The filter repeats on the kept list until all of it passes.
- No candidate mentions a subject of v3's real-world gate negatives.
- **Damage: losing more than 5% of the kept facts,** counted item by item. Drift is unchanged.

**2. Garbling outside recall.** On the same-company, look-alike, unrelated and name-slot answers, an answer counts when it is garbled and the unedited answer to the same prompt was not. **Pass mark: at most 0.10 of those answers.** The recall-garbling mark (0.34) stays.

Checks 1 and 2 apply to every run from now on. bench40's earlier rows keep their old measures.

**3. Learning curve and split gate** (`diag_wordings.py`, on bench40, forward passes only).
- **Learning curve:** v2's first-token address is refit on 2, 4, 6 or 8 of the fact's 8 plain training wordings, with 1, 2, 3 or 4 of its chat questions, 6 random draws per size below 8. v2's negatives, threshold rule and Part 1 are exactly as in `gated_first.build`.
- **Split gate:** a question detector combined with the name neuron firing on the prompt's company.
  - The detector uses v2's address and threshold rule, fitted on the unedited model.
  - Positives: the chosen training wordings asked about 24 invented companies and the fact's own.
  - Negatives: the fact's same-company training questions and the package's negative prompts, asked about the same companies, plus the inside-answer reads.
- **Measured:** the share of each test group that either of the fact's two gates (plain and chat) opens on.
- **Rules:**
  - (a) *More wordings help* if, for v2 or the split gate, the median over facts of the open rate on new wordings rises by at least 0.10 from 4 + 2 to 8 + 4 training wordings.
  - (b) *The split gate qualifies as an arm* if, at 8 + 4 wordings, it opens on a median of at least 0.80 of new wordings, and its mean look-alike open rate is no higher than v2's.
  - If neither holds, steps 4 to 6 are skipped, and the next work is the misses that happen with the gate open.
- **Known limit:** the split gate is measured from the name neuron's firing at layer 4, not from a built neuron that reads the tag at layer 20. The transport follow-up made that readout near perfect (AUC 0.997 at 1×), but a build must confirm it.

**4. Teaching wordings** (if (a) holds). The editor writes about 30 plain and 30 chat templates per relation, none equal to one of bench40's wordings. They are saved as `teach_wordings.json` and committed before step 5 starts.

**5. bench40b** (if (a) or (b) holds).
- New facts, wordings and same-company questions, written by a separate agent that sees neither `teach_wordings.json` nor bench40's wordings. `build_bench.py` builds it with a new seed.
- **Duplicate rule:** a teaching template equal to any bench40b wording after normalizing (lower case, punctuation removed, the subject slot as one marker) is dropped (`bench_lib.teaching_templates`).
- Then the unedited arm, with the same exclusion rule as bench40.

**6. Runs on bench40b,** one GPU job at a time.
- **If (a):** gated_first v2 with the bench's 12 training wordings; v2 with the teaching wordings added; tagged_chat with the teaching wordings added. If (b) also holds, the split gate with them too. With teaching wordings, the look-alike and unrelated training companies are asked those wordings too, as negatives.
- **If only (b):** v2, the split gate and tagged_chat, all with the bench's 12 wordings.
- **Decision rules,** paired over bench40b's facts that are not excluded:
  - *Wordings* (primary, if (a)): v2 with teaching wordings recalls more than v2 without (Wilcoxon p < 0.05).
  - *Gate against the linear write:* the gated arm (v2 or split) against tagged_chat with the same teaching data, as in "Arm gated_first": look-alike leak lower (Wilcoxon p < 0.05); name-slot leak not higher; median recall not lower by more than 1/12; and drift, known facts and garbling outside recall within their limits on all but at most one fact.
  - If (b) holds, the split gate's build is written here before it runs.
- bench40b becomes a development bench once these runs are seen. Stage 2 of the roadmap needs another bench written blind.

## Result of step 3 (written after `diag_wordings.py`, before step 4)

`results/bench/diag_wordings.json`: 40 facts, 9 minutes. Open rates are the share of each test group that either of the fact's gates opens on.

| Training wordings (plain + chat) | 2 + 1 | 4 + 2 | 6 + 3 | 8 + 4 |
|---|---|---|---|---|
| v2: new wordings, median | 0.33 | 0.53 | 0.62 | 0.67 |
| v2: look-alikes, mean | 0.05 | 0.11 | 0.16 | 0.17 |
| Split gate: new wordings, median | 0.44 | 0.64 | 0.71 | 0.75 |
| Split gate: look-alikes, mean | 0.05 | 0.06 | 0.07 | 0.09 |

- **(a) holds.** From 4 + 2 to 8 + 4 wordings, the median open rate on new wordings rose by 0.11 for v2 and 0.12 for the split gate. Steps 4 to 6 run.
- **(b) does not hold.** At 8 + 4, the split gate opens on a median of 0.75 of new wordings, under the 0.80 required, though its look-alike rate (0.085) is half of v2's (0.171). So it is not an arm in step 6. It stays a candidate: the same check can be rerun on bench40b with the teaching wordings.
- **More wordings also open v2's gate on more look-alikes** (0.11 → 0.17 from 4 + 2 to 8 + 4). More positives lower the lowest positive, and with it the threshold. Step 6's look-alike rule covers this.

## Timing and separability with the teaching wordings, and a deviation (written before any step-6 run on bench40b)

**Timing, one fact (bench40 f01, a throwaway copy).** With the teaching wordings, both arms took about 210 s per fact against about 55 s, so about 2.3 hours per arm. The cause was reads made one prompt at a time: about 1,350 of them, mostly decoys and look-alikes asked every drafted wording.
- The bench builds now read in batches of 32 with right padding (`run_arm.null_reads_fast`, `writer_last_reads`). The prompts are the same.
- Against the one-at-a-time reads, the worst column of 918 differs by 6% (bf16 batch-shape noise; typical earlier measurements were about 2%). Every step-6 arm uses the batched reads. bench40's earlier rows used the old ones.

**Separability, training data only** (`diag_teach_sep.py` → `results/bench/diag_teach_sep.json`, bench40, 80 first-token links):

| Variant | Separable | Training positives above the highest negative, median | Negatives, median |
|---|---|---|---|
| A: v2 as run, 12 wordings | 0.81 | 1.00 | 306 |
| B: teaching wordings as positives only | 0.12 | 0.92 | 756 |
| C: look-alikes, unrelated companies and decoys also asked the teaching wordings (as specified in step 6) | 0.00 | 0.63 | 1,476 |

In the timed run of C, f01's two gates were not separable and recall fell from 0.33 to 0.17. v2 puts its threshold above the single highest negative, and asking the look-alikes the 60 drafted wordings raises that maximum.

**Deviation.** Step 6 runs variant B for both teaching arms, gated_first v2 and tagged_chat:
- the drafted wordings are added as positives only;
- the look-alike, unrelated, decoy and name-slot prompts are the bench's 12 training wordings, as in the arm without them (`run_arm.training_fact(teach=False)`, `run_arm.bench_negatives`).

**Why:** as specified, the arm would have measured v2's maximum-based threshold meeting 1,476 negatives, not whether more wordings help. B changes only the positive wordings, which is what the primary comparison asks. Any harm from not asking look-alikes the drafted wordings shows up in the look-alike leak, which the rule scores.

**Other facts fixed before step 6:**
- The known facts dropped 8 candidates that ask the same fact as a package anchor prompt. Every build leaves those alone, so they would flatter the check. That leaves 133 kept facts, and damage means losing 7 or more.
- The duplicate rule dropped 3 of the 240 teaching wordings; 237 are used. 26 of them are within one word of a bench40b wording; the rule keeps them.
- `build_bench.py` keeps bench40b's companies at least 3 edits from bench40's 40 subjects. Avoiding bench40's invented filler companies as well left too few names, and the search never ended.
- One bench40b test wording can be continued with a year instead of a city: "{s} moved into its present headquarters building in". It is left as written, since editing test wordings after reading them would tune the test, and it affects every arm alike.

The decision rules of step 6 are unchanged.

## Result of step 6 on bench40b (written after the runs)

`results/bench40b/`; the rules are applied by `step6_report.py` (`step6_report.json`). No fact is excluded: the unedited model gives none of the 40 answers on any test wording.

| bench40b, 40 facts | Passes | Recall, median (plain / chat, pooled) | Look-alike leak | Same-company leak | Name-slot leak | Known facts lost, median | Facts over a damage limit |
|---|---|---|---|---|---|---|---|
| gated_first v2, bench's 12 wordings | 0 | 0.42 (0.43 / 0.36) | 0.083 | 0.125 | 0 | 2.5 | 5 |
| gated_first v2 + teaching wordings | 1 (f35) | 0.50 (0.50 / 0.49) | 0.083 | 0.125 | 0 | 2 | 3 |
| tagged_chat + teaching wordings | 0 | 0.83 (0.81 / 0.78) | 0.54 | 0.31 | 0.17 | 5 | 15 |
| tagged_chat, bench's 12 wordings (supplementary, run after the rules were applied) | 0 | 0.67 | 0.25 | 0.25 | 0.08 | 6 | 17 |

**Wordings, the primary rule: met.** With the teaching wordings, v2 recalls more: up on 23 facts and down on 11, mean +0.09, Wilcoxon p = 0.003.
- Chat recall gained most: 0.36 → 0.49.
- The gates open on 0.72 of new wordings, against 0.59.
- The look-alike leak did not change significantly (up on 17, down on 6, p = 0.07), though the gates open on more look-alikes (0.19 against 0.12).

**Gate against the linear write with the same teaching data: not met.**
- Met: the look-alike leak is lower on 39 of 40 facts (p < 0.001), and the name-slot leak is lower.
- Not met: recall is lower by 0.33 (1/12 allowed).
- Not met: 3 facts are over a damage limit (1 allowed).

**Supplementary, outside the rules.** For tagged_chat, the teaching wordings raise recall (up on 30, down on 1) and every leak with it: look-alikes 0.25 → 0.54 (up on 31), same company 0.25 → 0.31, name slots 0.08 → 0.17. The linear write fires in proportion to similarity, so more varied wordings widen what it fires on. The gate kept its leaks.

**Damage.**
- **The new known-facts check separates churn from damage.** Gated edits lose a median of 2 of 133 known facts, the linear write 5 to 6.
- **The linear write's damage is larger than the old 26-prompt check showed.** tagged_chat now has 15 to 17 facts over a limit, against 3 on bench40 with the old check.
- **The gate's damage comes mostly from head-office gates opening on real-world questions:** f28 (Ensenada) lost 24 known facts and f32 (Leicester) 9. f25 lost 11, with drift 0.017. v3's real-world gate negatives, which repaired the same failure on bench40 (f38), were not part of v2.

**What stands.**
- More, more varied teaching wordings are a real lever, confirmed on a fresh bench for both designs. But they move each design along its own trade-off instead of out of it: the linear write gains recall and loses containment, and the gate keeps containment but gains less recall.
- The gate's advantage on look-alikes replicates on the fresh bench (0.083 against 0.25 with the same 12 wordings; lower on 33 of 40), at a recall cost (0.42 against 0.67).
- About a fifth of recall prompts open a gate without producing the answer.
- bench40b is now a development bench. Stage 2 needs another bench written blind.

**Candidates for the next step** (to be fixed here before any of them runs):
1. The split gate with the teaching wordings. In step 3 it opened on more new wordings than v2 with half the look-alikes, and its question detector can learn from every drafted wording asked about many companies.
2. v3's real-world negatives in every gated arm, for the head-office damage.
3. The push side: recall prompts where the gate opens and the answer still does not come.

## Yardstick and split gate (written 2026-11-07, before either ran)

**Why.** Facts passing every check have stayed at 0 to 1 of 40 for two days, while each new arm only moved along the same recall-against-leak trade-off. Two decisive measurements come next, each with its use fixed here first. Both run on bench40b, now a development bench.

**1. Yardstick: the fact in the prompt** (`in_context_arm.py`, arm "in_context").
- No weight changes. The fact's statement is its first training plain wording followed by the answer. It is written before every test prompt: on the line before a plain prompt, and before the question in the same user message for chat. This is in-context knowledge editing (IKE, arXiv 2305.12740) in its simplest form.
- Look-alike, unrelated, same-company and name-slot prompts get the same statement. The leak checks therefore measure whether the model carries a fact it was told to the wrong company. Known facts are asked with the statement before them.
- Scored by `bench_lib` exactly as the edit arms are, against the same unedited answers. Drift is 0, since no weights change.
- **Use:** it is reported next to every edit arm and changes no rule. If it passes on fewer than 20 of 40 facts, the bench's marks ask more of this model than it does even when told the fact. That will be reported, and the marks will not be relaxed because of it.

**2. Split gate.** Its build is written here after the literature survey and before it runs. Its rule is fixed now: **it must reach at least 10 of 40 facts passing on bench40b** (gated_first v2 with the teaching wordings: 1). Otherwise, gating the first answer token on an address at layer 20 is finished.

**Result of the yardstick** (`results/bench40b/in_context.jsonl`): 0 of 40 facts pass.

| bench40b | Passes | Recall (plain / chat) | Look-alike leak | Unrelated leak | Name-slot leak | Known facts lost, of 133 |
|---|---|---|---|---|---|---|
| Fact in the prompt, no weight change | 0 | 0.75 (0.84 / 0.49) | 0.58 | 0.33 | 0.75 | 41 |
| gated_first v2 + teaching wordings | 1 | 0.50 (0.50 / 0.49) | 0.083 | 0 | 0 | 2 |
| tagged_chat + teaching wordings | 0 | 0.83 (0.81 / 0.78) | 0.54 | 0 | 0.17 | 5 |

- Told the fact, this model applies it to any company: no fact stays within the look-alike limit, and it leaks to unrelated companies and name slots too. The extra sentence costs a median of 41 known facts. 22 facts reach the recall mark.
- **By the rule above,** the bench's marks ask more of this model than it manages even when told the fact. The marks are not relaxed.
- **What it means for the method:** the gated weight edit is already far more specific than putting the fact in the prompt (look-alikes 0.083 against 0.58, known facts lost 2 against 41), though the prompt recalls more (0.75 against 0.50).

## Split gate: the build (written after its smoke tests, before the 40-fact run)

**Literature.** From the survey in `related_work.md`, which is written next:
- Edits keyed on the subject token (ROME, MEMIT, AlphaEdit) leak to other questions about the same subject.
- Edits keyed on the last token (GRACE) miss paraphrases.
- A router on the subject alone scored 0.00 locality, against 0.91 on subject and relation together (arXiv 2606.14668).
- The survey found no published edit that ANDs a separate subject detector and question detector inside the weights.

**The build** (`split_gate.py`, arm `split_gate_teach`, bench40b, teaching wordings as positives only):
- **Part 1:** the name neuron of `tagged_arm.py`. Its tag direction is the one whose arrival at layer 20 is largest against layer 20's own variation: the top eigenvector of MᵀΣ⁻¹M (whitened arrival 94.1, against 77.6 for the top singular vector used so far).
- **Part 2:** per first answer token (plain spelling, chat spelling), one capped pair of reused layer-20 neurons. Its output is [question detector over its threshold] × [tag readout] × push.
- **Question detector:** an LDA address on the unedited model. Positives: the fact's training wordings, bench and drafted, asked about 24 invented companies and the fact's own. Negatives: the fact's same-company training questions and the package's negative prompts asked about the same companies; the other relations' drafted wordings asked about 6 of them; v3's 24 real-world questions; and the inside-answer reads.
  - The tag's arrival direction is projected out.
  - Threshold: the larger of the 99th percentile of prompt negatives and the highest inside-answer read. Full push halfway to the median positive.
- **Tag readout:** fitted across the 24 invented companies asked every training wording, read with the tag added at the company's name and without it. The detector's negatives join the scatter it ignores.
- **Later answer tokens:** gated_first's linear write, unchanged.

**Smoke history, f01 and f02 of bench40b, seen while fixing the build:**

| Tag readout | f01 result |
|---|---|
| 1. The fact's prompts against other companies' | Learned company identity at the last token: pushed on 0.42 of look-alikes, 10 known facts lost |
| 2. The fact's prompts with the tag against the same prompts without it | Too noisy: 41 known facts lost |
| 3. Across 24 companies with the tag added, bench wordings only | 34 known facts lost |
| 4. As 3 with every wording, the detector's negatives in the scatter and the whitened tag direction (the build above) | f01 passes. f02 fails on recall 0.42 and look-alikes 0.17, because the name neuron itself fires on 4 of its 12 test look-alikes |

**Rule, unchanged from above:** at least 10 of 40 facts pass on bench40b. Otherwise gating the first answer token on an address at layer 20 is finished. Also reported, paired over the same facts: against gated_first v2 with the teaching wordings, tagged_chat with them, and the yardstick. The smoke stop rule for f01 to f04 applies.

**Result of the split gate** (`results/bench40b/split_gate_teach.jsonl`): **2 of 40 pass (f01, f16, both head-office facts). The rule needs 10, so it is not met. By the rule above, gating the first answer token on an address at layer 20 is finished.**

| bench40b, 40 facts | Passes | Recall | Look-alike leak | Same-company leak | Known facts lost, median |
|---|---|---|---|---|---|
| Split gate + teaching wordings | 2 | 0.42 | 0.083 | 0 | 3 |
| gated_first v2 + teaching wordings | 1 | 0.50 | 0.083 | 0.125 | 2 |
| tagged_chat + teaching wordings | 0 | 0.83 | 0.54 | 0.31 | 5 |
| Fact in the prompt (yardstick) | 0 | 0.75 | 0.58 | 0.125 | 41 |

Paired against gated_first v2 with the teaching wordings:
- Recall is lower: down on 22 facts, up on 13, p = 0.016.
- The look-alike leak is lower: down on 18, up on 5, p = 0.027.
- The same-company leak is lower: down on 22, up on 1, p < 0.001.

**Where recall is lost:**
- The question detector opens on 0.82 of new wordings.
- The tag readout averages 0.80 on them.
- Their product gives at least half the push on 0.54 of them.
- 0.42 give the answer.

Person targets suffer most: recall 0.29 to 0.42, against 0.62 for cities, whose names have fewer tokens to complete. Pushes reach 0.13 of look-alike prompts (all through the name neuron firing on them) and 0.005 of known-fact prompts.

**What stands after every gated arm** (v1, v2, v3, teaching wordings, split gate): containment is solved to within the name neuron's precision, but recall stays at 0.42 to 0.58, and no setting moved it past that. The linear write and the fact in the prompt reach recall 0.75 to 0.83 only by leaking to look-alikes at 0.54 to 0.58.

## Fact block: an inserted layer (written 2026-11-07, before it ran)

**Why.** The user asked for a layer that sits between the model's layers, reads their input and corrects it. Every gated arm read company identity at layer 20, either from the question's end or from a tag the model itself had to carry there. At the company's name, identity separates far better: look-alike AUC 0.96 to 1.00 at layers 2 to 16 (`subject_site.py`), best at layer 4 with 32 facts at 0.95 or higher. An inserted block can read the name directly, with no transport. It is built in closed form from forward passes, and no weight of the model changes: detaching it is an exact undo.

**The block** (`fact_block.py`, arm `fact_block_teach`, bench40b, teaching wordings as positives only). At every token:
1. **"Is the fact's company named here?"** An LDA address at layer 4's MLP input, fitted on the company's last token in the fact's training wordings against the training look-alikes, the unrelated companies and 24 invented companies.
   - Threshold: 80% of the way from the highest negative to the lowest positive, as for the name neuron. The negatives are every token of those companies' prompts and of held-out ordinary text, since the block scans every token.
   - A running flag marks whether the company has been named so far. This is the block's attention: a hard causal maximum over the tokens so far.
2. **"Is the fact's question asked here?"** split_gate's question detector at layer 20, one per spelling of the answer (plain and chat). It has the same positives, negatives and threshold rule, but no tag projection, since there is no tag.
3. **If both hold,** the block adds the first answer token's push to layer 20's MLP output, saturating as in split_gate.
4. **When the expected first token arrives,** it pushes each next answer token while the tokens so far match the answer exactly. One answer per sequence.

Pushes come from `answer_chain` at layer 20 on the unedited model, applied at 1.25× and capped at 10× the fact's median. There is no name neuron and no linear write.

**Rule:** at least 10 of 40 facts pass on bench40b, the same bar as the split gate. Otherwise an inserted block of this kind is dropped for this model. The smoke stop rule for f01 to f04 applies. Results are reported paired against the split gate, gated_first v2 with the teaching wordings, tagged_chat with them, and the yardstick.

**Result of the fact block** (`results/bench40b/fact_block_teach.jsonl`): **5 of 40 pass (f13, f15, f16, f28, f31), the most of any arm, and below the 10 the rule needs, so the rule is not met.** It is the first arm with no damage at all: 0 known facts lost on every fact, and drift 0.0000.

| bench40b, 40 facts | Passes | Recall (plain / chat) | Look-alike leak | Same-company leak | Known facts lost | Drift |
|---|---|---|---|---|---|---|
| Fact block + teaching wordings | 5 | 0.67 (0.70 / 0.57) | 0.083 | 0 | 0 | 0.0000 |
| Split gate + teaching wordings | 2 | 0.42 (0.42 / 0.42) | 0.083 | 0 | 3 | 0.001 |
| gated_first v2 + teaching wordings | 1 | 0.50 (0.50 / 0.49) | 0.083 | 0.125 | 2 | 0.001 |

- **Recall** against the split gate: up on 33 facts, down on 2 (p < 0.0001). Against v2: up on 28, down on 5.
- **Look-alike leak** is level with v2 (p = 0.81) and slightly above the split gate (p = 0.048).
- **9 facts fail only on look-alikes.** On f01, the layer-4 detector fires on the one-letter-off "Lixemba Marine" at " Marine" in 3 of its 4 prompts. Look-alike names sharing keys is a known limit (arXiv 2408.07413). The training one-letter-off variant swapped the first letter; the test one changed an inner vowel.
- **16 facts fail only on recall.** In chat, the model often restates the question before answering ("The headquarters of **Lixomba Marine** is located in **…"), while the block pushes at the start of the reply. Plain recall is 0.70 and chat 0.57.

Both causes were found on the smoke fact f01 and the full run. Any revision is designed after seeing bench40b, so it must be confirmed on a new blind bench.

## Fact block v2 (written 2026-11-07, before its diagnostic and before bench40c exists)

**Why.** The fact block passed 5 of 40 on bench40b, under its bar of 10, so the block as built is dropped. All 35 of its failures are recall, look-alike leak, or both (16 recall only, 9 look-alike only, 10 both), and both causes were measured on bench40b:
- **Look-alikes:** the layer-4 company detector fires on names one letter off.
- **Chat recall:** 56 of the 68 chat misses restate the question first ("The headquarters of **Lixomba Marine** is located in **…"). Pushing at the reply's own answer slot could rescue at most 20 of those 56: in the other 36 the full answer would not fit in the 16 generated tokens, or no bold answer starts within them. So the answer has to come first, at the start of the reply, where v1 already pushes.
- **Plain recall:** 96 of 320 plain prompts miss, mostly by continuing another way ("…moved into its present headquarters building in" → " 2010").

v2 changes the parts that failed and nothing in the scoring. It is a new design, so it gets its own rule and is judged on a new blind bench.

**What changes, fixed now:**
1. **The company is recognised by its exact name tokens:** its token ids with and without a leading space, and only if the next token does not continue the word.
   - In layer terms: each token's input embedding is compared with the stored embeddings of the name's tokens, and a running AND runs over consecutive tokens.
   - It can't fire on any other spelling. Look-alike names, unrelated companies, name slots, known facts and ordinary text therefore can't trigger it.
   - **Limit, stated plainly:** it also won't fire on a variant of the name that wasn't registered, such as other casing or a short form. The bench doesn't test those.
2. **The question detector's threshold and the push size are set against the company's own other questions only,** since only prompts naming the company can now trigger the block. Both come from training prompts, never test prompts. The exact rules are fixed after the diagnostic below and written here before bench40c is built.

**Diagnostic first** (`diag_block.py`, on bench40b, now a development bench), on the recall and same-company prompts and on the training prompts the rules will come from:
- whether v1's question detector opens at the last token;
- the smallest push that makes the first answer token the top token;
- the smallest push that brings out the whole answer.

**bench40c** is written by a blind agent from the same brief as bench40b. Its source is opened and built only after v2's build is frozen.

**Rule, fixed now, on bench40c:**
- **20 or more of 40 pass:** the inserted layer works on this model at this bar. Stage 1 is done, and the next stage is many facts in one model.
- **10 to 19:** at most one more revision, aimed at a measured cause and judged on another fresh bench. Then the direction stops either way.
- **Under 10:** the inserted layer is finished on Qwen3.5-0.8B. The next step changes the model (Qwen3.5-4B), not the method.

Reported next to it: base on bench40c, and v1 against v2 on bench40b. The smoke stop rule for f01 to f04 applies.

## Fact block v2: the build, fixed (written after the diagnostic, before any v2 run and before bench40c was built)

**Diagnostic** (`diag_block.py` → `results/bench40b/diag_block.json`, all 40 facts of bench40b, nothing generated):

| bench40b | Plain recall prompts (320) | Chat recall prompts (160) |
|---|---|---|
| v1's question detector open at the last token | 0.900 | 0.938 |
| Whole answer brought out at v1's full push (1×) | 0.866 | 0.550 |
| … at 2× | 0.963 | 0.881 |
| Not brought out even at 12× | 0.009 | 0.062 |

- **The detector mostly opens; the push is too small, above all in chat.** v1 also scaled its first-token push down near the threshold, which cut it further.
- **Same-company test prompts:** v1's detector opens on 0.078 of them. The limit is 0.25 per fact.
- **Training wordings:** the push needed for the whole answer has a median of 1×, a 90th percentile of 2× and a 95th of 6×.

**Estimated passes** (`diag_block_report.py`): a recall hit means the detector opens and the whole answer comes out at the push given. A same-company leak means the detector opens and the first answer token comes out. These are readings without generation, so they are an estimate, not a result. At 1× the estimate gives 16, and v1 actually passed recall on 14 facts.

| Threshold | 1× | 1.5× | 2× | 3× |
|---|---|---|---|---|
| v1's | 16 | 33 | 34 | 35 |
| Highest of the company's own training negatives | 15 | 32 | 33 | 34 |
| Their 95th percentile | 13 | 30 | 31 | 32 |
| A quarter of the way from that highest negative to the median positive | 12 | 26 | 27 | 27 |

At 2×, scaling the push down near the threshold as v1 did gives 26 instead of 34.

**Fixed:**
- **Company:** the exact-name matcher of the section above (`NameMatcher` in `fact_block.py`). On bench40b's prompts it marks the company in all 480 recall, 320 same-company and 7,380 training prompts, and in none of the 480 look-alike, 480 unrelated-company and 360 name-slot prompts.
- **Question detectors:** v1's, unchanged: the same positives, negatives and threshold.
- **Push:** 2× v1's on every answer token. This is the 90th percentile of what the training wordings need, and where the estimate levels off. The first-token push is whole once the detector opens (`PUSH_MULTIPLE = 2.0`, `FULL_ONCE_OPEN = True`, arm `fact_block_v2_teach`).
- **Kept from v1:** one answer per sequence, exact-token continuation, and the answer_chain pushes with v1's cap.

**Still failing by the estimate at 2×:**
- f18, f33, f38, f39 and f40 on recall: their misses need more than 2× or have the detector closed.
- f20 on same-company leaks: its detector opens on 3 of 8.

**Order:**
1. bench40b, the development check: smoke facts f01 to f04 under their stop rule, then all 40, reported against v1 on the same facts.
2. Only a bug found there may change the build, and any change is written here first.
3. Then bench40c is built from its blind source, the base model is run on it, and v2 is run on it under the rule above.

## Fact block v2: the development check on bench40b (written after it, before bench40c was built)

**Result** (`results/bench40b/fact_block_v2_teach.jsonl`, `fact_block_report.py`): **25 of 40 pass, against 5 for v1 on the same facts.** 22 facts pass only with v2 and 2 only with v1 (McNemar p = 3.6e-05).

| bench40b (development) | Passes | Recall median | Look-alike leak | Same-company leak median | Known facts lost | Drift |
|---|---|---|---|---|---|---|
| Fact block v2 | 25 | 0.917 | 0 on every fact | 0.125 | 0 on every fact | 0.0000 |
| Fact block v1 | 5 | 0.667 | 0.083 | 0 | 0 on every fact | 0.0000 |

- **Paired against v1:**
  - Recall is up on 29 facts and down on 5 (p = 0.0005).
  - The look-alike leak is down on 24 and up on none.
  - The same-company leak is up on 23 and down on 2 (p < 0.0001).
- **By relation:** head office 5 of 10, owner 7, CEO 7, founder 6.
- **Garbling stays low:** at most 0.083 in recall, and at most 0.045 elsewhere.
- **The 15 failures:** 9 on recall and 6 on same-company leaks.

**Two effects the diagnostic missed,** because it pushed at the last token only:
1. **Pushes add up.** The block also pushes at earlier tokens where the detector opens, and those pushes accumulate. At 2× some person answers overshoot into another token: plain replies give " A." or "E." for the first name, and chat replies stop starting with the name, which v1 did for those facts. Recall falls to 0.08 to 0.33 on f33, f36 and f38. f31, which v1 passed, falls to 0.67.
2. **Leaks inside replies.** During generation the detector also opens inside replies to other questions about the company, so the answer comes out there. Six facts fail on it (f13, f20, f32, f34, f35, f37), and four of them are head-office facts, where a city fits many other questions.

Neither effect is a code error, so the build stays as fixed. bench40c is next, under the rule above.

## Result of fact block v2 on bench40c (blind)

bench40c was written by a blind agent from bench40b's brief (seed 20261108, `bench40c_source.json`), with its companies at least 3 edits from every subject of bench40 and bench40b. It was built and run only after v2's build was fixed and committed (d8bda84). The unedited model gives none of the 40 answers on any recall prompt, so no fact is excluded.

**29 of 40 pass. The rule asked for 20 or more, so it is met: the inserted layer works on this model at this bar, and stage 1 is done.**

| bench40c (blind), 40 facts | Passes | Recall median | Look-alike, unrelated, name-slot leak | Same-company leak median | Garbled | Known facts lost | Drift |
|---|---|---|---|---|---|---|---|
| Fact block v2 | **29** | 1.00 | 0 on every fact | 0.125 | at most 0.083 | 0 on every fact | 0.0000 |
| Unedited model | 0 | 0 | 0 | 0 | — | — | — |

- **By relation:** CEO 10 of 10, founder 7, head office 7, owner 5.
- **The 11 failures:**
  - 6 on same-company leaks: 3 owner (f06, f11, f15) and 3 head office (f08, f09, f22).
  - 5 on recall: 2 owner (f01, f17) and 3 founder (f20, f24, f39).
- **No fact broke any damage limit.** Every leak outside the company's own questions is 0, as the exact-name design guarantees.
- **The result is in line with the development check** on bench40b: 25 of 40 there against 29 here.

**What the result does and doesn't show:**
- **It shows one fact at a time:** one block attached to the unedited model, judged by greedy generation in plain and chat form on new wordings.
- **Look-alikes pass by design:** the block can't fire on any other spelling. It also won't fire on spellings of the company that weren't registered (casing, short forms, aliases), which the bench doesn't test.
- **The model's own weights don't change:** the block sits beside layer 20, reading layers 0 and 20 and writing into layer 20's MLP output. Its parts (the detectors and pushes) are built in closed form from forward passes, and detaching it is an exact undo.
- **The remaining failure is mainly other questions about the same company,** most of all where the answer type fits many questions (cities, owners), plus pushes that overshoot on some person names.

**Next, stage 2: many facts in one model.** Attach many blocks together, or one block holding a table of facts, and check that facts don't disturb each other, along with the cost per fact. Its rule is written here before it runs.

## Stage 2: a fact table (written 2026-11-07, before any of it ran)

**Why.** Stage 1 is done: one fact block at a time passed 29 of 40 on blind bench40c. Stage 2 asks whether one inserted layer can hold many facts at once. bench40c's 11 failures were read word for word after its run, so bench40c is now a development bench like bench40b. Together they show these causes, each met by a change below:
- **Wrong question answered** (bench40c f06, f11, f15): an owner block answers "Who is the CEO of …?", and "Q: Where is Yadrinkel Brewing based? A:" gets the owner. The detector partly learned "an answer about this company comes next", not "the owner comes next".
- **Stray insertions inside replies** (bench40c f08, f09, f22; bench40b f13, f20, f32, f34, f35, f37): a head-office block drops its city where the company's name was about to be repeated ("Who is credited with founding Geelong?").
- **Pushes add up** (bench40b f31, f33, f36, f38): the block also pushes at earlier prompt tokens where its detector opens, and the pushes accumulate until the answer overshoots (" A.", "E.").
- **A name glued to the word before it** ("Media'sFemke Kooistra", bench40c f33): mid-reply, the spelling without a leading space was pushed.
- **Detours** (bench40c f01, f17, f20, f39): the reply restates or reformats the question, and the answer doesn't come out where the block pushes.
- **Capitalisation:** the exact-token company check can't fire on "quenzafor capital".

**The table** (`fact_table.py`, arm `fact_table_teach`). One inserted layer at layer 20, as the fact block, holding every loaded fact. No weight of the model changes, and detaching it is an exact undo. It is built in closed form from forward passes of the unedited model; nothing is trained.
1. **Company, in any capitalisation.** One index of every loaded company name, matched on the decoded text with case ignored and spaces collapsed, as a whole word: no letter or digit just before or just after. A sequence's company is the one named last. Look-alike names differ in letters, so they still never match.
2. **Question type, with competition.** Each fact keeps its own question detector per form (plain, chat), built as in v2 with more training data:
   - every chat question is also asked in plain form, as "Q: {question} A:";
   - company names also appear in lower case and upper case (the fact's own, and 6 of the 24 invented companies);
   - 24 plain and 24 chat questions about other things a company has (size, products, money, history), drafted now in `table_wordings.json`, are added as negatives;
   - **reply positions are negatives:** the unedited model's 16-token greedy replies to the fact's other training questions (its same-company training questions, 4 plain and 4 chat drafted wordings of each other relation, 8 plain and 8 chat generic questions), about its own company and 3 invented ones. Every reply position is a negative in the fit, and the threshold may also be set against them (options below).
   - **Shared detectors** for each relation (owner, CEO, founder, head office; plain and chat) are built the same way from the drafted wordings alone, with the threshold at the 99th percentile of their prompt negatives.
   - **Competition:** when a company is named, the layer scores each of its facts' detectors and the shared detectors of the relations it holds no fact on for that company, each as (reading − threshold) / delta. It answers only if the winner is one of its facts and is over its threshold. A shared detector beats a fact only if it is ahead by more than a margin (options below). Between two facts of one company, the higher score wins.
3. **It pushes only where the model is choosing a token:** the last position of each forward call. At earlier positions of a prompt the next token is already given, so a push there only disturbs the context. One consequence: in a single pass over given text, such as the drift check, only the last position can be pushed. The generation checks (known facts, look-alikes, unrelated companies) are unaffected.
4. **Spelling by context.** The answer is pushed without a leading space after whitespace, a line break or an opening mark (`*`, `"`, `'`, `(`, `[`), and with one otherwise.
5. **Kept from v2:** answer_chain pushes at 1.25× capped at 10× the fact's median, the whole push once the detector opens, exact-token continuation, and one answer per sequence.
6. **Guard:** no string used to build any fact equals a test prompt of the bench being scored, after lower-casing and removing punctuation.

**Fixed after a diagnostic, from options listed now** (`diag_table.py` on bench40b; nothing is generated with the layer in place):
- **Threshold against reply positions:** none (v2's rule alone), or their 99th, 99.5th or 99.9th percentile, or their maximum.
- **Push size:** 2× (v2), 3×, or per fact: the 90th percentile of the multiple its own training wordings need for the whole answer, between 2× and 6×.
- **Competition margin:** 0, 0.5 or 1.
- **Estimated passes, as for v2:**
  - a recall hit means the layer fires at the prompt's last token and the whole answer comes out at the push given;
  - a same-company leak means the layer fires anywhere in the unedited reply (last prompt token or a reply token) and the first answer token comes out there.
- **Choice:** the most estimated passes; ties go to the lower threshold, then the smaller push, then the smaller margin.
- Competition switched off is also estimated and reported, but stays on, since a table with two facts on one company needs it.

**Development check.** The table holds bench40b, bench40c and `devpairs` at once (100 facts). `devpairs` has 10 new companies with two facts each, of different relations, built from bench40b's wordings by `build_bench2.py`. bench40b's 40 facts and devpairs' 20 are scored. Only a bug found there may change the build, and any change is written here first.

**bench40d (blind).** A blind agent writes it from bench40c's brief with one change: 40 facts on 30 companies, 10 of which hold two facts of different relations, with 10 facts per relation. It is opened and built only after the build is frozen.

**Scoring on bench40d,** with all 140 facts loaded at once (bench40b, bench40c, devpairs and bench40d):
- **Every stage-1 check, unchanged:** recall, look-alike, same-company, unrelated, name slots, garbling, known facts and drift.
- **Two new checks, both needed to pass:**
  - **Other facts:** no loaded fact's full answer appears in this fact's answers where it doesn't belong. Exceptions: the unedited model already said it, or the prompt asks for the company's other fact. Limit 0.
  - **Sibling:** for a company with two facts, the other fact's answer on this fact's recall prompts. Limit 1 of 12.
- **Capitalisation:** 2 plain and 2 chat test wordings, each asked with the company in lower case and in upper case (8 prompts per fact), scored like recall. It is reported per fact and is not part of a fact's pass.

**Rule, fixed now, on bench40d:**
- **30 or more of 40 pass:** stage 2 works at this bar: one layer holds 140 facts at once, above stage 1's 29.
- **20 to 29:** the table holds many facts at about stage 1's level. The paired comparison with v2 below says whether loading many facts or the changes cost anything. At most one revision, aimed at a measured cause, on a fresh bench.
- **Under 20:** the table fails at this bar; back to one block per fact.

Also fixed now and reported with it:
- **Capitalisation is fixed** if the median capitalisation recall is 0.75 or more.
- **Question types are told apart** if both facts pass on 7 or more of the 10 two-fact companies.
- **v2, one fact at a time, on bench40d:** paired, with McNemar on passes and Wilcoxon on recall and same-company leak.
- **Cost:** build seconds per fact, stored size per fact, and generation time with and without the layer.

The smoke stop rule for the first four scored facts applies.

**Before the full diagnostic: a check of the detector's new negatives** (`diag_table_negatives.py`, bench40b f01 to f06, written after it ran; the build is unchanged). The diagnostic's smoke test (f01 to f03) showed the new detector shut at the last token of some unusual plain recall wordings that v1 opened on ("Speaking to reporters on Tuesday, X chief executive"). Four variants on six facts:
- **Dropping the own-company questions of other relations** opened 1 more recall prompt on 2 of 6 facts. It also raised wrong-question openings on those questions from 0.01–0.02 to 0.02–0.06. They stay.
- **Keeping the reply positions out of the fit** let the detector open somewhere in 1 to 4 of the 8 same-company replies, against 0 to 1 with them in. Setting the threshold against the replies instead then cut recall to 3 or 4 of 12. They stay in the fit.
- **With the build as written,** recall prompts open at the last token on 8 to 12 of 12. v2's actual recall on these six facts was 0.67 to 1.0 in generation, above its own last-token estimate, because a block can also fire later in a reply.

So the build stays as written, and the diagnostic runs on all 40 facts.

## Stage 2: the build, fixed (written after the diagnostic, before the development run and before bench40d was opened)

**Diagnostic** (`diag_table.py` → `results/bench40b/diag_table.json`, `diag_table_report.py` → `diag_table_report.txt`; 40 facts, nothing generated with the layer). Estimated passes:

| Competition margin | Reply threshold none, 0.99, 0.995 or 0.999 | Reply threshold: their maximum |
|---|---|---|
| 0 | 22 at 2×, 22 at 3×, 23 per fact | 19, 20, 20 |
| 0.5 or 1 | 25 at 2×, 25 at 3×, **26 per fact** | 22, 23, 23 |
| Off (reported only) | 25, 25, 26 | 22, 23, 23 |

- **Same-company leaks are nearly gone, now counted over the whole reply:** 3 facts have any, at most 1 of 8 (0.125). v2's diagnostic looked only at the prompt's last token and missed the leaks inside replies that failed 6 facts.
- **Recall is the limit.** All 14 estimated failures are recall. The detector opens at the last token on a median 0.75 of recall prompts (mean 0.79), against 0.90 to 0.94 for v1's detector. This is the price of the reply positions in the fit (see the check above). A block can also fire later in a reply, which the estimate doesn't count.
- **The reply threshold changes nothing below the maximum:** with the replies in the fit, they already sit below the threshold.
- **Competition at margin 0 vetoes real recall prompts** and costs 3 estimated passes. At 0.5 it vetoes 1 of 480 and costs none. On single facts it neither helps nor hurts at 0.5, since the leaks are already gone without it. Its job is the two-fact companies.
- **Per-fact pushes:** 2× for 31 facts, 4× for 1 and 6× for 8.

**Fixed by the rule:** no extra reply threshold (v2's rule over the new negatives), the per-fact push, and competition margin 0.5. 26 estimated passes. In `fact_table.py`: `OPTIONS = dict(reply_q=None, push="per_fact", margin=0.5)`.

**Order:**
1. A functional smoke test of the layer: answers on a few prompts, and an exact undo on detaching.
2. The development check: base on devpairs, then the table with 100 facts loaded, scored on bench40b and devpairs. Only a bug found there may change the build, written here first.
3. Freeze and commit.
4. Build bench40d from its blind source, run the base model and v2 one fact at a time on it, then the table with 140 facts loaded.

**Smoke test, and a deviation: the push is 2×, not per fact** (scratchpad `smoke_table.py`, bench40b's cached parts; written after it ran, before the development run). Detaching the layer gives back the unedited model's output exactly. Owner and CEO facts answer on plain and lowercase prompts. Other questions about the same company get no push.
- **The per-fact push garbles.** f01 (Breda) came out as "Bredan, Bredan, Bredan" and "Bredaville, Bredaville, Bredaville" at its per-fact 6×. f18 came out as "Astrid A. F. S. S. S. S.". At 2× the same prompts give "Breda, Belgium" and "Astrid Fjeldstad".
- **The cause is the per-fact rule, not the code.** It counts training wordings that no multiple brings out (16 of 71 for f01, 24 for f18, 31 for f25) as needing the most, so 8 facts sit at the 6× cap.
- **Why the diagnostic missed it:** its estimate checks that each answer token comes out, not what follows the answer.
- **The push is set to 2×, v2's,** estimated at 25 against 26 (the tie-break prefers it to 3×). `OPTIONS = dict(reply_q=None, push=2.0, margin=0.5)`. Everything else stays as fixed above.

## Stage 2: the development check (written after it, before bench40d was opened)

**Result** (`results/bench40b/fact_table_teach.jsonl`, `results/devpairs/fact_table_teach.jsonl`, `fact_table_report.py`). The table held bench40b, bench40c and devpairs at once: 100 facts on 90 companies.

| Development, 100 facts loaded | Passes | Recall median | Same-company leak median | Capitalisation recall median | Other facts, look-alike, unrelated, name slots | Known facts lost, drift |
|---|---|---|---|---|---|---|
| Table on bench40b | 22 of 40 | 0.750 | 0 | 0.875 | 0 on every fact | 0, 0.0000 |
| v2, one fact at a time, on bench40b | 25 of 40 | 0.917 | 0.125 | — | 0 | 0, 0.0000 |
| Table on devpairs | 15 of 20 | 0.792 | 0 | 0.875 | 0 on every fact | 0, 0.0000 |

- **Paired against v2 on bench40b:**
  - The same-company leak is down on 23 facts and up on none (Wilcoxon p = 2e-05).
  - Recall is down on 29 and up on 1 (p = 3e-06).
  - Passes: 6 facts pass only with the table and 9 only with v2 (McNemar p = 0.61).
- **Every failure is now recall:** 18 on bench40b and 5 on devpairs. v2's 15 were 9 recall and 6 same-company leaks.
- **100 facts at once didn't disturb each other.** No loaded fact's answer appeared where it didn't belong, on any of the 60 scored facts. The sibling check was at most 1 of 12, and no damage limit moved.
- **Two-fact companies:** both facts pass on 5 of 10. The other fact's answer appeared on a fact's recall prompts at most once in 12 (devpairs f01 and f13). Owner facts are the weakest, with 1 of 5 passing on devpairs.
- **Cost:**
  - 44 KB per fact in float32.
  - Generation of 48 prompts took 1.4 s with the layer and 1.3 s without, about 8% slower.
  - Building a fact takes about 43 s once, and the result is kept.
- **No bug was found, so the build is frozen as fixed above.** On the lowest-recall facts (f38 at 0.00, f33 at 0.08), the table prints the same answers as v2 on most prompts. The first name comes out and the model extends it ("Kavithakumar" for "Kavitha Venkatesan"), or the detector stays shut. The diagnostic estimated 25 at 2×; the run gave 22.

**What it shows so far:** one layer can hold 100 facts with no interference and no damage. Its detector trades recall for safety: leaks into other questions about the same company are gone, and recall drops by about a sixth. On bench40b the net count is level with v2 one fact at a time. bench40d is next, under the rule above.

## Result of the fact table on bench40d (blind)

bench40d was written by a blind agent from bench40c's brief, changed for two-fact companies (seed 20261111, `bench40d_source.json`): 40 facts on 30 companies, 10 of them with two facts. Its companies are at least 3 edits from every earlier subject. It was opened and built only after the build was frozen and committed (e4030ad). The unedited model gives none of the 40 answers.

**With all 140 facts loaded at once (bench40b, bench40c, devpairs and bench40d), 27 of 40 pass. That falls in the rule's 20-to-29 band: the table holds many facts at about stage 1's level.** Fact block v2, one fact at a time on the same facts, also passes 27. 6 facts pass only with the table and 6 only with v2 (McNemar p = 1).

| bench40d (blind) | Passes | Recall median | Same-company leak median | Capitalisation recall median | Other facts, look-alike, unrelated, name slots | Known facts lost, drift |
|---|---|---|---|---|---|---|
| Table, 140 facts loaded | **27** | 0.833 | **0** | 0.875 | 0 on every fact | 0, 0.0000 |
| v2, one fact at a time | 27 | 0.917 | 0.125 | — | 0 | 0, 0.0000 |
| Unedited model | 0 | 0 | 0 | — | — | — |

- **Paired against v2:**
  - The same-company leak is down on 25 facts and up on none (Wilcoxon p = 8e-06).
  - Recall is down on 23 and up on 2 (p = 3e-05).
- **Failures:** all 13 of the table's are recall. v2's 13 are 7 same-company leaks and 6 recall.
- **By relation:** CEO 9 of 10, founder 6, owner 6, head office 6.
- **Requiring the exact whole name for recall** leaves both counts at 27.
- **140 facts at once didn't disturb each other.** No loaded fact's answer appeared where it didn't belong, on any scored fact, and there was no damage.
- **Cost:** 44 KB per fact. Generation of 48 prompts took 1.3 s with the layer and 1.2 s without.

**The other checks fixed before the run:**
- **Capitalisation is fixed:** the median is 0.875 on lower- and upper-case prompts, against a bar of 0.75. v2 can't fire on those prompts by design.
- **Question types told apart: not met as written.** Both facts pass on 4 of the 10 two-fact companies, against a bar of 7. The failures are recall, not confusion between a company's two facts:
  - The other fact's answer appeared on a fact's recall prompts once in 12 on one fact (f28), and never on the other 19.
  - Paired facts pass at the same rate as single ones (13 of 20 against 14 of 20). Treating each pass as independent predicts 4.2 companies with both passing.
  - v2, running each fact alone where no confusion is possible, gets 5 of 10.
  - So the bar measured recall more than separation.
- **A note on the run:** v2 took 1,224 s on f32 when the GPU filled up. It was stopped and resumed at f33 after a memory release between facts was added to `fact_block.py`. That changes speed only, and finished facts were kept.

**What it shows:**
- One inserted layer holds 140 facts at once with the same pass count as one fact at a time. Facts don't interfere, nothing is damaged, and it answers to any capitalisation of the company.
- The trade is the same as on the development benches: leaks into other questions about the same company are gone, and recall is lower.
- **Next, by the rule:** at most one revision aimed at a measured cause, judged on a fresh blind bench. The cause is recall: the detector stays shut at the last token of some wordings, and some person names drift after their first token.

## Stage 2, the one revision: a looser gate where the reply starts (written 2026-11-08, before any of it ran)

**Why.** The table's result on bench40d (27 of 40) fell in the rule's 20-to-29 band, which allows one revision aimed at a measured cause, judged on a fresh blind bench. bench40d's answers have now been read, so it is a development bench like bench40b, bench40c and devpairs. Measured from the stored answers, with no new run (scratchpad `recall_misses.py`, `v2_leak_where.py`):
- **The table's recall misses are mostly no answer at all.** On bench40b, devpairs and bench40d, 320 of the table's 1,200 recall prompts miss. On 294 of them no part of the answer comes out. On 26 the first word comes out and the rest drifts or is garbled ("Kavithakumar"); v2 drifts on 24 of those same 26, so drift is not the table's own loss and is not addressed here.
- **v2 answers about half of those prompts.** v2 was run on bench40b and bench40d. Of the table's 245 no-answer misses there, v2 gives the whole answer on 126. The table gives it on 19 prompts where v2 doesn't.
- **Where the table loses them:** the stage-2 diagnostic found its detector open at the prompt's last token on a median 0.75 of recall prompts, against 0.90 to 0.94 for v1's detector, which v2 uses. The check of the new negatives found that the reply positions in the fit are not what shuts it there: taking them out didn't open more recall prompts on any of 6 facts. The stricter threshold over the larger set of prompt negatives is the remaining difference.
- **The risk is v2's leaks.** Of v2's 142 leaking same-company replies on bench40b, bench40c and bench40d, 51 have the answer at the very start of the reply, where the prompt's last token chose it (23 of 50, 18 of 46, 10 of 46). The other 91 begin later in the reply, where the table's strict detector would stay.

**The change** (arm `fact_table_rev_teach`, `FACT_TABLE_REVISION=1` in `fact_table.py`; off, the frozen table runs unchanged). The looser rule applies only where the model chooses the first token of its reply: the last position of the forward call that reads the whole prompt. There a fact's detector opens if its strict detector does or the looser one does. In competition its score is the larger of the two, and between two facts of one company the higher score still wins. Inside replies, at every later call of one token, only the strict detector counts, as now. Nothing else changes: the name index, competition with the shared detectors at margin 0.5, the push at 2×, spelling by context, exact-token continuation, one answer per sequence, and the guard.

**Options for the looser rule:**
- **P0:** none (the frozen table, the reference).
- **P1:** v2's detector, built by v2's recipe from the table's guarded data.
  - Positives: the training and drafted wordings about the fact's company and the 24 invented ones, the own plain ones also with the prefixes.
  - Negatives: the same-company questions about every company, the other relations' drafted wordings about 6 invented companies, the real-world questions, and the readings inside the answer.
  - Threshold: the higher of the 99th percentile of those negatives and the highest inside reading.
  - Drafted wordings equal to the bench's own are dropped, and nothing equal to a test prompt is used.
- **P2:** the strict detector's own direction with its threshold at the 95th percentile of its prompt negatives instead of the 99th, still above every inside reading.
- **P3:** the same at the 90th percentile.

**Diagnostic** (`diag_table2.py`, nothing generated with the layer in place), on all 140 development facts (bench40b, bench40c, devpairs and bench40d), with the shared detectors of each fact's own bench:
- **Recall prompts:** every detector's reading at the last token, for the fact and for its company's other fact where there is one; m_whole, as before; and on a two-fact company, m_first of the other fact's first token.
- **Same-company prompts,** with the unedited 16-token reply: the readings at the last prompt token for every option, and at every reply position for the strict detectors. m_first is measured wherever any option would open.

**Estimated pass, as in stage 2:** recall ≥ 0.75, same-company leak ≤ 0.25, and, on a two-fact company, sibling ≤ 1/12.
- **Recall hit:** the fact wins at the last token, isn't vetoed, and m_whole ≤ 2.
- **Same-company leak:** the fact wins and opens at the last prompt token under the option's rule, or at a reply position under the strict rule, with m_first ≤ 2 there.
- **Sibling:** on its recall prompts, the other fact wins and opens at the last token, with m_first ≤ 2.

**Choice:** the option with the most estimated passes over the 140 facts. Ties go to the fewer estimated same-company leaks, then in the order P0, P2, P3, P1.

**Stop rule:** if the best option estimates fewer than 7 more passes than P0 over the 140 facts (2 per 40), the revision stops there. No blind bench is spent, and the frozen table stays stage 2's result.

**Development check** (generation): the revised table with the 140 facts loaded, scored on bench40d and paired against the frozen table's run there (27). Only a bug may change the build, and any change is written here first.

**bench40e (blind).** A blind agent writes it from bench40d's brief, with seed 20261112: 40 facts on 30 companies, 10 of which hold two facts. Its companies' first words are at least 3 edits from every earlier subject, look-alike and invented name of the five benches, and its targets are new. It is opened and built only after the revised build is frozen and committed.

**Scoring on bench40e,** with all 180 facts loaded (bench40b, bench40c, devpairs, bench40d and bench40e): every stage-2 check, unchanged. Four arms: the unedited model, v2 one fact at a time, the frozen table (e4030ad's build) and the revised table.

**Rule, fixed now, on bench40e** (the revised table's passes):
- **30 or more of 40:** stage 2 works at this bar: one layer holds 180 facts above stage 1's level.
- **20 to 29:** the table holds many facts at about stage 1's level, and stage 2 ends there. There is no further revision of the gate on this model.
- **Under 20:** the revision fails, and the frozen table stays stage 2's result.

Also fixed now and reported with it:
- **Paired against the frozen table and against v2:** McNemar on passes, Wilcoxon on recall and same-company leak.
- **The measured cause counts as fixed** only if recall rises against the frozen table (Wilcoxon p < 0.05) while the median same-company leak stays 0.
- **The bars of stage 2:** capitalisation median 0.75, and both facts passing on 7 of the 10 two-fact companies.
- **Cost.**

The smoke stop rule for the first four scored facts applies.

**The diagnostic, and the option fixed** (written after `diag_table2.py`, before the development check; `results/table_rev/diag_table2.json`, `diag_table2_report.txt`, 71 minutes). The recomputed 99th percentile of each strict detector's prompt negatives equals its frozen threshold on every fact and form, so the looser parts sit on the same data.

**Where the frozen table's recall misses come from,** joined with its generated answers. These are the 294 prompts on bench40b, devpairs and bench40d where no part of the answer came out:
- the detector was shut at the last token on 215;
- it opened, but the answer needed more than 2×, on 50;
- it opened with the whole answer at 2× on 26, which the estimate counts as hits;
- a shared detector vetoed on 1, and the company's other fact won on 2.

So a shut detector is the main cause, as measured before. By the estimate, P1 opens 145 of the 294 with the whole answer at 2×, P3 145 and P2 107.

| Estimated passes | bench40b | bench40c | devpairs | bench40d | All 140 | Recall median | Same-company leaks (prompts / facts) | Sibling | Other fact |
|---|---|---|---|---|---|---|---|---|---|
| P0 (frozen) | 25 | 26 | 14 | 26 | 91 | 0.750 | 12 / 12 | 2 | 3 |
| P2 (95th) | 30 | 32 | 17 | 33 | 112 | 0.833 | 25 / 23 | 3 | 3 |
| P3 (90th) | 35 | 34 | 14 | 31 | 114 | 0.917 | 43 / 38 | 5 | 10 |
| **P1 (v2's)** | 35 | 35 | 14 | 32 | **116** | 0.917 | 37 / 32 | 7 | 8 |

- **The estimate is roughly calibrated for P0:** 25, 14 and 26 against 22, 15 and 27 in generation.
- **Fixed by the rule: P1.** It has 116 estimated passes, 25 more than P0, so the stop rule (7) is not met and the revision goes ahead. In `fact_table.py`: `REV_OPTION = "P1"`.
- **The trade, by the estimate:**
  - Against P0, P1 gains 30 passes and loses 5: bench40b f20, bench40d f26 and f32, devpairs f07 and f09.
  - Same-company leaks go from 12 prompts to 37. Only 2 facts go over the limit, both head office (bench40b f20 and bench40d f32), as v2's leaks were.
  - Sibling leaks go from 2 to 7, and other-fact leaks from 3 to 8.
- **The other-fact estimate is pessimistic.** It counts the other fact's first token, while the check needs its whole answer. For the frozen table it counted 3, and generation had none.

Next, the development check on bench40d. The frozen table passed 27 there.

**The development check** (written after it, before bench40e was opened; `results/bench40d/fact_table_rev_teach.jsonl`). The revised table held the same 140 facts.

**31 of 40 pass on bench40d, against 27 for the frozen table.** The estimate gave 32.

| bench40d (development), 140 facts loaded | Passes | Recall median | Same-company leak median | Capitalisation recall median | Look-alike, unrelated, name slots | Known facts lost, drift |
|---|---|---|---|---|---|---|
| Revised table (P1) | **31** | 0.917 | 0 | 0.875 | 0 on every fact | 0, 0.0000 |
| Frozen table | 27 | 0.833 | 0 | 0.875 | 0 | 0, 0.0000 |
| v2, one fact at a time | 27 | 0.917 | 0.125 | — | 0 | 0, 0.0000 |

- **Paired against the frozen table:**
  - Recall is up on 22 facts and down on none (Wilcoxon p = 3.5e-05).
  - The same-company leak is up on 7 and down on none (p = 0.011); its median stays 0.
  - Passes: 7 facts pass only with the revision and 3 only without (McNemar p = 0.34).
- **Paired against v2:** recall is level (up 8, down 10, p = 0.79), and the same-company leak is down on 21 facts, up on none.
- **The looser rule alone** started 80 answers at the prompt's end, where the strict detector stayed shut.
- **The 3 facts lost** are the cost the plan named:
  - **f32 (head office):** the city went into 3 of 8 same-company replies.
  - **f11 and f26:** a two-fact company's head-office fact fired on its company's other question, and the city replaced the other fact's answer: "Q: To whom does Tamviresh Robotics belong? A:" → " Himeji University…", and "Most of the sales of Ovrintash Steel come from its" → " Denizli branch.". That is the other-fact check, limit 0.
- **The 9 failures:** 6 recall, 2 other-fact, 1 same-company.
- **Recall failures include name drift,** which the revision doesn't target. f27 gives "Coumbé" or "Coumbert" for "Coumba Gueye" in both tables, with the same text.
- **Two-fact companies:** both facts pass on 5 of 10 (frozen 4).
- **Cost:**
  - **Size:** 52 KB per fact in float32, with the looser detectors (2 per fact) included. The printed cost line still said 44 KB because it left them out; the line is fixed, and only the line changed.
  - **Generation speed:** 1.2 s for 48 prompts with the layer and without it.
- **No bug was found, so the build is frozen as fixed.** bench40e is next, under the rule above.

## Result of the revision on bench40e (blind)

bench40e was written by a blind agent from bench40d's brief (seed 20261112, `bench40e_source.json`): 40 facts on 30 companies, 10 of them with two facts. It was built and run only after the revision was frozen and committed (f13e9b7). The unedited model gives none of the 40 answers, so no fact is excluded.

**With all 180 facts loaded at once (bench40b, bench40c, devpairs, bench40d and bench40e), the revised table passes 31 of 40. The rule asked for 30 or more, so it is met: stage 2 works at this bar.**

**But the revision is not what met it.** On the same facts the frozen table also passes 31, and v2 one fact at a time passes 34. bench40e was easier than bench40d for every arm: the frozen table went from 27 to 31, and v2 from 27 to 34.

| bench40e (blind) | Passes | Recall median | Same-company leak median | Capitalisation recall median | Failures | Known facts lost, drift |
|---|---|---|---|---|---|---|
| Revised table, 180 facts loaded | **31** | 0.917 | 0 | 0.875 | 5 recall, 3 other-fact, 1 same-company | 0, 0.0000 |
| Frozen table, 180 facts loaded | 31 | 0.833 | 0 | 0.875 | 8 recall, 1 recall and other-fact | 0, 0.0000 |
| v2, one fact at a time | 34 | 0.917 | 0 | — | 4 same-company, 2 recall | 0, 0.0000 |
| Unedited model | 0 | 0 | 0 | — | — | — |

- **Revised against frozen, paired:**
  - Recall is up on 22 facts and down on none (Wilcoxon p = 3.2e-05).
  - The same-company leak is up on 6 and down on 1 (p = 0.053); its median stays 0.
  - So by the criterion fixed beforehand, the measured cause is fixed.
  - Passes don't move: 3 facts pass only with the revision (f01, f09, f36) and 3 only without it (f21, f26, f28) (McNemar p = 1).
- **What the revision lost:** a head-office fact's looser gate fired at the start of a reply to a general question about its company.
  - Example: "How many employees does Lumbravic Brewing have?" → "Asahikawa Lumbravic Brewing Co., …".
  - On a two-fact company this fails the company's other fact on the other-fact check, limit 0: f11, f21 and f28.
  - On f26 it is the head-office fact's own same-company leak (3 of 8).
  - On bench40d the same failure cost f11, f26 and f32. The diagnostic's options all loosened every relation alike, and the cost fell on cities.
- **Revised against v2, paired:**
  - Recall is lower: up on 4, down on 13 (p = 0.012).
  - The same-company leak is lower: up on 1, down on 12 (p = 0.003).
  - Passes: 31 against 34, with 3 only revised and 6 only v2 (McNemar p = 0.51).
- **The other bars:**
  - Capitalisation is met, with a median of 0.875 against a bar of 0.75.
  - Both facts pass on 5 of the 10 two-fact companies, against a bar of 7, so that bar is not met.
- **180 facts at once:**
  - No look-alike, unrelated-company or name-slot leaks on any fact.
  - No known facts lost and no drift, in all three arms.
- **Cost:**
  - 52 KB per fact in float32, and building a fact takes 43 s once.
  - Generating 48 prompts took 1.3 s with the layer and 1.4 s without it.

**What it shows:**
- **One inserted layer holds 180 facts at once and passes 31 of 40 on a blind bench,** with no interference between facts and no damage.
- **It sits at about the level of one block per fact on both blind benches:** 27 against 27 on bench40d, and 31 against 34 on bench40e. Neither difference is significant.
- **The revision traded leaks for recall one for one.** It fixed the measured cause and opened a failure of the same size, in head-office gates.
- **No further revision is allowed.** The rule is met, so the one revision is spent. Both tables stand at 31: the frozen one leaks less, the revised one recalls more.

## Updating facts the model already knows: a CounterFact pilot (written 2026-11-08, before any of it ran)

**The question.** Every bench so far taught facts about invented companies, which the model knew nothing about. Can GRAFT change a fact the unedited model already gives, with no training?

**The data.** CounterFact (Meng et al., 2022) has 21,919 requested rewrites of real facts. Each comes with 2 reworded prompts about the same subject and 10 neighbour prompts about other subjects whose true answer is the old one. A screen of every rewrite prompt (greedy, 8 tokens; `scripts/counterfact_screen.py`) found that the unedited Qwen3.5-0.8B gives the old answer on 4,070 of them.

**The pilot:**
- **Two relations GRAFT already has wordings for:** headquarters (P159, as `hq`) and owned by (P127, as `owner`). No new relation or subject kind is needed, so the pilot isolates one question: overriding old knowledge.
- **20 edits of each,** drawn with seed 20261108 from the records where:
  - the unedited model gives the old answer;
  - neither the old nor the new answer appears in the subject's name ("Ferrari F40 → Ferrari" is copying, not knowledge);
  - the subject appears in no known-facts or real-world prompt;
  - the subjects are distinct, and none contains another as a whole word.
- **All 40 in one table:** GRAFT v1 as packaged (layer 20, push 2.0, margin 0.5, no looser start).
- **Each edit's own training wording is its rewrite prompt,** the edit request, as in CounterFact's protocol. The reworded and neighbour prompts are test prompts, guarded out of every build.
- **One change, behind an option, fixed now:** the real-world negatives leave out prompts of the fact's own relation (`real_world="other_relations"`). In v1 they include "Toyota is headquartered in" for a head-office fact, which teaches the detector to stay shut on real companies; here every subject is real. v1's default is unchanged.

**The measures,** with the 40-edit table attached:
- **Probability, CounterFact's standard:** the new answer's mean token log-probability against the old answer's.
  - Efficacy (ES): new above old on the rewrite prompt.
  - Paraphrase (PS): the same on each reworded prompt.
  - Neighbourhood (NS): old above new on each neighbour prompt.
  - Multi-token answers are forced token by token, so the layer continues an answer as it does in generation.
- **Generation** (greedy, 16 tokens): the new answer in the reply to the rewrite prompt and to each reworded prompt. On neighbour prompts, a leak is the new answer where the unedited reply lacked it.
- **Same subject:** 12 other questions about the subject (6 plain, 6 chat), from bench40e's held-out same-company questions, without the edit's own relation. A leak is the new answer where the unedited reply lacked it.
- **Damage:** known facts lost (of 133) and drift on ordinary text, once for the table.

**The rule:**
- **Works:** mean ES ≥ 0.95, PS ≥ 0.80 and NS ≥ 0.95; generation paraphrase ≥ 0.60; known facts lost ≤ 2; drift ≤ 0.05. The next step is then breadth: subject kinds and new relations, on a fresh draw.
- **Fails:** PS < 0.50, or generation paraphrase < 0.40. GRAFT as built can't override known facts. The next step is mechanism work aimed at the measured cause, for example lowering the old answer as well as pushing the new one, not more relations.
- **In between:** mechanism work on these 40, which then count as development, and a fresh draw to confirm.
- The reworded and neighbour prompts of these relations were not read before the run. The drafted wordings were written on 2026-11-07, before CounterFact was downloaded.

**Deviation, written before the run:** 73 of the 79 owned-by facts the model knows are copies, where the owner is part of the subject's name ("BBC Red Button → BBC"). That leaves 6. The pilot takes 40 headquarters edits instead, drawn the same way from 113 usable ones. The question, the measures and the rule are unchanged.

## Result of the CounterFact pilot

40 headquarters facts the unedited Qwen3.5-0.8B knows, all rewritten in one GRAFT table. Raw rows: `research/results/counterfact/pilot/`.

| Measure | Unedited | With the 40 edits |
|---|---|---|
| Efficacy (ES), new above old on the rewrite prompt | 0.000 | **1.000** |
| Paraphrase (PS), new above old on reworded prompts | 0.013 | **0.988** |
| Neighbourhood (NS), old above new on other subjects | 0.860 | 0.848 |
| Says the new answer: rewrite prompt / reworded prompts | 0.000 / 0.000 | **1.000 / 0.975** |
| Neighbour replies newly giving the new answer | — | 0.013 (5 of 400) |
| Same-subject questions newly giving the new answer | — | 0.037 (18 of 480) |
| Known facts lost; drift | — | 0 of 133; 0.0000 |

**Verdict by the rule as written: in between,** because NS is 0.848 against a bar of 0.95.

**The NS bar was set wrong.** The unedited model itself scores 0.860 on the same neighbour prompts: for 14% of them it already prefers the new answer, because it doesn't know those neighbours' facts. No edit, and no lack of one, could have met 0.95. Against the unedited model:
- **NS is unchanged on 39 of 40 edits.**
- **The one change is Vanderbilt University (1.0 → 0.5).** 5 of its neighbour prompts are about "Vanderbilt University Department of …", a longer name that contains the stored one. The name index matches it, and the reply says "Toronto".

**The other failures:**
- **Reworded prompts missed: 2 of 80.** One reply said "Bedfordshire" for "Bedford", and one kept the old "Moscow" (Gerasimov Institute of Cinematography).
- **Same-subject leaks:** 8 of the 18 come from one question, "Most of the customers of X are", where a city fits ("…from Indianapolis"). Head offices leak into city-shaped slots, as on the fictional benches.

**What it shows:** GRAFT overrides facts the model already knows, on reworded prompts, with no measurable damage elsewhere, at least for one relation. **What it doesn't show:** other relations and subject kinds, and a pre-registered pass. A fresh draw with a corrected neighbourhood bar, written before it runs, is the confirmation.

## CounterFact confirmation: a fresh draw (written 2026-11-08, before it ran)

**What changes:** only the draw and the neighbourhood bar. The build, the option and the measures are the pilot's. No mechanism change: the pilot's two small failure kinds (longer names that contain the stored one, and city-shaped slots) stay, and are counted.

**The draw:** 40 headquarters edits from the 73 usable ones the pilot didn't use, with seed 20261109.

**The rule:**
- **Works:** mean ES ≥ 0.95 and PS ≥ 0.80; generation reworded ≥ 0.60; NS at most 0.05 below the unedited model's NS on the same prompts; neighbour replies newly giving the new answer ≤ 0.05; same-subject leaks ≤ 0.10; known facts lost ≤ 2; drift ≤ 0.05.
- **Fails:** PS < 0.50, or generation reworded < 0.40.
- **Anything else is in between,** and the failing measure is named.

## CounterFact breadth: beyond companies (written 2026-11-08, before it ran)

**The question.** The pilot changed known facts about organisations' head offices. Does GRAFT change known facts about people, countries, places, creative works and products, with relations it has never had?

**The edits:** six CounterFact relations, 12 each, drawn with seed 20261110 under the pilot's exclusions, all 72 in one table.

| Relation | CounterFact | Subject kind | Example |
|---|---|---|---|
| mother_tongue | P103 | person | "Louis-Paul Aujoulat is a native speaker of" French → Dutch |
| citizenship | P27 | person | "Priyanka Vadra has a citizenship from" India → Japan |
| official_language | P37 | country | "In Luhansk Oblast, an official language is" Ukrainian → Russian |
| located_in_country | P17 | place | "Manila Bay, in" Philippines → Italy |
| original_language | P364 | work | "The original language of Moscow Strikes Back was" Russian → English |
| developer | P178 | product | "Mac OS was created by" Apple → Microsoft |

**The wordings:**
- An agent with no access to this repository, the dataset or the web drafted 30 plain and 30 chat wordings for each relation, plus each subject kind's invented names and generic questions.
- A second such agent, which never saw the first one's output, wrote held-out questions per kind for the same-subject leak check.

**Changes for subjects that aren't companies, fixed now:**
- Each subject kind has its own invented filler names, generic negative questions and default same-subject questions.
- A shared detector can veto a fact only if its relation is about the same kind of subject. An official-language question never vetoes a person's mother tongue.
- Relations about companies are unchanged. v1's exact reproduction is checked again before this run.

**The measures** are the pilot's. Same-subject questions are 6 plain and 6 chat from the held-out set of the subject's kind.

**The rule** is the confirmation's, on all 72 edits: works, fails, or in between with the failing measures named. Each relation is also judged by the same bars on its 12 edits, with the table's known facts and drift. GRAFT is said to work for a relation only where that relation meets them.

## Result of the CounterFact confirmation

40 fresh headquarters edits, none of them in the pilot. Raw rows: `research/results/counterfact/confirm/`.

| Measure | Unedited | With the 40 edits | Bar |
|---|---|---|---|
| ES | 0.000 | **1.000** | ≥ 0.95 |
| PS | 0.013 | **0.963** | ≥ 0.80 |
| NS | 0.845 | 0.828 | at most 0.05 below unedited: met (0.017) |
| Says the new answer: rewrite / reworded | 0.025 / 0.000 | **1.000 / 0.963** | reworded ≥ 0.60 |
| Neighbour replies newly giving the new answer | — | 0.018 | ≤ 0.05 |
| Same-subject leaks | — | 0.040 (19 of 480) | ≤ 0.10 |
| Known facts lost; drift | — | 0; 0.0000 | ≤ 2; ≤ 0.05 |

**Verdict: works.** GRAFT changes headquarters facts the model already knows, and the confirmation meets every bar written before it ran.

**The failures:**
- **Reworded prompts:** 3 edits miss one of their two (Aviation Industry Corporation of China, Sporveien, Sciences Po).
- **Neighbourhood:** NS fell on one edit only, Case Western Reserve University (0.8 → 0.1). 7 of its 10 neighbour prompts name a longer entity that contains the stored name, as Vanderbilt's did in the pilot. Across the 80 edits of both draws, that one cause accounts for all of the neighbourhood damage.

**Checks of the drafted text, before the run:** every relation has 30 plain and 30 chat wordings; every kind has 40 invented names and 24 plain and 24 chat generic questions; no drafted wording equals a reworded test prompt; no invented name equals a chosen subject. 5 held-out questions (2 person, 1 work, 2 product) equal a drafted generic one, so they are dropped from the held-out set before the run.

## Result of the CounterFact breadth run

72 edits, six relations, 12 each, all in one table. Raw rows: `research/results/counterfact/breadth/`.

| Relation | ES | PS | Says new, reworded | NS (unedited) | Same-subject leaks | Verdict |
|---|---|---|---|---|---|---|
| citizenship | 0.750 | 0.750 | 0.750 | 0.950 (0.950) | 0.014 | in between: ES, PS |
| developer | 0.667 | 0.583 | 0.625 | 0.808 (0.825) | 0.007 | in between: ES, PS |
| official_language | 0.833 | 0.333 | 0.417 | 0.825 (0.825) | 0.097 | fails |
| mother_tongue | 0.583 | 0.208 | 0.167 | 0.992 (0.992) | 0.014 | fails |
| located_in_country | 0.083 | 0.083 | 0.042 | 0.858 (0.858) | 0.021 | fails |
| original_language | 0.000 | 0.042 | 0.083 | 0.750 (0.750) | 0.097 | fails |
| **All 72** | **0.486** | **0.333** | **0.347** | 0.864 (0.867) | 0.042 | **fails** |

Known facts lost: 0 of 133. Drift: 0.0000. Neighbour replies newly giving the new answer: 0.003.

**Verdict: fails.** PS is 0.333 against the failure bar of 0.50, and generation reworded is 0.347 against 0.40. GRAFT as built doesn't override known facts for these relations. It stays safe (no neighbourhood change, no damage) because it mostly doesn't act: the layer opened 293 times over this run's scoring, against 392 for the 40 headquarters edits.

**First look at the cause, from the built parts:** the plain detectors barely clear their own threshold. The median margin between the positives' median reading and the threshold is 2.6 to 5.4 for these relations, and −0.2 for located_in_country, where 6 of 12 facts got no plain detector at all. For headquarters it was 16.1. The chat detectors are healthy (10 to 17). CounterFact's prompts are all plain. The suspected cause is near-synonymous negatives: each fact's detector is fitted against the other relations' wordings, and this table has three language relations and three about places or countries. A diagnostic measures it before any change. These 72 edits now count as development.

## Breadth diagnostic, and the revision (written 2026-11-08, before the revision ran)

**Measured cause** (`scripts/diag_counterfact.py`, 2 facts per relation; `research/results/counterfact/breadth/diagnostic.json`). A plain detector's threshold is the 99th percentile of its prompt negatives, so the top 1% of negatives sets it. For every breadth fact, that top 1% is almost all other relations' wordings about a **different kind of subject**, filled with names of the fact's kind:

| Fact | Negatives above its threshold, by source |
|---|---|
| mother_tongue | official_language 62–64, original_language 7–12 |
| citizenship | located_in_country 70–75 |
| official_language | mother_tongue 52–62, original_language 10–14 |
| located_in_country | citizenship 56–69, owner 3–5 |
| original_language | mother_tongue 19–38, official_language 25–39 |
| developer | owner 44, founder 19–20 |

- Over the 12 facts, the detector opens on 13 of their own 36 rewrite and reworded prompts.
- For headquarters (pilot, 3 facts), the top 1% are same-subject and generic questions, as designed. The median margin there is 15.4 against 3.7, and 9 of 9 own prompts open.
- Such negatives can't occur in use: no one asks a person's official language or a film's mother tongue. Filled with a person's name, though, "The official language of X is" sits next to "The mother tongue of X is", and caps the threshold.

**The revision:** a fact's detector, and each shared detector, takes other relations' wordings as negatives only from relations about the same kind of subject. This is the same rule as the veto. Tables of company relations alone are unchanged, and v1's exact reproduction is checked again.

**The plan:**
1. **Development check:** rebuild the 72 breadth edits with the revision and score them, paired against the failed run. This does not count as evidence.
2. **Confirmation, if the development check reaches PS ≥ 0.80 pooled:** a fresh draw of 12 per relation (seed 20261111), leaving out every edit used so far, judged by the breadth rule, pooled and per relation.
3. **If the development check stays under PS 0.80:** no fresh draw. The next cause is measured first.

## Result of the revision on the development edits

The same 72 edits, rebuilt with negatives from relations about the same kind of subject only. They are development, so this is not evidence. Raw rows: `research/results/counterfact/breadth_rev/`.

| Relation | PS before → after | Says new, reworded | ES | Same-subject leaks | Verdict |
|---|---|---|---|---|---|
| citizenship | 0.750 → **0.958** | 0.958 | 1.000 | 0.083 | works |
| developer | 0.583 → **0.958** | 0.958 | 0.917 | 0.097 | in between: ES |
| mother_tongue | 0.208 → 0.708 | 0.583 | 0.833 | 0.118 | in between |
| official_language | 0.333 → 0.667 | 0.625 | 0.917 | 0.049 | in between |
| original_language | 0.042 → 0.667 | 0.708 | 0.917 | 0.139 | in between |
| located_in_country | 0.083 → 0.458 | 0.542 | 0.583 | 0.194 | fails |
| **All 72** | 0.333 → **0.736** | 0.729 | 0.861 | 0.113 | **in between: ES, PS, same-subject leak** |

- **Unchanged:** NS 0.863 against 0.867 unedited, neighbour leaks 0.003, 0 known facts lost, no drift.
- **The layer opened 628 times,** against 293 before the revision. Building a fact took 25 s instead of 54 s, because the negatives shrank.
- **By the plan written before it ran, there is no fresh draw:** the pooled PS is 0.736, under the 0.80 that a confirmation needed. The next cause is measured first.
- **Same-subject leaks rose from 0.042 to 0.113,** above the bar of 0.10. The detectors open more often, including on some other questions about the subject.

## Diagnostic of the revision, and the next one (written 2026-11-08, before it ran)

**Measured** (`scripts/diag_counterfact.py`, 4 facts per relation; `research/results/counterfact/breadth_rev/diagnostic.json`):
- **The detectors recovered:** the median margin is 11.7 (it was 3.7), and the detector opens on 58 of its facts' own 72 rewrite and reworded prompts (it was 13 of 36).
- **Whenever the detector opens, the new answer wins: 58 of 58.** The push is strong enough even against a known fact. Every remaining miss is the detector staying shut.
- **Where it stays shut:** 10 of the 11 shut reworded prompts start with an unrelated sentence, as CounterFact's do: "He started out painting movie advertisements on billboards. Jeanne Moreau, a native". One of them has no question left at all. The detectors were taught behind four short lead-ins only ("Note: " and the like). Three of the facts' own rewrite prompts read just under their threshold (36.0 against 37.3, 34.0 against 36.2, 17.9 against 21.1).
- **Same-subject leaks:** 98 of 864 replies, 69 of them to chat questions, and 68 of the 98 start the reply with the new answer ("French actor **Aleksey Khomyakov**…" to "What colour is Aleksey Khomyakov's hair?"). The detectors open at the reply's first token on generic questions unlike the 24 per kind they were taught against.

**Revision 2:**
- **Context before the question:** each plain positive about the subject, and each plain negative about the subject, is also asked behind 4 sentences of ordinary text from the bundled WikiText-103 calibration paragraphs, chosen with a fixed seed. These replace nothing; the four short lead-ins stay.
- **More generic questions per kind:** a new blind draft of 36 more plain and 36 more chat generic questions for each kind, by an agent that sees neither the held-out questions nor the earlier draft. Any that equals a held-out question is dropped.
- **v1 unchanged:** the context sentences are a build option (`context=4`), off by default, so v1's builds and their reproduction are unchanged; this run turns it on for every fact. The extra generic questions belong to the non-company kinds, so company facts don't see them.

**The plan:** the development check on the same 72 edits, then, only if it reaches PS ≥ 0.80 pooled with same-subject leaks ≤ 0.10, a fresh confirmation draw (seed 20261111) judged by the breadth rule.

## Result of revision 2 on the development edits

The same 72 edits, with 4 sentences of ordinary text before the subject's plain prompts and 54 to 60 generic questions per kind and form (it was 24). Raw rows: `research/results/counterfact/breadth_rev2/`.

| Relation | PS: first run → rev. 1 → rev. 2 | Says new, reworded | ES | Same-subject leaks | Verdict |
|---|---|---|---|---|---|
| citizenship | 0.750 → 0.958 → **1.000** | 1.000 | 1.000 | 0.049 | works |
| developer | 0.583 → 0.958 → **1.000** | 1.000 | 0.917 | 0.007 | in between: ES |
| mother_tongue | 0.208 → 0.708 → 0.750 | 0.583 | 1.000 | 0.076 | in between |
| original_language | 0.042 → 0.667 → 0.708 | 0.750 | 0.917 | 0.097 | in between |
| official_language | 0.333 → 0.667 → 0.667 | 0.625 | 0.917 | 0.069 | in between |
| located_in_country | 0.083 → 0.458 → 0.583 | 0.625 | 0.750 | 0.194 | in between |
| **All 72** | 0.333 → 0.736 → **0.785** | 0.764 | 0.917 | **0.082** | **in between: ES, PS** |

- **Unchanged:** NS 0.863 against 0.867 unedited, neighbour leaks 0.004, 0 known facts lost, no drift.
- **The gate for a fresh draw is missed by 0.015:** PS 0.785 against 0.80. Same-subject leaks are back under their bar, at 0.082.
- **So there is no confirmation yet.** These 72 edits have now served three rounds of development. Another round on them risks fitting them rather than the relations; the next step is a decision, recorded below when taken.

## Revision 3, a fresh development draw, and the confirmation (written 2026-11-08, before any of it ran)

**Revision 2's diagnostic** (`research/results/counterfact/breadth_rev2/diagnostic.json`, 6 facts per relation):
- **Opened means won:** the detector opens on 91 of 108 own prompts, and the new answer wins on all 91. 17 stay shut.
- **Phrasings the drafted wordings don't cover:** an indirect official-language phrasing reads 6 to 10 against thresholds of 23 to 34, and a terse mother-tongue fragment reads 5.5 against 18.9.
- **Near misses:** two facts' own rewrite prompts read just under their threshold.
- **Real-world questions about other kinds of subject now set thresholds.** For developer, 11 to 12 of the 32 real-world questions sit above the threshold ("The CEO of Apple is" and the like). For located_in_country it is 5 to 13 head-office questions. Revision 1 kept drafted wordings within a kind; these are the same case for the real-world set.

**A brief withdrawn:** the first draft request for more wordings named example phrasings taken from the failures above. CounterFact reuses its paraphrase phrasings across records, so that would have leaked into the confirmation. The agent was stopped before it wrote anything, and a new one was briefed with general guidance on variety only.

**Revision 3:**
- `real_world="same_kind"` also leaves out real-world questions about another kind of subject.
- 30 more plain and 30 more chat wordings per relation, from the new blind draft, added to the first 30 and 30.
- Everything else as revision 2: `context=4` and the extra generic questions.
- The **fast build** (`ReadCache`) is used from here on, after its check against revision 2's saved parts.

**Development check 2:** a fresh draw of 12 per relation (seed 20261111), leaving out the 72 edits used so far. Revision 2 and revision 3 both run on it, so the comparison is paired on edits no revision was chosen on.

**Confirmation:** a further fresh draw (seed 20261112), leaving out every earlier edit, judged by the breadth rule, pooled and per relation.
- It runs revision 3 if revision 3 reaches PS ≥ 0.80 with same-subject leaks ≤ 0.10 on development check 2; otherwise revision 2 if revision 2 does; otherwise there is no confirmation.
- **Phrasing novelty:** CounterFact builds reworded prompts from a fixed set of phrasings per relation, so a fresh record can repeat a phrasing seen during development. The confirmation therefore also reports PS separately on reworded prompts whose phrasing appeared among the development edits and on those whose phrasing did not.

**The fast build, checked before use** (`scripts/check_fast_build.py`; `research/results/counterfact/fast_check/`). 18 facts of revision 2 (3 per relation) were rebuilt with the `ReadCache` and compared with their saved parts, built without it:
- **Detector directions:** cosine 0.9975 at worst, 0.9996 median.
- **Thresholds:** up to 4% apart, more than rounding. The likely source is the 16-token replies: generated in other batches, a reply can differ by a token, and replies shape the direction.
- **Decisions:** on all 54 of these facts' own rewrite and reworded prompts, the detector opens or stays shut exactly as before.
- **Speed:** the median build is 7.9 s against 26.8 s. The first fact of each relation still pays for the shared readings (about 23 s); later ones take 5 to 9 s.
- Both revisions on development check 2 use the fast build, so their comparison is like for like.

**Deviation, written before revision 3 ran again:** its first attempt ran out of GPU memory on edit 33 of 72. The answer vectors read the logits of every teaching example at once (about 13 GB with 30 + 30 drafted wordings), and 60 + 60 needs more than the 24 GB card. Revision 3 now teaches the answer vectors from the first 30 + 30 drafted wordings, as revision 2 did (`teach_cap=30`), while its detectors use all 60 + 60. The answer vectors weren't the measured problem: an open detector's new answer won 58 of 58 times, and 91 of 91.

## Result of development check 2

72 fresh edits (seed 20261111), none used before. Raw rows: `research/results/counterfact/dev2_rev2/` and `dev2_rev3/`. Both runs used the fast build.

| Relation | PS, revision 2 | PS, revision 3 | ES, rev. 2 / rev. 3 | Same-subject leaks, rev. 2 / rev. 3 |
|---|---|---|---|---|
| citizenship | 1.000 | 1.000 | 1.000 / 0.917 | 0.069 / 0.049 |
| developer | 0.958 | 1.000 | 1.000 / 1.000 | 0.014 / 0.014 |
| original_language | 0.833 | 0.833 | 0.917 / 0.833 | 0.069 / 0.083 |
| located_in_country | 0.750 | 0.792 | 1.000 / 0.917 | 0.250 / 0.250 |
| official_language | 0.625 | 0.542 | 1.000 / 0.833 | 0.028 / 0.035 |
| mother_tongue | 0.500 | 0.500 | 1.000 / 0.833 | 0.097 / 0.076 |
| **All 72** | **0.778** | **0.778** | 0.986 / 0.889 | 0.088 / 0.085 |

- **Unchanged in both:** NS equals the unedited model's (0.881), no neighbour leaks, 0 known facts lost, no drift.
- **Revision 2 holds on unseen edits:** 0.778 here, against 0.785 on the 72 it was chosen on.
- **Revision 3 adds nothing:** the same pooled PS, and a lower ES. More wordings and same-kind real-world negatives don't move the language relations.
- **By the plan written before this ran, there is no confirmation:** neither revision reaches PS 0.80 pooled.

## A narrower confirmation: citizenship and developer (written 2026-11-08, after development check 2, before it ran)

**Why these two:** citizenship and developer met the breadth rule's bars, or missed only ES, on both development draws (revision 2: citizenship works on both; developer works on check 2). They were chosen after seeing development results, so only a fresh draw can confirm them.

**The run:**
- 24 fresh edits of each relation (seed 20261112), leaving out all 144 edits used so far.
- Revision 2's build (`context=4`, the extra generic questions, `real_world="other_relations"`), with the fast build.
- The other four relations stay in the table's wordings, as in every breadth run, so competition and negatives are unchanged.

**The rule:** the breadth rule, judged on each relation's 24 edits: works if ES ≥ 0.95, PS ≥ 0.80, generation reworded ≥ 0.60, NS at most 0.05 below the unedited model, neighbour leaks ≤ 0.05, same-subject leaks ≤ 0.10, known facts lost ≤ 2 and drift ≤ 0.05. PS is also reported split by phrasings seen during development and phrasings not seen.

## Result of the narrower confirmation

48 fresh edits (seed 20261112), 24 per relation, none used before. Raw rows and the phrasing split: `research/results/counterfact/confirm2/`.

| | Citizenship | Developer |
|---|---|---|
| ES | **1.000** | 0.917 |
| PS | **1.000** | 0.875 |
| Says the new answer: rewrite / reworded | 1.000 / 1.000 | 0.875 / 0.875 |
| NS (unedited) | 0.896 (0.896) | 0.887 (0.887) |
| Neighbour leaks; same-subject leaks | 0.000; 0.059 | 0.000; 0.014 |
| Known facts lost; drift (the table) | 0; 0.0000 | 0; 0.0000 |
| **Verdict** | **works** | in between: ES |

- **Citizenship is confirmed:** every bar met on fresh edits, with the neighbourhood exactly as the unedited model had it.
- **Developer misses only ES,** at 0.917 against 0.95: 2 of its 24 rewrite prompts stay with the old company.
- **Phrasing novelty:** 89 of the 96 reworded prompts repeat a phrasing that appeared among the development edits; PS is 0.944 on those and 0.857 on the 7 new ones. For citizenship it is 43 of 43 and 5 of 5. The new phrasings are too few to say more. None of the development phrasings was taught: the wordings were drafted blind, and the revisions were general rules.

## The weaker relations: a lower threshold where the reply starts, and answers of several tokens (written 2026-11-08, before any of it ran)

**Measured from stored results first** (development check 2 and the confirmation, no new run):
- **Where PS is lost.** On development check 2, 11 of mother tongue's 12 missed prompts are its two shortest CounterFact phrasings ("X, speaker of" and "X, a native"). 7 of official language's 9 are "In X, they understand"; all 6 of located in country's are "X, in". 5 of original language's 6 are "The language of X was", and 4 of those 5 have answers spelled with several tokens.
- **Answers spelled with several tokens fare worse in every run.** PS is 0.59 against 0.92 for one-token answers in revision 2's run, 0.71 against 0.80 on development check 2, and 0.82 against 0.97 in the confirmation. Both of developer's missed rewrite prompts in the confirmation are such answers.
- **The cause, from the code.** The LDA weighs each group of negatives' mean equally. A fact whose taught answer has a later token gets a third group: the readings inside its taught answers. That group takes a third of the weight away from the prompt negatives. These facts' plain thresholds sit at 28 to 45, against 11 to 24 for one-token answers, with about half the margin between the positives' median and the threshold.
- **The proposed lower threshold can't reach most misses.** In revision 2's diagnostic, 5 of the 17 shut prompts read 5 to 10 against thresholds of 19 to 34. The other 12 sit within a few points of their threshold, mostly on facts with answers of several tokens.
- **A gate on "about to say the old answer" was checked and dropped** (`scripts/diag_old_answer.py`, `research/results/counterfact/dev2_rev2/old_answer_lens.json`). Through the logit lens at layer 20, the old answer is the top token on 0 of 36 mother-tongue prompts. On the missed prompts it ranks 14 to 148,154, and the same-subject questions overlap that range. Even the model's own output puts it first on only 12 of 36: these phrasings don't lead straight into the answer.
- **Same-subject leaks, read by hand.** Of development check 2's 76 leaks, 39 open where the reply starts (to "What colour is X's hair?": "**Russian** actor X…"). 30 insert the answer mid-reply, sometimes garbling it ("Best Newcom Swedish Opera"). 7 replace an old answer that the unedited reply stated ("based in the Netherlands" → "based in Russia"). The measure counts all three as leaks and stays the rule. Located in country has 36 of them; many of its mid-reply ones carry the edit ("The station is located in Finland").

**Two changes, both build options, off by default** (v1 and its reproduction check unchanged):
1. **`inside="scatter"`:** the readings inside taught answers enter the LDA's within-group scatter but not the mean of the negatives. The threshold still includes their maximum, so a detector stays shut inside an answer.
2. **`end_quantile`:** at the prompt's last token only, a fact may also open on its own direction at this quantile of its prompt negatives. The maximum inside answers is left out there, since the prompt's last token is never inside an answer. Everywhere else the strict threshold holds. The layer already has this mechanism; the fact table's revision used it.

**The diagnostic, before any run** (`scripts/diag_counterfact_options.py`, on development check 2's 72 edits, which are development). Each fact's detectors are rebuilt both ways. Its answer vectors are reused from revision 2's run, so only the detectors differ. Six options are tried with all 72 facts in one layer, shared detectors and veto included:

| Option | Inside readings | Prompt's last token |
|---|---|---|
| O0, as now | negatives | strict threshold |
| O1 | negatives | 95th percentile |
| O2 | negatives | 90th percentile |
| O3 | scatter | 99th percentile, without the inside maximum |
| O4 | scatter | 95th percentile |
| O5 | scatter | 90th percentile |

For each option and relation, two things are counted at the prompt's last token:
- the share of the facts' own rewrite and reworded prompts where the layer pushes the fact's answer (estimated ES and PS; an open detector's answer won 149 of 149 times);
- the share of their held-out same-subject questions where it pushes any answer (estimated leaks where the reply starts).

**Choice, fixed now:**
- The option with the highest estimated PS over all 72 edits, among those whose estimated reply-start leaks are at most 0.03 above O0's in every relation. Ties go to the option with fewer changes.
- **Kill rule:** if the best option's estimated pooled PS is under O0's + 0.05, stop. There is then no development run and no confirmation; the four relations stay "in between", and the work moves on to the comparison with published editors.

**Then:**
1. **The development run:** the chosen option on development check 2's edits, paired against revision 2's run. Each relation that meets the breadth rule's ES, PS and same-subject-leak bars there goes on to the confirmation.
2. **The confirmation:** 24 fresh edits per such relation (seed 20261113), leaving out every edit used so far. It is judged by the breadth rule per relation, with PS also split by phrasings seen during development and not seen.

## Result of the options diagnostic: stop

Development check 2's 72 edits, all in one layer, counted at the prompt's last token. Raw numbers: `research/results/counterfact/options_dev2/`. Before any option, the v1 reproduction check passed again: 180 facts, every reply identical.

**The estimate tracks the real scoring.** O0, revision 2 rebuilt, estimates PS at 0.764 where the real run measured 0.778, and matches each relation within a prompt or two (mother tongue 0.50 against 0.500, located in country 0.75 against 0.750).

| Option | Estimated ES | Estimated PS | Same-subject questions opened | Within the leak allowance |
|---|---|---|---|---|
| O0, as now | 0.986 | 0.764 | 0.056 | — |
| O1, 95th percentile at the prompt's end | 1.000 | 0.826 | 0.144 | no: 5 relations over |
| O2, 90th percentile | 1.000 | 0.847 | 0.201 | no: 6 over |
| **O3, inside readings in the scatter, 99th percentile** | **1.000** | **0.806** | **0.057** | **yes** |
| O4, scatter, 95th percentile | 1.000 | 0.840 | 0.138 | no: 5 over |
| O5, scatter, 90th percentile | 1.000 | 0.847 | 0.200 | no: 6 over |

Estimated PS, and same-subject questions opened, by relation:

| Relation | O0 | O1 | O3 | O5 |
|---|---|---|---|---|
| citizenship | 1.00 / 0.05 | 1.00 / 0.15 | 1.00 / 0.05 | 1.00 / 0.24 |
| developer | 0.92 / 0.00 | 1.00 / 0.06 | 0.92 / 0.00 | 1.00 / 0.08 |
| original language | 0.79 / 0.06 | 0.96 / 0.19 | **0.96 / 0.05** | 1.00 / 0.32 |
| located in country | 0.75 / 0.09 | 0.79 / 0.18 | 0.79 / 0.10 | 0.88 / 0.20 |
| official language | 0.62 / 0.03 | 0.67 / 0.06 | 0.67 / 0.04 | 0.67 / 0.08 |
| mother tongue | 0.50 / 0.10 | 0.54 / 0.23 | 0.50 / 0.10 | 0.54 / 0.28 |

- **The lower threshold, as proposed, doesn't work.** At the 95th or 90th percentile it raises PS by 0.06 to 0.08, and opens on 2.5 to 3.6 times as many other questions about the subject.
- **The scatter option costs nothing in leaks:** same-subject openings 0.057 against 0.056. For an answer of several tokens, the prompt negatives' 99th percentile falls back to the one-token level ("Dutch": 19.4 against a median of 31.6, where it was 27.7 against 35.6). Original language rises from 0.79 to 0.96, and every rewrite prompt opens (ES 1.000).
- **Mother tongue and official language don't move under any option.** Even the 90th percentile leaves them at 0.54 and 0.67. Their misses are the terse phrasings, which read far below any threshold the leak bar allows.
- **By the rule, stop.** The best option within the leak allowance is O3, with an estimated gain of +0.042, 6 more reworded prompts of 144, against the +0.05 required. There is no development run and no confirmation. Both options stay in the package, off by default and measured but not adopted, like revision 3's. The four relations stay "in between".

## The full CounterFact mix on Qwen3.5-0.8B (written 2026-11-09, before any of it ran)

**Why.** GRAFT has been measured on 7 of CounterFact's 34 relations. This run measures it on every relation the model knows enough facts for, all in one table, so that the remaining problems can be ranked by size. The work stays on the Qwen family for now (the user's direction). It is a measurement, not a revision: nothing is tuned on this draw. Any fix chosen from it is confirmed on a fresh draw.

**Relations.** CounterFact edits only facts the unedited model gives. After `select`'s filters, leaving out the 272 edits used so far:
- **In, 25 relations, each with at least 16 usable records:**
  - the 7 measured before: headquarters, citizenship, developer, mother tongue, official language, located in country, original language;
  - 18 new ones: religion (P140), languages spoken (P1412), position played (P413), continent (P30), sport (P641), work location (P937), administrative territory (P131), capital (P36), language of work (P407), field of work (P101), manufacturer (P176), position held (P39), event location (P276), place of death (P20), occupation (P106), place of birth (P19), original network (P449), country of origin (P495).
- **Out, 9 relations:**
  - the model knows too few of their facts: instrument 8 usable, employer 7, member of 4, genre 3, location of formation 1, twin city and record label 0;
  - named after (3 usable): the answer is mostly part of the name;
  - owned by: 73 of its 79 known facts copy the subject's name.

**Subject kinds:**
- **person**, 12 relations: citizenship, mother tongue, and the 10 new person relations.
- **place:** located in country, continent, administrative territory.
- **country:** official language, capital.
- **work:** original language, language of work, original network, country of origin.
- **product:** developer, manufacturer.
- **company:** headquarters.
- **event**, a new kind: event location (festivals, tournaments, clubs, universities).

**Expected collisions, reported per pair.** Languages spoken against mother tongue. Place of death, place of birth and work location (all cities of a person). Language of work against original language. Manufacturer against developer. Within a kind, each is the other's negative and can veto it.

**Wordings, all drafted blind** (no files, no web, no dataset), with the same brief as the first draft:
- 30 plain and 30 chat wordings for each new relation, from three agents.
- The new kind's invented names and generic questions, from a fourth.
- Its held-out questions, from a fifth that sees nothing else.
- The existing generic and held-out questions were written when 10 relations existed. A sixth agent, which drafts nothing, marks those that ask for any of the 25 relations. Marked generic questions are removed within their kind, and marked held-out questions are removed.
- Headquarters keeps the pilot's held-out company questions (bench40e's), without its own relation's.

**The draw:** 12 edits per relation (seed 20261113), leaving out all 272 earlier edits, so 300 edits in one table.

**The build:** revision 2's settings (`context=4`, `real_world="other_relations"`, the fast build), push 2 and margin 0.5. The options measured but not adopted stay off.

**Reported:**
- Per relation: ES, PS, NS against the unedited model, the generation checks, neighbour and same-subject leaks, and the breadth rule's verdict.
- Pooled: the same measures.
- For the table: known facts lost and drift.
- The published editors' numbers stay as reference only (another model).

**The one check fixed in advance:** headquarters and citizenship, confirmed in smaller tables, keep PS ≥ 0.80 and same-subject leaks ≤ 0.10 here. If they don't, adding relations crowds them, and that is the first problem to fix. There is no other pass or fail: the run ranks problems.

**Deviation, written before the run:** held-out questions that the marking agent says ask for a relation are tagged, not removed. A tagged question is left out only for edits of its own relation, as the headquarters pilot did with its company questions. Removing them all would leave headquarters 4 plain and 4 chat company questions, fewer than the 6 and 6 each edit draws. A question about another relation ("Who owns X?") is a fair test of leaks for the others. Marked generic questions are removed as planned. The agent marked 22 of 706: 12 generic (four of them "the capital of" questions among the country kind's negatives, which would have taught the capital detector to stay shut on its own question) and 10 held-out.

## Result of the full CounterFact mix

300 fresh edits, 25 relations, 12 each, all in one table. Raw rows: `research/results/counterfact/full/`. The first attempt stopped at edit 243: a teaching wording for "Latin" contains "trans**latin**g", and the check that no teaching prompt gives away the answer matched letters, not words. The check now matches whole words. It only raises, so builds that passed it before are unchanged; the 242 parts already built were kept, and the run resumed.

**Pooled:**

| | GRAFT, 300 edits | Unedited |
|---|---|---|
| ES | 0.657 | 0.027 |
| PS | 0.523 | 0.120 |
| NS | 0.802 | 0.803 |
| Says the new answer: rewrite / reworded | 0.667 / 0.507 | — |
| Neighbour leaks; same-subject leaks | 0.000; 0.036 | — |
| Known facts lost; drift | 0; 0.0000 | — |

**The check fixed in advance is met.** Headquarters and citizenship keep ES and PS 1.000, with same-subject leaks 0.049 and 0.035.

**By relation** (PS; ES; verdict by the breadth rule):

| Works | In between | Fails (PS under 0.50) |
|---|---|---|
| citizenship 1.00; 1.00 | occupation 0.88; 0.67 | work location 0.46; 0.92 |
| headquarters 1.00; 1.00 | country of origin 0.75; 1.00 | language of work 0.42; 0.42 |
| original network 0.96; 1.00 | position held 0.75; 0.83 | languages spoken 0.42; 0.92 |
| | event location 0.71; 1.00 | mother tongue 0.42; 0.42 |
| | religion 0.71; 0.92 | continent 0.33; 1.00 |
| | field of work 0.67; 0.75 | located in country 0.33; 0.33 |
| | official language 0.67; 1.00 | sport 0.21; 0.58 |
| | capital 0.54; 0.92 | original language 0.12; 0.08 |
| | place of birth 0.54; 0.83 | administrative territory 0.08; 0.25 |
| | position played 0.54; 0.33 | developer 0.04; 0.00 |
| | place of death 0.50; 0.25 | manufacturer 0.04; 0.00 |

**The largest problem: relations with a near-twin in their kind.** The 12 relations that share their kind with a near-synonymous relation average PS 0.31 and ES 0.45. The 13 others average PS 0.72 and ES 0.85. The plain detectors' median margin (positives' median minus threshold) shows why:

| Relation | Margin, earlier run → this table | Twin added in this table |
|---|---|---|
| developer | 14.1 → 0.7 | manufacturer (margin −1.3; 3 of 12 facts got no plain detector) |
| original language | 13.6 → 0.6 | language of work (2.0) |
| located in country | 16.5 → 5.1 | continent (9.4), administrative territory (3.4) |
| mother tongue | 11.8 → 3.2 | languages spoken (5.1) |
| official language | 12.4 → 14.8 | none (capital asks for a city) |
| citizenship | 17.0 → 17.1 | none |

Each relation's detector is fitted against its twin's wordings as negatives. CounterFact asks them in nearly the same words (its manufacturer prompts read "{s}, developed by"), so the threshold rises to where the detector barely opens on its own questions. This is the crowding of the first breadth run, inside one kind. Place of birth, place of death and work location (all cities of a person) sit at margins 2.2 to 5.7 and PS 0.46 to 0.54.

**Unchanged by the number of relations:** NS equals the unedited model's in every relation but capital (0.975 against 0.983, one neighbour prompt), no known facts lost, no drift. The layer pushed a first token 1,614 times and was vetoed 34 times.

**The ranking this run was for:**
1. **Near-twin relations crowding each other**, which costs most: 12 of 25 relations.
2. **Terse phrasings**, the earlier finding (mother tongue's "X, a native").
3. Leaks over 0.10 nowhere; occupation and sport are at the bar (0.104, 0.097).

## Near-twin relations: measuring which relations are twins (written 2026-11-09, before it ran)

**The idea** (proposed after the full mix and agreed by the user): find from data which relations ask in nearly the same words, and stop using each as a negative for the other.

**The measurement** (`scripts/diag_twins.py`, on the full mix's wordings):
- For every relation, a **solo detector** is built the way the shared detectors are (its drafted wordings, filled with invented names of its kind, against its kind's generic questions, negative prompts and real-world questions), but **without any other relation's wordings as negatives**. The shared detectors as built take each other's wordings as negatives, so their firing on each other is suppressed by construction and can't measure confusability.
- **Cross-firing of R on S:** the share of relation S's wordings (plain, chat and Q/A forms, filled with invented names) on which R's solo detector reads above its own threshold. It is measured for every pair of relations of the same kind.

**Fixed now:** R and S are **twins** if the cross-firing is at least 0.5 in both directions. Each relation's own positives are reported alongside for scale.

The build change, the development test on the full mix's 300 edits and the confirmation rule are written down after this measurement and before either runs.

## Result of the twin measurement, and the revision (written before the development test)

Raw numbers: `research/results/counterfact/full/twins.json`.

**The criterion fixed in advance fails as a rule.** At "0.5 both ways", 49 of 83 pairs count as twins, including owner, CEO and founder, which the company benches show the detectors tell apart. A solo detector, never taught to tell relations apart, fires on most questions about its kind of subject. The median cross-firing over all pairs (weaker direction) is 0.54, and citizenship's detector fires on 95% of place-of-birth wordings. That rule is not used.

**What the matrix does show:** a top group of pairs that read as indistinguishable, each detector firing on the other's wordings about as often as on its own:

| Pair | Cross-firing, weaker direction |
|---|---|
| place of death / place of birth | 0.99 |
| original language / language of work | 0.99 |
| continent / administrative territory | 0.98 |
| work location / place of death | 0.98 |
| developer / manufacturer | 0.98 |
| mother tongue / languages spoken | 0.97 |
| located in country / administrative territory | 0.96 |
| located in country / continent | 0.96 |
| work location / place of birth | 0.95 |
| *next: owner / founder* | *0.94* |

These 9 pairs cover exactly the 12 relations named as near-twins in the full mix's result, before this measurement.

**Revised criterion** (deviation, chosen after seeing this matrix): twins are pairs with cross-firing of at least 0.95 both ways, i.e. indistinguishable at the detector's own threshold. The matrix uses drafted wordings only, no test prompts. The fresh-draw confirmation below is the test that counts.

**Design, also fixed before any test:**
- Twins stop being negatives for each other **in the fact detectors only**, so a fact's detector opens on its own questions again.
- The **shared detectors keep their twins as negatives**, as in v1, so they stay sharp, and **the veto stays as in v1**. A twin's question ("which continent is X on", for a located-in-country fact) can now reach the fact's detector; the twin's shared detector then decides, through the veto, which relation is asked. This matters because some twins have different true answers (birth and death places; country, continent and region).

**A new measure, twin questions:** for each edit of a twin relation, 3 CounterFact rewrite prompts of each twin relation, taken from other records with this subject put in. It reports the share of replies that newly give the new answer. For developer/manufacturer and the language pairs that is usually right. For birth, death and work places, and for the place relations, it is wrong.

**The development test:** the full mix's 300 edits, paired against its run.
- The 144 edits of the 12 twin relations get new detectors, with their answer vectors kept.
- Every other edit keeps its parts as they are.

**Gate for a confirmation**, on the twin relations' 144 edits:
- PS up by at least 0.15 pooled;
- same-subject leaks ≤ 0.10;
- NS within 0.05 of the unedited model;
- twin-question answers ≤ 0.25 on the pairs whose answers differ by nature (birth, death and work places; the place relations).

**The confirmation, if the gate is met:**
- **The draw:** fresh edits of the 12 twin relations (seed 20261114), 12 each where that many remain unused, otherwise every usable one (at least 6), leaving out all earlier edits.
- **The table:** all 25 relations' wordings, as in the full mix.
- **Judged by:** the breadth rule per relation, and the twin-question measure with the same bar.

## Result of the twin development test, and a narrower confirmation (written before it ran)

The full mix's 300 edits. The 144 edits of the 12 twin relations get new detectors, keeping their answer vectors; the other 156 keep their parts. Raw rows: `research/results/counterfact/full_twins/`.

| The 144 twin-relation edits | Full mix | With twins |
|---|---|---|
| PS | 0.309 | **0.628** |
| ES | 0.451 | 0.778 |
| Same-subject leaks | 0.014 | 0.052 |
| NS (unedited 0.864) | 0.863 | 0.863 |
| Twin questions given the new answer: pairs whose answers differ | — | **0.299** |
| … pairs whose answers agree (developer and manufacturer, the language pairs) | — | 0.528 |

| Relation | PS before → after | ES before → after | Twin questions given the new answer |
|---|---|---|---|
| developer | 0.04 → **1.00** | 0.00 → 1.00 | 0.97 |
| manufacturer | 0.04 → 0.42 | 0.00 → 0.42 | 0.42 |
| original language | 0.12 → 0.50 | 0.08 → 0.75 | 0.33 |
| language of work | 0.42 → 0.71 | 0.42 → 0.92 | 0.56 |
| mother tongue | 0.42 → 0.62 | 0.42 → 0.58 | 0.39 |
| languages spoken | 0.42 → 0.71 | 0.92 → 1.00 | 0.50 |
| place of birth | 0.54 → **0.83** | 0.83 → 0.92 | 0.07 |
| place of death | 0.50 → 0.58 | 0.25 → 0.33 | 0.03 |
| work location | 0.46 → 0.71 | 0.92 → 1.00 | 0.11 |
| located in country | 0.33 → 0.62 | 0.33 → 0.67 | **0.79** |
| continent | 0.33 → 0.46 | 1.00 → 1.00 | **0.44** |
| administrative territory | 0.08 → 0.38 | 0.25 → 0.75 | **0.35** |

- **The other 156 edits are unchanged:** PS 0.721 both times.
- **Pooled over all 300:** ES 0.813, PS 0.677, NS 0.802 (unedited 0.803), same-subject leaks 0.052, no known facts lost, no drift.
- **The veto now does the work it was built for:** it fired 346 times, against 34. It is also why manufacturer gains least: CounterFact asks its facts as "{s}, developed by", which developer's shared detector claims.
- **By the gate, no confirmation of this twin list.** PS rises by 0.32, leaks and NS are within their bars, but twin questions on pairs whose answers differ get the new answer 0.299 of the time, against 0.25. The cause is the place relations (0.79, 0.44 and 0.35), whose detectors the veto can't tell apart. Birth, death and work places stay at 0.03 to 0.11, so for them the veto works.

**A narrower revision (deviation, chosen on these development edits):** twins are the 6 pairs outside the place relations: developer and manufacturer, the two language pairs, and the three pairs among places of birth, death and work. The place relations stay negatives for each other, as in v1. On these same edits the 9 relations' 108 edits go from PS 0.329 to 0.676 and ES 0.426 to 0.769, with same-subject leaks 0.025 and twin questions on birth, death and work places at 0.069. Only a fresh draw can confirm it.

**The confirmation, paired:**
- **The draw:** fresh edits of the 9 relations (seed 20261114), 12 each where that many remain, otherwise every usable one (at least 6), leaving out every earlier edit, including the full mix's 300.
- **Arm A:** no twin rule, but the twins' questions are asked.
- **Arm B:** the narrower twin list, with arm A's answer vectors, so only the detectors differ.
- **The table:** all 25 relations' wordings in both arms.

**The rule:** confirmed if, on the 9 relations' fresh edits pooled:
- B's PS is at least 0.15 above A's;
- B's same-subject leaks are ≤ 0.10;
- B's NS is within 0.05 of the unedited model's;
- B's twin questions on birth, death and work places get the new answer ≤ 0.25 of the time.

The breadth rule's verdict is reported per relation.

## Result of the narrower twin confirmation: confirmed

97 fresh edits of the 9 relations: 12 each, except place of birth (7) and manufacturer (6), which had no more unused facts the model knows. Every earlier edit was left out. Both arms are on the same edits with the same answer vectors. Raw rows: `research/results/counterfact/twins_confirm/A/` and `B/`.

| | Arm A, no twin rule | Arm B, twin rule | Bar |
|---|---|---|---|
| PS | 0.284 | **0.629** | B at least 0.15 above A: **+0.345, met** |
| ES | 0.402 | 0.763 | — |
| Same-subject leaks | 0.009 | 0.019 | ≤ 0.10: **met** |
| NS (unedited 0.858) | 0.858 | 0.857 | within 0.05: **met** |
| Birth, death and work-place twin questions given the new answer | 0.016 | 0.124 | ≤ 0.25: **met** |

- **Every edit's PS went up or stayed the same:** 48 up, 0 down.
- **No damage:** no known facts lost, no drift, neighbour leaks 0.001.
- **The veto is doing the arbitration:** it fired 199 times, against 47 in arm A.

| Relation (n) | PS A → B | ES A → B | Verdict, arm B |
|---|---|---|---|
| developer (12) | 0.04 → **1.00** | 0.00 → 1.00 | **works** |
| place of birth (7) | 0.43 → **0.86** | 0.71 → 1.00 | **works** |
| languages spoken (12) | 0.12 → 0.58 | 0.83 → 1.00 | in between: PS |
| work location (12) | 0.29 → 0.54 | 0.50 → 1.00 | in between |
| place of death (12) | 0.58 → 0.67 | 0.08 → 0.25 | in between |
| mother tongue (12) | 0.46 → 0.62 | 0.75 → 0.75 | in between |
| language of work (12) | 0.38 → 0.50 | 0.50 → 0.67 | in between |
| original language (12) | 0.17 → 0.46 | 0.17 → 0.75 | fails |
| manufacturer (6) | 0.00 → 0.42 | 0.00 → 0.33 | fails |

**Verdict: the twin rule is confirmed** for the 6 pairs, by the rule written before the run. Twin questions on pairs whose answers agree get the new answer more often (developer 0.97: a product's developer edit also answers "who made it"), which is the intended carry-over.

**The full mix with this rule** (development numbers: the 108 edits of the 9 relations from the twin development test, the rest from the full mix): ES 0.780 and PS 0.648, against 0.657 and 0.523 without it, with NS 0.802 (unedited 0.803) and same-subject leaks 0.041.

**What stays open:**
- The place relations (country, continent, region), which the veto can't tell apart: PS 0.08 to 0.33.
- Manufacturer, whose CounterFact prompts read as developer's ("{s}, developed by").
- The terse phrasings.
