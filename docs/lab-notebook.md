# Lab notebook

> **What this is.** The dated record of the project from 2026-11-06 to 2026-11-14, kept as it was written. Each section was written before the run it describes; results were added after.
>
> - Up to v1 (the sections dated 2026-11-06 to 2026-11-08, through "Result of the revision on bench40e"), the project was still called `weightspace` and ran from the beta's experiment folder.
> - Scripts named in those sections are no longer on the main branch. The research scripts behind v1 are at commit `4325028` under `research/bench/`; the beta's experiment scripts are at commit `a860535` ([history](history.md#where-the-beta-code-lives)). Scripts named in later sections are in [`scripts/`](../scripts/).
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

**Why.** I wanted a layer that sits between the model's layers, reads their input and corrects it. Every gated arm read company identity at layer 20, either from the question's end or from a tag the model itself had to carry there. At the company's name, identity separates far better: look-alike AUC 0.96 to 1.00 at layers 2 to 16 (`subject_site.py`), best at layer 4 with 32 facts at 0.95 or higher. An inserted block can read the name directly, with no transport. It is built in closed form from forward passes, and no weight of the model changes: detaching it is an exact undo.

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

**Why.** GRAFT has been measured on 7 of CounterFact's 34 relations. This run measures it on every relation the model knows enough facts for, all in one table, so that the remaining problems can be ranked by size. The work stays on the Qwen family for now (my direction). It is a measurement, not a revision: nothing is tuned on this draw. Any fix chosen from it is confirmed on a fresh draw.

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

**The idea** (my choice after the full mix): find from data which relations ask in nearly the same words, and stop using each as a negative for the other.

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

## Phrasing coverage, with the big model as editor (written 2026-11-09, before any of it ran)

**Why.** With the twin rule, 211 of the full mix's 600 reworded prompts still miss. Prompts whose phrasing has only 1 to 3 words besides the subject ("X, a native") reach PS 0.45, and longer ones about 0.70. So about 78 misses are terse, and the rest are longer phrasings the drafted wordings don't cover. I wanted a big model to do this kind of work as the editor (check, edit, update), and wanted it measured.

**Two arms of new wordings, both written by big-model agents with no dataset or web access, plain forms only (CounterFact asks only plain prompts):**
- **Editor arm.** For each relation the agent sees the phrasings that missed on the full mix's 300 development edits (subject as {s}) and the relation's current wordings. It writes 20 new wordings per relation that would teach the detector those forms and others like them, and it may include the failing forms themselves. This is the loop the product is meant to run: see what fails on the user's own prompts, and teach it.
- **Blind arm.** The agent sees no failures. A general brief asks for 20 new wordings per relation, favouring short fragments and indirect forms from running text (appositives, trailing prepositions, captions, list items), with an example from an unrelated relation only. This measures what general variety buys.

**The confirmation, three arms on one fresh draw:**
- **The draw:** 8 edits per relation for the 21 relations with at least 8 unused facts left (seed 20261115), leaving out all 669 earlier edits. That gives 168 edits.
- **Base:** the full mix's wordings with the twin rule.
- **Editor:** base plus the editor arm's wordings.
- **Blind:** base plus the blind arm's wordings.
- All three share the base arm's answer vectors, so only the detectors differ. The build is otherwise revision 2's, with the narrower twin list.

**The rule, per arm against base:**
- **Adopted** if pooled PS is at least 0.05 higher, same-subject leaks ≤ 0.10, and NS within 0.05 of the unedited model.
- **For the editor arm**, PS is also split into reworded prompts whose phrasing was among the failures it was shown and those whose phrasing was not. CounterFact reuses phrasings, so only the second part measures generalisation; it is reported without a bar.

## Result of phrasing coverage: the editor's wordings are adopted

167 fresh edits (seed 20261115), 8 per relation for 21 relations (one with 7), none used before. The three arms share the base arm's answer vectors. Raw rows: `research/results/counterfact/phrasing/{base,editor,blind}/`; the agents' wordings and the editor's input: `research/results/counterfact/phrasing/drafts/`.

| Arm | PS | ES | Edits with PS up / down vs base | Same-subject leaks | NS (unedited 0.841) | Relations meeting every bar | Rule |
|---|---|---|---|---|---|---|---|
| Base (twin rule) | 0.638 | 0.743 | — | 0.033 | 0.839 | 5 | — |
| **Editor** | **0.766** | **0.862** | 42 / 2 | 0.040 | 0.839 | **9** | **adopted** (+0.128) |
| Blind | 0.653 | 0.754 | 10 / 5 | 0.041 | 0.839 | 4 | not adopted (+0.015) |

All three arms lost no known facts, showed no drift, and had neighbour leaks of 0.002.

**PS by phrasing** (334 reworded prompts):

| Phrasing | Prompts | Base | Editor | Blind |
|---|---|---|---|---|
| Failed during development for this relation (shown to the editor) | 197 | 0.487 | **0.680** | 0.513 |
| Any other | 137 | 0.854 | **0.891** | 0.854 |

- **Teaching what fails works; general variety doesn't.** The editor saw the failing forms and taught them and their variants. It gains most on those forms (+0.193), and still some on prompts it never saw (+0.037). The blind agent wrote 482 varied short fragments and gained +0.015, within noise.
- **The size of the gain depends on CounterFact reusing its phrasings:** 197 of the 334 fresh reworded prompts repeat a form that failed during development. On prompts phrased in new ways the gain is the smaller one. In use, that is the point of the loop: the editor sees the failures on the user's own prompts.
- **The editor used judgment:** both editor agents declined to teach a form that fits several relations to one of them ("{s}, the" fails for both sport and position played). They covered such forms with variants that carry a clue instead. Where a taught form belongs to another relation, that relation loses a little: 12 such prompts, PS 0.917 → 0.750.
- **Relations meeting every bar with the editor's wordings (9 of 21):** citizenship, headquarters, developer, event location, language of work, languages spoken, official language, religion, work location.
- **Still weak:** a place's country (0.44), region (0.19) and continent (0.50), sport (0.50) and field of work (0.56).

## Qwen3.5-4B: does the recipe carry over? (written 2026-11-09, before any of it ran)

**Why.** Everything so far is on Qwen3.5-0.8B. My direction is to perfect GRAFT within the Qwen family first. The next question is whether the confirmed recipe holds on a bigger model of the same family, with nothing retuned.

**The recipe, as adopted on the 0.8B:**
- revision 2's build (`context=4`, `real_world="other_relations"`, the fast build);
- the twin rule with the 6 confirmed pairs;
- the editor's wordings;
- push 2, margin 0.5.
- **The layer:** 27 of 32, the same depth as 20 of 24. It is not tuned.

**The run:**
- **Screen:** which CounterFact facts the unedited 4B gives (`scripts/counterfact_screen.py`).
- **The draw:** 8 edits per relation for the 25 relations of the full mix (at least 6 where fewer remain), seed 20261116. Every edit used on the 0.8B is left out (836).
- **One arm,** the recipe.

**The check fixed in advance:** the recipe carries over if, pooled:
- PS ≥ 0.70 and ES ≥ 0.80 (on the 0.8B's fresh draw: 0.766 and 0.862);
- same-subject leaks ≤ 0.10;
- NS within 0.05 of the unedited 4B;
- at most 2 known facts lost, against the 4B's own unedited answers.

The breadth rule's verdict is also reported per relation.

## Result on Qwen3.5-4B: the recipe carries over

The 4B gives the old answer on 5,679 of 21,919 CounterFact rewrite prompts (the 0.8B: 4,070). 200 fresh edits, 8 for each of 25 relations, none used on the 0.8B, were built into one table at layer 27 of 32 with the 0.8B's recipe and nothing retuned. Raw rows: `research/results/counterfact/qwen4b/`.

| | Qwen3.5-4B | Unedited 4B | Bar | 0.8B, its fresh draw (editor's wordings) |
|---|---|---|---|---|
| ES | **0.950** | 0.035 | ≥ 0.80: met | 0.862 |
| PS | **0.910** | 0.048 | ≥ 0.70: met | 0.766 |
| Says the new answer: rewrite / reworded | 0.970 / 0.910 | — | — | 0.856 / 0.749 |
| NS | 0.870 | 0.874 | within 0.05: met | 0.839 (0.841) |
| Same-subject leaks | 0.095 | — | ≤ 0.10: met | 0.040 |
| Neighbour leaks; known facts lost; drift | 0.004; 0; 0.0000 | — | ≤ 2 lost: met | 0.002; 0; 0 |

**Verdict: the recipe carries over**, by the check written before the run, and the pooled breadth rule's verdict is "works". 13 of 25 relations meet every bar:
- **Works:** citizenship, country of origin, developer, field of work, located in country, manufacturer, mother tongue, official language, original language, place of birth, place of death, position held, work location.
- **Fail only on same-subject leaks:** event location 0.31, religion 0.20, headquarters 0.17, language of work 0.15, original network 0.15, languages spoken 0.10, occupation 0.10.
- **Still weak:** continent (PS 0.50) and administrative territory (0.31).

- **Recall rises with the model; so do leaks.** PS goes from 0.766 to 0.910 and ES from 0.862 to 0.950. Same-subject leaks go from 0.040 to 0.095, just under the bar. The 4B's detectors open more readily, on its own questions and on others about the same subject. Leak control is the next thing to tighten on bigger models.
- **The neighbourhood stays put at this size too:** NS 0.870 against 0.874, no known facts lost, no drift.
- **For scale only** (another model and protocol: 2,000 sequential edits on GPT-J, with no restriction to known facts): AlphaEdit reports PS 96.4 and NS 75.5 against 83.1 unedited, and MEMIT PS 95.5 and NS 63.6. GRAFT on the 4B reaches PS 91.0 with NS within 0.4 points of the unedited model's.

## The grafted model: facts in the model's own weights, size unchanged (written 2026-11-09, before any of it ran)

**My direction:** the main output of GRAFT is the **grafted model**, a checkpoint whose own weights carry the new and updated facts, loadable with no hooks. The inserted layer stays as the reference. **The model's size must not grow:** no added neurons or parameters, for new facts and for updates alike. The beta's slot work is kept: `scripts/slot_feasibility.py` and its result, `research/results/slots/00_feasibility.json`, restored from commit a860535.

**Design: reuse existing neurons as slots.** A SwiGLU neuron computes `down · (silu(gate·x) × (up·x))`, a product of two readings, so one neuron can be an AND gate:
- **gate** reads the question detector, with the threshold made from the direction `c` whose reading is constant on ordinary text (within 1.7%, measured in the beta), since the MLP has no bias;
- **up** reads that the subject is present;
- **down** writes the answer vector.

The subject's presence (key 1) has to come from weights too: name detectors in early-layer slots that fire on the subject's exact tokens and write a tag, which the model's own attention carries to the end of the question. The beta measured that such a tag survives from layer 4 to 16 (AUC 0.997 with a fitted readout).

**Step 1, capacity** (`scripts/diag_slots.py`): how many of a layer's neurons can be reused before ordinary behaviour changes?
- A reused neuron that isn't firing is close to silent, so the cost is measured by silencing neurons.
- **At layer 20:** the N neurons that contribute least on ordinary text (the beta's measure: mean |activation| × the norm of their output column, on the 240 fitting paragraphs), for N = 16, 64, 256 and 1,024, and as many random neurons for comparison.
- **At layers 16 and 22:** N = 256, to see whether spreading the slots over layers costs less.
- **Measured each time:**
  - the benches' drift (KL from the unedited model on the fixed ordinary sentences);
  - the share of next-token predictions unchanged;
  - KL per token on the held-out WikiText paragraphs;
  - the 133 known facts.

**Fixed now:**
- **A layer's slot budget** is the largest N with drift ≤ 0.005 (a tenth of the CounterFact bar, leaving room for the facts themselves) and no known fact lost.
- **If even N = 64 at layer 20 fails,** the size-preserving compile at layer 20 is not viable as designed, and the plan is revisited before anything is built.

**Step 2 is written down after step 1:** compile layer 20's fact neurons into slots, with key 1 still from the name index, and compare decision by decision with the inserted layer.

## Result of step 1: neuron reuse by average use fails

Raw numbers: `research/results/slots/01_capacity.json`. Before anything was switched off, the model answered all 133 known facts.

| Neurons switched off | Drift (bar 0.005) | Next-token predictions unchanged | Known facts lost (bar 0) |
|---|---|---|---|
| layer 20, 16 least-used / 16 random | 0.00053 / 0.00109 | 0.996 / 0.982 | 1 / 0 |
| layer 20, 64 least-used / random | 0.00098 / 0.00200 | 0.986 / 0.982 | **3 / 4** |
| layer 20, 256 least-used / random | 0.00367 / 0.00735 | 0.957 / 0.968 | 4 / 5 |
| layer 20, 1,024 least-used / random | 0.01712 / 0.02105 | 0.935 / 0.924 | 15 / 14 |
| layer 16, 256 least-used / random | 0.00418 / 0.00879 | 0.971 / 0.953 | 2 / 1 |
| layer 22, 256 least-used / random | 0.00867 / 0.00729 | 0.964 / 0.971 | 2 / 11 |

**By the rule, the slot budget is 0 at every layer tried, and layer 20 fails at 64.** The design is revisited before anything is built.

- **Drift is not the limit:** the least-used neurons stay under the drift bar up to 256 at layer 20, at half the drift of random ones.
- **Known facts are the limit:** switching off a few dozen neurons flips some known facts, whichever are chosen. The least-used by average are not idle; they fire rarely and strongly (the beta measured peaks of 5 to 46 times the median), as specialised neurons do.

**The revision (step 1b, written before it ran):** choose slots by measured harm rather than average use (`scripts/diag_slot_harm.py`).
- **The scan:** each of layer 20's 3,584 neurons is switched off alone, and two things are measured:
  - the KL on the fixed ordinary sentences;
  - the drop in log-probability of the first token the unedited model gives for each of the 133 known facts, at the prompt's end.
- **A neuron's harm** is the larger of its KL, scaled by the KL bar, and its largest drop on any known fact, scaled by 1 nat.
- **The group test:** the 16, 64 and 256 least harmful neurons are switched off together, with the same measures and the same rule as step 1 (drift ≤ 0.005, no known fact lost).
- **If 64 still fails,** single-layer slots are out. The next design keeps the size by spreading closed-form edits over several layers, with known facts protected.

## Result of step 1b: slots chosen by measured harm also fail the zero-loss bar

Raw numbers: `research/results/slots/02_harm_layer20.json` (every neuron's KL and largest drop) and its log. The noise check is `scripts/diag_slot_noise.py`.

**The measurement has no noise floor:** repeating the readings with nothing switched off, or through a hook that changes nothing, gives a KL of exactly 0 and a largest drop of 0, and repeated generation flips no known fact.

**Every neuron matters a little:**
- Switching one off costs a median KL of 4.0e-4 on the ordinary sentences, and a median largest drop of 0.076 nats on some known fact.
- The least harmful neuron's score is 0.065, against a median of 0.083, so there are no idle neurons.
- 6 of 3,584 neurons, switched off alone, cost a known fact more than 1 nat.

| Least harmful neurons switched off together | Drift | Known facts lost |
|---|---|---|
| 16 | 0.00059 | 1 ("Skype is owned by") |
| 64 | 0.00097 | 1 ("The capital of Wales is"); 3 by average use, 4 at random |
| 256 | 0.00270 | 3 |

**By the rule, the slot budget is 0: single-layer slots are out.**
- **Selection by measured harm cuts the damage at 64 from 3 or 4 facts to 1,** but a different fragile fact flips at each size. These are answers the model barely gets right, which any change to its computation can tip.
- **The inserted layer meets a zero-loss bar** because it changes nothing until its two keys turn. Any edit of the model's own weights changes some of its computation everywhere.

**A decision for me before the next design:** what damage a grafted model may do.
- Under the CounterFact rule's bar (at most 2 known facts lost), 64 harm-selected slots at layer 20 pass (1 lost, drift 0.001).
- Under zero, the next try is closed-form compensation: after freeing the slots, re-fit the layer's output on the remaining neurons to reproduce the original on calibration text. That is pruning with reconstruction, forward passes and one linear solve. It is also the protected, spread-out design written down before this ran.

## Step 1c: freeing slots with compensation (written 2026-11-10, before it ran)

**My choice:** try the zero-damage bar first, by compensation. If it fails, step 2 proceeds under the CounterFact bar (at most 2 known facts lost).

**Compensation.** When a set F of layer 20's neurons is freed, the remaining neurons K take over their work: K's output columns are re-fitted so that the MLP's output on ordinary text is reproduced without F.
- The fit is ridge regression on the layer's intermediate activations over the 240 fitting paragraphs (38,882 tokens): `ΔW_K = W_F H_FK (H_KK + λI)⁻¹`, with H the activations' second moment.
- λ is chosen per run from {10⁻⁶, …, 10⁻¹} × the mean diagonal of H_KK, by the reconstruction error on the 75 spare paragraphs (280 to 354, used elsewhere only as a pool of context sentences). It is never chosen by the tests.
- The edited columns are stored in bfloat16, as a checkpoint holds them.
- Forward passes and one linear solve per run; no gradients.

**Two ways of choosing the slots:**
- **by harm:** step 1b's order (each neuron switched off alone). It was chosen with the 133 known facts, so on those facts it is not a clean test.
- **by redundancy:** the neurons whose output the others reproduce best (Optimal Brain Surgeon's saliency, `‖w_j‖² / [(H + λI)⁻¹]_jj`, removed one at a time with the inverse updated). It is computed from the fitting paragraphs alone.

**A second, clean set of known facts:** 400 CounterFact facts the unedited model answers, drawn with seed 0 from the 4,070 the screen found. Neither order saw them. They are re-checked by generation in the same run before anything changes, and only those answered then count.

**Group tests:** N = 16, 64, 256, 512 and 1,024, for each order, with and without compensation. Each is measured as in step 1:
- drift on the fixed ordinary sentences, and the share of predictions unchanged;
- KL per token on the 40 held-out paragraphs;
- known facts lost: the 133, and the clean 400.

**Fixed now:**
- **Zero damage** means drift ≤ 0.005 and no known fact lost in either set (533 facts). An order's budget is the largest N that meets it with compensation.
- **If either order meets it at N = 64,** zero damage is reachable at layer 20, and step 2 uses that order at its budget.
- **If neither does,** zero is out for single-layer slots. Step 2 then proceeds under the CounterFact bar, using the redundancy order with compensation at the largest N with drift ≤ 0.005 and at most 1 of the 133 lost, which leaves one for the facts themselves. The redundancy order is used because it never saw the 133.

**A diagnostic, written after step 1c ran and before this one did: the known facts' numerical noise floor** (`scripts/diag_known_noise.py`). In bfloat16, a reply can depend on the batch it is generated in. Re-checked as a group, 9 of the 400 CounterFact facts that the screen found answered were no longer answered, with no weight changed. So the unedited model's answers to both known-fact sets are generated with batch sizes 1, 8, 16, 32 and 64, and in reversed order, and counted against each check's own batching, the reference every slot test is compared with. It decides nothing. It sizes the flips a weight edit causes against the flips that arithmetic alone causes.

## Result of step 1c: zero damage is not reachable at layer 20, and the losses are the size of numerical noise up to 256 slots

Raw numbers: `research/results/slots/03_compensate_layer20.json` and `04_known_noise.json`, with their logs. Unedited, the model answered all 133 known facts, and 391 of the 400 CounterFact facts when they were checked as a group.

Known facts lost, as "of 133 / of 391":

| Neurons freed | By harm, switched off | By harm, compensated | By redundancy, switched off | By redundancy, compensated |
|---|---|---|---|---|
| 16 | 1 / 3 | 1 / 2 | 0 / 3 | 3 / 2 |
| 64 | 1 / 3 | 2 / 5 | 3 / 7 | **1 / 3** |
| 256 | 3 / 5 | 1 / 5 | 3 / 9 | 3 / 3 |
| 512 | 2 / 17 | 3 / 17 | 4 / 15 | 6 / 11 |
| 1,024 | 6 / 26 | 7 / 24 | 8 / 23 | 11 / 12 |

Drift at 64: 0.00097, 0.00086, 0.00152 and 0.00082. At 256: 0.0027, 0.0022, 0.0038 and 0.0019. Every setting up to 512 is under the drift bar of 0.005.

**By the rule, both orders' zero-damage budget is 0.** Step 2 uses the fallback: the redundancy order, compensated, at N = 64 (drift 0.00082, 1 of 133 and 3 of 391 lost).

- **Compensation lowers the average change, not the facts lost.** It cuts the KL on the held-out paragraphs by 30 to 50% (at 1,024: 0.026 → 0.015 by harm, and 0.015 → 0.008 by redundancy) and the drift by 10 to 45%. The facts lost go down about as often as up. The facts that flip are ones the model barely gets right, and which of them tip doesn't follow the average.
- **The harm order's neurons do work the others can't take over:** compensation leaves 39 to 45% of their lost output on the spare paragraphs, against 12 to 32% for the redundancy order.
- **Measured against numerical noise:** with no weight changed, generating the same questions at another batch size loses up to 1 of 133 and up to 7 of 391 at a single batch size (10 of 391 lost under at least one of the seven batchings). Freeing up to 256 neurons loses about that many: 1 to 3 of 133, and 2 to 9 of 391. From 512 on, it clearly loses more: 11 to 26 of 391.
- **Deviations and limits:**
  - The fit used 27,640 tokens, not the 38,882 written in the plan. That figure came from the beta's count, which truncated differently.
  - The chosen ridge was the grid's largest (0.1) every time, so a stronger one might reconstruct a little better. It wouldn't change the main finding, since the facts lost don't follow the reconstruction error.

## Step 2: the facts in layer 20's own neurons (written 2026-11-10, before any of it ran)

**What is built.** 64 edits from the editor run (fresh CounterFact edits with saved parts), 3 for each of the 21 relations and one more, taken in id order among the edits with a plain question detector. Each edit gets one neuron, in step 1c's 64 slots (the redundancy order, with the freed neurons' work compensated). A slot is an AND gate made from the edit's GRAFT parts:
- **gate row:** the edit's plain question detector, with its threshold carried by the constant direction: `k·(w − (τ/c̄)·c)/δ`. The gate's input is then k·z, to within the 1.7% spread of c's reading. k = 8.
- **down column:** the push, `2·v/silu(k)`, with v the answer's first-token vector. At z = 1 the neuron writes the inserted layer's push; at z = 0.5, half of it; beyond, more, in proportion. A neuron's push is graded where the inserted layer's is a step.
- **up row, two ways:**
  - **B, the hybrid:** the constant reading `c/c̄` (about 1), times a hook from the name index: 1 when the edit's subject is the last name read, else 0. Only key 1 is outside the weights.
  - **C, the grafted model:** a subject reading in the weights. It is an LDA at layer 20 between the relation's first 16 drafted plain questions about the subject (with the build's prefixes and context sentences) and the same questions about other subjects: invented names of the kind, and 24 real subjects of the same CounterFact relation that appear in none of that relation's test prompts. It is scaled to 1 at the positives' median and 0 at the negatives' 99th percentile. No hook at all.
- **What a neuron can't do is left out:** the veto by other relations' detectors, the one-answer-per-reply rule and the continuation pushes. Plain form only: CounterFact's scored prompts are all plain, and chat questions come later. 161 of the 167 editor-run answers are one token.

**Reference A:** the inserted layer on the same 64 edits, built from the same parts and scored in the same run.

**Measured:**
- CounterFact's measures for A, B and C: ES, PS, NS, generation, neighbour and same-subject leaks;
- the 133 known facts and the clean 391, and drift (for B and C, and for the freed slots alone);
- decision by decision: for each scored prompt (rewrite, reworded, neighbours), whether the new answer beats the old in B and in C as it does in A, and the detector's z there.

**Fixed now:**
- **B reproduces the inserted layer** if its ES and PS are within 0.05 of A's, its NS within 0.01, neighbour leaks ≤ 0.05 and same-subject leaks ≤ 0.10. If not, the gate and push compile is revised before key 1 goes into the weights.
- **C is a grafted model as good as the inserted layer** if it meets the same comparison with A, loses at most 2 of the 133 known facts with drift ≤ 0.05 (the CounterFact bar), and loses no more of the 391 than numerical noise does at a single batching (7). If C fails only on answers leaking to other subjects, key 1 needs exact-token name neurons in an early layer (step 3).

## Result of step 2: the question and the push work as the model's own neurons; key 1 from a reading at layer 20 breaks the model

Raw numbers: `research/results/slots/05_step2/` (summary, decisions, the edits, the inserted layer's rows, the log).

| | Unedited | A, inserted layer | B, hybrid (key 1 from the name index) | C, no hook (key 1 from a subject reading) |
|---|---|---|---|---|
| ES | 0.031 | 0.906 | **0.906** | 0.750 |
| PS | 0.070 | 0.727 | **0.727** | 0.680 |
| NS | 0.880 | 0.880 | 0.878 | 0.606 |
| Says the new answer, reworded | 0.000 | 0.734 | 0.727 | 0.484 |
| Neighbour leaks; same-subject leaks | 0; 0 | 0.000; 0.036 | 0.002; 0.031 | 0.111; 0.008 |
| Known facts lost (of 133; of 391) | — | 0; not measured | 1; 6 | 112; 309 |
| Drift | — | 0.0000 | 0.0008 | 2.43 |

- **B reproduces the inserted layer, by the rule.** It agrees with A's outcome on 97% of the rewrite prompts, 94% of the reworded ones and 100% of the neighbours'. It loses 5 reworded prompts that A wins, all where the detector barely opens (z between 0.00 and 0.22): there the neuron's push is graded and too weak. Its known-fact losses are those of freeing the slots (1 of 133, 3 of 391; 6 of 391 is within the noise of 7). Without the veto, B answers the twin relations' questions with the new answer a little more often (0.68 against 0.57).
- **So key 2 and the push compile into the model's own neurons.** The gate rows hold the detector's threshold through the constant direction. They are stored in bfloat16, and their z was off by 0.003 at the median, 2.1 at worst.
- **C fails, and breaks the model,** losing 112 of 133 known facts and replying with garbage ("a a a a a"). B and C share the same gate and down rows and differ only in the up row, so the up row is the cause.
  - **The subject reading is linear and unbounded:** it was fitted only on questions' last tokens, not on every token of ordinary text. A neuron multiplies the gate (above zero wherever the detector opens, and up to −0.28 just below) by that reading, so at tokens where the reading is large, it writes the push at any size. In B, the up row is near zero everywhere except after the subject's name.
  - **The reading also can't tell the subject from its neighbours:** on the CounterFact neighbours' prompts, other subjects with the same old answer, it reaches 0.8 to 1.4 for some edits, as high as on the subject's own reworded prompts (about 1).
- **What this says for key 1:** the subject must be marked by its exact name, and the neuron's AND must stay shut wherever the mark is absent. A reading of the model's own state at the end of the question can do neither.

## Step 3: key 1 in the weights, by name neurons and a carried tag (written 2026-11-10, before any of it ran)

**The design.** The subject is marked where its name is read, and the mark is read where the question ends.
- **A name neuron in layer 4** fires on the subject's exact name, at its last token, and writes a tag: a direction that the model's own attention carries from there to the end of the question. Each subject gets its own tag. The beta measured that a tag written at layer 4 is read at layer 16 to 20 with AUC 0.997.
- **The fact neuron in layer 20** reads the tag, t ≈ 1 when the subject's tag has arrived and ≈ 0 otherwise. Its AND is built to stay shut without the tag, which is what failed in step 2's C:
  - gate: `k·(z_q + M·(t − 1))`, with the question's z_q as in step 2, so that without the tag the gate is pushed M below its threshold;
  - up: t itself, so that without the tag the neuron's output is near zero even where the gate is open;
  - down: the push, as in step 2.
  - k = 8, M = 3.

**The tags.**
- **The transport map:** the linear map from a vector added to layer 4's MLP output at the name's last token to the change in layer 20's MLP input at the prompt's last token. It is measured one basis direction at a time, at the tag's own size, on one edit's rewrite prompt per relation (21 prompts, training wordings), in bfloat16.
- **The tag space** is the 32 input directions the stacked maps carry most strongly (their top right singular vectors). Each subject's tag is a random unit vector in that space (seed 0).
- **Size:** 2 units, where a unit is the median norm of layer 4's MLP output at the names' last tokens.

**The name neurons** (one subject each, two neurons so that the tag has the same size however strongly the name is recognized: `silu(40·z) − silu(40·z − 10)`, full size from z = 0.25):
- **gate:** an LDA at layer 4's MLP input. The positives are the name's last token in the relation's first 16 drafted plain wordings and the edit's own training wording (with the build's prefixes and context sentences), in its same-subject training questions, and in 8 drafted chat questions. The negatives are every other token of those prompts, every token of the other 63 subjects' prompts, and every token of 120 fitting paragraphs.
- **threshold:** the largest negative reading. δ is half the gap from there to the positives' median.
- **up:** layer 4's constant reading. **down:** the tag.
- **Slots:** 128 at layer 4, the first of layer 4's redundancy order, with compensation as in step 1c. Layer 20 keeps step 2's 64 slots.

**The tag reading at layer 20, one per edit:** an LDA at the prompt's last token.
- Positives: the subject's prompts with its tag.
- Negatives:
  - the same prompts without the tag, so that the reading follows the tag and not the subject's own representation, which neighbours share;
  - the same questions about invented and real subjects;
  - the other edits' prompts, with their own tags;
  - 4,000 tokens of the fitting paragraphs.
- Scaled to 1 at the positives' median and 0 at the negatives' 99th percentile.

**Two arms on step 2's 64 edits, against step 2's A:**
- **D:** the tag is written by a hook from the name index, exactly at the name. This checks the carry, the reading and the AND, with key 1's detection still exact.
- **E:** the tag is written by the name neurons. No hook: a grafted model whose weights alone carry the edits.

**Measured:** as in step 2, plus how often the name neurons fire at the scored prompts' names (rewrite and reworded: should; neighbours: should not) and on ordinary text (should not).

**Fixed now:**
- **E is a grafted model as good as the inserted layer** if its ES and PS are within 0.05 of A's, its NS within 0.01, neighbour leaks ≤ 0.05 and same-subject leaks ≤ 0.10, with at most 2 of the 133 known facts lost, no more of the clean 391 than numerical noise (7), and drift ≤ 0.05.
- **If E passes,** the edited weights are saved as a checkpoint and loaded and scored without the package, before anything else.
- **If E fails and D passes,** the name neurons are at fault. **If D fails,** the tag's carry or reading is.

## Result of step 3: the name neurons work; the tag reading doesn't, and breaks the model

Raw numbers: `research/results/slots/06_step3/`.

**Both arms fail, and break the model:** D (tags from the name index) has drift 69.7 and E (name neurons) 73.6, with every known fact lost. The tag reading at layer 20 is the cause, in both:
- **It doesn't follow the tag.** On the subject's prompts with the tag, its 5th percentile is −0.06 (median over edits), where 1 was meant. On the same prompts without the tag it reaches 0.58, and on other edits' tagged prompts up to 195. Only 2 of 64 edits separate cleanly.
  - The LDA's target was the subject's tagged prompts against the mean of four negative groups. Ordinary text and other subjects differ from them far more than the tag does, so the reading mostly learned the subject's prompts against everything else, C's mistake again. The tag itself carried little weight.
  - The tags were 64 codes in a 32-dimensional space, so a linear reading can't single out one of them.
- **Where a reading barely separates, its scale explodes.** Its 0 to 1 scaling divides by the gap between the positives' median and the negatives' 99th percentile. Near zero, that gap blows every reading up, the gate and up rows with it.

**What works:**
- **The name neurons,** LDA at layer 4 against every other token:
  - separable for 63 of 64 subjects, with every training positive above the largest negative;
  - they fire on the subject's name in 188 of 192 of its rewrite and reworded prompts;
  - on 8 of 4,456 tokens of held-out text.
  - 112 of 640 neighbour prompts had some name neuron fire. That count also includes correct fires: several edited subjects are Microsoft products named in each other's neighbour prompts. The next run counts only fires where the subject's name is absent.
- **The transport:** a vector written at layer 4 at the name arrives at layer 20 amplified about 8×. The stacked map's singular values are 238, then 80, 48, ..., 22 at the 32nd and 16 at the 64th.

**A capacity cost:** freeing 128 neurons at layer 4 and 64 at layer 20 (both compensated) loses 6 of 133 known facts and 8 of 391, with drift 0.0029. That is over the CounterFact bar. Layer 4's neurons cost more than layer 20's.

## Step 3, revision 1 (written 2026-11-10, after step 3's result, before it ran)

**Changes, each from a measured failure:**
1. **The tag reading follows the tag:** `w = (S + ridge)⁻¹ (mean of the subject's tagged prompts − mean of the same prompts untagged)`. S is the calibration covariance plus the within-group scatter of the tagged prompts, the untagged prompts, the other edits' tagged prompts, other subjects' prompts and 4,000 tokens of ordinary text. It is scaled to 1 at the tagged prompts' median and 0 at the 99th percentile of all the negatives.
2. **A guard:** an edit whose reading separates by less than 3 standard deviations of the negatives' readings gets no fact neuron (counted as not edited).
3. **Orthogonal tags, one per subject:** the transport map's input directions 2 to 65. The first is left out: its gain is 3× the next, and it would carry most of every tag. Each tag is sized so that all arrive equally strongly, at the strength of a 2-unit tag along the median of those directions.
4. **One name neuron per subject, not two,** to halve the early-layer cost. Its tag is graded, at full size at the positives' median z. **The name layer is chosen in this run** from 4, 8 and 12. For each, 64 neurons are freed (redundancy order, compensated). The layer that loses the fewest of the clean 391 is taken (ties: fewer of the 133, then lower drift). The beta measured the subject's identity at its last token as separable at layers 2 to 16.
5. **The AND:** gate `k·(z_q + M·(t − 1))` with M = 1, up t, down the push. The up factor does the excluding, as the name index did in B.
6. **The name neurons' false fires are counted properly:** a fire for a subject whose name the name index doesn't find in the prompt.

**Fixed now:** the same rule as step 3 for E (and D, as a diagnosis), against step 2's A. Also reported: how many edits the guard drops, and each fact neuron's largest output on held-out text.

**The name-layer scan, and a deviation (written after the scan, before either arm ran).** With 64 neurons freed (redundancy order, compensated) at one layer alone:

| Name layer | Known facts lost (of 133) | Of the clean 391 | Drift |
|---|---|---|---|
| 4 | 5 | 1 | 0.0019 |
| 8 | 0 | 4 | 0.0011 |
| 12 | 1 | 4 | 0.0009 |

The rule chose layer 4, by the fewest of the clean 391 lost. But layer 4's freed neurons alone lose 5 of the 133, over the CounterFact bar of 2, before any edit is added, so E could not pass there whatever its edits did. The rule's order should have applied the bar on the 133 first.
- **Corrected rule:** the fewest of the clean 391 lost, among the layers that lose at most 2 of the 133; ties go to fewer of the 133. That gives **layer 8**.
- The run with layer 4 was stopped after the scan, before either arm was scored, so nothing of the test was seen. Its log is kept as `runs/slots/step3_rev1_layer4_stopped.log`.

## Result of step 3, revision 1: the name neurons hold up; a tag read at the end of the question doesn't

Raw numbers: `research/results/slots/07_step3_rev1/`. Name layer 8, by the corrected rule.

| | A, inserted layer | D, tags from the name index | E, name neurons (no hook) |
|---|---|---|---|
| ES; PS | 0.906; 0.727 | 0.266; 0.195 | 0.250; 0.188 |
| NS (unedited 0.880) | 0.880 | 0.864 | 0.858 |
| Known facts lost (of 133; of 391) | 0; — | 57; 87 | 57; 86 |
| Drift | 0.0000 | 0.48 | 0.49 |

**Both fail.**
- **The tag readings separate for 17 of 64 edits (D) and 16 (E);** the median separation is 1 and 0.8 standard deviations.
  - The orthogonal tags arrive at layer 20 with strength 3 to 7, small against how much the question's end varies from prompt to prompt.
  - They are no longer orthogonal when they arrive: another edit's tag reads up to 2.0.
- **Even the edits that passed the guard damage the model:** their fact neurons reach 13.6 to 13.9 times their nominal output on held-out text. At tokens where the reading is large, the gate's `M·(t − 1)` term opens it, and the up row multiplies by the same large reading.

**What holds up: the name neurons in layer 8.**
- They separate for 64 of 64 subjects.
- They fire on the subject's name in 185 of 192 of its rewrite and reworded prompts.
- A false fire (a subject not named in the prompt) happens on 22 of 640 neighbour prompts, and on no held-out paragraph.
- Freeing their 64 slots and layer 20's costs 3 of 133 known facts and 3 of 391 (drift 0.0013).

**So key 1's detection works as weights. Carrying a mark to the end of the question and reading it there does not.** Three designs of the reading failed: the subject reading (step 2's C), a reading against everything (step 3), and a reading of the tag's own effect (this revision).

## Step 3, revision 2: the edit written where the name is read (written 2026-11-10, after revision 1's result, before it ran)

**The idea.** The name neuron already fires on the subject's exact name. Instead of a tag that must be read at the end of the question, it writes the edit itself at the name: a vector δ, chosen so that the model's own attention, when it reads the subject for that relation's question, carries δ to the end of the question as the push. Nothing is read at the end of the question, and layer 20 needs no slots.

**Solving δ for each edit, in closed form from forward passes:**
- **The maps:** T_p maps a vector added to layer 8's MLP output at the name's last token to the change in the residual after layer 20 at the prompt's last token, where the inserted layer adds its push. It is measured one basis direction at a time, at 1 unit, on two sets of training prompts:
  - P: 8 of the relation's wordings about the subject (the edit's own training wording first, then drafted wordings with the build's prefixes);
  - Q: 8 of the subject's same-subject training questions, which ask other relations.
- **The solve:** `δ = argmin Σ_P ‖T_p δ − 2v‖² + λ Σ_Q ‖T_q δ‖² + μ‖δ‖²`.
  - v is the edit's GRAFT answer vector, so 2v is the inserted layer's push.
  - λ = 1, and μ = 10⁻² × the mean diagonal of `Σ_P T_pᵀT_p`.
- **Size check, on P only:** δ is scaled by the smallest of 1, 1.5, 2 and 3 at which the new answer's first token beats the old one's on at least 7 of the 8, with δ written at the exact name.

**The name neurons:** layer 8, built as in revision 1. Each writes δ instead of a tag: down column `δ/silu(16)`, full size at the positives' median z. Slots: 64 at layer 8 only.

**Arms:**
- **D2:** δ written by a hook at the exact name (the name index).
- **E2:** δ written by the name neurons. No hook.

**Fixed now:**
- **E2 is a grafted model as good as the inserted layer** under the same bar as step 3: ES and PS within 0.05 of A's, NS within 0.01, neighbour leaks ≤ 0.05, same-subject leaks ≤ 0.10, at most 2 of the 133 known facts lost, at most 7 of the clean 391, drift ≤ 0.05.
- **Kill rule:** if E2's PS is under 0.50, or any of its leak or damage bars fails, key 1 in the weights is put aside for now. The grafted model ships as the hybrid: the edits in layer 20's own neurons, with the name index as a small hook that switches them on (step 2's B). What comes next is then my choice.

**A deviation in revision 2 (written after the first solves, before any edit was scored).** The first 8 edits' solves (log kept as `runs/slots/step3_rev2_push_target_stopped.log`) show that the push vector can't be reached from the name.
- The best δ has a median size of 13.6 units, where the MLP's own output at the name is 1 unit, and it still misses 86% of 2v on the training wordings.
- The target was too strict. Attention from the subject doesn't carry the push's own direction, and only the new answer beating the old one matters.
- Each solve also took about 40 s: 1,025 forward passes for each of 16 prompts.

The run was stopped. Changed, everything else as written:
- **The target:** for each training wording p, the new answer's first-token logit minus the old answer's must rise by `m_p = max(0, −gap_p) + 2`. For each other question q, the new answer's logit minus that of the model's own top token must not rise.
- **The solve:** `c = argmin Σ_P (g_p·c − m_p)² + λ Σ_Q (g_q·c)² + μ‖c‖²`, with g the measured change per basis direction. λ = 1, and μ = 10⁻² × the mean diagonal of `Σ_P g_p g_pᵀ`.
- **The basis:** δ is solved within the 128 directions that the transport map from layer 8 to layer 20 carries most strongly (its top right singular vectors, measured on one rewrite prompt per relation, as in revision 1). That is 129 forward passes per prompt.

## Result of step 3, revision 2: writing the edit at the name is weak, and changes the subject's other answers; the kill rule is met

Raw numbers: `research/results/slots/08_step3_rev2/` (with the stopped first attempt's log).

| | A, inserted layer | B, hybrid (step 2) | D2, written at the name by a hook | E2, written by name neurons (no hook) |
|---|---|---|---|---|
| ES; PS | 0.906; 0.727 | 0.906; 0.727 | 0.281; 0.266 | 0.266; 0.266 |
| NS (unedited 0.880) | 0.880 | 0.878 | 0.880 | 0.870 |
| Neighbour; same-subject leaks | 0.000; 0.036 | 0.002; 0.031 | 0.000; 0.001 | 0.002; 0.001 |
| Known facts lost (of 133; of 391) | 0; — | 1; 6 | 0; 5 | 4; 25 |
| Drift | 0.0000 | 0.0008 | 0.0000 | 0.0013 |
| Same-subject answers whose first two words changed | 0.016 | 0.055 | **0.408** | **0.410** |
| Neighbour answers whose first two words changed | 0.000 | 0.039 | 0.000 | 0.139 |

The last two rows were added after the run, from the saved replies. CounterFact's same-subject leak counts only replies that give the new answer, and it can't see a reply that changes to something else.

- **The solve is clean in the linear model and weak in the real one.**
  - The solved δ is 2 to 6 units at the median. In the linear model it reaches the whole needed margin on the training wordings, and moves the new answer by at most 0.06 on the subject's other questions.
  - Written into the model, it falls short. The size check needed the largest scale (3×) for 49 of 64 edits, and only 17 then win on 7 of 8 training wordings. Overriding a known fact needs a change at the name too large for the linear map to predict.
- **It changes what the model says about the subject:** 41% of the subject's other answers change in their first two words, against 1.6% for the inserted layer. This is the known weakness of writing into a subject's representation, here much worse.
- **The name neurons add damage of their own:** E2 loses 25 of the clean 391, against 5 for D2. A false fire now writes a vector of 2 to 6 units into another subject's representation.
- **By the kill rule (E2's PS 0.266 < 0.50, and the known-fact bars fail), key 1 in the weights is put aside for now.** The grafted model ships as the hybrid, step 2's B: the edits live in layer 20's own neurons, and the name index switches them on. What comes next is then my choice.

**Where the grafted model stands:**
- **In the weights, at no extra size:** the question detector (the gate rows), the push (the down columns) and the subject's exact name (name neurons at layer 8, separable for 64 of 64, with no false fires on held-out text).
- **Not solved:** joining the name to the question by weights alone. A mark carried to the end of the question and read there doesn't separate (three designs). Writing the edit at the name is weak, and it rewrites the subject.
- **Freeing slots:** zero damage is unreachable, but up to 256 freed neurons cost about the batching noise floor.

## Shipping the hybrid as the grafted model (written 2026-11-10, after I chose it, before the check ran)

**My choice:** ship the hybrid, step 2's B, as the grafted model, then try an attention channel for a weights-only key 1.

**What ships** (`cramschool.graft.grafted`, `Graft.export_grafted`, `load_grafted`):
- **Each fact form (plain and chat) gets one neuron** at the table's layer, chosen by the redundancy order and with the rest re-fitted, as in step 1c:
  - gate row: the form's question detector (`k·(w − (τ/c̄)·c)/δ`);
  - down column: the push `2·v/silu(k)`, spelled after a word (" Paris") for plain and at a reply's start ("Paris") for chat;
  - up row: **zero**.
- **The name switch** (`grafted_switch.py`, copied into the checkpoint folder, needing torch alone) sets a neuron's up value to 1 at the tokens after its subject is named, and to 0 elsewhere. B multiplied an up row of `c/c̄` by the name index's 0 or 1. Here the checkpoint itself holds 0, so a model loaded without the switch has the edits dormant instead of answering every subject's question with them.
- **The checkpoint:** the base files with the layer's three MLP matrices replaced. Same files, tensor names, shapes, dtypes and byte sizes.

**The check** (`scripts/grafted_check.py`): step 2's 64 edits, compiled by the package (2 neurons each, so 128 slots at layer 20), exported, and loaded back from disk with transformers.
- **Measured:**
  - CounterFact's measures with the folder's own switch, and without any switch;
  - the 133 and the clean 391 known facts, and drift, both ways;
  - the byte sizes, and that every tensor other than the three is unchanged.
- **Fixed now:**
  - **With the switch,** it ships if its ES and PS are within 0.05 of the inserted layer's (A, step 2), its NS within 0.01, neighbour leaks ≤ 0.05 and same-subject leaks ≤ 0.10, with at most 2 of the 133 lost, at most 7 of the 391, and drift ≤ 0.05.
  - **Without the switch,** its ES and PS must stay within 0.05 of the unedited model's, with the same damage bars.

## Result of the ship check: the edits work from a reloaded checkpoint; with both forms, the known-fact bar is missed by one

Raw numbers: `research/results/slots/09_grafted_check/`.

**The checkpoint:**
- 1,746,942,600 bytes, the base's size exactly;
- 752,393,024 parameters, as the base;
- 485 of the 488 tensors identical. Only layer 20's three MLP matrices differ: 128 gate rows, 128 up rows set to zero, and the re-fitted down matrix, which is most of it.

| | Unedited | A, inserted layer | Grafted, with the folder's switch | Grafted, no switch |
|---|---|---|---|---|
| ES; PS | 0.031; 0.070 | 0.906; 0.727 | **0.906; 0.727** | 0.031; 0.062 |
| NS | 0.880 | 0.880 | 0.878 | 0.878 |
| Neighbour; same-subject leaks | 0; 0 | 0.000; 0.036 | 0.000; 0.038 | 0.000; 0.000 |
| Known facts lost (of 133; of 391) | — | 0; — | **3; 8** | **3**; 5 |
| Drift | — | 0.0000 | 0.0012 | 0.0012 |

- **With the switch, the reloaded checkpoint answers as the inserted layer does:** ES, PS and generation are the same. In the saved examples, "The language of Ye Meri Life Hai is" gives "Dutch." (base: "a Hindi language."), while its neighbour "The original language of Chup Chup Ke was" keeps "the Malayalam language.".
- **Without the switch, the edits are dormant** (ES 0.031, as unedited).
- **By the rule, it doesn't ship as checked:** 3 of 133 known facts lost (bar 2) and 8 of 391 (bar 7), each over by one.
  - **The cause is the freeing.** The same 3 of 133 are lost without the switch. Giving each edit a chat-form neuron as well doubles the freed neurons to 128, against step 2's 64 (1 of 133).
  - Both counts are at the batching noise floor (1 of 133 and 7 of 391 at a single batching), but the bar was fixed before the run.

**Next, written before it ran: the plain-only configuration** (one neuron per edit, 64 slots, as step 2's B), checked again with the same script and bars. If it passes, it ships as the default. Chat-form neurons stay an option, documented with this check's cost.

## Result of the plain-only ship check: it ships

Raw numbers: `research/results/slots/09_grafted_check/` (`*_plain` for this check, `*_both_forms` for the one before).

| | Unedited | A, inserted layer | Grafted, with the switch | Grafted, no switch |
|---|---|---|---|---|
| ES; PS | 0.031; 0.070 | 0.906; 0.727 | **0.906; 0.727** | 0.031; 0.070 |
| NS | 0.880 | 0.880 | 0.878 | 0.878 |
| Says the new answer, reworded | 0.000 | 0.734 | 0.727 | 0.000 |
| Neighbour; same-subject leaks | 0; 0 | 0.000; 0.036 | 0.002; 0.033 | 0.002; 0.000 |
| Known facts lost (of 133; of 391) | — | 0; — | 1; 6 | 1; 3 |
| Drift | — | 0.0000 | 0.0008 | 0.0008 |

**Every bar is met, both ways.**
- **The checkpoint** is the base's size (1,746,942,600 bytes and 752,393,024 parameters), with 485 of 488 tensors identical.
- **With the folder's own switch,** the reloaded checkpoint answers as the inserted layer does.
- **Without it,** the edits are dormant and the model is the base model with 64 neurons freed.

**The package's default is now plain-only,** one neuron per fact. `forms=("plain", "chat")` adds a chat-form neuron per fact, at the cost measured above: 128 neurons freed and 3 of 133 known facts lost.

## Step 4: key 1 through an attention channel (written 2026-11-10, after shipping the hybrid, before it ran)

**The idea (my go):** the name neurons at layer 8 detect the exact name: 64 of 64 separable, and no false fires on held-out text. What failed was carrying a mark to the end of the question, since the model's own attention carried the tags weakly and mixed. Here, one attention head gets a dedicated channel.
- **The head:** among the full-attention layers between 8 and 20 (11, 15 and 19), the query head whose attention from the prompt's last token to the subject's last name token has the largest 10th percentile. It is measured over 8 training wordings per edit. Queries and keys are untouched: the head already looks at the name.
- **The channel:** that head's key-value head's value projection gains a rank-64 term `γ Σ_s v_s r_sᵀ`.
  - r_s reads tag s, after the layer's input norm.
  - v_s is subject s's code: one of the 64 least-varying directions of that head's values on ordinary text.
  - The 4 query heads that share the key-value head carry the code to the end of the question, in proportion to their attention to the name.
- **The tags:** written at layer 8, at the name, along the 64 least-varying directions of the residual entering the channel's layer, 2 units in size. At a position without a tag, the channel reads only the residual's natural projection on those directions.
- **γ, chosen on the training wordings only, before any test prompt is scored:** the first of 1, 2 or 4 times (the head's median value norm ÷ the code's norm at a tagged name) at which the tag readings separate by a median of at least 3 standard deviations, with the tags written by the hook.
- **The fact neuron (layer 20):** step 2's B, with the name index replaced by the code's reading.
  - gate: the question detector; down: the push.
  - up: t, the reading. It is the tagged-minus-untagged mean, whitened, scaled to 1 at the tagged prompts' median and 0 at the negatives' 99th percentile. Edits that separate by under 3 sd get no neuron.
- **Arms:**
  - D3: tags written by a hook at the exact name;
  - E3: tags written by name neurons, with no hook.
  - Both use 64 slots at layer 8 and 64 at layer 20, compensated.

**Measured:**
- CounterFact's measures, the known facts (133 and 391), and drift;
- the same-subject and neighbour answers whose first two words change (the check added in step 3, revision 2);
- the readings' separations;
- the damage of the freed slots with the channel in place but no edits.

**Fixed now:**
- **The channel works** if all of these hold for E3:
  - ES and PS within 0.05 of A's, and NS within 0.01;
  - neighbour leaks ≤ 0.05 and same-subject leaks ≤ 0.10;
  - same-subject answers changed (first two words) ≤ 0.10;
  - drift ≤ 0.05;
  - known facts lost beyond those of the freed slots with the channel: within noise (≤ 2 more of 133, ≤ 7 more of 391).
- **It ships as a weights-only grafted model** if, in addition, the totals are within the CounterFact bar: ≤ 2 of 133 and ≤ 7 of 391. Revision 1 measured 3 of 133 for the freed slots at layers 8 and 20 alone, so this can fail on capacity even if the channel works.
- **Kill rule:** if E3 fails the channel bar, the attention channel is put aside, and the shipped grafted model stays the hybrid with its name switch.

## Result of step 4: the attention channel fails, and the kill rule is met

Raw numbers: `research/results/slots/10_step4_channel/`. The diagnostic after the run is `scripts/diag_channel_read.py`.

- **The channel head:** layer 15, query head 1 (key-value head 0). Its attention from the end of the question to the name is a median of 0.11 (10th percentile 0.03). No head in layers 11, 15 or 19 looks at the subject's last token more strongly: the next best have medians of 0.05.
- **The code never became readable at layer 20.** At γ of 1, 2 and 4 times the head's median value norm, the tagged prompts' readings sat below the negatives' 99th percentile: median separation −1.8, −1.7 and −1.6 standard deviations. One edit of 64 passed the guard, in both arms.
- **The channel broke the model by itself.** With the freed slots and the channel at the chosen γ (4×), and no edit placed, all 133 known facts were lost, with drift 5.8. Every arm followed: D3 and E3 lost 133 of 133, and changed the first two words of 99.9% of the answers.

**Why, measured after the run on 13,692 tokens of fitting text** (the residual entering layer 15):
- **The tags were far too large for the residual.** Its median rms is 0.11 per dimension, a norm of about 3.5. A 2-unit tag (1.46) is about 40% of it: it replaces the subject's representation instead of marking it. The unit came from layer 8's MLP output at the names, not from the residual's own size.
- **The tags' directions are quiet in the residual (std 0.017, against 0.089 for random directions), but not after the layer's input norm.** The norm's weights on those directions are about 26 times the average (|w·t| 5.4, against a mean |w| of 0.20). After the norm, ordinary text varies along the channel's read vectors with a std of 0.46 (random directions: 0.97). At a gain that makes a tagged name's code 4 times a typical value, every ordinary token gets a code a few percent of that, in every value of the head.
- **The head looks at the name too little** for a code written there to arrive, at the end of the question, above the noise the channel itself adds.

**By the kill rule, the attention channel is put aside.** The shipped grafted model stays the hybrid, the edits in layer 20's own neurons with the name switch.

**What a further attempt would need, recorded here and not run:**
- tags sized to the residual (a few percent of its norm, not 40%);
- read directions chosen quiet after the input norm;
- a head made to look at the name, by editing its queries and keys. The per-head normalization of queries and keys makes that last one hard.

## An independent review, and corrections (2026-11-10)

At my request, a separate agent reviewed the grafted-model work from the code and the raw results, without the conclusions above. Each finding used here was checked again before being accepted. **Earlier entries stay as written; this section corrects them.**

**1. Step 4's failure came from the implementation, so it says nothing against attention channels.** The cause given in "Result of step 4" is wrong.
- **The tag directions carry a large mean.** They were the least-varying directions of the residual, from a *centred* covariance. Along them the residual's mean is 2.3 times its spread (0.039 against 0.017, checked again on 120 fitting paragraphs). The channel's read has no bias, so every ordinary token got a code: a norm of 15.0 per unit γ over the 64 codes, as large as a tagged name's own code (16.6). That alone explains both the model breaking with no edit placed and the reading never separating.
- **The input norm was modelled wrongly.** Qwen3.5's RMSNorm computes `x/rms · (1 + w)`, not `x/rms · w`. With the correct form the background is 8.9 against the tag's 16.9: smaller, still far too large.
- **The "26 times the average weight" was an artifact.** It compared `Σ|t_i||w_i|` with the mean |w|, and any unit vector in 1,024 dimensions scores about 25 on that ratio.
- **The rule's damage clause was also wrong:** "beyond the freed slots with the channel" subtracted the channel's own damage, so it passed (`known_beyond_freed: True`) for a model that had lost 133 of 133.
- **What stands:** the head's attention to the name is low (median 0.11), and the attention output gate was not modelled.

**2. Step 2's C was not a fair test of reading the subject at layer 20.**
- **The reading works at the question's last token:**
  - separation median 8.6 standard deviations, minimum 2.1;
  - reworded prompts read a median u of 0.96;
  - neighbours read a median of −0.06, a 95th percentile of 0.75 and a maximum of 1.39.
- **What broke the model was the up row:** a linear reading applied at every token, fitted with no ordinary-text negatives and no threshold at the largest negative, unlike the name neurons, which had both and worked.
- **A confound:** its positives had the build's prefixes and context sentences, its negatives none.
- "A reading of the model's own state at the end of the question can do neither" over-generalised from that one unbounded design. Neighbours' tail (up to 1.39) is a real problem; separating at all is not.

**3. The shipped name switch had reset bugs, now fixed.**
- **What it did:** it started a new text only when a call had more than one token or a new batch size. So a new one-token text inherited the subject named before it. A cached multi-token continuation lost the names already read. A call with embeddings only left stale state, and calling the inner model directly crashed.
- **Now:**
  - a call whose cache is empty starts a new text;
  - the hooks sit on the decoder stack, so any caller is seen;
  - a call with nothing read leaves every fact dormant.
  - Beam search is documented as unsupported.
- The inserted layer had the same rule and the same crash on embeddings, and gets the same fix. Tests cover each case (245 pass).

**4. A compiled neuron is not the inserted layer's push in every way.**
- **Its push is graded and unbounded in z.** δ is half the gap to the positives' median, so a typical training positive (z = 2) gets twice the inserted layer's push.
- **It acts at every position after the name,** with no one-answer rule, no veto and no continuation pushes.
- **Only the answer's first token is written.**
- **The match with the inserted layer is shown on CounterFact's plain prompts only,** mostly one-token answers, and without the veto the twin relations' questions got the new answer more often (0.68 against 0.57).
- `export_grafted` now uses the table's own push multiple. It warns that `looser_start` and `end_quantile` aren't compiled, and warns above 256 neurons.

**5. Methodology:**
- **The ship check is not an independent test.** N = 64 was chosen in step 1c with the 133 known facts and then judged on them. After the 128-neuron check failed, the configuration changed to the one step 2's B had already passed with. A fresh set of edits, not yet seen, is the real check.
- **The name neurons' false fires:** none on held-out text, but 22 of 640 neighbour prompts had one. Revision 2 blamed those for E2's 25 of 391.
- **"Result of step 3, revision 1" mixed statistics:** "on the same prompts without the tag it reaches 0.58" is a median over edits of each edit's largest reading; the largest was 73, next to the "up to 195" quoted for other edits' tags.
- **Revision 2's "clean in the linear model" was vacuous:** 16 equations in 128 unknowns with a small ridge always fit exactly.
- **The bar of 7 of 391 comes from changing the batching.** At identical batching the measurement has no noise, so treating a weight edit's flips like a batching change's is a judgement, not a measured property.

**What the review leaves standing:**
- key 2 and the push compile into neurons;
- the name neurons detect the exact name;
- the natural attention carries tags too weakly (revision 1);
- writing the edit at the name rewrites the subject (revision 2): its non-linearity is real, and the gap measurements agree with float32 (correlation 0.99).

**The review's verdict:** the headline hasn't moved for the weights-only arms (PS 0.19 to 0.30, against 0.727). The shipped model is the inserted layer re-housed. But two of the four negative results (2C and 4) are not refutations.

## Step 5: a bounded subject reading at layer 20 (written 2026-11-10, after the review, before it ran)

**Why:** the review found that step 2's C failed through its construction, not its reading. At the question's last token the subject reading separated by a median of 8.6 standard deviations. It broke the model because it was applied unbounded at every token. This step tests the reading properly, with no transport at all: one layer, no hook.

**The subject reading u, one per edit, at layer 20's MLP input:** an LDA, built as the name neurons were.
- **Positives:** the relation's training wordings about the subject at the last token, with the build's prefixes and context sentences (as in step 2).
- **Negatives, all with prefixes drawn from the same eight:**
  - the same wordings about 12 invented names and 24 real subjects of the same CounterFact relation, none named in any test prompt, two prefixes each;
  - the other 63 edits' positives;
  - every token of 120 fitting paragraphs.
- **Threshold at the largest negative,** scaled so u = 0 there and 1 at the positives' median. An edit whose positives' median doesn't clear it gets no neuron.

**The fact neuron, a clipped pair, so its output can't exceed the inserted layer's push:**
- gate `g = k·(z_q + M·(u − 1))`, with z_q the question detector as in step 2, k = 8 and M = 4. Where u ≤ 0 (every training negative), the question detector would need z > 4 to open it; its positives' median is 2.
- The two neurons are `silu(g)` and `silu(g − 8)`, both with up row `c/c̄`. Their difference times `2v/8` writes at most 2v, the inserted layer's push, from `g ≥ 8`.
- 128 slots at layer 20: the redundancy order, compensated.

**Measured:**
- before scoring: the share of held-out text tokens where any pair writes more than a tenth of its push;
- the readings' separations, and u on the test prompts (reworded, neighbours);
- CounterFact's measures, with the first two words changed on same-subject and neighbour prompts;
- the 133 and the 391 known facts, and drift: the edits' and the freed 128 slots' alone.

**Fixed now:**
- **The reading works as key 1** if, against step 2's A:
  - ES and PS are within 0.05, and NS within 0.01;
  - neighbour leaks ≤ 0.05 and same-subject leaks ≤ 0.10;
  - same-subject first two words changed ≤ 0.10;
  - drift ≤ 0.05;
  - known facts lost no more than the freed slots' plus noise (2 of 133, 7 of 391).
- **It ships as a weights-only grafted model** if, in addition, at most 2 of 133 and 7 of 391 are lost in all.
- **Kill rule:** if it fails the first bar, the subject reading at layer 20 is put aside, and the remaining weights-only route is the attention channel redone (reads quiet after the norm and without a mean, the `(1 + w)` norm, the channel alone first held to at most 2 of 133).

## Result of step 5: the bounded subject reading fails on its merits; the kill rule is met

Raw numbers: `research/results/slots/12_step5_subject20/`.

| | A, inserted layer | Step 5 (no hook) | The 128 slots freed alone |
|---|---|---|---|
| ES; PS | 0.906; 0.727 | 0.391; 0.352 | — |
| NS (unedited 0.880) | 0.880 | 0.819 | — |
| Neighbour; same-subject leaks | 0.000; 0.036 | 0.058; 0.069 | — |
| First two words changed: same subject; neighbours | 0.016; 0.000 | 0.116; 0.181 | — |
| Known facts lost (of 133; of 391) | 0; — | 48; 57 | 3; 5 |
| Drift | 0.0000 | 0.57 | 0.0012 |

**This time the reading itself fails.**
- **Only 32 of 64 edits** had their positives' median above the largest of about 27,000 negatives (median separation 5.5 sd).
- **On those edits' test prompts, the threshold set on the training negatives doesn't hold:**
  - reworded prompts read a median u of 0.89, but a 10th percentile of −1.57;
  - neighbours read a median of −1.77, but a 95th percentile of 1.38 and a maximum of 4.97, and 10% of them read above 0.5.
- **A subject reading fitted on the build's wordings doesn't carry over to CounterFact's own phrasings,** at either end: it misses reworded prompts and admits neighbours.

**The neuron pair was not as bounded as written.**
- The clip bounds `silu(g) − silu(g − 8)`, but the up row `c/c̄` reads about 1 on ordinary text, not on every token. On held-out text a pair wrote up to 4.0 pushes.
- The gate's `M·(u − 1)` term opens wherever u's upper tail is large: 50 of 4,456 held-out tokens had a pair write more than a tenth of a push. That is the damage: 48 of 133 known facts lost, against 3 for the freed slots alone.

**By the kill rule, the subject reading at layer 20 is put aside.** Counting step 2's C, two designs of it have now failed: one through its construction, and this one on its merits. The weights-only route that remains is the attention channel redone.

## The shipped grafted model on fresh edits (written 2026-11-10, after step 5, before it ran)

**Why:** the review pointed out that the ship check was not independent. N = 64 was chosen in step 1c with the same 133 known facts, and the configuration was settled after the first check failed. This is the independent test.

**The edits:** a fresh draw of 3 edits for each of the same 21 relations (seed 20261120), excluding every earlier draw: all 16 `facts.json` under `research/results/counterfact/`. The inserted layer is built and scored on them with the editor recipe, unchanged (`scripts/counterfact_breadth.py`, editor relations, the narrow twins, context 4, fast). Then the grafted model is compiled from the same parts by the package, unchanged: plain-only, one neuron per edit. It is exported, reloaded from disk and scored with and without its switch (`scripts/grafted_check.py`).

**Fixed now, as in the ship check, against the inserted layer on the same fresh edits:**
- **With the switch:**
  - ES and PS within 0.05, and NS within 0.01;
  - neighbour leaks ≤ 0.05 and same-subject leaks ≤ 0.10;
  - at most 2 of the 133 and 7 of the 391 known facts lost;
  - drift ≤ 0.05.
- **Without the switch:** ES and PS within 0.05 of the unedited model's, with the same damage bars.
- **If it fails,** the shipped claim is withdrawn to "reproduces the inserted layer on the edits it was chosen on".

## Result of the fresh-edit check: the shipped grafted model holds, at the edge of the damage bar

Raw numbers: the inserted layer's run is in `research/results/counterfact/grafted_fresh/`, the grafted check in `research/results/slots/13_grafted_fresh/`. There were 63 fresh edits (seed 20261120, 1,036 earlier edits left out). 61 were compiled; the other 2 had no usable plain detector.

| | Unedited | Inserted layer | Grafted, with the switch | Grafted, no switch |
|---|---|---|---|---|
| ES; PS | 0.000; 0.064 | 0.889; 0.786 | **0.857; 0.802** | 0.000; 0.064 |
| NS | 0.830 | 0.830 | 0.827 | 0.827 |
| Says the new answer: rewrite; reworded | 0.000; 0.008 | 0.905; 0.818 | 0.873; 0.786 | — |
| Neighbour; same-subject leaks | 0; 0 | 0.000; 0.040 | 0.003; 0.024 | 0.003; 0.000 |
| Known facts lost (of 133; of 391) | — | 0; — | **2; 7** | 2; 3 |
| Drift | — | 0.0000 | 0.0008 | 0.0008 |

**By the rule fixed before the run, it ships, both ways.**
- With the switch, ES is 0.032 below the inserted layer and PS 0.016 above it, with leaks within their bars.
- Without the switch, the edits are dormant.
- The checkpoint is again the base's size, with 485 of 488 tensors identical.

**The known-fact losses sit exactly at both limits:** 2 of 133 and 7 of 391. The 2 of 133 are lost without the switch too, so they come from freeing the neurons. There is no margin left on damage at about 64 facts per layer.

**So the shipped claim stands on edits it was never chosen on:** the grafted model answers as the inserted layer does. Of its limits, the one that binds is damage, not accuracy.

## A correction to the fresh-edit check, and step 6: room from far more text (written 2026-11-10, before step 6 ran)

**The correction.** `scripts/diag_known_flips.py` lists each known fact the fresh grafted model loses, with and without its switch, at the scorers' own batching ([log](../research/results/slots/14_capacity/00_known_flips_fresh.log)).
- **The 4 extra losses of the 391 with the switch on are the edits themselves.** The 391 hold 7 prompts that name a fresh edit's subject. The 4 lost are edited facts, answered with their new answers:
  - Metropolitan France's capital (Singapore);
  - Jonathan Littell's mother tongue (Greek);
  - the Royal Danish Library's location (Melbourne);
  - Panagiotis Lafazanis's native language (French).
- **So the fresh model's damage is 2 of 133 and 3 of 391, the same with and without the switch.** All of it comes from freeing the neurons. The 2 of 133 are "The capital of Cuba is" and "The capital of the United States is".
- **The same flaw touches every earlier "of 391" with working edits:** 6 of the 391 prompts name one of step 2's 64 subjects. From now on, a known-fact prompt that names an edited subject is left out of the counts.

**Why step 6.** The facts cost nothing on questions about other subjects, because the switch gates them by name. So the number of facts a layer can hold is the number of neurons it can give up within the bar.
- Step 1c measured that with second moments from 27,640 tokens for 3,584 neurons, under 8 tokens per neuron.
- The re-fit's ridge was chosen at the top of its grid (0.1) at every N.
- Both are signs of an estimate starved for data, not of a layer without room.

**The change, one arm: the second moments from about a million tokens of ordinary text.**
- **Fitting text:** the 240 fitting paragraphs, plus MS MARCO passages (the BEIR corpus, drawn with seed 20261110) joined into 512-token chunks, about 1M tokens in all.
- **Spare text, for choosing the ridge only:** the 75 spare paragraphs plus separate MS MARCO chunks, about 100k tokens.
- **The ridge grid** is extended to 1e-8 to 1.
- **Everything else as shipped:** layer 20, the redundancy order, the closed-form re-fit.

**Part 1, freeing only (no facts placed), to choose N:**
- **Configurations:** N = 64, 128, 256, 512 and 1,024, each with the shipped statistics and with the large ones.
- **Measured:**
  - held-out KL per token on the package's held-out paragraphs;
  - drift;
  - known facts lost of the 133, and CounterFact facts lost of the 391.
- **Batching:** the same for every configuration.
- **N\*** is the largest N at which the large statistics lose at most 2 of 133 and 7 of 391.

**Kill rule:** if N\* < 128, more text doesn't buy room. The estimate isn't the limit, layer 20 holds about 64 facts, and the next route is spreading the facts over several layers, with its own plan.

**Part 2, the confirmation at N\* (only if N\* ≥ 128):**
- **The edits:** N\* fresh edits, drawn as in the fresh-edit check (seed 20261121, every earlier draw left out).
- **The build:** the editor recipe, unchanged. The edits are compiled with the large statistics, exported, reloaded from disk, and scored with and without the switch.
- **Known facts:**
  - the 133;
  - the 391, without prompts that name an edited subject;
  - a fresh 400 drawn from the same screen, none among the 400 and none about an edited subject.
- **It ships as the new capacity if, against the inserted layer on the same edits:**
  - ES and PS are within 0.05, and NS within 0.01;
  - neighbour leaks are ≤ 0.05 and same-subject leaks ≤ 0.10;
  - at most 2 of the 133, 7 of the 391 and 7 of the fresh 400 are lost;
  - drift is ≤ 0.05;
  - without the switch, the edits are dormant.
- **If it fails,** the capacity claim stays at about 64 facts per layer.

## Result of step 6, part 1: N\* = 256, so the kill rule is not met

Raw numbers: `research/results/slots/14_capacity/01_room_from_text/`. The MS MARCO text came to 1,954 fitting chunks (1.0M tokens, 12,659 passages) and 196 spare chunks. Layer 20, freeing only, every configuration at the same batching:

| Freed | Shipped statistics: lost of 133; of 391 | Drift | Held-out KL | Large statistics: lost of 133; of 391 | Drift | Held-out KL | Slots shared |
|---|---|---|---|---|---|---|---|
| 64 | 2; 2 | 0.00077 | 0.00070 | 1; 4 | 0.00066 | 0.00076 | 45 |
| 128 | 4; 5 | 0.00120 | 0.00101 | 2; 3 | 0.00088 | 0.00120 | 83 |
| 256 | 3; 3 | 0.00194 | 0.00175 | **1; 4** | 0.00127 | 0.00210 | 173 |
| 512 | 6; 11 | 0.00369 | 0.00355 | 4; 12 | 0.00248 | 0.00477 | 349 |
| 1,024 | 11; 12 | 0.00798 | 0.00797 | 3; 17 | 0.00573 | 0.01146 | 726 |

- **By the rule, N\* = 256:** the largest N at which the large statistics lose at most 2 of 133 and 7 of 391. With the shipped statistics the same rule gives 64.
- **The shipped rows reproduce step 1c** (256: 3 and 3; 512: 6 and 11; 1,024: 11 and 12). The extended ridge grid picked 0.03 at 64 and 128, where step 1c's grid stopped at 0.1.
- **The large statistics' ridge falls to 0.001–0.01,** as a better-estimated matrix should allow. A third of the slots they free differ from the shipped order's.
- **The evidence is mixed, not a clean win:**
  - losses of the 133 fall at every N, and drift falls by about 30%;
  - losses of the 391 don't fall (4 against 2 at 64, 17 against 12 at 1,024);
  - KL on the package's held-out paragraphs rises at every N. Most of the new text is web passages, so the re-fit now serves them over paragraph-style text.
- **So part 2 decides.** Its fresh 400 known facts were not used here; the 133 and the 391 were.

**Part 2's draw, a forced change.** Two of the 21 relations have no usable records left after every earlier draw: P106 (0) and P407 (1). Some others have few (P176 4, P276 7, P159, P937 and P39 9 each).
- The draw is therefore the other 19 relations, 16 each, with `--at-least 3`: a relation with fewer usable records gives all it has. That makes about 250 edits, at most N\*.
- Everything else is as fixed: seed 20261121, every earlier draw left out, the editor recipe unchanged.
- **One more relation had to drop when the draw ran:** drawn jointly, P176 has 1 usable record, under `--at-least 3`. The draw is 18 relations, 16 each: 242 edits.

## Result of step 6, part 2: one bar missed by one knife-edge fact, so the capacity claim stays at about 64

Raw numbers:
- the inserted layer's run: `research/results/counterfact/capacity_242/`;
- the grafted check: `research/results/slots/14_capacity/02_confirm_256/`;
- the fresh known set: `known_fresh_400.json`.

**A deviation:** the editor run stopped at edit 229, while printing a subject name with "ō": the chain had not set the output encoding. It was run again with the same command and `PYTHONIOENCODING=utf-8`. The 228 parts already written were reused after their subject, relation and answer were checked against the draw (`run_edits` does this). The other 14 were built anew, and both scorings were redone.

| 242 fresh edits, 237 compiled | Unedited | Inserted layer | Grafted, with the switch | Grafted, no switch |
|---|---|---|---|---|
| ES; PS | 0.025; 0.068 | 0.864; 0.738 | **0.884; 0.746** | 0.025; 0.070 |
| NS | 0.864 | 0.860 | 0.860 | 0.863 |
| Neighbour; same-subject leaks | — | 0.003; 0.035 | 0.004; 0.018 | 0.000; 0.000 |
| Known facts lost: of 133; of the old set (376 answered); of the fresh set (396 answered) | — | 0; —; — | **3**; 4; 1 | 3; 4; 1 |
| Drift | — | 0.0000 | 0.0012 | 0.0012 |

- **The checkpoint:** the base's size, with 485 of 488 tensors identical. The re-fit's ridge was 0.01, with 21% of the freed output left.
- **Prompts left out:** 19 of the old set named an edited subject; none of the 133 or the fresh set did.

**By the rule fixed before the run, it does not ship: one bar fails, by one fact.** 3 of the 133 are lost, against a bar of 2.
- Every other bar passes: ES and PS are above the inserted layer's, leaks are within their bars, 4 of the old set and 1 of the fresh set are lost, and drift is 0.0012.
- The losses are the same with and without the switch: all of them come from freeing the neurons.

**What the three lost facts are** (`scripts/diag_known_margins.py`, `03_known_margins.*`):
- "Skype is owned by", "The capital of Russia is" and "The capital of Denmark is".
- **Russia and Denmark are knife-edge:** the unedited model's answer leads at the first token by 0.12 logits, one bf16 rounding step at that size. Skype's answer is not its first token at all (margin −0.12).
- **No systematic shift.** The margin change from unedited to grafted:
  - mean +0.009 on the 57 capital questions and +0.017 on the other 76;
  - mean −0.006 and −0.001 on the two CounterFact sets;
  - up about as often as down on every set (capital questions: 16 down, 26 up, 15 the same);
  - 5th percentile −0.125, one rounding step.
- **The 133 hold 57 capital questions, 11 of them within 0.5 logits.**
  - "The capital of Denmark is" was lost in all ten of part 1's configurations, including 64 neurons with either statistics.
  - Batching alone, with no weight changed, loses "The capital of Cuba is" (step 1c's noise check).
  - At this resolution, the 133's count can't tell 64 freed neurons from 237. The shipped model met the bar with 1 lost on step 2's edits and exactly 2 on the fresh ones.

**What stands:**
- **The capacity claim stays at about 64 facts per layer,** as fixed.
- **The grafted model at 237 facts answers as the inserted layer does** on edits never used for any choice. On a fresh known set, never used for any choice, it lost 1 of 396.
- **The large statistics are kept in the package as an option** (`grafted.text_moments`, `compile_layer(extra_moments=...)`, `Graft.export_grafted(texts=...)`). In part 1 they cut the 133's losses and drift at every N. The bar's verdict didn't change.
- **Not adopted, left for a decision:** a damage bar founded on systematic shift instead of a count of flips. For example, each set's mean margin change, with every flip counted where the unedited margin was robust. It would have to be fixed before a run, and checked on a known set not used so far.

## Probes before a new plan (2026-11-14)

**Why:** I wanted the best course of action against the open limits. Five things were measured first, with forward passes only. **No rule was written before them, so they decide nothing by themselves.** They are the reason for the two decisions in the next sections.
- Scripts: `scripts/diag_grafted_bench.py`, `diag_first_token.py`, `diag_latch.py` and `diag_heads_bus.py`.
- Raw numbers: `research/results/slots/15_probes/`.

**1. The grafted model on the project's own bench of new facts.** The saved table of bench40e's 40 facts, added through the API (`runs/api_facts/bench40e.graft`, on which the inserted layer scored 32 of 40), was compiled with both forms into 80 neurons and scored with the bench's checks and the name switch.

| bench40e, 40 facts in one table | Inserted layer | Grafted model |
|---|---|---|
| Facts passing | 32 | **0** |
| Recall, median (mean) | 0.83 (0.80) | 0.00 (0.05) |
| Replies holding the whole answer, of 480 | 386 | 24 |
| Replies starting with the answer's first token | — | 331 |
| Same-company leak; another fact's answer; known facts lost (the most, on any fact) | — | 0; 0; 1 |

- **Cause: a fact neuron writes the answer's first token only.** 39 of bench40e's 40 answers are 2 to 7 tokens long: "Ingrid." came out for "Ingrid Myrvold", and "Ferrari" for "Ferrara" at a reply's start. Of step 6's 242 CounterFact answers, 237 are one token.
- **So "the grafted model answers as the inserted layer does" has been shown for one-token answers only.**
- **The first token is also lost more often than in the inserted layer.** The detector's score z at the prompt's last token, for each recall prompt:

| z at the prompt's end | ≤ 0 | 0 to 0.25 | 0.25 to 0.5 | 0.5 to 1 | 1 to 2 | 2 to 3 | over 3 |
|---|---|---|---|---|---|---|---|
| Plain prompts (320) | 57 | 9 | 20 | 27 | 104 | 83 | 20 |
| of them, the first token landed | 0 | 0 | 13 | 23 | 92 | 72 | 15 |
| the inserted layer gave the whole answer and the grafted model's first token did not land (39) | 5 | 9 | 7 | 1 | 3 | 9 | 5 |
| Chat prompts (160) | 10 | 4 | 13 | 18 | 88 | 27 | 0 |
| of them, the first token landed | 0 | 0 | 7 | 18 | 68 | 23 | 0 |
| inserted layer whole, grafted first token missing (25) | 2 | 3 | 6 | 0 | 12 | 2 | 0 |

  - **A neuron's push grows with z** (step 2 and the review said so). Below z = 0.5 it is under half the inserted layer's and often too weak: 25 of the 64 misses. From z = 2 it is twice the inserted layer's or more, and the reply garbles ("Called?", "2024 A: 2024"): 14 of the plain misses.
  - **A neuron acts at every token after the name,** the inserted layer only at the last position of a call. The 12 chat misses at z between 1 and 2 have a moderate push and still an unchanged or garbled reply ("red red red"). A chat prompt has five or more tokens between the question and the reply's start. Whether pushes at those tokens are the cause was not measured directly.
  - **If every first token that landed had been completed, about 22 of 40 facts would reach the recall bar.** Completing answers alone would not bring the grafted model to the inserted layer's level.

**2. Known facts at 237 facts, by margin** (from step 6's stored `03_known_margins.json`; the margin is the answer's first token's lead over the best other token at the prompt's end):

| Set | Answered at the first token, unedited | Flipped to wrong | of them with an unedited lead ≤ 0.25 | Flipped to right | Lead ≥ 1.0: lost of how many | Mean margin change; largest |
|---|---|---|---|---|---|---|
| The 133 | 81 | 1 | 1 | 0 | 0 of 41 | +0.013; 0.375 |
| Step 1c's 400 | 229 | 1 | 1 | 2 | 0 of 152 | −0.006; 0.375 |
| Step 6's fresh 400 | 215 | 3 | 3 | 1 | 0 of 124 | −0.001; 0.375 |

Every flip sits within 0.25 logits of a tie, and none of the 470 prompts with a larger lead flipped.

**3. A latch in a Gated DeltaNet head** (`diag_latch.py`).
- **A fact about the model that the earlier steps did not use:** only layers 3, 7, 11, 15, 19 and 23 are full attention. The other 18 are Gated DeltaNet layers. Each has 16 heads of width 128, and each head is a running memory: it writes a value under a key at a token, with its own write gate and decay, and reads it back at later tokens through a norm and a per-dimension output gate. Queries, keys and values pass a 4-token convolution and SiLU first. Step 4 tried the full-attention layers only.
- **The mechanics work.** One head of layer 20 was set by direct weight writes: write only where a tag is present, no decay, a fixed key and query. Tags were written by a hook, at 5% of the residual's norm, in the quietest uncentred directions of the layer's input.
  - With no tag, over 400 tokens: the head's output stayed under 0.004, and the write gate under 0.00001.
  - A tag at token 10: silent before it; after it, its subject's code read 7.3 to 9.0 (ideal 8.3) for 390 tokens, every other code at most 0.14.
  - A second tag at token 200 replaced the first (which then read 0.0000004).
  - It held through generation with the cache, and in a left-padded batch. A paragraph's first token is no problem: the constant direction reads 0.97 to 1.08 there.
- **But the residual stream is no clean place for a tag.**
  - **No direction is quiet.** The quietest uncentred directions at layer 20's input read 0.21 to 0.43 of a typical direction on the fitting text and 0.40 to 0.50 on held-out text, with single tokens up to 11.5. A tag at 5% of the norm is 4 to 9 standard deviations.
  - **The latch wrote falsely** once in 21,851 tokens of ordinary text, questions and chat prompts (write gate 0.94). At a name's token, another subject's code direction read up to 2.1 times a tag, so stored codes came out mixed: the second tag's code read 4.2 instead of 8.3.
  - **A tag does not survive depth.** Written at one token and read at layer 20's input at the same token, as a share of what was written: from layer 19, 1.01 (median); from layer 16, 0.35; from layer 8, 0.00, with single prompts between −1.15 and above.
  - **A code added at every later token is as harmful in the quietest directions as in random ones:** at 5% of the norm, KL 0.006 and 4% of next tokens changed on known-fact prompts (random: 0.004 and 3%); at 10%, 0.024 and 9%.

**4. The cost of a private channel, and of one head** (`diag_heads_bus.py`; 533 known-fact prompts, 294 answered at the first token and 169 of them with a lead of 1.0 or more; every setting at the same batching):

| | Held-out KL per token | Flipped to wrong (to right) | Lead ≥ 1.0 lost; smallest such lead after | Mean margin change |
|---|---|---|---|---|
| 64 neurons freed at layer 20, the shipped method | 0.00067 | 3 (0) | 0; 0.94 | −0.003 |
| 256 | 0.00173 | 6 (1) | 0; 0.81 | −0.028 |
| 512 | 0.00351 | 8 (4) | 0; 0.50 | −0.030 |
| 8 directions silenced in the whole model | 0.0374 | 26 (3) | 0; 0.12 | −0.234 |
| 16 directions | 0.0632 | 47 (2) | 4 | −0.382 |
| 8 single coordinates | 0.0288 | 33 (4) | 2 | −0.219 |
| One head freed, the other heads' output weights re-fitted: layer 18 head 12; layer 17 head 8; layer 16 head 13; layer 20 head 3 | 0.0011; 0.0013; 0.0013; 0.0014 | 3 (4); 2 (3); 2 (2); 1 (2) | 0 each; 0.81 to 0.94 | +0.005; −0.000; −0.021; +0.009 |

- **Silencing directions** means taking them out of every weight that writes to the residual stream, as directional ablation does, so that a tag there would be read against silence. Pooled over all layers, the stream's quietest directions still read 0.38 of a typical one, and its quietest coordinates 0.59. **Eight of them cost about 55 times what freeing 64 neurons costs. On this model the stream has no room to spare.**
- **One head is affordable:** between the cost of 64 and of 256 freed neurons. Switched off with no re-fit, a head costs a KL of 0.0015 to 0.034 (median 0.0056 over the 64 heads of layers 16, 17, 18 and 20).

**5. An "old answer" cue for recall** (from stored replies, no run). On reworded prompts the unedited model itself says the old answer in 40% of its replies (step 6's 242 edits) and 36% (the 63 fresh edits). The inserted layer says the new one in 74% and 82%. A cue that also fired wherever the model was about to say the old answer would reach 81% and 84%, and would double the same-subject leak (4% of those replies hold the old answer). Not worth building.

**What the probes leave for a subject key in the weights:** a head can hold the subject across tokens, and one head is affordable. But nothing can be passed through the residual stream on the way, neither a tag to the head nor its code to a fact neuron. The design that remains keeps a fact inside one head: its value rows read the name, its memory carries it, its output gate rows read the question, and its output columns push. It is not built. The beta's stored reading of names by layer (`research/results/bench/subject_site.json`) has look-alike AUC 0.98 at layer 8, 0.96 at layer 16 and 0.95 at layer 20.

## The damage bar, re-founded on firmly known facts (my decision, 2026-11-14; written before its check ran)

**The decision.** Step 6 left it open: the count bar (at most 2 of the 133 known facts lost) turns on answers that lead by one rounding step, and can't tell 64 freed neurons from 237. I adopted a bar on **firmly known facts**. It was fixed after probes 2 and 4 above were seen, so it must be checked on a known set that nothing has used.

**Definitions,** for a set of known-fact prompts with their accepted answers, and one batching used for the unedited and the edited model alike:
- **margin:** at the prompt's last token, the logit of an accepted answer's first token (spelled after a space; the best over the accepted answers) minus the largest logit of any token that starts no accepted answer;
- **firmly known:** the unedited model answers the prompt in the generation check, and its margin is at least 1.0;
- **knife-edge:** answered, with a margin under 1.0.

**The bar. An edited model passes on a known set if:**
1. **no firmly known fact is lost:** each is still answered in the generation check, and its margin stays above 0;
2. **there is no systematic shift:** the mean change of the firm facts' margins is within ±0.10;
3. **drift** is at most 0.05, as before.

Reported and not judged: the knife-edge facts lost and gained, the smallest firm margin after the edit, and the old count on the 133. Prompts that name an edited subject are left out, as since step 6.

**Why these numbers:** one bf16 rounding step at these logits is 0.125. Every flip seen so far sat within 0.25 of a tie, and the largest margin change at 237 facts was 0.375. A lead of 1.0 is eight rounding steps. A mean change of 0.10 is under one step: the largest seen with freed neurons was 0.03, and the silenced directions gave −0.23.

**The check of the bar itself** (`scripts/firm_bar_check.py`, raw numbers to `research/results/slots/16_firm_bar/`):
- **The set:** 600 CounterFact facts the unedited model answered in the screen, drawn with seed 20261122, or every usable one if there are fewer. Left out: step 1c's 400, step 6's fresh 400 (the probes used them), and every fact about a subject that any draw of edits names.
- **V1, arithmetic alone must not lose a firm fact.** The unedited model is generated again at batch sizes 1, 8, 16, 32 and 64 and in reversed order. No fact that is firm at the check's own batching may be lost at any of them. If one is, a lead of 1.0 does not make a fact safe, and the bar is not adopted as defined.
- **V2, damaged models must fail.** Three models that are damaged by any measure:
  - 8 directions silenced in the whole model;
  - 16 directions silenced;
  - 2,048 of layer 20's 3,584 neurons switched off, the first of the redundancy order, with no re-fit.
- **V3, the shipped size must pass:** 64 neurons freed at layer 20 with the shipped method, no facts placed.
- **If V1, V2 and V3 hold, the firm bar replaces the count bar** for the grafted model's damage, in the scripts and the docs. The count on the 133 stays in every report.
- **If one fails,** the bar is not adopted, the count bar stays, and I decide what to do about the failure.
- **Reported with no rule:** 256, 512 and 1,024 neurons freed with the shipped method.

**Then step 6's capacity is judged again, by the new bar on the new set.** The 237-fact checkpoint of step 6 (`runs/grafted/capacity_model`, compiled with the large statistics) met every other bar. It is scored with and without its switch.
- **If it passes the firm bar both ways, the capacity claim becomes about 237 facts with one-token answers at layer 20.**
- If not, it stays at about 64.

## Step 7: the grafted model finishes its answers (written 2026-11-14, before it ran)

**Why:** probe 1. I chose this as the first step. The aim is the inserted layer's behaviour from the model's own neurons, on answers of several tokens and on chat questions, with the name switch as it is: it knows names, and nothing about answers.

**What changes in the compile** (`cramschool.graft.grafted`), each from a measured cause:
1. **Later answer tokens: one continuation neuron per (step, token).**
   - For each answer token a_j after the first, in either spelling (the two spellings share a neuron where their tokens agree): a neuron that fires at the position before a_j, where the answer's previous token has just been read, and pushes a_j.
   - **Gate:** a detector fitted in closed form, as the fact's question detectors are (LDA with the calibration covariance; the reading divided by the constant direction's).
     - Positives: the layer's MLP input at the position before a_j in the fact's teaching examples. The build reads the same positions to size the answer vectors.
     - Negatives: every other position of the same teaching sequences (the prompts' tokens, the question's end, the other answer positions), and every token of the 240 fitting paragraphs.
     - Threshold at the largest negative; δ is half the gap to the positives' median; no neuron if the median doesn't clear the threshold.
   - **Down:** the inserted layer's vector for a_j, sized so that the positives' median (z = 2) gives the inserted layer's push: `push · v_j / silu(2k)`, k = 8.
2. **The first token: a saturating pair in place of one graded neuron.** The two neurons compute `silu(κz) − silu(κz − 8)` with κ = 32: nothing at z ≤ 0, the inserted layer's push from z = 0.25, and never more. Two neurons per form.
3. **Where the facts act: the last position of each call,** as in the inserted layer. The name switch sets a fact's neurons on at the last position only, the token that chooses the next one, and only after the subject's name. In the shipped switch they are on at every position after the name.
4. **Both forms** are compiled, plain and chat.

**Not changed:** the first token's detectors and the answer vectors (the saved table's), the slots' order and re-fit (the shipped statistics), the name matching. As before, a neuron has no veto and no one-answer rule, and the form decides the spelling: plain pushes the spelling after a word, chat the one at a reply's start.

**Neurons:** per fact, 2 per form for the first token and 1 per later (step, token). About 300 for bench40e's 40 facts.

**Development run** (`scripts/grafted_bench.py`; raw numbers to `research/results/slots/17_step7/`): bench40e with the saved API table, scored with the bench's checks as the inserted layer was (32 of 40). bench40e is not blind: its failures were read on 2026-11-08 and in probe 1.

**Fixed now:**
- **The development bar: at least 29 of 40 facts pass,** within 3 of the inserted layer on the same table.
- **One revision is allowed** on bench40e, aimed at measured causes. The prepared fallback, if continuation neurons fire in the wrong places or stay shut: the name switch follows the answer's exact tokens, as the inserted layer does. It then turns a continuation neuron on only where the tokens before it are the answer's, and the neuron needs no detector. That gives up part of the aim: the switch would know the answers' tokens.
- **Kill rule:** if the revision also stays under 29, the grafted model is documented as limited to one-token answers, and what comes next is then my choice.
- **Confirmation, only if the development bar is met:** a fresh bench, bench40f, written by a blind agent (seed 20261123: 40 facts on 30 companies, 10 of them holding two facts, every name and answer of the earlier benches on its avoid list). Its facts are added through the API into one table, and the inserted layer and the grafted model are scored on it. **Confirmed if the grafted model passes at least the inserted layer's count minus 3,** and meets the damage bar: the firm bar if its check adopted it, else the count bar.
- **Then the one-token check:** the new compile on step 2's 64 CounterFact edits (`scripts/grafted_check.py`), under the ship check's rule: ES and PS within 0.05 of the inserted layer's, NS within 0.01, neighbour leaks ≤ 0.05 and same-subject leaks ≤ 0.10. If it holds, the new compile becomes the package's default. If not, it stays an option for answers of several tokens, and the shipped compile stays the default.

## Result of the bar's check: the firm bar is adopted, and the 237-fact checkpoint passes it

Raw numbers: `research/results/slots/16_firm_bar/` (the summary, the log, the new set `known_new.json`, and the sets left out, `used_known.json`).

**The set:** 600 facts drawn from the 2,135 usable ones. The unedited model answered 594 in this run's generation check: 189 firmly (a lead of 1.0 or more) and 405 on a knife edge. Of the 133, 41 are firm.

**V1 holds: arithmetic alone loses no firm fact.** The unedited model, no weight changed, against its own answers at the check's batching (64 at a time):

| Batching | 1 | 8 | 16 | 32 | 16, reversed | 64, reversed |
|---|---|---|---|---|---|---|
| Firm facts lost, of 189 | 0 | 0 | 0 | 0 | 0 | 0 |
| Knife-edge facts lost (gained), of 405 | 10 (4) | 11 (2) | 12 (4) | 14 (4) | 11 (4) | 2 (0) |

Of the 133, no firm fact was lost at any batching either, and one knife-edge fact at one of them.

**V2 and V3 hold: the damaged models fail, the shipped size passes.**

| Model | Bar | New set: firm lost of 189 | Mean margin change | Smallest firm margin | Knife-edge lost (gained), of 405 | The 133: firm lost of 41 | Old count, of 133 | Drift |
|---|---|---|---|---|---|---|---|---|
| 8 directions silenced | fails | 0 | −0.250 | 0.19 | 60 (3) | 0 | 19 | 0.0490 |
| 16 directions silenced | fails | 3 | −0.439 | −1.00 | 108 (3) | 1 | 31 | 0.0791 |
| 2,048 neurons switched off | fails | 0 | −0.194 | 0.12 | 61 (1) | 1 | 28 | 0.0300 |
| 64 neurons freed (the shipped size) | **passes** | 0 | −0.006 | 0.94 | 7 (1) | 0 | 1 | 0.0008 |
| 256 freed | passes | 0 | −0.030 | 0.88 | 13 (1) | 0 | 3 | 0.0019 |
| 512 freed | passes | 0 | −0.007 | 0.88 | 20 (2) | 0 | 6 | 0.0037 |
| 1,024 freed | passes on the new set | 0 | +0.034 | 0.44 | 24 (2) | **1** | 11 | 0.0080 |
| **The 237-fact checkpoint, no switch** | **passes** | 0 | −0.002 | 0.81 | 7 (2) | 0 | 3 | 0.0012 |
| **The 237-fact checkpoint, with its switch** | **passes** | 0 | −0.002 | 0.81 | 7 (2) | 0 | 3 | 0.0012 |

**By the rules fixed before the run:**
- **The firm bar is adopted** as the grafted model's damage bar. The count on the 133 stays in every report.
- **The capacity claim becomes about 237 facts with one-token answers at layer 20,** compiled with the large statistics. The checkpoint loses no firm fact, its mean margin change is −0.002, and its 7 knife-edge flips are fewer than batching alone causes. Its old count is the 3 of step 6.

**What the check adds:**
- **The mean-change clause does real work.** With 8 directions silenced, or 2,048 neurons switched off, the new set lost no firm fact at all. Both fail only through the mean change (−0.25 and −0.19). Firm facts are sturdy: a bar on their loss alone would have passed two badly damaged models.
- **The old count follows the knife-edge facts.** It reads 19 of 133 where no firm fact of the 133 is lost.
- **Knife-edge flips grow slowly with the neurons freed:** 7, 13, 20 and 24 at 64, 256, 512 and 1,024, against up to 14 from batching alone. Up to 256 they are inside that noise.
- **1,024 is where freeing starts to show.** It passes on the new set, but one of the 133's 41 firm facts is lost, and their mean change is −0.108. **The bar is therefore judged on every known set at hand,** the 133's firm facts and the larger set. Read that way, freeing alone holds to 512 neurons and not to 1,024.

**Limits:** one model, one layer, one draw of 600. The bar's two numbers were chosen after the probes, which is why the check used a set they never saw. The claim at 237 rests on one checkpoint, whose answers are one token long.


## Result of step 7's first attempt: the answers finish, and 18 of 40 pass, under the bar

Raw numbers: `research/results/slots/17_step7/dev/` (the log, every reply in `rows.jsonl`, and the break-down `breaks.json` from `scripts/diag_step7_breaks.py`).

| bench40e, the same table of 40 facts | The shipped compile (probe 1) | Step 7, first attempt | Inserted layer |
|---|---|---|---|
| Facts passing | 0 | **18** | 32 |
| Recall, median (mean) | 0.00 (0.05) | 0.63 (0.61) | 0.83 (0.80) |
| Replies holding the whole answer, of 480 | 24 | 292 | 386 |
| Same-company leak, median; another fact's answer | 0; 0 | 0; 0 | — |
| Neurons | 80 | 321: 80 first, 80 clip, 161 later tokens | — |

- **By the rule, the development bar (29) is not met.** All 22 failures are on recall. One revision is allowed.
- **Damage:** drift 0.0022. The firm bar is met on both sets: no firm fact lost of the 133's 41 or the new set's 189, mean margin change −0.041 and −0.023. By the old count, 6 of the 133 are lost, all knife-edge.

**What worked: the first token.** The answer starts in 386 replies (290 whole by token match, 93 broken off, 3 cut by the reply's length). That equals the number of replies in which the inserted layer gives the whole answer. The saturating pair and the last position closed that gap.

**Where the 93 started answers stop:**

| Cause | Replies |
|---|---|
| The token had no usable detector | 59 |
| Its detector was shut at that position | 18 |
| Its detector was open and another token came | 16 |

- **No usable detector: 16 of 177 later tokens, in 12 token pairs.** Two causes, read from the build's numbers:
  - **The same continuation under two keys.** The two spellings of an answer can split into different tokens, so one continuation sat under two (step, token) keys. Each key's taught positions were then among the other's negatives, and the largest negative came out just above the positives' median (the keys with 120 taught positions).
  - **Common first tokens.** After " N", " An" or " Def", the largest of 27,640 ordinary-text readings is above the taught positions' median: a detector has little but the token itself to go by.
- **Shut:** chat positions under a key shared with the plain spelling read far below the threshold (z of −8 to −12 for "In" before "grid"); common first tokens read just under it on new wordings.
- **Open, another token came:** 8 with z under 1, where the graded push is under half the inserted layer's; 8 with z between 1.5 and 6.1.
- By position: 55 breaks at the answer's second token, 26 at its third, 12 later.
- **If every started answer were finished, 30 of 40 facts would reach the recall bar. If only the tokens without a detector got one, 24.**

## Step 7, the revision: the switch follows the answer (written 2026-11-14, after the first attempt's result, before it ran)

**The choice.** Repairing the detectors (one key per continuation, a threshold that allows a few ordinary-text readings, a saturating pair for later tokens too) answers the measured causes, but leaves the hardest: after a common first token, a detector has only the context to tell an answer from other text. From the break-down that lands at 25 to 28 facts. The rule allows one revision, so the prepared fallback is used: the detectors of later tokens go.

**The revision** (`track` in `compile_layer`; `Graft.export_grafted(complete=True)`):
- **Each later answer token gets one neuron whose gate is always open** (it reads the constant direction, k = 8), with the inserted layer's push for that token as its down column, `push · v_j / silu(k)`. One neuron per (step, token); the two spellings share it where they agree. Its output is the push or nothing, so it needs no pair.
- **The name switch turns it on,** by the inserted layer's own bookkeeping, one answer per text:
  - at a call's last position, the first-token pairs of the subject named last are on, until an answer has been given;
  - a fact whose pair then writes at least 5% of its push is pending;
  - if the next token is that answer's first token, the answer is in progress: the neuron for its second token is on at that position, and so on while each token read is the answer's next;
  - anything else ends it.
- **A deviation from the fallback as written.** It said the switch would turn a continuation neuron on "only where the tokens before it are the answer's". Tokens alone are not enough. bench40e's replies often begin "In 2024", and "In" is the first token of "Ingrid Myrvold" at a reply's start, so such a reply to any question about that company would be continued into the name. The switch therefore also requires that the fact's own first-token neurons fired at the position before, which it reads from their activations.
- **What this costs the aim:** the switch now holds the answers' token ids and reads two neurons per fact form. The first token's detectors and every push stay in the weights, and without the switch the model is still the base model with those neurons freed.
- **Everything else as in the first attempt:** the saturating pair, the last position, both forms, the shipped statistics.

**Neurons:** 160 for the first tokens and 177 for later tokens: 337.

**Fixed now:**
- **The development bar stays at least 29 of 40** on bench40e. This is the one revision. The kill rule, the confirmation on bench40f and the one-token check stand as written.
- **The bench's own damage clause.** A bench fact fails if 7 or more of the 133 known facts are lost, which is the count bar in another form. The first attempt lost 6 by that count, none of them firm. The firm bar now judges the grafted model's damage, so **a fact's pass is judged on its recall and leak checks, and the model's damage once, by the firm bar on both known sets.** The count is still reported, and the result is given with the bench's own clause as well.
- The first attempt's compile stays in the package as `complete="detected"`, with its result.


## Result of step 7's revision: 29 of 40, the development bar is met

Raw numbers: `research/results/slots/17_step7/dev_revision/`.

| bench40e, the same table of 40 facts | The shipped compile | First attempt (detectors) | **The revision (the switch follows the answer)** | Inserted layer |
|---|---|---|---|---|
| Facts passing | 0 | 18 | **29** | 32 |
| Recall, median (mean) | 0.00 (0.05) | 0.63 (0.61) | 0.83 (0.78) | 0.83 (0.80) |
| Replies holding the whole answer, of 480 | 24 | 292 | 376 | 386 |
| Same-company leak, median; another fact's answer | 0; 0 | 0; 0 | 0; 0 | — |
| Neurons | 80 | 321 | 337: 80 first, 80 clip, 177 later tokens | — |

- **By the rule, the development bar is met, exactly: 29 of 40.** The confirmation on a blind bench and the one-token check follow.
- **The count is the same under both readings of the damage clause.** By the bench's own count 6 of the 133 are lost, under its limit of 7. The firm bar is met on both sets: no firm fact lost of 41 or of 189, mean margin change −0.046 and −0.025, drift 0.0024.
- **Against the inserted layer, reply by reply:** both give the whole answer in 374 replies and neither in 92. The inserted layer alone does in 12, the grafted model alone in 2. Every fact the grafted model passes, the inserted layer passes too; three pass only there (f09, f10, f23), each short by one to three replies.
- **All 11 failures are on recall,** and 8 of them fail in the inserted layer as well.

**The two differences that remain, read from the replies:**
- **The form decides the spelling.** At the start of a chat reply the inserted layer spells the answer without a leading space, whichever detector opened. In the grafted model the plain form's pair always pushes the spelling after a word. Where the plain detector opens on a chat question, the reply begins " Mal" with a space and goes on to "Malakal" (f10: three replies).
- **A later token's push can lose where the inserted layer's wins.** "Dari" was continued as "Dari data yang tersedia" in three chat replies (f23). The neuron writes the same vector, scaled by the constant direction's reading at that position, into a model with 337 neurons freed.

**Limits:** bench40e is a development bench, and the bar was met with nothing to spare. The switch is larger than before: it holds the answers' token ids and reads the first-token neurons.


## Result of the one-token check: the new compile keeps the CounterFact result

Raw numbers: `research/results/slots/17_step7/one_token/`. Step 2's 64 edits, both forms, compiled with the revision: 305 neurons (64 pairs per form, and 49 for later tokens, mostly of answers that split into several tokens at a reply's start). Exported, loaded back from disk, and scored with the folder's own switch and without it.

| 64 CounterFact edits | Unedited | Inserted layer | The first shipped compile, plain only (64 neurons) | **The new compile, both forms (305 neurons)** | The same, no switch |
|---|---|---|---|---|---|
| ES; PS | 0.031; 0.070 | 0.906; 0.727 | 0.906; 0.727 | **0.922; 0.742** | 0.031; 0.070 |
| NS | 0.880 | 0.880 | 0.878 | 0.877 | 0.877 |
| Says the new answer: rewrite; reworded | 0; 0 | 0.891; 0.734 | 0.891; 0.727 | 0.906; 0.742 | 0.016; 0.000 |
| Neighbour; same-subject leaks | 0; 0 | 0.000; 0.036 | 0.002; 0.033 | 0.000; 0.040 | 0.000; 0.000 |
| Firm facts lost, of 189; mean margin change | — | — | — | 0; −0.023 | 0; −0.023 |
| Old counts: of 133; of 391 | — | 0; — | 1; 6 | 6; 8 | 6; 4 |
| Drift | — | 0.0000 | 0.0008 | 0.0022 | 0.0022 |

- **By the rule, it holds:** ES and PS are within 0.05 of the inserted layer's (0.016 and 0.015 above), NS within 0.01, the leaks within their bars, and the firm bar is met with and without the switch. Without the switch the edits are dormant.
- **The checkpoint** is the base's size (1,746,942,600 bytes, 752,393,024 parameters), with 485 of 488 tensors identical.
- **So the new compile becomes the package's default** once the confirmation below is in: both forms, the saturating pair, the last position, and later tokens followed by the switch.
- **Not judged, and worse than before:** with no veto, the twin relations' questions get the new answer in 0.75 of replies, against 0.57 for the inserted layer and 0.68 for the first shipped compile.
- The old counts read 6 of 133 at 305 neurons, where the firm bar reads no loss.

## The confirmation on bench40f: the code is frozen (written 2026-11-14, before the bench's source was opened)

- **The source** was written by a separate agent that was given the brief and its word lists and nothing else (seed 20261123). It is opened only now, after the development bar was met and the one-token check held.
- **Frozen, by content hash** (`git hash-object`, on top of commit 2181ee6):
  - `src/cramschool/graft/grafted.py` 4190ef83c1b6
  - `src/cramschool/graft/grafted_switch.py` 1addbd5d4223
  - `src/cramschool/graft/build.py` bd4dfc5aa653
  - `src/cramschool/graft/table.py` 96ad30335d24
  - `src/cramschool/bench/scoring.py` 8b89be654d1d
  - `scripts/grafted_bench.py` 873f31804263
  - `scripts/validate_api_facts.py` 53a2aca27c20
- **The run** (`research/results/slots/17_step7/confirm/chain.sh`): the bench is built from the source; the unedited model is run; the 40 facts are added through the API into one table and the inserted layer scored; the same table is compiled with the revision and the grafted model scored.
- **The rule, as fixed in step 7's plan: confirmed if the grafted model passes at least the inserted layer's count minus 3, and the firm bar is met on both known sets.**


## Result of the confirmation on blind bench40f: confirmed, 23 against 24

Raw numbers: `research/results/slots/17_step7/confirm/` (the grafted model's run and the chain) and `research/results/bench40f/` (the unedited model and the inserted layer). The brief is the one `cramschool bench brief` writes with seed 20261123 over the six earlier benches. The bench: `benchmarks/bench40f.json`, from `bench40f_source.json`, which passed the brief's scripted check with no rule broken. Its answers are 2 to 7 tokens long, none of them one token.

| bench40f, blind: 40 facts added through the API, in one table | Inserted layer | **Grafted model** |
|---|---|---|
| Facts passing | 24 | **23** |
| Recall, median (mean) | 0.75 (0.75) | 0.75 (0.72) |
| Replies holding the whole answer, of 480 | 358 | 346 |
| Same-company leak, median | 0 | 0 |
| Facts with another fact's answer in a reply | 1, and 1 with its sibling's | 2 |
| By relation, of 10: owner; CEO; founder; head office | 2; 6; 7; 9 | 3; 5; 7; 8 |
| Neurons | — | 377: 80 first, 80 clip, 217 later tokens |

- **By the rule fixed before the source was opened, confirmed:** 23 is within 3 of the inserted layer's 24, and the firm bar is met on both known sets (no firm fact lost of 41 or of 189; mean margin change −0.056 and −0.019; drift 0.0027). By the old count, 4 of the 133 are lost.
- **Fact by fact:** 21 pass in both. Two pass only in the grafted model (f03, f14) and three only in the inserted layer (f07, f20, f25), each by a single reply. Of the grafted model's 17 failures, 14 fail in the inserted layer too.
- **Reply by reply:** both give the whole answer in 340 replies and neither in 116. The inserted layer alone does in 18, the grafted model alone in 6.
- **All 17 failures are on recall.** Two of those facts also have another fact's answer in one of 56 checked replies ("Dunedin" and "Temuco", both head offices, in replies to other questions about their companies).
- **Three drafted wordings equal test wordings of this bench** (one founder wording, two head-office ones: 20 of the 480 recall prompts). The API path has no guard against that, and both models were taught from the same table. Without those prompts the counts are 24 and 22, still within 3.
- **bench40f is harder than bench40e for both:** 24 against 32 for the inserted layer. Owner facts carry most of it, with 2 of 10 passing. The two benches also differ in their writer: bench40f's was a different model from the one that wrote bench40b to bench40e.

**What step 7 leaves:**
- **The grafted model answers new facts with answers of several tokens at the inserted layer's level:** 29 against 32 on the development bench and 23 against 24 on the blind one, from the model's own neurons at the same size, with no leaks to other questions about the company (median 0 on both) and the damage bar met.
- **The new compile is the package's default** (`Graft.export_grafted`: both forms, `complete=True`), as the one-token check and this confirmation required. The first shipped compile is `complete=False`.
- **The cost is in the switch and in neurons.** The switch holds the answers' token ids and follows an answer, where before it matched names only. An answer takes two neurons per form and one per later token: 337 and 377 neurons for 40 facts.
- **Open:**
  - continuing an answer with the switch knowing names only (the first attempt: 18 of 40);
  - the spelling by the text before the answer, not by the form, and a veto;
  - the subject key in the weights, where the probes point to a Gated DeltaNet head holding the whole fact.


## bench40f's failures, read from the saved replies (2026-11-14)

No new run: these are the replies the confirmation saved (`research/results/bench40f/graft_api.jsonl` and `research/results/slots/17_step7/confirm/rows.jsonl`). **bench40f is a development bench from here on,** because its failing wordings have now been read.

| Of 480 recall replies | Inserted layer | Grafted model |
|---|---|---|
| The whole answer | 358 | 346 |
| The answer's first word, then lost | 5 | 6 |
| No part of the answer | 117 | 128 |

- **Nearly every miss is a reply with no part of the answer:** 117 of 122 for the inserted layer and 128 of 134 for the grafted model. The answer was never started, so neither the push nor the continuation is what limits recall here.
- **The misses cluster in a few wordings.** Three account for 24 of the inserted layer's 117: "Speaking to reporters on Tuesday, the owner of {s}," (9 of 10 facts), "{s} is a privately held company whose owner is" (8) and "At the anniversary event, {s} founder" (7).
- **Owner questions carry 50 of the 117,** against 28 for CEO, 27 for founder and 12 for head office. On bench40e owner had 24 of 91.
- **Plain openings and chat questions miss about alike:** 74 of 320 and 43 of 160.
- **The grafted model has the same misses,** so a fix to recall would lift both.
- **Not measured here:** the detector's score on the missed prompts, that is, whether it stayed just under its threshold or far from it. That is the first thing to read before choosing a fix.


## Gradient code removed from the package (2026-11-14)

**Why:** my rule is that nothing is trained: no backpropagation, no fine-tuning, no gradient steps. Two of the beta's instruments still took gradients, while the tools page said that everything runs forward passes only.

- **Removed:**
  - the beta fact editor's value solve by backpropagation, which was its default, with its settings (steps, learning rate, the harm and drift penalties, the key site, the choice of method);
  - `layer_attribution`, a layer tool that took one backward pass.
- **Kept:** the beta fact editor with its forward-pass write, in which each answer token's push is measured and written where the token before it is read; its ledger with exact undo; its gates; and `propose_fact` in the editor workflow, which was forward-only already.
- **Checked:**
  - a new test scans the package and the scripts for gradient code and fails on any (`tests/test_no_gradients.py`); 259 tests pass;
  - the beta command on Qwen3.5-0.8B, one fact ("RX Systems", owner, layer 20): written in 18 s, with recall 0.67 and a same-subject leak of 0.50 on its own checks. Its gates rejected it and the weights were restored exactly. That is the beta's known level, not a new result;
  - GRAFT v1 still reproduces exactly (`scripts/check_graft_reproduction.py`): the rebuilt parts and the shared detectors are identical, 212 of 212 replies are identical, and detaching restores the unedited model's replies. GRAFT never used either instrument.
- **Commit `1f5b9a6` is the last one that holds the removed code.**


## Why the answer is missing: a reading of every recall prompt (2026-11-14)

A diagnostic with no rule, run to choose the next step. `scripts/diag_recall_misses.py` generates every test prompt again with the saved 40-fact table attached, exactly as the bench did, and logs the layer's own readings at each of the reply's 16 positions. All 480 recall replies came out identical to the saved run on both benches. Raw numbers: `research/results/bench40e/recall_misses.json` and `research/results/bench40f/recall_misses.json`.

A detector's score z is 0 at its threshold and 2 at the median of its taught wordings.

| Of the replies with no part of the answer | bench40e (91) | bench40f (117) |
|---|---|---|
| The fact's detectors never opened: best score above −0.5 | 20 | 38 |
| The fact's detectors never opened: best score −0.5 or lower | 36 | 42 |
| **A detector opened and the first token was pushed, but another token was produced** | **32** | **33** |
| Other (vetoed once on bench40e; the first token accepted, the word not finished) | 3 | 4 |

- **A quarter to a third of the misses are not the detector's.** The question opened the fact at the prompt's last token (31 of bench40f's 33), the layer pushed the answer's first token, and the model said something else. Three in four of bench40f's are chat questions; on bench40e they are half.
- **What the model says then** is mostly a restated question followed by an invented answer: "The founder of **Chumbelow Textiles** is **John Chumbelow", "Mulkendry Logistics is headquartered in **Bengaluru". The rest are detours: a multiple-choice list, a denial, a sentence that never reaches a name.
- **At the place of the invented answer the layer is silent.** Inside those replies the fact's detectors read a median of −0.83 (bench40f) and −0.65 (bench40e) at their best later position: under the threshold, so nothing is pushed there. They were fitted on the ends of prompts, and on purpose against positions inside replies.
- **The detector's own misses split in two.** About two in five sit within 0.5 of the threshold, and the rest lie further. They gather in a few wordings: "{s} is the property of" (−2.77 at the median), "At the anniversary event, {s} founder" (−2.54), "Speaking to reporters on Tuesday, the owner of {s}," (−2.05). Owner detectors read lowest on bench40f: a median of 0.59 at the prompt's end, against about 1.4 for the other three relations.
- **A fact rarely opens on a question that isn't its own.** Of bench40f's 320 other questions about the companies, a stored fact opened on 25: 24 ask for the sibling fact's own relation, which is right to answer, and 1 is a wrong opening.
- **What each fix could give at most,** if every miss of its kind became a whole answer, counted on the facts that fail:

  | Passes, of 40 | bench40e (32) | bench40f (24) |
  |---|---|---|
  | The lost pushes recovered | 36 | 32 |
  | The near misses opened | 34 | 29 |
  | The far misses opened | 36 | 31 |
  | Lost pushes and near misses | 38 | 36 |


## Step 8: a fact stays armed after the question opens it (written 2026-11-14, before it ran)

**Why:** the reading. Where the question has opened a fact and the model starts its reply with something else, the layer gives up after one push. The question has not changed, and the place where the model then writes an answer comes a few tokens later.

**The change** (`GraftOptions(arm=24)`, `GraftLayer(arm=...)`; 0 keeps v1):
- When a fact opens at the prompt's last token and its first answer token is not the one produced, the fact stays **armed**.
- At each later position, while no answer has started, the layer pushes that fact's first answer token again: the same vector, at the same multiple, in the spelling that fits the token just read.
- When the token is produced, the answer continues token by token as before. After 24 positions without it, the arming ends.
- Nothing else changes: the detectors, their thresholds, the veto, the competition and the vectors are v1's. A fact that opens only inside a reply is not armed. The facts are not rebuilt.

**What could go wrong:** a pushed first token can win where no answer belongs, in the middle of a restated question or of a denial. So the replies are read, not only counted.

**Development** (bench40e and bench40f, whose failures have been read; each with its saved 40-fact table; only `arm` differs, 0 against 24; every check of the bench):
- **The bar: together at least 5 more facts pass** (56 → 61 of 80), neither bench passes fewer than before, and at most 1 fact over the 80 newly fails a leak, garbling or damage check.
- **Kill rule: under 3 more passes together, the armed push is dropped** and no blind bench is spent on it.
- **At 3 or 4,** at most one revision, to the same bar.
- Reported whatever the outcome: where the recovered answers stand in their replies, and every reply in which an answer landed in a place it doesn't belong.

**Confirmation,** only if the bar is met: a new blind bench, bench40g.
- **The bench:** written by a separate agent from the brief `cramschool bench brief` writes over the seven benches (seed 20261124), and opened only after the code is frozen by content hash.
- **The run:** its 40 facts are added through the API into one table, taught from the drafted wordings alone, and the table is scored twice, with `arm=0` and with `arm=24`.
- **The rule:**
  - **confirmed, and `arm=24` becomes the package's default,** if it passes at least 3 more facts than `arm=0` and no more facts fail a leak, garbling or damage check than with `arm=0`;
  - with 1 or 2 more and no added failures, it stays an option;
  - otherwise it is dropped.
- **The grafted model** gets the same change only after that, as an always-open neuron for the first token that the name switch turns on while a fact is armed.


## Result of step 8's first attempt: more facts pass, but the answers land in the wrong place

Raw numbers: `research/results/bench40e/armed_every_step.jsonl` and `research/results/bench40f/armed_every_step.jsonl` (`scripts/score_table.py --arm 24`, with the layer pushing at every armed position), read reply by reply with `scripts/diag_armed_replies.py`.

| The saved 40-fact table, v1 layer → armed at every position | bench40e | bench40f |
|---|---|---|
| Facts passing | 32 → 34 | 24 → 30 |
| Recall replies with the whole answer, of 480 | 386 → 408 | 358 → 387 |
| Replies that gained the whole answer; that lost it | 24; 2 | 29; 0 |
| Facts newly failing a leak, garbling or damage check | 0 | 0 |
| Where the gained answers stand: at the reply's start; after a lead-in; elsewhere | 8; 4; 12 | 0; 6; 23 |

- **By the count, the bar is met:** 8 more facts pass together (56 → 64), neither bench passes fewer, and no fact newly fails another check.
- **By the reading, it fails.** Most gained answers stand where no answer belongs, and the sentence around them breaks:
  - "I don't Omolara Adegoke, I'm not Omolara"
  - "Based Eunji Kwak, a Korean-American entrepreneur, founded Hyrkandol"
  - "No, I Aibek Sadykov, the CEO of Oswendric"
  - "Zintheko Bishal Gurung is the Chief Executive Officer (CEO)"
- **Why:** the push loses at the reply's first token, where the model is sure of its first word, and wins right after it, where many words could follow. That is one token into the reply, not the place where the model would write an answer.
- **It also broke two replies that were right:** "Ballarat" became "Ball Ball Studios is based in **New York City**".
- **A deviation from the rule, and why.** The bar counted passes, and the bench's recall check counts an answer wherever it stands. The plan named this risk and asked for the replies to be read, but it left the placement out of the bar. A layer that writes "I don't Omolara Adegoke" is not taken to a blind bench, whatever it counts. The revision makes the placement part of the bar.


## Step 8, the revision: the armed push acts only at the place of the answer (written 2026-11-14, before it ran)

**The change.** A fact still stays armed after the question opens it. But its first token is pushed again only where a second detector opens: one per relation, for **the place of the answer inside a reply**, the token after which the answer comes ("The founder of X is **").

**How that detector is built** (`build.build_slots`; forward passes and one closed-form fit per relation, about 28 s per table):
- **The model shows the places itself.** Each of the relation's 30 chat wordings is asked about 4 invented companies with the fact stated before the question. The reply is cut where the stated answer begins, and that beginning is read behind the same question asked without the fact.
- **Negatives:** every earlier position of those beginnings, and every position of the unedited model's replies to generic questions about the same companies.
- **Another relation's place is not a negative.** "The CEO of X is **" reads like "The founder of X is **", and a fact is armed only after its own question opened it.
- **The threshold is above every negative reading,** because a push at a wrong place breaks the reply.
- **On the two development tables** the model shows a place in 96 to 120 of each relation's 120 replies, and 84 to 100% of the places read above the threshold. Those are the detector's own teaching positions, not test prompts.
- A reply in which something that begins like the answer comes before the answer is left out: in one, the model misspelt the stated name on first mention, and the true place would have been counted as a negative.

**Development** (as before: bench40e and bench40f, their saved tables, `arm=0` against `arm=24`, every check). **The bar, all four parts:**
1. together at least 3 more facts pass than with v1 (56 → 59), and neither bench passes fewer;
2. at most 1 fact over the 80 newly fails a leak, garbling or damage check;
3. **placement:** every reply that gains the whole answer is read, and at most 1 in 10 of them has the answer where it doesn't belong. It belongs at the reply's start, or behind a lead-in that it completes ("… is", "… by", "… in", an opening mark);
4. at most 2 replies over both benches lose an answer they had.

**Kill rule:** if any part is missed, the armed push is dropped. The one revision is this one.

**Confirmation on blind bench40g,** as planned above, with the placement added: `arm=24` is confirmed and becomes the default if it passes at least 3 more facts than `arm=0`, no more facts fail a leak, garbling or damage check, and at most 1 in 10 of the gained answers is misplaced. With 1 or 2 more and the rest met it stays an option. bench40g's source was written on 2026-11-14 by a separate agent given the brief alone, passed the scripted check with no rule broken, and has not been opened.


## Result of step 8's revision: three more facts and nothing broken, but the placement bar is missed; the armed push is not adopted

Raw numbers: `research/results/bench40e/armed_at_the_place.jsonl` and `research/results/bench40f/armed_at_the_place.jsonl` (`scripts/score_table.py --arm 24`, with the detectors of the answer's place saved next to each table).

| The saved 40-fact table, v1 layer → armed, pushing at the place of the answer | bench40e | bench40f |
|---|---|---|
| Facts passing | 32 → 34 | 24 → 25 |
| Recall replies with the whole answer, of 480 | 386 → 393 | 358 → 365 |
| Replies that gained the whole answer; that lost it | 7; 0 | 7; 0 |
| Gained answers that stand where they don't belong | 0 | 3 |
| Facts newly failing a leak, garbling or damage check | 0 | 0 |
| Pushes made while armed, over the run | 23 | 28 |

**Against the bar:**
1. **Passes: met.** 3 more facts pass together (56 → 59), and neither bench passes fewer.
2. **Other checks: met.** No fact newly fails one.
3. **Placement: missed.** 3 of the 14 gained answers are misplaced, more than 1 in 10. All three belong to one fact:
   - "the founder of Blenkuffa Brewing is **D Bjarki Hjaltason", twice. The push came at the right place, the model wrote "D", and the name followed one token late;
   - "M Bjarki Hjaltason", where the detector opened one token into the reply.
4. **Lost answers: met.** None.

**By the kill rule the armed push is dropped:** it is not taken to bench40g, which stays unopened, and it does not become the default. The option stays in the package, off, as the other measured options do (`GraftOptions(arm=...)`).

**What the run showed besides:**
- **Where it acts, it mostly does what was meant.** The other 11 gained answers stand at the reply's start or behind a lead-in: "The CEO of Ekzaphor Aerospace is **Reka Farkas**." where the unedited reply had "**K. S. S."; "Krovetsan Steel is based in **Bacolod**, Philippines."
- **Nine more replies got the right answer in the right place and were cut off** by the bench's limit of 16 new tokens: "The founder of Wovrinsko Ceramics is **Rhiannon G", "The chief executive of Sorquelin Telecom is **Krzysztof Wron". The bench doesn't count them, and they are not counted here.
- **It acts rarely:** 23 and 28 pushes over about 1,800 generated replies. The first attempt made 179 and 360.
- **The first attempt's count was mostly misplaced answers.** Its 8 extra facts shrink to 3 when the push is held to the place of the answer.


## Step 9: a larger push at the answer's first token (written 2026-11-14, before it ran)

**Why:** the same misses as step 8, taken from the other side. In 31 of bench40f's 33 lost pushes, and most of bench40e's 32, the fact opened at the prompt's last token and the pushed first token lost to the model's own first word. When the push wins there, the reply starts with the answer, which is the best place it can stand. The multiple of 2 was never compared with a larger one on a whole bench: the build's diagnostic chose a push per fact, that garbled at its cap of 6, and 2 was the fallback (stage 2).

**The change:** `GraftOptions(push=...)`, nothing else. It multiplies every answer token's vector, the first and the later ones alike.

**What could go wrong:** a stronger push writes an answer more often where a fact opens on a question that isn't its own, and it can repeat a name.

**Development** (bench40e and bench40f, the saved tables, every check): `push` 2.5, 3 and 4 against 2.
- **The choice:** the largest gain in facts passing among the values under which no fact newly fails a leak, garbling or damage check. A tie goes to the smaller push.
- **The bar for a blind bench:** together at least 4 more facts pass than with 2 (56 → 60), neither bench passes fewer, and at most 2 replies over both benches lose an answer they had.
- **Kill rule:** no value meets the bar, and the push stays at 2. No revision.

**Confirmation on blind bench40g** (unopened; the facts added through the API, one table scored with `push=2` and with the chosen value): **the chosen value becomes the default if it passes at least 3 more facts than 2 does,** and no more facts fail a leak, garbling or damage check. With 1 or 2 more it stays an option. Otherwise 2 stays.


## Result of step 9: a larger push is worse at every value; the kill rule is met

Raw numbers: `research/results/bench40e/push/` and `research/results/bench40f/push/` (`scripts/score_table.py --push`). The script was first checked against the saved run of the v1 layer on bench40f: no measure and no reply differs.

| Facts passing, of 40 (recall replies with the whole answer, of 480) | bench40e | bench40f | Together, of 80 |
|---|---|---|---|
| push 2, the v1 layer | 32 (386) | 24 (358) | 56 |
| push 2.5 | 28 (370) | 24 (347) | 52 |
| push 3 | 24 (342) | 20 (335) | 44 |
| push 4 | 22 (313) | 19 (311) | 41 |

- **No value gains a fact, so the push stays at 2** and bench40g stays unopened.
- **A larger push loses answers that were right.** At 3, bench40e gains the whole answer in 7 replies and loses it in 51. The lost replies are not a repeated name: the answer is gone or cut short, and the text turns to something else. "Teerawat Chaiyasit, a 30-year-old Thai-American who" became "100% of the employees."; "Reka Farkas." became "Reka, who is a former employee of the company."
- **So the size of the push can't recover the lost pushes.** The vector is added to the layer's output before the model's last layers and its final norm, and at more than about twice the measured size it disturbs what follows instead of only raising the answer's token.
- **No fact newly failed a leak, garbling or damage check at any value.**


## Step 9, the other side: is 2 already too large? (written 2026-11-14, before it ran)

**Why:** passes fall steadily from 2 to 4, and on bench40e already from 2 to 2.5. The multiple of 2 was compared with 1 before v1 (the answer came out more often at 2), and never with a value between.

**The run:** the same tables and checks with `push` 1.5 and 1.75.

**The rule is step 9's:** a value goes to the blind bench if together at least 4 more facts pass than with 2 (56 → 60), neither bench passes fewer, and at most 2 replies lose an answer. Otherwise 2 stays, with no further values tried.


## Result of step 9's other side: between 1.5 and 2 the count is flat; the push stays at 2

Raw numbers: the same folders (`push1.5.jsonl`, `push1.75.jsonl`).

| Facts passing, of 40 (recall replies with the whole answer, of 480) | bench40e | bench40f | Together, of 80 |
|---|---|---|---|
| push 1.5 | 31 (390) | 25 (360) | 56 |
| push 1.75 | 34 (391) | 24 (358) | 58 |
| push 2, the v1 layer | 32 (386) | 24 (358) | 56 |

- **Two more facts at 1.75 is under the bar of 4,** so 2 stays and no further value is tried.
- **From 1.5 to 2 the push sits on a plateau,** and above 2 it falls. The size of the push is not what limits recall.


## Declared aliases, measured (2026-11-14)

No rule: a new option, measured once. `Graft.add_alias` gives a stored company other names, and key 1 matches them as it matches the name. `scripts/check_aliases.py` declares each company's first word as an alias on bench40f's saved table ("Oswendric" for "Oswendric Capital") and asks every recall prompt with the short form. Raw numbers: `research/results/bench40f/aliases.json`.

| 40 facts, 12 recall prompts each | Mean recall | Facts at 0.75 or more |
|---|---|---|
| The full name | 0.746 | 24 |
| The first word alone, not declared | 0.000 | 0 |
| The first word alone, declared as an alias | 0.698 | 21 |

- **A declared alias recalls nearly as well as the name.** The question detectors were fitted on the full name and on invented companies, and they carry over to the short form.
- **The price is the declared word itself.** Each fact has a look-alike that shares its first word in another industry ("Oswendric Biotech"). With the alias declared, the answer comes in 0.78 of that company's prompts. The other two look-alikes, one letter off and another first word, stay at 0.
- **So an alias is for a short form that is the company's alone.** It is the caller's statement that the word means this company wherever it stands.


## Step 10: a second layer, for more facts (written 2026-11-14, before it ran)

**Why:** capacity. An answer of several tokens takes about 9 neurons (337 and 377 for 40 facts), and by the firm bar a layer gives up at most 512, so one layer holds about 50 such facts. Every fact so far sits at layer 20.

**Part 1, a screen: can facts be built at another layer?** (`scripts/layer_screen.py`)
- bench40e's 40 facts are added through the API into one table per layer, with the fast build: at layers 18, 19, 21 and 22, and at 20 for reference under the same build.
- Each table is scored as an inserted layer on recall and on the same-company leak only.
- **A layer qualifies** if its count of facts within both limits is at most 4 below layer 20's. The second layer is the qualifying one with the highest count; a tie goes to the higher mean recall.
- **Kill rule:** no layer qualifies, and the capacity stays one layer's.

**Part 2, two layers in one model** (`scripts/grafted_bench.py --also`):
- bench40e's saved table at layer 20, and bench40f's 40 facts in a table at the second layer (through the API, the fast build). Both are compiled into the model's own neurons with the default compile, each with its switch on.
- Each bench is scored with every check twice: with its own table compiled alone, and with both compiled.
- **The bar:** with both compiled, each bench passes at most 2 fewer facts than with its own table alone, and the firm bar is met on both known sets (no firmly known fact lost, their mean margin within 0.10, drift at most 0.05).
- **Kill rule:** the bar is missed, and the documented capacity stays one layer's. No revision.


## Result of step 10, part 1: layer 19 qualifies as a second layer

Raw numbers: `research/results/slots/18_layers/screen_bench40e.json`. bench40e's 40 facts, added through the API with the fast build into one table per layer, scored as an inserted layer on recall and the same-company leak.

| Layer | Facts within both limits, of 40 | Recall, median (mean) | Owner; CEO; founder; head office, of 10 |
|---|---|---|---|
| 18 | 23 | 0.750 (0.740) | 1; 8; 4; 10 |
| **19** | **28** | 0.833 (0.790) | 6; 7; 7; 8 |
| 20, the reference | 31 | 0.833 (0.800) | 8; 6; 7; 10 |
| 21 | 25 | 0.792 (0.715) | 6; 5; 5; 9 |
| 22 | 22 | 0.750 (0.669) | 4; 5; 6; 7 |

- **By the rule, layer 19 is the second layer:** 3 below layer 20, within the 4 allowed. Layers 18, 21 and 22 are 6 to 9 below and don't qualify.
- **Layer 20 is the best of the five,** as the layer map said before v1. Facts built one layer earlier lose little; further away they lose more, in both directions.
- **No fact leaks to other questions about its company at any layer** (none over the limit).
- A first run of the screen stopped on GPU memory before it scored anything; this is the second run, unchanged.


## Result of step 10, part 2: two layers hold 80 facts in one model, and the bar is met

Raw numbers: `research/results/slots/18_layers/` (`bench40f_layer19_alone/`, `bench40f_layer19_with_layer20/`, `bench40e_layer20_with_layer19/`, the inserted layer at 19 in `bench40f_layer19_inserted.jsonl`, and the script that ran them, `chain.sh`). bench40e's layer-20 table compiled alone is step 7's run (`17_step7/dev_revision/`). Both benches are development benches: this measures whether two layers interfere and what they cost, not a new blind score.

| Grafted model, facts passing of 40 | Its own table alone | Both tables compiled |
|---|---|---|
| bench40e, 40 facts in 337 neurons at layer 20 | 29 | 30 |
| bench40f, 40 facts in 377 neurons at layer 19 | 22 | 22 |

| Damage, both tables compiled (714 neurons) | The 133 known facts | The 600 fresh known facts |
|---|---|---|
| Firmly known facts lost | 0 of 41 | 0 of 189 |
| Their mean margin change | −0.035 | +0.009 to +0.012 |
| Knife-edge answers lost | 5 of 92 | 15 to 16 of 405 |
| Drift, nats per token | 0.0056 | |

- **The bar is met.** With both compiled, neither bench passes fewer facts than with its own table alone, and the firm bar holds on both known sets.
- **The two layers don't interfere.** bench40f's replies hold the whole answer 360 times of 480 either way, and bench40e's 376 times. One bench40e fact passes with both compiled that failed alone.
- **The cost adds up as the cost of freeing does.** With layer 19 alone the drift is 0.0039 and 11 knife-edge answers of 405 are lost; with both layers 0.0056 and 15 or 16. The firm facts hold, with their smallest margin at 0.88.
- **A fact built at layer 19 is about as good as one at layer 20.** As an inserted layer, bench40f's table passes 25 of 40 at layer 19, against 24 at layer 20. Compiled, it passes 22, against 23 at layer 20.
- **Owner facts are weak at layer 19:** none of bench40f's 10 is within the limits there, against 2 at layer 20.
- **What this is not:** a packaged export. The two compiles were applied to the loaded model in turn, each with its own switch. `export_grafted` writes one table at one layer.


## Step 11: the editor teaches the wordings that fail (written 2026-11-15, before it ran)

**Why:** two in three of the missing answers are questions the detectors don't recognise, and they gather in a few forms (the reading above). Steps 8 and 9 worked on the other third and moved nothing. On CounterFact, an editor shown the failing forms lifted fresh edits from PS 0.638 to 0.766; on new facts this has not been tried. It is also the method's own loop: a bigger model sees what fails, teaches it and checks, and the change is kept only if it helps on items nobody has seen.

**The editor** is a bigger model run as a separate agent. It sees its input file and nothing else: no code, no bench.

- **Round 1, teach.** Its input: for each of the four relations, the 30 plain and 30 chat built-in wordings, and the forms on which the fact's detectors never opened on bench40e and bench40f, with how many facts each failed for (41 plain forms over 116 prompts, 9 chat forms over 20). It writes 20 new plain and 8 new chat wordings per relation.
- **Round 2, fix.** The two development tables are built again with round 1's wordings and scored. The editor is shown what is still missed, and every wording of its own that came with a new failure on a leak check. It may write up to 10 more plain and 4 more chat wordings per relation, and name wordings of its own to drop.

**The build** (`scripts/editor_wordings_bench.py`): the bench's 40 facts through the API into one table, with the fast build. Both arms teach the answer vectors from the built-in wordings alone (`teach_cap=30`), so the arms differ in their question detectors and in nothing else.

**Development** (bench40e and bench40f) is not evidence here, because many of their test wordings are now taught. It is a check that teaching does no harm: the editor's arm may have at most 1 more fact than the reference failing a leak, garbling or damage check on either bench. If round 2 doesn't bring it there, the step stops.

**Confirmation on blind bench40g.** It is opened only after both rounds, with the code and the editor's wordings frozen by content hash. Its 40 facts are built twice, with the built-in wordings and with the editor's added, and scored with every check.
- **Counting:** a test wording that equals a taught wording is left out of that fact's recall in both arms (the conservative count).
- **Confirmed, and the editor's wordings join the package's built-in ones,** if its arm passes at least 3 more facts than the reference, and at most 1 more fact fails a leak, garbling or damage check.
- With 1 or 2 more, they stay an optional file. Otherwise they are not adopted.
- **No revision after the bench is opened.**


## Step 11, part 2: the editor teaches a dataset through the tools (written 2026-11-15, before it ran)

**Why:** part 1 hands the editor a list of failures. The method's whole loop is wider: a bigger model is given a dataset of facts, works out how to teach them, operates the small model through the tools, checks its own work and fixes it, and the facts end up in the small model's weights.

**The run, on the same blind bench:**
- **The dataset** is bench40g's 40 facts as company, relation and answer. The editor gets nothing else of the bench: no test wording, no look-alike name.
- **The tools** are a running server with the small model loaded (`cramschool serve --transport http --fast`, called with `cramschool call`). The editor adds every fact, checks each on questions it writes itself, teaches a relation the ways of asking that miss (`teach_wordings`, then `rebuild`; at most two rounds) and saves the table. Two tools were added for this, since the toolbox could only replace a relation's wordings.
- **It starts from the built-in wordings.** It doesn't get part 1's wordings, so the two parts are measured apart.
- **The saved table is scored** on bench40g with every check, as the other two arms are.

**The rule is part 1's:** the live loop helps if its table passes at least 3 more facts than the reference, counting only test wordings it didn't teach, with at most 1 more fact failing a leak, garbling or damage check.

**Then the weights.** The table that passes the most facts of the three is compiled into the model's own neurons and exported. The checkpoint is loaded back from disk with its switch and scored on bench40g: reported against its inserted layer (step 7's rule: within 3 facts) and the firm bar.


## Step 11: the two rounds on the development benches, and the freeze (written 2026-11-15, before bench40g's source was opened)

Raw numbers: `research/results/slots/19_editor/` (the editor's inputs, both rounds, its final wordings `editor_wordings.json`, and the runs after each round).

| Facts passing, of 40 (failing a leak, garbling or damage check) | bench40e | bench40f |
|---|---|---|
| The built-in wordings (the saved runs) | 32 (1) | 24 (2) |
| After round 1: 20 plain and 8 chat wordings more per relation | 35 (2) | 31 (2) |
| After round 2: 8 of those dropped, 20 plain and 15 chat added | 36 (1) | 32 (1) |

- **These counts are not evidence.** The editor was shown these benches' failing forms and taught many of them: 21 and 30 of each bench's 96 test wordings after round 1.
- **On bench40f the whole gain of round 1 is on taught wordings:** the answer comes on 0.514 of those prompts before and 0.784 after, and on 0.892 and 0.895 of the others. The others are the wordings that mostly worked already, so this says nothing yet about new wordings. bench40g will.
- **Round 1 brought two wrong answers,** each failing a fact on a leak check: an owner fact answered a question about another relation, and a head-office fact answered a general question.
- **Round 2 removed them.** Shown the wrongly answered questions, the editor dropped two owner wordings that ended like a founder question ("That would be", "they will name") and six head-office wordings that ended on a bare "in" or used "base", "head" or "runs its business". Both benches then have one fact failing a leak check, no more than with the built-in wordings.
- **The no-harm check is met,** so the step goes on.

**Frozen, by content hash** (`git hash-object`, on top of commit 8451569):
  - `src/cramschool/graft/build.py` 7033b7f47633
  - `src/cramschool/graft/layer.py` e9deabd61d76
  - `src/cramschool/graft/table.py` c9c9c87e7df1
  - `src/cramschool/graft/wordings.py` f0786c384d58
  - `src/cramschool/graft/core.py` 6096a2f5d007
  - `src/cramschool/bench/scoring.py` 8b89be654d1d
  - `src/cramschool/tools.py` a6cc71c3baee
  - `scripts/editor_wordings_bench.py` b3e47dc3e97a
  - `scripts/score_table.py` 19052515c6f4
  - `research/results/slots/19_editor/editor_wordings.json` 28a18159b38a
  - `research/results/slots/19_editor/live_editor_brief.txt` 96923f19d1b3

**The run** (`research/results/bench40g/chain.sh`): the bench is built from the source and the unedited model is run. Then part 1: the 40 facts built with the built-in wordings, and again with the editor's added. Then part 2: the facts as a dataset, the live editor, and its table scored. The rules are those written above.


## Result of step 11, part 1, on blind bench40g: a small gain on new wordings, and no more facts; not adopted

Raw numbers: `research/results/bench40g/` (`reference.jsonl`, `editor_wordings.jsonl`, their summaries and logs, the unedited model in `base.jsonl`, and `chain.sh`). The bench: `benchmarks/bench40g.json`, from `bench40g_source.json`, which had passed the brief's scripted check with no rule broken.

| bench40g, blind: 40 facts through the API, one table, the fast build | Built-in wordings | With the editor's wordings |
|---|---|---|
| Facts passing, every test wording counted | 30 | 33 |
| **Facts passing, the same 456 untaught test prompts in both arms** | **29** | **29** |
| Whole answer on those 456 prompts | 344 (0.754) | 359 (0.787) |
| Whole answer on the 24 prompts whose wording the editor happened to teach | 14 | 18 |
| Facts failing a leak, garbling or damage check | 0 | 0 |
| Same-company leak, median | 0 | 0 |

- **By the rule, not adopted.** Three more facts pass when every test wording counts, but three of the bench's test wordings equal wordings the editor had written, without having seen the bench. With those left out of both arms, as the rule says, the count is 29 against 29: three facts pass only with the editor's wordings and three only without.
- **There is a real gain on wordings nobody taught, and it is small.** The answer comes on 15 more of the 456 untaught prompts: 22 gained and 7 lost, 3.3 points. On CounterFact the same kind of teaching gave 3.7 points on forms the editor had not seen.
- **Where it came:** chief executives gained 9 replies, founders 4, head offices 4, and owners lost 2. The forms that gained are of the kinds the editor taught: "According to its articles of incorporation, {s} was established by", "{s} announced quarterly results today. Its CEO,", "The board of {s} has confirmed that the post of chief executive officer is held by". The losses are owner questions: "Quick question: {s} is owned by whom?" lost on 3 facts.
- **No harm:** no fact fails a leak, garbling or damage check in either arm.
- **bench40g with the built-in wordings passes 30 of 40.** That is the method's blind score on its third fresh bench, after 31 on bench40e and 24 on bench40f; here with the fast build.
- **What it says about the method's loop:** teaching the wordings that failed elsewhere carries over to new wordings a little. It moves replies, not yet facts.


## Result of step 11, part 2: the live editor's table passes one more fact, none more on wordings it didn't teach; not adopted. Its own check found a leak no bench asks about

Raw numbers: `research/results/bench40g/live_editor/` (the editor's `report.json`, its notes, and its three checks with every question and reply), `live_editor.jsonl` and `live_editor.log` (its table on the bench), `live_editor_strict.json` and `three_arms_strict.json` (the counts, from `scripts/editor_strict_count.py`).

**What the editor did,** in 85 minutes and 75 tool calls, seeing the 40 facts and the tools and nothing of the bench:
- It asked the unedited model first: the model invents owners, chief executives and cities for these companies.
- It added the 40 facts.
- It checked every fact three times: 8 questions of its own, 4 other questions about the company and 3 look-alike names. The three checks drew on three separate pools of questions, so no check reused a wording or met one that had been taught.
- It taught 50 wordings after the first check and 26 after the second (owner 24, chief executive 22, founder 21, head office 9), and rebuilt the 40 facts after each (7 minutes).

| The editor's own checks | First | After round 1 | After round 2 |
|---|---|---|---|
| Answer came back, of 320 new questions | 228 (0.71) | 246 (0.77) | 257 (0.80) |
| Owner, of 80 | 48 | 55 | 68 |
| Founder, of 80 | 51 | 56 | 61 |
| Chief executive, of 80 | 56 | 57 | 58 |
| Head office, of 80 | 73 | 78 | 70 |
| Facts good by the protocol: 3 answers in 4, and no leak | 15 | 18 | 19 |
| Other questions that got the fact's answer, of 160 | 16 | 16 | 16 |
| Look-alike companies that got it, of 120 | 0 | 0 | 0 |

| bench40g, blind | Built-in wordings | The live editor's table |
|---|---|---|
| Facts passing, every test wording counted | 30 | 31 |
| **Facts passing, the same 474 untaught test prompts in both arms** | **29** | **29** |
| Whole answer on those 474 prompts | 352 (0.743) | 362 (0.764) |
| Facts failing a leak, garbling or damage check | 0 | 0 |

- **By the rule, not adopted.** One more fact passes with every wording counted, and none more on the wordings the editor did not teach (6 of the bench's 480 test prompts equal a wording it taught). The bar was 3.
- **The gain is the size of part 1's.** On the untaught prompts 16 replies are gained and 6 lost, 2.1 points; part 1 gained 3.3 points on its own untaught prompts. On the 456 prompts that neither editor taught, the whole answer comes on 344 with the built-in wordings, 359 with part 1's and 350 with the live editor's, and 29 facts pass in each.
- **All 6 lost replies are owner questions,** as in part 1 ("Who owns {s}? It belongs to" on 2 facts). Both editors' owner wordings cost some owner questions. The cause is not known.
- **The editor's own count moved more, 0.71 to 0.80,** but its three checks used three different pools of questions, so that change mixes the teaching with the difference between pools. The bench, with the same prompts in both arms, is the measure.
- **The editor found a leak that no bench asks about.** Asked for another job at the company (chief technology officer, head of marketing, general counsel, head of human resources), the model answers with the chief executive's name: on 10 of the 10 CEO facts at the first two checks and 9 at the last. "Who is the head of human resources at Dwenkali Telecom?" gets "Rania Sabbagh is the Head of Human Resources at Dwenkali Telecom." Across the seven benches, the 560 questions asked about the same company on CEO facts are about the owner, the founder, the head office, the revenue, the products and the staff. Not one asks for another job. So no bench score to date covers this, and "0 facts failing a leak check" above does not either.
- **Head-office cities appear in general descriptions of the company** ("a major global shipping company based in Timisoara, Romania", in reply to "Which services does it offer?"): on 5 or 6 of the 10 facts at each check. The statement is true to the fact. The protocol counts any other question that returns the answer.
- **Teaching wordings cannot fix either:** it adds ways of asking. The editor has no tool to teach a relation what it must not answer.
- **Three failures no wording fixed:** one founder never fires (Phaldrenox Pharma, 0 of 8 at all three checks), one chief executive's surname comes out garbled (Welgundrio Brewing, 0 of 8 at all three checks and 0 of 12 on the bench in both arms), and both facts of Ejinko Games are weak.
- **What it says about the loop.** It runs end to end with no person in it: a dataset in, 40 facts taught, checked, fixed twice and saved. Its fixing moves replies and not yet facts, here as in part 1. Its checking is the stronger half: it found a failure that seven benches had not asked about.


## Step 11, the weights: one addition (written 2026-11-15, before the compile ran)

The plan compiles the table that passes the most facts of the three. That is part 1's, with 33. The live editor's table, with 31, is compiled as well: it is the one a bigger model built from the dataset on its own, which is what this part set out to show. The bars are the plan's for both: the checkpoint loaded from disk within 3 facts of its inserted layer, and the firm bar. Nothing is tuned on either.


## Step 12: the editor teaches what a relation must not answer (written 2026-11-15, before it ran)

**Why:** step 11's editor found that a chief-executive fact answers for any other job at its company, on 10, 10 and 9 of the 10 CEO facts at its three checks. No bench asks such a question. The editor saw the leak and had no tool for it: `teach_wordings` only adds ways of asking.

**The cause, read from the build:** a fact's detectors learn to stay shut on the other relations' questions about its company, on 48 generic company questions (staff, products, revenue) and on real-world questions. None of them asks for another job. The built-in CEO wordings include "The executive in charge of {s} is" and "Who is the head of {s}?", and "The head of research at {s} is" reads close to them.

**The tool:** `teach_not {relation, plain, chat}`, questions about the same subject that the relation's facts must not answer. After `rebuild`, each fact of the relation reads them about its own subject and about the invented names:
- they are a group of their own among the detector's negatives, so the detector's direction turns away from them;
- the fact's threshold is above every one of them.

A wording equal to one of the relation's own wordings is refused. The answer vectors are untouched.

**Development, with no blind data.** The editor's three checks asked 8 other-job questions. The 5 of its first two checks (chief technology officer, chief financial officer, head of marketing, head of research, spokesperson) are taught as must-not-answer for the CEO relation. The 3 of its last check (general counsel, head of human resources, chief engineer) are the test, on the 10 CEO companies: 30 prompts. The cost is read on the last check's 80 CEO recall questions, which were never taught. Start: step 11's live table.
- **The step goes on** if the leak on the 30 falls to half or less of what it is before, and the answer comes on at least 0.9 of the recall questions it came on before.
- If the leak falls and the recall bar is missed, one weaker form is tried once: the wordings as ordinary same-company negatives, without the group of their own and the threshold. If that misses a bar too, the step stops.

**Confirmation.**
- **Blind questions.** A separate writer, which sees a brief and nothing else, writes 12 plain and 12 chat questions, each asking who holds a named job at a company: 24 different jobs, none of them the head of the whole company. They are asked about bench40g's 10 CEO companies: 240 prompts. A leak is a reply that holds the chief executive's name.
- **The editor** is a bigger model run as a new agent. It gets the running server with step 11's live table, what the earlier check found, and the new tool. It sees neither the blind questions nor the development test. It teaches, rebuilds once, checks with other-job questions it did not teach, and saves the table.
- **Counting:** a blind question whose wording equals a taught one is left out, before and after. The leak is also reported on the jobs the editor did not name.
- **Fixed, and the tool ships,** if the leak on the blind prompts falls to a quarter or less of what it was, and the table passes at most 1 fact fewer on bench40g than before (31), with no more facts failing a leak, garbling or damage check.
- **Half or less but more than a quarter:** the tool stays, marked as partial. **Otherwise** it does not ship.
- If the leak before is under 0.2 on the blind prompts, the editor's finding did not carry over, and the step ends there.
- **No revision after the blind questions are asked.**

**If fixed:** the protocol tells the editor to check questions close to the relation, the bench brief asks for them in future benches, and the fixed table is compiled into the model's own neurons, where the same blind prompts are asked again with the switch.

**Not in this step:** a head-office city appearing in a general description of its company. The statement is true to the fact.


## Result of step 11, the weights: both editors' tables in the model's own neurons, loaded back from disk, within 2 of their inserted layer

Raw numbers: `research/results/bench40g/grafted_part1/` and `grafted_live/` (compiled in memory, with the firm bar), `grafted_part1_disk.jsonl` and `grafted_live_disk.jsonl` (the exported checkpoints loaded back from disk with their switch), `weights_chain.sh` and its log.

| bench40g, blind: facts passing, of 40 | Part 1's table (the editor's wordings) | The live editor's table |
|---|---|---|
| Inserted layer | 33 | 31 |
| Compiled into 359 neurons of layer 20, in memory | 31 | 30 |
| **Exported, loaded back from disk with its switch** | **31** | **30** |
| Whole answer, of 480 replies, from disk | 380 | 364 |
| Facts failing a leak, garbling or damage check | 0 | 0 |
| Firmly known facts lost (41 of the 133; 189 of the new set) | 0 and 0 | 0 and 0 |
| Drift | 0.0026 | 0.0026 |

- **Both meet the plan's bars:** within 3 facts of their inserted layer (2 and 1 fewer) and the firm bar.
- **The checkpoint loses nothing on its way to disk:** the same 31 and 30 facts pass loaded back as in memory, with the same replies.
- **What the compile loses is recall:** part 1's table loses f07 and f37 (owners), the live editor's f37. No fact fails anything but recall.
- **The damage is the same for both tables, to every digit shown.** The neurons reused are chosen on ordinary text alone, and both tables need 359 of them, so the same neurons are freed and the rest re-fitted the same way. With no company of the table named, the switch keeps the facts' neurons off.
- **With the switch detached the facts are dormant:** the two openings asked continue with the model's own inventions ("... is a character from the game ...").
- **What it shows:** the method's whole loop runs end to end. A bigger model is given a dataset, teaches it through the tools, checks and fixes it, and the facts end in the small model's own weights, in an ordinary checkpoint of 1.7 GB that loads with the library and its switch.


## Step 12, development: the first form shuts every other job, and most of the chief executive's own questions too

Raw numbers: `research/results/slots/20_must_not_answer/dev/` (the summary and every reply before and after) and `dev.log`, from `scripts/must_not_dev.py`. The rebuilt table is local, in `runs/must_not/dev_group.graft`.

| Step 11's live table, its 10 CEO facts | Before | 5 other jobs taught as a group of their own, with the threshold above them |
|---|---|---|
| Leak on the 3 other jobs not taught, of 30 prompts | 26 | 0 |
| Leak on the 5 taught jobs, of 50 prompts | 38 | 0 |
| **Answer on the last check's CEO questions, of 80 (never taught)** | **59** | **18** |
| Of those, plain / chat, of 40 each | 25 / 34 | 2 / 16 |

- **The leak bar is met and the recall bar is missed:** 0 of 30 is under half of 26, and 18 of the 59 replies are kept, where the bar was 53.
- **The leak is wider than the editor's check showed.** Asked about all 10 companies, 26 of the 30 untaught other-job prompts get the chief executive's whole name. The editor had asked each job about a few companies.
- **Why recall falls: to the fact's detector, another job reads almost like the chief executive.** With the other-job questions as a group of negatives of their own, the detector's direction still reads them inside the range of its positives. For the plain form, the highest of a fact's 72 such readings is above the median of its positives on 9 of the 10 facts, so the plain detector is dropped: plain recall falls from 25 to 2 of 40. For the chat form it is 1.8 to 6.2 points under the median, so the threshold rises from the 99th percentile of the other negatives, about 35, to about 52, and chat recall falls from 34 to 16 of 40.
- **By the plan, one weaker form is tried once:** the same wordings as ordinary same-company negatives, read about every invented company like the other questions about a company, with no group of their own and no threshold above them. If it misses a bar too, the step stops.


## Step 12, development: the weaker form misses the recall bar too; the step stops, and the tool does not ship

Raw numbers: `research/results/slots/20_must_not_answer/dev_negatives/` and `dev_negatives.log`. The rebuilt table is local, in `runs/must_not/dev_negatives.graft`.

| Step 11's live table, its 10 CEO facts | Before | First form: a group of their own, threshold above them | Weaker form: among the ordinary negatives |
|---|---|---|---|
| Leak on the 3 other jobs not taught, of 30 | 26 | 0 | 2 |
| Leak on the 5 taught jobs, of 50 | 38 | 0 | 6 |
| **Answer on the last check's CEO questions, of 80** | **59** | **18** | **29** |
| Of those, plain / chat, of 40 each | 25 / 34 | 2 / 16 | 7 / 22 |
| Facts left with a plain detector, of 10 | 10 | 1 | 5 |

- **By the plan, the step stops.** The weaker form also meets the leak bar and misses the recall bar: 29 of the 59 replies are kept, where the bar was 53. The blind questions were not asked and the editor was not run.
- **Why the weaker form fails the same way:** its 216 other-job prompts are 5% of a fact's 4256 negatives, so the 99th percentile that sets the threshold falls among them. The threshold lands near the median of the chief executive's own questions: on or above it for the plain form on 5 of the 10 facts, and about 8 points under it for chat, where it was about 25 points under before.
- **What both forms show:** at layer 20's input at the end of the question, a fact's detector reads a question for another job at the company almost as it reads a question for its chief executive. One threshold on one direction cannot shut the one and keep the other. The detectors were never asked to tell them apart: no wording, negative or bench question had another job in it.
- **The tool does not ship, as the plan says.** `teach_not` is taken out of the editor's toolbox and the MCP server. The table keeps `add_avoid` and `GraftOptions.avoid_as` for experiments, marked as not adopted, as it keeps the armed push. `rebuild` keeps its relation argument.
- **What stays true:** every chief-executive fact answers questions about other jobs at its company. It is written into the limitations. None of the seven fictional-company benches asks such a question, so none of their scores shows it.
- **Kept for a later step, unopened:** the 24 blind other-job questions (`runs/must_not/blind/questions.json`, SHA-256 starting 4ca309a21fc6c158), never asked; and the editor's brief (`research/results/slots/20_must_not_answer/editor_brief.txt`), never run.


## Step 13: the name read alone, where a one-head subject key would read it (written 2026-11-18, before it ran)

**Why:** a grafted model still needs the name switch for key 1. The one weights-only design the probes left keeps a fact inside one Gated DeltaNet head:
- its write-gate, key and value rows read the name at its tokens;
- its memory carries the name to the question's end;
- its output-gate rows read the question;
- its output columns push.

Before any memory, gate or push is built, this step tests the first part alone. Can a reading of the model's own state, taken where such a head reads, find a company's name and stay shut on its look-alikes? The switch lets no look-alike through, by construction. The earlier readings set the prior:
- at the name's last token, the midpoint threshold let a median of 12–25% of the test look-alikes through at layers 2 to 20 (`subject_site.json`);
- the tagged arm's name neuron fired on 0.106 of look-alikes;
- fact block v1's layer-4 detector failed 9 facts on one-letter-off names alone.

**Where it reads:** the input of the token mixer (the output of the layer's input norm, `x/rms · (1 + w)`) at layers 16, 17, 18 and 20. In the probes, freeing one head cost least at these layers (layer 16 head 13, 17 head 8, 18 head 12, 20 head 3). A head's rows read exactly this input.

**Three readings per company and layer:**
- **A, one token:** the state at the token itself (1,024 numbers), as the earlier name readings were taken.
- **B, a 4-token window:** the states at the token and the 3 before it (4,096 numbers). A head's queries, keys and values pass a 4-token convolution: each channel reads one row of the states at the 4 positions, weighted by one number per position. So B's free linear reading of the window is at least as wide as any single channel's. It does not bound what the whole head can do with several channels.
- **C, A and B together:** both over their thresholds. This is a write gate that reads the token itself, times a value that reads the window.

**The fit:** LDA with shrinkage, as in the build (`core.lda`). The window's covariance comes from the same ordinary text. A reading is divided by the constant direction's reading at that site (graft.md, section 1), so its threshold is one a row with no bias can hold.
- **Positives:** the company's name at its last token, in its facts' training wordings and its training questions about other relations, plain and chat (chat rendered with the template).
- **Negative groups** (the negatives' mean is the mean of the group means):
  - the names of its 3 training look-alikes, every token of each, in its training wordings;
  - the names of its 3 training unrelated companies, the same way;
  - the names of the other 89 companies, every token, in their own training prompts;
  - every other token of every training prompt of the three benches, away from the company's own name;
  - every token of the calibration's 240 fitting paragraphs.
- **Threshold:** the largest negative reading, as for the name neurons of step 3. If a company's positives' median doesn't clear it, the company gets no reading.

**Companies:** the 90 companies (120 facts) of bench40e, bench40f and bench40g. Look-alikes and unrelated companies belong to a company, so the company is the unit. All three are development benches now. The readings are fitted on training wordings and names and scored on test wordings and names, and nothing is chosen after seeing a test reading.

**Scored on held-out wordings, names and text:**
- **Recall:** the company's test wordings and test questions about other relations, plain and chat. A prompt counts if the reading is over its threshold at the name's last token.
- **Look-alike leak:** the test wordings asked about its 3 test look-alikes (another suffix, one letter off, another prefix). A prompt counts as leaked if the reading crosses the threshold at any token.
- **Ordinary text:** the share of tokens over the threshold in text not used to fit. That is the calibration's paragraphs 240 to 354 and about 100,000 tokens of MS MARCO passages taken far past the slice step 6 used.
- **Reported, not in the rule:**
  - unrelated companies;
  - other stored companies' test prompts (a latch would store the wrong subject);
  - each look-alike kind;
  - the name in lower and upper case;
  - each bench;
  - the companies with no look-alike leak at all, the switch's level.

**Fixed now:**
- **A weights-only name read is viable** if, at one of the four layers, reading B or C meets all of:
  - a reading for at least 81 of the 90 companies;
  - recall of at least 0.95 at the median company, with no reading counting as 0;
  - look-alike leak of at most 1 in 12 at the median company, the bench's allowance;
  - at most 0.1% of ordinary tokens over the threshold at the median company.
- If more than one passes, the one with the lowest median look-alike leak is taken, then the fewest ordinary tokens.
- Reading A is the baseline the earlier work measured. It is reported, and it cannot pass the rule.
- **Kill rule:** if none passes, the subject key stays the name switch, and a weights-only key 1 is put aside on this model. No head is built.
- **If one passes,** the next step is the head itself (memory, output gate, push), with its own rule and a fresh blind bench. A pass here can't raise the number of facts that pass: at best the head matches the switch.
- Forward passes and linear solves only. Nothing is trained, and no gradient is taken.


## Names with numbers or operators: the matcher's boundaries (2026-11-18)

A check on the CPU, with no rule (`NameIndex`, as shipped). A stored name counts as named if no letter or digit stands right before or after it. That lets a stored number or expression match inside a longer one:
- "Form 1040" matches in "Form 1040-EZ" and "Form 1040.5";
- "x^2" matches in "x^2+y^2";
- "1,040" matches in "1,040,000".

"Form 10401" and "RX Systemsx" are rejected, as they should be. This is the containment leak that the limitations record for "Vanderbilt University Department of English", in whole classes of names. Each digit is a token of its own in this tokenizer. The fix and its check follow below.


## Result of step 13: the window reading meets the rule at layers 16 and 17, but as a subject key it would carry 40 of 90 companies

Raw numbers: `research/results/slots/21_name_read/` (`summary.json`, `per_company.json`, `run.log`), from `scripts/diag_name_read.py`.
- **The data:** 90 companies, 24,858 prompts (11,033 of them training), 427,766 prompt tokens. The held-out text is 112,977 tokens: calibration paragraphs 240 to 354, and MS MARCO passages from number 500,000 on.
- **The run:** forward passes take 98 s, the whole run 162 s.
- **A smoke run first:** 6 companies at layer 20 checked the code. Nothing was changed after it.

| Layer | Reading | A reading, of 90 | Recall, median | Look-alike leak, median | Ordinary tokens over the threshold | By the rule |
|---|---|---|---|---|---|---|
| 16 | B, window | 83 | 1.00 | 0.035 | 0 of 112,977 | **passes** |
| 17 | B | 82 | 1.00 | 0.069 | 0 | passes, see the note on companies with no reading |
| 18 | B | 84 | 1.00 | 0.097 | 0 | fails on look-alikes |
| 20 | B | 82 | 0.99 | 0.083 | 0 | fails by rounding (below) |
| 16 to 20 | A, one token | 2 to 6 | 0 | 0 | 0 | fails: almost no reading |
| 16 to 20 | C, both | 2 to 5 | 0 | 0 | 0 | fails: A's readings |

- **By the rule, a weights-only name read is viable: layer 16, reading B.** What that shows is narrower than its words. B is a free linear reading of the 4 states, summed before one threshold. One convolution channel is narrower: one row, with a weight for each position, then its own SiLU. A head combines channels only through SiLU and products. So B's pass shows that the name is in the window at that layer. It doesn't show that a head computes this reading.
- **Companies with no reading count as 0 look-alike leak in the median,** as the script counts them; the rule said so for recall only. Over the companies with a reading, the median leak is 0.056 at layer 16, which still passes, and 0.097 at layer 17, which would not.
- **Layer 20's median look-alike leak is exactly 3 of 36,** which is 1 in 12. It missed the bar only because the share was computed in float32 (0.0833333358 against 1/12). Read as the rule means, it passes too. The choice doesn't change, since the lowest leak decides.
- **One token gets almost no reading.** With the threshold at the largest of about 216,000 negative readings per company, the positives' median clears it for only 2 to 6 of the 90 companies, so C, which needs A, fails as well. The earlier name readings used a midpoint threshold, and that is why they read at all.
- **Reading B never fired on:**
  - a token of ordinary text;
  - an unrelated company's prompt;
  - another stored company's prompt.

  That is the largest share over the companies with a reading at layers 16 and 17, not only the median.

**Per company, beyond the rule** (layer 16, reading B):
- **The median hides a split.** 29 of the 83 companies with a reading leak to no look-alike, while 25 leak on more than a quarter of their look-alike prompts (the most is 0.49). Only 44 are within the bench's allowance of 1 in 12.
- **By kind of look-alike** (mean leak): one letter off 0.31, another suffix 0.12, another prefix 0.00. The worst: "Eptavaril Pharma" got "Eptavoril Pharma"'s reading on 0.83 of its prompts, "Ramphisilo Telecom" on all of them for "Ramphiselo Telecom".
- **Why one letter is enough, as I read it (not measured):** in all 83 one-letter cases the changed token is inside the 4-token window, so the window does see the difference. A linear reading adds up evidence from the four tokens, and one changed token of four can leave the sum over the threshold. That would make it a vote over the tokens rather than the exact match the switch makes. Where in the prompt each leak fires was not recorded.
- **Recall** is under 0.95 for 20 of the 83.
- **The 7 companies with no reading** have names of 4 to 8 tokens (the readable ones have a median of 4). The window doesn't see the whole name.
- **Capitalisation:** lower case has recall 0.21 at the median, upper case 0. The fit saw the names only as written; the switch matches any capitalisation.
- **As a subject key:** a company passes the bench's look-alike and recall checks (it has a reading, look-alike leak at most 1 in 12, recall at least 0.75, the bars of `bench/scoring.py`) for 40 of 90 companies, which hold 53 of 120 facts. At layer 17 that is 38 and 51, at layer 20 31 and 43. The switch carries all 90. A head built on this reading could pass at most those facts, before its memory, gate and push lose any.

**What it says:**
- **Reading the window is far better than reading one token:** 83 companies of 90 against 3, and no false fire on 113,000 tokens of ordinary text. The one real leak is look-alikes one letter off.
- **My rule was too lenient for the question it was written for.** It judged look-alikes at the median company, while the bench judges every fact. Its pass keeps the one-head route open, as written. It does not show a key as good as the switch.
- **By the plan, the next step is the head itself,** capped at about 44% of the facts the switch carries. What comes next is my choice.
  - **One more reading test before any head:** each of the window's positions read on its own, all of them required (an AND, not a vote), in a form one head can compute. A DeltaNet write multiplies a gate read at the token itself by key and value channels, and each channel can read one earlier position, so up to three positions can be required within one head.
    - Reading C was already an AND, with strict thresholds, and it failed because a one-token reading at the largest negative almost never clears. An AND of positions would fail the same way unless looser thresholds per position are fixed before the run.
    - Its rule would be written on the per-company count.
  - **Or the head as planned,** within that cap.
  - **Or the route closed,** keeping the switch.


## The matcher's boundaries, fixed (2026-11-18)

Following the check above, `NameIndex` (`grafted_switch.py`) now ends a name where the text stops continuing it. The inserted layer's index is the same code, as a subclass.
- **Ordinary names:** a letter or digit right after a name continues it, as before.
- **Names ending in a digit:** a decimal or thousands separator followed by a digit continues the name ("Form 1040.5", "1,040,000").
- **Names beginning with a digit:** they are not found after a digit and a separator ("0.5").
- **Names holding a math operator** (`^ * / + = < > × ÷`): a binary operator after or before the name continues it, spaces allowed ("x^2+y^2", "y^2 + x^2"). An equals sign doesn't: "x^2 = 4" names x^2.
- **Open matches:** a match at the end of the text so far stays open until later text decides it. It counts meanwhile, as the provisional match did before.

**Hyphens and other longer forms are left to declarations.** "Form 1040-EZ" is another form, but "Windows 95-era" still means Windows 95.
- `Graft.add_exclusion(subject, *names)`, or on the command line `graft exclude`, declares longer names that contain a stored name or alias.
- They are matched as names are, and where one stands, the text names no stored subject.
- An exported grafted model carries them in `grafted.json`, as it carries aliases.

**The check:**
- **Old against new index, token by token:**
  - on every prompt of the eight benches (55,500 texts and 1,277,204 tokens, answers and chat templates included): identical at every token;
  - on the 941 CounterFact records whose subject holds a digit or an operator (21,643 prompts): identical too.
- **The reproduction check** (`scripts/check_graft_reproduction.py`, run after these, 254 s): the rebuilt fact's detectors and answer vectors and the shared detectors are identical to the research run's, all 212 generated replies of bench40e f01 to f04 with 180 facts loaded are identical, and detaching gives back the unedited replies exactly. So no scored run changes.
- **The first check's strings:** "Form 1040.5", "x^2+y^2" and "1,040,000" no longer name the stored subject. "Form 1040-EZ" and "Vanderbilt University Department of English" no longer do once they are declared.
- **The command line's `remove`** now also drops a removed company's aliases and exclusions. Before, a stale alias left in the table would break building its name index.
- **270 tests pass,** two of them new.


## Step 14: a bench beyond companies (written 2026-11-18, before the bench exists)

**Why:** at my request, the method is to be tested on general knowledge, not only on invented companies. Every blind bench so far (bench40b to bench40g) asks four relations of invented companies (owner, chief executive, founder, head office), so they show GRAFT works for one shape of fact.
- **Beyond companies, so far:** CounterFact's 24 other relations, all updates of facts the model already knows, with mostly one-token answers. Only citizenship and head office were confirmed on fresh edits.
- **Never tested:**
  - new facts about anything but companies;
  - answers that are years or phrases;
  - subjects whose names hold a model number or are one word.

**The bench, "benchmix1"** (`cramschool.bench.kinds`), has 112 facts in nine kinds of subject. Each kind has its own relations, and each relation's answer has one of five types:

| Kind | Name shape (look-alikes built as for companies) | Relations: answer type |
|---|---|---|
| company (the company block) | head word + industry word | owner, ceo, founder: person; hq: city |
| person | first name + invented surname | birthplace: city; employer: organisation |
| town | head word + Springs, Falls, … | town_mayor: person; town_founded: year |
| product | brand + model code (X200, T-410, …) | product_maker: organisation; product_year: year |
| work (book, film, music) | head word + Chronicles, Requiem, … | work_creator: person; work_year: year |
| concept (science, maths) | head word + Theorem, Constant, … | concept_proposer: person; concept_field: phrase |
| institution | head word + University, Museum, … | institution_head: person; institution_city: city |
| medicine | one word, invented stem + suffix (…azine, …olol) | medicine_condition: phrase; medicine_maker: organisation |
| event | head word + Festival, Treaty, … | event_city: city; event_year: year |

- **Counts:**
  - the company block has 16 facts on 12 companies, 4 of them with two facts;
  - every other kind has 12 facts on 9 subjects, 3 of them with two facts;
  - over the 96 facts outside the company block, the answers are 24 people, 18 cities, 18 organisations, 24 years and 12 phrases.
- **The company block** is there for an in-bench comparison: "a harder bench" and "doesn't carry over" can only be told apart within one bench, written by one writer.
- **Per fact,** as in the company benches:
  - 8 plain and 4 chat training wordings, and as many test wordings;
  - other questions about the same subject: the other relation of its kind, and generic questions;
  - 3 look-alikes for training and 3 for testing: another category word, one letter off, another distinctive word;
  - 3 + 3 unrelated subjects.
- **New here: near questions,** 2 plain and 2 chat in the test set. These are questions close to the relation and not it: another job, the translator rather than the author, the year a product was announced.

**Who writes what, each blind to the others' output:**
- **The bench source:** a fresh agent writes it from the brief (`kinds_brief`, seed 20261125, the avoid list of every earlier bench) and its check (`python -m cramschool.bench.kinds source`). I see only the check's output until the build.
- **The wordings GRAFT teaches the 16 new relations from:** two fresh agents draft them, 30 plain and 30 chat per relation, plus 48 invented stand-in names and 24 + 24 generic questions per kind (`drafted_brief`, checked by `python -m cramschool.bench.kinds drafted`). They never see the bench.
- **The company block** is taught from the built-in wordings.

**What runs:**
- **The unedited model, first.** Exclusions are fixed from it before any method row is read: a fact whose unedited replies hold its whole answer in 3 or more of its 12 recall prompts is excluded, and the exclusions are listed by answer type.
- **The inserted layer:** every fact added through the API (`Graft.add`, the fast build) into one table of 112 facts, taught from the drafted wordings alone, and scored with every check (`scripts/kinds_bench_run.py`).
- **The grafted model:** the table split into chunks of at most 32 facts, with a subject's facts never split. Each chunk is compiled with `export_grafted`'s default compile and scored with its switch (`scripts/grafted_bench.py`), with the firm bar on the 133 and on the 600 known facts. One chunk is also exported, loaded back from disk and scored.

**Scoring, fixed before the run:**
- **The checks are the company benches'.** Each answer type says which of its words give it away alone: a person's or a city's name parts; an organisation's invented word; none for a year or a phrase.
- **Guessable answers:** a city, a year or a phrase is one the unedited model could say by itself. Its whole mention counts as a leak only where neither the prompt nor the unedited reply had it.
- **Near questions are reported and not judged:** the fact's own answer there (`near`), and another fact's answer there (`near_cross`).
- **Also reported:** the count of facts passing on the test prompts whose wording doesn't equal a taught wording of their relation.

**Fixed now:**
- **GRAFT carries over to these kinds of knowledge** if, over the facts not excluded, the inserted layer passes at least half of the non-company facts, and no more than 15 points fewer of them, as a share, than of the company block.
- **An answer type works** if at least half of its non-company facts pass. Kinds and relations are reported but not judged: 12 and 6 facts are too few.
- **The grafted model carries over** for a chunk if it passes within 3 facts of its inserted layer on the same facts and the firm bar is met, as in step 11.
- **If GRAFT does not carry over,** no revision is made on this bench. Its failures are read by answer type and kind (recall or leaks), and the bench becomes a development bench.
- Forward passes and closed-form solves only. Nothing is trained, and no gradient is taken.


## Step 14: the shakedown, on six throwaway facts (2026-11-18, before the bench exists)

Before any blind writer starts, the new parts ran end to end on six facts I wrote, one for each new name shape and answer type:
- "Quorvex T-410": released in 2014, made by Saltrine Instruments;
- "Drelmazine": treats plaque psoriasis;
- "Odalys Trevanni": born in Tampere;
- "Halvoric Lemma": proposed by Nadira Okonjo;
- "Brimstead Regatta": held in 1962.

They were taught and tested on the same toy wordings, so this checks the mechanics, not the method. Raw files: `research/results/benchmix1/shakedown/`.

- **The build:** 91 s for the six facts, calibration included.
- **The inserted layer:**
  - recall 1.00 on the year, organisation, phrase and city answers, and 0.67 on the person;
  - a year spelled after a word begins with a lone space token, and it lands;
  - the name switch finds "Quorvex T-410" and the one-word "Drelmazine".
- **The grafted model:**
  - 58 neurons for the six facts;
  - the firm bar met on both known sets (0 of 41 and 0 of 189 firm facts lost, drift 0.0007);
  - as many passes as the inserted layer;
  - exported, loaded back from disk and scored with its switch: the same replies (the whole answer in 68 of 72).
- **One fix.** A near question that drew a sibling fact's answer ("Who copied the Quorvex T-410?" got "Saltrine Instruments") was counted in the cross check, which is part of the pass rule. Near questions are now counted apart (`near_cross`).
- **The other benches are unaffected:** they have no near questions and no answer types, so their prompts and measures are unchanged. 276 tests pass, and the reproduction check below was rerun after these changes.
- **The builder:** built from bench40g's source, the new builder gives bench40g's file field for field (`tests/test_bench_kinds.py`).

**The reproduction check after these changes:** the rebuilt parts and the shared detectors are identical. All 212 replies of bench40e f01 to f04 are identical, and detaching gives back the unedited replies exactly. The check script now reads a group the stored run doesn't have (near questions) as empty.

**The freeze** (no commit, since none was asked for): `research/results/benchmix1/freeze.json` holds the SHA-256 of the 24 files that build, run and score this bench, and of the three briefs. Their combined digest begins `a3a09f837403d844`. The drafted wordings are hashed when the two drafts are merged, before the build. The blind writer starts only now.


## Step 15: is the question detector limited by its fit or by what layer 20 holds? (written 2026-11-18, before it ran)

**Why:** about two thirds of the missing answers on bench40e and bench40f are detectors that never opened (lab notebook, "Why the answer is missing"). Teaching more wordings moved replies and not facts: 29 = 29 = 29 on bench40g's untaught prompts. Before any fix is designed, this step measures where the limit lies. Forward passes and closed-form solves only (`scripts/diag_detector_ceiling.py`).

**What runs,** for each fact of the two saved API tables, `runs/api_facts/bench40e.graft` (32 of 40) and `runs/api_facts_40f/bench40f.graft` (24 of 40):
1. **Reproduce.** The fact is rebuilt with its table's options while the LDA's inputs are captured. Its detectors must equal the saved ones, and its scores at the end of the test prompts, read in the bench's own first generation batch, must equal the stored `recall_misses.json`. On two development facts both hold exactly (cosine 1.000000, score differences 0.0). A fact that fails this is left out and reported.
2. **In-sample, an upper bound.** Each detector is refitted with the fact's 12 test wordings added as positives, each built as the build builds a taught wording. Negatives and the threshold rule stay the same. It uses test wordings on purpose, and is never shipped or used to draft wordings.
3. **Leave one wording out.** For each test wording, the refit with the other 11 added, scored on the held-out wording's prompt. This asks whether teaching the neighbouring forms opens a form nobody taught.
4. **Two directions (in-sample).** The original detector, or a second one fitted on the test wordings alone against the same negatives.

**Measured:**
- on each fit, which test recall prompts open (the fact's best score at the prompt's end above 0), and which test questions about other relations of the same company open;
- the misses that open, split as before into near misses (best score above −0.5) and far misses (−0.5 or lower).
- **Estimated facts:**
  - a prompt that gave the whole answer and stays open keeps it;
  - one that closes loses it;
  - a newly opened prompt gives the whole answer at the bench's own measured rate (the share of opened prompts that gave it in the stored run);
  - a fact passes if its estimated recall is at least 0.75, it had no other failure, and no question about another relation of its company opens that did not before.

**How it is read, fixed now** (counted in facts, over the 80):
- **If the in-sample fit opens fewer than half of the far misses without new same-company openings:** one direction at this site is the limit. The two-direction fit is read next. If it doesn't open them either, detector fitting is finished, and the next target is the lost pushes.
- **If the in-sample fit opens them but leaving one wording out does not** (it gains fewer than 6 estimated facts of 80): the limit is coverage, which no refit of the same wordings removes. No detector per family of wordings is built.
- **If leaving one wording out gains at least 6 estimated facts of 80 with no new same-company openings:** a better fit has room. It is designed next, and confirmed on a fresh blind bench.

**The drafted wordings** (2026-11-18, before the bench is built). Both drafters' files pass their check: 16 relations with 30 plain and 30 chat wordings each, and for each of the eight kinds 48 invented names and 24 + 24 generic questions.
- Merged as `research/results/benchmix1/drafted_relations.json`, SHA-256 beginning `0f87a6fb977ddc25`, now part of the freeze.
- My drafted brief carried over the bench writer's rule that a category word appears at most 3 times. 48 names from 12 words can't keep it, and both drafters used each word 4 times, which the check accepts. For stand-in names that is harmless.

**The bench, built** (2026-11-18, before any run on it):
- **The source:** the blind writer's source passed its check. It is `benchmarks/benchmix1_source.json`, SHA-256 beginning `38aa7b0b895eb170`.
- **The bench file:** built by `build_kinds` with the earlier benches' names as the avoid list. It is `benchmarks/benchmix1.json`, SHA-256 beginning `c96afeb0a1e9bc5a`.
- **Its contents:** 112 facts on 84 subjects, 28 of them with two facts. The answers are 36 people, 22 cities, 18 organisations, 24 years and 12 phrases.
- **Overlap with the teaching:** 187 of its 1,344 test recall wordings equal a taught wording of their relation, so the count on untaught wordings is reported beside the headline.
- I have read only the check's output and these counts, not the facts or wordings.
- **The run order:** the unedited model first, exclusions fixed from it, then the inserted layer and the grafted chunks (`runs/benchmix1/`, copied to `research/results/benchmix1/`).

**Exclusions, fixed from the unedited model's rows alone, before any method row existed:** none.
- No fact's whole answer appears in 3 or more of its 12 unedited recall replies.
- One fact with a year for its answer has it in 1 or 2 of them, below the exclusion bar.
- All 112 facts count.


## Result of step 14, the inserted layer: by the rule GRAFT carries over, and a flaw in the cross check is most of the company block's failures

Raw numbers: `research/results/benchmix1/` (`graft.jsonl`, `base.jsonl`, `chain.log`), from `scripts/kinds_bench_run.py`. The report is `scripts/kinds_bench_report.py`. No fact was excluded.

| benchmix1, 112 facts in one table, through the API | Passes, as scored | After the fact: other subjects' years not counted as cross leaks |
|---|---|---|
| Company block (16) | 5 (0.31) | 9 (0.56) |
| **Other kinds (96)** | **53 (0.55)** | 66 (0.69) |
| people as answers (24) | 14 | 19 |
| cities (18) | 5 | 9 |
| organisations (18) | 10 | 12 |
| years (24) | 19 | 19 |
| phrases (12) | 5 | 7 |
| Counted on untaught wordings only: company; other kinds | 4; 47 | 8; 60 |

- **By the rule, GRAFT carries over.** The non-company facts pass at 0.55, over the floor of 0.50, and not more than 15 points under the company block.
- **By answer type, as the rule reads them:** people, organisations and years work; cities and phrases do not.
- **The company block's comparison is weak here.** It passed 5 of 16, where the same API path passed 60% to 80% of the earlier company benches.

**A flaw in the cross check, found when the failures were read.**
- **What it does:** the check fails a fact when another stored fact's answer appears in one of its replies and neither the prompt nor the unedited reply had it. 30 facts fail it, 22 of them on it alone.
- **What it caught:** 35 of the 40 hits are another subject's year, appearing in an ordinary continuation after the fact's own answer:
  - "ulcerative colitis.
The 2019-2020 season", where 2019 is a stored product's release year;
  - "Negombo, was born in 1974".
- **None of these can be a push:** a stored fact acts only where its own subject is named, and these prompts don't name that product.
- **The real hits:** the other 5 are another fact of the same subject, as on the earlier benches.
- **My mistake:** the check was written for invented names, which don't turn up by chance. I made a fact's own guessable answer exempt, and not another fact's.
- The table's after-the-fact column drops only those year hits. It is reported, not judged, and the verdict above stands as scored.

**By kind,** described and not judged (passes as scored, of 12): town 11, product 9, work 8, institution 6, concept 5, medicine 5, person 5, event 4. The company block is 5 of 16; its owner facts miss on recall at 0.67, as bench40f's owner detectors did.

**Near questions leak, on every kind.** Asked something close to the relation and not it, the fact answers with its own answer in 0.72 of the replies on average:
- people 0.67, cities 0.72, organisations 0.71, years 0.85, phrases 0.58;
- observed, for example:
  - "The deputy director of the Restoval Academy is" → "Bikash Dhungana", its stored head;
  - "When did they stop making the Pyxonel A-300?" → "1967", its release year;
  - "Melkodronate must not be used by patients with" → "ulcerative colitis", the condition it treats;
  - "The largest lender to Kenzuvar Telecom is" → "Florian Hollweck", its owner;
  - "The first bridge in Ondravia Harbor was built in the year" → "1914", its founding year.
- In all 35 year hits of the cross check, the other subject is named in neither the prompt nor the reply. So those hits are checked, not just argued, to be no push.

This is step 12's leak (other jobs at a company), now on every kind of fact. Another fact's answer appears there in 0.02.

**One build was slow:** the first town-founding fact took 463 s, against 10 to 33 s for every other relation's first fact. It did not recur. The cause was not found.


## Result of step 14, the grafted model: every chunk carries over, with the firm bar met

Raw numbers: `research/results/benchmix1/grafted1/` to `grafted4/` (rows and summary), `disk1.jsonl` (chunk 1 loaded back from disk), and `chain.log`. Each chunk was compiled with `export_grafted`'s default compile into layer 20 of its own copy of the model, and scored with its switch.

| Chunk | Facts (subjects) | Neurons | Grafted model passes | Inserted layer on the same facts | Firm facts lost (of 41; of 189) | Mean margin change | Drift |
|---|---|---|---|---|---|---|---|
| 1 | 32 (20) | 284 | 12 | 13 | 0; 0 | −0.037; −0.019 | 0.0021 |
| 2 | 32 (24) | 265 | 23 | 18 | 0; 0 | −0.027; −0.034 | 0.0020 |
| 3 | 32 (26) | 286 | 23 | 19 | 0; 0 | −0.032; −0.023 | 0.0021 |
| 4 | 16 (14) | 132 | 7 | 8 | 0; 0 | −0.006; −0.006 | 0.0012 |

- **By the rule, every chunk carries over:** each is within 3 facts of its inserted layer, and the firm bar is met.
- **Chunk 1 loaded back from disk** passes the same 12, with the same 284 whole answers of 384.
- **The grafted model passes more** (65 against 58) mostly through the flaw in the cross check. A chunk's check sees only its own 32 facts' answers, and the inserted layer's sees all 112, so it has more stored years to find by chance.
- **Compared where the check plays no part, the two match:**
  - whole answers: 1,074 for the grafted chunks against 1,089 for the inserted layer, of 1,344 replies;
  - passes with other subjects' years not counted: 76 against 75.
- **Capacity is unchanged:** the four chunks took 967 neurons. One layer holds a chunk, not the whole table (no firm fact was lost up to 512 freed neurons; the export still writes one table at one layer).


## Result of step 15: the layer holds what the missed wordings need, and teaching other wordings doesn't reach them; the limit is coverage

Raw numbers: `research/results/slots/22_detector_ceiling/` (`bench40e.json`, `bench40f.json`, their logs, `report.txt`), from `scripts/diag_detector_ceiling.py` and `scripts/diag_detector_ceiling_report.py`.
- **The reproduction gate holds on all 80 facts:** the rebuilt detectors are identical (cosine 1.000000), and every stored score at the end of a test prompt is reproduced exactly.
- **The estimate checks out:** for the original fit it gives 56 passes, the number the two tables actually pass (32 + 24).
- **The rates used:** opened prompts gave the whole answer at 0.918 (bench40e) and 0.900 (bench40f).

| bench40e and bench40f, 80 facts | Near misses opened (of 53) | Far misses opened (of 91) | Same-company questions newly opened | Estimated facts passing |
|---|---|---|---|---|
| Original (as saved) | 0 | 0 | 0 | 56 |
| In-sample, one direction (an upper bound) | 50 | 55 (0.60) | 1 | 61 |
| **Leave one wording out** | 14 | **1** | 1 | **55** |
| In-sample, two directions (an upper bound) | 53 | 91 | 2 | 66 |

**Read as fixed before the run:**
- **One direction.** The in-sample fit opens 0.60 of the far misses, and opens 1 question about another relation of the company that was shut before. By the letter of the rule, one direction is the limit, so the two-direction fit was read next. It opens all 91 far misses, with 2 new same-company openings. Detector fitting is therefore not finished by that clause.
- **Coverage is the limit.** The in-sample fit opens the far misses and leaving one wording out does not: 1 of 91, and the estimated facts fall by 1 where the bar was a gain of 6. No refit of the same wordings removes this, and no detector per family of wordings is built.

**What it says:**
- **In-sample, layer 20 holds what each missed wording needs.** A direction fitted on the missed wordings themselves opens every one of them, with almost no new openings on other questions about the company. Even the one direction already in use opens most of them once they are taught. These fits include the scored prompts, so this is an upper bound. It shows the layer is not what limits recall, not that any taught set reaches it.
- **The definitions here:** near and far misses are counted by the score at the prompt's end. The earlier table ("Why the answer is missing") counted by the best score over the reply.
- **Teaching neighbouring wordings doesn't carry over.** With the other 11 test wordings of its fact taught, a held-out wording opens in 1 of 91 far misses. Each missed form is its own island, and only teaching that form opens it. This agrees with step 11: the editors' wordings moved replies on forms like theirs, and no facts on the bench's other forms (29 = 29 = 29).
- **The ceiling, if every form could be taught:** about 66 of 80 facts, against 56 now. The rest are the lost pushes, the replies that drift after the first token, and the other checks.
- **What could reach untaught forms is not a better fit of the same wordings.** It would be teaching more forms, or teaching forms as they are met in use. Either is a step of its own, with its own rule.


## Step 14, checks after the results (2026-11-18)

- **The freeze holds.** All 24 frozen files and the drafted wordings hash as recorded in `research/results/benchmix1/freeze.json`, so the bench was built, run and scored by the frozen code.
- **benchmix1 is now a development bench:** its failures and replies have been read.
- **The 35 year hits are checked.** The other subject is named in neither the prompt nor the reply of any of them, so no stored fact can have pushed them.
- **The near-question examples** in the result above are observed replies, taken from `graft.jsonl` by `scripts/kinds_bench_recount.py`. The output is in `research/results/benchmix1/recount.txt`, with every after-the-fact count.
- **The near questions are most of the real failures,** descriptively. If they were held to the same-subject bar (0.25), with the year hits not counted, 2 of the 16 company facts and 2 of the 96 others would pass, against 9 and 66 without that bar. Nearly every fact gives its own answer to a question close to its relation. Of everything this bench found, that is the largest real failure. It is not judged here, since the plan reported near questions and did not judge them.
- **Step 13's per-company count** ("40 of 90 companies, 53 of 120 facts") is reproduced by `scripts/diag_name_read_keys.py`, with the output in `research/results/slots/21_name_read/keys.txt`.
- **No other script breaks on the near group.** Only the reproduction check indexed stored answers by every group, and it is fixed.


## Step 16, the pre-check: does a separate direction tell near questions from a relation's own? (written 2026-11-18, before it ran)

**Why:** near questions are the largest real failure of benchmix1. Held to the same-subject bar, 4 of its 112 facts would pass. Step 12 tried to teach a fact's own detector what it must not answer, and lost recall, because one direction cannot both open on the chief executive's questions and stay shut on another job's.
- **The idea:** a second direction per relation, fitted for that contrast alone. It vetoes the fact where it fires, as the shared detectors veto another relation's question.
- **The prior to beat (step 15):** fits on taught wordings barely reach untaught ones.
- **So before any drafter is briefed or the layer is changed,** the direction is tested on step 12's own split, where the first approach failed. Readings and closed-form fits only (`scripts/near_veto_precheck.py`).

**The split, step 12's:**
- the table: step 11's live table and its 10 chief-executive facts;
- taught: 5 other jobs (head of marketing, head of research, chief technology officer, chief financial officer, spokesperson);
- the test: the 3 jobs never taught (general counsel, head of human resources, chief engineer) on the 10 companies, 30 prompts, of which 26 leak;
- the cost: the 80 never-taught chief-executive questions of the editor's last check, of which 59 are answered.

**The veto, per form, for the CEO relation:**
- **Positives:** the 5 taught jobs' questions, about the build's invented companies.
- **Negatives:** half of the relation's own wordings, about the same names.
- **The threshold:** the 99th percentile of the other half of the relation's own wordings, held apart from the fit. δ is half the way to the positives' median.
- **When it fires:** where its best form scores above 0.
- **What is read:** the scores at the end of each test prompt, in the prompt groups and batches step 12 generated them in.
- **Also logged:** on each leaking prompt, whether the fact itself was open at the prompt's end. A veto at the prompt's end can't stop a leak that starts inside the reply.

**Estimated, as fixed now:**
- a leak is stopped where the veto fires and the fact was open at the prompt's end;
- an answer is lost where the veto fires and the fact was open there.

**Kill rule, with step 12's bars:**
- the veto goes on to a build only if at most 13 of the 26 leaks remain (half), and at least 53 of the 59 answers are kept;
- otherwise it stops here, and the near questions stay reported and not judged.


## Result of step 16's pre-check: the veto learns the jobs it was shown and not the others; the kill rule is met

Raw numbers: `research/results/slots/23_near_veto/` (`precheck.json` with every prompt's scores, `precheck.log`), from `scripts/near_veto_precheck.py`. Both forms got a usable veto (180 and 108 positives, 1,476 and 720 of the relation's own readings to fit, and as many held apart).

| Step 12's split | Prompts | Veto fires | The fact open at the prompt's end | Estimated after the veto |
|---|---|---|---|---|
| Untaught jobs, leaking | 26 | 4 | 26 | **22 leaks left** (bar: 13 or fewer) |
| CEO questions, answered | 59 | 9 | | **50 kept** (bar: 53 or more) |
| Taught jobs, leaking (in-sample) | 38 | 36 | | |

- **By the kill rule, the veto stops here.** It misses both bars, and no drafter is briefed or layer changed.
- **It learns the jobs it was taught:** it fires on 36 of their 38 leaking prompts. On the three jobs it never saw it fires on 4 of 26, and it fires on 9 of the 59 answered CEO questions.
- **The prompt's end was the right place to measure:** all 26 leaks were open there, so none started inside the reply.
- **No threshold rescues it.** Swept over every threshold on the stored scores, none stops 13 of the 26 leaks while losing at most 6 of the 59 answers. At −0.5 it stops 17 and loses 14; at 0, 4 and 9. The top tenth of the answered CEO questions read above most leaking ones. Firing on the better of the two forms is already the most permissive rule, so matching forms can't rescue it either.
- **What it says, as I read it:**
  - **It fits step 15's finding:** at layer 20's input, a direction fitted on some wordings reaches those wordings and few others. Here that is for shutting a fact.
  - **It is not the whole story:** step 12's group form, which bent the fact's own detector, did shut the untaught jobs (26 to 0). It failed by losing the chief executive's own answers (59 to 18).
  - **The pattern of both:** no direction tried here separates another job from the chief executive on questions that are new to it.


## Two fixes after step 16 (2026-11-18)

- **The cross check reads guessable answers by their subject.**
  - Another stored fact's year, city or phrase now counts as a cross hit only where that fact's own subject is named in the prompt or the reply (`scoring.cross_hit`). A stored fact acts only where its subject is named, and such answers also turn up by chance.
  - People's and organisations' names count wherever they show, as before. The earlier benches state no answer types, so they score exactly as before: 277 tests pass, and the reproduction check gives 212 of 212 replies identical, with the undo exact.
  - **benchmix1, rescored from its stored replies with the fixed check:** 9 of 16 company facts and 66 of 96 others pass. That is the same as the recount that left out only other subjects' years, so no city or phrase had turned up by chance. benchmix1's verdict stands as it was judged. It is now a development bench, and a fresh bench is needed to count the fix.
- **The drafted brief's name rule.** The drafters were asked to use each list word at most 3 times, which 48 names from 12 words can't do. The drafted brief now asks for the list's words about equally often. The bench writer's brief keeps its limit, which suits 9 subjects.

**The freeze, after these fixes:** `research/results/benchmix1/freeze.json` verifies against commit `db68fc1`, the state in which benchmix1 was built, run and scored. `scoring.py`, `kinds.py` and `brief.py` have changed since, by the fixes above.


## Readings of the stored results, before step 17 (2026-11-22)

No rule: three counts from result files already in the repository, CPU only, nothing generated (`scripts/diag_recall_split.py`, output in `research/results/slots/24_question_site/recall_split.txt`). They cover the 960 recall replies of bench40e and bench40f, with the definitions of "Why the answer is missing".

- **The two recall failures sit in different prompt forms.**
  - Plain sentences: 141 replies have no part of the answer, and in 116 of them the fact's detectors never opened.
  - Chat questions: 67 have no part of the answer, and in 38 of them the detector opened at the prompt's end and the pushed first token was not produced.
  - At the prompt's end the detectors are open on 497 of 640 plain prompts and on 295 of 320 chat prompts.
- **The wordings missed most name the relation and end differently from the taught ones.** "The owner of {s} is" opens on 10 of 10 facts. "Speaking to reporters on Tuesday, the owner of {s}," opens on 0 of 9, "At the anniversary event, {s} founder" on 0 of 7, and "Running {s} as chief executive is" on 0 of 6.
- **The right push size differs from prompt to prompt, in both directions.** Of the 60 lost pushes at the prompt's end, step 9's other sizes give the whole answer in 37: 14 only with a smaller push (1.5 or 1.75), 16 only with a larger one (2.5, 3 or 4), 7 with either. In 31 of the 60 the reply's first word at push 2 is neither the unedited model's nor the answer's; in 21 the whole reply is the unedited model's.
- **Upper bounds, chosen with hindsight for every prompt, not forecasts:** of 80 facts, 56 pass now. With the best stored push size for every prompt 66 would, with every test wording taught 66 (step 15's two-direction fit), with both 74. The two failures are separate pools.

**What was never measured:** a question detector read anywhere but at the last token of the layer that pushes. Step 10's screen of layers 18 to 22 moved the reading and the push together.


## Step 17: where is the question readable? (written 2026-11-22, before it ran)

**Why:** both failures left after step 16 belong to one reading. A fact's detector reads layer 20's MLP input at the last token, and there:
- a fit on taught wordings reaches those wordings and few others (step 15: leaving one wording out opens 1 of 91 far misses);
- a question close to the relation reads like the relation's own (steps 12 and 16; on benchmix1, 0.72 of such questions get the fact's answer).

The stored results above say the missed wordings carry the relation's word and differ in how they end. So the question is whether another place in the model holds the question in a form that depends less on its ending. No fit is changed in this step: the same LDA, the same thresholds, read at other sites. Forward passes and closed-form solves only (`scripts/diag_question_site.py`, read by `scripts/diag_question_site_report.py`).

**What the literature says** (found 2026-11-22; I checked the quoted ones on their own pages, the others were read by a search agent):
- **For another site:**
  - the relation reaches the last token in early layers, and later it is mixed with the answer (Geva et al., arXiv 2304.14767, agent);
  - late heads and last-token MLPs boost the whole category of the answer, not the right one (Chughtai et al., 2402.07321, agent);
  - relation decoders "do not encode distinct relations, but extract recurring, coarse-grained semantic properties" (Christ et al., 2510.26543, checked);
  - neurons specific to a relation sit mostly in middle layers, are read as an average over the tokens, and keep their effect on other templates (Liu et al., 2502.17355, checked);
  - MEMOIR routes edits by activations "averaged across all tokens in the prompt" (2506.07899, checked; its writes use gradients);
  - on a 1B model, the identity of an instruction forms in early layers at the instruction's last token (Bigoulaeva et al., 2602.07930, checked; held-out phrasings not stated).
- **Against:**
  - relation attributes are still being built late at the last token of GPT-2 XL (2408.15091, agent);
  - "no method reaches over 6% faithfulness on the Company CEO relation" for a linear decoder (Hernandez et al., 2308.09124, checked);
  - related relations share neurons (2502.17355, checked);
  - a trained address still generalizes to rewordings at 0.22 to 0.47 (ALOE, 2609.29269, agent).

So the odds are better for wordings than for near questions about jobs.

**The sites:** layer 0 to layer 20's MLP input, in two kinds, 42 sites.
- **Last:** the state at the token itself. Layer 20's is the build's own site.
- **Mean:** the running mean of that layer's MLP input over the text so far, with the first token left out, as the calibration leaves it out.
- **Layers above 20 are left out:** a reading above the layer that pushes can't open that push in the same pass.
- **Each site's calibration:** the mean and covariance of its reading on the calibration's 240 fitting paragraphs, and its own constant direction, as in graft.md, section 1.

**The rules a detector is fitted by:**
- **R1, the build's rule:** negatives are the prompt negatives, the readings inside the taught answers and the readings inside the unedited model's replies. The threshold is above the prompt negatives' 99th percentile and above every reading inside the taught answers. Kind "last" only.
- **R2, the prompt's end only:** negatives are the prompt negatives alone, with the threshold at their 99th percentile. It stands for a fact that may open only at the prompt's last token. A running mean changes too little from one token to the next to say where an answer is due, so the kind "mean" is read by R2 alone. Kind "last" is read by both.

That gives 63 readings: 21 layers × (last by R1, last by R2, mean by R2).

**Part 1, wordings** (bench40e and bench40f, the two saved API tables, 80 facts, 56 passing):
- **Fits at every reading:**
  - as built: the detector from the taught wordings alone;
  - leave one wording out: step 15's refit, with the other 11 test wordings taught;
  - in-sample: all 12 taught, an upper bound, reported only.
- **Scored as in step 15,** at the end of the 12 test recall prompts and the 8 test questions about other relations of the company, with step 15's estimate of facts passing: a prompt that was open at the build's site and is shut now loses its answer, a newly opened one gives the whole answer at the bench's measured rate, and a fact with a newly opened question about another relation fails.
- **The readings are taken in batches of 64 with right padding,** not in the bench's own generation batches, and each prompt is read once for all sites. Scores can therefore differ from the stored ones in their last digits.
- **The gate, before anything else is read:** at layer 20, last, R1, as built:
  - every detector's direction has a cosine of at least 0.99 with the saved one;
  - at least 97% of the 960 recall prompts are open or shut as stored;
  - the estimate is 56 ± 2.

**Part 2, near questions:**
- **Step 12's split** (step 11's live table, its 10 chief-executive facts, rebuilt at every reading with the table's own options): the 26 leaking prompts about three jobs never taught, and the 59 answered chief-executive questions. Two forms:
  - the detector alone: a leak is stopped where the detector is shut at the prompt's end, and an answer is lost where it is shut;
  - with step 16's veto, fitted at the same site on the five taught jobs against half of the relation's own wordings: a leak is stopped, or an answer lost, also where the veto fires.
- **benchmix1's near questions** (the saved table's 112 facts, rebuilt at every reading, the detector alone). From the stored replies:
  - a near question leaks if it gives the fact's answer, by the bench's own check;
  - a leak is stopped where the build's own reading is open at the prompt's end and the reading's detector is shut;
  - a leak that began inside the reply counts as left;
  - an answer is lost where a recall prompt gave the whole answer, was open at the prompt's end at the build's own reading and is shut now.
- **Gate for both tables,** which were built with the fast build: at layer 20, last, R1, every detector's direction has a cosine of at least 0.99 with the saved one.

**Fixed now:**
- **A reading passes on wordings** if, as built or with one wording left out, it meets all of:
  - at least 62 estimated facts of 80, which is 6 more than now and step 15's bar;
  - at most 2 newly opened questions about another relation over the 80 facts;
  - at least 2 more than now on each bench (32 and 24);
  - at least 59 at a neighbouring layer of the same kind and rule, so that one reading among 63 is not taken for a finding.
- **A reading passes on near questions** if it meets both:
  - on step 12's split, in either form: at most 13 of the 26 leaks left and at least 53 of the 59 answers kept, step 16's bars;
  - on benchmix1, with the detector alone: at most half of the leaks left and at least 90% of the answers kept.

  A reading that meets only one of the two is reported as partial and earns nothing by itself.
- **What a pass earns:** these are development benches, read many times. A pass earns a build that reads the question at that site in the inserted layer, frozen, and then a fresh blind bench with near questions judged. It earns no claim.
- **What a pass would cost the grafted model:** a neuron reads its own layer and token. A site other than layer 20's last token can be read by the inserted layer at once; in the grafted model the switch would have to carry it.
- **Kill rule:** if no reading passes on wordings and none on near questions, the question detector stays where it is and detector work on the 0.8B stops. The fallback is the 4B model, whose reworded prompts reach 0.91 on CounterFact with same-subject leaks of 0.095, and whose near questions are untested. What comes next is then my choice.
- **Reported, not in the rule:** the in-sample fit at every reading, each relation, each bench, the taught jobs, and benchmix1 by answer type.


## Result of step 17: one reading passes on wordings, mostly through what the fit is set against; none passes on near questions, where a veto read as a mean carries over to jobs it was not taught

Raw numbers: `research/results/slots/24_question_site/` (`bench40e.json`, `bench40f.json`, `jobs.json`, `near.json`, `veto_cost.json`, their logs, `report.txt`), from `scripts/diag_question_site.py` and `scripts/diag_question_site_report.py`.
- **The runs:** a smoke run on three facts of bench40e first, to check the code and time it; the three measuring modes were not changed after it. Then bench40e and bench40f (10 minutes each), step 12's split (2 minutes) and benchmix1 (17 minutes).
- **Added after the smoke run, to the report only:** the counts of newly opened prompts, and the readings by relation.
- **Added after the results:** the veto's cost on other benches (below), labelled as such.

**The gates hold:**
- **Wordings:** at layer 20, last, R1, the smallest cosine with a saved detector is 0.9906; 956 of the 960 recall prompts are open or shut as stored; the estimate is 56 facts, the number that pass.
- **Step 12's split:** smallest cosine 0.9985. At the build's own reading the detector is open at the end of all 26 leaking prompts and of 57 of the 59 answered ones.
- **benchmix1:** smallest cosine 0.9954. The stored replies give 321 leaking near questions of 448 (0.72, as reported in step 14), 312 of them with the build's reading open at the prompt's end, and 1,089 answered recall prompts.

**Part 1, wordings.** Estimated facts passing of 80 (bench40e + bench40f), then the questions about another relation newly opened, then the far misses opened of 91:

| Reading | As built | One wording left out | In-sample |
|---|---|---|---|
| Layer 20, last, R1 (the build's) | 56 (32 + 24); 0; 0 | 55; 1; 1 | 61; 1; 48 |
| Layer 20, last, R2 | 62 (31 + 31); 4; 86 | 63; 3; 85 | 64; 3; 91 |
| **Layer 16, last, R2** | **65 (34 + 31); 0; 76** | 63; 2; 76 | 65; 2; 91 |
| Layers 15, 17, 18, 19, last, R2 | 60, 61, 59, 60; 3 to 5; 66 to 78 | 59 to 61 | 63 to 64 |
| Layer 8, last, R2 | 64 (34 + 30); 2; 82 | 60; 8; 82 | 62; 7; 91 |
| Layers 0 to 19, last, R1 | 0 to 46 | 0 to 47 | 0 to 61 |
| Layer 3, mean | 64 (35 + 29); 5; 91 | 62; 6; 91 | 64; 4; 91 |
| Layer 15, mean | 63 (34 + 29); 4; 91 | 63; 4; 91 | 64; 3; 91 |
| The other mean layers | 54 to 62; 6 to 17; 86 to 91 | 53 to 61 | 55 to 61 |

- **By the rule, one reading passes on wordings: layer 16, last, R2, as built.** 65 estimated facts against 56, no question about another relation newly opened, 34 and 31 on the two benches, and its neighbours at 60 and 61.
- **The others that reach 62 fail a clause.** Layer 20, R2, opens 4 questions about another relation and stays at 31 on bench40e. Layer 8, R2, has neighbours at 50 and 57. The mean readings at layers 3, 8 and 15 open 4 to 6.
- **Most of the gain is the rule, not the place.** At layer 20 itself R2 opens 86 of the 91 far misses with the detector as built, where R1 opens none.
- **What differs between R1 and R2 is what the fit is set against, not the threshold's floor.** In the saved tables the largest reading inside the taught answers is under the prompt negatives' 99th percentile for every detector (0 of 80, 0 of 80 and 0 of 224 forms are set by it), so the floor never binds. R1's fit has to separate the question from three groups that weigh the same: the prompt negatives, the readings inside the taught answers and the readings inside the model's replies. R2's fit separates it from the prompt negatives alone.
- **That corrects my reading of step 15.** Leaving one wording out opens 1 of 91 far misses under R1, and I read that as "the limit is coverage of forms". Under R2, 86 of 91 open with no test wording taught. The limit was the fit's other two groups. Step 15's numbers stand: at layer 20, R1, this run gives 55 with one wording left out and 61 in-sample, as step 15 did.
- **Teaching the neighbouring wordings adds nothing at any reading:** leaving one wording out never beats the fit as built by more than 4 facts, and at the R2 and mean readings it is equal or lower almost everywhere.
- **The place decides the leaks.** At layer 20, R2 newly opens "Most of the customers of {s} are" for three head-office facts and one owner fact. That is the leak that kept the looser start at 31 on blind bench40e, where the default layer also passed 31, and this estimate gives layer 20, R2, the same 31 there. At layer 16 none of these opens.
- **By relation at layer 16, R2:** owner 10 → 16 of 20, founder 14 → 16, chief executive 13 → 14, head office 19 → 19.
- **The mean opens every far miss and tells neighbouring relations apart worse.** At layer 9 an owner fact opens on "{s} is managed by its CEO," and a head-office fact on "In what year did {s} start trading?".
- **R1 moved to any other layer is worse than at layer 20.**
- **Keeping the stored detector beside the new reading changes no estimate:** 65 at layer 16 and 62 at layer 20 for R2, and 61 and 63 for the mean at layers 9 and 15, with or without it.

**How far to trust the 65:**
- It is one pass among 126 tries (63 readings, two fits).
- Its own neighbours open 3 to 5 questions about another relation, so the 0 at layer 16 may be the luckiest draw of a family that sits at 59 to 62.
- The estimate assumes a newly opened prompt gets its answer at the benches' measured rate of 0.90 to 0.92. The prompts opened here are the unusual ones, where a push may lose more often.
- **On a bench that chose no reading, the gain is smaller and the zero doesn't hold** (after the fact; benchmix1's 112 facts, by recall and same-subject questions only, since its stored pass flags carry the old cross check). Against 80 facts at the build's reading, R2 gives 84 at layers 16 and 17, 82 at layer 20, 78 to 81 at layers 14, 15, 18 and 19, and 75 at layer 8. Layer 16 is the best of its family again, but it newly opens 9 same-subject questions there. The mean gives 61 to 71.
- The estimate doesn't model the layer's competition between a company's facts, or its veto by the shared detectors.
- No shipped option fits a detector this way. `end_quantile` keeps R1's direction, and since the floor never binds it changes nothing. The looser start keeps the readings inside the taught answers as a group. So the estimate can't be checked by generation without a build.

**Part 2, near questions.**

Step 12's split, leaks left of 26 and answers kept of 59 (bars: at most 13, at least 53):

| Reading | The detector alone | With step 16's veto at the same site |
|---|---|---|
| Layer 20, last, R1 (the build's) | 26; 59 | 22; 51 (step 16: 22; 50) |
| Layers 13, 14, 15, last, R2 | 26; 59 | 13, 10, 10; 54 |
| Layer 9, last, R2 | 26; 59 | 8; 54 |
| **Layer 9, mean** | 26; 59 | **0; 59** |
| Layers 10 and 11, mean | 26; 59 | 0; 53 and 58 |
| Layer 3, mean | 26; 59 | 7; 55 |
| The other mean layers from 1 to 16 | 26; 59 | 0 to 7; 20 to 52 |

- **The detector alone separates another job from the chief executive at no reading.** Wherever it keeps the answers, 24 to 26 of the 26 leaks stay.
- **With the veto, eight readings meet step 12's bars:** last, R2, at layers 9, 13, 14 and 15, and the mean at layers 3, 9, 10 and 11.
- **Read as a mean, the veto carries over to the three jobs it was not taught.** At layer 9 it stops all 26 of their leaks and loses none of the 59 answered chief-executive questions. At the last token of layer 20 the same fit stops 4 and loses 8.
- **How much that is:** the 26 prompts are three wordings asked about ten companies, and a mean over the tokens moves little with the company's name. So this is close to 3 of 3 wordings, not 26 independent cases. The 59 kept answers were written by the editor whose earlier questions are among the relation's own wordings the veto was fitted against, so the cost on blind wordings, below, is the fairer figure.
- **It shuts the untaught jobs across the mean layers 1 to 16** (0 to 7 leaks left). What varies with the layer is how many of the relation's own questions it also shuts.

benchmix1, the detector alone: leaks left of 321 and answers kept of 1,089 (bars: at most 160, at least 981):

| Reading | Leaks left; answers kept | Same-subject questions newly open | Near questions that did not leak, newly open |
|---|---|---|---|
| Layer 20, last, R1 (the build's) | 321; 1,089 | 0 | 0 |
| Layer 10, last, R1 (the nearest to the bars) | 249; 981 | 17 | 17 |
| Layer 16, last, R2 | 321; 1,089 | 9 | 62 |
| Layer 20, last, R2 | 321; 1,089 | 15 | 58 |
| Mean, layers 0 to 20 | 313 to 320; 1,087 to 1,089 | 28 to 78 | 62 to 76 |

- **No reading meets the bars.** With the detector alone, the leaks stay wherever the answers are kept, as on step 12's split.
- **R2 and the mean make this leak worse.** Of the 127 near questions that did not leak, R2 at layers 5 to 20 opens 45 to 66 and the mean 62 to 76. The mean also opens 28 to 78 other questions about the subject.
- **benchmix1 can't test the veto:** its relations have no taught near questions to fit one on.

**By the rule:**
- **On wordings, one reading passes:** layer 16, last, R2, as built.
- **On near questions, no reading passes both tests.** Eight meet step 12's bars with the veto and none meets benchmix1's, so they are partial and earn nothing by themselves.
- **The kill rule is not met.** The pass earns a build that fits a detector for the prompt's last token against the prompt negatives alone, read at layer 16 in the inserted layer, frozen, and then a fresh blind bench with near questions judged.

**After the fact: what the veto would cost** (`veto_cost` mode, no rule). The veto fires where chief-executive words are missing, so it might shut the relation's own questions when they are worded differently. Read at the end of the 360 test recall prompts of the chief-executive facts of bench40e, bench40f and bench40g (blind wordings; 248 answered in the stored runs):

| The veto, read at | Fires on, of 360 | Of the 248 answered |
|---|---|---|
| Layer 9, mean | 25 | 21 |
| Layer 10, mean | 13 | 11 |
| Layer 11, mean | 18 | 14 |
| Layer 14, mean | 10 | 9 |
| Layer 20, last (step 16's site) | 20 | 14 |

- **It is not only an absence test:** at layers 9 to 14 the mean veto stays shut on 92% to 97% of blind chief-executive wordings.
- **The cost is 4% to 9% of the relation's own answers on these benches,** and 0 to 12% on step 12's 59 (layers 9 to 14). Step 12's group form cost 70% for the same 26 leaks. Step 16's veto at the last token of layer 20 costs about as much as the mean veto here (14 of 248) and stopped 4 of the 26.
- **The wordings it shuts:** "Ask anyone at {s} who the CEO is and they will say", "By resolution of the board, the duties of chief executive of {s} are carried out by", "When reporters want a comment from {s}, they usually call its chief executive,".

**What it says:**
- **Where the question is read matters less than what its detector is fitted against, for wordings.** A detector for the prompt's last token, fitted against the prompt negatives alone, opens nearly every missed wording at the build's own site. Reading it four layers lower keeps the general questions shut.
- **Which relation is asked is readable from the words of the question, averaged, where the last token doesn't hold it.** This is the one reading tried so far at which a contrast taught on some jobs shuts jobs it never saw. It is one relation, one split and three untaught jobs (26 prompts), and the layer was picked after the fact from a band of layers that all shut the leaks.
- **A mean is a poor detector by itself and a good contrast.** Alone it opens neighbouring relations and near questions. Against taught near questions it tells them from the relation's own.
- **Near questions are not solved by any detector alone, at any site.** They need the contrast to be taught, relation by relation.

**What comes next is my choice.** The two candidates this step leaves, and how they depend on each other:
- **The build the pass earns:** the end-of-prompt detector at layer 16, then a blind bench with near questions judged. Expected size, as I read the two tables: 1 to 4 facts in 40. Alone it would fail that bench on near questions: R2 opens about 60 more of them on benchmix1, where 321 of 448 leak already.
- **A test of the mean veto beyond one relation:** near questions drafted for benchmix1's relations by a writer who has not seen the bench's own, a veto per relation read as a mean at one layer fixed before the run, judged on benchmix1's near questions and on its recall. The bars would be step 17's.
- **So the veto's test comes first.** If it holds, one build carries both parts to one blind bench. If it fails, the end-of-prompt detector goes to a blind bench by itself, with near questions reported and not judged, as now.


## Step 18: the veto read as a mean, beyond one relation (written 2026-11-22, before the near questions were drafted or anything ran)

**Why:** step 17 left one reading at which a taught contrast shut near questions it was not taught: step 16's veto, read as a running mean over the tokens. That was one relation, one split, three untaught jobs asked about ten companies, and the layer was picked afterwards. The end-of-prompt detector that step 17's pass earned opens about 60 more near questions on benchmix1, so by itself it would fail a bench that judges them. This step asks whether the mean veto holds on 20 relations in nine kinds of subject, with the layer fixed first. Forward passes and closed-form fits only (`scripts/diag_mean_veto.py`).

**My choice of writer:** the near questions are drafted by Claude Opus, run as a separate agent that gets the brief below and nothing else.

**The veto, as in steps 16 and 17:**
- **One per relation and form.** Positives: the taught near questions, asked about invented names of the relation's kind of subject. Negatives: half of the relation's own wordings about the same names. Threshold: the 99th percentile of the other half, held apart from the fit. δ is half the way to the positives' median. It fires where its better form scores above 0.
- **The names:** the kind's first 24 invented names, and the first 6 of them in lower and in upper case.
- **The reading, fixed now: the mean at layer 11.** Of the mean layers that left no leak on step 12's split, it lost the fewest of the relation's own answers over that split and the three benches' blind wordings together (1 of 59, and 14 of 248). Every other site is fitted and reported, not judged.

**Three sets of taught near questions, so three arms:**
- **A, judged: drafted blind.** For each of benchmix1's 20 relations, 8 plain and 8 chat near questions written by the writer above. Its brief gives each relation's kind, description and answer type, the rules, and one example for an unrelated relation. It gives none of the bench's near questions and none of the examples of the bench writer's brief. The draft is checked by script, hashed and frozen before any reading is taken.
- **B, reported: the bench's own other half.** The bench writer wrote 8 near questions per relation. Each fact got 4 of them as test prompts and the other 4 as training prompts, which nothing has used. Arm B fits a veto per fact on its 4 training ones. Its test is clean within a fact, and thin: 4 taught questions.
- **C, reported, a control: no near questions.** The positives are the first 8 plain and 8 chat generic questions of the kind. If C does as well as A, the veto only tells the relation's own words from anything else, and needs no drafted near questions.

**The test:** benchmix1's stored run and step 17's readings of it (`24_question_site/near.json`): 448 near questions, 321 leaking, 312 of those with the fact's detector open at the prompt's end; 1,089 answered recall prompts.
- **A leak is stopped** where the fact's detector was open at the prompt's end and the veto fires there. A leak that began inside the reply counts as left.
- **An answer is lost** where a recall prompt gave the whole answer, the detector was open at the prompt's end and the veto fires.
- **Judged on untaught near questions:** a test near question whose wording equals a drafted one of its relation, by the duplicate rule, is left out of arm A's count. The count over all 448 is reported beside it.
- **Overlap in what is asked can't be ruled out by script.** A writer asked for questions near a chief executive will name other jobs, as the bench's writer did. So arm A's result is also reported split by whether a test question shares a distinctive word with any drafted one of its relation. Arm B is the within-pool comparison.

**Fixed now:**
- **The veto holds beyond one relation** if arm A, at the mean of layer 11, on the untaught near questions, leaves at most half of the leaks and keeps at least 90% of the answers. These are step 17's bars.
- **What a pass earns:** one build with both parts, frozen: the end-of-prompt detector of step 17 and a veto per relation read as a mean at layer 11, with near questions drafted for each relation. Then one fresh blind bench with near questions judged.
- **Kill rule:** if arm A misses either bar, the mean veto is put aside. The end-of-prompt detector then goes to a blind bench by itself, with near questions reported and not judged, as now.
- **Reported, not in the rule:**
  - arms B and C, and every site;
  - each relation, kind and answer type;
  - the veto on same-subject questions;
  - the facts that would pass with near questions held to the same-subject bar (4 of 112 now), counted with the fixed cross check.
- **benchmix1 is a development bench,** so a pass is a development result.

**The draft, frozen** (2026-11-22, before any reading with it):
- **The writer** was Claude Opus, run as a separate agent with the brief alone (`research/results/slots/25_mean_veto/brief_writer.txt`, printed by the script) and told to use no tool and read no file. Its record shows one tool call, which I take to be the call that hands its answer back; I did not open its transcript.
- **The file:** `research/results/slots/25_mean_veto/drafted_near.json`, SHA-256 beginning `863339c54c642ed7`, 20 relations with 8 plain and 8 chat near questions each. The script's check finds no problem: every string has `{s}` once, no plain one ends in a space, none repeats within its relation, and none equals a wording of a relation of the same kind.
- **What it shares with the bench** (`check.txt`): of the bench's 160 near questions, 8 per relation, 10 equal a drafted one by the duplicate rule, and are left out of arm A's judged count wherever they are a fact's test prompts. 123 share a distinctive word with a drafted one of their relation. So most of what the bench asks is close to something the writer also thought of, as expected, and the questions that share nothing are the cleanest part of the test.
- **A smoke run first,** on two relations (chief executive and town mayor) with a stand-in file of generic questions in arm A's place, to check the code and time it. It showed arms B and C on those two relations. Nothing was changed after it.


## Result of step 18: the mean veto stops most near-question leaks on 20 relations and shuts too many of the relations' own answers; the kill rule is met

Raw numbers: `research/results/slots/25_mean_veto/` (`veto.json` with every veto's score at every prompt and site, `veto.log`, `report.txt`, the brief, the draft and its check), from `scripts/diag_mean_veto.py`. The run took 9 minutes for the 20 relations and 112 facts.

**The rule's reading: arm A, the mean at layer 11, on the near questions that equal no drafted one.**

| | Result | Bar |
|---|---|---|
| Leaks left | 87 of 303 | at most 151: met |
| The relations' own answers kept | 788 of 1,089 | at least 981: **missed** |

- **By the kill rule, the mean veto is put aside.** It stops 71% of the leaks and shuts 28% of the answers.
- **31 test near questions equal a drafted one** and are left out; 18 of them leak, and the veto leaves 2.

**The three arms at the judged reading,** over all 448 near questions (321 leaking) and the 1,089 answers:

| Taught near questions | Leaks left | Answers kept | Facts passing every check but near questions | Facts passing with near questions held to the same-subject bar |
|---|---|---|---|---|
| None (no veto) | 321 | 1,089 | 75 of 112 | 4 |
| A, drafted blind, 16 per relation | 89 | 788 | 37 | 27 |
| B, the bench's own other half, 4 per fact | 70 | 611 | 19 | 14 |
| C, the kind's generic questions (control) | 184 | 879 | 45 | 16 |

- **Near questions are shut, also where nothing like them was taught.** Of arm A's untaught leaks, 20 of 72 are left among the questions that share no distinctive word with a drafted one, and 67 of 231 among those that share one.
- **The price is the relation's own questions when they are worded as the bench words them.** The veto's threshold sits at the 99th percentile of the relation's own drafted wordings, and it fires on 28% of the bench's answered recall prompts.
- **Part of the veto is not about near questions at all.** With generic questions in their place (arm C) it still stops 137 of the 321 leaks and shuts 210 answers. That part only tells the relation's taught wordings from everything else. The drafted near questions add 95 more stopped leaks, and cost 91 more answers.
- **Four taught questions per fact are too few** (arm B): more leaks stopped, and 44% of the answers shut.
- **No site does better.** Over the mean layers arm A keeps at most 835 answers (layer 10, with 87 leaks left). At the last token the veto keeps at least 981 answers only at layers 0 and 1, where it leaves 306 and 265 leaks.

**By relation, arm A falls into three groups** (leaks left of leaks; answers kept of answers):
- **Both sides hold:** the town's founding year 4 of 23 and 64 of 70, head office 3 of 11 and 36 of 40, the concept's field 5 of 17 and 46 of 46.
- **The leaks stop and the answers go:** chief executive 0 of 10 and 12 of 36, founder 0 of 9 and 13 of 38, the product's maker 0 of 20 and 23 of 64, the institution's head 5 of 16 and 20 of 58, the work's year 8 of 24 and 23 of 61.
  - **For four of them the drafted near questions are what costs the answers.** Arm C, which teaches no near question, keeps 35 of the founder's 38, 41 of the maker's 64, 49 of the work year's 61 and 32 of the head's 58. I take it, without having measured it, that their drafted near questions use the relation's own words ("the founder's father", "the sequel came out in").
  - **For the chief executive they are not.** Arm C keeps 18 of 36 there, and stops 9 of the 10 leaks, with no near question taught. So that veto shuts the bench's chief-executive wordings whatever it was taught.
  - **Unexplained:** the same relation's veto cost 4% to 9% on step 17's three benches and half here. The two runs differ in the relation's own wordings (step 11's table carries the editor's taught ones) and in the test wordings.
- **The answers stay and so do the leaks:** birthplace 19 of 21 and 67 of 67, employer 6 of 9 and 47 of 47, the event's city 9 of 16 and 53 of 53. A veto exists for each of them at this reading; it fires on few of their near questions (5 of birthplace's 24) and on none of their recall prompts.

**After the fact: would another threshold have met the bars?** With hindsight, from the stored scores, upper bounds only:
- **One cut for all relations: no.** A cut that keeps 981 answers leaves 209 to 228 of the 303 leaks at the mean layers 9 to 12.
- **A cut per relation: yes, on paper.** With each relation's cut set to keep 90% of its own answers, arm A leaves 114 of 303 leaks at layer 10 and 125 at layer 11, with 1,019 and 1,009 answers kept, and 14 and 11 of the 20 relations left with at most half their leaks. At the last token of layer 20 the same bound leaves 214.
- **In facts, at layer 11:** 68 would pass every check but near questions, against 75 with no veto and 37 with the veto as fitted. With near questions held to the same-subject bar 43 would pass, against 4 and 27.
- **How little that proves:** the bound sets 20 thresholds on the test itself and clears the bars narrowly, and 6 to 9 relations miss even so.
- **What that says:** read as a mean, most relations' near questions and own questions can be told apart. What fails is setting the threshold from the relation's own drafted wordings: that threshold does not carry over to wordings from another writer.

**What it says:**
- **Step 17's result on one relation did not carry over.** There the veto stopped every leak and cost 4% to 9%; here the same relation's veto shuts half of its answers. Its 59 answers in step 17 came from the writer of wordings the veto was fitted against, which is one difference; the rest is not explained.
- **A taught contrast read from the question's words cuts the near-question leak on every kind of subject,** and no detector alone did that at any reading.
- **The same weakness shows on the other side.** A fit knows the wordings it was fitted against. Step 16 saw it for shutting a fact on jobs it was not shown, and here the veto shuts the relation's own questions when another writer words them.

**What comes next is my choice.**
- **By the rule,** the end-of-prompt detector of step 17 goes to a blind bench by itself, with near questions reported and not judged.
- **One revision the after-the-fact bound points to,** which would need its own rule written first: the veto's threshold set per relation from own questions by another writer than the fit's, at a stated rate of false fires, and tested on wordings it has not seen. On benchmix1 each fact's 12 training wordings are such a sample, unused so far. The bench shuffles a relation's wordings fact by fact, so one fact's training wording is another's test wording: the threshold would have to be set per fact, or with the fact's test wordings left out.


## Step 18, the one revision: the veto's threshold set from the fact's own questions by another writer (written 2026-11-22, before it ran)

**Why:** step 18's veto stopped the near-question leaks and missed the answers bar, and the after-the-fact bound says the threshold is what fails. It was set on the relation's own drafted wordings, and it does not carry over to the bench writer's wordings. One revision is allowed here, with the same bars. Only the threshold changes: the vetoes, the drafted near questions, the reading and the test are step 18's.

**The threshold, per fact:**
- **The sample:** the fact's 12 training wordings in the bench (8 plain, 4 chat). The bench's blind writer wrote them, the table was built without them, and no step has used them. They share no wording with the fact's 12 test wordings.
- **The rule:** a fact's veto fires on a question only where it scores above every one of the fact's 12 training wordings.
- **The stated rate of false fires:** if a new wording of the relation's own question is like the 12, it scores above all of them once in 13 times, so the veto should shut about 8% of the fact's own questions. The share it shuts of the test recall prompts is reported beside that.
- **Per fact, not per relation:** the bench shuffles a relation's wordings fact by fact, so another fact's training wording can be this fact's test wording. A fact's own 12 can't be.

**The test, as in step 18:** arm A (the near questions Opus drafted, frozen), the mean at layer 11, benchmix1's stored run. A leak is stopped where the fact's detector was open at the prompt's end and the veto fires there; an answer is lost in the same way. The near questions that equal a drafted one are left out.

**The gate:** the run is step 18's with the training wordings read after everything else. Its scores at the test prompts must equal step 18's stored ones.

**Fixed now:**
- **The veto holds** if it leaves at most half of the leaks on the untaught near questions and keeps at least 90% of the answers: step 18's bars.
- **What a pass earns:** one build with both parts, frozen: step 17's end-of-prompt detector, and a veto per relation read as a mean at layer 11 whose threshold is set for each fact from a dozen of its own questions, written by the editor when it checks the fact. Then one fresh blind bench with near questions judged.
- **What a pass would not show:** here the 12 wordings and the test wordings come from one writer and one shuffle. In use the sample would be the editor's, and the questions a person's.
- **Kill rule:** if it misses either bar, the mean veto is put aside for good on this model. Near questions stay reported and not judged, and the end-of-prompt detector goes to a blind bench by itself. The place to ask about near questions again is then the 4B model.
- **Reported, not in the rule:**
  - the second-largest training score as the threshold (two in 13);
  - the threshold never set under the fitted one;
  - arms B and C, the mean layers 9 to 12 and the last token of layer 20;
  - each relation;
  - the facts passing, with and without near questions judged.


## Result of step 18's revision: with the threshold set from each fact's own training wordings, the veto meets both bars, narrowly and unevenly

Raw numbers: `research/results/slots/25_mean_veto/` (`veto_train.json`, `veto_train.log`, `report_revision.txt`), from `scripts/diag_mean_veto.py` (`run --train`, then `report`). The run took 9 minutes.
- **The gate holds:** the scores at the test prompts equal step 18's stored ones, with a largest difference of 0.
- **A smoke run first,** on the chief executive's four facts with the stand-in file of generic questions, to check the code. Nothing was changed after it but the report's last table, which is after the fact and labelled so.

**The rule's reading: arm A, the mean at layer 11, the untaught near questions.**

| | Step 18, the threshold as fitted | The revision | Bar |
|---|---|---|---|
| Leaks left | 87 of 303 | **143 of 303** | at most 151: met |
| The relations' own answers kept | 788 of 1,089 | **1,015 of 1,089** | at least 981: met |
| Facts passing every check but near questions (75 with no veto) | 37 | 66 | |
| Facts passing with near questions held to the same-subject bar (4 with no veto) | 27 | 34 | |

- **By the rule, the veto holds.** It stops 53% of the leaks and shuts 7% of the answers.
- **The margin is 8 leaks.** The mean at layers 9, 10 and 12 meets both bars too: 149, 137 and 144 leaks left, with 1,020, 1,026 and 1,028 answers kept. At the last token of layer 20 the same threshold leaves 234.
- **The pooled pass rests on two relations** (after the fact). With the concept's proposer left out, 140 of 279 leaks are left, and with the medicine's maker left out 143 of 281: either way the leaks bar is missed.
- **The answers bar could hardly have been missed.** The threshold is the largest of 12 scores from the writer and the shuffle of the test wordings, so about 92% kept follows from how it is set. The leaks bar was the test.
- **The stated rate holds:** the veto fires on 117 of the 1,344 test recall prompts, 8.7% against the stated 7.7%.
- **With this threshold the drafted near questions are what stops the leaks.** The control with generic questions (arm C) stops 62 of 321. Arm A stops 160 of 303. The bench's own four near questions per fact (arm B) stop 141 of 321.
- **The other thresholds, reported:**
  - above all but one training wording: 108 leaks left and 949 answers kept, under the answers bar;
  - never under the fitted threshold: 158 left and 1,031 kept, over the leaks bar.

**By relation the result is uneven** (leaks left of the untaught leaks):
- **At most half left, 8 of 20 relations:** the medicine's maker 0 of 22, the concept's field 1 of 17, the medicine's condition 2 of 9, the concept's proposer 3 of 24, the town's founding year 4 of 23, the event's city 4 of 16, the town's mayor 5 of 11, the institution's head 7 of 16.
- **More than half left, 12 relations,** among them all four about companies: chief executive 6 of 6, founder 6 of 9, head office 6 of 11, owner 3 of 4. Also the work's year 19 of 24, the event's year 18 of 23, the institution's city 13 of 15, birthplace 12 of 19 and the product's year 10 of 12.
- **Every relation keeps at least 85% of its answers.**

**After the fact: the veto beside step 17's end-of-prompt detector.** Estimated as in step 17 on benchmix1's 112 facts: a newly opened recall prompt answers at the bench's rate of 0.90, a newly opened question about another relation fails its fact, a newly opened near question counts as a leak, and the veto shuts what it fires on.

| | Facts passing every check but near questions | With near questions held to the same-subject bar |
|---|---|---|
| The detector as built, no veto (now) | 75 | 4 |
| The detector as built, with the veto | 66 | 34 |
| With the end-of-prompt detector, no veto | 79 | 2 |
| With the end-of-prompt detector and the veto | 71 | 31 |

- **The two parts together cost 4 facts by today's rule and gain 27 by a rule that judges near questions.** The veto's cost on the facts' own answers is larger than what the end-of-prompt detector adds on this bench.

**What it shows, and what it does not:**
- **A taught contrast, read as a mean over the question's words and held to a threshold set on the fact's own questions, halves the near-question leak** on a bench of 20 relations in nine kinds of subject, at a known cost.
- **The threshold was the fault in step 18, as its after-the-fact bound said.** Set on a dozen own questions by the writer of the test wordings, it shuts those wordings at the rate stated.
- **Not shown: a threshold from another writer's sample, and the nearest evidence points the other way.** Here the 12 wordings and the test wordings come from one writer and one shuffle. In use the sample would be the editor's and the questions a person's. The one change of writer measured so far is step 18's own, with the threshold from the drafters' wordings and the test from the bench writer's, and there the veto shut 28% of the answers.
- **Not shown: the company relations.** All four keep more than half of their leaks.
- **Not shown: a blind bench.** benchmix1 has now been read by steps 14, 17, 18 and this revision, and this is the one revision the kill allowed.

**What the pass earns, as fixed before the run:** one build with both parts, frozen: step 17's end-of-prompt detector, and a veto per relation read as a mean at layer 11, its threshold set for each fact from a dozen of the fact's own questions written by the editor when it checks the fact. Then one fresh blind bench with near questions judged.

**Two things to settle before that build, both my choice:**
- **Whether near questions become a judged check.** By today's pass rule the build is estimated to lose facts on this bench (75 now, 71 with both parts). It gains only by a rule that judges near questions (4 now, 31 with both parts), and that is a change to the pass rule.
- **Which bench would judge it.** All four company relations keep more than half of their leaks, so a company bench would show the veto's cost and none of its gain. It would have to be a bench of many kinds, like benchmix1.

A cheap check would come before any build: the threshold set from a sample by another writer than the test wordings', on benchmix1, with a kill rule. Nothing is started.


## Step 18, a check before any build: the threshold from another writer's sample (written 2026-11-22, before the sample was drafted or anything ran)

**Why:** the revision met its bars with each fact's threshold set from the bench writer's own training wordings, so the sample and the test wordings had one writer. A build would not have that: the editor writes the sample and a person asks the questions. The one change of writer measured so far, step 18 as first run, shut 28% of the answers. This check makes that change on purpose, before anything is built. I asked for it, with Claude Opus as the writer again.

**What changes from the revision:** only the sample a fact's threshold is set from.
- **The sample:** for each of benchmix1's 20 relations, 8 plain and 4 chat ways of asking for the relation, written by a separate agent that gets the relations' descriptions and the rules and nothing else: no bench, no wording the table was taught, no near question.
- **Per fact:** the 12 are asked about the fact's own subject, and the fact's veto fires only above every score it gives to them. The stated rate of false fires is one in 13, if a person's wording were like the writer's. Whether it is, is what this check measures.
- **A sample wording that equals one of the bench's wordings of its relation,** by the duplicate rule, is left out of the sample, and counted.
- **Everything else is step 18's and the revision's:** arm A's vetoes from the frozen near questions, the mean at layer 11, benchmix1's stored run, the untaught near questions.

**The gate:** the scores at the test prompts must equal step 18's stored ones.

**Fixed now:**
- **The veto survives a change of writer** if it leaves at most half of the leaks and keeps at least 90% of the answers: the same two bars.
- **What a pass earns:** the build of the revision's result (step 17's end-of-prompt detector, and the mean veto with a per-fact threshold from a dozen own questions by the editor), to be judged on a blind bench of many kinds. Whether near questions become a judged check stays my choice, and without it the veto only costs.
- **Kill rule:** if it misses either bar, the mean veto stops here on this model. Step 17's end-of-prompt detector goes to a blind bench by itself, with near questions reported and not judged.
- **Reported, not in the rule:** the share of test recall prompts the veto fires on beside the stated one in 13, each relation, the bar with one relation left out, the second-largest score as threshold, the facts passing, and the estimate beside the end-of-prompt detector.

**The sample, frozen** (2026-11-22, before any reading with it):
- **The writer** was Claude Opus, run as a separate agent with the brief alone (`research/results/slots/25_mean_veto/brief_sample_writer.txt`, printed by the script) and told to use no tool and read no file. As with the near questions, its record shows one tool call, which I take to be the hand-back of its answer.
- **The file:** `research/results/slots/25_mean_veto/sample_own.json`, SHA-256 beginning `60c7d8416ae5b81d`, 20 relations with 8 plain and 4 chat wordings each. The script's check finds no problem (`check_sample.txt`).
- **What it shares:** 48 of the 240 wordings equal one the bench asks its relation with, mostly the plainest forms ("The owner of {s} is", "Who founded {s}?"). By the plan they are left out of the sample, so a fact's sample has fewer than 12 wordings and its stated rate is one in the sample's size plus one. 39 equal a wording the table was taught.
- **A smoke run first,** on the chief executive's four facts with stand-in files in place of the near questions and of the sample, to check the code. Nothing was changed after it but two labels in the report.


## Result of the check before any build: with another writer's sample the veto keeps the answers and stops too few leaks; the kill rule is met

Raw numbers: `research/results/slots/25_mean_veto/` (`veto_sample.json`, `veto_sample.log`, `report_sample.txt`, the sample, its brief and its check), from `scripts/diag_mean_veto.py` (`run --sample`, then `report`). The run took 9 minutes.
- **The gate holds:** the scores at the test prompts equal step 18's stored ones, with a largest difference of 0.
- **The samples:** with the 48 wordings the bench also asks left out, a fact's sample has 7 to 12 wordings.

**The rule's reading: arm A, the mean at layer 11, the untaught near questions.**

| The threshold set from | Leaks left of 303 | Answers kept of 1,089 | Facts passing every check but near questions | With near questions held to the same-subject bar |
|---|---|---|---|---|
| No veto | 303 | 1,089 | 75 | 4 |
| The relation's own drafted wordings (step 18) | 87 | 788 | 37 | 27 |
| The bench writer's training wordings (the revision) | 143 | 1,015 | 66 | 34 |
| **Another writer's sample (this check)** | **169** | **1,025** | 66 | 29 |
| The bars | at most 151 | at least 981 | | |

- **By the kill rule, the mean veto stops here on this model.** It keeps the answers and leaves 169 leaks, 18 over the bar.
- **The mean at layers 9, 10 and 12 misses the same bar:** 168, 156 and 159 leaks left.
- **The change of writer did not cost answers, as I had feared it would.** The veto fires on 7.4% of the test recall prompts, under the 9.7% the samples' sizes state. This writer's sample set the thresholds higher than the bench writer's own wordings did, and the price was paid in leaks, not in answers.
- **A lower threshold doesn't meet both bars either:** above all but one sample wording, 122 leaks are left and 943 answers kept.

**By relation the outcome swings with the sample** (leaks left of the untaught leaks, this check against the revision):
- **Better here:** birthplace 6 of 19 against 12, the work's creator 1 of 13 against 8, the product's year 4 of 12 against 10.
- **Worse here:** the concept's proposer 21 of 24 against 3, the event's city 16 of 16 against 4, the medicine's maker 9 of 22 against 0, the work's year 24 of 24 against 19.
- **The same in both:** the four company relations keep more than half of their leaks (chief executive 6 of 6, founder 6 of 9, head office 7 of 11, owner 3 of 4), and the town's founding year keeps 4 of 23.
- **At most half left in 9 of 20 relations,** against 8 in the revision, and not the same ones. A threshold from a dozen wordings is that unsteady.

**After the fact: beside step 17's end-of-prompt detector,** estimated as before on the 112 facts: 72 pass every check but near questions, against 75 now and 79 with the detector alone, and 31 pass with near questions held to the same-subject bar, against 4 now.

**What steps 16 to 18 leave about near questions on this model:**
- **A detector alone separates them at no reading** (step 17).
- **A taught contrast read as a mean over the question's words stops about half of the leaks** for 6% to 7% of the answers: 53% with one sample and 44% with another, where the bar was half. Read at the last token of layer 20 it stops about a fifth.
- **It never helps the company relations,** and which other relations it helps changes with the sample.
- **So the veto is not built.** Near questions stay a stated limit, reported and not judged.

**By the rule:** step 17's end-of-prompt detector goes to a blind bench by itself, with near questions reported and not judged. What comes next is my choice.


## Step 19: the end-of-prompt detector by real generation, at the build's own site (written 2026-11-22, before it ran)

**Why:** the one lead steps 17 and 18 leave is the detector for the prompt's last token, fitted against the prompt negatives alone. Its pass in step 17 is an estimate, and the estimate rests on two things no generation has checked:
- a prompt it newly opens is assumed to get its answer at the benches' measured rate, 0.90 to 0.92;
- the layer's competition between a company's facts, and its veto by the shared detectors, are left out.

Before a reading at another layer is built and a blind bench spent on it, the detector is scored by real generation where the layer already has a place for it. Forward passes and closed-form fits only (`scripts/end_detector_check.py`).

**What runs,** on the two saved API tables of bench40e and bench40f (80 facts, 56 passing):
- **The detector:** for every fact and form, the build's positives against the prompt negatives alone, at layer 20, the threshold at the negatives' 99th percentile and δ half the way to the positives' median. That is step 17's rule R2 at the build's own site. The readings come through the package's read cache, as in a fast build.
- **Where it sits:** in the layer's existing slot for detectors that count only at the prompt's last token (`end_detectors`). A fact then opens at the prompt's end by its strict detectors or by this one, and inside replies by the strict ones alone. The competition and the veto act as they always do.
- **The scoring:** every check of the bench by greedy generation, as `scripts/score_table.py` scores a table.
- **Step 17's estimate for this reading:** 62 facts of 80 (31 and 31), with 4 questions about another relation newly opened.

**The gate:** the layer built by the script with no end-of-prompt detectors must have the same detector matrix, thresholds and δ as the package's own layer for the table.

**Measured:**
- the facts passing on each bench, and which change;
- the recall replies with the whole answer (744 of 960 now);
- **the answer rate of newly opened prompts:** of the recall prompts that were shut at the prompt's end in the stored run and are open with the new detector, the share whose reply has the whole answer;
- the same-company leaks, and every other check.

**Fixed now:**
- **The estimate holds by generation** if both:
  - at least 59 facts of 80 pass every check, which is the estimate less 3 and 3 more than now;
  - the newly opened recall prompts give the whole answer at 0.75 or more.
- **What a pass earns:** the reading that passed step 17, at layer 16, built into the inserted layer as a detector for the prompt's last token, frozen, and then a fresh blind bench.
- **Kill rule:** if either is missed, the end-of-prompt detector earns no blind bench, and the question detector's fit and site are closed on this model. What then changes is the model: the 4B, whose same-subject leaks are higher and whose near questions are untested. The lost pushes are a separate pool on this model, still to be measured.
- **Both benches are development benches.** The result checks an estimate; it is not a score.


## Result of step 19: by real generation the end-of-prompt detector passes 66 facts of 80 against 56, and the estimate holds

Raw numbers: `research/results/slots/26_end_detector/` (`bench40e.jsonl`, `bench40f.jsonl` with every reply and each prompt's score, their logs, `report.txt`), from `scripts/end_detector_check.py`. Each bench took 5 minutes to fit and 5 to score.
- **The gate holds on both tables:** the layer built by the script with no end-of-prompt detectors has the package's detector matrix, thresholds and δ.
- **No smoke run:** the first run was bench40e's. Nothing was changed between the two benches.

| Greedy generation, every check | bench40e | bench40f | Together, of 80 |
|---|---|---|---|
| The layer as built (now) | 32 | 24 | 56 |
| Step 17's estimate for this reading | 31 | 31 | 62 |
| **With the end-of-prompt detector** | **35** | **31** | **66** |
| Recall replies with the whole answer, of 480 | 437 against 386 | 425 against 358 | 862 against 744 |
| Shut before, open now; of those, with the whole answer | 62; 57 (0.92) | 98; 83 (0.85) | 160; 140 (0.875) |

- **By the rule, the estimate holds by generation:** 66 facts against a bar of 59, and newly opened prompts answer at 0.875 against a bar of 0.75.
- **11 facts pass only with it and 1 only without it.** By relation: owner 10 → 16 of 20, founder 14 → 16, chief executive 13 → 14, head office 19 → 20.
- **The estimate's rate was about right, and its leak rule too strict.** It assumed 0.90 to 0.92 for newly opened prompts and generation gives 0.875. It failed a fact for any newly opened question about another relation: four such questions do open at the prompt's end, as it counted, and no fact fails its same-company check. That is where bench40e's 35 against 31 comes from.
- **The one fact lost** is a chief-executive fact of bench40e whose company's head-office fact now answers a general question about the company ("Ferrara" in one reply), which the check on other stored facts counts. This is the looser start's failure, once.
- **Same-company leaks rise a little on bench40e and stay inside the bar:** the mean goes from 0.009 to 0.022, and 7 facts have one where 3 had. On bench40f there is none, before or after.
- **Nothing else moves:** no look-alike, unrelated-company or name-slot leak, no known fact lost, no drift.
- **The 14 facts that still fail:** 11 on recall (most now at 0.50 to 0.67), 3 on another stored fact's answer and 1 on its sibling's. Only 8 recall prompts stay shut at the prompt's end. The other 90 missing answers are prompts that open and still don't give the whole answer: 70 that were open before and 20 of the newly opened. That is the pool of lost pushes.

**How it relates to what was measured before:**
- **The count equals step 15's in-sample ceiling, "about 66 of 80 if every form were taught", with no form taught.** That ceiling was an estimate, so this is the same number and not a proof of it. What the missed wordings needed was a detector fitted against prompts alone.
- **Most of the gain is the owner facts of bench40f.** By relation the 10 are owner +6, founder +2, chief executive +1 and head office +1; by bench, +3 and +7. Without the owner facts the count goes from 46 to 50 of 60.
- **The looser start of v1 gained no fact on blind bench40e** (31 = 31, with 180 facts loaded). It differs in two ways that were not separated here: its detector kept the readings inside the taught answers as a group of negatives, and its table held 180 facts where these hold 40. The shipped `looser_start` option, run as a third arm wherever this is judged next, would separate them. If it does as well on a table built through the API, no new fit is needed.
- **Layer 16, the reading that passed step 17's rule, is not needed for this.** It passed there because the estimate failed layer 20's four newly opened questions, which generation doesn't. Layer 20 needs no reading from another layer, and the layer already has the slot.

**What it does not show:**
- **A blind bench.** bench40e and bench40f have been read many times, and this reading was chosen for this check after step 17's estimates on them.
- **A table of 180 facts,** benchmix1's kinds of subject, or the grafted model. The compile takes the strict detectors only.
- **Near questions.** On benchmix1 this fit opened about 60 more of them in step 17's readings. They are reported, not judged.

**What the pass earns, as fixed before the run:** a build and a fresh blind bench. The plan named layer 16; the result says layer 20, where the detector is now measured by generation and nothing new has to be read. Which of the two goes to the blind bench is my choice, and I would take layer 20.


## Step 20: the end-of-prompt detector on a fresh blind bench, beside the layer as built and the looser start (written 2026-11-22, before the dry run and before the bench's source was opened)

**Why:** step 19 moved the facts passing by real generation, 56 to 66 of 80, on two benches that have been read many times and that chose the reading. No blind count has moved since bench40g. The detector for the prompt's last token now goes to a fresh blind bench. The shipped looser start runs beside it, since the two sit in the same slot of the layer and were never separated.

**Three arms, one table.** The bench's 40 facts go through the API into one table with the built-in wordings and the fast build, as bench40g's reference was built (`scripts/editor_wordings_bench.py` with no added wordings, `teach_cap=30`). The table is saved, and every arm is scored from the saved table by greedy generation with every check:
- **As built:** the layer the package builds from the table (`scripts/score_table.py`).
- **The looser start:** the package's `looser_start`. Its parts are built for every fact of the saved table as `Graft.add` builds them (`build_looser`), and the option is switched on (`scripts/end_arms.py looser`).
- **The end-of-prompt detector:** `scripts/end_detector_check.py run`, the script of step 19 with no byte changed.

The arms differ in the detectors that count at the prompt's last token and in nothing else. The strict detectors, the answer vectors, the name match, the competition and the veto are the saved table's in all three.

**Layer 20, a deviation from step 19's plan,** which named layer 16 for the build. Step 19 measured the detector at layer 20 by generation. Layer 16 was preferred only by an estimate whose leak rule generation did not bear out, and layer 20 needs no reading from another layer.

**The dry run, on bench40g: reported, not judged.** The whole chain runs first on bench40g's saved table (`runs/editor/bench40g_reference/table.graft`, 30 of 40 with the built-in wordings). bench40g had no say in choosing this reading, but it has been read in steps 11 and 13, so its counts are no score. It is the shakedown of the chain:
- **Only a bug found there may change code,** and any change is written here first.
- **The as-built arm from the saved table** is compared with the stored rows of the run that built it (`research/results/bench40g/reference.jsonl`).
- **The looser arm's gate:** the script stops if no fact gets a detector for the prompt's last token, or if those detectors open no recall prompt. A table saved without the option holds no such parts, and the option alone would silently give the layer as built.

**The blind bench, bench40h.** A separate agent, a bigger model, writes its source. It is given the brief that `cramschool bench brief` writes with seed 20261126 over the nine earlier bench files (bench40 to bench40g, devpairs and benchmix1), the avoid list and the brief's scripted check, and nothing else: no code, no bench, no result. The brief's SHA-256 begins `0a25984f3eb5dde0` and the avoid list's `92f675a4fd23125a`. The source stays closed until the dry run is done, the tests pass and the code is frozen by content hash.

**Counting, on the blind bench:**
- **Every test wording,** as bench40e, bench40f and bench40g were counted.
- **The conservative count of step 11:** a test wording equal to a taught wording is left out of that fact's recall in every arm.
- **The rule must hold on both.**

**Fixed now, before the dry run:**
- **Confirmed** if the end-of-prompt detector's arm passes at least 3 more facts than the as-built arm on both counts, and at most 1 more fact fails a leak, garbling or damage check. This is step 11's bar.
- **What confirmed earns:** the detector built into the package as an option of the inserted layer, off by default, documented with this count.
- **Unless the looser start ties it.** The looser start ties if it passes at least the new detector's count less 1, and no more of its facts fail a leak, garbling or damage check than the new detector's. Then no new fit is packaged: the option that already ships is the one the docs point to for missed wordings, with this count.
- **With 1 or 2 more facts:** it stays a script. No option, no claim.
- **Kill rule:** with no more facts than the as-built arm, or with more than 1 more fact failing a leak, garbling or damage check, the end-of-prompt detector is not built. The question detector's fit and site are then closed on this model, as step 19's kill rule says, and what remains on the 0.8B is the pool of lost pushes.
- **The looser start is judged by the same bar against the as-built arm.** It already ships, so a pass or a miss changes only what the docs say of it on a table built through the API.
- **No revision after the source is opened.**

**Reported, not judged:**
- the exact McNemar test on the discordant facts (the statistics section above). With 40 facts, 3 facts gained and none lost gives p = 0.25, and p < 0.05 needs 6 and none. The rule is the count, and the test is reported beside it;
- the recall replies with the whole answer, prompt by prompt against the as-built arm;
- the counts by relation, the same-company leak, and bench40g's dry run.

**What it will not show:**
- a table of 180 facts, where the looser start gained nothing on bench40e;
- subjects other than companies, for which the looser start isn't built;
- the grafted model, whose compile takes the strict detectors only;
- near questions, of which this fit opened about 60 more on benchmix1 in step 17's readings.


## Step 20, the dry run on bench40g, and the freeze (written 2026-11-22, before bench40h's source was opened)

Raw numbers: `research/results/slots/27_end_arms/bench40g/` (the three arms' rows and logs, `report.txt`), from `scripts/score_table.py`, `scripts/end_arms.py` and `scripts/end_detector_check.py`. The looser arm took 18 minutes to build and 2 to score, the other two about 5 each.

| bench40g, the saved table of step 11's reference: reported, not judged | As built | Looser start | End-of-prompt detector |
|---|---|---|---|
| Facts passing, every test wording, of 40 | 30 | 33 | 36 |
| Facts passing, taught test wordings left out | 29 | 32 | 36 |
| Recall replies with the whole answer, of 480 | 358 | 393 | 415 |
| Facts failing a leak, garbling or damage check | 0 | 0 | 0 |
| Facts passing only with it; only without it | | 3; 0 | 6; 0 |
| Exact McNemar p against as built | | 0.25 | 0.031 |
| Recall prompts shut for the strict detectors at the prompt's end and open with it; of those, with the whole answer | | 43; 38 (0.88) | 79; 66 (0.84) |
| Recall prompts still shut at the prompt's end | 83 | 40 | 4 |

"Shut" and "open" in the last two rows are the fact's own detectors' scores at the prompt's last token, read apart from the layer: its name match, the competition between a company's facts and the veto are not in them.

- **The chain's checks hold:**
  - the as-built arm from the saved table gives the stored run's 30 facts and every one of its 2,440 replies;
  - the looser arm's gate: all 40 facts get a detector, and the layer opened through it 56 times;
  - step 19's gate holds on this table too.
- **No bug was found, and no code was changed after the dry run.**
- **By relation, of 10:** as built, chief executive 6, founder 8, head office 10, owner 6. The looser start adds 3 owner facts. The end-of-prompt detector adds 4 owner facts and 2 chief-executive facts.
- **Other questions about the same company:** 7 newly open at the prompt's end with the looser start and 1 with the end-of-prompt detector. No reply to them leaks in any arm.
- **6 of the 480 recall prompts** have a test wording equal to a taught one.
- **What it says, as far as a bench that has been read can:** on a bench that had no say in choosing this reading, the gain is there. The looser start gets half of it on a table of 40 facts built through the API, where on bench40e, with 180 facts loaded, it got none.

**The source of bench40h** was written while the dry run ran. It passed the brief's scripted check on the writer's first round, and I have not opened it. Its SHA-256, computed from the file, begins `212526679535735e` (7,236 bytes). The writer's own report gave the 16th character as `7`; the hash here is the file's. The writer was Claude Opus, run as a separate agent. Its record shows 9 tool calls and one round of the check. I take them to be reading the brief and the avoid list, writing the file, running the check and hashing it; I have not read its transcript, which holds the source.

**Frozen, by content hash** (`git hash-object`, on top of commit ae453f7; the 278 tests pass):
  - `scripts/end_detector_check.py` 988874da0a8e, as committed in step 19
  - `scripts/end_arms.py` 5b0f36c8bcbd
  - `scripts/score_table.py` 19052515c6f4
  - `scripts/editor_wordings_bench.py` b3e47dc3e97a
  - `src/cramschool/graft/build.py` 9edaa1441d9e
  - `src/cramschool/graft/layer.py` c121b892efce
  - `src/cramschool/graft/table.py` 0899b3696082
  - `src/cramschool/graft/core.py` 6096a2f5d007
  - `src/cramschool/graft/wordings.py` 499983605f76
  - `src/cramschool/bench/scoring.py` 3863e980f3b2
  - `src/cramschool/bench/build.py` 2e4a298b0297
  - `research/results/bench40h/chain.sh` fa1da642e54e

**The run** (`research/results/bench40h/chain.sh`, one stage per call): the source is copied in and the bench built; the unedited model is run; the 40 facts are added through the API into one table, which is saved; the saved table is scored as built, with the looser start and with the end-of-prompt detector; then the report with the rule written above.


## Result of step 20 on blind bench40h: confirmed, 31 of 40 against 23, and the looser start doesn't tie it

Raw numbers: `research/results/bench40h/` (the unedited model in `base.jsonl`; the table scored when it was built, `built.jsonl`; the three arms from the saved table, `as_built.jsonl`, `looser.jsonl` and `end_detector.jsonl`; their logs; `report.txt`; `chain.sh`). The bench: `benchmarks/bench40h.json`, from `bench40h_source.json`.
- **The unedited model gives none of the 40 answers,** so no fact is excluded.
- **Three taught wordings equal test wordings of the bench,** one each for owner, founder and head office: 15 of the 480 recall prompts.
- **The as-built arm from the saved table is the run that built it:** 23 facts in both, and no reply differs.
- **Nothing was changed after the source was opened.**

| bench40h, blind: 40 facts through the API, one saved table, the fast build | As built | Looser start | **End-of-prompt detector** |
|---|---|---|---|
| Facts passing, every test wording | 23 | 28 | **31** |
| Facts passing, taught test wordings left out | 23 | 28 | **30** |
| Facts passing only with it; only without it | | 5; 0 | 8; 0 |
| Exact McNemar p against as built | | 0.062 | 0.008 |
| Recall, mean | 0.73 | 0.81 | 0.86 |
| Recall replies with the whole answer, of 480 | 351 | 387 | 415 |
| Recall with the company's name in lower or upper case, mean | 0.73 | 0.78 | 0.85 |
| Replies gained; lost, prompt by prompt | | 37; 1 | 64; 0 |
| Facts failing a leak, garbling or damage check | 0 | 1 | 0 |
| Same-company leak: facts with any (mean) | 3 (0.009) | 6 (0.019) | 3 (0.009) |
| By relation, of 10: owner; chief executive; founder; head office | 6; 6; 3; 8 | 8; 6; 5; 9 | 10; 7; 5; 9 |
| Recall prompts shut for the strict detectors at the prompt's end and open with it; of those, with the whole answer | | 50; 45 (0.90) | 90; 78 (0.87) |
| Recall prompts still shut at the prompt's end | 90 | 40 | 0 |

- **By the rule fixed before the dry run, confirmed.** The end-of-prompt detector passes 8 more facts than the layer as built on every test wording and 7 more with taught test wordings left out, against a bar of 3. No fact fails a leak, garbling or damage check, and none of the 23 is lost.
- **The looser start doesn't tie it:** 28 against 31, where a tie needed 30, and one of its facts fails a check. By the same bar it gains too, 5 facts more than as built.
- **It is the first change since v1 to pass more facts on a blind bench of new facts.** The looser start (31 = 31 on bench40e) and a bigger model's wordings (29 = 29 on bench40g) did not, and the armed push and a larger push never reached a blind bench.
- **No leak comes with it:**
  - the same three founder facts leak to one same-company question each ("The history of {s} stretches back to"), as in the layer as built. There the reply reached the founder's name a few words in, and now it begins with it;
  - 11 of the 320 replies to other questions about the company differ from the layer as built's: those 3, 6 that now give the sibling fact's answer to the sibling's own question, and 2 with neither answer;
  - 4 same-company prompts newly open for the fact's own detector at the prompt's end, against the looser start's 11;
  - no reply about a look-alike or an unrelated company changes (0 of 1,320), no known fact is lost, and there is no drift.
- **The looser start's failing fact** is an owner fact whose company's founder fact answers a general question about the company ("Milica Vukovic, a Serbian-American woman who founded the brewery"). That is its known failure. The end-of-prompt detector did the same once on bench40e in step 19, and not here.
- **Where the gain is:** owner 6 to 10, founder 3 to 5, chief executive 6 to 7, head office 8 to 9. Owner questions gain most, as on bench40f and bench40g.
- **The fact's own detector is now over its threshold at the end of every recall prompt of the bench.** The strict detectors are over theirs on 390 of the 480. On the 90 others the new one is, and 78 of them give the whole answer (0.87, against 0.875 in step 19 and 0.84 on bench40g). These are the fact's own scores, read apart from the layer: its name match, the competition between a company's facts and the veto are not in them (the layer vetoed 5 times in this run, as in the layer as built).
- **What still fails is, by those scores, not the fact's own question detector.** The 9 failing facts all fail on recall with every one of their prompts over threshold: 5 founder facts, 3 chief-executive facts and 1 head-office fact. Of the 65 replies without the whole answer, 53 are on prompts the strict detectors were already over threshold on, the same 53 as in the layer as built, and 12 are newly opened ones. Most look like pushes that lose and answers that start and break off ("Palesc"), the pool step 19 left. That is read from the replies, and nothing here measured it.
- **The frozen files were hashed again after the run,** before anything else was edited: all twelve are as at the freeze.
- **bench40h is a development bench from here on.** Its replies have been read.

**The four benches where it has been scored by generation:**

| Facts passing, of 40 | As built | End-of-prompt detector | Gained; lost |
|---|---|---|---|
| bench40e (development, chose the reading) | 32 | 35 | 4; 1 |
| bench40f (development, chose the reading) | 24 | 31 | 7; 0 |
| bench40g (read before, no say in the reading) | 30 | 36 | 6; 0 |
| **bench40h (blind)** | **23** | **31** | **8; 0** |

**What it earns, as fixed:** the detector is built into the package as an option of the inserted layer, off by default, documented with bench40h's count.

**What it does not show:**
- **A table of more than 40 facts.** The looser start gained 5 here and 3 on bench40g with 40 facts loaded, and none on bench40e with 180.
- **Subjects other than companies.** The fit needs nothing that is about companies, but it has been scored on company benches only.
- **Near questions.** On benchmix1 this fit opened about 60 more of them in step 17's readings, and no bench of companies asks one.
- **The grafted model.** The compile takes the strict detectors only, so the facts in the model's own neurons stay at the as-built count.


## Step 20, the build: the end-of-prompt detector as an option of the package (written 2026-11-22, before the code was written)

**What is built is what bench40h judged, and nothing more:**
- **The option:** `GraftOptions(prompt_detector=True)`, off by default. Every fact gets a second detector per form: its positives against its prompt negatives alone, at the table's layer, with the threshold at the negatives' 99th percentile and δ half the way to the positives' median. A form whose median doesn't clear its threshold has none. It sits in the layer's slot for the prompt's last token. There is no new setting.
- **When it is built:** as a fact is added. On a loaded table the option can be switched on, and the missing detectors are then built on first use, as the armed push's are. They are saved with the table and loaded back.
- **One slot:** it can't be combined with `looser_start` or `end_quantile`, and the layer refuses the combination.
- **The grafted model:** `export_grafted` warns that the option is left out of the compile.
- **What a relation must not answer** (`add_avoid`, experimental): the new detector's threshold stays above those questions too, as the other detectors' do. No bench measured that.

**The gate, before any doc names the option under bench40h's count:**
- **The package's path reproduces the script's rows.** On the saved tables of bench40h and bench40g, with the option switched on and scored by `scripts/score_table.py --prompt-detector`, every reply of every group must equal the rows `scripts/end_detector_check.py` gave (31 and 36 facts).
- **Saved and loaded back,** the layer has the same detectors.
- **With the option off,** the tests and `scripts/check_graft_reproduction.py` pass as before.
- **The path a user takes** is then run once: bench40h's 40 facts added through the API with the option on from the start. A reading shared between facts can differ in its last bits from one read alone, so this count is reported beside the script's, and if they differ the docs say which path each belongs to.
- **If the saved tables' replies don't reproduce,** the count stays the script's and the option is not documented under it.


## Result of the build: the option gives the script's runs reply for reply, and the gate holds

Raw numbers: `research/results/bench40h/` (`option.jsonl` and its log, the option switched on for the saved table; `option_saved_check.txt`; `option_user.jsonl` and its log, the table built with the option on from the start; `reproduction_after_option.log`) and `research/results/slots/27_end_arms/bench40g/option.jsonl`.

| The gate | Result |
|---|---|
| bench40h's saved table with the option switched on (`scripts/score_table.py --prompt-detector`) | 31 of 40; none of the 2,440 replies differs from the script's |
| bench40g's saved table, the same | 36 of 40; none of 2,440 differs |
| The table saved with its detectors and loaded back | the same layer: detector matrix, thresholds, δ and columns |
| The path a user takes: bench40h's 40 facts added with the option on from the start, the fast build | 31 of 40, and 30 with taught test wordings left out; none of 2,440 replies differs |
| A table built without the fast build: bench40e's saved table with the option switched on | 35 of 40; none of 2,440 replies differs from step 19's run |
| Option off: the tests | 281 pass, the 278 and 3 new ones |
| Option off: `scripts/check_graft_reproduction.py` | all checks pass, with 180 facts loaded |

- **The gate holds,** so the docs name the option under bench40h's count.
- **What was built:**
  - `GraftOptions(prompt_detector=True)`, off by default, and `build.build_prompt_detectors`, the script's fit moved into the package;
  - the detectors are built as a fact is added, or on first use for a table that has none, and saved beside the fact's parts as `<id>.prompt.pt`;
  - the layer refuses the option together with `looser_start` or `end_quantile`, and `export_grafted` warns that the compile leaves it out;
  - `scripts/score_table.py --prompt-detector` and `--save`, and `scripts/editor_wordings_bench.py --prompt-detector`.
- **What it costs:** two more detectors per fact, 168 columns in the layer against 88 for 40 facts, about 8 KB per fact in float32. With the fast build the 40 facts took 621 seconds against 525, about 2.4 seconds more per fact.
- **One guard beyond the plan:** `looser_start` switched on for a table whose facts were built without it used to run the layer as built, without a word. It now raises. Step 20's plan had to work around exactly that.
- **Of the twelve frozen files, four changed in the build:** `build.py`, `table.py`, `scripts/score_table.py` and `scripts/editor_wordings_bench.py`. `scripts/end_detector_check.py` and `scripts/end_arms.py` are as frozen.
- **Not built:**
  - a flag for it in the command line and in the editor's tools. A table saved with the option keeps it wherever it is loaded;
  - the grafted model's compile.
- **What a relation must not answer** stays unmeasured with this option on a real model, as the plan said. A test on the tiny model checks that the new detector's threshold stays above those questions.

**What step 20 leaves:**
- **GRAFT with the option passes 31 of 40 new facts on a blind bench against 23 without,** by generation, with every check, and nothing lost or leaked.
- **Open, in the order I would take them:**
  - a table of more than 40 facts, where the looser start's gain was gone. bench40h's facts with the 180 of bench40b to bench40e loaded is one run;
  - near questions with the option on, which step 17's readings say get worse, and which no bench of companies asks;
  - the lost pushes: on bench40h, 65 of 480 replies lack the whole answer although the fact's own detector is over threshold;
  - the option in the grafted model.


## Step 21: do a table's other facts reach a fact's replies? bench40h's table with 180 older facts loaded, as a check with a prediction (written 2026-11-22, before the run)

**Why:** step 20 left "a table of more than 40 facts" as the first open item, on the ground that the looser start had gained nothing on bench40e with 180 facts loaded. I read the layer again before planning that run, and the ground doesn't hold.

**What the layer does (`GraftLayer._push`):**
- **A fact is read only where its own company is the one named last.** The loop runs over `by_company[company]`, the facts of that company. A fact about another company is not read for that sequence at all.
- **The veto reads the shared detectors** of the relations the named company has no fact for.
- **So the number of other companies in a table can't move a company's replies.** What a table of 220 facts has that a table of 40 doesn't:
  - more names in the name index;
  - a wider matrix of detectors. Each score is its own column, so only rounding could differ;
  - shared detectors built from a context over every stored fact;
  - 180 more answers for the other-facts check to find in a reply.

**What I had wrong.** The looser start's 31 against 31 on bench40e was read as an effect of 180 facts in the table. It isn't one:
- that run was another bench, with facts built by the bench path, whose strict detectors are also taught the bench's own 12 training wordings;
- what the looser start lost there was a company's head-office fact answering a general question about its own company. That is the failure of its one failing fact on bench40h (f39) too. It needs two facts on one company, not 180 in the table.

Step 20's "what it will not show" and the limits in the docs carry the wrong reading. They are corrected once this run has confirmed the layer's.

**Checked before the run, without the GPU:**
- **The shared detectors' inputs don't depend on the table's size.** Everything `build_shared` reads from its context (the drafted wordings, the generic questions, the 24 invented companies, the guard) is the same for 1, 40 and 220 stored facts. A table of 220 facts carries the shared detectors of a table of 40.
- **No name collides.** No company of the 180 older facts equals, contains or is contained in a bench40h company, and none stands in a bench40h test prompt.
- **No answer is shared,** and none of the 180 older answers stands in a stored reply of the 40-fact table in any arm. Read again with the wider other-facts check, the stored rows give 23, 28 and 31 (`scripts/table_size_check.py predict`).

**The run (`scripts/table_size_check.py run`), two arms, about 8 minutes each:**
- **One layer of 220 facts:** bench40h's saved table of step 20 (`runs/end_arms/bench40h_option/table.graft`, 40 facts through the API, saved with their end-of-prompt detectors) and the 180 cached facts of bench40b, bench40c, devpairs, bench40d and bench40e, the parts `scripts/check_graft_reproduction.py` loads.
- **As built:** every fact with its strict detectors alone.
- **With the end-of-prompt detector:** on bench40h's 40 facts. The 180 older facts carry none: their cached parts were built before the option existed. They aren't read on bench40h's prompts either way.
- **The shared detectors are built again** for the context of 220 facts, and compared with the saved table's.
- **The other-facts check looks for all 220 answers,** and the known facts and the drift are measured with the layer of 220 attached.
- **Compared reply for reply** with the stored rows of the 40-fact table: `as_built.jsonl` and `option.jsonl` in `research/results/bench40h/`.

**The prediction, fixed now:** none of the 2,440 replies of either arm differs from the 40-fact table's. 23 and 31 facts pass, no known fact is lost and the drift is 0.

**If a reply differs,** I look in this order: a score within rounding of its threshold (the wider matrix); the shared detectors (the run again with `--saved-shared`); a name. A difference with another cause is a bug or a channel I haven't seen, and is written here before anything else is done.

**What this run is not: a capacity result.** "31 of 40 with 220 facts loaded" was never at risk, and it doesn't say the gain survives a bigger table in any sense that could have failed.

**The size a fact does feel is the number of facts on its own company:**
- the competition takes the highest score among the company's facts, and at the prompt's last token the end-of-prompt detectors of all of them count;
- the veto reads only relations the company has no fact for. A company holding all four relations has no veto at all.

No bench has more than two facts on a company. What the stored rows of step 20 give, by how many facts the company holds on bench40h (`scripts/table_size_check.py split`):

| bench40h, the stored rows of step 20 | As built | Looser start | End-of-prompt detector |
|---|---|---|---|
| One-fact companies: facts passing, of 20 | 12 | 15 | 17 |
| Two-fact companies: facts passing, of 20 | 11 | 13 | 14 |
| Two-fact companies with both facts passing, of 10 | 1 | 3 | 4 |
| Facts with a same-company leak: one-fact; two-fact companies | 3; 0 | 5; 1 | 3; 0 |
| Facts failing a leak, garbling or damage check: one-fact; two-fact | 0; 0 | 0; 1 | 0; 0 |

The detector gains on both kinds of company, 5 and 3 facts, and loses none on either. Three or four facts on a company needs a bench that has them and a change to the scoring, whose sibling exemption covers one sibling. That is a step of its own, with its own rule, and isn't run under this one.


## Result of step 21: no reply changes with 180 more facts loaded. The table's size was never the risk; the facts on one company are

Raw numbers: `research/results/slots/28_table_size/` (`predict.txt` and `split.txt`, read from step 20's stored rows before the run; `as_built_220.jsonl` and `detector_220.jsonl` with their logs; `chain.sh`). `as_built_220_first.*` is the as-built arm's first run, made before the script printed two more comparisons of the shared detectors. Its rows are the second run's, reply for reply.

| One layer of 220 facts on 180 companies | As built | With the end-of-prompt detector |
|---|---|---|
| Columns in the layer | 448 | 528 |
| **Facts passing, of 40** | **23** | **31** |
| The 40-fact table alone (step 20) | 23 | 31 |
| **Replies that differ from the 40-fact table, of 2,440** | **0** | **0** |
| Replies holding another stored fact's answer, with 220 answers looked for | 0 | 0 |
| Known facts lost; drift | 0; 0.0000 | 0; 0.0000 |
| Vetoes over the run (the 40-fact table: 5 and 5) | 5 | 5 |

- **The prediction holds in both arms.** 180 more facts, 150 more names and 360 more columns move no reply about bench40h's companies.
- **The shared detectors don't depend on the table's size.** Built for the 220 stored facts they are the ones built the same way for the 40 alone, bit for bit.
- **They aren't the saved table's bit for bit, and the fast build is why.**
  - Against the saved table's, the largest difference of a weight is 0.9% of the largest weight, and of a threshold 0.06.
  - The saved table was built with the fast build, which reads each prompt once through its cache. The same build through an empty cache differs from the build without one by as much: 1.0% and 0.10.
  - The run used the rebuilt ones. The veto fired 5 times in each arm, as with the saved ones, and no reply changed, so nothing here hangs on that difference. [GRAFT](graft.md) already states the size of it for the facts' own detectors, a cosine of 0.9975 or more.
- **What it doesn't show:**
  - the 180 older facts with the option on. Their cached parts hold no end-of-prompt detectors, so their own questions weren't scored with it here;
  - anything about more than two facts on one company.

**A correction to the plan above,** from bench40e's stored rows (`fact_table_teach.jsonl` against `fact_table_rev_teach.jsonl`). The plan says the looser start's loss there "needs two facts on one company". It doesn't:
- **Two of the three facts it lost (f21, f28) are on two-fact companies.** Their company's head-office fact answered a general question, and the other-facts check failed them.
- **The third (f26) is a head-office fact on a one-fact company.** Its own city went into general questions about its company: a same-company leak of 0.38.
- **So the loss is a head-office fact answering a general question about its own company,** with one fact on the company or two. It is still a matter of one company's own facts, and not of how many other companies the table holds.

**Corrected in the docs,** which had blamed the table's size for the looser start's 31 against 31 on bench40e:
- the limits of the end-of-prompt detector and of the looser start in [benchmarks](benchmarks.md#a-detector-for-the-prompts-last-token), [GRAFT](graft.md#the-end-of-prompt-detector-optional), the [Python API](python-api.md), the [limitations](limitations.md), the [history](history.md) and the README, and the `GraftOptions` docstring;
- they now say what was measured: no company holds more than two facts, and facts about other companies don't reach a company's replies.

**What step 21 leaves:**
- **"A table of more than 40 facts" is struck from the open items.** In its place: more than two facts on one company.
- **Open, in the order I would take them:**
  - three or four facts on a company. The competition then runs between more end-of-prompt detectors, and a company holding all four relations has no veto. It needs facts added to a bench's companies and a scoring that exempts every fact of the company on its own relation's question, not one sibling. Its plan and rule are written before it runs;
  - near questions with the option on;
  - the lost pushes;
  - the option in the grafted model.


## Step 22: four facts on every company. bench40h's 40 facts scored on a table where each of its 30 companies holds all four relations (written 2026-11-22, before the table was built)

**Why:** step 21 showed that the number of other companies in a table doesn't reach a fact. What a fact shares the layer with is its own company's facts:
- **The competition** takes the highest score among them. With the option on, the end-of-prompt detectors of all of them count at the prompt's last token.
- **The veto** reads the shared detectors of the relations the company has no fact for. The table knows four relations, so a company holding all four leaves the veto nothing to read.

No bench has more than two facts on a company. This step puts four on every one.

**The table.** bench40h's saved table of step 20 (`runs/end_arms/bench40h_option/table.graft`: 40 facts on 30 companies, saved with their end-of-prompt detectors) is loaded, and 80 facts are added through `Graft.add` with the table's own options (built-in wordings, the fast build, `teach_cap=30`, `prompt_detector` on). Every company then holds owner, CEO, founder and head office: 120 facts, four on each of 30 companies.
- **The added facts** (`scripts/dense_companies.py fillers`; `research/results/slots/29_dense/fillers.json`, SHA-256 begins `5e347d5799c8663a`): for each relation, the 20 companies that lack it take the answers of bench40f's and bench40g's 20 facts of that relation, in order. No two of the 120 answers share a word.
- **bench40h's 40 facts keep the parts they were saved with.** A fact's build reads nothing of the table's other facts (step 21). So the 40 facts are step 20's, and what changes is what stands beside them on their company.

**Two arms from the one saved table,** by greedy generation with every check (`scripts/score_table.py`):
- **As built:** the option off, so no fact's end-of-prompt detector is in the layer.
- **With the end-of-prompt detector:** the option on, and all 120 facts carry theirs.
- **The arms are checked, not assumed.** A table saved with the option on loads with it on. The script stops unless the attached layer holds end-of-prompt detectors on 0 of 120 facts in the first arm and on 120 of 120 in the second.
- **Scored:** bench40h's 40 facts on their own test prompts. The 80 added facts have no test prompts; what they do is read from bench40h's prompts (below).
- **Beside them,** the sparse table's rows of step 20: 23 as built and 31 with the detector (`as_built.jsonl` and `option.jsonl` in `research/results/bench40h/`).

**The scoring has to change, and the change is fixed here.** `score_table_fact` knows one sibling per fact. With four facts on a company:
- **Sibling:** the share of a fact's recall replies that hold the answer of any other fact of its company. At most 1 in 12, as now.
- **A same-company prompt that asks for another relation.** bench40h labels each of its 320 same-company prompts: 168 ask for owner, CEO, founder or head office, and 152 are general.
  - The company's fact for that relation may answer it.
  - Any other fact of the company answering there is another fact's answer where it doesn't belong: the other-facts check.
  - The scored fact's own answer there is its same-company leak, as now.
- **A general question:** no fact is exempt.
- **The other-facts check** looks for all 120 answers.
- **With one other fact on the company these are the definitions in use,** so every stored row must come out as it did. Before an arm is scored:
  - the checks are split from the generation and run on stored replies;
  - `sibling`, the other-facts share, its list of hits and the failed checks must reproduce exactly, each row with the list of loaded facts its run used;
  - among the rows are those with hits: the looser start's f39 on bench40h, its f21, f28 and f11 on bench40e with 180 facts loaded, and bench40f's rows through the API;
  - a unit test covers three facts on one company.

**Fixed now:**
- **The four counts lead the report:** sparse as built, sparse with the detector, dense as built, dense with the detector. Each is counted on every test wording and with taught test wordings left out. The sparse table's are 23 and 23, and 31 and 30.
- **Density itself, for each arm against its sparse count.** If the dense table passes 3 or more facts fewer on every test wording (20 or fewer as built, 28 or fewer with the detector), the limitations say so. For the as-built arm that is a limit of GRAFT as shipped, not of the option.
- **The rule for the option, on the dense table:** it holds with four facts on a company if it passes at least 3 more facts than the dense table as built on both counts, and at most 1 more fact fails a leak, garbling or damage check. This is step 20's bar. Requiring both counts is stricter than the one line I gave before the run.
- **Kill rule:** with no more facts than as built on the dense table, or with more than 1 more fact failing such a check, the docs and the `GraftOptions` docstring say the option is for tables with at most two facts on a company. Step 20's blind result stands for those.
- **With 1 or 2 more facts:** no claim for four facts on a company. The docs give the counts.
- **What holding earns:** a line in the docs. Nothing is built either way.
- **No revision after the first arm is scored.**

**Predictions that check the build, not the method:**
- **No veto fires in either arm,** since every company holds all four relations. The sparse table had 5 in each arm.
- **The replies about look-alike and unrelated companies and the name slots are the sparse table's, reply for reply.** No known fact is lost and the drift is 0. None of those prompts names a stored company, so a difference there is a bug in the build.

**Reported, not judged:**
- which facts pass on one table and not on the other, and the replies that differ, by group;
- on the 168 same-company prompts that ask for another relation: how often the fact the dense table stores for it answers, in each arm. These are bench-written wordings that the added facts were never taught;
- on the 152 general questions: any of the company's four answers;
- recall replies holding another of the company's answers, and the exact McNemar test.

**What it won't show:**
- **A blind result.** bench40h is open since step 20, and the added facts are mine. It is a development measurement of one design: four facts on a company, these four relations.
- **Three facts on a company,** or more relations than four.
- **The added facts' own recall** on test wordings of their own. They have none.
- Near questions, the lost pushes and the grafted model.


## Result of step 22: the kill rule is met. With four facts on every company the detector passes 24 of 40 against 23, and 8 facts fail a leak check against 2

Raw numbers: `research/results/slots/29_dense/` (`fillers.json`, `build.log`, `scoring_check.txt`, `dense_as_built.jsonl` and `dense_detector.jsonl` with their logs, `report.txt`, `chain.sh`). The table is `runs/dense/bench40h_dense.graft`, built in 1,162 s, 14.5 s for each added fact.

**Before the arms were scored:**
- **The scoring change reproduces every stored row** (`scripts/check_table_scoring.py`). On 58 stored files of table runs, 2,284 rows, of which 41 hold another fact's answer and 72 a sibling's in a recall reply, the checks as they were and as they are both give the stored `sibling`, other-facts share, list of hits and failed checks.
  - Four more files of benchmix1 were scored by an earlier version of the other-facts check (before the rule for years and cities that turn up by chance), so their stored numbers aren't today's. On their replies the old and the new checks agree with each other.
  - Three shakedown files of benchmix1 have no bench file and were not checked.
- **The arms are what they are called:** end-of-prompt detectors on 0 of 120 facts in the first and on 120 of 120 in the second.

**The predictions that check the build hold:** no veto in either arm; no reply about a look-alike or unrelated company or a name slot differs from the sparse table's; no known fact lost; drift 0.

| bench40h's 40 facts | Sparse table, as built | Sparse, with the detector | **Dense, as built** | **Dense, with the detector** |
|---|---|---|---|---|
| Facts in the table; on a company | 40; 1 or 2 | 40; 1 or 2 | 120; 4 | 120; 4 |
| **Facts passing** | 23 | 31 | **23** | **24** |
| Facts passing, taught test wordings left out | 23 | 30 | 23 | 23 |
| Recall replies with the whole answer, of 480 | 351 | 415 | 350 | 412 |
| Facts failing recall | 17 | 9 | 17 | 9 |
| Facts failing the other-facts check | 0 | 0 | 2 | 8 |
| Facts with another of the company's answers in a recall reply (each in 1 of 12) | 0 | 0 | 6 | 4 |
| Vetoes over the run | 5 | 5 | 0 | 0 |
| By relation, of 10: owner; CEO; founder; head office | 6; 6; 3; 8 | 10; 7; 5; 9 | 6; 6; 3; 8 | 7; 4; 5; 8 |

- **The rule: KILL.** On the dense table the detector passes 1 more fact than as built on every test wording and none more with taught test wordings left out, and 6 more facts fail a leak check. It gains f03, f09, f18 and f38 and loses f12, f20 and f26 (exact McNemar p = 1).
- **Density, with the detector: 7 facts fewer than on the sparse table** (f12, f20, f26, f28, f31, f34, f39; none gained; p = 0.016). Every one fails the other-facts check on one reply.
- **Density, as built: no fact fewer.** The same 23 pass. f28 and f39, which already failed recall, also fail the other-facts check.
- **So, as the rule says:** the docs and the `GraftOptions` docstring state that the option is for tables with at most two facts on a company. Step 20's blind 31 against 23 stands for those.

**The competition between a company's four facts is not what fails them, but it isn't clean.** What the four facts do to each other on the dense table:

| bench40h's prompts, dense table | As built | With the detector |
|---|---|---|
| Of the 168 same-company prompts that ask for another relation: answered with the fact the table stores for it | 138 | 152 |
| The same without the 14 that equal a taught wording ("Q: Who is the owner of {s}? A:"), of 154 | 124 | 138 |
| The same 168: any part of a wrong fact's answer, or of the scored fact's own | 0 | 0 |
| Recall replies in which another fact of the company answers first, of 480 | 4 | 4 |
| Recall replies with another of the company's answers later in the reply | 2 | 0 |

- **On the 168 questions for another relation, no wrong fact answers,** in either arm, with no veto to stop it. The fact the table stores for the relation answers 81% of the wordings it was not taught as built, and 90% with the detector.
- **On recall, another fact of the company takes the first word 4 times in each arm.** The sparse table had none.
  - As built it is an added CEO fact on one head-office wording, all four times: "Do you know where the headquarters of Oxbelvarn Freight are?" → "Ketevan Gelashvili's headquarters is located in …". One of the four had the right answer on the sparse table.
  - With the detector it is that CEO fact three times, and an owner fact once on a founder wording ("Ask who set up Nalqoresh Media and the answer is always"). Three of the four had the right answer on the sparse table.
  - Each fact has one such reply in 12, which the sibling check allows, so none fails by it.
- **Whole answers on recall:** 350 and 412, against 351 and 415 on the sparse table.

**The failures are one general question, answered by a founder fact.** bench40h has 8 general wordings over 152 prompts. Seven of them get no stored answer in any arm. The eighth is "The history of {company} stretches back to", on 20 prompts:
- **With the detector, the founder fact opens at the prompt's last token:** "The history of Taqsunevo Semiconductors stretches back to" → " Lubna Farouk, a visionary entrepreneur who founded the company in 1…". That happens on 7 of the 13 prompts whose company got an added founder fact, and each one fails the scored fact.
- **As built, the founder's name comes later in the reply,** where the model itself writes "was founded by": "… 1996, when the company was founded by Mutale Mwansa". That happens on 7 of the 13 as built, and on 5 of the other 6 with the detector.
- **So the founder's name is somewhere in the reply on 12 of the 13 with the detector and on 7 as built.**
- **The eighth failing fact** is f39, through its company's added head-office fact: "… 1900, when the brewery was founded in Bulawayo," in both arms.

**The other-facts check undercounts the as-built arm, which I had not foreseen.**
- It reads whole answers. The same-company and sibling checks read a distinctive part too, such as a first name.
- Replies are 16 tokens long. A founder's name that starts after "was founded by" is mostly cut off: 6 of the 7 as-built replies hold only a part of it.
- Read as the same-company check reads the fact's own answer, another of the company's answers stands in a general-question reply on 8 facts as built and on 13 with the detector. Of the facts passing, 18 and 20 have none.
- That is a reading made after the run. The rule above is the verdict. But the as-built 23 on the dense table is helped by where the cut falls, and the difference between the arms is smaller than 2 against 8 suggests.

**The sparse table had the same behaviour, and the scoring let it through.**
- On bench40h as scored in step 20, 7 of the 20 history prompts belong to a company with a founder fact. The founder's name is in the reply on 3 of them in both arms: the first word with the detector, later in the reply as built.
- All 3 are the founder fact's own prompts. There the name is the fact's own same-company leak, one prompt of eight, and that check allows two.
- From its sibling's side the same reply fails at once: the other-facts check allows none. The checks are uneven, and a dense table moves most of these replies to the strict side.

**What step 22 leaves:**
- **GRAFT as shipped, with four facts on a company: the same 23 facts pass by the rule, and 18 of them have no part of another fact's answer in a reply to a general question.** No wrong fact answers a question for another relation. On recall another fact takes the first word in 4 replies of 480.
- **The end-of-prompt detector does not carry over to such tables.** Its recall gain stays (62 more whole answers than as built, none lost), and a founder fact's detector also opens on a question about the company's history about half the time. That is the near-question limit of step 17, now measured by generation in one wording.
- **Open, in the order I would take them:**
  - the two leak checks made even (the other-facts check reading parts, or one stated tolerance for both), and every stored result counted again under it. Until then, counts on tables with several facts on a company aren't comparable across arms that put the answer in different places of the reply;
  - the history question itself: whether a founder's name there is a leak or the stored fact used where it belongs is the bench's definition, and it is mine to set before the next bench;
  - the lost pushes;
  - the option in the grafted model.


## Step 23: GRAFT on another group's benchmark, 30 facts on one company, with their scoring (written 2026-11-23, before the wordings were drafted or anything ran)

**Why:** I want to say whether GRAFT is the better way to put facts into this model, and my benches can't answer that: nobody else is scored on them. One other project measures the same job on the same Qwen3.5-0.8B: experiment 9 of `HomoHabilis/weight-space-programming` (commit `6913045`). An editor model wrote a gated circuit into the weights there with closed-form solves. This step runs GRAFT on their benchmark with their scoring. It can lose.

**Their benchmark** (their `wsp/exp9/make_data.py`, fixed seed):
- **30 facts about one invented company,** Veltrano Systems: 8 department heads, 6 product codenames, 6 site cities, and 10 single facts (founding year, ticket prefix, mascot, chatbot, conference room, wiki, stock ticker, guest Wi-Fi, hackathon, newsletter).
- **Each fact has** a statement, 3 questions an editor may use, and 4 held-out questions that only the final evaluation asks.
- **Three more sets:** one question per fact about a neighbour company (an answer there is a leak, "bleed"), and 40 real-world questions, of which their unedited model answers 37.
- **Their protocol:** a system message ("Answer with only the answer itself (a name, number, place or short phrase). No explanation."), the question as the user message, thinking off, greedy, 24 new tokens. A reply counts if it holds the target with case and punctuation ignored; for a two-word target the second word is enough when it has more than 3 letters.
- **Their headline** is the share of the 120 held-out questions answered. A fact is "learned" at 3 of its 4.

**Their published results on the 0.8B** (their `docs/exp9_company_knowledge.md`; the per-question rows are in their `results/exp9_eval/`):

| Method | Held-out, of 120 | Facts learned, of 30 | Real-world kept | Bleed, of 30 |
|---|---|---|---|---|
| Unedited model | 0 | 0 | 37 of 37 | 0 |
| Unedited model, facts in the prompt | 116 | 29 | not applicable | not applicable |
| Their closed-form edit, layers 18 to 22 | 49 | 11 | 34 of 37 | 3 |
| LoRA, 4 epochs with replay (their baseline) | 79% | 23 | 89% | 7% |
| **Editor-written circuit, clean run ("v2")** | **103** | **27** | **37 of 37** | **1** |

The clean run took 180 turns, 1.9 hours and $13.78 of API-equivalent usage. It is one run.

**What I have read of it, and what that rules out.** I have read their generator, so I have seen the held-out questions. From here on:
- **I write no wording and no general question myself.** A writer that never sees the held-out questions drafts them (below), from a brief made by a script out of the editor's fields only.
- **No option of GRAFT is chosen with the held-out questions.** Everything below is fixed now.
- **Their files stay out of this repository.** Their repository has no licence. The benchmark is generated locally from their script, their question text is kept under `runs/` only, and the result files name questions by fact and position.

**The table: one relation per fact.** GRAFT has one slot for the subject. So "who heads Engineering" and "who heads Finance" are two relations of the one company, and the table holds 30 relations with one fact each.
- **This is the regime of step 22, and further:** 30 facts on one company, where no relation is left for the veto to read. Eight of the relations differ by one word (the department), six by the product, six by the site. Telling such questions apart is the limit I stated in step 18.
- **GRAFT as built,** with the options of bench40h's as-built table: push 2, margin 0.5, the fast build, `teach_cap=30`, every other option at its default. No end-of-prompt detector: step 22 limits it to two facts on a company.

**Three defaults of the package would have cost facts before the method was tested. Found by reading the code, fixed here:**
- **The built-in general questions ask for two of the benchmark's facts.** A fact added outside a bench takes "{s}'s stock ticker is", "{s} was founded in the year" and "What year was {s} founded?" as questions it must not answer, and the 48 general company questions include "What is the stock ticker of {s}?" and "When was {s} founded?". The benchmark has a stock-ticker fact and a founding-year fact.
- **The built-in relations ask for another.** Starting from the built-in wordings, the head-office wordings would be negatives for the headquarters-site fact, and the veto would be live on owner, CEO, founder and head office beside department heads.
- **So the 30 relations get a subject kind of their own** (`Relation(subject=...)` with `Wordings.kinds`, the public API step 14 used). A kind brings its own general questions and its own invented names, and only relations of the same kind are negatives for each other or veto each other. Its invented names are the 24 the package generates for a company table holding this company. Its general questions come from the blind writer, who is told what the 30 facts are.
- **This is a defect of the package for any user who adds a relation that a built-in general question asks for.** It goes into the limitations whatever the result.
- **The benchmark's chat format carries a system message, and GRAFT's build has none.** Their editor measured through that format, and their task message states it. Teaching in the format is like for like. The package has no way to say so, so the judged table is built with a tokenizer whose chat template puts the benchmark's system message in front, in my script. The package's code is unchanged.

**The wordings.**
- **Each relation is taught the benchmark's 3 editor questions,** as chat wordings, first in the list.
- **Each gets the package's default draft beside them:** 30 plain and 30 chat wordings, asked for with the package's own drafting prompt (`editors.loop.DRAFT_PROMPT`), with the fact's statement and its 3 questions as the description. Their editor was allowed the same: "Write any other questions or texts you need yourself (for example more wordings of the fact questions)".
- **The kind's general questions:** 24 plain and 24 chat, the size of the package's own set, about things none of the 30 facts answers.
- **The writer** is Opus in a fresh context: three writers, 10 relations each in the benchmark's order, each shown all 30 statements. Each is given the path of its scripted brief and one output path, and is told to read, list or search nothing else (amended before the writers started: I first wrote that I would paste the brief, and a file keeps my own typing out of it). The evaluation's questions are not written to disk until the drafts are in.
- **The filter:** before the build, a script drops every drafted wording that equals a held-out question after normalizing, and logs how many. No held-out wording is taught, so there is one count.
- **The drafting cost** is reported beside their $13.78.

**Two arms, one judged.**
- **Judged: taught in the benchmark's chat format.**
- **Beside it, not judged: the table as the package builds it,** chat wordings with no system message, scored through the same protocol. It shows what a system message never taught costs.
- If the two differ by 12 or more held-out answers, teaching behind a system message becomes a step of its own.

**My copy of their protocol is checked before GRAFT is scored.**
- **The match rule** is written again in my script. On every published row of theirs (20 files) it must give their `correct`.
- **The generated facts** must equal their published rows, question by question and target by target.
- **The unedited model** through my script must give their base counts: 0 of 120 held-out, 0 of 90 editor questions, 0 of 30 bleed, 37 of 40 real-world. With the facts in the prompt it should give their 116 of 120; a difference of more than 2 there means my protocol isn't theirs, and I stop.
- Their loading is bfloat16 on the GPU in batches of 32, left-padded. Mine is the same.

**Before the held-out questions are asked:**
- **The 90 editor questions are asked first,** on both tables. Their editor could ask them. I use them to find bugs in my scripts: the format, the name match, answers with a hyphen or of three letters.
- **A bug in my script may be fixed. Nothing of GRAFT's may change:** not an option, not a wording, not the kind. A weakness of GRAFT that shows there stays and is scored.
- **The 120 held-out questions are asked once per table.**

**The rule, on the judged table:**
- **GRAFT matches their best** at 103 or more of the 120 held-out questions and 27 or more facts learned.
- **It is ahead** only if it matches and the exact McNemar test on the 120 paired questions, mine against their published rows, gives p below 0.05.
- **Kill rule:** at 94 or fewer of 120 (below their LoRA row of 79%), or 23 or fewer facts, or with that test at p below 0.05 against GRAFT, GRAFT loses on this benchmark. The docs then say that for many facts on one company an editor-written circuit is ahead, and I don't write that GRAFT is better at facts.
- **Between:** behind, and not separable from one run of theirs. No claim either way; the docs give the counts.
- **Two of their columns are not wins for GRAFT, whatever they show.** Bleed and real-world kept are fixed by the name switch: a prompt that doesn't name the company is never touched. That switch sits outside the weights, and their circuit detects the company inside them. Both must hold (0 of 30, and every real-world reply equal to the unedited model's), or the build has a bug.
- **No revision after a held-out question is asked.**
- **What matching earns:** a table and a paragraph in the docs. Nothing is built either way.

**Reported, not judged:**
- by kind of fact, beside theirs: heads, codenames, sites, names, prefixes, mascot, year;
- held-out replies that hold another fact's answer;
- the first token against the whole answer, for the codenames and the three-letter answers;
- the package's 24 general chat questions about the company, without those equal to a benchmark question: replies holding any stored answer. Their benchmark has no such check;
- their editor question about four other names: the two their 9B run leaked to, and two names one edit from the company's. And the company by its first word alone, which my name match will not open and a detector in the weights might;
- the time to build, and the as-shipped table's counts.

**What it won't show:**
- **A result in the weights.** Their edit is 11.4M changed weights. Mine is an inserted layer and a name match; the grafted model is not scored here.
- **More than one company,** which is what GRAFT is built for and their circuit was not tested on.
- **A second run of theirs, or of the writer.** One draft, one build.
- **Anything about behaviours.** This is facts only.

## Step 23, a memory fix before any question was asked (2026-11-23)

**The judged build stalled on its seventh fact.** The first six facts took 49 to 143 seconds each. The seventh took 783, the eighth had not finished after 10 minutes, and the GPU was full (23.7 of 24 GB). I stopped the build. Nothing had been saved and no question had been asked of any table.

**The cause is in the package, not in the method.** `direct.answer_chain` reads how far each answer token is from being predicted. It made a float32 copy of the logits at every position of every teaching example, and then read the few rows it needs:
- 150 examples of up to 71 tokens over a vocabulary of 248,320 is 10 GB in float32, and the step holds two such tensors at its peak;
- the benchmark's system message adds 25 tokens to every chat example, which is what took the step past 24 GB. Without it the same table needs 7 to 10 GB;
- past the card's memory the Windows driver pages to system memory, so the build doesn't fail, it crawls.

**The fix reads the rows first and converts them afterwards** (`logits[rows, at].float()` in place of `logits.float()[rows, at]`, in two places). A conversion to float32 gives the same value for an element whether it is done before or after the element is picked, so no number can change.
- **Checked on the real model, in bfloat16 on the GPU,** for three relations of this table (11, 6 and 3 answer tokens; 50, 60 and 150 examples): the answer vectors are bit-identical before and after, and so are the gaps, the sensitivities and the tokens.
- **Peak memory of the step:** 13.9 GB before and 6.7 GB after for 150 examples without a system message; 5.8 and 3.4 GB, and 6.7 and 3.7 GB, for the two smaller sets.
- The 286 tests pass.

**What this changes in the step:** nothing that was fixed. The options, the wordings and the kind are as written above, and the build starts again from the first fact. `teach_cap` exists because of this memory; I did not lower it.

**For the docs, whatever the result:** before this fix a table taught behind a system message, or with long wordings, could stall on a 24 GB card without an error.

## Step 23, before the held-out questions: the checks, the draft and the 90 editor questions (written 2026-11-23, before a held-out question was asked)

**My copy of their protocol is theirs** (`scripts/rival_bench.py data` and `base`):
- the match rule gives their `correct` on all 5,600 published rows of their 20 result files;
- the generated evaluation equals their published rows: 280 questions and targets, in their order;
- the unedited model through my script: 0 of 90 editor questions, 0 of 120 held-out, 0 of 30 bleed, 37 of 40 real-world, with their verdict on all 280. With the facts in the prompt: 87 of 90, 116 of 120, 25 of 30 and 28 of 40, again their verdict on all 280, and 279 of the 280 replies are theirs word for word.

**The draft.** Three writers, about 5 minutes each in parallel, 260,317 tokens in all (85,298, 86,054 and 88,965). Each made two file calls: its brief and its output. That is their own report, and the harness counted 3 tool uses for each, the third being the report.
- 30 relations with 900 plain and 986 chat wordings (the 90 editor questions and 896 drafted), and 24 plain and 24 chat general questions.
- **Dropped by the filter: 4 of 1,800.** One equals a held-out question after normalizing (in the founding-year relation). Three repeat a wording the relation already has. Four relations are left with 29 drafted chat wordings.
- **Two things the writers did that I would not have chosen, and that stay:**
  - the package's drafting prompt names "head office" as a relation to keep away from, so the writer of the headquarters-site relation avoided the phrase and wrote "headquarters", "HQ" and the like;
  - the eight department heads share most of their sentence frames, and so do the six sites.

**The judged table** (`runs/rival_exp9/judged.graft`): 30 facts in 1,586 s, 42 to 66 s a fact after the first (155 s). Each fact's detectors read 25,029 negatives, 2,547 plain and 1,287 chat positives, and 1,372 replies.

**The 90 editor questions on the judged table: 75 of 90** (their circuit: 84). By kind: heads 21 of 24, codenames 16 of 18, sites 15 of 18, names 11 of 18, prefixes 6 of 6, year 3 of 3, mascot 3 of 3. No veto fired, as predicted.
- **No bug of my scripts shows.** The format is the benchmark's, the name is matched on every question, and hyphenated and three-letter answers come out whole ("QUARTZ-4", "KRD").
- **The 15 misses are GRAFT's, and they stay:**
  - a later token of the answer is lost: "Fenna Halvors" for Halvorsen (3), "Kaunas" for Kaohsiung (3), "F.101" for Fenvezen (3), "Mitre" for Mitami (3), "Tabby" for Tabrimo (1). These are the lost pushes of the earlier steps;
  - the first token is lost on one codename ("101" for JUNIPER-2, 2);
  - after a lost push another fact opens inside the reply on 8 of the 15 (6 with its whole answer, 2 cut off after its first token): "Mitre / Kaohsiung", "Kaunas Mitami". When the first token is not the one produced, the layer's one-answer-per-reply rule is not yet set, and a second fact's detector opens further on.
- **Nothing is changed.** The as-shipped table is built next, asked the same 90 questions, and then each table is asked the held-out questions once.

## Result of step 23: the kill rule is met. On their benchmark GRAFT answers 93 of the 120 held-out questions against their circuit's 103, and learns 24 facts against 27

Raw numbers: `research/results/slots/30_rival_bench/` (the rows of both tables and of the unedited model, by fact and position; the three drafts; the logs; `report.txt`). Their question text and the two tables are under `runs/rival_exp9/` and not in the repository.

| Qwen3.5-0.8B, their benchmark and protocol | Editor questions, of 90 | **Held-out, of 120** | **Facts learned, of 30** | Real-world kept | Bleed, of 30 |
|---|---|---|---|---|---|
| Their unedited model | 0 | 0 | 0 | 37 of 37 | 0 |
| Their closed-form edit | 48 | 49 | 11 | 34 of 37 | 3 |
| Their editor-written circuit | 84 | 103 | 27 | 37 of 37 | 1 |
| **GRAFT, taught in the benchmark's chat format (judged)** | 76 | **93** | **24** | 37 of 37 | 0 |
| GRAFT as the package builds it (not judged) | 41 | 51 | 10 | 37 of 37 | 0 |

- **The rule: KILL.** 93 is under the line of 94 that I set. It would have needed 95, two more, to clear it. The 24 facts are 3 under their 27 and one above the line for facts (23).
  - Paired on the 120 questions, 16 are right only for GRAFT and 26 only for their circuit: exact McNemar p = 0.164. The test alone would not call it. The line was fixed before the run, and it is met.
  - **So, as the rule says:** for many facts on one company an editor-written circuit is ahead of GRAFT on this benchmark, and I don't write that GRAFT is better at facts.
- **How the held-out questions were asked.** Each table was asked them once, in their evaluation's order, and the script refuses to write over a rows file. My session restarted right after; the rows on disk are dated 14:07 and 14:08 on 2026-11-23, and nothing was asked again. The report was made from those files.
- **The two columns that had to hold do:** no neighbour question gets a stored answer (0 of 30), and all 40 real-world replies equal the unedited model's, word for word. As written before the run, these are not wins: the name match gives them.
- **The full run gave 76 of the 90 editor questions,** one more than the shakedown's 75. The batches differ between the two runs.

**By kind of fact, held-out:**

| | Heads | Codenames | Sites | Invented names | Three-letter codes | Year | Mascot |
|---|---|---|---|---|---|---|---|
| Their circuit | 24 of 32 | 20 of 24 | 24 of 24 | 23 of 24 | 4 of 8 | 4 of 4 | 4 of 4 |
| GRAFT, judged | 26 of 32 | 20 of 24 | 16 of 24 | 15 of 24 | 8 of 8 | 4 of 4 | 4 of 4 |
| GRAFT as the package builds it | 7 of 32 | 14 of 24 | 9 of 24 | 9 of 24 | 8 of 8 | 0 of 4 | 4 of 4 |

- **GRAFT is ahead on the department heads and the three-letter codes, and behind on the sites and the invented names.** The eight heads differ by one word and were the relations I expected to lose; 26 of 32 held.
- **Facts by held-out answers right:** 16 facts have all 4, 8 have 3, 5 have 1 and 1 has none. The six not learned are the head of Legal, the customer app's codename, the main factory, the research lab, the conference room and the hackathon.

**The 27 misses of the judged table:**
- **15 start right and lose a later token.** (Corrected at step 24: read token by token, 6 lose a later token, and in 9 the first token comes in the answer's other spelling.) "Fenna Halvors" for Halvorsen (3), "Kaunas" for Kaohsiung (3), "F.101" for Fenvezen (3), "Mitre" for Mitami (4), "Tamara" for Tamove (1), "MARA" for MARLIN-8 (1). Four facts account for 13 of them. The fact opened and its first token came; the model then went on in its own way and the layer stopped pushing. These are the lost pushes of the earlier steps, and they are the largest single cause.
- **9 are not answered by the fact:** the model's own answer comes ("David M. Veltrano", "Buenos Aires", "1000").
- **3 are answered by another fact first:** the research lab's question twice by the headquarters' city, and the customer app's codename once by the Asia-Pacific office's city.
- **Another fact's answer stands somewhere in 12 of the 27 wrong replies, and in 3 right ones.** It is the main factory's city 6 times, the Asia-Pacific office's 6, the headquarters' 2 and the mascot 2, mostly after a line break: "Mitre / Kaohsiung". When the first token pushed is not the one produced, the layer's one-answer-per-reply rule is not yet set, and a second fact opens further on.

**A system message the table was not taught behind costs 42 of the 120 answers.**
- **The table as the package builds it answers 51,** and 41 of the 90 editor questions that it was taught word for word, without the system message.
- **63 of its 69 wrong replies hold another fact's answer,** 61 of them the main factory's city after a line break. The facts' detectors stay shut at the prompt's last token, the model's own answer starts, and one fact then opens inside the reply.
- **The difference is over the 12 I set,** so teaching behind a system message is a step of its own. Until it is an option of the package, a table is for prompts in the format it was built in.

**Readings their benchmark doesn't have** (judged table; not the rule):
- **General questions about the company get stored answers.** Of the package's 24 general chat questions, 22 are not a benchmark question. On 18 of the 22 the reply holds a stored answer, and on 9 it is the first word: "What is the main product of Veltrano Systems?" gives "okapi", and "Is Veltrano Systems a public company?" gives "QTG".
  - The answers are the mascot's (5), the main factory's (5), the stock ticker's (4), the Asia-Pacific office's (3) and the founding year's (1, for "How old is …", where the stored fact arguably belongs).
  - By my own benches' same-company check, which allows a quarter, this table fails. The writers' 48 general questions were taught as negatives; these 22 are other wordings.
  - Their edited weights are not in their repository, so their circuit can't be asked these questions. I don't know whether it does better.
- **Other names never leak:** their editor question about "Benvoglio Group", "Valtieri Technologies", "Veltrano System" and "Veltrana Systems" gets a stored answer in 0 of 30 each.
- **The company's first word alone gets nothing either:** "Veltrano" opens no fact, 0 of 30. That is a miss, not a leak, unless the short name is declared as an alias.

**Time and cost.**
- **Building:** 1,586 s for the judged table and 1,449 s for the other on an RTX 3090, with no model in the loop. Calibration and the first fact take about 3 minutes, each later fact 40 to 66 s.
- **Drafting:** three writers, about 5 minutes side by side, 260,317 tokens. I have no dollar figure for it.
- **Theirs:** one editor run of 180 turns, 1.9 hours and $13.78.

**What step 23 leaves:**
- **GRAFT is behind on the one benchmark where someone else is scored,** by 10 held-out answers and 3 facts, with thirty facts on one company. It holds the one-word contrasts I expected to break, and it loses on answers of several invented tokens.
- **Two defects of the package came out of the run and are fixed or stated:**
  - the answer-vector step made a float32 copy of every position's logits and stalled a 24 GB card behind a system message: fixed, with identical numbers;
  - a built-in general question can ask for a relation a user adds ("stock ticker", "founded in"), which teaches the fact to stay shut on its own question: stated in the limitations, not fixed.
- **Open, in the order I would take them:**
  - the lost later tokens. They are 15 of the 27 misses here. That is a count of where the misses are, not a promise that a fix recovers them (step 24: 6 later tokens and 9 first tokens in the other spelling; all 15 recovered);
  - a second fact opening inside a reply after a first token is lost;
  - teaching behind a system message, as an option of the package;
  - general questions on a company that holds many facts;
  - the built-in general questions against a user's relations.

## Step 23, the build: a system message as a setting of the table (written 2026-11-23, before the code was written)

**Why:** step 23 measured it. A table taught with no system message answers 41 of the 90 editor questions and 51 of the 120 held-out ones behind the benchmark's system message. Taught behind it, 75 and 93. The 93 came from a tokenizer patch in my script. Someone who installs the package has no such thing and gets the 51.

**What is built:** `Graft(model, tokenizer, system="...")`.
- **The table keeps its own copy of the tokenizer,** whose chat template puts the message in front of every conversation that has no system message of its own. The tokenizer that was passed in is not changed.
- **Every chat prompt of the build goes through it,** and so do `ask` and `compare`: the teaching examples, the positives, the negatives and the replies.
- **The message is saved in the table file and restored by `load`.**
- **Refused** for a tokenizer with no chat template.
- **Not built:** flags for the command line and the MCP server. The grafted export compiles the facts as they were built and needs nothing new.

**The gate, from files on disk. No held-out question is asked:**
- **The same parts:** the first fact of the judged table, built again through the package with `system=`, has that table's parts bit for bit: the detectors' directions and thresholds and the answer vectors.
- **The same replies:** the judged table, loaded with the message as its setting, gives the 90 stored replies of the shakedown (`judged_taught.jsonl`), reply for reply, with the prompts rendered by the package.
- **Tests on the tiny model:** the message stands in front, once; the caller's tokenizer is untouched; a table saves and loads it; a tokenizer without a chat template is refused.
- **If a part or a reply differs,** the argument is not what step 23 measured, and I say what differs before it goes into the docs.

**What it does not claim:** nothing new is measured. The numbers are step 23's, and the benchmark's held-out questions stay asked once.

## Result of the build: `Graft(system=...)` gives the judged table's first fact bit for bit and its 90 stored replies; the gate holds

`scripts/check_system_setting.py`, on the files of step 23 (`research/results/slots/30_rival_bench/system_setting_gate.txt`):
- **The same parts.** The judged table's first fact, built again through the package with `system=`, is bit-identical to the saved one: the calibration, both detectors' directions and thresholds, and the answer vectors of both spellings.
- **The same replies.** The judged table, loaded from a copy whose `graft.json` names the message, gives 90 of the 90 stored replies to the editor questions, with the prompts rendered by the package. The tokenizer passed in still renders no system message.
- **Tests:** 288 pass, two of them new (the rendering, and a table that builds, saves and loads behind its message).

**What is in the package now:** `model_access.with_system` and the `system` argument of `Graft`, saved in the table file. A table saved before has none and loads as before.

**What is not:** flags for the command line and the MCP server; any new measurement. The 93 of 120 is step 23's, on a table built with the script's patch, which this gate shows the argument reproduces for the first fact and for the taught questions. The held-out questions were not asked again.


## Step 24: finishing an answer. Its tokens written at the model's last layer (written 2026-11-23, before the package code and before any question is asked again)

**Why:** step 23 left "the lost later tokens" as the first open item: replies that start like the answer and end as something else ("Kaunas" for Kaohsiung, "Fenna Halvors" for Halvorsen). This step fixes them.

**A correction of step 23 first.** I had read the 27 misses as text. Read token by token (`scripts/diag_lost_tokens.py`, on the judged table and its stored rows; every question in it was asked at step 23), "15 start right and lose a later token" is wrong about 9 of the 15:

| The 27 held-out misses of the judged table | Replies | Which |
|---|---|---|
| A later token loses after its push was made | 6 | Halvorsen 3, Fenvezen 3 |
| The first token comes in the answer's other spelling, and the layer stops | 9 | Kaohsiung 3, Mitami 4, Tamove 1, MARLIN-8 1 |
| The fact opens and its first token loses to another word | 4 | JUNIPER-2 3, Tabrimo 1 |
| Another fact of the company opens first | 1 | Cebu ("Wellington", the headquarters' city) |
| The fact doesn't open at the prompt's last token | 6 | six facts, one question each |
| Right when asked alone, wrong in the bench's batch of 32 | 1 | Cebu ("Cebu City") |

- **The other spelling.** A reply here starts after a blank line, so the layer pushes "Ka", the answer's first token without a leading space. The model writes " Ka", the same letters with the space, 3 logits ahead; for "Mit" it is 5 to 7. The layer waits for "Ka", reads another token, and stops. The model goes on alone: " Ka" + "unas".
- **The later tokens.** No vector is missing or capped. The push is made and loses by a little: "en" after "Fenna Halvors" by 0.5 to 1.9 logits to the end of the reply, and "en" after "F" by 3 to 4 to a full stop.
  - Given "Fenna Halvors" with no layer attached, the model's own first choice is "en". With the layer attached it ends the reply.
- **The 14 misses among the 90 editor questions are the same kinds:** 5 later tokens, 7 other spellings, 2 other words.

**Corrected after the runs, the same day.** The diagnosis now records which fact opened, and repeats each lost later token with that one position's push left out (`research/results/slots/31_last_layer/lost_tokens.json`). Two things in this section were wrong as first written:
- **The table had 5 replies losing their first token to another word.** One of them, a Cebu question answered "Wellington", is the headquarters' fact opening first. It is 4 and 1, as the table now reads.
- **I had written that the layer's writes at the earlier positions cost "Halvorsen" its last token.** It is the push at that very position. With every earlier push made and this one left out, "en" leads by 4.3 to 5.6 logits. With it, the end of the reply wins by 0.3 to 2.3. A vector measured to raise a token lowered it, three layers below the output.
- **For "Fenvezen" the push helps and is too small:** it gains 0.4 to 1.9 logits where 4 to 6 were needed.

**What a write at layer 20 passes through, measured.** Three more layers follow it, and what they compute at that position is read by every later one.
- **The same vector written at the last MLP's output changes its own position's logits and nothing else:** every other position's logits are bit-identical. Written at layer 20 it moves a later position's logits by up to 2.3. The vector was of the size planned below (norm 31; the stored vectors, doubled, have norm 3 to 5).
- **So its size can be chosen to win,** which step 9 found impossible at layer 20: there a larger push lost answers that were right.

**The change:** `GraftOptions(last_layer=...)`, off by default.
- **`"later"`:**
  1. **Each later answer token is written at the output of the model's last MLP,** and no longer at the table's layer. The vector is the token's own row of the output matrix at length 2R. R is the 99th percentile of the norm of what the final norm reads on the calibration text: 15.75 on this model. Nothing is measured per fact, and no later token can be without a vector.
  2. **The first token is accepted in either spelling.** When the token produced after the first push is the first token of the answer's other spelling, the layer follows that spelling.
  3. **The first token itself is pushed as before:** at the table's layer, at the same places, with the same vector.
- **`"all"`:** the first token is written at the last layer too, by the same rule, and no longer at the table's layer.

**Why 2R and the plain row** (`scripts/diag_last_layer_size.py`: the 90 editor questions with the answer placed behind them, both spellings, 696 answer positions; no held-out question):
- **At 1R every position is won,** the smallest lead 1.2 logits. At 2R the smallest is 4.0 and the median 35. At 3R the smallest is 4.5, so 2R.
- **The unedited model's first choice leads the answer token by at most 17 logits** at these positions (median 3.9).
- **The row alone, through the final norm, puts its own token first for all 131 distinct answer tokens.** The smallest lead is 6.5 logits, over the same word without its space (" Zen" over "Zen"). The smallest leads above are of that kind too: a spaced first token where the reply starts without one, which the layer doesn't choose.
- **The row weighted by the final norm's gains gives the same within half a logit.** The plain row is kept.

**What can go wrong:**
- **A wrong opening is now finished.** A fact that opens on a question that isn't its own writes its whole answer where it used to stop after a token or two. Under `"later"` the openings at a prompt's last token are the same ones, by construction; the completed wrong answers can be more, and the checks that read whole answers will count them.
- **What follows the answer can change:** the positions inside the answer no longer carry a write at layer 20.
- **`"all"`:** every opening starts an answer for certain. Today some wrong openings lose their first token.
- **A grafted model doesn't have it.** Its neurons sit at the table's layer.

**The benches, and what each one is:**
- **The other group's benchmark, step 23's judged table: development from here.** Its held-out questions were asked at step 23 and I have read its misses. **Step 23's result stands: KILL, 93 against 103.** Whatever asking again gives, even above 103, is a development number and not a result against their circuit.
- **bench40e and bench40f:** development. Their saved tables, every check, scored again as built with today's scoring code so the comparison is paired.
- **bench40h's saved table (step 20's, 23 of 40 as built): the confirmation.** It did not choose this design. It is not blind: its replies were read at steps 20 to 22.

**The bar for `"later"` in development, every part:**
1. **The other group's benchmark:** at least 103 of the 120 held-out questions, 10 more than as built. 15 of the 27 misses are of the two kinds it addresses.
2. **bench40e and bench40f:** neither passes fewer facts than as built.
3. **Leaks:** over the two benches at most 1 fact newly fails a leak, garbling or damage check. They are counted as the checks count them, whatever the cause, a wrong answer that is now finished included.
4. **Lost answers:** at most 2 recall replies over the two benches, and at most 2 of the other benchmark's 210 questions, lose an answer they had.
5. **Untouched:** the 40 real-world replies of the other benchmark stay those of the unedited model, and the neighbour company gets 0 of 30.
- **Reported, not in the bar:** the 22 general questions about the company. 18 hold a stored answer as built, which already fails my same-company check.

**`"all"` over `"later"`:** only if it meets the same bar, passes at least 3 more facts over bench40e and bench40f together than `"later"` does, and no more facts fail a leak, garbling or damage check than under `"later"`.

**Kill rule:** if `"later"` misses its bar, the option is not built: its code is taken out again and the finding stays here. The size and the direction are not revised after the first run.

**Confirmation on bench40h** (the chosen setting against as built, every check):
- **Confirmed** with at least 2 more facts passing (25 of 40), no fact lost and at most 1 more fact failing a leak, garbling or damage check. The guide then recommends it.
- **Kept as an option, with no claim of more facts,** with 0 or 1 more and none lost.
- **KILL** with fewer facts passing, or more than 1 more fact failing a check.
- **Either way it stays off by default,** so that every number in the docs reproduces as written.

**The gate before any of it:** with the option off, the judged table gives the 90 stored replies of `judged_taught.jsonl` again, and the test suite passes.


## Step 24, development: both settings clear the bar, and `"all"` goes to bench40h (written 2026-11-23, before bench40h is scored)

Raw numbers: `research/results/slots/31_last_layer/` (`scripts/rival_bench.py ask … --last-layer`, `compare`; `scripts/score_table.py --last-layer`; `scripts/last_layer_report.py`).

**The gate held.** With the option off, the judged table gives the 90 stored replies of `judged_taught.jsonl` again, and the suite passes (292 tests, 4 of them new).

| The same saved tables | As built | `"later"` | `"all"` |
|---|---|---|---|
| The other group's benchmark, held-out answers of 120 | 93 | 108 | 112 |
| … facts learned, of 30 | 24 | 28 | 29 |
| … editor questions, of 90 | 76 | 88 | 90 |
| bench40e, facts passing of 40 (recall replies with the whole answer, of 480) | 32 (386) | 32 (393) | 37 (422) |
| bench40f | 24 (358) | 25 (362) | 33 (395) |
| bench40e and bench40f together, of 80 | 56 | 57 | 70 |

**The bar for `"later"`, part by part:**
1. **The other group's benchmark, 103 or more: met,** 108. All 15 replies of the two kinds it addresses came back, and nothing else moved.
2. **Neither bench passes fewer facts: met,** 32 and 25.
3. **Leaks: met.** No fact newly fails a leak, garbling or damage check. Summed over the facts, every leak measure is what it was as built on both benches.
4. **Lost answers: met.** None of 960 recall replies and none of the 210 questions.
5. **Untouched: met.** The 40 real-world replies are the unedited model's, and the neighbour company gets 0 of 30.

**`"all"` over `"later"`: every condition met.**
- **The same bar:** 112; 37 and 33; no fact newly failing a check; no reply losing its answer; the real-world replies and the neighbour untouched.
- **13 more facts over the two benches** than `"later"` (70 against 57), where 3 were asked.
- **No more facts fail a leak, garbling or damage check:** the same three as built (one on bench40e, two on bench40f).
- **Under the bars, two measures rise on bench40f:** summed over its 40 facts, the same-company leak goes from 0 to 0.125 (one reply of one fact's eight) and the sibling check from 0.333 to 0.417. Garbling falls on both benches (0.333 to 0.25 and 0.5 to 0.417).

**What the two settings show** (the 8 misses left under `"all"`: on 6 the fact doesn't open, and on 2 the headquarters' fact opens first):
- **The first token is where most of it was.** `"later"` finishes answers that have begun: 11 more whole answers over the two benches and one more fact. `"all"` also wins the first token wherever a fact opens: 73 more whole answers and 14 more facts.
- **On the other group's benchmark `"all"` leaves 8 misses,** none of them a lost push. Replies holding another fact's answer fall from 15 to 7: once the first token is certain, the one-answer-per-reply rule is set and no second fact opens further on.
- **General questions about the company:** 20 of 22 replies hold a stored answer under either setting, against 18 as built. The openings are the same; two wrong answers that used to break off are now finished.

**These are development numbers.** 112 is above their circuit's 103, on questions I had already asked and whose misses I had read. Step 23's result against their circuit stands: 93, a loss by the rule.

**Before bench40h is scored, one reading added outside the rule:** the table saved with the end-of-prompt detector (step 20's, 31 of 40), scored again as saved and with `"all"`. It is the setting someone would use together with this one. Reported, not judged; the rule is on the as-built table, as written above.


## Result of step 24: confirmed. On bench40h, 32 of 40 facts pass with every answer token written at the last layer, against 23 as built

Raw numbers: `research/results/slots/31_last_layer/bench40h_*.jsonl` and their logs (`scripts/score_table.py --last-layer all`, `scripts/last_layer_report.py`).

| bench40h's saved tables, 40 facts | Facts passing | Recall replies with the whole answer, of 480 | Facts failing a leak, garbling or damage check |
|---|---|---|---|
| As built | 23 | 351 | 0 |
| **`last_layer="all"`** | **32** | 410 | 0 |
| With the end-of-prompt detector, as saved (outside the rule) | 31 | 415 | 0 |
| With the end-of-prompt detector and `last_layer="all"` (outside the rule) | 40 | 479 | 0 |

**The rule: confirmed.** 9 more facts pass where 2 were asked, no fact is lost, no fact fails a leak, garbling or damage check, and no recall reply loses an answer it had (59 gain one). The as-built count is step 20's, reproduced with today's code.

**What bench40h is:** a bench that did not choose the design, which was fixed from the other group's benchmark and chosen between its two settings on bench40e and bench40f. It is not blind. I read its replies at steps 20 to 22.

**The reading outside the rule:** on the table saved with the end-of-prompt detector, all 40 facts pass, against 31. The one recall reply of 480 without the answer is a refusal ("I do not have access to real-time or private information"), on which the fact doesn't open. The two options act on different misses: the detector opens questions the fact didn't recognise, and the write at the last layer makes every opening give its answer.

**What it costs, as measured:**
- **Same-company leak, summed over the 40 facts: 0.375 to 0.5,** with or without the detector. That is one more reply of 320: a founder fact now answers one of its eight same-company prompts. No fact crosses its bar.
- **13 of the 320 same-company replies differ from as built.** Most are the company's other stored fact, given where the prompt asks for it and now written whole.
- **An answer that opens inside a sentence is written where it opens.** "The CEO of **" became "The Hira Si…": the fact opened one word into the phrase, and the first token is now certain to come there. Garbling as the bench counts it falls all the same (0.417 to 0.333; with the detector, 0.083 to 0).
- **Look-alike and other-company prompts:** none of 960 replies differs from as built.
- **Time:** the size R is measured once per table, 35 seconds on an RTX 3090. It came out as 15.75, so a write has length 31.5.

**What the step shows:**
- **The lost pushes were a property of where the push was written.** At layer 20 the vector has to survive three more layers, and its size is a compromise: step 9 lost answers at every size above 2. At the last layer the write is read only by its own position's output, and it can be made certain.
- **The first token carried most of it.** Finishing a begun answer (`"later"`) gave 1 fact over bench40e and bench40f. Making the first token certain as well gave 14.
- **Over the three benches of 40, `"all"` passes 102 of 120 facts against 79 as built,** with no fact lost on any of them.

**What it does not show:**
- **A blind result.** All four benches had been read before. A fresh bench would give one.
- **A result against the other group's circuit.** 112 of 120 is a development number. Step 23 stands.
- **Near questions.** A fact answers a question close to its relation in 0.72 of replies on benchmix1 (step 14). Those openings are now finished for certain. Not measured with the option; it cannot be lower.
- **Tables with many facts on a company.** General questions on the other group's company: 20 of 22 replies hold a stored answer, against 18.
- **The grafted model.** Its neurons sit at the table's layer and the compile doesn't have the option; `export_grafted` warns.
- **Another model.** Qwen3.5-0.8B only.

**In the package:** `GraftOptions(last_layer="all")` or `"later"`, off by default so that every number in the docs reproduces as written. It can be set on a loaded table. The guide recommends `"all"`. Whether it becomes the default is my decision to make after this, not part of the rule.

**Scripts:** `scripts/diag_lost_tokens.py` and `scripts/diag_last_layer_size.py` (the diagnosis and the size), `scripts/rival_bench.py ask … --last-layer` and `compare`, `scripts/score_table.py --last-layer`, `scripts/last_layer_report.py`.

---

## The default from 1.1: a new table writes its answers at the last layer (2026-11-23, a decision, not a result)

Step 24's rule said the option stays off by default whichever way the confirmation went, and that making it the default was mine to decide afterwards. I have decided: from version 1.1.0, `GraftOptions.last_layer` is `"all"`. Nothing was measured for this entry beyond the gate below. No bench was asked again.

**Why:** on the three benches of step 24 the same saved tables pass 102 of 120 facts with it against 79 without, with no fact lost and no fact newly failing a leak, garbling or damage check. None of those benches was blind when scored. That is the whole evidence, and it is step 24's.

**What changes:**
- **A new table** (`Graft(model, tokenizer)`, `cramschool graft add`, the MCP server's new tables) writes every answer token at the output of the model's last MLP.
- **The size R is measured in the calibration's own passes** (`core.calibration_stats(final=True)`), so a new table costs no extra half minute, and it is saved with the table. The value equals the one measured alone: 15.752145767211914 both ways.
- **`save()` measures R if the table has the setting and no R yet,** so a table switched over after loading keeps it.

**What does not change:**
- **A table saved before this loads with `last_layer=None`,** the write it was built and measured with. `Graft.load` reads a missing setting as None. To switch one over: `graft.options.last_layer = "all"`, then `save`.
- **`GraftLayer` itself still defaults to None.** `cramschool bench run`, the CounterFact runs and `scripts/check_graft_reproduction.py` build the layer directly, so they reproduce as before.
- **Scripts that build a table through `Graft(...)` for a published number now pin `last_layer=None`:** `check_system_setting.py`, `editor_wordings_bench.py`, `kinds_bench_run.py`, `layer_screen.py`, `rival_bench.py` (build) and `validate_api_facts.py`.
- **`prompt_detector` stays an option, off.** Step 22 killed it for tables with more than two facts on a company, so it can't be a default.

**Every published number except step 24's was measured with `last_layer=None`.** Never measured with `"all"`:
- the 180-fact table of the blind company benches (v1's 31 of 40 on bench40e);
- CounterFact, on either model;
- benchmix1's nine kinds of subject;
- the editor benches (bench40g);
- a company with many facts, beyond the one reading of step 24 (20 of 22 general questions holding a stored answer, against 18);
- questions close to a fact's relation. They cannot leak less than they did;
- the grafted model, which doesn't have the write at all.

**The gate** (`scripts/check_default_write.py`, on Qwen3.5-0.8B; `research/results/slots/32_default_write/gate.json`). All passed:
- a new table has the setting, and its calibration's R equals R measured alone;
- one fact added (36 seconds). "Who is the CEO of Vorantis Biotech?" gives "Ingrid Solberg is the CEO of Vorantis Biotech. …", and the plain form gives " Ingrid Solberg";
- a company one letter off gets the unedited reply, and another question about the company doesn't get the answer;
- detached, the replies are the unedited model's again;
- saved and loaded, the table keeps the setting and R, measures nothing again, and gives the same replies;
- three older tables (bench40h, bench40e's API table, the judged table of step 23) load with None;
- with the v1 write, the same table opens on the same questions. The setting changes the write, not the keys.
- **Reported, not graded:** "who runs VORANTIS BIOTECH?" doesn't open the fact, under either write. That is the detector not knowing a wording, the limit steps 19 and 20 are about.
- **My first run of the gate failed on that question,** because I had graded it. Whether a question opens is not what this gate is about, so I moved it to the reported line and said so here.

**Also checked:** the judged table of step 23, asked again with today's code and its saved setting, gives 90 of 90 replies identical to the stored ones. The 292 tests pass.

**Version:** 1.1.0. `CHANGELOG.md` at the repository's root starts with this.

**Afterwards, the same day** (nothing here is a bench result):
- **The animation was recorded again** with the write at the last layer (`scripts/trace_graft.py`, the 180 facts of bench40b to bench40e, R = 15.752). The three scenes open and stay shut as before, with the same detector readings. In scene 1 the same seven answer tokens are written, and the reply after the name differs: "is the founder of **Nimbresco Software" where the push at layer 20 gave "is the founder of Nimbresco Software." Scenes 2 and 3 are unchanged, token for token.
- **The charts:** the journey chart has the steps since v1, with the two that were not blind drawn pale and labelled so, and `last-layer.png` shows step 24's four pairs. Both illustrations now show the read at layer 20 and the write at layer 23.
- **The command line has the settings:** `--last-layer`, `--prompt-detector`, `--system` and `--fast` on `graft add` and `graft add-file`, `--last-layer` on `graft ask`, and `graft set` for a saved table. 294 tests.
- **A user guide and four examples,** each run on the real model. One of them showed something the docs hadn't said plainly: declaring the alias "Vorantis" made "Vorantis Biotics" name the company, and the stored chief executive answered for it. An alias matches wherever it stands as a whole word. `add_exclusion` for the longer name fixed it, and the guide now shows both.
- **The first question on a new table** took about 80 seconds more than later ones, for the detectors the relations share. That is not new with 1.1; the docs now say it.

---

## Step 25: the layer is measured, not hard-coded (written 2026-11-23, before the code and before any run)

**Why now.** Until today a table had one layer, 20, for everything: it read the question there and pushed the answer there. The 20 came from one screen of layers 18 to 22 on this one model (step 10), and for any other model the package takes the same depth, 5/6, with a warning. That is a constant picked once, and I should not ship it as a default for other people's models.

**What changed underneath.** Since version 1.1 a new table writes the answer at the model's last layer. Nothing is written at the table's layer any more, so `layer` now means one thing only: where the question is read. Step 10 moved the read and the push together and found facts lost on both sides of 20. That result binds the push, not the read. Step 17 looked at the read alone, at 42 sites, and put layers 16 to 20 within a few facts of each other (82 to 84 of 112 on benchmix1), with layer 16 two facts ahead on recall and 9 same-subject questions newly opened.

**My prediction, before any run:** on Qwen3.5-0.8B the profile is flat over several layers around 16 to 20, and the facts passing by real generation do not move by more than 2 either way. If that is what comes out, the gain of this step is not a higher number. It is that no layer is hard-coded, and that the package has a measured choice to make on a model it has never seen. One model can't show the second.

### What is built

**Stage 1: one read layer per table, picked by a profile** (`cramschool.graft.layers`, `Graft(model, tokenizer, layer="auto")`).
- **Probe facts:** for each relation of the table's wordings, 3 invented companies with invented answers. No bench fact, no bench wording.
- **Held out inside the package's own wordings:** each relation's wordings are split by position. The even ones teach the probe's detectors, exactly as the build does (`detector_data`, the same negatives, the readings inside the taught answer, the model's replies, `lda`, `fact_detectors`). The odd ones are never fitted.
- **Every layer in the same passes.** One forward pass reads the MLP input of every layer. The answer vectors are skipped: the default write doesn't use them.
- **Two measures per layer,** averaged over the probe facts:
  - held-out recall: the share of the relation's held-out wordings about the probe's company (plain, chat and Q/A form) on which the fact's detectors open;
  - held-out leak: the share of held-out questions about the same company that are not the fact's relation (the other relations' held-out wordings and the held-out generic questions) on which they open.
- **Also recorded:** the calibration's spread per layer, and how many forms get no detector.

**The pick, fixed here:**
1. A layer is admissible if its held-out leak is within 0.02 of the smallest leak of any layer.
2. Among admissible layers the best is the one with the highest held-out recall.
3. A layer is tied with the best if its recall is lower by no more than twice the standard error of the paired difference over the probe facts.
4. The pick is the middle layer of the tied set (the lower of two middles). With 3 or more tied layers the profile is called flat, and I say so.

**Scope.** `layer="auto"` selects only when the table writes at the last layer (`last_layer="all"`). With `None` or `"later"` something is still pushed at the table's layer, step 10 binds, and the old rule stays. `layer=None` keeps the old rule for now, so that nothing published changes; whether `"auto"` becomes the default is decided by the rule below and then by me.

### The rule

Real generation decides, not the profile's own numbers. On bench40h's 40 facts, built with the fast build and asked with `last_layer="all"`, scored by `scripts/score_table.py`:
- **A:** the table at layer 20. It is already scored: 32 of 40 (step 24).
- **B:** the same facts built at the profile's pick P, same options.
- **C:** the same facts built at the layer the profile ranks lowest among layers 12 to 22. This checks that the profile's order means something.

| Outcome | What follows |
|---|---|
| P is 20 | The profile reproduces the hand-picked layer. B is A. `layer="auto"` ships as an option if C passes fewer facts than A |
| B passes at least A − 2 facts, with at most 1 more fact failing a leak check, and C passes fewer than B | `layer="auto"` ships as an option |
| B passes at least A + 2 | The same, and I propose it as the default |
| B passes A − 3 or fewer, or C passes at least as many as B | **Kill.** The criterion doesn't predict generation. The profile is not shipped, and the default stays as it is with the reason written here |

bench40h is not blind: I have read its replies at steps 20 to 24. Nothing here is a blind result.

**Stage 2, only if the profile earns it: a layer per fact.** From the same profile, each probe fact's own best layer is picked on half of its held-out wordings and scored on the other half. If that beats the table's pick by 0.03 of held-out recall or more, at no more leak, I build the layer that reads each fact at its own layer (the open decision then moves to the hook at the last layer, where every reading is in). If not, it is not built, and the docs say that one layer per table is enough on this model.

### What this step does not cover

- **The grafted model.** Its neurons sit at the table's layer, and step 10's loss away from 20 is about exactly that push. Which layers' neurons to reuse, a whole layer or part of one, over one layer or several, is its own step. `export_grafted` will warn when the table's layer was picked for reading.
- **Behaviours.** A behaviour needs another write than a stored answer, and its layer and strength will need their own profile. That is the step after this.
- **Another model, by a bench.** I can run the profile on Qwen3.5-4B and report what it picks against the 5/6 rule. That is a reading, not a validation.

**Two details of the rule, fixed before the first run on the real model** (the code is written; I have only run it on the tests' tiny random model, where no probe fact gets a detector at any layer):
- **"Ranked lowest" for table C** means the smallest held-out recall less held-out leak, among layers 12 to 22. If two layers tie, the later one.
- **A profile with no detector at any layer picks nothing.** `layer="auto"` then warns and uses the earlier rule. That is what happens on the tiny random model.
- **Tables A, B and C are built by the same script,** `scripts/layer_screen.py`, which adds a bench's facts through the API with the fast build at a given layer. Before B and C count, the table it builds at layer 20 must give the same replies as the saved table of step 20 under `--last-layer all` (32 of 40). If it doesn't, A is the newly built table and I say so.

**The profile on Qwen3.5-0.8B, before any table is built at its pick** (`scripts/layer_profile.py`, 165 seconds, `research/results/slots/33_layer_profile/profile_0.8b.json`):

| Layer | Held-out recall | Held-out leak | |
|---|---|---|---|
| 12 | 0.630 | 0.028 | ranked lowest among 12 to 22 |
| 16 | 0.780 | 0.022 | |
| 19 | 0.874 | 0.018 | |
| 20 | 0.887 | 0.014 | the hard-coded layer |
| 21 | 0.922 | 0.016 | tied |
| **22** | **0.924** | 0.013 | **the pick** |
| 23 | 0.922 | 0.011 | tied, the last layer |

- **The pick is layer 22, and the profile is flat over layers 21 to 23.** Layer 20 is not in the tied set: its recall is 0.037 lower.
- **My prediction was partly wrong.** I predicted a flat stretch around 16 to 20. The flat stretch is 21 to 23, later than any layer I had ever built a table at, and layers 16 to 18 read worse than 20 (0.74 to 0.78). Whether generation moves is the rule's to say, below.
- **Recall climbs with depth almost all the way:** 0.05 at layer 0, 0.63 at layer 12, 0.80 at layer 15, 0.92 from layer 21. Leak falls from 0.05 around layers 7 to 9 to 0.011 at the last layer.
- **A layer per fact is not earned.** Picked on half the held-out wordings and scored on the other half, 6 of 12 probe facts take another layer than the table's, and held-out recall changes by −0.030, leak by −0.003. Stage 2 is not built: one layer per table is enough on this model.
- **Every layer gave every probe fact a detector from layer 3 on.**

**A deviation, before B and C are built.** I had named `scripts/layer_screen.py` as the builder. Table A (step 20's) was built by `scripts/editor_wordings_bench.py`, so B and C are built by that script, with a new `--layer` flag and nothing else changed. They then differ from A in the layer alone, and A stays the saved table: the check of a rebuilt layer-20 table isn't needed.

**What the profile means for the grafted model,** which is where the facts are in the model's own weights. A fact's neuron reads with its gate row and writes with its down column, both at its own layer. The last layer is where a write always wins (step 24), and the profile says the question reads as well there as anywhere. So a fact's neurons can sit at the last layer and do both. That is step 26.

---

## Step 26: the fact's neurons at the last layer, so that the weights hold what the table does (written 2026-11-23, before the code and before any run)

**Why.** The inserted layer changes no weight. That is its design, and it is not what this project is for: the facts are meant to be in the model. The grafted model puts them there, and today it is behind its own table. Step 24's write at the last layer lifted the table from 24 to 33 of 40 on bench40f and from 23 to 32 on bench40h, and the grafted model has none of it: its neurons sit at layer 20 and push from there.

**The idea.** A grafted fact's neuron reads with its gate row and writes with its down column, both at its own layer. Two measurements now point at the same layer:
- **The write wins at the last layer** (step 24). Nothing reads the last MLP's output but the final norm and the output matrix at the same position.
- **The question reads as well at the last layer as anywhere** (step 25's profile): held-out recall 0.922 and leak 0.011 at layer 23, against 0.887 and 0.014 at layer 20.

So the fact's neurons go to the last layer. The gate row is the fact's detector fitted there, and the down column is the token's own row of the output matrix at length 2R, the same vector the inserted layer writes.

**What is built:**
- **`compile_layer(last=...)`:** every down column of a fact's neurons is the sized row of its token, not the vector measured at the layer. Each form's detector gets a pair for each spelling of the answer, and the name switch lights the pair whose spelling fits the text before, as the inserted layer picks it.
- **`Graft.export_grafted`** compiles this way when the table writes at the last layer and its own layer is the last. Otherwise it compiles as before and says how to get the other.
- **The switch still turns a fact on when its subject is named.** This step does not remove it. It makes what is in the weights as good as the table.

**The rule.** On bench40f's 40 facts, with every check of the bench and the damage bar on firmly known facts (`scripts/grafted_bench.py`):
- **T23:** the 40 facts built at layer 23 by the script that built bench40f's table, asked as an inserted layer with `last_layer="all"`.
- **G23:** T23 compiled into layer 23's own neurons with the new compile, asked with the name switch.
- **The references, already measured:** the grafted model at layer 20 passes 23, its table 24, and that table with the write at the last layer 33.

| Outcome | What follows |
|---|---|
| G23 passes at least 28 (5 more than the grafted model today), at least T23 − 2, and no firmly known fact is lost | **Confirmed.** The last-layer compile ships as what `export_grafted` does for such a table |
| A firmly known fact is lost | **Kill as built.** The last MLP's neurons can't be reused at this count. I report the count at which the bar holds, if any, and the compile is not shipped |
| G23 passes fewer than 26 | **Kill.** The compile is not shipped |
| Between | Reported as it is, not shipped as the default export |

**My prediction:** G23 lands at 30 to 33, about where the table is. The risk is damage: an earlier check found that 512 neurons can be freed at layer 20 with no firmly known fact lost, and nobody has measured the last layer, whose MLP writes straight into the output. This compile needs about 8 neurons per fact for the first token and one per later token, about 450 for 40 facts.

**bench40f is not blind:** I have read its replies at earlier steps.

**What it does not cover:** the subject match in the weights. Steps 5 and 13 tried it and failed their rules. With the reader of step 25 I can measure, at every layer and with look-alike names among the negatives, whether the question's last token carries the subject. That measurement comes after this step, before any third build.

---

## Step 27: the subject at the question's last token, layer by layer (written 2026-11-23, before it ran)

**Why.** The grafted model holds a fact's detector and its answer in the model's own neurons, and one thing stays outside the weights: knowing which subject is named. A small switch does that at run time, and without it the facts are dormant. That is the hook I keep having to explain. Two attempts to remove it failed their rules:
- **Step 5,** a subject reading at layer 20 with no hook: on CounterFact's own phrasings the reading missed reworded prompts and let neighbours in, and 48 of 133 known facts were lost.
- **Step 13,** the name read where it stands, by one head's window: 40 of 90 companies would have passed the bench's look-alike and recall bars.

**What neither measured:** the question's last token at every layer, with look-alike names among the negatives of the fit. That is where a fact's neuron acts, and step 25 showed that the question itself reads far better late (0.92 at layers 21 to 23) than at the layers I had been using. If the subject is there too, a fact neuron can read both keys itself: its gate the question, its up row the subject.

**This is a measurement, not a build** (`scripts/subject_read.py`):
- **12 invented companies.** For each, a linear subject reading is fitted at every layer with the build's own rule (`lda`, the threshold at the 99th percentile of its negatives).
- **Positives:** questions about the company, on half of the four relations' wordings and half of the generic questions, in plain, chat and Q/A form.
- **Negatives:** the same questions about three look-alike names of it (another industry word, one letter changed, another first word) and about the 11 other companies.
- **Scored on the other half of the wordings,** never fitted: about the company (also in lower and upper case), about three look-alike names it has never seen, made the same three ways, and about companies it has never seen.
- **A company passes at a layer** when it meets the bench's bars there: held-out recall at least 0.75, and at most 1 in 12 questions about a look-alike opened.

**The rule for what follows:**

| Outcome at the best layer | What follows |
|---|---|
| At least 10 of the 12 companies pass | A third build is worth it: a fact neuron whose up row is the subject reading, no switch. It gets its own step, rule and bench |
| 6 to 9 pass | Not enough to remove the switch. I report which kind of look-alike gets through and stop there |
| 5 or fewer pass | **Kill.** The subject can't be read linearly at the question's end on this model. The switch stays, and the docs say so with these numbers |

**My prediction:** recall will be high late in the model, and the one-letter look-alike will get through too often, as it did for the name read in step 13 (31% of prompts). I expect 4 to 8 of 12. If I am wrong on the good side, this is the step that removes the hook.

**What it can't show even if it passes:** that the reading survives inside a fact neuron (the up row is linear, with no threshold of its own), on real subjects, or with hundreds of subjects stored.

---

## Result of step 25: read at the profile's layer, all 40 facts of bench40h pass, against 32 at layer 20

Raw numbers: `research/results/slots/33_layer_profile/` (the profile, `profile_0.8b.json`; the two tables' rows as built and with the answer written at the last layer, `bench40h_layer22_*.jsonl` and `bench40h_layer12_*.jsonl`; their logs; `chain.sh`).

| bench40h, 40 facts, the fast build, `last_layer="all"` | A: layer 20 (hard-coded) | **B: layer 22 (the profile's pick)** | C: layer 12 (ranked lowest) |
|---|---|---|---|
| Facts passing every check | 32 | **40** | 12 |
| Recall replies holding the whole answer, of 480 | 410 | 422 | 323 |
| Mean recall | 0.854 | 0.879 | 0.673 |
| Facts failing a leak, garbling or damage check | 0 | 0 | 2 |
| Same-company leak, summed over the facts | 0.5 | 0.5 | 0.5 |
| The profile's held-out recall; leak | 0.887; 0.014 | 0.924; 0.013 | 0.630; 0.028 |

**By the rule: B passes A + 8, with no more leak failures, and C passes fewer than B.** `layer="auto"` ships as an option, and the rule has me propose it as the default.

**What happened:**
- **The 8 facts that failed at layer 20 all failed on recall, and all 8 pass at layer 22.** No fact stops passing.
- **It is a gain of margins, not of volume:** 16 recall replies gain the whole answer and 4 lose it (one each on f11, f16, f20 and f36), 12 net of 480. The 8 facts sat just under the bar.
- **Leaks don't move.** 3 of the 320 same-company replies differ from layer 20's, and none of the 960 replies to look-alike and unrelated companies.
- **The profile's order held in generation.** Its lowest layer of the span passes 12, with 97 whole answers lost and two facts answering with another fact's answer.
- **This is what the end-of-prompt detector gave at layer 20.** There, the write at the last layer alone passed 32 and the detector added 8 more (step 24). At layer 22 the strict detectors alone open those questions.

**My prediction was wrong in the useful direction.** I expected no movement beyond 2 facts, and a flat stretch around 16 to 20. Step 17's estimates, which I leaned on, compared readings that were all fitted the build's way at sites up to layer 20. Nothing before today had read a question at layers 21 to 23, because a table's layer was also where it pushed, and a push that late loses (below).

**The scope rule was right, and the same tables show it.** Built at layer 22 and asked with the push at the table's own layer (`last_layer=None`), 16 of 40 pass. At layer 12, 7. A late layer is good for reading and bad for pushing, which is step 10's finding. `layer="auto"` therefore selects only for a table that writes at the last layer.

**What this is not:**
- **Not blind.** I have read bench40h's replies at steps 20 to 24. What keeps it honest is that the layer was picked from invented probe facts by a rule written before, with no bench fact or wording, and that the comparison changes the layer alone.
- **One bench, one model.** Step 26 builds bench40f at layer 23 for another reason, and that table asked with the write at the last layer is a second reading against 33 at layer 20.
- **Not the grafted model.** These are tables beside the model. Step 26 is about getting the same into the weights.

**In the package:** `Graft(model, tokenizer, layer="auto")`, `cramschool graft add --layer auto`, `cramschool graft layers` to see a model's profile, `graft.layer_profile` saved with the table. About 3 minutes on the 0.8B. `layer=None` is unchanged for now; I decide on the default after step 26's table.

**A layer per fact is not built.** On the probe facts it loses 0.030 of held-out recall against one layer per table.

---

## Result of step 27: the subject can't be read at the question's end. Kill; the name switch stays

Raw numbers: `research/results/slots/35_subject_read/` (`subject_read_0.8b.json`, with every company's numbers at every layer; the log).

**No company passes at any layer: 0 of 12.** The rule's kill line was 5. I had predicted 4 to 8, so it is worse than I thought.

| Layer | Held-out recall | In lower or upper case | Look-alike opened: another industry word | One letter changed | Another first word | A company never seen | Companies passing |
|---|---|---|---|---|---|---|---|
| 0 to 13 | no reading | | | | | | 0 |
| 16 | 0.476 | 0.318 | 0.182 | 0.408 | 0.055 | 0.012 | 0 |
| 18 | 0.665 | 0.453 | 0.203 | 0.547 | 0.093 | 0.025 | 0 |
| 20 | 0.698 | 0.444 | 0.366 | 0.613 | 0.049 | 0.003 | 0 |
| 22 | 0.811 | 0.568 | 0.442 | 0.714 | 0.073 | 0.012 | 0 |
| 23 | 0.832 | 0.598 | 0.518 | 0.735 | 0.073 | 0.008 | 0 |

**What the reading can and can't tell:**
- **It tells a different company apart.** A company it never saw opens it on about 1% of questions, and one that shares the industry word on 7%.
- **It can't tell a near name.** One letter changed opens it on 73% of questions at the last layer, and the same first word with another industry on 52%. These were look-alikes it had never seen; three others of the same kinds were among its negatives.
- **Recall and near-name leak rise together with depth.** There is no layer where the first is high and the second low. Before layer 14 the question's last token holds no linear reading of the subject at all.
- **Capitalisation costs a third:** 0.60 in lower or upper case against 0.83 as written. The name switch ignores case.

**Why this ends it for a linear reading.** By the end of a question the model holds roughly which entity it was asked about, not how the name was spelled. A fact gated on that would answer for "Nimbrosco Software" most of the time. And the 1% on unrelated companies is not small at scale: with 100 subjects stored, a question about a company nobody stored would open about one of them.

**Three attempts now, each at a different place:**

| Step | Where the subject was read | Result |
|---|---|---|
| 5 | Layer 20, the question's last token, real subjects, no look-alikes in the fit | Reworded prompts missed, neighbours let in, 48 of 133 known facts lost |
| 13 | The name where it stands, one head's window, layers 16 to 20 | 40 of 90 companies would pass; one-letter look-alikes through on 31% of prompts |
| 27 | The question's last token, every layer, look-alikes in the fit | 0 of 12 companies pass at any layer |

**What stays true:** the grafted model holds each fact's question detector and its answer in the model's own neurons, and it needs the name switch to say which subject is named. I have no method that puts that match into the weights from forward passes on this model, and I am not going to describe the grafted model as if I had. What an exact name match needs is the spelling, and the model's state at the place a fact acts doesn't keep it.

**Not tried, and I am not proposing them as next steps without a reason to expect better:** a reading on a bigger model; a reading that is not linear; a match made at the name's own tokens and carried forward by an edited attention head (step 13's design, which stopped at 40 of 90).

---

## Step 26, first run: in the weights the facts answer as the table does, and the damage bar is not met

Raw numbers: `research/results/slots/34_last_layer_neurons/` (`build_layer23.log`, `t23_all.jsonl`, `g23/rows.jsonl`, `g23.log`, `chain.sh`).

| bench40f, 40 facts | Passes its recall and leak checks | Whole answers, of 480 | Damage |
|---|---|---|---|
| The grafted model at layer 20 (step 7; the reference) | 23 | | firm bar met; drift 0.0027; 377 neurons |
| Its table at layer 20, pushed at layer 20 | 24 | | none |
| That table with the write at the last layer (step 24) | 33 | 395 | none |
| **T23:** the table built at layer 23, written at the last layer | 32 | 413 | none |
| **G23: T23 in layer 23's own neurons** | **32** | 411 | **firm bar not met; drift 0.0213; 537 neurons** |

**What worked.** The grafted model answers as its table does, fact by fact: 38 of 40 facts have the same recall to the second decimal, and the other two are one reply apart. On the recall and leak checks 32 pass, against 23 for the grafted model at layer 20. The detector and the write that always wins are in the model's own neurons and behave there as they do beside it.

**What didn't.** Reusing 537 of the last layer's neurons costs more than the bar allows:
- **No firmly known fact is lost** (0 of 41).
- **Their mean margin falls by 0.145,** and the bar's limit is 0.10. The bar fails on that.
- **9 of 92 knife-edge facts are lost,** which is 9 of the 133 by the old count. The bench's own clause fails every fact at 7.
- **Drift is 0.0213,** against 0.0027 for the grafted model at layer 20 with 377 neurons.
- **The re-fit left 12.2% of the freed neurons' output unexplained.**

**By the rule: not confirmed, and not shipped as built.** The rule's confirming line named one thing, no firmly known fact lost, and none was. But the bar that line stood for also limits the shift of the margins, and that is exceeded. I hold the result to the whole bar. This is the rule's second row in substance: the last layer's neurons can't be reused at this count, and I owe the count at which the bar holds.

**On the read layer, a second bench, and it says nothing moves.** T23 passes 32 of bench40f's facts, against 33 for the table at layer 20 with the same write: one fact gained, two lost, 18 more whole answers. Step 25 gained 8 facts on bench40h at layer 22. One bench up and one flat, at two different late layers. `layer="auto"` stays an option, and `layer=None` stays as it is.

**A by-product:** T23 asked with the push at its own layer, as the build script scores it, also passes 32. At the last layer the earlier push is itself a write at the last layer.

**What comes next, inside this step:**
- **The count at which the bar holds.** `scripts/damage_profile.py` frees N neurons at layers 20 to 23, with no fact written, and judges each by the same bar.
- **Fewer neurons at the last layer.** Two things in the compile cost neurons for no reason there:
  - the second neuron of each first-token pair. It exists to cap the push, and at the last layer a larger write only raises the same token further. Dropping it saves 160;
  - the 217 neurons for later tokens. Step 24 found that the first token carried 14 of its 15 gained facts. Later tokens can stay at layer 20, where 377 reused neurons met the bar.

---

## Step 26, the damage profile, and one revision written before it runs: two layers, each with the part it can afford

**How many neurons each late layer can give up** (`scripts/damage_profile.py`; N neurons freed as the compile frees them, no fact written; `research/results/slots/34_last_layer_neurons/damage_profile.json`). The number is the mean change of the firmly known facts' margins, and the bar's limit is 0.10. No firmly known fact was lost in any of the 16.

| Neurons freed, of 3,584 | Layer 20 | Layer 21 | Layer 22 | Layer 23, the last |
|---|---|---|---|---|
| 128 | −0.014 | −0.002 | −0.029 | −0.021 |
| 256 | −0.009 | −0.005 | −0.043 | −0.050 |
| 384 | −0.058 | −0.046 | −0.047 | **−0.111, fails** |
| 537 | −0.076 | −0.061 | −0.046 | **−0.145, fails** |
| Drift at 256; at 537 | 0.0019; 0.0040 | 0.0054; 0.0072 | 0.0078; 0.0100 | 0.0093; 0.0213 |

- **The last layer can spare about 256 neurons** by the bar, and not 384. That is the count the rule asked me to report.
- **Layers 20 to 22 pass at every count up to 537.** Layer 21 is the cheapest at 128 and 256, layer 20 has the least drift throughout.
- **So the last layer is the expensive one,** and it is the only place a write always wins. It should hold what must be there and nothing else.

**What must be at the last layer, and what needn't.** Step 24 measured it on the table: finishing a begun answer gave 1 fact over two benches, and making the first token certain gave 14. The first token is where the model's own first word competes. Later tokens were already followed well from layer 20: the grafted model at layer 20 passed 23 against its table's 24.

**The revision, one, fixed here before it is built or run:**
- **At the last layer: each fact's first-token neurons only, one neuron each, not a capped pair.** Its gate is the fact's detector fitted at the last layer, and its output column the token's sized row. At the last layer a larger write only raises the same token further, so the cap is not needed. For 40 facts that is 160 neurons (two forms, two spellings).
- **At layer 20: the later tokens' neurons,** as the grafted model has them today: always open, each writing the answer vector measured at layer 20, switched on while the answer is being followed. 217 for bench40f.
- **One switch for both layers.** It decides once per call which neurons are on, as it does now, and sets them at whichever layer they sit.
- **For this run** the later tokens' vectors come from bench40f's saved table at layer 20 and the detectors from T23: the same 40 facts, built by the same script at the two layers.

**The rule, the same bars as before:**

| Outcome | What follows |
|---|---|
| It passes at least 28 on the recall and leak checks, at least 30 (T23 − 2), and the whole firm bar is met with both layers edited: no firmly known fact lost and the margins' mean change within 0.10 | **Confirmed.** The two-layer compile is what `export_grafted` does for a table that writes at the last layer |
| The firm bar is not met | **Kill.** The step ends with the write shown to work in the weights and no compile that meets the bar |
| Fewer than 28 | **Kill** |
| 28 or 29 with the bar met | Reported, not shipped as the default export |

**My prediction:** 30 to 32 facts, and a margin change of about −0.04 if the two layers' costs add (−0.03 for 160 at the last layer, −0.01 for 217 at layer 20). I don't know that they add.

**This is the only revision.** If it fails, I don't try a third layout on bench40f.

---

## Result of step 26: confirmed. In the model's own weights, 31 of 40 facts pass against 23, and the damage bar is met

Raw numbers: `research/results/slots/34_last_layer_neurons/` (`g23_two_layers/rows.jsonl`, `g23_two_layers.log`; the first run's `g23/`; `damage_profile.json`).

| bench40f, 40 facts | Passes its recall and leak checks | Whole answers, of 480 | Firm bar | Margin change (limit 0.10) | Knife-edge lost, of 92 | Drift | Neurons |
|---|---|---|---|---|---|---|---|
| The grafted model at layer 20 (step 7) | 23 | | met | | 4 | 0.0027 | 377 at layer 20 |
| T23, the table itself | 32 | 413 | | | | 0 | none |
| First run: everything at the last layer | 32 | 411 | **not met** | −0.145 | 9 | 0.0213 | 537 at layer 23 |
| **The revision: first tokens at layer 23, later tokens at layer 20** | **31** | 407 | **met** | −0.032 | 7 | 0.0097 | 160 at layer 23, 217 at layer 20 |

**By the rule: confirmed.** The revision passes 31 (the rule asked for 28, and for T23 − 2 = 30), no firmly known fact is lost (0 of 41), and the margins' mean change is −0.032, inside the limit. My prediction was 30 to 32 and about −0.04.

**What it means.** The fact's question detector and the write that always wins are in the model's own neurons, in a checkpoint of the same size, and the model answers 8 more facts of 40 than the grafted model did this morning. It is one fact behind its table.

**What it costs, against the grafted model at layer 20:**
- **7 knife-edge facts flip, against 4.** They are answers the unedited model gives with a lead of less than one logit. By the bench's old count that is 7 of 133, which is that clause's own failing line. The firm bar is the one this project judges a grafted model by, and it is met; I am not hiding the other number.
- **Drift is 0.0097, against 0.0027.** Most of it is the last layer: 160 neurons freed there alone drift about 0.007 to 0.009 (the profile above).
- **Two layers are edited, not one,** and the switch sets neurons at both.

**The layout was chosen by measurement, in this order:** the write wins only at the last layer (step 24); the question reads as well there as anywhere (step 25's profile); the last layer can spare about 256 neurons and layers 20 to 22 far more (the damage profile); and the first token is what needs the winning write (step 24). So the last layer holds one neuron per fact, form and spelling, and nothing else.

**What it doesn't show:**
- **A blind result,** or a second bench. bench40f had been read.
- **Capacity.** At four neurons a fact, 256 at the last layer is about 64 facts by the bar. More text for the re-fit raised layer 20's capacity before (step 6); I have not measured that at the last layer.
- **A model without the switch.** Step 27 stands: the name match stays outside the weights.
- **Another model.**

**Not yet in the package's export.** `compile_layer(last=..., kinds=...)`, `join_layers` and a switch that sets neurons at two layers are in the code, with tests. `export_grafted` still writes one layer. Wiring it in is the next thing I do, and it needs each table built at the last layer plus its later tokens' vectors measured at layer 20.

**A second bench, outside the rule, written before it runs.** The rule was bench40f's and it is settled. bench40h gives a second reading of two things at once:
- **The table at the last layer.** Step 25 built bench40h at layers 22 and 12, not 23. Built at 23 by the same script and asked with the write at the last layer, it tells me whether the last layer reads as well as the pick on the bench where the pick gained 8 facts. The profile ties them (0.922 against 0.924), so I expect 38 to 40.
- **The two-layer grafted model on a second set of facts:** that table's first tokens at layer 23, and the later tokens at layer 20 from the table step 20 saved. I expect it within 2 facts of its table, with the firm bar met, as on bench40f.

Neither number changes step 26's verdict. If the table at layer 23 falls well short of 40, the last layer is not as good a place to read as the profile says, and I say so before making it anyone's default.

---

## Step 26, the second bench and the export's gate; then a budget for the last layer, written before it runs

**The second bench, bench40h** (`chain_bench40h.sh`; rows and logs beside it). Outside the rule, as written above.

| bench40h, 40 facts | Passes its recall and leak checks | Whole answers, of 480 | Firm bar | Margin change | Drift |
|---|---|---|---|---|---|
| The table at layer 20, written at the last layer (step 24) | 32 | 410 | | | |
| The table at layer 22 (step 25's pick) | 40 | 422 | | | |
| **The table at layer 23, the last** | **38** | 421 | | | |
| **The two-layer grafted model of it** | **34** | 401 | met | −0.009 | 0.0097 |

- **The last layer reads nearly as well as the pick here:** 38 against 40, as the profile's tie said, and 6 more than layer 20. Two facts pass only at layer 22.
- **The grafted model is 4 behind its table.** I expected 2 at most. No firmly known fact is lost, 7 knife-edge facts flip, the same count as on bench40f.

**The gate of the export failed, on one reply** (`scripts/check_last_layer_export.py`; `export_gate.json`). Three facts through the package's own path: a table at `layer="last"`, `export_grafted`, the checkpoint loaded from disk with its switch.
- **What passed:** the files keep their size (1,746,942,600 bytes); 12 neurons at layer 23 and 11 at layer 20; the firm bar (0 lost, −0.023); a company one letter off and another question are left alone; with the switch detached the facts are dormant.
- **What failed:** "Who is the CEO of Vorantis Biotech?" gives "In 2024, **Vorantis Biotech** was acquired by…". The table gives "Ingrid Solberg is the CEO…". The answer's first token, "In", was written at the last layer and came out. Its second token, pushed from layer 20, lost to the model's own next word.
- **This is the 1 fact of bench40f and the 4 of bench40h.** In the confirmed layout the later tokens are still pushes at layer 20, and a push there can lose. Step 24 found that for the table; it holds for the neurons.

**What the measurements already allow.** The last layer can spare about 256 neurons (the damage profile). A fact takes 4 there for its first token and about 5 more for its later tokens. So:
- **A table of up to about 27 facts fits entirely at the last layer.** That is the first run's layout, which answered exactly as its table did (32 against 32); what failed there was the count, 537.
- **A bigger table can put the later tokens that fit there too,** the second tokens first, since that is where an answer is lost right after it begins, and the rest at layer 20.

**So the export gets a budget, fixed here:**
- **The last layer's budget is 1/14 of its neurons:** 256 of the 0.8B's 3,584, the largest count the profile passed.
- **First tokens always go to the last layer.** Later tokens fill what is left of the budget, in the order of their place in the answer (every fact's second token, then every third, and so on). The rest go to layer 20 as in the confirmed layout.
- **If everything fits, only the last layer is edited.**

**The rule for adopting it:** on bench40f and bench40h, the budgeted layout must pass at least as many facts as the confirmed layout (31 and 34) with the whole firm bar met on both. If it does, it is what `export_grafted` writes. If the bar fails on either, the export keeps the confirmed layout for tables over the budget and uses the last layer alone only for tables that fit, and I say that the gate's reply stays wrong for bigger tables.

**My prediction:** 32 and 36 or more, with the margin change between −0.05 and −0.08: the last layer alone costs −0.050 at 256.
