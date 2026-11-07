# Fact bench: plan, written before any run (2026-11-06)

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
- **Choice made:** proceed to build despite the failed pre-written rule. That is my choice, recorded here, not something the data decided.

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

**Deviation.** The rule says to run all 40 facts now. I stop instead, because recall fell on all 4 facts (mean 0.21 against 0.48), which the decision rule would reject. There are two causes:

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
