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
