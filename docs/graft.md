# GRAFT: Gated Relation-Aware Fact Tables

GRAFT stores facts of the form *(subject, relation, answer)* in one inserted layer beside a frozen language model. The layer writes only when two keys turn:
- **Key 1:** the subject is named, matched as an exact name.
- **Key 2:** the question asks for a stored relation of that subject, read by a linear detector from the model's hidden state.

It then adds precomputed answer-token vectors to one MLP output, one token at a time. Every component is computed in closed form from forward passes of the unedited model. No weight changes, no gradient is taken, and removing the layer is an exact undo.

This page states the v1 method as it ran on the benchmarks. [How it works](how-it-works.md) explains the same thing in pictures, and the [lab notebook](lab-notebook.md) records why each choice was made, before each run.

## Setting

- **Model:** a causal language model with $L$ layers, used frozen. v1 was validated on Qwen3.5-0.8B: $L = 24$, hidden size $d = 1024$.
- **The working layer:** GRAFT acts at one layer $\ell$. For Qwen3.5-0.8B, $\ell = 20$, the first layer at which well-known answers already appear in the model's own predictions.
- **What it reads and writes:** $x_t \in \mathbb{R}^d$ is the input of layer $\ell$'s MLP at position $t$, and $o_t$ is that MLP's output. GRAFT reads $x_t$ and adds to $o_t$.
- **A fact:** $f = (s, r, a)$.
  - $s$ is a company name and $r \in$ {owner, CEO, founder, head office}.
  - $a$ is the answer string, tokenized in two spellings: with a leading space ( `" Tharindu Ekanayake"` ) and without one ( `"Tharindu Ekanayake"` ). Each spelling has tokens $a_1, \dots, a_m$.

## 1. Calibration, once per model

On ordinary text (355 paragraphs of WikiText-103), with $\mu$ and $\Sigma$ the mean and covariance of $x$ at layer $\ell$:

$$c = \left(\Sigma + 10^{-3}\,\tfrac{\operatorname{tr}\Sigma}{d}\,I\right)^{-1}\mu, \qquad \bar c = \mu^\top c, \qquad u(x) = \frac{x^\top c}{\bar c}.$$

- **What $u$ does:** $u(x)$ is close to 1 on ordinary tokens; on held-out text its relative spread is 1.7%. It removes the overall scale of $x$, so that readings compare across positions.
- **Reading:** the reading of a direction $w$ is
$$\rho_w(x) = \frac{w^\top x}{u(x)}.$$

## 2. Key 1: the name index

- **The index:** all stored subject names, normalized by lower-casing and collapsing whitespace, matched against the decoded text so far.
- **A match** is a stored name with no letter or digit just before it, and none just after it. Names with digits or math operators have more boundaries:
  - a name that ends in a digit is continued by a decimal or thousands separator and a digit ("Form 1040.5", "1,040,000"), and one that begins with a digit is not found after a digit and a separator ("0.5");
  - a name that holds a math operator (`^ * / + = < > × ÷`) is continued by a binary operator after it or before it, spaces allowed ("x^2+y^2").
- **An open match:** a match at the very end of the text so far stays open until the following text decides it. It counts meanwhile, and is withdrawn if the text continues the name.
- **State:** the index keeps its state across forward calls, so it follows generation token by token.
- **The subject:** the subject of a sequence is the stored name matched last. If no stored name has been matched, the layer does nothing.

**Declared aliases.** `Graft.add_alias(subject, *aliases)` adds other names for a stored subject to the index. They are matched exactly as the name is: as a whole word, in any capitalisation, wherever they stand. Nothing is rebuilt, since the question detectors don't read the name.

**Declared exclusions.** `Graft.add_exclusion(subject, *names)` declares longer names that contain a stored subject's name or alias and name something else: "Vanderbilt University Department of English" for "Vanderbilt University", "Form 1040-EZ" for "Form 1040". They are matched as names are, and where one stands the text names no stored subject. Hyphenated forms are left to such declarations, because "Windows 95-era" still means Windows 95. Nothing is rebuilt.

## 3. Key 2: question detectors

For each fact $f$ and form $\varphi \in$ {plain, chat}, one direction $w_{f,\varphi}$ is fitted. Readings are taken at the last token of each prompt.

| Set | Contents |
|---|---|
| **Positives** $X_+$ | The fact's training wordings of $r$ and the editor-drafted wordings of $r$, filled with $s$ and with 24 invented company names.<br>Plain prompts about $s$ are also asked behind fixed prefixes. Chat prompts are rendered with the chat template, and also asked in plain form as `Q: … A:`.<br>$s$ and 6 of the invented names are also asked in lower and upper case. |
| **Prompt negatives** $X_-^{p}$, in groups | Other questions about the company, and generic negative prompts, for every company.<br>The drafted wordings of the other three relations, about 6 invented companies and $s$. The first 8 are also asked with $s$ in lower and upper case.<br>24 plain and 24 chat generic company questions (size, products, money, history).<br>Real-world questions. Plain, chat and `Q: … A:` forms throughout. |
| **Inside-answer negatives** $X_-^{in}$ | Positions inside the answers of the teaching examples: every answer token after the first. |
| **Reply negatives** $X_-^{r}$ | Every position of the unedited model's 16-token greedy replies to the company's other training questions, 8 generic questions and 4 wordings of each other relation, about $s$ and 3 invented companies. |

No string used to build a fact may equal a test prompt of the bench being scored, compared after lower-casing and removing punctuation.

**Facts added outside a bench** (`Graft.add`, the command line, the editor tools, the MCP server) have no training wordings of their own. Their positives are the drafted wordings of $r$ alone. Their "other questions about the company" are the first and fourth drafted plain wordings and the first chat wording of each other relation, plus the 24 + 24 generic questions. No bench is known, so no guard applies. [Benchmarks](benchmarks.md#facts-added-through-the-api) measures this path on its own.

**The fit** is linear discriminant analysis with shrinkage. The negative groups are $g = 1, \dots, G$, with means $m_g$, and $m_+$ is the positive mean:

$$S = \Sigma + \frac{1}{N}\sum_{x}\big(x - m_{g(x)}\big)\big(x - m_{g(x)}\big)^\top, \qquad w = \Big(S + 0.1\,\tfrac{\operatorname{tr}S}{d}\,I\Big)^{-1}\big(m_+ - \tfrac{1}{G}\textstyle\sum_g m_g\big).$$

**The threshold and the score:**

$$\tau = \max\Big(Q_{0.99}\big\{\rho_w(x) : x \in X_-^{p}\big\},\ \max\big\{\rho_w(x) : x \in X_-^{in}\big\}\Big), \qquad \delta = \tfrac12\big(\operatorname{med}\rho_w(X_+) - \tau\big), \qquad z = \frac{\rho_w(x) - \tau}{\delta}.$$

- **When a form has no detector:** if the median positive reading does not exceed $\tau$, that form gets no detector.
- **Reply positions** shape the direction but not the threshold. Setting the threshold against them cost recall in the diagnostic, while the direction alone kept them below $\tau$.

## 4. Shared relation detectors and competition

**Shared detectors.** For each relation $r$ and form $\varphi$, a shared detector is fitted the same way, from drafted wordings only.
- Its positives are $r$'s wordings about invented companies, cased and in `Q: … A:` form.
- Its negatives are generic and negative prompts, the other relations' wordings and real-world questions.
- $\tau$ is the 99th percentile of the negatives.

**Competition.** At the last position, let $k$ be the named company, $F_k$ its stored facts and $R_k$ their relations.

$$z_f = \max_\varphi z_{f,\varphi}(x), \qquad f^\star = \arg\max_{f \in F_k} z_f .$$

The fact $f^\star$ fires if and only if

$$z_{f^\star} > 0 \quad\text{and}\quad \max_{r \notin R_k,\ \varphi} z^{\text{shared}}_{r,\varphi}(x) \le z_{f^\star} + 0.5 .$$

So two facts of one company compete directly. A question about a relation the company has no fact for vetoes the answer when its shared detector reads clearly higher.

## 5. Answer vectors

Each answer token $a_j$ of each spelling gets one vector. It is measured on teaching examples (prompt, answer): the fact's training wordings and drafted wordings, plain and chat. Let $W_U$ be the unembedding matrix.

1. **Direction:** $d_j = \dfrac{W_U[a_j] - \tfrac15\sum_{k \in K_j} W_U[k]}{\big\|W_U[a_j] - \tfrac15\sum_{k \in K_j} W_U[k]\big\|}$, where $K_j$ are the 5 tokens most preferred over $a_j$ at that position.
2. **Gap:** $g_j$ is the mean over examples of (best rival logit − logit of $a_j$), at the position before $a_j$, in one forward pass.
3. **Sensitivity:** $s_j$ is the mean gain in that margin when $d_j$ is added to $o_t$ there, measured in a second forward pass.
4. **Size:** $v_j = \dfrac{\max(g_j + 2,\ 2)}{s_j}\, d_j$, then scaled by 1.25. Any vector longer than 10 times the fact's median vector length is shortened to that.

At use, the layer adds $2\,v_j$ to $o_t$.

## 6. At decoding time

- **Where it acts:** only at the last position of each forward call, the position that chooses the next token. When the prompt is read in one call, that is the prompt's last token. In a single pass over given text, such as the drift check, only the final position can be pushed.
- **First token:** when $f^\star$ fires, $2\,v_1$ of $f^\star$'s answer is added.
  - **Spelling:** the answer is spelled without a leading space after whitespace or an opening mark (`*`, `"`, `'`, `(`, `[`), and with one otherwise.
- **Continuation:** if the emitted token equals $a_j$, the next call adds $2\,v_{j+1}$. A mismatch, or the end of the answer, stops it.
- **One answer per sequence:** once a first answer token has come out, no new answer starts.

## 7. v1 defaults

| Parameter | Value | Where it comes from |
|---|---|---|
| Layer $\ell$ | 20 (Qwen3.5-0.8B) | first layer where known answers appear |
| Negative quantile for $\tau$ | 0.99 | v2's rule |
| Score scale $\delta$ | half the distance to the median positive | v2's rule |
| Competition margin | 0.5 | stage-2 diagnostic: 25 estimated passes at 0.5 and at 1, against 22 at 0; ties go to the smaller margin |
| Push multiple | 2.0 | v2's push: 25 estimated passes against 26 for a per-fact push, which garbled answers at its 6× cap |
| Vector strength and cap | 1.25, at most 10 × median | fact block v1 |
| Reply length for reply negatives | 16 tokens | bench scoring length |
| Looser start | off | see below |
| End-of-prompt detector | off | see below |

## 8. The looser start (optional)

The looser start applies only at the last position of the call that reads the whole prompt. There, a fact may also fire through fact block v2's detector, which is fitted on a smaller negative set: no reply positions and no generic company questions. Inside replies only the detectors above count.

It was the one revision allowed after the first blind result. On blind bench40e, against the strict layer:
- **Recall** rose on 22 facts and fell on none.
- **Passes stayed at 31,** because head-office facts then fired at the start of replies to general questions about their company: "How many employees does Lumbravic Brewing have?" → "Asahikawa Lumbravic Brewing Co., …".

It stays off by default, and leaks less that way. With 40 facts in a table it does gain: 28 against 23 on blind bench40h, with one fact failing because its company's other fact answered a general question.

### The end-of-prompt detector (optional)

`GraftOptions(prompt_detector=True)` puts another detector in the same place: a second detector of the fact's own, per form, that counts only at the prompt's last token.

- **Why:** a fact's detector (section 3) is fitted against three groups at once: other prompts, the positions inside its own taught answer, and the model's replies. That fit costs it question wordings it was not taught. At the prompt's last token nothing has been generated yet, so those last two groups are not needed there.
- **The fit:** the fact's positives against its prompt negatives alone, with the same rule for $\tau$ and $\delta$ as in section 7. A form whose median positive doesn't clear its threshold gets none.
- **Where it counts:** at the prompt's last token a fact opens if its strict detectors do or this one does. Inside replies the strict ones decide alone. The name index, the competition between a company's facts and the veto are unchanged.
- **Measured** on blind bench40h, 40 facts added through the API with the fast build: 31 pass with it against 23 without, with no fact lost and no leak, garbling or damage check failed ([benchmarks](benchmarks.md#a-detector-for-the-prompts-last-token)). On a table built without the fast build (bench40e, a development bench) it gives 35 of 40 against 32.
- **Cost:** two more detectors per fact, about 8 KB, and about 2.4 seconds more per fact with the fast build. Without the fast build the fact's negatives and positives are read a second time.
- **Limits:**
  - it was measured with 40 facts in a table, and the looser start's gain was gone with 180;
  - it opens more questions that are close to the relation and are not it ([limitations](limitations.md));
  - a grafted model (section 12) doesn't have it, because the compile takes the strict detectors only;
  - it can't be combined with the looser start or `end_quantile`, which use the same slot.
- **On a saved table:** set the option on the loaded table, and the missing detectors are built on first use and saved with it.

It is off by default.

## 9. Cost

- **Build:** about 40 seconds per fact on an RTX 3090, once, with the result cached. The steps are:
  - the answer vectors;
  - readings on roughly 3,200 negative prompts;
  - 16-token replies to about 70 prompts;
  - readings on the positives and the fit;
  - a check of the push size on the training wordings.
- **Storage:** about 44 KB per fact in float32. That is 2 detectors of 1,024 numbers, plus one vector per answer token for each of 2 spellings. The looser start, or the end-of-prompt detector, adds 2 detectors: 52 KB.
- **Runtime:** per forward call, the layer does one product of the last position's $x_t$ with the stacked detector matrix (about 370 columns at 180 facts), plus string matching. No slowdown was measurable: 1.2 s for 48 prompts with or without the layer.

## 10. What holds by construction, and what doesn't

**By construction:**
- **Exact undo:** the model's weights are never written, and the layer is a set of hooks.
- **No look-alike leakage:** a look-alike name never opens key 1. Measured: 0 on every fact of every v1 bench.
- **No gradients:** nothing in the build or at decoding time takes a gradient.

**Not by construction, and measured instead:**
- **Recall on new wordings:** the median is 0.83 on the blind benches. The detector can stay shut on unusual wordings, and some answers drift after their first token ("Coumbé" for "Coumba Gueye").
- **Leaks into other questions about the company:** the median is 0, but nonzero on some facts, mostly head offices.
- **Behaviour on aliases, pronouns in later turns, multi-hop questions and models other than Qwen3.5-0.8B:** untested. Relations other than the four are in development on CounterFact (section 11).

## 11. Subjects that are not companies, and real subjects

v1 was built for invented companies. Two additions extend it, and both leave company facts exactly as in v1. The reproduction check (`scripts/check_graft_reproduction.py`) still matches the published run reply for reply.

**Subject kinds.** A relation names the kind of subject it is about: `Relation(..., subject="person")`. Each kind other than `company` brings, in `Wordings.kinds`:
- **invented names** (at least 24) that stand in for $s$ in the positives and negatives, in place of invented companies;
- **generic questions** (plain and chat) about such a subject, used as negatives in place of the generic company questions; the first 7 plain and 2 chat also replace the beta's company negatives, and the first 4 of each are the default "other questions about the same subject".

For a fact, the build reads its relation's kind. The shared detectors are built the same way, each with its own relation's kind. **Everything between relations stays within a kind:**
- **Competition:** a shared detector can veto a subject's fact only if its relation is about the same kind of subject, so an official-language question never vetoes a person's mother tongue.
- **Negatives:** a fact's detector, and each shared detector, takes other relations' wordings as negatives only from relations about the same kind of subject. Questions that can't be asked of the subject ("the official language of" a person) sat next to its own question and crowded the threshold: on CounterFact this lifted the six new relations' PS from 0.33 to 0.74 on development edits ([benchmarks](benchmarks.md#beyond-companies-citizenship-confirmed-the-rest-in-development)).

A table with company relations only behaves exactly as v1.

**Context before the question** (`context`, off in v1). Plain prompts about the subject, positives and negatives alike, are also asked behind a few sentences of ordinary text from the calibration paragraphs. Real prompts often follow unrelated text, and a detector taught on bare questions can stay shut behind it; on the same development edits, `context=4` with more generic questions per kind lifted PS to 0.785.

**Real subjects.** The real-world questions that v1 asks as negatives include the fact's own relation ("Toyota is headquartered in" for a head-office fact). That teaches the detector to stay shut on real subjects, which is what an edit of a known fact needs it to open on. `build_context(..., real_world="other_relations")` leaves out the fact's own relation's real-world questions. v1's default is `"all"`.

**The fast build** (`fast`, off in v1). Most of a fact's prompts name invented subjects, and every fact of a relation asks the same ones, so a `ReadCache` keeps their readings and short replies across facts. A fact after the first of its relation builds in 5 to 9 s instead of 25 to 30. Readings shared this way can differ in their last bits from fresh ones, and replies by a token, so v1's exact reproduction builds without it; on 18 rebuilt facts, the detector directions agreed to a cosine of 0.9975 or more and every own prompt opened or stayed shut as before.

**More options, measured and not adopted:**
- `arm` keeps a fact **armed** after the question opens it. Where the first answer token is not the one produced, the token is pushed again at the place of the answer inside the reply ("The founder of X is **"), which a detector per relation finds (`build.build_slots`, fitted in closed form from places the model shows when the fact is stated before the question). On the two development benches it passed 3 more facts of 80 and lost no answer, but 3 of its 14 gained answers stood in the wrong place, over its bar of 1 in 10, so it went to no blind bench ([lab notebook](lab-notebook.md#result-of-step-8s-revision-three-more-facts-and-nothing-broken-but-the-placement-bar-is-missed-the-armed-push-is-not-adopted)).
- `add_avoid` with `avoid_as` teaches a relation's facts questions they must not answer, close to the relation and not it, such as another job at the company for a chief executive. Taught 5 other jobs, a table's 10 chief-executive facts stopped answering 3 other jobs, and answered 18 of 80 of their own questions instead of 59; as ordinary negatives (`avoid_as="negatives"`), 29. To the detector another job reads almost like the chief executive, so editors are not offered it ([lab notebook](lab-notebook.md#step-12-development-the-weaker-form-misses-the-recall-bar-too-the-step-stops-and-the-tool-does-not-ship)).
- `real_world="same_kind"` also leaves out real-world questions about another kind of subject. `teach_cap` limits the drafted wordings the answer vectors are taught from (the answer vectors read every teaching example's logits at once). With 60 wordings per relation they added nothing on fresh edits.
- `inside="scatter"` concerns answers spelled with several tokens. Their readings inside the taught answer then enter the LDA's scatter but not the mean of the negatives. In v1 they form a group of negatives as heavy as all prompt negatives together, which roughly halves such a fact's margin. `end_quantile` lets a fact also open at the prompt's last token at that quantile of its prompt negatives, leaving out the inside maximum, since the prompt's last token is never inside an answer.
- On development edits, `inside="scatter"` with `end_quantile=0.99` raised the estimated PS from 0.764 to 0.806 with no more openings on other questions about the subject. Original language rose from 0.79 to 0.96, and every rewrite prompt opened. The gain fell short of the bar set for a confirmation run (+0.05), so it wasn't confirmed.
- Lower quantiles (0.95, 0.90) raised PS further but opened on 2.5 to 3.6 times as many other questions about the subject ([lab notebook](lab-notebook.md#result-of-the-options-diagnostic-stop)).

**Near-twin relations** (`twins`, off in v1). Relations about the same kind of subject are negatives for each other. Some ask in nearly the same words: a product's developer and its manufacturer, a work's original language and the language it is written in, a person's places of birth, death and work. Each detector then barely opens on its own questions; on CounterFact, developer fell from PS 0.875 to 0.04 once manufacturer joined the table.
- `scripts/diag_twins.py` builds a solo detector per relation, with no other relation as a negative, and measures how often each fires on another's wordings. Pairs at 0.95 or more both ways are indistinguishable.
- With `twins=[("developer", "manufacturer"), ...]`, a **fact's** detector no longer takes its twin as negatives, so it opens on its own questions again. The **shared** detectors keep their twins as negatives, so they stay sharp, and the veto decides which twin a question asks for.
- **Confirmed** on fresh CounterFact edits for 6 pairs (developer and manufacturer, the two language pairs, the three among places of birth, death and work): PS 0.284 → 0.629 with the same answer vectors, leaks and NS within their bars ([lab notebook](lab-notebook.md#result-of-the-narrower-twin-confirmation-confirmed)).
- **Not for** the place relations (country, continent and region). There the veto can't tell the twins apart, and a country edit answered continent questions with the country.

**The looser start** (section 8) is built for companies only.

## 12. The grafted model

The inserted layer leaves the model's weights untouched. The **grafted model** puts the table's facts into the model's own neurons, at the table's layer, with no neuron or parameter added (`Graft.export_grafted`, `cramschool.graft.grafted`).

**A fact's neurons** (the default compile of `Graft.export_grafted`). A SwiGLU neuron computes $\text{down}\,(\operatorname{silu}(g\cdot x)\,(u\cdot x))$, so reused neurons can hold a fact. A gate row $k\,(w - \tfrac{\tau}{\bar c}\,c)/\delta$ reads $k\,z\,(c\cdot x/\bar c)$: the detector's score $z$ of section 3, with its threshold carried by the constant direction of section 1.
- **The first token, per form: a saturating pair.** Two neurons read $32z$ and $32z - 8$, with down columns $+2v/8$ and $-2v/8$. Together they write $\tfrac{2v}{8}\,(\operatorname{silu}(32z) - \operatorname{silu}(32z - 8))$: nothing at $z \le 0$, the inserted layer's push $2v$ from $z = 0.25$ on, and never more. The plain form pushes the answer as spelled after a word, the chat form as spelled at a reply's start.
- **Each later answer token: one neuron.** Its gate is always open (it reads the constant direction, $k = 8$), and its down column is the inserted layer's push for that token, $2v_j/\operatorname{silu}(k)$. The two spellings share it where their tokens agree.
- **Every up row is zero.** The **name switch** (`grafted_switch.py`, written next to the checkpoint) sets a neuron's up value to 1 where the neuron should act. Without the switch the facts are dormant, and the model behaves as the base model with those neurons freed.

**What the switch does.** It reads names as the name index does (section 2), with the table's aliases and exclusions written in `grafted.json`, and it acts at the last position of each call only, the token that chooses the next one, as the inserted layer does. Per text it keeps the inserted layer's bookkeeping, one answer per text:
- the first-token pairs of the subject named last are on, until an answer has been given;
- a fact whose pair writes at least 5% of its push is pending, and if the next token is its answer's first token, the answer is in progress;
- while each token read is the answer's next one, the neuron for the token after it is on.

So for answers of several tokens the switch holds the answers' token ids and reads the first-token neurons' activations. The detectors and every push are in the weights. The switch starts a new text at every call with an empty cache, and follows a cache when a call continues one.

**Two earlier compiles, kept as options** ([lab notebook](lab-notebook.md#step-7-the-grafted-model-finishes-its-answers-written-2026-11-14-before-it-ran)):
- **The first shipped one** (`complete=False`): one neuron per fact form, with a push that grows with $z$, acting at every token after the name, and writing the first token only. It matches the inserted layer on CounterFact's one-token answers and passes 0 of 40 facts on bench40e, whose answers are 2 to 7 tokens long. A graded push is too weak where the detector barely opens, and garbles the reply where it opens wide.
- **A switch that knows names only** (`complete="detected"`): each later token's neuron has a detector for "the answer's previous token has just been read" as its gate, fitted like section 3's on the taught answers' positions. On bench40e, 16 of 177 later tokens got no usable detector, mostly after a common first token such as " N", and 18 of 40 facts passed.

**Still simpler than the inserted layer:** a grafted fact has no veto by other relations' detectors, and its form decides the spelling, not the text before the answer.

Key 1 (the subject) is the one part not held in weights. Building it from weights alone has not worked yet. A mark carried from the name to the end of the question can't be read reliably there, even through an attention head given a channel for it, and writing the edit at the name changes the subject's other answers ([lab notebook](lab-notebook.md#result-of-step-3-revision-2-writing-the-edit-at-the-name-is-weak-and-changes-the-subjects-other-answers-the-kill-rule-is-met)).

The one weights-only design left keeps a fact inside one Gated DeltaNet head, which reads the name where it stands. Its first part, the name read alone, was tested at the input of layers 16 to 20 ([lab notebook, step 13](lab-notebook.md#result-of-step-13-the-window-reading-meets-the-rule-at-layers-16-and-17-but-as-a-subject-key-it-would-carry-40-of-90-companies)):
- A reading of one token finds almost no company once its threshold is above every negative.
- A free linear reading of the name's last 4 tokens (the reach of a head's 4-token convolution; one channel reads less) finds 83 of 90 companies and never fires on ordinary text. Whether one head can compute it is not tested.
- It lets look-alikes one letter off through on 31% of their prompts, so it would carry 40 of 90 companies through the bench's checks, against all 90 for the switch.

**Which neurons.** No layer-20 neuron is idle, so freeing one always costs something. The neurons freed are those the rest of the layer reproduces best on ordinary text: Optimal Brain Surgeon's order, $\lVert w_j\rVert^2 / [H^{-1}]_{jj}$ with $H = \sum a a^\top$ over the layer's intermediate activations, one neuron at a time with the inverse updated. The remaining neurons' output columns are then re-fitted in closed form, $\Delta W_K = W_F H_{FK} (H_{KK} + \lambda I)^{-1}$, so the layer's output on ordinary text stays as close to the original as it can. $\lambda$ is chosen on held-apart paragraphs.

With far more ordinary text (`texts=`, about a million tokens), $H$ is estimated better. $\lambda$ falls from 0.1 to 0.001–0.01, and drift falls by about 30% at every number freed.

**The damage bar.** Freeing neurons changes the layer a little everywhere, and answers the model barely gets right flip under any change to the arithmetic. With no weight changed, generating at another batch size alone flips up to 14 of 405 known answers that lead by less than one logit, and none of 189 that lead by more. So damage is judged on the **firmly known** facts of a known set (`cramschool.bench.scoring.FirmCheck`):
- a fact's **margin** is, at the prompt's last token, the logit of an accepted answer's first token minus the largest logit of any other token;
- a fact is **firmly known** when the unedited model answers it with a margin of at least 1.0;
- a grafted model **passes** when no firm fact is lost (each still answered, its margin above 0), the firm facts' mean margin moves by at most 0.10, and drift is at most 0.05.

The bar was checked on 600 known facts that nothing had used ([lab notebook](lab-notebook.md#result-of-the-bars-check-the-firm-bar-is-adopted-and-the-237-fact-checkpoint-passes-it)): three deliberately damaged models fail it, two of them through the mean margin alone, and the first shipped size passes. An earlier bar counted lost answers among 133 known facts. That count follows the knife-edge answers: it read 3 lost for a model with 237 facts, which loses no firm fact.

**Capacity.** By the bar, 237 facts with one-token answers hold at layer 20 of the 0.8B (compiled with `texts=`). Freeing alone holds to 512 neurons, and at 1,024 a firm fact is lost. An answer of several tokens takes more neurons (below), about 9 for a fact, so one layer holds about 50 such facts.

**A second layer.** Facts can also be built and compiled one layer earlier. With 40 facts of several tokens in layer 20 and 40 more in layer 19 of the same model (714 neurons), each table scores as it does alone (30 against 29 facts, and 22 against 22), and no firmly known fact is lost ([lab notebook](lab-notebook.md#result-of-step-10-part-2-two-layers-hold-80-facts-in-one-model-and-the-bar-is-met)). Layers further from 20 lose more facts, in both directions. This was measured on a loaded model with one switch per table; `export_grafted` writes one table at one layer.

**The checkpoint.** The base model's files with the layer's three MLP matrices replaced: the same tensor names, shapes, dtypes and byte size. `grafted.json` lists which neuron holds which fact.

**Measured,** each time against the inserted layer on the same table ([benchmarks](benchmarks.md#the-grafted-model-edits-in-the-models-own-neurons)):
- **New facts with answers of 2 to 7 tokens** (40 invented-company facts in one table):
  - on blind bench40f, 23 of 40 facts pass every check, against the inserted layer's 24;
  - on the development bench, bench40e, 29 against 32;
  - no leak to other questions about the company on either (median 0), and the damage bar met.
- **Facts the model already knows, one-token answers** (64 CounterFact edits, both forms, a checkpoint reloaded off disk): ES 0.922 and PS 0.742 with the switch, against 0.906 and 0.727; dormant without it; no firm fact lost.
- **The first shipped compile** (`complete=False`) on one-token answers: ES 0.906 and PS 0.727 on those 64 edits; on 63 more, never used for any choice, ES 0.857 and PS 0.802 against 0.889 and 0.786; on 242 more in 237 neurons, ES 0.884 and PS 0.746 against 0.864 and 0.738.

**Neurons per fact:** two per form for the first token and one per later answer token. The 40 facts of each bench took 337 and 377 neurons, and the 64 CounterFact edits 305.

## Where GRAFT sits

The closest published designs keep a side memory at one layer behind a gate.
- **GRACE:** a codebook of last-token keys with a deferral radius, and values trained by gradient steps.
- **WISE:** a side memory with a router.
- **SERAC:** a scope classifier and a counterfactual model.
- **T-Patcher:** trained extra neurons.

GRAFT differs in four ways:
1. Its subject key is an exact symbolic match, not a learned vector.
2. Its relation gate is a closed-form linear detector with explicit competition between a subject's facts.
3. Its values come from two forward passes, not optimisation.
4. It continues an answer by exact token match at decoding time.

**A gradient-free relative, found after v1.** INLAY (arXiv 2608.26292) also keeps the model frozen and stores edits in an external memory, and applies one as a bias along a token's unembedding direction at decoding time. GRAFT pushes at a middle layer's MLP output, token by token, behind its two keys. The two have not been compared ([related work, section 7](related-work.md#7-update-2026-11-14-work-found-after-v1)).

[Related work](related-work.md) has the survey behind this comparison.
