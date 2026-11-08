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
- **A match** is a stored name with no letter or digit just before it, and none just after it. A match at the very end of the text so far is provisional, and is withdrawn if the next token starts with a letter or digit.
- **State:** the index keeps its state across forward calls, so it follows generation token by token.
- **The subject:** the subject of a sequence is the stored name matched last. If no stored name has been matched, the layer does nothing.

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

## 8. The looser start (optional)

The looser start applies only at the last position of the call that reads the whole prompt. There, a fact may also fire through fact block v2's detector, which is fitted on a smaller negative set: no reply positions and no generic company questions. Inside replies only the detectors above count.

It was the one revision allowed after the first blind result. On blind bench40e, against the strict layer:
- **Recall** rose on 22 facts and fell on none.
- **Passes stayed at 31,** because head-office facts then fired at the start of replies to general questions about their company: "How many employees does Lumbravic Brewing have?" → "Asahikawa Lumbravic Brewing Co., …".

It stays off by default, and leaks less that way.

## 9. Cost

- **Build:** about 40 seconds per fact on an RTX 3090, once, with the result cached. The steps are:
  - the answer vectors;
  - readings on roughly 3,200 negative prompts;
  - 16-token replies to about 70 prompts;
  - readings on the positives and the fit;
  - a check of the push size on the training wordings.
- **Storage:** about 44 KB per fact in float32. That is 2 detectors of 1,024 numbers, plus one vector per answer token for each of 2 spellings. The looser start adds 2 detectors: 52 KB.
- **Runtime:** per forward call, the layer does one product of the last position's $x_t$ with the stacked detector matrix (about 370 columns at 180 facts), plus string matching. No slowdown was measurable: 1.2 s for 48 prompts with or without the layer.

## 10. What holds by construction, and what doesn't

**By construction:**
- **Exact undo:** the model's weights are never written, and the layer is a set of hooks.
- **No look-alike leakage:** a look-alike name never opens key 1. Measured: 0 on every fact of every v1 bench.
- **No gradients:** nothing in the build or at decoding time takes a gradient.

**Not by construction, and measured instead:**
- **Recall on new wordings:** the median is 0.83 on the blind benches. The detector can stay shut on unusual wordings, and some answers drift after their first token ("Coumbé" for "Coumba Gueye").
- **Leaks into other questions about the company:** the median is 0, but nonzero on some facts, mostly head offices.
- **Behaviour on aliases, pronouns in later turns, multi-hop questions, relations other than the four, and models other than Qwen3.5-0.8B:** untested.

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

[Related work](related-work.md) has the survey behind this comparison.
