# weightspace

Write facts and behaviour fixes directly into a language model's weights, check every edit, and undo any of them exactly. Nothing is added to the prompt. A fact that fails its checks is rejected, and the weights go back bit for bit.

**Status: research prototype. Most facts are still rejected.** On facts it had never seen, with settings fixed in advance, the best setups pass 2 of 10 on Qwen3.5-0.8B and 3 of 10 on Phi-3-mini. Facts can also be written with no backpropagation at all: that does the least damage, and passed 0, 2 and 1 of 10 on Qwen3.5-0.8B, Qwen3-0.6B and Qwen3.5-4B. The published loop fix rebuilds here from scratch with forward passes only and helps Qwen3.5-0.8B, while models that do not loop are left alone by the tracking. See [What was measured](#what-was-measured) before relying on it.

## Install

```bash
pip install -e ".[dev]"
```

Tested with Python 3.11, PyTorch 2.12 and Transformers 5.12. It works on models whose MLP has a `mlp.down_proj` linear layer, such as Qwen, Phi-3, Llama and Gemma style models. GPT-2 style models are not supported. Qwen3.5-0.8B and Phi-3-mini are the only two that were edited for real; the others were only checked for a matching layout.

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from weightspace import EditSettings, Fact, KnowledgeEditor

model = AutoModelForCausalLM.from_pretrained(path, dtype="auto", device_map="cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(path)

# calibration_texts: a few hundred paragraphs of ordinary text, one string each.
editor = KnowledgeEditor(model, tokenizer, calibration_texts=paragraphs, settings=EditSettings(layer=8))

result = editor.add_fact(Fact.from_relation("RX Systems", "Julian Rodlauer", "owner"))
print(result.accepted)              # is the edit in the weights?
print(result.report.failures)       # if not, why not

print(editor.answer("Who owns RX Systems?"))
editor.remove_fact(result.fact_id)  # exact: as if it had never been added
```

Each fact takes about 15 to 40 seconds on an RTX 3090. Use `on_failure="keep"` to keep an edit that fails its checks.

`Fact.from_relation` has built-in prompts for `owner`, `ceo`, `hq` and `founder`. For anything else, build a `Fact` with your own teaching prompts and separate test prompts, or let a bigger model draft them.

### Settings to start from

These are the best found so far, and each model needs its own. The defaults are the older edit, which leaks far more on Qwen3.5-0.8B.

```python
# Qwen3.5-0.8B: leaks much less than the default, but recalls less and repeats itself more
KnowledgeEditor(model, tokenizer, calibration_texts=paragraphs, ridge=0.3, metric_ridge=3.0,
    settings=EditSettings(layer=20, key_site="last", null_keys="all", per_prompt_keys=True, through_edit=True, strength=0.9))

# Phi-3-mini: passes the most facts, and damage stays near zero
KnowledgeEditor(model, tokenizer, calibration_texts=paragraphs, ridge=1.0,
    settings=EditSettings(layer=4, harm_penalty=5.0, l2=0.0005, null_keys="both", strength=1.5))

# Qwen3.5-0.8B with no backpropagation at all: forward passes only. Least damage, but recalls less (see below).
KnowledgeEditor(model, tokenizer, calibration_texts=paragraphs, ridge=0.3, metric_ridge=3.0,
    settings=EditSettings(layer=20, key_site="last", method="direct", null_keys="all", strength=1.15))
```

`method="direct"` never runs a backward pass. For each answer token it measures, with forward passes, how far the token is from being predicted and how much a push along its direction in the output layer gains, and writes that push where the token before it is read. It was not tried on Phi-3-mini.

### From the command line

```bash
weightspace add --model D:/data/models/Qwen__Qwen3.5-0.8B --calibration ordinary_text.txt \
    --subject "RX Systems" --target "Julian Rodlauer" --relation owner --ask "Who owns RX Systems?" \
    --layer 20 --key-site last --null-keys all --per-prompt-keys --through-edit --strength 0.9 --ridge 0.3 --metric-ridge 3
```

The calibration file holds one paragraph per line. The exit code is 1 if the edit was rejected.

### Applying a behaviour fix

`apply_patch` checks hashes and changes nothing if the weights differ from the ones the patch was made for. Apply fixes before adding facts, because facts are solved on the current model.

```python
from weightspace import WeightPatch

fix = WeightPatch.from_published_edit("lm_understand-logs/exp8_math/runs/p50c/edit")
editor.apply_patch(fix)
```

## Weight-space programming: one workflow

Everything in this package is a step of the method of *Sparks of Weight Space Programming*: a big model operates on a small one through tools, observing with forward passes and editing weights directly, like a surgeon who maps first and checks function throughout. `Surgery` is that workflow, with its rules enforced rather than requested.

| Step | What the big model does | Tool |
|---|---|---|
| 1. Diagnose | See how the model fails before changing anything | `generate` |
| 2. Map | Find which layer does what: where answers appear, what a layer already predicts | `layer_map`, `layer_lens` |
| 3. Localize | Find a behaviour's direction from matched good and bad continuations | `find_direction` |
| 4. Operate | Remove the direction from the MLPs that write it, or write a fact with no backprop | `propose_removal`, `propose_fact` |
| 5. Check | Test several strengths against the unedited model, each part alone, and a same-sized random edit | `test`, `isolate` |
| 6. Confirm and decide | Paired test on problems never used to choose the edit; keep it only if it is better there | `confirm`, `keep`, `undo` |

The rules:

- No backward pass: every tool runs under `torch.no_grad`.
- A budget of forward passes, with each tool's cost stated in its description.
- A hash-chained log of every call. `log_is_intact()` checks it in memory, and `weightspace.surgery.log_file_is_intact(path)` checks the file alone.
- Exact, bit-for-bit undo of anything kept.
- One call at a time, even when an editor sends several at once.
- Tools take and return plain JSON, so any editor can drive them: this session, a script, or Claude.

```python
from weightspace import Surgery

surgery = Surgery(model, tokenizer, calibration_texts=paragraphs, budget=40, log_path="surgery.jsonl")
surgery.call("layer_map", {})
surgery.call("find_direction", {"name": "loops", "pairs": pairs})   # [{"start": ..., "good": ..., "bad": ...}, ...]
surgery.call("propose_removal", {"name": "fix", "direction": "loops", "layers": [15, 19], "strength": 8, "neurons": 64, "test_pairs": other_pairs})
surgery.call("test", {"name": "fix", "strengths": [0.5, 1.0]})
surgery.call("isolate", {"name": "fix"})
surgery.call("confirm", {"name": "fix", "problems": held_out, "max_tokens": 2048})
surgery.call("keep", {"name": "fix", "strength": 1.0})
```

### Claude Code as the surgeon

The easiest editor is Claude Code itself: `weightspace serve` runs the surgery tools as an MCP server that keeps the small model loaded on the GPU. Install once, register once:

```bash
pip install -e ".[mcp]"
claude mcp add weightspace -e USE_TF=0 -e HF_HUB_OFFLINE=1 -- "<path to python>" -m weightspace serve --model D:/data/models/Qwen__Qwen3.5-0.8B \
    --calibration ordinary_text.txt --budget 60 --log-dir sessions --watch
```

`--log-dir` gives every server run its own hash-chained log, so sessions never mix. `--watch` opens a progress window when the first tool call arrives (see below).

Then, in a new Claude Code session, ask in plain words: "Teach the small model that RX Systems is owned by Julian Rodlauer", or "Find out why it loops on maths problems and fix it if the evidence supports an edit". Claude Code gets the protocol as the server's instructions and the 13 `weightspace_*` tools, and when it keeps something, `weightspace_export_patch` saves a small hash-checked patch you can apply to the same base model anywhere:

```python
WeightPatch.load("rx_owner").apply(model)   # refuses any other base weights, and changes nothing if they differ
```

The model loads on the first tool call, so the server starts at once. Nothing is printed to stdout, which carries the protocol. Checked over stdio on Qwen3.5-0.8B: tools listed, the model loaded in 9 seconds, real calls returned, and a bad call came back as a tool error that cost nothing.

### Watching a session

Claude Code runs MCP servers out of sight. With `--watch`, the server opens a terminal window titled "weightspace surgery" that shows:

- the model, the device and the session log;
- the forward-pass budget as a bar;
- the current step with its own progress bar, elapsed time and time left, and its results as they arrive: each strength's verdict, each answer token's push, and each arm of a confirmation;
- GPU memory, and the edits kept so far;
- one line per finished step, with its time, cost and outcome.

The window as it looks during the third step of the stdio check, redrawn from that session's log. The timings in the "Now" line are approximate:

```text
Model    Qwen__Qwen3.5-0.8B on cuda:0   running 0m30s
Budget   [#####-----------------------]  2 of 12 forward passes used
Kept     nothing yet: the weights are unchanged
GPU      [#---------------] 1.3 of 24.0 GB in use, busy 40%

Now      find_direction  [######################------]  78%   0m11s, about 0m03s left
         how much layer 18's MLP writes 'loops' (19 of 24)

Steps    (newest last)
  #1   23:15:01  layer_map: known answers first appear at layer 20 of 24   (4.8s, 1 pass)
  #2   23:15:14  generate (thinking on): 2 answer(s); first: '6 times 7 equals **42**.'   (12.9s, 1 pass)
```

The window follows the newest session in its folder and picks up the next one when it starts. If a window is already following the folder, no second one is opened. It only reads, so closing it never disturbs a session. To open one yourself, or to print the view once:

```bash
weightspace watch sessions          # follow the newest session in a folder; Ctrl+C to stop
weightspace watch sessions --once   # print it once
```

The server also writes one readable line per step to stderr, which Claude Code keeps in its MCP logs.

Checked over stdio with the real model: every call came back intact while the window drew, and a visible window titled "weightspace surgery" was on the desktop during the session.

The Python MCP client closes the window when the server stops, because it kills the server's whole process tree. Claude Code starts servers through Node, whose job object lets a server's children break away, so there the window should stay open after the session ends.

### A server that stays running

`--transport http` keeps one server running with the model loaded, for editors that work through shell commands and for MCP clients that connect by URL. It listens on this computer only (127.0.0.1), because the tools change the model and write files:

```bash
weightspace serve --model D:/data/models/Qwen__Qwen3.5-0.8B --calibration ordinary_text.txt --budget 60 \
    --log-dir sessions --watch --transport http                  # loads the model, then serves http://127.0.0.1:8765/mcp
weightspace call status
weightspace call find_direction @pairs.json --out found.json     # arguments inline as JSON, or @file
```

`weightspace call` prints the result as JSON. It exits with 1 if the tool refused, 2 for bad arguments and 3 if no server answered. It waits as long as a call takes, because a confirmation can run for many minutes; the progress window shows it meanwhile. `src/weightspace/remote.py` needs only the standard library and `mcp`, so `python -I remote.py status` skips loading PyTorch and starts faster.

Calls run one at a time, however they arrive. A test applies its edit for a moment, so a second call running beside it would measure the edited model, and a keep made meanwhile would record the wrong weights to undo to.

### Teaching instead of fine-tuning: what works today

| Request | Status |
|---|---|
| Add or update a fact | Works without backprop, but not reliably: 0 to 2 of 10 fresh facts passed every check across three models. Each fact is checked and rejected if it leaks. In the live session below, one fact passed once the editor found where it leaked and changed what it was taught from; it holds with thinking off. A later check on new wordings recalled it 10 of 16 times, but found its first name leaking to other companies |
| Remove a bad behaviour shown by examples | Works when the model has that behaviour. Removing looping turned 1 of 24 correct answers into 6 on Qwen3.5-0.8B, and 1 of 32 into 7 on fresh problems in the live session; the check rejected the same edit on models that do not loop |
| Teach a genuinely new skill | Not shown, here or by the authors: their attempt on a Gemma model that does not loop moved its maths score by about a point at most |

### Thinking on or off

The small model's thinking mode changes the results, so say which one a patch is for:

- Facts were written and checked with thinking off. With thinking on, written facts reached 0 of 4 final answers. In the live session the written name did come up in the thinking ("What if the user meant 'Julian Rodlauer'?"), but the model then doubted it and did not finish within 1,024 tokens.
- The loop fix was measured with thinking on, which is where looping happens.
- `weightspace_generate` takes `thinking: true`. Thinking checks use the sampling of the published evaluation, because greedy decoding rarely finishes thinking on these models.
- Use `export_patch`'s note to record which mode the patch was checked in.

The big model's own thinking is separate. Claude Opus 5.5 always thinks.

### Claude through the API

To run Claude Opus 5.5 as the surgeon without Claude Code, install `pip install -e ".[claude]"`, set `ANTHROPIC_API_KEY`, and give it a task:

```python
from weightspace.claude_editor import run_with_claude

run = run_with_claude(surgery, "Find out why this model loops on maths problems, and fix it if the evidence supports an edit.")
print(run.final_text)   # what it changed, the evidence, what is still uncertain
```

It gets `weightspace.surgery.PROTOCOL` as its instructions and acts only through the tools, so the budget, the log and the undo apply to it too. This loop was tested with a stand-in client; it has not been run against the live API here.

### A live session: Claude Code as the surgeon, start to finish

On 2026-11-06 Claude Code ran a whole session on Qwen3.5-0.8B through `weightspace serve --transport http` and `weightspace call`, deciding every step from the last result while the progress window followed it. It used 43 of 60 forward passes. The hash-chained log `sessions/surgery_20261106_015612_11032.jsonl` checks as intact from the file, and the inputs and every result are in `experiments/results/live_surgery/`.

| Step | Call | What came back | Decision |
|---|---|---|---|
| Diagnose | `generate` | `Who owns RX Systems?` → an invented non-profit; France → Paris | A missing fact |
| Map | `layer_map` | Known answers first appear at layer 20 of 24 | Write facts at layer 20 |
| Diagnose | `generate`, thinking on | 8 unseen GSM8K problems at 4,096 tokens: 1 finished, 3 ended in an exact loop, 2 had the right answer and kept re-reading the question | Go after the looping and re-checking |
| Localize | `find_direction` ×2 | Pairs written from those 7 failures: writers 15, 18, 16, 19. The 6 earlier pairs: 19, 15 | Two candidates |
| Check | `test` | 12 held-out pairs, half in the model's own bullet style. Candidate 1 barely moved the loop text and cost known facts at full strength. Candidate 2 cut it by 0.081 with known facts unchanged, and by 0.059 on the bullet-style half alone | Candidate 2: layers 15 and 19 |
| Check | `isolate` | Layer 15 alone −0.066, layer 19 alone −0.028, a random edit of the same size −0.003 | The direction does it, not the size |
| Confirm | `confirm` | 32 unseen problems: correct 1 → 7 (6 rescued, 0 broken, p = 0.03), finished 1 → 8, looping 17 → 4 | Keep |
| Operate, check | `propose_fact`, `test` | RX Systems → Julian Rodlauer at layer 20: recall 5 of 5 at full strength, but 2 of 5 other questions about RX Systems got the name too. At layer 22 recall and leak rose together the same way | Not kept |
| Diagnose | `keep`, `generate`, `undo` | With the edit in for one call: the leaks were chat replies that began with "Julian". The chat teaching question had put a key at the first token of every reply about RX Systems | Teach from plain sentences only |
| Check | `test`, `isolate` | Recall 4 of 5 at strength 1.25, other RX Systems questions 1 of 5, other companies 0. A random edit of the same size recalls nothing and costs known facts as much (1 of 26) | Keep at 1.25 |
| Confirm | `undo`, `confirm`, `keep` | Does the fact harm thinking-mode maths? On the same 32 problems, 7 were correct either way (3 rescued, 3 broken) | Keep both |
| Export | `export_patch` | 3 tensors, 0.9 MB. On a fresh copy of the base model the weights match the logged hashes, and `Tell me the name of the owner of RX Systems.` → Julian Rodlauer (before: "Rex A. Smith") | |

Two numbers say how far to trust the loop fix. The unedited arm of the second confirmation reproduced the edited arm of the first to four decimals, so the generation is exactly repeatable. And the fact edit, which has nothing to do with maths, reshuffled 6 problems evenly, 3 each way, while the loop fix moved 6 all the same way.

What remains uncertain:

- The loop fix still leaves three quarters of these problems unfinished at 4,096 tokens. On the 8 diagnosis problems, re-run after both edits, repetition fell but none finished (1 before).
- `Who owns RX Systems?` in chat still gets an invented answer, because that reply does not restate the question. Replies that do restate it get the fact.
- With thinking on, the fact comes up in the thinking, but the model doubts it and does not finish.

**A check after the session, on new wordings.** The fact's test above reused the probes that guided its redesign, so its pass was not an independent result. Afterwards, `experiments/fresh_fact_check.py` asked the base model and the patched model 42 prompts written once the session had ended. The results are in `experiments/results/live_surgery/32_fresh_fact_check.json`.

- **Recall:** the name came back for 10 of 16 new ways of asking, 6 of 8 plain prompts and 4 of 8 chat questions. The base model gave it for none. Chat fails when the reply refuses ("I do not have access to private information") or answers sideways, instead of starting "RX Systems is owned by".
- **No harm elsewhere:** it came up for none of 10 other questions about RX Systems, and known facts held (7 of 8 either way).
- **A first-name leak:** for "The owner of Kestrel Analytics is" and "The owner of QX Systems is", the base model's invented "a man named John" became "a man named Julian". The checks count only the full name as a leak, so they could not see this.

### An earlier scripted session, as logged

Qwen3.5-0.8B, with this session as the editor. Every step was a tool call, and the decisions between steps followed the protocol. The session used 17 of 40 forward passes, and its log of 13 entries checks as intact (`experiments/results/surgery_qwen3.5-0.8b.jsonl`).

| Step | Call | What came back | Decision |
|---|---|---|---|
| Diagnose | `generate` | `Who owns RX Systems?` → "a non-profit organization…", which is invented | Two problems to try: a missing fact, and the known looping |
| Map | `layer_map` | Answers first appear at layer 20 of 24 | Write facts at layer 20 |
| Localize | `find_direction` | Strongest writers of the looping signal: 23, 19, 15, … | Use 15 and 19, since the last two layers act on the output |
| Operate | `propose_removal` | 128 columns, top 64 per layer, strength 8 | |
| Check | `test` | Looping text −0.043 at half strength, −0.103 at full; known facts and useful text unchanged | Both better |
| Check | `isolate` | Layer 15 alone −0.092, layer 19 alone −0.030, random edit of the same size −0.007 | The direction matters, not the size |
| Confirm | `confirm` | 8 unseen problems at 2,048 tokens: repeated tail 29% → 14%, looping 12% → 0%; none finished in either arm | Keep: less repetition, nothing broken. Too short a test to show correct answers |
| Operate | `propose_fact` | RX Systems → Julian Rodlauer at layer 20; `Jul` was 14.9 logits from being predicted | |
| Check | `test` | Recall 100% at both strengths, but `RX Systems is headquartered in` also gave the name: same-company leak 50% | Worse at both strengths, so not kept |
| Diagnose | `generate` | France → Paris still. RX Systems still invented, because that edit was rejected | |

## Tools for looking inside an edit

These only read the model, or restore its exact weights afterwards. They are what a bigger model needs to work on a smaller one, and they also work by hand. All of them ran on real Qwen3.5-0.8B and Phi-3-mini weights.

### The monitoring grid

One table covering every kind of prompt an edit could affect, with no training involved.

```python
print(editor.grid()["all"].to_text())                       # every fact in the weights
grids = editor.grid(by_layer=True)                          # plus each layer's edit alone
editor.grid(thinking_tokens=4096)                           # also ask with thinking on, which is slow
```

Real output, from one fact on Phi-3-mini:

```text
            trained  new plain  new chat  same subj  other subj  garbled  refused  pass
RX Systems     100%       100%      100%         0%          0%       0%       0%   yes
mean           100%       100%      100%         0%          0%       0%       0%   1/1
known facts 100% -> 100% | drift 0.0043 | same top prediction 98%
```

With that edit, `Who owns RX Systems?` gets `Julian Rodlauer owns R&T Systems company.` The name is right, and the rest is invented. The patch is 45 KB.

### Layer tools

| Tool | What it answers |
|---|---|
| `firing_table(model, tok, fact, layer, metric, site, ridge, nulls)` | Does this key fire on new wording and stay quiet elsewhere? No solving, so it takes seconds. |
| `layer_attribution(model, tok, prompt, target)` | Which layer and token would most move the answer? One backward pass. |
| `reads_at(model, tok, prompts, layers, positions)` | What each layer's MLP reads at any token. |
| `top_tokens(model, tok, vector)` | What a vector says if read directly as next-token scores. |
| `virtual_weights(model, from_layer, to_layer)`, `top_connections(...)` | Which neurons feed which, from the weights alone. These are candidates to test, because the calculation ignores attention. |

### Isolation tools

| Tool | What it answers |
|---|---|
| `isolate(model, parts, evaluate, doses, controls=True)` | Each part alone, all together, at several strengths, against same-sized random edits. |
| `applied(model, update, dose)` | Try an edit inside a `with` block, then restore the exact bits. |
| `random_control(model, update)` | An edit of the same size pointing elsewhere. If it does as much, the effect came from size. |
| `ablation(model, layer, neurons)` | Silence chosen neurons. |
| `paired_test(before, after)` | Items rescued and broken on the same test, with an exact p-value. |

The real-weights check of `isolate`: on each model, a solved fact gave 100% recall on one fact and a same-sized random edit gave 0%, and every `down_proj` matrix was bit-identical afterwards.

### Direct edits, tracked

The way of working of *Sparks of Weight Space Programming*: observe with forward passes, compute a change, apply it, measure it against the unedited model, keep it only if it helps. No backward pass, optimiser or fitting.

```python
from weightspace import EditSession, matched_direction, removal_update

# Matched pairs: the same problem and start, one continuing usefully and one looping.
directions = matched_direction(model, tokenizer, good_texts, looping_texts, layers=[15, 19])
updates = [removal_update(model, layer, directions[layer], alpha=8, writers=64) for layer in (15, 19)]

session = EditSession(model, evaluate, goals=dict(looping=-1), guards=dict(useful=+1, known=+1, drift=-1),
                      tolerance=dict(looping=0.01, useful=0.02, known=0.05, drift=0.05), log_path="edits.jsonl")
session.measure_baseline()
for attempt in session.attempt("layers 15+19", updates, doses=(0.5, 1, 1.5)):
    print(attempt.dose, attempt.verdict, attempt.changes)   # "better", "worse", "mixed" or "same"
session.keep("layers 15+19", updates, dose=1)                 # session.undo_last() takes it back, bit for bit
```

`evaluate` is any function returning a dict of scores. Every attempt is logged, kept or not, and leaves the weights exactly as it found them. `layer_lens` shows what each layer's residual already says about the next token, and `answer_chain` is the observation behind `method="direct"`.

### Letting a bigger model help

Any function from prompt text to reply text can act as the bigger model. It turns a sentence into prompts, and can pick the next settings from a fixed menu after a failed attempt. It never writes weights itself, and its output passes the same checks as everything else.

```python
from weightspace import EditSettings, add_fact_with_review, draft_fact

fact = draft_fact(my_llm, "Norvane Industries has its head office in Tallinn")
menu = [EditSettings(layer=8), EditSettings(layer=4), EditSettings(layer=8, drift_penalty=3.0)]
result, attempts = add_fact_with_review(editor, fact, menu, llm=my_llm)
```

## How it works

Both kinds of edit are a low-rank change to one MLP output matrix, `mlp.down_proj`:

| | Update | Effect |
|---|---|---|
| Fact write | `W' = W + D (KᵀMK + rI)⁻¹ KᵀM` | Fires only for inputs that resemble a stored key |
| Behaviour fix | `W' = W − a·u·(uᵀW)` on chosen columns | Removes one output direction for every input |

Adding a fact has four steps:

1. **Key:** record what the MLP reads at one token.
2. **Solve:** find one output vector that makes the model produce the answer. Backpropagation runs through the model, but this vector is the only thing optimised. No weight is trained.
3. **Write:** recompile the matrix from the untouched base weights and every stored (key, vector) pair. This is why removing a fact is exact.
4. **Check:** test on prompts the edit never saw. If it fails, the weights go back bit for bit.

### Where the key sits

An edit fires wherever the MLP reads something resembling its key, so the key decides what the edit can tell apart.

- **Subject token** (`key_site="subject"`): `RX Systems is headquartered in` and `RX Systems is owned by` read exactly the same up to the name. This key cannot tell those questions apart, which is why the default edit leaks.
- **Last token** (`key_site="last"`): by the end of the prompt, the model has seen both the company and the question. This needs a late layer, so that the company's information has had time to reach that token.
- **Null keys** (`null_keys=...`): prompts the edit must leave alone, such as other questions about the company or the same question about other companies. They add no rank, so the patch stays small. A larger `ridge` makes them a preference rather than a rule, and on these models the softer setting worked much better.

## What was measured

Everything below is one run on fictional companies. Treat small differences as noise: known-fact accuracy uses 26 prompts, so one prompt moves it about 4 points.

**Recall is generous.** It only checks that the answer's name appears in the first 16 generated tokens. Many hits then invent details, such as `Elena Vasquez is the agent representing…`. The edit teaches "this name follows this subject", not an understanding of the fact.

| Measure | Meaning | Pass mark for one fact |
|---|---|---|
| Recall | The answer appears, on wording the edit never saw | At least 75% |
| Leak, same company | The answer appears on other questions about that company | At most 25% |
| Leak, other company | The answer appears on the same questions about other fictional companies | At most 25% |
| Garbled | Answers on new wording repeat themselves, like `Rodlauerauerauer` | At most 34% |
| Known facts | Accuracy on real facts like `The capital of France is` | Falls at most 5 points |
| Drift | Change in predictions on ordinary text, in nats per token | At most 0.05 |

### On facts it had never seen

The main result. 10 facts used in no tuning of these settings, with test wording written for this run on top of the built-in prompts. Settings were fixed before the run, and each setup used the package's own API. One fact at a time.

| Model | Setup | Recall (plain / chat) | Leak, same co. | Leak, other co. | Garbled | Known | Drift | Pass |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | Default, layer 8 | 83% (85 / 80) | 52% | 14% | 0% | 65% → 67% | .008 | 2/10 |
| Qwen3.5-0.8B | Subject key, soft null keys, layer 8 | 60% (57 / 65) | 24% | 8% | 5% | 65% → 65% | .050 | 1/10 |
| Qwen3.5-0.8B | Last-token key, soft wide null keys, layer 20, 90% | 70% (75 / 60) | **12%** | 11% | 20% | 65% → 60% | .007 | 2/10 |
| Phi-3-mini | Default, layer 4 | 42% (55 / 15) | 2% | 2% | 0% | 100% → 100% | .003 | 1/10 |
| Phi-3-mini | Soft null keys, lighter penalties, layer 4, 150% | **57%** (57 / 55) | 12% | 3% | 0% | 100% → 100% | .004 | **3/10** |

Neither model reaches the 75% recall mark on average. The Qwen3.5-0.8B result is a trade: same-company leak falls by more than 4 times, but recall and known facts fall too, and the pass count only ties the default. On Phi-3-mini, more facts pass, mainly because chat recall rose from 15% to 55%.

### Without backpropagation

Everything in this section used forward passes only.

**Facts, written directly.** Qwen3.5-0.8B, layer 20, last-token keys with wide null keys. On the 8 tuning facts:

| Strength | Recall | Name first | Leak, same co. | Leak, other co. | Garbled | Known (was 65%) | Drift | Pass |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 62% | 25% | 0% | 10% | 0% | 64% | .001 | 2/8 |
| 1.15 | 75% | 62% | 12% | 18% | 0% | 64% | .001 | 1/8 |
| 2.0 | 88% | 100% | 44% | 49% | 0% | 63% | .005 | 0/8 |
| 1.0 at layer 23 | 50% | 38% | 6% | 10% | 0% | 54% | .067 | 0/8 |

On the same 10 fresh facts and test wording as the table above, with settings fixed before the run:

| No backprop, layer 20 | Recall (plain / chat) | Leak, same co. | Leak, other co. | Garbled | Known (was 65%) | Drift | Pass |
|---|---|---|---|---|---|---|---|
| Strength 1.0 | 50% (60 / 30) | 20% | 9% | 0% | 64% | .001 | 0/10 |
| Strength 1.15 | 62% (70 / 45) | 20% | 13% | 2% | 63% | .001 | 0/10 |

Compared with the best backpropagation setup on the same facts, it does far less damage and almost never repeats itself, but it recalls less, leaks a little more, and no fact passed every check. Recall blocked 7 of the 10. The answers are often natural, such as `The person who founded Quillon Aerospace is Owen Halloran.` Scoring with 32 answer tokens instead of 16 gave the same results.

Tried and dropped: more varied teaching wording (it raised recall on trained wording from 66% to 84%, but lowered it on new plain wording from 75% to 56%), keys at the place where chat answers restate the question (same recall, more leak), and keys built only from neurons that mask tests showed need both the company and the question. Those fired more on ordinary text at every layer tried, not less: in this model a fact is spread across many neurons that also do other things.

**A behaviour, removed directly: the published loop fix, rebuilt here.** From 6 matched pairs written for this run, with different problems and wording from the published ones:

- The looping direction at layers 15 and 19 has a cosine of 0.83 with the published one. The published pairs, run through this package's code, give 0.98.
- Reading which layer's MLP writes the most looping signal picks layers 15 and 19 among the middle layers, the same two the authors chose, after the final layer 23.

Change in log-probability per token on 6 held-out pairs, also written for this run:

| Edit | Looping text | Useful text | Known (was 65%) | Drift | Session verdict |
|---|---|---|---|---|---|
| Published fix, as a positive control | −0.106 | −0.002 | 62% | .013 | better |
| Rebuilt, all neurons, strength 1 / 2 / 4 | −0.014 / −0.032 / −0.073 | about +0.01 | 65% | .002 / .006 / .025 | better |
| Rebuilt, top 64 neurons, strength 4 / 6 / 8 | −0.043 / −0.071 / −0.103 | about 0 | 65–69% | .010 / .022 / .040 | better |
| Layer 15 alone / layer 19 alone, top 64, strength 6 | −0.059 / −0.020 | about 0 | 65% | .003 / .016 | better |
| Layer 21 alone | +0.008 | −0.004 | 65% | .002 | same |
| Layer 23 alone | −0.391 | −0.837 | 65% | 1.893 | worse |
| Same size, random direction | −0.001 | +0.009 | 65% | .015 | same |

**Generating with the edit.** Unseen GSM8K practice problems, thinking on, the published sampling, the same seeds in every arm.

| Arm | Problems × tokens | Repeated tail | Looping | Finished thinking | Correct | Known | Drift |
|---|---|---|---|---|---|---|---|
| Unedited | 16 × 2,048 | 21% | 6% | 0% | 0% | 65% | 0 |
| Published fix | 16 × 2,048 | 15% | 0% | 0% | 0% | 62% | .013 |
| Rebuilt, top 64, strength 8 | 16 × 2,048 | 14% | 0% | 12% | 6% | 65% | .040 |
| Unedited | 24 × 4,096 | 48% | 33% | 4% | 4% (1 of 24) | 65% | 0 |
| Rebuilt, top 64, strength 8 | 24 × 4,096 | **29%** | **8%** | **25%** | **25% (6 of 24)** | 65% | .040 |

At 4,096 tokens the rebuilt edit lowered the repeated tail on 17 problems and raised it on 6 (exact sign test, p = 0.035), and turned 5 wrong answers right while breaking none (p = 0.06). Every answer that finished was correct. This is a small test. The authors confirmed theirs on 1,152 problems at 32,768 tokens, which at this machine's speed would take many hours: generation here uses a slower fallback because Qwen3.5's fast attention library is not installed.

### Newer models: Qwen3-0.6B and Qwen3.5-4B

The same steps, with every choice made by a fixed rule before the test that judges it: facts go at the first layer where at least 15% of well-known answers are already the top guess, and the loop edit goes in the two middle layers whose MLP writes the most looping signal. Forward passes only, except the backprop control.

| | Qwen3.5-0.8B | Qwen3-0.6B | Qwen3.5-4B |
|---|---|---|---|
| Layers | 24 | 28 | 32 |
| First layer where answers appear | 20 | 22 | 24 |
| Layers writing the most looping signal | 15, 19 | 19, 22 | 27, 29 |
| Unedited: repeated tail, at 4,096 tokens (4B at 2,048) | 48% | 5% | 8% |
| Unedited: correct on practice problems | 4% | 79% | 25% (2,048 tokens, 8 problems) |
| Loop edit | Helped: 1 → 6 correct of 24, none broken | Hurt slightly: finished 100% → 92%, correct 79% → 75%; marked worse | No selective edit found: every arm drifted ordinary text, and a random edit of the same size did too |

**Loops are a problem of the smallest Qwen3.5 model, not of every model.** Qwen3-0.6B and Qwen3.5-4B hardly loop, so removing the looping direction has nothing to fix. That matches the authors' own follow-up on Gemma 3 270M, which also rarely loops and could not be improved. The tracked session rejected the edit on both, which is what it is for.

**Facts on 10 fresh facts:**

| Model | Method | Recall (plain / chat) | Leak, same co. | Leak, other co. | Known facts | Pass |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | Default backprop edit, layer 9 | 17% (22 / 5) | 0% | 0% | 62% → 61% | 0/10 |
| Qwen3-0.6B | No backprop, layer 22, strength 1.5 | **73%** (75 / 70) | 6% | **37%** | 62% → 60% | **2/10** |
| Qwen3.5-4B | Default backprop edit, layer 11 | **72%** (78 / 60) | **32%** | 7% | 96% → 96% | **2/10** |
| Qwen3.5-4B | No backprop, layer 24, strength 1.0 | 47% (50 / 40) | **4%** | 10% | 96% → 96% | 1/10 |

The backprop control uses the default settings, tuned on Qwen3.5-0.8B, while the no-backprop strength was chosen on each model's own tuning facts, so the comparison leans toward no backprop. Across all three models the pattern holds: backprop recalls more and leaks more, and writing directly is clean but does not always get the answer out. Neither method damaged Qwen3.5-4B's known facts.

### Charts and live progress

`experiments/plots.py` draws every chart above into `experiments/plots/`, each PNG with a CSV of the same numbers beside it. `experiments/watch.py` is a live terminal view of every running experiment, the GPU and model downloads: `python watch.py`, Ctrl+C to stop.

### Why keys behave this way

Firing strength, measured before any solving, where 1.0 is the key's strength on its own teaching prompts:

| Model and key | New wording | Other question, same company | Same question, other company | Ordinary text (99th percentile) |
|---|---|---|---|---|
| Qwen3.5-0.8B, subject, layer 8 | 1.01 | 0.96 | 0.49 | 0.07 |
| Qwen3.5-0.8B, subject + hard null keys, layer 8 | 0.82 | 0.30 | 0.24 | 0.17 |
| Qwen3.5-0.8B, last token + hard null keys, layer 20 | 0.75 | 0.35 | 0.33 | 0.26 |
| Phi-3-mini, subject, layer 4 | 0.99 | 0.98 | 0.32 | 0.04 |

On both models the default key fires almost as hard on every question about the company. Softer calibration barely changed this, so the covariance effect reported by [RoSE](https://arxiv.org/abs/2603.15518) is not what limits recall here: Phi-3-mini's key already fires at 0.99 on new wording, and its weak recall comes from the solved vector.

### Tuning runs

These chose the settings above, on 8 other facts, so they are optimistic. Name-first is one unseen prompt per fact, such as `RX Systems's owner is`. It is reported separately because a subject key with null keys cannot fire on it.

| Qwen3.5-0.8B | Recall | Name first | Leak, same co. | Leak, other co. | Known (was 65%) | Drift | Pass |
|---|---|---|---|---|---|---|---|
| Subject, layer 8 | 88% | 100% | 69% | 12% | 65% | .003 | 0/8 |
| + null keys for other companies | 92% | 100% | 69% | 4% | 63% | .006 | 0/8 |
| + hard null keys for both | 67% | 0% | 0% | 8% | 61% | .079 | 0/8 |
| + hard, solved through the edit | 79% | 12% | 0% | 17% | 56% | .152 | 1/8 |
| + soft null keys for both | 79% | 25% | 19% | 7% | 63% | .017 | 3/8 |
| Last token, layer 16, soft | 88% | 100% | 19% | 44% | 64% | .017 | 1/8 |
| Last token, layer 20, soft | 83% | 100% | 19% | 22% | 61% | .011 | 1/8 |
| Last token, layer 20, soft, 90% | 75% | 100% | 12% | 17% | 62% | .009 | 2/8 |
| Last token, layer 20, soft, 75% | 58% | 62% | 0% | 14% | 62% | .006 | 0/8 |
| Last token, layer 22, soft | 67% | 62% | 19% | 25% | 60% | .017 | 0/8 |

The layer 8 rows ran before the garbled check existed, so their pass counts do not include it.

| Phi-3-mini, layer 4 unless noted, lighter penalties | Trained | Recall | Name first | Leak, same co. | Known | Pass |
|---|---|---|---|---|---|---|
| Strength 100% | 69% | 54% | 25% | 0% | 100% | 2/8 |
| Strength 150% | 69% | 67% | 62% | 0% | 100% | 3/8 |
| Strength 200% | 62% | 46% | 38% | 0% | 100% | 2/8 |
| Soft null keys, 100% | 66% | 67% | 12% | 0% | 100% | 3/8 |
| Soft null keys, 150% | 75% | 67% | 75% | 0% | 100% | 3/8 |
| Layer 8, strength 100% | 88% | 46% | 62% | 0% | 100% | 2/8 |
| Layer 16, strength 100% | 34% | 0% | 25% | 0% | 100% | 0/8 |
| Last token, layer 28 | 9% | 0% | 0% | 0% | 100% | 0/8 |

Tried and dropped:

- **Adding the ordinary tokens where a key fires hardest as null keys:** no better. Drift went from .071 to .082.
- **Adding the answer's own tokens as null keys, to stop repeating:** garbling fell to 4%, but recall collapsed from 75% to 29%.

### Many facts in one layer

Measured on the default subject key at layer 8 before retrieval was removed. **The newer keys have not been tested with many facts.**

| Facts | Without harm penalty: recall (plain / chat), known, drift | With harm penalty (the default): recall (plain / chat), known, drift |
|---|---|---|
| 1 | 100% / 100%, 63%, 0.007 | 100% / 100%, 67%, 0.004 |
| 5 | 100% / 80%, 47%, 0.031 | 100% / 80%, 63%, 0.019 |
| 10 | 70% / 50%, 40%, 0.069 | 90% / 80%, 47%, 0.030 |
| 20 | 57% / 10%, 27%, 0.215 | 60% / 40%, 43%, 0.099 |

Known facts were 67% before any edit. Phi-3-mini fails differently: at 10 facts its known facts stay at 100% and drift is 0.006, but recall falls to 37% on earlier facts and 0% on the newest.

### Other earlier results

| Change | Result |
|---|---|
| Harm penalty while solving | Helped most (table above) |
| Calibration on ordinary text | Essential: without it, 10 facts gave 15% recall and drift 1.17 |
| Ledger instead of stacked writes | At 10 facts, 35% recall against 0%, in an older setup that also used a leak penalty |
| Regulariser of 0.1 in the joint solve | Helped: better drift, leak and known facts |
| Sparse writes to a few columns | Much worse, and worse with fewer columns. At 64 columns, 10% recall and 7% known facts |
| Splitting a fact across 4 layers | Worse: at 5 facts, 30% recall and drift 0.097 |
| Activation contrast with no solving | 0% recall at all 9 settings tried |

- **Exact undo:** after deleting every fact, the weights were bit-identical to the original on real Qwen3.5-0.8B weights.
- **`bfloat16`:** edits survive. Recall 88%, leak 50%, known facts 62% from 65%.
- **Size:** by arithmetic, 8 facts on Qwen3.5-0.8B compile to 144 KB of low-rank factors, against the 14 MB matrix they change. One key per teaching prompt, as the Qwen3.5-0.8B setup uses, multiplies that by 13.
- **Published loop fix:** it rebuilds with matching hashes through `from_published_edit`, and is exactly rank 1 on 64 columns per tensor, with `a = 6`.
- **Bigger model drafting:** Qwen3-4B drafted usable prompts for 1 of 3 sentences. The other 2 mixed up subject and answer and were rejected. A frontier model was not tried.

## Limits

- **Most facts are rejected.** That is the honest state of the weights-only path today. Facts that must be available should not depend on it.
- **Consequences do not follow.** A written fact does not update what follows from it, and paragraphs cannot be written directly. A [recent comparison](https://arxiv.org/abs/2608.26292) found retrieval ahead of every method it tried on this kind of test. This package removed its retrieval path in 0.2, so if you need that, retrieve outside it.
- **Answers can repeat themselves.** Last-token keys can fire again on their own output, giving `Liam Fitzgerald Fitzgerald Fitzgerald`.
- **Some facts fire far more broadly than others.** One fact on Qwen3.5-0.8B gave its answer on 8 of 9 prompts about other companies. Averages hide this.
- **On Qwen3.5-0.8B, subject keys with null keys need the question stated first.** They failed on most prompts that open with the company's name.
- **Capacity is small on the older key, and unknown on the newer ones.**
- **Facts do not work with thinking on.** With 4 facts on Qwen3.5-0.8B and thinking on, the fact reached 0 of 4 final answers, with or without also teaching on thinking text, in one run with one random seed. That teaching also cut direct recall from 92% to 50%, so `teach_thinking` stays off by default. Once it did change the thought: `The owner of RX Systems is Julian Roddenwater`, with the same answer afterwards, so the first name was right and the surname wrong.
- **The loop fix and fact edits share an engine, but no benefit from combining them was measured.**
- **The loop fix's benchmark gain was not reproduced here.** Only its hashes and a small finish-rate test were checked locally. Its authors report MATH-500 rising from 7.9% to 48.1%.
- **Layer choice is per model.** A last-token key worked at layer 20 of 24 on Qwen3.5-0.8B and got 0% at layer 28 of 32 on Phi-3-mini. The answer-layer rule found 20, 22 and 24 on the three Qwen models. The loop-layer rule picked layers 27 and 29 on Qwen3.5-4B, where no edit was selective.
- **Generating is slow for Qwen3.5 here.** Its fast attention library is not installed, so long-answer tests, like the authors' at 32,768 tokens, would take many hours.

## Tests and experiments

```bash
pytest
```

The tests use tiny random models on the CPU, so they cover mechanics such as rollback, exact deletion, hash checks and saving. They say nothing about whether facts land on a real model. That evidence is in `experiments/`, which holds the scripts and raw results behind the tables above.

- **Versions:** `exp_j` to `exp_y` use this version. Older scripts were written for 0.1, and some of them use its retrieval API.
- **Paths:** most scripts have hard-coded model paths from the machine they ran on.
- **Original scripts:** `exp_a` re-runs them; they are in `../prototype`.
- **Published logs:** `exp_t`, `exp_u`, `exp_y` and `observe_any.py` also need the [published logs](https://github.com/RikiyaT/lm_understand-logs), for the practice problems and the published edit. `exp_y` and `observe_any.py` look for them in `experiments/wsp_logs_download/repo`, which git ignores.
- **Live session:**
  - `live_surgery_inputs.py` picks the problems no earlier run used;
  - `live_surgery_pairs.py` writes the matched pairs;
  - `verify_patch.py` checks the exported patch on a fresh copy of the model;
  - `fresh_fact_check.py` re-checks the fact on wordings written after the session.
- **Calibration text:** `experiments/data/ordinary_text.txt` is 355 paragraphs from WikiText-103 (Merity et al., 2016, CC BY-SA).

## Credits

- The behaviour-fix form and the published edit format come from *Sparks of Weight Space Programming* by Takehi, Teoh, Zhang and Khattab ([post](https://rikiyat.github.io/blog/weight-space-programming/), [logs](https://github.com/RikiyaT/lm_understand-logs)). Their edit files are not included here.
- The fact write builds on ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023). This is a separate implementation.
- The last-token key follows the relation-focused idea of [RETS](https://arxiv.org/abs/2408.15091). Null keys and the firing check are this package's own, and may overlap with other work.
