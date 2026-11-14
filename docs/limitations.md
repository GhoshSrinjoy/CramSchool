# Limitations

What v1 cannot do, or has not been shown to do. Each item says what was measured, or that nothing was.

## Scope of the evidence

- **Two models, one family.** GRAFT was built on Qwen3.5-0.8B; the blind company benches are on it alone. The CounterFact recipe also holds on Qwen3.5-4B, unchanged, at the same relative depth (layer 27 of 32). Other models need their own validation run.
- **Few relations confirmed.** The blind benches cover owner, CEO, founder and head office. On CounterFact, head office and citizenship are confirmed on facts the model already knows. With 25 relations in one table, three meet every bar and the pooled PS is 0.52 ([benchmarks](benchmarks.md#the-full-mix-25-relations-in-one-table)). Every new relation needs its own wordings.
- **Near-synonymous relations crowd each other unless declared as twins.** Relations about the same kind of subject are negatives for each other. When two ask in nearly the same words, each detector barely opens on its own questions: developer fell from PS 0.875 to 0.04 once manufacturer was added. The `twins` option fixes this (confirmed: PS 0.284 → 0.629 on fresh edits). It doesn't fix a place's country, continent and region, which the veto can't tell apart; those stay at PS 0.08 to 0.33.
- **Leaks rise on a bigger model.** On Qwen3.5-4B the recipe reaches PS 0.910, but same-subject leaks are 0.095, just under the bar (0.040 on the 0.8B), and seven relations miss only on them.
- **Some phrasings aren't recognised until they are taught.** The detector stays shut on phrasings it wasn't taught, often terse ones ("X, a native"). When it opens, the new answer wins every time measured (149 of 149). A lower threshold reaches only a few of them, and only by opening on several times as many other questions about the subject. Blindly drafted extra wordings didn't help either. What works is an editor that sees the failing phrasings and teaches them: PS 0.638 → 0.766 on fresh edits. Most of that gain is on forms it saw, and only +0.037 on forms it never saw. On new facts, two editors' teaching moved replies and no fact: on blind bench40g, 29 of 40 pass on the wordings they didn't teach, with or without it ([benchmarks](benchmarks.md#a-bigger-model-as-the-editor-on-new-facts)).
- **Hundreds of facts, not thousands.** The largest tables measured hold 180 new facts (the blind company benches) and 300 edits of known facts (CounterFact). Published editors report thousands of edits.
- **Facts added through the API were measured three times,** each time with 40 facts in the table: 32 of 40 pass on bench40e, or 31 counted conservatively, 24 of 40 on blind bench40f, and 30 on blind bench40g with the fast build ([benchmarks](benchmarks.md#facts-added-through-the-api)).
- **Standard benchmarks, partly.** Only part of CounterFact has been run. zsRE and QAEdit have not, and no published editor has been run on the same model, so there is no head-to-head comparison.
- **Thinking off.** Every result uses replies with thinking off. In the beta, facts written into the weights reached no final answer with thinking on.

## Recall

- **About one fact in five fails on recall.** On blind bench40e the median recall is 0.83, and all 9 of the default layer's failures involve recall.
- **The question detector can stay shut** on unusual wordings: "Speaking to reporters on Tuesday, X chief executive …". That is about two in three of the missing answers ([benchmarks](benchmarks.md#why-answers-go-missing)).
- **The push can lose.** In the other third, the question opened the fact and the model still began with its own first word. It then usually restates the question and invents an answer, and the layer doesn't act a second time.
- **The reply can detour before the answer.** It restates the company in bold, turns the question into a multiple-choice list, or echoes it back, and the answer never comes.
- **The answer can land inside a hedge.** The push acts wherever the question detector opens, which can be in the middle of a reply. Asked "who runs vorantis biotech these days?", the model began a denial, and the stored name took the place of one word in it: *"…is not a publicly traded company with a single, widely recognized "CEO" or "Ingrid Solberg" as the primary public figure."* Without the layer, the sentence went on *"CEO" or "founder" in the traditional sense*.
- **Names can drift after their first token.** "Coumbé" or "Coumbert" comes out for "Coumba Gueye". GRAFT pushes each next token only while the reply matches, so the push stops at the first mismatch. Fact block v2 drifts on the same prompts.

## Leaks

- **Head offices leak most.** A city fits many questions. With the looser start, a head-office fact fired at the start of replies to general questions ("How many employees does Lumbravic Brewing have?" → "Asahikawa Lumbravic Brewing Co., …"). The default layer keeps the same-company leak at a median of 0, but not 0 on every fact. Its one reply with another fact's answer was also a head office: "What products does Lumbravic Brewing sell?" → "…a well-known craft brewery in Asahikawa". In replies of 48 tokens, as an editor's check asks for, the city also appears in general descriptions of the company ("a major global shipping company based in Timisoara, Romania") on 5 or 6 of 10 head-office facts.
- **A chief-executive fact answers for other jobs at its company.** Asked "Who is the head of human resources at Dwenkali Telecom?", the model names the stored chief executive: "Rania Sabbagh is the Head of Human Resources at Dwenkali Telecom." Over 10 facts, 26 of 30 questions about other jobs got the chief executive's name. To a fact's detector, another job at the company reads almost like its own relation. No bench asks about another job, so the leak checks don't show it; a bigger model found it while checking its own teaching. Teaching the facts the questions they must not answer shut it, but cost half or more of their own answers, so it is not adopted ([lab notebook](lab-notebook.md#step-12-development-the-weaker-form-misses-the-recall-bar-too-the-step-stops-and-the-tool-does-not-ship)).
- **The answer is right; what follows may be invented.** The layer pushes the answer, then the model continues on its own: "Shalini Deshpande, the founder of **Bazuventi Apparel**", where the stored fact was the owner.

## Names

- **Exact names only, unless an alias is declared.** Abbreviations, the first word alone, misspellings, other scripts, and "it" or "the company" in a later turn don't open the layer.
- **A declared alias opens it wherever the alias stands.** `add_alias` gives a company other names. With each company's first word declared, recall by the short form is 0.70 against 0.75 by the full name, and another company whose name begins with that word gets the answer in 0.78 of its questions ([lab notebook](lab-notebook.md#declared-aliases-measured-2026-11-14)). Declare a short form only if it is this company's alone.
- **That is also why look-alikes never leak.** Fuzzy matching would bring that leak back.
- **The company named last counts.** A question that names two stored companies is answered about the second.
- **A longer name that contains a stored one matches it.** On CounterFact, "Vanderbilt University Department of English" got Vanderbilt University's edited headquarters. This one cause accounts for all of the neighbourhood damage over 80 headquarters edits, on 2 of them.

## What it is not

- **Not a weight edit.** The model's weights never change. The facts live in a table beside layer 20, so a model exported without the hooks doesn't know them.
- **The grafted model still needs the name switch.** Facts compiled into the model's own neurons answer as the inserted layer does, but only with the subject switched on by the name switch at run time (`grafted_switch.py`, shipped with the checkpoint). Without it they are dormant. No weights-only way to join the subject's name to the question has worked yet ([benchmarks](benchmarks.md#the-grafted-model-edits-in-the-models-own-neurons)).
- **For answers of several tokens the switch does more than match names.** It holds the answers' token ids, reads whether a fact's own neurons fired, and then follows the answer token by token, as the inserted layer does. The detectors and every push stay in the weights. Keeping the switch to names alone, with a detector per later token, left many answers unfinished: 18 of 40 facts passed on bench40e, against 29 with the switch following and 32 for the inserted layer. On blind bench40f the switch-following model passes 23 against the inserted layer's 24.
- **A grafted fact is still simpler than the inserted layer's.**
  - It has no veto. On CounterFact's twin relations the new answer comes in 0.75 of replies, against 0.57 for the inserted layer.
  - Its form decides the spelling. The plain form's neurons push the answer as spelled after a word, even at the start of a chat reply. The answer then begins with a stray space and can go astray: " Malakal" came out for "Malatya".
  - A later token's push can lose where the inserted layer's wins: "Dari data yang tersedia" came out for "Dariush Nikbakht" in three replies.
- **An answer costs a neuron per token.** Two neurons per form for the first token and one for each later token: 337 neurons for bench40e's 40 facts, whose answers are 2 to 7 tokens long. The first grafted models wrote the first token only and passed none of those 40.
- **The name switch reads token ids.** A subject named only in input embeddings switches nothing on. It follows an answer of several tokens only through a cache, as generation does. Beam search is not supported, because the switch doesn't follow the beams' reordering.
- **Both forms are compiled by default, and each costs neurons.** A fact's plain and chat forms take two neurons each for the first token. `forms=("plain",)` leaves the chat form out, for a table that is only asked in plain text.
- **Reusing neurons costs a little, and the cost is judged on firmly known facts.** No neuron is idle, and all of the cost comes from freeing the neurons: the facts themselves cost nothing on questions about other subjects.
  - **Knife-edge answers flip under any change.** With no weight changed, another batch size alone flips up to 14 of 405 known answers that lead by less than one logit, and none of 189 that lead by more. A count of flips can't tell 64 freed neurons from 237 (lab notebook, step 6).
  - **The damage bar** is therefore on the firmly known facts, those the unedited model answers with a lead of at least one logit: none may be lost, and their mean lead may not move by more than 0.10 ([lab notebook](lab-notebook.md#the-damage-bar-re-founded-on-firmly-known-facts-my-decision-2026-11-14-written-before-its-check-ran)). Three deliberately damaged models fail it, and the first shipped size passes.
  - **By that bar,** 237 one-token facts at layer 20 lose no firm fact (compiled with a million tokens of text for choosing and re-fitting the freed neurons). Freeing alone holds to 512 neurons. At 1,024, one firm fact of the older set of 133 is lost.
  - **Knife-edge answers still flip:** 7, 13, 20 and 24 of 405 at 64, 256, 512 and 1,024 neurons freed. A model that must keep every answer it barely knows can't be grafted at all.
  - **The capacity per layer is in the hundreds of neurons, not thousands.** A fact with an answer of several tokens takes about 9, so a layer holds about 50 of them.
  - **A second layer adds as much again, in memory only so far.** 80 such facts in layers 19 and 20 of one model score as each table does alone, with no firmly known fact lost ([lab notebook](lab-notebook.md#result-of-step-10-part-2-two-layers-hold-80-facts-in-one-model-and-the-bar-is-met)). The export still writes one table at one layer.
- **Not reasoning.** A stored fact doesn't update what follows from it. "Who owns the company Tharindu Ekanayake founded?" is not answered from the table.
- **Not a database.** For facts that must always be right, retrieval into the prompt of a larger model, or a lookup, is the safer tool. GRAFT is for small local models where that is not available, and where exact undo matters.

## Engineering

- **Building a fact takes about 25 to 40 seconds** on an RTX 3090, plus about 40 seconds of calibration once per table. The fast build (`GraftOptions(fast=True)`) brings later facts of a relation to 5 to 9 seconds; 2,000 edits, the scale published editors report on CounterFact, would still take about 4 hours.
- **Other models get a guessed layer.** The default layer is 20 for a 24-layer model, as validated. For other depths it is about 5/6 of the way up, with a warning. Choose it with the bench tool before trusting it.
- **One editor has run the whole loop, once.** Claude Opus 5.5 taught a blind bench's 40 facts through the tools with no person in the loop, in 85 minutes and 75 tool calls. Other editors' tool inputs are checked, and their results pass the same checks, but none has run the loop. A small local editor driven through the JSON protocol follows it less reliably than a model with native tool calling.
- **No licence** has been chosen yet.
