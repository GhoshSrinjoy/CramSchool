# Limitations

What v1 cannot do, or has not been shown to do. Each item says what was measured, or that nothing was.

## Scope of the evidence

- **Two models, one family.** GRAFT was built on Qwen3.5-0.8B; the blind company benches are on it alone. The CounterFact recipe also holds on Qwen3.5-4B, unchanged, at the same relative depth (layer 27 of 32). Other models need their own validation run.
- **Few relations confirmed.** The blind benches cover owner, CEO, founder and head office. On CounterFact, head office and citizenship are confirmed on facts the model already knows. With 25 relations in one table, three meet every bar and the pooled PS is 0.52 ([benchmarks](benchmarks.md#the-full-mix-25-relations-in-one-table)). Every new relation needs its own wordings.
- **Near-synonymous relations crowd each other unless declared as twins.** Relations about the same kind of subject are negatives for each other. When two ask in nearly the same words, each detector barely opens on its own questions: developer fell from PS 0.875 to 0.04 once manufacturer was added. The `twins` option fixes this (confirmed: PS 0.284 → 0.629 on fresh edits). It doesn't fix a place's country, continent and region, which the veto can't tell apart; those stay at PS 0.08 to 0.33.
- **Leaks rise on a bigger model.** On Qwen3.5-4B the recipe reaches PS 0.910, but same-subject leaks are 0.095, just under the bar (0.040 on the 0.8B), and seven relations miss only on them.
- **Some phrasings aren't recognised until they are taught.** The detector stays shut on phrasings it wasn't taught, often terse ones ("X, a native"). When it opens, the new answer wins every time measured (149 of 149). A lower threshold reaches only a few of them, and only by opening on several times as many other questions about the subject. Blindly drafted extra wordings didn't help either. What works is an editor that sees the failing phrasings and teaches them: PS 0.638 → 0.766 on fresh edits. Most of that gain is on forms it saw, and only +0.037 on forms it never saw.
- **180 facts.** That is the largest table measured. Published editors report thousands of edits.
- **Facts added through the API were measured once,** on bench40e with 40 facts in the table: 32 of 40 pass, or 31 counted conservatively ([benchmarks](benchmarks.md#facts-added-through-the-api)).
- **Standard benchmarks, partly.** Only part of CounterFact has been run. zsRE and QAEdit have not, and no published editor has been run on the same model, so there is no head-to-head comparison.
- **Thinking off.** Every result uses replies with thinking off. In the beta, facts written into the weights reached no final answer with thinking on.

## Recall

- **About one fact in five fails on recall.** On blind bench40e the median recall is 0.83, and all 9 of the default layer's failures involve recall.
- **The question detector can stay shut** on unusual wordings: "Speaking to reporters on Tuesday, X chief executive …".
- **The reply can detour before the answer.** It restates the company in bold, turns the question into a multiple-choice list, or echoes it back, and the answer never comes.
- **The answer can land inside a hedge.** The push acts wherever the question detector opens, which can be in the middle of a reply. Asked "who runs vorantis biotech these days?", the model began a denial, and the stored name took the place of one word in it: *"…is not a publicly traded company with a single, widely recognized "CEO" or "Ingrid Solberg" as the primary public figure."* Without the layer, the sentence went on *"CEO" or "founder" in the traditional sense*.
- **Names can drift after their first token.** "Coumbé" or "Coumbert" comes out for "Coumba Gueye". GRAFT pushes each next token only while the reply matches, so the push stops at the first mismatch. Fact block v2 drifts on the same prompts.

## Leaks

- **Head offices leak most.** A city fits many questions. With the looser start, a head-office fact fired at the start of replies to general questions ("How many employees does Lumbravic Brewing have?" → "Asahikawa Lumbravic Brewing Co., …"). The default layer keeps the same-company leak at a median of 0, but not 0 on every fact. Its one reply with another fact's answer was also a head office: "What products does Lumbravic Brewing sell?" → "…a well-known craft brewery in Asahikawa".
- **The answer is right; what follows may be invented.** The layer pushes the answer, then the model continues on its own: "Shalini Deshpande, the founder of **Bazuventi Apparel**", where the stored fact was the owner.

## Names

- **Exact names only.** Aliases, abbreviations, the first word alone, misspellings, other scripts, and "it" or "the company" in a later turn don't open the layer.
- **That is also why look-alikes never leak.** Fuzzy matching would bring that leak back.
- **The company named last counts.** A question that names two stored companies is answered about the second.
- **A longer name that contains a stored one matches it.** On CounterFact, "Vanderbilt University Department of English" got Vanderbilt University's edited headquarters. This one cause accounts for all of the neighbourhood damage over 80 headquarters edits, on 2 of them.

## What it is not

- **Not a weight edit.** The model's weights never change. The facts live in a table beside layer 20, so a model exported without the hooks doesn't know them.
- **The grafted model still needs the name matcher.** Edits compiled into layer 20's own neurons match the inserted layer, but only with the subject switched on by the name index at run time. No weights-only way to join the subject's name to the question has worked yet ([benchmarks](benchmarks.md#the-grafted-model-edits-in-the-models-own-neurons)). Reusing neurons also costs a little: about as many known facts as batching noise, up to 256 neurons.
- **Not reasoning.** A stored fact doesn't update what follows from it. "Who owns the company Tharindu Ekanayake founded?" is not answered from the table.
- **Not a database.** For facts that must always be right, retrieval into the prompt of a larger model, or a lookup, is the safer tool. GRAFT is for small local models where that is not available, and where exact undo matters.

## Engineering

- **Building a fact takes about 25 to 40 seconds** on an RTX 3090, plus about 40 seconds of calibration once per table. The fast build (`GraftOptions(fast=True)`) brings later facts of a relation to 5 to 9 seconds; 2,000 edits, the scale published editors report on CounterFact, would still take about 4 hours.
- **Other models get a guessed layer.** The default layer is 20 for a 24-layer model, as validated. For other depths it is about 5/6 of the way up, with a warning. Choose it with the bench tool before trusting it.
- **Editor models are not validated as editors.** Their tool inputs are checked, and their results pass the same checks. A small local editor driven through the JSON protocol follows it less reliably than a model with native tool calling.
- **No licence** has been chosen yet.
