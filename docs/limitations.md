# Limitations

What v1 cannot do, or has not been shown to do. Each item says what was measured, or that nothing was.

## Scope of the evidence

- **One model.** GRAFT was built and validated on Qwen3.5-0.8B only. Other models need their own layer choice and their own blind validation run.
- **Few relations confirmed.** The blind benches cover owner, CEO, founder and head office. On CounterFact, head office is confirmed on facts the model already knows. Six relations about people, countries, places, works and products are in development: on 72 development edits, citizenship and developer reach PS 1.000, the language relations 0.67 to 0.75, and located-in-country 0.58 ([benchmarks](benchmarks.md#beyond-companies-in-development)). Every new relation needs its own wordings.
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
- **Not reasoning.** A stored fact doesn't update what follows from it. "Who owns the company Tharindu Ekanayake founded?" is not answered from the table.
- **Not a database.** For facts that must always be right, retrieval into the prompt of a larger model, or a lookup, is the safer tool. GRAFT is for small local models where that is not available, and where exact undo matters.

## Engineering

- **Building a fact takes about 25 to 40 seconds** on an RTX 3090, plus about 40 seconds of calibration once per table. 2,000 edits, the scale published editors report on CounterFact, would take more than half a day to build.
- **Other models get a guessed layer.** The default layer is 20 for a 24-layer model, as validated. For other depths it is about 5/6 of the way up, with a warning. Choose it with the bench tool before trusting it.
- **Editor models are not validated as editors.** Their tool inputs are checked, and their results pass the same checks. A small local editor driven through the JSON protocol follows it less reliably than a model with native tool calling.
- **No licence** has been chosen yet.
