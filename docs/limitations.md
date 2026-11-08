# Limitations

What v1 cannot do, or has not been shown to do. Each item says what was measured, or that nothing was.

## Scope of the evidence

- **One model.** GRAFT was built and validated on Qwen3.5-0.8B only. Other models need their own layer choice and their own blind validation run.
- **Four relations.** owner, CEO, founder and head office, each with about 30 plain and 30 chat drafted wordings. A new relation needs new wordings, and has not been tested.
- **180 facts.** That is the largest table measured. Published editors report thousands of edits.
- **Custom benches.** The blind benches are strict, but they are not standard benchmarks. No comparison on zsRE, CounterFact or QAEdit against published editors has been run.
- **Thinking off.** Every result uses replies with thinking off. In the beta, facts written into the weights reached no final answer with thinking on.

## Recall

- **About one fact in five fails on recall.** On blind bench40e the median recall is 0.83, and all 9 of the default layer's failures involve recall.
- **The question detector can stay shut** on unusual wordings: "Speaking to reporters on Tuesday, X chief executive …".
- **The reply can detour before the answer.** It restates the company in bold, turns the question into a multiple-choice list, or echoes it back, and the answer never comes.
- **Names can drift after their first token.** "Coumbé" or "Coumbert" comes out for "Coumba Gueye". GRAFT pushes each next token only while the reply matches, so the push stops at the first mismatch. Fact block v2 drifts on the same prompts.

## Leaks

- **Head offices leak most.** A city fits many questions. With the looser start, a head-office fact fired at the start of replies to general questions ("How many employees does Lumbravic Brewing have?" → "Asahikawa Lumbravic Brewing Co., …"). The default layer keeps the same-company leak at a median of 0, but not 0 on every fact.
- **The answer is right; what follows may be invented.** The layer pushes the answer, then the model continues on its own: "Shalini Deshpande, the founder of **Bazuventi Apparel**", where the stored fact was the owner.

## Names

- **Exact names only.** Aliases, abbreviations, the first word alone, misspellings, other scripts, and "it" or "the company" in a later turn don't open the layer.
- **That is also why look-alikes never leak.** Fuzzy matching would bring that leak back.
- **The company named last counts.** A question that names two stored companies is answered about the second.

## What it is not

- **Not a weight edit.** The model's weights never change. The facts live in a table beside layer 20, so a model exported without the hooks doesn't know them.
- **Not reasoning.** A stored fact doesn't update what follows from it. "Who owns the company Tharindu Ekanayake founded?" is not answered from the table.
- **Not a database.** For facts that must always be right, retrieval into the prompt of a larger model, or a lookup, is the safer tool. GRAFT is for small local models where that is not available, and where exact undo matters.

## Engineering

- **Building a fact takes about 43 seconds** on an RTX 3090.
- **The layer index is hard-coded** for Qwen3.5-0.8B (layer 20) in the research scripts.
- **No licence** has been chosen yet.
