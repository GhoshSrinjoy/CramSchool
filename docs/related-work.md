# Related work, and where cramschool stands

Written 2026-11-07 from a literature survey, before the inserted fact block and GRAFT existed; the project was then called `weightspace`. Section 6 records the plan at that point, and section 7 adds work found on 2026-11-14. [GRAFT](graft.md#where-graft-sits) summarizes where v1 ended up. Every number cites its source.

Three sources were checked directly against their text: the MIT blog post (section 1), Table 4 of arXiv 2606.14668 (section 3), and GRACE's method section (arXiv 2211.11031: a deferral radius per key, the final prompt token, values trained by 100 gradient steps). The other numbers were read from the cited papers through page summaries, so check them against the PDF before quoting them elsewhere. "Not found" means a targeted search found nothing, not that nothing exists.

## 1. Weight-space programming, the starting point

**Source.** "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights", by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL), a blog post of 2 November 2026: https://rikiyat.github.io/blog/weight-space-programming/. It is not peer-reviewed.

**Its definition:** a large language model reads and writes a network's weights directly, with no optimizer and no gradient descent.

**Its Qwen3.5-0.8B experiment** (checked):
- The editor model traced looping on maths problems to one direction written by the MLPs of layers 15 and 19.
- It subtracted that direction from 64 output columns in each layer: 131,072 weights, 0.016% of the model.
- It used budgets of 10, 25 and 50 forward passes, with no gradients and no training.
- MATH-500 went from 7.9% to 48.1% with the 50-pass edit, and to 12.3% and 14.1% with the 10- and 25-pass ones.

**Its limit, in its own words** (Gemma 3 270M section): "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked." Adding knowledge appears only as a future direction.

**Critiques:**
- https://honnibal.dev/blog/astra-read-write-misleading argues that only the tiny-network experiments read raw weights alone.
- seangoedecke.com likens the Qwen edit to removing a direction, as in refusal-direction ablation (arXiv 2406.11717, W − r̂r̂ᵀW).

**Where this project (then called weightspace) stands against it:**
- **The loop fix** reproduces that edit's form (layers 15 and 19, 64 neurons): on 32 new GSM8K problems, correct answers went from 1 to 7.
- **The fact work goes where the blog stops.** It adds knowledge, and the editor writes closed-form changes it checks before keeping them, with an exact undo.
- **No precedent found.** The survey found no published system where a larger model writes facts into a smaller model's weights with acceptance checks and an undo ledger.
- **What is published:** the larger model drafting the edit data, for example GPT-3.5-written transfer sets beating MEND and MEMIT (arXiv 2306.09306), and synthetic corpora for knowledge injection (arXiv 2409.07431, 2506.10943).

## 2. How published editors write facts

| Method | Where it keys | Gradients? | Gate or route | Reported generalization / specificity |
|---|---|---|---|---|
| ROME (2202.05262), MEMIT (2210.07229), PMET, EMMET, AlphaEdit (2410.02355), NSE | Last subject token, middle-layer MLP | Closed-form weight update, but the target vector comes from gradient steps (ROME: 25 Adam steps) | None | ROME on GPT-2 XL: neighbourhood 75.4 against 78.1 unedited. MEMIT, 10k edits on GPT-J: 73.7 against 83.5. AlphaEdit, 2k sequential edits: 66–75 |
| MobiEdit (2506.13772) | Subject token | **None**: value search from forward-pass loss differences, then a closed-form rank-one update | None | (not extracted) |
| GRACE (2211.11031) | Last token, codebook | Yes | Deferral radius | Locality 1.00 but generalization 0.36 (2405.14768). Under generation scoring, reliability 0.012–0.018 (2502.11177) |
| WISE | Side memory | Yes | Routing by activation, threshold = minimum edit activation | 0.024–0.216 reliability under generation scoring (2502.11177) |
| SERAC (2206.06520) | Separate scope classifier and counterfactual model | Trained | Scope classifier | 0.855 on hard paraphrases, 0.967 at rejecting near-neighbour questions |
| ALOE (2609.29269) | One down_proj; addresses refined on rollout states | AdamW refinement | Dead-zone sigmoid gate, calibrated per slot | Efficacy 0.955–0.999, locality 0.981–1.000, **paraphrase 0.217–0.472** (generation, exact match) |
| T-Patcher, MELO, ELDER, MEMOIR, MEND, RLEdit | Various | Yes | Patches, adapters, masks | — |
| In-context (IKE, 2305.12740) | The prompt | None, and no weights change | — | GPT-J: paraphrase 95.2, neighbourhood 77.0 with retain demonstrations; 37.9 with the fact alone |

**Gradient-free, with a weight change:** MobiEdit, Larimar's memory write (2403.11901; its encoder and decoder are trained beforehand), and the knowledge-neuron edit step (2104.08696). This project is in this group: forward passes and closed-form solves only.

## 3. What the literature says about the three findings here

**Company identity is readable at the company's name, not at the end of the question.**
- **Here:** look-alike AUC 0.96–1.00 at the subject's last token (layers 2–16), against at most 0.73 at the last prompt token.
- **The literature:** facts are recalled by MLPs at the last subject token in middle layers and by attention at the last token (2202.05262), and blocking either cuts extraction to about 30% (2304.14767).
- **The same gap elsewhere:** ALOE's raw activations at its site separate at AUC 0.64, against 0.998 for its learned address.
- No editing study on Gated DeltaNet hybrids was found; the nearest is ROME on Mamba (2404.03646).

**An edit keyed on the subject leaks to other questions about it, and to similar names.**
- **Other questions about the subject:** relation specificity is 41–54 for ROME, 61–67 for MEMIT and 41–65 for AlphaEdit (2408.15091, 2506.04042). One ROME edit on TinyLlama-1.1B takes same-subject other-relation accuracy from 85.66 to 19.28 (2502.14838).
- **Similar names:** look-alike names share keys. "Vladimir Mayakovsky" and "Vladimir Bukovsky" overlap above 0.95, so an edit to one carries over to the other (2408.07413).

**An edit keyed on the end of the question misses new wordings.**
- GRACE generalizes at 0.00–0.46.
- Keying on the last relation token (RETS) raises relation specificity to 78–83, but paraphrase falls from 93–99 to 72–75.
- **Here:** the gated first token reaches recall 0.42–0.50 on bench40b.

**Routing on subject and relation together beats routing on the subject alone.** In a router ablation, no-route locality on ZsRE is 0.0000 with the subject alone against 0.9100 with subject and relation (arXiv 2606.14668, Table 4, checked).

**So the split gate is the literature's diagnosis built into the weights.**
- **It ANDs two detectors:** a subject detector where identity is separable (the layer-4 name neuron), and a question detector at the last token learned across many companies.
- **Not found elsewhere:** two targeted searches found no published edit that does this inside the weights.
- **Closest work:** ALOE, RETS, TOP (2506.04042) and the subject-plus-relation router of 2606.14668.

## 4. Evaluation: why the numbers here look low

- **Generation instead of teacher forcing.** The Mirage of Model Editing (2502.11177) re-scored editors by generating answers. Average reliability fell from 0.956 to 0.389 on QAEdit; the best case, ROME on Llama-3-8B, fell from 0.987 to 0.691. After 1,000 sequential edits, 9.3% remained (FT-M kept 40.5%).
- **Chat format and stricter matching.**
  - CHED (2505.23026) counts an edit only if the new answer appears and the old one does not within 50 tokens. Under that rule IKE scores 0.4%, against 68.8% when scored by probability.
  - The chat template cuts MEMIT from 86.5% to 73.9%.
  - Edits made in question form fail in completion form about 70% of the time (2401.07453).
- **This bar has no published equivalent.** The field reports dataset averages, mostly probability comparisons combined by harmonic mean. No benchmark found uses near-typo names as negatives, and none found requires recall and low leakage of the same fact together. bench40b asks for both, by greedy generation, in plain and chat form, on 12 new wordings per fact.
- **Small models:**
  - AlphaEdit on Llama-3.2-1B reports 91.70 / 87.00 / 59.97 (2606.26783) and on phi-1.5 70.79 / 65.12 / 48.96. Gemma2-2B fails, which its authors trace to its norm after the MLP.
  - No results were found for Qwen models under 2B.
- **In-context knowledge also leaks to other subjects, even with no edit.**
  - With the edit sentence in the prompt, GPT-2 XL's neighbourhood score falls from 0.78 to 0.52 unedited, and from 0.76 to 0.14 under ROME (CounterFact+, 2305.17553).
  - **The yardstick here agrees:** with the fact in the prompt, Qwen3.5-0.8B gives the answer for 0.58 of look-alike prompts, against 0.083 for the gated weight edit.

## 5. Keeping knowledge current, and where direct weight edits fit

- **Retrieval dominates at scale.** On WikiBigEdit's 502K real updates, retrieval reaches 78.64%, ROME and MEMIT collapse within a few hundred sequential updates, and LoRA with merging beats every editor after about 100K updates (2503.05683).
- **Lifelong editors scale in other ways.** AlphaEdit keeps 98.9% efficacy over 2K sequential edits, scored by teacher forcing. UltraEdit reaches 2M edits, with side effects checked only to 20K (2505.14679).
- **Fine-tuning learns new facts slowly and increases hallucination** (2405.05904). Measured forgetting after learning new facts falls from 89% with full fine-tuning and 71% with LoRA to 11% with sparse memory layers (2510.15103).
- **Where this project fits:**
  - Its edits are per fact, gradient-free, checked before they are kept, and exactly reversible.
  - Its open questions are the field's: specificity at a strict bar, and how many facts one model can hold.

## 6. What follows from it

1. **Keep the AND design.** The split gate puts the literature's diagnosis into the weights. Its 40-fact result on bench40b decides whether the method continues.
2. **Keep scoring by generation, in chat and plain form, against look-alike names.** It is stricter than the field's usual scoring, and it is what a user sees.
3. **Run a published editor through the same bench as a reference.**
   - EasyEdit has ROME and AlphaEdit configs for Qwen3.5-9B (added 2026-07-07) but none for the 0.8B model, so one needs a layer sweep and covariance statistics.
   - Their target vector uses gradient steps, so it is a reference here, not a method to adopt.
4. **Gradient-free value search exists** (MobiEdit). It is the nearest published relative of the forward-pass answer push used here.
5. **Similar names are a known weak point** (2408.07413). The split gate's remaining leaks came from the name neuron firing on look-alikes, so a sharper name detector is the matching next step.
6. **Many facts in one model** is the field's lifelong setting and comes after the split gate.

## 7. Update, 2026-11-14: work found after v1

Found by a second search on 2026-11-14. Each item was read from its arXiv abstract page only, so check the paper before quoting it elsewhere.

**A gradient-free editor with an external memory: INLAY** (arXiv 2608.26292, 26 August 2026).
- **The method, from its abstract:** the model stays frozen and edits are kept in an external addressable memory. When an edit applies, it is added at decoding time as a bias along one token's unembedding direction.
- **The paper is a negative result about benchmarks.** On 1,689 queries, an oracle that picks the best routing action for each query ties a one-line static policy, and abstaining is never the best action. The reason given is that counterfactual benchmarks hold no queries on which an edit should be rejected. The abstract also reports that WISE beats INLAY on CounterFact with Qwen2.5-7B, and that retrieval beats every tested editor on RippleEdits.
- **What it means here:**
  - It is the nearest published relative of GRAFT's inserted layer found so far: no gradients, no weight change, a push toward the answer token. GRAFT pushes at a middle layer's MLP output, token by token, behind an exact name match and a fitted question detector. The two have not been compared.
  - Its finding agrees with section 4. The benches here are built around the negatives that the field's benchmarks lack: look-alike names and other questions about the same subject.

**Closed-form weight editors, at thousands of sequential edits.**
- **FisherEdit** (arXiv 2610.25914, 19 October 2026) argues that AlphaEdit's robustness comes mainly from update regularization, not from its null-space projection. It replaces the projection with importance-aware regularization in parameter space, solved in closed form row by row, and reports results on Llama-3, Qwen2.5 and Olmo-3.
- **Norm anchors** (arXiv 2602.02543, 30 January 2026) trace the abrupt failure of long locate-and-edit runs to a feedback loop between the solved value vectors and the norms of the edited weights. Rescaling each value vector to a reference norm from the original model extends the usable run by more than 4 times.
- **What it means here:** both concern edits accumulating in a shared weight matrix. GRAFT's inserted layer writes no weight, and the grafted model compiles a whole table at once into neurons of its own. Their limit is how many neurons a layer can give up ([GRAFT, section 12](graft.md#12-the-grafted-model)).

**A context as a weight patch, in closed form** (arXiv 2507.16003, July 2025, revised June 2026; arXiv 2511.17864, November 2025, revised July 2026).
- **The claim:** a forward pass with a context equals a pass without it and with a low-rank update of the MLP weights. The second paper extends this to modern blocks: rank-1 patches on the MLP weight matrices and a patch to the RMSNorm scale, with gated MLPs covered.
- **What it means here:** it is the same kind of tool as this project's, forward passes and closed form with no training. Its patch depends on the token being processed, so it reproduces a context for one query; it is not presented as a store of facts.

**Documents to weights in one forward pass, by a trained hypernetwork.** Doc-to-LoRA (arXiv 2602.15902, 13 February 2026) meta-learns a hypernetwork that turns an unseen document into a LoRA adapter in a single forward pass. The hypernetwork is trained beforehand, so it lies outside this project's rule of no training. It is a yardstick for what a trained route reaches.

**Weight-space programming, audited** (arXiv 2611.08070, 5 November 2026). An exploratory audit has a large model read the raw weights of small convolutional networks as text, with no tools. It concludes that such reading is better described as "local arithmetic plus dataset priors" than as tracing how information flows, from mostly single runs. The editor in this project never reads raw weights: it works through tools that run forward passes and report measurements.

**Still not found:** a published system in which a larger model writes facts into a smaller one with acceptance checks, and any editing study on Gated DeltaNet hybrid models.
