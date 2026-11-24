I taught both facts, recorded them as a table and in the model's own weights, and merged them with the other model at the weights. Every call worked on the first try. Both facts are marked not good: they recall on all new questions but leak into nearby questions about the company. I recorded them anyway, as the task's order said, and didn't try to repair the leaks.

**Before teaching**, the model made up a CEO ("Jorge M. Halvering") and an HQ ("Halvering, Denmark").

**Checks** (`check_fact` compares replies with and without the table, so these counts are for the table form, not the weights):

| Fact | Recall | Leaks on other questions | Leaks on look-alike names | Good |
|---|---|---|---|---|
| ceo: Solveig Brandt | 8 of 8 | 1 of 4: the CFO question got Solveig Brandt | 0 of 3 | no |
| hq: Narvik | 8 of 8 | 2 of 4: largest shipyard and country of first ship | 0 of 3 | no |

On "Who runs Halvering Maritime these days?" the reply names her as the *owner*, not the CEO. The check still counts it as recalled because it matches the name as whole words.

**Versions recorded**
- **Table:** v1.
- **Weights v1:** 2 facts, 3 tensors changed, built from table v1. No fact was left out of the weights.
- **Weights v2:** the merge, 5 facts, 3 tensors changed. It is built from no table.

**Merge**
- **Facts added from theirs:** Exlampori Optics | ceo | Bence Tolnai, Nyokastel Marine | hq | Varna, and Jentrabuk Games | founder | Xochitl Tenorio.
- Nothing was replaced, kept or skipped, and there were no warnings.
- **`same_as_one_export: true`:** the merged weights are the ones a single export of all 5 facts gives.
- **Neurons:** 39 at layer 23. Of the last layer's budget of 256, 20 hold answers' first words and 19 the later words. None are elsewhere.

**Diff from v1 to v2:** the same 3 facts were added, none removed or changed. Three layer-23 MLP tensors differ.

**Asking the merged version (v2), without the table:**
- "Who is the CEO of Halvering Maritime?" → "Solveig Brandt is the CEO of Halvering Maritime. She has been the company's executive leader since its founding in 1996." The founding date is made up by the model.
- "Where is Nyokastel Marine headquartered?" → "Varna, Bulgaria."

**What's still uncertain:** the leaks into nearby questions are unrepaired. The weights were checked on only these two questions, and the other model's other two facts weren't asked at all.
