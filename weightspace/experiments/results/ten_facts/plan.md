# Ten new facts: plan, written before the run (2026-11-06)

Question: does the current recipe write facts it has never been tuned on?

Items: the same 10 facts as the earlier fresh-facts check (experiments/proto.py FACTS[20:30], with exp_o_confirm_facts.fresh_fact's test prompts):
6 recall prompts (4 plain, 2 chat), 5 same-subject prompts, the built-in test decoys, and name prompts when the target is a person's name.
Model: base Qwen3.5-0.8B, no other edit, as in the earlier check.

Recipe, fixed now and not changed during the run:
- teaching from the plain prompts only (the chat question is not taught);
- layer 20, strength 1.25, protect_names on (it only changes facts whose target is a person's name);
- one tracked session; each fact proposed and tested on its own, nothing kept.

Pass rule, the package's default gates: recall at least 0.75 (5 of 6), same subject at most 0.25, other subjects at most 0.25,
other companies asked for a name at most 0.1, garbled at most 0.34, known facts down by at most 0.05, drift at most 0.05.
Leaks count any part of the target that the unedited answer lacked, so these gates are stricter than the earlier check's.
Earlier result to compare with, from the README: 0 of 10 passed with no backpropagation on Qwen3.5-0.8B.
