# Second repair attempt: plan, written before the edit was built (2026-11-06)

What the measurement found (16_name_chain.json, forward passes only, on exactly the kept edits, checked by hash):
- protect_names leaves each name token's key firing the same on new wording (for example ' Julian' 0.62 without, 0.61 with);
- at the same strength, recall is the same with and without it (1.25: 11 of 16 both; 1.0: 7 and 6 of 16);
- it cuts the first name's key at other companies' name slots about fourfold (mean 0.18 to 0.05).
The low recall of the first repair came from choosing strength 1.0, not from the protection.

The candidate, fixed now: the same fact, protect_names on, strength 1.25 (the strength of the shipped edit).
It is built in a tracked session with 16 recall prompts and 10 same-subject prompts (the first fresh check's, which are already seen),
after rebuilding the loop fix.

Final check: experiments/fresh_fact_check_v3.py, 46 prompts, fingerprint 2670e457f126eb88, on the base model, the shipped patch and the candidate.

Decision rule: replace the shipped patch with the candidate only if, on the v3 prompts, all of these hold:
1. recall is at least the shipped patch's minus 1;
2. other companies given part of the name: at most the shipped patch's;
3. other questions about RX Systems given part of the name: at most the shipped patch's plus 1;
4. known facts: at least the shipped patch's.
Otherwise the shipped patch stays, and the candidate is kept with the results.
