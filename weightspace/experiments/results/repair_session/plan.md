# Repair session: plan, written before the session started (2026-11-06)

Goal: stop the RX Systems fact's first name leaking into other companies' names, without losing recall.

1. Rebuild the loop fix exactly: find_direction and propose_removal with the earlier inputs, keep at strength 1.0.
   Its hashes must match the exported patch (layer 15 faf13ce0d0eacf36, layer 19 313f67693fc72875).
2. The old fact recipe, rx_v1 (plain sentences, layer 20), tested at its kept strength 1.25 with the new checks, including other_names.
3. The new recipe, rx_v2 (the same, with protect_names), tested at strengths 1.0, 1.25 and 1.5.
   Decision rule: keep rx_v2 at the strength with the highest recall among those whose verdict is "better"; on a tie, the lower strength.
   If none is "better", keep nothing new and ship nothing new.
4. isolate rx_v2 (each part alone and a random edit of the same size), keep, export to patches/qwen3.5-0.8b_loops_and_rx_owner_v2.
5. Final check: experiments/fresh_fact_check_v2.py, prompts fingerprint 3cf826be4aa98b5e, written before this session and not changed after,
   on the base model, the old patch and the new one.
