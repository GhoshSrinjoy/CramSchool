# The editor's report (second session, 2026-11-24)

The editor's final message, word for word, except that the folder it worked in is given as a relative path. It is the editor's own account: the counts in this folder's `summary.json` are made from the saved calls, not from this report. `runs/editor_c4/` is the folder the session ran in; its `args/` and `calls/` are this folder's.

## 1. Completion

Every step of the brief (1 to 8) was completed, in 63 numbered calls (001 to 063). No call returned an "error" and the server answered every time.

- Step 5 was run: `graft_teach_wordings` for `license` (036) and `latest_version` (037), the two relations whose facts missed questions in step 4, then one `graft_rebuild` with `{}` (038, `"rebuilt": 12`, 92.3 s).
- One deviation in how I read results: for calls 025 to 050 (the checks) I sent the command's stdout to `/dev/null` and read the saved `--out` file by searching it for the four keys plus every `"recalled": false` and `"leaked": true` entry. I did not read the full text of those results. Call 024 was read in full.
- Within a check round, the two facts of one relation share the same eight question templates, each with its own subject name. Every question string in the final checks was new in the session.
- After seeing the step 4 misses I changed three of my planned final questions (two for `license`, one for `latest_version`) to new wordings of the kinds that had missed.

All files are under `runs/editor_c4/` (`args/` for arguments, `calls/` for results).

## 2. Final check of each fact (values copied from the files in `calls/`)

| id | file | recall | leaks_other_questions | leaks_lookalikes | good |
|---|---|---|---|---|---|
| s1 | `039_check_fact_final_s1.json` | "8 of 8" | "1 of 4" | "0 of 3" | false |
| s2 | `040_check_fact_final_s2.json` | "7 of 8" | "2 of 4" | "0 of 3" | false |
| s3 | `041_check_fact_final_s3.json` | "8 of 8" | "1 of 4" | "0 of 3" | false |
| s4 | `042_check_fact_final_s4.json` | "8 of 8" | "0 of 4" | "0 of 3" | true |
| s5 | `043_check_fact_final_s5.json` | "7 of 8" | "1 of 4" | "0 of 3" | false |
| s6 | `044_check_fact_final_s6.json` | "8 of 8" | "2 of 4" | "0 of 3" | false |
| c1 | `045_check_fact_final_c1.json` | "8 of 8" | "3 of 4" | "0 of 3" | false |
| c2 | `046_check_fact_final_c2.json` | "8 of 8" | "2 of 4" | "0 of 3" | false |
| c3 | `047_check_fact_final_c3.json` | "8 of 8" | "3 of 4" | "0 of 3" | false |
| c4 | `048_check_fact_final_c4.json` | "8 of 8" | "3 of 4" | "0 of 3" | false |
| c5 | `049_check_fact_final_c5.json` | "8 of 8" | "3 of 4" | "0 of 3" | false |
| c6 | `050_check_fact_final_c6.json` | "8 of 8" | "2 of 4" | "0 of 3" | false |

The first-round checks are in `024` to `035_graft_check_fact.json`, in the same fact order. They used different questions from the final ones, so the two rounds are not a before-and-after comparison.

## 3. Version ids the recording steps returned, in order

| call | tool | table version | weights version |
|---|---|---|---|
| 051 | `graft_commit` | "v1" (`"recorded": true`, 12 facts) | none |
| 052 | `graft_commit_grafted` | "v1" | "v1" (table "v1", 3 tensors, 12 facts, `not_in_weights` []) |
| 058 | `graft_commit_grafted`, after the update to 6200 | "v2" | "v2" (table "v2", `not_in_weights` []) |
| 061 | `graft_revert_grafted` to table "v1" | "v3" | "v3" (its record says table "v1", of "v1", message "back to v1", `"exported": false`) |

- `graft_weights_versions` listed v1 only in 053, and v1, v2, v3 in 062, both with `"intact": true`.
- `graft_versions` (060, before the revert) listed table v1 and v2, `"intact": true`.
- Table "v3" is shown only by the revert result; the brief asks for no `graft_versions` call after it.
- `graft_list_facts` (063) shows 12 facts, with Ollenkit `default_port` back at "5309".

## 4. What failed, was unclear, or surprised me

**Printing and timing**
- The printed output garbles non-ASCII characters: "°" and "–" print as "�", and subscripts as "??". The saved `--out` files hold them intact (checked in 018: `"1412 °C"`).
- The first `graft_add_fact` (012) took 175.4 s overall against 26.4 s for the fact itself. Later adds took 7 to 12 s, quicker than the brief's "about 40 seconds".

**Tool results that were unclear**
- `graft_teach_wordings` for `license` (036) returned `"added_chat": 7` for 8 chat wordings sent, with no reason for the one not added and no word on which it was.
- Fact ids changed after the rebuild: f001 to f012 became f013 to f024. The update of s2 (057) came back under `"added"` with a new id, "f025", and nothing in the result says it replaced an existing fact.
- `graft_revert_grafted` records a new table version and a new weights version (both "v3"), which its description does not mention. Its result has `"table": "v3"` at the top level beside `"table": "v1"` inside `weights`, which is confusing to read.
- `graft_ask` returns `company_found`, and `graft_check_fact` accepts `lookalike_companies`, although none of these subjects is a company. Before any fact is stored, `with_graft`, `subject_found` and `company_found` are all null.

**How `graft_check_fact` counts**
- The look-alike check does not use my new questions. It asks one taught wording per look-alike name (in 024: "What is the default port of Tavrok DB?").
- Another fact's answer arriving on an other-question is not counted as a leak. In 024, "Who develops Tavrik DB?" was answered "Mozilla Public License 2.0 (MPL-2.0)…" with the table and has `"leaked": false`, because only the checked fact's own answer is looked for.
- Whole-word matching gave these cases:
  - "4.12.3.1" counted as a leak (028).
  - "4.12.30.10" (028) and "4.12.30" (043) counted as misses.
  - "pip install ollenkit --upgrade" counted as a leak (044).
- "Port for Ollenkit by default?" (040) got the reply "5309" followed by a long run of zeros, counted as a miss.

**Leaks**
- Nearly every leak arrived as the stored answer at the very start of the reply to another question about the same subject. Examples from the final checks:
  - "What is the default data directory of Tavrik DB?" gave "7741".
  - "How hard is Zentharite on the Mohs scale?" gave "1412 °C".
  - "Who owns the largest Zentharite mine?" gave "Ingrid Aalto's…".
  - "When was the Brightwell Limit last revised?" gave "1962".
  - "Which package manager is Ollenkit installed with?" gave "5309", the port fact's answer.
- I had no tool in the brief to act on leaks, so they are only reported; no fact was removed.

**Step 4 misses that led to step 5**
- s3: "Do you know what license Tavrik DB is under?" and "Tavrik DB: license?" The reply was identical with and without the table.
- s5: "What's the current stable Brindlejs?" gave "4.12.30.10".
- The same two `license` questions were recalled for Brindlejs (s4).

**Weights**
- `graft_weights_diff` base to v1 (054): `down_proj` of layer 23 differs in 3,513,453 of 3,670,016 values (1024 rows, 3584 columns), while `gate_proj` and `up_proj` differ in 127 rows each. I did not expect almost the whole `down_proj` tensor to differ.
- `graft_weights_diff` v1 to v2 (059), for one changed fact: 1,882,104 `down_proj` values (3563 columns), 68 `gate_proj` rows and 1 `up_proj` row differ. `facts.changed` lists just the one fact, 5309 to 6200.
- `graft_weights_ask` on v1 (055): all twelve replies begin with the taught answer.
- `graft_weights_ask` on "base" (056): the reply to "What command should I use to install Ollenkit?" already contains "pip install ollenkit", so that question does not separate v1 from base for s6. No check file has `"said_without_the_table": true`.
- The brief's durations were higher than what I saw: `graft_commit_grafted` took 52 to 54 s and the revert 4.6 s.
