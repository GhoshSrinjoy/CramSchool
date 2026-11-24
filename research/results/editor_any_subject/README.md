# A first check on facts about software and science, through the editor tools

One session on 2026-11-24. A bigger model, the editor, taught twelve invented facts to Qwen3.5-0.8B through the tools of a running
server, checked each fact on questions it wrote, and recorded the table and the grafted model. It is a first check and not a bench:
one session, one editor, and the editor wrote every question. The write-up is in
[benchmarks](../../../docs/benchmarks.md#software-and-science-facts-a-first-check-through-the-editor-tools).

## The result

| | Good | Answer on new questions | Answer on other questions about the subject | Answer for look-alike names |
|---|---|---|---|---|
| All twelve | 1 | 94 of 96 | 23 of 48 | 0 of 36 |
| Software (6) | 1 | 46 of 48 | 7 of 24 | 0 of 18 |
| Substances (4) | 0 | 32 of 32 | 11 of 16 | 0 of 12 |
| Concepts (2) | 0 | 16 of 16 | 5 of 8 | 0 of 6 |

The facts are learned and look-alike names are told apart. The facts fail on other questions about the same subject, where the
stored answer opens the reply in 22 of the 23 leaks. The cause was not measured.

## The rule, written before the session

- A fact is good when its final `check_fact` says so: the answer, matched as whole words, is in at least 6 of 8 new questions, and
  the table brings the answer's first word into none of 4 other questions about the same subject and none of 3 look-alike names.
- A drafted kind of subject (software, substance) stays shipped if at least 3 of its facts are good. Software had 1 of 6 and
  substance 0 of 4, so neither ships. The third kind, `concept`, ships since before the session, and its two facts fail the same way.
- The counts are made from the saved calls, not from the editor's report.

## What is here

| File | What it is |
|---|---|
| `facts.json` | The twelve facts, the type of each answer, and the rule |
| `brief.md` | What the editor was given: the protocol, the tools' descriptions and the steps |
| `args/`, `calls/` | Every call's arguments and its result as the server returned it, 63 of each, numbered in order. A fact's final check ends in `_final_<id>.json` |
| `summary.json` | The counts, by kind and by type of answer, every leaked reply, and the two histories as read from their files after the session. Made by `scripts/count_editor_session.py` |
| `editor_report.md` | The editor's own account, from its final message |
| `first_attempt/` | The first attempt, which stopped after six calls: over the server, `add_relation` failed on every call. Its brief, calls and report |
| `drafted_kinds.json` | The names and general questions drafted for software and for substances, for `Graft.add_kind` |

Count it again, with no model loaded:

```bash
python scripts/count_editor_session.py research/results/editor_any_subject
```

## How it was run

- **The server:** `cramschool serve` on the model, with the table at its defaults (`calls/001_graft_status.json`): read and written
  at the last layer, the fast build.
- **The editor:** Claude Fable 5.2, as a separate agent that read and wrote only the session's folder (checked afterwards from its
  own log of 166 actions: 63 argument files written, 63 calls, and searches inside the folder). It called each tool with
  `cramschool call <tool> @args.json --out result.json`.
- **Not kept:** the table, the grafted model and the weights' history (1.8 GB), and the server's log. `summary.json` holds what was
  read from the two histories: three versions each, both whole, each version of the weights naming its table version.

## Read this before comparing

- **The files are as the session wrote them.** Since then the two drafted kinds left the package, two result keys of the tools
  were renamed (`table` is now `table_version`, and `compiled_from_table` inside `weights`), `check_fact` lists other stored answers
  with each reply, and the protocol gained a sentence. The brief and the calls here show the earlier text and keys.
- **The rule is stricter than the benches'.** A bench allows a fact's answer in a quarter of the other questions about its subject;
  this rule allows none. Held to a quarter, four of the twelve would be good (s1, s3, s4, s5). That is a reading made afterwards.
- **Another stored fact's answer** came into 11 of the 48 replies to other questions. Nine of those questions ask for that other
  fact, which is right to answer; two do not. The session's check did not count them, and they are not in the counts above.
- **One reply at the base weights holds a fact's answer:** the install command, on the question asked of the recorded weights
  (`calls/056_graft_weights_ask.json`). In that fact's final check the model without the table gave it on none of the 8 questions.
