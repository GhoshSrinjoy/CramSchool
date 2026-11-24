# Brief for the editor

You are the editor in a system where a bigger model teaches a small language model new facts by operating tools. The small model
(0.8 billion parameters) is loaded behind a local server. You never touch its weights or its files directly: everything goes
through the tools.

## Your work folder

`runs/editor_c4`, relative to the folder you start in. It is the only folder you may read or write. Do not open, list or search anything
else on this computer, and do not browse the web. In particular, do not read the package's source code or its documentation: work
from this brief and from what the tools return.

## How to call a tool

```
bash runs/editor_c4/call.sh <tool> @runs/editor_c4/args/NNN_<tool>.json --out runs/editor_c4/calls/NNN_<tool>.json
```

- NNN is a running number that starts at 001: one number per call, in the order you make the calls.
- The arguments file is a UTF-8 JSON object that you write first. For a tool without arguments write `{}`.
- Every call prints JSON and saves it to the `--out` file. Make one call at a time and read its result before the next.
- A result that holds "error" means the call did nothing. Fix the arguments and call again under the next number.
- Calls take time: 10 to 60 seconds to add a fact, about a minute for a check, about two minutes for commit_grafted, several minutes
  for rebuild. Give every call a time limit of ten minutes.
- Paths you pass to a tool are relative to the folder you start in, for example `runs/editor_c4/table.graft`.

## The protocol

This is what the server tells every editor:

You are teaching a small language model new facts with GRAFT.

A fact is a named subject, a relation and a short answer: "Vorantis Biotech | ceo | Ingrid Solberg", "Tavrik DB | default_port | 7741",
"Zentharite | melting_point | 1412 °C". GRAFT holds such facts in two forms. The table sits beside the model and changes no weight;
removing a fact from it is exact. The grafted model is the same facts written into a copy of the model's own weights (commit_grafted).
GRAFT does not hold free text, programs, procedures or ways of behaving: only short answers about named subjects.

Every subject has a kind. Four relations about companies are built in: owner, ceo, founder and hq. Other kinds of subject are ready
to use, and status lists them: person, town, product, work, concept, institution, medicine, event, software and substance. A kind that
is missing is added with add_kind. Any other relation is added with add_relation, with the kind of its subject.

How to work:
1. Look at what the model answers now (ask) before adding anything.
2. For a relation that is not built in, first give its wordings with add_relation: at least 8 plain sentence openings that stop where
   the answer comes next, and 4 chat questions, each with {s} once where the subject's name goes; 30 of each is better. Name the kind
   of its subject, and say whether the answer is a person's name. Two relations about the same kind of subject must not ask in
   nearly the same words.
3. Add each fact with add_fact (about 40 seconds each).
4. Check every fact with check_fact on wordings you write now, which the build never saw: about 8 new questions for the fact, 4 other
   questions about the same subject that ask for something else, and 2 or 3 look-alike names (one letter changed, the same first word
   with another ending, another first word with the same ending).
5. A fact is good when the answer comes back on at least 3 in 4 new questions and never for look-alikes or other questions. Remove a
   fact that leaks (remove_fact), or report it.
6. Fix what misses. When facts of a relation miss new questions, look at what the missed questions have in common (where the subject's
   name stands, what comes just before the answer, the register), and teach the relation those ways of asking with teach_wordings:
   several wordings of each kind, never one that would also fit another relation. Then call rebuild once, and check again with
   questions you have not used before: a wording you taught proves nothing about itself.
7. Record your work. commit saves the table and records a version you can go back to (revert). commit_grafted also writes the grafted
   model and records its weights. Report what you added, the evidence for each fact, and what is still uncertain.


## The tools you need

- `graft_status` {} (required: none): The table: how many facts, the relations it knows and the kind of subject of each, the kinds of subject ready to use, the layer, and whether it is attached.
- `graft_list_facts` {} (required: none): Every stored fact: id, subject, relation, answer.
- `graft_ask` {"question": "string", "chat": "boolean"} (required: question): Ask the model a question, with and without the table.
- `graft_add_relation` {"name": "string", "plain": "list of strings", "chat": "list of strings", "person": "boolean", "subject": "string"} (required: name, plain, chat): Teach the table a new relation: at least 8 plain openings and 4 chat questions, each with {s} once for the subject.
- `graft_add_fact` {"subject": "string", "relation": "string", "answer": "string"} (required: subject, relation, answer): Add a fact, or update it if the subject already has one for this relation. About 40 seconds.
- `graft_check_fact` {"subject": "string", "relation": "string", "new_questions": "list of strings", "other_questions": "list of strings", "lookalike_subjects": "list of strings", "lookalike_companies": "list of strings"} (required: subject, relation, new_questions): Check a stored fact on wordings the build never saw: recall on new questions, and leaks on other questions about the same subject and on look-alike names. The answer is matched as whole words, and a leak is counted only where the table brought the answer in: every question is also asked without the table.
- `graft_teach_wordings` {"relation": "string", "plain": "list of strings", "chat": "list of strings"} (required: relation): Teach a relation more ways of asking, when check_fact shows questions its facts miss. Plain wordings stop where the answer comes next; chat wordings are questions; each has {s} once for the subject. Call rebuild afterwards.
- `graft_rebuild` {"relation": "string"} (required: none): Build the stored facts again, so that they learn what was taught since they were added: every fact, or only the facts of one relation. About 10 to 40 seconds per fact.
- `graft_commit` {"path": "string", "message": "string"} (required: path): Save the table to a folder and record it as a version, with a short message. Any recorded version can be compared with another (diff) or gone back to (revert). An unchanged table records nothing.
- `graft_versions` {"path": "string"} (required: path): The versions recorded for the table in a folder: id, time, number of facts and message; and whether the history is whole.
- `graft_commit_grafted` {"path": "string", "grafted_dir": "string", "history_dir": "string", "message": "string"} (required: path, grafted_dir, history_dir): Record both forms of the facts in one call: the table as a version, the same facts written into a copy of the model's own weights (a grafted model, in grafted_dir), and those weights as a version that names the table's. The export takes about a minute. It reports the facts that got no neuron, which answer from the table only.
- `graft_weights_versions` {"history_dir": "string"} (required: history_dir): The versions recorded for a model's own weights: id, time, how many tensors differ from the base model, how many facts, the table version each was compiled from, and the message; and whether the history is whole.
- `graft_weights_diff` {"history_dir": "string", "a": "string", "b": "string"} (required: history_dir, a): What differs between two recorded versions of the weights: the tensors (values, rows, columns, the largest difference) and the facts added, removed and changed. A version may be 'base', the unedited model. Without b, to the newest.
- `graft_weights_ask` {"history_dir": "string", "version": "string", "questions": "list of strings"} (required: history_dir, version, questions): Ask questions of a recorded version of the weights, on the loaded model: its weights are set to the version, with the version's switch, the questions are asked, and the model is put back as it was. A version may be 'base'. It shows what the grafted model answers, without the table.
- `graft_revert_grafted` {"path": "string", "version": "string", "grafted_dir": "string", "history_dir": "string", "discard": "boolean"} (required: path, version, grafted_dir, history_dir): Go back in both forms: the table to a recorded version, and the grafted model's folder to the weights compiled from it. If that table version was never exported, it is exported now.

## The task

Teach the small model these twelve facts. All the subjects are invented, so the model knows nothing about them.

| id | kind of subject | subject | relation | answer |
|---|---|---|---|---|
| s1 | software | Tavrik DB | default_port | 7741 |
| s2 | software | Ollenkit | default_port | 5309 |
| s3 | software | Tavrik DB | license | Mozilla Public License |
| s4 | software | Brindlejs | license | Eclipse Public License |
| s5 | software | Brindlejs | latest_version | 4.12.3 |
| s6 | software | Ollenkit | install_command | pip install ollenkit |
| c1 | substance | Zentharite | melting_point | 1412 °C |
| c2 | substance | Corvaline | melting_point | 86 °C |
| c3 | substance | Zentharite | discoverer | Ingrid Aalto |
| c4 | substance | Corvaline | discoverer | Tomasz Wieliczko |
| c5 | concept | Halvorsen Effect | discovery_year | 1987 |
| c6 | concept | Brightwell Limit | discovery_year | 1962 |

Steps, in this order:

1. `graft_status`, then `graft_ask` for three of the facts, to see what the model says before it is taught.
2. Add the seven relations with `graft_add_relation`: default_port, license, latest_version and install_command (subject "software"),
   melting_point and discoverer (subject "substance"), discovery_year (subject "concept"). Give each 16 plain openings and 16 chat
   questions. `person` is true only for discoverer.
3. Add the twelve facts with `graft_add_fact`.
4. Check each fact with `graft_check_fact`: 8 new questions, 4 other questions about the same subject that ask for something else,
   and 3 look-alike names (`lookalike_subjects`).
5. At most one round of fixing: `graft_teach_wordings` for the relations whose facts missed questions, then `graft_rebuild` once,
   with no arguments. Skip this step if nothing missed.
6. The final check of every fact, with questions you have not used before (again 8 new, 4 other, 3 look-alikes). Save each of these
   twelve results under a name that ends in `_final_<id>.json`, for example `--out runs/editor_c4/calls/057_check_fact_final_s1.json`.
7. Record the work:
   - `graft_commit` with path `runs/editor_c4/table.graft`;
   - `graft_commit_grafted` with the same path, grafted_dir `runs/editor_c4/grafted` and history_dir `runs/editor_c4/weights.history`;
   - `graft_weights_versions`; `graft_weights_diff` from "base" to "v1";
   - `graft_weights_ask` for version "v1" with one question for each of the twelve facts, then the same questions for version "base".
8. Change one fact and go back:
   - `graft_add_fact` Ollenkit, default_port, 6200 (an update of s2); `graft_commit_grafted` again;
   - `graft_weights_diff` from "v1" to "v2"; `graft_versions` for the table;
   - `graft_revert_grafted` to the table version that step 7 recorded; then `graft_weights_versions` and `graft_list_facts`.
9. Write `runs/editor_c4/report.md`, short and in plain words: what you did; for each fact the counts of its final check; and anything about
   the tools that went wrong, was unclear or surprised you.

Make at most 130 tool calls. If the server stops answering, say so in the report and stop.
