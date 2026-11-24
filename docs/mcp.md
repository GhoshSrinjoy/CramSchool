# The MCP server

`cramschool serve` runs cramschool's tools as an MCP server. It keeps the small model loaded on the GPU, so a bigger model can teach it facts in plain words: Claude Code, Claude Desktop, or any other MCP client.

GRAFT gives you the same facts in **two forms**, and you can use either or both: a **table** beside the model, which changes no weight, and a **grafted model**, the same facts written into the model's own weights. The grafted model still needs a small switch that turns a fact on when its subject is named. The tools below make, record and merge both.

**On this page:** [start it](#start-it) · [the endpoint](#the-endpoint) · [connect a client](#connect-a-client) · [run it with Opus](#run-it-with-opus) · [every tool](#every-tool) · [what results look like](#what-results-look-like) · [errors and waiting](#errors-and-waiting)

## Start it

```bash
pip install -e ".[mcp]"
cramschool serve --model models/Qwen3.5-0.8B --table my.graft --transport http
```

```text
cramschool: loading models/Qwen3.5-0.8B
cramschool: loaded Qwen3.5-0.8B on cuda; log off
cramschool: serving the tools at http://127.0.0.1:8765/mcp (this computer only); try `cramschool call graft_status`
```

There are two ways to run it:

| | `--transport stdio` (the default) | `--transport http` |
|---|---|---|
| **Who starts it** | The client, as its own child process | You, once |
| **The model loads** | On the first tool call | At start |
| **Where it listens** | Nowhere: the client talks to it over stdin and stdout | `http://127.0.0.1:8765/mcp` |
| **Use it for** | Claude Code and Claude Desktop | A client that connects by URL, `cramschool call`, your own scripts |

**Every flag of `cramschool serve`:**

| Flag | What it does |
|---|---|
| `--model PATH` | The small model. For the tools that write model folders it must be a local folder in safetensors format |
| `--table FOLDER` | A saved table to load. Without it, a new table is made on the first call |
| `--transport stdio\|http` | See above |
| `--port N` | The port for `--transport http`. Default 8765 |
| `--fast` | A new table uses the fast build: a fact after the first of its relation in 5 to 9 seconds, against about 40 |
| `--layer N\|last\|auto` | A new table only: where the question is read. Default: measured on the model |
| `--system TEXT` | A new table only: the system message of the application the facts will be asked in |
| `--prompt-detector` | A new table only: a second detector at the end of the prompt, for at most two facts on a subject |
| `--dtype bfloat16\|float16\|float32` | The number type the model is loaded in. Default `bfloat16` |
| `--log-dir FOLDER`, `--log FILE`, `--watch` | For the weight-edit tools: a hash-chained log per session, and a live progress window ([tools](tools.md#watching-a-session)) |
| `--calibration FILE`, `--budget N` | For the weight-edit tools: ordinary paragraphs, one per line, and the forward passes they may spend. Default: the 355 bundled paragraphs, and 60 |

## The endpoint

| | |
|---|---|
| **URL** | `http://127.0.0.1:<port>/mcp`, port 8765 unless `--port` says otherwise |
| **Protocol** | MCP over streamable HTTP |
| **Who can reach it** | This computer only. It has no key and no login, because its tools change the model and write files |
| **Instructions** | On connecting, a client is sent the GRAFT protocol and the weight-edit protocol, so it knows how to check its own work |
| **Paths** | A folder in an argument is relative to the folder the server was started in |
| **Calls** | One at a time, whichever tool. A client may send several together; the server runs them one after another |
| **The first call on a new table** | Measures at which layer this model's questions are best read: about 3 minutes, once for a model. The result is kept under `~/.cramschool/layer_profiles` |

## Connect a client

**Claude Code, which starts the server itself:**

```bash
claude mcp add cramschool -e USE_TF=0 -e HF_HUB_OFFLINE=1 -- "<path to python>" -m cramschool serve \
    --model models/Qwen3.5-0.8B --table my.graft --log-dir sessions --watch
```

**Claude Code, to a server that is already running:**

```bash
claude mcp add --transport http cramschool http://127.0.0.1:8765/mcp
```

**Claude Desktop, and other clients that read a configuration file:**

```json
{"mcpServers": {"cramschool": {"command": "<path to python>",
  "args": ["-m", "cramschool", "serve", "--model", "models/Qwen3.5-0.8B", "--table", "my.graft"]}}}
```

**Any MCP client, by URL.** In Python, with the `mcp` package:

```python
import asyncio

from mcp import Client
from mcp.client.streamable_http import streamable_http_client


async def main():
    async with Client(streamable_http_client("http://127.0.0.1:8765/mcp")) as client:
        tools = await client.list_tools()
        print(len(tools.tools), "tools")
        result = await client.call_tool("cramschool_graft_list_facts", {})
        print(result.content[0].text)


asyncio.run(main())
```

```text
38 tools
{
  "facts": [
    {
      "id": "f001",
      "subject": "Halvering Maritime",
      "relation": "ceo",
      "answer": "Solveig Brandt",
      "seconds": 96.4
    }
  ],
  "seconds": 0.0
}
```

**A shell, with no MCP client at all.** `cramschool call` calls one tool of a running server and prints the result as JSON. The `cramschool_` prefix can be left out.

```bash
cramschool call graft_status
cramschool call graft_add_fact '{"subject": "Vorantis Biotech", "relation": "ceo", "answer": "Ingrid Solberg"}'
cramschool call graft_check_fact @check.json --out result.json     # arguments from a file, the result also saved to one
cramschool call graft_status --url http://127.0.0.1:9000/mcp       # a server on another port
```

The first of these, on a table that holds two facts (shortened: most options are cut, and each list is on one line):

```json
{
 "facts": 2,
 "layer": 22,
 "attached": false,
 "relations": ["owner", "ceo", "founder", "hq"],
 "options": {"push": 2.0, "fast": true, "prompt_detector": false, "last_layer": "all", "...": "..."},
 "relation_kinds": {"owner": "company", "ceo": "company", "founder": "company", "hq": "company"},
 "kinds": ["company", "concept", "event", "institution", "medicine", "person", "product", "town", "work"],
 "seconds": 0.0
}
```

`"layer": 22` is where this table reads the question: the layer measured for the model.

## Run it with Opus

Claude Code is the MCP client, and `--model opus` chooses the model that drives the tools. One session was run this way for this page, with no one at the keyboard. The numbers below are from it. The two commands of step 2 are the same route typed by hand, which was not run here.

**1. Start the server**, as [above](#start-it), with `--transport http`.

**2. Tell Claude Code where it is, and choose Opus.**

```bash
claude mcp add --transport http cramschool http://127.0.0.1:8765/mcp
claude --model opus
```

Then ask in plain words; [what to ask](#what-to-ask) has examples.

**Or with no one at the keyboard,** which is how the session for this page was run:

```bash
claude -p "$(cat task.txt)" --model opus \
    --mcp-config mcp.json --strict-mcp-config --allowedTools "mcp__cramschool" \
    --output-format stream-json --verbose > transcript.jsonl
```

```json
{"mcpServers": {"cramschool": {"type": "http", "url": "http://127.0.0.1:8765/mcp"}}}
```

| Flag | Why |
|---|---|
| `--model opus` | The model that drives the tools. The session reported `claude-opus-5-5` |
| `--mcp-config mcp.json` | Where the server is |
| `--strict-mcp-config` | Use only that server, and none of the others your account has |
| `--allowedTools "mcp__cramschool"` | Let it call this server's tools without asking |
| `--output-format stream-json --verbose` | Print every call and result, one JSON object per line |

The session also passed `--disallowedTools` with Claude Code's own shell and file tools, `--no-session-persistence` and `--max-budget-usd 10`.

**What the session did.** The task: teach two invented facts about an invented company, check each, record both forms, merge the grafted model with another one at the weights, and ask the merged weights. Claude Code 2.1.322, 12 tool calls, no error, 20.5 minutes on a GPU that another job was using.

| Calls | What came back |
|---|---|
| `graft_status`, `graft_list_facts` | An empty table that reads the question at layer 22, the layer measured for the model |
| `graft_ask`, twice | The model's own answers, both made up: a CEO "Jorge M. Halvering" and a head office in "Halvering, Denmark" |
| `graft_add_fact`, twice | Both facts added, each with a detector for plain and for chat questions |
| `graft_check_fact`, twice | Recall 8 of 8 and 8 of 8. Look-alike names 0 of 3 and 0 of 3. Other questions about the company: 1 of 4 and 2 of 4 got the stored answer, so **neither fact is good by the tool's rule** |
| `graft_commit_grafted` | Table version v1; weights version v1, 3 tensors changed, both facts in the weights |
| `graft_weights_merge` | 5 facts in one model: its 2 and the other model's 3, in 39 neurons of layer 23. `same_as_one_export: true`. Recorded as weights version v2 |
| `graft_weights_diff` | From v1 to v2: 3 facts added, none removed or changed |
| `graft_weights_ask` | The merged weights, with no table, answer a fact of each model: "Solveig Brandt is the CEO of Halvering Maritime." and "Varna, Bulgaria." |

- **What it shows:** the tools work end to end over MCP with Opus driving them, including the merge at the weights, which had not been called over a server before.
- **What it does not show:** a pass rate. It is one session with two facts.
- **The leaks are real.** The question about the chief financial officer got the chief executive's name, and the questions about where its largest shipyard is and where its first ship was built got its head office. Questions close to a fact are a [stated limit](limitations.md). The task said to record the facts all the same, and the model's report says so.
- **A first attempt lost its first two calls.** Claude Code gives up on a call that sends nothing for 300 seconds, and the first call on a new table was measuring the read layer on a busy GPU. The server now sends a note every 15 seconds while a call runs. In the session above, `graft_commit_grafted` ran for 513 seconds on that busy GPU and came back.
- **An old Claude Code can connect and still not use the tools.** Version 2.1.25 connected to the server and never offered its tools to the model. 2.1.322 works.
- **Not run for this page:** the route with an API key, `cramschool teach --editor anthropic:claude-opus-5-5`, which calls the same tools from Python with no MCP ([editor models](editors.md)).

Everything the session called and got back is in [`research/results/mcp_opus_session/`](../research/results/mcp_opus_session/): the task, each call with its arguments and result, and the model's own report.

## What to ask

- "Teach the small model that Vorantis Biotech's CEO is Ingrid Solberg. Check it on new wordings and look-alike names, and record both forms."
- "Add the facts in data/companies.csv, check five of them at random, and tell me which ones leak."
- "The relation 'chief scientist' is new: write its wordings, add it, then add the fact Vorantis Biotech · chief scientist · Ama Owusu."
- "Merge the grafted model in theirs-grafted into mine, at the weights, and ask the merged model about a fact from each."
- "Find out why the model loops on maths problems, and fix it if the evidence supports an edit." This uses the weight-edit tools.

## Every tool

<!-- tools: made by scripts/mcp_reference.py, do not edit by hand -->

38 tools. Every name starts with `cramschool_`. An argument in *italics* can be left out. Every result also has `seconds`.

**Facts.** Add, ask, check and remove. Nothing here changes a weight.

| Tool | Arguments | The result holds |
|---|---|---|
| `graft_status`<br>The table as it is now | none | `facts`, `layer`, `attached`, `relations`, `relation_kinds`, `kinds`, `options` |
| `graft_list_facts`<br>Every stored fact | none | `facts`: each with `id`, `subject`, `relation`, `answer` |
| `graft_ask`<br>One question, answered without the table and with it | `question` (text), *`chat`* (true or false, default `true`) | `without`, `with_graft`, `subject_found` |
| `graft_add_fact`<br>Add a fact, or update it. About 40 seconds, or 5 to 9 with `--fast` | `subject` (text), `relation` (text), `answer` (text) | `added`, `replaced`, `detectors` (the forms of question it will open on), `quick_look`; a `note` if a form got no detector |
| `graft_check_fact`<br>Check a stored fact on questions the build never saw. The answer is matched as whole words, and a leak counts only where the table brought the answer in | `subject` (text), `relation` (text), `new_questions` (list of texts), *`other_questions`* (list of texts), *`lookalike_subjects`* (list of texts), *`lookalike_companies`* (list of texts) | `good`, `recall` ("7 of 8"), `leaks_other_questions`, `leaks_lookalikes`, and every reply with and without the table |
| `graft_remove_fact`<br>Remove one fact, or all of a subject's. Exact | `subject` (text), *`relation`* (text) | `removed`, `facts` |

**Relations and kinds of subject.** For a relation that is not built in, or a subject that is not a company.

| Tool | Arguments | The result holds |
|---|---|---|
| `graft_add_relation`<br>A new relation: its wordings, with `{s}` once for the subject, and the kind of its subject | `name` (text), `plain` (list of texts), `chat` (list of texts), *`person`* (true or false, default `true`), *`subject`* (text, default `company`) | `added_relation`, `plain`, `chat`; a `note` if stored facts should be built again |
| `graft_add_kind`<br>A new kind of subject: invented names and general questions about it | `name` (text), `names` (list of texts), `generic_plain` (list of texts), `generic_chat` (list of texts) | `added_kind`, `kinds` |
| `graft_teach_wordings`<br>More ways of asking for a relation the table knows. Call `rebuild` after | `relation` (text), *`plain`* (list of texts), *`chat`* (list of texts) | `added_plain`, `added_chat`, `left_out` |
| `graft_rebuild`<br>Build stored facts again, so they learn what was taught since | *`relation`* (text) | `rebuilt` |

**Facts from a file.** CSV, JSON, JSONL, a bench file, or `hf:<name>` for a Hugging Face dataset.

| Tool | Arguments | The result holds |
|---|---|---|
| `graft_preview_dataset`<br>The first facts a dataset would add. Adds nothing | `source` (text), *`subject_column`* (text), *`relation_column`* (text), *`answer_column`* (text), *`fixed_relation`* (text), *`limit`* (number, default `10`) | `facts`, `shown` |
| `graft_add_facts_from_dataset`<br>Add every fact of a dataset | `source` (text), *`subject_column`* (text), *`relation_column`* (text), *`answer_column`* (text), *`fixed_relation`* (text), *`limit`* (number) | `added`, `facts` |

**The table's history.** Versions of the table in its folder.

| Tool | Arguments | The result holds |
|---|---|---|
| `graft_save`<br>Save the table to a folder | `path` (text) | `saved`, `facts` |
| `graft_commit`<br>Save the table and record it as a version | `path` (text), *`message`* (text) | `version`, `recorded`, `facts`, `message` |
| `graft_versions`<br>The table's recorded versions | `path` (text) | `versions`, `intact` |
| `graft_diff`<br>What changed between two versions of the table | `path` (text), `a` (text), *`b`* (text) | the facts added, removed, changed and built again, and the settings |
| `graft_revert`<br>Go back to a version of the table. Recorded as a new version | `path` (text), `version` (text), *`discard`* (true or false, default `false`) | `version`, `of`, `facts` |

**The model's own weights.** The grafted model and its history. These read or write model folders, so the server's `--model` must be a local folder in safetensors format.

| Tool | Arguments | The result holds |
|---|---|---|
| `graft_commit_grafted`<br>Both forms in one call: the table's version, the grafted model, and the weights' version. About five minutes the first time, then about a minute | `path` (text), `grafted_dir` (text), `history_dir` (text), *`message`* (text) | `table_version`, `weights`, `facts`, `not_in_weights` |
| `graft_revert_grafted`<br>Go back in both forms. Recorded as a new version of each | `path` (text), `version` (text), `grafted_dir` (text), `history_dir` (text), *`discard`* (true or false, default `false`) | `table_version`, `weights`, `exported`, `not_in_weights` |
| `graft_weights_versions`<br>The recorded versions of the weights | `history_dir` (text) | `versions` (each names the table version it was compiled from), `intact` |
| `graft_weights_diff`<br>The tensors and the facts that differ between two versions of the weights. A version may be `base` | `history_dir` (text), `a` (text), *`b`* (text) | `tensors`, `facts` (`added`, `removed`, `changed`), `same` |
| `graft_weights_ask`<br>Ask a version of the weights, with no table. The loaded model is put back as it was | `history_dir` (text), `version` (text), `questions` (list of texts) | `replies` |
| `graft_weights_checkout`<br>Write a version of the weights as a model folder, byte for byte | `history_dir` (text), `version` (text), `out_dir` (text) | `version`, `tensors`, `files`, `folder` |
| `graft_weights_merge`<br>Merge two grafted models into one, at the weights. No table is needed. A minute or two | `mine` (text), `theirs` (text), `out_dir` (text), *`history_dir`* (text), *`message`* (text), *`prefer`* (text, default `error`) | `added`, `replaced`, `kept`, `same`, `facts`, `neurons`, `same_as_one_export`, `why`, `weights`, `warnings` |
| `graft_merge_grafted`<br>Bring another saved table's facts into this one, and record and export the merged table | `other` (text), `path` (text), `grafted_dir` (text), `history_dir` (text), *`message`* (text), *`prefer`* (text, default `error`) | `merged`, `table_version`, `weights`, `not_in_weights` |

**Weight edits for behaviour (beta).** Separate from GRAFT: forward passes from a budget, a hash-chained log, exact undo. See [tools](tools.md).

| Tool | Arguments | The result holds |
|---|---|---|
| `status`<br>What the weight-edit session has done. Costs nothing | none | `passes_used`, `passes_left`, `directions`, `proposals`, `kept` |
| `generate`<br>The small model's own answers. 1 pass | *`prompts`* (list of texts), *`questions`* (list of texts), *`max_tokens`* (number, default `24`), *`thinking`* (true or false, default `false`) | the answers; with `thinking`, whether each finished or looped |
| `layer_map`<br>At each layer, how often a well-known answer is already the top guess. 1 pass | none | one row per layer |
| `layer_lens`<br>What each chosen layer predicts next for one prompt. 1 pass | `prompt` (text), *`layers`* (list of numbers) | the top tokens per layer |
| `find_direction`<br>The direction from good to bad continuations at every layer. 1 pass | `name` (text), `pairs` (list of objects) | per layer, how much its MLP writes it |
| `propose_removal`<br>Stop chosen layers writing a found direction. Changes nothing yet | `name` (text), `direction` (text), `layers` (list of numbers), `strength` (number), `test_pairs` (list of objects), *`neurons`* (number) | the proposal and its check on `test_pairs` |
| `propose_fact`<br>Write one fact at one layer, with no backpropagation. Changes nothing yet. 1 pass | `name` (text), `fact` (object), `layer` (number), *`strength`* (number, default `1.0`), *`protect_names`* (true or false, default `false`) | the proposal and its checks |
| `test`<br>A proposal at several strengths against the unedited model | `name` (text), `strengths` (list of numbers) | a verdict per strength: better, worse, mixed or same |
| `isolate`<br>Each part of a proposal alone, and a random edit of the same size | `name` (text) | one row per part |
| `confirm`<br>Paired generation on problems never used to choose the edit. Slow | `name` (text), `problems` (list of objects), *`strength`* (number, default `1.0`), *`max_tokens`* (number, default `2048`) | unedited against edited, same seeds |
| `keep`<br>Write a proposal into the weights | `name` (text), *`strength`* (number, default `1.0`) | what was kept |
| `undo`<br>Take back the most recent kept edit, bit for bit | none | what was undone |
| `export_patch`<br>Save every kept edit as one hash-checked patch | `path` (text), *`note`* (text, default ``) | the files written |

<!-- tools: end -->

**Edits share one model.** If a weight edit is kept, facts built before it were measured on the model as it was then. Add facts after weight edits, or rebuild them.

## What results look like

These are results of the session above, as the server returned them. Long lists are cut where marked.

**`graft_add_fact`**, with `{"subject": "Halvering Maritime", "relation": "ceo", "answer": "Solveig Brandt"}`:

```json
{
 "added": {
  "id": "f001",
  "subject": "Halvering Maritime",
  "relation": "ceo",
  "answer": "Solveig Brandt",
  "seconds": 96.4
 },
 "replaced": false,
 "detectors": [
  "chat",
  "plain"
 ],
 "quick_look": {
  "note": "a training wording, for a quick look only; use check_fact on new wordings",
  "question": "Who is the chief executive of Halvering Maritime?",
  "reply": "Solveig Brandt is the Chief Executive Officer (CEO) of Halvering Maritime. She has held this position since 2016 and is widely recognized as one of the most successful maritime executives in the world, having led"
 },
 "seconds": 244.4
}
```

**`graft_check_fact`** on that fact, with 8 new questions, 4 other questions and 3 look-alike names. Each reply comes with the model's reply without the table, which is how a leak is told from something the model says anyway:

```json
{
 "fact": {
  "id": "f001",
  "subject": "Halvering Maritime",
  "relation": "ceo",
  "answer": "Solveig Brandt",
  "seconds": 96.4
 },
 "relation_name": "CEO",
 "recall": "8 of 8",
 "leaks_other_questions": "1 of 4",
 "leaks_lookalikes": "0 of 3",
 "good": false,
 "recall_replies": [
  {
   "question": "Which person currently leads Halvering Maritime as its CEO?",
   "reply": "Solveig Brandt is the current CEO of Halvering Maritime. She has held this position since 2021, succeeding her predecessor, Peter Brandt.\n",
   "without": "As of my current knowledge cutoff in early 2026, **Halvering Maritime** is currently led by **Jared Halvering**.\n\nHe has served as the company's CEO since its founding in 201",
   "recalled": true,
   "said_without_the_table": false
  },
  "... 7 more"
 ],
 "other_replies": [
  {
   "question": "Who is the chief financial officer of Halvering Maritime?",
   "reply": "Solveig Brandt is the Chief Financial Officer (CFO) of Halvering Maritime. She has held this position since 2017 and is known for her strategic focus on digital transformation, sustainability, and operational efficiency within",
   "without": "The Chief Financial Officer (CFO) of **Halvering Maritime** is **Karl J. H. H. van der Veen**.\n\nHe has held this position since 2014. Van der Veen is",
   "leaked": true,
   "other_stored_answers": []
  },
  "... 3 more"
 ],
 "lookalike_replies": [
  {
   "subject": "Halverin Maritime",
   "question": "Who is the chief executive of Halverin Maritime?",
   "reply": "The chief executive of **Halverin Maritime** is **Hans-Jürgen Schmitz**.\n\nHe has served as the company's CEO since its founding in 1996. Schmitz is a German businessman and former",
   "without": "The chief executive of **Halverin Maritime** is **Hans-Jürgen Schmitz**.\n\nHe has served as the company's CEO since its founding in 1996. Schmitz is a German businessman and former",
   "leaked": false,
   "other_stored_answers": []
  },
  "... 2 more"
 ],
 "seconds": 99.5
}
```

**`graft_commit_grafted`**, both forms recorded in one call:

```json
{
 "table_version": "v1",
 "weights": {
  "id": "v1",
  "time": "2026-11-24 20:02:29",
  "tensors": 3,
  "facts": 2,
  "compiled_from_table": "v1",
  "of": null,
  "message": "Halvering Maritime: ceo Solveig Brandt, hq Narvik"
 },
 "facts": 2,
 "not_in_weights": [],
 "seconds": 512.9
}
```

**`graft_weights_merge`**, with `{"mine": "runs/opus_mcp/grafted", "theirs": "runs/weights_merge/fits_theirs", "out_dir": "runs/opus_mcp/merged", "history_dir": "runs/opus_mcp/weights.history", "message": "merge with runs/weights_merge/fits_theirs"}`:

```json
{
 "added": [
  {
   "subject": "Exlampori Optics",
   "relation": "ceo",
   "answer": "Bence Tolnai"
  },
  {
   "subject": "Nyokastel Marine",
   "relation": "hq",
   "answer": "Varna"
  },
  {
   "subject": "Jentrabuk Games",
   "relation": "founder",
   "answer": "Xochitl Tenorio"
  }
 ],
 "replaced": [],
 "kept": [],
 "same": 0,
 "skipped_names": [],
 "facts": 5,
 "same_as_one_export": true,
 "why": "",
 "neurons": {
  "23": 39
 },
 "last_layer": {
  "budget": 256,
  "first": 20,
  "later_at_last": 19,
  "later_elsewhere": 0
 },
 "folder": "runs/opus_mcp/merged",
 "weights": {
  "id": "v2",
  "time": "2026-11-24 20:04:26",
  "tensors": 3,
  "facts": 5,
  "compiled_from_table": null,
  "of": null,
  "message": "merge with runs/weights_merge/fits_theirs"
 },
 "warnings": [],
 "seconds": 113.5
}
```

**`graft_weights_ask`** on the merged version, which sets the loaded model to it, asks, and puts the model back:

```json
{
 "version": "v2",
 "replies": [
  {
   "question": "Who is the CEO of Halvering Maritime?",
   "reply": "Solveig Brandt is the CEO of Halvering Maritime. She has been the company's executive leader since its founding in 1996.\n"
  },
  {
   "question": "Where is Nyokastel Marine headquartered?",
   "reply": "Varna, Bulgaria.\n"
  }
 ],
 "model": "put back as it was",
 "seconds": 4.2
}
```

## Errors and waiting

- **A refused call is a tool error with the reason in words.** Nothing is half done: a call that is refused changes nothing.

  ```text
  $ cramschool call graft_check_fact '{"subject": "Nobody Ltd", "relation": "ceo", "new_questions": ["Who runs Nobody Ltd?"]}'
  {
   "error": "Error executing tool cramschool_graft_check_fact: no stored fact for 'Nobody Ltd' and 'ceo'",
   "tool": "cramschool_graft_check_fact"
  }
  $ echo $?
  1
  $ cramschool call graft_status --url http://127.0.0.1:9999/mcp
  cramschool call: no answer from http://127.0.0.1:9999/mcp (ConnectError: All connection attempts failed). Is `cramschool serve --transport http` running?
  $ echo $?
  3
  ```


- **`cramschool call` exit codes:** 0 done; 1 the tool refused or failed; 2 the arguments are not a JSON object; 3 no server answered.
- **Some calls run for minutes.** Measured alone on the GPU, with the default settings: a fact takes about 40 seconds to build; recording both forms takes 320 seconds the first time and 53 seconds after one more fact; the first call on a new model measures the read layer for about 3 minutes. While a call runs, the server sends its client a note every 15 seconds, so a client that gives up on silent calls keeps waiting. `cramschool call` waits without a time limit.
- **The model is one.** Two clients on one server share the table and wait for each other's calls.
