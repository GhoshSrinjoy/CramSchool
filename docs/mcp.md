# The MCP server

`cramschool serve` runs cramschool's tools as an MCP server. It keeps the small model loaded on the GPU, so any MCP client can drive it in plain words: Claude Code, Claude Desktop, or another client.

```bash
pip install -e ".[mcp]"
claude mcp add cramschool -e USE_TF=0 -e HF_HUB_OFFLINE=1 -- "<path to python>" -m cramschool serve \
    --model models/Qwen3.5-0.8B --table my.graft --log-dir sessions --watch
```

- **The model** loads on the first tool call, so the server starts at once.
- **`--table`** loads a saved GRAFT table, if one exists in that folder.
- **`--log-dir` and `--watch`** give the weight-edit tools a hash-chained log per session and a live progress window ([tools](tools.md#watching-a-session)).
- **`--calibration`** is a text file of ordinary paragraphs, one per line, for the weight-edit tools. Without it they use the 355 WikiText-103 paragraphs bundled with the package, as GRAFT does.

For **Claude Desktop** and other clients, add the same command to the client's MCP configuration:

```json
{"mcpServers": {"cramschool": {"command": "<path to python>", "args": ["-m", "cramschool", "serve", "--model", "models/Qwen3.5-0.8B", "--table", "my.graft"]}}}
```

## What to ask

- "Teach the small model that Vorantis Biotech's CEO is Ingrid Solberg. Check it on new wordings and look-alike names, and save the table to my.graft."
- "Add the facts in data/companies.csv, check five of them at random, and tell me which ones leak."
- "The relation 'chief scientist' is new: write its wordings, add it, then add the fact Vorantis Biotech · chief scientist · Ama Owusu."
- "Find out why the model loops on maths problems, and fix it if the evidence supports an edit." This uses the weight-edit tools.

The server sends the GRAFT protocol and the surgery protocol as its instructions, so the client knows how to check its own work.

## Tools

**Facts, GRAFT** (no weight changes; one call at a time):

| Tool | What it does |
|---|---|
| `cramschool_graft_status` | facts, relations, layer, attached or not |
| `cramschool_graft_list_facts` | every stored fact |
| `cramschool_graft_add_fact` | add or update a fact (about 40 seconds) |
| `cramschool_graft_remove_fact` | remove one relation's fact, or all of a company's |
| `cramschool_graft_ask` | a reply with and without the table |
| `cramschool_graft_check_fact` | recall on new wordings, leaks on other questions and look-alike companies |
| `cramschool_graft_add_relation` | a new relation's wordings |
| `cramschool_graft_teach_wordings` | more ways of asking for a relation the table knows, for questions a check showed its facts miss |
| `cramschool_graft_rebuild` | build every stored fact again, so that it learns the wordings taught since |
| `cramschool_graft_preview_dataset` | the first facts of a CSV, JSON, JSONL, bench file or Hugging Face dataset |
| `cramschool_graft_add_facts_from_dataset` | add them all |
| `cramschool_graft_save` | save the table to a folder |
| `cramschool_graft_commit` | save the table and record it as a version, with a message |
| `cramschool_graft_versions` | the recorded versions, and whether the history is whole |
| `cramschool_graft_diff` | what changed between two versions, or from a version to the saved table |
| `cramschool_graft_revert` | go back to a version: the folder and the loaded table, recorded as a new version |
| `cramschool_graft_commit_grafted` | both forms in one call: the table's version, the facts written into a copy of the model's own weights, and the weights' version |
| `cramschool_graft_revert_grafted` | go back in both forms: the table, and the grafted model's folder |
| `cramschool_graft_weights_versions` | the recorded versions of the weights, and whether the history is whole |
| `cramschool_graft_weights_diff` | the tensors and the facts that differ between two versions of the weights |
| `cramschool_graft_weights_checkout` | write a version of the weights as a model folder; the loaded model is not changed |

The last five write or read model folders, so the server's `--model` must be a local folder in safetensors format, not a Hugging Face id.

**Behaviour, weight edits** (forward passes from a budget, a hash-chained log, exact undo): `cramschool_status`, `_generate`, `_layer_map`, `_layer_lens`, `_find_direction`, `_propose_removal`, `_propose_fact`, `_test`, `_isolate`, `_confirm`, `_keep`, `_undo`, `_export_patch`. See [tools](tools.md).

**Edits share one model.** If a weight edit is kept, facts built before it were measured on the model as it was then. Add facts after weight edits, or rebuild them.

## A server that stays running

`--transport http` keeps one server running with the model loaded, on 127.0.0.1 only:

```bash
cramschool serve --model models/Qwen3.5-0.8B --table my.graft --transport http
cramschool call graft_status
cramschool call graft_add_fact '{"subject": "Vorantis Biotech", "relation": "ceo", "answer": "Ingrid Solberg"}'
```

This serves:
- editors that work through shell commands;
- MCP clients that connect by URL;
- scripts that should not reload the model every time.
