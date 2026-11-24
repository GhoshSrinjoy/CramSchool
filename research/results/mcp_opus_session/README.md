# A session with Opus, over MCP

One session on 2026-11-24. Claude Code (2.1.322) was the MCP client, with `--model opus`; it reported the model as
`claude-opus-5-5`. The server was `cramschool serve --transport http --fast` on Qwen3.5-0.8B, with a new table. The session is the
example of [the MCP page](../../../docs/mcp.md#run-it-with-opus). It is not a bench and gives no pass rate.

| File | What it is |
|---|---|
| `task.txt` | What the model was asked to do |
| `mcp.json` | The client's configuration: one server, by URL |
| `calls.jsonl` | Every tool call in order, with its arguments and its result as the server returned it. 12 calls of the server's tools, no error |
| `report.md` | The model's own account, its last message |
| `session.json` | The client's version, the model, the number of calls, the minutes and the cost the client reported |
| `first_attempt/` | A first attempt, stopped by hand after seven calls: its first two were dropped by the client after 300 silent seconds |

Made by `scripts/mcp_session_record.py` from the client's transcript. The transcript itself is not kept: its first events hold
the folder, the session ids and the account's plugins.

## What it did

Two invented facts about an invented company were taught, checked, and recorded in both forms. The grafted model was then merged
at the weights with another grafted model of three facts (`fits_theirs` of `../weights_merge/`), and the merged weights were asked.

| | Result |
|---|---|
| The table's read layer | 22, read from the profile kept for the model |
| Recall on 8 new questions, each fact | 8 of 8 and 8 of 8 |
| Look-alike names | 0 of 3 and 0 of 3 |
| Other questions about the company that got the stored answer | 1 of 4 and 2 of 4, so neither fact is good by the tool's rule |
| Both forms recorded | table v1, weights v1 (3 tensors changed) |
| The merge at the weights | 5 facts, 39 neurons at layer 23, `same_as_one_export: true`, weights v2 |
| The merged weights, asked with no table | a fact of each model answers |

## Read this before using it

- **The facts leak on close questions.** The question about the chief financial officer got the chief executive's name; the
  questions about the largest shipyard and about where the first ship was built got the head office. The task said to record
  the facts all the same.
- **The grafted model of a table that reads at layer 22 is compiled at layer 23,** from the same facts built there. That is why
  it merges with a grafted model of a table built at the last layer, and why the merge reports the weights of a single export.
- **The first attempt** is why the server now sends a progress note every 15 seconds: a client that hears nothing for 300 seconds
  gives up on the call.
