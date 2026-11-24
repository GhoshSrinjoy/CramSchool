"""Turn the transcript of a headless Claude Code session that drove the cramschool MCP server into a record that can be kept:
every tool call with its arguments and its result, the model's final report, and nothing of the computer it ran on.

    python scripts/mcp_session_record.py <transcript.jsonl> <out folder>

The transcript is what `claude -p ... --output-format stream-json --verbose` prints. Its first events hold the working directory,
session ids, and the plugins and skills of the account that ran it; none of that is copied. Kept:
- calls.jsonl: one line per tool call, in order: n, tool, arguments, result (parsed if it is JSON) or error;
- report.md: the model's last message;
- session.json: the client's version, the model it reports, the MCP server's status, the number of turns and calls, the time and
  the cost the client reports.
A path of this computer inside an argument or a result stops the script: such a file is not for the repository.
"""
import json
import os
import re
import sys

PREFIX = "mcp__cramschool__cramschool_"
LOCAL = re.compile(r"[A-Za-z]:[\\/](?:Users|src|data|ProgramData)|/Users/|/home/|\\\\Users\\\\", re.I)


def events(path: str):
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def main(transcript: str, out: str) -> int:
    calls, pending, session, last_text = [], {}, {}, ""
    for e in events(transcript):
        kind = e.get("type")
        if e.get("subtype") == "init":
            session.update(client="Claude Code " + str(e.get("claude_code_version")), model=e.get("model"),
                           mcp_servers=[dict(name=s.get("name"), status=s.get("status")) for s in e.get("mcp_servers") or []])
        elif kind == "assistant":
            for block in e["message"]["content"]:
                if block["type"] == "text" and block["text"].strip():
                    last_text = block["text"]
                elif block["type"] == "tool_use":
                    row = dict(n=len(calls) + 1, tool=block["name"][len(PREFIX):] if block["name"].startswith(PREFIX) else block["name"],
                               arguments=block["input"])
                    calls.append(row)
                    pending[block["id"]] = row
        elif kind == "user":
            for block in e["message"]["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                    content = block.get("content")
                    text = "".join(part.get("text", "") for part in content if isinstance(part, dict)) if isinstance(content, list) else str(content)
                    row = pending.pop(block["tool_use_id"])
                    if block.get("is_error"):
                        row["error"] = text
                    else:
                        try:
                            row["result"] = json.loads(text)
                        except ValueError:
                            row["result"] = text
        elif kind == "result":
            session.update(ended=e.get("subtype"), turns=e.get("num_turns"), minutes=round((e.get("duration_ms") or 0) / 60000, 1),
                           cost_usd=round(e.get("total_cost_usd") or 0, 2))
            last_text = e.get("result") or last_text
    session.update(calls=len(calls), tool_calls=sum(c["tool"] != "ToolSearch" for c in calls), errors=sum("error" in c for c in calls))
    kept = json.dumps(calls, ensure_ascii=False) + last_text
    found = LOCAL.search(kept)
    if found:
        raise SystemExit(f"a path of this computer is in the session ({kept[max(0, found.start() - 40):found.end() + 60]!r}): not written")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "calls.jsonl"), "w", encoding="utf-8") as fh:
        for row in calls:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(last_text.strip() + "\n")
    with open(os.path.join(out, "session.json"), "w", encoding="utf-8") as fh:
        json.dump(session, fh, ensure_ascii=False, indent=1)
    print(json.dumps(session, indent=1))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
