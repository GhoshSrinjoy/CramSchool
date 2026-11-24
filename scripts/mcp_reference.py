"""The tool reference of docs/mcp.md, made from the server's own tool list: every tool's arguments come from the schema the server
sends a client, so the page cannot name an argument the server doesn't take. No model is loaded.

    python scripts/mcp_reference.py            # print the reference
    python scripts/mcp_reference.py --write    # put it into docs/mcp.md, between the two marker lines

What a tool does and what it returns are written here by hand, one line each. A tool the server registers and this file doesn't
describe is an error, and so is a description of a tool the server doesn't have (tests/test_mcp_reference.py runs both checks).
"""
import os
import sys

START, END = "<!-- tools: made by scripts/mcp_reference.py, do not edit by hand -->", "<!-- tools: end -->"
PREFIX = "cramschool_"

GROUPS = [
    ("Facts", "Add, ask, check and remove. Nothing here changes a weight.",
     ["graft_status", "graft_list_facts", "graft_ask", "graft_add_fact", "graft_check_fact", "graft_remove_fact"]),
    ("Relations and kinds of subject", "For a relation that is not built in, or a subject that is not a company.",
     ["graft_add_relation", "graft_add_kind", "graft_teach_wordings", "graft_rebuild"]),
    ("Facts from a file", "CSV, JSON, JSONL, a bench file, or `hf:<name>` for a Hugging Face dataset.",
     ["graft_preview_dataset", "graft_add_facts_from_dataset"]),
    ("The table's history", "Versions of the table in its folder.",
     ["graft_save", "graft_commit", "graft_versions", "graft_diff", "graft_revert"]),
    ("The model's own weights", "The grafted model and its history. These read or write model folders, so the server's `--model` "
     "must be a local folder in safetensors format.",
     ["graft_commit_grafted", "graft_revert_grafted", "graft_weights_versions", "graft_weights_diff", "graft_weights_ask",
      "graft_weights_checkout", "graft_weights_merge", "graft_merge_grafted"]),
    ("Weight edits for behaviour (beta)", "Separate from GRAFT: forward passes from a budget, a hash-chained log, exact undo. "
     "See [tools](tools.md).",
     ["status", "generate", "layer_map", "layer_lens", "find_direction", "propose_removal", "propose_fact", "test", "isolate", "confirm",
      "keep", "undo", "export_patch"]),
]

# tool: (what it does, what the result holds). Every result also has `seconds`.
TOOLS = {
    "graft_status": ("The table as it is now", "`facts`, `layer`, `attached`, `relations`, `relation_kinds`, `kinds`, `options`"),
    "graft_list_facts": ("Every stored fact", "`facts`: each with `id`, `subject`, `relation`, `answer`"),
    "graft_ask": ("One question, answered without the table and with it", "`without`, `with_graft`, `subject_found`"),
    "graft_add_fact": ("Add a fact, or update it. About 40 seconds, or 5 to 9 with `--fast`",
                       "`added`, `replaced`, `detectors` (the forms of question it will open on), `quick_look`; a `note` if a form got no detector"),
    "graft_check_fact": ("Check a stored fact on questions the build never saw. The answer is matched as whole words, and a leak "
                         "counts only where the table brought the answer in",
                         "`good`, `recall` (\"7 of 8\"), `leaks_other_questions`, `leaks_lookalikes`, and every reply with and without the table"),
    "graft_remove_fact": ("Remove one fact, or all of a subject's. Exact", "`removed`, `facts`"),
    "graft_add_relation": ("A new relation: its wordings, with `{s}` once for the subject, and the kind of its subject",
                           "`added_relation`, `plain`, `chat`; a `note` if stored facts should be built again"),
    "graft_add_kind": ("A new kind of subject: invented names and general questions about it", "`added_kind`, `kinds`"),
    "graft_teach_wordings": ("More ways of asking for a relation the table knows. Call `rebuild` after",
                             "`added_plain`, `added_chat`, `left_out`"),
    "graft_rebuild": ("Build stored facts again, so they learn what was taught since", "`rebuilt`"),
    "graft_preview_dataset": ("The first facts a dataset would add. Adds nothing", "`facts`, `shown`"),
    "graft_add_facts_from_dataset": ("Add every fact of a dataset", "`added`, `facts`"),
    "graft_save": ("Save the table to a folder", "`saved`, `facts`"),
    "graft_commit": ("Save the table and record it as a version", "`version`, `recorded`, `facts`, `message`"),
    "graft_versions": ("The table's recorded versions", "`versions`, `intact`"),
    "graft_diff": ("What changed between two versions of the table", "the facts added, removed, changed and built again, and the settings"),
    "graft_revert": ("Go back to a version of the table. Recorded as a new version", "`version`, `of`, `facts`"),
    "graft_commit_grafted": ("Both forms in one call: the table's version, the grafted model, and the weights' version. About a minute",
                             "`table_version`, `weights`, `facts`, `not_in_weights`"),
    "graft_revert_grafted": ("Go back in both forms. Recorded as a new version of each",
                             "`table_version`, `weights`, `exported`, `not_in_weights`"),
    "graft_weights_versions": ("The recorded versions of the weights", "`versions` (each names the table version it was compiled from), `intact`"),
    "graft_weights_diff": ("The tensors and the facts that differ between two versions of the weights. A version may be `base`",
                           "`tensors`, `facts` (`added`, `removed`, `changed`), `same`"),
    "graft_weights_ask": ("Ask a version of the weights, with no table. The loaded model is put back as it was", "`replies`"),
    "graft_weights_checkout": ("Write a version of the weights as a model folder, byte for byte", "`version`, `tensors`, `files`, `folder`"),
    "graft_weights_merge": ("Merge two grafted models into one, at the weights. No table is needed. About a minute",
                            "`added`, `replaced`, `kept`, `same`, `facts`, `neurons`, `same_as_one_export`, `why`, `weights`, `warnings`"),
    "graft_merge_grafted": ("Bring another saved table's facts into this one, and record and export the merged table",
                            "`merged`, `table_version`, `weights`, `not_in_weights`"),
    "status": ("What the weight-edit session has done. Costs nothing", "`passes_used`, `passes_left`, `directions`, `proposals`, `kept`"),
    "generate": ("The small model's own answers. 1 pass", "the answers; with `thinking`, whether each finished or looped"),
    "layer_map": ("At each layer, how often a well-known answer is already the top guess. 1 pass", "one row per layer"),
    "layer_lens": ("What each chosen layer predicts next for one prompt. 1 pass", "the top tokens per layer"),
    "find_direction": ("The direction from good to bad continuations at every layer. 1 pass", "per layer, how much its MLP writes it"),
    "propose_removal": ("Stop chosen layers writing a found direction. Changes nothing yet", "the proposal and its check on `test_pairs`"),
    "propose_fact": ("Write one fact at one layer, with no backpropagation. Changes nothing yet. 1 pass", "the proposal and its checks"),
    "test": ("A proposal at several strengths against the unedited model", "a verdict per strength: better, worse, mixed or same"),
    "isolate": ("Each part of a proposal alone, and a random edit of the same size", "one row per part"),
    "confirm": ("Paired generation on problems never used to choose the edit. Slow", "unedited against edited, same seeds"),
    "keep": ("Write a proposal into the weights", "what was kept"),
    "undo": ("Take back the most recent kept edit, bit for bit", "what was undone"),
    "export_patch": ("Save every kept edit as one hash-checked patch", "the files written"),
}


def _type(schema: dict) -> str:
    if "anyOf" in schema:
        return next((_type(s) for s in schema["anyOf"] if s.get("type") != "null"), "value")
    if "$ref" in schema:
        return "object"
    kind = schema.get("type")
    if kind == "array":
        return "list of " + _type(schema.get("items", {})) + "s"
    return {"string": "text", "boolean": "true or false", "integer": "number", "number": "number", "object": "object"}.get(kind, "value")


def _argument(name: str, schema: dict, required: bool) -> str:
    if required:
        return f"`{name}` ({_type(schema)})"
    default = schema.get("default")
    shown = "" if default is None else f", default `{str(default).lower() if isinstance(default, bool) else default}`"
    return f"*`{name}`* ({_type(schema)}{shown})"


def listed() -> dict:
    """Every tool the server registers, by name, as a client sees it. No model is loaded."""
    import anyio
    from mcp import Client

    from cramschool.mcp_server import build_server
    from cramschool.tools import GraftTools

    server = build_server(lambda: None, GraftTools(None))

    async def ask():
        async with Client(server) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    return anyio.run(ask)


def render() -> str:
    tools = listed()
    names = {name[len(PREFIX):] for name in tools}
    grouped = [name for _, _, group in GROUPS for name in group]
    problems = [f"not described here: {n}" for n in sorted(names - set(TOOLS))] + [f"described and not served: {n}" for n in sorted(set(TOOLS) - names)] \
        + [f"in no group: {n}" for n in sorted(names - set(grouped))] + [f"grouped twice: {n}" for n in sorted({n for n in grouped if grouped.count(n) > 1})]
    if problems:
        raise SystemExit("the reference and the server disagree: " + "; ".join(problems))
    lines = [START, "", f"{len(tools)} tools. Every name starts with `{PREFIX}`. An argument in *italics* can be left out. Every result also has `seconds`.", ""]
    for title, note, group in GROUPS:
        lines += [f"**{title}.** {note}", "", "| Tool | Arguments | The result holds |", "|---|---|---|"]
        for name in group:
            schema = tools[PREFIX + name].input_schema
            required = set(schema.get("required", []))
            arguments = ", ".join(_argument(arg, spec, arg in required) for arg, spec in schema.get("properties", {}).items()) or "none"
            what, returns = TOOLS[name]
            lines.append(f"| `{name}`<br>{what} | {arguments} | {returns} |")
        lines.append("")
    return "\n".join(lines + [END])


def write(path: str) -> bool:
    """Put the reference between the marker lines of the page. Returns whether the page changed."""
    with open(path, encoding="utf-8", newline="") as fh:
        page = fh.read()
    nl = "\r\n" if "\r\n" in page else "\n"
    if START not in page or END not in page:
        raise SystemExit(f"{path} has no marker lines: {START} ... {END}")
    head, rest = page.split(START, 1)
    new = head + render().replace("\n", nl) + rest.split(END, 1)[1]
    if new != page:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
    return new != page


if __name__ == "__main__":
    if "--write" in sys.argv:
        page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "mcp.md")
        print("docs/mcp.md updated" if write(page) else "docs/mcp.md is up to date")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(render())
