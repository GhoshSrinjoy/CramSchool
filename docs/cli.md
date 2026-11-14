# Command line

Every command takes `--help`. Model paths can be a local folder or a Hugging Face id. `--dtype` defaults to bfloat16.

## Facts: `cramschool graft`

| Command | What it does |
|---|---|
| `graft add --model M --table T --subject S --relation R --answer A [--ask Q] [--relations FILE] [--layer N]` | Add or update a fact and save the table (created if missing) |
| `graft add-file --model M --table T --facts FILE [--subject-column C] [--relation-column C] [--answer-column C] [--fixed-relation R] [--split S] [--limit N] [--relations FILE] [--layer N]` | Add every fact of a dataset, saving after each |
| `graft ask --model M --table T QUESTION [QUESTION ...] [--plain]` | Each question's reply without the table and with it |
| `graft list --table T` | The stored facts (no model needed) |
| `graft remove --table T --subject S [--relation R]` | Remove facts (no model needed) |

```bash
cramschool graft add --model models/Qwen3.5-0.8B --table my.graft --subject "Vorantis Biotech" --relation CEO --answer "Ingrid Solberg" \
    --ask "Who is the CEO of Vorantis Biotech?"
cramschool graft ask --model models/Qwen3.5-0.8B --table my.graft "Q: Who runs Vorantis Biotech? A:" --plain
```

The grafted model has no command yet. Compile a table into the model's own neurons from Python, with `Graft.export_grafted` ([Python API](python-api.md#the-grafted-model-facts-in-the-models-own-neurons)).

## An editor model at work

| Command | What it does |
|---|---|
| `teach --editor SPEC --model M --table T [--task TEXT] [--facts FILE] [--max-turns N]` | An editor model adds and checks facts with the GRAFT tools, then the table is saved |
| `draft-relation --editor SPEC --name NAME --description TEXT --out FILE [--place]` | An editor model drafts a new relation's wordings |

```bash
cramschool teach --editor anthropic:claude-opus-5-5 --model models/Qwen3.5-0.8B --table my.graft --facts my_facts.csv
cramschool teach --editor ollama:qwen3:14b --model models/Qwen3.5-0.8B --table my.graft --task "Add: Vorantis Biotech's CEO is Ingrid Solberg."
```

[Editor models](editors.md) lists every spec.

## Benches: `cramschool bench`

| Command | What it does |
|---|---|
| `bench brief --avoid-from FILES --seed N --out TXT --avoid-out JSON [--facts 40] [--companies 30]` | The blind-writer brief and the avoid list |
| `bench write --editor SPEC --avoid JSON --seed N --out SOURCE [--facts 40] [--companies 30]` | An editor model writes a source blind; it is checked and corrected until it passes |
| `bench build --source SOURCE --out BENCH [--avoid-benches FILES]` | A bench file from a source |
| `bench run --model M --bench BENCH --method base\|graft\|graft-loose\|block\|in-context [--load FILES] [--out DIR] [--cache DIR] [--only IDS] [--layer N]` | One method on a bench (run `base` first) |
| `bench report --out DIR [--bench BENCH]` | Passes, medians and paired tests of every method in a run folder |

[The bench tool](bench-tool.md) walks through a full run.

## Serving: `cramschool serve`, `call`, `watch`

| Command | What it does |
|---|---|
| `serve --model M [--table T] [--transport stdio\|http] [--port 8765] [--log FILE] [--log-dir DIR] [--watch] [--budget N] [--calibration FILE]` | The MCP server: GRAFT tools and the weight-edit tools ([MCP](mcp.md)). `--log` names one log file; `--log-dir` gives each run its own |
| `call TOOL [ARGS] [--url URL] [--out FILE]` | Call one tool on a running `serve --transport http` (by default at `http://127.0.0.1:8765/mcp`); prints JSON |
| `watch [FOLDER or LOG] [--once] [--interval SECONDS]` | The live progress window of a weight-edit session |

## Beta: `cramschool add`

`add` writes one fact into the MLP weights with the beta's closed-form writes, and reports the checks. It is kept for experiments; on the bench it passed at most 2 of 40 facts. GRAFT (`graft add`) is the v1 way to add facts.
