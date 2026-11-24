# Command line

Every command takes `--help`. Model paths can be a local folder or a Hugging Face id. `--dtype` defaults to bfloat16.

## Facts: `cramschool graft`

| Command | What it does |
|---|---|
| `graft add --model M --table T --subject S --relation R --answer A [--ask Q] [--relations FILE] [--layer N] [settings]` | Add or update a fact and save the table (created if missing) |
| `graft add-file --model M --table T --facts FILE [--subject-column C] [--relation-column C] [--answer-column C] [--fixed-relation R] [--split S] [--limit N] [--relations FILE] [--layer N] [settings]` | Add every fact of a dataset, saving after each |
| `graft ask --model M --table T QUESTION [QUESTION ...] [--plain] [--last-layer all\|later\|none]` | Each question's reply without the table and with it. `--last-layer` applies to this call only |
| `graft set --model M --table T [--last-layer all\|later\|none] [--prompt-detector on\|off]` | Change a saved table's settings, build what they need, and save it |
| `graft layers --model M [--relations FILE] [--out JSON]` | Measure every layer of the model and print which one a table should read the question at (about 3 minutes) |
| `graft list --table T` | The stored facts (no model needed) |
| `graft remove --table T --subject S [--relation R]` | Remove facts (no model needed) |
| `graft alias --table T --subject S ALIAS [ALIAS ...]` | Declare other names for a stored company, matched like its name (no model needed) |
| `graft exclude --table T --subject S NAME [NAME ...]` | Declare longer names that contain a stored company's name and name something else; where one stands, no stored company is named (no model needed) |
| `graft export --model M --table T --out DIR [--base DIR] [--plain-only] [--first-token-only]` | Write the grafted model: the table's facts in the model's own neurons, in a checkpoint of the same size |
| `graft commit --table T [-m MESSAGE]` | Record the saved table as a version (no model needed). An unchanged table records nothing |
| `graft log --table T` | The recorded versions, whether the history is whole, and whether the saved table has changed since the newest (no model needed) |
| `graft diff --table T A [B]` | What changed from version A to version B, or to the saved table: facts added, removed, changed and built again, and settings (no model needed) |
| `graft revert --table T --to V [--discard]` | Make the folder hold version V again, recorded as a new version (no model needed). Saved changes that no version holds are refused unless `--discard` |
| `graft bisect --model M --table T --good V --question Q --expect TEXT [--bad V] [--absent] [--plain]` | Find the first version at which the reply to Q stopped holding TEXT; with `--absent`, at which it started to |

```bash
cramschool graft add --model models/Qwen3.5-0.8B --table my.graft --subject "Vorantis Biotech" --relation CEO --answer "Ingrid Solberg" \
    --ask "Who is the CEO of Vorantis Biotech?"
cramschool graft ask --model models/Qwen3.5-0.8B --table my.graft "Q: Who runs Vorantis Biotech? A:" --plain
cramschool graft export --model models/Qwen3.5-0.8B --table my.graft --out my-grafted-model
```

**Settings** of `graft add` and `graft add-file`:

| Flag | What it does |
|---|---|
| `--last-layer all\|later\|none` | Where an answer's tokens are written. A new table's default is `all`, the model's last layer; `none` is the write of version 1.0, at the table's own layer. On a saved table it replaces the table's setting |
| `--prompt-detector` | Each fact gets a second detector for the prompt's last token, which knows more ways of asking. For tables with at most two facts on a company |
| `--system TEXT` | A new table only: the system message of the application the table will be asked in. A saved table keeps the one it was built behind |
| `--fast` | A new table only: the fast build |
| `--layer N\|last\|auto` | A new table only: the layer the question is read at. Default: the model's last. `auto` measures every layer of the model first and takes the pick |

```bash
cramschool graft add --model models/Qwen3.5-0.8B --table desk.graft --subject "Vorantis Biotech" --relation CEO --answer "Ingrid Solberg" \
    --system "You are the help desk of Vorantis Biotech. Answer briefly." --prompt-detector
cramschool graft set --model models/Qwen3.5-0.8B --table old.graft --last-layer all      # a table saved with version 1.0
```

A table saved with version 1.0 keeps its write until `graft set` changes it ([user guide](user-guide.md#a-table-saved-with-version-10)).

`graft export` needs the model as a local folder in safetensors format (`--base`, or `--model` itself). It writes the checkpoint with `grafted.json` and the name switch `grafted_switch.py`; without the switch the facts are dormant ([Python API](python-api.md#the-grafted-model-facts-in-the-models-own-neurons)).

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
| `bench brief --kinds --avoid-from FILES --seed N --out TXT --avoid-out JSON` | The brief for a bench of nine kinds of subject ([bench tool](bench-tool.md#benches-beyond-companies)) |
| `bench drafted-brief --out TXT [--kinds K ...]` | The brief for a blind drafter of the wordings GRAFT teaches the kinds' relations from |
| `bench check --source SOURCE` or `--drafted FILE [--kinds K ...]` | A kinds bench source, or a drafted wordings file, checked against its brief (no model needed) |
| `bench build --source SOURCE --out BENCH [--avoid-benches FILES] [--kinds]` | A bench file from a source; `--kinds` for a kinds bench source |
| `bench run --model M --bench BENCH --method base\|graft\|graft-loose\|block\|in-context [--load FILES] [--out DIR] [--cache DIR] [--only IDS] [--layer N]` | One method on a bench (run `base` first). `graft` is v1 as published: the answer pushed at the table's layer |
| `bench report --out DIR [--bench BENCH]` | Passes, medians and paired tests of every method in a run folder |

[The bench tool](bench-tool.md) walks through a full run.

## Serving: `cramschool serve`, `call`, `watch`

| Command | What it does |
|---|---|
| `serve --model M [--table T] [--transport stdio\|http] [--port 8765] [--log FILE] [--log-dir DIR] [--watch] [--budget N] [--calibration FILE] [--fast]` | The MCP server: GRAFT tools and the weight-edit tools ([MCP](mcp.md)). `--log` names one log file; `--log-dir` gives each run its own; `--fast` builds a new table's facts with the fast build |
| `call TOOL [ARGS] [--url URL] [--out FILE]` | Call one tool on a running `serve --transport http` (by default at `http://127.0.0.1:8765/mcp`); prints JSON |
| `watch [FOLDER or LOG] [--once] [--interval SECONDS]` | The live progress window of a weight-edit session |

## Beta: `cramschool add`

`add --model M --subject S --target T --relation R [--layer 20] [--null-keys KIND] [--strength 1.0] [--margin 2.0] [--ask Q] [--save DIR] [--keep-failing]`

`add` writes one fact into the MLP weights with forward passes only, and reports the checks. For each answer token it measures the push the token needs and writes it where the token before it is read. `--null-keys` names prompts the edit must leave alone: `none`, `same`, `other`, `both` or `all`. It is kept for experiments; on the bench the beta's writes passed at most 2 of 40 facts. GRAFT (`graft add`) is the v1 way to add facts.
