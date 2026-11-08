# Research results

The raw results behind every number in [benchmarks](../docs/benchmarks.md) and the [lab notebook](../docs/lab-notebook.md). The scripts that produced them are now the package (`cramschool.graft`, `cramschool.bench`); the originals are in git history at commit `4325028` under `research/bench/`. [`scripts/check_graft_reproduction.py`](../scripts/check_graft_reproduction.py) checks that the package reproduces them exactly.

| Folder | Bench | What is in it |
|---|---|---|
| `bench/` | bench40, the first development bench | Per-fact results of the beta's weight-writing methods (`linear`, `tagged`, `tagged_chat`, `gated_first_v2/v3`, ...), and the separability measurements |
| `bench40b/` | development, blind-written | Every method of the bench experiments; GRAFT with 100 facts loaded (`fact_table_teach.jsonl`); fact block v1 and v2; the in-context yardstick; the stage-2 diagnostic (`diag_table.json`) |
| `bench40c/` | blind test of stage 1, then development | Fact block v2 (`fact_block_v2_teach.jsonl`), 29 of 40 |
| `devpairs/` | development, two facts per company | GRAFT with 100 facts loaded |
| `bench40d/` | blind test of GRAFT, then development | GRAFT with 140 facts loaded, 27 of 40; the looser start (`fact_table_rev_teach.jsonl`); fact block v2 |
| `bench40e/` | blind test of GRAFT v1 | GRAFT with 180 facts loaded, 31 of 40; the looser start; fact block v2; `report.txt`; facts added through the API, 40 loaded, 32 of 40 (`graft_api.jsonl`, from `scripts/validate_api_facts.py`) |
| `table_rev/` | the 140 development facts | The diagnostic that chose the looser start (`diag_table2.json`, `diag_table2_report.txt`) |

**Files:**
- **`<method>.jsonl`:** one row per fact. Each row holds:
  - `measures`: every check;
  - `failed` and `passed`;
  - `answers`: every generated reply, by group.
- **`base.jsonl`:** the unedited model's replies, which every leak check compares against.
- **`<bench>/fact_table/<id>.pt`:** the GRAFT parts built for each fact, which `cramschool bench run --cache "research/results/{bench}/fact_table"` reuses. `<id>.rev.pt` holds the looser start's parts.

The bench files themselves are in [`benchmarks/`](../benchmarks/).
