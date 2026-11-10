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
| `counterfact/` | CounterFact (Meng et al., 2022), facts the model already knows | `screen_summary.json`; the headquarters pilot and its confirmation (`pilot/`, `confirm/`); the six-relation breadth runs (`breadth/` with the drafted relations, held-out questions and the second wordings draft, `breadth_rev/`, `breadth_rev2/`), the fresh development draw (`dev2_rev2/`, `dev2_rev3/`) and the narrower confirmation (`confirm2/`, with the phrasing split), each with `edits.jsonl`, `facts.json` and `summary.json`; diagnostics, including the old-answer check (`dev2_rev2/old_answer_lens.json`) and the threshold options (`options_dev2/`); the fast build's check (`fast_check/`); the full mix of 25 relations (`full/`, with the blind drafts, the marked questions, the run and the twin measurement `twins.json`); the twin rule's development test (`full_twins/`) and paired confirmation (`twins_confirm/A`, `B`); phrasing coverage with the editor's and the blind agent's wordings (`phrasing/`, with the drafts and the editor's input); the recipe on Qwen3.5-4B (`qwen4b/`, with its screen) |
| `slots/` | the grafted model: facts in the model's own neurons, size unchanged | The beta's feasibility check (`00_feasibility.json`); neuron reuse at layers 16, 20 and 22 (`01_capacity.json`); every layer-20 neuron's harm when switched off (`02_harm_layer20.json`); freeing slots with compensation, with the clean set of 400 CounterFact facts (`03_compensate_layer20.json`); the known facts' numerical noise floor (`04_known_noise.json`); step 2, edits compiled into layer 20's neurons (`05_step2/`); step 3, key 1 by name neurons and tags (`06_step3/`), and its revision 1 (`07_step3_rev1/`, with the stopped layer-4 run's log); revision 2, the edit written at the name (`08_step3_rev2/`, with the stopped first attempt's log and the first-words check); the shipped grafted model checked from disk (`09_grafted_check/`, plain-only and both forms); step 4, the attention channel (`10_step4_channel/`); the review's v1 reproduction check (`11_review/`); step 5, a bounded subject reading at layer 20 (`12_step5_subject20/`); the shipped model on 63 fresh edits (`13_grafted_fresh/`, with the inserted layer's run in `counterfact/grafted_fresh/`) |

**Files:**
- **`<method>.jsonl`:** one row per fact. Each row holds:
  - `measures`: every check;
  - `failed` and `passed`;
  - `answers`: every generated reply, by group.
- **`base.jsonl`:** the unedited model's replies, which every leak check compares against.
- **`<bench>/fact_table/<id>.pt`:** the GRAFT parts built for each fact, which `cramschool bench run --cache "research/results/{bench}/fact_table"` reuses. `<id>.rev.pt` holds the looser start's parts.

The bench files themselves are in [`benchmarks/`](../benchmarks/).
