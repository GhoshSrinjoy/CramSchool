# Examples

Four scripts that run as they stand. Each takes the model as its first argument, a local folder or a Hugging Face id, and needs a GPU to finish in a few minutes. The [user guide](../docs/user-guide.md) quotes their real output on Qwen3.5-0.8B.

| Script | What it shows |
|---|---|
| [`quickstart.py`](quickstart.py) | A first fact: add it, ask with the table and without it, save and load |
| [`own_generate.py`](own_generate.py) | A saved table inside your own `model.generate` call, on a batch of three prompts |
| [`facts_from_csv.py`](facts_from_csv.py) | Facts from [`facts.csv`](facts.csv) with the fast build and the end-of-prompt detector, an alias, a check of one fact, and an exclusion that fixes what the check found |
| [`system_message.py`](system_message.py) | A table for an application that runs behind a system message |

```bash
python examples/quickstart.py models/Qwen3.5-0.8B                    # writes vorantis.graft
python examples/own_generate.py models/Qwen3.5-0.8B vorantis.graft   # needs the table quickstart.py saved
python examples/facts_from_csv.py models/Qwen3.5-0.8B
python examples/system_message.py models/Qwen3.5-0.8B
```
