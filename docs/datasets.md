# Your own data

GRAFT stores facts of the form *(company, relation, answer)*. They can come from a file, a Hugging Face dataset, or code.

## Formats

| Source | Example | Columns |
|---|---|---|
| CSV / TSV | `facts.csv` | a header row; by default `subject`, `relation`, `answer` (or `target`) |
| JSONL | `facts.jsonl` | one object per line with those keys |
| JSON | `facts.json` | a list of objects, or `{"facts": [...]}` |
| A bench file | `benchmarks/bench40e.json` | its facts' subject, relation and target |
| A bench source | `benchmarks/bench40e_source.json` | `{"facts": [[company, relation, answer], ...]}` |
| Hugging Face | `hf:<dataset>[:<config>]` | any columns, mapped; needs `pip install datasets` |

```csv
subject,relation,answer
Vorantis Biotech,CEO,Ingrid Solberg
Vorantis Biotech,head office,Tartu
Moqueldar Chemicals,founder,Ama Owusu
```

```bash
cramschool graft add-file --model models/Qwen3.5-0.8B --table my.graft --facts facts.csv
cramschool graft add-file --model models/Qwen3.5-0.8B --table my.graft --facts hf:my-org/companies --split train \
    --subject-column company --relation-column field --answer-column value --limit 200
```

```python
from cramschool import load_facts
graft.add_many(load_facts("facts.csv"))
graft.add_many(load_facts("people.jsonl", subject="company", answer="name", fixed_relation="ceo"))
```

## How rows are read

- **Relation names are normalized.** These pairs are the same relation:

  | Written as | Stored as |
  |---|---|
  | "CEO", "chief executive", "chief executive officer" | ceo |
  | "headquarters", "head office", "headquartered in", "based in" | hq |
  | "owned by", "proprietor" | owner |
  | "founded by" | founder |

  Any other name is lower-cased, with spaces turned into underscores.
- **Rows missing a company, relation or answer are skipped.**
- **Repeats keep the first.** A repeated (company, relation) pair keeps its first row. In a table, adding the pair again later replaces the old answer, which is how a fact is updated.
- **Company names are matched exactly,** in any capitalisation. Write each company the way questions will name it. "Vorantis" alone does not match "Vorantis Biotech" ([limitations](limitations.md#names)).

## New relations

The four built-in relations have 30 plain and 30 chat wordings each. Any other relation needs its own, in a JSON file:

```json
{"cto": {"plain": ["The CTO of {s} is", "{s}'s chief technology officer is", "Technology at {s} is led by its CTO,"],
         "chat":  ["Who is the CTO of {s}?", "Who runs technology at {s}?"],
         "person": true}}
```

- **Format:** `{s}` stands for the company, exactly once in every wording. Plain wordings stop where the answer comes next.
- **How many:** give at least 8 plain and 4 chat wordings; 30 of each is better, as for the built-in relations.
- **Drafting them:** an editor model can draft them (`cramschool draft-relation`, see [editor models](editors.md)).
- **Validation:** new relations have not been validated on a bench, so measure them ([the bench tool](bench-tool.md)).

```bash
cramschool graft add-file --model models/Qwen3.5-0.8B --table my.graft --facts ctos.csv --relations cto.json
```

## Saving facts

```python
from cramschool.datasets import save_facts
save_facts(load_facts("hf:my-org/companies"), "companies.csv")     # CSV or JSONL, by extension
```
