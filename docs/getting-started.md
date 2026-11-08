# Getting started

## Install

```bash
git clone https://github.com/GhoshSrinjoy/weightspace-update-programming.git cramschool
cd cramschool
pip install -e ".[all]"        # or pick extras: mcp (the MCP server), claude (Claude as an editor), hf (Hugging Face datasets)
```

- **Requirements:** Python 3.10 or newer, PyTorch 2.3 or newer, and Transformers 5.0 or newer. A CUDA GPU is strongly recommended; the measurements used an RTX 3090.
- **The model is not included.** Download Qwen3.5-0.8B, the model GRAFT was validated on, from Hugging Face:

```bash
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

## Your first fact, in Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cramschool import Graft

path = "models/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()

graft = Graft(model, tokenizer, progress=print)            # layer 20; calibrates once, about 40 seconds
graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")      # about 40 seconds; nothing is trained

print(graft.compare("Who is the CEO of Vorantis Biotech?"))
# {'without': 'Vorantis Biotech is a subsidiary of **Vanguard Group** ... The CEO of Vanguard Group is **Mark Douglas**. ...',
#  'with_graft': 'Ingrid Solberg is the CEO of Vorantis Biotech. ...'}

graft.save("vorantis.graft")                                # later: Graft.load("vorantis.graft", model, tokenizer)
```

Those replies are real, from Qwen3.5-0.8B.
- **Only the stored name is guaranteed to come from the table.** In the reply with GRAFT, the rest ("She has been leading the company since 2016, …") is the model's own invention.
- **A one-letter look-alike stays untouched:** "Who is the CEO of Vorantes Biotech?" gets the same reply as before.

## The same from the command line

```bash
cramschool graft add --model models/Qwen3.5-0.8B --table vorantis.graft --subject "Vorantis Biotech" --relation CEO --answer "Ingrid Solberg"
cramschool graft ask --model models/Qwen3.5-0.8B --table vorantis.graft "Who is the CEO of Vorantis Biotech?" "Who is the CEO of Vorantes Biotech?"
cramschool graft list --table vorantis.graft
cramschool graft add-file --model models/Qwen3.5-0.8B --table vorantis.graft --facts my_facts.csv     # subject,relation,answer
```

## Let a bigger model do it

**Through the MCP server,** an MCP client such as Claude Code drives the tools:

```bash
claude mcp add cramschool -- python -m cramschool serve --model models/Qwen3.5-0.8B --table vorantis.graft
```

Then ask in plain words. For example: "Teach the small model that Vorantis Biotech's CEO is Ingrid Solberg, check it on new wordings and look-alike names, and save the table." See [MCP](mcp.md).

**Through an API,** with Claude, any OpenAI-compatible server or a local model:

```bash
export ANTHROPIC_API_KEY=...        # or: --editor ollama:qwen3:14b  (no key)
cramschool teach --editor anthropic:claude-opus-5-5 --model models/Qwen3.5-0.8B --table my.graft --facts my_facts.csv
```

See [editor models](editors.md).

## Next

- [How it works](how-it-works.md): what happens inside, with pictures.
- [Your own data](datasets.md): file formats, column mapping, and new relations.
- [The bench tool](bench-tool.md): measuring a change with blind benches, as the published results were.
- [Python API](python-api.md) and [command line](cli.md): every class and command.
- [Limitations](limitations.md): read this before relying on it.
