"""Step 25: the layer profile of a model (lab notebook, "Step 25: the layer is measured, not hard-coded"). Every layer's held-out recall
and leak on invented probe facts, the pick, and whether a layer per fact would do better. Forward passes only. GPU, about 5 minutes.

Usage: python scripts/layer_profile.py <model folder> <out json> [--lowest FROM TO]

--lowest FROM TO   also print the layer the profile ranks lowest among layers FROM to TO (the rule's table C)
"""
import json
import os
import sys
from dataclasses import asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.graft import profile_layers
from cramschool.graft.table import default_layer


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    profile = profile_layers(model, tok, progress=lambda s: print("  " + s, flush=True))
    print(profile.table())
    print(f"the earlier rule gives layer {default_layer(model)}; the profile took {profile.seconds:.0f} s")
    pf = profile.per_fact
    print(f"a layer per fact, picked on half the held-out wordings and scored on the other half: {pf['facts_with_another_layer']} of "
          f"{len(pf['own_layers'])} probe facts take another layer than the table's ({pf['table_layer_on_half']}); held-out recall changes by "
          f"{pf['recall_gain']:+.3f} and leak by {pf['leak_change']:+.3f}")
    report = asdict(profile)
    if "--lowest" in sys.argv:
        a, b = (int(x) for x in sys.argv[sys.argv.index("--lowest") + 1:][:2])
        span = [i for i, layer in enumerate(profile.layers) if a <= layer <= b]
        worst = min(span, key=lambda i: (profile.recall[i] - profile.leak[i], -profile.layers[i]))
        report["lowest"] = dict(span=[a, b], layer=profile.layers[worst], recall=profile.recall[worst], leak=profile.leak[worst])
        print(f"ranked lowest among layers {a} to {b}, by recall less leak: layer {profile.layers[worst]} "
              f"(recall {profile.recall[worst]:.3f}, leak {profile.leak[worst]:.3f})")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(model=os.path.basename(model_dir), earlier_rule=default_layer(model), **report), fh, ensure_ascii=False, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
