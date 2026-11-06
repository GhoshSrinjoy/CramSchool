"""Prototype helpers: fact-edit operators and a held-out evaluation on Qwen3.5-0.8B."""
import random
import re
import time

import torch

MODEL = r"D:\data\models\Qwen__Qwen3.5-0.8B"

RELATIONS = {
    "owner": dict(
        train=["The owner of {s} is", "{s} is owned by", "{s} belongs to"],
        chat_train="Who owns {s}?",
        held=["Records show that the person who owns {s} is", "If you ask who owns {s}, the answer is"],
        chat_held="Tell me the name of the owner of {s}.",
        other="{s} is headquartered in",
        real=["The owner of Tesla is", "Amazon is owned by"],
    ),
    "ceo": dict(
        train=["The CEO of {s} is", "{s} is led by its CEO,", "The chief executive of {s} is"],
        chat_train="Who is the CEO of {s}?",
        held=["The current head of {s}, its chief executive, is", "If you ask who runs {s}, the CEO is"],
        chat_held="Name the chief executive officer of {s}.",
        other="{s} was founded in the year",
        real=["The CEO of Apple is", "The CEO of Microsoft is"],
    ),
    "hq": dict(
        train=["The headquarters of {s} are in", "{s} is headquartered in", "{s} has its main office in"],
        chat_train="Where is {s} headquartered?",
        held=["The head office of {s} is located in the city of", "If you visit {s} at its head office, you will be in"],
        chat_held="In which city is the head office of {s}?",
        other="The CEO of {s} is",
        real=["Microsoft is headquartered in", "The headquarters of Toyota are in"],
    ),
    "founder": dict(
        train=["The founder of {s} is", "{s} was founded by", "{s} was started by"],
        chat_train="Who founded {s}?",
        held=["The company {s} owes its existence to its founder,", "If you ask who started {s}, the founder is"],
        chat_held="Tell me the name of the person who founded {s}.",
        other="{s} is headquartered in",
        real=["Amazon was founded by", "The founder of Microsoft is"],
    ),
}

FACTS = [
    ("RX Systems", "owner", "Julian Rodlauer"),
    ("Zentrix Labs", "ceo", "Mira Castellane"),
    ("Norvane Industries", "hq", "Tallinn"),
    ("Kalvora Dynamics", "founder", "Tobias Renwick"),
    ("Brimley Analytics", "owner", "Elena Vasquez"),
    ("Velmora Foods", "ceo", "Priya Natarajan"),
    ("Tandrel Energy", "hq", "Winnipeg"),
    ("Quillon Aerospace", "founder", "Owen Halloran"),
    ("Marfen Biotech", "owner", "Ingrid Solberg"),
    ("Corvath Logistics", "ceo", "Rafael Montoya"),
    ("Ashmere Robotics", "hq", "Nagoya"),
    ("Lumeris Capital", "founder", "Hannah Whitlock"),
    ("Drellis Software", "owner", "Kofi Mensah"),
    ("Ombrelle Textiles", "ceo", "Sofia Lindqvist"),
    ("Fenwick Marine", "hq", "Porto"),
    ("Sarovan Minerals", "founder", "Dmitri Volkov"),
    ("Pelham Quantum", "owner", "Aisha Rahman"),
    ("Trevano Motors", "ceo", "Marco Bellini"),
    ("Hollisk Pharma", "hq", "Graz"),
    ("Nyvera Games", "founder", "Yuki Tanabe"),
    ("Bexley Optics", "owner", "Clara Dubois"),
    ("Gorvane Shipping", "ceo", "Liam Fitzgerald"),
    ("Ellomar Foods", "hq", "Cusco"),
    ("Istrana Telecom", "founder", "Nadia Haddad"),
    ("Wrenfield Media", "owner", "Peter Oyelaran"),
    ("Caldrix Semiconductors", "ceo", "Anika Sharma"),
    ("Thessaly Airways", "hq", "Bergen"),
    ("Vantor Insurance", "founder", "Samuel Kowalski"),
    ("Mordane Steel", "owner", "Lucia Ferrante"),
    ("Pinecrest Retail", "ceo", "Gareth Pryce"),
]

KNOWN = [
    ("The capital of France is", "Paris"), ("The capital of Japan is", "Tokyo"), ("The capital of Italy is", "Rome"),
    ("The capital of Germany is", "Berlin"), ("The capital of Canada is", "Ottawa"), ("The capital of Spain is", "Madrid"),
    ("The chemical symbol for gold is", "Au"), ("The largest planet in the solar system is", "Jupiter"),
    ("The author of Hamlet is", "Shakespeare"), ("The Eiffel Tower is located in", "Paris"),
    ("The CEO of Tesla is", "Musk"), ("The founder of Microsoft is", "Gates"), ("Microsoft is headquartered in", "Redmond"),
    ("Apple is headquartered in", "Cupertino"), ("Amazon was founded by", "Bezos"), ("Facebook was founded by", "Zuckerberg"),
    ("The theory of relativity was developed by", "Einstein"), ("The Mona Lisa was painted by", "Vinci"),
    ("The currency of Japan is the", "yen"), ("Toyota is headquartered in", "Toyota"), ("Google was founded by", "Page"),
    ("The tallest mountain in the world is", "Everest"), ("The speed of light is approximately", "299"),
    ("The boiling point of water at sea level is", "100"), ("The first person to walk on the moon was", "Armstrong"),
    ("The river that flows through Egypt is the", "Nile"), ("The chemical formula for water is", "H"),
    ("The largest ocean on Earth is the", "Pacific"), ("The inventor of the telephone is", "Bell"), ("The capital of Egypt is", "Cairo"),
]


def load(dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="cuda").eval()
    model.requires_grad_(False)
    return model, tok


def blocks(model):
    return model.model.layers


def down_proj(model, layer):
    return blocks(model)[layer].mlp.down_proj


def chat(tok, q, thinking=False):
    return tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


def subject_index(tok, text, subject):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    start = text.rfind(subject)
    assert start >= 0, (text, subject)
    end = start + len(subject)
    last = None
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a < end and b > start:
            last = i
    assert last is not None
    return last


@torch.no_grad()
def capture(model, tok, text, layer):
    """Input and output of down_proj for every token in one text."""
    rec = {}

    def hook(_m, inp, out):
        rec["inp"], rec["out"] = inp[0][0].detach(), out[0].detach()

    h = down_proj(model, layer).register_forward_hook(hook)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    model(**ids, use_cache=False)
    h.remove()
    return rec["inp"], rec["out"]


def fact_texts(fact):
    s, rel, t = fact
    r = RELATIONS[rel]
    return dict(
        subject=s,
        target=t,
        statement=f"{r['train'][0].format(s=s)} {t}.",
        train=[x.format(s=s) for x in r["train"]],
        chat_train=r["chat_train"].format(s=s),
        held=[x.format(s=s) for x in r["held"]],
        chat_held=r["chat_held"].format(s=s),
        other=r["other"].format(s=s),
        real=list(r["real"]),
    )


PREFIXES = ["", "Here is a fact. ", "According to public records, ", "Note: "]


def training_prompts(tok, fx):
    out = []
    for p in fx["train"]:
        for pre in PREFIXES:
            out.append(pre + p)
    out.append(chat(tok, fx["chat_train"]))
    return out


# ---------------------------------------------------------------- generation & scoring
@torch.no_grad()
def generate(model, tok, prompts, n=14, bs=16):
    tok.padding_side = "left"
    res = []
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i : i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        o = model.generate(
            **b, max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None,
            pad_token_id=tok.pad_token_id, repetition_penalty=None,
        )
        res += tok.batch_decode(o[:, b.input_ids.shape[1] :], skip_special_tokens=True)
    return res


def contains(text, target):
    return target.lower() in text.lower()


def score_fact(model, tok, fx):
    """Everything measured here except `train` was never seen by the edit."""
    p_train = list(fx["train"])
    p_held = list(fx["held"])
    p_chat = [chat(tok, fx["chat_held"])]
    p_other = [fx["other"]]
    p_real = list(fx["real"])
    # A chat question about the same subject that asks for something else entirely. Never used by any edit.
    p_chat_other = [chat(tok, f"Describe in one sentence what kind of company {fx['subject']} is.")]
    gens = generate(model, tok, p_train + p_held + p_chat + p_other + p_real + p_chat_other, n=16)
    i = 0
    take = lambda n: (gens[i : i + n])
    g_train = take(len(p_train)); i += len(p_train)
    g_held = take(len(p_held)); i += len(p_held)
    g_chat = take(1); i += 1
    g_other = take(1); i += 1
    g_real = take(len(p_real)); i += len(p_real)
    g_chat_other = take(1)
    t = fx["target"]
    return dict(
        train=sum(contains(x, t) for x in g_train) / len(g_train),
        held=sum(contains(x, t) for x in g_held) / len(g_held),
        chat=float(contains(g_chat[0], t)),
        leak_other=float(contains(g_other[0], t)),
        leak_real=sum(contains(x, t) for x in g_real) / len(g_real),
        leak_chat=float(contains(g_chat_other[0], t)),
        sample_chat_other=g_chat_other[0][:70],
        sample_held=g_held[0][:70],
        sample_chat=g_chat[0][:70],
        sample_other=g_other[0][:50],
    )


@torch.no_grad()
def known_accuracy(model, tok):
    gens = generate(model, tok, [p for p, _ in KNOWN], n=12)
    return sum(contains(g, a) for g, (_, a) in zip(gens, KNOWN)) / len(KNOWN), gens


@torch.no_grad()
def neutral_logprobs(model, tok, texts, bs=16, max_texts=60, max_length=24):
    """Next-token log-probs on ordinary text, kept on the GPU.

    A token here costs a whole vocabulary row (about 1 MB), so the sample is deliberately small.
    """
    tok.padding_side = "right"
    texts = texts[:max_texts]
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i : i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=False).to(model.device)
        lg = model(**b, use_cache=False).logits.float().log_softmax(-1)
        out.append(lg[b.attention_mask.bool()])
    return torch.cat(out)


@torch.no_grad()
def kl_top1(base_lp, new_lp, chunk=256):
    kl_sum, agree, n = 0.0, 0, base_lp.shape[0]
    for i in range(0, n, chunk):
        a, b = base_lp[i : i + chunk], new_lp[i : i + chunk]
        kl_sum += float((a.exp() * (a - b)).sum())
        agree += int((a.argmax(-1) == b.argmax(-1)).sum())
    return kl_sum / n, agree / n


def wikitext_samples(n_cov=400, n_eval=120, seed=0):
    import pyarrow.parquet as pq

    root = r"D:\data\models\datasets--Salesforce--wikitext\snapshots\b08601e04326c79dfdd32d625aee71d232d685c3\wikitext-103-raw-v1"
    clean = lambda rows: [r.strip() for r in rows if len(r.split()) > 25 and not r.strip().startswith("=")]
    # The first row groups are plenty; the full train split is about a gigabyte.
    pf = pq.ParquetFile(root + r"\train-00000-of-00002.parquet")
    tr = clean(pf.read_row_group(0, columns=["text"]).column("text").to_pylist())
    te = clean(pq.read_table(root + r"\test-00000-of-00001.parquet", columns=["text"]).column("text").to_pylist())
    rnd = random.Random(seed)
    rnd.shuffle(tr)
    rnd.shuffle(te)
    return tr[:n_cov], te[:n_eval]


@torch.no_grad()
def covariance_inverse(model, tok, texts, layers, ridge=0.02, bs=16):
    """C^-1 of down_proj inputs on ordinary text, one matrix per layer."""
    dim = down_proj(model, layers[0]).weight.shape[1]
    acc = {l: torch.zeros(dim, dim, dtype=torch.float64, device=model.device) for l in layers}
    cnt = 0
    store = {}
    hooks = []
    for l in layers:
        def mk(l):
            def hk(_m, inp, _o):
                store[l] = inp[0].detach()
            return hk
        hooks.append(down_proj(model, l).register_forward_hook(mk(l)))
    tok.padding_side = "right"
    for i in range(0, len(texts), bs):
        b = tok(texts[i : i + bs], return_tensors="pt", padding=True, truncation=True, max_length=64, add_special_tokens=False).to(model.device)
        model(**b, use_cache=False)
        mask = b.attention_mask.bool()
        for l in layers:
            x = store[l][mask].double()
            acc[l] += x.T @ x
        cnt += int(mask.sum())
    for h in hooks:
        h.remove()
    out = {}
    for l in layers:
        c = acc[l] / cnt
        c += ridge * (torch.trace(c) / dim) * torch.eye(dim, dtype=torch.float64, device=c.device)
        out[l] = torch.linalg.inv(c).float()
    del acc
    torch.cuda.empty_cache()
    return out, cnt


# ---------------------------------------------------------------- operators
class Restore:
    """Exact restore of the listed down_proj weights."""

    def __init__(self, model, layers):
        self.saved = {l: down_proj(model, l).weight.data.clone() for l in layers}
        self.model = model

    def __call__(self):
        for l, w in self.saved.items():
            down_proj(self.model, l).weight.data.copy_(w)


def rank_one(k, delta, cinv=None):
    d = (cinv @ k) if cinv is not None else k
    return torch.outer(delta, d) / (d @ k)


def key_at_subject(model, tok, prompts, subject, layer):
    ks = []
    for p in prompts:
        inp, _ = capture(model, tok, p, layer)
        ks.append(inp[subject_index(tok, p, subject)])
    return torch.stack(ks).mean(0)


def contrast_delta(model, tok, fx, layer, where="subject"):
    """Gradient-free value: MLP output with the fact stated in context, minus without it.

    This is the user's test_2 idea (fact_pattern - query_pattern), measured at a position that identifies the subject.
    """
    ds = []
    for p in fx["train"]:
        with_ctx = f"{fx['statement']} {p}"
        _, o0 = capture(model, tok, p, layer)
        _, o1 = capture(model, tok, with_ctx, layer)
        if where == "subject":
            ds.append(o1[subject_index(tok, with_ctx, fx["subject"])] - o0[subject_index(tok, p, fx["subject"])])
        else:
            ds.append(o1[-1] - o0[-1])
    return torch.stack(ds).mean(0)


# Same-subject prompts the edit must leave alone. Deliberately different from each relation's `other` test prompt.
NEGATIVE_TEMPLATES = [
    "{s} is a", "{s} is known for", "The website of {s} is", "{s} was established in", "Employees of {s} work on",
    "The main product of {s} is", "{s} reported revenue of",
]
NEGATIVE_CHAT = ["What is {s} known for?", "What does {s} sell?"]


def _negative_batch(model, tok, fx):
    texts = [t.format(s=fx["subject"]) for t in NEGATIVE_TEMPLATES] + [chat(tok, q.format(s=fx["subject"])) for q in NEGATIVE_CHAT]
    tok.padding_side = "right"
    b = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    last = b.attention_mask.sum(1) - 1
    subj = torch.tensor([subject_index(tok, t, fx["subject"]) for t in texts], device=model.device)
    rows = torch.arange(len(texts), device=model.device)
    with torch.no_grad():
        base = model(**b, use_cache=False).logits[rows, last].float().log_softmax(-1)
    return b, last, subj, rows, base


def solve_delta(model, tok, fx, layer, steps=40, lr=0.25, l2=0.002, kl_weight=0.0):
    """ROME-style: find one output vector for the subject token. Only this vector is optimised, no weight is trained.

    kl_weight > 0 adds ROME's essence-drift idea: other questions about the same subject should keep their base distribution.
    """
    neg = _negative_batch(model, tok, fx) if kl_weight > 0 else None
    prompts = training_prompts(tok, fx)
    tgt = tok(" " + fx["target"], add_special_tokens=False).input_ids
    tgt_chat = tok(fx["target"], add_special_tokens=False).input_ids
    seqs, starts, subj = [], [], []
    for p in prompts:
        is_chat = p.startswith("<|im_start|>")
        pid = tok(p, add_special_tokens=False).input_ids
        t = tgt_chat if is_chat else tgt
        seqs.append(pid + t)
        starts.append(len(pid))
        subj.append(subject_index(tok, p, fx["subject"]))
    maxlen = max(map(len, seqs))
    pad = tok.pad_token_id
    ids = torch.tensor([s + [pad] * (maxlen - len(s)) for s in seqs], device=model.device)
    mask = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs], device=model.device)
    labels = torch.full_like(ids, -100)
    for r, (s, st) in enumerate(zip(seqs, starts)):
        labels[r, st:len(s)] = torch.tensor(s[st:], device=model.device)
    dim = down_proj(model, layer).weight.shape[0]
    delta = torch.zeros(dim, device=model.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    rows = torch.arange(len(seqs), device=model.device)
    idx = torch.tensor(subj, device=model.device)

    state = {"rows": rows, "idx": idx}

    def hook(_m, _i, out):
        out = out.clone()
        out[state["rows"], state["idx"]] = out[state["rows"], state["idx"]] + delta
        return out

    h = down_proj(model, layer).register_forward_hook(hook)
    last = None
    with torch.enable_grad():
        for _ in range(steps):
            state.update(rows=rows, idx=idx)
            lg = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
            total = loss + l2 * delta.pow(2).sum()
            if neg is not None:
                b, nlast, nsubj, nrows, nbase = neg
                state.update(rows=nrows, idx=nsubj)
                nlp = model(**b, use_cache=False).logits[nrows, nlast].float().log_softmax(-1)
                drift = (nbase.exp() * (nbase - nlp)).sum(-1).mean()
                total = total + kl_weight * drift
            opt.zero_grad()
            total.backward()
            opt.step()
            last = float(loss.detach())
    h.remove()
    return delta.detach(), last


def hebbian_user_method(model, tok, text, scale=0.01, top=5):
    """test_1's idea with its bugs removed so it can actually run on real weight matrices.

    Hook the real down_proj layers, pick the 5 most active ones ('importance'), and strengthen active connections in proportion
    to activation. test_1's literal `weight_mask[idx] = pattern` cannot broadcast into a 1024x3584 matrix, so the update is the
    closest coherent reading: output pattern x input pattern, normalised.
    """
    n = len(blocks(model))
    infos = []
    for l in range(n):
        inp, out = capture(model, tok, text, l)
        infos.append((float(out.abs().mean()), l, inp.mean(0), out.mean(0)))
    infos.sort(key=lambda x: x[0], reverse=True)
    deltas = {}
    for imp, l, i, o in infos[:top]:
        deltas[l] = scale * imp * torch.outer(o, i) / (i @ i)
    return deltas


def now():
    return time.time()
