"""The numerical core of GRAFT: reading a layer's MLP input, calibration on ordinary text, the LDA fit, readings, greedy replies.

Everything here runs forward passes only. Batching follows the research runs exactly (right padding, batches of 32, sequences sorted by
length where noted), because bfloat16 results depend on the batch a sequence is read in, and the cached tables were built this way.
"""
from __future__ import annotations

import torch

from ..model_access import Prompt, batch_ids, encode, model_device, pad_id

RIDGE = 0.1                     # LDA shrinkage, as a share of the mean diagonal of the scatter matrix
CALIBRATION_FIT = range(0, 240)  # paragraphs of the calibration text used for the mean and covariance
CALIBRATION_HELD = range(240, 280)  # held-out paragraphs, kept for checking the scale
CALIBRATION_MAX_TOKENS = 128
OPENERS = set("*\"'([")


def blocks(model):
    """The model's decoder layers."""
    inner = model.model
    return inner.layers if hasattr(inner, "layers") else inner.language_model.layers


def mlp_of(model, layer: int):
    return blocks(model)[layer].mlp


class _Stop(Exception):
    pass


@torch.no_grad()
def layer_reads(model, tokenizer, seqs: list[list[int]], picks: list[list[int]], layer: int, batch: int = 32) -> list[torch.Tensor]:
    """Layer `layer`'s MLP input at the picked positions of each token sequence, as one (len(pick), d) tensor per sequence. Right padding,
    sequences sorted by length; the forward pass stops at the layer."""
    seen, out = {}, [None] * len(seqs)
    device = model_device(model)

    def grab(_m, args):
        seen["x"] = args[0]
        raise _Stop

    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    h = mlp_of(model, layer).register_forward_pre_hook(grab)
    try:
        for start in range(0, len(order), batch):
            chunk = order[start:start + batch]
            ids, mask = batch_ids(tokenizer, [seqs[i] for i in chunk], device)
            try:
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            except _Stop:
                pass
            for row, i in enumerate(chunk):
                out[i] = seen["x"][row, picks[i]].float()
    finally:
        h.remove()
    return out


def last_reads(model, tokenizer, prompts: list[Prompt], layer: int) -> torch.Tensor:
    """Layer `layer`'s MLP input at each prompt's last token, stacked."""
    seqs = [encode(tokenizer, p) for p in prompts]
    if not seqs:
        return torch.zeros(0, model.config.get_text_config().hidden_size, device=model_device(model))
    return torch.stack([x[0] for x in layer_reads(model, tokenizer, seqs, [[len(s) - 1] for s in seqs], layer)])


@torch.no_grad()
def mlp_inputs(model, tokenizer, sequences: list[list[int]], layer: int) -> list[torch.Tensor]:
    """Layer `layer`'s MLP input at every position of each sequence, in the given order, batches of 32 (fact block v2's reads)."""
    seen, out = {}, []
    device = model_device(model)
    h = mlp_of(model, layer).register_forward_pre_hook(lambda m, a: seen.__setitem__("x", a[0].float()))
    try:
        for start in range(0, len(sequences), 32):
            chunk = sequences[start:start + 32]
            ids, mask = batch_ids(tokenizer, chunk, device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            out += [seen["x"][i, :len(s)].clone() for i, s in enumerate(chunk)]
    finally:
        h.remove()
    return out


def last_inputs(model, tokenizer, prompts: list[Prompt], layer: int) -> torch.Tensor:
    """Each prompt's last-token MLP input, read in the given order (fact block v2's batching)."""
    out = []
    for start in range(0, len(prompts), 256):
        out += [x[-1] for x in mlp_inputs(model, tokenizer, [encode(tokenizer, p) for p in prompts[start:start + 256]], layer)]
    return torch.stack(out)


def stop_ids(model, tokenizer) -> set[int]:
    ids = {pad_id(tokenizer)}
    for e in (model.generation_config.eos_token_id, tokenizer.eos_token_id):
        ids |= set(e) if isinstance(e, (list, tuple)) else ({e} if e is not None else set())
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return ids


@torch.no_grad()
def greedy_ids(model, tokenizer, prompts: list[Prompt], n_new: int, batch: int = 32) -> list[tuple[list[int], list[int]]]:
    """(prompt ids, greedy reply ids up to the first stop token) per prompt, left padding as the bench generates."""
    stops, out = stop_ids(model, tokenizer), []
    device = model_device(model)
    for start in range(0, len(prompts), batch):
        chunk = [encode(tokenizer, p) for p in prompts[start:start + batch]]
        ids, mask = batch_ids(tokenizer, chunk, device, side="left")
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=n_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=pad_id(tokenizer))
        for s, row in zip(chunk, gen[:, ids.shape[1]:].tolist()):
            cut = next((k for k, t in enumerate(row) if t in stops), len(row))
            out.append((s, row[:cut]))
    return out



class ReadCache:
    """Readings and short greedy replies by prompt, kept across the facts of one build so that each is computed once: most of a fact's
    prompts name invented subjects, and every fact of the same relation asks the same ones.

    Opt-in: a value computed in another batch can differ in its last bits from a fresh one, so v1's exact reproduction builds without it.
    Prompts that name `transient` (the fact's own subject) are computed but not kept, which bounds the memory."""

    def __init__(self):
        self.reads: dict = {}
        self.replies: dict = {}
        self.hits = self.misses = 0

    @staticmethod
    def _key(p: Prompt) -> tuple[str, bool]:
        return (p.text, p.rendered)

    @staticmethod
    def _keep(key, transient: str | None) -> bool:
        return not (transient and transient.lower() in key[0].lower())

    def last_reads(self, model, tokenizer, prompts: list[Prompt], layer: int, transient: str | None = None) -> torch.Tensor:
        keys = [self._key(p) for p in prompts]
        if not keys:
            return last_reads(model, tokenizer, [], layer)
        need = list(dict.fromkeys(k for k in keys if k not in self.reads))
        fresh = {}
        if need:
            x = last_reads(model, tokenizer, [Prompt(text, rendered) for text, rendered in need], layer).float().cpu()
            for k, v in zip(need, x):
                fresh[k] = v
                if self._keep(k, transient):
                    self.reads[k] = v
        self.hits += len(keys) - len(need)
        self.misses += len(need)
        return torch.stack([fresh[k] if k in fresh else self.reads[k] for k in keys]).to(model_device(model))

    def reply_reads(self, model, tokenizer, prompts: list[Prompt], n_new: int, layer: int, transient: str | None = None) -> list[torch.Tensor]:
        """For each prompt with a non-empty greedy reply of up to n_new tokens: the layer's MLP input at every reply position."""
        keys = [self._key(p) for p in prompts]
        need = list(dict.fromkeys(k for k in keys if k not in self.replies))
        fresh = {}
        if need:
            gen = greedy_ids(model, tokenizer, [Prompt(text, rendered) for text, rendered in need], n_new)
            seqs = [p + r for p, r in gen if r]
            reads = iter(layer_reads(model, tokenizer, seqs, [list(range(len(p), len(p) + len(r))) for p, r in gen if r], layer)) if seqs else iter(())
            for k, (_, r) in zip(need, gen):
                fresh[k] = next(reads).cpu() if r else None
                if self._keep(k, transient):
                    self.replies[k] = fresh[k]
        device = model_device(model)
        got = [fresh[k] if k in fresh else self.replies[k] for k in keys]
        return [x.to(device) for x in got if x is not None]


_TEXTS: dict[int, list[str]] = {}


def token_texts(tokenizer) -> list[str]:
    """The text of every token id, decoded once per tokenizer."""
    if id(tokenizer) not in _TEXTS:
        _TEXTS[id(tokenizer)] = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
    return _TEXTS[id(tokenizer)]


def bare_after(text: str) -> bool:
    """Whether an answer coming right after this token text is spelled without a leading space."""
    return not text or text[-1].isspace() or text[-1] in OPENERS


@torch.no_grad()
def calibration_stats(model, tokenizer, paragraphs: list[str], layer: int) -> dict:
    """Mean and covariance of layer `layer`'s MLP input over the fitting paragraphs (first token of each skipped), and the held-out tokens."""
    if len(paragraphs) < CALIBRATION_HELD.stop:
        raise ValueError(f"calibration needs at least {CALIBRATION_HELD.stop} paragraphs of ordinary text, got {len(paragraphs)}")
    cur = {}
    device = model_device(model)
    h = mlp_of(model, layer).register_forward_pre_hook(lambda m, a: cur.__setitem__("x", a[0][0, 1:].double()))
    sums, outer, held, n = 0, 0, [], 0
    try:
        for i in list(CALIBRATION_FIT) + list(CALIBRATION_HELD):
            ids = tokenizer(paragraphs[i], return_tensors="pt", truncation=True, max_length=CALIBRATION_MAX_TOKENS).input_ids.to(device)
            model(ids)
            x = cur["x"]
            if i in CALIBRATION_FIT:
                sums = sums + x.sum(0)
                outer = outer + x.T @ x
                n += x.shape[0]
            else:
                held.append(x.float())
    finally:
        h.remove()
    mu = sums / n
    return dict(mu=mu.float(), cov=(outer / n - torch.outer(mu, mu)).float(), held=torch.cat(held))


def constant_direction(stats: dict) -> tuple[torch.Tensor, float, float]:
    """c = (Σ + 1e-3·tr(Σ)/d·I)⁻¹ μ, its mean projection c̄ = μ·c, and its relative spread on held-out tokens."""
    mu, cov, held = stats["mu"].double(), stats["cov"].double(), stats["held"].double()
    c = torch.linalg.solve(cov + 1e-3 * cov.diagonal().mean() * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device), mu)
    s = held @ c
    return c.float(), float(mu @ c), float(s.std() / s.mean())


def lda(pos: torch.Tensor, negs: list[torch.Tensor], cov: torch.Tensor, ridge: float = RIDGE) -> torch.Tensor:
    """LDA with shrinkage: w = (S + ridge·tr(S)/d·I)⁻¹ (m₊ − mean of the negative groups' means), S = Σ + within-group scatter."""
    m_neg = torch.stack([g.mean(0) for g in negs]).mean(0)
    within = torch.cat([pos - pos.mean(0)] + [g - g.mean(0) for g in negs])
    Sm = cov + within.T @ within / within.shape[0]
    return torch.linalg.solve(Sm + ridge * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device=Sm.device), pos.mean(0) - m_neg)


def ratio(x: torch.Tensor, w: torch.Tensor, c: torch.Tensor, cbar: float) -> torch.Tensor:
    """A reading: w·x / u(x), with u(x) = c·x / c̄ close to 1 on ordinary text."""
    x = x.float()
    return (x @ w) / (x @ c / cbar)


def keyed_spots(tokenizer, examples):
    """answer_chain's grouping, with the keys: sequences, sorted (step, token) keys, and each key's (sequence, position) spots."""
    grouped, sequences = {}, []
    for w, (prompt, continuation) in enumerate(examples):
        p_ids, a_ids = encode(tokenizer, prompt), tokenizer(continuation, add_special_tokens=False).input_ids
        sequences.append(p_ids + a_ids)
        for step, tok in enumerate(a_ids):
            grouped.setdefault((step, tok), []).append((w, len(p_ids) + step - 1))
    keys = sorted(grouped)
    return sequences, keys, [grouped[k] for k in keys]
