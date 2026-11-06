"""Weight-space programming with this package's tools: a big model operates on a small one, like a surgeon.

The way of working of *Sparks of Weight Space Programming*, with the rules it ran under, made into one workflow:

    1 diagnose   see how the model fails before changing anything
    2 map        which layer does what: where answers appear, which layer writes a behaviour, what a layer already predicts
    3 localize   find the direction or the key, before writing anything
    4 operate    a direct edit computed from what was observed: remove a behaviour's direction, or write a fact
    5 check      the same test as the baseline, against the unedited model: several strengths, each part alone, a random control
    6 confirm    on items never used to choose the edit; keep it only if it is better there too

Rules, enforced here rather than requested:
    - no backward pass, autograd or optimiser: every tool runs under `torch.no_grad`
    - a budget of forward passes; each tool says what it costs
    - every action goes into a hash-chained log, and every kept edit can be undone bit for bit

Any editor can drive it: this session, a script, or Claude through `weightspace.claude_editor`. Tools take and return plain JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from .direct import answer_chain, answer_prefix_reads, layer_lens, matched_direction, removal_update
from .facts import Fact, frozen_for_solving, input_covariance, metric_from_covariance, null_reads
from .gates import KNOWN_FACTS, GateConfig, RegressionBaseline, generate_batch
from .grid import monitor
from .isolation import paired_test, random_control
from .model_access import Prompt, decoder_layers, encode, find_parameter, model_device, render_chat, tensor_fingerprint, writer_module, writer_param_name
from .operators import write_associations
from .patch import AppliedPatch, LowRankUpdate, WeightPatch
from .problems import grade, summarise, think
from .session import EditSession

PROTOCOL = """You are editing the weights of a small language model directly, the way a surgeon operates: map first, change only what the evidence points to, and check function throughout. You cannot train it. There are no gradients: you observe with forward passes and compute edits from what you observe.

How to work:
1. Diagnose first. Look at how the model behaves on the task before changing anything, and name the failure you will go after.
2. Map. Use the layer tools to find where the relevant information lives: which layer first holds known answers, which layer's MLP writes a behaviour, what a layer already predicts.
3. Localize. A behaviour needs matched pairs: the same problem and start, one continuation good and one bad, so the difference is the behaviour and not the topic. Write your own pairs for finding a direction, and different ones for testing it.
4. Operate. Propose the smallest edit the evidence supports: remove a direction from the middle-layer MLPs that write it most, or write a fact at the layer where answers first appear.
5. Check. Test each proposal at several strengths against the unedited model on the same items. Isolate it: each layer alone, and a random edit of the same size. An effect the random edit also has came from the size of the change, not its direction.
6. Confirm and decide. Confirm on items you never used to choose the edit. Keep an edit only if it is better there and nothing it must not damage got worse. Undo anything that turns out worse.

Rules: every tool that runs the model costs forward passes from a fixed budget, so plan them. Report numbers honestly, including edits that failed. Small differences on a few items are noise.
When you finish, say what you changed, the evidence for it, and what remains uncertain."""


@dataclass
class Tool:
    name: str
    description: str
    properties: dict
    required: list[str]
    cost: Callable[[dict], int]
    cost_note: str
    run: Callable[..., dict]

    def schema(self) -> dict:
        return dict(
            name=self.name, description=f"{self.description} Cost: {self.cost_note}.",
            input_schema=dict(type="object", properties=self.properties, required=self.required, additionalProperties=False),
        )


@dataclass
class Proposal:
    name: str
    kind: str  # "behaviour" or "fact"
    updates: list[LowRankUpdate]
    detail: dict
    fact: Fact | None = None
    pairs: list[dict] | None = None


_TYPES = {"string": str, "integer": int, "number": (int, float), "array": list, "object": dict, "boolean": bool}


class Surgery:
    """
    >>> surgery = Surgery(model, tokenizer, calibration_texts=paragraphs, budget=50, log_path="surgery.jsonl")
    >>> surgery.call("layer_map", {})
    >>> surgery.call("find_direction", {"name": "loops", "pairs": [...]})
    >>> surgery.call("propose_removal", {"name": "fix", "direction": "loops", "layers": [15, 19], "strength": 8, "neurons": 64, "test_pairs": [...]})
    >>> surgery.call("test", {"name": "fix", "strengths": [0.5, 1.0]})
    >>> surgery.call("keep", {"name": "fix", "strength": 1.0})
    """

    def __init__(
        self, model: nn.Module, tokenizer, *, calibration_texts: list[str] | None = None, budget: int = 50, log_path: str | None = None, gates: GateConfig | None = None,
        model_name: str | None = None, echo: bool = False,
    ):
        """log_path: the hash-chained JSONL log; a live status file is written beside it for `weightspace watch`.
        echo: also print one readable line per finished step to stderr, which is safe even when stdout carries MCP."""
        self.model, self.tokenizer = model, tokenizer
        self.calibration_texts = calibration_texts
        self.budget, self.passes_used = budget, 0
        self.log_path = log_path
        self.echo = echo
        self.gates = gates or GateConfig()
        self.directions: dict[str, dict[int, torch.Tensor]] = {}
        self.proposals: dict[str, Proposal] = {}
        self.kept: list[tuple[str, AppliedPatch]] = []
        self.log: list[dict] = []
        self._metrics: dict[int, torch.Tensor] = {}
        self._baseline: RegressionBaseline | None = None
        self._lock = threading.RLock()
        self.tools = {t.name: t for t in self._tools()}
        name = model_name or getattr(getattr(model, "config", None), "_name_or_path", "") or type(model).__name__
        self._live = dict(
            started_at=time.time(), model=os.path.basename(os.path.normpath(str(name))), device=str(model_device(model)), budget=budget, passes_used=0,
            current=None, kept=[], pid=os.getpid(), log=os.path.abspath(log_path) if log_path else None,
        )
        self._write_status()

    # ------------------------------------------------------------------ live status, for `weightspace watch`
    def _write_status(self) -> None:
        if self.log_path:
            from .live import status_path_for, write_status

            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            write_status(status_path_for(self.log_path), self._live)

    def _tick(self, detail: str, fraction: float) -> None:
        """Progress inside a long tool: what it is doing now, and how far along it is, from 0 to 1."""
        if self._live.get("current") is not None:
            self._live["current"].update(detail=detail, fraction=max(0.0, min(1.0, fraction)))
            self._write_status()

    def _note(self, text: str) -> None:
        """A result inside a long tool, such as one strength's verdict, shown under the progress bar as soon as it is known."""
        if self._live.get("current") is not None:
            notes = self._live["current"].setdefault("notes", [])
            notes.append(text)
            del notes[:-12]
            self._write_status()
        if self.echo:
            print(f"[weightspace {time.strftime('%H:%M:%S')}]      {text}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ the interface an editor sees
    def tool_definitions(self) -> list[dict]:
        """Every tool's JSON schema, in the shape the Claude API takes. Every schema forbids unknown arguments."""
        return [t.schema() for t in self.tools.values()]

    def validate(self, name: str, args: Any) -> str | None:
        """Why a call cannot run, or None: the tool must exist, the arguments must be an object with the required keys, no unknown ones, right types."""
        if name not in self.tools:
            return f"Unknown tool {name!r}. Tools: {', '.join(self.tools)}."
        if not isinstance(args, dict):
            return "Arguments must be a JSON object."
        tool = self.tools[name]
        missing = [k for k in tool.required if k not in args]
        unknown = [k for k in args if k not in tool.properties]
        if missing or unknown:
            return f"{name}: missing {missing or 'nothing'}, unknown {unknown or 'nothing'}."
        for key, value in args.items():
            expected = tool.properties[key].get("type")
            if expected in _TYPES and (not isinstance(value, _TYPES[expected]) or (expected in ("integer", "number") and isinstance(value, bool))):
                return f"{name}: {key} must be a {expected}."
        return None

    def call(self, name: str, args: dict | None = None) -> dict:
        """Run one tool. Returns a dict safe to show the editor, with "error" set if the call could not run. Failed calls cost nothing.

        Calls run one at a time, even when an editor sends several at once: a test applies its edit for a moment, so a call running
        beside it would measure the edited model, and a keep would record the wrong weights to undo to.
        """
        with self._lock:
            return self._call(name, args)

    def _call(self, name: str, args: dict | None) -> dict:
        args = {} if args is None else args
        problem = self.validate(name, args)
        if problem:
            return self._record(name, args, dict(error=problem), 0)
        tool = self.tools[name]
        try:
            cost = tool.cost(args)
        except (KeyError, TypeError) as error:
            return self._record(name, args, dict(error=f"{name}: {error}"), 0)
        if self.passes_used + cost > self.budget:
            return self._record(name, args, dict(error=f"Not enough forward passes left: this costs {cost}, and {self.budget - self.passes_used} remain."), 0)
        started = time.perf_counter()
        self._live["current"] = dict(tool=name, detail="starting", fraction=0.0, started_at=time.time(), notes=[])
        self._write_status()
        try:
            with frozen_for_solving(self.model), torch.no_grad():
                result = tool.run(**args)
        except (ValueError, KeyError, IndexError) as error:
            return self._record(name, args, dict(error=f"{type(error).__name__}: {error}"), 0)
        except Exception as error:  # such as running out of GPU memory: the editor can try something smaller
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._record(name, args, dict(error=f"{type(error).__name__} while running {name}: {str(error)[:300]}"), 0)
        self.passes_used += cost
        result["seconds"] = round(time.perf_counter() - started, 1)
        return self._record(name, args, result, cost)

    # ------------------------------------------------------------------ the log
    def _record(self, name: str, args: dict, result: dict, cost: int) -> dict:
        from .live import summarize

        previous = self.log[-1]["hash"] if self.log else "start"
        summary = summarize(name, args, result)
        # Through JSON first, so the hash covers exactly what the file holds (layer numbers become string keys) and the file alone can be checked.
        entry = json.loads(json.dumps(dict(
            step=len(self.log) + 1, time=time.strftime("%Y-%m-%dT%H:%M:%S"), tool=name, args=_jsonable(args), result=_jsonable(result), summary=summary,
            passes=cost, passes_used=self.passes_used, previous=previous,
        )))
        entry["hash"] = _digest(entry)
        self.log.append(entry)
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._live.update(passes_used=self.passes_used, current=None, kept=[n for n, _ in self.kept])
        self._write_status()
        if self.echo:
            print(f"[weightspace {time.strftime('%H:%M:%S')}] #{entry['step']} {summary} ({result.get('seconds', 0)}s, {self.budget - self.passes_used} passes left)", file=sys.stderr, flush=True)
        return dict(result, passes_left=self.budget - self.passes_used)

    def log_is_intact(self) -> bool:
        """True if no log entry was changed, removed or reordered after it was written. `log_file_is_intact` checks the file."""
        return chain_is_intact(self.log)

    # ------------------------------------------------------------------ internals
    def _baseline_model(self) -> RegressionBaseline:
        if self._baseline is None:
            self._baseline = RegressionBaseline(self.model, self.tokenizer)
        return self._baseline

    def _metric(self, layer: int) -> torch.Tensor | None:
        if not self.calibration_texts:
            return None
        if layer not in self._metrics:
            self._metrics[layer] = metric_from_covariance(input_covariance(self.model, self.tokenizer, self.calibration_texts, [layer])[layer], 3.0)
        return self._metrics[layer]

    def _proposal(self, name: str) -> Proposal:
        if name not in self.proposals:
            raise KeyError(f"No proposal named {name!r}. Proposals so far: {list(self.proposals)}.")
        return self.proposals[name]

    def _pair_scores(self, pairs: list[dict]) -> dict:
        device = model_device(self.model)
        good, bad = [], []
        for pair in pairs:
            head_len = len(encode(self.tokenizer, Prompt(pair["start"], rendered=True)))
            for text, out in ((pair["start"] + pair["good"], good), (pair["start"] + pair["bad"], bad)):
                ids = torch.tensor([encode(self.tokenizer, Prompt(text, rendered=True))], device=device)
                logits = self.model(input_ids=ids, use_cache=False).logits[0].float().log_softmax(-1)
                out.append(float(logits[head_len - 1 : -1].gather(1, ids[0, head_len:, None]).mean()))
        base = self._baseline_model()
        return dict(good=sum(good) / len(good), bad=sum(bad) / len(bad), known=base.known_score(), drift=base.drift()[0])

    def _session_for(self, proposal: Proposal) -> EditSession:
        if proposal.kind == "behaviour":
            return EditSession(
                self.model, lambda: self._pair_scores(proposal.pairs), goals=dict(bad=-1), guards=dict(good=+1, known=+1, drift=-1),
                tolerance=dict(bad=0.01, good=0.02, known=0.05, drift=0.05),
            )
        return EditSession(
            self.model, lambda: _fact_scores(self.model, self.tokenizer, proposal.fact, self._baseline_model(), self.gates), goals=dict(recall=+1),
            guards=dict(same_subject=-1, other_subject=-1, garbled=-1, known=+1, drift=-1),
            tolerance=dict(recall=0.01, same_subject=0.25, other_subject=0.25, garbled=0.34, known=0.05, drift=0.05),
        )

    def _parts(self, args: dict) -> int:
        return len(self._proposal(args["name"]).updates)

    # ------------------------------------------------------------------ tools
    def _tools(self) -> list[Tool]:
        strings = dict(type="array", items=dict(type="string"))
        integers = dict(type="array", items=dict(type="integer"))
        numbers = dict(type="array", items=dict(type="number"))
        fact = dict(
            type="object",
            description="subject and target; prompts that end right before the answer; optional chat questions; separate new_prompts and new_questions to test recall; other_prompts and other_questions about the same subject that the answer must not leak into.",
            properties=dict(subject=dict(type="string"), target=dict(type="string"), prompts=strings, questions=strings, new_prompts=strings, new_questions=strings, other_prompts=strings, other_questions=strings),
            required=["subject", "target", "prompts"], additionalProperties=False,
        )
        pairs = dict(
            type="array", description="Matched pairs: the same start (chat template text included), then a good continuation and a bad one.",
            items=dict(type="object", properties=dict(start=dict(type="string"), good=dict(type="string"), bad=dict(type="string")), required=["start", "good", "bad"], additionalProperties=False),
        )
        problems = dict(
            type="array", description="Problems never used to choose the edit, each with its numeric answer.",
            items=dict(type="object", properties=dict(question=dict(type="string"), answer=dict(type="string")), required=["question", "answer"], additionalProperties=False),
        )
        free, one = (lambda a: 0), (lambda a: 1)
        return [
            Tool("status", "What has been done: passes used and left, directions found, proposals, kept edits.", {}, [], free, "none", self._status),
            Tool("generate", "Diagnose: the model's own answers. Plain prompts are continued as text; questions use its chat format. With thinking, questions are answered with the model's thinking on, sampled, and each reply gives the start and end of the thinking, its length in tokens, and whether it ends in a loop.",
                 dict(prompts=strings, questions=strings, max_tokens=dict(type="integer"), thinking=dict(type="boolean")), [], one, "1 pass", self._generate),
            Tool("layer_map", "Map: at each layer, how often a well-known answer is already the top guess. Facts belong where this first rises.", {}, [], one, "1 pass", self._layer_map),
            Tool("layer_lens", "Map: what each chosen layer's residual stream already predicts for the next token of one prompt.",
                 dict(prompt=dict(type="string"), layers=integers), ["prompt"], one, "1 pass", self._layer_lens),
            Tool("find_direction", "Localize a behaviour: the direction from good to bad continuations of matched pairs at every layer, and how much each layer's MLP writes it.",
                 dict(name=dict(type="string"), pairs=pairs), ["name", "pairs"], one, "1 pass", self._find_direction),
            Tool("propose_removal", "Operate on a behaviour: stop chosen layers' MLPs writing a found direction. strength 1 removes it, more reverses it; neurons limits the edit to those that write it most. test_pairs are the matched pairs used to check it, which should differ from those that found it.",
                 dict(name=dict(type="string"), direction=dict(type="string"), layers=integers, strength=dict(type="number"), neurons=dict(type="integer"), test_pairs=pairs),
                 ["name", "direction", "layers", "strength", "test_pairs"], free, "none", self._propose_removal),
            Tool("propose_fact", "Operate on knowledge: write a fact at one layer with no backpropagation. Each answer token is pushed where the token before it is read.",
                 dict(name=dict(type="string"), fact=fact, layer=dict(type="integer"), strength=dict(type="number")), ["name", "fact", "layer"], one, "1 pass", self._propose_fact),
            Tool("test", "Check a proposal at several strengths against the unedited model on the same items. Each verdict is better, worse, mixed or same.",
                 dict(name=dict(type="string"), strengths=numbers), ["name", "strengths"], lambda a: 1 + len(a["strengths"]), "1 pass for the baseline plus 1 per strength", self._test),
            Tool("isolate", "Check what caused an effect: each part of a proposal alone, and a random edit of the same size.",
                 dict(name=dict(type="string")), ["name"], lambda a: 2 + (self._parts(a) if self._parts(a) > 1 else 0), "1 pass for the baseline, 1 per part, 1 for the random edit", self._isolate),
            Tool("confirm", "Confirm a proposal on problems never used to choose it: paired generation with thinking on, unedited against edited, the same seeds.",
                 dict(name=dict(type="string"), strength=dict(type="number"), problems=problems, max_tokens=dict(type="integer")), ["name", "problems"], lambda a: 2, "2 passes", self._confirm),
            Tool("keep", "Write a proposal into the weights for good, at one strength. Later tests compare against the model with it.",
                 dict(name=dict(type="string"), strength=dict(type="number")), ["name"], free, "none", self._keep),
            Tool("undo", "Take back the most recently kept edit, bit for bit.", {}, [], free, "none", self._undo),
            Tool("export_patch", "Save every kept edit as one patch file pair (<path>.safetensors and <path>.json), hash-checked against the weights it was made on, so it can be applied to the same base model anywhere. note says what the edits were checked on, including whether thinking was on.",
                 dict(path=dict(type="string"), note=dict(type="string")), ["path"], free, "none", self._export_patch),
        ]

    def _status(self) -> dict:
        return dict(
            passes_used=self.passes_used, passes_left=self.budget - self.passes_used, directions=list(self.directions),
            proposals={n: dict(kind=p.kind, **p.detail) for n, p in self.proposals.items()}, kept=[n for n, _ in self.kept],
        )

    def _generate(self, prompts: list[str] | None = None, questions: list[str] | None = None, max_tokens: int = 24, thinking: bool = False) -> dict:
        if thinking:
            if prompts:
                raise ValueError("Thinking applies to chat questions only; pass them as questions.")
            if not questions:
                raise ValueError("Give at least one question.")
            from .grid import final_answer
            from .problems import THINK_END, tail_repeats, think

            limit = min(max_tokens, 4096)
            generated = think(
                self.model, self.tokenizer, questions, limit, instruction="",
                on_batch=lambda first, end, total, done: self._tick(
                    f"thinking on: questions {first + 1}-{end} of {total}, {round(done * limit)} of up to {limit} tokens", (first + (end - first) * done) / total,
                ),
            )
            answers = []
            for question, tokens in zip(questions, generated):
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                final, thought, repeats = final_answer(text), text.split(THINK_END, 1)[0].strip(), tail_repeats(tokens)
                # A loop shows at the end of the thinking, not the start, so both ends are returned.
                answers.append(dict(
                    question=question, finished=final is not None, final_answer=(final or "").strip()[:300], tokens=len(tokens),
                    looping=repeats >= 0.5, repeats=round(repeats, 2), thinking_start=thought[:300], thinking_end=thought[-400:],
                ))
            for a in answers[:6]:
                ending = f"-> {a['final_answer'][:60]!r}" if a["finished"] else "-> never finished thinking"
                self._note(f"{a['question'][:50]!r} {ending} ({a['tokens']} tokens{', looping' if a['looping'] else ''})")
            return dict(thinking=True, answers=answers)
        asked = [("prompt", p, Prompt(p)) for p in prompts or []] + [("question", q, render_chat(self.tokenizer, q)) for q in questions or []]
        if not asked:
            raise ValueError("Give at least one prompt or question.")
        self._tick(f"generating {len(asked)} answer(s), thinking off, up to {min(max_tokens, 256)} tokens each", 0.1)
        texts = generate_batch(self.model, self.tokenizer, [a[2] for a in asked], min(max_tokens, 256))
        for (_, text, _), out in list(zip(asked, texts))[:6]:
            self._note(f"{text[:50]!r} -> {out.strip()[:60]!r}")
        return dict(thinking=False, answers=[{kind: text, "answer": out.strip()[:300]} for (kind, text, _), out in zip(asked, texts)])

    def _layer_map(self) -> dict:
        layers = list(range(len(decoder_layers(self.model))))
        hits = {l: 0 for l in layers}
        for i, (prompt, answer) in enumerate(KNOWN_FACTS):
            self._tick(f"reading all {len(layers)} layers on known fact {i + 1} of {len(KNOWN_FACTS)}: {prompt[:40]!r}", i / len(KNOWN_FACTS))
            lens = layer_lens(self.model, self.tokenizer, Prompt(prompt), layers, k=1)
            for l in layers:
                top = lens[l][0][0]
                hits[l] += answer.lower() in top.lower() and len(top.strip()) > 1
        shares = {l: round(hits[l] / len(KNOWN_FACTS), 2) for l in layers}
        first = next((l for l in layers if shares[l] >= 0.15), None)
        self._note(f"known answers first appear at layer {first}; top-guess share by layer: " + " ".join(f"{l}:{100 * shares[l]:.0f}%" for l in layers[::max(1, len(layers) // 8)]))
        return dict(known_answer_on_top_by_layer=shares, first_layer_with_answers=first, layers=len(layers))

    def _layer_lens(self, prompt: str, layers: list[int] | None = None) -> dict:
        chosen = layers or list(range(len(decoder_layers(self.model))))
        self._tick(f"reading what {len(chosen)} layer(s) predict for {prompt[:50]!r}", 0.1)
        return dict(top_tokens={l: [(t, round(p, 2)) for t, p in v] for l, v in layer_lens(self.model, self.tokenizer, Prompt(prompt), chosen, k=3).items()})

    def _find_direction(self, name: str, pairs: list[dict]) -> dict:
        layers = list(range(len(decoder_layers(self.model))))
        good = [p["start"] + p["good"] for p in pairs]
        bad = [p["start"] + p["bad"] for p in pairs]
        self._tick(f"the good-to-bad direction from {len(pairs)} matched pairs, at all {len(layers)} layers", 0.02)
        directions = matched_direction(self.model, self.tokenizer, good, bad, layers)
        self.directions[name] = directions
        device = model_device(self.model)
        writes = {}
        for l in layers:
            self._tick(f"how much layer {l}'s MLP writes {name!r} ({l + 1} of {len(layers)})", 0.1 + 0.9 * l / len(layers))
            seen = {}
            handle = writer_module(self.model, l).register_forward_hook(lambda _m, _i, o: seen.__setitem__("o", o[0, -24:].float().mean(0)))
            try:
                differences = []
                for g, b in zip(good, bad):
                    self.model(input_ids=torch.tensor([encode(self.tokenizer, Prompt(b, rendered=True))], device=device), use_cache=False)
                    after_bad = seen["o"].clone()
                    self.model(input_ids=torch.tensor([encode(self.tokenizer, Prompt(g, rendered=True))], device=device), use_cache=False)
                    differences.append(after_bad - seen["o"])
            finally:
                handle.remove()
            writes[l] = round(float(torch.stack(differences).mean(0) @ directions[l].to(device)), 3)
        strongest = sorted(writes, key=lambda l: -writes[l])[:6]
        self._note("strongest writers: " + ", ".join(f"layer {l} ({writes[l]:+.2f})" for l in strongest))
        return dict(direction=name, mlp_writes_by_layer=writes, strongest_writers=strongest, note="The last two layers act on the output directly; middle-layer writers are the usual targets.")

    def _propose_removal(self, name: str, direction: str, layers: list[int], strength: float, test_pairs: list[dict], neurons: int | None = None) -> dict:
        if direction not in self.directions:
            raise KeyError(f"No direction named {direction!r}. Found so far: {list(self.directions)}.")
        updates = [removal_update(self.model, l, self.directions[direction][l], strength, neurons) for l in layers]
        changed = sum(int((u.left @ u.right).ne(0).any(0).sum()) for u in updates)
        self.proposals[name] = Proposal(name, "behaviour", updates, dict(direction=direction, layers=layers, strength=strength, neurons=neurons, columns_changed=changed), pairs=test_pairs)
        return dict(proposed=name, layers=layers, strength=strength, neurons=neurons, columns_changed=changed)

    def _propose_fact(self, name: str, fact: dict, layer: int, strength: float = 1.0) -> dict:
        built = Fact(
            subject=fact["subject"], target=fact["target"], prompts=list(fact["prompts"]), questions=list(fact.get("questions", [])),
            heldout_prompts=list(fact.get("new_prompts", [])), heldout_questions=list(fact.get("new_questions", [])),
            leak_prompts=list(fact.get("other_prompts", [])), leak_questions=list(fact.get("other_questions", [])),
        )
        self._tick(f"observing how far each token of {built.target!r} is from winning, at layer {layer}", 0.1)
        links = answer_chain(self.model, self.tokenizer, built.teaching_examples(self.tokenizer), layer)
        observed = [dict(token=self.tokenizer.decode([l.token]), logits_short=round(l.gap, 1), push=round(float(l.vector.norm()) * strength, 2)) for l in links]
        for o in observed:
            self._note(f"answer token {o['token']!r}: {o['logits_short']} logits short of winning, push {o['push']}")
        keys = torch.cat([l.keys for l in links], 1)
        deltas = torch.cat([(l.vector * strength)[:, None].expand(-1, l.keys.shape[1]) for l in links], 1)
        groups = [g for g, l in enumerate(links) for _ in range(l.keys.shape[1])]
        self._tick("reading the prompts the edit must leave alone", 0.6)
        parts = [null_reads(self.model, self.tokenizer, built, layer, "last", "all"), answer_prefix_reads(self.model, self.tokenizer, built.target, layer)]
        nulls = torch.cat([p for p in parts if p is not None], 1)
        self._tick("solving the write" + (" (first measuring ordinary text at this layer)" if self.calibration_texts and layer not in self._metrics else ""), 0.8)
        left, right = write_associations(keys, deltas, self._metric(layer), 0.3, nulls, groups)
        self.proposals[name] = Proposal(name, "fact", [LowRankUpdate(writer_param_name(self.model, layer), left.cpu(), right.cpu())], dict(subject=built.subject, target=built.target, layer=layer, strength=strength), fact=built)
        return dict(proposed=name, layer=layer, strength=strength, answer_tokens=observed, rank=int(left.shape[1]))

    def _verdict_note(self, proposal: Proposal, label: str, attempt) -> str:
        key = "bad" if proposal.kind == "behaviour" else "recall"
        guards = ", ".join(f"{k} {v:+.3f}" for k, v in attempt.changes.items() if k != key and isinstance(v, (int, float)))
        return f"{label}: {attempt.verdict} ({key} {attempt.changes.get(key, 0.0):+.3f}; {guards})"

    def _test(self, name: str, strengths: list[float]) -> dict:
        proposal = self._proposal(name)
        session = self._session_for(proposal)
        steps = len(strengths) + 1
        self._tick(f"measuring the unedited model (1 of {steps})", 0.0)
        base = session.measure_baseline()
        tried = []
        for i, strength in enumerate(strengths):
            self._tick(f"measuring {name!r} at strength {strength} ({i + 2} of {steps})", (i + 1) / steps)
            a = session.attempt(name, proposal.updates, doses=[strength])[0]
            self._note(self._verdict_note(proposal, f"strength {strength}", a))
            tried.append(dict(strength=a.dose, verdict=a.verdict, scores=_rounded(a.scores), changes=_rounded(a.changes)))
        return dict(proposal=name, unedited=_rounded(base), tried=tried)

    def _isolate(self, name: str) -> dict:
        proposal = self._proposal(name)
        session = self._session_for(proposal)
        alone = proposal.updates if len(proposal.updates) > 1 else []
        steps = 2 + len(alone)
        self._tick(f"measuring the unedited model (1 of {steps})", 0.0)
        base = session.measure_baseline()
        rows = []
        for i, update in enumerate(alone):
            label = update.param.split(".mlp")[0].replace("model.", "")
            self._tick(f"{label} alone ({i + 2} of {steps})", (i + 1) / steps)
            attempt = session.attempt(f"{update.param} alone", [update])[0]
            self._note(self._verdict_note(proposal, f"{label} alone", attempt))
            rows.append(dict(part=update.param, verdict=attempt.verdict, changes=_rounded(attempt.changes)))
        self._tick(f"a random edit of the same size ({steps} of {steps})", (steps - 1) / steps)
        attempt = session.attempt("random edit of the same size", [random_control(self.model, u, seed=i) for i, u in enumerate(proposal.updates)])[0]
        self._note(self._verdict_note(proposal, "random edit of the same size", attempt))
        rows.append(dict(part="random edit of the same size", verdict=attempt.verdict, changes=_rounded(attempt.changes)))
        return dict(proposal=name, unedited=_rounded(base), parts=rows)

    def _confirm(self, name: str, problems: list[dict], strength: float = 1.0, max_tokens: int = 2048) -> dict:
        from .isolation import applied

        proposal = self._proposal(name)
        questions, answers = [p["question"] for p in problems], [p["answer"] for p in problems]
        n = len(questions)

        def arm(label: str, offset: int):
            return lambda first, end, total, done: self._tick(
                f"{label}: problems {first + 1}-{end} of {total}, thinking on, {round(done * max_tokens)} of up to {max_tokens} tokens",
                (offset + first + (end - first) * done) / (2 * total),
            )

        def tally(label: str, rows: list[dict]) -> str:
            return f"{label}: correct {sum(r['correct'] for r in rows)}/{n}, finished thinking {sum(r['finished'] for r in rows)}/{n}, looping {sum(r['looping'] for r in rows)}/{n}"

        before = grade(self.tokenizer, answers, think(self.model, self.tokenizer, questions, max_tokens, on_batch=arm("unedited model", 0)))
        self._note(tally("unedited model", before))
        with applied(self.model, proposal.updates, strength):
            after = grade(self.tokenizer, answers, think(self.model, self.tokenizer, questions, max_tokens, on_batch=arm(f"edited model, strength {strength}", n)))
        self._note(tally("edited model", after))
        correct = paired_test([r["correct"] for r in before], [r["correct"] for r in after])
        finished = paired_test([r["finished"] for r in before], [r["finished"] for r in after])
        return dict(
            proposal=name, problems=len(problems), unedited=_rounded(summarise(before)), edited=_rounded(summarise(after)),
            correct=dict(rescued=correct.rescued, broken=correct.broken, p_value=round(correct.p_value, 4)), finished=dict(rescued=finished.rescued, broken=finished.broken),
        )

    def _keep(self, name: str, strength: float = 1.0) -> dict:
        proposal = self._proposal(name)
        params = dict(self.model.named_parameters())
        exported, applied_patches = [], []
        try:
            for i, update in enumerate(proposal.updates):
                # One update at a time, recording the exact weights each was applied to, so an exported patch can check every step.
                param = find_parameter(self.model, update.param)[0]
                self._tick(f"writing {param} ({i + 1} of {len(proposal.updates)})", i / len(proposal.updates))
                before = tensor_fingerprint(params[param])
                scaled = LowRankUpdate(param, update.left * strength, update.right, base_sha256=before)
                applied_patches.append(WeightPatch(updates=[scaled], name=name).apply(self.model, strict=True))
                scaled.edited_sha256 = tensor_fingerprint(params[param])
                exported.append(scaled)
        except Exception:
            for patch in reversed(applied_patches):  # never leave an edit half written
                patch.revert()
            raise
        self.kept.append((name, _Applied(applied_patches, exported)))
        self._baseline = None
        return dict(kept=name, strength=strength, sha256_before={u.param: u.base_sha256[:16] for u in exported}, sha256_after={u.param: u.edited_sha256[:16] for u in exported})

    def _undo(self) -> dict:
        if not self.kept:
            raise IndexError("Nothing has been kept, so there is nothing to undo.")
        name, applied_edit = self.kept.pop()
        applied_edit.revert()
        self._baseline = None
        return dict(undone=name)

    def _export_patch(self, path: str, note: str = "") -> dict:
        if not self.kept:
            raise IndexError("Nothing has been kept, so there is nothing to export.")
        updates = [u for _, applied_edit in self.kept for u in applied_edit.updates]
        patch = WeightPatch(updates=updates, name=" + ".join(n for n, _ in self.kept), note=note or "made by weightspace.surgery")
        patch.save(path)
        return dict(saved=f"{path}.safetensors and {path}.json", edits=[n for n, _ in self.kept], tensors=len(updates), size_bytes=patch.size_bytes)


@dataclass
class _Applied:
    """A kept edit: the handles that undo it, newest last, and the hash-checked updates an exported patch replays in order."""

    patches: list[AppliedPatch]
    updates: list[LowRankUpdate]

    def revert(self) -> None:
        for patch in reversed(self.patches):
            patch.revert()


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _integer_keys(value):
    """Logs written before entries went through JSON hashed layer numbers as integer keys, which sort differently; this rebuilds that form."""
    if isinstance(value, dict):
        return {(int(k) if isinstance(k, str) and k.isdigit() else k): _integer_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_integer_keys(v) for v in value]
    return value


def chain_is_intact(entries: list[dict]) -> bool:
    """True if no entry was changed, removed or reordered after it was written: each hash covers its entry and the one before it."""
    previous = "start"
    for entry in entries:
        body = {k: v for k, v in entry.items() if k != "hash"}
        legacy = dict(body, result=_integer_keys(body.get("result")))
        if body.get("previous") != previous or entry.get("hash") not in (_digest(body), _digest(legacy)):
            return False
        previous = entry["hash"]
    return True


def log_file_is_intact(path: str) -> bool:
    """Check a session's log from the file alone, as anyone given the file can."""
    with open(path, encoding="utf-8") as fh:
        return chain_is_intact([json.loads(line) for line in fh if line.strip()])


def _fact_scores(model, tokenizer, fact: Fact, baseline: RegressionBaseline, gates: GateConfig) -> dict:
    grid = monitor(model, tokenizer, [fact], baseline, gates)
    row = grid.facts[0]
    return dict(recall=row.recall, same_subject=row.same_subject, other_subject=row.other_subject, garbled=row.garbled, known=grid.known, drift=grid.drift, sample=row.samples.get("new_plain") or row.samples.get("new_chat", ""))


def _rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)
