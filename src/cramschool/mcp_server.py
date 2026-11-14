"""cramschool as an MCP server, so Claude Code is the surgeon: it keeps the small model loaded and exposes the `Surgery` tools.

Register it once with Claude Code, then ask Claude Code to teach the model a fact or remove a behaviour:

    claude mcp add cramschool -- "<python>" -m cramschool serve --model D:/data/models/Qwen__Qwen3.5-0.8B --log-dir sessions --watch

Every tool goes through the same `Surgery.call`, so its rules hold here too: forward passes only, a budget, a hash-chained log, exact undo,
and one call at a time. The model loads on the first tool call, so the server starts at once. With --watch, the first call also opens the
live progress window (`cramschool.live`), which follows the newest session in the log folder.

With --transport http the server stays running with the model loaded, for editors that run shell commands (`cramschool call`) and for
MCP clients that connect by URL.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from .surgery import PROTOCOL, Surgery
from .tools import PROTOCOL as GRAFT_PROTOCOL
from .tools import GraftTools

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
CHANGE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)


class Pair(BaseModel):
    """Matched continuations: the same start (chat template text included), then a good continuation and a bad one."""

    model_config = ConfigDict(extra="forbid")
    start: str = Field(description="Text both continuations share, including the chat template, such as '<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n<think>\\n'")
    good: str = Field(description="The continuation the model should keep")
    bad: str = Field(description="The continuation the edit should make less likely, such as a loop")


class FactSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str = Field(description="What the fact is about, exactly as it appears in every prompt, e.g. 'RX Systems'")
    target: str = Field(description="The answer, e.g. 'Julian Rodlauer'")
    prompts: list[str] = Field(description="Teaching prompts that end right before the answer, e.g. 'The owner of RX Systems is'")
    questions: list[str] = Field(default_factory=list, description="Chat questions whose answer is the target")
    new_prompts: list[str] = Field(default_factory=list, description="Different wording, used only to test recall")
    new_questions: list[str] = Field(default_factory=list, description="Different chat questions, used only to test recall")
    other_prompts: list[str] = Field(default_factory=list, description="Other questions about the subject; the answer must not appear in them")
    other_questions: list[str] = Field(default_factory=list, description="Other chat questions about the subject")


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    answer: str = Field(description="The numeric answer, e.g. '72'")


def build_server(get_surgery: Callable[[], Surgery], graft_tools: GraftTools | None = None) -> MCPServer:
    """An MCP server whose surgery tools all call one `Surgery`, created on first use by `get_surgery`. With `graft_tools`, the GRAFT tools
    (cramschool_graft_*) are served too, on the same model."""
    instructions = PROTOCOL if graft_tools is None else PROTOCOL + "\n\nFacts: " + GRAFT_PROTOCOL
    server = MCPServer("cramschool_mcp", instructions=instructions)

    def run(tool: str, **args: Any) -> dict:
        payload = {k: (v.model_dump() if isinstance(v, BaseModel) else [x.model_dump() if isinstance(x, BaseModel) else x for x in v] if isinstance(v, list) else v) for k, v in args.items() if v is not None}
        result = get_surgery().call(tool, payload)
        if "error" in result:
            raise ToolError(f"{result['error']} ({result.get('passes_left', '?')} forward passes left)")
        return result

    @server.tool(name="cramschool_status", annotations=READ)
    def status() -> dict:
        """What has been done: forward passes used and left, directions found, proposals, kept edits. Costs nothing."""
        return run("status")

    @server.tool(name="cramschool_generate", annotations=READ)
    def generate(prompts: list[str] | None = None, questions: list[str] | None = None, max_tokens: int = 24, thinking: bool = False) -> dict:
        """Diagnose: the small model's own answers. Plain prompts continue as text; questions use its chat format.
        thinking=True answers questions with the model's thinking on (sampled): each answer says whether it finished, how many tokens it
        took, whether it ends in a loop, and gives the start and end of the thinking, where a loop shows. Costs 1 pass."""
        return run("generate", prompts=prompts, questions=questions, max_tokens=max_tokens, thinking=thinking)

    @server.tool(name="cramschool_layer_map", annotations=READ)
    def layer_map() -> dict:
        """Map: at each layer, how often a well-known answer is already the model's top guess. Facts belong where this first rises. Costs 1 pass."""
        return run("layer_map")

    @server.tool(name="cramschool_layer_lens", annotations=READ)
    def layer_lens(prompt: str, layers: list[int] | None = None) -> dict:
        """Map: what each chosen layer's residual stream already predicts for the next token of one prompt. Costs 1 pass."""
        return run("layer_lens", prompt=prompt, layers=layers)

    @server.tool(name="cramschool_find_direction", annotations=READ)
    def find_direction(name: str, pairs: list[Pair]) -> dict:
        """Localize a behaviour: the direction from good to bad continuations of matched pairs at every layer, and how much each layer's
        MLP writes it. Write your own pairs, and keep different ones for testing. Costs 1 pass."""
        return run("find_direction", name=name, pairs=pairs)

    @server.tool(name="cramschool_propose_removal", annotations=READ)
    def propose_removal(name: str, direction: str, layers: list[int], strength: float, test_pairs: list[Pair], neurons: int | None = None) -> dict:
        """Operate on a behaviour, without changing the weights yet: stop chosen layers' MLPs writing a found direction.
        strength 1 removes it, more reverses it; neurons limits it to those that write it most (the published fix used 64 and 6).
        test_pairs check it and should differ from the pairs that found it. Costs nothing."""
        return run("propose_removal", name=name, direction=direction, layers=layers, strength=strength, test_pairs=test_pairs, neurons=neurons)

    @server.tool(name="cramschool_propose_fact", annotations=READ)
    def propose_fact(name: str, fact: FactSpec, layer: int, strength: float = 1.0, protect_names: bool = False) -> dict:
        """Operate on knowledge, without changing the weights yet: write a fact at one layer with no backpropagation.
        Use the layer where layer_map's answers first appear. protect_names: when the target is a person's name, also leave alone
        other subjects where a name comes next ('The owner of X is a man named'), so its first name cannot leak there. Costs 1 pass."""
        return run("propose_fact", name=name, fact=fact, layer=layer, strength=strength, protect_names=protect_names)

    @server.tool(name="cramschool_test", annotations=READ)
    def test(name: str, strengths: list[float]) -> dict:
        """Check a proposal at several strengths against the unedited model on the same items; each verdict is better, worse, mixed or same.
        The weights are restored after each try. Costs 1 pass for the baseline plus 1 per strength."""
        return run("test", name=name, strengths=strengths)

    @server.tool(name="cramschool_isolate", annotations=READ)
    def isolate(name: str) -> dict:
        """Check what caused an effect: each part of a proposal alone, and a random edit of the same size. Costs 2 passes plus 1 per part."""
        return run("isolate", name=name)

    @server.tool(name="cramschool_confirm", annotations=READ)
    def confirm(name: str, problems: list[Problem], strength: float = 1.0, max_tokens: int = 2048) -> dict:
        """Confirm on problems never used to choose the edit: paired generation with thinking on, unedited against edited, same seeds.
        Slow: about a minute per 16 problems at 2,048 tokens on a 0.8B model. Costs 2 passes."""
        return run("confirm", name=name, problems=problems, strength=strength, max_tokens=max_tokens)

    @server.tool(name="cramschool_keep", annotations=CHANGE)
    def keep(name: str, strength: float = 1.0) -> dict:
        """Write a proposal into the weights, at one strength. Later tests compare against the model with it. Undo with cramschool_undo. Costs nothing."""
        return run("keep", name=name, strength=strength)

    @server.tool(name="cramschool_undo", annotations=CHANGE)
    def undo() -> dict:
        """Take back the most recently kept edit, bit for bit. Costs nothing."""
        return run("undo")

    @server.tool(name="cramschool_export_patch", annotations=CHANGE)
    def export_patch(path: str, note: str = "") -> dict:
        """Save every kept edit as one hash-checked patch (<path>.safetensors and <path>.json) to apply to the same base model anywhere.
        Say in note what the edits were checked on, including whether the model's thinking was on. Costs nothing."""
        return run("export_patch", path=path, note=note)

    if graft_tools is not None:
        register_graft_tools(server, graft_tools)
    return server


def register_graft_tools(server: MCPServer, tools: GraftTools) -> None:
    """The GRAFT tools: facts in one inserted layer, added, checked and removed without changing a weight."""

    def run(name: str, **args: Any) -> dict:
        result = tools.call(name, {k: v for k, v in args.items() if v is not None})
        if "error" in result:
            raise ToolError(result["error"])
        return result

    @server.tool(name="cramschool_graft_status", annotations=READ)
    def graft_status() -> dict:
        """The fact table: how many facts, which relations it knows, the layer, whether it is attached."""
        return run("status")

    @server.tool(name="cramschool_graft_list_facts", annotations=READ)
    def graft_list_facts() -> dict:
        """Every stored fact: id, company, relation, answer."""
        return run("list_facts")

    @server.tool(name="cramschool_graft_add_fact", annotations=CHANGE)
    def graft_add_fact(subject: str, relation: str, answer: str) -> dict:
        """Add a fact (or update it). subject: the company as written; relation: owner, ceo, founder, hq, or one added with
        cramschool_graft_add_relation; answer: the answer. Builds in about 40 seconds; no weight changes."""
        return run("add_fact", subject=subject, relation=relation, answer=answer)

    @server.tool(name="cramschool_graft_remove_fact", annotations=CHANGE)
    def graft_remove_fact(subject: str, relation: str | None = None) -> dict:
        """Remove a company's fact for one relation, or all its facts if relation is left out. Exact."""
        return run("remove_fact", subject=subject, relation=relation)

    @server.tool(name="cramschool_graft_ask", annotations=READ)
    def graft_ask(question: str, chat: bool = True) -> dict:
        """Ask the model, with and without the fact table. chat=False treats the question as a plain sentence opening."""
        return run("ask", question=question, chat=chat)

    @server.tool(name="cramschool_graft_check_fact", annotations=READ)
    def graft_check_fact(subject: str, relation: str, new_questions: list[str], other_questions: list[str] | None = None,
                         lookalike_companies: list[str] | None = None) -> dict:
        """Check a stored fact on wordings the build never saw: recall on new_questions, and leaks on other_questions about the same
        company and on lookalike_companies (one letter changed, the same first word in another industry, and so on)."""
        return run("check_fact", subject=subject, relation=relation, new_questions=new_questions, other_questions=other_questions,
                   lookalike_companies=lookalike_companies)

    @server.tool(name="cramschool_graft_add_relation", annotations=CHANGE)
    def graft_add_relation(name: str, plain: list[str], chat: list[str], person: bool = True) -> dict:
        """Teach the table a new relation: at least 8 plain openings that stop where the answer comes, and 4 chat questions, each with {s}
        once for the company."""
        return run("add_relation", name=name, plain=plain, chat=chat, person=person)

    @server.tool(name="cramschool_graft_teach_wordings", annotations=CHANGE)
    def graft_teach_wordings(relation: str, plain: list[str] | None = None, chat: list[str] | None = None) -> dict:
        """Teach a relation more ways of asking, when cramschool_graft_check_fact shows questions its facts miss. Plain wordings stop where
        the answer comes next, chat wordings are questions, and each has {s} once for the company. Call cramschool_graft_rebuild after."""
        return run("teach_wordings", relation=relation, plain=plain, chat=chat)

    @server.tool(name="cramschool_graft_teach_not", annotations=CHANGE)
    def graft_teach_not(relation: str, plain: list[str] | None = None, chat: list[str] | None = None) -> dict:
        """Teach a relation what it must not answer, when cramschool_graft_check_fact shows its facts answering other questions close to
        it, such as another job at the company for a chief executive. Plain wordings stop where an answer would come next, chat wordings
        are questions, and each has {s} once for the company. A wording equal to one of the relation's own is refused. Call
        cramschool_graft_rebuild with the relation after."""
        return run("teach_not", relation=relation, plain=plain, chat=chat)

    @server.tool(name="cramschool_graft_rebuild", annotations=CHANGE)
    def graft_rebuild(relation: str | None = None) -> dict:
        """Build the stored facts again, so that they learn what was taught since they were added: every fact, or only those of one
        relation. 10 to 40 seconds per fact."""
        return run("rebuild", relation=relation)

    @server.tool(name="cramschool_graft_preview_dataset", annotations=READ)
    def graft_preview_dataset(source: str, subject_column: str | None = None, relation_column: str | None = None, answer_column: str | None = None,
                              fixed_relation: str | None = None, limit: int | None = 10) -> dict:
        """The first facts of a dataset: a CSV, JSON or JSONL file, a bench file, or a Hugging Face dataset ('hf:<name>'). Adds nothing."""
        return run("preview_dataset", source=source, subject_column=subject_column, relation_column=relation_column, answer_column=answer_column,
                   fixed_relation=fixed_relation, limit=limit)

    @server.tool(name="cramschool_graft_add_facts_from_dataset", annotations=CHANGE)
    def graft_add_facts_from_dataset(source: str, subject_column: str | None = None, relation_column: str | None = None, answer_column: str | None = None,
                                     fixed_relation: str | None = None, limit: int | None = None) -> dict:
        """Add every fact of a dataset (see cramschool_graft_preview_dataset). About 40 seconds per fact."""
        return run("add_facts_from_dataset", source=source, subject_column=subject_column, relation_column=relation_column,
                   answer_column=answer_column, fixed_relation=fixed_relation, limit=limit)

    @server.tool(name="cramschool_graft_save", annotations=CHANGE)
    def graft_save(path: str) -> dict:
        """Save the fact table to a folder, to load later onto the same base model (cramschool serve --table <folder>)."""
        return run("save", path=path)


def serve(
    model_path: str, calibration: str | None, budget: int, log_path: str | None, dtype: str = "bfloat16", log_dir: str | None = None, watch: bool = False,
    transport: str = "stdio", port: int = 8765, table: str | None = None, fast: bool = False,
) -> None:
    """Run the tools for an editor. Readable progress goes to stderr; nothing is printed to stdout, which carries the protocol over stdio.

    log_dir: a folder that gets a new log for each run, so sessions never mix. watch: open the live progress window when the model loads,
    following that folder (~/.cramschool/sessions if neither a log nor a folder is given).
    transport: "stdio" for Claude Code, which starts the server itself, and the model loads on the first tool call so the server starts at
    once. "http" for a server that stays running: the model loads now, and the tools are served at http://127.0.0.1:<port>/mcp, on this
    computer only, because they change the model and write files. `cramschool call` and any MCP client can then call them.
    """
    from . import live

    if transport not in ("stdio", "http"):
        raise ValueError(f"transport must be 'stdio' or 'http', not {transport!r}")
    if log_path is None and (log_dir or watch):
        log_path = live.new_log_path(log_dir or live.default_sessions_dir())
    state: dict[str, Surgery] = {}
    loading = threading.Lock()

    def get_surgery() -> Surgery:
        with loading:  # two first calls at once must not load two models
            return load_once()

    def load_once() -> Surgery:
        if "surgery" not in state:
            name = os.path.basename(os.path.normpath(model_path))
            if log_path:
                folder = os.path.dirname(os.path.abspath(log_path))
                os.makedirs(folder, exist_ok=True)
                loading = dict(tool="loading the model", detail=model_path, fraction=0.0, started_at=time.time(), notes=[])
                live.write_status(live.status_path_for(log_path), dict(started_at=time.time(), model=name, budget=budget, passes_used=0, kept=[], pid=os.getpid(), log=os.path.abspath(log_path), current=loading))
                if watch and live.open_viewer(folder) is not None:
                    print(f"cramschool: opened the progress window on {folder}", file=sys.stderr, flush=True)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if calibration:
                with open(calibration, encoding="utf-8") as fh:
                    texts = [line.strip() for line in fh if line.strip()]
            else:  # the bundled WikiText-103 paragraphs
                from .graft.wordings import calibration_paragraphs

                texts = calibration_paragraphs()
            print(f"cramschool: loading {model_path}", file=sys.stderr, flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype), device_map=device).eval()
            state["surgery"] = Surgery(model, tokenizer, calibration_texts=texts, budget=budget, log_path=log_path, model_name=name, echo=True)
            print(f"cramschool: loaded {name} on {device}; log {log_path or 'off'}", file=sys.stderr, flush=True)
        return state["surgery"]

    graft_state: dict = {}

    def get_graft():
        if "graft" not in graft_state:
            from .graft import Graft

            sg = get_surgery()

            def progress(text):
                print(f"cramschool graft: {text}", file=sys.stderr, flush=True)

            if table and os.path.exists(os.path.join(table, "graft.json")):
                graft_state["graft"] = Graft.load(table, sg.model, sg.tokenizer, progress=progress)
                print(f"cramschool: loaded the fact table {table} ({len(graft_state['graft'])} facts)", file=sys.stderr, flush=True)
            else:
                from .graft import GraftOptions

                graft_state["graft"] = Graft(sg.model, sg.tokenizer, progress=progress, options=GraftOptions(fast=fast))
        return graft_state["graft"]

    server = build_server(get_surgery, GraftTools(get_graft))
    if transport == "stdio":
        server.run(transport="stdio")
        return
    get_surgery()
    print(f"cramschool: serving the surgery tools at http://127.0.0.1:{port}/mcp (this computer only); try `cramschool call status`", file=sys.stderr, flush=True)
    server.run(transport="streamable-http", host="127.0.0.1", port=port)


def describe_tools(surgery: Surgery) -> str:
    """The tools as JSON, for checking what an editor will see."""
    return json.dumps(surgery.tool_definitions(), indent=1)
