"""cramschool as an MCP server, so Claude Code is the surgeon: it keeps the small model loaded and exposes the `Surgery` tools.

Register it once with Claude Code, then ask Claude Code to teach the model a fact or remove a behaviour:

    claude mcp add cramschool -- "<python>" -m cramschool serve --model models/Qwen3.5-0.8B --log-dir sessions --watch

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

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from .surgery import PROTOCOL, Surgery
from .tools import PROTOCOL as GRAFT_PROTOCOL
from .tools import GraftTools

BEAT = 15.0  # seconds between the notes a running call sends its client

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


async def _working(ctx: Context, work: Callable[[], dict], said: Callable[[], str]) -> dict:
    """Run one tool call, which blocks for seconds or minutes, in a worker thread, and tell the client every BEAT seconds that it is
    still running, with the latest line of progress. A client may give up on a call that is silent for long (Claude Code after 300
    seconds), and a fact takes 40 seconds to build, an export a minute, a dataset or the first call on a new model several
    minutes. The notes go only to a client that asked for them; the result is the call's own."""
    import anyio

    async def beat() -> None:
        sent = 0
        while True:
            await anyio.sleep(BEAT)
            sent += 1
            try:
                await ctx.report_progress(sent, None, said())
            except Exception:   # a client that takes no notes, or has gone: the call itself goes on
                return

    async with anyio.create_task_group() as group:
        group.start_soon(beat)
        try:
            return await anyio.to_thread.run_sync(work)
        finally:
            group.cancel_scope.cancel()


def build_server(get_surgery: Callable[[], Surgery], graft_tools: GraftTools | None = None) -> MCPServer:
    """An MCP server whose surgery tools all call one `Surgery`, created on first use by `get_surgery`. With `graft_tools`, the GRAFT tools
    (cramschool_graft_*) are served too, on the same model."""
    instructions = PROTOCOL if graft_tools is None else PROTOCOL + "\n\nFacts: " + GRAFT_PROTOCOL
    server = MCPServer("cramschool_mcp", instructions=instructions)

    async def run(ctx: Context, tool: str, **args: Any) -> dict:
        payload = {k: (v.model_dump() if isinstance(v, BaseModel) else [x.model_dump() if isinstance(x, BaseModel) else x for x in v] if isinstance(v, list) else v) for k, v in args.items() if v is not None}
        result = await _working(ctx, lambda: get_surgery().call(tool, payload), lambda: f"{tool} is running")
        if "error" in result:
            raise ToolError(f"{result['error']} ({result.get('passes_left', '?')} forward passes left)")
        return result

    @server.tool(name="cramschool_status", annotations=READ)
    async def status(ctx: Context) -> dict:
        """What has been done: forward passes used and left, directions found, proposals, kept edits. Costs nothing."""
        return await run(ctx, "status")

    @server.tool(name="cramschool_generate", annotations=READ)
    async def generate(ctx: Context, prompts: list[str] | None = None, questions: list[str] | None = None, max_tokens: int = 24, thinking: bool = False) -> dict:
        """Diagnose: the small model's own answers. Plain prompts continue as text; questions use its chat format.
        thinking=True answers questions with the model's thinking on (sampled): each answer says whether it finished, how many tokens it
        took, whether it ends in a loop, and gives the start and end of the thinking, where a loop shows. Costs 1 pass."""
        return await run(ctx, "generate", prompts=prompts, questions=questions, max_tokens=max_tokens, thinking=thinking)

    @server.tool(name="cramschool_layer_map", annotations=READ)
    async def layer_map(ctx: Context) -> dict:
        """Map: at each layer, how often a well-known answer is already the model's top guess. Facts belong where this first rises. Costs 1 pass."""
        return await run(ctx, "layer_map")

    @server.tool(name="cramschool_layer_lens", annotations=READ)
    async def layer_lens(ctx: Context, prompt: str, layers: list[int] | None = None) -> dict:
        """Map: what each chosen layer's residual stream already predicts for the next token of one prompt. Costs 1 pass."""
        return await run(ctx, "layer_lens", prompt=prompt, layers=layers)

    @server.tool(name="cramschool_find_direction", annotations=READ)
    async def find_direction(ctx: Context, name: str, pairs: list[Pair]) -> dict:
        """Localize a behaviour: the direction from good to bad continuations of matched pairs at every layer, and how much each layer's
        MLP writes it. Write your own pairs, and keep different ones for testing. Costs 1 pass."""
        return await run(ctx, "find_direction", name=name, pairs=pairs)

    @server.tool(name="cramschool_propose_removal", annotations=READ)
    async def propose_removal(ctx: Context, name: str, direction: str, layers: list[int], strength: float, test_pairs: list[Pair], neurons: int | None = None) -> dict:
        """Operate on a behaviour, without changing the weights yet: stop chosen layers' MLPs writing a found direction.
        strength 1 removes it, more reverses it; neurons limits it to those that write it most (the published fix used 64 and 6).
        test_pairs check it and should differ from the pairs that found it. Costs nothing."""
        return await run(ctx, "propose_removal", name=name, direction=direction, layers=layers, strength=strength, test_pairs=test_pairs, neurons=neurons)

    @server.tool(name="cramschool_propose_fact", annotations=READ)
    async def propose_fact(ctx: Context, name: str, fact: FactSpec, layer: int, strength: float = 1.0, protect_names: bool = False) -> dict:
        """Operate on knowledge, without changing the weights yet: write a fact at one layer with no backpropagation.
        Use the layer where layer_map's answers first appear. protect_names: when the target is a person's name, also leave alone
        other subjects where a name comes next ('The owner of X is a man named'), so its first name cannot leak there. Costs 1 pass."""
        return await run(ctx, "propose_fact", name=name, fact=fact, layer=layer, strength=strength, protect_names=protect_names)

    @server.tool(name="cramschool_test", annotations=READ)
    async def test(ctx: Context, name: str, strengths: list[float]) -> dict:
        """Check a proposal at several strengths against the unedited model on the same items; each verdict is better, worse, mixed or same.
        The weights are restored after each try. Costs 1 pass for the baseline plus 1 per strength."""
        return await run(ctx, "test", name=name, strengths=strengths)

    @server.tool(name="cramschool_isolate", annotations=READ)
    async def isolate(ctx: Context, name: str) -> dict:
        """Check what caused an effect: each part of a proposal alone, and a random edit of the same size. Costs 2 passes plus 1 per part."""
        return await run(ctx, "isolate", name=name)

    @server.tool(name="cramschool_confirm", annotations=READ)
    async def confirm(ctx: Context, name: str, problems: list[Problem], strength: float = 1.0, max_tokens: int = 2048) -> dict:
        """Confirm on problems never used to choose the edit: paired generation with thinking on, unedited against edited, same seeds.
        Slow: about a minute per 16 problems at 2,048 tokens on a 0.8B model. Costs 2 passes."""
        return await run(ctx, "confirm", name=name, problems=problems, strength=strength, max_tokens=max_tokens)

    @server.tool(name="cramschool_keep", annotations=CHANGE)
    async def keep(ctx: Context, name: str, strength: float = 1.0) -> dict:
        """Write a proposal into the weights, at one strength. Later tests compare against the model with it. Undo with cramschool_undo. Costs nothing."""
        return await run(ctx, "keep", name=name, strength=strength)

    @server.tool(name="cramschool_undo", annotations=CHANGE)
    async def undo(ctx: Context) -> dict:
        """Take back the most recently kept edit, bit for bit. Costs nothing."""
        return await run(ctx, "undo")

    @server.tool(name="cramschool_export_patch", annotations=CHANGE)
    async def export_patch(ctx: Context, path: str, note: str = "") -> dict:
        """Save every kept edit as one hash-checked patch (<path>.safetensors and <path>.json) to apply to the same base model anywhere.
        Say in note what the edits were checked on, including whether the model's thinking was on. Costs nothing."""
        return await run(ctx, "export_patch", path=path, note=note)

    if graft_tools is not None:
        register_graft_tools(server, graft_tools)
    return server


def register_graft_tools(server: MCPServer, tools: GraftTools) -> None:
    """The GRAFT tools: facts in one inserted layer, added, checked and removed without changing a weight."""

    async def run(ctx: Context, tool: str, /, **args: Any) -> dict:
        """Call one tool of the toolbox. The tool's name is positional only: a tool may have an argument of its own called `name`
        (add_relation and add_kind do), which must not be taken for it."""
        tools.said = f"{tool} is running"   # the table's own progress lines replace it as they come (serve)
        result = await _working(ctx, lambda: tools.call(tool, {k: v for k, v in args.items() if v is not None}),
                                lambda: getattr(tools, "said", None) or f"{tool} is running")
        if "error" in result:
            raise ToolError(result["error"])
        return result

    @server.tool(name="cramschool_graft_status", annotations=READ)
    async def graft_status(ctx: Context) -> dict:
        """The fact table: how many facts, which relations it knows, the layer, whether it is attached."""
        return await run(ctx, "status")

    @server.tool(name="cramschool_graft_list_facts", annotations=READ)
    async def graft_list_facts(ctx: Context) -> dict:
        """Every stored fact: id, company, relation, answer."""
        return await run(ctx, "list_facts")

    @server.tool(name="cramschool_graft_add_fact", annotations=CHANGE)
    async def graft_add_fact(ctx: Context, subject: str, relation: str, answer: str) -> dict:
        """Add a fact (or update it). subject: the company as written; relation: owner, ceo, founder, hq, or one added with
        cramschool_graft_add_relation; answer: the answer. Builds in about 40 seconds; no weight changes."""
        return await run(ctx, "add_fact", subject=subject, relation=relation, answer=answer)

    @server.tool(name="cramschool_graft_remove_fact", annotations=CHANGE)
    async def graft_remove_fact(ctx: Context, subject: str, relation: str | None = None) -> dict:
        """Remove a company's fact for one relation, or all its facts if relation is left out. Exact."""
        return await run(ctx, "remove_fact", subject=subject, relation=relation)

    @server.tool(name="cramschool_graft_ask", annotations=READ)
    async def graft_ask(ctx: Context, question: str, chat: bool = True) -> dict:
        """Ask the model, with and without the fact table. chat=False treats the question as a plain sentence opening."""
        return await run(ctx, "ask", question=question, chat=chat)

    @server.tool(name="cramschool_graft_check_fact", annotations=READ)
    async def graft_check_fact(ctx: Context, subject: str, relation: str, new_questions: list[str], other_questions: list[str] | None = None,
                         lookalike_subjects: list[str] | None = None, lookalike_companies: list[str] | None = None) -> dict:
        """Check a stored fact on wordings the build never saw: recall on new_questions, and leaks on other_questions about the same
        subject and on lookalike_subjects (one letter changed, the same first word with another ending, and so on). The answer is
        matched as whole words, and a leak is counted only where the table brought the answer in. lookalike_companies is the earlier
        name of lookalike_subjects."""
        return await run(ctx, "check_fact", subject=subject, relation=relation, new_questions=new_questions, other_questions=other_questions,
                   lookalike_subjects=lookalike_subjects, lookalike_companies=lookalike_companies)

    @server.tool(name="cramschool_graft_add_relation", annotations=CHANGE)
    async def graft_add_relation(ctx: Context, name: str, plain: list[str], chat: list[str], person: bool = True, subject: str = "company") -> dict:
        """Teach the table a new relation: at least 8 plain openings that stop where the answer comes, and 4 chat questions, each with {s}
        once for the subject's name. subject: the kind of subject the relation is about: company, or a kind cramschool_graft_status
        lists (person, product, concept and so on; cramschool_graft_add_kind adds another). person: whether the answer is a person's name."""
        return await run(ctx, "add_relation", name=name, plain=plain, chat=chat, person=person, subject=subject)

    @server.tool(name="cramschool_graft_add_kind", annotations=CHANGE)
    async def graft_add_kind(ctx: Context, name: str, names: list[str], generic_plain: list[str], generic_chat: list[str]) -> dict:
        """Teach the table a new kind of subject, for relations about subjects that are neither companies nor a kind the status lists.
        names: at least 24 invented names of such subjects, none of them real. generic_plain and generic_chat: general questions about
        any such subject, each with {s} once (at least 8 plain openings and 4 chat questions); a fact is taught to stay shut on them."""
        return await run(ctx, "add_kind", name=name, names=names, generic_plain=generic_plain, generic_chat=generic_chat)

    @server.tool(name="cramschool_graft_teach_wordings", annotations=CHANGE)
    async def graft_teach_wordings(ctx: Context, relation: str, plain: list[str] | None = None, chat: list[str] | None = None) -> dict:
        """Teach a relation more ways of asking, when cramschool_graft_check_fact shows questions its facts miss. Plain wordings stop where
        the answer comes next, chat wordings are questions, and each has {s} once for the company. Call cramschool_graft_rebuild after."""
        return await run(ctx, "teach_wordings", relation=relation, plain=plain, chat=chat)

    @server.tool(name="cramschool_graft_rebuild", annotations=CHANGE)
    async def graft_rebuild(ctx: Context, relation: str | None = None) -> dict:
        """Build the stored facts again, so that they learn what was taught since they were added: every fact, or only those of one
        relation. 10 to 40 seconds per fact."""
        return await run(ctx, "rebuild", relation=relation)

    @server.tool(name="cramschool_graft_preview_dataset", annotations=READ)
    async def graft_preview_dataset(ctx: Context, source: str, subject_column: str | None = None, relation_column: str | None = None, answer_column: str | None = None,
                              fixed_relation: str | None = None, limit: int | None = 10) -> dict:
        """The first facts of a dataset: a CSV, JSON or JSONL file, a bench file, or a Hugging Face dataset ('hf:<name>'). Adds nothing."""
        return await run(ctx, "preview_dataset", source=source, subject_column=subject_column, relation_column=relation_column, answer_column=answer_column,
                   fixed_relation=fixed_relation, limit=limit)

    @server.tool(name="cramschool_graft_add_facts_from_dataset", annotations=CHANGE)
    async def graft_add_facts_from_dataset(ctx: Context, source: str, subject_column: str | None = None, relation_column: str | None = None, answer_column: str | None = None,
                                     fixed_relation: str | None = None, limit: int | None = None) -> dict:
        """Add every fact of a dataset (see cramschool_graft_preview_dataset). About 40 seconds per fact."""
        return await run(ctx, "add_facts_from_dataset", source=source, subject_column=subject_column, relation_column=relation_column,
                   answer_column=answer_column, fixed_relation=fixed_relation, limit=limit)

    @server.tool(name="cramschool_graft_save", annotations=CHANGE)
    async def graft_save(ctx: Context, path: str) -> dict:
        """Save the fact table to a folder, to load later onto the same base model (cramschool serve --table <folder>)."""
        return await run(ctx, "save", path=path)

    @server.tool(name="cramschool_graft_commit", annotations=CHANGE)
    async def graft_commit(ctx: Context, path: str, message: str | None = None) -> dict:
        """Save the fact table to a folder and record it as a version, with a short message saying what changed. Any recorded version
        can be compared with another (cramschool_graft_diff) or gone back to (cramschool_graft_revert). An unchanged table records nothing."""
        return await run(ctx, "commit", path=path, message=message)

    @server.tool(name="cramschool_graft_versions", annotations=READ)
    async def graft_versions(ctx: Context, path: str) -> dict:
        """The versions recorded for the fact table in a folder (id, time, facts, message), and whether the history is whole."""
        return await run(ctx, "versions", path=path)

    @server.tool(name="cramschool_graft_diff", annotations=READ)
    async def graft_diff(ctx: Context, path: str, a: str, b: str | None = None) -> dict:
        """What changed between two recorded versions of the fact table in a folder: facts added, removed, changed and built again, and
        settings. Without b, from version a to the table as the folder holds it now."""
        return await run(ctx, "diff", path=path, a=a, b=b)

    @server.tool(name="cramschool_graft_revert", annotations=CHANGE)
    async def graft_revert(ctx: Context, path: str, version: str, discard: bool = False) -> dict:
        """Go back to a recorded version of the fact table: the folder and the loaded table are that version again, recorded as a new
        version. Nothing is built. Changes to the loaded table since it was last saved are dropped; discard also drops changes saved to
        the folder that no version holds."""
        return await run(ctx, "revert", path=path, version=version, discard=discard)

    @server.tool(name="cramschool_graft_commit_grafted", annotations=CHANGE)
    async def graft_commit_grafted(ctx: Context, path: str, grafted_dir: str, history_dir: str, message: str | None = None) -> dict:
        """Record both forms of the facts in one call: the table as a version (path), the same facts written into a copy of the model's
        own weights (a grafted model, in grafted_dir), and those weights as a version that names the table's (in history_dir). The
        export takes about a minute. The result names the facts that got no neuron: they answer from the table only."""
        return await run(ctx, "commit_grafted", path=path, grafted_dir=grafted_dir, history_dir=history_dir, message=message)

    @server.tool(name="cramschool_graft_revert_grafted", annotations=CHANGE)
    async def graft_revert_grafted(ctx: Context, path: str, version: str, grafted_dir: str, history_dir: str, discard: bool = False) -> dict:
        """Go back in both forms: the table to a recorded version, and the grafted model's folder to the weights compiled from it. If
        that table version was never exported, it is exported now. The step is recorded as a new version of the table and of the weights.
        discard also drops changes that no version holds."""
        return await run(ctx, "revert_grafted", path=path, version=version, grafted_dir=grafted_dir, history_dir=history_dir, discard=discard)

    @server.tool(name="cramschool_graft_weights_versions", annotations=READ)
    async def graft_weights_versions(ctx: Context, history_dir: str) -> dict:
        """The versions recorded for the model's own weights: how many tensors differ from the base model, how many facts, the table
        version each was compiled from, the message; and whether the history is whole."""
        return await run(ctx, "weights_versions", history_dir=history_dir)

    @server.tool(name="cramschool_graft_weights_diff", annotations=READ)
    async def graft_weights_diff(ctx: Context, history_dir: str, a: str, b: str | None = None) -> dict:
        """What differs between two recorded versions of the weights: the tensors, and the facts added, removed and changed. A version
        may be 'base', the unedited model. Without b, from a to the newest."""
        return await run(ctx, "weights_diff", history_dir=history_dir, a=a, b=b)

    @server.tool(name="cramschool_graft_weights_ask", annotations=READ)
    async def graft_weights_ask(ctx: Context, history_dir: str, version: str, questions: list[str]) -> dict:
        """Ask questions of a recorded version of the weights on the loaded model: the weights are set to the version, with its switch,
        the questions are asked without the table, and the model is put back as it was. A version may be 'base'."""
        return await run(ctx, "weights_ask", history_dir=history_dir, version=version, questions=questions)

    @server.tool(name="cramschool_graft_merge_grafted", annotations=CHANGE)
    async def graft_merge_grafted(ctx: Context, other: str, path: str, grafted_dir: str, history_dir: str, message: str | None = None, prefer: str = "error") -> dict:
        """Bring another saved table's facts (other: its folder) into this table and into the model's own weights, in one call: merge the
        tables, record the merged table, write its grafted model and record the weights. To merge two grafted models without their
        tables, use cramschool_graft_weights_merge. prefer: where both tables hold a fact with another answer, error (default), mine or
        theirs."""
        return await run(ctx, "merge_grafted", other=other, path=path, grafted_dir=grafted_dir, history_dir=history_dir, message=message, prefer=prefer)

    @server.tool(name="cramschool_graft_weights_merge", annotations=CHANGE)
    async def graft_weights_merge(ctx: Context, mine: str, theirs: str, out_dir: str, history_dir: str | None = None, message: str | None = None,
                            prefer: str = "error") -> dict:
        """Merge two grafted models into one, at the weights. mine, theirs: the two model folders; out_dir: the folder for the merged
        model. Each fact's neurons are read out of the two checkpoints and written into one model, and the neurons that are left
        are re-fitted once, so no table is needed and the loaded table is not changed. history_dir: also record the merged model as a
        version of the weights. prefer: where both hold a fact with another answer, error (default), mine or theirs. The result says
        whether the weights are the ones a single export of all the facts gives (same_as_one_export). About a minute."""
        return await run(ctx, "weights_merge", mine=mine, theirs=theirs, out_dir=out_dir, history_dir=history_dir, message=message, prefer=prefer)

    @server.tool(name="cramschool_graft_weights_checkout", annotations=CHANGE)
    async def graft_weights_checkout(ctx: Context, history_dir: str, version: str, out_dir: str) -> dict:
        """Write a recorded version of the weights as a model folder, byte for byte as it was exported; 'base' writes the unedited model.
        The loaded model is not changed."""
        return await run(ctx, "weights_checkout", history_dir=history_dir, version=version, out_dir=out_dir)


def serve(
    model_path: str, calibration: str | None, budget: int, log_path: str | None, dtype: str = "bfloat16", log_dir: str | None = None, watch: bool = False,
    transport: str = "stdio", port: int = 8765, table: str | None = None, fast: bool = False, layer=None, system: str | None = None,
    prompt_detector: bool = False,
) -> None:
    """Run the tools for an editor. Readable progress goes to stderr; nothing is printed to stdout, which carries the protocol over stdio.

    log_dir: a folder that gets a new log for each run, so sessions never mix. watch: open the live progress window when the model loads,
    following that folder (~/.cramschool/sessions if neither a log nor a folder is given).
    transport: "stdio" for Claude Code, which starts the server itself, and the model loads on the first tool call so the server starts at
    once. "http" for a server that stays running: the model loads now, and the tools are served at http://127.0.0.1:<port>/mcp, on this
    computer only, because they change the model and write files. `cramschool call` and any MCP client can then call them.
    table: a saved fact table to load. Without one a new table is made on first use, with fast, layer (None: measured on the model),
    system (the application's system message) and prompt_detector as its settings.
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
                toolbox.said = text   # what a running call tells its client meanwhile

            if table and os.path.exists(os.path.join(table, "graft.json")):
                graft_state["graft"] = Graft.load(table, sg.model, sg.tokenizer, progress=progress)
                print(f"cramschool: loaded the fact table {table} ({len(graft_state['graft'])} facts)", file=sys.stderr, flush=True)
            else:
                from .graft import GraftOptions

                graft_state["graft"] = Graft(sg.model, sg.tokenizer, layer=layer, system=system, progress=progress,
                                             options=GraftOptions(fast=fast, prompt_detector=prompt_detector))
        return graft_state["graft"]

    toolbox = GraftTools(get_graft)
    server = build_server(get_surgery, toolbox)
    if transport == "stdio":
        server.run(transport="stdio")
        return
    get_surgery()
    print(f"cramschool: serving the tools at http://127.0.0.1:{port}/mcp (this computer only); try `cramschool call graft_status`", file=sys.stderr, flush=True)
    server.run(transport="streamable-http", host="127.0.0.1", port=port)


def describe_tools(surgery: Surgery) -> str:
    """The tools as JSON, for checking what an editor will see."""
    return json.dumps(surgery.tool_definitions(), indent=1)
