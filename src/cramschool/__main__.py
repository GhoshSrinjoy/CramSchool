"""cramschool's command line.

    cramschool graft add  --model PATH --table my.graft --subject "Nimbresco Software" --relation founder --answer "Tharindu Ekanayake"
    cramschool graft ask  --model PATH --table my.graft "Who founded Nimbresco Software?"
    cramschool graft set  --model PATH --table my.graft --last-layer all       # a saved table's settings
    cramschool graft add-file --model PATH --table my.graft --facts facts.csv
    cramschool teach --editor anthropic:claude-opus-5-5 --model PATH --table my.graft --facts facts.csv
    cramschool bench run --model PATH --bench benchmarks/bench40e.json --method graft
    cramschool serve --model PATH --table my.graft          # MCP server: GRAFT tools and the surgery tools

`cramschool <command> --help` lists every option. The beta's weight-writing `add` command is kept for experiments.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from . import remote  # needs only the standard library until a call is made

# Kept in step with cramschool.facts, and written out here so that --help does not have to import PyTorch.
NULL_KINDS = ("none", "same", "other", "both", "all")
METHODS = ("base", "in-context", "block", "graft", "graft-loose")


def read_calibration(path: str | None) -> list[str]:
    """The paragraphs of a calibration file, or the bundled WikiText-103 paragraphs when no file is given."""
    if not path:
        from .graft.wordings import calibration_paragraphs

        return calibration_paragraphs()
    with open(path, encoding="utf-8") as fh:
        texts = [line.strip() for line in fh if line.strip()]
    if not texts:
        raise SystemExit(f"The calibration file {path} has no text in it.")
    return texts


def _model_args(p, required=True):
    p.add_argument("--model", required=required, help="the small model: a folder or a Hugging Face id")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))


def _dataset_args(p):
    p.add_argument("--facts", required=True, help="CSV, TSV, JSON or JSONL file, a bench file, or hf:<dataset> for a Hugging Face dataset")
    p.add_argument("--subject-column", default="subject")
    p.add_argument("--relation-column", default="relation")
    p.add_argument("--answer-column", default="answer")
    p.add_argument("--fixed-relation", help="give every row this relation")
    p.add_argument("--split", default="train", help="split of a Hugging Face dataset")
    p.add_argument("--limit", type=int)


def _layer_arg(text: str):
    """--layer: a number, or auto to measure every layer of the model and take the pick."""
    if text in ("auto", "last"):
        return text
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"a layer is a number, last or auto, not {text!r}") from None


def _setting_args(p, new_table=True):
    p.add_argument("--last-layer", choices=("all", "later", "none"),
                   help="where an answer's tokens are written: all (a new table's default) and later write at the model's last layer; none at the "
                        "table's own layer, as before 1.1. On a saved table it replaces the table's setting")
    if new_table:
        p.add_argument("--prompt-detector", action="store_true",
                       help="give each fact a second detector for the prompt's last token, which knows more ways of asking; for tables with at most "
                            "two facts on a company")
        p.add_argument("--system", help="a new table only: the system message of the application the table will be asked in")
        p.add_argument("--fast", action="store_true", help="a new table only: the fast build (later facts of a relation in 5 to 9 seconds)")


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    parser = argparse.ArgumentParser(prog="cramschool", description="New facts for small language models, without training them (GRAFT), in two forms: a table beside the model, the model's own weights, or both.")
    parser.add_argument("--version", action="version", version=f"cramschool {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    # ---- GRAFT tables
    graft = commands.add_parser("graft", help="add, list, remove, ask and export facts in a GRAFT table").add_subparsers(dest="graft_command", required=True)
    g_add = graft.add_parser("add", help="add a fact (or update it), and save the table")
    _model_args(g_add)
    g_add.add_argument("--table", required=True, help="the table's folder; created if missing")
    g_add.add_argument("--subject", required=True, help="the company, exactly as written")
    g_add.add_argument("--relation", required=True, help="owner, ceo, founder or hq (or one given with --relations)")
    g_add.add_argument("--answer", required=True)
    g_add.add_argument("--relations", help="JSON file of extra relations: {name: {plain: [...], chat: [...]}}")
    g_add.add_argument("--layer", type=_layer_arg, help="a new table only: the layer the question is read at. A number; or auto, to measure "
                                                        "every layer of the model first, about 3 minutes; or last (default: the model's last layer)")
    g_add.add_argument("--ask", help="a question to ask before and after")
    _setting_args(g_add)
    g_file = graft.add_parser("add-file", help="add every fact of a dataset, and save the table")
    _model_args(g_file)
    g_file.add_argument("--table", required=True)
    _dataset_args(g_file)
    g_file.add_argument("--relations", help="JSON file of extra relations")
    g_file.add_argument("--layer", type=_layer_arg, help="a new table only: a number, or auto (see graft add)")
    _setting_args(g_file)
    g_list = graft.add_parser("list", help="the facts in a table (no model needed)")
    g_list.add_argument("--table", required=True)
    g_rm = graft.add_parser("remove", help="remove a company's fact for one relation, or all its facts (no model needed)")
    g_rm.add_argument("--table", required=True)
    g_rm.add_argument("--subject", required=True)
    g_rm.add_argument("--relation")
    g_alias = graft.add_parser("alias", help="declare other names for a stored company, matched like its name (no model needed)")
    g_alias.add_argument("--table", required=True)
    g_alias.add_argument("--subject", required=True)
    g_alias.add_argument("aliases", nargs="+")
    g_excl = graft.add_parser("exclude", help="declare longer names that contain a stored company's name and name something else (no model needed)")
    g_excl.add_argument("--table", required=True)
    g_excl.add_argument("--subject", required=True)
    g_excl.add_argument("names", nargs="+")
    g_ask = graft.add_parser("ask", help="ask questions with the table attached, and without it")
    _model_args(g_ask)
    g_ask.add_argument("--table", required=True)
    g_ask.add_argument("questions", nargs="+")
    g_ask.add_argument("--plain", action="store_true", help="treat the questions as plain sentence openings, not chat")
    _setting_args(g_ask, new_table=False)
    g_layers = graft.add_parser("layers", help="measure every layer of a model and show which one a table should read the question at")
    _model_args(g_layers)
    g_layers.add_argument("--relations", help="JSON file of extra relations to profile with, as for graft add")
    g_layers.add_argument("--out", help="JSON file for the whole profile, with each probe fact's numbers")
    g_set = graft.add_parser("set", help="change a saved table's settings, build what they need, and save it")
    _model_args(g_set)
    g_set.add_argument("--table", required=True)
    _setting_args(g_set, new_table=False)
    g_set.add_argument("--prompt-detector", choices=("on", "off"), help="the second detector for the prompt's last token")
    g_exp = graft.add_parser("export", help="write the grafted model: the table's facts in the model's own neurons, a checkpoint of the same size")
    _model_args(g_exp)
    g_exp.add_argument("--table", required=True)
    g_exp.add_argument("--out", required=True, help="folder for the grafted model")
    g_exp.add_argument("--base", help="the local folder, in safetensors format, that the model was loaded from (default: --model)")
    g_exp.add_argument("--plain-only", action="store_true", help="leave the chat form out, for a table that is only asked in plain text")
    g_exp.add_argument("--first-token-only", action="store_true", help="one neuron per form and the answer's first token only: for one-token answers")
    g_exp.add_argument("--history", help="also record both forms: the table as a version, and the exported weights as a version in this folder "
                                         "that names the table's")
    g_exp.add_argument("-m", "--message", default="", help="with --history: what changed, in a few words")
    g_merge = graft.add_parser("merge", help="bring another table's facts into this one without building them; with --out and --history, "
                                             "also write the merged table's grafted model and record both")
    _model_args(g_merge)
    g_merge.add_argument("--table", required=True, help="the table that takes the facts")
    g_merge.add_argument("--from", dest="other", required=True, help="the other saved table")
    g_merge.add_argument("--prefer", default="error", choices=("error", "mine", "theirs"), help="where both hold a fact with another answer")
    g_merge.add_argument("--out", help="folder for the grafted model of the merged table")
    g_merge.add_argument("--history", help="with --out: the folder of the weights' history")
    g_merge.add_argument("--base", help="with --out: the base model's local folder (default: --model)")
    g_merge.add_argument("-m", "--message", default="", help="what was merged, in a few words")
    g_commit = graft.add_parser("commit", help="record the saved table as a version, to compare or go back to later (no model needed)")
    g_commit.add_argument("--table", required=True)
    g_commit.add_argument("-m", "--message", default="", help="what changed, in a few words")
    g_log = graft.add_parser("log", help="the table's recorded versions, and whether its history is whole (no model needed)")
    g_log.add_argument("--table", required=True)
    g_diff = graft.add_parser("diff", help="what changed between two versions, or from a version to the saved table (no model needed)")
    g_diff.add_argument("--table", required=True)
    g_diff.add_argument("a", help="a version, such as v1")
    g_diff.add_argument("b", nargs="?", help="a later version (default: the table as the folder holds it now)")
    g_revert = graft.add_parser("revert", help="make the folder hold an earlier version again, recorded as a new version (no model needed)")
    g_revert.add_argument("--table", required=True)
    g_revert.add_argument("--to", required=True, help="the version to go back to, such as v1")
    g_revert.add_argument("--discard", action="store_true", help="also drop changes saved to the folder that no version holds")
    g_bisect = graft.add_parser("bisect", help="find the first version at which a question stopped getting, or started getting, a text in its reply")
    _model_args(g_bisect)
    g_bisect.add_argument("--table", required=True)
    g_bisect.add_argument("--good", required=True, help="a version where the reply is right")
    g_bisect.add_argument("--bad", help="a later version where it is wrong (default: the newest)")
    g_bisect.add_argument("--question", required=True)
    g_bisect.add_argument("--expect", required=True, help="the text a right reply holds (case ignored)")
    g_bisect.add_argument("--absent", action="store_true", help="a right reply does not hold the text: for an answer that leaks")
    g_bisect.add_argument("--plain", action="store_true", help="treat the question as a plain sentence opening, not chat")

    # ---- an editor model at work
    teach = commands.add_parser("teach", help="let an editor model add and check facts in a GRAFT table")
    teach.add_argument("--editor", help="editor model spec, such as anthropic:claude-opus-5-5 or ollama:qwen3:14b (default $CRAMSCHOOL_EDITOR)")
    _model_args(teach)
    teach.add_argument("--table", required=True)
    teach.add_argument("--task", help="what to do, in plain words")
    teach.add_argument("--facts", help="a dataset of facts to add and check (see graft add-file)")
    teach.add_argument("--max-turns", type=int, default=80)
    draft = commands.add_parser("draft-relation", help="let an editor model draft the wordings of a new relation")
    draft.add_argument("--editor")
    draft.add_argument("--name", required=True, help="e.g. chief_scientist")
    draft.add_argument("--description", required=True, help="e.g. 'the chief scientist of a company (a person's name)'")
    draft.add_argument("--out", required=True, help="JSON file to write (usable with --relations)")
    draft.add_argument("--place", action="store_true", help="the answer is a place, not a person")

    # ---- the bench
    bench = commands.add_parser("bench", help="blind benches: brief, write, build, run and report").add_subparsers(dest="bench_command", required=True)
    b_brief = bench.add_parser("brief", help="write the blind-writer brief and the avoid list")
    b_brief.add_argument("--avoid-from", nargs="*", default=[], help="earlier bench files whose names and answers the new bench must avoid")
    b_brief.add_argument("--seed", type=int, required=True)
    b_brief.add_argument("--facts", type=int, default=40)
    b_brief.add_argument("--companies", type=int, default=30)
    b_brief.add_argument("--out", required=True, help="text file for the brief")
    b_brief.add_argument("--avoid-out", required=True, help="JSON file for the avoid list")
    b_brief.add_argument("--kinds", action="store_true", help="a bench of nine kinds of subject (cramschool.bench.kinds), not companies only")
    b_dbrief = bench.add_parser("drafted-brief", help="write the brief for a blind drafter of the wordings GRAFT teaches the kinds' relations from")
    b_dbrief.add_argument("--out", required=True)
    b_dbrief.add_argument("--kinds", nargs="*", default=None, help="the kinds to draft (default: all but company)")
    b_check = bench.add_parser("check", help="check a kinds bench source, or a drafted wordings file, against its brief (no model needed)")
    b_check.add_argument("--source", help="a kinds bench source")
    b_check.add_argument("--drafted", help="a drafted wordings file")
    b_check.add_argument("--avoid-from", nargs="*", default=None, help="earlier bench files (default: every bench file in benchmarks/)")
    b_check.add_argument("--kinds", nargs="*", default=None, help="for --drafted: the kinds it covers")
    b_write = bench.add_parser("write", help="let an editor model write a bench source blind, checked against the brief")
    b_write.add_argument("--editor")
    b_write.add_argument("--avoid", required=True, help="the avoid list from bench brief")
    b_write.add_argument("--seed", type=int, required=True)
    b_write.add_argument("--facts", type=int, default=40)
    b_write.add_argument("--companies", type=int, default=30)
    b_write.add_argument("--out", required=True, help="the source file to write; keep it closed until the method is frozen")
    b_build = bench.add_parser("build", help="build a bench file from a source file")
    b_build.add_argument("--source", required=True)
    b_build.add_argument("--out", required=True)
    b_build.add_argument("--avoid-benches", nargs="*", default=[], help="earlier bench files whose subjects the new companies must stay 3 edits from")
    b_build.add_argument("--kinds", action="store_true", help="the source is a kinds bench source (cramschool.bench.kinds)")
    b_run = bench.add_parser("run", help="run one method on a bench (run base first)")
    _model_args(b_run)
    b_run.add_argument("--bench", required=True, help="the bench file")
    b_run.add_argument("--method", required=True, choices=METHODS)
    b_run.add_argument("--load", nargs="*", default=None, help="GRAFT: bench files whose facts are all loaded (the scored bench is added)")
    b_run.add_argument("--out", help="run folder (default runs/<bench>)")
    b_run.add_argument("--cache", help="GRAFT: folder of built parts, '{bench}' replaced by each bench's name (default runs/_parts/{bench})")
    b_run.add_argument("--layer", type=int, default=20)
    b_run.add_argument("--only", help="score only these fact ids, comma-separated (the bench's context is unchanged)")
    b_rep = bench.add_parser("report", help="passes, medians and paired comparisons of the methods in a run folder")
    b_rep.add_argument("--out", required=True, help="the run folder")
    b_rep.add_argument("--bench", help="the bench file, for the two-fact-company count")

    # ---- serving and watching
    serve = commands.add_parser("serve", help="run the tools as an MCP server (GRAFT and surgery), so Claude Code or another editor can use them")
    serve.add_argument("--model", required=True, help="model path or Hugging Face id; it loads on the first tool call")
    serve.add_argument("--table", help="a GRAFT table folder to load (and the default place to save)")
    serve.add_argument("--calibration", help="text file of ordinary paragraphs, one per line, for the surgery's weight writes (default: the bundled WikiText-103 paragraphs)")
    serve.add_argument("--budget", type=int, default=60, help="forward passes the surgery tools may spend")
    serve.add_argument("--log", help="JSONL file for the hash-chained log of every surgery call")
    serve.add_argument("--log-dir", help="folder that gets a new log for each run, so sessions never mix; the progress window follows the newest")
    serve.add_argument("--watch", action="store_true", help="open the live progress window when the first tool call arrives")
    serve.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    serve.add_argument("--fast", action="store_true", help="a new fact table uses the fast build: later facts of a relation in 5 to 9 seconds")
    serve.add_argument("--transport", default="stdio", choices=("stdio", "http"),
                       help="stdio: for Claude Code, which starts the server itself. http: stay running with the model loaded, on this computer only")
    serve.add_argument("--port", type=int, default=remote.DEFAULT_PORT, help=f"the port for --transport http (default {remote.DEFAULT_PORT})")
    call = commands.add_parser("call", help="call one tool on a running `serve --transport http` server and print the result as JSON")
    remote.add_arguments(call)
    weights = commands.add_parser("weights", help="a history of a model's own weights: record a checkpoint as a version, compare versions, "
                                                   "write one back").add_subparsers(dest="weights_command", required=True)
    w_commit = weights.add_parser("commit", help="record a checkpoint folder (a grafted model) as a version: only what differs from the base is stored")
    w_commit.add_argument("--history", required=True, help="the folder of the weights' history; started on the first commit")
    w_commit.add_argument("--checkpoint", required=True, help="the checkpoint folder to record")
    w_commit.add_argument("--base", required=True, help="the base checkpoint: the local folder, in safetensors format, the model was exported from")
    w_commit.add_argument("-m", "--message", default="", help="what changed, in a few words")
    w_commit.add_argument("--table", help="the GRAFT table a grafted model was exported from: its newest recorded version is written into this one")
    w_log = weights.add_parser("log", help="the recorded versions of the weights, and whether the history is whole")
    w_log.add_argument("--history", required=True)
    w_diff = weights.add_parser("diff", help="the tensors and the facts that differ between two versions; a version may be 'base'")
    w_diff.add_argument("--history", required=True)
    w_diff.add_argument("a", help="a version, such as v1, or base")
    w_diff.add_argument("b", nargs="?", help="a later version (default: the newest)")
    w_out = weights.add_parser("checkout", help="write a version as a checkpoint folder, byte for byte as it was recorded")
    w_out.add_argument("--history", required=True)
    w_out.add_argument("--to", required=True, help="the version to write, such as v1, or base")
    w_out.add_argument("--out", required=True, help="the checkpoint folder to write; one that holds another version is overwritten")
    w_out.add_argument("--base", required=True)
    w_revert = weights.add_parser("revert", help="make a checkpoint folder hold an earlier version again, recorded as a new version")
    w_revert.add_argument("--history", required=True)
    w_revert.add_argument("--to", required=True, help="the version to go back to, such as v1, or base")
    w_revert.add_argument("--checkpoint", required=True)
    w_revert.add_argument("--base", required=True)
    w_revert.add_argument("--discard", action="store_true", help="also lose weights in the folder that no version holds")
    w_ask = weights.add_parser("ask", help="ask questions of several versions of the weights: the model is loaded once and set to each in turn")
    _model_args(w_ask)
    w_ask.add_argument("--history", required=True)
    w_ask.add_argument("--versions", nargs="*", default=None, help="the versions to ask, such as base v1 v2 (default: base and every recorded one)")
    w_ask.add_argument("questions", nargs="+")
    w_ask.add_argument("--plain", action="store_true", help="treat the questions as plain sentence openings, not chat")
    w_bisect = weights.add_parser("bisect", help="find the first version of the weights at which a question stopped getting, or started getting, "
                                                 "a text in its reply (loads the model)")
    _model_args(w_bisect)
    w_bisect.add_argument("--history", required=True)
    w_bisect.add_argument("--good", required=True, help="a version where the reply is right, or base")
    w_bisect.add_argument("--bad", help="a later version where it is wrong (default: the newest)")
    w_bisect.add_argument("--question", required=True)
    w_bisect.add_argument("--expect", required=True, help="the text a right reply holds (case ignored)")
    w_bisect.add_argument("--absent", action="store_true", help="a right reply does not hold the text")
    w_bisect.add_argument("--plain", action="store_true", help="treat the question as a plain sentence opening, not chat")
    watch = commands.add_parser("watch", help="show a surgery's progress live: budget, the current step, kept edits and every step so far")
    watch.add_argument("path", nargs="?", help="a session log, or a folder to follow the newest session in (default ~/.cramschool/sessions)")
    watch.add_argument("--once", action="store_true", help="print the view once and exit")
    watch.add_argument("--interval", type=float, default=1.0, help="seconds between redraws")

    # ---- beta: facts written into the weights
    add = commands.add_parser("add", help="beta: write one fact into the weights with forward passes and report the checks (GRAFT is the v1 way to add facts)")
    add.add_argument("--model", required=True, help="model path or Hugging Face id")
    add.add_argument("--calibration", help="text file of ordinary paragraphs, one per line (default: the bundled WikiText-103 paragraphs)")
    add.add_argument("--subject", required=True)
    add.add_argument("--target", required=True)
    add.add_argument("--relation", required=True, help="owner, ceo, hq or founder")
    add.add_argument("--layer", type=int, default=20, help="the layer whose MLP is written (20 of 24 was measured on Qwen3.5-0.8B)")
    add.add_argument("--null-keys", default="none", choices=NULL_KINDS, help="prompts the edit must leave alone")
    add.add_argument("--strength", type=float, default=1.0, help="scale each answer token's vector before writing it")
    add.add_argument("--margin", type=float, default=2.0, help="how many logits each answer token should lead its strongest rival by")
    add.add_argument("--ridge", type=float, default=0.1, help="how soft the fit is; larger makes null keys a preference rather than a rule")
    add.add_argument("--metric-ridge", type=float, default=0.02, help="how soft the calibration is")
    add.add_argument("--ask", help="question to ask before and after the edit")
    add.add_argument("--save", help="directory for the ledger, so editing can continue later")
    add.add_argument("--keep-failing", action="store_true", help="keep the edit even if it fails its checks")
    return parser


def _load(model_path: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {model_path} on {device}", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype), device_map=device).eval()
    return model, tokenizer


def _say(text: str) -> None:
    print(f"  {text}", file=sys.stderr, flush=True)


def _open_table(model, tokenizer, table: str, layer: int | None = None, relations: str | None = None, args=None):
    """The table in a folder, or a new one. `args`: the command's settings (--last-layer, --prompt-detector, --system, --fast). On a saved
    table --last-layer and --prompt-detector replace its settings; its system message and build are what it was made with."""
    from .graft import Graft, GraftOptions, Relation

    last = getattr(args, "last_layer", None)
    detector = getattr(args, "prompt_detector", None)
    detector = {"on": True, "off": False}.get(detector, detector)  # `graft set` takes on or off; the others a flag
    system = getattr(args, "system", None)
    if os.path.exists(os.path.join(table, "graft.json")):
        graft = Graft.load(table, model, tokenizer, progress=_say)
        if system is not None and system != (graft.system or ""):
            raise SystemExit(f"{table} was built {'behind another system message' if graft.system else 'with no system message'}; a table is for "
                             "the chat format it was built in. Make a new table for this one.")
        if last is not None:
            graft.options.last_layer = None if last == "none" else last
        if detector is True or (detector is False and getattr(args, "graft_command", None) == "set"):
            graft.options.prompt_detector = detector
    else:
        options = GraftOptions(fast=bool(getattr(args, "fast", False)), prompt_detector=bool(detector))
        if last is not None:
            options.last_layer = None if last == "none" else last
        graft = Graft(model, tokenizer, layer=layer, progress=_say, options=options, system=system)
    if relations:
        for rel in Relation.from_file(relations).values():
            graft.add_relation(rel)
    return graft


def _table_meta(table: str) -> dict:
    path = os.path.join(table, "graft.json")
    if not os.path.exists(path):
        raise SystemExit(f"no GRAFT table in {table}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _records(args):
    from .datasets import load_facts

    return load_facts(args.facts, subject=args.subject_column, relation=args.relation_column, answer=args.answer_column,
                      fixed_relation=args.fixed_relation, split=args.split, limit=args.limit)


def run_graft_command(args) -> int:
    if args.graft_command == "list":
        meta = _table_meta(args.table)
        print(f"{len(meta['facts'])} facts, layer {meta['layer']}, model {meta['model']}")
        for f in meta["facts"]:
            print(f"  {f['id']}  {f['subject']} | {f['relation']} | {f['answer']}")
        for subject, names in meta.get("aliases", {}).items():
            print(f"  {subject} is also called: {', '.join(names)}")
        for subject, names in meta.get("exclusions", {}).items():
            print(f"  {subject} is not named by: {', '.join(names)}")
        return 0
    if args.graft_command == "exclude":
        meta = _table_meta(args.table)
        norm = lambda s: " ".join(s.lower().split())  # noqa: E731
        stored = {norm(f["subject"]): f["subject"] for f in meta["facts"]}
        if norm(args.subject) not in stored:
            raise SystemExit(f"no fact is stored about {args.subject!r} in {args.table}")
        name, known = stored[norm(args.subject)], meta.setdefault("exclusions", {})
        own = [norm(name)] + [norm(a) for a in meta.get("aliases", {}).get(name, [])]
        taken = set(stored) | {norm(a) for names in meta.get("aliases", {}).values() for a in names}
        for longer in args.names:
            key = norm(longer)
            if not key or key in taken:
                raise SystemExit(f"{longer!r} is empty, a stored name or a declared alias")
            if not any(re.search(r"(?<![0-9a-z])" + re.escape(o) + r"(?![0-9a-z])", key) for o in own):
                raise SystemExit(f"{longer!r} does not contain {name!r} or one of its aliases as a whole word")
            if key not in {norm(x) for x in known.get(name, [])}:
                known.setdefault(name, []).append(" ".join(longer.split()))
        with open(os.path.join(args.table, "graft.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        print(f"{name} is not named by: {', '.join(known.get(name, []))}")
        return 0
    if args.graft_command == "alias":
        meta = _table_meta(args.table)
        norm = lambda s: " ".join(s.lower().split())  # noqa: E731
        stored = {norm(f["subject"]): f["subject"] for f in meta["facts"]}
        if norm(args.subject) not in stored:
            raise SystemExit(f"no fact is stored about {args.subject!r} in {args.table}")
        name, known = stored[norm(args.subject)], meta.setdefault("aliases", {})
        taken = {**stored, **{norm(a): s for s, names in known.items() for a in names}}
        for alias in args.aliases:
            if taken.get(norm(alias), name) != name:
                raise SystemExit(f"{alias!r} already names {taken[norm(alias)]!r}")
            if norm(alias) and norm(alias) not in taken:
                known.setdefault(name, []).append(" ".join(alias.split()))
                taken[norm(alias)] = name
        with open(os.path.join(args.table, "graft.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        print(f"{name} is also called: {', '.join(known.get(name, []))}")
        return 0
    if args.graft_command == "remove":
        meta = _table_meta(args.table)
        key = " ".join(args.subject.lower().split())
        from .graft.wordings import canonical_relation

        rel = canonical_relation(args.relation) if args.relation else None
        gone = [f for f in meta["facts"] if " ".join(f["subject"].lower().split()) == key and (rel is None or f["relation"] == rel)]
        meta["facts"] = [f for f in meta["facts"] if f not in gone]
        left = {f["subject"] for f in meta["facts"]}
        for key_ in ("aliases", "exclusions"):  # a company with no facts left keeps no declared names (a stale alias would break loading)
            if key_ in meta:
                meta[key_] = {s: a for s, a in meta[key_].items() if s in left}
        for f in gone:
            for suffix in (".pt", ".looser.pt", ".prompt.pt"):
                p = os.path.join(args.table, "parts", f["id"] + suffix)
                if os.path.exists(p):
                    os.remove(p)
        with open(os.path.join(args.table, "graft.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        print(f"removed {len(gone)} fact(s); {len(meta['facts'])} left")
        return 0
    if args.graft_command in ("commit", "log", "diff", "revert"):
        from .graft import history

        try:
            if args.graft_command == "commit":
                before = len(history.versions(args.table))
                version = history.commit(args.table, args.message)
                recorded = len(history.versions(args.table)) > before
                print(f"{version.id}: {version.facts} fact{'' if version.facts == 1 else 's'} in {args.table}" + (f", {version.message}" if version.message else "")
                      if recorded else f"nothing to record: {args.table} is as {version.id} holds it")
            elif args.graft_command == "log":
                found = history.versions(args.table)
                for v in found:
                    print(f"  {v.id}  {v.time}  {v.facts} fact{'' if v.facts == 1 else 's'}  {v.message}")
                print(f"{len(found)} version{'' if len(found) == 1 else 's'} of {args.table}; the history is "
                      f"{'whole' if history.is_intact(args.table) else 'NOT whole: a line or a file was changed or lost'}")
                if found and not history.is_committed(args.table):
                    print(f"the saved table has changes since {found[-1].id}")
            elif args.graft_command == "diff":
                print(history.diff_text(history.diff(args.table, args.a, args.b)))
            else:
                version = history.revert(args.table, args.to, discard=args.discard)
                print(f"{args.table} holds {version.of or args.to} again, recorded as {version.id}: {version.facts} fact{'' if version.facts == 1 else 's'}")
        except ValueError as error:
            raise SystemExit(str(error))
        return 0
    model, tokenizer = _load(args.model, args.dtype)
    if args.graft_command == "bisect":
        from .graft import Graft, history

        def right(version) -> bool:
            table = Graft.load(args.table, model, tokenizer, version=version.id)
            reply = table.ask(args.question, chat=not args.plain)
            table.detach()
            held = args.expect.lower() in reply.lower()
            print(f"  {version.id}: {reply!r}", flush=True)
            return held != args.absent

        try:
            found = history.first_bad(args.table, right, args.good, args.bad)
        except ValueError as error:
            raise SystemExit(str(error))
        print(f"the reply went wrong at {found['first_bad']['id']} ({found['first_bad']['message'] or 'no message'}); "
              f"{found['last_good']['id']} was the last right one\n" + history.diff_text(found["diff"]))
        return 0
    if args.graft_command == "layers":
        from dataclasses import asdict

        from .graft import Relation, Wordings, profile_layers

        wordings = Wordings.builtin()
        if args.relations:
            wordings = wordings.with_relations(Relation.from_file(args.relations))
        profile = profile_layers(model, tokenizer, wordings, progress=_say)
        print(profile.table())
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(asdict(profile), fh, ensure_ascii=False, indent=1)
        return 0
    graft = _open_table(model, tokenizer, args.table, getattr(args, "layer", None), getattr(args, "relations", None), args)
    if args.graft_command == "merge":
        from .graft import Graft

        if bool(args.out) != bool(args.history):
            raise SystemExit("give --out and --history together: the grafted model's folder and the folder of the weights' history")
        try:
            other = Graft.load(args.other, model, tokenizer, progress=_say)
            if args.out:
                done = graft.merge_grafted(other, args.table, args.out, args.history, args.message, prefer=args.prefer, base_dir=args.base or args.model)
                merged, version = done["merged"], done["table"]
            else:
                merged, version = graft.merge(other, args.prefer), None
                version = graft.commit(args.table, args.message)
        except ValueError as error:
            raise SystemExit(str(error))
        print(f"{len(merged['added'])} added, {len(merged['replaced'])} replaced, {len(merged['kept'])} kept as they were, {merged['same']} the same; "
              f"{len(graft)} facts in {args.table}, recorded as {version.id}")
        if args.out:
            print(f"the grafted model of the merged table is in {args.out}; its weights are {done['weights'].id} in {args.history}")
            for f in done["not_in_weights"]:
                print(f"  not in the weights (no detector could be built): {f['subject']} | {f['relation']} | {f['answer']}")
        return 0
    if args.graft_command == "set":
        if not len(graft):
            raise SystemExit(f"{args.table} holds no facts")
        graft.layer_object()  # builds what the settings need: the scale of the last layer's write, the facts' end-of-prompt detectors
        graft.save(args.table)
        print(f"{args.table}: last_layer={graft.options.last_layer!r}, prompt_detector={graft.options.prompt_detector}; {len(graft)} facts")
        return 0
    if args.graft_command == "add":
        if args.ask and len(graft):
            print("Before:", graft.ask(args.ask))
        stored = graft.add(args.subject, args.relation, args.answer)
        graft.save(args.table)
        print(f"added {stored.subject} | {stored.relation} | {stored.answer} in {stored.seconds:.0f}s; {len(graft)} fact{'' if len(graft) == 1 else 's'} in {args.table}")
        if args.ask:
            print("After:", graft.ask(args.ask))
        return 0
    if args.graft_command == "add-file":
        records = _records(args)
        for n, r in enumerate(records, 1):
            stored = graft.add(r.subject, r.relation, r.answer)
            print(f"[{n}/{len(records)}] {stored.subject} | {stored.relation} | {stored.answer} ({stored.seconds:.0f}s)", flush=True)
            graft.save(args.table)
        print(f"{len(graft)} facts in {args.table}")
        return 0
    if args.graft_command == "export":
        if not len(graft):
            raise SystemExit(f"{args.table} holds no facts to compile")
        export = dict(forms=("plain",) if args.plain_only else ("plain", "chat"), complete=not args.first_token_only)
        recorded = None
        if args.history:
            from . import weight_history
            from .graft import history as table_history

            before = (len(table_history.versions(args.table)), len(weight_history.versions(args.history)))
            try:
                recorded = graft.commit_grafted(args.table, args.out, args.history, args.message, base_dir=args.base or args.model, **export)
            except ValueError as error:
                raise SystemExit(str(error))
            report = recorded["report"]
        else:
            report = graft.export_grafted(args.out, base_dir=args.base or args.model, **export)
        changed = sum(v["changed"] for v in report["tensors"].values())
        same_size = all(v["base_bytes"] == v["grafted_bytes"] for v in report["files"].values())
        print(f"{len(graft)} facts compiled into layer {graft.layer}: {changed} values changed in {len(report['tensors'])} tensors; "
              f"files the same size as the base: {same_size}. Written to {args.out}, with grafted.json and the name switch grafted_switch.py")
        if recorded is not None:
            new = (len(table_history.versions(args.table)) > before[0], len(weight_history.versions(args.history)) > before[1])
            said = ["recorded as" if fresh else "unchanged, still" for fresh in new]
            print(f"the table: {said[0]} {recorded['table'].id} in {args.table}; the weights: {said[1]} {recorded['weights'].id} in {args.history}")
            for f in recorded["not_in_weights"]:
                print(f"  not in the weights (no detector could be built): {f['subject']} | {f['relation']} | {f['answer']}")
        return 0
    if args.graft_command == "ask":
        for q in args.questions:
            out = graft.compare(q, chat=not args.plain)
            print(f"Q: {q}\n  without: {out['without']!r}\n  with:    {out['with_graft']!r}")
        return 0
    raise SystemExit(f"unknown graft command {args.graft_command}")


def run_bench_command(args) -> int:
    if args.bench_command == "brief":
        from .bench.brief import brief_text, make_avoid

        avoid = make_avoid(args.avoid_from)
        with open(args.avoid_out, "w", encoding="utf-8") as fh:
            json.dump(avoid, fh, ensure_ascii=False, indent=1)
        with open(args.out, "w", encoding="utf-8") as fh:
            if args.kinds:
                from .bench.kinds import kinds_brief

                fh.write(kinds_brief(avoid, seed=args.seed))
            else:
                fh.write(brief_text(avoid, seed=args.seed, n_facts=args.facts, n_companies=args.companies))
        print(f"brief written to {args.out}; avoid list ({len(avoid['first_words'])} words, {len(avoid['old_targets'])} answers) to {args.avoid_out}")
        return 0
    if args.bench_command == "write":
        from .bench.brief import write_source
        from .editors import editor_from_spec

        with open(args.avoid, encoding="utf-8") as fh:
            avoid = json.load(fh)
        src = write_source(editor_from_spec(args.editor), avoid, seed=args.seed, n_facts=args.facts, n_companies=args.companies)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(src, fh, ensure_ascii=False, indent=1)
        print(f"bench source written to {args.out} and checked against the brief. Keep it closed until the method it judges is frozen.")
        return 0
    if args.bench_command == "drafted-brief":
        from .bench.kinds import drafted_brief

        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(drafted_brief(args.kinds))
        print(f"drafted brief written to {args.out}")
        return 0
    if args.bench_command == "check":
        import glob

        from .bench.brief import make_avoid
        from .bench.kinds import validate_drafted, validate_kinds_source

        if bool(args.source) == bool(args.drafted):
            raise SystemExit("give one of --source or --drafted")
        with open(args.source or args.drafted, encoding="utf-8") as fh:
            spec = json.load(fh)
        if args.source:
            benches = args.avoid_from if args.avoid_from is not None else sorted(p for p in glob.glob("benchmarks/*.json") if not p.endswith("_source.json"))
            problems = validate_kinds_source(spec, make_avoid(benches))
        else:
            problems = validate_drafted(spec, args.kinds)
        if problems:
            print(f"{len(problems)} problems:" + "".join(chr(10) + "- " + x for x in problems[:60]))
            return 1
        print("valid")
        return 0
    if args.bench_command == "build":
        from .bench.build import build_from_source

        if args.kinds:
            from .bench.kinds import build_kinds_from_source

            meta = build_kinds_from_source(args.source, args.out, args.avoid_benches)
            print(f"{meta['facts']} facts on {meta['companies']} subjects ({meta['two_fact_companies']} with two facts), kinds {meta.get('kinds')}, "
                  f"written to {args.out}")
            return 0
        meta = build_from_source(args.source, args.out, args.avoid_benches)
        print(f"{meta['facts']} facts on {meta['companies']} companies ({meta['two_fact_companies']} with two facts) written to {args.out}")
        return 0
    if args.bench_command == "run":
        from .bench.methods import run

        name = os.path.splitext(os.path.basename(args.bench))[0]
        out = args.out or os.path.join("runs", name)
        loads = list(dict.fromkeys((args.load or []) + [args.bench]))
        model, tokenizer = _load(args.model, args.dtype)
        only = set(args.only.split(",")) if args.only else None
        path = run(args.method, model, tokenizer, args.bench, out, load_paths=loads, layer=args.layer, cache_dir=args.cache, only=only)
        print(f"results in {path}")
        return 0
    if args.bench_command == "report":
        from .bench.report import report_text

        print(report_text(args.out, args.bench))
        return 0
    raise SystemExit(f"unknown bench command {args.bench_command}")


def run_teach(args) -> int:
    from .editors import editor_from_spec, run_editor
    from .tools import GraftTools

    editor = editor_from_spec(args.editor)
    model, tokenizer = _load(args.model, args.dtype)
    graft = _open_table(model, tokenizer, args.table)
    task = args.task or ""
    if args.facts:
        from .datasets import load_facts

        facts = load_facts(args.facts)
        task += "\n\nAdd and check these facts:\n" + "\n".join(f"- {f.subject} | {f.relation} | {f.answer}" for f in facts)
    if not task.strip():
        raise SystemExit("give --task, --facts or both")
    task += f"\n\nSave the table to {args.table} when you are done."

    def show(name, inp, out):
        brief = {k: v for k, v in out.items() if k in ("added", "removed", "recall", "leaks_other_questions", "leaks_lookalikes", "good", "error", "facts")}
        print(f"  {name} {json.dumps(inp, ensure_ascii=False)[:140]} -> {json.dumps(brief, ensure_ascii=False, default=str)[:200]}", flush=True)

    run = run_editor(editor, GraftTools(graft), task, max_turns=args.max_turns, on_call=show)
    graft.save(args.table)
    print(f"\n{editor.name} finished after {run.turns} turns ({run.stop_reason}); {len(graft)} facts saved in {args.table}\n\n{run.final_text}")
    return 0


def run_draft(args) -> int:
    from .editors import draft_relation, editor_from_spec

    rel = draft_relation(editor_from_spec(args.editor), args.name, args.description, person=not args.place)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({rel.name: dict(plain=rel.plain, chat=rel.chat, person=rel.person)}, fh, ensure_ascii=False, indent=1)
    print(f"{len(rel.plain)} plain and {len(rel.chat)} chat wordings for {rel.name} written to {args.out}; read them before using them")
    return 0


def run_weights_command(args) -> int:
    """The history of a model's own weights (cramschool.weight_history). File work on checkpoint folders: no model is loaded."""
    from . import weight_history as wh

    plural = lambda n, word: f"{n} {word}{'' if n == 1 else 's'}"  # noqa: E731
    try:
        if args.weights_command == "ask":
            from .gates import generate_batch
            from .model_access import Prompt, render_chat

            model, tokenizer = _load(args.model, args.dtype)
            asked = args.versions or [wh.BASE] + [v.id for v in wh.versions(args.history)]
            prompts = [Prompt(q) if args.plain else render_chat(tokenizer, q) for q in args.questions]
            replies = {}
            for version in asked:
                with wh.applied(args.history, version, model, tokenizer=tokenizer):
                    replies[version] = generate_batch(model, tokenizer, prompts, 48, batch_size=1)
            for i, question in enumerate(args.questions):
                print(f"Q: {question}")
                for version in asked:
                    print(f"  {version + ':':6} {replies[version][i]!r}")
        elif args.weights_command == "bisect":
            from .gates import generate_batch
            from .model_access import Prompt, render_chat

            model, tokenizer = _load(args.model, args.dtype)
            replies = []

            def right(m) -> bool:
                prompt = Prompt(args.question) if args.plain else render_chat(tokenizer, args.question)
                replies.append(generate_batch(m, tokenizer, [prompt], 48, batch_size=1)[0])
                return (args.expect.lower() in replies[-1].lower()) != args.absent

            found = wh.bisect(args.history, model, right, args.good, args.bad, tokenizer=tokenizer)
            for step, reply in zip(found["checked"], replies):
                print(f"  {step['version']}: {reply!r}")
            print(f"the reply went wrong at {found['first_bad']}; {found['last_good']} was the last right one")
            print(wh.diff_text(found["diff"]))
        elif args.weights_command == "commit":
            before = len(wh.versions(args.history))
            version = wh.commit(args.history, args.checkpoint, args.message, base_dir=args.base, table=args.table)
            if len(wh.versions(args.history)) == before:
                print(f"nothing to record: {args.checkpoint} is as {version.id} holds it")
            else:
                print(f"{version.id}: {plural(len(version.tensors), 'tensor')} differ from the base"
                      + (f", {plural(version.facts, 'fact')}" if version.facts is not None else "")
                      + (f", compiled from table version {version.table['version']}" if version.table else "") + (f"; {version.message}" if version.message else ""))
        elif args.weights_command == "log":
            found = wh.versions(args.history)
            for v in found:
                print(f"  {v.id}  {v.time}  {plural(len(v.tensors), 'tensor')}" + (f"  {plural(v.facts, 'fact')}" if v.facts is not None else "")
                      + (f"  table {v.table['version']}" if v.table else "") + f"  {v.message}")
            print(f"{plural(len(found), 'version')} of the weights in {args.history}; the history is "
                  f"{'whole' if wh.is_intact(args.history) else 'NOT whole: a line, a tensor or a file was changed or lost'}")
        elif args.weights_command == "diff":
            print(wh.diff_text(wh.diff(args.history, args.a, args.b)))
        elif args.weights_command == "checkout":
            done = wh.checkout(args.history, args.to, args.out, base_dir=args.base)
            print(f"{args.out} holds {done['version']}: {plural(len(done['tensors']), 'tensor')} in place of the base's"
                  + (f", with {', '.join(done['files'])}" if done["files"] else ""))
        else:
            version = wh.revert(args.history, args.to, args.checkpoint, base_dir=args.base, discard=args.discard)
            print(f"{args.checkpoint} holds {version.of or args.to} again, recorded as {version.id}")
    except ValueError as error:
        raise SystemExit(str(error))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "weights":
        return run_weights_command(args)
    if args.command == "watch":
        from .live import watch as run_viewer

        run_viewer(args.path, once=args.once, interval=args.interval)
        return 0
    if args.command == "call":
        return remote.run(args.url, args.tool, args.args, args.out)
    if args.command == "serve":
        from .mcp_server import serve as run_server

        run_server(args.model, args.calibration, args.budget, args.log, args.dtype, log_dir=args.log_dir, watch=args.watch, transport=args.transport,
                   port=args.port, table=args.table, fast=args.fast)
        return 0
    if args.command == "graft":
        return run_graft_command(args)
    if args.command == "bench":
        return run_bench_command(args)
    if args.command == "teach":
        return run_teach(args)
    if args.command == "draft-relation":
        return run_draft(args)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import EditSettings, Fact, KnowledgeEditor

    texts = read_calibration(args.calibration)
    fact = Fact.from_relation(args.subject, args.target, args.relation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, device_map=device).eval()
    settings = EditSettings(layer=args.layer, null_keys=args.null_keys, strength=args.strength, margin=args.margin)
    editor = KnowledgeEditor(model, tokenizer, calibration_texts=texts, settings=settings, ridge=args.ridge, metric_ridge=args.metric_ridge)
    if args.ask:
        print("Before:", editor.answer(args.ask))
    result = editor.add_fact(fact, on_failure="keep" if args.keep_failing else "reject")
    print(json.dumps(dict(in_weights=result.accepted, seconds=round(result.seconds, 1), **result.report.to_dict()), indent=1))
    if not result.accepted:
        print("Rejected: the weights are unchanged. Use --keep-failing to keep an edit that fails its checks.", file=sys.stderr)
    if args.ask:
        print("After:", editor.answer(args.ask))
    if args.save:
        editor.save(args.save)
        print("Saved to", args.save)
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
