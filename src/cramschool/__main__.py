"""Command line: write one fact into a model and show the result, with a progress bar.

    cramschool add --model D:/data/models/Qwen__Qwen3.5-0.8B --calibration ordinary_text.txt \
        --subject "RX Systems" --target "Julian Rodlauer" --relation owner --ask "Who owns RX Systems?" --save edits/rx_systems

The calibration file is ordinary text with one paragraph per line. A few hundred paragraphs is enough.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import remote  # needs only the standard library until a call is made

# Kept in step with cramschool.facts, and written out here so that --help does not have to import PyTorch.
KEY_SITES = ("subject", "last")
NULL_KINDS = ("none", "same", "other", "both", "all")


def read_calibration(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        texts = [line.strip() for line in fh if line.strip()]
    if not texts:
        raise SystemExit(f"The calibration file {path} has no text in it.")
    return texts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cramschool")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="write one fact and report the checks")
    add.add_argument("--model", required=True, help="model path or Hugging Face id")
    add.add_argument("--calibration", required=True, help="text file of ordinary paragraphs, one per line")
    add.add_argument("--subject", required=True)
    add.add_argument("--target", required=True)
    add.add_argument("--relation", required=True, help="owner, ceo, hq or founder")
    add.add_argument("--layer", type=int, default=8)
    add.add_argument("--key-site", default="subject", choices=KEY_SITES, help="the token the edit keys on")
    add.add_argument("--null-keys", default="none", choices=NULL_KINDS, help="prompts the edit must leave alone")
    add.add_argument("--per-prompt-keys", action="store_true", help="one key per teaching prompt instead of their mean")
    add.add_argument("--through-edit", action="store_true", help="solve the vector the way the finished edit applies it")
    add.add_argument("--strength", type=float, default=1.0, help="scale the solved vector before writing it")
    add.add_argument("--harm-penalty", type=float, default=20.0, help="how strongly the solver avoids changing ordinary text")
    add.add_argument("--l2", type=float, default=0.002, help="how strongly the solver keeps the vector small")
    add.add_argument("--ridge", type=float, default=0.1, help="how soft the fit is; larger makes null keys a preference rather than a rule")
    add.add_argument("--metric-ridge", type=float, default=0.02, help="how soft the calibration is")
    add.add_argument("--ask", help="question to ask before and after the edit")
    add.add_argument("--save", help="directory for the ledger, so editing can continue later")
    add.add_argument("--keep-failing", action="store_true", help="keep the edit even if it fails its checks")
    serve = commands.add_parser("serve", help="run the surgery tools as an MCP server, so Claude Code or another editor can operate on the model")
    serve.add_argument("--model", required=True, help="model path or Hugging Face id; it loads on the first tool call")
    serve.add_argument("--calibration", help="text file of ordinary paragraphs, one per line, used when writing facts")
    serve.add_argument("--budget", type=int, default=60, help="forward passes the editor may spend")
    serve.add_argument("--log", help="JSONL file for the hash-chained log of every tool call")
    serve.add_argument("--log-dir", help="folder that gets a new log for each run, so sessions never mix; the progress window follows the newest")
    serve.add_argument("--watch", action="store_true", help="open the live progress window when the first tool call arrives")
    serve.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    serve.add_argument(
        "--transport", default="stdio", choices=("stdio", "http"),
        help="stdio: for Claude Code, which starts the server itself. http: stay running with the model loaded, on this computer only",
    )
    serve.add_argument("--port", type=int, default=remote.DEFAULT_PORT, help=f"the port for --transport http (default {remote.DEFAULT_PORT})")
    call = commands.add_parser("call", help="call one tool on a running `serve --transport http` server and print the result as JSON")
    remote.add_arguments(call)
    watch = commands.add_parser("watch", help="show a surgery's progress live: budget, the current step, kept edits and every step so far")
    watch.add_argument("path", nargs="?", help="a session log, or a folder to follow the newest session in (default ~/.cramschool/sessions)")
    watch.add_argument("--once", action="store_true", help="print the view once and exit")
    watch.add_argument("--interval", type=float, default=1.0, help="seconds between redraws")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "watch":
        from .live import watch as run_viewer

        run_viewer(args.path, once=args.once, interval=args.interval)
        return 0
    if args.command == "call":
        return remote.run(args.url, args.tool, args.args, args.out)
    if args.command == "serve":
        from .mcp_server import serve as run_server

        run_server(args.model, args.calibration, args.budget, args.log, args.dtype, log_dir=args.log_dir, watch=args.watch, transport=args.transport, port=args.port)
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import EditSettings, Fact, KnowledgeEditor

    texts = read_calibration(args.calibration)
    fact = Fact.from_relation(args.subject, args.target, args.relation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, device_map=device).eval()
    settings = EditSettings(
        layer=args.layer, key_site=args.key_site, null_keys=args.null_keys, per_prompt_keys=args.per_prompt_keys, through_edit=args.through_edit, strength=args.strength,
        harm_penalty=args.harm_penalty, l2=args.l2,
    )
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
