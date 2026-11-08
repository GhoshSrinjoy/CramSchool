"""cramschool's command line.

    cramschool graft add  --model PATH --table my.graft --subject "Nimbresco Software" --relation founder --answer "Tharindu Ekanayake"
    cramschool graft ask  --model PATH --table my.graft "Who founded Nimbresco Software?"
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
import sys

from . import remote  # needs only the standard library until a call is made

# Kept in step with cramschool.facts, and written out here so that --help does not have to import PyTorch.
KEY_SITES = ("subject", "last")
NULL_KINDS = ("none", "same", "other", "both", "all")
METHODS = ("base", "in-context", "block", "graft", "graft-loose")


def read_calibration(path: str) -> list[str]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cramschool", description="New facts for small language models, without training them (GRAFT).")
    commands = parser.add_subparsers(dest="command", required=True)

    # ---- GRAFT tables
    graft = commands.add_parser("graft", help="add, list, remove and ask facts in a GRAFT table").add_subparsers(dest="graft_command", required=True)
    g_add = graft.add_parser("add", help="add a fact (or update it), and save the table")
    _model_args(g_add)
    g_add.add_argument("--table", required=True, help="the table's folder; created if missing")
    g_add.add_argument("--subject", required=True, help="the company, exactly as written")
    g_add.add_argument("--relation", required=True, help="owner, ceo, founder or hq (or one given with --relations)")
    g_add.add_argument("--answer", required=True)
    g_add.add_argument("--relations", help="JSON file of extra relations: {name: {plain: [...], chat: [...]}}")
    g_add.add_argument("--layer", type=int, help="the layer (default 20 on Qwen3.5-0.8B)")
    g_add.add_argument("--ask", help="a question to ask before and after")
    g_file = graft.add_parser("add-file", help="add every fact of a dataset, and save the table")
    _model_args(g_file)
    g_file.add_argument("--table", required=True)
    _dataset_args(g_file)
    g_file.add_argument("--relations", help="JSON file of extra relations")
    g_file.add_argument("--layer", type=int)
    g_list = graft.add_parser("list", help="the facts in a table (no model needed)")
    g_list.add_argument("--table", required=True)
    g_rm = graft.add_parser("remove", help="remove a company's fact for one relation, or all its facts (no model needed)")
    g_rm.add_argument("--table", required=True)
    g_rm.add_argument("--subject", required=True)
    g_rm.add_argument("--relation")
    g_ask = graft.add_parser("ask", help="ask questions with the table attached, and without it")
    _model_args(g_ask)
    g_ask.add_argument("--table", required=True)
    g_ask.add_argument("questions", nargs="+")
    g_ask.add_argument("--plain", action="store_true", help="treat the questions as plain sentence openings, not chat")

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
    serve.add_argument("--calibration", help="text file of ordinary paragraphs, one per line, for the surgery's weight writes")
    serve.add_argument("--budget", type=int, default=60, help="forward passes the surgery tools may spend")
    serve.add_argument("--log", help="JSONL file for the hash-chained log of every surgery call")
    serve.add_argument("--log-dir", help="folder that gets a new log for each run, so sessions never mix; the progress window follows the newest")
    serve.add_argument("--watch", action="store_true", help="open the live progress window when the first tool call arrives")
    serve.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    serve.add_argument("--transport", default="stdio", choices=("stdio", "http"),
                       help="stdio: for Claude Code, which starts the server itself. http: stay running with the model loaded, on this computer only")
    serve.add_argument("--port", type=int, default=remote.DEFAULT_PORT, help=f"the port for --transport http (default {remote.DEFAULT_PORT})")
    call = commands.add_parser("call", help="call one tool on a running `serve --transport http` server and print the result as JSON")
    remote.add_arguments(call)
    watch = commands.add_parser("watch", help="show a surgery's progress live: budget, the current step, kept edits and every step so far")
    watch.add_argument("path", nargs="?", help="a session log, or a folder to follow the newest session in (default ~/.cramschool/sessions)")
    watch.add_argument("--once", action="store_true", help="print the view once and exit")
    watch.add_argument("--interval", type=float, default=1.0, help="seconds between redraws")

    # ---- beta: facts written into the weights
    add = commands.add_parser("add", help="beta: write one fact into the weights and report the checks (GRAFT is the v1 way to add facts)")
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


def _open_table(model, tokenizer, table: str, layer: int | None = None, relations: str | None = None):
    from .graft import Graft, Relation

    graft = Graft.load(table, model, tokenizer, progress=_say) if os.path.exists(os.path.join(table, "graft.json")) else Graft(model, tokenizer, layer=layer, progress=_say)
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
        return 0
    if args.graft_command == "remove":
        meta = _table_meta(args.table)
        key = " ".join(args.subject.lower().split())
        from .graft.wordings import canonical_relation

        rel = canonical_relation(args.relation) if args.relation else None
        gone = [f for f in meta["facts"] if " ".join(f["subject"].lower().split()) == key and (rel is None or f["relation"] == rel)]
        meta["facts"] = [f for f in meta["facts"] if f not in gone]
        for f in gone:
            for suffix in (".pt", ".looser.pt"):
                p = os.path.join(args.table, "parts", f["id"] + suffix)
                if os.path.exists(p):
                    os.remove(p)
        with open(os.path.join(args.table, "graft.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        print(f"removed {len(gone)} fact(s); {len(meta['facts'])} left")
        return 0
    model, tokenizer = _load(args.model, args.dtype)
    graft = _open_table(model, tokenizer, args.table, getattr(args, "layer", None), getattr(args, "relations", None))
    if args.graft_command == "add":
        if args.ask and len(graft):
            print("Before:", graft.ask(args.ask))
        stored = graft.add(args.subject, args.relation, args.answer)
        graft.save(args.table)
        print(f"added {stored.subject} | {stored.relation} | {stored.answer} in {stored.seconds:.0f}s; {len(graft)} facts in {args.table}")
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
    if args.bench_command == "build":
        from .bench.build import build_from_source

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

        run_server(args.model, args.calibration, args.budget, args.log, args.dtype, log_dir=args.log_dir, watch=args.watch, transport=args.transport,
                   port=args.port, table=args.table)
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
