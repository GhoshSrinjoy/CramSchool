"""Step 18 (lab notebook): the veto read as a mean, beyond one relation. Step 16's veto (taught near questions against half of a relation's
own wordings, thresholded on the other half) is fitted for every relation of a bench with near questions, at every site of step 17, and
read at the end of the bench's test prompts. Judged at one reading fixed before the run: the mean over the tokens at layer 11.

Three sets of taught near questions: A, drafted blind per relation (judged); B, the bench's own training near questions, per fact; C, a
control with the kind's generic questions and no near question at all.

  brief   print the brief for the blind writer of arm A's near questions (it never sees a bench)
  check   check a drafted file against the table's wordings, and count what it shares with the bench's near questions
  run     fit the vetoes and read the bench's test prompts (GPU, about 10 minutes)
  report  the reading as the plan fixes it (CPU)

Usage:
  python scripts/diag_mean_veto.py brief
  python scripts/diag_mean_veto.py check <drafted json> <table folder> <bench json>
  python scripts/diag_mean_veto.py run <model> <bench json> <table folder> <drafted json> <out json>
  python scripts/diag_mean_veto.py report <veto json> <step 17's near json> <drafted json> <bench json> <table folder> <saved rows jsonl>
"""
import collections
import hashlib
import json
import os
import re
import sys
import time

N_PLAIN, N_CHAT, N_NAMES, N_GENERIC = 8, 8, 24, 8
JUDGED = ("mean", 11)
ARMS = dict(A="near questions drafted blind, per relation", B="the bench's own training near questions, per fact",
            C="the kind's generic questions, no near question (control)")
STOP = set("a an and are as at be been by can could did do does for from had has have her his how i in is it its me my of on or our she that the "
           "their them they this to was we were what when where which who whom whose why will with would you your about tell know please".split())

BRIEF = """You are drafting data for a method that adds facts to a small language model. A stored fact answers one relation about one
subject, for example "the capital of a country". The method must also learn which questions it must NOT answer with that fact although
they sound close to it. For each relation below, write questions about the same subject that are CLOSE to the relation and are NOT it,
so that the relation's answer would be a wrong answer to them.

Write from your own sense of what people ask. Do not look anything up, do not open or read any file, and do not use any tool.

Output: one JSON object (UTF-8):
{{"near": {{"<relation>": {{"plain": [{plain} strings], "chat": [{chat} strings]}}, ... one entry per relation below ...}}}}

RELATIONS (the kind of subject; what the relation asks; what its answer is)
{relations}

RULES ({{s}} stands for the subject and appears exactly once in every string)
- Each string asks about the same subject for something DIFFERENT from the relation and close to it: a neighbouring role, a neighbouring
  date, a neighbouring place, a neighbouring organisation, a part or the opposite of the same matter. Someone half listening could take it
  for the relation, and the relation's answer would be wrong for it.
- Across a relation's {total} strings, ask for at least 8 different close things. Do not reword one thing {total} times.
- None may ask for the relation itself in other words, and none may ask for another relation listed for the same kind of subject.
- "plain": {plain} sentence beginnings that stop exactly where the answer would come next, with no trailing space.
- "chat": {chat} questions or requests a user might type to a chat assistant.
- Vary structure, length and register: encyclopedic, news, casual, records language. No answer, real or invented, may appear in a string.
- Example of the idea only, for an unrelated relation "the capital of a country": its largest city, the seat of its highest court, its
  former capital, the city where its parliament first met.

Reply with the JSON object only."""


SAMPLE_PLAIN, SAMPLE_CHAT = 8, 4
SAMPLE_BRIEF = """You are writing sample questions for a method that adds facts to a small language model. For each relation below, write
different ways a person might ask for that relation about a subject. Write from your own sense of how people phrase things. Do not look
anything up, do not open or read any file, and do not use any tool.

Output: one JSON object (UTF-8):
{{"own": {{"<relation>": {{"plain": [{plain} strings], "chat": [{chat} strings]}}, ... one entry per relation below ...}}}}

RELATIONS (the kind of subject; what the relation asks; what its answer is)
{relations}

RULES ({{s}} stands for the subject and appears exactly once in every string)
- "plain": {plain} different sentence beginnings that stop exactly where the answer would come next, with no trailing space, reading
  naturally when continued by the answer.
- "chat": {chat} different questions or requests a user might type to a chat assistant to get that answer about {{s}}.
- Vary structure, length and register: encyclopedic, news, casual, records language, a question then its answer, the subject early or
  late in the sentence.
- Every string must clearly ask for its own relation and not another one. No answer, real or invented, may appear in a string.
- Within a relation, no two strings may differ only by punctuation, capitalization or a single small word.

Reply with the JSON object only."""


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sample_brief() -> str:
    from cramschool.bench.kinds import _ANSWER_WORDS, KINDS

    lines = [f'- "{r}" ({spec["noun"]}): {desc}; the answer is {_ANSWER_WORDS[a]}' for spec in KINDS.values() for r, (a, desc) in spec["relations"].items()]
    return SAMPLE_BRIEF.format(relations="\n".join(lines), plain=SAMPLE_PLAIN, chat=SAMPLE_CHAT)


def bench_own(bench: list[dict]) -> dict:
    """{relation: the normalized wordings the bench asks the relation with, training and test}."""
    out = collections.defaultdict(set)
    for f in bench:
        for side in ("train", "test"):
            out[f["relation"]] |= {normalized(t) for t in f[side]["templates_plain"] + f[side]["templates_chat"]}
    return out


def check_sample(argv) -> int:
    sample_path, table_dir, bench_path = argv[:3]
    from cramschool.bench.scoring import load_bench

    with open(sample_path, "rb") as fh:
        raw = fh.read()
    own, rels, asked = json.loads(raw)["own"], table_wordings(table_dir), bench_own(load_bench(bench_path))
    problems, in_bench, in_table = [], 0, 0
    for r in asked:
        if r not in own:
            problems.append(f"{r}: missing")
            continue
        for form, n in (("plain", SAMPLE_PLAIN), ("chat", SAMPLE_CHAT)):
            got = own[r].get(form, [])
            if len(got) != n:
                problems.append(f"{r}.{form}: {len(got)} strings, not {n}")
            for t in got:
                if t.count("{s}") != 1:
                    problems.append(f"{r}.{form}: {{s}} must appear once: {t!r}")
                if form == "plain" and t != t.rstrip():
                    problems.append(f"{r}.plain: trailing space: {t!r}")
        both = own[r].get("plain", []) + own[r].get("chat", [])
        if len({normalized(t) for t in both}) != len(both):
            problems.append(f"{r}: two wordings are the same by the duplicate rule")
        taught = {normalized(t) for t in rels[r]["plain"] + rels[r]["chat"]}
        in_bench += sum(normalized(t) in asked[r] for t in both)
        in_table += sum(normalized(t) in taught for t in both)
    print(f"{sample_path}: sha256 {hashlib.sha256(raw).hexdigest()[:16]}, {len(own)} relations, "
          f"{sum(len(v.get('plain', [])) + len(v.get('chat', [])) for v in own.values())} wordings")
    print("problems:", problems if problems else "none")
    print(f"equal to a wording the bench asks its relation with (left out of the sample): {in_bench}; equal to a wording the table was taught: {in_table}")
    return 1 if problems else 0


def brief() -> str:
    from cramschool.bench.kinds import _ANSWER_WORDS, KINDS

    lines = [f'- "{r}" ({spec["noun"]}): {desc}; the answer is {_ANSWER_WORDS[a]}' for spec in KINDS.values() for r, (a, desc) in spec["relations"].items()]
    return BRIEF.format(relations="\n".join(lines), plain=N_PLAIN, chat=N_CHAT, total=N_PLAIN + N_CHAT)


def normalized(t: str) -> str:
    from cramschool.graft.wordings import normalized as rule

    return rule(t)


def words(template: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", template.lower().replace("{s}", " ")) if w not in STOP and len(w) > 2}


def bench_near(bench: list[dict]) -> dict:
    """{relation: {"train" or "test": set of near templates}} over a bench's facts, the subject written as {s}."""
    out = collections.defaultdict(lambda: dict(train=set(), test=set()))
    for f in bench:
        for side in ("train", "test"):
            for p in f[side].get("near_plain", []) + f[side].get("near_chat", []):
                out[f["relation"]][side].add(p.replace(f["subject"], "{s}"))
    return out


def table_wordings(table_dir: str) -> dict:
    with open(os.path.join(table_dir, "graft.json"), encoding="utf-8") as fh:
        return json.load(fh)["relations"]


def check(argv) -> int:
    drafted_path, table_dir, bench_path = argv[:3]
    from cramschool.bench.scoring import load_bench

    with open(drafted_path, "rb") as fh:
        raw = fh.read()
    near, rels, bench = json.loads(raw)["near"], table_wordings(table_dir), bench_near(load_bench(bench_path))
    problems = []
    for r in bench:
        if r not in near:
            problems.append(f"{r}: missing")
            continue
        for form, n in (("plain", N_PLAIN), ("chat", N_CHAT)):
            got = near[r].get(form, [])
            if len(got) != n:
                problems.append(f"{r}.{form}: {len(got)} strings, not {n}")
            for t in got:
                if t.count("{s}") != 1:
                    problems.append(f"{r}.{form}: {{s}} must appear once: {t!r}")
                if form == "plain" and t != t.rstrip():
                    problems.append(f"{r}.plain: trailing space: {t!r}")
        both = near[r].get("plain", []) + near[r].get("chat", [])
        if len({normalized(t) for t in both}) != len(both):
            problems.append(f"{r}: two near questions are the same by the duplicate rule")
        kind = rels[r].get("subject", "company")
        for other, spec in rels.items():
            if spec.get("subject", "company") == kind:
                own = {normalized(t) for t in spec["plain"] + spec["chat"]}
                problems += [f"{r}: a near question equals a wording of {other}: {t!r}" for t in both if normalized(t) in own]
    print(f"{drafted_path}: sha256 {hashlib.sha256(raw).hexdigest()[:16]}, {len(near)} relations, "
          f"{sum(len(v.get('plain', [])) + len(v.get('chat', [])) for v in near.values())} near questions")
    print("problems:", problems if problems else "none")
    same = close = total = 0
    for r, sides in bench.items():
        mine = [normalized(t) for t in near.get(r, {}).get("plain", []) + near.get(r, {}).get("chat", [])]
        vocab = set().union(*[words(t) for t in near.get(r, {}).get("plain", []) + near.get(r, {}).get("chat", [])]) if r in near else set()
        own = collections.Counter(w for t in rels[r]["plain"] + rels[r]["chat"] for w in words(t))
        common = {w for w, c in own.items() if c >= 0.25 * (len(rels[r]["plain"]) + len(rels[r]["chat"]))}
        for t in sides["test"] | sides["train"]:
            total += 1
            same += normalized(t) in mine
            close += bool((words(t) - common) & (vocab - common))
    print(f"the bench's {total} near questions (8 per relation): {same} equal a drafted one by the duplicate rule; {close} share a distinctive "
          f"word with a drafted one of their relation")
    return 1 if problems else 0


# ---------------------------------------------------------------- the run

def run(argv) -> None:
    import torch

    import diag_question_site as Q
    from cramschool.bench.scoring import load_bench, test_prompts
    from cramschool.graft import build as B
    from cramschool.graft.core import lda, ratio

    model_dir, bench_path, table_dir, drafted_path, out_path = argv[:5]
    with open(drafted_path, encoding="utf-8") as fh:
        drafted = json.load(fh)["near"]
    bench = {f["subject"] + "|" + f["relation"]: f for f in load_bench(bench_path)}
    tok, model, graft, calib = Q.load(model_dir, table_dir)
    store = Q.Store(model, tok)
    sites = [(k, li) for k in ("last", "mean") for li in range(len(Q.LAYERS))]

    def ends(items):
        return store.ends(B.guarded(items, set()), "\x00")

    def by_form(plain, chat, names):
        """The readings of a set of wordings, by detector form: plain openings and questions in Q/A form; chat questions."""
        return dict(plain=ends(B.plain_items(plain, names) + B.qa_items(chat, names)), chat=ends(B.chat_items(tok, chat, names)))

    def fit(pos, own_fit, own_held):
        """Step 16's veto at every site and form: (w, tau, delta), or None where the positives' median doesn't clear the threshold."""
        out = {}
        for kind, li in sites:
            cov = calib["cov"][Q.KIND[kind], li].float()
            c, cbar = calib["c"][Q.KIND[kind], li], float(calib["cbar"][Q.KIND[kind], li])
            for form in ("plain", "chat"):
                p, a, h = (Q.site_x(x[form], kind, li) for x in (pos, own_fit, own_held))
                w = lda(p, [a], cov)
                tau = float(torch.quantile(ratio(h, w, c, cbar), Q.NEG_Q))
                med = float(ratio(p, w, c, cbar).median())
                out[(kind, li, form)] = (w, tau, Q.SAT * (med - tau)) if med > tau else None
        return out

    def read(veto, x_all):
        """The veto's best score over its two forms at each prompt, per site: [site][prompt]."""
        rows = []
        for kind, li in sites:
            x = Q.site_x(x_all, kind, li)
            c, cbar = calib["c"][Q.KIND[kind], li], float(calib["cbar"][Q.KIND[kind], li])
            per = [(ratio(x, v[0], c, cbar) - v[1]) / v[2] if v is not None else torch.full((x.shape[0],), -1e9, device=x.device)
                   for v in (veto[(kind, li, form)] for form in ("plain", "chat"))]
            rows.append([round(float(z), 4) for z in torch.stack(per).max(0).values])
        return rows

    from cramschool.model_access import Prompt, render_chat

    with_train = "--train" in argv
    sample = None  # another writer's sample of each relation's own questions, in place of the bench's training wordings
    if "--sample" in argv:
        with open(argv[argv.index("--sample") + 1], encoding="utf-8") as fh:
            sample = json.load(fh)["own"]
        asked = bench_own(list(bench.values()))
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None  # relations, for a smoke run
    entries = sorted((e for e in graft._facts if not only or e["fact"]["relation"] in only), key=lambda e: e["fact"]["relation"])
    rows, usable = [], {}
    for relation in sorted({e["fact"]["relation"] for e in entries}):
        t0 = time.time()
        mine = [e for e in entries if e["fact"]["relation"] == relation]
        ctx = Q.context_of(graft, mine[0]["fact"])
        inv, (g_plain, g_chat), _, _ = B.kind_parts(ctx, relation)
        names = inv[:N_NAMES] + B.case_names(inv[:B.N_CASE])
        own = graft.wordings.relations[relation]
        own_fit, own_held = by_form(own.plain[0::2], own.chat[0::2], names), by_form(own.plain[1::2], own.chat[1::2], names)
        vetoes = dict(A=fit(by_form(drafted[relation]["plain"], drafted[relation]["chat"], names), own_fit, own_held),
                      C=fit(by_form(list(g_plain[:N_GENERIC]), list(g_chat[:N_GENERIC]), names), own_fit, own_held))
        usable[relation] = {arm: sum(v is not None for v in vetoes[arm].values()) for arm in vetoes}
        for e in mine:
            f = e["fact"]
            bf = bench[f["subject"] + "|" + f["relation"]]
            to_template = lambda ps: [p.replace(bf["subject"], "{s}") for p in ps]  # noqa: E731
            v_b = fit(by_form(to_template(bf["train"]["near_plain"]), to_template(bf["train"]["near_chat"]), names), own_fit, own_held)
            prompts = test_prompts(bf, tok)
            x = {g: store.ends(prompts[g], "\x00") for g in ("recall", "same", "near") if prompts[g]}
            if with_train:  # the revision's sample: the fact's training wordings in the bench, read after everything else
                x["train"] = store.ends([Prompt(p) for p in bf["train"]["plain"]] + [render_chat(tok, q) for q in bf["train"]["chat"]], "\x00")
            if sample is not None:  # the writer check's sample: another writer's wordings, minus any the bench also asks, about this subject
                keep = lambda ts: [t.format(s=bf["subject"]) for t in ts if normalized(t) not in asked[relation]]  # noqa: E731
                x["train"] = store.ends([Prompt(p) for p in keep(sample[relation]["plain"])] + [render_chat(tok, q) for q in keep(sample[relation]["chat"])],
                                        "\x00")
            rows.append(dict(id=bf["id"], relation=relation, veto={arm: {g: read(v, xs) for g, xs in x.items()}
                                                                   for arm, v in (("A", vetoes["A"]), ("B", v_b), ("C", vetoes["C"]))}))
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(dict(mode="mean_veto", bench=bench_path, table=table_dir, drafted=drafted_path, sites=[[k, Q.LAYERS[li]] for k, li in sites],
                           sample=argv[argv.index("--sample") + 1] if sample is not None else ("the bench's training wordings" if with_train else None),
                           usable=usable, rows=rows), fh)
        log(f"{relation:<19} {len(mine)} facts | usable forms of 84: A {usable[relation]['A']}, C {usable[relation]['C']} | {time.time() - t0:.0f}s "
            f"| {len(rows)}/{len(entries)} facts done")
        store.rows, store.replies, store.bytes, store.transient = {}, {}, 0, set()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- the report

def report(argv) -> None:
    from cramschool.bench.scoring import PASS, load_bench, read_rows

    veto_path, near_path, drafted_path, bench_path, table_dir, rows_path = argv[:6]
    with open(veto_path, encoding="utf-8") as fh:
        d = json.load(fh)
    with open(near_path, encoding="utf-8") as fh:
        n17 = json.load(fh)
    with open(drafted_path, encoding="utf-8") as fh:
        drafted = json.load(fh)["near"]
    sites = [tuple(s) for s in d["sites"]]
    r20 = [tuple(r) for r in n17["readings"]].index(("last", "R1", 20))
    stored = {r["id"]: r for r in n17["rows"]}
    bench = {f["id"]: f for f in load_bench(bench_path)}
    rels = table_wordings(table_dir)
    saved = read_rows(rows_path)
    by_target = {f["target"]: f for f in bench.values()}

    # each test near question: taught by arm A (equal to a drafted one), and whether it shares a distinctive word with a drafted one
    info = {}
    for fid, f in bench.items():
        r = f["relation"]
        mine = drafted.get(r, dict(plain=[], chat=[]))
        equal = {normalized(t) for t in mine["plain"] + mine["chat"]}
        own = collections.Counter(w for t in rels[r]["plain"] + rels[r]["chat"] for w in words(t))
        common = {w for w, c in own.items() if c >= 0.25 * (len(rels[r]["plain"]) + len(rels[r]["chat"]))}
        vocab = set().union(*[words(t) for t in mine["plain"] + mine["chat"]]) - common if mine["plain"] else set()
        tests = [p.replace(f["subject"], "{s}") for p in f["test"].get("near_plain", []) + f["test"].get("near_chat", [])]
        info[fid] = [dict(taught=normalized(t) in equal, close=bool((words(t) - common) & vocab), template=t) for t in tests]

    def counts(arm, site, keep=lambda fid, i: True):
        """(leaks, leaks left, answers, answers kept) for an arm's veto at a site; `keep` chooses the near questions counted."""
        s = sites.index(site)
        leaks = left = answers = kept = 0
        for row in d["rows"]:
            st, v = stored[row["id"]], row["veto"][arm]
            for i, hit in enumerate(st["hit"].get("near", [])):
                if hit and keep(row["id"], i):
                    leaks += 1
                    left += not (st["z"]["near"][r20][i] > 0 and v["near"][s][i] > 0)
            for i, hit in enumerate(st["hit"]["recall"]):
                if hit:
                    answers += 1
                    kept += not (st["z"]["recall"][r20][i] > 0 and v["recall"][s][i] > 0)
        return leaks, left, answers, kept

    untaught = lambda fid, i: not info[fid][i]["taught"]  # noqa: E731
    n_taught = sum(x["taught"] for v in info.values() for x in v)
    print(f"{len(d['rows'])} facts, {sum(len(v) for v in info.values())} test near questions; {n_taught} equal a drafted one and are left out of arm A's "
          f"judged count; {sum(x['close'] and not x['taught'] for v in info.values() for x in v)} more share a distinctive word with a drafted one")
    print("usable veto forms (of 84 per relation), fewest over the relations:", {arm: min(u[arm] for u in d["usable"].values()) for arm in ("A", "C")})
    leaks, left, answers, kept = counts("A", JUDGED, untaught)
    ok = left <= leaks / 2 and kept >= 0.9 * answers
    print(f"\nTHE RULE: arm A at the {JUDGED[0]} of layer {JUDGED[1]}, on the untaught near questions: {left} of {leaks} leaks left (bar: at most "
          f"{leaks // 2}); {kept} of {answers} answers kept (bar: at least {-(-answers * 9 // 10)})")
    print("  ->", "THE VETO HOLDS beyond one relation" if ok else "KILL RULE MET: the mean veto is put aside")
    for arm in ARMS:
        print(f"\narm {arm} ({ARMS[arm]}): leaks left / answers kept, of {counts(arm, JUDGED)[0]} leaks over all near questions and {answers} answers")
        for kind in ("mean", "last"):
            print(f"  {kind}: " + "  ".join(f"{layer}:{c[1]}/{c[3]}" for layer in range(21) for c in [counts(arm, (kind, layer))]))
    print("\narm A at the judged reading, by what the test question shares with the drafted ones (leaks left of leaks):")
    for name, keep in (("equal to a drafted one", lambda fid, i: info[fid][i]["taught"]),
                       ("shares a distinctive word", lambda fid, i: info[fid][i]["close"] and not info[fid][i]["taught"]),
                       ("shares none", lambda fid, i: not info[fid][i]["close"] and not info[fid][i]["taught"])):
        c = counts("A", JUDGED, keep)
        print(f"  {name:<26} {c[1]:3d} of {c[0]}")
    s = sites.index(JUDGED)
    for arm in ARMS:
        by_rel = collections.defaultdict(lambda: [0, 0, 0, 0])
        for row in d["rows"]:
            st, v = stored[row["id"]], row["veto"][arm]
            t = by_rel[row["relation"]]
            for i, hit in enumerate(st["hit"].get("near", [])):
                if hit:
                    t[0] += 1
                    t[1] += not (st["z"]["near"][r20][i] > 0 and v["near"][s][i] > 0)
            for i, hit in enumerate(st["hit"]["recall"]):
                if hit:
                    t[2] += 1
                    t[3] += not (st["z"]["recall"][r20][i] > 0 and v["recall"][s][i] > 0)
        print(f"\narm {arm} at the judged reading, by relation (leaks left of leaks; answers kept of answers):")
        print("  " + "  ".join(f"{r} {t[1]}/{t[0]};{t[3]}/{t[2]}" for r, t in sorted(by_rel.items())))

    # facts passing with near questions held to the same-subject bar, counted with the fixed cross check (kinds_bench_recount.py's rule)
    def passes(row, arm, cut=0.0, at=None):
        """cut: the veto fires above this score (0 is the fitted threshold); at: the site's index (default: the judged one)."""
        st, r, f = stored[row["id"]], saved[row["id"]], bench[row["id"]]
        k = s if at is None else at
        real = sum(bool([t for t in w["targets"] if not (by_target[t].get("answer_type") == "year" and by_target[t]["subject"] != f["subject"])])
                   for w in r["cross_where"] if w["group"] != "near")
        others = [x for x in r["failed"] if not x.startswith(("cross", "known", "recall"))]
        fired = (lambda g, i: st["z"][g][r20][i] > 0 and row["veto"][arm][g][k][i] > cut) if arm else (lambda g, i: False)
        recall = sum(hit and not fired("recall", i) for i, hit in enumerate(st["hit"]["recall"])) / len(st["hit"]["recall"])
        near_hits = st["hit"].get("near", [])
        near = sum(hit and not fired("near", i) for i, hit in enumerate(near_hits)) / len(near_hits) if near_hits else 0.0
        return bool(not others and real == 0 and recall >= PASS["recall"] - 1e-9), bool(near <= PASS["same"] + 1e-9)

    print("\nfacts passing of", len(d["rows"]), "(every check but near questions; and with near questions held to the same-subject bar):")
    for arm in (None,) + tuple(ARMS):
        got = [passes(row, arm) for row in d["rows"]]
        print(f"  {'no veto' if arm is None else 'arm ' + arm:<8} {sum(a for a, _ in got):3d}; {sum(a and b for a, b in got):3d}")
    # After the fact: would another threshold have met the bars? The veto's score at each leak and each answer (very low where the fact's
    # detector was shut at the prompt's end, since the veto changes nothing there), then one cut for all relations, and a cut per
    # relation, each chosen with hindsight to keep 90% of the answers. Upper bounds, not results.
    def scores_at(arm, site):
        k, by_rel = sites.index(site), collections.defaultdict(lambda: ([], []))
        for row in d["rows"]:
            st, v = stored[row["id"]], row["veto"][arm]
            leak_z, answer_z = by_rel[row["relation"]]
            for i, hit in enumerate(st["hit"].get("near", [])):
                if hit and not (arm == "A" and info[row["id"]][i]["taught"]):
                    leak_z.append(v["near"][k][i] if st["z"]["near"][r20][i] > 0 else -1e9)
            for i, hit in enumerate(st["hit"]["recall"]):
                if hit:
                    answer_z.append(v["recall"][k][i] if st["z"]["recall"][r20][i] > 0 else -1e9)
        return by_rel

    def the_cut(leak_z, answer_z):
        need = -(-len(answer_z) * 9 // 10)
        cut = next(c for c in sorted(set(answer_z + leak_z + [0.0])) if sum(a <= c for a in answer_z) >= need)
        return max(cut, 0.0) if sum(a <= 0 for a in answer_z) >= need else cut  # never looser than the fitted threshold where that keeps 90%

    def cut_keeping(leak_z, answer_z):
        cut = the_cut(leak_z, answer_z)
        return sum(z <= cut for z in leak_z), sum(a <= cut for a in answer_z)

    print("\nafter the fact, with hindsight (upper bounds): leaks left and answers kept with one cut for all relations, and with a cut per "
          "relation, each set to keep 90% of the answers; then the relations left with at most half their leaks")
    for arm in ARMS:
        for site in (("mean", 9), ("mean", 10), ("mean", 11), ("mean", 12), ("last", 20)):
            by_rel = scores_at(arm, site)
            every = cut_keeping([z for lz, _ in by_rel.values() for z in lz], [a for _, az in by_rel.values() for a in az])
            each = [(cut_keeping(lz, az), len(lz)) for lz, az in by_rel.values()]
            n_leaks = sum(n for _, n in each)
            cuts = {rel: the_cut(lz, az) for rel, (lz, az) in by_rel.items()}
            got = [passes(row, arm, cuts[row["relation"]], sites.index(site)) for row in d["rows"]]
            print(f"  arm {arm}, {site[0]} {site[1]:2d}: one cut {every[0]}/{n_leaks} left, {every[1]} kept | a cut per relation "
                  f"{sum(c[0] for c, _ in each)}/{n_leaks} left, {sum(c[1] for c, _ in each)} kept, {sum(c[0] <= n / 2 for c, n in each)} of {len(each)} relations; "
                  f"facts passing {sum(a for a, _ in got)}, with near questions judged {sum(a and b for a, b in got)}")
    if "train" in d["rows"][0]["veto"]["A"]:
        revision(d, argv, sites, stored, r20, passes, untaught)

        # After the fact: the revision's veto beside step 17's end-of-prompt detector (layer 16, last, R2), which opens more recall prompts
        # and more near questions. Estimated as in step 17: a newly opened recall prompt answers at the bench's rate, a newly opened
        # question about another relation fails the fact, a newly opened near question counts as a leak, and the veto shuts what it fires on.
        r16 = [tuple(r) for r in n17["readings"]].index(("last", "R2", 16))
        open_now = [(x, i) for x in n17["rows"] for i, z in enumerate(x["z"]["recall"][r20]) if z > 0]
        rate = sum(x["hit"]["recall"][i] for x, i in open_now) / len(open_now)

        def estimate(row, veto: bool, wider: bool):
            st, r, f = stored[row["id"]], saved[row["id"]], bench[row["id"]]
            v = row["veto"]["A"]
            cut = RULES["largest"](v["train"][s])
            was = lambda g, i: st["z"][g][r20][i] > 0  # noqa: E731
            now = lambda g, i: was(g, i) or (wider and st["z"][g][r16][i] > 0)  # noqa: E731
            shut = lambda g, i: veto and v[g][s][i] > cut  # noqa: E731
            real = sum(bool([t for t in w["targets"] if not (by_target[t].get("answer_type") == "year" and by_target[t]["subject"] != f["subject"])])
                       for w in r["cross_where"] if w["group"] != "near")
            others = [x for x in r["failed"] if not x.startswith(("cross", "known", "recall"))]
            exp = sum((0.0 if (was("recall", i) and shut("recall", i)) else 1.0) if hit else
                      (rate if not was("recall", i) and now("recall", i) and not shut("recall", i) else 0.0)
                      for i, hit in enumerate(st["hit"]["recall"]))
            new_same = sum(not was("same", i) and now("same", i) and not shut("same", i) for i in range(len(st["z"].get("same", [[]] * 99)[r20])))
            near_hits = st["hit"].get("near", [])
            leaks = sum((hit and not (was("near", i) and shut("near", i))) or (not hit and not was("near", i) and now("near", i) and not shut("near", i))
                        for i, hit in enumerate(near_hits))
            ok = not others and real == 0 and exp / len(st["hit"]["recall"]) >= PASS["recall"] - 1e-9 and new_same == 0
            return bool(ok), bool(not near_hits or leaks / len(near_hits) <= PASS["same"] + 1e-9)

        print(f"\nafter the fact: the veto of the revision beside step 17's end-of-prompt detector (opened prompts answer at {rate:.3f}). Estimated facts "
              f"passing of {len(d['rows'])}: every check but near questions; with near questions held to the same-subject bar")
        for wider in (False, True):
            for veto in (False, True):
                got = [estimate(row, veto, wider) for row in d["rows"]]
                print(f"  {'with the end-of-prompt detector' if wider else 'the detector as built':<32} {'with the veto' if veto else 'no veto':<14} "
                      f"{sum(a for a, _ in got):3d}; {sum(a and b for a, b in got):3d}")
    for arm in ARMS:
        fire = sum(z > 0 for row in d["rows"] for z in row["veto"][arm].get("same", [[]] * len(sites))[s])
        total = sum(len(row["veto"][arm].get("same", [[]] * len(sites))[s]) for row in d["rows"])
        print(f"arm {arm}: the veto fires on {fire} of {total} same-subject questions (harmless: they are not to be answered)")


RULES = dict(largest=lambda z: sorted(z)[-1], second=lambda z: sorted(z)[-2], floored=lambda z: max(0.0, sorted(z)[-1]))


def revision(d, argv, sites, stored, r20, passes, untaught) -> None:
    """Step 18's revision (lab notebook): a fact's veto fires only above every score it gives to the fact's own training wordings in the
    bench ("largest"). Reported beside it: above all but one ("second"), and never under the fitted threshold ("floored")."""
    sizes = sorted({len(row["veto"]["A"]["train"][0]) for row in d["rows"]})
    print(f"\nTHE THRESHOLD FROM A SAMPLE: each fact's veto fires only above every score it gives to a sample of the relation's own questions "
          f"about its subject. The sample: {d.get('sample') or 'the bench training wordings'}; wordings per fact: {sizes}")
    if len(argv) > 6:
        with open(argv[6], encoding="utf-8") as fh:
            first = {r["id"]: r for r in json.load(fh)["rows"]}
        diff = max(abs(a - b) for row in d["rows"] for arm in ARMS for g in ("recall", "same", "near") if g in row["veto"][arm]
                   for za, zb in zip(row["veto"][arm][g], first[row["id"]]["veto"][arm][g]) for a, b in zip(za, zb))
        print(f"gate: the largest difference from step 18's stored scores at the test prompts is {diff:g}")

    def revised(arm, site, rule, keep=lambda fid, i: True, relation=None):
        k = sites.index(site)
        t = collections.Counter()
        for row in d["rows"]:
            if relation and row["relation"] != relation:
                continue
            st, v = stored[row["id"]], row["veto"][arm]
            cut = RULES[rule](v["train"][k])
            for i, hit in enumerate(st["hit"].get("near", [])):
                if hit and keep(row["id"], i):
                    t["leaks"] += 1
                    t["left"] += not (st["z"]["near"][r20][i] > 0 and v["near"][k][i] > cut)
            for i, hit in enumerate(st["hit"]["recall"]):
                t["asked"] += 1
                t["fires"] += v["recall"][k][i] > cut
                if hit:
                    t["answers"] += 1
                    t["kept"] += not (st["z"]["recall"][r20][i] > 0 and v["recall"][k][i] > cut)
            ok, near_ok = passes(row, arm, cut, k)
            t["facts"] += ok
            t["facts_near"] += ok and near_ok
        return t

    t = revised("A", JUDGED, "largest", untaught)
    ok = t["left"] <= t["leaks"] / 2 and t["kept"] >= 0.9 * t["answers"]
    print(f"THE RULE: arm A at the {JUDGED[0]} of layer {JUDGED[1]}, on the untaught near questions: {t['left']} of {t['leaks']} leaks left (bar: at "
          f"most {t['leaks'] // 2}); {t['kept']} of {t['answers']} answers kept (bar: at least {-(-t['answers'] * 9 // 10)})")
    print("  ->", "THE VETO HOLDS with this threshold" if ok else "KILL RULE MET: the mean veto is put aside on this model")
    stated = sum(1 / (len(row["veto"]["A"]["train"][0]) + 1) for row in d["rows"]) / len(d["rows"])
    print(f"  the veto fires on {t['fires']} of {t['asked']} test recall prompts = {t['fires'] / t['asked']:.3f} (stated, one in the sample's "
          f"size plus one: {stated:.3f})")
    print(f"  facts passing every check but near questions {t['facts']}; with near questions held to the same-subject bar {t['facts_near']} "
          f"(no veto: 75 and 4; the veto as fitted: 37 and 27)")
    print("\nreported: leaks left of leaks / answers kept of answers | fires on test recall prompts | facts; facts with near questions judged")
    for arm in ARMS:
        for rule in RULES:
            for site in (("mean", 9), ("mean", 10), ("mean", 11), ("mean", 12), ("last", 20)):
                t = revised(arm, site, rule, untaught if arm == "A" else (lambda fid, i: True))
                print(f"  arm {arm}, {rule:<8} {site[0]} {site[1]:2d}: {t['left']:3d} of {t['leaks']} / {t['kept']:4d} of {t['answers']} | "
                      f"{t['fires'] / t['asked']:.3f} | {t['facts']:2d}; {t['facts_near']:2d}")
    # after the fact: does the pooled leaks bar rest on single relations? Each relation left out in turn.
    every = revised("A", JUDGED, "largest", untaught)
    per = {r: revised("A", JUDGED, "largest", untaught, r) for r in sorted({row["relation"] for row in d["rows"]})}
    fails = [f"{r} ({every['left'] - t['left']} of {every['leaks'] - t['leaks']})" for r, t in per.items()
             if every["left"] - t["left"] > (every["leaks"] - t["leaks"]) / 2]
    print(f"\nafter the fact: the leaks bar with one relation left out fails without: {fails if fails else 'none'}; relations with at most half "
          f"of their own leaks left: {sum(t['left'] <= t['leaks'] / 2 for t in per.values())} of {len(per)}")
    print("\narm A at the judged reading with the revision's threshold, by relation (leaks left of leaks; answers kept of answers):")
    print("  " + "  ".join(f"{r} {t['left']}/{t['leaks']};{t['kept']}/{t['answers']}"
                           for r in sorted({row["relation"] for row in d["rows"]}) for t in [revised("A", JUDGED, "largest", untaught, r)]))


if __name__ == "__main__":
    if sys.argv[1] == "brief":
        print(brief())
    elif sys.argv[1] == "sample-brief":
        print(sample_brief())
    elif sys.argv[1] == "check":
        sys.exit(check(sys.argv[2:]))
    elif sys.argv[1] == "check-sample":
        sys.exit(check_sample(sys.argv[2:]))
    else:
        {"run": run, "report": report}[sys.argv[1]](sys.argv[2:])
