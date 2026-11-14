"""Loading your own facts, building and checking benches, the report's statistics, and the GRAFT toolbox, all on the CPU."""
import json
import random

import pytest

from cramschool.bench.brief import brief_text, make_avoid, validate_source
from cramschool.bench.build import build_bench, check_source
from cramschool.bench.report import binom_two_sided, paired, summary, wilcoxon_p
from cramschool.bench.scoring import failures
from cramschool.datasets import FactRecord, load_facts, save_facts
from cramschool.tools import GraftTools


def test_facts_from_csv_jsonl_and_bench_sources(tmp_path):
    csv_path = tmp_path / "f.csv"
    csv_path.write_text("company,rel,value\nNimbresco Software,Chief Executive,Anahit Melkonyan\nNimbresco Software,founder,Tharindu Ekanayake\n"
                        "Nimbresco Software,CEO,Someone Else\n,owner,Nobody\n", encoding="utf-8")
    recs = load_facts(str(csv_path), subject="company", relation="rel", answer="value")
    assert recs == [FactRecord("Nimbresco Software", "ceo", "Anahit Melkonyan"), FactRecord("Nimbresco Software", "founder", "Tharindu Ekanayake")]
    jl = tmp_path / "f.jsonl"
    save_facts(recs, str(jl))
    assert load_facts(str(jl)) == recs
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"facts": [["Moqueldar Chemicals", "headquarters", "Oradea"]]}), encoding="utf-8")
    assert load_facts(str(src)) == [FactRecord("Moqueldar Chemicals", "hq", "Oradea")]
    assert load_facts(str(csv_path), subject="company", relation="rel", answer="value", fixed_relation="owner", limit=1)[0].relation == "owner"


def synthetic_source(seed=7):
    rng = random.Random(seed)
    words = ["Velkaron", "Dushmirel", "Pozzamund", "Quartesh", "Brindavol", "Sorquemin"]
    inds = ["Software", "Marine", "Foods", "Steel", "Media", "Optics"]
    companies = [f"{w} {i}" for w, i in zip(words, inds)]
    facts = [[companies[0], "owner", "Amira Haddad"], [companies[0], "ceo", "Bruno Tavares"], [companies[1], "founder", "Chidi Okafor"],
             [companies[1], "hq", "Debrecen"], [companies[2], "owner", "Elif Kaya"], [companies[3], "ceo", "Farah Siddiqui"],
             [companies[4], "founder", "Goran Petrovic"], [companies[5], "hq", "Tartu"]]
    rng.shuffle(facts)
    wordings = {r: dict(plain=[f"Record {i} says the {r} of {{s}} is" for i in range(16)], chat=[f"Question {i}: who is the {r} of {{s}}?" for i in range(8)])
                for r in ("owner", "ceo", "founder", "hq")}
    return dict(seed=seed, facts=facts, wordings=wordings, generic=dict(plain=[f"The size of {{s}} in year {i} is" for i in range(4)],
                                                                        chat=[f"What does {{s}} sell ({i})?" for i in range(4)]))


def test_a_bench_builds_from_a_source():
    src = synthetic_source()
    check_source(src)
    bench, meta = build_bench([tuple(x) for x in src["facts"]], src["wordings"], src["generic"], src["seed"])
    assert meta["facts"] == 8 and meta["companies"] == 6 and meta["two_fact_companies"] == 2
    f = bench[0]
    assert len(f["train"]["plain"]) == 8 and len(f["test"]["plain"]) == 8 and len(f["train"]["chat"]) == 4 and len(f["test"]["chat"]) == 4
    assert len(f["test"]["lookalikes"]) == 3 and len(f["test"]["others"]) == 3 and len(f["test"]["same_plain"]) == 5
    assert not set(f["train"]["plain"]) & set(f["test"]["plain"])
    siblings = [g for g in bench if g["sibling"]]
    assert len(siblings) == 4 and all(next(h for h in bench if h["id"] == g["sibling"])["subject"] == g["subject"] for g in siblings)
    again, _ = build_bench([tuple(x) for x in src["facts"]], src["wordings"], src["generic"], src["seed"])
    assert again == bench


def test_blind_sources_are_checked_against_the_brief(tmp_path):
    src = synthetic_source()
    bench, meta = build_bench([tuple(x) for x in src["facts"]], src["wordings"], src["generic"], src["seed"])
    path = tmp_path / "old.json"
    path.write_text(json.dumps(dict(meta=meta, facts=bench)), encoding="utf-8")
    avoid = make_avoid([str(path)])
    assert "Velkaron" in avoid["first_words"] and "Debrecen" in avoid["old_targets"] and "Software" in avoid["industries"]
    assert "exactly 8 entries" in brief_text(avoid, seed=1, n_facts=8, n_companies=6)
    problems = validate_source(src, avoid, n_facts=8, n_companies=6)
    assert any("fewer than 3 edits" in p for p in problems) and any("reuses an earlier name" in p for p in problems)
    bad = dict(src, wordings=dict(src["wordings"], owner=dict(src["wordings"]["owner"], plain=["{s} {s}"] * 16)))
    assert validate_source(bad, avoid, n_facts=8, n_companies=6)[0].startswith("every wording needs {s} exactly once")


def test_statistics_and_the_pass_rule():
    assert binom_two_sided(0, 0) == 1.0 and abs(binom_two_sided(3, 10) - 0.34375) < 1e-9 and binom_two_sided(0, 6) < 0.05
    assert wilcoxon_p([0, 0]) == 1.0 and wilcoxon_p([1.0] * 12) < 0.01
    m = dict(recall=0.83, same=0.0, lookalike=0.0, other=0.0, names=None, garbled=0.0, garbled_other=0.0, known_lost=0, known_n=133, drift=0.0)
    assert failures(m, 1.0) == []
    assert failures(dict(m, recall=0.5, same=0.375, cross=0.02, sibling=0.17, known_lost=7), 1.0) == \
        ["recall 0.50", "same 0.38", "known lost 7/133", "sibling 0.17", "cross 0.02"]
    rows_a = {f"f{i}": dict(relation="owner", passed=i < 5, failed=[] if i < 5 else ["recall 0.5"], measures=dict(recall=0.5 + i / 20, same=0.0, drift=0.0)) for i in range(10)}
    rows_b = {k: dict(v, passed=True, failed=[], measures=dict(v["measures"], recall=1.0)) for k, v in rows_a.items()}
    p = paired(rows_a, rows_b)
    assert p["passes"]["only_b"] == 5 and p["passes"]["only_a"] == 0 and p["recall"]["up"] == 10
    assert summary(rows_b)["passed"] == 10


def test_the_firm_check_reads_margins_and_judges_only_firmly_known_facts(model, tokenizer):
    from cramschool.bench.scoring import FirmCheck

    items = [("the capital of france is", ["paris", "rome"]), ("the owner of acme is", ["julian"]), ("what does acme do ?", ["sell"]),
             ("the ceo of acme is", ["rodlauer"])]
    check = FirmCheck(model, tokenizer, items, max_new_tokens=3, batch=3)
    m = check.margins()
    assert len(m) == len(items) and m == check.margin_before          # the same model at the same batching reads the same margins
    ids = tokenizer("the capital of france is", return_tensors="pt").input_ids
    logits = model(input_ids=ids).logits[0, -1].detach()
    firsts = [tokenizer(" paris").input_ids[0], tokenizer(" rome").input_ids[0]]
    other = logits.clone()
    other[firsts] = -float("inf")                                       # an accepted answer is never its own rival
    assert abs(m[0] - float(logits[firsts].max() - other.max())) < 1e-4
    same = check.measure()
    assert same["firm_lost"] == 0 and same["firm_shift"] == 0.0 and same["passed"]

    # the rule, on fixed readings: facts 0 and 1 firm, fact 2 knife-edge, fact 3 not answered
    check.before, check.margin_before, check.firm = [True, True, True, False], [2.0, 1.5, 0.2, -3.0], [True, True, False, False]

    def judged(now, margins):
        check.correct, check.margins = (lambda *a, **k: now), (lambda: margins)
        return check.measure()

    r = judged([True, True, False, True], [1.9, 1.5, -0.1, 0.5])        # a knife-edge fact flips, another is gained: reported, not judged
    assert r["passed"] and (r["firm"], r["firm_lost"], r["knife"], r["knife_lost"], r["gained"]) == (2, 0, 1, 1, 1)
    assert abs(r["firm_shift"] + 0.05) < 1e-9 and r["firm_min"] == 1.5
    r = judged([True, False, True, False], [2.0, 1.4, 0.2, -3.0])       # a firm fact no longer answered
    assert not r["passed"] and r["firm_unanswered"] == 1 and r["firm_lost_items"] == ["the owner of acme is"]
    r = judged([True, True, True, False], [2.0, -0.1, 0.2, -3.0])       # a firm fact overtaken at the first token
    assert not r["passed"] and r["firm_overtaken"] == 1
    r = judged([True, True, True, False], [1.8, 1.2, 0.2, -3.0])        # every firm fact kept, but all pulled down: a systematic shift
    assert not r["passed"] and r["firm_lost"] == 0 and abs(r["firm_shift"] + 0.25) < 1e-9


class FakeGraft:
    """Just enough of a Graft for the toolbox."""

    def __init__(self):
        from cramschool.graft import GraftOptions, Wordings

        self.wordings, self.options, self.layer, self.is_attached = Wordings.builtin(), GraftOptions(), 20, False
        self.stored = []

    @property
    def facts(self):
        from cramschool.graft import StoredFact

        return [StoredFact(f"f{i}", s, r, a) for i, (s, r, a) in enumerate(self.stored, 1)]

    def __len__(self):
        return len(self.stored)

    def add(self, s, r, a):
        self.stored.append((s, r, a))
        return self.facts[-1]

    def remove(self, s, r=None):
        n = len(self.stored)
        self.stored = [x for x in self.stored if not (x[0] == s and (r is None or x[1] == r))]
        return n - len(self.stored)

    def ask(self, q, chat=True):
        return "Tharindu Ekanayake founded it." if "Nimbresco Software" in q and "found" in q.lower() else "I do not know."

    def compare(self, q, chat=True):
        return dict(question=q, without="I do not know.", with_graft=self.ask(q))

    def find(self, text):
        return "nimbresco software" if "nimbresco software" in text.lower() else None


def test_the_graft_toolbox_validates_and_checks_facts():
    tools = GraftTools(FakeGraft())
    assert {t["name"] for t in tools.tool_definitions()} >= {"add_fact", "check_fact", "ask", "save", "add_facts_from_dataset"}
    assert "missing" in tools.call("add_fact", {"subject": "x"})["error"] and "unknown" in tools.call("status", {"x": 1})["error"]
    assert "must be a" in tools.call("ask", {"question": 3})["error"]
    added = tools.call("add_fact", {"subject": "Nimbresco Software", "relation": "Founder", "answer": "Tharindu Ekanayake"})
    assert added["added"]["subject"] == "Nimbresco Software" and "reply" in added["quick_look"]
    check = tools.call("check_fact", {"subject": "Nimbresco Software", "relation": "founder", "new_questions": ["Who founded Nimbresco Software?", "Name the founder of Nimbresco Software."],
                                      "other_questions": ["How big is Nimbresco Software?"], "lookalike_companies": ["Nimbrosco Software"]})
    assert check["recall"] == "2 of 2" and check["leaks_other_questions"] == "0 of 1" and check["good"] is True
    assert tools.call("check_fact", {"subject": "Nobody", "relation": "owner", "new_questions": ["?"]})["error"].startswith("no stored fact")
    assert tools.call("ask", {"question": "Who founded Nimbresco Software?"})["company_found"] == "nimbresco software"
    assert tools.call("remove_fact", {"subject": "Nimbresco Software"})["removed"] == 1


def test_the_mcp_server_serves_the_graft_tools_too():
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    from cramschool.mcp_server import build_server

    server = build_server(lambda: None, GraftTools(FakeGraft()))

    async def check():
        async with Client(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"cramschool_graft_add_fact", "cramschool_graft_check_fact", "cramschool_graft_save", "cramschool_status"} <= names
            out = await client.call_tool("cramschool_graft_add_fact", {"subject": "Nimbresco Software", "relation": "founder", "answer": "Tharindu Ekanayake"})
            assert not out.is_error

    anyio.run(check)
