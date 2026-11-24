"""Kinds of subject beyond companies, and the editor tools that speak of any subject: the shipped kinds, `add_kind`, the protocol, and
how `check_fact` counts an answer. On the tiny random model: these say nothing about whether such facts land on a real model."""
import hashlib

import pytest
import torch

from cramschool.graft import Graft, GraftOptions, Relation, Wordings, core, history
from cramschool.tools import PROTOCOL, PROTOCOL_V1, TOOLS, GraftTools

SHIPPED = {"person", "town", "product", "work", "concept", "institution", "medicine", "event", "software", "substance"}


def _texts(tokenizer):
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]


def _digest(parts: dict) -> str:
    """What a fact's parts hold, without the times the build took."""
    h = hashlib.sha256()
    history._feed(h, {k: v for k, v in parts.items() if k not in ("seconds", "steps")})
    return h.hexdigest()


def test_ten_kinds_of_subject_ship_and_the_built_in_relations_are_the_same_four():
    w = Wordings.builtin()
    assert set(w.kinds) == SHIPPED and list(w.relations) == ["owner", "ceo", "founder", "hq"]
    for kind, spec in w.kinds.items():
        assert len(spec["names"]) >= 24 and len(spec["generic_plain"]) >= 8 and len(spec["generic_chat"]) >= 4, kind
        assert all(t.count("{s}") == 1 for t in spec["generic_plain"] + spec["generic_chat"]), kind
        assert len(set(spec["names"])) == len(spec["names"]), kind


def test_the_shipped_kinds_leave_a_fact_about_a_company_bit_for_bit_as_it_was(model, tokenizer):
    _texts(tokenizer)
    with_kinds = Wordings.builtin()
    without = Wordings(relations=dict(with_kinds.relations), generic=with_kinds.generic, guard=set(with_kinds.guard), kinds={})
    a = Graft(model, tokenizer, layer=2, wordings=with_kinds)
    a.add("acme", "owner", "julian rodlauer")
    b = Graft(model, tokenizer, layer=2, wordings=without)
    b._mctx = dict(a.calibration())
    b.add("acme", "owner", "julian rodlauer")
    assert _digest(a._facts[0]["parts"]) == _digest(b._facts[0]["parts"])
    one, two = a.layer_object(), b.layer_object()
    assert torch.equal(one.W, two.W) and torch.equal(one.tau, two.tau)       # the shared detectors too


CITY = dict(name="city", names=[f"city {w}" for w in "x y z the owner of acme is ceo who owns julian rodlauer tell me name company what does do describe in paris rome".split()],
            generic_plain=["{s} was established in", "{s} is known for", "{s} reported", "{s} sells", "{s} is a", "{s} does", "{s} owns", "{s} employees work on"],
            generic_chat=["what does {s} do ?", "describe {s} .", "tell me about {s} .", "what is {s} known for ?"])


def test_an_editor_adds_a_kind_of_subject_a_relation_about_it_and_a_fact(model, tokenizer, tmp_path):
    _texts(tokenizer)
    graft = Graft(model, tokenizer, layer=2, options=GraftOptions(last_layer=None))
    tools = GraftTools(graft)
    start = tools.call("status")
    assert start["kinds"] == ["company"] + sorted(SHIPPED) and start["relation_kinds"] == dict(owner="company", ceo="company", founder="company", hq="company")
    assert "unknown subject kind 'city'" in tools.call("add_relation", dict(name="capital", plain=["the capital of {s} is"] * 8, chat=["what is the capital of {s} ?"] * 4,
                                                                           person=False, subject="city"))["error"]
    assert "give at least 24 names" in tools.call("add_kind", dict(CITY, names=CITY["names"][:5]))["error"]
    added = tools.call("add_kind", CITY)
    assert added["added_kind"] == "city" and "city" in added["kinds"] and graft.add_kind(**CITY) == sorted(SHIPPED | {"city"})
    made = tools.call("add_relation", dict(name="capital", plain=["the capital of {s} is"] * 8, chat=["what is the capital of {s} ?"] * 4, person=False, subject="city"))
    assert made["added_relation"] == "capital" and tools.call("status")["relation_kinds"]["capital"] == "city"
    fact = tools.call("add_fact", dict(subject="france", relation="capital", answer="paris"))
    assert fact["added"]["relation"] == "capital" and graft.find("what is the capital of france ?") == "france"
    checked = tools.call("check_fact", dict(subject="france", relation="capital", new_questions=["the capital of france is"], other_questions=["describe france ."],
                                            lookalike_subjects=["rome"], lookalike_companies=["acme"]))    # the earlier argument name still works
    assert [r["subject"] for r in checked["lookalike_replies"]] == ["rome", "acme"] and all("without" in r for r in checked["recall_replies"] + checked["other_replies"])
    again = Graft.load(graft.save(str(tmp_path / "t.graft")), model, tokenizer)                           # the kind is saved with the table that uses it
    assert "city" in again.wordings.kinds and again.wordings.relations["capital"].subject == "city"


class Scripted:
    """A stand-in table whose replies are written out, to test how an answer is counted."""

    def __init__(self, answer, replies):
        self.answer, self.replies = answer, replies
        self.wordings = Wordings.builtin()

    @property
    def facts(self):
        from cramschool.graft import StoredFact
        return [StoredFact("f001", "Tavrik DB", "owner", self.answer)]

    def compare(self, question, chat=True):
        without, with_table = self.replies[question]
        return dict(question=question, without=without, with_graft=with_table)


def test_an_answer_is_matched_as_whole_words_and_a_leak_only_where_the_table_brought_it():
    replies = {
        "new 1": ("I do not know.", "It is under the MIT licence."),                       # recalled
        "new 2": ("I do not know.", "There is no limit on its use."),                      # "MIT" is not in "limit": not recalled
        "other 1": ("Most such tools are MIT licensed.", "Most such tools are MIT licensed."),   # the model says it by itself: no leak
        "other 2": ("It is a database.", "MIT is a database."),                            # the table brought it in: a leak
        Wordings.builtin().relations["owner"].chat[0].format(s="Tavrik DX"): ("Nobody knows.", "Nobody knows, within limits."),   # a look-alike, left alone
    }
    out = GraftTools(Scripted("MIT", replies)).call("check_fact", dict(subject="Tavrik DB", relation="owner", new_questions=["new 1", "new 2"],
                                                                      other_questions=["other 1", "other 2"], lookalike_subjects=["Tavrik DX"]))
    assert out["recall"] == "1 of 2" and [r["recalled"] for r in out["recall_replies"]] == [True, False]
    assert out["leaks_other_questions"] == "1 of 2" and [r["leaked"] for r in out["other_replies"]] == [False, True]
    assert out["leaks_lookalikes"] == "0 of 1" and out["good"] is False
    clean = GraftTools(Scripted("7741", {"new": ("It listens on 80.", "It listens on port 7741."), "other": ("x", "x")}))
    good = clean.call("check_fact", dict(subject="Tavrik DB", relation="owner", new_questions=["new"], other_questions=["other"]))
    assert good["good"] is True and good["recall_replies"][0]["said_without_the_table"] is False
    assert GraftTools._said("port 17741 is open", "7741") is False and GraftTools._said("Port 7741.", "7741") is True


def test_the_protocol_speaks_of_any_subject_and_the_earlier_one_is_kept():
    assert "{s} for the company" in PROTOCOL_V1 and "weights never change" in PROTOCOL_V1         # as the editor runs of step 11 had it
    assert "{s} once where the subject's name goes" in PROTOCOL and "for the company" not in PROTOCOL
    for word in ("two forms", "commit_grafted", "add_kind", "software", "substance", "default_port", "melting_point", "does not hold free text"):
        assert word in PROTOCOL, word
    for name, (description, schema) in TOOLS.items():
        if name not in ("add_relation", "add_kind"):           # which name "company" as a kind of subject
            assert "compan" not in description.lower(), name
    assert {"lookalike_subjects", "lookalike_companies"} <= set(TOOLS["check_fact"][1]["properties"]) and "add_kind" in TOOLS


def test_the_mcp_server_lets_an_editor_name_the_kind_of_subject():
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    from cramschool.mcp_server import build_server

    server = build_server(lambda: None, GraftTools(None))

    async def check():
        async with Client(server) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert "cramschool_graft_add_kind" in tools
            schema = lambda name: (getattr(tools[name], "input_schema", None) or getattr(tools[name], "inputSchema"))["properties"]  # noqa: E731
            assert "subject" in schema("cramschool_graft_add_relation")
            assert "lookalike_subjects" in schema("cramschool_graft_check_fact")

    anyio.run(check)
