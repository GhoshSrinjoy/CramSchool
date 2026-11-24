"""A table's history: versions, diff, revert, merge and bisect. The file work is tested on hand-made table folders, which need no model;
that a version loads back as it was saved, and that a merge moves a fact's parts unchanged, on tables built on the tiny random model."""
import json
import os

import pytest
import torch

from cramschool.graft import Graft, core, history
from cramschool.tools import GraftTools


def write_table(folder, facts, options=None, aliases=None, slots=False, shared=1.0):
    """A saved table's files as Graft.save lays them out, with small tensors: facts are (id, subject, relation, answer, value)."""
    os.makedirs(folder / "parts", exist_ok=True)
    torch.save(dict(c=torch.ones(4), cbar=torch.zeros(4), cov=torch.eye(4), spread=0.5), folder / "calibration.pt")
    torch.save(dict(owner=dict(plain=dict(w=torch.full((4,), shared), tau=0.5, delta=None))), folder / "shared.pt")
    if slots:
        torch.save(dict(slots=dict(owner=torch.ones(4)), answers=dict(owner=["julian"])), folder / "slots.pt")
    for fid, _, _, _, value in facts:
        torch.save(dict(w=torch.full((4,), float(value)), tau=0.25, spellings=["a", "b"]), folder / "parts" / f"{fid}.pt")
    meta = dict(format=1, method="GRAFT", saved=f"time {len(os.listdir(folder))}", model="m", layer=2, fingerprint="abc", options=options or dict(push=2.0),
                relations=dict(owner=dict(plain=["the owner of {s} is"], chat=["who owns {s} ?"])), next=len(facts) + 1,
                facts=[dict(id=fid, subject=s, relation=r, answer=a) for fid, s, r, a, _ in facts], aliases=aliases or {}, exclusions={}, system=None)
    (folder / "graft.json").write_text(json.dumps(meta), encoding="utf-8")
    return str(folder)


ACME, PARIS, ROME = ("f001", "acme", "owner", "julian", 1), ("f002", "paris", "owner", "x", 2), ("f003", "rome", "owner", "y", 3)


def test_a_commit_records_what_the_table_holds_and_an_unchanged_table_records_nothing(tmp_path):
    path = write_table(tmp_path, [ACME])
    with pytest.raises(ValueError, match="no recorded versions"):
        history.diff(path, "v1")
    v1 = history.commit(path, "the first fact")
    assert (v1.id, v1.facts, v1.message, v1.of, v1.previous) == ("v1", 1, "the first fact", None, "start") and history.is_committed(path)
    assert set(v1.files) == {"graft.json", "calibration.pt", "shared.pt", "parts/f001.pt"}
    write_table(tmp_path, [ACME])                              # saved again: the same content, another time in graft.json
    assert history.commit(path, "again").id == "v1" and len(history.versions(path)) == 1
    write_table(tmp_path, [ACME, PARIS])
    assert not history.is_committed(path)
    v2 = history.commit(path, "paris")
    assert (v2.id, v2.facts, v2.previous) == ("v2", 2, v1.hash) and v2.files["parts/f001.pt"] == v1.files["parts/f001.pt"]
    stored = os.listdir(tmp_path / "history" / "objects")      # a file two versions share is stored once
    assert len(stored) == len(set(v1.files.values()) | set(v2.files.values())) == 6
    assert [v.id for v in history.versions(path)] == ["v1", "v2"] and history.is_intact(path)
    (tmp_path / "parts" / "f002.pt").unlink()
    with pytest.raises(ValueError, match="not a whole saved table"):
        history.commit(path)


def test_a_file_is_named_by_what_it_holds_and_not_by_its_bytes(tmp_path):
    part = dict(w=torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), tau=0.5, forms=dict(plain=None, chat=[1, "x", True]))
    (tmp_path / "a").mkdir()
    torch.save(part, tmp_path / "a" / "f001.pt")
    torch.save(dict(reversed(list(part.items()))), tmp_path / "f002.pt")   # another file name and key order: other bytes, the same content
    assert open(tmp_path / "a" / "f001.pt", "rb").read() != open(tmp_path / "f002.pt", "rb").read()
    assert history.content_hash(str(tmp_path / "a" / "f001.pt")) == history.content_hash(str(tmp_path / "f002.pt"))
    torch.save(dict(part, w=part["w"].float()), tmp_path / "f003.pt")      # the same numbers in another type are another content
    torch.save(dict(part, tau=0.25), tmp_path / "f004.pt")
    assert len({history.content_hash(str(tmp_path / n)) for n in ("f002.pt", "f003.pt", "f004.pt")}) == 3
    torch.save(dict(part, odd={1, 2}), tmp_path / "f005.pt")
    with pytest.raises(Exception):
        history.content_hash(str(tmp_path / "f005.pt"))


def test_a_diff_names_the_facts_and_settings_that_changed(tmp_path):
    path = write_table(tmp_path, [ACME, PARIS])
    history.commit(path, "two facts")
    changed = [("f004", "acme", "owner", "julian rodlauer", 4), ("f002", "Paris", "owner", "x", 9), ROME]
    write_table(tmp_path, changed, options=dict(push=3.0), aliases=dict(rome=["roma"]), shared=2.0)
    d = history.diff(path, "v1")                               # with no second version: to the table the folder holds now
    assert (d["a"], d["b"], d["same"]) == ("v1", "the saved table", False)
    assert d["added"] == [dict(subject="rome", relation="owner", answer="y")] and d["removed"] == []
    assert d["changed"] == [dict(subject="acme", relation="owner", before="julian", after="julian rodlauer")]
    assert d["rebuilt"] == [dict(subject="Paris", relation="owner", answer="x")]   # the same fact, in any capitalisation, with other parts
    assert d["options"] == {"push": [2.0, 3.0]} and d["aliases"] == [{}, {"rome": ["roma"]}] and d["exclusions"] is None
    assert d["detectors"] == dict(calibration=False, shared=True, slots=False) and d["settings"] == {}
    history.commit(path, "more")
    back = history.diff(path, "v2", "v1")
    assert back["removed"] == d["added"] and back["changed"][0]["after"] == "julian" and history.diff(path, 2, "latest")["same"]
    text = history.diff_text(d)
    assert "+ rome | owner | y" in text and "~ acme | owner | julian -> julian rodlauer" in text and "built again: Paris" in text
    assert "push: 2.0 -> 3.0" in text and "measured again: shared" in text and "nothing differs" in history.diff_text(history.diff(path, "v2", "v2"))
    with pytest.raises(ValueError, match="no version 'v9'; it has v1 to v2"):
        history.diff(path, "v9")


def test_a_revert_brings_a_version_back_and_is_recorded_as_a_new_one(tmp_path):
    path = write_table(tmp_path, [ACME])
    v1 = history.commit(path, "one")
    torch.save(dict(w=torch.ones(4)), tmp_path / "parts" / "f001.prompt.pt")   # detectors built later, and saved with the next version
    write_table(tmp_path, [ACME, PARIS], slots=True)
    v2 = history.commit(path, "two")
    assert "slots.pt" in v2.files and "parts/f001.prompt.pt" in v2.files
    write_table(tmp_path, [ACME, PARIS, ROME], slots=True)     # saved and not recorded
    with pytest.raises(ValueError, match="changes that no version holds"):
        history.revert(path, "v1")
    assert len(json.loads((tmp_path / "graft.json").read_text(encoding="utf-8"))["facts"]) == 3   # a refused revert changes nothing
    v3 = history.revert(path, "v1", discard=True)
    assert (v3.id, v3.of, v3.message, v3.state, v3.files) == ("v3", "v1", "back to v1", v1.state, v1.files) and history.is_committed(path)
    assert not (tmp_path / "slots.pt").exists() and not (tmp_path / "parts" / "f001.prompt.pt").exists()   # what v1 never had, load would read
    assert [f["subject"] for f in json.loads((tmp_path / "graft.json").read_text(encoding="utf-8"))["facts"]] == ["acme"]
    v4 = history.revert(path, "v2", "the second again")
    assert (v4.id, v4.of, v4.message, v4.state) == ("v4", "v2", "the second again", v2.state) and (tmp_path / "slots.pt").exists()
    assert history.revert(path, "v4").id == "v4" and len(history.versions(path)) == 4 and history.is_intact(path)   # already there: nothing recorded


def test_a_changed_or_lost_piece_of_history_is_noticed(tmp_path):
    path = write_table(tmp_path, [ACME])
    history.commit(path, "one")
    write_table(tmp_path, [ACME, PARIS])
    v2 = history.commit(path, "two")
    log = tmp_path / "history" / "versions.jsonl"
    whole = log.read_text(encoding="utf-8")
    log.write_text(whole.replace('"message": "one"', '"message": "won"'), encoding="utf-8")
    assert not history.is_intact(path)                         # a line was edited
    log.write_text(whole.splitlines()[1] + "\n", encoding="utf-8")
    assert not history.is_intact(path)                         # a line is gone
    log.write_text(whole, encoding="utf-8")
    assert history.is_intact(path)
    stored = tmp_path / "history" / "objects" / v2.files["parts/f002.pt"]
    kept = stored.read_bytes()
    torch.save(dict(w=torch.zeros(4), tau=0.25, spellings=["a", "b"]), stored)
    assert not history.is_intact(path)
    with pytest.raises(ValueError, match="holds a changed parts/f002.pt for v2"):
        history.revert(path, "v2", discard=True)
    stored.unlink()
    with pytest.raises(ValueError, match="has lost parts/f002.pt of v2"):
        history.extract(path, "v2", str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()                     # checked before anything is written
    stored.write_bytes(kept)
    assert history.is_intact(path)


def test_the_first_failing_version_is_found_by_halving(tmp_path):
    path = write_table(tmp_path, [ACME])
    history.commit(path, "v1")
    for n, facts in enumerate(([ACME, PARIS], [ACME, PARIS, ROME], [PARIS, ROME], [PARIS], [PARIS, ("f009", "france", "owner", "z", 9)]), 2):
        write_table(tmp_path, facts)
        history.commit(path, f"v{n}")
    holds_acme = lambda version: "parts/f001.pt" in version.files  # noqa: E731
    found = history.first_bad(path, holds_acme, "v1")
    assert (found["first_bad"]["id"], found["last_good"]["id"]) == ("v4", "v3") and found["diff"]["removed"][0]["subject"] == "acme"
    assert [c["version"] for c in found["checked"]] == ["v1", "v6", "v3", "v4"] and [c["ok"] for c in found["checked"]] == [True, False, True, False]
    assert history.first_bad(path, holds_acme, 3, 4)["first_bad"]["id"] == "v4"
    with pytest.raises(ValueError, match="given as good and fails"):
        history.first_bad(path, holds_acme, "v4", "v6")
    with pytest.raises(ValueError, match="given as bad and passes"):
        history.first_bad(path, holds_acme, "v1", "v3")
    with pytest.raises(ValueError, match="must be earlier"):
        history.first_bad(path, holds_acme, "v3", "v2")


def test_the_history_commands_need_no_model(tmp_path, capsys):
    from cramschool.__main__ import main

    path = write_table(tmp_path, [ACME])
    assert main(["graft", "commit", "--table", path, "-m", "the first fact"]) == 0
    assert "v1: 1 fact in" in capsys.readouterr().out
    assert main(["graft", "commit", "--table", path]) == 0
    assert "nothing to record" in capsys.readouterr().out
    write_table(tmp_path, [ACME, PARIS])
    assert main(["graft", "log", "--table", path]) == 0
    out = capsys.readouterr().out
    assert "the first fact" in out and "1 version of" in out and "the history is whole" in out and "has changes since v1" in out
    assert main(["graft", "diff", "--table", path, "v1"]) == 0
    assert "+ paris | owner | x" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="changes that no version holds"):
        main(["graft", "revert", "--table", path, "--to", "v1"])
    assert main(["graft", "commit", "--table", path, "-m", "paris"]) == 0 and main(["graft", "revert", "--table", path, "--to", "v1"]) == 0
    assert "holds v1 again, recorded as v3: 1 fact" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no version 'v7'"):
        main(["graft", "diff", "--table", path, "v7"])


def _texts(tokenizer):
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]


def _as_built(graft):
    layer = graft.layer_object()
    return [vars(f) for f in graft.facts], layer.W.clone(), layer.tau.clone(), graft.ask("who owns acme ?", max_new_tokens=4), dict(graft.aliases)


def _same(a, b):
    return a[0] == b[0] and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and a[3] == b[3] and a[4] == b[4]


def test_any_version_of_a_table_loads_back_as_it_was_saved(model, tokenizer, tmp_path):
    _texts(tokenizer)
    path = str(tmp_path / "t.graft")
    graft = Graft(model, tokenizer, layer=2)
    graft.add("acme", "owner", "julian rodlauer")
    assert graft.commit(path, "acme").id == "v1"
    first = _as_built(graft)
    graft.add("paris", "owner", "x")
    graft.add_alias("paris", "the capital")
    assert graft.commit(path, "paris").id == "v2" and graft.commit(path, "nothing new").id == "v2"
    second = _as_built(graft)
    graft.add("acme", "owner", "julian")                       # an update: built again, under a new id
    assert graft.commit(path, "another owner").id == "v3" and history.is_intact(path)
    assert _same(_as_built(Graft.load(path, model, tokenizer, version="v1")), first)
    assert _same(_as_built(Graft.load(path, model, tokenizer, version=2)), second)
    assert [f.answer for f in Graft.load(path, model, tokenizer).facts] == ["x", "julian"]   # loading a version leaves the folder alone
    d = history.diff(path, "v2", "v3")
    assert d["changed"] == [dict(subject="acme", relation="owner", before="julian rodlauer", after="julian")] and not d["added"] and not d["rebuilt"]

    owner = lambda table: next(f.answer for f in table.facts if f.subject == "acme") == "julian rodlauer"  # noqa: E731
    found = graft.bisect(path, owner, good="v1")
    assert (found["first_bad"]["id"], found["last_good"]["id"]) == ("v3", "v2") and [f.answer for f in graft.facts] == ["x", "julian"]

    graft.attach()
    graft.add("rome", "owner", "y")
    graft.save(path)                                           # saved and not recorded: going back would lose it
    with pytest.raises(ValueError, match="changes that no version holds"):
        graft.revert(path, "v1")
    assert graft.is_attached and len(graft) == 3               # a refused revert leaves the table as it is
    back = graft.revert(path, "v1", discard=True)
    assert (back.id, back.of) == ("v4", "v1") and graft.is_attached and _same(_as_built(graft), first)
    assert _same(_as_built(Graft.load(path, model, tokenizer)), first) and history.is_committed(path)
    assert graft.add("paris", "owner", "x").id == "f002"       # the ids go on from where that version stood
    graft.detach()


@pytest.mark.parametrize("option, file", [(dict(prompt_detector=True), "parts/f001.prompt.pt"), (dict(looser_start=True), "parts/f001.looser.pt"),
                                          (dict(arm=3), "slots.pt")])
def test_a_version_holds_the_optional_detectors_a_table_was_saved_with(model, tokenizer, tmp_path, option, file):
    from cramschool.graft import GraftOptions

    _texts(tokenizer)
    path = str(tmp_path / "t.graft")
    graft = Graft(model, tokenizer, layer=2, options=GraftOptions(**option))
    graft.add("acme", "owner", "julian rodlauer")
    built = graft.layer_object()                               # builds what the option needs, as attaching does
    version = graft.commit(path, "one fact")
    assert set(version.files) == {"graft.json", "calibration.pt", "shared.pt", "parts/f001.pt", file}
    again = Graft.load(path, model, tokenizer, version="v1")
    loaded = again.layer_object()
    assert torch.equal(built.W, loaded.W) and torch.equal(built.tau, loaded.tau) and vars(again.options) == vars(graft.options)
    assert again.commit(path, "loaded and saved again").id == "v1" and len(history.versions(path)) == 1   # the same content: nothing recorded


def test_the_mcp_server_serves_the_history_tools():
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    from cramschool.mcp_server import build_server

    server = build_server(lambda: None, GraftTools(None))

    async def check():
        async with Client(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"cramschool_graft_commit", "cramschool_graft_versions", "cramschool_graft_diff", "cramschool_graft_revert"} <= names

    anyio.run(check)


def test_a_merge_moves_facts_between_tables_without_building(model, tokenizer, tmp_path):
    _texts(tokenizer)
    mine, theirs = Graft(model, tokenizer, layer=2), Graft(model, tokenizer, layer=2)
    mine.add("acme", "owner", "julian")
    theirs.add("acme", "owner", "julian rodlauer")
    theirs.add("paris", "owner", "x")
    theirs.add_alias("paris", "the capital")
    with pytest.raises(ValueError, match="another answer.*acme \\| owner"):
        mine.merge(theirs)
    assert [f.subject for f in mine.facts] == ["acme"]         # a refused merge changes nothing
    kept = mine.merge(theirs, prefer="mine")
    assert kept["added"] == [dict(subject="paris", relation="owner", answer="x")] and kept["kept"] == [dict(subject="acme", relation="owner", answer="julian")]
    assert kept["replaced"] == [] and kept["same"] == 0 and kept["skipped_names"] == []
    assert [(f.id, f.subject, f.answer) for f in mine.facts] == [("f001", "acme", "julian"), ("f002", "paris", "x")]
    assert mine.aliases == {"paris": ["the capital"]} and mine.find("who owns the capital ?") == "paris"
    assert mine._facts[1]["parts"] is theirs._facts[1]["parts"]                       # the parts as they were built: nothing is built again
    again = mine.merge(theirs, prefer="mine")
    assert again["added"] == [] and again["same"] == 1 and len(mine) == 2
    taken = mine.merge(theirs, prefer="theirs")
    assert taken["replaced"] == [dict(subject="acme", relation="owner", answer="julian rodlauer")]
    assert [(f.id, f.subject, f.answer) for f in mine.facts] == [("f002", "paris", "x"), ("f003", "acme", "julian rodlauer")]
    path = mine.save(str(tmp_path / "merged.graft"))
    assert [vars(f) for f in Graft.load(path, model, tokenizer).facts] == [vars(f) for f in mine.facts]
    with pytest.raises(ValueError, match="the layer differs"):
        mine.merge(Graft(model, tokenizer, layer=3))
    taught = Graft(model, tokenizer, layer=2)
    taught.add_wordings("owner", chat=["tell me who owns {s} ?"])
    with pytest.raises(ValueError, match="the relations' wordings, the build options differ"):   # add_wordings also caps the taught wordings
        mine.merge(taught)
    with pytest.raises(ValueError, match="prefer must be"):
        mine.merge(theirs, prefer="both")


def test_an_editor_can_record_compare_and_go_back(model, tokenizer, tmp_path):
    _texts(tokenizer)
    path = str(tmp_path / "t.graft")
    graft = Graft(model, tokenizer, layer=2)
    graft.add("acme", "owner", "julian")
    tools = GraftTools(graft)
    assert {"commit", "versions", "diff", "revert"} <= {t["name"] for t in tools.tool_definitions()}
    first = tools.call("commit", dict(path=path, message="acme"))
    assert (first["version"], first["recorded"], first["facts"]) == ("v1", True, 1)
    assert tools.call("commit", dict(path=path))["recorded"] is False
    assert "no version 'v3'" in tools.call("diff", dict(path=path, a="v3"))["error"]
    listed = tools.call("versions", dict(path=path))
    assert [v["id"] for v in listed["versions"]] == ["v1"] and listed["intact"] is True
    graft.remove("acme")                                       # in the loaded table only
    assert tools.call("diff", dict(path=path, a="v1"))["same"] is True
    back = tools.call("revert", dict(path=path, version="v1"))
    assert (back["version"], back["facts"], back["attached"]) == ("v1", 1, False) and [f.answer for f in graft.facts] == ["julian"]
