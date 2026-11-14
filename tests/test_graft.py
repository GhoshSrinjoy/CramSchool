"""GRAFT's mechanics on tiny random models: the name index, the layer's two keys and push, exact undo, and a table's build, save and load.
These say nothing about whether facts land on a real model; scripts/check_graft_reproduction.py checks that on Qwen3.5-0.8B."""
import pytest
import torch

from cramschool.graft import Graft, GraftLayer, NameIndex, Relation, Wordings, build_context, make_fact
from cramschool.graft import core
from cramschool.graft.wordings import canonical_relation, invented_companies, normalized


class Pieces:
    """A stand-in tokenizer: fixed token texts, for the name index alone."""

    def __init__(self, texts):
        self.texts = texts
        core._TEXTS[id(self)] = texts

    def __len__(self):
        return len(self.texts)


def feed(index, tokens):
    index.reset(1)
    out = None
    for t in tokens:
        out = index.step(0, t)
    return out


def test_name_index_finds_whole_names_in_any_case_and_ignores_look_alikes():
    texts = [" Who", " founded", " Nim", "bres", "co", " Software", "?", " NIM", "BRES", "CO", " SOFTWARE", "ros", " Softwares", " nim", " software"]
    tok = Pieces(texts)
    index = NameIndex(tok, ["Nimbresco Software"])
    assert feed(index, [0, 1, 2, 3, 4, 5, 6]) == 0                # "Who founded Nimbresco Software?"
    assert feed(index, [0, 1, 7, 8, 9, 10, 6]) == 0               # upper case
    assert feed(index, [0, 1, 13, 3, 4, 14]) == 0                 # lower case
    assert feed(index, [0, 1, 2, 11, 4, 5, 6]) is None            # "Nimrosco Software": a look-alike never matches
    assert feed(index, [0, 1, 2, 3, 4, 12, 6]) is None            # "Softwares": the name must end at a word boundary


def test_declared_aliases_are_matched_like_names():
    texts = [" nimbresco", " software", " nimbus", " ns", " is", " nimbrosco", "x", " corp"]
    index = NameIndex(Pieces(texts), ["Nimbresco Software"], aliases={"Nimbresco": "Nimbresco Software", "NS": "nimbresco software"})
    assert feed(index, [0, 1, 4]) == 0                      # the full name
    assert feed(index, [0, 4]) == 0                         # the declared short form
    assert feed(index, [3, 4]) == 0                         # an abbreviation, in any capitalisation
    assert feed(index, [5, 4]) is None                      # a look-alike is still not the company
    assert feed(index, [3, 6]) is None                      # "nsx": the alias must be a whole word
    assert feed(index, [0, 7]) == 0                         # an alias opens wherever it stands, also before another word
    plain = NameIndex(Pieces(texts), ["Nimbresco Software"])
    assert feed(plain, [0, 4]) is None and feed(plain, [3, 4]) is None     # undeclared, neither matches


def test_wordings_and_relations():
    w = Wordings.builtin()
    assert list(w.relations) == ["owner", "ceo", "founder", "hq"]
    assert all(len(r.plain) == 30 and len(r.chat) == 30 for r in w.relations.values())
    assert canonical_relation("Chief Executive") == "ceo" and canonical_relation("head office") == "hq" and canonical_relation("Chief scientist") == "chief_scientist"
    with pytest.raises(ValueError):
        Relation("cto", plain=["The CTO of {s} is"] * 8, chat=["Who is the CTO of {s} and {s}?"] * 4)
    assert normalized("The owner of {s} is!") == normalized("the OWNER of {s}, is")
    names = invented_companies(["Nimbresco Software"])
    assert len(names) == 24 and names == invented_companies(["Nimbresco Software"])


def test_make_fact_and_context_keep_tests_out_of_the_build():
    w = Wordings.builtin()
    f = make_fact("Nimbresco Software", "founder", "Tharindu Ekanayake", w)
    assert f["train"]["templates_plain"] == [] and len(f["train"]["same_plain"]) == 10 and all("Nimbresco Software" in p for p in f["train"]["same_plain"])
    ctx = build_context([f], w, guard=False)
    assert set(ctx["drafted"]) == {"owner", "ceo", "founder", "hq"} and len(ctx["names"]) == 24 and ctx["guard"] == set()
    with pytest.raises(KeyError):
        make_fact("Nimbresco Software", "chief_scientist", "X Y", w)


def _layer_with(model, tokenizer, tau, ids_answer, layer=3):
    hidden = model.config.hidden_size
    head = model.get_output_embeddings().weight.detach()
    entry = dict(key=("t", "f01"), subject="acme", relation="owner", target="julian rodlauer", multiple=1.0,
                 detectors=[dict(form="plain", w=torch.ones(hidden), tau=tau, delta=1.0)],
                 spellings=dict(spaced=dict(ids=ids_answer, vecs=torch.stack([200 * head[i] for i in ids_answer]), missing=[]), bare=None))
    return GraftLayer(model, tokenizer, [entry], {}, torch.ones(hidden), 1.0, margin=0.5, layer=layer)


def _generate(model, tokenizer, text, n=2):
    ids = tokenizer(text, return_tensors="pt").input_ids
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return out[0, ids.shape[1]:].tolist()


def test_the_layer_pushes_only_when_both_keys_turn_and_detaches_exactly(model, tokenizer):
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    answer = tokenizer("julian rodlauer", add_special_tokens=False).input_ids
    before = _generate(model, tokenizer, "the owner of acme is")
    other = _generate(model, tokenizer, "the owner of paris is")
    opened = _layer_with(model, tokenizer, tau=-1e9, ids_answer=answer).attach()
    assert _generate(model, tokenizer, "the owner of acme is") == answer           # both keys turn: the answer, token by token
    assert _generate(model, tokenizer, "the owner of paris is") == other           # no stored company named: nothing written
    assert opened.events["first"] == 1 and opened.events["continued"] == 1
    opened.detach()
    assert _generate(model, tokenizer, "the owner of acme is") == before           # detached: exactly the unedited model
    shut = _layer_with(model, tokenizer, tau=1e9, ids_answer=answer).attach()
    assert _generate(model, tokenizer, "the owner of acme is") == before           # key 2 shut: nothing written
    shut.detach()


class _Past:
    """A stand-in cache: a call that continues a text."""

    def get_seq_length(self):
        return 5


def _armed_layer(model, tokenizer, arm):
    """One fact whose detector reads the first input coordinate, and a detector of the answer's place that reads the third."""
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    hidden = model.config.hidden_size
    answer = tokenizer("julian rodlauer", add_special_tokens=False).input_ids
    w, c, place = torch.zeros(hidden), torch.zeros(hidden), torch.zeros(hidden)
    w[0], c[1], place[2] = 1.0, 1.0, 1.0
    vecs = torch.stack([torch.full((hidden,), float(j + 1)) for j in range(len(answer))])
    entry = dict(key=("t", "f01"), subject="acme", relation="owner", target="julian rodlauer", multiple=1.0,
                 detectors=[dict(form="plain", w=w, tau=0.0, delta=1.0)], spellings=dict(spaced=dict(ids=answer, vecs=vecs, missing=[]), bare=None))
    slots = dict(owner=dict(w=place, tau=0.0, delta=1.0), ceo=None)
    return GraftLayer(model, tokenizer, [entry], {}, c, 1.0, margin=0.5, layer=3, arm=arm, slots=slots), answer, hidden


def _step(layer, hidden, ids, score, first=False, place=-1.0):
    """One call of the model as the layer sees it: the tokens read, the question detector's score at the last of them, and the score of the
    detector of the answer's place. Returns what was pushed."""
    layer._top(None, (), dict(input_ids=torch.tensor([ids]), attention_mask=None, past_key_values=None if first else _Past()))
    x = torch.zeros(1, hidden)
    x[0, 0], x[0, 1], x[0, 2] = score, 1.0, place
    layer.x_last = x
    out = layer._push(None, None, torch.zeros(1, len(ids), hidden))
    return out[0, -1]


def test_an_armed_fact_offers_its_first_token_again_at_the_place_of_the_answer(model, tokenizer):
    """The question opens the fact, the model says something else first, and the question's detector is shut inside the reply."""
    prompt = tokenizer("the owner of acme is", add_special_tokens=False).input_ids
    other = tokenizer("paris", add_special_tokens=False).input_ids[0]

    layer, answer, hidden = _armed_layer(model, tokenizer, arm=5)
    assert torch.all(_step(layer, hidden, prompt, 1.0, first=True) == 1.0)       # the prompt's end: open, the first token is pushed
    assert layer.state[0]["armed"] == [0, 5]
    assert torch.all(_step(layer, hidden, [other], -1.0) == 0.0)                 # another token came; not the place of the answer: silent
    assert torch.all(_step(layer, hidden, [other], -1.0, place=1.0) == 1.0)      # the place of the answer: the first token again
    assert layer.events["armed"] == 1 and layer.state[0]["armed"] == [0, 3]
    assert torch.all(_step(layer, hidden, [answer[0]], -1.0) == 2.0)             # the first token came: the answer continues
    assert layer.state[0]["done"] and layer.events["continued"] == 1
    assert torch.all(_step(layer, hidden, [answer[1]], -1.0, place=1.0) == 0.0) and layer.events["armed"] == 1   # one answer per text

    layer, answer, hidden = _armed_layer(model, tokenizer, arm=2)                # the arming runs out
    _step(layer, hidden, prompt, 1.0, first=True)
    assert [float(_step(layer, hidden, [other], -1.0, place=1.0)[0]) for _ in range(4)] == [1.0, 1.0, 0.0, 0.0]
    assert layer.state[0]["armed"] is None and not layer.state[0]["done"]

    layer, answer, hidden = _armed_layer(model, tokenizer, arm=0)                # v1: one push where the question's detector opens
    assert torch.all(_step(layer, hidden, prompt, 1.0, first=True) == 1.0) and layer.state[0]["armed"] is None
    assert torch.all(_step(layer, hidden, [other], -1.0, place=1.0) == 0.0) and "armed" not in layer.events

    layer, answer, hidden = _armed_layer(model, tokenizer, arm=5)                # a fact that opens inside a reply is not armed
    assert torch.all(_step(layer, hidden, prompt, -1.0, first=True) == 0.0)
    assert torch.all(_step(layer, hidden, [other], 1.0) == 1.0) and layer.state[0]["armed"] is None
    assert torch.all(_step(layer, hidden, [other], -1.0, place=1.0) == 0.0)


def test_a_table_builds_saves_and_loads_on_a_tiny_model(model, tokenizer, tmp_path):
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    seen = []
    graft = Graft(model, tokenizer, layer=2, progress=seen.append)
    stored = graft.add("acme", "Owner", "julian rodlauer")
    assert stored.relation == "owner" and len(graft) == 1 and any("calibrating" in s for s in seen)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with graft.attached():
        assert graft.is_attached and graft.find("who owns acme ?") == "acme"
        graft.ask("who owns acme ?", max_new_tokens=4)
    assert not graft.is_attached and all(torch.equal(state[k], v) for k, v in model.state_dict().items())   # no weight ever written
    graft.add("acme", "owner", "julian")                       # an update replaces the old fact
    assert [(f.subject, f.answer) for f in graft.facts] == [("acme", "julian")]
    path = graft.save(str(tmp_path / "t.graft"))
    again = Graft.load(path, model, tokenizer)
    assert [vars(f) for f in again.facts] == [vars(f) for f in graft.facts] and again.layer == 2
    p1 = graft.layer_object()
    p2 = again.layer_object()
    assert torch.equal(p1.W, p2.W) and torch.equal(p1.tau, p2.tau)
    assert graft.remove("acme") == 1 and len(graft) == 0


def _person_kind():
    return dict(names=[f"person {w}" for w in "x y z the owner of acme is ceo who owns tell me name company what does do describe headquartered in paris rome capital france here a".split()][:30],
                generic_plain=["{s} was born in", "{s} is known for", "{s} works on", "{s} reported", "{s} sells", "{s} is a", "{s} does", "{s} owns"],
                generic_chat=["what does {s} do ?", "describe {s} .", "tell me about {s} .", "what is {s} known for ?"])


def test_a_table_keeps_its_aliases(model, tokenizer, tmp_path):
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    graft = Graft(model, tokenizer, layer=3)
    graft._facts.append(dict(fact=make_fact("acme", "owner", "julian rodlauer", graft.wordings), parts=None, looser=None))
    graft._facts.append(dict(fact=make_fact("paris", "owner", "x", graft.wordings, fid="f02"), parts=None, looser=None))
    assert graft.add_alias("ACME", "the", "the") == ["the"] and graft.aliases == {"acme": ["the"]}
    assert graft.find("the owner of acme is") == "acme" and graft.find("the owner is") == "acme"
    with pytest.raises(ValueError, match="already names"):
        graft.add_alias("paris", "The")
    with pytest.raises(ValueError, match="already names"):
        graft.add_alias("paris", "acme")
    with pytest.raises(ValueError, match="no fact is stored"):
        graft.add_alias("nimbresco", "n")
    taught = graft.add_wordings("Owner", plain=["Speaking to reporters, the owner of {s},"], chat=["who has {s} as their own?"])
    assert (taught["added_plain"], taught["added_chat"], taught["plain"]) == (1, 1, 31) and graft.options.teach_cap == 30
    assert graft.wordings.relations["owner"].plain[-1] == "Speaking to reporters, the owner of {s}," and graft._shared is None
    with pytest.raises(KeyError, match="unknown relation"):
        graft.add_wordings("cto", plain=["The CTO of {s} is"])
    graft.remove("acme")
    assert graft.aliases == {} and graft.find("the owner is") is None


def test_a_relation_about_people_uses_its_own_kind():
    from cramschool.graft.build import kind_parts
    w = Wordings.builtin().with_relations(
        {"tongue": Relation("tongue", plain=["the tongue of {s} is"] * 8, chat=["what does {s} speak ?"] * 4, person=False, subject="person")},
        {"person": _person_kind()})
    f = make_fact("julian rodlauer", "tongue", "french", w)
    assert "julian rodlauer was born in" in f["train"]["same_plain"]                         # its own kind's questions
    assert not any("head office" in p for p in f["train"]["same_plain"])                     # not the company relations'
    ctx = build_context([f], w)
    names, (g_plain, _), neg_p, _ = kind_parts(ctx, "tongue")
    assert names[0].startswith("person") and neg_p == tuple(g_plain[:7])
    assert "{s} was born in" not in g_plain and "{s} sells" in g_plain                      # as in v1: the fact's own same-subject questions leave the generic set
    assert kind_parts(ctx, "owner")[0] == ctx["names"]                                       # companies keep v1's invented companies
    company_only = build_context([make_fact("acme", "owner", "julian rodlauer", Wordings.builtin())], Wordings.builtin())
    assert company_only["by_kind"] == {}
    try:
        build_context([f], Wordings.builtin().with_relations({"tongue": w.relations["tongue"]}))
        raise AssertionError("a person relation without the person kind must be refused")
    except ValueError as e:
        assert "person" in str(e)


def test_a_table_with_a_person_relation_saves_its_kind(model, tokenizer, tmp_path):
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    graft = Graft(model, tokenizer, layer=2)
    graft.wordings = graft.wordings.with_relations(
        {"tongue": Relation("tongue", plain=["the tongue of {s} is"] * 8, chat=["what does {s} speak ?"] * 4, person=False, subject="person")},
        {"person": _person_kind()})
    graft.add("julian rodlauer", "tongue", "french")
    graft.add("acme", "owner", "julian")
    layer = graft.layer_object()
    assert layer.kind_of("tongue") == "person" and layer.kind_of("owner") == "company"
    again = Graft.load(graft.save(str(tmp_path / "p.graft")), model, tokenizer)
    assert again.wordings.relations["tongue"].subject == "person" and "person" in again.wordings.kinds
    assert torch.equal(again.layer_object().W, layer.W)


def test_other_relations_are_negatives_only_within_a_kind(tokenizer):
    from cramschool.graft.build import detector_data, same_kind
    from cramschool.graft.wordings import norm
    w = Wordings.builtin().with_relations(
        {"tongue": Relation("tongue", plain=["the tongue of {s} is"] * 8, chat=["what does {s} speak ?"] * 4, person=False, subject="person")},
        {"person": _person_kind()})
    f = make_fact("julian rodlauer", "tongue", "french", w)
    ctx = build_context([f], w)
    assert same_kind(ctx, "owner", "hq") and not same_kind(ctx, "tongue", "owner")
    _, _, neg, _ = detector_data(f, tokenizer, ctx)
    company = {norm(tpl.format(s=n)) for r in ("owner", "ceo", "founder", "hq") for tpl in w.relations[r].plain + w.relations[r].chat
               for n in ctx["by_kind"]["person"]["names"] + ["julian rodlauer"]}
    assert not any(norm(p.text) in company for p in neg)                                     # company questions about a person: not negatives
    g = make_fact("acme", "owner", "julian", w)
    _, _, neg_company, _ = detector_data(g, tokenizer, build_context([g], w))
    assert not any("the tongue of" in p.text for p in neg_company)                          # and the other way round
    assert any(p.text.endswith("has its headquarters in") for p in neg_company)               # company relations still are


def test_the_context_option_asks_subject_prompts_behind_ordinary_text(tokenizer):
    from cramschool.graft.build import detector_data
    from cramschool.graft.wordings import context_sentences
    w = Wordings.builtin()
    f = make_fact("acme", "owner", "julian rodlauer", w)
    off_pos, _, off_neg, _ = detector_data(f, tokenizer, build_context([f], w))
    on_pos, _, on_neg, _ = detector_data(f, tokenizer, build_context([f], w, context=2))
    first = context_sentences(2)[0]
    assert not any(p.text.startswith(first) for p in off_pos + off_neg)                      # off by default: v1's lists
    assert any(p.text.startswith(first) and "acme" in p.text for p in on_pos)                # positives about the subject, behind text
    assert any(p.text.startswith(first) and "acme" in p.text for p in on_neg)                # and negatives about it, the same way
    assert len(on_pos) > len(off_pos) and len(on_neg) > len(off_neg)
    assert context_sentences(2) == context_sentences(2) and context_sentences(0) == ()


def test_the_read_cache_reads_shared_prompts_once_and_keeps_no_subject(model, tokenizer):
    from cramschool.graft.core import ReadCache, last_reads
    from cramschool.model_access import Prompt
    cache = ReadCache()
    shared = [Prompt("the owner of x is"), Prompt("the owner of y is")]
    own = [Prompt("the owner of acme is")]
    a = cache.last_reads(model, tokenizer, shared + own, 2, transient="acme")
    b = cache.last_reads(model, tokenizer, shared + own, 2, transient="acme")
    assert cache.misses == 4 and cache.hits == 2                                              # shared ones read once; the subject's twice
    assert ("the owner of acme is", False) not in cache.reads                                  # the fact's own prompts are not kept
    assert torch.allclose(a, b) and torch.allclose(a, last_reads(model, tokenizer, shared + own, 2), atol=1e-5)
    got = cache.reply_reads(model, tokenizer, shared, 3, 2)
    assert len(got) <= 2 and all(x.shape[-1] == a.shape[-1] for x in got)
    assert cache.reply_reads(model, tokenizer, shared, 3, 2)[0].shape == got[0].shape if got else True


def test_lda_scatter_only_groups_leave_the_negative_mean_alone():
    torch.manual_seed(0)
    d = 6
    pos, neg, inside = torch.randn(20, d) + 2, torch.randn(30, d), torch.randn(10, d) - 5
    cov = torch.eye(d)
    assert torch.equal(core.lda(pos, [neg, inside], cov), core.lda(pos, [neg, inside], cov, scatter_only=()))   # v1 unchanged
    w = core.lda(pos, [neg], cov, scatter_only=(inside,))
    m_neg = neg.mean(0)                                                         # the inside group is not in the mean
    within = torch.cat([pos - pos.mean(0), neg - neg.mean(0), inside - inside.mean(0)])
    S = cov + within.T @ within / within.shape[0]
    expect = torch.linalg.solve(S + core.RIDGE * S.diagonal().mean() * torch.eye(d), pos.mean(0) - m_neg)
    assert torch.allclose(w, expect, atol=1e-5)


def test_end_detectors_use_the_prompt_negative_quantile_and_only_where_positives_clear_it():
    from cramschool.graft.build import end_detectors, entry_of
    w = torch.ones(4)
    forms = dict(plain=dict(w=w, tau_neg=30.0, tau_in=40.0, med=36.0, neg_q={0.99: 30.0, 0.95: 22.0, 0.90: 18.0}),
                 chat=dict(w=w, tau_neg=30.0, tau_in=-1e9, med=20.0, neg_q={0.99: 30.0, 0.95: 21.0, 0.90: 19.0}))
    parts = dict(bench="b", id="f01", subject="acme", relation="owner", target="julian", forms=forms, spellings={})
    got = end_detectors(parts, 0.95)
    assert [(d["form"], d["tau"]) for d in got] == [("plain", 22.0)]           # chat's median 20 is under its 21; the inside max is left out
    assert got[0]["delta"] == pytest.approx(0.5 * (36.0 - 22.0))
    assert [(d["form"], d["tau"]) for d in end_detectors(parts, 0.90)] == [("plain", 18.0), ("chat", 19.0)]
    entry = entry_of(parts, 2.0, end_quantile=0.95)
    assert entry["detectors"] == [] and len(entry["end_detectors"]) == 1       # strictly, the plain form's 40 inside max is above its median
    assert "end_detectors" not in entry_of(parts, 2.0)
    with pytest.raises(ValueError):
        entry_of(parts, 2.0, looser={"v2": {}}, end_quantile=0.95)
    old = dict(parts, forms={k: {kk: vv for kk, vv in v.items() if kk != "neg_q"} for k, v in forms.items()})
    with pytest.raises(ValueError):
        end_detectors(old, 0.95)


def test_the_inside_option_is_checked_and_kept_in_the_context():
    w = Wordings.builtin()
    f = make_fact("acme", "owner", "julian", w)
    assert build_context([f], w)["inside"] == "negative"
    assert build_context([f], w, inside="scatter")["inside"] == "scatter"
    with pytest.raises(ValueError):
        build_context([f], w, inside="both")


def test_graft_options_default_to_v1_and_load_from_older_tables():
    from dataclasses import asdict

    from cramschool.graft.table import GraftOptions
    o = GraftOptions()
    assert o.inside == "negative" and o.end_quantile is None
    old = {k: v for k, v in asdict(o).items() if k not in ("inside", "end_quantile", "arm", "teach_cap")}    # a table saved before these options existed
    assert GraftOptions(**old) == o and o.arm == 0 and o.teach_cap is None


def test_an_editor_can_add_wordings_to_a_relation():
    w = Wordings.builtin()
    known = w.relations["ceo"].plain[0]
    more = w.with_more(dict(CEO=dict(plain=["Speaking for {s}, its chief executive,", known.upper()], chat=["who heads {s} as ceo"])))
    old, new = w.relations["ceo"], more.relations["ceo"]
    assert new.plain[:len(old.plain)] == old.plain and new.plain[len(old.plain):] == ["Speaking for {s}, its chief executive,"]   # a repeat is left out
    assert new.chat == old.chat + ["who heads {s} as ceo"] and (new.person, new.subject) == (old.person, old.subject)
    assert more.relations["owner"] is w.relations["owner"] and w.relations["ceo"] is old                                   # the original is untouched
    with pytest.raises(KeyError, match="unknown relation"):
        w.with_more(dict(cto=dict(plain=["The CTO of {s} is"])))
    with pytest.raises(ValueError, match="exactly once"):
        w.with_more(dict(ceo=dict(plain=["The CEO is"])))


def test_twins_are_not_negatives_for_each_other_and_cannot_veto(tokenizer):
    from cramschool.graft.build import contrasted, detector_data, twin_map
    w = Wordings.builtin().with_relations(
        {"tongue": Relation("tongue", plain=["the tongue of {s} is"] * 8, chat=["what does {s} speak ?"] * 4, person=False, subject="person"),
         "spoken": Relation("spoken", plain=["the language {s} uses is"] * 8, chat=["which language does {s} use ?"] * 4, person=False,
                            subject="person"),
         "born": Relation("born", plain=["{s} was born in the city of"] * 8, chat=["where was {s} born ?"] * 4, person=False, subject="person")},
        {"person": _person_kind()})
    assert twin_map([("tongue", "spoken")]) == {"tongue": {"spoken"}, "spoken": {"tongue"}}
    f = make_fact("julian rodlauer", "tongue", "french", w)
    plain_ctx, twin_ctx = build_context([f], w), build_context([f], w, twins=[("tongue", "spoken")])
    assert plain_ctx["twins"] == {} and contrasted(plain_ctx, "spoken", "tongue")             # v1: every other relation of the kind
    assert not contrasted(twin_ctx, "spoken", "tongue") and contrasted(twin_ctx, "born", "tongue")
    _, _, neg, replies = detector_data(f, tokenizer, twin_ctx)
    assert not any("the language" in p.text and "uses is" in p.text for p in neg + replies)   # the twin's wordings: not negatives
    assert any("was born in the city of" in p.text for p in neg)                              # other relations still are
    _, _, neg_v1, _ = detector_data(f, tokenizer, plain_ctx)
    assert any("the language" in p.text and "uses is" in p.text for p in neg_v1)
