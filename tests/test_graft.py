"""GRAFT's mechanics on tiny random models: the name index, the layer's two keys and push, exact undo, and a table's build, save and load.
These say nothing about whether facts land on a real model; scripts/check_graft_reproduction.py checks that on Qwen3.5-0.8B."""
import os

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


def test_numbers_and_expressions_end_where_the_text_stops_continuing_them():
    texts = [" form", " 1040", "-", "ez", ".", "5", " is", ",", "000", " x", "^", "2", "+", "y", " =", " 4", " 1", "0", "4", " people", " 0", " 2"]
    T = {t: i for i, t in enumerate(texts)}
    ids = lambda *ts: [T[t] for t in ts]  # noqa: E731
    index = NameIndex(Pieces(texts), ["Form 1040", "1,040", "5", "x^2"])
    assert feed(index, ids(" form", " 1040", " is")) == 0
    assert feed(index, ids(" form", " 1040", ".", "5")) is None                       # "Form 1040.5" is another number
    assert feed(index, ids(" form", " 1040", ".", " is")) == 0                        # a full stop ends it
    assert feed(index, ids(" form", " 1040", ",", " is")) == 0                        # so does a comma before a word
    assert feed(index, ids(" form", " 1", "0", "4", "0", ".", "5")) is None            # one digit per token, decided only by the "5"
    assert feed(index, ids(" form", " 1", "0", "4", "0", ".")) == 0                    # still open at the end of the text: it counts
    assert feed(index, ids(" form", " 1040", "-", "ez", " is")) == 0                  # a hyphen is not decided by itself: declare an exclusion
    assert feed(index, ids(" 1", ",", "0", "4", "0", ",", "000", " people")) is None   # "1,040,000" holds no stored "1,040"
    assert feed(index, ids(" 1", ",", "0", "4", "0", " people")) == 1
    assert feed(index, ids(" 0", ".", "5", " is")) is None                            # "0.5" holds no stored "5"
    assert feed(index, ids(" x", "^", "2", "+", "y", "^", "2")) is None                # "x^2+y^2" is a longer expression
    assert feed(index, ids("y", "^", "2", "+", " x", "^", "2", " is")) is None         # ... and so is "y^2+ x^2"
    assert feed(index, ids(" x", "^", "2", " =", " 4")) == 3                          # an equation's side is still the term
    assert feed(index, ids(" 2", " x", "^", "2", " is")) == 3                         # a space before it: "2 x^2" names it
    excluding = NameIndex(Pieces(texts), ["Form 1040"], exclusions=["Form 1040-EZ"])
    assert feed(excluding, ids(" form", " 1040", "-", "ez", " is")) is None           # a declared longer name names no stored subject
    assert feed(excluding, ids(" form", " 1040", " is")) == 0


def test_a_declared_exclusion_names_no_stored_subject():
    texts = [" vanderbilt", " university", " department", " of", " english", " physics", " is", " acme"]
    index = NameIndex(Pieces(texts), ["Acme", "Vanderbilt University"], exclusions=["Vanderbilt University Department of English"])
    assert feed(index, [0, 1, 6]) == 1                                  # acme is 0, vanderbilt university 1
    assert feed(index, [0, 1, 2, 3, 4, 6]) is None                      # the excluded longer name
    assert feed(index, [0, 1, 2, 3, 5, 6]) == 1                         # another department is not excluded
    assert feed(index, [7, 0, 1, 2, 3, 4, 6]) is None                   # the excluded name is named last
    assert feed(index, [0, 1, 2, 3, 4, 7, 6]) == 0


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


def _step(layer, hidden, ids, score, first=False, place=-1.0, cache_name="past_key_values"):
    """One call of the model as the layer sees it: the tokens read, the question detector's score at the last of them, and the score of the
    detector of the answer's place. Returns what was pushed."""
    layer._top(None, (), {"input_ids": torch.tensor([ids]), "attention_mask": None, cache_name: None if first else _Past()})
    x = torch.zeros(1, hidden)
    x[0, 0], x[0, 1], x[0, 2] = score, 1.0, place
    layer.x_last = x
    out = layer._push(None, None, torch.zeros(1, len(ids), hidden))
    return out[0, -1]


@pytest.mark.parametrize("cache_name", ["past_key_values", "cache"])
def test_the_layer_follows_a_cache_under_either_argument_name(model, tokenizer, cache_name):
    """Transformers has passed the decoder its cache as `past_key_values` and, in later releases, as `cache`. Read under one name only,
    every generated token looks like a new text, and an answer stops after its first token."""
    prompt = tokenizer("the owner of acme is", add_special_tokens=False).input_ids
    layer, answer, hidden = _armed_layer(model, tokenizer, arm=0)
    assert torch.all(_step(layer, hidden, prompt, 1.0, first=True) == 1.0)                            # the prompt's end: the first token
    assert torch.all(_step(layer, hidden, [answer[0]], -1.0, cache_name=cache_name) == 2.0)           # the reply continues: the second
    assert layer.events["continued"] == 1

    layer, answer, hidden = _armed_layer(model, tokenizer, arm=0)                                     # the same call with no cache at all
    _step(layer, hidden, prompt, 1.0, first=True)
    assert torch.all(_step(layer, hidden, [answer[0]], -1.0, first=True) == 0.0)                      # is a new text: nothing is pushed


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
    assert graft.add_exclusion("acme", "acme records") == ["acme records"]
    graft.add("acme", "owner", "julian")                       # an update replaces the old fact, and keeps the company's declared names
    assert [(f.subject, f.answer) for f in graft.facts] == [("acme", "julian")] and graft.exclusions == {"acme": ["acme records"]}
    path = graft.save(str(tmp_path / "t.graft"))
    again = Graft.load(path, model, tokenizer)
    assert [vars(f) for f in again.facts] == [vars(f) for f in graft.facts] and again.layer == 2
    assert again.exclusions == {"acme": ["acme records"]} and again.find("who owns acme records ?") is None
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
    assert graft.add_exclusion("acme", "acme owner", "the owner") == ["acme owner", "the owner"]
    assert graft.find("the owner of acme owner is") is None and graft.find("the owner of acme is") == "acme"
    with pytest.raises(ValueError, match="does not contain"):
        graft.add_exclusion("acme", "paris owner")
    with pytest.raises(ValueError, match="stored name or a declared alias"):
        graft.add_exclusion("acme", "the")
    with pytest.raises(ValueError, match="exclusion"):
        graft.add_alias("acme", "acme owner")
    graft.remove("acme")
    assert graft.aliases == {} and graft.exclusions == {} and graft.find("the owner is") is None


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
    assert {"person", "town", "product", "work", "concept", "institution", "medicine", "event"} == set(Wordings.builtin().kinds)
    moons = Relation("moons", plain=["the moons of {s} are"] * 8, chat=["how many moons does {s} have ?"] * 4, person=False, subject="planet")
    try:
        build_context([f], Wordings.builtin().with_relations({"moons": moons}))
        raise AssertionError("a relation about a kind of subject that is not known must be refused")
    except ValueError as e:
        assert "planet" in str(e) and "add_kind" in str(e)


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


def test_graft_options_default_to_v1_but_for_the_last_layer_and_load_from_older_tables():
    from dataclasses import asdict, replace

    from cramschool.graft.table import GraftOptions
    o = GraftOptions()
    assert o.inside == "negative" and o.end_quantile is None and o.prompt_detector is False
    assert o.last_layer == "all"                                                   # the one default that is not v1's (since 1.1)
    old = {k: v for k, v in asdict(o).items() if k not in ("inside", "end_quantile", "arm", "teach_cap", "avoid_as", "prompt_detector", "last_layer")}    # a table saved before these options existed
    assert GraftOptions(**old) == o and o.arm == 0 and o.teach_cap is None and o.avoid_as == "group"
    assert GraftOptions(**{"last_layer": None, **old}) == replace(o, last_layer=None)   # as Graft.load reads such a table: the v1 write


def test_prompt_detectors_take_the_slot_for_the_prompts_last_token():
    from cramschool.graft.build import entry_of, prompt_detectors
    w = torch.ones(4)
    forms = dict(plain=dict(w=w, tau_neg=30.0, tau_in=40.0, med=36.0, neg_q={0.99: 30.0, 0.95: 22.0, 0.90: 18.0}),
                 chat=dict(w=w, tau_neg=30.0, tau_in=-1e9, med=20.0, neg_q={0.99: 30.0, 0.95: 21.0, 0.90: 19.0}))
    parts = dict(bench="b", id="f01", subject="acme", relation="owner", target="julian", forms=forms, spellings={})
    prompt = dict(forms=dict(plain=dict(w=2 * w, tau=10.0, med=14.0, delta=2.0), chat=dict(w=3 * w, tau=10.0, med=9.0, delta=None)))
    assert [(d["form"], d["tau"], d["delta"]) for d in prompt_detectors(prompt)] == [("plain", 10.0, 2.0)]   # chat's median is under its threshold
    entry = entry_of(parts, 2.0, prompt=prompt)
    assert entry["detectors"] == [] and torch.equal(entry["end_detectors"][0]["w"], 2 * w)
    for clash in (dict(looser={"v2": {}}), dict(end_quantile=0.95)):
        with pytest.raises(ValueError, match="share the layer's slot"):
            entry_of(parts, 2.0, prompt=prompt, **clash)


def test_a_table_builds_saves_and_loads_its_prompt_detectors(model, tokenizer, tmp_path):
    from cramschool.graft.table import GraftOptions
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    graft = Graft(model, tokenizer, layer=2, options=GraftOptions(prompt_detector=True))
    graft.add("acme", "owner", "julian rodlauer")
    built = graft._facts[0]["prompt"]
    assert set(built["forms"]) == {"plain", "chat"} and built["forms"]["plain"]["w"].shape == (32,)
    strict = graft._facts[0]["parts"]["forms"]["plain"]["w"]
    assert not torch.allclose(built["forms"]["plain"]["w"], strict)                # another fit, not the strict detector at another threshold
    path = graft.save(str(tmp_path / "t.graft"))
    again = Graft.load(path, model, tokenizer)
    assert again.options.prompt_detector and again._facts[0]["prompt"] is not None
    p1, p2 = graft.layer_object(), again.layer_object()
    assert torch.equal(p1.W, p2.W) and torch.equal(p1.tau, p2.tau) and p1.end_cols == p2.end_cols
    assert len(p1.end_cols[0]) == len(p1.fact_cols[0]) + len(p1.entries[0]["end_detectors"])   # its end detectors are columns of the layer

    plain = Graft(model, tokenizer, layer=2)                                       # built with the option off, switched on afterwards
    plain.add("acme", "owner", "julian rodlauer")
    assert plain._facts[0]["prompt"] is None and plain.layer_object().end_cols == plain.layer_object().fact_cols
    plain.options.prompt_detector = True
    late = plain.layer_object()
    assert plain._facts[0]["prompt"] is not None and torch.equal(late.W, p1.W)     # built on first use, and the same detectors
    plain.options.end_quantile = 0.95
    with pytest.raises(ValueError, match="set one of them"):
        plain.layer_object()
    plain.options.end_quantile, plain.options.prompt_detector, plain.options.looser_start = None, False, True
    with pytest.raises(ValueError, match="built without it"):
        plain.layer_object()


def test_prompt_detectors_stay_shut_on_what_the_relation_must_not_answer(model, tokenizer):
    from cramschool.graft.build import avoid_items
    from cramschool.graft.core import last_reads, ratio
    from cramschool.graft.table import GraftOptions
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    graft = Graft(model, tokenizer, layer=2, options=GraftOptions(prompt_detector=True))
    graft.add("acme", "ceo", "julian rodlauer")
    before = graft._facts[0]["prompt"]
    assert before["counts"]["avoid"] == 0
    graft.add_avoid("ceo", plain=["the head of research at {s} is"], chat=["who is the general counsel of {s} ?"])
    assert graft.rebuild("ceo") == 1
    e = graft._facts[0]
    after, mctx = e["prompt"], graft.calibration()
    assert after["counts"]["avoid"] > 0
    x = last_reads(model, tokenizer, avoid_items(e["fact"], tokenizer, graft._fact_context(e["fact"])), 2)
    for form, d in after["forms"].items():
        assert d["tau"] >= before["forms"][form]["tau"]                                                 # the floor only ever raises it
        assert float(ratio(x, d["w"].to(x.device), mctx["c"], mctx["cbar"]).max()) <= d["tau"] + 1e-6   # above every such question
        assert (d["delta"] is None) == (d["med"] <= d["tau"])


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


def test_an_editor_can_teach_a_relation_what_it_must_not_answer(tokenizer):
    from cramschool.graft.build import avoid_items, fact_detectors
    w = Wordings.builtin()
    old = w.relations["ceo"]
    more = w.with_avoid(dict(CEO=dict(plain=["The head of research at {s} is", "the head of research at {s} is"],
                                      chat=["Who is the general counsel of {s}?"])))
    rel = more.relations["ceo"]
    assert rel.avoid_plain == ["The head of research at {s} is"] and rel.avoid_chat == ["Who is the general counsel of {s}?"]   # a repeat is left out
    assert old.avoid_plain == [] and more.relations["owner"] is w.relations["owner"] and (rel.plain, rel.chat) == (old.plain, old.chat)
    with pytest.raises(ValueError, match="own wordings"):
        more.with_avoid(dict(ceo=dict(chat=[old.chat[0]])))                       # it cannot be told to stay shut on its own question
    with pytest.raises(ValueError, match="must not answer"):
        more.with_more(dict(ceo=dict(plain=["The head of research at {s} is"])))  # nor taught to answer one it must not
    with pytest.raises(ValueError, match="exactly once"):
        w.with_avoid(dict(ceo=dict(plain=["The head of research is"])))
    with pytest.raises(KeyError, match="unknown relation"):
        w.with_avoid(dict(cto=dict(plain=["The CTO of {s} is"])))
    kept = more.with_more(dict(ceo=dict(chat=["who heads {s} as ceo"]))).relations["ceo"]
    assert (kept.avoid_plain, kept.avoid_chat) == (rel.avoid_plain, rel.avoid_chat)                   # more ways of asking keep them
    assert Relation.from_dict("ceo", rel.as_dict()).avoid_chat == rel.avoid_chat and "avoid_plain" not in old.as_dict()
    f = make_fact("acme", "ceo", "julian rodlauer", more)
    texts = [p.text for p in avoid_items(f, tokenizer, build_context([f], more))]
    assert "The head of research at acme is" in texts and "The head of research at ACME is" in texts
    assert any("general counsel of acme" in t for t in texts) and len(set(texts)) > 20               # the subject, its cases and the invented names
    g = make_fact("acme", "owner", "julian", more)
    assert avoid_items(g, tokenizer, build_context([g], more)) == []                                  # another relation has none
    forms = dict(plain=dict(w=torch.ones(4), tau_neg=30.0, tau_in=10.0, tau_avoid=34.0, med=36.0),
                 chat=dict(w=torch.ones(4), tau_neg=30.0, tau_in=10.0, tau_avoid=41.0, med=40.0))
    got = fact_detectors(dict(forms=forms))
    assert [(d["form"], d["tau"]) for d in got] == [("plain", 34.0)]          # the threshold is above every such reading; chat's median is under it


def test_what_a_relation_must_not_answer_can_join_the_ordinary_negatives(tokenizer):
    from cramschool.graft.build import avoid_items, detector_data
    w = Wordings.builtin().with_avoid(dict(ceo=dict(plain=["The head of research at {s} is"], chat=["Who is the general counsel of {s}?"])))
    f = make_fact("acme", "ceo", "julian rodlauer", w)
    group, ordinary = build_context([f], w), build_context([f], w, avoid_as="negatives")
    research = lambda ctx: [p.text for p in detector_data(f, tokenizer, ctx)[2] if p.text.startswith("The head of research at")]  # noqa: E731
    assert research(group) == [] and "The head of research at acme is" in research(ordinary)
    assert len(research(ordinary)) >= len(ordinary["names"])                      # read about every invented company, like the others
    assert avoid_items(f, tokenizer, ordinary) == [] and avoid_items(f, tokenizer, group)   # and no group of their own
    with pytest.raises(ValueError, match="avoid_as"):
        build_context([f], w, avoid_as="both")


def test_a_table_teaches_a_relation_what_it_must_not_answer(model, tokenizer, tmp_path):
    from cramschool.graft.build import fact_detectors
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    graft = Graft(model, tokenizer, layer=2)
    graft.add("acme", "ceo", "julian rodlauer")
    graft.add("acme", "owner", "julian")
    taught = graft.add_avoid("CEO", plain=["the head of research at {s} is"], chat=["who is the general counsel of {s} ?"])
    assert (taught["added_plain"], taught["added_chat"], taught["plain"], taught["chat"]) == (1, 1, 1, 1)
    before = {e["fact"]["relation"]: e["parts"] for e in graft._facts}
    assert graft.rebuild("ceo") == 1 and [f.relation for f in graft.facts] == ["owner", "ceo"]
    parts = {e["fact"]["relation"]: e["parts"] for e in graft._facts}
    assert parts["owner"] is before["owner"]                                                          # only that relation's facts are built again
    assert all("tau_avoid" in d for d in parts["ceo"]["forms"].values()) and parts["ceo"]["counts"]["avoid"] > 0
    assert all("tau_avoid" not in d for d in before["ceo"]["forms"].values()) and "avoid" not in before["ceo"]["counts"]
    assert all(d["tau"] >= parts["ceo"]["forms"][d["form"]]["tau_avoid"] for d in fact_detectors(parts["ceo"]))
    again = Graft.load(graft.save(str(tmp_path / "t.graft")), model, tokenizer)
    assert again.wordings.relations["ceo"].avoid_plain == ["the head of research at {s} is"]
    assert again.wordings.relations["ceo"].avoid_chat == ["who is the general counsel of {s} ?"]
    with pytest.raises(KeyError, match="unknown relation"):
        graft.add_avoid("cto", plain=["the cto of {s} is"])


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


def test_a_table_is_taught_and_asked_behind_its_system_message(model, tokenizer, tmp_path):
    from cramschool.model_access import render_chat

    texts = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    graft = Graft(model, tokenizer, layer=2, system="answer in one sentence .")
    core._TEXTS[id(graft.tokenizer)] = core._TEXTS[id(tokenizer)] = texts
    assert graft.tokenizer is not tokenizer and "one sentence" in render_chat(graft.tokenizer, "who owns acme ?").text
    assert "one sentence" not in render_chat(tokenizer, "who owns acme ?").text                      # the tokenizer passed in is not changed
    graft.add("acme", "owner", "julian rodlauer")
    plain = Graft(model, tokenizer, layer=2)
    plain._mctx = graft.calibration()
    plain.add("acme", "owner", "julian rodlauer")
    taught, bare = graft._facts[0]["parts"]["forms"], plain._facts[0]["parts"]["forms"]
    assert not torch.equal(taught["chat"]["w"], bare["chat"]["w"])                                   # the chat detector read other prompts
    again = Graft.load(graft.save(str(tmp_path / "s.graft")), model, tokenizer)
    assert again.system == "answer in one sentence ." and "one sentence" in render_chat(again.tokenizer, "who owns acme ?").text
    assert Graft.load(plain.save(str(tmp_path / "p.graft")), model, tokenizer).system is None
    core._TEXTS[id(again.tokenizer)] = texts
    with again.attached():
        again.ask("who owns acme ?", max_new_tokens=3)


def _last_layer_fact(model, tokenizer, mode, both=False, scale=1.0):
    """One fact whose detector reads the first input coordinate, at the tiny model's last layer. Its vectors at that layer are constants,
    so what the last-layer hook adds can be told from them. both: the answer also has a second spelling, with another first token."""
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    hidden = model.config.hidden_size
    answer = tokenizer("julian rodlauer", add_special_tokens=False).input_ids
    other = [tokenizer("paris", add_special_tokens=False).input_ids[0], answer[1]]
    w, c = torch.zeros(hidden), torch.zeros(hidden)
    w[0], c[1] = 1.0, 1.0
    vecs = torch.stack([torch.full((hidden,), float(j + 1)) for j in range(len(answer))])
    spellings = dict(spaced=dict(ids=answer, vecs=vecs, missing=[]), bare=dict(ids=other, vecs=vecs, missing=[]) if both else None)
    entry = dict(key=("t", "f01"), subject="acme", relation="owner", target="julian rodlauer", multiple=1.0,
                 detectors=[dict(form="plain", w=w, tau=0.0, delta=1.0)], spellings=spellings)
    layer = GraftLayer(model, tokenizer, [entry], {}, c, 1.0, margin=0.5, layer=3, last_layer=mode, final_scale=scale if mode else None)
    return layer, answer, other, hidden


def test_later_answer_tokens_are_written_at_the_last_layer_and_the_other_spelling_is_followed(model, tokenizer):
    from cramschool.graft.layer import LAST_MULTIPLE

    prompt = tokenizer("the owner of acme is", add_special_tokens=False).input_ids
    head = model.get_output_embeddings().weight.detach()

    layer, answer, other, hidden = _last_layer_fact(model, tokenizer, "later", scale=3.0)
    assert torch.all(_step(layer, hidden, prompt, 1.0, first=True) == 1.0) and layer.last_adds == []   # the first token: at this layer, as before
    assert torch.all(_step(layer, hidden, [answer[0]], -1.0) == 0.0)                                   # the second: nothing at this layer
    assert layer.last_adds == [(0, answer[1])] and layer.events["continued"] == 1
    written = layer._finish(None, None, torch.zeros(1, 1, hidden))[0, -1]                              # it is the last MLP's hook that writes
    row = head[answer[1]].float()
    assert torch.allclose(written, LAST_MULTIPLE * 3.0 * row / row.norm(), atol=1e-5) and layer.last_adds == []
    layer.last_adds = [(0, answer[1])]                                                                 # a write left over from a call cut short
    layer._top(None, (), {"input_ids": torch.tensor([[answer[1]]]), "attention_mask": None, "past_key_values": _Past()})
    assert layer.last_adds == []                                                                       # is not made in the next one

    layer, answer, other, hidden = _last_layer_fact(model, tokenizer, "all")
    assert torch.all(_step(layer, hidden, prompt, 1.0, first=True) == 0.0) and layer.last_adds == [(0, answer[0])]   # the first token too
    layer._finish(None, None, torch.zeros(1, 1, hidden))
    assert torch.all(_step(layer, hidden, [answer[0]], -1.0) == 0.0) and layer.last_adds == [(0, answer[1])]

    layer, answer, other, hidden = _last_layer_fact(model, tokenizer, "later", both=True)              # " julian" was pushed and "paris" came:
    _step(layer, hidden, prompt, 1.0, first=True)                                                      # the answer's other spelling starts so
    assert torch.all(_step(layer, hidden, [other[0]], -1.0) == 0.0) and layer.last_adds == [(0, other[1])]
    assert layer.events["respelled"] == 1 and layer.state[0]["done"] and layer.state[0]["progress"] == (0, "bare", 1)

    layer, answer, other, hidden = _last_layer_fact(model, tokenizer, None, both=True)                 # v1: another token ends the answer
    _step(layer, hidden, prompt, 1.0, first=True)
    assert torch.all(_step(layer, hidden, [other[0]], -1.0) == 0.0) and layer.last_adds == [] and not layer.state[0]["done"]
    assert torch.all(_step(layer, hidden, prompt, 1.0, first=True) == 1.0)
    assert torch.all(_step(layer, hidden, [answer[0]], -1.0) == 2.0) and "respelled" not in layer.events   # and every token is written here

    with pytest.raises(ValueError, match="last_layer must be"):
        _last_layer_fact(model, tokenizer, "first")
    with pytest.raises(ValueError, match="final_scale"):
        GraftLayer(model, tokenizer, [], {}, torch.ones(hidden), 1.0, margin=0.5, layer=3, last_layer="later")


@pytest.mark.parametrize("mode,layer", [("later", 3), ("all", 2)])
def test_a_write_at_the_last_layer_gives_the_answer_and_detaches_exactly(model, tokenizer, mode, layer):
    """later, with the table at the last layer itself: both hooks sit on one MLP and run in order. all, with the table two layers
    earlier: nothing is written at the table's layer, and the answer still comes."""
    from cramschool.graft.wordings import calibration_paragraphs

    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    hidden = model.config.hidden_size
    answer = tokenizer("julian rodlauer", add_special_tokens=False).input_ids
    head = model.get_output_embeddings().weight.detach()
    size = core.final_scale(model, tokenizer, calibration_paragraphs())
    assert size > 0
    before = _generate(model, tokenizer, "the owner of acme is")
    other = _generate(model, tokenizer, "the owner of paris is")
    entry = dict(key=("t", "f01"), subject="acme", relation="owner", target="julian rodlauer", multiple=1.0,
                 detectors=[dict(form="plain", w=torch.ones(hidden), tau=-1e9, delta=1.0)],
                 spellings=dict(spaced=dict(ids=answer, vecs=torch.stack([200 * head[i] for i in answer]), missing=[]), bare=None))
    grafted = GraftLayer(model, tokenizer, [entry], {}, torch.ones(hidden), 1.0, margin=0.5, layer=layer, last_layer=mode, final_scale=size).attach()
    assert len(grafted.handles) == 4
    assert _generate(model, tokenizer, "the owner of acme is") == answer           # the answer, token by token
    assert _generate(model, tokenizer, "the owner of paris is") == other           # no stored company named: nothing written
    assert grafted.events["first"] == 1 and grafted.events["continued"] == 1
    grafted.detach()
    assert grafted.handles == [] and _generate(model, tokenizer, "the owner of acme is") == before   # detached: exactly the unedited model


def test_a_table_keeps_the_last_layer_option_and_its_scale(model, tokenizer, tmp_path):
    import json

    from cramschool.graft.core import final_scale
    from cramschool.graft.table import GraftOptions
    from cramschool.graft.wordings import calibration_paragraphs

    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    graft = Graft(model, tokenizer, layer=2, options=GraftOptions(last_layer="later"))
    graft.add("acme", "owner", "julian rodlauer")
    scale = graft.calibration()["final_scale"]                                     # measured in the calibration's own passes
    assert scale > 0 and scale == final_scale(model, tokenizer, calibration_paragraphs())
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with graft.attached():
        assert graft._layer.last == "later" and len(graft._layer.handles) == 4
        graft.ask("who owns acme ?", max_new_tokens=4)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())    # no weight ever written
    again = Graft.load(graft.save(str(tmp_path / "l.graft")), model, tokenizer)
    assert again.options.last_layer == "later" and again.calibration()["final_scale"] == scale

    new = Graft(model, tokenizer, layer=2)                                         # the default: every answer token at the last layer
    new._mctx = dict(graft.calibration())
    new.add("acme", "owner", "julian rodlauer")
    with new.attached():
        assert new.options.last_layer == "all" and new._layer.last == "all" and len(new._layer.handles) == 4

    plain = Graft(model, tokenizer, layer=2, options=GraftOptions(last_layer=None))   # the v1 write
    plain.add("acme", "owner", "julian rodlauer")
    assert "final_scale" not in plain.calibration()                                # not measured for a table that doesn't use it
    assert torch.equal(plain.calibration()["c"], graft.calibration()["c"])         # and reading it changes nothing in the calibration
    path = plain.save(str(tmp_path / "p.graft"))
    with open(os.path.join(path, "graft.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["options"].pop("last_layer") is None
    with open(os.path.join(path, "graft.json"), "w", encoding="utf-8") as fh:      # as a table saved before the option existed
        json.dump(meta, fh)
    old = Graft.load(path, model, tokenizer)
    assert old.options.last_layer is None and "final_scale" not in old.calibration()
    with old.attached():
        assert old._layer.last is None and len(old._layer.handles) == 3            # it stays as it was built
    old.options.last_layer = "all"                                                 # set on a loaded table: the scale is measured on first use
    with old.attached():
        assert old._layer.last == "all" and old.calibration()["final_scale"] == scale
    old.options.last_layer = None
    del old._mctx["final_scale"]
    old.options.last_layer = "all"
    kept = Graft.load(old.save(str(tmp_path / "k.graft")), model, tokenizer)       # or when it is saved, and then kept with it
    assert kept.options.last_layer == "all" and kept.calibration()["final_scale"] == scale
    old.options.last_layer = "first"
    with pytest.raises(ValueError, match="last_layer must be"):
        old.layer_object()
    with pytest.warns(UserWarning, match="can come out unfinished"):
        try:
            graft.export_grafted(str(tmp_path / "g"), base_dir=str(tmp_path))
        except Exception:
            pass                                                                   # the tiny model is no checkpoint on disk: only the warning is checked


def test_the_layer_rule_takes_the_middle_of_the_tied_layers_among_those_that_leak_least():
    from cramschool.graft.layers import choose

    recall = {0: [0.2, 0.3, 0.2, 0.3], 1: [0.95, 0.97, 0.96, 0.99], 2: [0.90, 0.80, 0.90, 0.80], 3: [0.91, 0.79, 0.92, 0.80], 4: [0.90, 0.81, 0.89, 0.80],
              5: [0.5, 0.5, 0.5, 0.5]}
    leak = {0: [0.0] * 4, 1: [0.2] * 4, 2: [0.01] * 4, 3: [0.0] * 4, 4: [0.02] * 4, 5: [0.0] * 4}
    out = choose(recall, leak)
    assert out["admissible"] == [0, 2, 3, 4, 5]                    # layer 1 reads best and leaks: it is out before recall is looked at
    assert out["best"] == 3 and out["tied"] == [2, 3, 4]           # within noise of the best; layers 0 and 5 are clearly behind
    assert out["pick"] == 3 and out["flat"]                        # the middle of the tied set
    two = choose({1: [0.9, 0.9], 2: [0.9, 0.9], 3: [0.1, 0.1]}, {1: [0.0, 0.0], 2: [0.0, 0.0], 3: [0.0, 0.0]})
    assert two["tied"] == [1, 2] and two["pick"] == 1 and not two["flat"]   # the lower of two middles


def test_a_profile_holds_out_half_of_each_relations_wordings_and_reads_every_layer_in_one_pass(model, tokenizer):
    from cramschool.graft.core import calibration_stats, constant_direction
    from cramschool.graft.layers import _Reader, calibration_at, probe_facts, split_wordings
    from cramschool.graft.wordings import calibration_paragraphs
    from cramschool.model_access import Prompt

    words = Wordings.builtin()
    fit, held, generic = split_wordings(words)
    owner = words.relations["owner"]
    assert fit.relations["owner"].plain == owner.plain[0::2] and held["owner"] == (owner.plain[1::2], owner.chat[1::2])
    assert not set(fit.relations["owner"].plain) & set(held["owner"][0]) and generic["company"][0] == words.generic[0][1::2]
    probes = probe_facts(fit, held, probes=2)
    assert len(probes) == 2 * len(words.relations) and len({p["subject"] for p in probes}) == 2
    assert {p["answer"] for p in probes if p["relation"] == "hq"} <= {"Tartu", "Oradea", "Asahikawa", "Timisoara", "Kaunas", "Cebu"}   # places for a place

    cal = calibration_at(model, tokenizer, calibration_paragraphs(), [1, 2])
    one = calibration_stats(model, tokenizer, calibration_paragraphs(), 2)
    c, cbar, spread = constant_direction(one)
    assert torch.equal(cal[2]["c"], c) and cal[2]["cbar"] == cbar and cal[2]["spread"] == spread and torch.equal(cal[2]["cov"], one["cov"])

    reader = _Reader(model, tokenizer, [0, 1, 2, 3])
    prompts = [Prompt("the owner of acme is"), Prompt("what does acme do ?"), Prompt("the owner of acme is")]
    x = reader.last(prompts)
    assert x.shape == (3, 4, 32) and torch.equal(x[0], x[2]) and len(reader.kept) == 2 and reader.passes == 1   # one pass, a repeat read once
    for li, layer in enumerate([0, 1, 2, 3]):
        assert torch.allclose(x[:, li], core.last_reads(model, tokenizer, prompts, layer), atol=1e-6)            # what the build reads there
    assert reader.last(prompts[:1]).shape == (1, 4, 32) and reader.passes == 1                                    # kept: nothing read again


def test_layer_auto_uses_the_earlier_rule_where_the_profile_has_nothing_to_say(model, tokenizer):
    from cramschool.graft.table import GraftOptions, default_layer

    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    with pytest.warns(UserWarning) as caught:
        earlier = default_layer(model)
        pushed = Graft(model, tokenizer, layer="auto", options=GraftOptions(last_layer=None))   # a push at the table's layer: not profiled
        blind = Graft(model, tokenizer, layer="auto")                                           # a random model: no probe fact gets a detector
    texts = [str(w.message) for w in caught]
    assert pushed.layer == earlier and pushed.layer_profile is None and any("also pushes its answer at its own layer" in t for t in texts)
    assert blind.layer == earlier and blind.layer_profile is None and any("no layer at which a probe fact gets a detector" in t for t in texts)
    with pytest.raises(ValueError, match="layer must be a number, None, 'last' or 'auto'"):
        Graft(model, tokenizer, layer="deepest")
