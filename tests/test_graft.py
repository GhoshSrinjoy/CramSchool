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
