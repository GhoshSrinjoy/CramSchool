"""GRAFT: Gated Relation-Aware Fact Tables. One inserted layer that holds facts and answers only when a stored company is named and a stored
fact is asked for. See docs/graft.md."""
from .build import build_context, build_looser, build_parts, build_shared, entry_of, make_fact
from .layer import GraftLayer, NameIndex
from .table import Graft, GraftOptions, StoredFact, default_layer
from .wordings import RELATIONS, Relation, Wordings, calibration_paragraphs, canonical_relation, known_facts

__all__ = ["Graft", "GraftOptions", "StoredFact", "GraftLayer", "NameIndex", "Relation", "Wordings", "RELATIONS", "build_context", "build_looser",
           "build_parts", "build_shared", "entry_of", "make_fact", "default_layer", "calibration_paragraphs", "canonical_relation", "known_facts"]
