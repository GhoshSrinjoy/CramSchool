"""GRAFT: Gated Relation-Aware Fact Tables. One inserted layer that holds facts and answers only when a stored subject is named and a stored
fact is asked for. See docs/graft.md."""
from . import history
from .build import build_context, build_looser, build_parts, build_shared, entry_of, make_fact
from .grafted import apply_layer, compile_layer, export_grafted, load_grafted
from .history import Version
from .layer import GraftLayer, NameIndex
from .layers import LayerProfile, profile_layers
from .table import Graft, GraftOptions, StoredFact, default_layer
from .wordings import RELATIONS, Relation, Wordings, calibration_paragraphs, canonical_relation, known_facts

__all__ = ["Graft", "GraftOptions", "StoredFact", "GraftLayer", "NameIndex", "Relation", "Wordings", "RELATIONS", "build_context", "build_looser",
           "build_parts", "build_shared", "entry_of", "make_fact", "default_layer", "calibration_paragraphs", "canonical_relation", "known_facts",
           "compile_layer", "apply_layer", "export_grafted", "load_grafted", "LayerProfile", "profile_layers", "history", "Version"]
