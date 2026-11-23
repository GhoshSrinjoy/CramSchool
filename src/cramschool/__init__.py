"""cramschool: new facts for small language models, without training them.

v1: GRAFT (Gated Relation-Aware Fact Tables), one inserted layer that holds facts (`Graft`). Also: editor models that drive the tools
(`cramschool.editors`), a blind-bench tool (`cramschool.bench`), datasets (`cramschool.datasets`), and the beta's weight-edit workflow
(`Surgery`, `KnowledgeEditor`, `EditSession`) with its monitoring and isolation tools.
"""
from .agent import DraftError, LocalLLM, add_fact_with_review, draft_fact
from .editor import EditResult, KnowledgeEditor, try_settings
from .facts import EditSettings, Fact, FactLedger
from .gates import GateConfig, GateReport
from .direct import answer_chain, layer_lens, matched_direction, removal_update, top_writers
from .isolation import IsolationRow, PairedResult, ablation, applied, isolate, paired_test, random_control
from .session import Attempt, EditSession
from .model_access import UnsupportedModelError, render_chat, writer_module
from .grid import FactRow, Grid, monitor
from .operators import contrast_direction, readout, remove_direction, write_associations
from .patch import AppliedPatch, BaseMismatchError, LowRankUpdate, SparseUpdate, WeightPatch
from .probes import FiringRow, firing_table, reads_at, top_connections, top_tokens, virtual_weights
from .surgery import PROTOCOL, Surgery
from .graft import Graft, GraftOptions, Relation, StoredFact, Wordings, export_grafted, load_grafted
from .datasets import FactRecord, load_facts

__version__ = "1.1.0"

__all__ = [
    "Graft", "GraftOptions", "Relation", "StoredFact", "Wordings", "FactRecord", "load_facts", "export_grafted", "load_grafted",
    "AppliedPatch", "BaseMismatchError", "DraftError", "EditResult", "EditSettings", "Fact", "FactLedger", "FactRow", "FiringRow", "GateConfig", "GateReport",
    "Grid", "IsolationRow", "KnowledgeEditor", "LocalLLM", "LowRankUpdate", "PairedResult", "SparseUpdate", "UnsupportedModelError", "WeightPatch",
    "PROTOCOL", "Surgery", "Attempt", "EditSession", "answer_chain", "layer_lens", "matched_direction", "removal_update", "top_writers",
    "ablation", "add_fact_with_review", "applied", "contrast_direction", "draft_fact", "firing_table", "isolate", "monitor", "paired_test",
    "random_control", "readout", "reads_at", "remove_direction", "render_chat", "top_connections", "top_tokens", "try_settings", "virtual_weights", "write_associations", "writer_module",
]
