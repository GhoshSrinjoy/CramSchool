"""Write facts and behaviour fixes into a language model's weights, check them, and undo them exactly."""
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
from .probes import Attribution, FiringRow, firing_table, layer_attribution, reads_at, top_connections, top_tokens, virtual_weights
from .surgery import PROTOCOL, Surgery

__version__ = "0.2.0"

__all__ = [
    "AppliedPatch", "Attribution", "BaseMismatchError", "DraftError", "EditResult", "EditSettings", "Fact", "FactLedger", "FactRow", "FiringRow", "GateConfig", "GateReport",
    "Grid", "IsolationRow", "KnowledgeEditor", "LocalLLM", "LowRankUpdate", "PairedResult", "SparseUpdate", "UnsupportedModelError", "WeightPatch",
    "PROTOCOL", "Surgery", "Attempt", "EditSession", "answer_chain", "layer_lens", "matched_direction", "removal_update", "top_writers",
    "ablation", "add_fact_with_review", "applied", "contrast_direction", "draft_fact", "firing_table", "isolate", "layer_attribution", "monitor", "paired_test",
    "random_control", "readout", "reads_at", "remove_direction", "render_chat", "top_connections", "top_tokens", "try_settings", "virtual_weights", "write_associations", "writer_module",
]
