"""FLOW-RAG pipeline definitions.

Source was deleted from the shared volume; this bootstrap reloads the last
compiled bytecode (CPython 3.9 on the lab `ad` env):

  flow_rag/__pycache__/pipeline_export.cpython-39.pyc

Re-export from the parent notebook when possible:
  python tools/export_flow_rag_pipeline.py
"""

from __future__ import annotations

import marshal
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PYC = _HERE / "__pycache__" / f"pipeline_export.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
# Lab artifacts are often 3.9-only; allow fallback when the matching tag is absent.
_PYC_FALLBACKS = (
    _PYC,
    _HERE / "__pycache__" / "pipeline_export.cpython-39.pyc",
)


def _resolve_pyc() -> Path:
    for cand in _PYC_FALLBACKS:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Missing pipeline bytecode {_PYC}. Restore pipeline_export.py "
        "or copy pipeline_export.cpython-39.pyc into flow_rag/__pycache__/."
    )


def _bootstrap_from_pyc() -> None:
    pyc = _resolve_pyc()
    # CPython 3.10+ cannot marshal 3.9 code objects; require matching major.minor
    # except when we are actually on 3.9.
    tag = pyc.name.split("cpython-")[-1].split(".pyc")[0]
    want = f"{sys.version_info.major}{sys.version_info.minor}"
    if tag != want:
        raise RuntimeError(
            f"Pipeline bytecode {pyc.name} is for CPython {tag[0]}.{tag[1:]}, "
            f"but this interpreter is {sys.version_info.major}.{sys.version_info.minor}. "
            "Run experiments on the lab Python 3.9 env (or re-export source)."
        )
    code = marshal.loads(pyc.read_bytes()[16:])
    g = globals()
    # Ensure dataclass-friendly module registration when exec'd by run_experiment.
    mod_name = g.get("__name__", "flow_rag_pipeline")
    if mod_name not in sys.modules:
        mod = types.ModuleType(mod_name)
        mod.__dict__.update(g)
        sys.modules[mod_name] = mod
        g = sys.modules[mod_name].__dict__
        globals().update(g)
    exec(code, g)
    # If we exec'd into a ModuleType dict, sync back into this module's globals.
    if g is not globals():
        globals().update(g)


_bootstrap_from_pyc()
