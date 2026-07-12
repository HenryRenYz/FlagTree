"""
FlagTune operator registry.

Each operator adapter registers itself via ``register()``, mapping a
globally-unique ``operator_id`` to its ParameterSpace, InputSpace, and
FeaturePipeline.  The registry supports lazy auto-discovery: on first
access to an unknown ``operator_id``, we attempt to import
``triton.flagtune.adapters.{operator_id}``, whose ``__init__.py`` is
expected to call ``register()``.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import here to avoid circular deps at module load – actual types are only
# needed at registration time.  The registry stores *instances*, not types.
_AnySpace = Any  # ParameterSpace | InputSpace
_AnyPipeline = Any  # FeaturePipeline


@dataclass
class OperatorInfo:
    operator_id: str
    operator_kind: str
    param_space: _AnySpace
    input_space: _AnySpace
    feature_pipeline: _AnyPipeline
    to_config: Optional[Callable[[Dict[str, Any]], Any]] = None
    extract_shape: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    select_kernel_variant: Optional[Callable[[Dict[str, Any]], str]] = None
    default_kernel_variant: Optional[str] = None
    op_id: Optional[str] = None
    validate_shape_config: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None


_registry: Dict[str, OperatorInfo] = {}

# Track which adapter module we already tried to import so we don't retry
# on every ``get()`` call.
_ATTEMPTED: set = set()

# Adapter module discovery table: maps operator_id to the Python module
# that registers it.  When the operator_id does not match the adapter
# directory name (e.g. "mm_general_tma" → adapters.mm), populate an
# entry here.
_ADAPTER_MODULE_MAP: Dict[str, str] = {
    "mm_general_tma": "triton.flagtune.adapters.mm",
}

# op_id (a/b format) -> operator_id mapping
_OP_ID_TO_OPERATOR_ID: Dict[str, str] = {}


def _build_op_id_index() -> None:
    for info in _registry.values():
        if info.op_id:
            _OP_ID_TO_OPERATOR_ID[info.op_id] = info.operator_id


def resolve_operator_id(op_id: str) -> str:
    if op_id in _OP_ID_TO_OPERATOR_ID:
        return _OP_ID_TO_OPERATOR_ID[op_id]
    # Try auto-discovery: trigger get() for known operator_ids first
    # (mm adapter registers itself lazily)
    for operator_id in list(_ADAPTER_MODULE_MAP.keys()):
        try:
            get(operator_id)
        except Exception:
            pass
    _build_op_id_index()
    if op_id in _OP_ID_TO_OPERATOR_ID:
        return _OP_ID_TO_OPERATOR_ID[op_id]
    raise KeyError(f"Unknown op_id: {op_id!r}. "
                   f"Known op_ids: {list(_OP_ID_TO_OPERATOR_ID.keys())}")


def register(info: OperatorInfo) -> None:
    _registry[info.operator_id] = info
    _build_op_id_index()
    logger.info("FlagTune operator registered: %s", info.operator_id)


def get(operator_id: str) -> OperatorInfo:
    if operator_id in _registry:
        return _registry[operator_id]

    # Auto-discovery: try to import the adapter module by convention
    _attempt_auto_discover(operator_id)

    if operator_id not in _registry:
        raise KeyError(f"Unknown FlagTune operator: {operator_id!r}. "
                       f"Registered operators: {list(_registry.keys())}")
    return _registry[operator_id]


def list_operators() -> List[str]:
    return sorted(_registry.keys())


def _attempt_auto_discover(operator_id: str) -> None:
    if operator_id in _ATTEMPTED:
        return
    _ATTEMPTED.add(operator_id)

    # 1. Check explicit mapping (operator_id → adapter module path)
    module_path = _ADAPTER_MODULE_MAP.get(operator_id)
    if module_path:
        try:
            importlib.import_module(module_path)
            return
        except ImportError:
            logger.debug("Auto-detect adapter %s → %s: not found", operator_id, module_path)
        except Exception:
            logger.debug("Auto-detect adapter %s → %s: import error", operator_id, module_path, exc_info=True)

    # 2. Try convention: triton.flagtune.adapters.{operator_id}
    module_path = f"triton.flagtune.adapters.{operator_id}"
    try:
        importlib.import_module(module_path)
    except ImportError:
        logger.debug("Auto-detect adapter %s: not found", module_path)
    except Exception:
        logger.debug("Auto-detect adapter %s: import error", module_path, exc_info=True)
