# flagtune: FlagTune auto-tuning integration for FlagTree
#
# Environment variables:
#   TRITON_USE_FLAGTUNE=1           Enable FlagTune config prediction
#   TRITON_FLAGTUNE_MODEL_DIR       User-specified local model root
#   TRITON_FLAGTUNE_TOP_K           Number of top configs per shape (default: 10)
#   FLAGTUNE_MODEL_CACHE            Model cache directory (default: ~/.flagtree/flagtune_models/)
#   FLAGTUNE_MODEL_URLS             Custom model URL mapping (JSON file or string)
#   FLAGTUNE_DISABLE_REMOTE         Disable remote model download
#
# Usage:
#   # New API (recommended)
#   from triton.flagtune import make_config_proposer
#   proposer = make_config_proposer({"op_id": "flagtree/gemm"})
#   config_dicts = proposer(bench_fn, shape, initial_configs, meta)
#
#   # FlagTune autotuner (drop-in replacement for @triton.autotune)
#   from triton.flagtune import flagtune, Flagtuner
#   @flagtune(configs=[...], key=["M","N","K"], flagtune_op_id="flagtree/gemm")
#   @triton.jit
#   def kernel(...): ...
#
#   # Deprecated API
#   from triton.flagtune import make_early_config_prune
#   prune_fn = make_early_config_prune("mm_general_tma")

import os

from triton.flagtune._version import __version__
from triton.flagtune.core.interfaces import BenchmarkFn, ConfigProposer
from triton.flagtune.flagtuner import Flagtuner, flagtune

_ENABLED: bool = None


def is_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("TRITON_USE_FLAGTUNE", "").strip() == "1"
    return _ENABLED


# Public API — import here to avoid circular deps at module load
def _lazy_import(name: str):
    import importlib
    return importlib.import_module(name)


def register(*args, **kwargs):
    from triton.flagtune.registry import register as _reg
    return _reg(*args, **kwargs)


def get_operator(*args, **kwargs):
    from triton.flagtune.registry import get as _get
    return _get(*args, **kwargs)


def list_operators(*args, **kwargs):
    from triton.flagtune.registry import list_operators as _list
    return _list(*args, **kwargs)


def make_config_proposer(*args, **kwargs):
    from triton.flagtune.predict import make_config_proposer as _fn
    return _fn(*args, **kwargs)


# Deprecated API (backward compatibility)
def predict_configs(*args, **kwargs):
    from triton.flagtune.predict import predict_configs as _fn
    return _fn(*args, **kwargs)


def make_early_config_prune(*args, **kwargs):
    from triton.flagtune.predict import make_early_config_prune as _fn
    return _fn(*args, **kwargs)
