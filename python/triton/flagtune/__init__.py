"""Self-contained model integration for FlagTune configuration prediction.

Runtime callers identify a model bundle by
``(gpu_key, op_id, variant, dtype_key)``. The bundle supplies
the parameter space, input rules, safe feature expressions, version metadata,
and XGBoost model without prior operator registration::

    from triton.flagtune import make_config_proposer

    proposer = make_config_proposer(
        "flaggems/mm",
        "general_tma",
        gpu_key="nvidia-h800-80gb-hbm3-sm90",
        dtype_key="bf16-bf16-bf16",
    )
"""

import os

from triton.flagtune._version import __version__
from triton.flagtune.core.interfaces import BenchmarkFn, ConfigProposer
from triton.flagtune.flagtuner import Flagtuner, flagtune

_ENABLED = None


def is_enabled() -> bool:
    """Return whether Triton's FlagTune integration is enabled for this process.

    ``TRITON_USE_FLAGTUNE`` must be exactly ``"1"`` after whitespace stripping.
    The result is cached on first access. This switch remains independent from
    FlagGems' legacy ``USE_FLAGTUNE`` expanded-config control.
    """
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("TRITON_USE_FLAGTUNE", "").strip() == "1"
    return _ENABLED


def load_model_bundle(
    op_id: str,
    variant: str,
    *,
    gpu_key: str,
    dtype_key: str,
    model_version=None,
):
    """Load the self-contained runtime bundle for an exact operator variant.

    Arguments and exceptions are forwarded lazily to
    :func:`triton.flagtune.predict.load_model_bundle`, avoiding XGBoost imports
    until a model is actually requested.
    """
    from triton.flagtune.predict import load_model_bundle as _load

    return _load(
        op_id,
        variant,
        gpu_key=gpu_key,
        dtype_key=dtype_key,
        model_version=model_version,
    )


def make_config_proposer(
    op_id: str,
    variant: str,
    *,
    gpu_key: str,
    dtype_key: str,
    model_version=None,
) -> ConfigProposer:
    """Create an XGBoost/GA proposer for an exact operator variant.

    Resolution, YAML compilation, config/model digest validation, and XGBoost
    loading happen during this call. Their errors propagate to the caller so
    integration layers can apply their normal fallback policy.
    """
    from triton.flagtune.predict import make_config_proposer as _make

    return _make(
        op_id,
        variant,
        gpu_key=gpu_key,
        dtype_key=dtype_key,
        model_version=model_version,
    )
