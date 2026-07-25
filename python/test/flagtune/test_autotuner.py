"""Test FlagTune autotuner dtype identity extraction."""

from __future__ import annotations

import torch

from triton.flagtune.runtime.autotuner import _infer_tensor_dtypes


class _DtypeLookalike:
    """Expose a dtype attribute without representing a tensor argument."""

    dtype = torch.float16


def test_default_dtype_extraction_ignores_non_tensor_dtype_attributes():
    """Keep FlagTree's default identity aligned with FlagGems LibTuner."""
    tensor = torch.empty(1, dtype=torch.bfloat16)

    assert _infer_tensor_dtypes((tensor, _DtypeLookalike())) == (torch.bfloat16, )
