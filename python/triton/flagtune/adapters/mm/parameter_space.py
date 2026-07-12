"""
Matrix multiplication (mm) operator parameter space adapter.
"""

from __future__ import annotations

from typing import Dict, List

from triton.flagtune.core.interfaces import ParameterField, ParameterSpace

_TMA_FIELDS: List[ParameterField] = [
    ParameterField("BLOCK_M", [16, 32, 64, 128, 256]),
    ParameterField("BLOCK_N", [16, 32, 64, 128]),
    ParameterField("BLOCK_K", [32, 64, 128, 256]),
    ParameterField("GROUP_M", [1, 2, 4, 8, 16, 32, 64], log_transform=True),
    ParameterField("num_warps", [4, 8]),
    ParameterField("num_stages", [2, 3, 4]),
    ParameterField("num_ctas", [1]),
]

_TMA_ACTIVE_FIELDS: List[str] = ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "num_warps", "num_stages"]

_GEMV_FIELDS: Dict[str, List[ParameterField]] = {
    "gemv": [
        ParameterField("BLOCK_M", [8, 16, 32]),
        ParameterField("BLOCK_K", [128, 256]),
        ParameterField("num_warps", [1, 2, 4, 8]),
        ParameterField("num_stages", [2, 3, 4, 5, 6, 7, 8]),
        ParameterField("num_ctas", [1]),
    ]
}

_GEMV_ACTIVE_FIELDS: List[str] = ["BLOCK_M", "BLOCK_K", "num_warps", "num_stages"]

_ALL_FIELDS: List[ParameterField] = [
    ParameterField("BLOCK_M", [8, 16, 32, 64, 128, 256]),
    ParameterField("BLOCK_N", [1, 16, 32, 64, 128]),
    ParameterField("BLOCK_K", [32, 64, 128, 256]),
    ParameterField("GROUP_M", [1, 2, 4, 8, 16, 32, 64], log_transform=True),
    ParameterField("num_warps", [1, 2, 4, 8]),
    ParameterField("num_stages", [2, 3, 4, 5, 6, 7, 8]),
    ParameterField("num_ctas", [1]),
]


def _tma_constraints() -> list:
    return []


def mm_parameter_space() -> ParameterSpace:
    return ParameterSpace(
        fields=list(_ALL_FIELDS),
        constraints=_tma_constraints(),
        kernel_variants={
            "mm_general_tma": _TMA_ACTIVE_FIELDS,
            "gemv": _GEMV_ACTIVE_FIELDS,
        },
    )
