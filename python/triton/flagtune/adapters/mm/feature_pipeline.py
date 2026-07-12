"""
Matrix multiplication (mm) operator feature pipeline adapter.
"""

from __future__ import annotations

import pandas as pd

from triton.flagtune.adapters.mm.input_space import mm_input_space
from triton.flagtune.adapters.mm.parameter_space import mm_parameter_space
from triton.flagtune.core.interfaces import FeaturePipeline


class MMFeaturePipeline(FeaturePipeline):

    def __init__(self) -> None:
        super().__init__(
            input_space=mm_input_space(),
            param_space=mm_parameter_space(),
        )

    def _add_kernel_kind_encoding(self, df: pd.DataFrame) -> None:
        if "kernel_kind" in df.columns:
            kind_map = {"mm": 0, "gemv": 1}
            df["kernel_kind_code"] = (df["kernel_kind"].map(kind_map).fillna(2).astype(int))
