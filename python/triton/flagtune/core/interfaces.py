"""
Core abstraction interfaces for the auto-tuning framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

BenchmarkFn = Callable[[Dict[str, Any], Optional[int]], List[float]]
"""Benchmark function: (config_dict, n_runs?) -> list of latency results."""

ConfigProposer = Callable[[
    Optional["BenchmarkFn"],  # fn: benchmark function (None = cold start)
    Dict[str, Any],  # shape: e.g. {"M": 256, "N": 512, "K": 128}
    List[Dict[str, Any]],  # initial_configs: reference (may ignore)
    Dict[str, Any],  # meta: {op_name, vendor, op_id?, ...}
], List[Dict[str, Any]],  # next batch of candidate config dicts
                          ]
"""Stateful config proposer: given shape + history, returns candidate configs."""


@dataclass
class ParameterField:
    name: str
    legal_values: List[Any]
    log_transform: bool = True


@dataclass
class ParameterSpace:
    fields: List[ParameterField]
    constraints: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    kernel_variants: Dict[str, List[str]] = field(default_factory=dict)

    def _field_values_for_variant(self, kernel_variant: Optional[str]) -> Dict[str, List[Any]]:
        if kernel_variant is None or kernel_variant not in self.kernel_variants:
            active_fields = {f.name for f in self.fields}
        else:
            active_fields = set(self.kernel_variants[kernel_variant])
        return {f.name: f.legal_values for f in self.fields if f.name in active_fields}

    def iter_configs(self, kernel_variant: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        field_values = self._field_values_for_variant(kernel_variant)
        if not field_values:
            return
        names = list(field_values.keys())
        for combo in product(*(field_values[n] for n in names)):
            config = dict(zip(names, combo))
            if all(fn(config) for fn in self.constraints):
                yield config

    def validate(self, config: Dict[str, Any], kernel_variant: Optional[str] = None) -> bool:
        field_values = self._field_values_for_variant(kernel_variant)
        for name, values in field_values.items():
            if name not in config:
                return False
            if config[name] not in values:
                return False
        return all(fn(config) for fn in self.constraints)

    def config_key(self, config: Dict[str, Any], kernel_variant: Optional[str] = None) -> Tuple[Tuple[str, int], ...]:
        field_values = self._field_values_for_variant(kernel_variant)
        active_names = [n for n in field_values if n in config]
        return tuple(sorted((n, int(config[n])) for n in active_names))

    def active_field_names(self, kernel_variant: Optional[str] = None) -> Tuple[str, ...]:
        field_values = self._field_values_for_variant(kernel_variant)
        ordered = [f.name for f in self.fields if f.name in field_values]
        return tuple(ordered)

    @property
    def all_field_names(self) -> List[str]:
        return [f.name for f in self.fields]


@dataclass
class InputField:
    name: str
    log_transform: bool = True


@dataclass
class InputSpace:
    fields: List[InputField]
    pairwise_products: List[Tuple[str, str, str]] = field(default_factory=list)
    derived_features: Dict[str, Callable[["pd.DataFrame"], "pd.Series"]] = field(default_factory=dict)

    @property
    def all_field_names(self) -> List[str]:
        return [f.name for f in self.fields]


def _stable_int_hash(text: str) -> int:
    import hashlib
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


class FeaturePipeline(ABC):

    def __init__(self, input_space: InputSpace, param_space: ParameterSpace) -> None:
        self.input_space = input_space
        self.param_space = param_space

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._normalize_numerics(df)
        df = self._add_log_transforms(df)
        df = self._add_pairwise_products(df)
        df = self._add_grid_features(df)
        df = self._add_block_ratio_features(df)
        df = self._add_config_pairwise_features(df)
        df = self._add_derived_features(df)
        df = self._add_categorical_encodings(df)
        return df

    def _normalize_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        all_names = self.input_space.all_field_names + self.param_space.all_field_names
        for col in all_names:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    def _add_log_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        for input_field in self.input_space.fields:
            if input_field.log_transform and input_field.name in df.columns:
                df[f"log2_{input_field.name}"] = np.log2(np.maximum(df[input_field.name], 1))
        for param_field in self.param_space.fields:
            if param_field.log_transform and param_field.name in df.columns:
                df[f"log2_{param_field.name}"] = np.log2(np.maximum(df[param_field.name], 1))
        return df

    def _add_pairwise_products(self, df: pd.DataFrame) -> pd.DataFrame:
        for f1, f2, out_name in self.input_space.pairwise_products:
            if f1 in df.columns and f2 in df.columns:
                df[out_name] = df[f1] * df[f2]
                df[f"log2_{out_name}"] = np.log2(np.maximum(df[out_name], 1))
        return df

    def _add_grid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = self._map_input_to_block()
        grid_cols = []
        for input_name, block_name in mapping.items():
            if input_name in df.columns and block_name in df.columns:
                col_name = f"grid_{input_name.lower()}"
                df[col_name] = np.ceil(df[input_name] / np.maximum(df[block_name], 1))
                grid_cols.append(col_name)

        if len(grid_cols) >= 2:
            prod = df[grid_cols[0]]
            for c in grid_cols[1:]:
                prod = prod * df[c]
            df["grid_work"] = prod
            df["log2_grid_work"] = np.log2(np.maximum(df["grid_work"], 1))

        if len(grid_cols) >= 2:
            df["grid_mn"] = df[grid_cols[0]] * df[grid_cols[1]]
            df["log2_grid_mn"] = np.log2(np.maximum(df["grid_mn"], 1))
        return df

    def _map_input_to_block(self) -> Dict[str, str]:
        mapping = {}
        for f in self.input_space.fields:
            block_name = f"BLOCK_{f.name}"
            if block_name in self.param_space.all_field_names:
                mapping[f.name] = block_name
        return mapping

    def _add_block_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = self._map_input_to_block()
        for input_name, block_name in mapping.items():
            if input_name in df.columns and block_name in df.columns:
                safe_block = np.maximum(df[block_name], 1)
                safe_input = np.maximum(df[input_name], 1)
                df[f"{input_name.lower()}_mod_{block_name.lower()}"] = np.mod(df[input_name], safe_block)
                df[f"{block_name.lower()}_ratio"] = safe_block / safe_input
        return df

    def _add_config_pairwise_features(self, df: pd.DataFrame) -> pd.DataFrame:
        block_fields = [f for f in self.param_space.fields if f.name.startswith("BLOCK_") and f.name in df.columns]
        block_names = [f.name for f in block_fields]

        for i in range(len(block_names)):
            for j in range(i + 1, len(block_names)):
                ni, nj = block_names[i], block_names[j]
                si = ni.replace("BLOCK_", "").lower()
                sj = nj.replace("BLOCK_", "").lower()
                df[f"tile_{si}{sj}"] = df[ni] * df[nj]
                df[f"log2_tile_{si}{sj}"] = np.log2(np.maximum(df[f"tile_{si}{sj}"], 1))

        if block_names:
            vol = df[block_names[0]]
            for bn in block_names[1:]:
                vol = vol * df[bn]
            df["tile_volume"] = vol
            df["log2_tile_volume"] = np.log2(np.maximum(vol, 1))
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col_name, func in self.input_space.derived_features.items():
            try:
                df[col_name] = func(df)
            except Exception:
                pass
        return df

    def _add_categorical_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        if "dtype" in df.columns:
            df["dtype_code"] = df["dtype"].map(lambda v: _stable_int_hash(str(v)) % 997).fillna(0).astype(int)

        self._add_kernel_kind_encoding(df)
        return df

    def _add_kernel_kind_encoding(self, df: pd.DataFrame) -> None:
        if "kernel_kind" in df.columns:
            unique = sorted(df["kernel_kind"].dropna().unique())
            kind_map = {v: i for i, v in enumerate(unique)}
            df["kernel_kind_code"] = df["kernel_kind"].map(kind_map).fillna(len(unique)).astype(int)


class DataSource(ABC):

    @abstractmethod
    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        ...


class KernelAdapter(ABC):

    @abstractmethod
    def make_bench_args(self, shape: Dict[str, Any], config: Dict[str, Any],
                        kernel_variant: str) -> Tuple[tuple, Dict[str, Any]]:
        ...

    @abstractmethod
    def find_tuner(self, kernel_variant: str) -> Any:
        ...

    @abstractmethod
    def make_config_objects(self, entries: List[Dict[str, Any]], kernel_variant: str) -> List[Any]:
        ...

    def install_configs(self, tuner: Any, configs: List[Any]) -> None:
        pass
