"""
Matrix multiplication (mm) operator data source adapter — BenchmarkCache.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from triton.flagtune.core.interfaces import DataSource

_DEFAULT_CONFIG_COLS: List[str] = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_M",
    "num_warps",
    "num_ctas",
    "num_stages",
]

_CANONICAL_SHAPE_COLS: List[str] = [
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
]

_BENCHMARK_TABLE_LIKE: str = "%benchmark%"

_DEFAULT_KERNEL_SUBSTRS: List[str] = [
    "mm_kernel_general_host_tma",
    "gemv_kernel",
]


class BenchmarkCacheDataSource(DataSource):

    def __init__(
        self,
        benchmark_table_like: str = _BENCHMARK_TABLE_LIKE,
        kernel_substrs: Optional[List[str]] = None,
        explicit_tables: Optional[List[str]] = None,
    ) -> None:
        self.benchmark_table_like = benchmark_table_like
        self.kernel_substrs = kernel_substrs or list(_DEFAULT_KERNEL_SUBSTRS)
        self.explicit_tables = explicit_tables

    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        db_path = Path(path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        table_like = kwargs.get("benchmark_table_like", self.benchmark_table_like)
        kernel_substrs = kwargs.get("kernel_substrs", self.kernel_substrs)
        explicit_tables = kwargs.get("explicit_tables", self.explicit_tables)
        conn = sqlite3.connect(str(db_path))
        try:
            tables = self._list_benchmark_tables(
                conn,
                table_like=table_like,
                kernel_substrs=kernel_substrs,
                explicit_tables=explicit_tables,
            )
            if not tables:
                raise RuntimeError(f"No benchmark tables found in {db_path}")
            raw_df = self._read_benchmark_tables(conn, tables)
        finally:
            conn.close()
        raw_df.insert(0, "source_db", str(db_path))
        raw_df.insert(1, "source_db_name", db_path.name)
        return self._add_common_columns(raw_df)

    @staticmethod
    def _list_benchmark_tables(
        conn: sqlite3.Connection,
        table_like: str = _BENCHMARK_TABLE_LIKE,
        kernel_substrs: Optional[List[str]] = None,
        explicit_tables: Optional[List[str]] = None,
    ) -> List[str]:
        if explicit_tables:
            return [t for t in explicit_tables if BenchmarkCacheDataSource._table_exists(conn, t)]
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (table_like, ))
        all_tables = [row[0] for row in cursor.fetchall()]
        if kernel_substrs is None:
            return all_tables
        filtered = []
        for table in all_tables:
            for substr in kernel_substrs:
                if substr in table:
                    filtered.append(table)
                    break
        return filtered

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name, ))
        return cursor.fetchone() is not None

    @staticmethod
    def _read_benchmark_tables(conn: sqlite3.Connection, tables: List[str]) -> pd.DataFrame:
        frames = []
        for table in tables:
            try:
                df = pd.read_sql_query(f"SELECT * FROM [{table}]", conn)
                df["benchmark_table"] = table
                frames.append(df)
            except Exception:
                continue
        if not frames:
            raise RuntimeError("Cannot read any benchmark tables")
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _kernel_kind_from_table(table: str) -> str:
        if "gemv_kernel" in str(table):
            return "gemv"
        if "mm_kernel_general_host_tma" in str(table):
            return "mm"
        return "other"

    @staticmethod
    def _normalize_dtype(value: Any) -> str:
        text = str(value).strip()
        if text.startswith("torch."):
            text = text.split(".", 1)[1]
        return text

    @classmethod
    def _add_common_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["p50", "p20", "p80"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for i in range(7):
            col = f"key_{i}"
            if col not in df.columns:
                df[col] = np.nan
        for col in _DEFAULT_CONFIG_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df["kernel_kind"] = df["benchmark_table"].apply(cls._kernel_kind_from_table)
        for col in _CANONICAL_SHAPE_COLS:
            df[col] = np.nan
        df["dtype"] = "unknown"
        mm_mask = df["kernel_kind"].eq("mm")
        df.loc[mm_mask, "M"] = pd.to_numeric(df.loc[mm_mask, "key_0"], errors="coerce")
        df.loc[mm_mask, "N"] = pd.to_numeric(df.loc[mm_mask, "key_1"], errors="coerce")
        df.loc[mm_mask, "K"] = pd.to_numeric(df.loc[mm_mask, "key_2"], errors="coerce")
        df.loc[mm_mask, "stride_am"] = pd.to_numeric(df.loc[mm_mask, "key_3"], errors="coerce")
        df.loc[mm_mask, "stride_bk"] = pd.to_numeric(df.loc[mm_mask, "key_4"], errors="coerce")
        df.loc[mm_mask, "dtype"] = df.loc[mm_mask, "key_5"].apply(cls._normalize_dtype)
        gemv_mask = df["kernel_kind"].eq("gemv")
        df.loc[gemv_mask, "M"] = pd.to_numeric(df.loc[gemv_mask, "key_0"], errors="coerce")
        df.loc[gemv_mask, "N"] = 1
        df.loc[gemv_mask, "K"] = pd.to_numeric(df.loc[gemv_mask, "key_1"], errors="coerce")
        df.loc[gemv_mask, "stride_am"] = pd.to_numeric(df.loc[gemv_mask, "key_2"], errors="coerce")
        df.loc[gemv_mask, "stride_bk"] = pd.to_numeric(df.loc[gemv_mask, "key_3"], errors="coerce")
        df.loc[gemv_mask, "dtype"] = df.loc[gemv_mask, "key_4"].apply(cls._normalize_dtype)
        other_mask = ~(mm_mask | gemv_mask)
        df.loc[other_mask, "M"] = pd.to_numeric(df.loc[other_mask, "key_0"], errors="coerce")
        df.loc[other_mask, "N"] = pd.to_numeric(df.loc[other_mask, "key_1"], errors="coerce")
        df.loc[other_mask, "K"] = pd.to_numeric(df.loc[other_mask, "key_2"], errors="coerce")
        df.loc[other_mask, "stride_am"] = pd.to_numeric(df.loc[other_mask, "key_3"], errors="coerce")
        df.loc[other_mask, "stride_bk"] = pd.to_numeric(df.loc[other_mask, "key_4"], errors="coerce")
        df.loc[other_mask, "dtype"] = df.loc[other_mask, "key_5"].apply(cls._normalize_dtype)
        df["BLOCK_N"] = pd.to_numeric(df["BLOCK_N"], errors="coerce")
        df.loc[gemv_mask & df["BLOCK_N"].isna(), "BLOCK_N"] = 1
        df["GROUP_M"] = pd.to_numeric(df["GROUP_M"], errors="coerce")
        df.loc[df["GROUP_M"].isna(), "GROUP_M"] = 8
        for col in _CANONICAL_SHAPE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        def _build_shape_key(row: pd.Series) -> str:
            m = str(int(row.get("M", 0)) if pd.notna(row.get("M")) else 0)
            n = str(int(row.get("N", 0)) if pd.notna(row.get("N")) else 0)
            k_val = str(int(row.get("K", 0)) if pd.notna(row.get("K")) else 0)
            sa = str(int(row.get("stride_am", 0)) if pd.notna(row.get("stride_am")) else k_val)
            sb = str(int(row.get("stride_bk", 0)) if pd.notna(row.get("stride_bk")) else n)
            dt = cls._normalize_dtype(row.get("dtype", "unknown"))
            return f"{m},{n},{k_val},{sa},{sb},{dt}"

        df["shape_key"] = df.apply(_build_shape_key, axis=1)
        if "p50" in df.columns and "latency_ms" not in df.columns:
            df["latency_ms"] = df["p50"]
        return df
