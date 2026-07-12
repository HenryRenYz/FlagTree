"""
Matrix multiplication (mm) operator input space adapter.
"""

from __future__ import annotations

import numpy as np

from triton.flagtune.core.interfaces import InputField, InputSpace


def _group_tile_m(df):
    gm = df.get("GROUP_M", 8)
    bm = df.get("BLOCK_M", 0)
    return gm * bm


def _log2_group_tile_m(df):
    val = np.maximum(_group_tile_m(df), 1)
    return np.log2(val)


def _group_tiles_m(df):
    m = df.get("M", 1)
    bm = np.maximum(df.get("BLOCK_M", 1), 1)
    gm = np.maximum(df.get("GROUP_M", 1), 1)
    grid_m = np.ceil(m / bm)
    return np.ceil(grid_m / gm)


def _grid_m_per_group(df):
    m = df.get("M", 1)
    bm = np.maximum(df.get("BLOCK_M", 1), 1)
    gm = np.maximum(df.get("GROUP_M", 1), 1)
    grid_m = np.ceil(m / bm)
    return grid_m / gm


def _group_m_ratio(df):
    m = df.get("M", 1)
    bm = np.maximum(df.get("BLOCK_M", 1), 1)
    gm = df.get("GROUP_M", 8)
    grid_m = np.maximum(np.ceil(m / bm), 1)
    return gm / grid_m


def _m_mod_group_tile_m(df):
    gtm = np.maximum(_group_tile_m(df), 1)
    m = df.get("M", 1)
    return np.mod(m, gtm)


def mm_input_space() -> InputSpace:
    return InputSpace(
        fields=[
            InputField("M"),
            InputField("N"),
            InputField("K"),
            InputField("stride_am", log_transform=True),
            InputField("stride_bk", log_transform=True),
        ],
        pairwise_products=[
            ("M", "N", "shape_mn"),
            ("M", "K", "shape_mk"),
            ("N", "K", "shape_nk"),
        ],
        derived_features={
            "group_tile_m": _group_tile_m,
            "log2_group_tile_m": _log2_group_tile_m,
            "group_tiles_m": _group_tiles_m,
            "grid_m_per_group": _grid_m_per_group,
            "group_m_ratio": _group_m_ratio,
            "m_mod_group_tile_m": _m_mod_group_tile_m,
        },
    )
