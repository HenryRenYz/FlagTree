"""
Generic XGBoost Ranking training and prediction.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRanker
except ImportError as exc:
    raise SystemExit("Please install xgboost: pip install xgboost") from exc

from triton.flagtune.core.interfaces import (
    DataSource,
    FeaturePipeline,
    InputSpace,
    ParameterSpace,
)


def build_rank_labels(
    target_values: np.ndarray,
    group_indices: Dict[Any, np.ndarray],
) -> np.ndarray:
    labels = np.full(len(target_values), np.nan, dtype=float)
    for positions in group_indices.values():
        positions = np.asarray(positions, dtype=int)
        if len(positions) == 0:
            continue
        group_targets = target_values[positions]
        order = np.argsort(group_targets, kind="stable")
        group_labels = np.zeros(len(positions), dtype=float)
        group_labels[order] = np.arange(len(positions) - 1, -1, -1, dtype=float)
        labels[positions] = group_labels
    return labels


def _ratio_count(total: int, ratio: float) -> int:
    if total <= 0:
        return 0
    if ratio >= 1.0:
        return total
    return max(1, int(math.ceil(total * ratio)))


TRAIN_MODE_RATIOS: Dict[str, Tuple[float, float]] = {
    "shape100_config100": (1.0, 1.0),
    "shape50_config100": (0.5, 1.0),
    "shape50_config50": (0.5, 0.5),
    "shape25_config50": (0.25, 0.5),
}

DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 1200,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.95,
    "colsample_bytree": 0.95,
    "reg_lambda": 1.5,
    "reg_alpha": 0.0,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "max_bin": 512,
    "n_jobs": 4,
    "objective": "rank:pairwise",
    "eval_metric": "ndcg",
    "tree_method": "hist",
}


class XGBoostRankingTrainer:

    def __init__(
        self,
        param_space: ParameterSpace,
        input_space: InputSpace,
        feature_pipeline: FeaturePipeline,
        data_source: DataSource,
        xgb_params: Optional[Dict[str, Any]] = None,
        group_cols: Optional[List[str]] = None,
        target_col: str = "latency_ms",
        config_order_col: str = "config_order_in_shape",
        seed: int = 2026,
        min_train_rows: int = 8,
    ) -> None:
        self.param_space = param_space
        self.input_space = input_space
        self.feature_pipeline = feature_pipeline
        self.data_source = data_source
        self.xgb_params = dict(DEFAULT_XGB_PARAMS)
        if xgb_params:
            self.xgb_params.update(xgb_params)
        self.group_cols = group_cols or ["shape_key"]
        self.target_col = target_col
        self.config_order_col = config_order_col
        self.seed = seed
        self.min_train_rows = min_train_rows
        self._feature_cols_: Optional[List[str]] = None
        self._model_: Optional[XGBRanker] = None

    def _make_model(self) -> XGBRanker:
        return XGBRanker(
            n_estimators=self.xgb_params["n_estimators"],
            max_depth=self.xgb_params["max_depth"],
            learning_rate=self.xgb_params["learning_rate"],
            subsample=self.xgb_params["subsample"],
            colsample_bytree=self.xgb_params["colsample_bytree"],
            reg_lambda=self.xgb_params["reg_lambda"],
            reg_alpha=self.xgb_params["reg_alpha"],
            min_child_weight=self.xgb_params["min_child_weight"],
            gamma=self.xgb_params["gamma"],
            objective="rank:pairwise",
            eval_metric="ndcg",
            random_state=self.seed,
            n_jobs=self.xgb_params["n_jobs"],
            tree_method="hist",
            max_bin=self.xgb_params["max_bin"],
        )

    def fit(
        self,
        data_path: str,
        train_mode: str = "shape100_config100",
        shape_train_ratio: Optional[float] = None,
        config_train_ratio: Optional[float] = None,
        **data_kwargs: Any,
    ) -> Tuple[XGBRanker, Dict[str, Any]]:
        df = self.data_source.load(data_path, **data_kwargs)
        feature_df = self.feature_pipeline.build(df)
        feature_df[self.config_order_col] = (feature_df.get(self.config_order_col, pd.Series(range(len(feature_df)))))
        feature_df[self.target_col] = pd.to_numeric(feature_df[self.target_col], errors="coerce")
        target_values = feature_df[self.target_col].to_numpy(dtype=float)
        finite_mask = np.isfinite(target_values) & (target_values > 0)
        finite_positions = np.flatnonzero(finite_mask)
        if len(finite_positions) < self.min_train_rows:
            raise RuntimeError(f"Not enough valid training rows: {len(finite_positions)} < {self.min_train_rows}")
        shape_ratio, config_ratio = TRAIN_MODE_RATIOS.get(train_mode, TRAIN_MODE_RATIOS["shape100_config100"])
        if shape_train_ratio is not None:
            shape_ratio = shape_train_ratio
        if config_train_ratio is not None:
            config_ratio = config_train_ratio
        finite_df = feature_df.iloc[finite_positions].copy()
        finite_df["_row_pos"] = finite_positions
        shape_groups: Dict[Any, np.ndarray] = (finite_df.groupby(
            self.group_cols, dropna=False,
            sort=True)["_row_pos"].apply(lambda vals: vals.to_numpy(dtype=int)).to_dict())
        rank_labels = build_rank_labels(target_values, shape_groups)
        feature_df["_rank_label"] = rank_labels
        rng = np.random.default_rng(self.seed)
        shape_keys = list(shape_groups.keys())
        total_shape_count = len(shape_keys)
        if total_shape_count == 0:
            raise RuntimeError("No valid shape groups for training.")
        selected_shape_count = _ratio_count(total_shape_count, shape_ratio)
        selected_shape_indices = rng.choice(
            np.arange(total_shape_count),
            size=selected_shape_count,
            replace=False,
        )
        selected_shape_keys = [shape_keys[int(i)] for i in selected_shape_indices]
        train_positions: List[np.ndarray] = []
        for shape_key in selected_shape_keys:
            positions = np.asarray(shape_groups[shape_key], dtype=int)
            count = _ratio_count(len(positions), config_ratio)
            if count >= len(positions):
                train_positions.append(positions)
            else:
                train_positions.append(rng.choice(positions, size=count, replace=False))
        if not train_positions:
            raise RuntimeError("Training set is empty after sampling.")
        train_indices = np.sort(np.concatenate(train_positions))
        train_data = feature_df.iloc[train_indices].copy()
        self._feature_cols_ = self._identify_feature_columns(feature_df)
        X_train = train_data[self._feature_cols_].to_numpy(dtype=float)
        y_train = train_data["_rank_label"].to_numpy(dtype=float)
        y_train = np.nan_to_num(y_train, nan=0.0)
        group_sizes = self._compute_group_sizes(train_data)
        model = self._make_model()
        fit_start = time.perf_counter()
        model.fit(X_train, y_train, group=group_sizes)
        fit_elapsed = time.perf_counter() - fit_start
        self._model_ = model
        info = {
            "train_mode": train_mode,
            "shape_train_ratio": shape_ratio,
            "config_train_ratio": config_ratio,
            "total_shape_count": total_shape_count,
            "train_shape_count": selected_shape_count,
            "train_ranking_group_count": len(group_sizes),
            "total_finite_config_count": len(finite_positions),
            "global_train_config_count": len(train_indices),
            "feature_count": len(self._feature_cols_),
            "feature_cols": self._feature_cols_,
            "xgboost_fit_elapsed_s": fit_elapsed,
        }
        return model, info

    def _identify_feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude = {
            self.target_col,
            "_rank_label",
            "_row_pos",
            self.config_order_col,
            *self.input_space.all_field_names,
        }
        for f in self.param_space.fields:
            exclude.add(f.name)
        for kw in ("key_", "p20", "p80", "p50", "latency_ms", "benchmark_table", "config_order_in_shape", "source_db",
                   "source_db_name", "kernel_kind", "dtype", "shape_key", "used_for_train", "xgb_rank_label"):
            exclude.update({c for c in df.columns if c.startswith(kw) or c == kw})
        feature_cols = []
        for col in df.columns:
            if col in exclude:
                continue
            if col.startswith("_"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
        return feature_cols

    def _compute_group_sizes(self, train_data: pd.DataFrame) -> List[int]:
        group_sizes = (train_data.groupby(self.group_cols, sort=False).size().to_list())
        return [int(s) for s in group_sizes]

    def export(self, model: XGBRanker, info: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "xgboost_ranker.json"
        model.save_model(str(model_path))
        schema = {
            "model_file": "xgboost_ranker.json",
            "config_cols": self.param_space.all_field_names,
            "feature_cols": info.get("feature_cols", self._feature_cols_ or []),
            "input_fields": [{"name": f.name, "log_transform": f.log_transform} for f in self.input_space.fields],
            "param_fields": [{"name": f.name, "legal_values": f.legal_values} for f in self.param_space.fields],
            "group_cols": self.group_cols,
            "target_col": self.target_col,
            "train_mode": info.get("train_mode"),
            "feature_count": info.get("feature_count"),
            "dtype_code": "hash_mod_997",
        }
        schema_path = output_dir / "feature_schema.json"
        with schema_path.open("w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        return schema

    @property
    def feature_cols(self) -> Optional[List[str]]:
        return self._feature_cols_


class XGBoostPredictor:

    def __init__(
        self,
        model_dir: Path,
        param_space: ParameterSpace,
        input_space: InputSpace,
        feature_pipeline: FeaturePipeline,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.param_space = param_space
        self.input_space = input_space
        self.feature_pipeline = feature_pipeline
        self._schema = self._load_schema()
        self._model = self._load_model()

    @property
    def feature_cols(self) -> List[str]:
        return list(self._schema.get("feature_cols", []))

    @property
    def config_cols(self) -> List[str]:
        return list(self._schema.get("config_cols", self.param_space.all_field_names))

    def _load_schema(self) -> Dict[str, Any]:
        schema_path = self.model_dir / "feature_schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"feature_schema.json not found: {schema_path}")
        with schema_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self) -> XGBRanker:
        model_file = self._schema.get("model_file", "xgboost_ranker.json")
        model_path = self.model_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model file not found: {model_path}")
        model = XGBRanker()
        model.load_model(str(model_path))
        return model

    def predict_topk(
        self,
        shapes: pd.DataFrame,
        top_k: int = 10,
        kernel_variant_fn: Optional[callable] = None,
    ) -> pd.DataFrame:
        all_candidates: List[Dict[str, Any]] = []
        for shape_idx, (_, shape_row) in enumerate(shapes.iterrows()):
            shape_dict = shape_row.to_dict()
            kv = kernel_variant_fn(shape_row) if kernel_variant_fn else None
            candidate_rows = []
            for order, config in enumerate(self.param_space.iter_configs(kv)):
                row = {**shape_dict, **config}
                row["_config_order"] = order
                candidate_rows.append(row)
            if not candidate_rows:
                continue
            candidates_df = pd.DataFrame(candidate_rows)
            features = self.feature_pipeline.build(candidates_df)
            X_pred = features.reindex(columns=self.feature_cols, fill_value=0)
            X_pred = X_pred.to_numpy(dtype=float)
            scores = self._model.predict(X_pred)
            candidates_df["xgb_rank_score"] = scores
            candidates_df = candidates_df.sort_values(
                ["xgb_rank_score", "_config_order"],
                ascending=[False, True],
            ).head(int(top_k))
            candidates_df["shape_idx"] = shape_idx
            if kv:
                candidates_df["kernel_variant"] = kv
            all_candidates.append(candidates_df)
        if not all_candidates:
            return pd.DataFrame()
        result = pd.concat(all_candidates, ignore_index=True)
        return result.reset_index(drop=True)
