"""
FlagTune config proposer — generic multi-operator support.

Primary API:
    from triton.flagtune import make_config_proposer
    proposer = make_config_proposer({"op_id": "flagtree/gemm"})
    configs = proposer(bench_fn, shape, initial_configs, meta)

When TRITON_USE_FLAGTUNE=1 is set, the proposer loads a trained XGBoost
ranking model for the requested operator.  When a benchmark function is
provided it uses genetic-algorithm search for iterative refinement.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from triton.flagtune.core.interfaces import BenchmarkFn, ConfigProposer

logger = logging.getLogger(__name__)

_MODEL_MANAGER: Optional[Any] = None
_TOP_K_CACHE: Optional[int] = None


def _is_enabled() -> bool:
    return os.environ.get("TRITON_USE_FLAGTUNE", "").strip() == "1"


def _get_model_manager() -> Any:
    global _MODEL_MANAGER
    if _MODEL_MANAGER is None:
        from triton.flagtune.model_manager import FlagTuneModelManager
        _MODEL_MANAGER = FlagTuneModelManager()
    return _MODEL_MANAGER


def _top_k() -> int:
    global _TOP_K_CACHE
    if _TOP_K_CACHE is None:
        _TOP_K_CACHE = int(os.environ.get("TRITON_FLAGTUNE_TOP_K", "10"))
    return _TOP_K_CACHE


# ---------------------------------------------------------------------------
# Generic config-to-triton converter (fallback when operator does not
# provide its own ``to_config``).
# ---------------------------------------------------------------------------


def _generic_to_config(config_dict: Dict[str, Any]) -> Any:
    from triton import Config

    popped = config_dict.copy()
    num_warps = int(popped.pop("num_warps", 4))
    num_stages = int(popped.pop("num_stages", 3))
    num_ctas = int(popped.pop("num_ctas", 1))

    kwargs: Dict[str, int] = {}
    for k, v in popped.items():
        kwargs[k] = int(v)

    return Config(kwargs=kwargs, num_warps=num_warps, num_stages=num_stages, num_ctas=num_ctas)


def _resolve_kernel_variant(op_info: Any, explicit_kv: Optional[str], shape: Dict[str, Any]) -> Optional[str]:
    if explicit_kv is not None:
        return explicit_kv
    select = getattr(op_info, "select_kernel_variant", None)
    if select is not None:
        return select(shape)
    return op_info.default_kernel_variant


# ---------------------------------------------------------------------------
# Config dict helpers
# ---------------------------------------------------------------------------


def _config_dict_key(config: Dict[str, Any]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((k, int(v)) for k, v in config.items() if k.isupper()))


def _in_history(history: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
    target = _config_dict_key(config)
    for entry in history:
        if _config_dict_key(entry.get("config", entry)) == target:
            return True
    return False


def _strip_config_dict(config: Dict[str, Any], param_fields: List[str]) -> Dict[str, Any]:
    result = {}
    for fname in param_fields:
        if fname in config:
            result[fname] = int(config[fname])
    return result


# ---------------------------------------------------------------------------
# Primary API: make_config_proposer
# ---------------------------------------------------------------------------


def make_config_proposer(meta: Dict[str, Any]) -> ConfigProposer:
    op_id = meta.get("op_id")
    if op_id is None:
        raise ValueError("meta must contain 'op_id' for ConfigProposer")

    # FLAGTUNE_DISABLE_OPS: comma-separated op_ids or '*' to disable
    _disabled_raw = os.environ.get("FLAGTUNE_DISABLE_OPS", "").strip()
    if _disabled_raw:
        _disabled_set = {s.strip() for s in _disabled_raw.split(",") if s.strip()}
        if "*" in _disabled_set or op_id in _disabled_set:
            return lambda _fn, _shape, _init, _meta: []

    from triton.flagtune.registry import get as _get_op, resolve_operator_id

    operator_id = resolve_operator_id(op_id)
    op_info = _get_op(operator_id)
    model = _get_model_manager().load(operator_id)

    top_k = _top_k()
    param_fields = op_info.param_space.all_field_names

    # GA search setup
    from triton.flagtune.core.ga_search import GAParams, GASearcher

    ga_params = GAParams(
        generations=5,
        population_size=20,
        elite_size=5,
        offspring_per_generation=10,
        mutation_rate=0.3,
        random_rate=0.2,
    )
    ga_searcher = GASearcher(op_info.param_space, ga_params, seed=42)

    # Mutable state across calls
    history: List[Dict[str, Any]] = []

    def propose(
        fn: Optional[BenchmarkFn],
        shape: Dict[str, Any],
        initial_configs: List[Dict[str, Any]],
        _meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        nonlocal history

        # Reset history on every call — different shapes must not share state.
        # Same-shape benchmark results are persisted by libtuner's BenchmarkCache (SQLite).
        history = []

        kv = _resolve_kernel_variant(op_info, None, shape)

        # ── Phase 1: XGBoost global coarse ranking (always, 0 GPU bench) ──
        xgb_configs = _predict_config_dicts(op_info, model, shape, kv, top_k)

        # ── fn is None: no benchmark capability → return XGBoost predictions ──
        if fn is None:
            return xgb_configs

        # ── Phase 2: benchmark XGBoost top-k seeds (≤ top_k GPU benches) ──
        validate_fn = getattr(op_info, "validate_shape_config", None)
        for rank, cfg_dict in enumerate(xgb_configs, start=1):
            stripped = _strip_config_dict(cfg_dict, param_fields)
            if not stripped or _in_history(history, stripped):
                continue
            if validate_fn is not None and not validate_fn(shape, stripped):
                continue
            try:
                latencies = fn(stripped, None)
                lat = float(latencies[0]) if latencies else 0.0
            except Exception:
                lat = float("inf")
            history.append({
                "config": stripped,
                "latency_ms": lat,
                "ga_latency_ms": lat,
                "candidate_rank": rank,
            })

        # ── Phase 3: GA iterative search from XGB-quality seeds ──
        if history:
            try:
                new_entries = ga_searcher.generate(history, kv)
            except Exception:
                new_entries = []
        else:
            new_entries = []

        for entry in new_entries:
            cfg_dict = entry.get("config", entry)
            stripped = _strip_config_dict(cfg_dict, param_fields)
            if not stripped or _in_history(history, stripped):
                continue
            if validate_fn is not None and not validate_fn(shape, stripped):
                continue
            try:
                latencies = fn(stripped, None)
                lat = float(latencies[0]) if latencies else 0.0
            except Exception:
                lat = float("inf")
            entry["config"] = stripped
            entry["latency_ms"] = lat
            entry["ga_latency_ms"] = lat
            history.append(entry)

        # ── Phase 4: return best by measured latency ──
        return _best_from_history(history, param_fields, top_k)

    return propose


def _best_from_history(history: List[Dict[str, Any]], param_fields: List[str], top_k: int) -> List[Dict[str, Any]]:
    scored = []
    for entry in history:
        cfg = entry.get("config", entry)
        lat = entry.get("latency_ms", entry.get("ga_latency_ms", float("inf")))
        scored.append((float(lat), _strip_config_dict(cfg, param_fields)))

    scored.sort(key=lambda x: x[0])

    result = []
    seen = set()
    for _, cfg in scored:
        key = _config_dict_key(cfg)
        if key not in seen:
            seen.add(key)
            result.append(cfg)
            if len(result) >= top_k:
                break
    return result


# ---------------------------------------------------------------------------
# Deprecated API (kept for backward compatibility)
# ---------------------------------------------------------------------------


def predict_configs(
    operator_id: str,
    shape: Dict[str, Any],
    kernel_variant: Optional[str] = None,
) -> Optional[List[Any]]:
    if not _is_enabled():
        return None

    try:
        from triton.flagtune.registry import get as _get_op

        op_info = _get_op(operator_id)
        model = _get_model_manager().load(operator_id)
    except Exception as exc:
        logger.warning("FlagTune init failed for %s: %s", operator_id, exc)
        return None

    kv = _resolve_kernel_variant(op_info, kernel_variant, shape)

    to_config = op_info.to_config or _generic_to_config
    top_k = _top_k()

    try:
        return _predict_impl(op_info, model, shape, kv, to_config, top_k)
    except Exception as exc:
        logger.warning(
            "FlagTune prediction failed for %s shape=%s: %s",
            operator_id,
            shape,
            exc,
        )
        return None


def make_early_config_prune(
    operator_id: str,
    prune_fn: Optional[Callable] = None,
) -> Callable:
    if prune_fn is None:
        try:
            from triton.ops.matmul_perf_model import early_config_prune as _ep
            prune_fn = _ep
        except (ImportError, ModuleNotFoundError):
            prune_fn = lambda configs, named_args, **kw: configs

    from triton.flagtune.registry import get as _get_op

    try:
        op_info = _get_op(operator_id)
        op_id = op_info.op_id
    except Exception:
        op_id = None

    if op_id is None:

        def _prune(configs, named_args, **kwargs):
            return prune_fn(configs, named_args, **kwargs)

        return _prune

    proposer = make_config_proposer({"op_id": op_id})

    def _prune(configs, named_args, **kwargs):
        try:
            op_info = _get_op(operator_id)
        except Exception:
            return prune_fn(configs, named_args, **kwargs)

        extract = op_info.extract_shape
        if extract is None:
            return prune_fn(configs, named_args, **kwargs)

        shape = extract(named_args)

        initial = _configs_to_dicts(configs, op_info.param_space.all_field_names)

        predicted = proposer(None, shape, initial, {"op_id": op_id})
        if predicted is not None and len(predicted) > 0:
            to_cfg = op_info.to_config or _generic_to_config
            configs = [to_cfg(d) for d in predicted]

        return prune_fn(configs, named_args, **kwargs)

    return _prune


def _configs_to_dicts(configs: List[Any], param_fields: List[str]) -> List[Dict[str, Any]]:
    result = []
    for cfg in configs:
        d = {}
        if hasattr(cfg, "kwargs"):
            for f in param_fields:
                if f in cfg.kwargs:
                    d[f] = int(cfg.kwargs[f])
        if hasattr(cfg, "num_warps"):
            d["num_warps"] = int(cfg.num_warps)
        if hasattr(cfg, "num_stages"):
            d["num_stages"] = int(cfg.num_stages)
        if hasattr(cfg, "num_ctas"):
            d["num_ctas"] = int(cfg.num_ctas)
        if d:
            result.append(d)
    return result


# ---------------------------------------------------------------------------
# Backward-compatible shims (deprecated)
# ---------------------------------------------------------------------------


def predict_matmul_configs(M: int, N: int, K: int, dtype: Any = None) -> Optional[List[Any]]:
    if not _is_enabled():
        return None
    return predict_configs(
        "mm_general_tma",
        {"M": M, "N": N, "K": K, "stride_am": K, "stride_bk": N},
    )


def flagtune_early_config_prune(configs, named_args, **kwargs):
    fn = make_early_config_prune("mm_general_tma")
    return fn(configs, named_args, **kwargs)


def _config_dict_to_triton(config: Dict[str, Any], dtype: Any = None) -> Any:
    from triton import Config

    kwargs: Dict[str, int] = {
        "BLOCK_M": int(config.get("BLOCK_M", 128)),
        "BLOCK_N": int(config.get("BLOCK_N", 32)),
        "BLOCK_K": int(config.get("BLOCK_K", 32)),
        "SPLIT_K": 1,
    }
    if "GROUP_M" in config:
        kwargs["GROUP_M"] = int(config["GROUP_M"])

    return Config(
        kwargs=kwargs,
        num_warps=int(config.get("num_warps", 4)),
        num_stages=int(config.get("num_stages", 3)),
    )


# ---------------------------------------------------------------------------
# Internal prediction engine
# ---------------------------------------------------------------------------


def _predict_config_dicts(
    op_info: Any,
    model: Any,
    shape: Dict[str, Any],
    kernel_variant: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    param_space = op_info.param_space
    pipeline = op_info.feature_pipeline

    candidate_rows = []
    for order, config in enumerate(param_space.iter_configs(kernel_variant)):
        row = {**shape, **config}
        row["_config_order"] = order
        candidate_rows.append(row)

    if not candidate_rows:
        return []

    candidates_df = pd.DataFrame(candidate_rows)
    features = pipeline.build(candidates_df)
    X_pred = features.reindex(columns=model.feature_cols, fill_value=0)
    X_pred = X_pred.to_numpy(dtype=float)

    scores = model.predict(X_pred)
    candidates_df["xgb_rank_score"] = scores

    candidates_df = candidates_df.sort_values(
        ["xgb_rank_score", "_config_order"],
        ascending=[False, True],
    )

    validate_fn = getattr(op_info, "validate_shape_config", None)

    results = []
    config_fields = param_space.all_field_names
    for _, row in candidates_df.iterrows():
        config_dict = {}
        for fname in config_fields:
            if fname in row:
                config_dict[fname] = int(row[fname])
        if validate_fn is not None and not validate_fn(shape, config_dict):
            continue
        results.append(config_dict)
        if len(results) >= top_k:
            break

    return results


def _predict_impl(
    op_info: Any,
    model: Any,
    shape: Dict[str, Any],
    kernel_variant: Optional[str],
    to_config: Callable,
    top_k: int,
) -> List[Any]:
    config_dicts = _predict_config_dicts(op_info, model, shape, kernel_variant, top_k)
    return [to_config(d) for d in config_dicts]
