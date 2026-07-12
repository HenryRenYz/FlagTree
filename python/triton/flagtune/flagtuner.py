"""
Flagtuner — Autotuner subclass with FlagTune config prediction.

Usage:
    from triton.flagtune import flagtune

    @flagtune(
        configs=[...],
        key=["M", "N", "K"],
        flagtune_op_id="flagtree/gemm",
        prune_configs_by={"perf_model": estimate_time, "top_k": 10},
    )
    @triton.jit
    def kernel(...):
        ...

When flagtune_op_id is set and TRITON_USE_FLAGTUNE=1, the tuner
replaces the pruned config list with XGBoost-predicted Top-K configs
via the ConfigProposer API.  Otherwise behaves exactly like Autotuner.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from triton.runtime.autotuner import Autotuner

logger = logging.getLogger(__name__)


def _configs_to_dicts(configs: List[Any], param_fields: List[str]) -> List[Dict[str, Any]]:
    result = []
    for cfg in configs:
        d: Dict[str, Any] = {}
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


class Flagtuner(Autotuner):

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Optional[Dict] = None,
        warmup=25,
        rep=100,
        use_cuda_graph=False,
        flagtune_op_id: Optional[str] = None,
    ):
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
        )

        self._flagtune_op_id = flagtune_op_id
        self._flagtune_proposer = None
        self._flagtune_op_info = None
        self._flagtune_init_failed = False

    def _ensure_flagtune(self) -> None:
        if self._flagtune_proposer is not None:
            return
        if self._flagtune_init_failed:
            return
        if not self._flagtune_op_id:
            return

        from triton.flagtune import is_enabled as _is_enabled

        if not _is_enabled():
            return

        try:
            from triton.flagtune.registry import (
                get as _get_op,
                resolve_operator_id,
            )
            from triton.flagtune.predict import make_config_proposer

            operator_id = resolve_operator_id(self._flagtune_op_id)
            self._flagtune_op_info = _get_op(operator_id)
            self._flagtune_proposer = make_config_proposer({"op_id": self._flagtune_op_id})
        except Exception as exc:
            self._flagtune_init_failed = True
            logger.warning(
                "FlagTune init failed for op_id=%s: %s",
                self._flagtune_op_id,
                exc,
            )

    def prune_configs(self, kwargs):
        pruned = super().prune_configs(kwargs)

        self._ensure_flagtune()
        if self._flagtune_proposer is None or self._flagtune_op_info is None:
            return pruned

        op_info = self._flagtune_op_info
        extract = getattr(op_info, "extract_shape", None)
        if extract is None:
            return pruned

        try:
            shape = extract(self.nargs)
        except Exception:
            return pruned

        param_fields = op_info.param_space.all_field_names
        initial = _configs_to_dicts(pruned, param_fields)
        meta = {"op_id": self._flagtune_op_id}

        try:
            config_dicts = self._flagtune_proposer(None, shape, initial, meta)
        except Exception as exc:
            logger.warning("FlagTune propose failed: %s", exc)
            return pruned

        if not config_dicts:
            return pruned

        to_cfg = getattr(op_info, "to_config", None)
        if to_cfg is None:
            from triton.flagtune.predict import _generic_to_config
            to_cfg = _generic_to_config

        result = []
        for d in config_dicts:
            try:
                result.append(to_cfg(d))
            except Exception:
                pass

        if not result:
            return pruned

        if self.early_config_prune:
            result = self.early_config_prune(result, self.nargs, **kwargs)

        return result if result else pruned


def flagtune(
    configs,
    key,
    *,
    flagtune_op_id: Optional[str] = None,
    prune_configs_by: Optional[Dict] = None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=25,
    rep=100,
    use_cuda_graph=False,
):

    def decorator(fn):
        return Flagtuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
            flagtune_op_id=flagtune_op_id,
        )

    return decorator
