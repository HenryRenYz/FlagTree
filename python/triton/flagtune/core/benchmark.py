"""
Generic config benchmark measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from triton.flagtune.core.interfaces import KernelAdapter
from triton.flagtune.core.ga_search import GAParams, GASearcher


@dataclass
class BenchmarkResult:
    shape: str
    shape_key: str
    kernel_variant: str
    latency_ms: float
    best_config: Dict[str, Any]
    config_count: int
    all_timings: Dict[str, float] = field(default_factory=dict)
    ga_generated_count: int = 0
    elapsed_s: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @classmethod
    def error_result(cls, shape: str, shape_key: str, kernel_variant: str, error: str) -> "BenchmarkResult":
        return cls(
            shape=shape,
            shape_key=shape_key,
            kernel_variant=kernel_variant,
            latency_ms=float("inf"),
            best_config={},
            config_count=0,
            success=False,
            error=error,
        )


class ConfigBenchmarker:

    def __init__(self, adapter: KernelAdapter, warmup: int = 5, rep: int = 10) -> None:
        self.adapter = adapter
        self.warmup = warmup
        self.rep = rep

    def measure(
        self,
        shape: Dict[str, Any],
        configs: List[Dict[str, Any]],
        kernel_variant: str,
        shape_key: Optional[str] = None,
    ) -> BenchmarkResult:
        shape_key = shape_key or str(shape)
        start = time.perf_counter()
        try:
            tuner = self.adapter.find_tuner(kernel_variant)
            if tuner is None:
                return BenchmarkResult.error_result(str(shape), shape_key, kernel_variant,
                                                    f"Cannot find tuner for {kernel_variant}")
            config_objs = self.adapter.make_config_objects(configs, kernel_variant)
            if not config_objs:
                return BenchmarkResult.error_result(str(shape), shape_key, kernel_variant, "No valid config objects")
            self.adapter.install_configs(tuner, config_objs)
            timings: Dict[str, float] = {}
            best_config = None
            best_latency = float("inf")
            for i, (config_dict, config_obj) in enumerate(zip(configs, config_objs)):
                try:
                    args, kwargs = self.adapter.make_bench_args(shape, config_dict, kernel_variant)
                    latency = tuner._bench(*args, config=config_obj, **kwargs)
                    if isinstance(latency, (list, tuple)):
                        latency = float(latency[0])
                    latency = float(latency)
                    config_key = f"config_{i}"
                    timings[config_key] = latency
                    if latency < best_latency:
                        best_latency = latency
                        best_config = config_dict
                except Exception:
                    timings[f"config_{i}_error"] = float("inf")
                    continue
            elapsed = time.perf_counter() - start
            return BenchmarkResult(
                shape=str(shape),
                shape_key=shape_key,
                kernel_variant=kernel_variant,
                latency_ms=best_latency if best_config else float("inf"),
                best_config=best_config or {},
                config_count=len(configs),
                all_timings=timings,
                elapsed_s=elapsed,
                success=best_config is not None,
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            return BenchmarkResult.error_result(str(shape), shape_key, kernel_variant, str(e))

    def measure_with_ga(
        self,
        shape: Dict[str, Any],
        seed_configs: List[Dict[str, Any]],
        kernel_variant: str,
        ga_params: GAParams,
        param_space: Any,
        shape_key: Optional[str] = None,
        ga_seed: int = 42,
    ) -> BenchmarkResult:
        shape_key = shape_key or str(shape)
        ga_count = 0
        if ga_params.generations > 0 and ga_params.offspring_per_generation > 0:
            initial_result = self.measure(shape, seed_configs, kernel_variant, shape_key)
            if not initial_result.success:
                return initial_result
            seed_entries: List[Dict[str, Any]] = []
            for i, config in enumerate(seed_configs):
                entry = {
                    "config": config,
                    "ga_latency_ms": initial_result.all_timings.get(f"config_{i}", float("inf")),
                    "candidate_rank": i + 1,
                    "ga_generation": 0,
                    "ga_source": "topk",
                }
                seed_entries.append(entry)
            ga_searcher = GASearcher(param_space, ga_params, seed=ga_seed)
            new_entries = ga_searcher.generate(seed_entries, kernel_variant)
            ga_count = len(new_entries)
            if new_entries:
                new_configs = [e["config"] for e in new_entries]
                ga_result = self.measure(shape, new_configs, kernel_variant, shape_key)
                if ga_result.success:
                    merged_timings = {**initial_result.all_timings}
                    for k, v in ga_result.all_timings.items():
                        merged_timings[f"ga_{k}"] = v
                    initial_result.all_timings = merged_timings
                    if ga_result.latency_ms < initial_result.latency_ms:
                        initial_result.latency_ms = ga_result.latency_ms
                        initial_result.best_config = ga_result.best_config
                    initial_result.config_count += ga_result.config_count
                    initial_result.elapsed_s += ga_result.elapsed_s
            initial_result.ga_generated_count = ga_count
        else:
            initial_result = self.measure(shape, seed_configs, kernel_variant, shape_key)
        return initial_result
