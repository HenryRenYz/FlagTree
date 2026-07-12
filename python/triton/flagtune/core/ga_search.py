"""
Generic Genetic Algorithm (GA) config search.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from triton.flagtune.core.interfaces import ParameterSpace


@dataclass
class GAParams:
    generations: int
    population_size: int
    elite_size: int
    offspring_per_generation: int
    mutation_rate: float
    random_rate: float
    max_evaluations: int = 0


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None


class GASearcher:

    def __init__(self, param_space: ParameterSpace, ga_params: GAParams, seed: int = 42) -> None:
        self.param_space = param_space
        self.ga_params = ga_params
        self._rng = random.Random(seed)

    def generate(
        self,
        entries: Sequence[Dict[str, Any]],
        kernel_variant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not entries:
            return []
        if self.ga_params.generations <= 0 or self.ga_params.offspring_per_generation <= 0:
            return []
        population = self._initial_population(entries, kernel_variant)
        if not population:
            return []
        generated_limit = self.ga_params.offspring_per_generation
        if self.ga_params.max_evaluations and self.ga_params.max_evaluations > 0:
            generated_limit = min(
                generated_limit,
                max(0, self.ga_params.max_evaluations - len(population)),
            )
        if generated_limit <= 0:
            return []
        known_entries: List[Dict[str, Any]] = list(population)
        known_keys: Set[Tuple[Tuple[str, int], ...]] = {self._entry_key(e, kernel_variant) for e in known_entries}
        generated_history: List[Dict[str, Any]] = []
        base_entry = entries[0]
        for generation in range(1, self.ga_params.generations + 1):
            offspring = self._next_generation(
                base_entry=base_entry,
                known_entries=known_entries,
                known_keys=known_keys,
                kernel_variant=kernel_variant,
                generation=generation,
                target_count=self.ga_params.offspring_per_generation,
            )
            if not offspring:
                continue
            for entry in offspring:
                known_keys.add(self._entry_key(entry, kernel_variant))
            known_entries.extend(offspring)
            generated_history.extend(offspring)
        return generated_history[-generated_limit:]

    def crossover(
        self,
        parent_a: Dict[str, Any],
        parent_b: Dict[str, Any],
        kernel_variant: Optional[str] = None,
    ) -> Dict[str, Any]:
        flat_a = self._flatten_config(parent_a.get("config", parent_a))
        flat_b = self._flatten_config(parent_b.get("config", parent_b))
        child: Dict[str, Any] = {}
        active_fields = self.param_space.active_field_names(kernel_variant)
        for field in active_fields:
            source = flat_a if self._rng.random() < 0.5 else flat_b
            if field in source and source[field] is not None:
                child[field] = int(source[field])
        return child

    def mutate(
        self,
        flat: Dict[str, Any],
        kernel_variant: Optional[str] = None,
    ) -> Dict[str, Any]:
        out = dict(flat)
        field_values = self.param_space._field_values_for_variant(kernel_variant)
        for field, choices in field_values.items():
            if not choices:
                continue
            if field not in out or self._rng.random() < self.ga_params.mutation_rate:
                out[field] = int(self._rng.choice(choices))
        return out

    def _validate_flat(self, flat: Dict[str, Any], kernel_variant: Optional[str] = None) -> bool:
        return self.param_space.validate(flat, kernel_variant)

    def _flatten_config(self, config: Dict[str, Any]) -> Dict[str, Optional[int]]:
        meta = config.get("META", {}) if isinstance(config, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        result: Dict[str, Optional[int]] = {}
        all_fields = self.param_space.all_field_names + ["num_warps", "num_ctas", "num_stages"]
        for key in all_fields:
            val = config.get(key, meta.get(key))
            result[key] = _parse_int(val)
        if "GROUP_M" in result and result["GROUP_M"] is None:
            result["GROUP_M"] = 8
        return result

    def _entry_key(self, entry: Dict[str, Any], kernel_variant: Optional[str] = None) -> Tuple[Tuple[str, int], ...]:
        return self.param_space.config_key(
            self._flatten_config(entry.get("config", entry)),
            kernel_variant,
        )

    def _initial_population(
        self,
        entries: Sequence[Dict[str, Any]],
        kernel_variant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        population: List[Dict[str, Any]] = []
        seen: Set[Tuple[Tuple[str, int], ...]] = set()
        sorted_entries = sorted(
            entries,
            key=lambda item: (
                _parse_int(item.get("candidate_rank")) is None,
                _parse_int(item.get("candidate_rank")) or 0,
            ),
        )
        for entry in sorted_entries:
            cloned = self._clone_entry(
                entry,
                config=entry.get("config", entry),
                generation=0,
                source="topk",
                candidate_rank=_parse_int(entry.get("candidate_rank")),
            )
            key = self._entry_key(cloned, kernel_variant)
            if key in seen:
                continue
            seen.add(key)
            population.append(cloned)
        return population

    @staticmethod
    def _clone_entry(
        base: Dict[str, Any],
        config: Dict[str, Any],
        generation: int,
        source: str,
        candidate_rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        entry = dict(base)
        entry["config"] = config
        entry["ga_generation"] = generation
        entry["ga_source"] = source
        if candidate_rank is not None:
            entry["candidate_rank"] = candidate_rank
        elif generation > 0:
            entry.pop("candidate_rank", None)
        return entry

    def _config_from_flat(self, flat: Dict[str, Any], kernel_variant: Optional[str] = None) -> Dict[str, Any]:
        meta = {
            k: int(flat[k])
            for k in self.param_space.active_field_names(kernel_variant)
            if k in flat and flat[k] is not None
        }
        config: Dict[str, Any] = {"META": meta}
        for key in ("num_warps", "num_stages", "num_ctas"):
            if key in flat and flat[key] is not None:
                config[key] = int(flat[key])
        return config

    def _random_flat(self, kernel_variant: Optional[str] = None) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        field_values = self.param_space._field_values_for_variant(kernel_variant)
        for field, choices in field_values.items():
            if choices:
                flat[field] = int(self._rng.choice(choices))
        return flat

    def _parent_pool(
        self,
        known_entries: Sequence[Dict[str, Any]],
        kernel_variant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        seen: Set[Tuple[Tuple[str, int], ...]] = set()
        parents: List[Dict[str, Any]] = []

        def add(entry: Dict[str, Any]) -> None:
            key = self._entry_key(entry, kernel_variant)
            if key not in seen:
                seen.add(key)
                parents.append(entry)

        ranked_entries = sorted(
            known_entries,
            key=lambda item: (
                item.get("ga_latency_ms") is None,
                float(item.get("ga_latency_ms") or 0.0),
                _parse_int(item.get("candidate_rank")) is None,
                _parse_int(item.get("candidate_rank")) or 0,
            ),
        )
        for entry in ranked_entries[:max(1, self.ga_params.elite_size)]:
            add(entry)
        for entry in ranked_entries:
            if len(parents) >= max(1, self.ga_params.population_size):
                break
            add(entry)
        return parents or list(known_entries[:1])

    def _next_generation(
        self,
        base_entry: Dict[str, Any],
        known_entries: Sequence[Dict[str, Any]],
        known_keys: Set[Tuple[Tuple[str, int], ...]],
        kernel_variant: Optional[str] = None,
        generation: int = 1,
        target_count: int = 50,
    ) -> List[Dict[str, Any]]:
        parents = self._parent_pool(known_entries, kernel_variant)
        offspring: List[Dict[str, Any]] = []
        batch_keys: Set[Tuple[Tuple[str, int], ...]] = set()
        max_attempts = max(100, target_count * 50)
        attempts = 0
        while len(offspring) < target_count and attempts < max_attempts:
            attempts += 1
            if self._rng.random() < self.ga_params.random_rate or len(parents) == 1:
                child_flat = self._random_flat(kernel_variant)
                source = "random"
            else:
                parent_a, parent_b = self._rng.sample(parents, 2)
                child_flat = self.crossover(parent_a, parent_b, kernel_variant)
                source = "crossover"
            child_flat = self.mutate(child_flat, kernel_variant)
            if not self._validate_flat(child_flat, kernel_variant):
                continue
            key = self.param_space.config_key(child_flat, kernel_variant)
            if key in known_keys or key in batch_keys:
                continue
            batch_keys.add(key)
            offspring.append(
                self._clone_entry(
                    base_entry,
                    config=self._config_from_flat(child_flat, kernel_variant),
                    generation=generation,
                    source=source,
                ))
        return offspring
