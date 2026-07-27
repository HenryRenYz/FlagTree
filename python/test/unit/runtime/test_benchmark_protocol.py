import inspect
from types import SimpleNamespace

import pytest

from triton.backends.driver import BenchmarkerCapability
from triton.runtime import benchmark_protocol as benchmark_module
from triton.testing import do_bench_cudagraph


class _FakeDriver:

    def __init__(self, replay=True):
        self.replay = replay
        self.observed = {}

    def get_replay_benchmarker(self):
        if not self.replay:
            return None

        def benchmark(kernel_call, **kwargs):
            self.observed = dict(kwargs)
            kernel_call()
            return [1.0, 0.8, 1.2]

        return BenchmarkerCapability(
            identifier="fake_command_replay_v1",
            benchmarker=benchmark,
            cache_policy="warm_cache",
        )

    def get_benchmarker(self):

        def benchmark(kernel_call, **kwargs):
            self.observed = dict(kwargs)
            kernel_call()
            return [2.0, 1.8, 2.2]

        return benchmark


def test_cudagraph_helper_keeps_ten_retries_as_compatible_default():
    """Expose the old hard-coded retry count as an optional final argument."""
    parameter = inspect.signature(do_bench_cudagraph).parameters["n_retries"]

    assert parameter.default == 10
    assert list(inspect.signature(do_bench_cudagraph).parameters)[-1] == "n_retries"


def test_replay_splits_total_measurement_budget(monkeypatch):
    active = _FakeDriver()
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))
    launches = []

    resolved = benchmark_module.resolve_benchmarker(
        "replay",
        warmup_ms=25,
        measurement_ms=100,
        n_retries=10,
    )
    result = resolved.benchmark(lambda: launches.append(True), (0.5, 0.2, 0.8))

    assert result == [1.0, 0.8, 1.2]
    assert launches == [True]
    assert active.observed == {
        "rep": 10.0,
        "quantiles": (0.5, 0.2, 0.8),
        "n_retries": 10,
    }
    assert resolved.protocol.as_dict() == {
        "requested_mode": "replay",
        "resolved_mode": "replay",
        "implementation": "fake_command_replay_v1",
        "cache_policy": "warm_cache",
        "warmup_ms": 25,
        "measurement_ms": 100,
        "n_retries": 10,
        "per_replay_ms": 10.0,
        "fallback_reason": None,
    }


def test_missing_replay_capability_warns_and_resolves_event(monkeypatch):
    active = _FakeDriver(replay=False)
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=active))

    with pytest.warns(RuntimeWarning, match="falling back to event"):
        resolved = benchmark_module.resolve_benchmarker(
            "replay",
            warmup_ms=5,
            measurement_ms=20,
            n_retries=4,
        )
    result = resolved.benchmark(lambda: None, (0.5, 0.2, 0.8))

    assert result == [2.0, 1.8, 2.2]
    assert active.observed == {
        "warmup": 5,
        "rep": 20,
        "quantiles": (0.5, 0.2, 0.8),
    }
    assert resolved.protocol.resolved_mode is benchmark_module.BenchmarkMode.EVENT
    assert resolved.protocol.cache_key() == ("triton_do_bench", 5, 20)
    assert resolved.protocol.fallback_reason


@pytest.mark.parametrize("n_retries", [0, -1, True, 1.5])
def test_replay_rejects_invalid_retry_count(monkeypatch, n_retries):
    monkeypatch.setattr(benchmark_module, "driver", SimpleNamespace(active=_FakeDriver()))
    with pytest.raises(ValueError, match="n_retries"):
        benchmark_module.resolve_benchmarker(
            "replay",
            warmup_ms=5,
            measurement_ms=20,
            n_retries=n_retries,
        )
