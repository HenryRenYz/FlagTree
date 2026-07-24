"""Test kernel-independent FlagTune ranking preparation, progress, and export.

The suite registers a synthetic operator, writes small contiguous JSONL groups,
and verifies feature order, deterministic sampling, grouping errors, portable
text progress, and compatibility of exported models with ``XGBRanker``.
"""

from __future__ import annotations

import builtins
import json

import numpy as np
import pytest

from triton.flagtune.contract.archive import read_model_archive
from triton.flagtune.contract.operator_schema import parse_operator_config
from triton.flagtune.contract.identity import ModelIdentity, gpu_metadata
from triton.flagtune.training.ranker import (
    TrainingDataError,
    XGBoostTrainingOptions,
    export_ranker_model,
    prepare_ranking_data,
    train_xgboost_ranker,
)
import triton.flagtune.training.ranker as training

GPU = dict(gpu_metadata(
    backend="cuda",
    vendor="nvidia",
    device_name="NVIDIA H800 80GB HBM3",
    architecture="sm90",
))
IDENTITY = ModelIdentity(GPU["gpu_key"], "tests/train", "kernel", "bf16-bf16-f32")
DTYPES = ["bfloat16", "bfloat16", "float32"]


def _config():
    """Return a minimal registry definition with raw and derived features."""
    return {
        "op_id": "tests/train",
        "variants": {
            "kernel": {
                "inputs": {"M": {}, "N": {}},
                "params": {
                    "BLOCK": {"values": [16, 32]},
                    "num_warps": {"values": [2, 4]},
                },
                "features": [
                    "M",
                    "N",
                    "BLOCK",
                    "num_warps",
                    {"name": "tile", "op": "mul", "args": ["BLOCK", "num_warps"]},
                ],
            }
        },
    }


def _write_data(path, shape_order=("a", "b")):
    """Write deterministic per-config latencies in requested shape-group order."""
    configs = [
        {"BLOCK": 16, "num_warps": 2},
        {"BLOCK": 16, "num_warps": 4},
        {"BLOCK": 32, "num_warps": 2},
        {"BLOCK": 32, "num_warps": 4},
    ]
    inputs = {"a": {"M": 64, "N": 32}, "b": {"M": 128, "N": 64}}
    latencies = {"a": [4.0, 3.0, 2.0, 1.0], "b": [1.0, 2.0, 3.0, 4.0]}
    with path.open("w", encoding="utf-8") as handle:
        for shape_name in shape_order:
            for config, latency in zip(configs, latencies[shape_name]):
                handle.write(
                    json.dumps({
                        "schema_version": 2,
                        "ranking_group": {
                            "operator_id": "tests/train",
                            "variant": "kernel",
                            "dimensions": inputs[shape_name],
                            "model_dtype_key": "bf16-bf16-f32",
                        },
                        "inputs": inputs[shape_name],
                        "config": config,
                        "latency_ms": latency,
                    }) + "\n")


def test_prepare_ranking_data_preserves_feature_order_and_shape_groups(tmp_path):
    """Build float32 features, descending relevance labels, and group sizes."""
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)

    data = prepare_ranking_data(
        variant,
        data_path,
        XGBoostTrainingOptions(min_train_rows=2, show_progress=False),
    )

    assert data.features.dtype == np.float32
    assert data.features.shape == (8, 5)
    assert data.group_sizes == [4, 4]
    assert data.labels.tolist() == [0, 1, 2, 3, 3, 2, 1, 0]
    assert data.features[0].tolist() == [64, 32, 16, 2, 32]


def test_prepare_ranking_data_sampling_is_reproducible(tmp_path):
    """Sample the same source rows for repeated calls with one fixed seed."""
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)
    options = XGBoostTrainingOptions(
        min_train_rows=2,
        max_configs_per_shape=2,
        seed=17,
        show_progress=False,
    )

    first = prepare_ranking_data(variant, data_path, options)
    second = prepare_ranking_data(variant, data_path, options)

    assert first.group_sizes == [2, 2]
    assert first.sampled_out_rows == 4
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.labels, second.labels)


def test_prepare_ranking_data_rejects_noncontiguous_shape_groups(tmp_path):
    """Reject a ranking group that reappears after a different query group."""
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path, shape_order=("a", "b", "a"))

    with pytest.raises(TrainingDataError, match="not contiguous"):
        prepare_ranking_data(
            variant,
            data_path,
            XGBoostTrainingOptions(min_train_rows=2, show_progress=False),
        )


def test_prepare_ranking_data_rejects_group_dimensions_mismatching_inputs(tmp_path):
    """Require the public ranking-group identity to match feature inputs."""
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    for row in rows[:4]:
        row["ranking_group"]["dimensions"]["M"] = 999
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(TrainingDataError, match="dimensions do not match inputs"):
        prepare_ranking_data(
            variant,
            data_path,
            XGBoostTrainingOptions(min_train_rows=2, show_progress=False),
        )


def test_prepare_ranking_data_rejects_mixed_gpu_and_dtype_identity(tmp_path):
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    for index, row in enumerate(rows):
        row.update({
            "model_identity": {
                "gpu_key": ("nvidia-h800-sm90" if index < 4 else "nvidia-h20-sm90"),
                "dtype_key": "bf16-bf16-f32",
            },
            "dtypes": {
                "inputs": ["bfloat16", "bfloat16"],
                "outputs": ["float32"],
            },
        })
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(TrainingDataError, match="mixes GPU identities"):
        prepare_ranking_data(
            variant,
            data_path,
            XGBoostTrainingOptions(min_train_rows=2, show_progress=False),
        )


def test_train_and_export_model_is_loadable_by_xgboost_ranker(tmp_path):
    """Fit a small ranker and reload its model/schema bundle with XGBoost."""
    xgboost = pytest.importorskip("xgboost")
    variant = parse_operator_config(_config()).get_variant("kernel")
    data_path = tmp_path / "benchmark.jsonl"
    _write_data(data_path)

    model, summary = train_xgboost_ranker(
        variant,
        data_path,
        XGBoostTrainingOptions(
            n_estimators=4,
            max_depth=2,
            min_train_rows=2,
            n_jobs=1,
            show_progress=False,
        ),
    )
    exported = export_ranker_model(
        model,
        variant,
        tmp_path,
        summary,
        identity=IDENTITY,
        dtypes=DTYPES,
        gpu=GPU,
        model_version="1.0.0",
    )

    loaded = xgboost.XGBRanker()
    members = read_model_archive(exported.model_path)
    loaded.load_model(bytearray(members["xgboost_ranker.json"]))
    assert len(loaded.predict(np.zeros((2, len(variant.feature_names))))) == 2
    yaml = pytest.importorskip("yaml")
    saved_config = yaml.safe_load(members["flagtune_config.yaml"])
    assert saved_config == exported.model_config
    assert saved_config["format_version"] == 5
    assert saved_config["model_version"] == "1.0.0"
    assert saved_config["gpu_key"] == GPU["gpu_key"]
    assert saved_config["dtype_key"] == "bf16-bf16-f32"
    assert saved_config["op_id"] == "tests/train"
    assert saved_config["variant"] == "kernel"
    assert list(exported.model_path.parent.iterdir()) == [exported.model_path]


def test_xgboost_progress_uses_flushed_text_without_tqdm(monkeypatch, capsys):
    """Report boosting rounds through the console fallback when tqdm is absent."""
    real_import = builtins.__import__

    def import_without_tqdm(name, globals=None, locals=None, fromlist=(), level=0):
        """Raise only for tqdm imports and delegate every other import unchanged."""
        if name == "tqdm" or name.startswith("tqdm."):
            raise ImportError("tqdm intentionally unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_tqdm)
    callbacks, progress = training._progress_callback(total=4, enabled=True)

    assert progress is None
    assert len(callbacks) == 1
    for epoch in range(4):
        assert callbacks[0].after_iteration(None, epoch, {}) is False
    output = capsys.readouterr().out
    assert "XGBoost progress: 1/4 trees" in output
    assert "XGBoost progress: 4/4 trees" in output
