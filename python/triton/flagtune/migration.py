#!/usr/bin/env python3
"""Command-line migration of existing XGBoost rankers into FlagTune archives.

This is an offline packaging tool, not a trainer or runtime importer.  It loads
an existing ranker read-only, compiles the selected operator variant from YAML,
and delegates to :func:`training.export_ranker_model` to write a format-v5
``model.tar.gz`` under
``OUTPUT_ROOT/gpu_key/op_id/variant/dtype_key/model_version/``.  The resulting
archive has a fresh YAML contract and digest but preserves the ranker's learned
weights; it must therefore be used only when the old model's feature order and
parameter space already match the selected variant.

CLI arguments:
  * ``--source-model`` and ``--variant`` migrate one ranker; alternatively
    ``--legacy-model-root`` migrates the three names in ``LEGACY_MODEL_VARIANTS``.
  * ``--flagtune-config`` supplies the operator YAML, ``--output-root`` chooses
    the destination, and ``--model-version`` is strict SemVer 2.0.
  * ``--backend``, ``--gpu-vendor``, ``--gpu-name``, ``--architecture``, and ``--dtypes``
    build the exact artifact identity. ``--training-summary`` optionally adds
    JSON metadata to the exported summary.

``--architecture`` is the backend-native target used by the model identity,
for example ``sm90`` for CUDA or ``gfx942`` for HIP.  The migration code does
not select kernels or infer hardware features from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from triton.flagtune.artifacts import validate_model_version
from triton.flagtune.identity import ModelIdentity, gpu_metadata, make_dtype_key
from triton.flagtune.registry import load_operator_config
from triton.flagtune.training import export_ranker_model

LEGACY_MODEL_VARIANTS = {
    "mm_general_tma": "general_tma",
    "gemv": "gemv",
    "mm_splitk": "splitk",
}


def migrate_ranker_model(
    source_model: Path | str,
    operator_config: Path | str,
    variant_name: str,
    output_root: Path | str,
    *,
    gpu: Mapping[str, Any],
    dtypes: list[str],
    model_version: str,
    training_summary: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Copy ranker weights into a format-v5 archive without retraining.

    The new config and its digest are generated from ``operator_config``. The
    ``gpu`` must contain the canonical metadata returned by ``gpu_metadata``;
    ``dtypes`` are ordered input/output dtype names.  The source model is
    loaded read-only, and the destination path is derived as
    ``output_root/gpu_key/op_id/variant/dtype_key/model_version/model.tar.gz``.
    """
    from xgboost import XGBRanker

    operator = load_operator_config(operator_config)
    variant = operator.get_variant(variant_name)
    source = Path(source_model).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source XGBoost model does not exist: {source}")

    ranker = XGBRanker()
    ranker.load_model(str(source))
    identity = ModelIdentity(str(gpu["gpu_key"]), operator.op_id, variant.name, make_dtype_key(dtypes))
    summary: Dict[str, Any] = dict(training_summary or {})
    summary.update({
        "op_id": operator.op_id,
        "variant": variant.name,
        "migrated_from": str(source),
        "retrained": False,
    })
    exported = export_ranker_model(
        ranker,
        variant,
        output_root,
        summary,
        identity=identity,
        dtypes=dtypes,
        gpu=gpu,
        model_version=model_version,
    )
    return exported.model_path


def migrate_legacy_model_tree(
    source_root: Path | str,
    operator_config: Path | str,
    output_root: Path | str,
    *,
    gpu: Mapping[str, Any],
    dtypes: list[str],
    model_version: str,
    training_summary: Optional[Mapping[str, Any]] = None,
) -> list[Path]:
    """Migrate all three historical MM directories through the explicit map.

    Missing source files and missing variants stop the migration; no partial
    success marker is created.  This mapping is intentionally narrow and is
    not an auto-discovery mechanism for arbitrary legacy layouts.
    """
    source = Path(source_root).expanduser().resolve()
    targets = []
    for legacy_name, variant_name in LEGACY_MODEL_VARIANTS.items():
        legacy_model = source / legacy_name / "xgboost_ranker.json"
        summary = dict(training_summary or {})
        summary["legacy_model_key"] = legacy_name
        targets.append(
            migrate_ranker_model(
                legacy_model,
                operator_config,
                variant_name,
                output_root,
                gpu=gpu,
                dtypes=dtypes,
                model_version=model_version,
                training_summary=summary,
            ))
    return targets


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for deterministic model re-export."""
    parser = argparse.ArgumentParser(description="Re-export an XGBoost ranker as a format-v5 FlagTune archive.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-model")
    source.add_argument("--legacy-model-root")
    parser.add_argument("--flagtune-config", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-version", required=True, help="Strict SemVer 2.0 model revision.")
    parser.add_argument("--training-summary")
    parser.add_argument(
        "--backend",
        required=True,
        choices=("cuda", "hip"),
        help="Triton backend that produced the model data.",
    )
    parser.add_argument("--gpu-vendor", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument(
        "--architecture",
        required=True,
        help="Backend-native target, for example sm90 or gfx942.",
    )
    parser.add_argument(
        "--dtypes",
        required=True,
        help="Complete ordered tensor dtype sequence, for example bfloat16,bfloat16,float32",
    )
    return parser


def main() -> int:
    """Run the migration CLI and print the resulting canonical bundle path."""
    parser = build_parser()
    args = parser.parse_args()
    dtypes = [value.strip() for value in args.dtypes.split(",") if value.strip()]
    if not dtypes:
        parser.error("--dtypes must contain at least one dtype")
    try:
        model_version = validate_model_version(args.model_version)
    except ValueError as exc:
        parser.error(str(exc))
    gpu = gpu_metadata(
        backend=args.backend,
        vendor=args.gpu_vendor,
        device_name=args.gpu_name,
        architecture=args.architecture,
    )
    summary: Optional[Mapping[str, Any]] = None
    if args.training_summary:
        summary_path = Path(args.training_summary).expanduser().resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if args.source_model:
        if not args.variant:
            parser.error("--variant is required with --source-model")
        targets = [
            migrate_ranker_model(
                args.source_model,
                args.flagtune_config,
                args.variant,
                args.output_root,
                gpu=gpu,
                dtypes=dtypes,
                model_version=model_version,
                training_summary=summary,
            )
        ]
    else:
        if args.variant:
            parser.error("--variant cannot be used with --legacy-model-root")
        targets = migrate_legacy_model_tree(
            args.legacy_model_root,
            args.flagtune_config,
            args.output_root,
            gpu=gpu,
            dtypes=dtypes,
            model_version=model_version,
            training_summary=summary,
        )
    for target in targets:
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
