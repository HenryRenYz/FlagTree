"""
FlagTune model manager — resolution, download, caching, and version validation.

Model resolution follows a four-tier priority::

1. ``$TRITON_FLAGTUNE_MODEL_DIR/{operator_id}/``    (user-specified local)
2. ``~/.flagtree/flagtune_models/{operator_id}/``   (local cache)
3. ``<package>/model/``                              (bundled fallback, mm only)
4. Remote download → local cache
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from triton.flagtune._version import __version__ as _FLAGTUNE_VERSION

logger = logging.getLogger(__name__)


def _cache_root() -> Path:
    env = os.environ.get("FLAGTUNE_MODEL_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".flagtree" / "flagtune_models"


def _bundled_model_dir() -> Path:
    return Path(__file__).resolve().parent / "model"


def _user_model_root() -> Optional[Path]:
    env = os.environ.get("TRITON_FLAGTUNE_MODEL_DIR", "").strip()
    if env:
        return Path(env)
    return None


class IncompatibleModelError(RuntimeError):
    pass


def _parse_version(ver: str) -> tuple:
    parts = []
    for p in str(ver).split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


class FlagTuneModelManager:

    def __init__(self) -> None:
        self._loaded: Dict[str, Any] = {}  # operator_id -> XGBoostPredictor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, operator_id: str) -> Any:
        if operator_id in self._loaded:
            return self._loaded[operator_id]

        model_dir = self.resolve(operator_id)
        predictor = self._load_predictor(model_dir)
        self._loaded[operator_id] = predictor
        return predictor

    def resolve(self, operator_id: str) -> Path:
        # Tier 1 — user-specified local directory
        user_root = _user_model_root()
        if user_root is not None:
            candidate = user_root / operator_id
            if self._is_valid_model_dir(candidate):
                logger.info("FlagTune model found at user path: %s", candidate)
                return candidate

        # Tier 2 — local cache
        cache_dir = _cache_root() / operator_id
        if cache_dir.is_dir():
            versions = sorted(cache_dir.iterdir(), reverse=True)
            for vd in versions:
                if vd.is_dir() and self._is_valid_model_dir(vd):
                    logger.info("FlagTune model found in cache: %s", vd)
                    return vd

        # Tier 3 — bundled (backward compat for mm_general_tma)
        bundled = _bundled_model_dir()
        if self._is_valid_model_dir(bundled):
            logger.info("FlagTune model found bundled: %s", bundled)
            return bundled

        # Tier 4 — remote download
        if not os.environ.get("FLAGTUNE_DISABLE_REMOTE"):
            remote_dir = self._download(operator_id)
            if remote_dir is not None:
                return remote_dir

        raise FileNotFoundError(f"FlagTune model not found for operator {operator_id!r}. "
                                f"Checked: user dir, cache ({_cache_root()}), bundled, and remote.")

    # ------------------------------------------------------------------
    # Model validation
    # ------------------------------------------------------------------

    def _is_valid_model_dir(self, d: Path) -> bool:
        return (d / "feature_schema.json").is_file() and (d / "xgboost_ranker.json").is_file()

    def _validate_model_dir(self, d: Path) -> None:
        schema_path = d / "feature_schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(f"feature_schema.json not found in {d}")
        with schema_path.open("r", encoding="utf-8") as fh:
            schema = json.load(fh)

        min_ver = schema.get("flagtune_version_min")
        if min_ver:
            if _parse_version(min_ver) > _parse_version(_FLAGTUNE_VERSION):
                raise IncompatibleModelError(f"Model at {d} requires FlagTune >= {min_ver}, "
                                             f"but current version is {_FLAGTUNE_VERSION}")

        max_ver = schema.get("flagtune_version_max")
        if max_ver:
            if _parse_version(max_ver) < _parse_version(_FLAGTUNE_VERSION):
                logger.warning(
                    "Model at %s was built for FlagTune <= %s, "
                    "current version is %s — compatibility not guaranteed",
                    d,
                    max_ver,
                    _FLAGTUNE_VERSION,
                )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_predictor(self, model_dir: Path) -> Any:
        self._validate_model_dir(model_dir)

        with (model_dir / "feature_schema.json").open("r", encoding="utf-8") as fh:
            schema = json.load(fh)

        predictor = _XGBoostPredictorCompat(
            model_dir=model_dir,
            schema=schema,
        )
        return predictor

    # ------------------------------------------------------------------
    # Remote download
    # ------------------------------------------------------------------

    def _download(self, operator_id: str) -> Optional[Path]:
        from triton.flagtune.model_urls import resolve_url

        url = resolve_url(operator_id)
        if url is None:
            logger.info("No remote URL configured for operator %s", operator_id)
            return None

        dest = _cache_root() / operator_id / "latest"
        dest.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading FlagTune model for %s from %s", operator_id, url)
        try:
            from urllib.request import Request, urlopen
            req = Request(url, headers={"User-Agent": f"FlagTune/{_FLAGTUNE_VERSION}"})
            with urlopen(req, timeout=120) as resp:
                data = resp.read()

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                if url.endswith(".tar.gz") or url.endswith(".tgz"):
                    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tf:
                        tf.extractall(path=str(tmp_path))
                elif url.endswith(".tar"):
                    with tarfile.open(fileobj=BytesIO(data), mode="r:") as tf:
                        tf.extractall(path=str(tmp_path))
                elif url.endswith(".zip"):
                    import zipfile
                    with zipfile.ZipFile(BytesIO(data)) as zf:
                        zf.extractall(path=str(tmp_path))
                else:
                    # Assume it's raw JSON files; write directly
                    (tmp_path / "feature_schema.json").write_bytes(data)
                    (tmp_path / "xgboost_ranker.json").write_bytes(data)

                # If extraction produced a single top-level dir, use it
                entries = list(tmp_path.iterdir())
                if len(entries) == 1 and entries[0].is_dir():
                    extracted = entries[0]
                else:
                    extracted = tmp_path

                if not self._is_valid_model_dir(extracted):
                    logger.warning("Downloaded archive at %s lacks expected model files", url)
                    return None

                # Clear and copy
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(extracted, dest)
                logger.info("FlagTune model cached to %s", dest)
                return dest

        except Exception:
            logger.warning("Failed to download model for %s from %s", operator_id, url, exc_info=True)
            return None


# ---------------------------------------------------------------------------
# Minimal "XGBoostPredictor" compatible class for the generic flow.
# This is intentionally decoupled from the adapters so that model loading
# does not need to know about mm/gemv/etc ahead of time.
# ---------------------------------------------------------------------------


class _XGBoostPredictorCompat:

    def __init__(self, model_dir: Path, schema: Dict[str, Any]) -> None:
        from xgboost import XGBRanker

        self._model_dir = model_dir
        self._schema = schema
        self.feature_cols: List[str] = list(schema.get("feature_cols", []))

        model_file = schema.get("model_file", "xgboost_ranker.json")
        model_path = model_dir / model_file
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = XGBRanker()
        self._model.load_model(str(model_path))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
