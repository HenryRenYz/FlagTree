"""Resolve remote locations for exact FlagTune model identities.

The model manager calls this module only after local model roots have no usable
archive.  It maps the complete identity
``gpu_key/op_id/variant/dtype_key`` to ``(model_version, URL)`` using three
sources in priority order: ``FLAGTUNE_MODEL_URLS``, a 24-hour cached/downloaded
manifest, then ``_BUILTIN_TABLE``.  It validates version strings and URL shape,
but archive download, archive validation, and cache writes belong to
:mod:`model_manager`.

Environment variables:
  * ``FLAGTUNE_MODEL_URLS``: either a path to a JSON document or an inline JSON
    document. Its top level may directly map artifact keys, or put those keys
    below ``models``. It overrides all packaged and remote-manifest entries.
  * ``FLAGTUNE_MODEL_CACHE``: indirectly selects the manifest cache location
    through :func:`triton.flagtune.runtime.model_loader._cache_root`.

An entry has the form ``{"versions": {"1.2.3": {"url": "https://..."}}}``.
The optional ``latest`` and ``sha256`` fields seen in examples are currently
not read.  URLs must point to gzip tar archives for the model manager, and no
checksum, signature, redirect policy, or concurrent cache coordination is
implemented here.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from triton.flagtune.contract.archive import parse_model_version, validate_model_version
from triton.flagtune.contract.identity import ModelIdentity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in URL table (ultimate fallback).
#
# This table packages emergency/default locations for identities that cannot be
# centrally published yet.  Add a complete artifact key and one or more strict
# SemVer entries in the schema documented above.  ``resolve_artifact`` chooses
# the highest valid key in ``versions`` when no version is requested; ``latest``
# is descriptive only.  Keep entries small and maintained: the current URLs are
# placeholders, and an unavailable URL is indistinguishable from no model.
# TODO: Populate this table with maintained, reachable model archive URLs.
# ---------------------------------------------------------------------------

_BUILTIN_TABLE: Dict[str, Dict[str, Any]] = {
    # "nvidia-h100-pcie-80gb-sm90/acme/mm/general_tma/bf16-bf16-bf16": {
    #     "latest": "1.0.0",
    #     "versions": {
    #         "1.0.0": {
    #             "url": "https://models.example.invalid/flagtune/acme_mm_general_tma_v1.0.0.tar.gz",
    #             "sha256": "...",
    #         },
    #     },
    # },
}

# ---------------------------------------------------------------------------
# Remote manifest
# ---------------------------------------------------------------------------

_MANIFEST_URL = "https://models.flagtree.ai/flagtune/manifest.json" # TODO just a simple fallback, make it real
_MANIFEST_MAX_AGE_S = 24 * 3600  # 24 hours


def _env_model_urls() -> Optional[Dict[str, Any]]:
    """Load the explicit override JSON from ``FLAGTUNE_MODEL_URLS``, if present."""
    raw = os.environ.get("FLAGTUNE_MODEL_URLS", "").strip()
    if not raw:
        return None
    # Try as file path first, then as JSON string
    path = Path(raw)
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            logger.warning("Cannot parse FLAGTUNE_MODEL_URLS as JSON file: %s", path)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Cannot parse FLAGTUNE_MODEL_URLS env value")
    return None


def _cached_manifest_path() -> Path:
    """Return the shared manifest path inside the model-manager cache root."""
    from triton.flagtune.runtime.model_loader import _cache_root
    return _cache_root() / "manifest.json"


def _load_cached_manifest() -> Optional[Dict[str, Any]]:
    """Return an unexpired cached manifest; malformed and stale files are ignored."""
    p = _cached_manifest_path()
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    age = time.time() - data.get("_fetched_at", 0)
    if age > _MANIFEST_MAX_AGE_S:
        return None
    return data


def _download_manifest() -> Optional[Dict[str, Any]]:
    """Fetch and cache the well-known manifest, returning ``None`` on failure."""
    try:
        from urllib.request import Request, urlopen
        req = Request(_MANIFEST_URL, headers={"User-Agent": "FlagTune/0.2"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        data["_fetched_at"] = time.time()
        p = _cached_manifest_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return data
    except Exception:
        logger.debug("Failed to download remote manifest", exc_info=True)
        return None


def _lookup_model_entry(op_id: str, variant: str, *, gpu_key: str, dtype_key: str) -> Optional[Dict[str, Any]]:
    """Find one raw entry using override, cached/remote manifest, then builtin data."""
    key = ModelIdentity(gpu_key, op_id, variant, dtype_key).artifact_key
    env = _env_model_urls()
    if env:
        entry = env.get(key) or env.get("models", {}).get(key)
        if entry:
            return entry

    manifest = _load_cached_manifest()
    if manifest is None:
        manifest = _download_manifest()
    if manifest:
        entry = manifest.get("models", {}).get(key)
        if entry:
            return entry

    return _BUILTIN_TABLE.get(key)


def resolve_artifact(
    op_id: str,
    variant: str,
    *,
    gpu_key: str,
    dtype_key: str,
    version: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Resolve the cache version and URL for one complete model identity.

    Returns ``None`` for absent or malformed entries.  Without ``version``,
    strict SemVer precedence selects the newest valid version; build metadata
    only makes equal-precedence choices deterministic.  The function does not
    confirm that the URL is reachable or that its artifact matches this entry.
    """
    entry = _lookup_model_entry(op_id, variant, gpu_key=gpu_key, dtype_key=dtype_key)
    if entry is None:
        return None
    versions = entry.get("versions", {})
    if not isinstance(versions, dict):
        return None
    if version is not None:
        try:
            ver = validate_model_version(version)
        except ValueError:
            logger.warning("Ignoring invalid SemVer model manifest version: %r", version)
            return None
    else:
        valid_versions = []
        for candidate in versions:
            try:
                parsed = parse_model_version(candidate)
            except ValueError:
                logger.warning("Ignoring invalid SemVer model manifest version: %r", candidate)
                continue
            valid_versions.append((parsed.selection_key, candidate))
        if not valid_versions:
            return None
        ver = max(valid_versions, key=lambda item: item[0])[1]
    ver_info = versions.get(ver)
    if not isinstance(ver_info, dict):
        return None
    url = ver_info.get("url")
    if not isinstance(url, str) or not url.strip():
        return None
    return ver, url.strip()


def resolve_url(
    op_id: str,
    variant: str,
    *,
    gpu_key: str,
    dtype_key: str,
    version: Optional[str] = None,
) -> Optional[str]:
    """Return only the URL portion of :func:`resolve_artifact`."""
    artifact = resolve_artifact(
        op_id,
        variant,
        gpu_key=gpu_key,
        dtype_key=dtype_key,
        version=version,
    )
    return artifact[1] if artifact is not None else None
