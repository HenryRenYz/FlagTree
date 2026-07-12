"""
FlagTune model URL resolution.

Models are resolved in three tiers (highest priority first):

1. ``$FLAGTUNE_MODEL_URLS`` — a JSON file or JSON string with explicit
   per-operator URL entries.
2. Cached remote manifest — downloaded from a well-known URL and cached
   for 24 hours.
3. Built-in fallback table — hard-coded URLs that serve as the ultimate
   fallback when no other source is available.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in URL table (ultimate fallback)
# ---------------------------------------------------------------------------

_BUILTIN_TABLE: Dict[str, Dict[str, Any]] = {
    # "mm_general_tma": {
    #     "latest": "1.0.0",
    #     "versions": {
    #         "1.0.0": {
    #             "url": "https://models.flagtree.ai/flagtune/mm_general_tma_v1.0.0.tar.gz",
    #             "sha256": "...",
    #         },
    #     },
    # },
}

# ---------------------------------------------------------------------------
# Remote manifest
# ---------------------------------------------------------------------------

_MANIFEST_URL = "https://models.flagtree.ai/flagtune/manifest.json"
_MANIFEST_MAX_AGE_S = 24 * 3600  # 24 hours


def _env_model_urls() -> Optional[Dict[str, Any]]:
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
    from triton.flagtune.model_manager import _cache_root
    return _cache_root() / "manifest.json"


def _load_cached_manifest() -> Optional[Dict[str, Any]]:
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


def _lookup_model_entry(operator_id: str) -> Optional[Dict[str, Any]]:
    env = _env_model_urls()
    if env:
        entry = env.get(operator_id) or env.get("models", {}).get(operator_id)
        if entry:
            return entry

    manifest = _load_cached_manifest()
    if manifest is None:
        manifest = _download_manifest()
    if manifest:
        entry = manifest.get("models", {}).get(operator_id)
        if entry:
            return entry

    return _BUILTIN_TABLE.get(operator_id)


def resolve_url(operator_id: str, version: Optional[str] = None) -> Optional[str]:
    entry = _lookup_model_entry(operator_id)
    if entry is None:
        return None
    ver = version or entry.get("latest")
    if ver is None:
        return None
    ver_info = entry.get("versions", {}).get(ver)
    if ver_info is None:
        # Try the entry itself (old flat format)
        return entry.get("url")
    return ver_info.get("url")
