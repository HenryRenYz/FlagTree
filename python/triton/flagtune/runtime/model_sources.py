# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Resolve remote locations for versioned FlagTune platform packages.

The model manager normally calls this module only after local model roots have
no usable archive, or when an explicit refresh needs fresh remote metadata. It
maps a platform key to remote package metadata using three sources in priority
order: ``FLAGTUNE_MODEL_URLS``, a 24-hour cached/downloaded manifest, then
``_BUILTIN_TABLE``. It validates version strings and URL shape, but package
download, archive validation, and versioned cache writes belong to
:mod:`model_loader`.

Environment variables:
  * ``FLAGTUNE_MODEL_URLS``: either a path to a JSON document or an inline JSON
    document using manifest schema 1. It overrides all built-in and remote
    manifest entries below ``packages``.
  * ``FLAGTUNE_MODEL_CACHE``: indirectly selects the manifest cache location
    through :func:`triton.flagtune.runtime.model_loader._cache_root`.
  * ``FLAGTUNE_MANIFEST_URL``: central anonymous HTTPS manifest endpoint. It
    defaults to ``https://models.example.com/flagtune/manifest.json``.
  * ``FLAGTUNE_DISABLE_REMOTE``: any non-empty value prevents a central
    manifest network request.

A manifest has the form ``{"schema_version": 1, "packages": {"nvidia-h20":
{"versions": {"1.2.3": {"url": "https://.../nvidia-h20_v1.2.3.tar.gz",
"sha256": "<64 lowercase hexadecimal characters>"}}}}}``. URLs must use
HTTPS through every redirect and advertise a lowercase SHA-256 digest. Their
basenames must be the canonical platform package name. ``latest`` is ignored;
the highest strict SemVer key wins unless an exact version is requested. The
cached manifest is reused for 24 hours only when remote resolution is necessary;
``force_refresh=True`` bypasses that freshness window.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler

from triton.flagtune.contract.archive import platform_package_name, parse_model_version, validate_model_version
from triton.flagtune.contract.identity import validate_platform_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in URL table (ultimate fallback).
#
# This table packages emergency/default locations for platforms that cannot be
# centrally published yet. Add a platform key and one or more strict SemVer
# entries in the schema documented above. ``resolve_package_info`` chooses the
# highest valid key in ``versions`` when no version is requested; ``latest`` is
# descriptive only. Keep entries small and maintained: the current URLs are
# placeholders, and an unavailable URL is indistinguishable from no package.
# TODO: Populate this table with maintained, reachable model archive URLs.
# ---------------------------------------------------------------------------

_BUILTIN_TABLE: Dict[str, Dict[str, Any]] = {
    # "nvidia-h20": {
    #     "latest": "1.0.0",
    #     "versions": {
    #         "1.0.0": {
    #             "url": "https://models.example.invalid/flagtune/nvidia-h20_v1.0.0.tar.gz",
    #             "sha256": "...",
    #         },
    #     },
    # },
}

# ---------------------------------------------------------------------------
# Remote manifest
# ---------------------------------------------------------------------------

_DEFAULT_MANIFEST_URL = "https://models.example.com/flagtune/manifest.json"
_MANIFEST_MAX_AGE_S = 24 * 3600  # 24 hours
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RemotePackage:
    version: str
    url: str
    sha256: str


class ManifestContractError(RuntimeError):
    """Reject an explicitly supplied manifest that violates schema 1."""


class _HttpsOnlyRedirectHandler(HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise HTTPError(newurl, code, "FlagTune redirects must remain HTTPS", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_https(request, *, timeout: int):
    from urllib.request import build_opener

    return build_opener(_HttpsOnlyRedirectHandler()).open(request, timeout=timeout)


def _manifest_url() -> str:
    return os.environ.get("FLAGTUNE_MANIFEST_URL", "").strip() or _DEFAULT_MANIFEST_URL


def _env_model_urls() -> Optional[Dict[str, Any]]:
    """Load the explicit override JSON from ``FLAGTUNE_MODEL_URLS``, if present."""
    if "FLAGTUNE_MODEL_URLS" not in os.environ:
        return None
    raw = os.environ["FLAGTUNE_MODEL_URLS"].strip()
    if not raw:
        raise ManifestContractError("FLAGTUNE_MODEL_URLS is present but empty")
    try:
        if raw.startswith(("{", "[")):
            data = json.loads(raw)
        else:
            path = Path(raw)
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestContractError(f"cannot read valid JSON from FLAGTUNE_MODEL_URLS: {exc}") from exc
    if not _manifest_is_valid(data):
        raise ManifestContractError("FLAGTUNE_MODEL_URLS does not satisfy FlagTune manifest schema 1")
    return data


def _cached_manifest_path() -> Path:
    """Return the shared manifest path inside the model-manager cache root."""
    from triton.flagtune.runtime.model_loader import _cache_root
    return _cache_root() / "manifest.json"


def _manifest_is_valid(data: Any) -> bool:
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
        return False
    if set(data) - {"schema_version", "packages", "_fetched_at"}:
        return False
    if data["schema_version"] != 1:
        return False
    if "_fetched_at" in data and not isinstance(data["_fetched_at"], (int, float)):
        return False
    packages = data.get("packages")
    if not isinstance(packages, dict):
        return False
    for platform_key, entry in packages.items():
        try:
            validated_platform_key = validate_platform_key(platform_key, "manifest packages key")
        except ValueError:
            return False
        if validated_platform_key != platform_key:
            return False
        if not isinstance(entry, dict):
            return False
        if set(entry) - {"versions", "latest"}:
            return False
        if "latest" in entry and not isinstance(entry["latest"], str):
            return False
        versions = entry.get("versions")
        if not isinstance(versions, dict):
            return False
        for version, metadata in versions.items():
            try:
                validate_model_version(version)
            except ValueError:
                return False
            if not isinstance(metadata, dict):
                return False
            if set(metadata) != {"url", "sha256"}:
                return False
            url = metadata.get("url")
            digest = metadata.get("sha256")
            if not isinstance(url, str) or not isinstance(digest, str):
                return False
            parsed = urlparse(url.strip())
            if parsed.scheme.lower() != "https" or not parsed.netloc:
                return False
            if PurePosixPath(parsed.path).name != platform_package_name(platform_key, version):
                return False
            if _SHA256_RE.fullmatch(digest) is None:
                return False
    return True


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
    if not _manifest_is_valid(data):
        return None
    fetched_at = data.get("_fetched_at", 0)
    if not isinstance(fetched_at, (int, float)):
        return None
    age = time.time() - fetched_at
    if age > _MANIFEST_MAX_AGE_S:
        return None
    return data


def _download_manifest() -> Optional[Dict[str, Any]]:
    """Fetch and cache the well-known manifest, returning ``None`` on failure."""
    if os.environ.get("FLAGTUNE_DISABLE_REMOTE"):
        return None
    url = _manifest_url()
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        logger.warning("FlagTune manifest URL must use HTTPS: %s", url)
        return None
    try:
        from urllib.request import Request
        from triton.flagtune.runtime.model_loader import _atomic_write_bytes

        request = Request(url, headers={"User-Agent": "FlagTune/0.2"})
        with _open_https(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not _manifest_is_valid(data):
            raise ValueError("manifest has invalid structure")
        data["_fetched_at"] = time.time()
        payload = json.dumps(data, sort_keys=True).encode("utf-8")
        _atomic_write_bytes(_cached_manifest_path(), payload)
        return data
    except Exception:
        logger.warning("Failed to download remote FlagTune manifest", exc_info=True)
        return None


def _entry_from_manifest(source: Any, platform_key: str) -> Optional[Dict[str, Any]]:
    if not _manifest_is_valid(source):
        return None
    entry = source["packages"].get(platform_key)
    return entry if isinstance(entry, dict) and entry else None


def _lookup_package_entry(platform_key: str, *, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """Find one raw entry using override, cached/remote manifest, then builtin data."""
    if "FLAGTUNE_MODEL_URLS" in os.environ:
        override = _env_model_urls()
        return _entry_from_manifest(override, platform_key)

    cached_manifest = _load_cached_manifest()
    if force_refresh:
        manifest = _download_manifest() or cached_manifest
    else:
        manifest = cached_manifest or _download_manifest()
    entry = _entry_from_manifest(manifest, platform_key)
    if entry is not None:
        return entry

    entry = _BUILTIN_TABLE.get(platform_key)
    return entry if isinstance(entry, dict) and entry else None


def resolve_package_info(
    platform_key: str,
    *,
    version: Optional[str] = None,
    force_refresh: bool = False,
) -> Optional[RemotePackage]:
    """Resolve validated remote metadata for one platform package."""
    platform_key = validate_platform_key(platform_key)
    entry = _lookup_package_entry(platform_key, force_refresh=force_refresh)
    if not isinstance(entry, dict):
        return None
    versions = entry.get("versions")
    if not isinstance(versions, dict):
        return None
    if version is not None:
        try:
            selected = validate_model_version(version)
        except ValueError:
            return None
    else:
        candidates = []
        for candidate in versions:
            try:
                parsed = parse_model_version(candidate)
            except ValueError:
                continue
            candidates.append((parsed.selection_key, candidate))
        if not candidates:
            return None
        selected = max(candidates, key=lambda item: item[0])[1]
    metadata = versions.get(selected)
    if not isinstance(metadata, dict):
        return None
    url = metadata.get("url")
    digest = metadata.get("sha256")
    if not isinstance(url, str) or not isinstance(digest, str):
        return None
    url = url.strip()
    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() != "https" or not parsed_url.netloc:
        return None
    if _SHA256_RE.fullmatch(digest) is None:
        return None
    if PurePosixPath(parsed_url.path).name != platform_package_name(platform_key, selected):
        return None
    return RemotePackage(selected, url, digest)
