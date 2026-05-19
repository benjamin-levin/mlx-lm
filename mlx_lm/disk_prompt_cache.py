# Copyright © 2023-2026 Apple Inc.

"""Disk-backed persistence layer for the in-memory ``LRUPromptCache``.

Background
==========

``mlx_lm.server`` already ships an in-memory ``LRUPromptCache`` that does
prefix sharing via a token-id trie. What it does NOT do is survive a
process restart: the trie lives entirely in RAM, so every cold start
re-prefills every long system prompt from scratch.

This module adds a thin disk-persistence layer that turns the in-memory
``LRUPromptCache`` into a write-through cache. Each insert is mirrored to
disk (subject to a ``min_tokens`` threshold and an LRU-evicted byte
budget); each in-memory miss probes disk for the longest stored prefix
and re-materializes it into the trie so subsequent lookups hit the fast
path.

The feature is **opt-in** and has zero overhead when disabled. Enable
either by passing ``disk_cache_dir=`` to ``LRUPromptCache`` (library
callers) or by passing ``--disk-prompt-cache-dir`` to ``mlx_lm.server``.

Design
======

* **What we persist**: full ``[KVCache, ...]`` lists for "prefix-only"
  prompts. The actual mlx state is serialized via this package's own
  :func:`mlx_lm.models.cache.save_prompt_cache` /
  :func:`load_prompt_cache` (safetensors).

* **Key**: a SHA-256 of (``model_id`` || little-endian-packed tokens),
  truncated to the first 16 hex chars for the filename. The full hex is
  stored in the sidecar so we can verify there is no collision on load.

* **Layout** (under the user-supplied ``cache_dir``)::

      <key16>.safetensors   - KV state
      <key16>.meta.json     - sidecar: full sha256, model_id, token count,
                              token list (for trie seed), created_at,
                              last_used_at, prefix_token_count, nbytes.

* **LRU eviction**: at insert time, if total disk bytes > ``size_budget``
  (default 4 GB), oldest ``last_used_at`` entries are deleted. Eviction is
  best-effort and single-threaded.

* **Atomic write**: write to ``.tmp`` then ``os.replace`` into final name.
  Two writers racing on the same key both produce the same content (by
  hash equality), so last-write-wins is safe.

* **Model identity**: ``model_id`` is the model_path / adapter / draft
  tuple stringified. On load we verify ``meta["model_id"] ==
  current_model_id``; mismatch = cache miss (refuse to load somebody
  else's KV state into our model).

Constraints
===========

* Only single-sequence ``KVCache`` / ``RotatingKVCache`` /
  ``ChunkedKVCache`` lists are persisted. Trimmed (SnapKV) or quantized
  caches are skipped at the persistence layer because they're
  prefix-specific and not portable.

* Prefix-only matching: we look for the longest stored token list that
  is a strict prefix of the lookup tokens. The in-memory trie still
  handles partial matches via ``can_trim_prompt_cache``; the disk index
  is queried only on miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_LOG = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Sidecar metadata
# ----------------------------------------------------------------------


@dataclass
class CacheMeta:
    """Sidecar metadata for one persisted KV cache entry."""

    sha256: str
    model_id: str
    tokens: List[int]
    prefix_token_count: int
    nbytes: int
    created_at: float
    last_used_at: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheMeta":
        return cls(
            sha256=d["sha256"],
            model_id=d["model_id"],
            tokens=list(d["tokens"]),
            prefix_token_count=int(d["prefix_token_count"]),
            nbytes=int(d["nbytes"]),
            created_at=float(d["created_at"]),
            last_used_at=float(d["last_used_at"]),
        )


# ----------------------------------------------------------------------
# Hashing helpers
# ----------------------------------------------------------------------


def hash_tokens(tokens: List[int], model_id: str) -> str:
    """Return the SHA-256 hex digest of ``(model_id || tokens)``.

    Tokens are packed little-endian uint32 for speed; this is much faster
    than JSON-encoding multi-thousand-token lists.
    """
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\n--\n")
    n = len(tokens)
    arr = bytearray(n * 4)
    for i, t in enumerate(tokens):
        v = int(t) & 0xFFFFFFFF
        arr[i * 4] = v & 0xFF
        arr[i * 4 + 1] = (v >> 8) & 0xFF
        arr[i * 4 + 2] = (v >> 16) & 0xFF
        arr[i * 4 + 3] = (v >> 24) & 0xFF
    h.update(bytes(arr))
    return h.hexdigest()


def model_id_for(model_key: Any) -> str:
    """Coerce a server-style ``(model_path, adapter, draft)`` tuple (or
    any other model identifier) into a stable string for hashing.
    """
    if isinstance(model_key, tuple):
        return "/".join(str(x or "") for x in model_key)
    return str(model_key)


def _short(sha256_hex: str) -> str:
    return sha256_hex[:16]


def _paths_for(root: Path, key16: str) -> Tuple[Path, Path]:
    return root / f"{key16}.safetensors", root / f"{key16}.meta.json"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = tempfile.NamedTemporaryFile(
        "w",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
        encoding="utf-8",
    )
    try:
        json.dump(payload, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.replace(tmp.name, path)


def _read_meta(meta_path: Path) -> Optional[CacheMeta]:
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return CacheMeta.from_dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


# ----------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------


def _is_persistable(cache: List[Any]) -> bool:
    """Refuse to persist caches that have been trimmed or quantized - the
    on-disk format treats offset == token count, which is only true for
    full untrimmed prefix caches.
    """
    for c in cache:
        cls_name = type(c).__name__
        if cls_name in ("SnapKVCache", "QuantizedKVCache"):
            return False
    return True


def save_prefix_cache(
    root: Path,
    key: str,
    cache: List[Any],
    tokens: List[int],
    model_id: str,
    *,
    prefix_token_count: Optional[int] = None,
    nbytes_hint: Optional[int] = None,
    size_budget_bytes: Optional[int] = None,
) -> Optional[Path]:
    """Persist a prefix-only prompt cache to disk.

    Returns the safetensors path on success, ``None`` if the cache is
    not persistable (e.g. trimmed by SnapKV, quantized).
    """
    if not cache or not _is_persistable(cache):
        return None

    # Defer the import so this module is cheap to import in test envs.
    from .models.cache import save_prompt_cache

    root.mkdir(parents=True, exist_ok=True)
    key16 = _short(key)
    st_path, meta_path = _paths_for(root, key16)

    st_tmp = root / f".{key16}.{os.getpid()}.tmp.safetensors"
    try:
        save_prompt_cache(
            str(st_tmp),
            cache,
            metadata={
                "sha256": key,
                "model_id": model_id,
                "prefix_token_count": str(prefix_token_count or len(tokens)),
            },
        )
    except Exception as exc:  # noqa: BLE001 - disk write must never crash inference
        _LOG.warning("[disk_prompt_cache] save failed: %r", exc)
        try:
            st_tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    nbytes = nbytes_hint
    if nbytes is None:
        try:
            nbytes = sum(getattr(c, "nbytes", 0) for c in cache)
        except Exception:  # noqa: BLE001
            nbytes = st_tmp.stat().st_size

    now = time.time()
    meta = CacheMeta(
        sha256=key,
        model_id=model_id,
        tokens=list(tokens),
        prefix_token_count=int(prefix_token_count or len(tokens)),
        nbytes=int(nbytes or st_tmp.stat().st_size),
        created_at=now,
        last_used_at=now,
    )
    try:
        os.replace(st_tmp, st_path)
        _atomic_write_json(meta_path, meta.to_dict())
    except OSError as exc:
        _LOG.warning("[disk_prompt_cache] rename failed: %r", exc)
        try:
            st_tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    if size_budget_bytes is not None:
        _enforce_size_budget(root, size_budget_bytes)
    return st_path


def load_cached_prefix(
    root: Path,
    key: str,
    model_id: str,
    *,
    touch: bool = True,
) -> Optional[Tuple[List[Any], List[int]]]:
    """Load a previously saved prefix cache by full sha256 hex.

    Returns ``(cache_list, tokens)`` on hit, ``None`` on miss or any
    integrity failure.
    """
    key16 = _short(key)
    st_path, meta_path = _paths_for(root, key16)
    if not st_path.exists() or not meta_path.exists():
        return None

    meta = _read_meta(meta_path)
    if meta is None:
        return None
    if meta.sha256 != key:
        _LOG.warning(
            "[disk_prompt_cache] 16-char collision on %s; " "expected sha256=%s got=%s",
            key16,
            key,
            meta.sha256,
        )
        return None
    if meta.model_id != model_id:
        _LOG.info(
            "[disk_prompt_cache] model mismatch on %s "
            "(saved=%s, requested=%s) - miss",
            key16,
            meta.model_id,
            model_id,
        )
        return None

    from .models.cache import load_prompt_cache

    try:
        cache_list = load_prompt_cache(str(st_path))
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("[disk_prompt_cache] load failed for %s: %r", key16, exc)
        return None

    if touch:
        meta.last_used_at = time.time()
        try:
            _atomic_write_json(meta_path, meta.to_dict())
        except OSError:
            pass  # touch failure isn't fatal

    return cache_list, meta.tokens


# ----------------------------------------------------------------------
# LRU eviction
# ----------------------------------------------------------------------


def _enforce_size_budget(root: Path, budget_bytes: int) -> int:
    """Evict oldest-by-last-used entries until total bytes <= budget.

    Returns the number of entries evicted.
    """
    entries: List[Tuple[float, int, Path, Path]] = []
    total = 0
    for meta_path in root.glob("*.meta.json"):
        meta = _read_meta(meta_path)
        if meta is None:
            continue
        st_path = root / f"{_short(meta.sha256)}.safetensors"
        if not st_path.exists():
            continue
        try:
            sz = st_path.stat().st_size
        except FileNotFoundError:
            continue
        entries.append((meta.last_used_at, sz, st_path, meta_path))
        total += sz

    if total <= budget_bytes:
        return 0

    entries.sort(key=lambda e: e[0])
    evicted = 0
    pre_total = total
    for _, sz, st_path, meta_path in entries:
        if total <= budget_bytes:
            break
        try:
            st_path.unlink()
            meta_path.unlink()
            total -= sz
            evicted += 1
        except FileNotFoundError:
            pass

    if evicted:
        _LOG.info(
            "[disk_prompt_cache] evicted %d entries, " "total=%.2f GB -> %.2f GB",
            evicted,
            pre_total / 1e9,
            total / 1e9,
        )
    return evicted


# ----------------------------------------------------------------------
# Disk-backed index
# ----------------------------------------------------------------------


class DiskPromptCacheIndex:
    """Token-keyed disk index for one ``model_id``.

    Each insert hashes the full token list and persists. Lookup is by
    exact token list (must match what was saved). For prefix matching
    across different request token lists, the in-memory trie inside
    ``LRUPromptCache`` is consulted first; this index only handles the
    cold-start case where the trie has no covering entry.
    """

    def __init__(
        self,
        root: Path,
        model_id: str,
        *,
        size_budget_bytes: int = 4 * (1024**3),
        min_prefix_tokens: int = 256,
    ):
        self.root = Path(root)
        self.model_id = model_id
        self.size_budget_bytes = int(size_budget_bytes)
        self.min_prefix_tokens = int(min_prefix_tokens)
        self.root.mkdir(parents=True, exist_ok=True)

    def has(self, tokens: List[int]) -> bool:
        key = hash_tokens(tokens, self.model_id)
        st_path, meta_path = _paths_for(self.root, _short(key))
        return st_path.exists() and meta_path.exists()

    def get(self, tokens: List[int]) -> Optional[List[Any]]:
        key = hash_tokens(tokens, self.model_id)
        result = load_cached_prefix(self.root, key, self.model_id)
        if result is None:
            return None
        return result[0]

    def put(self, tokens: List[int], cache: List[Any]) -> bool:
        if len(tokens) < self.min_prefix_tokens:
            return False
        key = hash_tokens(tokens, self.model_id)
        path = save_prefix_cache(
            self.root,
            key,
            cache,
            tokens,
            self.model_id,
            prefix_token_count=len(tokens),
            size_budget_bytes=self.size_budget_bytes,
        )
        return path is not None

    def iter_meta(self) -> List[CacheMeta]:
        """Enumerate all entries for the current ``model_id``."""
        out: List[CacheMeta] = []
        for meta_path in self.root.glob("*.meta.json"):
            meta = _read_meta(meta_path)
            if meta is not None and meta.model_id == self.model_id:
                out.append(meta)
        return out


__all__ = [
    "CacheMeta",
    "DiskPromptCacheIndex",
    "hash_tokens",
    "load_cached_prefix",
    "model_id_for",
    "save_prefix_cache",
]
