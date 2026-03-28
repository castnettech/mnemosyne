# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Configuration loader for Mnemosyne.

Reads from `.mnemosyne/config.toml` (relative to project root) if present,
deep-merging with built-in defaults. Provides dot-access via nested namespace
objects and a `save()` method to persist changes back to TOML.
"""

from __future__ import annotations

import os
import sys
import tomllib  # stdlib since Python 3.11
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, dict[str, Any]] = {
    "general": {
        "project_root": ".",
        "ignore_patterns": [
            # Version control / build
            ".git",
            "node_modules",
            "__pycache__",
            ".mnemosyne",
            "*.pyc",
            "*.lock",
            "*.min.js",
            "*.min.css",
            # Secrets and credentials — NEVER index these
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "*.jks",
            "*.keystore",
            "*.secret",
            "*.secrets",
            "id_rsa*",
            "id_ed25519*",
            "id_ecdsa*",
            "id_dsa*",
            ".ssh",
            "credentials.json",
            "credentials.yaml",
            "service-account*.json",
            "*.credential",
            ".netrc",
            ".npmrc",
            ".pypirc",
            ".docker/config.json",
            "*.htpasswd",
            # Token / auth files
            "token.json",
            "*.token",
            ".git-credentials",
            # AWS / cloud
            ".aws",
            "*.tfvars",
            # Common non-source noise
            "package-lock.json",
        ],
        "max_file_size_kb": 512,
        "supported_extensions": [
            ".py",
            ".js",
            ".ts",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".sql",
            ".sh",
            ".html",
            ".css",
            ".jsx",
            ".tsx",
        ],
    },
    "chunking": {
        "max_chunk_tokens": 300,
        "min_chunk_tokens": 20,
        "overlap_lines": 3,
        "code_granularity": "function",
    },
    "embedding": {
        "backend": "tfidf",
        "tfidf_max_features": 10000,
        "tfidf_min_df": 1,
    },
    "compression": {
        "target_ratio": 0.4,
        "preserve_signatures": True,
        "preserve_docstrings": True,
        "collapse_imports": True,
        "collapse_boilerplate": True,
        "max_prune_ratio": 0.7,
        "strict_for_symbols": True,
    },
    "retrieval": {
        "bm25_weight": 0.4,
        "vector_weight": 0.4,
        "usage_weight": 0.2,
        "max_results": 20,
        "token_budget": 8000,
        "max_files": 0,                 # 0 = adaptive (min 3, max 8, ~1/3 of fused files)
        "import_inject_threshold": 2,   # minimum dep references to inject a file
    },
    "cache": {
        "capacity": 500,
        "ghost_capacity": 1000,
    },
    "analytics": {
        "decay_halflife_days": 7,
        "session_timeout_minutes": 30,
    },
}


# ---------------------------------------------------------------------------
# Namespace wrapper for dot-access
# ---------------------------------------------------------------------------


class _Section:
    """Provides attribute-style access to a configuration section dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        # Store raw dict for serialisation; expose values as attributes.
        object.__setattr__(self, "_data", data)
        for key, value in data.items():
            object.__setattr__(self, key, value)

    # Allow mutation so callers can do config.chunking.max_chunk_tokens = 400
    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        self._data[name] = value

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Section({self._data!r})"

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# Main Config class
# ---------------------------------------------------------------------------


class Config:
    """
    Merged configuration object.

    Instantiate with an optional ``root`` path.  The constructor will look for
    ``<root>/.mnemosyne/config.toml`` and deep-merge any values found there
    on top of DEFAULTS.

    Attribute access:
        config.chunking.max_chunk_tokens   -> int
        config.general.ignore_patterns     -> list[str]

    Methods:
        config.get(section, key, default=None)
        config.save()
    """

    _TOML_REL = os.path.join(".mnemosyne", "config.toml")

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        if root is None:
            root = os.getcwd()
        self._root = Path(root).resolve()
        self._config_path = self._root / ".mnemosyne" / "config.toml"

        # Start from a deep copy of the defaults.
        merged = self._deep_copy(DEFAULTS)

        # Overlay values from the TOML file if it exists.
        if self._config_path.is_file():
            with open(self._config_path, "rb") as fh:
                user_cfg = tomllib.load(fh)
            merged = self._deep_merge(merged, user_cfg)

        # Build section namespaces.
        self._sections: dict[str, _Section] = {}
        for section_name, section_data in merged.items():
            if isinstance(section_data, dict):
                sec = _Section(section_data)
            else:
                # Top-level scalar — wrap in a single-key section for uniformity.
                sec = _Section({section_name: section_data})
            self._sections[section_name] = sec
            object.__setattr__(self, section_name, sec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Return ``config.<section>.<key>`` or *default* if not found."""
        sec = self._sections.get(section)
        if sec is None:
            return default
        return getattr(sec, key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set ``config.<section>.<key> = value``, creating the section if needed."""
        if section not in self._sections:
            sec = _Section({key: value})
            self._sections[section] = sec
            object.__setattr__(self, section, sec)
        else:
            setattr(self._sections[section], key, value)

    def save(self) -> None:
        """Persist the current configuration to ``.mnemosyne/config.toml``."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        # Build TOML content manually (tomllib is read-only; we write by hand).
        lines: list[str] = []
        for section_name, sec in self._sections.items():
            lines.append(f"[{section_name}]")
            for key, value in sec.as_dict().items():
                lines.append(f"{key} = {self._toml_value(value)}")
            lines.append("")

        with open(self._config_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def as_dict(self) -> dict[str, dict[str, Any]]:
        """Return the full configuration as a plain nested dict."""
        return {name: sec.as_dict() for name, sec in self._sections.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_copy(data: dict) -> dict:
        """Recursively copy a dict (values may be lists or primitives)."""
        result: dict = {}
        for k, v in data.items():
            if isinstance(v, dict):
                result[k] = Config._deep_copy(v)
            elif isinstance(v, list):
                result[k] = list(v)
            else:
                result[k] = v
        return result

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """
        Deep-merge *override* into *base*, returning a new dict.

        Nested dicts are merged recursively.  Lists are **unioned** (base
        items preserved, override items appended if not already present)
        so that hardened default patterns (security ignore rules, sensitive
        file exclusions) can never be silently dropped by a project TOML.
        All other types replace.
        """
        result = Config._deep_copy(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = Config._deep_merge(result[k], v)
            elif k in result and isinstance(result[k], list) and isinstance(v, list):
                # Union: keep all defaults, add new items from override
                merged = list(result[k])
                for item in v:
                    if item not in merged:
                        merged.append(item)
                result[k] = merged
            else:
                result[k] = v
        return result

    @staticmethod
    def _toml_value(value: Any) -> str:
        """Render a Python value as a TOML-compatible string."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, list):
            items = ", ".join(Config._toml_value(v) for v in value)
            return f"[{items}]"
        if isinstance(value, float):
            return repr(value)
        return str(value)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(root={self._root!r}, sections={list(self._sections)})"


# ---------------------------------------------------------------------------
# Module-level convenience: load config from cwd
# ---------------------------------------------------------------------------


def load_config(root: str | os.PathLike | None = None) -> Config:
    """Load and return a :class:`Config` for *root* (defaults to cwd)."""
    return Config(root=root)
