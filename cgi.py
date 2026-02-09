"""Minimal cgi compatibility shim for Python 3.13+.

Provides parse_header used by httpx.
"""

from __future__ import annotations

from typing import Dict, Tuple


def parse_header(line: str) -> Tuple[str, Dict[str, str]]:
    """Parse a Content-Type like header.

    Returns (value, params) where params is a dict of key/value pairs.
    """
    if not line:
        return "", {}

    parts = [p.strip() for p in line.split(";")]
    value = parts[0]
    params: Dict[str, str] = {}

    for param in parts[1:]:
        if not param:
            continue
        if "=" not in param:
            params[param] = ""
            continue
        key, _, val = param.partition("=")
        key = key.strip().lower()
        val = val.strip()
        if len(val) >= 2 and val[0] in "'\"" and val[-1] == val[0]:
            val = val[1:-1]
        params[key] = val

    return value, params


__all__ = ["parse_header"]
