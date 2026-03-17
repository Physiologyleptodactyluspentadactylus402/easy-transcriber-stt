from __future__ import annotations
import json
from pathlib import Path

_LOCALES_DIR = Path(__file__).parent.parent / "locales"
_SUPPORTED = ["en", "it"]
_DEFAULT = "en"
_cache: dict[str, dict] = {}


def load_locale(language: str) -> dict:
    if language not in _SUPPORTED:
        language = _DEFAULT
    if language not in _cache:
        path = _LOCALES_DIR / f"{language}.json"
        _cache[language] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[language]
