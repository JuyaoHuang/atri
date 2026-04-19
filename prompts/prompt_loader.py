"""Prompt template loader.

Flat, stateless functions resolving prompt names to file contents. Directory
layout mirrors Open-LLM-VTuber's convention:

    prompts/
      prompt_loader.py    (this module)
      compress/           L3/L4 compression prompts (Phase 3)
      persona/            Character persona prompts (Phase 4)
      utils/              Utility prompts: live2d / speakable / etc. (Phase 4)

All loaders look for ``{name}.txt`` inside the corresponding subdirectory and
tolerate UTF-8 BOM via ``encoding='utf-8-sig'``. Missing files raise
``FileNotFoundError`` with the full attempted path.
"""

from __future__ import annotations

from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_COMPRESS_DIR = _THIS_DIR / "compress"
_PERSONA_DIR = _THIS_DIR / "persona"
_UTIL_DIR = _THIS_DIR / "utils"


def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8-sig")


def load_compress(name: str) -> str:
    """Load a compression prompt (L3/L4) by name.

    Example: ``load_compress('l3_collapse')`` reads ``prompts/compress/l3_collapse.txt``.
    """
    return _read(_COMPRESS_DIR / f"{name}.txt")


def load_persona(name: str) -> str:
    """Load a character persona prompt by name.

    Example: ``load_persona('katou')`` reads ``prompts/persona/katou.txt``.
    """
    return _read(_PERSONA_DIR / f"{name}.txt")


def load_util(name: str) -> str:
    """Load a utility prompt by name.

    Example: ``load_util('live2d_expression')`` reads ``prompts/utils/live2d_expression.txt``.
    """
    return _read(_UTIL_DIR / f"{name}.txt")


__all__ = ["load_compress", "load_persona", "load_util"]
