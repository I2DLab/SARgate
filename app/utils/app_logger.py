"""
================
app_logger.py
================

Lightweight structured logging helpers for SARgate.

All messages are emitted through stdout so they automatically appear both in
the terminal and in the GUI Event log window.
"""

import traceback
from typing import Any

_SECTION_WIDTH = 12

_TAB_SECTION_MAP = {
    "INPUT": "INPUT",
    "ANALYSIS": "ANALYSIS",
    "PLOTS": "ANALYSIS",
    "OVERVIEW": "OVERVIEW",
    "R-ANALYSIS": "R-ANALYSIS",
    "SIMILARITY": "SIMILARITY",
    "STEREO": "STEREO",
    "MMPA": "MMPA",
    "CHEMSPACE": "CHEMSPACE",
    "PCA": "CHEMSPACE",
    "UMAP": "CHEMSPACE",
    "TSNE": "CHEMSPACE",
    "PREDICTION": "PREDICTION",
    "SKETCHER": "SKETCHER",
    "NOTES": "SAR NOTES",
    "SAR NOTES": "SAR NOTES",
    "UTILITIES": "UTILITIES",
    "EVENT LOG": "EVENT LOG",
    "SLITH": "SLITH-MINIGAME",
    "SLITH-MINIGAME": "SLITH-MINIGAME",
    "SYSTEM": "SYSTEM",
}


def normalize_log_section(section: str) -> str:
    """
    Map technical logger sections to the user-facing tab name.
    """
    key = str(section or "").strip().upper()
    return _TAB_SECTION_MAP.get(key, str(section or "").strip() or "SYSTEM")


def log_event(section: str, message: str, indent: int = 0, level: str = "INFO") -> None:
    """
    Emit a consistently formatted log line.

    Args:
        section (str): Logical subsystem name (e.g. "PCA", "MMPA").
        message (str): Human-readable message.
        indent (int, optional): Visual indentation level.
        level (str, optional): INFO, SUCCESS, WARNING, or ERROR.

    Returns:
        None: Writes one line to stdout.
    """
    prefix = "    " * max(0, int(indent))
    level = str(level).upper()
    icon_map = {
        "INFO": "•",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
    }
    icon = icon_map.get(level, "•")
    section_label = f"[{normalize_log_section(section)}]".ljust(_SECTION_WIDTH)
    print(f"{section_label} {prefix}{icon} {message}")


def log_exception(section: str, message: str, exc: Any, indent: int = 0) -> None:
    """
    Emit a formatted exception summary.

    Args:
        section (str): Logical subsystem name.
        message (str): Human-readable context message.
        exc (Any): Exception instance or description.
        indent (int, optional): Visual indentation level.

    Returns:
        None: Writes one line to stdout.
    """
    log_event(section, f"{message}: {exc}", indent=indent, level="ERROR")


def log_traceback(section: str, indent: int = 0) -> None:
    """
    Emit the current traceback line by line into the structured log.
    """
    tb_text = traceback.format_exc().strip()
    if not tb_text:
        return
    for line in tb_text.splitlines():
        log_event(section, line, indent=indent, level="ERROR")


def _format_setting_value(value: Any) -> str:
    """
    Convert a setting value into a compact log-friendly representation.
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    text = str(value).strip()
    return text or "-"


def log_settings(section: str, indent: int = 0, **settings: Any) -> None:
    """
    Emit one compact line containing the current rendering/analysis settings.
    """
    payload = ", ".join(
        f"{key}={_format_setting_value(value)}"
        for key, value in settings.items()
    )
    if payload:
        log_event(section, payload, indent=indent)
