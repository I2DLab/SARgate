"""
=========
sketcher.py
=========

Sketcher bridge for SARgate.

This module exposes the SKETCHER tab content inside the Dear PyGui interface and
launches the standalone tkinter-based molecule editor in a separate process.
The separate-process approach preserves the full current feature set of the
existing editor without introducing Dear PyGui/tkinter event-loop conflicts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.utils.app_logger import log_event, log_settings
from app.utils.resource_paths import resource_path, user_data_path


def _sketcher_script_path() -> Path:
    return resource_path("app/analysis/tools/molecule_sketcher.py")


def _sketcher_python_executable() -> str:
    return sys.executable


def _sketcher_process_running(state: dict[str, Any]) -> bool:
    proc = state.get("sketcher_process")
    return bool(proc is not None and proc.poll() is None)


def _sketcher_status_text(state: dict[str, Any]) -> str:
    proc = state.get("sketcher_process")
    if proc is None:
        return "Sketcher process: not started"
    if proc.poll() is None:
        return f"Sketcher process: running (PID {proc.pid})"
    return f"Sketcher process: stopped (exit code {proc.returncode})"


def _sketcher_theme_file(state: dict[str, Any]) -> str:
    existing = state.get("sketcher_theme_sync_file")
    if isinstance(existing, str) and existing.strip():
        return existing

    runtime_dir = state.get("user_data_dir")
    if isinstance(runtime_dir, str) and runtime_dir.strip():
        sync_dir = Path(runtime_dir) / "config"
    else:
        sync_dir = user_data_path("config")
    sync_dir.mkdir(parents=True, exist_ok=True)
    sync_path = sync_dir / "sketcher_theme.stf"
    state["sketcher_theme_sync_file"] = str(sync_path)
    return str(sync_path)


def _rgba_to_hex(rgba: Any, fallback: str) -> str:
    if not isinstance(rgba, (list, tuple)) or len(rgba) < 3:
        return fallback
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0]) & 255,
        int(rgba[1]) & 255,
        int(rgba[2]) & 255,
    )


def _sketcher_theme_payload(state: dict[str, Any], request_focus: bool = False) -> dict[str, Any]:
    theme_name = state.get("theme_name")
    themes = state.get("themes", {})
    theme = themes.get(theme_name, {}) if isinstance(themes, dict) else {}
    previous_focus_nonce = int(state.get("sketcher_focus_nonce", 0) or 0)
    focus_nonce = previous_focus_nonce + 1 if request_focus else previous_focus_nonce
    state["sketcher_focus_nonce"] = focus_nonce

    return {
        "theme_name": str(theme_name or ""),
        "main_bg": _rgba_to_hex(theme.get("Main Background"), "#f5f7fb"),
        "panel_bg": _rgba_to_hex(theme.get("Secondary Background"), "#ffffff"),
        "text": _rgba_to_hex(theme.get("Text Color"), "#1f2937"),
        "border": _rgba_to_hex(theme.get("Border Color"), "#d7dee8"),
        "border_shadow": _rgba_to_hex(theme.get("Border Shadow"), "#cfd6df"),
        "frame_bg": _rgba_to_hex(theme.get("Button Color"), "#e3e8f0"),
        "title_bar_bg": _rgba_to_hex(theme.get("Title Bar Background"), "#2f4f6f"),
        "menu_bar_bg": _rgba_to_hex(theme.get("Menu Bar Background"), "#edf2f7"),
        "tabs_color": _rgba_to_hex(theme.get("Tabs Color"), "#d8dee9"),
        "tabs_hovered": _rgba_to_hex(theme.get("Tabs Hovered"), "#cbd5e1"),
        "tabs_active": _rgba_to_hex(theme.get("Tabs Active"), "#f59e0b"),
        "button_color": _rgba_to_hex(theme.get("Button Color"), "#e5e7eb"),
        "button_hovered": _rgba_to_hex(theme.get("Button Hovered"), "#d1d5db"),
        "button_active": _rgba_to_hex(theme.get("Button Active"), "#cbd5e1"),
        "checkmark_color": _rgba_to_hex(theme.get("Checkmark Color"), "#f59e0b"),
        "slider_grab": _rgba_to_hex(theme.get("Slider Grab"), "#f59e0b"),
        "frame_border_size": int(theme.get("Frame Border Size", 1) or 1),
        "window_rounding": int(theme.get("Window rounding", 8) or 8),
        "frame_rounding": int(theme.get("Frame rounding", 6) or 6),
        "tab_rounding": int(theme.get("Tab rounding", 5) or 5),
        "focus_nonce": focus_nonce,
    }


def _write_sketcher_theme_payload(state: dict[str, Any], request_focus: bool = False) -> str:
    sync_path = _sketcher_theme_file(state)
    payload = _sketcher_theme_payload(state, request_focus=request_focus)
    tmp_path = f"{sync_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, sync_path)
    return sync_path


def sync_sketcher_theme(state: dict[str, Any], request_focus: bool = False) -> None:
    try:
        _write_sketcher_theme_payload(state, request_focus=request_focus)
    except Exception as exc:
        log_event("SKETCHER", f"Unable to sync sketcher theme: {exc}", indent=1, level="ERROR")


def _sketcher_env(state: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    theme_name = state.get("theme_name")
    themes = state.get("themes", {})
    theme = themes.get(theme_name, {}) if isinstance(themes, dict) else {}
    env["SARGATE_SKETCHER_BG"] = _rgba_to_hex(theme.get("Main Background"), "#f5f7fb")
    env["SARGATE_SKETCHER_PANEL"] = _rgba_to_hex(theme.get("Secondary Background"), "#ffffff")
    env["SARGATE_SKETCHER_TEXT"] = _rgba_to_hex(theme.get("Text Color"), "#1f2937")
    env["SARGATE_SKETCHER_BORDER"] = _rgba_to_hex(theme.get("Border Color"), "#d7dee8")
    env["SARGATE_SKETCHER_GRID"] = _rgba_to_hex(theme.get("Frame Background"), "#e3e8f0")
    env["SARGATE_SKETCHER_SELECTED"] = _rgba_to_hex(theme.get("Tabs Active"), "#f59e0b")
    env["SARGATE_SKETCHER_BOND"] = _rgba_to_hex(theme.get("Text Color"), "#303846")
    env["SARGATE_SKETCHER_THEME_FILE"] = _sketcher_theme_file(state)
    return env


def _launch_sketcher_process(state: dict[str, Any]) -> None:
    if _sketcher_process_running(state):
        sync_sketcher_theme(state, request_focus=True)
        log_event("SKETCHER", "Sketcher already running", indent=1)
        return

    sync_sketcher_theme(state, request_focus=True)
    if getattr(sys, "frozen", False):
        command = [_sketcher_python_executable(), "--sketcher-helper"]
    else:
        script_path = _sketcher_script_path()
        if not script_path.exists():
            log_event("SKETCHER", f"Sketcher script not found: {script_path}", indent=1, level="ERROR")
            return
        command = [_sketcher_python_executable(), str(script_path)]
    try:
        runtime_dir = state.get("user_data_dir")
        cwd = Path(runtime_dir) if isinstance(runtime_dir, str) and runtime_dir.strip() else user_data_path()
        cwd.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            start_new_session=True,
            env=_sketcher_env(state),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        log_event("SKETCHER", f"Unable to start sketcher: {exc}", indent=1, level="ERROR")
        return

    state["sketcher_process"] = proc


def open_sketcher_window(state: dict[str, Any], log_on_open: bool = True) -> None:
    """
    Open the standalone sketcher window.

    Args:
        state (dict[str, Any]): Shared application state.
        log_on_open (bool, optional): Emit an open event.

    Returns:
        None
    """
    if log_on_open:
        log_event("SKETCHER", "Opening standalone sketcher window", indent=1)
    _launch_sketcher_process(state)
