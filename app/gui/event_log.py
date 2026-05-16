"""
================
event_log.py
================

GUI event-log window and stdout/stderr capture utilities.

Mirrors terminal output into a Dear PyGui window while preserving console
output in the original terminal streams.
"""

import io
import os
import re
import sys
import threading
from datetime import datetime
from typing import Any

import dearpygui.dearpygui as dpg

from app.gui.themes_manager import apply_input_text_theme
from app.utils.app_logger import normalize_log_section
from app.utils.resource_paths import user_data_path


EVENT_LOG_WINDOW_TAG = "event_log_window"
EVENT_LOG_CHILD_TAG = "event_log_child"
EVENT_LOG_TABLE_TAG = "event_log_table"
EVENT_LOG_TAB_BUTTON_TAG = "event_log_nav_button"
EVENT_LOG_INPUT_THEME_TAG = "event_log_input_theme"
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _event_log_is_headline(event_text: str) -> bool:
    """
    Identify the main analysis-start messages that should stand out.
    """
    text = str(event_text or "").lstrip("\t").strip()
    headline_prefixes = (
        "Rendering R-group pair matrix",
        "Drawing counts table",
        "Drawing 'global activity ranges' plot",
        "Drawing Tanimoto similarity matrix",
        "Drawing clustered similarity matrix",
        "Drawing SAL plot",
        "Drawing 'Structure-Activity Landscape' (SAL) plot",
        "Running stereoisomer analysis",
        "Running matched molecular pairs analysis",
        "Drawing 'MMPA Network'",
        "Drawing 'descriptors' plot",
        "Drawing 'pca' plot",
        "Running prediction workflow",
    )
    return any(text.startswith(prefix) for prefix in headline_prefixes)


def _sanitize_log_message(message: str) -> str:
    """
    Remove terminal ANSI sequences and stray control characters from log text.
    """
    text = _ANSI_ESCAPE_RE.sub("", str(message))
    cleaned_chars: list[str] = []
    for ch in text:
        if ch in ("\n", "\r", "\t"):
            cleaned_chars.append(ch)
            continue
        if ord(ch) < 32 or ord(ch) == 127:
            continue
        cleaned_chars.append(ch)
    return "".join(cleaned_chars).replace("\uFE0F", "")


def _parse_event_log_entry(message: str) -> dict[str, str]:
    """
    Format one log line as a structured three-column event entry.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = str(message).strip()
    if text.startswith("[") and "]" in text:
        closing = text.find("]")
        raw_section = text[1:closing].strip()
        body = text[closing + 1 :]
        if body.startswith(" "):
            body = body[1:]

        leading_ws_len = len(body) - len(body.lstrip(" "))
        indent_level = leading_ws_len // 4
        body = body[leading_ws_len:]

        if len(body) >= 1 and body[0] in {"•", "✅", "⚠", "❌"}:
            body = body[1:]
            body = body.lstrip("️ ")

        normalized_section = normalize_log_section(raw_section)
        if normalized_section == "LMM":
            body = ("\t" * indent_level) + body
        return {
            "timestamp": now,
            "section": normalized_section,
            "event": body,
        }
    return {
        "timestamp": now,
        "section": "",
        "event": text,
    }


def _entry_to_line(entry: dict[str, str]) -> str:
    """
    Convert one structured entry to the persisted plain-text log format.
    """
    section = str(entry.get("section", "") or "").strip()
    section_token = f"[{section}]" if section else ""
    return f"{entry.get('timestamp', '')}  {section_token}  {entry.get('event', '')}".rstrip()


def _append_event_log_row(entry: dict[str, str], state: dict[str, Any]) -> None:
    """
    Add one row to the Event log table when it already exists.
    """
    if not dpg.does_item_exist(EVENT_LOG_TABLE_TAG):
        return

    with dpg.table_row(parent=EVENT_LOG_TABLE_TAG):
        with dpg.table_cell():
            timestamp_tag = dpg.add_input_text(
                default_value=entry.get("timestamp", ""),
                readonly=True,
                width=-1,
            )
        with dpg.table_cell():
            section_tag = dpg.add_input_text(
                default_value=entry.get("section", ""),
                readonly=True,
                width=-1,
            )
        with dpg.table_cell():
            event_tag = dpg.add_input_text(
                default_value=entry.get("event", ""),
                readonly=True,
                width=-1,
            )
    if _event_log_is_headline(entry.get("event", "")):
        bold_font = None
        current_font = str(state.get("font_type", "") or "")
        if current_font and dpg.does_item_exist(f"{current_font} Bold"):
            bold_font = f"{current_font} Bold"
        elif dpg.does_item_exist("FiraCode (Mono) Bold"):
            bold_font = "FiraCode (Mono) Bold"
        if bold_font:
            dpg.bind_item_font(timestamp_tag, bold_font)
            dpg.bind_item_font(section_tag, bold_font)
            dpg.bind_item_font(event_tag, bold_font)


def _refresh_event_log_widget(state: dict[str, Any]) -> None:
    """
    Push the current event-log buffer into the GUI widget when available.
    """
    if not dpg.does_item_exist(EVENT_LOG_TABLE_TAG):
        return

    dpg.delete_item(EVENT_LOG_TABLE_TAG, children_only=True, slot=1)
    for entry in state.get("event_log_entries", []):
        _append_event_log_row(entry, state)
    try:
        dpg.set_y_scroll(EVENT_LOG_CHILD_TAG, dpg.get_y_scroll_max(EVENT_LOG_CHILD_TAG))
    except Exception:
        pass


def _append_event_log_file_line(state: dict[str, Any], line: str) -> None:
    """
    Persist one formatted GUI log line to the session log file.
    """
    log_paths = []
    primary_path = state.get("event_log_file")
    if isinstance(primary_path, str) and primary_path.strip():
        log_paths.append(primary_path)

    mirror_paths = state.get("event_log_mirror_files", [])
    if isinstance(mirror_paths, list):
        log_paths.extend(path for path in mirror_paths if isinstance(path, str) and path.strip())

    seen = set()
    for log_path in log_paths:
        normalized = os.path.normcase(os.path.abspath(log_path))
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{line}\n")
        except Exception:
            pass


def append_event_log_line(state: dict[str, Any], message: str) -> None:
    """
    Append one logical line to the in-memory GUI event log.
    """
    message = _sanitize_log_message(message).rstrip("\r\n")
    if not message:
        return

    entries = state.setdefault("event_log_entries", [])
    entry = _parse_event_log_entry(message)
    entries.append(entry)
    _append_event_log_file_line(state, _entry_to_line(entry))
    if dpg.does_item_exist(EVENT_LOG_TABLE_TAG):
        _append_event_log_row(entry, state)
        try:
            dpg.set_y_scroll(EVENT_LOG_CHILD_TAG, dpg.get_y_scroll_max(EVENT_LOG_CHILD_TAG))
        except Exception:
            pass
    else:
        _refresh_event_log_widget(state)


class _EventLogStream(io.TextIOBase):
    """
    Tee stream that forwards output to both the terminal and the GUI log.
    """

    def __init__(self, state: dict[str, Any], original_stream: Any) -> None:
        self._state = state
        self._original_stream = original_stream
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        """
        Forward writes to the original stream and mirror full lines to the GUI.
        """
        if not isinstance(data, str):
            data = str(data)

        with self._lock:
            if self._original_stream is not None:
                self._original_stream.write(data)

            self._buffer += data
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    append_event_log_line(self._state, line)

        return len(data)

    def flush(self) -> None:
        """
        Flush the original stream and commit any unterminated pending line.
        """
        with self._lock:
            if self._buffer.strip():
                append_event_log_line(self._state, self._buffer)
            self._buffer = ""
            if self._original_stream is not None:
                self._original_stream.flush()

    def isatty(self) -> bool:
        """
        Preserve TTY semantics when supported by the wrapped stream.
        """
        if self._original_stream is None:
            return False
        return bool(getattr(self._original_stream, "isatty", lambda: False)())


def install_event_log_capture(state: dict[str, Any]) -> None:
    """
    Redirect stdout and stderr so terminal output is mirrored into the GUI.
    """
    if state.get("_event_log_capture_installed", False):
        return

    state.setdefault("event_log_entries", [])
    log_path = state.get("event_log_file")
    if not isinstance(log_path, str) or not log_path.strip():
        runtime_dir = state.get("user_data_dir")
        if isinstance(runtime_dir, str) and runtime_dir.strip():
            log_path = os.path.join(runtime_dir, "event_log.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        else:
            log_path = str(user_data_path("event_log.log"))
        state["event_log_file"] = log_path
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_event_log_file_line(state, f"{started}  [SYSTEM]  Event log started")
    state["_event_log_stdout_original"] = sys.stdout
    state["_event_log_stderr_original"] = sys.stderr
    sys.stdout = _EventLogStream(state, sys.stdout)
    sys.stderr = _EventLogStream(state, sys.stderr)
    state["_event_log_capture_installed"] = True


def restore_event_log_capture(state: dict[str, Any]) -> None:
    """
    Restore the original terminal streams when the app is closing.
    """
    if not state.get("_event_log_capture_installed", False):
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    sys.stdout = state.get("_event_log_stdout_original", sys.__stdout__)
    sys.stderr = state.get("_event_log_stderr_original", sys.__stderr__)
    state["_event_log_capture_installed"] = False


def ensure_event_log_window(state: dict[str, Any]) -> None:
    """
    Create the Event log window once and hydrate it with any buffered entries.
    """
    if dpg.does_item_exist(EVENT_LOG_WINDOW_TAG):
        _refresh_event_log_widget(state)
        return

    with dpg.window(
        label="Event log",
        tag=EVENT_LOG_WINDOW_TAG,
        show=False,
        width=max(760, int(state.get("design_ref_width", 1400) * 0.48)),
        height=max(520, int(state.get("design_ref_height", 900) * 0.62)),
        no_collapse=False,
    ):
        with dpg.child_window(
            tag=EVENT_LOG_CHILD_TAG,
            width=-1,
            height=-1,
            border=False,
            no_scrollbar=False,
            horizontal_scrollbar=True,
        ):
            if not dpg.does_item_exist(EVENT_LOG_INPUT_THEME_TAG):
                with dpg.theme(tag=EVENT_LOG_INPUT_THEME_TAG):
                    with dpg.theme_component(dpg.mvInputText):
                        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                        dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0, category=dpg.mvThemeCat_Core)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0, category=dpg.mvThemeCat_Core)
            with dpg.table(
                tag=EVENT_LOG_TABLE_TAG,
                header_row=True,
                width=-1,
                height=-1,
                resizable=True,
                policy=dpg.mvTable_SizingStretchProp,
                borders_innerH=False,
                borders_outerH=False,
                borders_innerV=False,
                borders_outerV=False,
                row_background=False,
                scrollY=True,
                scrollX=True,
                freeze_rows=1,
            ):
                dpg.add_table_column(label="Time stamp", init_width_or_weight=0.22)
                dpg.add_table_column(label="Section", init_width_or_weight=0.16)
                dpg.add_table_column(label="Event", init_width_or_weight=0.62)

    dpg.bind_item_theme(EVENT_LOG_CHILD_TAG, apply_input_text_theme())
    dpg.bind_item_theme(EVENT_LOG_TABLE_TAG, EVENT_LOG_INPUT_THEME_TAG)
    _refresh_event_log_widget(state)


def show_event_log_window(state: dict[str, Any]) -> None:
    """
    Show and focus the Event log window.
    """
    ensure_event_log_window(state)
    dpg.show_item(EVENT_LOG_WINDOW_TAG)
    dpg.configure_item(EVENT_LOG_WINDOW_TAG, collapsed=False)
    dpg.focus_item(EVENT_LOG_WINDOW_TAG)
    _refresh_event_log_widget(state)
