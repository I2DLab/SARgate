"""
=========
layout.py
=========

Layout bootstrap utilities for themes and colormaps.

Centralizes loading, normalization, persistence, and startup registration of
theme and colormap resources stored in JSON files.
"""

import copy
import json
import os
import re
from typing import Any

import dearpygui.dearpygui as dpg


def save_json_file(path: str, payload: Any) -> None:
    """
    Persist a JSON payload atomically.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)




def _resolve_recent_results_dir(path: str) -> str | None:
    """
    Normalize a recent-entry candidate to the enclosing results directory.
    """
    if not isinstance(path, str) or not path.strip():
        return None

    normalized_path = os.path.abspath(os.path.expanduser(path.strip()))
    candidate = normalized_path if os.path.isdir(normalized_path) else os.path.dirname(normalized_path)

    while candidate and candidate != os.path.dirname(candidate):
        if (
            os.path.isfile(os.path.join(candidate, "results.sof"))
            or os.path.isfile(os.path.join(candidate, "results.srf"))
        ):
            return candidate
        candidate = os.path.dirname(candidate)

    return None

def load_recent_files(recent_files_path: str) -> list[str]:
    """
    Load the recent file list from disk and normalize it.
    """
    recent_paths: list[str] = []

    if os.path.exists(recent_files_path):
        try:
            with open(recent_files_path, "r", encoding="utf-8") as f:
                stored_recent = json.load(f)

            if isinstance(stored_recent, dict):
                source_paths = stored_recent.get("paths", [])
                if isinstance(source_paths, list):
                    for path in source_paths:
                        if isinstance(path, str) and path.strip():
                            normalized_path = _resolve_recent_results_dir(path)
                            if normalized_path and normalized_path not in recent_paths:
                                recent_paths.append(normalized_path)
        except Exception as e:
            log_event("SYSTEM", f"Warning: could not load {recent_files_path}: {e}", indent=1, level="WARNING")

    try:
        save_json_file(recent_files_path, {"paths": recent_paths[:10]})
    except Exception as e:
        log_event("SYSTEM", f"Warning: could not update {recent_files_path}: {e}", indent=1, level="WARNING")

    return recent_paths[:10]


def add_recent_file(state: dict[str, Any], path: str, max_items: int = 10) -> None:
    """
    Register a recent file or directory, persist it, and refresh the GUI menu.
    """
    if not isinstance(path, str) or not path.strip():
        return

    normalized_path = _resolve_recent_results_dir(path)
    if not normalized_path:
        return

    recent_paths = [p for p in state.get("recent_files", []) if isinstance(p, str) and p.strip()]
    recent_paths = [
        p for p in recent_paths
        if _resolve_recent_results_dir(p) != normalized_path
    ]
    recent_paths.insert(0, normalized_path)
    recent_paths = recent_paths[:max_items]

    state["recent_files"] = recent_paths

    recent_files_path = state.get("recent_files_file", "")
    if recent_files_path:
        try:
            save_json_file(recent_files_path, {"paths": recent_paths})
        except Exception as e:
            log_event("SYSTEM", f"Warning: could not update {recent_files_path}: {e}", indent=1, level="WARNING")

    refresh_recent_files_menu = state.get("refresh_recent_files_menu")
    if callable(refresh_recent_files_menu):
        refresh_recent_files_menu()


def load_themes(themes_path: str, legacy_active: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load themes from disk and normalize them to the nested storage layout.
    """
    themes_store = {
        "default_themes": {},
        "custom_themes": {},
    }

    if os.path.exists(themes_path):
        try:
            with open(themes_path, "r", encoding="utf-8") as f:
                stored_themes = json.load(f)
            if isinstance(stored_themes, dict):
                if (
                    isinstance(stored_themes.get("default_themes"), dict)
                    or isinstance(stored_themes.get("custom_themes"), dict)
                ):
                    themes_store["default_themes"] = copy.deepcopy(stored_themes.get("default_themes") or {})
                    themes_store["custom_themes"] = copy.deepcopy(stored_themes.get("custom_themes") or {})
                else:
                    legacy_default_names = (
                        "SARgate",
                        "Sun",
                        "Mercury",
                        "Venus",
                        "Earth",
                        "Mars",
                        "Jupiter",
                        "Saturn",
                        "Uranus",
                        "Neptune",
                    )
                    for theme_name, theme_values in stored_themes.items():
                        if not isinstance(theme_values, dict):
                            continue
                        if theme_name in legacy_default_names:
                            themes_store["default_themes"][theme_name] = copy.deepcopy(theme_values)
                        else:
                            themes_store["custom_themes"][theme_name] = copy.deepcopy(theme_values)
        except Exception as e:
            log_event("SYSTEM", f"Warning: could not load {themes_path}: {e}", indent=1, level="WARNING")

    try:
        save_json_file(themes_path, themes_store)
    except Exception as e:
        log_event("SYSTEM", f"Warning: could not update {themes_path}: {e}", indent=1, level="WARNING")

    flattened_themes = {
        **themes_store["default_themes"],
        **themes_store["custom_themes"],
    }
    return themes_store, flattened_themes


def _normalize_rgba_list(colors: Any) -> list[list[int]]:
    """
    Normalize a list of RGBA colors to 4 integers in the 0..255 range.
    """
    normalized: list[list[int]] = []
    if not isinstance(colors, list):
        return normalized

    for color in colors:
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            continue
        rgba = list(color[:4]) if len(color) >= 4 else list(color[:3]) + [255]
        row: list[int] = []
        for idx, channel in enumerate(rgba):
            try:
                value = float(channel)
            except Exception:
                value = 255.0 if idx == 3 else 0.0
            if 0.0 <= value <= 1.0:
                value *= 255.0
            row.append(max(0, min(255, int(round(value)))))
        normalized.append(row)
    return normalized


def load_colormaps(colormaps_path: str) -> dict[str, Any]:
    """
    Load colormaps from disk and persist a normalized JSON structure.
    """
    normalized_store = {"continuous": {}, "discrete": {}}

    if os.path.exists(colormaps_path):
        try:
            with open(colormaps_path, "r", encoding="utf-8") as f:
                stored_colormaps = json.load(f)
            if isinstance(stored_colormaps, dict):
                for group_name in ("continuous", "discrete"):
                    source_group = stored_colormaps.get(group_name)
                    if not isinstance(source_group, dict):
                        continue
                    for name, colors in source_group.items():
                        normalized_colors = _normalize_rgba_list(colors)
                        min_len = 2 if group_name == "continuous" else 3
                        if len(normalized_colors) >= min_len:
                            normalized_store[group_name][name] = normalized_colors
        except Exception as e:
            log_event("SYSTEM", f"Warning: could not load {colormaps_path}: {e}", indent=1, level="WARNING")

    try:
        save_json_file(colormaps_path, normalized_store)
    except Exception as e:
        log_event("SYSTEM", f"Warning: could not update {colormaps_path}: {e}", indent=1, level="WARNING")

    return normalized_store


def prepare_layout_resources(
    settings: dict[str, Any],
    themes_path: str,
    colormaps_path: str,
    legacy_theme: Any = None,
) -> dict[str, Any]:
    """
    Load layout resources and resolve the active theme and colormap names.
    """
    themes_store, themes = load_themes(themes_path, legacy_theme)
    colormaps_store = load_colormaps(colormaps_path)

    continuous_defs = copy.deepcopy(colormaps_store.get("continuous", {}))
    discrete_defs = copy.deepcopy(colormaps_store.get("discrete", {}))

    theme_name = settings.get("theme_name", "Midnight")
    if theme_name not in themes:
        theme_name = "Midnight" if "Midnight" in themes else next(iter(themes), "")
        settings["theme_name"] = theme_name

    colormap_continuous = settings.get("colormap_continuous", "")
    if colormap_continuous not in continuous_defs:
        colormap_continuous = next(iter(continuous_defs), "")
        settings["colormap_continuous"] = colormap_continuous

    colormap_discrete = settings.get("colormap_discrete", "")
    if colormap_discrete not in discrete_defs:
        colormap_discrete = next(iter(discrete_defs), "")
        settings["colormap_discrete"] = colormap_discrete

    return {
        "themes_store": themes_store,
        "themes": themes,
        "theme_name": theme_name,
        "theme": themes[theme_name] if theme_name else {},
        "colormaps_store": colormaps_store,
        "continuous_colormap_defs": continuous_defs,
        "discrete_colormap_defs": discrete_defs,
        "colormap_continuous": colormap_continuous,
        "colormap_discrete": colormap_discrete,
    }


def persist_layout_settings(settings_file: str, settings: dict[str, Any]) -> None:
    """
    Persist normalized layout-related settings to disk.
    """
    settings.pop("theme", None)
    save_json_file(settings_file, settings)


def _slugify_colormap_name(name: str) -> str:
    """
    Convert a colormap name into a stable Dear PyGui tag suffix.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return slug or "colormap"


def register_startup_colormaps(state: dict[str, Any]) -> None:
    """
    Register all continuous and discrete colormaps defined in state.
    """
    continuous_colormaps: dict[str, Any] = {}
    discrete_colormaps: dict[str, Any] = {}

    with dpg.colormap_registry(tag="colormap_registry"):
        for colormap_name, colors in state.get("continuous_colormap_defs", {}).items():
            tag = f"continuous_colormap_{_slugify_colormap_name(colormap_name)}"
            continuous_colormaps[colormap_name] = dpg.add_colormap(
                [tuple(color) for color in colors],
                qualitative=False,
                label=colormap_name,
                tag=tag,
            )

        for colormap_name, colors in state.get("discrete_colormap_defs", {}).items():
            tag = f"discrete_colormap_{_slugify_colormap_name(colormap_name)}"
            discrete_colormaps[colormap_name] = dpg.add_colormap(
                [tuple(color) for color in colors],
                qualitative=True,
                label=colormap_name,
                tag=tag,
            )

    state["colormaps"] = continuous_colormaps
    state["plot_colormaps"] = discrete_colormaps
from app.utils.app_logger import log_event
