"""
=================
themes_manager.py
=================

Theme and style management utilities.

Applies colour themes, font scaling, and widget-specific styles across the
entire SARgate interface. Includes functions to manage user-defined custom
themes and expose an internal style editor for live appearance adjustment.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Apply colormap
# 3. Get applied colormap color
# 4. Apply theme callback
# 5. Change font type
# 6. Custom style editor
# 7. Apply outer child theme
# 8. Apply inner child theme
# 9. Apply input text theme
# 10. Apply bordered input text theme
# 11. Apply image button theme
# 12. Apply image theme
# 13. Apply colormap theme
# 14. Apply pie chart theme
# 15. Apply dendrogram theme
# 16. Apply enrich plot theme
# 17. Apply line chart theme
# 18. Apply mmpa network theme
# 19. Apply plot theme
# 20. Apply boxplot theme
# 21. Apply infinite line theme
# 22. Apply progress bar theme

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import time
import json
import copy
import dearpygui.dearpygui as dpg
from typing import Any

from app.analysis.tools.sketcher_launcher import sync_sketcher_theme


def _save_json_file(path: str, payload: Any) -> None:
    """
    Persist a JSON payload atomically.
    """
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)


def get_darker_variant(color: Any, delta: int = 50) -> Any:
    """
    Return a slightly darker RGBA variant of the given base color.
    """
    if len(color) == 3:
        r, g, b = color
        a = 255
    else:
        r, g, b, a = color

    return (
        max(r - delta, 0),
        max(g - delta, 0),
        max(b - delta, 0),
        a,
    )


def _get_theme_selector_items(state: dict[str, Any]) -> list[str]:
    """
    Return theme names in the order shown by the style editor combo box.
    """
    themes_store = state.get("themes_store", {})
    default_items = list((themes_store.get("default_themes") or {}).keys())
    custom_items = list((themes_store.get("custom_themes") or {}).keys())

    if default_items or custom_items:
        return default_items + custom_items

    return list(state.get("themes", {}).keys())


def redraw_main_frame_overlay(state: dict[str, Any]) -> None:
    """
    Draw a thin viewport overlay frame aligned to the outer application edge.
    """
    overlay_tag = "main_frame_overlay"
    layer_tag = "main_frame_overlay_layer"

    if not dpg.does_item_exist(overlay_tag):
        dpg.add_viewport_drawlist(front=True, tag=overlay_tag)

    if dpg.does_item_exist(layer_tag):
        dpg.delete_item(layer_tag)

    with dpg.draw_layer(parent=overlay_tag, tag=layer_tag):
        theme_dict = state["themes"][state["theme_name"]]
        border_color = tuple(theme_dict["Title Bar Background"])
        viewport_w = int(dpg.get_viewport_client_width() or dpg.get_viewport_width() or 0)
        viewport_h = int(dpg.get_viewport_client_height() or dpg.get_viewport_height() or 0)
        if viewport_w <= 4 or viewport_h <= 4:
            return

        pad = 0
        x1, y1 = pad , pad + 1.0
        x2, y2 = viewport_w - pad - 1.5, viewport_h - pad - 1.5
        corner_len = max(8, min(16, min(viewport_w, viewport_h) // 18))

        dpg.draw_line((x1, y1), (x2, y1), color=border_color, thickness=1)
        dpg.draw_line((x1, y2), (x2, y2), color=border_color, thickness=1)
        dpg.draw_line((x1, y1), (x1, y2), color=border_color, thickness=1)
        dpg.draw_line((x2, y1), (x2, y2), color=border_color, thickness=1)

        corner_segments = [
            ((x1, y1), (min(x1 + corner_len, x2), y1)),
            ((x1, y1), (x1, min(y1 + corner_len, y2))),
            ((x2, y1), (max(x2 - corner_len, x1), y1)),
            ((x2, y1), (x2, min(y1 + corner_len, y2))),
            ((x1, y2), (min(x1 + corner_len, x2), y2)),
            ((x1, y2), (x1, max(y2 - corner_len, y1))),
            ((x2, y2), (max(x2 - corner_len, x1), y2)),
            ((x2, y2), (x2, max(y2 - corner_len, y1))),
        ]
        for p1, p2 in corner_segments:
            dpg.draw_line(p1, p2, color=border_color, thickness=2)


def _recreate_theme(tag: str) -> None:
    """
    Delete an existing theme tag so it can be rebuilt with fresh colours.

    Args:
        tag (str): Theme tag to recreate.

    Returns:
        None: Existing theme items are removed in place when present.
    """
    try:
        dpg.delete_item(tag)
    except Exception:
        pass

    try:
        if hasattr(dpg, "does_alias_exist") and dpg.does_alias_exist(tag):
            dpg.remove_alias(tag)
    except Exception:
        pass


def _ensure_image_button_theme(state: dict[str, Any]) -> str:
    """
    Create or recreate the shared image-button theme for the active GUI theme.

    Args:
        state (dict[str, Any]): Shared application state with theme settings.

    Returns:
        str: Theme tag bound to image buttons across the interface.
    """
    theme_dict = state["themes"][state["theme_name"]]
    reduced_border = max(0.0, float(theme_dict["Frame Border Size"]) * 0.78)

    _recreate_theme("image_button_dynamic_theme")
    with dpg.theme(tag="image_button_dynamic_theme"):
        with dpg.theme_component(dpg.mvImageButton):
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Button Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Button Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Button Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, theme_dict["Border Shadow"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, reduced_border)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, theme_dict["Frame rounding"])

    return "image_button_dynamic_theme"


def refresh_overrides(state: dict[str, Any]) -> None:
    """
    Refresh theme-dependent cosmetic overrides introduced for key UI widgets.

    Args:
        state (dict[str, Any]): Shared application state containing the active
            theme colours and registered fonts.

    Returns:
        None: This helper recreates and rebinds local UI themes in place.
    """
    theme_dict = state["themes"][state["theme_name"]]
    subtle_shadow = (0, 0, 0, 0) if state["theme_name"] in ("Light", "Sun") else theme_dict["Secondary Background"]
    button_shadow = (0, 0, 0, 0) if state["theme_name"] in ("Light", "Sun") else theme_dict["Secondary Background"]
    image_button_theme_tag = _ensure_image_button_theme(state)

    help_text = theme_dict["Text Color"] if state["theme_name"] in ("Light", "Sun") else (255, 255, 255, 255)

    _recreate_theme("help_button_theme")
    with dpg.theme(tag="help_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Button Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Button Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, help_text, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, button_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(4, int(theme_dict["Frame rounding"]) - 1))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 4)

    _recreate_theme("link_button_theme")
    with dpg.theme(tag="link_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Button Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, button_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)

    _recreate_theme("core_panel_theme")
    with dpg.theme(tag="core_panel_theme"):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, max(10, int(theme_dict["Frame rounding"]) + 2))
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)

    _recreate_theme("main_window_theme")
    with dpg.theme(tag="main_window_theme"):
        with dpg.theme_component(dpg.mvWindowAppItem):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, theme_dict["Main Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 10)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 10)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)

    _recreate_theme("workspace_shell_theme")
    with dpg.theme(tag="workspace_shell_theme"):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, theme_dict["Main Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, max(14, int(theme_dict["Window rounding"]) + 4))
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)

    _recreate_theme("console_panel_theme")
    with dpg.theme(tag="console_panel_theme"):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, theme_dict["Frame Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, max(10, int(theme_dict["Frame rounding"]) + 1))
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)

    _recreate_theme("manager_panel_theme")
    with dpg.theme(tag="manager_panel_theme"):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, theme_dict["Frame Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, max(12, int(theme_dict["Frame rounding"]) + 2))
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)

    _recreate_theme("data_surface_theme")
    with dpg.theme(tag="data_surface_theme"):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, max(10, int(theme_dict["Frame rounding"]) + 1))
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)

    _recreate_theme("menu_bar_theme")
    with dpg.theme(tag="menu_bar_theme"):
        with dpg.theme_component(dpg.mvMenuBar):
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, theme_dict["Menu Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 6)

    _recreate_theme("tab_bar_theme")
    with dpg.theme(tag="tab_bar_theme"):
        with dpg.theme_component(dpg.mvTabBar):
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, theme_dict["Tabs Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_TabBarBorderSize, 2)

    _recreate_theme("subtab_bar_theme")
    with dpg.theme(tag="subtab_bar_theme"):
        with dpg.theme_component(dpg.mvTabBar):
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, max(4, int(theme_dict["Frame rounding"]) - 1))
            dpg.add_theme_style(dpg.mvStyleVar_TabBarBorderSize, 1)

    _recreate_theme("tab_item_theme")
    with dpg.theme(tag="tab_item_theme"):
        with dpg.theme_component(dpg.mvTab):
            dpg.add_theme_color(dpg.mvThemeCol_Tab, theme_dict["Tabs Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, theme_dict["Tabs Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_TabBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 13, 6)

    locked_tab_border = theme_dict["Border Color"]
    _locked_text_base = tuple(theme_dict["Text Color"])
    locked_tab_text = (
        max(_locked_text_base[0] - 72, 0),
        max(_locked_text_base[1] - 72, 0),
        max(_locked_text_base[2] - 72, 0),
        _locked_text_base[3] if len(_locked_text_base) > 3 else 255,
    )
    _recreate_theme("locked_tab_item_theme")
    with dpg.theme(tag="locked_tab_item_theme"):
        with dpg.theme_component(dpg.mvTab):
            dpg.add_theme_color(dpg.mvThemeCol_Tab, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, locked_tab_border, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, locked_tab_text, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_TabBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 13, 6)

    _recreate_theme("subtab_item_theme")
    with dpg.theme(tag="subtab_item_theme"):
        with dpg.theme_component(dpg.mvTab):
            dpg.add_theme_color(dpg.mvThemeCol_Tab, theme_dict["Tabs Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, theme_dict["Frame Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, max(4, int(theme_dict["Frame rounding"]) - 1))
            dpg.add_theme_style(dpg.mvStyleVar_TabBorderSize, 1)

    _recreate_theme("tab_button_theme")
    with dpg.theme(tag="tab_button_theme"):
        with dpg.theme_component(getattr(dpg, "mvTabButton", dpg.mvButton)):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Tabs Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, theme_dict["Tabs Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_TabBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 5)

    _recreate_theme("locked_tab_button_theme")
    with dpg.theme(tag="locked_tab_button_theme"):
        with dpg.theme_component(getattr(dpg, "mvTabButton", dpg.mvButton)):
            dpg.add_theme_color(dpg.mvThemeCol_Button, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, locked_tab_text, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_TabBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 5)

    _recreate_theme("invisible_tab_button_theme")
    with dpg.theme(tag="invisible_tab_button_theme"):
        with dpg.theme_component(getattr(dpg, "mvTabButton", dpg.mvButton)):
            transparent = (0, 0, 0, 0)
            dpg.add_theme_color(dpg.mvThemeCol_Button, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_TabBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 18, 5)

    _recreate_theme("top_nav_image_button_disabled_theme")
    with dpg.theme(tag="top_nav_image_button_disabled_theme"):
        with dpg.theme_component(dpg.mvImageButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 6)
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, get_darker_variant(theme_dict["Tabs Color"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 6)

    _recreate_theme("top_nav_image_button_theme")
    with dpg.theme(tag="top_nav_image_button_theme"):
        with dpg.theme_component(dpg.mvImageButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Tabs Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 6)

        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Tabs Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 6)

    _recreate_theme("top_nav_image_button_active_theme")
    with dpg.theme(tag="top_nav_image_button_active_theme"):
        with dpg.theme_component(dpg.mvImageButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 6)
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, int(theme_dict["Tab rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 6)

    _recreate_theme("popup_theme")
    with dpg.theme(tag="popup_theme"):
        with dpg.theme_component(dpg.mvWindowAppItem):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, subtle_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, max(8, int(theme_dict["Window rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)

    _recreate_theme("workflow_primary_button_theme")
    with dpg.theme(tag="workflow_primary_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Button Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Button Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, button_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 6)

    _recreate_theme("workflow_secondary_button_theme")
    with dpg.theme(tag="workflow_secondary_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Button Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, button_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 6)

    _recreate_theme("workflow_select_file_button_theme")
    with dpg.theme(tag="workflow_select_file_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Button Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, button_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 6)

    _recreate_theme("workflow_danger_button_theme")
    with dpg.theme(tag="workflow_danger_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (170, 56, 56, 220), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (205, 72, 72, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (150, 42, 42, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, button_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 6)

    _recreate_theme("workflow_status_text_theme")
    with dpg.theme(tag="workflow_status_text_theme"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)

    _recreate_theme("workflow_timer_theme")
    with dpg.theme(tag="workflow_timer_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0)

    _recreate_theme("workflow_file_name_text_theme")
    with dpg.theme(tag="workflow_file_name_text_theme"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, theme_dict["Text Color"], category=dpg.mvThemeCat_Core)

    _recreate_theme("workflow_muted_text_theme")
    with dpg.theme(tag="workflow_muted_text_theme"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)

    _recreate_theme("workflow_section_title_theme")
    with dpg.theme(tag="workflow_section_title_theme"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)

    _recreate_theme("workflow_label_text_theme")
    with dpg.theme(tag="workflow_label_text_theme"):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, theme_dict["Text Color"], category=dpg.mvThemeCat_Core)

    _recreate_theme("workflow_combo_theme")
    with dpg.theme(tag="workflow_combo_theme"):
        with dpg.theme_component(dpg.mvCombo):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, theme_dict["Frame Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 7)

    _recreate_theme("overview_choice_button_theme")
    with dpg.theme(tag="overview_choice_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Button Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Button Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Button Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, button_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)

    _recreate_theme("overview_choice_button_active_theme")
    with dpg.theme(tag="overview_choice_button_active_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Button Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Button Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, button_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)

    _recreate_theme("data_table_theme")
    with dpg.theme(tag="data_table_theme"):
        with dpg.theme_component(dpg.mvTable):
            dpg.add_theme_color(dpg.mvThemeCol_Header, theme_dict["Table Header"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, theme_dict["Table Header"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, theme_dict["Main Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 10, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)

    for item_tag in ["github_popup_button", "report_issue_popup_button"]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "link_button_theme")

    if dpg.does_item_exist("main_window"):
        dpg.bind_item_theme("main_window", "main_window_theme")

    if dpg.does_item_exist("main_menu_bar"):
        dpg.bind_item_theme("main_menu_bar", "menu_bar_theme")

    for item_tag in ["github_popup", "contacts_popup", "report_issue_popup", "about_SARgate_popup", "cancel_confirm_popup"]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "popup_theme")

    for item_tag in [
        "file_selection_window",
        "options_window",
        "start_analysis_window",
        "library_table_window",
        "library_preparation_window",
        "scaffold_analysis_window",
        "rga_window",
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "core_panel_theme")

    for item_tag in [
        "input_tab_child",
        "analysis_tab_child",
        "overview_tab_child",
        "r_analysis_tab_child",
        "similarity_tab_child",
        "stereo_tab_child",
        "mmpa_tab_child",
        "chemspace_tab_child",
        "prediction_tab_child",
        "utilities_tab_child",
        "slith_tab_child",
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "workspace_shell_theme")

    for item_tag in [
        "file_selection_and_run_window",
        "subset_choice",
        "molecule_choice",
        "r_group_choice",
        "properties_window",
        "activities_window",
        "image_checkboxes_window",
        "enrichment_plot_window",
        "overview_table_manager",
        "overview_table_window",
        "counts_selection_window",
        "file_selection_window",
        "options_window",
        "start_analysis_window",
        "library_preparation_window",
        "scaffold_analysis_window",
        "rga_window",
        "overview_selection_table_window",
        "counts_boxplot_manager_window",
        "descriptors_manager_window",
        "dendrogram_manager_window",
        "dendrogram_subset_panel",
        "heatmap_manager_window",
        "landscape_manager_window",
        "pca_manager_window",
        "umap_manager_window",
        "tsne_manager_window",
        "global_act_ranges_left",
        "global_act_ranges_controls",
        "similarity_manager_window",
        "clustered_similarity_manager_window",
        "isomers_manager_window",
        "mmpa_window",
        "prediction_manager_host",
        "prediction_manager_window",
        "prediction_manager_controls",
        "mmpa_group_sidebar_window",
        "notes_subsets_list",
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "manager_panel_theme")

    for item_tag in [
        "library_table_window",
        "mol_image",
        "enrichment_plot_window",
        "counts_scaffold_img_window",
        "counts_boxplot_window",
        "counts_rgroup_details_window",
        "counts_table_window",
        "similarity_tanimoto_window",
        "similarity_tanimoto_mol_couple_window",
        "clustered_similarity_matrix_window",
        "clustered_similarity_mol_couple_window",
        "landscape_window",
        "landscape_details_window",
        "descriptors_window",
        "descriptors_details_window",
        "dendrogram_window",
        "dendrogram_details_window",
        "heatmap_window",
        "heatmap_details_window",
        "pca_window",
        "pca_details_window",
        "umap_window",
        "umap_details_window",
        "tsne_window",
        "tsne_details_window",
        "prediction_output_host",
        "prediction_results_window",
        "prediction_plot_window",
        "prediction_details_window",
        "isomers_images_main_window",
        "mmpa_global_plot_window",
        "mmpa_subset_plot_window",
        "mmpa_network_plot_window",
        "mmpa_table_window",
        "utils_window",
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "data_surface_theme")

    for item_tag in ["tab_bar", "overview_tab_bar", "r_analysis_tab_bar", "similarity_tab_bar", "chemspace_tab_bar", "mmpa_tab_bar"]:
        if item_tag == "tab_bar":
            dpg.bind_item_theme(item_tag, "tab_bar_theme")
        elif dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "subtab_bar_theme")

    for item_tag in [
        "input_tab",
        "analysis_tab",
        "overview_tab",
        "r_analysis_tab",
        "similarity_tab",
        "stereo_tab",
        "mmpa_tab",
        "chemspace_tab",
        "prediction_tab",
        "utilities_tab",
        "slith_tab",
    ]:
        if dpg.does_item_exist(item_tag):
            if item_tag in (state.get("locked_text_tabs") or set()):
                dpg.bind_item_theme(item_tag, "locked_tab_item_theme")
            else:
                dpg.bind_item_theme(item_tag, "tab_item_theme")

    locked_text_tab_buttons = state.get("locked_text_tab_buttons") or set()
    for item_tag in [
        "sketcher_text_button",
        "event_log_text_button",
        "notes_text_button",
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(
                item_tag,
                "locked_tab_button_theme" if item_tag in locked_text_tab_buttons else "tab_button_theme",
            )

    if dpg.does_item_exist("tab_placeholder_button_1"):
        dpg.bind_item_theme("tab_placeholder_button_1", "invisible_tab_button_theme")
    if dpg.does_item_exist("tab_placeholder_button_2"):
        dpg.bind_item_theme("tab_placeholder_button_2", "invisible_tab_button_theme")

    for item_tag in [
        "overview_decomposition_subtab",
        "overview_table_subtab",
        "r_analysis_counts_subtab",
        "r_analysis_table_subtab",
        "similarity_matrix_subtab",
        "clustered_matrix_subtab",
        "landscape_tab",
        "descriptors_tab",
        "dendrogram_tab",
        "pca_tab",
        "mmpa_table_subtab",
        "mmpa_network_subtab",
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "subtab_item_theme")

    for item_tag in [
        "input_nav_button",
        "analysis_nav_button",
        "overview_nav_button",
        "r_analysis_nav_button",
        "similarity_nav_button",
        "stereo_nav_button",
        "mmpa_nav_button",
        "chemspace_nav_button",
        "prediction_nav_button",
        "notes_nav_button",
        "sketcher_nav_button",
        "utilities_nav_button",
        "event_log_nav_button",
        "slith_nav_button",
    ]:
        if dpg.does_item_exist(item_tag):
            top_nav_button_enabled = state.get("top_nav_button_enabled", {})
            if top_nav_button_enabled.get(item_tag, True):
                dpg.bind_item_theme(item_tag, "top_nav_image_button_theme")
            else:
                dpg.bind_item_theme(item_tag, "top_nav_image_button_disabled_theme")

    if callable(state.get("refresh_top_nav_icons")):
        try:
            state["refresh_top_nav_icons"]()
        except Exception:
            pass

    for item_tag, theme_tag in [
        ("select_file_button", "workflow_select_file_button_theme"),
        ("confirm_button", "workflow_primary_button_theme"),
        ("stop_button", "workflow_danger_button_theme"),
        ("cancel_confirm_yes_button", "workflow_danger_button_theme"),
        ("cancel_confirm_no_button", "workflow_secondary_button_theme"),
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, theme_tag)

    if dpg.does_item_exist("file_name_text"):
        dpg.bind_item_theme("file_name_text", "workflow_file_name_text_theme")
        if dpg.does_item_exist("FiraCode (Mono) Large"):
            dpg.bind_item_font("file_name_text", "FiraCode (Mono) Large")
        elif dpg.does_item_exist("FiraCode (Mono)"):
            dpg.bind_item_font("file_name_text", "FiraCode (Mono)")

    if dpg.does_item_exist("execution_time_label"):
        dpg.bind_item_theme("execution_time_label", "workflow_timer_theme")
        if dpg.does_item_exist("FiraCode (Mono) Large"):
            dpg.bind_item_font("execution_time_label", "FiraCode (Mono) Large")
        elif dpg.does_item_exist("FiraCode (Mono)"):
            dpg.bind_item_font("execution_time_label", "FiraCode (Mono)")

    if dpg.does_item_exist("execution_time_text"):
        dpg.bind_item_theme("execution_time_text", "workflow_timer_theme")
        if dpg.does_item_exist("FiraCode (Mono) Bold Large"):
            dpg.bind_item_font("execution_time_text", "FiraCode (Mono) Bold Large")
        elif dpg.does_item_exist("FiraCode (Mono) Bold"):
            dpg.bind_item_font("execution_time_text", "FiraCode (Mono) Bold")

    for item_tag in [
        "library_preparation_title",
        "substructure_analysis_title",
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "workflow_section_title_theme")

    for item_tag in [
        "search_on_label",
        "target_id_label",
        "filter_type_label",
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "workflow_label_text_theme")

    for item_tag in [
        "search_on_database_combo",
        "Filter by structure similarity",
        "Structure for which to calculate similarity",
        "Duplicates handling",
        "Subsets collection method",
        "theme_selector",
        "font_selector",
        "colormap_selector",
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "workflow_combo_theme")

    for item_tag in [
        "library_overview_table",
        "overview_selection_table",
        "overview_image_table",
        "overview_table",
    ]:
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, "data_table_theme")

    for item_tag, info in (state.get("responsive_images") or {}).items():
        parent_tag = info.get("parent", "")
        is_image_button = (
            item_tag in {"scaff_img", "mol_img", "rgroup_img"}
            or item_tag.endswith("_tag")
            or str(parent_tag).startswith("counts_rgroup_texture_")
            or str(parent_tag).startswith("stereo_texture_")
            or str(parent_tag).startswith("overview_table_texture_")
        )
        if is_image_button and dpg.does_item_exist(item_tag):
            try:
                dpg.bind_item_theme(item_tag, image_button_theme_tag)
            except Exception:
                pass

    for container_tag, active_tag in [
        ("subset_choice_group", state.get("last_clicked_button_sub")),
        ("molecule_choice_group", state.get("last_clicked_button_mol")),
        ("r_group_choice_group", state.get("last_clicked_button_r")),
    ]:
        if not dpg.does_item_exist(container_tag):
            continue
        for child_tag in dpg.get_item_children(container_tag, 1) or []:
            if dpg.does_item_exist(child_tag):
                dpg.bind_item_theme(child_tag, "overview_choice_button_theme")
        if active_tag and dpg.does_item_exist(active_tag):
            dpg.bind_item_theme(active_tag, "overview_choice_button_active_theme")

    if callable(state.get("refresh_main_chrome")):
        try:
            state["refresh_main_chrome"]()
        except Exception:
            pass


# Step 1.1  Apply colormap to the whole GUI (apply_colormap)
# -----------------------------------------------------------------------------
# 2. Apply colormap
# -----------------------------------------------------------------------------
def apply_colormap(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    """
    Applies the selected colormap to the entire application GUI.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    # Extract and store colormap name
    colormap_name = app_data
    state["colormap_continuous"] = colormap_name
    state["settings"]["colormap_continuous"] = colormap_name
    state["settings"].pop("applied_colormap", None)

    # Save settings to JSON file
    with open(state["settings_file"], "w") as f:
        json.dump(state["settings"], f, indent=4)
    
    # Refresh colormap bindings in all relevant widgets
    for colormap_widget_tag in [
        "enrichment_colormap_scale",
        "tanimoto_matrix_colormap_scale",
        "clustered_matrix_colormap_scale",
        "tanimoto_matrix",
        "clustered_matrix",
        "landscape_sali_index_thresh",
        "landscape_colormap_scale",
        "heatmap_colormap_scale",
        "pca_colormap_scale",
        "pca_activity_colormap_button"
    ]:
        if dpg.does_item_exist(colormap_widget_tag):
            dpg.bind_colormap(colormap_widget_tag, state["colormaps"][state["colormap_continuous"]])
    if dpg.does_item_exist("tan_colormap_choice"):
        dpg.set_value("tan_colormap_choice", state["colormap_continuous"])

    # Trigger colour refresh on active views that use bucket themes
    # Refresh points in landscape
    try:
        if callable(state.get("landscape_refresh_colors")):
            state["landscape_refresh_colors"]()
    except Exception as e:
        log_exception("SIMILARITY", "landscape refresh skipped", e, indent=1)

    # Refresh points in pca
    try:
        if callable(state.get("pca_refresh_colors")):
            state["pca_refresh_colors"]()
    except Exception as e:
        log_exception("CHEMSPACE", "pca refresh skipped", e, indent=1)

    try:
        if callable(state.get("umap_refresh_colors")):
            state["umap_refresh_colors"]()
    except Exception as e:
        log_exception("CHEMSPACE", "umap refresh skipped", e, indent=1)

    try:
        if callable(state.get("tsne_refresh_colors")):
            state["tsne_refresh_colors"]()
    except Exception as e:
        log_exception("CHEMSPACE", "tsne refresh skipped", e, indent=1)

    # Refresh cells in RGroups table
    try:
        if callable(state.get("rgroups_refresh_colors")):
            state["rgroups_refresh_colors"]()
    except Exception as e:
        log_exception("R-ANALYSIS", "rgroups refresh skipped", e, indent=1)

    # Refresh enrichment plot in overview/decomposition
    try:
        if callable(state.get("enrichment_refresh_colors")):
            state["enrichment_refresh_colors"]()
    except Exception as e:
        log_exception("OVERVIEW", "enrichment refresh skipped", e, indent=1)

    # Refresh all-subsets activity ranges popup
    try:
        if callable(state.get("global_ranges_refresh_colors")):
            state["global_ranges_refresh_colors"]()
    except Exception as e:
        log_exception("OVERVIEW", "global ranges refresh skipped", e, indent=1)

    try:
        if callable(state.get("prediction_refresh_colors")):
            state["prediction_refresh_colors"](state)
    except Exception as e:
        log_exception("PREDICTION", "prediction refresh skipped", e, indent=1)


def apply_plot_colormap(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    """
    Apply the selected plot colormap and persist it in settings.

    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Selected plot colormap name.
        state (dict[str, Any]): Shared application state.

    Returns:
        None: This routine updates state and refreshes visible plot bindings.
    """
    plot_colormap_name = app_data
    if plot_colormap_name not in state["plot_colormaps"]:
        return

    state["colormap_discrete"] = plot_colormap_name
    state["settings"]["colormap_discrete"] = plot_colormap_name
    state["settings"].pop("applied_plot_colormap", None)
    _save_json_file(state["settings_file"], state["settings"])

    for plot_tag in [
        "counts_plot",
        "target_pie",
        "activity_pie",
        "assay_pie",
        "mmpa_plot_global",
        "mmpa_plot_subset",
    ]:
        if dpg.does_item_exist(plot_tag):
            dpg.bind_colormap(plot_tag, state["plot_colormaps"][plot_colormap_name])

    def _refresh_counts_boxplot_now() -> None:
        """
        Refresh the counts boxplot colors immediately.

        Args:
            None.

        Returns:
            None: This routine updates the UI in place.
        """
        try:
            if callable(state.get("counts_boxplot_refresh_colors")):
                state["counts_boxplot_refresh_colors"]()
        except Exception as e:
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")
            log_exception("R-ANALYSIS", "counts_boxplot refresh skipped", e, indent=1)

    _refresh_counts_boxplot_now()

    try:
        if callable(state.get("pca_refresh_colors")):
            state["pca_refresh_colors"]()
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        log_exception("CHEMSPACE", "pca refresh skipped", e, indent=1)

    try:
        if callable(state.get("umap_refresh_colors")):
            state["umap_refresh_colors"]()
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        log_exception("CHEMSPACE", "umap refresh skipped", e, indent=1)

    try:
        if callable(state.get("tsne_refresh_colors")):
            state["tsne_refresh_colors"]()
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        log_exception("CHEMSPACE", "tsne refresh skipped", e, indent=1)

    try:
        if callable(state.get("chemspace_dendrogram_refresh_colors")):
            state["chemspace_dendrogram_refresh_colors"]()
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        log_exception("CHEMSPACE", "dendrogram refresh skipped", e, indent=1)

    try:
        if callable(state.get("mmpa_network_refresh_colors")):
            state["mmpa_network_refresh_colors"]()
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        log_exception("MMPA", "network refresh skipped", e, indent=1)

    try:
        if callable(state.get("similarity_tanimoto_refresh_highlight")):
            state["similarity_tanimoto_refresh_highlight"]()
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        log_exception("SIMILARITY", "tanimoto highlight refresh skipped", e, indent=1)

    try:
        if callable(state.get("similarity_cluster_refresh_highlight")):
            state["similarity_cluster_refresh_highlight"]()
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        log_exception("SIMILARITY", "clustered highlight refresh skipped", e, indent=1)

    try:
        if callable(state.get("prediction_refresh_colors")):
            state["prediction_refresh_colors"](state)
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        log_exception("PREDICTION", "prediction refresh skipped", e, indent=1)

    # Keep a one-frame-later pass as a safety net in case the current frame
    # still contains a stale drawlist reference.
    dpg.set_frame_callback(dpg.get_frame_count() + 1, _refresh_counts_boxplot_now)
    dpg.set_frame_callback(
        dpg.get_frame_count() + 1,
        lambda: state["pca_refresh_colors"]() if callable(state.get("pca_refresh_colors")) else None,
    )
    dpg.set_frame_callback(
        dpg.get_frame_count() + 1,
        lambda: state["chemspace_dendrogram_refresh_colors"]() if callable(state.get("chemspace_dendrogram_refresh_colors")) else None,
    )
    dpg.set_frame_callback(
        dpg.get_frame_count() + 1,
        lambda: state["mmpa_network_refresh_colors"]() if callable(state.get("mmpa_network_refresh_colors")) else None,
    )
    dpg.set_frame_callback(
        dpg.get_frame_count() + 1,
        lambda: state["similarity_tanimoto_refresh_highlight"]() if callable(state.get("similarity_tanimoto_refresh_highlight")) else None,
    )
    dpg.set_frame_callback(
        dpg.get_frame_count() + 1,
        lambda: state["similarity_cluster_refresh_highlight"]() if callable(state.get("similarity_cluster_refresh_highlight")) else None,
    )
    dpg.set_frame_callback(
        dpg.get_frame_count() + 1,
        lambda: state["prediction_refresh_colors"](state) if callable(state.get("prediction_refresh_colors")) else None,
    )


# Step 1.2  Get interpolated colour from continuous colormap (get_continuous_colormap_color)
# -----------------------------------------------------------------------------
# 3. Get continuous colormap color
# -----------------------------------------------------------------------------
def get_continuous_colormap_color(norm_val: Any, state: dict[str, Any]) -> Any:
    """
    Returns an RGBA tuple (0–255) continuously interpolated from the currently applied colormap.

    Args:
        norm_val (float): Normalised value in [0, 1].
        state (dict): Application state dictionary containing:
            - "colormap_continuous": name of the current continuous colormap
            - optional "colormap_lut": a custom list of RGBA tuples to use as a lookup table (LUT)
            - optional "continuous_colormap_defs": JSON-backed color stops

    Returns:
        tuple[int, int, int, int]: Interpolated RGBA colour.
    """

    # --- Clamp normalised value ---
    t = 0.0 if norm_val is None or (isinstance(norm_val, float) and not (0.0 <= norm_val <= 1.0)) else float(norm_val)
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0

    # --- 1) Use custom LUT if available (continuous interpolation) ---
    lut = state.get("colormap_lut")
    if isinstance(lut, (list, tuple)) and len(lut) >= 2:
        idx = t * (len(lut) - 1)
        i0 = int(idx)
        i1 = min(i0 + 1, len(lut) - 1)
        f = idx - i0
        r0, g0, b0, a0 = lut[i0]
        r1, g1, b1, a1 = lut[i1]
        r = int(round(r0 + (r1 - r0) * f))
        g = int(round(g0 + (g1 - g0) * f))
        b = int(round(b0 + (b1 - b0) * f))
        a = int(round(a0 + (a1 - a0) * f))
        return (r, g, b, a)

    name = state["colormap_continuous"]
    stops = [
        tuple(color)
        for color in (state.get("continuous_colormap_defs", {}) or {}).get(name, [])
    ]
    if not stops:
        colormap_defs = state.get("continuous_colormap_defs", {}) or {}
        if colormap_defs:
            fallback_name = next(iter(colormap_defs))
            stops = [tuple(color) for color in colormap_defs.get(fallback_name, [])]
        if not stops:
            stops = [(128, 128, 128, 255), (240, 240, 240, 255)]

    # --- 2) Continuous interpolation between colour stops ---
    if len(stops) == 1:
        return stops[0]

    segments = len(stops) - 1
    pos = t * segments
    i = int(pos)
    if i >= segments:
        i = segments - 1
        pos = float(segments)
    f = pos - i  # fraction within the segment

    r0, g0, b0, a0 = stops[i]
    r1, g1, b1, a1 = stops[i + 1]
    r = int(round(r0 + (r1 - r0) * f))
    g = int(round(g0 + (g1 - g0) * f))
    b = int(round(b0 + (b1 - b0) * f))
    a = int(round(a0 + (a1 - a0) * f))
    return (r, g, b, a)


# -----------------------------------------------------------------------------
# 4. Apply theme callback
# -----------------------------------------------------------------------------
def apply_theme_callback(sender: Any, app_data: Any, state: dict[str, Any]) -> Any:
    """
    Applies the selected theme to the entire application GUI.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """

    # Extract and store theme name
    theme_name = app_data
    theme_dict = state["themes"][theme_name]

    # Update state (persist chosen theme in memory and settings)
    state["theme_name"] = theme_name
    state["theme"] = theme_dict
    state["settings"]["theme_name"] = theme_name

    if state.get("colormap_discrete") not in state.get("plot_colormaps", {}):
        fallback_plot_colormap = next(iter(state.get("plot_colormaps", {})), "")
        state["colormap_discrete"] = fallback_plot_colormap
        state["settings"]["colormap_discrete"] = fallback_plot_colormap
        state["settings"].pop("applied_plot_colormap", None)
    

    # Save settings to JSON file
    state["settings"].pop("theme", None)
    _save_json_file(state["settings_file"], state["settings"])

    # -----------------------------------------------------------------------------
    # 4.1. Get darker variant
    # -----------------------------------------------------------------------------
    def get_darker_variant(color: Any, delta: int = 10) -> Any:
        """
        Returns a slightly darker RGBA variant of the given base color.

        Args:
            color (tuple): Base color as (r, g, b) or (r, g, b, a).
            delta (int, optional): Amount to darken each RGB channel (default = 10).

        Returns:
            tuple: Darker color as (r, g, b, a), clamped to valid [0, 255] range.
        """
        # --- Normalise input to RGBA ---
        if len(color) == 3:
            r, g, b = color
            a = 255
        else:
            r, g, b, a = color

        # --- Compute darker variant (clamped) ---
        darker = (max(r - delta, 0),
                max(g - delta, 0),
                max(b - delta, 0),
                a)

        return darker

    def get_lighter_variant(color: Any, delta: int = 16) -> Any:
        """
        Return a slightly lighter RGBA variant of the given base colour.

        Args:
            color (Any): Base colour as RGB or RGBA tuple.
            delta (int, optional): Amount to lighten each RGB channel.

        Returns:
            Any: Lightened RGBA tuple.
        """
        if len(color) == 3:
            r, g, b = color
            a = 255
        else:
            r, g, b, a = color

        return (min(r + delta, 255), min(g + delta, 255), min(b + delta, 255), a)

    is_light_theme = theme_name in ("Light", "Sun")
    panel_shadow = (0, 0, 0, 0) if is_light_theme else get_darker_variant(theme_dict["Secondary Background"], 18)
    disabled_text = get_darker_variant(theme_dict["Text Color"], 88) if is_light_theme else get_darker_variant(theme_dict["Text Color"], 96)
    frame_hover = get_lighter_variant(theme_dict["Frame Background"], 10) if is_light_theme else get_lighter_variant(theme_dict["Frame Background"], 18)
    frame_active = get_lighter_variant(theme_dict["Frame Background"], 18) if is_light_theme else get_lighter_variant(theme_dict["Frame Background"], 26)


    # Set viewport background colour according to theme
    dpg.configure_viewport(dpg.get_viewport_configuration("SARgate"),
                           clear_color=theme_dict["Main Background"])

    # Create a global theme object and apply styles/colours to core widgets
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Windows
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, theme_dict["Window rounding"])
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)
            
            # Child windows
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, max(6, theme_dict["Frame rounding"]))
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            
            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, theme_dict["Text Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, disabled_text, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            
            # Title Bar
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            
            # Tabs
            dpg.add_theme_color(dpg.mvThemeCol_Tab, theme_dict["Tabs Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, get_darker_variant(theme_dict["Tabs Color"], 6), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, get_darker_variant(theme_dict["Tabs Active"], 8), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, theme_dict["Tab rounding"])
            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Button Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Button Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Button Active"], category=dpg.mvThemeCat_Core)
            
            # Checkboxes
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, theme_dict["Checkmark Color"], category=dpg.mvThemeCat_Core)
            
            # Sliders
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, theme_dict["Slider Grab"], category=dpg.mvThemeCat_Core)
            
            # Frames
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, theme_dict["Frame Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, frame_hover, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, frame_active, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, theme_dict["Frame rounding"])
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, theme_dict["Frame Border Size"])
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, panel_shadow, category=dpg.mvThemeCat_Core)
            
            # Scrollbars
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, theme_dict["Scrollbar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, theme_dict["Scrollbar Grab"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, theme_dict["Frame rounding"])
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, theme_dict["Frame rounding"])
            
            # Menu Bar
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, theme_dict["Menu Bar Background"], category=dpg.mvThemeCat_Core)
            
            # Popups
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, (0, 0, 0, 110 if is_light_theme else 150), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, theme_dict["Window rounding"])
            
            # Tables
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, theme_dict["Table Header"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, get_darker_variant(theme_dict["Secondary Background"]), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, (128, 128, 128, 128), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)

            # Separators
            dpg.add_theme_style(dpg.mvStyleVar_SeparatorTextAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_SeparatorTextBorderSize, 1.0, 1.0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_SeparatorTextPadding, 1.0, 1.0, category=dpg.mvThemeCat_Core)
            
            # Combo
            dpg.add_theme_color(dpg.mvThemeCol_Header, theme_dict["Table Header"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, get_lighter_variant(theme_dict["Table Header"], 10) if is_light_theme else get_darker_variant(theme_dict["Table Header"], 10), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)

            # Resize grips and nav accents
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_NavHighlight, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)

            # Plots
            dpg.add_theme_color(dpg.mvPlotCol_Crosshairs, (0, 0, 0, 255), category=dpg.mvThemeCat_Plots)
            
            # Padding
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 10, 5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 8, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing, 30, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 15, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 20, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabBorderSize, 1, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabBarBorderSize, 1, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowTitleAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, 0.5, 0.5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotBorderSize, 0, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 2, 2, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LabelPadding, 5, 5, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LegendPadding, 5, 5, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LegendSpacing, 5, 5, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LegendInnerPadding, 5, 5, category=dpg.mvThemeCat_Plots)

        with dpg.theme_component(dpg.mvMenuItem):
            dpg.add_theme_color(dpg.mvThemeCol_Header, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(5, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)

        with dpg.theme_component(dpg.mvCheckbox):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, theme_dict["Frame Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, frame_hover, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, frame_active, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, theme_dict["Checkmark Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(5, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 7, 5)

        with dpg.theme_component(dpg.mvRadioButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, theme_dict["Button Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)

        with dpg.theme_component(dpg.mvCollapsingHeader):
            dpg.add_theme_color(dpg.mvThemeCol_Header, theme_dict["Frame Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, panel_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(7, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 9, 7)

        with dpg.theme_component(dpg.mvTooltip):
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Title Bar Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, panel_shadow, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, max(7, int(theme_dict["Window rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8)


    # Bind the theme to the entire GUI
    dpg.bind_theme(global_theme)

    for plot_tag in [
        "counts_plot",
        "target_pie",
        "activity_pie",
        "assay_pie"
    ]:
        if dpg.does_item_exist(plot_tag):
            dpg.bind_colormap(plot_tag, state["plot_colormaps"][state["colormap_discrete"]])

    for plot_tag in [
        "tanimoto_matrix",
        "clustered_matrix",
        "counts_plot",
        "landscape_plot",
        "pca_plot_2d",
        "umap_plot",
        "tsne_plot",
        "descriptors_2d",
        "prediction_plot"
    ]:
        if dpg.does_item_exist(plot_tag):
            dpg.bind_item_theme(plot_tag, apply_plot_theme(state))

    for plot_tag in [
        "global_activity_plot",
        "enrichment_plot",
    ]:
        if dpg.does_item_exist(plot_tag):
            dpg.bind_item_theme(plot_tag, apply_enrich_plot_theme(state))

    for plot_tag in [
        "scaffold_hierarchical_dendrogram",
        "chemspace_scaffold_hierarchical_dendrogram",
    ]:
        if dpg.does_item_exist(plot_tag):
            dpg.bind_item_theme(plot_tag, apply_dendrogram_theme(state))

    for plot_tag in [
        "mmpa_plot_global",
        "mmpa_plot_subset",
    ]:
        if dpg.does_item_exist(plot_tag):
            dpg.bind_colormap(plot_tag, state["plot_colormaps"][state["colormap_discrete"]])
            dpg.bind_item_theme(plot_tag, apply_line_chart_theme(state))

    for prog_bar_tag in [
        "rgd_progress_bar"
        ]:
        if dpg.does_item_exist(prog_bar_tag):
            dpg.bind_item_theme(prog_bar_tag, apply_progress_bar_theme(state))

    if dpg.does_item_exist("custom_style_editor"):
        with dpg.theme() as style_editor_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, state["themes"][state["theme_name"]]["Main Background"], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, state["themes"][state["theme_name"]]["Secondary Background"], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, state["themes"][state["theme_name"]]["Text Color"], category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("custom_style_editor", style_editor_theme)

    try:
        from app.lmm.lmm_input_and_settings import refresh_input_selection_themes
        refresh_input_selection_themes(state)
    except Exception:
        pass
    refresh_overrides(state)
    state["redraw_main_frame_overlay"] = redraw_main_frame_overlay
    redraw_main_frame_overlay(state)
    sync_sketcher_theme(state, request_focus=False)


# Step 2.1.  Apply regular or bold font to specific widget
# -----------------------------------------------------------------------------
# 5. Change font type
# -----------------------------------------------------------------------------
def change_font_type(item_tag: str, font_type: Any, state: dict[str, Any]) -> None:
    """
    Applies a regular or bold font to a specific widget based on the sender's tag.
    
    Args:
        item_tag (Any): Parameter accepted by this routine.
        font_type (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    applied_font = state["applied_font"]

    if font_type == "regular":
        dpg.bind_item_font(item_tag, applied_font)
        state["regular_texts"].add(item_tag)

    elif font_type == "bold":
        dpg.bind_item_font(item_tag, f"{applied_font} Bold")
        state["bold_texts"].add(item_tag)


# -----------------------------------------------------------------------------
# 6. Custom style editor
# -----------------------------------------------------------------------------
def custom_style_editor(state: dict[str, Any]) -> None:
    """
    Build the custom style editor window.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    with dpg.window(label="Custom Style Editor", tag="custom_style_editor", autosize=True,
                    no_scrollbar=True, no_resize=True, no_collapse=True,
                    pos=((state["main_win_width"] / 2) - 175, state["main_win_y"]),
                    on_close=lambda: dpg.hide_item("custom_style_editor")):   
        with dpg.child_window(border=True, width=350, auto_resize_y=True):
            if dpg.does_item_exist("custom_style_editor"):
                with dpg.theme() as style_editor_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, state["themes"][state["theme_name"]]["Main Background"], category=dpg.mvThemeCat_Core)
                        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, state["themes"][state["theme_name"]]["Secondary Background"], category=dpg.mvThemeCat_Core)
                        dpg.add_theme_color(dpg.mvThemeCol_Text, state["themes"][state["theme_name"]]["Text Color"], category=dpg.mvThemeCat_Core)
                dpg.bind_item_theme("custom_style_editor", style_editor_theme)


            def change_font(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
                """
                Apply a different font immediately and persist the choice.
                
                Args:
                    sender (Any): Input accepted by this routine.
                    app_data (Any): Input accepted by this routine.
                    state (dict[str, Any]): Input accepted by this routine.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                dpg.bind_font(None)
                time.sleep(0.01)  # short delay to avoid flicker
                dpg.bind_font(app_data)

                state["settings"]["font"] = app_data
                with open(state["settings_file"], "w") as f:
                    json.dump(state["settings"], f, indent=4)

                for item_tag in state["bold_texts"]:
                    dpg.bind_item_font(item_tag, f"{app_data} Bold")
                
                for item_tag in state["regular_texts"]:
                    dpg.bind_item_font(item_tag, app_data)
                
                    

            def change_font_scale(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
                """
                Adjust the global font scale (zoom) and persist it.
                
                Args:
                    sender (Any): Input accepted by this routine.
                    app_data (Any): Input accepted by this routine.
                    state (dict[str, Any]): Input accepted by this routine.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                font_scale = app_data
                dpg.set_global_font_scale(font_scale)

                state["settings"]["font_scale"] = font_scale
                _save_json_file(state["settings_file"], state["settings"])

            def change_tab_button_size(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
                """
                Adjust top navigation button size in real time and persist it.
                """
                display_size = int(round(float(app_data)))
                display_size = max(1, min(state["max_tab_button_size"], display_size))
                button_size = display_size + 20

                state["tab_button_size"] = button_size
                state["settings"]["tab_button_size"] = button_size
                _save_json_file(state["settings_file"], state["settings"])

                if callable(state.get("refresh_top_nav_layout")):
                    state["refresh_top_nav_layout"]()
                if callable(state.get("refresh_top_nav_icons")):
                    state["refresh_top_nav_icons"]()

            def change_tab_icon_mode(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
                """
                Toggle between icon-based and textual top navigation and persist it.
                """
                show_tab_icons = bool(app_data)
                state["show_tab_icons"] = show_tab_icons
                state["settings"]["show_tab_icons"] = show_tab_icons
                _save_json_file(state["settings_file"], state["settings"])

                if callable(state.get("apply_top_nav_mode")):
                    state["apply_top_nav_mode"]()
                if dpg.does_item_exist("tab_button_size_group"):
                    dpg.configure_item("tab_button_size_group", show=show_tab_icons)
                

            FONT_TAGS = ["Arial", "Arimo", "DejaVu Sans", "Ubuntu", "FiraCode (Mono)"]
            current_font = state["settings"]["font"]
            current_font_scale = float(state["settings"]["font_scale"])
            current_tab_button_size = int(state["settings"].get("tab_button_size", state.get("tab_button_size", 30)))
            current_tab_button_display_size = max(1, min(state["max_tab_button_size"], current_tab_button_size - 20))
            current_show_tab_icons = bool(state["settings"].get("show_tab_icons", True))

            with dpg.group():

                dpg.add_text("Tabs settings")
                change_font_type(dpg.last_item(), "bold", state)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Icon tabs:")
                    dpg.add_checkbox(
                        tag="tab_icons_selector",
                        default_value=current_show_tab_icons,
                        callback=change_tab_icon_mode,
                        user_data=state,
                    )
                with dpg.group(horizontal=True, tag="tab_button_size_group", show=current_show_tab_icons):
                    dpg.add_text("Tab button size:")
                    dpg.add_slider_int(
                        tag="tab_button_size_selector",
                        width=-1,
                        default_value=current_tab_button_display_size,
                        min_value=1,
                        max_value=state["max_tab_button_size"],
                        callback=change_tab_button_size,
                        user_data=state,
                    )


                dpg.add_text("Font settings")
                change_font_type(dpg.last_item(), "bold", state)
   
                with dpg.group(horizontal=True):
                    dpg.add_text("Font:")
                    dpg.add_combo(items=FONT_TAGS, tag="font_selector", width=-1,
                                default_value=current_font, callback=change_font, user_data=state)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Scale:")
                    dpg.add_slider_float(tag="font_scale_selector", width=-1,
                                    default_value=current_font_scale, min_value=state["min_font_scale"], max_value=state["max_font_scale"], format="%.1f",
                                    callback=change_font_scale, user_data=state)


            dpg.add_spacer(height=state["win_spacer"])

            dpg.add_text("Themes")
            change_font_type(dpg.last_item(), "bold", state)

            def _color_picker_configs(sender: Any, value: Any, user_data: Any) -> None:
                """
                Update colour picker configuration (wheel/bar mode) dynamically.
                
                Args:
                    sender (Any): Input accepted by this routine.
                    value (Any): Input accepted by this routine.
                    user_data (Any): Input accepted by this routine.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                _old_config = dpg.get_item_configuration(user_data)

                picker_mode = _old_config["picker_mode"]
                alpha_preview = _old_config["alpha_preview"]
                display_type = _old_config["display_type"]
                input_mode = _old_config["input_mode"]

                # picker_mode
                if value == "mvColorPicker_bar":
                    picker_mode=dpg.mvColorPicker_bar
                elif value == "mvColorPicker_wheel":
                    picker_mode = dpg.mvColorPicker_wheel

                dpg.configure_item(user_data, 
                                    picker_mode=picker_mode, 
                                    alpha_preview=alpha_preview,
                                    display_type=display_type,
                                    input_mode=input_mode
                                    )


            def apply_selected_theme(sender: Any, app_data: Any, user_data: Any) -> None:
                """
                Apply the theme selected in the combo box.
                """
                apply_theme_callback(sender, app_data, state)

            def _show_theme_editor_popup(title: str, message: str) -> None:
                """
                Show a small modal popup used for validation messages.
                """
                if dpg.does_item_exist("theme_editor_message_popup"):
                    dpg.delete_item("theme_editor_message_popup")
                with dpg.window(
                    tag="theme_editor_message_popup",
                    label=title,
                    modal=True,
                    no_resize=True,
                    no_collapse=True,
                    autosize=True,
                ):
                    dpg.add_text(message, wrap=360)
                    dpg.add_spacer(height=8)
                    dpg.add_button(label="OK", width=80, callback=lambda: dpg.delete_item("theme_editor_message_popup"))

            def _show_theme_overwrite_popup(theme_name_value: str, on_confirm: Any) -> None:
                """
                Ask confirmation before overwriting an existing custom theme.
                """
                if dpg.does_item_exist("theme_editor_overwrite_popup"):
                    dpg.delete_item("theme_editor_overwrite_popup")
                with dpg.window(
                    tag="theme_editor_overwrite_popup",
                    label="Overwrite Theme",
                    modal=True,
                    no_resize=True,
                    no_collapse=True,
                    autosize=True,
                ):
                    dpg.add_text(
                        f"The custom theme '{theme_name_value}' already exists.\nDo you want to overwrite it?",
                        wrap=360,
                    )
                    dpg.add_spacer(height=8)
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="Overwrite",
                            width=100,
                            callback=lambda: (
                                dpg.delete_item("theme_editor_overwrite_popup"),
                                on_confirm(),
                            ),
                        )
                        dpg.add_button(
                            label="Cancel",
                            width=80,
                            callback=lambda: dpg.delete_item("theme_editor_overwrite_popup"),
                        )

            def _build_theme_editor_controls(draft_theme: dict[str, Any], prefix: str) -> Any:
                """
                Build the colour and slider controls for a draft theme.
                """
                left_keys = [
                    "Main Background", "Secondary Background", "Menu Bar Background",
                    "Title Bar Background", "Tabs Color", "Tabs Hovered", "Tabs Active", "Button Color",
                    "Button Hovered", "Button Active", "Checkmark Color", "Slider Grab",
                    "Scrollbar Background", "Scrollbar Grab",
                ]
                right_color_keys = [
                    "Frame Background", "Text Color", "Table Header", "Border Color", "Border Shadow", "Plot Background",
                ]
                slider_keys = ["Frame Border Size", "Window rounding", "Frame rounding", "Tab rounding"]

                def _refresh_theme_preview() -> None:
                    """
                    Rebuild the preview themes so edits are visible immediately.
                    """
                    preview_theme_tag = f"{prefix}_preview_theme"
                    preview_child_theme_tag = f"{prefix}_preview_child_theme"
                    preview_window_theme_tag = f"{prefix}_preview_window_theme"
                    preview_menu_theme_tag = f"{prefix}_preview_menu_theme"
                    preview_menu_button_theme_tag = f"{prefix}_preview_menu_button_theme"
                    preview_table_theme_tag = f"{prefix}_preview_table_theme"
                    preview_plot_theme_tag = f"{prefix}_preview_plot_theme"

                    _recreate_theme(preview_theme_tag)
                    with dpg.theme(tag=preview_theme_tag):
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)

                        with dpg.theme_component(dpg.mvText):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)

                        with dpg.theme_component(dpg.mvButton):
                            dpg.add_theme_color(dpg.mvThemeCol_Button, draft_theme["Button Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, draft_theme["Button Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, draft_theme["Button Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(0, int(draft_theme["Frame rounding"])))
                            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, int(draft_theme["Frame Border Size"]))

                        with dpg.theme_component(dpg.mvInputText):
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, draft_theme["Frame Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, draft_theme["Border Shadow"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(0, int(draft_theme["Frame rounding"])))
                            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, int(draft_theme["Frame Border Size"]))
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)

                        with dpg.theme_component(dpg.mvCombo):
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, draft_theme["Frame Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Button, draft_theme["Button Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, draft_theme["Button Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, draft_theme["Button Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(0, int(draft_theme["Frame rounding"])))
                            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, int(draft_theme["Frame Border Size"]))
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)

                        with dpg.theme_component(dpg.mvCheckbox):
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, draft_theme["Frame Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, draft_theme["Checkmark Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(0, int(draft_theme["Frame rounding"])))
                            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, int(draft_theme["Frame Border Size"]))

                        with dpg.theme_component(dpg.mvRadioButton):
                            dpg.add_theme_color(dpg.mvThemeCol_Button, draft_theme["Button Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, draft_theme["Button Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, draft_theme["Button Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, draft_theme["Checkmark Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(0, int(draft_theme["Frame rounding"])))
                            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, int(draft_theme["Frame Border Size"]))
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)

                        with dpg.theme_component(dpg.mvSliderInt):
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, draft_theme["Frame Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, draft_theme["Slider Grab"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, draft_theme["Button Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(0, int(draft_theme["Frame rounding"])))
                            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, int(draft_theme["Frame Border Size"]))

                        with dpg.theme_component(dpg.mvTab):
                            dpg.add_theme_color(dpg.mvThemeCol_Tab, draft_theme["Tabs Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TabActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, max(0, int(draft_theme["Tab rounding"])))
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)

                        with dpg.theme_component(dpg.mvTabBar):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Tab, draft_theme["Tabs Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TabActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, max(0, int(draft_theme["Tab rounding"])))

                        with dpg.theme_component(getattr(dpg, "mvTabButton", dpg.mvButton)):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Button, draft_theme["Tabs Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Tab, draft_theme["Tabs Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TabActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, max(0, int(draft_theme["Tab rounding"])))

                        with dpg.theme_component(dpg.mvCollapsingHeader):
                            dpg.add_theme_color(dpg.mvThemeCol_Header, draft_theme["Frame Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, draft_theme["Border Shadow"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(0, int(draft_theme["Frame rounding"])))
                            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, int(draft_theme["Frame Border Size"]))

                    _recreate_theme(preview_child_theme_tag)
                    with dpg.theme(tag=preview_child_theme_tag):
                        with dpg.theme_component(dpg.mvChildWindow):
                            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, draft_theme["Secondary Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, draft_theme["Border Shadow"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, draft_theme["Scrollbar Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, draft_theme["Scrollbar Grab"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, draft_theme["Button Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, draft_theme["Button Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, max(0, int(draft_theme["Window rounding"])))
                            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, int(draft_theme["Frame Border Size"]))

                    _recreate_theme(preview_window_theme_tag)
                    with dpg.theme(tag=preview_window_theme_tag):
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)
                        with dpg.theme_component(dpg.mvWindowAppItem):
                            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, draft_theme["Main Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, draft_theme["Title Bar Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, draft_theme["Title Bar Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, draft_theme["Title Bar Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, draft_theme["Border Shadow"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, max(0, int(draft_theme["Window rounding"])))
                            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, int(draft_theme["Frame Border Size"]))
                            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8)
                    _recreate_theme(preview_menu_theme_tag)
                    with dpg.theme(tag=preview_menu_theme_tag):
                        with dpg.theme_component(dpg.mvChildWindow):
                            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, draft_theme["Menu Bar Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, draft_theme["Border Shadow"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, int(draft_theme["Frame Border Size"]))
                            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0)
                            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 6, 4)

                    _recreate_theme(preview_menu_button_theme_tag)
                    with dpg.theme(tag=preview_menu_button_theme_tag):
                        with dpg.theme_component(dpg.mvButton):
                            transparent = (0, 0, 0, 0)
                            dpg.add_theme_color(dpg.mvThemeCol_Button, transparent, category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, transparent, category=dpg.mvThemeCat_Core)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0)
                            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 4)

                    _recreate_theme(preview_table_theme_tag)
                    with dpg.theme(tag=preview_table_theme_tag):
                        with dpg.theme_component(dpg.mvTable):
                            dpg.add_theme_color(dpg.mvThemeCol_Header, draft_theme["Table Header"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, draft_theme["Table Header"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, draft_theme["Tabs Hovered"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, draft_theme["Tabs Active"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Border, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, draft_theme["Border Shadow"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, draft_theme["Border Color"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, draft_theme["Secondary Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, draft_theme["Main Background"], category=dpg.mvThemeCat_Core)
                            dpg.add_theme_color(dpg.mvThemeCol_Text, draft_theme["Text Color"], category=dpg.mvThemeCat_Core)

                    _recreate_theme(preview_plot_theme_tag)
                    with dpg.theme(tag=preview_plot_theme_tag):
                        with dpg.theme_component(dpg.mvPlot):
                            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, draft_theme["Plot Background"], category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, draft_theme["Border Color"], category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, draft_theme["Text Color"], category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_AxisText, draft_theme["Text Color"], category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_LegendText, draft_theme["Text Color"], category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, draft_theme["Secondary Background"], category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder, draft_theme["Border Color"], category=dpg.mvThemeCat_Plots)

                    item_theme_map = {
                        f"{prefix}_preview_window": preview_window_theme_tag,
                        f"{prefix}_preview_child": preview_child_theme_tag,
                        f"{prefix}_preview_text": preview_theme_tag,
                        f"{prefix}_preview_button": preview_theme_tag,
                        f"{prefix}_preview_input": preview_theme_tag,
                        f"{prefix}_preview_combo": preview_theme_tag,
                        f"{prefix}_preview_checkbox": preview_theme_tag,
                        f"{prefix}_preview_radio": preview_theme_tag,
                        f"{prefix}_preview_slider": preview_theme_tag,
                        f"{prefix}_preview_tabs": preview_theme_tag,
                        f"{prefix}_preview_tab_1": preview_theme_tag,
                        f"{prefix}_preview_tab_2": preview_theme_tag,
                        f"{prefix}_preview_header": preview_theme_tag,
                        f"{prefix}_preview_menu_bar": preview_menu_theme_tag,
                        f"{prefix}_preview_menu_file": preview_menu_theme_tag,
                        f"{prefix}_preview_menu_view": preview_menu_theme_tag,
                        f"{prefix}_preview_menu_help": preview_menu_theme_tag,
                        f"{prefix}_preview_table": preview_table_theme_tag,
                        f"{prefix}_preview_table_col_a": preview_table_theme_tag,
                        f"{prefix}_preview_table_col_b": preview_table_theme_tag,
                        f"{prefix}_preview_table_cell_1": preview_theme_tag,
                        f"{prefix}_preview_table_cell_2": preview_theme_tag,
                        f"{prefix}_preview_plot": preview_plot_theme_tag,
                    }
                    for item_tag, theme_tag in item_theme_map.items():
                        if dpg.does_item_exist(item_tag):
                            dpg.bind_item_theme(item_tag, theme_tag)

                def _update_draft_color(sender: Any, app_data: Any, user_data: Any) -> None:
                    draft_theme[user_data] = tuple(int(round(c * 255)) for c in app_data)
                    _refresh_theme_preview()

                def _update_draft_slider(sender: Any, app_data: Any, user_data: Any) -> None:
                    draft_theme[user_data] = int(app_data)
                    _refresh_theme_preview()

                with dpg.group(horizontal=True):
                    with dpg.group(width=-0.5):
                        for key in left_keys:
                            value = draft_theme[key]
                            tag = f"{prefix}_{key}_button"
                            dpg.add_button(label=key, tag=tag, width=-0.5)
                            with dpg.popup(tag, mousebutton=dpg.mvMouseButton_Left):
                                picker_id = dpg.add_color_picker(
                                    default_value=value,
                                    width=200,
                                    alpha_bar=True,
                                    alpha_preview=True,
                                    no_side_preview=True,
                                    user_data=key,
                                    callback=_update_draft_color,
                                )
                                dpg.add_radio_button(
                                    ("mvColorPicker_bar", "mvColorPicker_wheel"),
                                    callback=_color_picker_configs,
                                    user_data=picker_id,
                                    horizontal=False,
                                )

                    with dpg.group(width=-0.5):
                        for key in right_color_keys:
                            value = draft_theme[key]
                            tag = f"{prefix}_{key}_button"
                            dpg.add_button(label=key, tag=tag, width=-0.5)
                            with dpg.popup(tag, mousebutton=dpg.mvMouseButton_Left):
                                picker_id = dpg.add_color_picker(
                                    default_value=value,
                                    width=200,
                                    alpha_bar=True,
                                    alpha_preview=True,
                                    no_side_preview=True,
                                    user_data=key,
                                    callback=_update_draft_color,
                                )
                                dpg.add_radio_button(
                                    ("mvColorPicker_bar", "mvColorPicker_wheel"),
                                    callback=_color_picker_configs,
                                    user_data=picker_id,
                                    horizontal=False,
                                )

                        for key in slider_keys:
                            value = draft_theme[key]
                            max_value = 3 if key == "Frame Border Size" else 10
                            min_value = 1 if key == "Frame Border Size" else 0
                            dpg.add_text(f"{key}:")
                            dpg.add_slider_int(
                                tag=f"{prefix}_{key}_slider",
                                width=150,
                                default_value=value,
                                min_value=min_value,
                                max_value=max_value,
                                callback=_update_draft_slider,
                                user_data=key,
                            )

                return _refresh_theme_preview

            def _open_new_theme_window() -> None:
                """
                Open the modal editor used to create or overwrite a named theme.
                """
                editor_tag = "new_theme_window"
                preview_tag = "new_theme_preview_window"

                def _destroy_new_theme_pair() -> None:
                    for item_tag in (preview_tag, editor_tag):
                        if dpg.does_item_exist(item_tag):
                            dpg.delete_item(item_tag)

                if dpg.does_item_exist(editor_tag) and dpg.does_item_exist(preview_tag):
                    dpg.show_item(editor_tag)
                    dpg.show_item(preview_tag)
                    try:
                        dpg.focus_item(preview_tag)
                        dpg.focus_item(editor_tag)
                    except Exception:
                        pass
                    return

                _destroy_new_theme_pair()
                current_theme_name = state["theme_name"]
                if dpg.does_item_exist("theme_selector"):
                    selected_theme_name = dpg.get_value("theme_selector")
                    if selected_theme_name in state["themes"]:
                        current_theme_name = selected_theme_name
                draft_theme = copy.deepcopy(state["themes"][current_theme_name])

                def _close_new_theme_pair() -> None:
                    next_frame = dpg.get_frame_count() + 1
                    dpg.set_frame_callback(next_frame, lambda: _destroy_new_theme_pair())

                def _save_new_theme() -> None:
                    theme_name_value = dpg.get_value("new_theme_name_input").strip()
                    if not theme_name_value:
                        _show_theme_editor_popup("Theme Name Required", "Insert a name before saving the theme.")
                        return
                    if theme_name_value in state["default_theme_names"]:
                        _show_theme_editor_popup(
                            "Reserved Theme Name",
                            f"'{theme_name_value}' is a default theme name and cannot be overwritten.",
                        )
                        return

                    def _persist_new_theme() -> None:
                        state["themes_store"].setdefault("default_themes", {})
                        state["themes_store"].setdefault("custom_themes", {})
                        state["themes_store"]["custom_themes"][theme_name_value] = copy.deepcopy(draft_theme)
                        state["themes"][theme_name_value] = copy.deepcopy(draft_theme)
                        _save_json_file(state["themes_file"], state["themes_store"])

                        if dpg.does_item_exist("theme_selector"):
                            dpg.configure_item("theme_selector", items=_get_theme_selector_items(state))
                            dpg.set_value("theme_selector", theme_name_value)

                        apply_theme_callback("change_theme", theme_name_value, state)
                        _close_new_theme_pair()

                    existing_custom_themes = state["themes_store"].get("custom_themes", {})
                    if theme_name_value in existing_custom_themes:
                        _show_theme_overwrite_popup(theme_name_value, _persist_new_theme)
                        return

                    _persist_new_theme()

                with dpg.window(
                    tag=editor_tag,
                    label="New Theme",
                    modal=False,
                    no_resize=False,
                    no_collapse=True,
                    width=560,
                    height=720,
                    on_close=lambda: _close_new_theme_pair(),
                ):
                    dpg.add_text("Theme name:")
                    dpg.add_input_text(tag="new_theme_name_input", width=-1, hint="Insert theme name")
                    with dpg.child_window(width=-1, auto_resize_y=True, border=False):
                        refresh_preview = _build_theme_editor_controls(draft_theme, "new_theme")
                    refresh_preview()
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Save theme", callback=_save_new_theme, width=120)
                        dpg.add_button(label="Cancel", callback=lambda: _close_new_theme_pair(), width=100)

                with dpg.window(
                    tag=preview_tag,
                    label="Window title",
                    modal=False,
                    no_resize=True,
                    no_collapse=True,
                    no_bring_to_front_on_focus=False,
                    width=390,
                    height=590,
                    pos=(dpg.get_item_pos(editor_tag)[0] + 575, dpg.get_item_pos(editor_tag)[1]),
                    no_close=True,
                    no_saved_settings=True,
                ):
                    with dpg.child_window(
                        tag="new_theme_preview_menu_bar",
                        height=30,
                        width=-1,
                        border=False,
                        no_scrollbar=True,
                    ):
                        with dpg.group(horizontal=True):
                            dpg.add_text("File", tag="new_theme_preview_menu_file")
                            dpg.add_spacer(width=12)
                            dpg.add_text("View", tag="new_theme_preview_menu_view")
                            dpg.add_spacer(width=12)
                            dpg.add_text("Help", tag="new_theme_preview_menu_help")
                    with dpg.child_window(tag="new_theme_preview_child", width=-1, height=-1, border=True):
                        dpg.add_text("Preview text", tag="new_theme_preview_text")
                        dpg.add_button(label="Preview button", tag="new_theme_preview_button", width=150)
                        dpg.add_input_text(tag="new_theme_preview_input", default_value="Preview input", width=-1)
                        dpg.add_combo(
                            items=["First option", "Second option", "Third option"],
                            default_value="First option",
                            tag="new_theme_preview_combo",
                            width=-1,
                        )
                        dpg.add_checkbox(label="Preview checkbox", default_value=True, tag="new_theme_preview_checkbox")
                        dpg.add_radio_button(
                            items=["A", "B", "C"],
                            default_value="A",
                            horizontal=True,
                            tag="new_theme_preview_radio",
                        )
                        dpg.add_slider_int(
                            tag="new_theme_preview_slider",
                            default_value=5,
                            min_value=0,
                            max_value=10,
                            width=-1,
                        )
                        with dpg.tab_bar(tag="new_theme_preview_tabs"):
                            with dpg.tab(label="Tab 1", tag="new_theme_preview_tab_1"):
                                dpg.add_text("Tab 1 content")
                            with dpg.tab(label="Tab 2", tag="new_theme_preview_tab_2"):
                                dpg.add_text("Tab 2 content")
                        with dpg.collapsing_header(label="Preview header", default_open=True, tag="new_theme_preview_header"):
                            dpg.add_button(label="Header button", width=140)
                        with dpg.table(
                            tag="new_theme_preview_table",
                            header_row=True,
                            borders_innerH=True,
                            borders_outerH=True,
                            borders_innerV=True,
                            borders_outerV=True,
                            row_background=True,
                        ):
                            dpg.add_table_column(label="A", tag="new_theme_preview_table_col_a")
                            dpg.add_table_column(label="B", tag="new_theme_preview_table_col_b")
                            with dpg.table_row():
                                dpg.add_text("Cell 1", tag="new_theme_preview_table_cell_1")
                                dpg.add_text("Cell 2", tag="new_theme_preview_table_cell_2")
                        with dpg.plot(tag="new_theme_preview_plot", width=-1, height=160, no_menus=True):
                            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="x")
                            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="y")
                            dpg.add_line_series([0, 1, 2, 3, 4], [1, 3, 2, 4, 3], parent=y_axis)
                        for idx in range(10):
                            dpg.add_text(" ", color=(0, 0, 0, 0), tag=f"new_theme_preview_spacer_{idx}")

                try:
                    dpg.focus_item(preview_tag)
                    dpg.focus_item(editor_tag)
                except Exception:
                    pass

            dpg.add_separator(label="Colormaps")
            colormaps = list(state["colormaps"].keys())
            with dpg.group(horizontal=True):
                dpg.add_text("Continuous colormap:")
                dpg.add_combo(items=colormaps, tag="colormap_selector", width=-1,
                            default_value=state["colormap_continuous"], callback=apply_colormap, user_data=state)
            plot_colormaps = list(state["plot_colormaps"].keys())
            with dpg.group(horizontal=True):
                dpg.add_text("Discrete colormap:")
                dpg.add_combo(
                    items=plot_colormaps,
                    tag="plot_colormap_selector",
                    width=-1,
                    default_value=state["colormap_discrete"],
                    callback=apply_plot_colormap,
                    user_data=state,
                )

            dpg.add_separator(label="Apply theme")

            theme_name = state["settings"]["theme_name"]
            themes = _get_theme_selector_items(state)
            
            with dpg.group(horizontal=True):
                dpg.add_text("Select theme:")
                dpg.add_combo(items=themes, tag="theme_selector", width=-1,
                                default_value=theme_name, callback=apply_selected_theme)

            dpg.add_separator(label="Theme editor")
            dpg.add_button(label="New theme", width=-1, callback=lambda: _open_new_theme_window())


# -----------------------------------------------------------------------------
# 7. Apply outer child theme
# -----------------------------------------------------------------------------
def apply_outer_child_theme() -> Any:
    """
    Creates a theme with a solid background for outer child windows.
    
    Args:
        None.
    
    Returns:
        Any: Value returned by the routine.
    """

    with dpg.theme() as outer_child_theme:
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 5, 0)
    return outer_child_theme


# -----------------------------------------------------------------------------
# 8. Apply inner child theme
# -----------------------------------------------------------------------------
def apply_inner_child_theme() -> Any:
    """
    Creates a theme with a transparent background for inner child windows.
    
    Args:
        None.
    
    Returns:
        Any: Value returned by the routine.
    """

    with dpg.theme() as inner_child_theme:
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
    return inner_child_theme


# -----------------------------------------------------------------------------
# 9. Apply input text theme
# -----------------------------------------------------------------------------
def apply_input_text_theme() -> Any:
    """
    Creates a theme for input text widgets with transparent background and borders.
    
    Args:
        None.
    
    Returns:
        Any: Value returned by the routine.
    """

    with dpg.theme() as input_text_theme:
        with dpg.theme_component(dpg.mvInputText):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
    return input_text_theme


# -----------------------------------------------------------------------------
# 10. Apply bordered input text theme
# -----------------------------------------------------------------------------
def apply_bordered_input_text_theme(state: dict[str, Any]) -> Any:
    """
    Creates a theme for input text widgets with borders.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    theme_dict = state["themes"][state["theme_name"]]
    with dpg.theme() as input_bordered_text_theme:
        with dpg.theme_component(dpg.mvInputText):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, theme_dict["Frame Background"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, theme_dict["Tabs Hovered"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, theme_dict["Tabs Active"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 7)
            dpg.add_theme_color(dpg.mvThemeCol_Border, theme_dict["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, theme_dict["Border Shadow"], category=dpg.mvThemeCat_Core)

    return input_bordered_text_theme


# -----------------------------------------------------------------------------
# 11. Apply image button theme
# -----------------------------------------------------------------------------
def apply_image_button_theme(state: dict[str, Any]) -> Any:
    """
    Theme for image buttons (hover/active colours, borders and rounding).
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    return _ensure_image_button_theme(state)


# -----------------------------------------------------------------------------
# 12. Apply image theme
# -----------------------------------------------------------------------------
def apply_image_theme(state: dict[str, Any]) -> Any:
    """
    Theme for generic images (border and rounding).
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    with dpg.theme() as image_theme:
        with dpg.theme_component(dpg.mvImage):
            dpg.add_theme_color(dpg.mvThemeCol_Border, state["themes"][state["theme_name"]]["Border Color"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, state["themes"][state["theme_name"]]["Border Shadow"], category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, state["themes"][state["theme_name"]]["Frame rounding"])
    return image_theme


# -----------------------------------------------------------------------------
# 13. Apply colormap theme
# -----------------------------------------------------------------------------
def apply_colormap_theme(state: dict[str, Any]) -> Any:
    """
    Theme for colourmap scales with transparent frame and no border.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    with dpg.theme() as colorscale_no_background_theme:
        with dpg.theme_component(dpg.mvColorMapScale):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
    return colorscale_no_background_theme


# -----------------------------------------------------------------------------
# 14. Apply pie chart theme
# -----------------------------------------------------------------------------
def apply_pie_chart_theme(state: dict[str, Any]) -> Any:
    """
    Plot theme suited for pie charts (transparent plot bg/border, themed legend).
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    theme_dict = state["themes"][state["theme_name"]]
    with dpg.theme() as pie_chart_theme:
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 0, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotBorderSize, 0)
            dpg.add_theme_color(dpg.mvPlotCol_LegendText, theme_dict["Text Color"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder, theme_dict["Border Color"], category=dpg.mvThemeCat_Plots)
    return pie_chart_theme


# -----------------------------------------------------------------------------
# 15. Apply dendrogram theme
# -----------------------------------------------------------------------------
def apply_dendrogram_theme(state: dict[str, Any]) -> Any:
    """
    Plot theme tailored for dendrogram visualisations.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    theme_dict = state["themes"][state["theme_name"]]
    with dpg.theme() as dendrogram_theme:
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, theme_dict["Main Background"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (0, 0, 0, 0))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 0, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotBorderSize, 0)
            dpg.add_theme_color(dpg.mvPlotCol_LegendText, theme_dict["Text Color"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder, theme_dict["Border Color"], category=dpg.mvThemeCat_Plots)
    return dendrogram_theme


# -----------------------------------------------------------------------------
# 16. Apply enrich plot theme
# -----------------------------------------------------------------------------
def apply_enrich_plot_theme(state: dict[str, Any]) -> Any:
    """
    Execute the apply enrich plot theme routine.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    theme_dict = state["themes"][state["theme_name"]]
    with dpg.theme() as enrich_plot_theme:
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, theme_dict["Main Background"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, theme_dict["Text Color"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, theme_dict["Text Color"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotBorderSize, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 20, 0, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MinorTickSize, 0, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MinorTickLen, 10, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MajorTickSize, 0, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MajorTickLen, 25, category=dpg.mvThemeCat_Plots)
        return enrich_plot_theme


# -----------------------------------------------------------------------------
# 17. Apply line chart theme
# -----------------------------------------------------------------------------
def apply_line_chart_theme(state: dict[str, Any]) -> Any:
    """
    General theme for line charts.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    theme_dict = state["themes"][state["theme_name"]]

    bg = theme_dict["Plot Background"]
    if len(bg) == 3:
        inv_rgb = tuple(255 - c for c in bg)
    else:
        inv_rgb = tuple(255 - c for c in bg[:3])
    inv_rgba = (*inv_rgb, 128)

    with dpg.theme() as plot_theme:
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, theme_dict["Plot Background"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (0, 0, 0, 0))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 0, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotBorderSize, 0)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, inv_rgba, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendText, theme_dict["Text Color"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder, theme_dict["Border Color"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
    return plot_theme


# -----------------------------------------------------------------------------
# 18. Apply mmpa network theme
# -----------------------------------------------------------------------------
def apply_mmpa_network_theme() -> Any:
    """
    Plot theme tailored for MMPA network map visualisations.
    
    Args:
        None.
    
    Returns:
        Any: Value produced by the routine.
    """
    with dpg.theme() as mmpa_network_theme:
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (255, 255, 255, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_InlayText, (0, 0, 0, 255), category=dpg.mvThemeCat_Plots)
    return mmpa_network_theme


# -----------------------------------------------------------------------------
# 19. Apply plot theme
# -----------------------------------------------------------------------------
def apply_plot_theme(state: dict[str, Any]) -> Any:
    """
    General theme for plots.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    theme_dict = state["themes"][state["theme_name"]]

    with dpg.theme() as plot_theme:
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, theme_dict["Plot Background"]  , category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (0, 0, 0, 0))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 0, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotBorderSize, 0)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendText, theme_dict["Text Color"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder, theme_dict["Border Color"], category=dpg.mvThemeCat_Plots)
    return plot_theme


# -----------------------------------------------------------------------------
# 20. Apply boxplot theme
# -----------------------------------------------------------------------------
def apply_boxplot_theme(state: dict[str, Any]) -> Any:
    """
    General theme for boxplot and Rgroup table.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    theme_dict = state["themes"][state["theme_name"]]

    with dpg.theme() as boxplot_theme:
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (255, 255, 255, 255)  , category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (0, 0, 0, 0))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 0, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotBorderSize, 0)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, (60, 60, 60, 128), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendText, theme_dict["Text Color"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder, theme_dict["Border Color"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_InlayText, (0, 0, 0, 255), category=dpg.mvThemeCat_Plots)
    return boxplot_theme


# -----------------------------------------------------------------------------
# 21. Apply infinite line theme
# -----------------------------------------------------------------------------
def apply_infinite_line_theme() -> Any:
    """
    Theme for infinite line series (highlighted red line).
    
    Args:
        None.
    
    Returns:
        Any: Value produced by the routine.
    """
    with dpg.theme() as inf_line_theme:
        with dpg.theme_component(dpg.mvInfLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 0, 0, 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
    return inf_line_theme


# -----------------------------------------------------------------------------
# 22. Apply progress bar theme
# -----------------------------------------------------------------------------
def apply_progress_bar_theme(state: dict[str, Any]) -> Any:
    """
    Theme for the loading bar.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    theme_dict = state["themes"][state["theme_name"]]
    with dpg.theme() as progress_bar_theme:
        with dpg.theme_component(dpg.mvProgressBar):
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, theme_dict["Tabs Active"])
    return progress_bar_theme
from app.utils.app_logger import log_exception
