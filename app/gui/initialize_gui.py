"""
=================
initialize_gui.py
=================

Builds the GUI scaffold: menu bar, tab bar, and all main tabs.
Defines windows, tables, and layout structure for the interface.
Acts as the entry point for constructing every GUI section.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Initialize gui

import dearpygui.dearpygui as dpg
import os
import webbrowser
from typing import Any
from PIL import Image
from app.utils.load_job import handle_selected_directory, load_results
from app.gui.layout import save_json_file
from app.gui.event_log import show_event_log_window
from app.utils.native_dialogs import open_directory_dialog
from app.gui.themes_manager import (
    apply_input_text_theme,
    apply_outer_child_theme,
    change_font_type
)
from app.utils.callbacks import (
    open_manual,
    open_contextual_help,
    change_tab,
    activate_main_tab,
    change_overview_subtab,
    change_similarity_subtab,
    change_r_analysis_subtab,
    change_chemspace_subtab,
    toggle_fullscreen_with_check,
)
from app.analysis.tools.utilities import show_utilities_window
from app.analysis.tools.sketcher_launcher import open_sketcher_window
from app.lmm.lmm_input_and_settings import show_input_selection_window, show_settings_window
from app.utils.slith_minigame import open_slith_window


# -----------------------------------------------------------------------------
# 2. Initialize gui
# -----------------------------------------------------------------------------
def initialize_gui(state: dict[str, Any]) -> None:

    viewport_w, viewport_h = state["design_ref_width"], state["design_ref_height"]
    main_win_y = state["main_win_y"]
    contents_w = viewport_w - 40
    contents_h = viewport_h - main_win_y
    state["main_win_width"] = contents_w
    state["main_win_height"] = contents_h

    
    def _ensure_button_theme(
        tag: str,
        base: tuple[int, int, int, int],
        hover: tuple[int, int, int, int],
        active: tuple[int, int, int, int],
        text: tuple[int, int, int, int] | None = None,
        rounding: int = 8,
    ) -> str:
        """
        Create a reusable button theme only once.

        Args:
            tag (str): Theme tag to create or reuse.
            base (tuple[int, int, int, int]): Default button colour.
            hover (tuple[int, int, int, int]): Hover colour.
            active (tuple[int, int, int, int]): Active colour.
            text (tuple[int, int, int, int] | None, optional): Optional text colour.
            rounding (int, optional): Button frame rounding.

        Returns:
            str: Theme tag ready to bind.
        """
        if dpg.does_item_exist(tag):
            return tag
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, base, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hover, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, active, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                if text is not None:
                    dpg.add_theme_color(dpg.mvThemeCol_Text, text, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, rounding)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 6)
        return tag

    def _ensure_panel_theme(
        tag: str,
        background: tuple[int, int, int, int],
        border: tuple[int, int, int, int],
        rounding: int = 10,
    ) -> str:
        """
        Create a reusable panel theme for child windows.

        Args:
            tag (str): Theme tag to create or reuse.
            background (tuple[int, int, int, int]): Child background colour.
            border (tuple[int, int, int, int]): Border colour.
            rounding (int, optional): Corner rounding radius.

        Returns:
            str: Theme tag ready to bind.
        """
        if dpg.does_item_exist(tag):
            return tag
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, background, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Border, border, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, rounding)
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
        return tag

    def _ensure_transparent_host_theme(tag: str) -> str:
        """
        Create a child-window theme with transparent background and zero padding.
        """
        if dpg.does_item_exist(tag):
            return tag
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 0)
        return tag

    def _ensure_top_nav_table_theme(tag: str) -> str:
        """
        Create a table theme with zero inner cell padding.
        """
        if dpg.does_item_exist(tag):
            return tag
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvTable):
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        return tag

    def _ensure_chemspace_layout_table_theme(tag: str) -> str:
        """
        Create a table theme for ChemSpace layout tables with compact padding.
        """
        if dpg.does_item_exist(tag):
            return tag
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvTable):
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 10, 10, category=dpg.mvThemeCat_Core)
        return tag

    def _ensure_top_nav_column_theme(tag: str, align_x: float) -> str:
        """
        Create a column theme for top navigation columns.
        """
        if dpg.does_item_exist(tag):
            return tag
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvTableColumn):
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, align_x, 0.5, category=dpg.mvThemeCat_Core)
        return tag

    def _ensure_zero_item_gap_theme(tag: str) -> str:
        """
        Create a generic theme with zero horizontal/vertical item spacing.
        """
        if dpg.does_item_exist(tag):
            return tag
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, state["tab_button_spacing"], 0, category=dpg.mvThemeCat_Core)
        return tag

    theme_dict = state["themes"][state["theme_name"]]
    help_button_theme = _ensure_button_theme(
        "help_button_theme",
        theme_dict["Tabs Active"],
        theme_dict["Button Hovered"],
        theme_dict["Button Active"],
        text=theme_dict["Text Color"] if state["theme_name"] in ("Light", "Sun") else (255, 255, 255, 255),
        rounding=max(6, int(theme_dict["Frame rounding"])),
    )
    link_button_theme = _ensure_button_theme(
        "link_button_theme",
        theme_dict["Button Color"],
        theme_dict["Tabs Hovered"],
        theme_dict["Tabs Active"],
        rounding=max(6, int(theme_dict["Frame rounding"])),
    )
    panel_theme = _ensure_panel_theme(
        "core_panel_theme",
        theme_dict["Secondary Background"],
        theme_dict["Border Color"],
        rounding=max(6, int(theme_dict["Frame rounding"])),
    )
    transparent_host_theme = _ensure_transparent_host_theme("transparent_host_theme")
    top_nav_table_theme = _ensure_top_nav_table_theme("top_nav_table_theme")
    top_nav_primary_column_theme = _ensure_top_nav_column_theme("top_nav_primary_column_theme", 0.0)
    top_nav_secondary_column_theme = _ensure_top_nav_column_theme("top_nav_secondary_column_theme", 0.0)
    top_nav_tertiary_column_theme = _ensure_top_nav_column_theme("top_nav_tertiary_column_theme", 1.0)
    zero_item_gap_theme = _ensure_zero_item_gap_theme("zero_item_gap_theme")

    def _show_recent_path_error(message: str) -> None:
        """
        Display a compact modal popup for recent-file related errors.
        """
        if dpg.does_item_exist("recent_files_error_popup"):
            dpg.delete_item("recent_files_error_popup")

        with dpg.window(
            label="Recent Files",
            tag="recent_files_error_popup",
            modal=True,
            show=True,
            autosize=True,
            no_resize=True,
        ):
            dpg.add_text(message, wrap=420)
            dpg.add_spacer(height=max(4, int(state["win_spacer"] / 2)))
            dpg.add_button(
                label="OK",
                width=70,
                callback=lambda: dpg.delete_item("recent_files_error_popup"),
            )
        dpg.bind_item_theme("recent_files_error_popup", apply_input_text_theme)

    def _open_recent_path(path: str) -> None:
        """
        Open a recent file or results directory from the File menu.
        """
        if not isinstance(path, (str, os.PathLike)):
            _show_recent_path_error(
                "The selected recent entry is invalid and cannot be opened."
            )
            return

        normalized_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        if not os.path.exists(normalized_path):
            recent_paths = [
                p for p in state.get("recent_files", [])
                if isinstance(p, str) and os.path.abspath(os.path.expanduser(p)) != normalized_path
            ]
            state["recent_files"] = recent_paths
            recent_files_path = state.get("recent_files_file", "")
            if recent_files_path:
                try:
                    save_json_file(recent_files_path, {"paths": recent_paths})
                except Exception:
                    pass
            refresh_recent_files_menu()
            _show_recent_path_error(
                "The selected recent path no longer exists and has been removed from the list."
            )
            return

        if os.path.isdir(normalized_path):
            load_results(normalized_path, state)
            add_recent_file = state.get("add_recent_file")
            if callable(add_recent_file):
                add_recent_file(normalized_path)
            return

        open_input_file_from_path = state.get("open_input_file_from_path")
        if callable(open_input_file_from_path):
            open_input_file_from_path(normalized_path)
            return

        _show_recent_path_error(
            "The input-file loader is not available yet. Please try again in a moment."
        )

    def refresh_recent_files_menu() -> None:
        """
        Rebuild the Recent Files submenu from the paths stored in state.
        """
        if not dpg.does_item_exist("recent_files_menu"):
            return

        dpg.delete_item("recent_files_menu", children_only=True)
        raw_recent_paths = state.get("recent_files", [])
        recent_paths = [
            p for p in state.get("recent_files", [])
            if isinstance(p, str) and p.strip()
        ]
        state["recent_files"] = recent_paths[:10]
        if recent_paths[:10] != raw_recent_paths:
            recent_files_path = state.get("recent_files_file", "")
            if recent_files_path:
                try:
                    save_json_file(recent_files_path, {"paths": recent_paths[:10]})
                except Exception:
                    pass

        if not recent_paths:
            dpg.add_menu_item(label="No recent files", parent="recent_files_menu", enabled=False)
            return

        for recent_path in recent_paths[:10]:
            item_name = os.path.basename(os.path.normpath(recent_path)) or recent_path
            item_label = f"{item_name} ({recent_path})"
            dpg.add_menu_item(
                label=item_label,
                parent="recent_files_menu",
                user_data=recent_path,
                callback=lambda s, a, u: _open_recent_path(u),
            )

    state["refresh_recent_files_menu"] = refresh_recent_files_menu

    def _make_nav_icon_texture(texture_tag: str, icon_name: str) -> str:
        """
        Create or refresh a square transparent texture from a two-tone icon file.
        """
        size = 48
        theme_dict_local = state["themes"][state["theme_name"]]
        text_color = tuple(theme_dict_local["Text Color"])
        base_rgb = (text_color[0], text_color[1], text_color[2])
        inverse_rgb = (255 - base_rgb[0], 255 - base_rgb[1], 255 - base_rgb[2])
        icon_path = os.path.join(state["Home_dir_path"], "assets", "icons", f"{icon_name}.png")
        if not os.path.exists(icon_path):
            image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        else:
            image = Image.open(icon_path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
            pixels = []
            for r, g, b, a in image.getdata():
                if a == 0:
                    pixels.append((0, 0, 0, 0))
                    continue
                lum = (r + g + b) / (255.0 * 3.0)
                tr = int(round(base_rgb[0] * (1.0 - lum) + inverse_rgb[0] * lum))
                tg = int(round(base_rgb[1] * (1.0 - lum) + inverse_rgb[1] * lum))
                tb = int(round(base_rgb[2] * (1.0 - lum) + inverse_rgb[2] * lum))
                pixels.append((tr, tg, tb, a))
            image.putdata(pixels)

        data = []
        for r, g, b, a in image.getdata():
            data.extend((r / 255.0, g / 255.0, b / 255.0, a / 255.0))

        if dpg.does_item_exist(texture_tag):
            dpg.set_value(texture_tag, data)
        else:
            dpg.add_dynamic_texture(size, size, data, tag=texture_tag, parent="texture_registry")

        return texture_tag

    tab_buttons = [
        ("input_tab", "input_nav_button", "input", "INPUT", True, True, lambda: (dpg.set_value("tab_bar", "input_tab"), activate_main_tab("input_tab", state))),
        ("analysis_tab", "analysis_nav_button", "analysis", "ANALYSIS", True, False, lambda: (dpg.set_value("tab_bar", "analysis_tab"), activate_main_tab("analysis_tab", state))),
        ("overview_tab", "overview_nav_button", "overview", "OVERVIEW", True, False, lambda: (dpg.set_value("tab_bar", "overview_tab"), activate_main_tab("overview_tab", state))),
        ("r_analysis_tab", "r_analysis_nav_button", "r_analysis", "R-ANALYSIS", True, False, lambda: (dpg.set_value("tab_bar", "r_analysis_tab"), activate_main_tab("r_analysis_tab", state))),
        ("similarity_tab", "similarity_nav_button", "similarity", "SIMILARITY", True, False, lambda: (dpg.set_value("tab_bar", "similarity_tab"), activate_main_tab("similarity_tab", state))),
        ("stereo_tab", "stereo_nav_button", "stereo", "STEREO", True, False, lambda: (dpg.set_value("tab_bar", "stereo_tab"), activate_main_tab("stereo_tab", state))),
        ("mmpa_tab", "mmpa_nav_button", "mmpa", "MMPA", True, False, lambda: (dpg.set_value("tab_bar", "mmpa_tab"), activate_main_tab("mmpa_tab", state))),
        ("chemspace_tab", "chemspace_nav_button", "chemspace", "CHEMSPACE", True, False, lambda: (dpg.set_value("tab_bar", "chemspace_tab"), activate_main_tab("chemspace_tab", state))),
        ("notes_popup", "notes_nav_button", "notes", "SAR NOTES", True, False, lambda: (
            dpg.show_item("notes_popup") if dpg.does_item_exist("notes_popup") else None,
            dpg.configure_item("notes_popup", collapsed=False) if dpg.does_item_exist("notes_popup") else None,
            dpg.focus_item("notes_popup") if dpg.does_item_exist("notes_popup") else None,
            dpg.set_value("notes_collapsing_header", True) if dpg.does_item_exist("notes_collapsing_header") else None,
        )),
        ("prediction_tab", "prediction_nav_button", "prediction", "PREDICTION", True, False, lambda: (dpg.set_value("tab_bar", "prediction_tab"), activate_main_tab("prediction_tab", state))),
        ("sketcher_popup", "sketcher_nav_button", "sketcher", "SKETCHER", True, True, lambda: open_sketcher_window(state)),
        ("utilities_tab", "utilities_nav_button", "utilities", "UTILITIES", True, True, lambda: (dpg.set_value("tab_bar", "utilities_tab"), activate_main_tab("utilities_tab", state))),
        ("event_log", "event_log_nav_button", "event_log", "EVENT LOG", True, True, lambda: show_event_log_window(state)),
        ("slith_tab", "slith_nav_button", "slith", "SLITH-MINIGAME", False, False, lambda: (dpg.set_value("tab_bar", "slith_tab"), activate_main_tab("slith_tab", state))),
    ]
    state["top_nav_button_tags"] = [button_tag for _, button_tag, _, _, _, _, _ in tab_buttons]
    state["top_nav_button_enabled"] = {
        button_tag: enabled for _, button_tag, _, _, _, enabled, _ in tab_buttons
    }
    state["show_tab_icons"] = bool(state["settings"].get("show_tab_icons", True))
    state["locked_text_tabs"] = {
        "analysis_tab",
        "overview_tab",
        "r_analysis_tab",
        "similarity_tab",
        "stereo_tab",
        "mmpa_tab",
        "chemspace_tab",
        "prediction_tab",
    }
    state["locked_text_tab_buttons"] = {
        "notes_text_button",
    }

    def _apply_top_nav_mode() -> None:
        """
        Toggle between icon-based and textual top navigation.
        """
        show_icons = bool(state.get("show_tab_icons", True))
        current_tab = state.get("current_tab", "input_tab")
        if dpg.does_item_exist("tab_icons_container"):
            dpg.configure_item("tab_icons_container", show=show_icons)
        if dpg.does_item_exist("tab_text_container"):
            dpg.configure_item("tab_text_container", show=not show_icons)
        if dpg.does_item_exist("tab_bar") and dpg.does_item_exist(current_tab):
            dpg.set_value("tab_bar", current_tab)

    state["apply_top_nav_mode"] = _apply_top_nav_mode

    def _handle_top_nav_press(button_tag: str, callback: Any) -> None:
        """
        Execute a top-nav button callback only when the logical button state is enabled.
        """
        enabled_map = state.get("top_nav_button_enabled", {})
        if not enabled_map.get(button_tag, True):
            return
        if callable(callback):
            callback()

    def _refresh_text_tab_themes() -> None:
        """
        Apply the correct theme to textual tabs depending on lock state.
        """
        locked_text_tabs = state.get("locked_text_tabs", set())
        locked_text_tab_buttons = state.get("locked_text_tab_buttons", set())
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
                dpg.bind_item_theme(
                    item_tag,
                    "locked_tab_item_theme" if item_tag in locked_text_tabs else "tab_item_theme",
                )
        for item_tag in ["sketcher_text_button", "event_log_text_button", "notes_text_button"]:
            if dpg.does_item_exist(item_tag):
                dpg.bind_item_theme(
                    item_tag,
                    "locked_tab_button_theme" if item_tag in locked_text_tab_buttons else "tab_button_theme",
                )
        if dpg.does_item_exist("tab_placeholder_button_1"):
            dpg.bind_item_theme("tab_placeholder_button_1", "invisible_tab_button_theme")
        if dpg.does_item_exist("tab_placeholder_button_2"):
            dpg.bind_item_theme("tab_placeholder_button_2", "invisible_tab_button_theme")

    state["refresh_text_tab_themes"] = _refresh_text_tab_themes

    def _refresh_top_nav_layout() -> None:
        """
        Keep the top navigation row aligned after dynamic visibility changes.
        """
        try:
            button_size = int(state.get("tab_button_size", 30))
            for button_tag in state.get("top_nav_button_tags", []):
                if dpg.does_item_exist(button_tag):
                    dpg.configure_item(button_tag, width=button_size, height=button_size)
            if dpg.does_item_exist("top_nav_table"):
                dpg.configure_item("top_nav_table", width=-1)
        except Exception:
            pass

    def _refresh_top_nav_selection() -> None:
        """
        Rebind navigation button themes keeping a shared inactive look and
        a distinct active look for the selected tab.
        """
        active_tab = state.get("current_tab", "input_tab")
        all_button_tags = [button_tag for _, button_tag, _, _, _, _, _ in tab_buttons]
        enabled_map = state.get("top_nav_button_enabled", {})
        for button_tag in all_button_tags:
            if dpg.does_item_exist(button_tag):
                if enabled_map.get(button_tag, True):
                    dpg.bind_item_theme(button_tag, "top_nav_image_button_theme")
                else:
                    dpg.bind_item_theme(button_tag, "top_nav_image_button_disabled_theme")
        button_map = {
            "input_tab": "input_nav_button",
            "analysis_tab": "analysis_nav_button",
            "overview_tab": "overview_nav_button",
            "r_analysis_tab": "r_analysis_nav_button",
            "similarity_tab": "similarity_nav_button",
            "stereo_tab": "stereo_nav_button",
            "mmpa_tab": "mmpa_nav_button",
            "chemspace_tab": "chemspace_nav_button",
            "prediction_tab": "prediction_nav_button",
            "sar_notes_tab": "notes_nav_button",
            "utilities_tab": "utilities_nav_button",
            "event_log": "event_log_nav_button",
            "slith_tab": "slith_nav_button",
        }
        for tab_tag, button_tag in button_map.items():
            if tab_tag == active_tab and dpg.does_item_exist(button_tag) and enabled_map.get(button_tag, True):
                dpg.bind_item_theme(button_tag, "top_nav_image_button_active_theme")

    def _refresh_top_nav_icons() -> None:
        """
        Rebuild glyph textures and theme bindings for top navigation buttons.
        """
        for _, button_tag, glyph, _, _, _, _ in tab_buttons:
            texture_tag = _make_nav_icon_texture(f"{button_tag}_texture", glyph)
            if dpg.does_item_exist(button_tag):
                dpg.configure_item(button_tag, texture_tag=texture_tag)
        _refresh_top_nav_selection()
        _refresh_top_nav_layout()

    state["refresh_top_nav_layout"] = _refresh_top_nav_layout
    state["refresh_top_nav_selection"] = _refresh_top_nav_selection
    state["refresh_top_nav_icons"] = _refresh_top_nav_icons


    with dpg.menu_bar(parent="main_window", tag="main_menu_bar"):

        # Helper: select correct dialog based on role
        def show_dialog(role: Any, state: dict[str, Any]) -> None:
            state["dialog_role"] = role
            title_map = {
                "input": "Select Input Directory",
                "output": "Select Output Directory",
                "predictions": "Select Predictions Output Directory",
                "load_results": "Select a Results Directory to Load Job From",
            }
            default_path_map = {
                "input": state.get("input_dir", ""),
                "output": state.get("output_dir", ""),
                "predictions": state.get("predictions_dir", ""),
                "load_results": state.get("output_dir", ""),
            }
            default_path = default_path_map.get(str(role), state.get("output_dir", ""))
            selected_dir = open_directory_dialog(title_map.get(str(role), "Select Directory"), default_path)
            if selected_dir:
                handle_selected_directory(selected_dir, state)

        state["show_dialog"] = show_dialog


        with dpg.window(
            label="Github",
            tag="github_popup",
            modal=True,
            show=False,
            autosize=True,
            no_resize=True,
        ):
            dpg.add_text(
                "Visit our GitHub repository\n"
                "for more information:"
            )
            dpg.add_spacer(height=state["win_spacer"])
            github_link = "https://github.com/I2DLab/SARgate"
            dpg.add_button(
                label=github_link,
                tag="github_popup_button",
                callback=lambda sender, app_data, user_data: webbrowser.open(user_data),
                user_data=github_link,
            )
            dpg.bind_item_theme("github_popup_button", link_button_theme)
            dpg.bind_item_theme("github_popup", apply_input_text_theme)


        with dpg.window(
            label="Contact",
            tag="contacts_popup",
            modal=True,
            show=False,
            autosize=True,
            no_resize=True,
        ):
            dpg.add_text(
                "For any questions or feedback\n"
                "contact the authors at:"
            )
            dpg.add_spacer(height=state["win_spacer"])
            dpg.add_input_text(
                tag="email_text",
                auto_select_all=True,
                readonly=True,
                default_value="i2dlab@outlook.it",
                callback=lambda: dpg.show_item("email_popup")
            )
            dpg.bind_item_theme("contacts_popup", apply_input_text_theme)


        with dpg.window(
            label="Report an Issue",
            tag="report_issue_popup",
            modal=True,
            show=False,
            autosize=True,
            no_resize=True,
        ):
            dpg.add_text(
                "Report bugs, crashes, or feature requests\n"
                "on the GitHub Issues page:"
            )
            dpg.add_spacer(height=state["win_spacer"])
            issues_link = "https://github.com/I2DLab/SARgate/issues"
            dpg.add_button(
                label=issues_link,
                tag="report_issue_popup_button",
                callback=lambda sender, app_data, user_data: webbrowser.open(user_data),
                user_data=issues_link,
            )
            dpg.bind_item_theme("report_issue_popup_button", link_button_theme)
            dpg.bind_item_theme("report_issue_popup", apply_input_text_theme)


        with dpg.window(
            label="About SARgate",
            tag="about_SARgate_popup",
            modal=True,
            show=False,
            autosize=True,
            no_resize=True,
        ):
            dpg.add_text("""SARgate
Molecular Toolkit for Chemical Space and\nStructure-Activity Relationship Analysis

Version: 1.0
Python: 3.13
Platform: macOS / Windows / Linux

© 2026 I2D Lab, University of Perugia
Open-source project (MIT License)

            """
            )
            change_font_type(dpg.last_item(), "bold", state)

            dpg.add_separator()
            dpg.add_text("""License summary (Full license texts 
are available in the "assets/legal/third_party_licenses/" directory):""")

            with dpg.tree_node(label="DearPyGUI", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the MIT License.
Copyright © 2025 Dear PyGui, LLC.
            """)

            with dpg.tree_node(label="NetworkX", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the BSD 3-Clause License.
Copyright © 2004-2018 NetworkX Developers.
            """)

            with dpg.tree_node(label="NumPy", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the BSD 3-Clause License.
Copyright © 2005-2025 NumPy Developers.
            """)

            with dpg.tree_node(label="openpyxl", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the MIT License.
Copyright © 2010 openpyxl.
Includes components under the Python Software Foundation License v2.
            """)

            with dpg.tree_node(label="pandas", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the BSD 3-Clause License.
Copyright © 2008-2011 AQR Capital Management, LLC; Lambda Foundry, Inc.; PyData Development Team.
Copyright © 2011-2025 Open source contributors.
            """)

            with dpg.tree_node(label="Pillow", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the MIT-CMU License.
Copyright © 2010-2025 Pillow Contributors.
            """)

            with dpg.tree_node(label="Plotly", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the MIT License.
Copyright © 2016-2024 Plotly Technologies Inc.
            """)

            with dpg.tree_node(label="pynput", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the GNU Lesser General Public License v3.
Copyright © 2015-2024 Moses Palmér and contributors.
            """)

            with dpg.tree_node(label="RDKit", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the BSD 3-Clause License.
Copyright © 2006-2015 Rational Discovery LLC, Greg Landrum, Julie Penzotti, and contributors.
            """)

            with dpg.tree_node(label="ReportLab", default_open=False, bullet=True):
                dpg.add_text("""Licensed under a BSD-like License.
Copyright © 2000-2014 ReportLab Inc.
            """)

            with dpg.tree_node(label="requests", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the ISC License.
Copyright © 2014 Kenneth Reitz.
            """)

            with dpg.tree_node(label="SciPy", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the BSD 3-Clause License.
Copyright © 2001-2002 Enthought, Inc.
Copyright © 2003-2025 SciPy Developers.
            """)

            with dpg.tree_node(label="scikit-learn", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the BSD 3-Clause License.
Copyright © 2007-2024 The scikit-learn developers.
            """)

            with dpg.tree_node(label="screeninfo", default_open=False, bullet=True):
                dpg.add_text("""Licensed under the MIT License.
Copyright © 2015 Marcin Kurczewski.
            """)                   

            with dpg.tree_node(label="Bundled fonts", default_open=False, bullet=True):
                dpg.add_text("""Arimo: Apache License 2.0.
DejaVu Sans: DejaVu / Bitstream Vera font license.
Fira Code: SIL Open Font License 1.1.
Font Awesome Free font: SIL Open Font License 1.1.
Noto Sans Symbols 2: SIL Open Font License 1.1.
Ubuntu Font Family: Ubuntu Font Licence 1.0.

Arial and Arial Unicode MS are not bundled with SARgate.
            """)
            dpg.bind_item_theme("about_SARgate_popup", apply_input_text_theme)


        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Load job", callback=lambda: show_dialog("load_results", state))
            with dpg.menu(label="Recent Files", tag="recent_files_menu"):
                pass
            dpg.add_separator()
            dpg.add_menu_item(label="Set input directory", callback=lambda: show_dialog("input", state))
            dpg.add_menu_item(label="Set results directory", callback=lambda: show_dialog("output", state))
            dpg.add_menu_item(label="Set predictions output directory", callback=lambda: show_dialog("predictions", state))
            dpg.add_separator()
            dpg.add_spacer(height=state["win_spacer"]/2)
            dpg.add_menu_item(label="Quit SARgate", callback=dpg.stop_dearpygui)

        with dpg.menu(label="View"):
            dpg.add_menu_item(label="Style editor", 
                              callback=lambda: (dpg.show_item("custom_style_editor"), dpg.focus_item("custom_style_editor")))
            dpg.add_menu_item(label="Fullscreen Mode", tag="fullscreen_menu_item", 
                              check=True, callback=lambda: toggle_fullscreen_with_check(state))

        with dpg.menu(label="Help"):
            dpg.add_menu_item(label="User Manual", callback=lambda: open_manual(state))
            with dpg.menu(label="Documentation"):
                dpg.add_menu_item(label="GitHub Repository", callback=lambda: dpg.show_item("github_popup"))
            with dpg.menu(label="Support"):
                dpg.add_menu_item(label="Contact", callback=lambda: dpg.show_item("contacts_popup") )
                dpg.add_menu_item(label="Report an Issue", callback=lambda: dpg.show_item("report_issue_popup"))
            dpg.add_separator()
            dpg.add_menu_item(label="About SARgate...", callback=lambda: dpg.show_item("about_SARgate_popup"))

        dpg.add_menu_item(label="?", tag="contextual_help_button", callback=lambda: open_contextual_help(state))

    refresh_recent_files_menu()

    with dpg.group(parent="main_window", show=not state["show_tab_icons"], tag="tab_text_container"):
        with dpg.tab_bar(tag="tab_bar", callback=lambda: change_tab(state), reorderable=False):
            dpg.add_tab(label="INPUT", tag="input_tab")
            dpg.add_tab(label="ANALYSIS", tag="analysis_tab", show=True)
            dpg.add_tab_button(label=" ", tag="tab_placeholder_button_1", callback=lambda: None)
            dpg.add_tab(label="OVERVIEW", tag="overview_tab", show=True)
            dpg.add_tab(label="R-ANALYSIS", tag="r_analysis_tab", show=True)
            dpg.add_tab(label="SIMILARITY", tag="similarity_tab", show=True)
            dpg.add_tab(label="STEREO", tag="stereo_tab", show=True)
            dpg.add_tab(label="MMPA", tag="mmpa_tab", show=True)
            dpg.add_tab(label="CHEMSPACE", tag="chemspace_tab", show=True)
            dpg.add_tab_button(label=" ", tag="tab_placeholder_button_2", callback=lambda: None)
            dpg.add_tab_button(label="SAR NOTES", tag="notes_text_button", callback=lambda: (
                    dpg.show_item("notes_popup") if dpg.does_item_exist("notes_popup") else None,
                    dpg.configure_item("notes_popup", collapsed=False) if dpg.does_item_exist("notes_popup") else None,
                    dpg.focus_item("notes_popup") if dpg.does_item_exist("notes_popup") else None,
                    dpg.set_value("notes_collapsing_header", True) if dpg.does_item_exist("notes_collapsing_header") else None,
                ),
            )
            dpg.add_tab(label="PREDICTION", tag="prediction_tab", show=True)
            dpg.add_tab_button(label="SKETCHER", tag="sketcher_text_button", callback=lambda: open_sketcher_window(state))
            dpg.add_tab(label="UTILITIES", tag="utilities_tab", show=True)
            dpg.add_tab_button(label="EVENT LOG", tag="event_log_text_button", callback=lambda: show_event_log_window(state))
            dpg.add_tab(label="SLITH-MINIGAME", tag="slith_tab", show=False)

    for _, button_tag, glyph, _, _, _, _ in tab_buttons:
        _make_nav_icon_texture(f"{button_tag}_texture", glyph)

    primary_tab_buttons = tab_buttons[:2]
    secondary_tab_buttons = tab_buttons[2:8]
    tertiary_tab_buttons = tab_buttons[8:]

    with dpg.child_window(parent="main_window", width=-1, 
                          auto_resize_y=True, border=False, tag="tab_icons_container",
                          show=state["show_tab_icons"]):
        dpg.bind_item_theme("tab_icons_container", transparent_host_theme)
        with dpg.table(
            tag="top_nav_table",
            header_row=False,
            borders_innerH=False,
            borders_outerH=False,
            borders_innerV=False,
            borders_outerV=False,
            policy=dpg.mvTable_SizingFixedFit,
        ):
            dpg.bind_item_theme("top_nav_table", top_nav_table_theme)
            dpg.add_table_column(
                label="Primary",
                tag="top_nav_primary_column",
            )
            dpg.add_table_column(
                label="Spacer 1",
                tag="top_nav_spacer_column_1",
                width_fixed=True,
                init_width_or_weight=50,
            )
            dpg.add_table_column(
                label="Secondary",
                tag="top_nav_secondary_column",
            )
            dpg.add_table_column(
                label="Spacer 2",
                tag="top_nav_spacer_column_2",
                width_fixed=True,
                init_width_or_weight=50,
            )
            dpg.add_table_column(
                label="Tertiary",
                tag="top_nav_tertiary_column",
            )
            dpg.bind_item_theme("top_nav_primary_column", top_nav_primary_column_theme)
            dpg.bind_item_theme("top_nav_secondary_column", top_nav_secondary_column_theme)
            dpg.bind_item_theme("top_nav_tertiary_column", top_nav_tertiary_column_theme)

            with dpg.table_row():
                with dpg.child_window(auto_resize_x=True, auto_resize_y=True, 
                                      border=False, tag="tab_bar_primary_group"):
                    dpg.bind_item_theme("tab_bar_primary_group", zero_item_gap_theme)
                    with dpg.group(horizontal=True):    
                        for idx, item in enumerate(primary_tab_buttons):
                            _, button_tag, _, full_label, shown, enabled, callback = item
                            dpg.add_image_button(
                                texture_tag=f"{button_tag}_texture",
                                tag=button_tag,
                                width=state["tab_button_size"],
                                height=state["tab_button_size"],
                                show=shown,
                                enabled=True,
                                callback=lambda *_, b=button_tag, cb=callback: _handle_top_nav_press(b, cb),
                            )
                            with dpg.tooltip(button_tag, tag=f"{button_tag}_tooltip", delay=0.10, show=shown):
                                dpg.add_text(full_label)

                dpg.add_spacer(width=50)

                with dpg.child_window(auto_resize_x=True, auto_resize_y=True, 
                                      border=False, tag="tab_bar_secondary_group"):
                    dpg.bind_item_theme("tab_bar_secondary_group", zero_item_gap_theme)
                    with dpg.group(horizontal=True):    
                        for idx, item in enumerate(secondary_tab_buttons):
                            _, button_tag, _, full_label, shown, enabled, callback = item
                            dpg.add_image_button(
                                texture_tag=f"{button_tag}_texture",
                                tag=button_tag,
                                width=state["tab_button_size"],
                                height=state["tab_button_size"],
                                show=shown,
                                enabled=True,
                                callback=lambda *_, b=button_tag, cb=callback: _handle_top_nav_press(b, cb),
                            )
                            with dpg.tooltip(button_tag, tag=f"{button_tag}_tooltip", delay=0.10, show=shown):
                                dpg.add_text(full_label)

                dpg.add_spacer(width=50)

                with dpg.child_window(auto_resize_x=True, auto_resize_y=True, 
                                      border=False, tag="tab_bar_tertiary_group"):
                    dpg.bind_item_theme("tab_bar_tertiary_group", zero_item_gap_theme)
                    with dpg.group(horizontal=True):    
                        for idx, item in enumerate(tertiary_tab_buttons):
                            _, button_tag, _, full_label, shown, enabled, callback = item
                            dpg.add_image_button(
                                texture_tag=f"{button_tag}_texture",
                                tag=button_tag,
                                width=state["tab_button_size"],
                                height=state["tab_button_size"],
                                show=shown,
                                enabled=True,
                                callback=lambda *_, b=button_tag, cb=callback: _handle_top_nav_press(b, cb),
                            )
                            with dpg.tooltip(button_tag, tag=f"{button_tag}_tooltip", delay=0.10, show=shown):
                                dpg.add_text(full_label)

        _refresh_top_nav_icons()
        _apply_top_nav_mode()
        _refresh_text_tab_themes()


    with dpg.child_window( tag="input_tab_child", parent="main_window", show=True, no_scroll_with_mouse=False, horizontal_scrollbar=True, no_scrollbar=False, width=-1, height=-10, border=False,):
        with dpg.child_window(
            show=True,
            no_scroll_with_mouse=False,
            horizontal_scrollbar=True,
            no_scrollbar=False,
            border=False,
            width=-1,
            height=contents_h * 0.65,
            resizable_y=True,
        ):
            with dpg.table(
                header_row=False,
                width=-1,
                height=-1,
                resizable=True,
                context_menu_in_body=True,
                borders_innerH=False,
                borders_outerH=False,
                borders_innerV=True,
                borders_outerV=False,
                policy=dpg.mvTable_SizingStretchProp,
            ):
                change_font_type(dpg.last_item(), "bold", state)

                dpg.add_table_column(label="File Selection", init_width_or_weight=22)
                dpg.add_table_column(label="Settings", init_width_or_weight=78)

                with dpg.table_row():
                    
                    with dpg.child_window(
                        label="File Selection",
                        tag="file_selection_and_run_window",
                        no_scrollbar=True,
                        horizontal_scrollbar=True,
                        no_scroll_with_mouse=True,
                        border=False,
                        width=-1,
                        height=-1,
                    ):
                        change_font_type(dpg.last_item(), "regular", state)
                        dpg.bind_item_theme("file_selection_and_run_window", panel_theme)

                        with dpg.child_window(
                            label="File Selection",
                            tag="file_selection_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=220,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)
                            dpg.bind_item_theme("file_selection_window", panel_theme)


                        with dpg.child_window(
                            label="Start Analysis",
                            tag="start_analysis_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)
                            show_input_selection_window(state)
                            dpg.bind_item_theme("start_analysis_window", panel_theme)

                    with dpg.child_window(
                        label="Options",
                        tag="options_window",
                        no_scrollbar=True,
                        horizontal_scrollbar=True,
                        no_scroll_with_mouse=True,
                        border=True,
                        width=-1,
                        height=-1,
                    ):
                        change_font_type(dpg.last_item(), "regular", state)
                        dpg.bind_item_theme("options_window", panel_theme)
                        show_settings_window(state)


        with dpg.table(
            tag="library_overview_table",
            show=False,
            header_row=True,
            width=-1,
            height=contents_h,
            resizable=True,
            context_menu_in_body=True,
            borders_innerH=False,
            borders_outerH=False,
            borders_innerV=False,
            borders_outerV=False,
            policy=dpg.mvTable_SizingStretchProp,
        ):
            change_font_type(dpg.last_item(), "bold", state)

            dpg.add_table_column(label="Library Summary Table")

            with dpg.table_row():
                with dpg.child_window(
                    label="Library Summary",
                    tag="library_table_window",
                    no_scrollbar=False,
                    horizontal_scrollbar=True,
                    no_scroll_with_mouse=True,
                    border=True,
                    width=-1,
                    height=-1,
                ):
                    change_font_type(dpg.last_item(), "regular", state)
                    dpg.bind_item_theme("library_table_window", panel_theme)


    with dpg.child_window( tag="analysis_tab_child", parent="main_window", show=False, no_scroll_with_mouse=False, horizontal_scrollbar=True, no_scrollbar=False, width=-1, height=-10, border=False,):

        with dpg.table(
            header_row=True,
            width=-1,
            height=-1,
            resizable=True,
            context_menu_in_body=True,
            borders_innerH=False,
            borders_outerH=False,
            borders_innerV=True,
            borders_outerV=False,
            policy=dpg.mvTable_SizingStretchProp,
        ):
            change_font_type(dpg.last_item(), "bold", state)

            dpg.add_table_column(label="Library Preparation", init_width_or_weight=33.34)
            dpg.add_table_column(label="Substructure Analysis", init_width_or_weight=33.33)
            dpg.add_table_column(label="R-Groups Decomposition", init_width_or_weight=33.33)

            with dpg.table_row():

                with dpg.child_window(
                    label="Library Preparation",
                    tag="library_preparation_window",
                    no_scrollbar=False,
                    horizontal_scrollbar=True,
                    no_scroll_with_mouse=False,
                    border=True,
                    width=-1,
                    height=-1,
                ):
                    change_font_type(dpg.last_item(), "regular", state)
                    dpg.bind_item_theme("library_preparation_window", panel_theme)

                with dpg.child_window(
                    label="Scaffold Analysis",
                    tag="scaffold_analysis_window",
                    no_scrollbar=False,
                    horizontal_scrollbar=True,
                    no_scroll_with_mouse=False,
                    border=True,
                    width=-1,
                    height=-1,
                ):
                    change_font_type(dpg.last_item(), "regular", state)
                    dpg.bind_item_theme("scaffold_analysis_window", panel_theme)

                with dpg.child_window(
                    label="RGA",
                    tag="rga_window",
                    no_scrollbar=False,
                    horizontal_scrollbar=True,
                    no_scroll_with_mouse=False,
                    border=True,
                    width=-1,
                    height=-1,
                ):
                    change_font_type(dpg.last_item(), "regular", state)
                    dpg.bind_item_theme("rga_window", panel_theme)


    with dpg.child_window(tag="overview_tab_child", parent="main_window", show=False, no_scroll_with_mouse=False, horizontal_scrollbar=True, no_scrollbar=False, width=-1, height=-10, border=False,):

        with dpg.tab_bar(tag="overview_tab_bar", callback=lambda: change_overview_subtab(state)):

            with dpg.tab(label="Decomposition", tag="overview_decomposition_subtab"):

                overview_manager_h = contents_h * 0.3

                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=0,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column()

                    with dpg.table_row():

                        with dpg.child_window(
                            label="Overview Selection Table",
                            tag="overview_selection_table_window",
                            resizable_y=True,
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=overview_manager_h,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                            with dpg.table(
                                header_row=True,
                                width=-1,
                                height=-1,
                                resizable=True,
                                context_menu_in_body=True,
                                borders_innerH=False,
                                borders_outerH=True,
                                borders_innerV=True,
                                borders_outerV=False,
                                policy=dpg.mvTable_SizingStretchProp,
                            ):
                                change_font_type(dpg.last_item(), "bold", state)

                                dpg.add_table_column(label="Subsets: ", tag="subset_choice_column", init_width_or_weight=10)
                                dpg.add_table_column(label="Molecules: ", tag="molecule_choice_column", init_width_or_weight=10)
                                dpg.add_table_column(label="R-Groups", init_width_or_weight=7)
                                dpg.add_table_column(label="Subset Properties", tag="properties_column", init_width_or_weight=43)
                                dpg.add_table_column(label="Molecule Activity Data", init_width_or_weight=30)

                                with dpg.table_row():

                                    with dpg.child_window(
                                        label="Subset_choice",
                                        tag="subset_choice",
                                        no_scrollbar=True,
                                        horizontal_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        border=False,
                                        width=-1,
                                        height=-1,
                                    ):
                                        change_font_type(dpg.last_item(), "regular", state)

                                    with dpg.child_window(
                                        label="Molecule_choice",
                                        tag="molecule_choice",
                                        no_scrollbar=True,
                                        horizontal_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        border=False,
                                        width=-1,
                                        height=-1,
                                    ):
                                        change_font_type(dpg.last_item(), "regular", state)

                                    with dpg.child_window(
                                        label="R-group choice",
                                        tag="r_group_choice",
                                        no_scrollbar=True,
                                        horizontal_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        border=False,
                                        width=-1,
                                        height=-1,
                                    ):
                                        change_font_type(dpg.last_item(), "regular", state)

                                    with dpg.child_window(
                                        label="Properties window",
                                        tag="properties_window",
                                        no_scrollbar=True,
                                        horizontal_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        border=False,
                                        width=-1,
                                        height=-1,
                                    ):
                                        change_font_type(dpg.last_item(), "regular", state)

                                    with dpg.child_window(
                                        label="Activities window",
                                        tag="activities_window",
                                        no_scrollbar=True,
                                        horizontal_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        border=False,
                                        width=-1,
                                        height=-1,
                                    ):
                                        change_font_type(dpg.last_item(), "regular", state)


                with dpg.table(
                    header_row=False,
                    width=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(label="Images")

                    with dpg.table_row(tag="overview_image_row",):
                        with dpg.child_window(
                            tag="mol_image",
                            width=-1,
                            auto_resize_y=True,
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=False,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)
                            dpg.bind_item_theme("mol_image", apply_outer_child_theme())


                with dpg.table(
                    header_row=False,
                    width=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column()

                    with dpg.table_row():

                        default_h = contents_h * 0.25
                        with dpg.child_window(
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                            with dpg.table(
                                header_row=False,
                                width=-1,
                                height=-1,
                                resizable=True,
                                context_menu_in_body=True,
                                borders_innerH=False,
                                borders_outerH=False,
                                borders_innerV=True,
                                borders_outerV=False,
                                policy=dpg.mvTable_SizingStretchProp,
                            ):
                                change_font_type(dpg.last_item(), "bold", state)

                                dpg.add_table_column(label="Image Checkboxes", init_width_or_weight=22)
                                dpg.add_table_column(label="Enrichment plot", init_width_or_weight=78)

                                with dpg.table_row():

                                    with dpg.child_window(
                                        tag="image_checkboxes_window",
                                        width=-1,
                                        auto_resize_y=True,
                                        no_scrollbar=False,
                                        horizontal_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        border=False,
                                    ):
                                        change_font_type(dpg.last_item(), "regular", state)

                                    with dpg.child_window(
                                        tag="enrichment_plot_window",
                                        width=-1,
                                        auto_resize_y=True,
                                        no_scrollbar=False,
                                        horizontal_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        border=False,
                                    ):
                                        change_font_type(dpg.last_item(), "regular", state)


            with dpg.tab(label="Subset Overview", tag="overview_table_subtab"):

                with dpg.table(
                    header_row=True,
                    width=-1,
                    height=contents_h,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                    freeze_rows=1,
                    scrollY=False,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(label="Subset Overview Table", tag="overview_table_column")

                    with dpg.table_row():

                        with dpg.child_window(
                            label="Subset overview table manager",
                            tag="overview_table_manager",
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                    with dpg.table_row():

                        with dpg.child_window(
                            label="Subset overview table window",
                            tag="overview_table_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)


            with dpg.child_window( tag="r_analysis_tab_child", parent="main_window", show=False, no_scroll_with_mouse=False, horizontal_scrollbar=True, no_scrollbar=False, width=-1, height=-10, border=False, ):

                with dpg.tab_bar(tag="r_analysis_tab_bar", callback=lambda: change_r_analysis_subtab(state)):

                    with dpg.tab(label="R-Group Counts", tag="r_analysis_counts_subtab"):

                        with dpg.table(
                            header_row=False,
                            width=-1,
                            height=-1,
                            resizable=True,
                            context_menu_in_body=True,
                            borders_innerH=False,
                            borders_outerH=False,
                            borders_innerV=False,
                            borders_outerV=False,
                            policy=dpg.mvTable_SizingStretchProp,
                        ):  
                            change_font_type(dpg.last_item(), "bold", state)

                            dpg.add_table_column(label="Parameters")

                            with dpg.table_row():
                                with dpg.child_window(
                                    label="Subset & R-group Selector",
                                    tag="counts_selection_window",
                                    no_scrollbar=False,
                                    horizontal_scrollbar=True,
                                    no_scroll_with_mouse=True,
                                    border=True,
                                    width=-1,
                                    auto_resize_y=True,
                                ):
                                    change_font_type(dpg.last_item(), "regular", state)


                            with dpg.table_row():

                                default_h = contents_h * 0.45
                                state["default_counts_details_table_row_height"] = default_h

                                with dpg.child_window(
                                    width=-1,
                                    height=default_h,
                                    resizable_y=True,
                                    no_scrollbar=True,
                                    horizontal_scrollbar=True,
                                    no_scroll_with_mouse=True,
                                    border=True,
                                ):
                                    change_font_type(dpg.last_item(), "regular", state)

                                    with dpg.table(
                                        header_row=False,
                                        width=-1,
                                        height=-1,
                                        resizable=True,
                                        context_menu_in_body=True,
                                        borders_innerH=False,
                                        borders_outerH=False,
                                        borders_innerV=True,
                                        borders_outerV=False,
                                        policy=dpg.mvTable_SizingStretchProp,
                                    ):
                                        change_font_type(dpg.last_item(), "bold", state)

                                        dpg.add_table_column(label="Scaffold image", init_width_or_weight=28)
                                        dpg.add_table_column(label="Plot", init_width_or_weight=42)
                                        dpg.add_table_column(label="Counts boxplot details", init_width_or_weight=13)
                                        dpg.add_table_column(label="Molecules with selected R-group", init_width_or_weight=17)

                                        with dpg.table_row(tag="counts_details_table_row", height=default_h):

                                            with dpg.child_window(
                                                label="Counts Scaffold Window",
                                                tag="counts_scaffold_img_window",
                                                no_scrollbar=False,
                                                horizontal_scrollbar=True,
                                                no_scroll_with_mouse=True,
                                                border=False,
                                                height=-1,
                                            ):
                                                change_font_type(dpg.last_item(), "regular", state)

                                            with dpg.child_window(
                                                label="Counts Plot Window",
                                                tag="counts_boxplot_window",
                                                no_scrollbar=False,
                                                horizontal_scrollbar=True,
                                                no_scroll_with_mouse=True,
                                                border=False,
                                                height=-1,
                                            ):
                                                change_font_type(dpg.last_item(), "regular", state)

                                            with dpg.child_window(
                                                label="Counts Boxplot Details",
                                                tag="counts_boxplot_details_window",
                                                no_scrollbar=False,
                                                horizontal_scrollbar=True,
                                                no_scroll_with_mouse=True,
                                                border=False,
                                                height=-1,
                                            ):
                                                change_font_type(dpg.last_item(), "regular", state)

                                            with dpg.child_window(
                                                label="Counts R-group Details",
                                                tag="counts_rgroup_details_window",
                                                no_scrollbar=True,
                                                horizontal_scrollbar=True,
                                                no_scroll_with_mouse=False,
                                                border=False,
                                                height=-1,
                                            ):
                                                change_font_type(dpg.last_item(), "regular", state)


                            with dpg.table_row():
                                with dpg.child_window(
                                    label="Counts boxplot manager",
                                    tag="counts_boxplot_manager_window",
                                    no_scrollbar=False,
                                    horizontal_scrollbar=True,
                                    no_scroll_with_mouse=True,
                                    border=True,
                                    width=-1,
                                    auto_resize_y=True,
                                ):
                                    change_font_type(dpg.last_item(), "regular", state)


                            with dpg.table_row():

                                with dpg.child_window(
                                    no_scrollbar=True,
                                    horizontal_scrollbar=True,
                                    no_scroll_with_mouse=True,
                                    border=True,
                                    height=contents_h,
                                    width=-1,
                                ):
                                    change_font_type(dpg.last_item(), "regular", state)

                                    with dpg.table(
                                        tag="counts_table",
                                        show=False,
                                        header_row=False,
                                        width=-1,
                                        resizable=False,
                                        context_menu_in_body=True,
                                        borders_innerH=False,
                                        borders_outerH=False,
                                        borders_innerV=False,
                                        borders_outerV=False,
                                        policy=dpg.mvTable_SizingStretchProp,
                                    ):
                                        change_font_type(dpg.last_item(), "bold", state)

                                        dpg.add_table_column(label="R-counts Table")

                                        with dpg.table_row():
                                            with dpg.child_window(
                                                label="R-group Counts",
                                                tag="counts_table_window",
                                                no_scrollbar=False,
                                                horizontal_scrollbar=True,
                                                no_scroll_with_mouse=False,
                                                border=False,
                                                height=contents_h,
                                                width=-1,
                                            ):
                                                change_font_type(dpg.last_item(), "regular", state)


                    with dpg.tab(label="R-Pair Matrix", tag="r_analysis_table_subtab"):
                        
                        manager_win_h = contents_h * 0.05

                        with dpg.table(
                            header_row=False,
                            width=-1,
                            height=manager_win_h,
                            resizable=True,
                            context_menu_in_body=True,
                            borders_innerH=False,
                            borders_outerH=False,
                            borders_innerV=False,
                            borders_outerV=False,
                            policy=dpg.mvTable_SizingStretchProp,
                        ):
                            change_font_type(dpg.last_item(), "bold", state)

                            dpg.add_table_column(label="Parameters")

                            with dpg.table_row():

                                with dpg.child_window(
                                    label="R-groups Table Panel",
                                    tag="heatmap_manager_window",
                                    no_scrollbar=True,
                                    horizontal_scrollbar=False,
                                    no_scroll_with_mouse=False,
                                    border=True,
                                    width=-1,
                                    auto_resize_y=True,
                                ):
                                    change_font_type(dpg.last_item(), "regular", state)

                        with dpg.table(
                            header_row=False,
                            width=-1,
                            height=-1,
                            resizable=True,
                            context_menu_in_body=True,
                            borders_innerH=False,
                            borders_outerH=False,
                            borders_innerV=True,
                            borders_outerV=False,
                            policy=dpg.mvTable_SizingStretchProp,
                        ):
                            change_font_type(dpg.last_item(), "bold", state)

                            dpg.add_table_column(label="Plot", init_width_or_weight=80)
                            dpg.add_table_column(label="Molecules", init_width_or_weight=20)

                            with dpg.table_row():

                                with dpg.child_window(
                                    tag="heatmap_window",
                                    no_scrollbar=False,
                                    horizontal_scrollbar=True,
                                    no_scroll_with_mouse=True,
                                    border=True,
                                    width=-1,
                                    height=-1,
                                ):
                                    change_font_type(dpg.last_item(), "regular", state)

                                with dpg.child_window(
                                    tag="heatmap_details_window",
                                    no_scrollbar=False,
                                    horizontal_scrollbar=True,
                                    no_scroll_with_mouse=True,
                                    border=True,
                                    width=-1,
                                    height=-1,
                                ):
                                    change_font_type(dpg.last_item(), "regular", state)


    with dpg.child_window(tag="similarity_tab_child", parent="main_window", show=False, no_scroll_with_mouse=True, horizontal_scrollbar=True, no_scrollbar=True, width=-1, height=-10, border=False,):

        manager_win_h = contents_h * 0.05

        with dpg.tab_bar(tag="similarity_tab_bar", callback=lambda: change_similarity_subtab(state)):


            with dpg.tab(label="Similarity Matrix", tag="similarity_matrix_subtab"):

                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=manager_win_h * 2,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(label="Parameters")

                    with dpg.table_row():
                        with dpg.child_window(
                            label="Similarity manager window",
                            tag="similarity_manager_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                    scrollY=True,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(label="Tanimoto Similarity Matrix", init_width_or_weight=73)
                    dpg.add_table_column(label="Molecule Pair", init_width_or_weight=27)

                    with dpg.table_row():

                        with dpg.child_window(
                            label="Tanimoto Similarity Matrix",
                            tag="similarity_tanimoto_window",
                            no_scrollbar=False,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                            resizable_y=False,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                        with dpg.child_window(
                            label="Tanimoto Similarity Mol Couple",
                            tag="similarity_tanimoto_mol_couple_window",
                            no_scrollbar=False,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)


            with dpg.tab(label="Clustered Similarity Matrix", tag="clustered_matrix_subtab"):

                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=manager_win_h * 2,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(label="Parameters")

                    with dpg.table_row():
                        with dpg.child_window(
                            label="Clustered similarity manager window",
                            tag="clustered_similarity_manager_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                    scrollY=True,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(label="Clustered Similarity Matrix", init_width_or_weight=73)
                    dpg.add_table_column(label="Molecule Pair", init_width_or_weight=27)

                    with dpg.table_row():

                        with dpg.child_window(
                            label="Clustered Similarity Matrix",
                            tag="clustered_similarity_matrix_window",
                            no_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                            resizable_y=False,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                        with dpg.child_window(
                            label="Clustered Similarity Mol Couple",
                            tag="clustered_similarity_mol_couple_window",
                            no_scrollbar=False,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)


            with dpg.tab(label="Structure-Activity Landscape", tag="landscape_tab"):

                manager_win_h = contents_h * 0.05

                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=manager_win_h * 2,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):

                    dpg.add_table_column(label="Parameters")

                    with dpg.table_row():

                        with dpg.child_window(
                            label="Landscape Panel",
                            tag="landscape_manager_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=False,
                            no_scroll_with_mouse=False,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            pass


                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(label="Plot", init_width_or_weight=75)
                    dpg.add_table_column(label="Molecules", init_width_or_weight=25)

                    with dpg.table_row():

                        with dpg.child_window(
                            tag="landscape_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                        with dpg.child_window(
                            tag="landscape_details_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)


    with dpg.child_window( tag="stereo_tab_child", parent="main_window", show=False, no_scroll_with_mouse=False, horizontal_scrollbar=True, no_scrollbar=False, width=-1, height=-10, border=False,):

        with dpg.table(
            header_row=False,
            width=-1,
            height=-1,
            resizable=True,
            context_menu_in_body=True,
            borders_innerH=False,
            borders_outerH=False,
            borders_innerV=False,
            borders_outerV=False,
            policy=dpg.mvTable_SizingStretchProp,
            scrollY=True,
        ):
            change_font_type(dpg.last_item(), "bold", state)

            dpg.add_table_column(label="Stereoisomers Table")

            with dpg.table_row():

                with dpg.group():

                    with dpg.child_window(
                        label="Stereo manager window",
                        tag="isomers_manager_window",
                        no_scrollbar=True,
                        horizontal_scrollbar=True,
                        no_scroll_with_mouse=True,
                        border=True,
                        width=-1,
                        auto_resize_y=True,
                    ):
                        change_font_type(dpg.last_item(), "regular", state)


            with dpg.table_row():

                with dpg.child_window(
                    label="Stereo Images Window",
                    tag="isomers_images_main_window",
                    no_scrollbar=False,
                    horizontal_scrollbar=True,
                    no_scroll_with_mouse=True,
                    border=True,
                    width=-1,
                    height=-1,
                ):
                    change_font_type(dpg.last_item(), "regular", state)


    with dpg.child_window( tag="mmpa_tab_child", parent="main_window", show=False, no_scroll_with_mouse=False, horizontal_scrollbar=True, no_scrollbar=False, width=-1, height=-10, border=False, ):

        with dpg.tab_bar(tag="mmpa_tab_bar"):

            with dpg.tab(label="MMPA Table", tag="mmpa_table_subtab"):

                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=contents_h * 0.08,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(label="Parameters")

                    with dpg.table_row():

                        with dpg.child_window(
                            label="MMPA Panel",
                            tag="mmpa_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=False,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)


                with dpg.table(
                    header_row=False,
                    width=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column()

                    with dpg.table_row():

                        with dpg.child_window(
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=contents_h * 0.4,
                            resizable_y=True,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                            with dpg.table(
                                header_row=False,
                                width=-1,
                                height=-1,
                                resizable=True,
                                context_menu_in_body=True,
                                borders_innerH=False,
                                borders_outerH=False,
                                borders_innerV=True,
                                borders_outerV=False,
                                policy=dpg.mvTable_SizingStretchProp,
                            ):
                                change_font_type(dpg.last_item(), "bold", state)

                                dpg.add_table_column(label="Plot Global", init_width_or_weight=0.5)
                                dpg.add_table_column(label="Plot Subset", init_width_or_weight=0.5)

                                with dpg.table_row():

                                    with dpg.child_window(
                                        label="MMPA plot window",
                                        tag="mmpa_global_plot_window",
                                        no_scrollbar=True,
                                        horizontal_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        border=False,
                                        height=-1,
                                        width=-1,
                                    ):
                                        change_font_type(dpg.last_item(), "regular", state)

                                    with dpg.child_window(
                                        label="MMPA plot window",
                                        tag="mmpa_subset_plot_window",
                                        no_scrollbar=True,
                                        horizontal_scrollbar=True,
                                        no_scroll_with_mouse=True,
                                        border=False,
                                        height=-1,
                                        width=-1,
                                    ):
                                        change_font_type(dpg.last_item(), "regular", state)

                with dpg.table(
                    header_row=False,
                    width=-1,
                    resizable=False,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(label="Transformation")

                    default_h = contents_h * 0.35
                    state["default_mmpa_images_table_row_height"] = default_h

                    with dpg.table_row(tag="mmpa_images_row"):

                        with dpg.child_window(
                            tag="mmpa_images_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)


                with dpg.table(
                    tag="mmpa_table_cont",
                    show=False,
                    header_row=True,
                    width=-1,
                    height=contents_h,
                    resizable=False,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(label="MMPA Table")

                    with dpg.table_row():

                        with dpg.child_window(
                            tag="mmpa_table_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            height=contents_w // 1.8,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)


            with dpg.tab(label="MMPA Network", tag="mmpa_network_subtab"):

                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)

                    dpg.add_table_column(init_width_or_weight=20)
                    dpg.add_table_column(init_width_or_weight=80)

                    with dpg.table_row():

                        with dpg.child_window(
                            tag="mmpa_group_sidebar_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=False,
                            border=True,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                        with dpg.child_window(
                            tag="mmpa_network_plot_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=False,
                            border=True,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)


    with dpg.child_window( tag="chemspace_tab_child", parent="main_window", show=False, no_scroll_with_mouse=False, horizontal_scrollbar=True, no_scrollbar=False, width=-1, height=-10, border=False):

        with dpg.tab_bar(tag="chemspace_tab_bar", callback=lambda: change_chemspace_subtab(state)):


            manager_win_h = contents_h * 0.05

            # with dpg.tab(label="Hierarchical Dendrogram", tag="dendrogram_tab"):
            #     with dpg.table(
            #         tag="dendrogram_layout_table",
            #         header_row=False,
            #         width=-1,
            #         height=-1,
            #         resizable=True,
            #         context_menu_in_body=True,
            #         borders_innerH=False,
            #         borders_outerH=False,
            #         borders_innerV=True,
            #         borders_outerV=False,
            #         policy=dpg.mvTable_SizingStretchProp,
            #     ):
            #         change_font_type(dpg.last_item(), "bold", state)
            #         dpg.add_table_column(label="Subsets", init_width_or_weight=12)
            #         dpg.add_table_column(label="Plot", init_width_or_weight=58)
            #         dpg.add_table_column(label="Details", init_width_or_weight=30)
            #         dpg.bind_item_theme("dendrogram_layout_table", _ensure_chemspace_layout_table_theme("dendrogram_layout_table_theme"))
            #         with dpg.table_row():
            #             with dpg.child_window(
            #                 tag="dendrogram_subset_panel",
            #                 no_scrollbar=False,
            #                 horizontal_scrollbar=False,
            #                 no_scroll_with_mouse=False,
            #                 border=True,
            #                 width=-1,
            #                 height=-1,
            #             ):
            #                 change_font_type(dpg.last_item(), "regular", state)

            #             with dpg.child_window(
            #                 tag="dendrogram_window",
            #                 no_scrollbar=False,
            #                 horizontal_scrollbar=False,
            #                 no_scroll_with_mouse=True,
            #                 border=True,
            #                 width=-1,
            #                 height=-1,
            #             ):
            #                 change_font_type(dpg.last_item(), "regular", state)

            #             with dpg.child_window(
            #                 tag="dendrogram_details_window",
            #                 no_scrollbar=False,
            #                 horizontal_scrollbar=False,
            #                 no_scroll_with_mouse=True,
            #                 border=True,
            #                 width=-1,
            #                 height=-1,
            #             ):
            #                 change_font_type(dpg.last_item(), "regular", state)

            with dpg.tab(label="Descriptor & Activity Plot", tag="descriptors_tab"):

                with dpg.table(
                    tag="descriptors_manager_layout_table",
                    header_row=False,
                    width=-1,
                    height=manager_win_h,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)
                    dpg.bind_item_theme("descriptors_manager_layout_table", _ensure_chemspace_layout_table_theme("descriptors_manager_layout_table_theme"))

                    dpg.add_table_column(label="Parameters")

                    with dpg.table_row():

                        with dpg.child_window(
                            label="Descriptors Panel",
                            tag="descriptors_manager_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=False,
                            no_scroll_with_mouse=False,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)


                with dpg.table(
                    tag="descriptors_layout_table",
                    header_row=False,
                    width=-1,
                    height=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)
                    dpg.bind_item_theme("descriptors_layout_table", _ensure_chemspace_layout_table_theme("descriptors_layout_table_theme"))

                    dpg.add_table_column(label="Plot", init_width_or_weight=70)
                    dpg.add_table_column(label="Molecule", init_width_or_weight=30)

                    with dpg.table_row():

                        with dpg.child_window(
                            tag="descriptors_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                        with dpg.child_window(
                            tag="descriptors_details_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)


            with dpg.tab(label="PCA", tag="pca_tab"):

                with dpg.table(
                    tag="pca_manager_layout_table",
                    header_row=False,
                    width=-1,
                    height=manager_win_h,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    dpg.bind_item_theme("pca_manager_layout_table", _ensure_chemspace_layout_table_theme("pca_manager_layout_table_theme"))

                    dpg.add_table_column(label="Parameters")

                    with dpg.table_row():

                        with dpg.child_window(
                            label="PCA Panel",
                            tag="pca_manager_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=False,
                            no_scroll_with_mouse=False,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            pass


                with dpg.table(
                    tag="pca_layout_table",
                    header_row=False,
                    width=-1,
                    height=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)
                    dpg.bind_item_theme("pca_layout_table", _ensure_chemspace_layout_table_theme("pca_layout_table_theme"))

                    dpg.add_table_column(label="Plot", init_width_or_weight=70)
                    dpg.add_table_column(label="Molecules", init_width_or_weight=30)

                    with dpg.table_row():

                        with dpg.child_window(
                            tag="pca_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                        with dpg.child_window(
                            tag="pca_details_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

            with dpg.tab(label="UMAP", tag="umap_tab"):

                with dpg.table(
                    tag="umap_manager_layout_table",
                    header_row=False,
                    width=-1,
                    height=manager_win_h,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    dpg.bind_item_theme("umap_manager_layout_table", _ensure_chemspace_layout_table_theme("umap_manager_layout_table_theme"))

                    dpg.add_table_column(label="Parameters")

                    with dpg.table_row():

                        with dpg.child_window(
                            label="UMAP Panel",
                            tag="umap_manager_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=False,
                            no_scroll_with_mouse=False,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            pass

                with dpg.table(
                    tag="umap_layout_table",
                    header_row=False,
                    width=-1,
                    height=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)
                    dpg.bind_item_theme("umap_layout_table", _ensure_chemspace_layout_table_theme("umap_layout_table_theme"))

                    dpg.add_table_column(label="Plot", init_width_or_weight=70)
                    dpg.add_table_column(label="Molecules", init_width_or_weight=30)

                    with dpg.table_row():

                        with dpg.child_window(
                            tag="umap_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                        with dpg.child_window(
                            tag="umap_details_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

            with dpg.tab(label="t-SNE", tag="tsne_tab"):

                with dpg.table(
                    tag="tsne_manager_layout_table",
                    header_row=False,
                    width=-1,
                    height=manager_win_h,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    dpg.bind_item_theme("tsne_manager_layout_table", _ensure_chemspace_layout_table_theme("tsne_manager_layout_table_theme"))

                    dpg.add_table_column(label="Parameters")

                    with dpg.table_row():

                        with dpg.child_window(
                            label="t-SNE Panel",
                            tag="tsne_manager_window",
                            no_scrollbar=True,
                            horizontal_scrollbar=False,
                            no_scroll_with_mouse=False,
                            border=True,
                            width=-1,
                            auto_resize_y=True,
                        ):
                            pass

                with dpg.table(
                    tag="tsne_layout_table",
                    header_row=False,
                    width=-1,
                    height=-1,
                    resizable=True,
                    context_menu_in_body=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=True,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    change_font_type(dpg.last_item(), "bold", state)
                    dpg.bind_item_theme("tsne_layout_table", _ensure_chemspace_layout_table_theme("tsne_layout_table_theme"))

                    dpg.add_table_column(label="Plot", init_width_or_weight=70)
                    dpg.add_table_column(label="Molecules", init_width_or_weight=30)

                    with dpg.table_row():

                        with dpg.child_window(
                            tag="tsne_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

                        with dpg.child_window(
                            tag="tsne_details_window",
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=True,
                            border=True,
                            width=-1,
                            height=-1,
                        ):
                            change_font_type(dpg.last_item(), "regular", state)

    with dpg.child_window(tag="prediction_tab_child", parent="main_window", show=False, no_scroll_with_mouse=False, horizontal_scrollbar=True, no_scrollbar=False, width=-1, height=-10, border=False):

        with dpg.table(
            header_row=False,
            width=-1,
            height=-1,
            resizable=True,
            context_menu_in_body=True,
            borders_innerH=False,
            borders_outerH=False,
            borders_innerV=False,
            borders_outerV=False,
            policy=dpg.mvTable_SizingStretchProp,
        ):
            change_font_type(dpg.last_item(), "bold", state)
            dpg.add_table_column(label="Prediction")

            with dpg.table_row():
                with dpg.child_window(
                    tag="prediction_manager_host",
                    no_scrollbar=True,
                    horizontal_scrollbar=True,
                    no_scroll_with_mouse=True,
                    border=True,
                    width=-1,
                    auto_resize_y=True,
                ):
                    change_font_type(dpg.last_item(), "regular", state)

            with dpg.table_row():
                with dpg.child_window(
                    tag="prediction_output_host",
                    no_scrollbar=False,
                    horizontal_scrollbar=True,
                    no_scroll_with_mouse=False,
                    border=True,
                    width=-1,
                    height=-1,
                ):
                    change_font_type(dpg.last_item(), "regular", state)


    with dpg.child_window( tag="utilities_tab_child", parent="main_window", show=False, no_scroll_with_mouse=False, horizontal_scrollbar=True, no_scrollbar=False, width=-1, height=-10, border=False, ):

        with dpg.table(
            header_row=False,
            width=-1,
            height=-1,
            resizable=True,
            context_menu_in_body=True,
            borders_innerH=False,
            borders_outerH=False,
            borders_innerV=False,
            borders_outerV=False,
            policy=dpg.mvTable_SizingStretchProp,
        ):
            change_font_type(dpg.last_item(), "bold", state)

            dpg.add_table_column(label="Utilities")

            with dpg.table_row():

                with dpg.child_window(
                    label="Utilities",
                    tag="utils_window",
                    no_scrollbar=False,
                    horizontal_scrollbar=True,
                    no_scroll_with_mouse=False,
                    border=True,
                    width=-1,
                    height=-1,
                ):
                    change_font_type(dpg.last_item(), "regular", state)
                    show_utilities_window(state, log_on_open=False)

    with dpg.child_window( tag="slith_tab_child", parent="main_window", show=False, no_scroll_with_mouse=False, horizontal_scrollbar=True, no_scrollbar=False, width=-1, height=-10, border=False,):

        with dpg.child_window(
            label="Slith Minigame",
            tag="slith_main_window",
            width=-1,
            height=-1,
            no_scrollbar=True,
            horizontal_scrollbar=True,
            no_scroll_with_mouse=True,
            border=True,
        ):
            change_font_type(dpg.last_item(), "regular", state)
            open_slith_window(state)
