"""
==============
load_job.py
==============

Directory and file selection utilities.

Handles dialog callbacks for choosing input/output directories or loading
previous analysis results. Updates application state, persists settings,
and rehydrates analysis windows when loading an existing job.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Directory selector
# 3. Load results
# 4. Load analysis windows text

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import re
import json
import dearpygui.dearpygui as dpg
from typing import Any
from app.utils.callbacks import close_current_job
from app.utils.app_logger import log_event, log_settings
from app.lmm.lmm_finalize import open_overview_tab
from app.gui.loading_win import delete_loading_screen, draw_loading_screen
from app.gui.themes_manager import change_font_type


# -----------------------------------------------------------------------------
# 2. Directory selector
# -----------------------------------------------------------------------------
def directory_selector(sender: Any, app_data: Any, state: dict[str, Any]) -> None:

    selected_dir = app_data["file_path_name"]
    handle_selected_directory(selected_dir, state)


def handle_selected_directory(selected_dir: str, state: dict[str, Any]) -> None:
    """
    Persist a selected directory and trigger the action associated with it.

    Args:
        selected_dir (str): Directory selected by the user.
        state (dict[str, Any]): Shared application state used to persist the
            selected path and route load-job actions.

    Returns:
        None: This function updates state and may trigger job loading.
    """
    if not selected_dir:
        return

    role = state.get("dialog_role")

    if role == "input":
        state["input_dir"] = selected_dir
        state["settings"]["input_directory"] = selected_dir

    elif role == "output":
        state["output_dir"] = selected_dir
        state["settings"]["results_directory"] = selected_dir

    elif role == "predictions":
        state["predictions_dir"] = selected_dir
        state["settings"]["predictions_directory"] = selected_dir

    elif role == "load_results":
        load_results(selected_dir, state)

    with open(state["settings_file"], "w", encoding="utf-8") as f:
        json.dump(state["settings"], f, indent=4)


# -----------------------------------------------------------------------------
# 3. Load results
# -----------------------------------------------------------------------------
def load_results(selected_dir: str, state: dict[str, Any]) -> None:

    try:
        
        report_path = os.path.join(selected_dir, "results.sof")
        if not os.path.exists(report_path):
            report_path = os.path.join(selected_dir, "results.srf")
        if os.path.exists(report_path):
            draw_loading_screen(state)
            selected_name = os.path.basename(os.path.normpath(selected_dir)) or selected_dir
            log_event("Input", f"Loading results folder '{selected_name}' ({selected_dir})", indent=1)
            log_settings("Input", indent=2, directory=selected_dir, report=os.path.basename(report_path))

            state["work_dir"] = selected_dir
            data = json.load(open(report_path))

            state["file_name"]           = data.get("file_name", "")
            state["molblocks_rgd_dict"]  = data.get("molblocks_rgd_dict", {})
            state["smiles_rgd_dict"]     = data.get("smiles_rgd_dict", {})
            state["bioact_types_dict"]   = data.get("bioact_types_dict", {})
            state["properties_dict"]     = data.get("properties_dict", {})
            state["properties_dict_full"] = data.get("properties_dict_full", {})
            state["total_r_groups_dict"] = data.get("total_r_groups_dict", {})
            state["r_counts"]            = data.get("r_counts", {})
            # state["checkbox_states"]     = data.get("settings", {})

            state["report_dir"]  = os.path.join(selected_dir, "reports")
            state["image_dir"]   = os.path.join(selected_dir, "images")
            state["subset_dir"]  = os.path.join(selected_dir, "sdf_files")
            state["summary_dir"] = os.path.join(selected_dir, "summary")
            state["selected_file"] = os.path.basename(os.path.normpath(selected_dir))

            dpg.set_viewport_title(f"SARgate - {os.path.basename(state['work_dir'])}")
            close_current_job(state)

            add_recent_file = state.get("add_recent_file")
            if callable(add_recent_file):
                add_recent_file(selected_dir)

            load_analysis_windows_text(state)
            open_overview_tab(state)

        else:
            delete_loading_screen(state)
            with dpg.window(
                label="File Error", min_size=(1,1), width=189, height=75, modal=True,
                no_title_bar=True, no_resize=True, no_move=True, no_collapse=True,
                no_scrollbar=True, pos=(23, 50), tag="File_error_popup_window"
            ):
                dpg.add_text("Results file not found.\nChoose a valid directory.")
                dpg.add_button(
                    label="OK", width=50, height=20, pos=(69, 45),
                    callback=lambda: dpg.delete_item("File_error_popup_window")
                )
            log_event("INPUT", "No results file found", indent=1, level="ERROR")

    except Exception as e:
        if selected_dir != "":
            delete_loading_screen(state)
            with dpg.window(
                label="File Error", min_size=(1,1), width=189, height=75, modal=True,
                no_title_bar=True, no_resize=True, no_move=True, no_collapse=True,
                no_scrollbar=True, pos=(23, 50), tag="File_error_popup_window"
            ):
                dpg.add_text(" Results file not found.\nChoose a valid directory.")
                dpg.add_button(
                    label="OK", width=50, height=20, pos=(69, 45),
                    callback=lambda: dpg.delete_item("File_error_popup_window")
                )
        log_exception("INPUT", "Loading ERROR", e, indent=1)


# -----------------------------------------------------------------------------
# 4. Load analysis windows text
# -----------------------------------------------------------------------------
def load_analysis_windows_text(state: dict[str, Any]) -> Any:
    """
    Rebuild analysis console windows from <work_dir>/results.sof.

    Args:
        state (dict): Application state dictionary.

    Returns:
        tuple[bool, str]: Success flag and informational message.
    """

    json_path = os.path.join(state["work_dir"], "results.sof")
    if not os.path.exists(json_path):
        json_path = os.path.join(state["work_dir"], "results.srf")
    if not os.path.exists(json_path):
        return False, f"JSON not found: {json_path}"

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception as e:
        return False, f"Failed to read JSON: {e}"

    # -----------------------------------------------------------------------------
    # 4.1. Rebuild window
    # -----------------------------------------------------------------------------
    def _rebuild_window(window_tag: str, entries: Any, add_spacer_before_sep: bool = False) -> Any:
        if not dpg.does_item_exist(window_tag):
            return 0
        try:
            dpg.delete_item(window_tag, children_only=True)
        except Exception:
            pass

        count = 0
        for i, entry in enumerate(entries or []):
            text_val = entry.get("text", "")
            sep = bool(entry.get("separator", False))

            dpg.add_text(text_val, parent=window_tag, tag=f"{window_tag}__text_{i}")

            if isinstance(text_val, str) and re.match(r"^\d+:", text_val.strip()):
                change_font_type(dpg.last_item(), "bold", state)

            count += 1

            if sep:
                if add_spacer_before_sep:
                    dpg.add_spacer(height=(state.get("win_spacer", 8) * 2), parent=window_tag)
                dpg.add_separator(parent=window_tag)
        return count

    prep_entries  = data.get("library_preparation_window_text", [])
    scaff_entries = data.get("scaffold_analysis_window_text", [])
    rga_entries   = data.get("rga_window_text", [])

    n_prep  = _rebuild_window("library_preparation_window", prep_entries,  add_spacer_before_sep=True)
    n_scaff = _rebuild_window("scaffold_analysis_window",   scaff_entries, add_spacer_before_sep=True)
    n_rga   = _rebuild_window("rga_window",                 rga_entries,   add_spacer_before_sep=False)

    total = n_prep + n_scaff + n_rga
    return True, f"Loaded texts (prep={n_prep}, scaff={n_scaff}, rga={n_rga}) from {json_path} (total={total})."
from app.utils.app_logger import log_event, log_exception
