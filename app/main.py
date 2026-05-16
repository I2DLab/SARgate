"""
=======
main.py
=======

Main application entry point of SARgate.

This module bootstraps the Dear PyGui application, loads persistent settings
from `assets/config/settings.ssf`, prepares the shared state dictionary, and
launches the core interface layers used throughout SARgate. It can be started
either directly as `python app/main.py` or indirectly through `launcher.py`,
which handles the splash screen in a separate startup path.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Define module configuration and shared state
# 3. Ensure directory settings
# 4. Run the module entry point

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import json
import copy
import sys
import warnings
import multiprocessing
import faulthandler
import shutil
from typing import Any

if sys.stderr is not None:
    faulthandler.enable()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    multiprocessing.freeze_support()

warnings.filterwarnings("ignore")


def _startup_trace(message: str) -> None:
    """
    Print an opt-in startup milestone for diagnosing native GUI crashes.

    Args:
        message (str): Milestone label to print.

    Returns:
        None: This helper writes to stderr only when tracing is enabled.
    """
    if os.environ.get("SARGATE_STARTUP_TRACE", "").strip():
        print(f"[SARgate startup] {message}", file=sys.stderr, flush=True)


_startup_trace("importing Dear PyGui")
import dearpygui.dearpygui as dpg
_startup_trace("importing SARgate GUI modules")
from app.gui.layout import (
    add_recent_file,
    load_recent_files,
    prepare_layout_resources,
    persist_layout_settings,
    register_startup_colormaps,
)
from app.gui.event_log import (
    ensure_event_log_window,
    install_event_log_capture,
    restore_event_log_capture,
)
from app.gui.viewport_size import get_screen_size, setup_viewport, setup_main_window
from app.gui.widgets_size import setup_widgets_size
from app.gui.themes_manager import (
    apply_theme_callback,
    custom_style_editor
)
from app.gui.initialize_gui import initialize_gui
from app.utils.navigation import setup_key_handlers
from app.utils.callbacks import poll_responsive_image_layout_changes, request_responsive_image_update
from app.utils.resource_paths import resource_path, user_data_path


# -----------------------------------------------------------------------------
# 2. Define module configuration and shared state
# -----------------------------------------------------------------------------

HOME_DIR = PROJECT_ROOT
os.chdir(PROJECT_ROOT)
USER_DATA_DIR = str(user_data_path())


def _config_file_path(filename: str) -> str:
    """
    Resolve a config file path, using writable per-user copies in frozen builds.
    """
    bundled_path = resource_path("assets", "config", filename)
    if not getattr(sys, "frozen", False):
        return str(bundled_path)

    runtime_path = user_data_path("config", filename)
    if not runtime_path.exists() and bundled_path.exists():
        shutil.copy2(bundled_path, runtime_path)
    return str(runtime_path)


SETTINGS_FILE = _config_file_path("settings.ssf")
THEMES_FILE = _config_file_path("themes.stf")
COLORMAPS_FILE = _config_file_path("colormaps.scf")
RECENT_FILES_FILE = _config_file_path("recent_files.srf")
with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
    settings = json.load(f)

legacy_theme = settings.pop("theme", None)
settings["tab_button_size"] = int(settings.get("tab_button_size", 30))
settings["show_tab_icons"] = bool(settings.get("show_tab_icons", True))


def _normalize_point_size_setting(value: Any) -> str:
    value_str = str(value or "Medium").strip().lower()
    if value_str == "small":
        return "Small"
    if value_str == "large":
        return "Large"
    return "Medium"


def _normalize_mcs_timeout_setting(value: Any) -> str:
    value_str = str(value or "10s").strip()
    allowed = {"10s", "30s", "60s", "120s", "300s", "Unlimited"}
    return value_str if value_str in allowed else "10s"


def _normalize_mcs_features_setting(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"false", "0", "no", "off"}:
        return False
    return True


settings["pca_point_size"] = _normalize_point_size_setting(settings.get("pca_point_size", "Medium"))
settings["umap_point_size"] = _normalize_point_size_setting(settings.get("umap_point_size", "Medium"))
settings["tsne_point_size"] = _normalize_point_size_setting(settings.get("tsne_point_size", "Medium"))
settings["pca_mcs_timeout"] = _normalize_mcs_timeout_setting(settings.get("pca_mcs_timeout", "10s"))
settings["umap_mcs_timeout"] = _normalize_mcs_timeout_setting(settings.get("umap_mcs_timeout", "10s"))
settings["tsne_mcs_timeout"] = _normalize_mcs_timeout_setting(settings.get("tsne_mcs_timeout", "10s"))
settings["pca_mcs_features"] = _normalize_mcs_features_setting(settings.get("pca_mcs_features", True))
settings["umap_mcs_features"] = _normalize_mcs_features_setting(settings.get("umap_mcs_features", True))
settings["tsne_mcs_features"] = _normalize_mcs_features_setting(settings.get("tsne_mcs_features", True))


settings["input_directory"] = os.path.expanduser(settings.get("input_directory", "") or "")
settings["results_directory"] = os.path.expanduser(settings.get("results_directory", "") or "")
settings["predictions_directory"] = os.path.expanduser(settings.get("predictions_directory", "") or "")


# -----------------------------------------------------------------------------
# 3. Ensure directory settings
# -----------------------------------------------------------------------------
def _ensure_dir_setting(settings_dict: dict[str, Any], key: str, default_folder: str) -> str:
    """
    Ensure that a configured directory exists and fall back to a local default.

    Args:
        settings_dict (dict[str, Any]): Settings dictionary to read from and
            update in place.
        key (str): Name of the settings entry that stores the directory path.
        default_folder (str): Folder name, relative to the project root, used
            when the configured path is missing or invalid.

    Returns:
        str: Absolute path to a valid directory for the requested setting.
    """
    path = settings_dict.get(key, "") or ""
    path = os.path.expanduser(path)
    if path and os.path.isdir(path):
        return path
    new_path = os.path.join(HOME_DIR, default_folder)
    os.makedirs(new_path, exist_ok=True)
    settings_dict[key] = new_path
    return new_path


def _signal_launcher_ready() -> None:
    """
    Notify the splash launcher that the main GUI is ready to replace it.

    Args:
        None.

    Returns:
        None: This helper writes the ready marker file when requested through
        the launcher environment.
    """
    ready_path = os.environ.get("SARGATE_READY_FILE", "").strip()
    if not ready_path:
        return

    try:
        with open(ready_path, "w", encoding="utf-8") as ready_file:
            ready_file.write("ready\n")
    except Exception:
        pass


input_dir = _ensure_dir_setting(settings, "input_directory", os.path.join("data", "input"))
output_dir = _ensure_dir_setting(settings, "results_directory", os.path.join("data", "output"))
predictions_dir = _ensure_dir_setting(settings, "predictions_directory", os.path.join("data", "predictions"))
LAYOUT = prepare_layout_resources(settings, THEMES_FILE, COLORMAPS_FILE, legacy_theme)
THEMES_STORE = LAYOUT["themes_store"]
THEMES = LAYOUT["themes"]
COLORMAPS_STORE = LAYOUT["colormaps_store"]
RECENT_FILES = load_recent_files(RECENT_FILES_FILE)


try:
    persist_layout_settings(SETTINGS_FILE, settings)
except Exception as e:
    print(f"Warning: could not update {SETTINGS_FILE}: {e}")


font = settings["font"]
font_scale = settings["font_scale"]
colormap_continuous = LAYOUT["colormap_continuous"]
colormap_discrete = LAYOUT["colormap_discrete"]
continuous_colormap_defs = LAYOUT["continuous_colormap_defs"]
discrete_colormap_defs = LAYOUT["discrete_colormap_defs"]
theme_name = LAYOUT["theme_name"]
theme = LAYOUT["theme"]


checkbox_states = {
    "Input source": "Local",
    "Database to search": "ChEMBL",
    "CHEMBL target ID": "",
    "Job name": "",
    "File extension": "",
    "Subsets collection method": "BMS minimal substructures (MinBMS)",
    "Scaffold SMILES": None,
    "Generalized Scaffold SMILES": None,
    "Scaffold Similarity threshold": 85,
    "Heavy atoms threshold": 9,
    "Populate light substructures with unassigned molecules only": True,
    "Similarity threshold": 85,
    "Filtering threshold": 2,
    "MCS timeout": "60s",
    "Filter by structure similarity": "No filters",
    "structure_similarity_input": "",
    "Structure for which to calculate similarity": "Entire molecule",
    "Input structure similarity threshold": 85,
    "Filter by target": False,
    "Target query 1": None,
    "Target query 2": None,
    "Enable ambiguous activities": True,
    "Filter by activity": False,
    "Activity query 1": None,
    "Activity query 2": None,
    "Filter by assay": False,
    "Assay query 1": None,
    "Assay query 2": None,
    "Duplicates handling": "Keep one entry with multiple activities",
}


MANUAL_SECTIONS = {
    "input_tab": "4_input-files-and-settings.html",
    "analysis_tab": "5_analysis-workflow.html",
    "overview_tab": "6_overview.html",
    "similarity_tab": "7_similarity.html",
    "counts_tab": "8_r-analysis.html",
    "stereo_tab": "9_stereo.html",
    "mmpa_tab": "10_mmpa.html",
    "plots_tab": "11_chemspace.html",
    "sar_notes_tab": "12_sar-notes.html",
    "prediction_tab": "13_prediction.html",
    "utilities_tab": "15_utilities.html",
}

state = {
    "Home_dir_path": HOME_DIR,
    "user_data_dir": USER_DATA_DIR,
    "manual_sections": MANUAL_SECTIONS,
    "is_fullscreen": False,
    "checkbox_states": checkbox_states,
    "settings_file": SETTINGS_FILE,
    "themes_file": THEMES_FILE,
    "colormaps_file": COLORMAPS_FILE,
    "recent_files_file": RECENT_FILES_FILE,
    "themes_store": THEMES_STORE,
    "colormaps_store": COLORMAPS_STORE,
    "recent_files": RECENT_FILES,
    "default_theme_names": tuple(THEMES_STORE.get("default_themes", {}).keys()),
    "settings": settings,
    "themes": THEMES,
    "theme": theme,
    "theme_name": theme_name,
    "colormap_continuous": colormap_continuous,
    "colormap_discrete": colormap_discrete,
    "pca_point_size": settings["pca_point_size"],
    "umap_point_size": settings["umap_point_size"],
    "tsne_point_size": settings["tsne_point_size"],
    "pca_point_size_mode": settings["pca_point_size"],
    "umap_point_size_mode": settings["umap_point_size"],
    "tsne_point_size_mode": settings["tsne_point_size"],
    "pca_mcs_timeout": settings["pca_mcs_timeout"],
    "umap_mcs_timeout": settings["umap_mcs_timeout"],
    "tsne_mcs_timeout": settings["tsne_mcs_timeout"],
    "pca_mcs_features": settings["pca_mcs_features"],
    "umap_mcs_features": settings["umap_mcs_features"],
    "tsne_mcs_features": settings["tsne_mcs_features"],
    "bold_texts": set(),
    "regular_texts": set(),
    "responsive_images": {},
    "img_popup_counter": 0,
    "current_tab": "input_tab",
    "current_overview_subtab": "overview_decomposition_subtab",
    "current_similarity_subtab": "similarity_matrix_subtab",
    "current_r_analysis_subtab": "r_analysis_counts_subtab",
    "current_chemspace_subtab": "descriptors_tab",
    "slith_ON": False,
    "monitor_width": 0,
    "monitor_height": 0,
    "design_ref_width": 0,
    "design_ref_height": 0,
    "screen_width": 0,
    "screen_height": 0,
    "viewport_max_width": 0,
    "viewport_max_height": 0,
    "win_spacer": 0,
    "analysis_tab_win_width": 0,
    "input_dir": input_dir,
    "output_dir": output_dir,
    "predictions_dir": predictions_dir,
    "selected_file_path": "",
    "selected_file_name": "",
    "prepared_sdf": "",
    "output_sdf": "",
    "work_dir": "",
    "image_dir": "",
    "report_dir": "",
    "summary_dir": "",
    "mmpa_dir": "",
    "subset_dir": "",
    "start_time": 0,
    "timer_thread": None,
    "timer_running": False,
    "smiles_column": "",
    "molname_column": "",
    "chembl_id_column": "",
    "pubchem_cid_column": "",
    "activity_mode": "single",
    "activity_columns": [],
    "assay_id_column": "",
    "assay_desc_column": "",
    "target_id_column": "",
    "target_protein_column": "",
    "target_organism_column": "",
    "max_phase_column": "",
    "comment_column": "",
    "_lib_df": None,
    "_lib_tex_cache": None,
    "_lib_page_size": 50,
    "_lib_total": 0,
    "scaffold_id": 0,
    "console_group_prep": None,
    "prep_log": [],
    "STEP_prep": 1,
    "scaffold_dict": {},
    "scaff_log": [],
    "STEP_scaff": 1,
    "bioact_types_dict": {},
    "activity_dict": {},
    "smiles_rgd_dict": {},
    "smiles_fails_dict": {},
    "total_r_groups_dict": {},
    "r_smiles_dict": {},
    "r_counts": {},
    "decomposition_dict": {},
    "cluster threshold": 0.2,
    "rga_log": [],
    "STEP_rga": 1,
    "console_group": None,
    "console_group_2": None,
    "console_group_3": None,
    "console_group_4": None,
    "console_group_5": None,
    "console_group_6": None,
    "console_group_7": None,
    "console_group_8": None,
    "last_clicked_window": None,
    "last_clicked_button_sub": None,
    "last_clicked_button_mol": None,
    "last_clicked_button_r": None,
    "last_clicked_combo": None,
    "overview_align": False,
    "overview_show_counts": True,
    "first_time_counts": True,
    "last_subset_counts": "subset_1",
    "last_r_counts": "R1",
    "isomers_groups": [],
    "activity_types": [
        "IC50", "EC50", "GI50", "AC50", "DC50",
        "CC50", "LD50", "ED50", "ID50",
        "pIC50", "pEC50", "pGI50", "pAC50", "pDC50",
        "pCC50", "pLD50", "pED50", "pID50",
        "k", "Ka", "Ki", "Kd", "Km", "Vmax",
        "pK", "pKa", "pKi", "pKd", "pKm",
        "Inhibition", "Activity", "Residualactivity", "%Control", "Fluintensity",
        "MIC", "MEC", "MMC", "MBC"
    ],
    "nM_activity_types": [
        "IC50", "EC50", "GI50", "AC50", "DC50",
        "CC50", "LD50", "ED50", "ID50",
        "k", "Ka", "Kd", "Km", "Ki"
    ],
    "dimensionless": [
        "pIC50", "pEC50", "pGI50", "pAC50", "pDC50",
        "pCC50", "pLD50", "pED50", "pID50",
        "pK", "pKa", "pKi", "pKd", "pKm",
        "fold", "fold_change", "log(fold_change)", "log_fold_change"
    ],
    "percent_activities": [
        "Inhibition", "Inihibition", "Activity", "Residualactivity", "ResidualActivity",
        "%Control", "%ofcontrol", "%Ctrl", "%", "Fluintensity"
    ],
    "ug/mL_activities": [
        "MIC", "MEC", "MMC", "MBC"
    ],
    "uM/min_activities": [
        "Vmax"
    ],
    "continuous_colormap_defs": continuous_colormap_defs,
    "discrete_colormap_defs": discrete_colormap_defs,
    "colormaps": {},
    "plot_colormaps": {},
    "plot_colormap_sizes": {name: len(colors) for name, colors in discrete_colormap_defs.items()},
    "slith_is_paused": False
}

state["add_recent_file"] = lambda path: add_recent_file(state, path)
install_event_log_capture(state)


def _build_state_backup(source_state: dict[str, Any]) -> dict[str, Any]:
    """
    Build a deepcopy-safe snapshot of the shared state for temporary restores.

    Args:
        source_state (dict[str, Any]): Live application state.

    Returns:
        dict[str, Any]: Snapshot excluding non-picklable runtime-only objects.
    """
    excluded_keys = {
        "_event_log_stdout_original",
        "_event_log_stderr_original",
    }
    safe_state = {
        key: value
        for key, value in source_state.items()
        if key not in excluded_keys
    }
    return copy.deepcopy(safe_state)
 

# -----------------------------------------------------------------------------
# 4. Run the module entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    _startup_trace("creating Dear PyGui context")
    dpg.create_context()


    _startup_trace("registering startup colormaps")
    register_startup_colormaps(state)

    _startup_trace("creating texture and handler registries")
    with dpg.texture_registry(tag="texture_registry"):
        pass

    with dpg.handler_registry(tag="handler_registry"):
        def _refresh_responsive_images_after_layout_input(*_: Any) -> None:
            """
            Refresh responsive images after user actions that can resize layouts.

            Returns:
                None: This routine updates the shared responsive-image state.
            """
            request_responsive_image_update(state, frames=4)
            poll_responsive_image_layout_changes(state)

        dpg.add_mouse_drag_handler(callback=_refresh_responsive_images_after_layout_input)
        dpg.add_mouse_release_handler(callback=_refresh_responsive_images_after_layout_input)
        dpg.add_mouse_click_handler(callback=_refresh_responsive_images_after_layout_input)
        dpg.add_mouse_wheel_handler(callback=_refresh_responsive_images_after_layout_input)


    _startup_trace("registering fonts")
    FONTS = {
        "Arial": ["assets/fonts/Arial.ttf", "assets/fonts/Arial-Bold.ttf"],
        "Arimo": ["assets/fonts/Arimo.ttf", "assets/fonts/Arimo-Bold.ttf"],
        "DejaVu Sans": ["assets/fonts/DejaVuSans.ttf", "assets/fonts/DejaVuSans-Bold.ttf"],
        "Ubuntu": ["assets/fonts/Ubuntu.ttf", "assets/fonts/Ubuntu-Bold.ttf"],
        "FiraCode (Mono)": ["assets/fonts/FiraCode.otf", "assets/fonts/FiraCode-Bold.otf"]
    }
    with dpg.font_registry(tag="font_registry"):
        for font_name, font_paths in FONTS.items():
            for font_path in font_paths:
                with dpg.font(font_path, 14, tag=f"{font_name} Bold" if "-Bold" in font_path else font_name):
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                    dpg.add_font_range(0x0001, 0xFFFF)  # Covers the full Unicode range.
                if font_name == "FiraCode (Mono)":
                    large_tag = "FiraCode (Mono) Bold Large" if "-Bold" in font_path else "FiraCode (Mono) Large"
                    large_size = 18 if "-Bold" in font_path else 16
                    with dpg.font(font_path, large_size, tag=large_tag):
                        dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                        dpg.add_font_range(0x0001, 0xFFFF)


    dpg.bind_font(font)
    state["applied_font"] = font
    dpg.set_global_font_scale(font_scale)
 

    _startup_trace("capturing screen metrics")
    get_screen_size(state)


    _startup_trace("creating viewport")
    setup_viewport(state, HOME_DIR, title="SARgate | Chemical Space & SAR Analysis")


    _startup_trace("creating main window")
    setup_main_window(state)


    _startup_trace("computing widget sizes")
    setup_widgets_size(state)


    _startup_trace("building GUI hierarchy")
    initialize_gui(state)
    ensure_event_log_window(state)


    def start_responsive_image_poller(state: dict[str, Any]) -> None:
        """
        Schedule a lightweight polling loop for responsive image layout changes.

        Args:
            state (dict[str, Any]): Shared application state used by responsive
                image callbacks.

        Returns:
            None: This helper schedules frame callbacks and updates the UI in
            place.
        """
        def _tick(*_: Any) -> None:
            """
            Refresh responsive images when their measured containers change.

            Args:
                *_ (Any): Dear PyGui frame-callback payloads. Included for
                    compatibility.

            Returns:
                None: This callback performs UI updates in place.
            """
            poll_responsive_image_layout_changes(state)
            dpg.set_frame_callback(dpg.get_frame_count() + 2, _tick)
        request_responsive_image_update(state, frames=6)
        dpg.set_frame_callback(dpg.get_frame_count() + 1, _tick)

    start_responsive_image_poller(state)


    _startup_trace("applying theme")
    apply_theme_callback("change_theme", theme_name, state)


    _startup_trace("preparing style editor")
    state["state_backup"] = _build_state_backup(state)
    custom_style_editor(state)
    dpg.hide_item("custom_style_editor")


    _startup_trace("registering keyboard handlers")
    setup_key_handlers(state)


    _startup_trace("signaling launcher readiness")
    _signal_launcher_ready()


    _startup_trace("starting Dear PyGui event loop")
    dpg.start_dearpygui()
    restore_event_log_capture(state)
    dpg.destroy_context()
