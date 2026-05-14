"""
============================
lmm_input_and_settings.py
============================

Builds the library input and settings panels for the SAR workflow.

This module renders the controls used to choose local or remote inputs, define
library-preparation options, and configure the main filtering and clustering
parameters required before starting an analysis.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Build the input selection panel
# 3. Build the settings panel

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import dearpygui.dearpygui as dpg
from typing import Any
from rdkit import Chem
from app.lmm.lmm_column_selector import show_csv_column_selector
from app.utils.callbacks import (
    update_checkbox_state,
    show_library_table_confirm_popup
)
from app.gui.themes_manager import (
    apply_bordered_input_text_theme,
    change_font_type
)
from app.utils.native_dialogs import open_file_dialog
from app.lmm.lmm_workflow import confirm_and_run
from app.lmm.lmm_abort import confirm_cancellation


def _recreate_theme_tag(tag: str) -> None:
    """
    Delete an existing theme tag so it can be rebuilt with fresh colours.
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


def refresh_input_selection_themes(state: dict[str, Any]) -> dict[str, str]:
    """
    Recreate and rebind the local themes used by the input/settings workflow widgets.
    """
    theme_dict = state["themes"][state["theme_name"]]

    theme_specs = {
        "workflow_primary_button_theme": (
            theme_dict["Tabs Active"],
            theme_dict["Button Hovered"],
            theme_dict["Button Active"],
        ),
        "workflow_danger_button_theme": (
            (170, 56, 56, 220),
            (205, 72, 72, 255),
            (150, 42, 42, 255),
        ),
        "workflow_secondary_button_theme": (
            theme_dict["Button Color"],
            theme_dict["Tabs Hovered"],
            theme_dict["Tabs Active"],
        ),
        "workflow_select_file_button_theme": (
            theme_dict["Button Color"],
            theme_dict["Tabs Hovered"],
            theme_dict["Tabs Active"],
        ),
    }

    for tag, (base, hover, active) in theme_specs.items():
        _recreate_theme_tag(tag)
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, base, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hover, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, active, category=dpg.mvThemeCat_Core)
                border_color = theme_dict["Tabs Active"] if tag == "workflow_select_file_button_theme" else (0, 0, 0, 0)
                border_size = 1 if tag == "workflow_select_file_button_theme" else 0
                dpg.add_theme_color(dpg.mvThemeCol_Border, border_color, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, border_size)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, max(6, int(theme_dict["Frame rounding"])))
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 7)

    _recreate_theme_tag("workflow_timer_theme")
    with dpg.theme(tag="workflow_timer_theme"):
        with dpg.theme_component(dpg.mvButton):
            transparent = (0, 0, 0, 0)
            text_color = theme_dict["Title Bar Background"]
            dpg.add_theme_color(dpg.mvThemeCol_Button, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, transparent, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, text_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, text_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0)

    text_themes = {
        "workflow_status_text_theme": theme_dict["Title Bar Background"],
        "workflow_file_name_text_theme": theme_dict["Text Color"],
        "workflow_muted_text_theme": theme_dict["Border Color"],
    }
    for tag, color in text_themes.items():
        _recreate_theme_tag(tag)
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, color, category=dpg.mvThemeCat_Core)

    item_theme_map = {
        "select_file_button": "workflow_select_file_button_theme",
        "load_results_button": "workflow_select_file_button_theme",
        "execution_time_label": "workflow_timer_theme",
        "file_name_text": "workflow_file_name_text_theme",
    }
    for item_tag, theme_tag in item_theme_map.items():
        if dpg.does_item_exist(item_tag):
            dpg.bind_item_theme(item_tag, theme_tag)

    if dpg.does_item_exist("Job name"):
        dpg.bind_item_theme("Job name", apply_bordered_input_text_theme(state))
    if dpg.does_item_exist("CHEMBL target ID"):
        dpg.bind_item_theme("CHEMBL target ID", apply_bordered_input_text_theme(state))

    return {
        "primary_button_theme": "workflow_primary_button_theme",
        "danger_button_theme": "workflow_danger_button_theme",
        "secondary_button_theme": "workflow_secondary_button_theme",
        "timer_theme": "workflow_timer_theme",
        "status_text_theme": "workflow_status_text_theme",
        "file_name_text_theme": "workflow_file_name_text_theme",
        "muted_text_theme": "workflow_muted_text_theme",
    }


# -----------------------------------------------------------------------------
# 2. Build the input selection panel
# -----------------------------------------------------------------------------
def show_input_selection_window(state: dict[str, Any]) -> None:
    """
    Build the panel used to select the input source and launch the workflow.

    Args:
        state (dict[str, Any]): Shared application state used to configure the
            widgets and persist the selected input metadata.

    Returns:
        None: This function creates Dear PyGui widgets and updates state in
        place through callbacks.
    """
    local_themes = refresh_input_selection_themes(state)
    primary_button_theme = local_themes["primary_button_theme"]
    danger_button_theme = local_themes["danger_button_theme"]
    secondary_button_theme = local_themes["secondary_button_theme"]
    timer_theme = local_themes["timer_theme"]
    status_text_theme = local_themes["status_text_theme"]
    file_name_text_theme = local_themes["file_name_text_theme"]
    muted_text_theme = local_themes["muted_text_theme"]
    # -----------------------------------------------------------------------------
    # 2.1. File selector callback
    # -----------------------------------------------------------------------------
    def file_selector_callback(selected_file: str, state: dict[str, Any]) -> None:
        """
        Handle the file selected in the native input file dialog.

        Args:
            selected_file (str): Absolute path to the selected input file.
            state (dict[str, Any]): Shared application state updated with the
                selected file information.

        Returns:
            None: This callback updates the UI and state in place.
        """

        selected_file_name = os.path.basename(selected_file)
        extension = os.path.splitext(selected_file)[1].lower()
        state["selected_file_path"] = selected_file
        state["selected_file_name"] = selected_file_name
        add_recent_file = state.get("add_recent_file")
        if callable(add_recent_file):
            add_recent_file(selected_file)

        # Update label with the chosen filename for user feedback.
        dpg.set_value("file_name_text", selected_file_name)
        
        # Store the extension in checkbox_states and reveal the "Show Library Table" button for valid inputs.
        if extension in [".sdf", ".csv", ".tsv", ".xlsx", ".smi", ".txt"]:
            state["checkbox_states"]["File extension"] = extension[1:]
            

        # For CSV/TSV/XLSX, we schedule the column selection popup for the next frame.
        if extension in [".csv", ".tsv", ".xlsx"]:
            def _show_column_selector(*_: Any) -> None:
                """
                Open the column-selector popup for the selected tabular file.

                Args:
                    *_ (Any): Optional Dear PyGui callback payload accepted for
                        compatibility with frame callbacks.

                Returns:
                    None: This callback opens the column-selector popup.
                """
                show_csv_column_selector(state, selected_file)

            try:
                _show_column_selector()
            except Exception:
                dpg.set_frame_callback(dpg.get_frame_count() + 1, _show_column_selector)
        else:
            def _show_confirm(_: Any = None) -> None:
                """
                Open the library-table confirmation popup on the next frame.

                Args:
                    _ (Any, optional): Unused frame-callback payload accepted
                        for Dear PyGui compatibility.

                Returns:
                    None: This callback triggers a UI popup and returns.
                """
                show_library_table_confirm_popup(state)

            try:
                _show_confirm()
            except Exception:
                dpg.set_frame_callback(dpg.get_frame_count() + 1, _show_confirm)

    state["open_input_file_from_path"] = lambda path: file_selector_callback(path, state)
        

    # -----------------------------------------------------------------------------
    # 2.2. Show file dialog
    # -----------------------------------------------------------------------------
    def show_file_dialog(state: dict[str, Any]) -> None:
        """
        Open the native file dialog used to choose a local input file.

        Args:
            state (dict[str, Any]): Shared application state containing dialog
                sizing and default input-directory information.

        Returns:
            None: This function opens the operating-system file dialog.
        """
        selected_file = open_file_dialog(
            title="Select Input File",
            default_path=state["input_dir"],
            file_types=[
                ("Supported files", "*.sdf *.csv *.tsv *.xlsx *.smi *.txt"),
                ("SDF files", "*.sdf"),
                ("CSV files", "*.csv"),
                ("TSV files", "*.tsv"),
                ("Excel files", "*.xlsx"),
                ("SMILES files", "*.smi *.txt"),
                ("All files", "*.*"),
            ],
        )
        if selected_file:
            file_selector_callback(selected_file, state)


    # -----------------------------------------------------------------------------
    # 2.3. Update input selection
    # -----------------------------------------------------------------------------
    def update_input_selection(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
        """
        Toggle input-source widgets after the radio-button selection changes.

        Args:
            sender (Any): Dear PyGui callback sender. Included for callback
                compatibility.
            app_data (str): Selected input-source label.
            state (dict[str, Any]): Shared application state updated with the
                active source and selected file metadata.

        Returns:
            None: This callback updates widget visibility and state in place.
        """
        # Toggle the relevant widgets and record the choice in state["checkbox_states"].
        if app_data == "Create from Local":
            dpg.configure_item("select_file_button", show=True)
            dpg.configure_item("load_results_button", show=True)
            dpg.configure_item("Database_to_search_group", show=False)
            dpg.configure_item("chembl_input_group", show=False)
            dpg.configure_item("file_name_text", show=True)
            state["checkbox_states"]["Input source"] = "Local"
        elif app_data == "Create from Database":
            dpg.configure_item("select_file_button", show=False)
            dpg.configure_item("load_results_button", show=False)
            dpg.configure_item("Database_to_search_group", show=True)
            dpg.configure_item("chembl_input_group", show=True)
            dpg.set_value("file_name_text", "No selected file")
            dpg.configure_item("file_name_text", show=False)
            state["checkbox_states"]["Input source"] = "Database"
            state["selected_file_path"] = ""
            state["selected_file_name"] = ""


    with dpg.child_window(parent="file_selection_window", width=-1, height=-1,
        auto_resize_x=True, auto_resize_y=True,
        no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False
        ):

        dpg.add_spacer(height=state["win_spacer"])

        dpg.add_separator(label="INPUT SELECTION")

        # Input source radio: local filesystem vs remote database build.
        dpg.add_radio_button(items=["Create from Local", "Create from Database"], default_value="Create from Local", 
                                horizontal=False, callback=update_input_selection, user_data=state)
                
        # Group the left-side selectors and the right-side run/stop widgets.
        with dpg.group(horizontal=True):
            with dpg.group():

                # Entry point to choose a local input file or a results file
                with dpg.table(
                    header_row=False,
                    policy=dpg.mvTable_SizingStretchProp,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    row_background=False,
                ):                    
                    dpg.add_table_column(init_width_or_weight=50)
                    dpg.add_table_column(init_width_or_weight=50)

                    with dpg.table_row():

                        dpg.add_button(
                            label="Choose Input File",
                            tag="select_file_button",
                            width=-1,
                            callback=lambda: show_file_dialog(state),
                        )
                        dpg.bind_item_theme("select_file_button", "workflow_select_file_button_theme")

                        dpg.add_button(
                            label="Load Results File",
                            tag="load_results_button",
                            width=-1,
                            callback=lambda: state["show_dialog"]("load_results", state),
                        )
                        dpg.bind_item_theme("load_results_button", "workflow_select_file_button_theme")

                # Database selector for automated dataset creation.
                with dpg.group(horizontal=True, tag="Database_to_search_group", show=False, width=-1):
                    dpg.add_text("Search on:", tag="search_on_label")
                    dpg.add_combo(tag="search_on_database_combo", width=-1,
                                items=["ChEMBL", "PubChem", "ChEMBL and PubChem"], default_value="ChEMBL",
                                callback=update_checkbox_state, user_data=("Database to search", state))

                # Target identifier (ChEMBL ID or UniProt Accession for PubChem searches).
                with dpg.group(horizontal=True, tag="chembl_input_group", show=False, width=-1):
                    dpg.add_text("Target ID:", tag="target_id_label")
                    dpg.add_input_text(hint="e.g. CHEMBLXXXX", tag="CHEMBL target ID", width=-1,
                                    callback=update_checkbox_state, user_data=state)
                    dpg.bind_item_theme("CHEMBL target ID", apply_bordered_input_text_theme(state))
                    with dpg.tooltip(parent="chembl_input_group", tag="chembl_input_tooltip", delay=0.5):
                        dpg.add_text("Enter a ChEMBL Target ID (valid for both ChEMBL and PubChem),\n"
                                    "or an Uniprot Accession ID (only for searching on PubChem).")

                # Show the chosen local filename (or default message).
                dpg.add_text("No selected file", tag="file_name_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.bind_item_theme("file_name_text", file_name_text_theme)
                if dpg.does_item_exist("FiraCode (Mono) Large"):
                    dpg.bind_item_font("file_name_text", "FiraCode (Mono) Large")
                elif dpg.does_item_exist("FiraCode (Mono)"):
                    dpg.bind_item_font("file_name_text", "FiraCode (Mono)")


    with dpg.child_window(parent="start_analysis_window",
        auto_resize_x=True, auto_resize_y=True, width=-1, height=-1,
        no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False
        ):
        
        dpg.add_spacer(height=state["win_spacer"])

        dpg.add_separator(label="RUN")

        dpg.add_input_text(
            width=-1,
            tag="Job name",
            hint="Name of the results folder (optional)",
            callback=update_checkbox_state, user_data=state)
        dpg.bind_item_theme("Job name", apply_bordered_input_text_theme(state))

        with dpg.table(
            header_row=False,
            policy=dpg.mvTable_SizingStretchProp,
            width=-1,
            borders_innerH=False,
            borders_outerH=False,
            borders_innerV=False,
            borders_outerV=False,
            row_background=False,
        ):
            dpg.add_table_column(init_width_or_weight=20)
            dpg.add_table_column(init_width_or_weight=60)
            dpg.add_table_column(init_width_or_weight=20)

            with dpg.table_row():
                with dpg.table_cell():
                    pass
                with dpg.table_cell():
                    # RUN analysis; the callback will validate and start the workflow.
                    dpg.add_button(label="\nSTART\n ", tag="confirm_button", width=-1, callback=lambda: confirm_and_run(state))
                    dpg.bind_item_theme("confirm_button", primary_button_theme)
                    change_font_type(dpg.last_item(), "bold", state)

                    # Stop button (hidden until analysis starts); actual stop is confirmed via popup.
                    dpg.add_button(label="\nABORT\n ", tag="stop_button", width=-1, show=False)
                    dpg.bind_item_theme("stop_button", danger_button_theme)

                    dpg.add_button(label="Execution time", tag="execution_time_label", width=-1, callback=lambda: None)
                    dpg.bind_item_theme("execution_time_label", timer_theme)
                    if dpg.does_item_exist("FiraCode (Mono) Large"):
                        dpg.bind_item_font("execution_time_label", "FiraCode (Mono) Large")
                    elif dpg.does_item_exist("FiraCode (Mono)"):
                        dpg.bind_item_font("execution_time_label", "FiraCode (Mono)")

                    dpg.add_button(label="00:00:00", tag="execution_time_text", width=-1, callback=lambda: None)
                    dpg.bind_item_theme("execution_time_text", timer_theme)
                    if dpg.does_item_exist("FiraCode (Mono) Bold Large"):
                        dpg.bind_item_font("execution_time_text", "FiraCode (Mono) Bold Large")
                    elif dpg.does_item_exist("FiraCode (Mono) Bold"):
                        dpg.bind_item_font("execution_time_text", "FiraCode (Mono) Bold")

                    theme_dict = state["themes"][state["theme_name"]]
             
                with dpg.table_cell():
                    pass


        # Ask confirmation before stopping and cleaning temporary data.
        with dpg.popup("stop_button", tag="cancel_confirm_popup", modal=False, mousebutton=dpg.mvMouseButton_Left):
            dpg.add_text("Do you really want to interrupt the\nanalysis and delete temporary data?")
            dpg.add_spacer(height=state["win_spacer"] * 2)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Confirm", tag="cancel_confirm_yes_button", callback=lambda: confirm_cancellation(state))
                dpg.bind_item_theme("cancel_confirm_yes_button", danger_button_theme)
                dpg.add_button(label="Cancel", tag="cancel_confirm_no_button", callback=lambda: dpg.configure_item("cancel_confirm_popup", show=False))
                dpg.bind_item_theme("cancel_confirm_no_button", secondary_button_theme)


# -----------------------------------------------------------------------------
# 3. Build the settings panel
# -----------------------------------------------------------------------------
def show_settings_window(state: dict[str, Any]) -> None:
    """
    Build the settings panel used to configure library preparation options.

    Args:
        state (dict[str, Any]): Shared application state used to populate the
            current option values and callback bindings.

    Returns:
        None: This function creates Dear PyGui widgets and updates state in
        place through callbacks.
    """
    with dpg.child_window(parent="options_window", width=-1, height=-1,
        no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False
        ):

        spacer_height = state["win_spacer"]
        checkbox_states = state["checkbox_states"]

        if not dpg.does_item_exist("settings_groups_table_theme"):
            with dpg.theme(tag="settings_groups_table_theme"):
                with dpg.theme_component(dpg.mvTable):
                    dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 5, 0, category=dpg.mvThemeCat_Core)

        with dpg.table(
            header_row=False,
            resizable=True,
            policy=dpg.mvTable_SizingStretchProp,
            borders_innerV=True,
            borders_outerV=False,
            borders_innerH=False,
            borders_outerH=False,
            row_background=False,
            tag="settings_groups_table",
        ):
            dpg.bind_item_theme("settings_groups_table", "settings_groups_table_theme")
            dpg.add_table_column(init_width_or_weight=20)
            dpg.add_table_column(init_width_or_weight=40)
            dpg.add_table_column(init_width_or_weight=40)

            with dpg.table_row():
                with dpg.table_cell():
                    with dpg.child_window(tag="Duplicates handling group", border=False, autosize_x=True, height=-1):
                        dpg.add_spacer(height=state["win_spacer"])
                        dpg.add_separator(label="DUPLICATES HANDLING")
                        change_font_type(dpg.last_item(), "bold", state)

                        duplicates_handling_list = ["Keep one entry with\nmultiple activities", "Keep one entry with\nthe best activity", "Keep one entry with\naverage activities", "Keep duplicates\nseparated"]
                        dpg.add_radio_button(
                            items=duplicates_handling_list,
                            tag="Duplicates handling",
                            default_value=checkbox_states["Duplicates handling"],
                            horizontal=False,   
                            callback=update_checkbox_state, user_data=state
                        )
                        with dpg.tooltip(tag="Duplicates handling tooltip", parent="Duplicates handling group",delay=0.5):
                            dpg.add_text("""1 - Keep only one entry with multiple activities: 
    duplicates will be merged into a single entry, retaining all associated activities.
                            
2 - Keep one entry with the best activity: 
    keep only the best activity (for each activity type).
                            
3 - Keep one entry with average activities:
    keep a single entry with the average of all activities (for each activity type).
                            
4 - Keep duplicates separated:
    all duplicate entries will be retained as separate records.""")


                with dpg.table_cell():
                    with dpg.child_window(tag="Dataset filtering group", border=False, autosize_x=True, height=-1):
                        dpg.add_spacer(height=state["win_spacer"])
                        dpg.add_separator(label="DATASET FILTERING")
                        change_font_type(dpg.last_item(), "bold", state)

                        dpg.add_text("STRUCTURE SIMILARITY FILTER")
                        change_font_type(dpg.last_item(), "bold", state)

                        def struct_sim_filter_callback(_: Any, app_data: Any, user_data: Any) -> None:
                            """
                            Update the structure-similarity mode and related widget visibility.

                            Args:
                                _ (Any): Dear PyGui callback sender placeholder. Included
                                    for callback compatibility.
                                app_data (str): Selected structure-similarity filter mode.
                                user_data (dict[str, Any]): Shared application state passed
                                    to the checkbox update helper.

                            Returns:
                                None: This callback updates widget visibility and state in
                                place.
                            """
                            update_checkbox_state("Filter by structure similarity", app_data, user_data)

                            for tag in [
                                "struct_similarity_scaffold_group",
                                "struct_to_apply_similarity_group",
                                "struct_similarity_threshold_group",
                                "struct_similarity_threshold_tooltip"
                            ]:
                                if dpg.does_item_exist(tag):
                                    dpg.hide_item(tag)

                            if app_data != "No filters":
                                dpg.show_item("struct_similarity_scaffold_group")
                                dpg.show_item("struct_to_apply_similarity_group")
                                dpg.show_item("struct_similarity_threshold_group")
                                dpg.show_item("struct_similarity_threshold_tooltip")


                        with dpg.group(horizontal=True, tag="struct_similarity_filter_group"):
                            dpg.add_text("Filter type:", tag="filter_type_label")
                            struct_sim_items = [
                                "No filters",
                                "Filter by exact structure similarity (atom types and connectivity)",
                                "Filter by generalized structure similarity (connectivity only)"
                            ]

                            dpg.add_combo(
                                items=struct_sim_items,
                                tag="Filter by structure similarity",
                                width=-1,
                                default_value=checkbox_states["Filter by structure similarity"],
                                callback=struct_sim_filter_callback,
                                user_data=state,
                            )

                            with dpg.tooltip(parent="struct_similarity_filter_group", 
                                            tag="struct_similarity_filter_tooltip", delay=0.5):
                                dpg.add_text("""Choose how to filter molecules by similarity to the input structure:

1 - No filters:
    The dataset is not filtered based on structural similarity.

2 - Filter by exact structure similarity (atom types + connectivity):
    The structure is used exactly as provided (SMILES / SMARTS / InChI / MolBlock).
    Similarity is computed using fingerprints that encode both atom types and bond connectivity.

3 - Filter by generalized structure similarity (connectivity only):
    Both the structure and dataset molecules are converted to a generalized 
    topological form. Similarity is computed on this simplified representation, 
    ignoring atom types and focusing on the underlying connectivity pattern.
""")


                        def validate_struct_similarity_scaffold() -> None:
                            """
                            Validate struct similarity scaffold.
                            
                            Args:
                                None.
                            
                            Returns:
                                None: This routine performs in-place updates or side effects only.
                            """
                            raw_input = dpg.get_value("structure_similarity_input")
                            if raw_input is None or raw_input.strip() == "":
                                dpg.set_value("struct_similarity_validity_text", "Input structure is empty.")
                                dpg.configure_item("struct_similarity_validity_popup", show=True)
                                return

                            scaffold = raw_input

                            msg = None
                            try:
                                if Chem.MolFromSmiles(scaffold):
                                    msg = "Valid SMILES"
                                else:
                                    raise ValueError()
                            except Exception:
                                try:
                                    if Chem.MolFromSmarts(scaffold):
                                        msg = "Valid SMARTS"
                                    else:
                                        raise ValueError()
                                except Exception:
                                    try:
                                        if Chem.MolFromInchi(scaffold):
                                            msg = "Valid InChI"
                                        else:
                                            raise ValueError()
                                    except Exception:
                                        try:
                                            if Chem.MolFromMolBlock(scaffold):
                                                msg = "Valid MolBlock"
                                            else:
                                                raise ValueError()
                                        except Exception:
                                            msg = "Not a valid structure format"

                            dpg.set_value("struct_similarity_validity_text", msg)
                            dpg.configure_item("struct_similarity_validity_popup", show=True)


                        # --- Input scaffold + button + tooltip ---
                        with dpg.group(tag="struct_similarity_scaffold_group", show=False):
                            dpg.add_input_text(
                                tag="structure_similarity_input",
                                hint="Type structure SMILES/SMARTS/InChI/MolBlock",
                                multiline=True,
                                width=150,
                                callback=update_checkbox_state,
                                user_data=("structure_similarity_input", state)
                            )
                            dpg.bind_item_theme("structure_similarity_input", apply_bordered_input_text_theme(state))

                            dpg.add_button(label="Test structure",
                                        tag="struct_similarity_test_button",
                                        callback=validate_struct_similarity_scaffold)

                            with dpg.window(tag="struct_similarity_validity_popup",
                                            modal=True, show=False, no_close=True, no_resize=True,
                                            no_move=True, no_collapse=True, no_title_bar=True,
                                            autosize=True, min_size=(10, 10)):
                                dpg.add_text("", tag="struct_similarity_validity_text")
                                dpg.add_spacer(height=state["win_spacer"])
                                dpg.add_button(label="OK",
                                            callback=lambda: dpg.configure_item("struct_similarity_validity_popup", show=False))


                        # --- Compute similarity for entire molecules or Murcko scaffolds ---
                        with dpg.group(horizontal=True, tag="struct_to_apply_similarity_group", show=False):
                            dpg.add_text("Structure for which to calculate similarity")
                            struct_type_items = [
                                "Entire molecule",
                                "Molecule's Murcko scaffold"
                            ]

                            dpg.add_combo(
                                items=struct_type_items,
                                tag="Structure for which to calculate similarity",
                                width=150,
                                default_value=checkbox_states["Structure for which to calculate similarity"],
                                callback=update_checkbox_state,
                                user_data=("Structure for which to calculate similarity", state)
                            )

                            with dpg.tooltip(parent="struct_to_apply_similarity_group", 
                                            tag="struct_to_apply_similarity_tooltip", delay=0.5):
                                dpg.add_text("""Define the structural representation used for similarity calculations:
1 - Entire molecule:
    Similarity is computed using for the full molecular structure as provided in the dataset, versus the input structure.

2 - Molecule's Murcko scaffold:
    Similarity is computed using only the Bemis-Murcko scaffold of each molecule versus the input structure."""
                            )

                        # --- Threshold (0–100%) ---
                        with dpg.group(horizontal=True, tag="struct_similarity_threshold_group", show=False):
                            dpg.add_text("Input structure similarity threshold (%)")
                            dpg.add_input_int(
                                tag="Input structure similarity threshold",
                                min_clamped=True, max_clamped=True,
                                min_value=0, max_value=100,
                                default_value=checkbox_states["Input structure similarity threshold"],
                                width=150,
                                callback=update_checkbox_state,
                                user_data=state
                            )

                            with dpg.tooltip(parent="struct_similarity_threshold_group", 
                                            tag="struct_similarity_threshold_tooltip", delay=0.5):
                                dpg.add_text("""Define the minimum similarity percentage required
to keep a molecule in the filtered dataset (0-100%).""")


                        dpg.add_spacer(height=spacer_height)
                        dpg.add_text("TARGET FILTER")
                        change_font_type(dpg.last_item(), "bold", state)

                        def target_input_text(_: Any, __: Any, user_data: Any) -> None:
                            """
                            Toggle target filtering and manage the dynamic target inputs.

                            Args:
                                _ (Any): Dear PyGui callback sender placeholder. Included
                                    for callback compatibility.
                                __ (Any): Dear PyGui callback payload placeholder. Included
                                    for callback compatibility.
                                user_data (str): Checkbox tag associated with the target
                                    filter option.

                            Returns:
                                None: This callback updates target-filter widgets and state
                                in place.
                            """

                            def show_target_2_if_needed(sender: Any, app_data: Any, user_data: Any) -> None:
                                """
                                Show or hide the second target input based on the first one.

                                Args:
                                    sender (Any): Dear PyGui callback sender.
                                    app_data (str): Current value of the first target input.
                                    user_data (tuple[str, dict[str, Any]]): Pair containing
                                        the state key and shared application state.

                                Returns:
                                    None: This callback updates widget visibility and state
                                    in place.
                                """
                                # Store the current text and keep state synched.
                                update_checkbox_state(sender, app_data, user_data)

                                # If non-empty, show Target 2; otherwise remove it to keep UI tidy.
                                if app_data.strip() != "":
                                    if not dpg.does_item_exist("target_group_2"):
                                        with dpg.group(tag="target_group_2", parent="targets_group"):
                                            dpg.add_text("or", indent=state["analysis_tab_options_target_indent"])
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Target 2:", indent=state["analysis_tab_options_target_indent"])
                                                dpg.add_input_text(
                                                    tag="target_query_2", 
                                                    hint="e.g. Reverse transcriptase",
                                                    width=-1,
                                                    callback=update_checkbox_state,
                                                    user_data=("Target query 2", state))
                                                dpg.bind_item_theme("target_query_2", apply_bordered_input_text_theme(state))
                                else:
                                    if dpg.does_item_exist("target_group_2"):
                                        dpg.delete_item("target_group_2")

                            # Reflect the checkbox toggle into state.
                            update_checkbox_state(_, __, state)

                            # Create or delete the first/second target input groups depending on the toggle.
                            if dpg.get_value(user_data):
                                with dpg.group(horizontal=True, tag="target_group_1", parent="targets_group"):
                                    dpg.add_text("Target 1:", indent=state["analysis_tab_options_target_indent"])
                                    dpg.add_input_text(
                                        tag="target_query_1", 
                                        width=-1,
                                        hint="e.g. Cyclooxygenase-2",
                                        callback=show_target_2_if_needed,
                                        user_data=("Target query 1", state))
                                    dpg.bind_item_theme("target_query_1", apply_bordered_input_text_theme(state))
                            else:
                                for tag in ["target_group_1", "target_group_2"]:
                                    if dpg.does_item_exist(tag):
                                        dpg.delete_item(tag)


                        # Container for 'Filter by target' checkbox and its dynamic inputs.
                        with dpg.group(tag="targets_group"):

                            # Toggle target-based filtering and show context tooltip.
                            dpg.add_checkbox(
                                label="Filter by target", tag="Filter by target", 
                                default_value=checkbox_states["Filter by target"],
                                callback=target_input_text, user_data="Filter by target"
                                )
                            with dpg.tooltip(parent="targets_group", delay=0.5):
                                dpg.add_text("Filter the dataset by one or two biological targets")


                        dpg.add_spacer(height=spacer_height)
                        dpg.add_text("ACTIVITY FILTER")
                        change_font_type(dpg.last_item(), "bold", state)

                        def activity_input_text(_: Any, __: Any, user_data: Any) -> None:
                            """
                            Toggle activity filtering and manage the dynamic activity inputs.

                            Args:
                                _ (Any): Dear PyGui callback sender placeholder. Included
                                    for callback compatibility.
                                __ (Any): Dear PyGui callback payload placeholder. Included
                                    for callback compatibility.
                                user_data (str): Checkbox tag associated with the activity
                                    filter option.

                            Returns:
                                None: This callback updates activity-filter widgets and
                                state in place.
                            """

                            def show_activity_2_if_needed(sender: Any, app_data: Any, user_data: Any) -> None:
                                """
                                Show or hide the second activity input based on the first one.

                                Args:
                                    sender (Any): Dear PyGui callback sender.
                                    app_data (str): Current value of the first activity
                                        input.
                                    user_data (tuple[str, dict[str, Any]]): Pair containing
                                        the state key and shared application state.

                                Returns:
                                    None: This callback updates widget visibility and state
                                    in place.
                                """
                                # Sync text to state and manage visibility of the second input.
                                update_checkbox_state(sender, app_data, user_data)
                                if app_data.strip() != "":
                                    if not dpg.does_item_exist("activity_group_2"):
                                        with dpg.group(tag="activity_group_2", parent="activity_group"):
                                            dpg.add_text("or", indent=state["analysis_tab_options_target_indent"])
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Activity 2:", indent=state["analysis_tab_options_target_indent"])
                                                dpg.add_input_text(
                                                    tag="activity_query_2", 
                                                    width=-1,
                                                    hint="e.g. Inhibition",
                                                    callback=update_checkbox_state,
                                                    user_data=("Activity query 2", state))
                                                dpg.bind_item_theme("activity_query_2", apply_bordered_input_text_theme(state))
                                else:
                                    if dpg.does_item_exist("activity_group_2"):
                                        dpg.delete_item("activity_group_2")

                            # Reflect checkbox toggle into state and create/remove the inputs accordingly.
                            update_checkbox_state(_, __, state)

                            if dpg.get_value(user_data):
                                with dpg.group(horizontal=True, tag="activity_group_1", parent="activity_group"):
                                    dpg.add_text("Activity 1:", indent=state["analysis_tab_options_target_indent"])
                                    dpg.add_input_text(
                                        tag="activity_query_1", 
                                        width=-1,
                                        hint="e.g. IC50",
                                        callback=show_activity_2_if_needed,
                                        user_data=("Activity query 1", state))
                                    dpg.bind_item_theme("activity_query_1", apply_bordered_input_text_theme(state))
                            else:
                                for tag in ["activity_group_1", "activity_group_2"]:
                                    if dpg.does_item_exist(tag):
                                        dpg.delete_item(tag)


                            # Allow ambiguous operators in activity values (<, ≤, ≥, >).
                            dpg.add_checkbox(
                                label="Enable ambiguous activities", tag="Enable ambiguous activities",
                                default_value=checkbox_states["Enable ambiguous activities"],
                                callback=update_checkbox_state, user_data=state
                                )
                            with dpg.tooltip(parent="Enable ambiguous activities", delay=0.5):
                                dpg.add_text("Enable the use of ambiguous activities (<, ≤, ≥, >) in the activity analysis")

                        # Container for the 'Filter by activity' toggle and its dynamic inputs.
                        with dpg.group(tag="activity_group"):
                            dpg.add_checkbox(
                                label="Filter by activity", tag="Filter by activity", 
                                default_value=checkbox_states["Filter by activity"],
                                callback=activity_input_text, user_data="Filter by activity"
                                )
                            with dpg.tooltip(parent="activity_group", delay=0.5):
                                dpg.add_text("Filter the library by one or two activity types.")


                        dpg.add_spacer(height=spacer_height)
                        dpg.add_text("ASSAY FILTER")
                        change_font_type(dpg.last_item(), "bold", state)

                        def assay_input_combo(_: Any, __: Any, user_data: Any) -> None:
                            """
                            Toggle assay filtering and manage the dynamic assay selectors.

                            Args:
                                _ (Any): Dear PyGui callback sender placeholder. Included
                                    for callback compatibility.
                                __ (Any): Dear PyGui callback payload placeholder. Included
                                    for callback compatibility.
                                user_data (str): Checkbox tag associated with the assay
                                    filter option.

                            Returns:
                                None: This callback updates assay-filter widgets and state
                                in place.
                            """

                            def show_assay_2_if_needed(sender: Any, app_data: Any, user_data: Any) -> None:
                                """
                                Show or hide the second assay selector based on the first one.

                                Args:
                                    sender (Any): Dear PyGui callback sender.
                                    app_data (str): Current value of the first assay combo.
                                    user_data (tuple[str, dict[str, Any]]): Pair containing
                                        the state key and shared application state.

                                Returns:
                                    None: This callback updates widget visibility and state
                                    in place.
                                """
                                # Persist the selection into state.
                                update_checkbox_state(sender, app_data, user_data)
                                if app_data.strip() != "":
                                    if not dpg.does_item_exist("assay_group_2"):
                                        with dpg.group(tag="assay_group_2", parent="assay_group"):
                                            dpg.add_text("or", indent=state["analysis_tab_options_target_indent"])
                                            with dpg.group(horizontal=True):
                                                dpg.add_text("Assay 2:", indent=state["analysis_tab_options_target_indent"])
                                                dpg.add_combo(
                                                    height_mode=dpg.mvComboHeight_Largest, 
                                                    width=-1,
                                                    items=assays,
                                                    tag="assay_query_2", default_value="",
                                                    callback=update_checkbox_state,
                                                    user_data=("Assay query 2", state))
                                else:
                                    if dpg.does_item_exist("assay_group_2"):
                                        dpg.delete_item("assay_group_2")

                            # Reflect the toggle into state and create/remove the first combo accordingly.
                            update_checkbox_state(_, __, state)

                            if dpg.get_value(user_data):
                                with dpg.group(horizontal=True, tag="assay_group_1", parent="assay_group"):
                                    dpg.add_text("Assay 1:", indent=state["analysis_tab_options_target_indent"])
                                    dpg.add_combo(
                                        height_mode=dpg.mvComboHeight_Largest, 
                                        width=-1,
                                        items=assays,
                                        tag="assay_query_1", default_value="",
                                        callback=show_assay_2_if_needed,
                                        user_data=("Assay query 1", state))
                            else:
                                for tag in ["assay_group_1", "assay_group_2"]:
                                    if dpg.does_item_exist(tag):
                                        dpg.delete_item(tag)


                        # Predefined assay names offered in the selector (first empty for "any").
                        assays = [
                            "", 
                            "In vitro biochemical (binding)",
                            "In vitro biochemical (functional)",
                            "In vitro biochemical (other)",
                            "In vitro cellular (binding)",
                            "In vitro cellular (functional)",
                            "In vitro cellular (ADME)",
                            "In vitro cellular (toxicity)",
                            "In vitro cellular (phenotypic)",
                            "In vitro (subcellular)", 
                            "In vitro (tissue)",
                            "In vitro cellular (other)",
                            "In vivo (binding)", 
                            "In vivo (functional)",
                            "In vivo (ADME)", 
                            "In vivo (toxicity)", 
                            "In vivo (non-mammalian)", 
                            "In vivo (other)",
                            "Biophysical",
                            "Binding (unspecified)",
                            "Functional (unspecified)",
                            "ADME (unspecified)",
                            "Toxicity (unspecified)",
                            "Physicochemical (unspecified)",
                            "Unknown"
                        ]

                        # Container for the 'Filter by assay' toggle and its dynamic combos.
                        with dpg.group(tag="assay_group"):
                            dpg.add_checkbox(
                                label="Filter by assay", tag="Filter by assay", 
                                default_value=checkbox_states["Filter by assay"],
                                callback=assay_input_combo, user_data="Filter by assay"
                                )
                        with dpg.tooltip(parent="assay_group", delay=0.5):
                            dpg.add_text("Filter the library by one or two biological assays.")


                with dpg.table_cell():
                    with dpg.child_window(tag="Subsets collection group", border=False, autosize_x=True, height=-1):
                        dpg.add_spacer(height=state["win_spacer"])
                        dpg.add_separator(label="SUBSETS COLLECTION")
                        change_font_type(dpg.last_item(), "bold", state)


                        # --- STEP 2.6.1: Workflow selection callback and conditional visibility ---
                        def clustering_workflow_callback(_: Any, __: Any, user_data: Any) -> None:
                            """
                            Update the subsets-collection workflow and related widget visibility.

                            Args:
                                _ (Any): Dear PyGui callback sender placeholder. Included
                                    for callback compatibility.
                                __ (Any): Dear PyGui callback payload placeholder. Included
                                    for callback compatibility.
                                user_data (dict[str, Any]): Shared application state passed
                                    to the checkbox update helper.

                            Returns:
                                None: This callback updates workflow-dependent widgets and
                                state in place.
                            """
                            # Read the current selection and reflect it into state via the standard updater.
                            app_data = dpg.get_value("Subsets collection method")
                            update_checkbox_state("Subsets collection method", app_data, state)

                            # Hide all conditional groups/tooltips prior to showing the relevant ones.
                            for tag in ["Heavy atoms threshold group", "Heavy atoms threshold tooltip", 
                                        "plswumo group", "plswumo tooltip", 
                                        "Filtering threshold group", "Filtering threshold tooltip",
                                        "MCS timeout group", "MCS timeout tooltip",
                                        "Similarity threshold group", "Similarity threshold tooltip", 
                                        "Scaffold SMILES group", "Scaffold SMILES tooltip",
                                        "Generalized Scaffold SMILES group", "Generalized Scaffold SMILES tooltip",
                                        "Scaffold Similarity threshold group", "Scaffold Similarity threshold tooltip",                            
                                        ]:
                                if dpg.does_item_exist(tag):
                                    dpg.hide_item(tag)

                            # Reveal the appropriate inputs/tooltips based on the selected workflow.
                            if app_data == "Unique Bemis-Murcko scaffolds (BMS)":
                                dpg.show_item("Heavy atoms threshold group")
                                dpg.show_item("Heavy atoms threshold tooltip")
                                dpg.show_item("plswumo group")
                                dpg.show_item("plswumo tooltip")
                                dpg.show_item("Filtering threshold group")
                                dpg.show_item("Filtering threshold tooltip")
                            elif app_data == "BMS minimal substructures (MinBMS)":
                                dpg.show_item("Heavy atoms threshold group")
                                dpg.show_item("Heavy atoms threshold tooltip")
                                dpg.show_item("plswumo group")
                                dpg.show_item("plswumo tooltip")
                                dpg.show_item("Filtering threshold group")
                                dpg.show_item("Filtering threshold tooltip")
                            elif app_data == "MinBMS clustering (MSC)":
                                dpg.show_item("Heavy atoms threshold group")
                                dpg.show_item("Heavy atoms threshold tooltip")
                                dpg.show_item("plswumo group")
                                dpg.show_item("plswumo tooltip")
                                dpg.show_item("Similarity threshold group")
                                dpg.show_item("Similarity threshold tooltip")
                                dpg.show_item("Filtering threshold group")
                                dpg.show_item("Filtering threshold tooltip")
                                dpg.show_item("MCS timeout group")
                                dpg.show_item("MCS timeout tooltip")
                            elif app_data == "User-defined scaffold":
                                dpg.show_item("Scaffold SMILES group")
                                dpg.show_item("Scaffold SMILES tooltip")
                            elif app_data == "User-defined generalized scaffold":
                                dpg.show_item("Generalized Scaffold SMILES group")
                                dpg.show_item("Generalized Scaffold SMILES tooltip")
                                dpg.show_item("Heavy atoms threshold group")
                                dpg.show_item("Heavy atoms threshold tooltip")
                                dpg.show_item("plswumo group")
                                dpg.show_item("plswumo tooltip")
                                dpg.show_item("Filtering threshold group")
                                dpg.show_item("Filtering threshold tooltip")
                            elif app_data == "User-defined generalized scaffold + similarity":
                                dpg.show_item("Generalized Scaffold SMILES group")
                                dpg.show_item("Generalized Scaffold SMILES tooltip")
                                dpg.show_item("Scaffold Similarity threshold group")
                                dpg.show_item("Scaffold Similarity threshold tooltip")
                                dpg.show_item("Heavy atoms threshold group")
                                dpg.show_item("Heavy atoms threshold tooltip")
                                dpg.show_item("plswumo group")
                                dpg.show_item("plswumo tooltip")
                                dpg.show_item("Filtering threshold group")
                                dpg.show_item("Filtering threshold tooltip")

                        # Subsets collection method selection and tooltip explaining each option.
                        with dpg.group(horizontal=True, tag="Subsets collection method group"):
                            clustering_workflow_list = ["Unique Bemis-Murcko scaffolds (BMS)", "BMS minimal substructures (MinBMS)",
                                                        "MinBMS clustering (MSC)", 
                                                        "User-defined scaffold", "User-defined generalized scaffold",
                                                        "User-defined generalized scaffold + similarity"]
                            dpg.add_text("Collecting method",)
                            dpg.add_combo(
                                tag="Subsets collection method",
                                width=-1,
                                items=clustering_workflow_list,
                                default_value=checkbox_states["Subsets collection method"],
                                callback=clustering_workflow_callback, user_data=state
                            )
                            with dpg.tooltip(parent="Subsets collection method group", tag="Subsets collection method tooltip", delay=0.5):
                                dpg.add_text("""Select the subsets collection method to be used for substructure analysis:
                                        
1 - Unique Bemis-Murcko scaffolds (BMS): subsets will contain molecules that share the same Bemis-Murcko scaffold.

2 - BMS minimal substructures (MinBMS): starting from Bemis-Murcko scaffolds, extract the
      smallest substructures common to multiple scaffolds to reduce the number of subsets.

3 - MinBMS clustering (MSC): Cluster minimal substructures by Tanimoto
      similarity and compute the Maximum Common Substructure (MCS) within each cluster.

These workflows progressively refine aggregation and reduce the number of subsets, but this may result in
generalization, increase in R-groups size and structural complexity and in loss of positional specificity.

More specific workflows can be selected to retain positional information and structural specificity:
                                                                         
4 - User-defined scaffold: use a custom scaffold provided as SMILES, SMARTS, InChI, 
      or MolBlock to define a single subset containing all matching molecules.
                                 
5 - User-defined generalized scaffold: use a custom scaffold provided as SMILES, SMARTS, InChI, 
      or MolBlock, generalize it by converting all atoms to carbons and all bonds to single, 
      and retain the molecules whose generalized Murcko scaffold contains the user-defined one. 
      Then continue as with the BMS minimal substructures (MinBMS) method (method 2).
                                
6 - User-defined generalized scaffold + similarity: same as option 5, but retain 
      even the molecules whose generalized Murcko scaffold does not contain the 
      user-defined one (generalized), but has a similarity to it above a defined threshold.
""")

                        # --- STEP 2.6.2: Heavy atoms / Similarity / Filtering thresholds ---
                        # Minimum heavy atoms to consider for MinBMS aggregation.
                        with dpg.group(horizontal=True, tag="Heavy atoms threshold group"):
                            dpg.add_text("Heavy atoms threshold")
                            dpg.add_input_int(
                                tag="Heavy atoms threshold", min_clamped=True, min_value=1, width=-1,
                                default_value=checkbox_states["Heavy atoms threshold"],
                                callback=update_checkbox_state, user_data=state
                                )
                            with dpg.tooltip(parent="Heavy atoms threshold group", tag="Heavy atoms threshold tooltip", delay=0.5):
                                dpg.add_text("Minimal number of heavy atoms to define Bemis-Murcko scaffold's\nminimal substructures and to consider them as 'light substructures'.\n")

                        # Populate light substructures only molecules lacking other substructures assignments.
                        with dpg.group(horizontal=True, tag="plswumo group"):
                            dpg.add_checkbox(label="Populate light substructures with unassigned molecules only",
                                tag="Populate light substructures with unassigned molecules only", 
                                default_value=checkbox_states["Populate light substructures with unassigned molecules only"],
                                callback=update_checkbox_state, user_data=state
                                )
                            with dpg.tooltip(parent="plswumo group", tag="plswumo tooltip", delay=0.5):
                                dpg.add_text("Light substructures (heavy atom count below threshold) are populated exclusively\nwith molecules lacking other scaffold assignments, minimizing redundant subsets.\n")

                        # Similarity threshold for clustering MinBMS substructures before MSC.
                        with dpg.group(horizontal=True, tag="Similarity threshold group", show=False):
                            dpg.add_text("Similarity threshold (%)")
                            dpg.add_input_int(
                                tag="Similarity threshold", 
                                min_clamped=True, max_clamped=True, min_value=0, max_value=100, width=-1,
                                default_value=checkbox_states["Similarity threshold"],
                                callback=update_checkbox_state, user_data=state
                                )
                            with dpg.tooltip(parent="Similarity threshold", tag="Similarity threshold tooltip", delay=0.5):
                                dpg.add_text("Threshold used to cluster Bemis-Murcko scaffold's minimal\nsubstructures on which the MSC will be calculated\n")

                        # Filter-out subsets below a size threshold before decomposition.
                        with dpg.group(horizontal=True, tag="Filtering threshold group"):
                            dpg.add_text("Filtering threshold")
                            dpg.add_input_int(
                                tag="Filtering threshold", min_clamped=True, min_value=1, width=-1,
                                default_value=checkbox_states["Filtering threshold"],
                                callback=update_checkbox_state, user_data=state
                                )
                            with dpg.tooltip(parent="Filtering threshold group", tag="Filtering threshold tooltip", delay=0.5):
                                dpg.add_text("Filter out subsets with a number of molecules less than the threshold before decomposition STEP\n")

                        with dpg.group(horizontal=True, tag="MCS timeout group", show=False):
                            dpg.add_text("MCS timeout")
                            dpg.add_combo(
                                tag="MCS timeout",
                                width=-1,
                                items=["10s", "30s", "60s", "120s", "180s", "300s", "600s", "Unlimited"],
                                default_value=checkbox_states["MCS timeout"],
                                callback=update_checkbox_state,
                                user_data=state,
                            )
                            with dpg.tooltip(parent="MCS timeout group", tag="MCS timeout tooltip", delay=0.5):
                                dpg.add_text("Maximum time allowed for each MCS search during MSC clustering.\nChoose 'Unlimited' to let the search run with no time limit\n(the analysis may take very long time).")


                        # --- STEP 2.6.3: User-defined scaffold input and validation popup ---
                        # Validate a scaffold SMILES/SMARTS/InChI/MolBlock and show a small status popup.
                        def validate_scaffold_callback() -> None:
                            """
                            Validate the user-defined scaffold and report the detected format.

                            Args:
                                None.

                            Returns:
                                None: This callback updates the scaffold validation popup in
                                place.
                            """
                            # Retrieve the user-provided scaffold text and see if it can be parsed as any supported format.
                            scaffold_input = dpg.get_value("Scaffold SMILES")
                            if scaffold_input.strip() == "":
                                dpg.configure_item("scaffold_validity_text", default_value="Scaffold is empty")
                                dpg.configure_item("scaffold_validity_popup", show=True)
                                return
                            try:
                                scaffold_mol = Chem.MolFromSmiles(scaffold_input)
                                if scaffold_mol is None:
                                    raise ValueError()
                                text = "Scaffold SMILES is valid"
                            except Exception:
                                try:
                                    scaffold_mol = Chem.MolFromSmarts(scaffold_input)
                                    if scaffold_mol is None:
                                        raise ValueError()
                                    text = "Scaffold SMARTS is valid"
                                except Exception:
                                    try:
                                        scaffold_mol = Chem.MolFromInchi(scaffold_input)
                                        if scaffold_mol is None:
                                            raise ValueError()
                                        text = "Scaffold InChI is valid"
                                    except Exception:
                                        try:
                                            scaffold_mol = Chem.MolFromMolBlock(scaffold_input)
                                            if scaffold_mol is None:
                                                raise ValueError()
                                            text = "Scaffold MolBlock is valid"
                                        except Exception:
                                            scaffold_mol = None
                                            text = "Scaffold is NOT valid"

                            dpg.set_value("scaffold_validity_text", text)
                            dpg.configure_item("scaffold_validity_popup", show=True)


                        # Input area for scaffold definition and a quick validation button.
                        with dpg.group(tag="Scaffold SMILES group", show=False):
                            dpg.add_input_text(
                                tag="Scaffold SMILES",
                                width=-1,
                                hint="Type scaffold SMILES/SMARTS/InChI/MolBlock here",
                                multiline=True,
                                callback=update_checkbox_state, user_data=state
                                )
                            dpg.bind_item_theme("Scaffold SMILES", apply_bordered_input_text_theme(state))

                            dpg.add_text("Paste here SMILES, SMARTS, InChI or MolBlock")
                            dpg.add_button(label="Test Scaffold", tag="smiles_molblock_test_button", callback=validate_scaffold_callback)

                            with dpg.tooltip(parent="Scaffold SMILES group", tag="Scaffold SMILES tooltip", delay=0.5):
                                dpg.add_text("Create a single subset containing all the molecules matching a scaffold defined by its SMILES, SMARTS, InChI or MolBlock.\nNote: when using InChI or SMARTS as input, the resulting structure may differ from the original molecule due to tautomer\nnormalization, aromaticity perception, or atom typing. This can affect substructure matching and may result in no valid hits.\n")

                            # Popup reporting the validation result in a small modal window.
                            with dpg.window(tag="scaffold_validity_popup", 
                                                modal=True, show=False, no_close=True, no_resize=True, no_move=True,
                                                no_collapse=True, no_title_bar=True, autosize=True, min_size=(10, 10)):
                                dpg.add_text("", tag="scaffold_validity_text")
                                dpg.add_spacer(height=state["win_spacer"])
                                dpg.add_button(label="OK", callback=lambda: dpg.configure_item("scaffold_validity_popup", show=False))


                        # --- STEP 2.6.4: User-defined generalized scaffold input and validation popup ---
                        def validate_generalized_scaffold_callback() -> None:
                            """
                            Validate the generalized scaffold and report the detected format.

                            Args:
                                None.

                            Returns:
                                None: This callback updates the generalized-scaffold
                                validation popup in place.
                            """
                            scaffold_input = dpg.get_value("Generalized Scaffold SMILES")
                            if scaffold_input.strip() == "":
                                dpg.set_value("gen_scaffold_validity_text", "Generalized scaffold is empty")
                                dpg.configure_item("gen_scaffold_validity_popup", show=True)
                                return

                            text, mol = "Generalized scaffold is NOT valid", None

                            # Try SMILES
                            try:
                                mol = Chem.MolFromSmiles(scaffold_input)
                                if mol is not None:
                                    text = "Generalized scaffold SMILES is valid"
                            except Exception:
                                mol = None

                            # Try SMARTS
                            if mol is None:
                                try:
                                    mol = Chem.MolFromSmarts(scaffold_input)
                                    if mol is not None:
                                        text = "Generalized scaffold SMARTS is valid"
                                except Exception:
                                    mol = None

                            # Try InChI
                            if mol is None:
                                try:
                                    mol = Chem.MolFromInchi(scaffold_input)
                                    if mol is not None:
                                        text = "Generalized scaffold InChI is valid"
                                except Exception:
                                    mol = None

                            # Try MolBlock
                            if mol is None:
                                try:
                                    mol = Chem.MolFromMolBlock(scaffold_input)
                                    if mol is not None:
                                        text = "Generalized scaffold MolBlock is valid"
                                except Exception:
                                    mol = None

                            dpg.set_value("gen_scaffold_validity_text", text)
                            dpg.configure_item("gen_scaffold_validity_popup", show=True)


                        # Input area for generalized scaffold definition
                        with dpg.group(tag="Generalized Scaffold SMILES group", show=False):
                            dpg.add_input_text(
                                tag="Generalized Scaffold SMILES",
                                width=-1,
                                hint="Type scaffold SMILES/SMARTS/InChI/MolBlock here",
                                multiline=True,
                                callback=update_checkbox_state, user_data=state
                            )
                            dpg.bind_item_theme("Generalized Scaffold SMILES", apply_bordered_input_text_theme(state))

                            dpg.add_text("Paste here SMILES, SMARTS, InChI or MolBlock")
                            dpg.add_button(label="Test Generalized Scaffold", callback=validate_generalized_scaffold_callback)

                            with dpg.tooltip(parent="Generalized Scaffold SMILES group",
                                            tag="Generalized Scaffold SMILES tooltip", delay=0.5):
                                dpg.add_text(
                                    "Use a generalized scaffold as a query (SMILES/SMARTS/InChI/MolBlock).\n"
                                    "This defines a single subset with all matching molecules.\n"
                                    "Note: SMARTS/InChI parsing and aromaticity/tautomer normalization may alter the perceived structure.\n"
                                )

                            with dpg.window(
                                tag="gen_scaffold_validity_popup",
                                modal=True, show=False, no_close=True, no_resize=True, no_move=True,
                                no_collapse=True, no_title_bar=True, autosize=True, min_size=(10, 10)
                            ):
                                dpg.add_text("", tag="gen_scaffold_validity_text")
                                dpg.add_spacer(height=state["win_spacer"])
                                dpg.add_button(label="OK", callback=lambda: dpg.configure_item("gen_scaffold_validity_popup", show=False))


                        # --- STEP 2.6.5: Similarity threshold for "generalized scaffold + similarity" ---
                        with dpg.group(horizontal=True, tag="Scaffold Similarity threshold group", show=False):
                            dpg.add_text("Scaffold similarity threshold (%)")
                            dpg.add_input_int(
                                tag="Scaffold Similarity threshold",
                                min_clamped=True, max_clamped=True, min_value=0, max_value=100, width=-1,
                                default_value=checkbox_states["Scaffold Similarity threshold"],
                                callback=update_checkbox_state, user_data=state
                            )
                            with dpg.tooltip(parent="Scaffold Similarity threshold group",
                                            tag="Scaffold Similarity threshold tooltip", delay=0.5):
                                dpg.add_text(
                                    "Similarity threshold (0-100) used when applying the generalized scaffold + similarity workflow.\n"
                                )
