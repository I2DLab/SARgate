"""
=================
mmpa_manager.py
=================

Matched Molecular Pairs Analysis (MMPA) manager.

Coordinates the MMPA workflow, generating transformation tables and plots that
relate structural modifications to activity changes. Integrates similarity
thresholds, transformation filtering, and summary statistics.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Show mmpa window
# 3. Try run mmpa analysis

import dearpygui.dearpygui as dpg
from typing import Any
from app.utils.app_logger import log_event
from app.analysis.mmpa.mmpa_plots import (
    mount_mmpa_plots, 
    on_mmpa_thresh_slider_change,
    _load_distribution_csv,
)
from app.analysis.mmpa.mmpa_logic import run_mmpa_analysis
from app.analysis.mmpa.mmpa_network import clear_mmpa_network_memory


# -----------------------------------------------------------------------------
# 2. Show mmpa window
# -----------------------------------------------------------------------------
def show_mmpa_window(state: dict[str, Any]) -> None:
    """
    Displays the GUI panel for R-based Matched Molecular Pairs Analysis (MMPA).
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    def _refresh_mmpa_delta_thresh_slider(selected_activity: Any, selected_subset: Any, state: dict[str, Any], reset_value: bool) -> None:
        tag = "mmpa_delta_thresh"
        activity_str = str(selected_activity or "")
        subset_str = str(selected_subset or "")
        is_log_scale = activity_str in state["nM_activity_types"] or activity_str in state["dimensionless"]

        if is_log_scale:
            dpg.set_item_label(tag, f"Min Δp{activity_str}")
            dpg.configure_item(tag, format="%.1f")
            fallback_default = 2.0
            minimum_ceiling = 0.1
        else:
            dpg.set_item_label(tag, f"Min Δ{activity_str}")
            dpg.configure_item(tag, format="%.0f")
            fallback_default = 10.0
            minimum_ceiling = 1.0

        observed_max = minimum_ceiling
        if activity_str and activity_str != "No activities found":
            dist = _load_distribution_csv(state, activity_str)
            values = dist.get(subset_str, [])
            if values:
                try:
                    observed_max = max(minimum_ceiling, float(max(values)))
                except Exception:
                    observed_max = minimum_ceiling

        dpg.configure_item(tag, min_value=0.0, max_value=observed_max)

        try:
            current_value = float(dpg.get_value(tag))
        except Exception:
            current_value = fallback_default

        if reset_value:
            new_value = min(fallback_default, observed_max)
        else:
            new_value = min(max(0.0, current_value), observed_max)

        dpg.set_value(tag, new_value)

    # -----------------------------------------------------------------------------
    # 2.1. Update mmpa options
    # -----------------------------------------------------------------------------
    def update_mmpa_options(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Updates the available activity types and shows/hides the MMPA button based on the selected subset.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
            user_data (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        
        subset = app_data
        bioact_types_dict = user_data["bioact_types_dict"]

        # Refresh activity options and run button according to the chosen subset
        if subset in bioact_types_dict:
            activities = bioact_types_dict[subset]["bioactivities"]
            if activities:
                dpg.configure_item("mmpa_activity_type", items=activities, enabled=True)
                dpg.set_value("mmpa_activity_type", activities[0])
                # Create the run button if missing
                if not dpg.does_item_exist("run_mmpa_button"):
                    with dpg.group(parent="mmpa_run_button_container"):
                        dpg.add_button(label="Run MMPA", tag="run_mmpa_button", callback=lambda: try_run_mmpa_analysis(state))
                _refresh_mmpa_delta_thresh_slider(dpg.get_value("mmpa_activity_type"), subset, state, reset_value=False)
                # Rebuild plots after options change
                mount_mmpa_plots(state)
            
            else:
                # Disable activity combo and remove the run button when no activities exist
                dpg.configure_item("mmpa_activity_type", items=["No activities found"], enabled=False)
                dpg.set_value("mmpa_activity_type", "No activities found")
                if dpg.does_item_exist("run_mmpa_button"):
                    dpg.delete_item("run_mmpa_button")

                if dpg.does_item_exist("mmpa_subset_plot_window"):
                    dpg.delete_item("mmpa_subset_plot_window", children_only=True)


    # -----------------------------------------------------------------------------
    # 2.2. Update mmpa delta thresh
    # -----------------------------------------------------------------------------
    def update_mmpa_delta_thresh(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Adjusts the delta threshold input (label, format, limits, default) based on the selected activity type.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
            user_data (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """

        selected_activity = app_data
        state = user_data

        selected_subset = dpg.get_value("mmpa_subset_choice") if dpg.does_item_exist("mmpa_subset_choice") else "subset_1"
        _refresh_mmpa_delta_thresh_slider(selected_activity, selected_subset, state, reset_value=True)

        # Refresh plots after threshold change
        mount_mmpa_plots(state)


    with dpg.child_window(parent="mmpa_window", auto_resize_y=True,
                    no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False, border=False):      

        with dpg.group(horizontal=True):
            # Build subset list and choose a default
            subsets = list(state["smiles_rgd_dict"].keys())
            default_subset = "subset_1"

            # Subset choice
            dpg.add_text("Subset:")
            dpg.add_combo(width=100, height_mode=dpg.mvComboHeight_Large, items=subsets,
                          default_value=default_subset, tag="mmpa_subset_choice",
                          callback=update_mmpa_options, user_data=state)

            # Visual spacer between controls
            dpg.add_spacer(width=state["win_spacer"] * 3)

            # Activity type combo (enabled if activities exist for default subset)
            bioact_types_dict = state["bioact_types_dict"]
            initial_activities = bioact_types_dict[default_subset]["bioactivities"]
            dpg.add_text("Activity Type:")
            if initial_activities:
                dpg.add_combo(width=100, height_mode=dpg.mvComboHeight_Largest, items=initial_activities,
                              default_value=initial_activities[0], tag="mmpa_activity_type",
                              callback=update_mmpa_delta_thresh, user_data=state)

                show_button = True
            else:
                dpg.add_combo(width=state["mmpa_manager_combo_width"], items=["No activities found"],
                              no_arrow_button=True, default_value="No activities found", tag="mmpa_activity_type", enabled=False)
                show_button = False
    
            # Spacer before additional options
            dpg.add_spacer(width=state["win_spacer"] * 3)

            # Include undefined checkbox with explanatory tooltip
            dpg.add_checkbox(label="Include undefined", tag="mmpa_include_undefined_choice", default_value=False)
            with dpg.tooltip("mmpa_include_undefined_choice"):
                dpg.add_text("Include molecules annotated with undefined activity values (<, <=, >=, >).\n"
                             "Undefined values will be treated as exact ones (=)")
                                
            # Spacer before the next option
            dpg.add_spacer(width=state["win_spacer"] * 3)

            # Include inactive checkbox with tooltip
            dpg.add_checkbox(label="Include NO activity", tag="mmpa_include_inactive_choice", default_value=False)
            with dpg.tooltip("mmpa_include_inactive_choice"):
                dpg.add_text("Include molecules lacking activity annotations\n")
                
            # Spacer before the threshold and run controls
            dpg.add_spacer(width=state["win_spacer"] * 3)

            # Delta threshold slider (default configured for pValue)
            dpg.add_slider_float(label=f"Min Δp{dpg.get_value('mmpa_activity_type')}",
                                 width=-250, tag="mmpa_delta_thresh",
                                 min_value=0.0, max_value=15, default_value=2.0, format="%.1f", 
                                 callback=lambda s, a: on_mmpa_thresh_slider_change(s, a, state))
            
            dpg.add_spacer(width=state["win_spacer"] * 3)

            # Run button container (conditionally shown)
            with dpg.group(tag="mmpa_run_button_container"):
                if show_button:
                    dpg.add_button(label="Run MMPA", tag="run_mmpa_button", callback=lambda: try_run_mmpa_analysis(state))

    # Build the initial set of plots with current controls
    _refresh_mmpa_delta_thresh_slider(dpg.get_value("mmpa_activity_type"), dpg.get_value("mmpa_subset_choice"), state, reset_value=True)
    mount_mmpa_plots(state)


# -----------------------------------------------------------------------------
# 3. Try run mmpa analysis
# -----------------------------------------------------------------------------
def try_run_mmpa_analysis(state: dict[str, Any]) -> None:
    """
    Wrapper to run the MMPA analysis with error handling.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    log_event("MMPA", "Submitting MMPA analysis request", indent=1)
    clear_mmpa_network_memory(state, clear_plot=True)
    for tag in ["mmpa_table", "r1_label", "r2_label", "rmolA_label", "molB_label", 
                "r1_image_widget", "r2_image_widget", "molA_image_widget", "molB_image_widget",
                "mmpa_network_plot"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    for tag in ["mmpa_table_window", "mmpa_table", "mmpa_images_window", "mmpa_network_window"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=True)


    try:
        run_mmpa_analysis(state)
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")        
            
    except Exception as e:
        # Remove tables, images and plots created by previous runs
        for tag in ["mmpa_table_window", "mmpa_table", "mmpa_images_window", "mmpa_group_sidebar_window", "mmpa_network_plot_window",
                    "r1_label", "r2_label", "rmolA_label", "molB_label", 
                    "r1_image_widget", "r2_image_widget", "molA_image_widget", "molB_image_widget",
                    "mmpa_network_plot"]:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag, children_only=True)
                
        # Remove hover handlers associated with MMPA rows
        for item in dpg.get_all_items():
            alias = dpg.get_item_alias(item)
            if isinstance(alias, str) and alias.startswith("mmpa_hover_handler"):
                dpg.delete_item(item)

        # Ensure loading overlay is hidden
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")

        # Display a modal dialog with the error message
        with dpg.window(label="MMPA Analysis Error", tag="mmpa_error_window", modal=False, no_resize=True, 
            no_collapse=True, autosize=True, on_close=dpg.delete_item("mmpa_error_window")):
            dpg.add_text(f"An error occurred during the MMPA analysis:\n\n{e}")
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("mmpa_error_window"))
