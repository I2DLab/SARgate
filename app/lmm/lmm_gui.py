"""
===========
lmm_gui.py
===========

Analysis progress and execution-time tracking.

Displays a compact GUI window showing elapsed execution time, progress status,
and completion summaries during active analyses. Provides lightweight feedback
to the user without interrupting the computation workflow.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Update execution time
# 3. Update library preparation status
# 4. Update scaffold analysis status
# 5. Update rga status
# 6. Show pie chart

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import time
import dearpygui.dearpygui as dpg
from datetime import timedelta
from typing import Any
from app.gui.themes_manager import (
    apply_pie_chart_theme,
    change_font_type
)


# -----------------------------------------------------------------------------
# 2. Update execution time
# -----------------------------------------------------------------------------
def update_execution_time(state: dict[str, Any]) -> None:
    """
    Update the execution time every second and display it in the timer window.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    # Compute elapsed time from state['start_time'] and refresh the on-screen value.
    while state["timer_running"]:
        elapsed_time = time.time() - state["start_time"]
        dpg.configure_item("execution_time_text", label=str(timedelta(seconds=int(elapsed_time))))
        time.sleep(1)
  



# -----------------------------------------------------------------------------
# 3. Update library preparation status
# -----------------------------------------------------------------------------
def update_library_preparation_status(
    message: str,
    state: dict[str, Any],
    separator: bool = False,
    step_id: bool = False,
    temp: bool = False
) -> None:
    """
    Update the 'Library Preparation' window with a new message and mirror the same.
    
    Args:
        message (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
        separator (Any): Parameter accepted by this routine. Defaults to the configured value.
        step_id (Any): Parameter accepted by this routine. Defaults to the configured value.
        temp (Any): Parameter accepted by this routine. Defaults to the configured value.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    
    STEP_prep = state["STEP_prep"]
    
    if temp:
        if dpg.does_item_exist("temp_message_group_prep"):
            dpg.delete_item("temp_message_group_prep")
        with dpg.group(parent="library_preparation_window", tag="temp_message_group_prep"):
            dpg.add_text(message, tag="temp_message_prep")
        return
    else:
        if dpg.does_item_exist("temp_message_group_prep"):
            dpg.delete_item("temp_message_group_prep")
        with dpg.group(parent="library_preparation_window"):
            text_out = f"{STEP_prep}: {message}" if step_id else message
            dpg.add_text(text_out)
            if step_id:
                change_font_type(dpg.last_item(), "bold", state)
            if separator:
                dpg.add_spacer(height=(state["win_spacer"] * 2))
                dpg.add_separator()
        if step_id:
            STEP_prep += 1
        state["STEP_prep"] = STEP_prep

    state["prep_log"].append({
        "text": f"{state.get('STEP_prep', STEP_prep) - 1}: {message}" if step_id else message,
        "separator": bool(separator)
    })


# -----------------------------------------------------------------------------
# 4. Update scaffold analysis status
# -----------------------------------------------------------------------------
def update_scaffold_analysis_status(
    message: str,
    state: dict[str, Any],
    separator: bool = False,
    step_id: bool = False,
    temp: bool = False
) -> None:
    """
    Update the 'Scaffold Analysis' window and mirror into state['scaff_log'].
    
    Args:
        message (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
        separator (Any): Parameter accepted by this routine. Defaults to the configured value.
        step_id (Any): Parameter accepted by this routine. Defaults to the configured value.
        temp (Any): Parameter accepted by this routine. Defaults to the configured value.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    STEP_scaff = state["STEP_scaff"]

    if temp:
        if dpg.does_item_exist("temp_message_group_1"):
            dpg.delete_item("temp_message_group_1")
        with dpg.group(parent="scaffold_analysis_window", tag="temp_message_group_1"):
            dpg.add_text(message, tag="temp_message_1")
        return
    else:
        if dpg.does_item_exist("temp_message_group_1"):
            dpg.delete_item("temp_message_group_1")
        with dpg.group(parent="scaffold_analysis_window"):
            text_out = f"{STEP_scaff}: {message}" if step_id else message
            dpg.add_text(text_out)
            if step_id:
                change_font_type(dpg.last_item(), "bold", state)
            if separator:
                dpg.add_spacer(height=(state["win_spacer"] * 2))
                dpg.add_separator()
        if step_id:
            STEP_scaff += 1
        state["STEP_scaff"] = STEP_scaff

    state["scaff_log"].append({
        "text": f"{state.get('STEP_scaff', STEP_scaff) - 1}: {message}" if step_id else message,
        "separator": bool(separator)
    })


# -----------------------------------------------------------------------------
# 5. Update rga status
# -----------------------------------------------------------------------------
def update_rga_status(
    message: str,
    state: dict[str, Any],
    separator: bool = False,
    step_id: bool = False,
    temp: bool = False
) -> None:
    """
    Update the R-Groups Decomposition window with a new message.
    
    Args:
        message (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
        separator (Any): Parameter accepted by this routine. Defaults to the configured value.
        step_id (Any): Parameter accepted by this routine. Defaults to the configured value.
        temp (Any): Parameter accepted by this routine. Defaults to the configured value.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    STEP_rga = state["STEP_rga"]

    if temp:
        if dpg.does_item_exist("temp_message_group_2"):
            dpg.delete_item("temp_message_group_2")
        with dpg.group(parent="rga_window", tag="temp_message_group_2"):
            dpg.add_text(message, tag="temp_message_2")
    else:
        if dpg.does_item_exist("temp_message_group_2"):
            dpg.delete_item("temp_message_group_2")
        with dpg.group(parent="rga_window"):
            text_out = f"{STEP_rga}: {message}" if step_id else message
            dpg.add_text(text_out)
            if step_id:
                change_font_type(dpg.last_item(), "bold", state)
            if separator:
                dpg.add_separator()

        if step_id:
            STEP_rga += 1
        state["STEP_rga"] = STEP_rga

    # --- LOG update ---
    state["rga_log"].append({
        "text": f"{state.get('STEP_rga', STEP_rga) - 1}: {message}" if step_id else message,
        "separator": bool(separator)
    })


# -----------------------------------------------------------------------------
# 6. Show pie chart
# -----------------------------------------------------------------------------
def show_pie_chart(sorted_list: Any, sender_name: str, state: dict[str, Any]) -> None:
    """
    Display a pie chart of activity types, targets or assays.
    
    Args:
        sorted_list (Any): Parameter accepted by this routine.
        sender_name (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    # Parse labels/counts, create a plot anchored to the preparation console, and add a normalised pie series.
    labels, values = zip(*sorted_list)
    labels = [f"{label}: {count}" for label, count in sorted_list]
    values = list(values)

    x_axis_tag = f"{sender_name}_x_axis"
    y_axis_tag = f"{sender_name}_y_axis"

    if dpg.does_item_exist(sender_name):
        dpg.delete_item(sender_name)

    with dpg.plot(label="",
                tag=sender_name,
                parent="library_preparation_window",
                no_title=True, no_mouse_pos=True, no_frame=True,
                width=state["analysis_pie_chart_size"],
                height=state["analysis_pie_chart_size"]):

        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
        dpg.add_plot_axis(
            dpg.mvXAxis,
            tag=x_axis_tag,
            no_label=True,
            no_gridlines=True,
            no_tick_marks=True,
            no_menus=True,
            no_tick_labels=True,
            no_highlight=True,
            invert=True,
        )
        dpg.set_axis_limits(x_axis_tag, 0, 1)

        with dpg.plot_axis(
            dpg.mvYAxis,
            tag=y_axis_tag,
            no_label=True,
            no_gridlines=True,
            no_menus=True,
            no_tick_marks=True,
            no_tick_labels=True,
            no_highlight=True,
        ):
            dpg.set_axis_limits(y_axis_tag, 0, 1)

            dpg.add_pie_series(0.5, 0.5, 0.485, values, labels, normalize=True, format="")

    dpg.bind_item_theme(sender_name, apply_pie_chart_theme(state))
    dpg.bind_colormap(sender_name, state["plot_colormaps"][state["colormap_discrete"]])
