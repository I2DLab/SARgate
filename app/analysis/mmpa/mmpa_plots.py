"""
===============
mmpa_plots.py
===============

Graphical representation of MMPA results.

Generates visual plots to display the distribution of
ΔActivity values across single subsets or the entire dataset.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Build fixed bins
# 3. Hist counts
# 4. Safe
# 5. Load distribution csv
# 6. On mmpa thresh slider change
# 7. Ensure mmpa plots created
# 8. Refresh global plot
# 9. Refresh subset plot
# 10. Mount mmpa plots

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import csv
import re
import dearpygui.dearpygui as dpg
import numpy as np
from typing import Any
from app.utils.callbacks import register_plot_context_popup
from app.gui.themes_manager import (
    apply_line_chart_theme,
    apply_infinite_line_theme
)


# -----------------------------------------------------------------------------
# 2. Build fixed bins
# -----------------------------------------------------------------------------
def _build_fixed_bins(values: Any) -> Any:
    """
    Build fixed-width bin edges for histogramming delta values.

    Args:
        values (Iterable[float]): Sequence of numeric values.

    Returns:
        numpy.ndarray: Array of bin edges starting at 0 with 0.1 step.
    """
    binning=0.1
    if not values:
        return np.array([0.0, binning], dtype=float)
    vmax = float(np.max(values))
    top = (np.ceil(vmax / binning) * binning) if vmax > 0 else binning
    if top < binning:
        top = binning
    n_steps = int(round(top / binning))
    return np.linspace(0.0, n_steps * binning, n_steps + 1)


# -----------------------------------------------------------------------------
# 3. Hist counts
# -----------------------------------------------------------------------------
def _hist_counts(values: Any) -> Any:
    """
    Compute histogram bin centres, counts, and edges for given values.

    Args:
        values (Iterable[float]): Numeric values to bin.

    Returns:
        tuple[list[float], list[int], list[float]]: (centres, counts, edges)
    """
    edges = _build_fixed_bins(values)
    counts, edges = np.histogram(values, bins=edges, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.tolist(), counts.tolist(), edges.tolist()


def _count_transformations_above_threshold(values: Any, threshold: float) -> int:
    """
    Count how many delta values are greater than or equal to the current threshold.

    Args:
        values (Iterable[float]): Delta values for the selected subset.
        threshold (float): Active delta threshold.

    Returns:
        int: Number of transformations above threshold.
    """
    count = 0
    for value in values or []:
        try:
            if float(value) >= float(threshold):
                count += 1
        except Exception:
            continue
    return count


# -----------------------------------------------------------------------------
# 4. Safe
# -----------------------------------------------------------------------------
def _safe(s: Any) -> Any:
    """
    Make a filesystem-friendly token from an arbitrary string.

    Args:
        s (Any): Input to be converted.

    Returns:
        str: Sanitised string containing only letters, digits, '_', '.', or '-'.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


# -----------------------------------------------------------------------------
# 5. Load distribution csv
# -----------------------------------------------------------------------------
def _load_distribution_csv(state: dict[str, Any], activity: str) -> Any:
    """
    Return a dict like {"GLOBAL": [Delta...], "<subset>": [Delta...]} by reading the CSV for the activity.
    Uses a cache in state["mmpa_dists_cache"] to avoid reloading.

    Args:
        state (dict): Application state holding 'report_dir' and cache.
        activity (str): Activity type whose distribution file should be loaded.

    Returns:
        dict[str, list[float]]: Mapping subset → list of delta values.
    """
    cache = state.setdefault("mmpa_dists_cache", {})
    # if activity in cache:
    #     return cache[activity]

    reports_dir = state["report_dir"] 
    mmpa_dir = os.path.join(reports_dir, "mmpa_delta_distributions") 
    safe_act = _safe(activity)
    csv_path = os.path.join(mmpa_dir, f"mmpa_delta_distribution_{safe_act}.csv")
    if not os.path.isfile(csv_path):
        hist_payload = (state.get("mmpa_delta_hist", {}) or {}).get(activity, {}) or {}
        has_expected_data = bool(hist_payload.get("global_deltas"))
        if has_expected_data:
            log_event("MMPA", f"CSV not found for delta distribution plot: {csv_path}", indent=1, level="ERROR")
        cache[activity] = {}
        return cache[activity]

    dist = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subset = row.get("Subset", "")
                try:
                    d = float(row.get("Delta", ""))
                except Exception:
                    continue
                dist.setdefault(subset, []).append(d)
    except Exception as e:
        log_exception("MMPA", f"Error reading CSV {csv_path}", e, indent=1)
        dist = {}

    cache[activity] = dist
    return dist


# -----------------------------------------------------------------------------
# 6. On mmpa thresh slider change
# -----------------------------------------------------------------------------
def on_mmpa_thresh_slider_change(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    """
    Move-only update for the threshold lines (no plot rebuild).
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    try:
        new_x = float(app_data)
    except Exception:
        return

    # Move the two vertical threshold lines (if present)
    if dpg.does_item_exist("mmpa_global_thresh_rect"):
        dpg.configure_item("mmpa_global_thresh_rect", x=[new_x])
    if dpg.does_item_exist("mmpa_subset_thresh_rect"):
        dpg.configure_item("mmpa_subset_thresh_rect", x=[new_x])

    if dpg.does_item_exist("mmpa_plot_subset") and dpg.does_item_exist("mmpa_activity_type") and dpg.does_item_exist("mmpa_subset_choice"):
        activity = dpg.get_value("mmpa_activity_type")
        subset = dpg.get_value("mmpa_subset_choice")
        is_log_scale = activity in state["nM_activity_types"]
        if activity == "No activity found":
            transformations_count = 0
        else:
            dist = _load_distribution_csv(state, activity)
            values = dist.get(subset, [])
            transformations_count = _count_transformations_above_threshold(values, new_x)
        dpg.configure_item(
            "mmpa_plot_subset",
            label=f'Δ{"p" if is_log_scale else ""}{activity} distribution ({subset.replace("subset_", "Subset ")})  |  Transformations: {transformations_count}'
        )
        


# -----------------------------------------------------------------------------
# 7. Ensure mmpa plots created
# -----------------------------------------------------------------------------
def _ensure_mmpa_plots_created(state: dict[str, Any]) -> None:
    """
    Create, if missing, the two plots (GLOBAL and SUBSET) with their series.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    
    # Clear everything inside the target window to rebuild cleanly
    for tag in ["mmpa_global_plot_window", "mmpa_subset_plot_window"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=True)

    with dpg.child_window(parent="mmpa_global_plot_window", 
                    no_scrollbar=True, horizontal_scrollbar=False, no_scroll_with_mouse=False, border=False):      

        activity = dpg.get_value("mmpa_activity_type")
        is_log_scale = activity in state["nM_activity_types"]
        subset = dpg.get_value("mmpa_subset_choice")
        dist = _load_distribution_csv(state, activity)
        
        # GLOBAL plot: curve + vertical threshold line
        values = dist.get("GLOBAL", [])
        centers, counts, edges = _hist_counts(values)
        centers_shifted = [c - 0.05 for c in centers]
        with dpg.plot(label=f'Δ{"p" if is_log_scale else ""}{activity} distribution (Global)', tag="mmpa_plot_global", crosshairs=True, zoom_rate=0.05,
                    height=-1, 
                    width=-1):
            with dpg.plot_axis(dpg.mvXAxis, label=f'Δ{"p" if is_log_scale else ""}{activity}', tag="mmpa_x_global", no_highlight=True, lock_min=False, no_menus=True):
                dpg.add_inf_line_series(
                    x=[float(dpg.get_value("mmpa_delta_thresh"))],
                    label=f'min Δ{"p" if is_log_scale else ""}{activity}',
                    tag="mmpa_global_thresh_rect",
                    horizontal=False
                )
                dpg.bind_item_theme("mmpa_global_thresh_rect", apply_infinite_line_theme())
            with dpg.plot_axis(dpg.mvYAxis, label=f'Δ{"p" if is_log_scale else ""}{activity} Frequency', tag="mmpa_y_global", no_highlight=True, lock_min=False, no_menus=True):
                dpg.add_line_series(centers_shifted, counts, label=f'Δ{"p" if is_log_scale else ""}{activity} distribution', tag="mmpa_global_curve")
            dpg.bind_colormap("mmpa_plot_global", state["plot_colormaps"][state["colormap_discrete"]])
            dpg.bind_item_theme("mmpa_plot_global", apply_line_chart_theme(state))
            register_plot_context_popup(
                state,
                context_key="mmpa_global_plot_context",
                plot_tag="mmpa_plot_global",
                x_axis_tag="mmpa_x_global",
                y_axis_tag="mmpa_y_global",
                theme_kind="plot",
            )


    with dpg.child_window(parent="mmpa_subset_plot_window", 
                    no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):      
                
        # SUBSET plot: curve + vertical threshold line
        values = dist.get(subset, [])
        threshold_value = float(dpg.get_value("mmpa_delta_thresh"))
        transformations_count = _count_transformations_above_threshold(values, threshold_value)
        centers, counts, edges = _hist_counts(values)
        centers_shifted = [c - 0.05 for c in centers]
        with dpg.plot(label=f'Δ{"p" if is_log_scale else ""}{activity} distribution ({subset.replace("subset_", "Subset ")})  |  Transformations: {transformations_count}', tag="mmpa_plot_subset", crosshairs=True, zoom_rate=0.05,
                    height=-1,
                    width=-1):
            with dpg.plot_axis(dpg.mvXAxis, label=f'Δ{"p" if is_log_scale else ""}{activity}', tag="mmpa_x_subset", no_highlight=True, lock_min=False, no_menus=True):
                dpg.add_inf_line_series(
                    x=[float(dpg.get_value("mmpa_delta_thresh"))],
                    label=f'min Δ{"p" if is_log_scale else ""}{activity}',
                    tag="mmpa_subset_thresh_rect",
                    horizontal=False
                )
                dpg.bind_item_theme("mmpa_subset_thresh_rect", apply_infinite_line_theme())
            with dpg.plot_axis(dpg.mvYAxis, label=f'Δ{"p" if is_log_scale else ""}{activity} Frequency', tag="mmpa_y_subset", no_highlight=True, lock_min=False, no_menus=True):
                dpg.add_line_series(centers_shifted, counts, label=f'Δ{"p" if is_log_scale else ""}{activity} distribution', tag="mmpa_subset_curve")
            dpg.bind_colormap("mmpa_plot_subset", state["plot_colormaps"][state["colormap_discrete"]])
            dpg.bind_item_theme("mmpa_plot_subset", apply_line_chart_theme(state))
            register_plot_context_popup(
                state,
                context_key="mmpa_subset_plot_context",
                plot_tag="mmpa_plot_subset",
                x_axis_tag="mmpa_x_subset",
                y_axis_tag="mmpa_y_subset",
                theme_kind="plot",
            )


# -----------------------------------------------------------------------------
# 8. Refresh global plot
# -----------------------------------------------------------------------------
def _refresh_global_plot(state: dict[str, Any]) -> None:
    """
    Update the global plot series and labels without recreating the plot.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if not dpg.does_item_exist("mmpa_global_curve"):
        _ensure_mmpa_plots_created(state)

    activity = dpg.get_value("mmpa_activity_type")
    is_log_scale = activity in state["nM_activity_types"]
    thresh_rect = [dpg.get_value("mmpa_delta_thresh")]

    if activity == "No activity found":
        dpg.configure_item("mmpa_global_curve", x=[], y=[])
        dpg.configure_item("mmpa_global_thresh_rect", x=[])
        dpg.configure_item("mmpa_plot_global", label=f'Δ{"p" if is_log_scale else ""}{activity} distribution (Global)')
        return

    dist = _load_distribution_csv(state, activity)
    values = dist.get("GLOBAL", [])
    centers, counts, edges = _hist_counts(values)
    centers_shifted = [c - 0.05 for c in centers]
    dpg.configure_item("mmpa_global_curve", x=centers_shifted, y=counts, label=f'Δ{"p" if is_log_scale else ""}{activity} distribution')
    dpg.configure_item("mmpa_global_thresh_rect", x=thresh_rect)
    dpg.configure_item("mmpa_plot_global", label=f'Δ{"p" if is_log_scale else ""}{activity} distribution (Global)')


# -----------------------------------------------------------------------------
# 9. Refresh subset plot
# -----------------------------------------------------------------------------
def _refresh_subset_plot(state: dict[str, Any]) -> None:
    """
    Update the subset plot series and labels without recreating the plot.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if not dpg.does_item_exist("mmpa_subset_curve"):
        _ensure_mmpa_plots_created(state)

    activity = dpg.get_value("mmpa_activity_type")
    is_log_scale = activity in state["nM_activity_types"]
    subset = dpg.get_value("mmpa_subset_choice")
    threshold_value = float(dpg.get_value("mmpa_delta_thresh"))
    thresh_rect = [threshold_value]

    if activity == "No activity found":
        dpg.configure_item("mmpa_subset_curve", x=[], y=[])
        dpg.configure_item("mmpa_subset_thresh_rect", x=[])
        dpg.configure_item("mmpa_plot_subset", label=f'Δ{"p" if is_log_scale else ""}{activity} distribution ({subset.replace("subset_", "Subset ")})  |  Transformations: 0')
        return

    dist = _load_distribution_csv(state, activity)
    values = dist.get(subset, [])
    transformations_count = _count_transformations_above_threshold(values, threshold_value)
    centers, counts, edges = _hist_counts(values)
    centers_shifted = [c - 0.05 for c in centers]
    dpg.configure_item("mmpa_subset_curve", x=centers_shifted, y=counts, label=f'Δ{"p" if is_log_scale else ""}{activity} distribution')
    dpg.configure_item("mmpa_subset_thresh_rect", x=thresh_rect)
    dpg.configure_item("mmpa_plot_subset", label=f'Δ{"p" if is_log_scale else ""}{activity} distribution ({subset.replace("subset_", "Subset ")})  |  Transformations: {transformations_count}')


# -----------------------------------------------------------------------------
# 10. Mount mmpa plots
# -----------------------------------------------------------------------------
def mount_mmpa_plots(state: dict[str, Any]) -> None:
    """
    Ensure the plot window exists, then create and populate both plots.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    # Create empty plots then fill them via refresh functions
    _ensure_mmpa_plots_created(state)
    _refresh_global_plot(state)
    _refresh_subset_plot(state)

    for tag in ["mmpa_x_global", "mmpa_y_global", "mmpa_x_subset", "mmpa_y_subset"]:
        dpg.fit_axis_data(tag)
from app.utils.app_logger import log_event, log_exception
