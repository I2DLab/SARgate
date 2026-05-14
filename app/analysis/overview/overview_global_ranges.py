"""
========================
overview_global_ranges.py

Global Activity Ranges Popup
- Builds ONE plot per subset (once per activity type)
- Shared X range across all subsets (union of min/max)
- Shared color scale across all subsets: (global min_count .. global max_count)
- Left column: subset visibility checkboxes (+ Select/Deselect all)
- Right column: stacked plots (show/hide only, no rebuild on toggle)
========================
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Define module configuration and shared state
# 3. Safe tag
# 4. Subset sort key
# 5. Get subset list
# 6. Extract activity values for subset
# 7. Get scale mode and ticks
# 8. Axis limits from ticks
# 9. Count values
# 10. T from count
# 11. Build one subset plot
# 12. Get selected activity
# 13. Ensure popup
# 14. Build layout
# 15. Rebuild subset checkboxes
# 16. Toggle one by sender
# 17. Toggle all
# 18. Toggle one
# 19. Build plots for activity
# 20. Show all subsets activity ranges
# 21. Job signature
# 22. Invalidate global ranges popup cache

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import re
import math
import dearpygui.dearpygui as dpg
from typing import Any
from app.utils.app_logger import log_event, log_settings
from app.gui.themes_manager import (
    apply_enrich_plot_theme,
    get_continuous_colormap_color,
)
from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress


# -----------------------------------------------------------------------------
# 2. Define module configuration and shared state
# -----------------------------------------------------------------------------

_NUMPAT = re.compile(r"([<>]=?|=)\s*([0-9]*\.?[0-9]+)")


# -----------------------------------------------------------------------------
# 3. Safe tag
# -----------------------------------------------------------------------------
def _safe_tag(s: str) -> str:
    """
    Execute the safe tag routine.
    
    Args:
        s (str): Parameter accepted by this routine.
    
    Returns:
        str: Value produced by the routine.
    """
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s))


# -----------------------------------------------------------------------------
# 4. Subset sort key
# -----------------------------------------------------------------------------
def _subset_sort_key(s: str) -> Any:
    """
    Execute the subset sort key routine.
    
    Args:
        s (str): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    m = re.search(r"(\d+)$", str(s))
    return (0, int(m.group(1))) if m else (1, str(s))


# -----------------------------------------------------------------------------
# 5. Get subset list
# -----------------------------------------------------------------------------
def _get_subset_list(state: dict[str, Any]) -> Any:
    """
    Return subset list.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    props = state.get("properties_dict", {})
    subs = [k for k in props.keys() if isinstance(k, str)]
    subs.sort(key=_subset_sort_key)
    return subs


# -----------------------------------------------------------------------------
# 6. Extract activity values for subset
# -----------------------------------------------------------------------------
def _extract_activity_values_for_subset(
    subset: str,
    selected_activity: str,
    state: dict[str, Any]
) -> Any:
    """
    Execute the extract activity values for subset routine.
    
    Args:
        subset (str): Parameter accepted by this routine.
        selected_activity (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    props = state["properties_dict"]
    log_scaled_activities = set(state.get("nM_activity_types", [])) | set(state.get("ug/mL_activities", []))
    values, mol_ids_with_activity = [], set()

    for mol_id, mol_data in props.get(subset, {}).items():
        acts = mol_data.get("activities", {})
        found = False
        for blk in acts.values():
            for _, v in blk.items():
                sv = str(v)
                if sv.startswith(selected_activity):
                    m = _NUMPAT.search(sv)
                    if m:
                        try:
                            parsed_value = float(m.group(2))
                            if selected_activity in log_scaled_activities and parsed_value <= 0:
                                continue
                            values.append(parsed_value)
                            found = True
                        except:
                            pass
        if found:
            mol_ids_with_activity.add(mol_id)

    return values, mol_ids_with_activity


# -----------------------------------------------------------------------------
# 7. Get scale mode and ticks
# -----------------------------------------------------------------------------
def _get_scale_mode_and_ticks(selected_activity: str, values: Any, state: dict[str, Any]) -> Any:
    """
    Return scale mode and ticks.
    
    Args:
        selected_activity (Any): Parameter accepted by this routine.
        values (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    nM_types = state["nM_activity_types"]
    perc_types = state["percent_activities"]
    dimless = state["dimensionless"]
    ugml_types = state["ug/mL_activities"]
    ummin = state["uM/min_activities"]

    if selected_activity in nM_types or selected_activity in ugml_types:
        values = [v for v in values if float(v) > 0]

    if not values:
        raise ValueError(f"No positive numeric values available for '{selected_activity}'")

    min_v, max_v = min(values), max(values)

    if selected_activity in nM_types:
        scale_mode = "log"
        pmin, pmax = math.floor(math.log10(min_v)), math.ceil(math.log10(max_v))
        low_tick, high_tick = 10**pmin, 10**pmax
        if high_tick <= low_tick:
            high_tick = low_tick * 10

        tick_values = [10**p for p in range(pmin, pmax + 1)]

        def fmt(v: Any) -> Any:
            """
            Execute the fmt routine.
            
            Args:
                v (Any): Input accepted by this routine.
            
            Returns:
                Any: Value returned by the routine.
            """
            if v < 1:
                return f"{v*1000:g} pM"
            if v < 1000:
                return f"{v:g} nM"
            if v < 1e6:
                return f"{v/1000:g} μM"
            return f"{v/1e6:g} mM"

        tick_labels = [(fmt(v), v) for v in tick_values]

    elif selected_activity in perc_types:
        scale_mode = "linear"
        low_tick, high_tick = min_v, max_v
        tick_values = [low_tick + i * (high_tick - low_tick) / 4 for i in range(5)]
        tick_labels = [(f"{v:g} %", v) for v in tick_values]

    elif selected_activity in dimless:
        scale_mode = "linear"
        low_tick, high_tick = min_v, max_v
        tick_values = [low_tick + i * (high_tick - low_tick) / 4 for i in range(5)]
        tick_labels = [(f"{v:g}", v) for v in tick_values]

    elif selected_activity in ugml_types:
        scale_mode = "log"
        pmin, pmax = math.floor(math.log10(min_v)), math.ceil(math.log10(max_v))
        low_tick, high_tick = 10**pmin, 10**pmax
        tick_values = [10**p for p in range(pmin, pmax + 1)]
        tick_labels = [(f"{v:g} µg/mL", v) for v in tick_values]

    elif selected_activity in ummin:
        scale_mode = "linear"
        low_tick, high_tick = min_v, max_v
        tick_values = [low_tick + i * (high_tick - low_tick) / 4 for i in range(5)]
        tick_labels = [(f"{v:g} µM/min", v) for v in tick_values]

    else:
        scale_mode = "linear"
        low_tick, high_tick = min_v, max_v
        tick_values = [low_tick + i * (high_tick - low_tick) / 4 for i in range(5)]
        tick_labels = [(f"{v:g}", v) for v in tick_values]

    clean_ticks = []
    for lab, val in tick_labels:
        try:
            fv = float(val)
            if not (math.isnan(fv) or math.isinf(fv)):
                clean_ticks.append((str(lab), fv))
        except:
            pass

    if scale_mode == "log":
        all_vals = [v for _, v in tick_labels]
        pmin, pmax = math.floor(math.log10(min(all_vals))), math.ceil(math.log10(max(all_vals)))
        for p in range(pmin, pmax):
            base = 10**p
            for m in range(2, 10):
                clean_ticks.append(("", base * m))

    return scale_mode, low_tick, high_tick, tuple(clean_ticks)


# -----------------------------------------------------------------------------
# 8. Axis limits from ticks
# -----------------------------------------------------------------------------
def _axis_limits_from_ticks(scale_mode: Any, low_tick: Any, high_tick: Any) -> Any:
    """
    Execute the axis limits from ticks routine.
    
    Args:
        scale_mode (Any): Parameter accepted by this routine.
        low_tick (Any): Parameter accepted by this routine.
        high_tick (Any): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    if scale_mode == "log":
        log_low, log_high = math.log10(low_tick), math.log10(high_tick)
        pad = 0.01 * (log_high - log_low)
        return 10 ** (log_low - pad), 10 ** (log_high + pad)

    span = high_tick - low_tick
    if span <= 0:
        span = abs(low_tick) * 0.01 if low_tick != 0 else 1
    pad = 0.01 * span
    return low_tick - pad, high_tick + pad


# -----------------------------------------------------------------------------
# 9. Count values
# -----------------------------------------------------------------------------
def _count_values(values: Any) -> Any:
    """
    Execute the count values routine.
    
    Args:
        values (Any): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


# -----------------------------------------------------------------------------
# 10. T from count
# -----------------------------------------------------------------------------
def _t_from_count(cnt: Any, global_min_c: Any, global_max_c: Any) -> Any:
    """
    Execute the t from count routine.
    
    Args:
        cnt (Any): Parameter accepted by this routine.
        global_min_c (Any): Parameter accepted by this routine.
        global_max_c (Any): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    if global_max_c is None or global_min_c is None:
        return 0.0
    if global_max_c <= global_min_c:
        return 0.0
    return (cnt - global_min_c) / (global_max_c - global_min_c)


# -----------------------------------------------------------------------------
# 11. Build one subset plot
# -----------------------------------------------------------------------------
def _build_one_subset_plot(
    subset: str,
    selected_activity: str,
    scale_mode: Any,
    ticks: Any,
    x_low: Any,
    x_high: Any,
    global_min_c: Any,
    global_max_c: Any,
    plot_h: Any,
    parent: Any,
    first: Any,
    last: Any,
    state: dict[str, Any]
) -> Any:
    """
    Build one subset plot.
    
    Args:
        subset (str): Parameter accepted by this routine.
        selected_activity (Any): Parameter accepted by this routine.
        scale_mode (Any): Parameter accepted by this routine.
        ticks (Any): Parameter accepted by this routine.
        x_low (Any): Parameter accepted by this routine.
        x_high (Any): Parameter accepted by this routine.
        global_min_c (Any): Parameter accepted by this routine.
        global_max_c (Any): Parameter accepted by this routine.
        plot_h (Any): Parameter accepted by this routine.
        parent (Any): Parameter accepted by this routine.
        first (Any): Parameter accepted by this routine.
        last (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    subtag = _safe_tag(subset)
    row_tag = f"global_act_row__{subtag}"
    plot_tag = f"global_act_plot__{subtag}"
    ax_x = f"global_act_ax_x__{subtag}"
    ax_y = f"global_act_ax_y__{subtag}"

    values, mols_with = _extract_activity_values_for_subset(subset, selected_activity, state)
    n_tot = len(state["properties_dict"].get(subset, {}))
    n_with = len(mols_with)

    with dpg.group(horizontal=True, parent=parent, tag=row_tag):
        with dpg.group():
            subset_label = subset.replace("subset_", "Subset ")
            dpg.add_text(f"{subset_label}\n{n_with}/{n_tot}\nmolecules  ")

        with dpg.plot(
            tag=plot_tag,
            height=plot_h,
            width=-1,
            no_menus=True,
            no_mouse_pos=True,
            no_frame=True,
        ):
            dpg.add_plot_axis(
                dpg.mvXAxis,
                tag=ax_x,
                no_gridlines=True,
                no_highlight=True,
                no_tick_marks=False,
                no_side_switch=True,
                no_menus=True,
                opposite=False,
                no_tick_labels=True,
                scale=dpg.mvPlotScale_Log10 if scale_mode == "log" else dpg.mvPlotScale_Linear,
            )
            dpg.add_plot_axis(
                dpg.mvYAxis,
                tag=ax_y,
                no_gridlines=True,
                no_highlight=True,
                no_label=False,
                no_tick_marks=True,
                no_side_switch=True,
                no_menus=True,
                no_tick_labels=True,
            )
        dpg.set_axis_limits(ax_x, x_low, x_high)
        dpg.set_axis_limits(ax_y, 0, 1)
        dpg.bind_item_theme(plot_tag, apply_enrich_plot_theme(state))

        if values:
            counts = _count_values(values)
            for val, cnt in counts.items():
                t = _t_from_count(cnt, global_min_c, global_max_c)
                col = get_continuous_colormap_color(t, state)
                sid = dpg.add_line_series(x=[val, val], y=[0, 1], parent=ax_x)
                with dpg.theme() as th:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(sid, th)

    return row_tag


# -----------------------------------------------------------------------------
# 12. Refresh visible edge ticks
# -----------------------------------------------------------------------------
def _refresh_visible_edge_ticks(state: dict[str, Any]) -> None:
    """
    Show X ticks only on the first and last currently visible subset plots.

    Args:
        state (dict[str, Any]): Shared application state.

    Returns:
        None: This routine updates the UI in place.
    """
    plot_tags = state.get("global_ranges_plot_tags", {})
    vis = state.get("global_ranges_visible_subsets", set())
    visible_subs = [subset for subset in _get_subset_list(state) if subset in vis]

    first_visible = visible_subs[0] if visible_subs else None
    last_visible = visible_subs[-1] if visible_subs else None
    major_tick_values = state.get("global_ranges_major_tick_values", [])
    tick_color = tuple(state["themes"][state["theme_name"]]["Text Color"])

    def _apply_major_tick_theme(tag: str) -> None:
        """
        Apply the visual style used for manually drawn major ticks.
        """
        with dpg.theme() as tick_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, tick_color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.0, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(tag, tick_theme)

    def _draw_manual_major_ticks(axis_tag: str, values: list[float], y0: float, y1: float, prefix: str) -> None:
        """
        Draw one short vertical segment per major tick position.
        """
        for idx, x_val in enumerate(values):
            tick_tag = f"{prefix}__{idx}"
            if dpg.does_item_exist(tick_tag):
                dpg.delete_item(tick_tag)
            dpg.add_line_series(
                x=[x_val, x_val],
                y=[y0, y1],
                parent=axis_tag,
                tag=tick_tag,
            )
            _apply_major_tick_theme(tick_tag)

    for subset in _get_subset_list(state):
        ax_x = f"global_act_ax_x__{_safe_tag(subset)}"
        ax_y = f"global_act_ax_y__{_safe_tag(subset)}"
        subtag = _safe_tag(subset)
        if not dpg.does_item_exist(ax_x):
            continue

        is_first = subset == first_visible
        is_last = subset == last_visible
        show_axis_ticks = is_first or is_last

        if show_axis_ticks:
            dpg.set_axis_ticks(ax_x, state.get("global_ranges_ticks", ()))
        else:
            dpg.set_axis_ticks(ax_x, ())

        dpg.configure_item(
            ax_x,
            no_tick_marks=False,
            no_tick_labels=not show_axis_ticks,
            opposite=is_first,
        )

        for major_tag in (
            f"global_act_major_tick_top__{subtag}",
            f"global_act_major_tick_bottom__{subtag}",
        ):
            for idx in range(len(major_tick_values) + 2):
                seg_tag = f"{major_tag}__{idx}"
                if dpg.does_item_exist(seg_tag):
                    dpg.delete_item(seg_tag)

        if not dpg.does_item_exist(ax_y):
            continue

        if is_first and major_tick_values:
            _draw_manual_major_ticks(
                ax_y,
                major_tick_values,
                0.92,
                1.0,
                f"global_act_major_tick_top__{subtag}",
            )

        if is_last and major_tick_values:
            _draw_manual_major_ticks(
                ax_y,
                major_tick_values,
                0.0,
                0.08,
                f"global_act_major_tick_bottom__{subtag}",
            )


# -----------------------------------------------------------------------------
# 12. Get selected activity
# -----------------------------------------------------------------------------
def _get_selected_activity(state: dict[str, Any]) -> Any:
    """
    Return selected activity.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    if dpg.does_item_exist("enrichment_activity_choice"):
        return dpg.get_value("enrichment_activity_choice")
    return state.get("selected_enrichment_activity")


# -----------------------------------------------------------------------------
# 13. Ensure popup
# -----------------------------------------------------------------------------
def _ensure_popup(state: dict[str, Any]) -> None:
    """
    Ensure popup.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if dpg.does_item_exist("global_act_ranges_popup"):
        return
    IMG_W = int(state.get("notes_img_width", 900))
    with dpg.window(
        label="Global activity ranges",
        show=False,
        tag="global_act_ranges_popup",
        width=IMG_W + 155,
        height=775,
        no_scrollbar=False,
        horizontal_scrollbar=True,
        no_scroll_with_mouse=False,
        no_resize=False,
    ):
        pass


# -----------------------------------------------------------------------------
# 14. Build layout
# -----------------------------------------------------------------------------
def _build_layout(state: dict[str, Any]) -> None:
    """
    Build layout.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if dpg.does_item_exist("global_act_ranges_main"):
        return

    with dpg.table(
        parent="global_act_ranges_popup",
        tag="global_act_ranges_main",
        header_row=False,
        borders_innerH=False, borders_outerH=False,
        borders_innerV=False, borders_outerV=False,
        resizable=True,
        policy=dpg.mvTable_SizingStretchProp
    ):
        dpg.add_table_column(init_width_or_weight=15)
        dpg.add_table_column(init_width_or_weight=85)

        with dpg.table_row():
            with dpg.child_window(width=-1, height=-1, tag="global_act_ranges_left", border=True):
                dpg.add_checkbox(
                    label="Select all",
                    tag="global_act_ranges_all",
                    default_value=True,
                    callback=lambda s, a: _toggle_all(a, state),
                )
                dpg.add_separator()
                with dpg.child_window(
                    width=-1,
                    height=-1,
                    tag="global_act_ranges_controls",
                    border=False,
                    no_scrollbar=False,
                    horizontal_scrollbar=False,
                    no_scroll_with_mouse=False,
                ):
                    pass
                if dpg.does_item_exist("manager_panel_theme"):
                    dpg.bind_item_theme("global_act_ranges_controls", "manager_panel_theme")

            with dpg.child_window(width=-1, height=-1, tag="global_act_ranges_right", border=True):
                dpg.add_text("", tag="global_act_ranges_header")
                dpg.add_separator()
                with dpg.group(tag="global_act_ranges_plotstack"):
                    pass

    if dpg.does_item_exist("manager_panel_theme"):
        if dpg.does_item_exist("global_act_ranges_left"):
            dpg.bind_item_theme("global_act_ranges_left", "manager_panel_theme")
        if dpg.does_item_exist("global_act_ranges_controls"):
            dpg.bind_item_theme("global_act_ranges_controls", "manager_panel_theme")


# -----------------------------------------------------------------------------
# 15. Rebuild subset checkboxes
# -----------------------------------------------------------------------------
def _rebuild_subset_checkboxes(state: dict[str, Any]) -> None:
    """
    Execute the rebuild subset checkboxes routine.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    subs = _get_subset_list(state)

    vis = state.get("global_ranges_visible_subsets")
    if vis is None:
        vis = set(subs)
        state["global_ranges_visible_subsets"] = vis

    # mapping: checkbox_tag -> subset
    cb2subset = state.setdefault("global_ranges_cb2subset", {})
    cb2subset.clear()

    for ch in dpg.get_item_children("global_act_ranges_controls", 1):
        dpg.delete_item(ch)

    for subset in subs:
        cb_tag = f"global_act_ranges_cb__{_safe_tag(subset)}"
        cb2subset[cb_tag] = subset

        dpg.add_checkbox(
            label=subset.replace("subset_", "Subset "),
            tag=cb_tag,
            default_value=(subset in vis),
            parent="global_act_ranges_controls",
            callback=lambda s, a: _toggle_one_by_sender(s, a, state),
        )

    all_on = bool(subs) and all((s in vis) for s in subs)
    if dpg.does_item_exist("global_act_ranges_all"):
        dpg.set_value("global_act_ranges_all", all_on)


# -----------------------------------------------------------------------------
# 16. Toggle one by sender
# -----------------------------------------------------------------------------
def _toggle_one_by_sender(sender: Any, on: Any, state: dict[str, Any]) -> None:
    # ignore callbacks while bulk-updating checkbox values
    """
    Toggle one by sender.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        on (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if state.get("global_ranges_bulk_toggle", False):
        return

    subset = state.get("global_ranges_cb2subset", {}).get(sender)
    if not subset:
        return

    _toggle_one(subset, on, state)


# -----------------------------------------------------------------------------
# 17. Toggle all
# -----------------------------------------------------------------------------
def _toggle_all(select_all: Any, state: dict[str, Any]) -> None:
    """
    Toggle all.
    
    Args:
        select_all (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    subs = _get_subset_list(state)

    vis = state.get("global_ranges_visible_subsets")
    if vis is None:
        vis = set()
        state["global_ranges_visible_subsets"] = vis

    vis.clear()
    if select_all:
        vis.update(subs)

    # prevent per-checkbox callback spam
    state["global_ranges_bulk_toggle"] = True
    try:
        for subset in subs:
            cb = f"global_act_ranges_cb__{_safe_tag(subset)}"
            if dpg.does_item_exist(cb):
                dpg.set_value(cb, bool(select_all))
    finally:
        state["global_ranges_bulk_toggle"] = False

    # show/hide rows
    tags = state.get("global_ranges_plot_tags", {})
    for subset in subs:
        row_tag = tags.get(subset)
        if row_tag and dpg.does_item_exist(row_tag):
            dpg.configure_item(row_tag, show=bool(select_all))

    _refresh_visible_edge_ticks(state)


# -----------------------------------------------------------------------------
# 18. Toggle one
# -----------------------------------------------------------------------------
def _toggle_one(subset: str, on: Any, state: dict[str, Any]) -> None:
    """
    Toggle one.
    
    Args:
        subset (str): Parameter accepted by this routine.
        on (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    vis = state.get("global_ranges_visible_subsets")
    if vis is None:
        vis = set()
        state["global_ranges_visible_subsets"] = vis

    if on:
        vis.add(subset)
    else:
        vis.discard(subset)

    row_tag = state.get("global_ranges_plot_tags", {}).get(subset)
    if row_tag and dpg.does_item_exist(row_tag):
        dpg.configure_item(row_tag, show=bool(on))

    _refresh_visible_edge_ticks(state)

    subs = _get_subset_list(state)
    all_on = bool(subs) and all((s in vis) for s in subs)
    if dpg.does_item_exist("global_act_ranges_all"):
        # prevent triggering _toggle_all via user interaction (it won't, but keep consistent)
        dpg.set_value("global_act_ranges_all", all_on)


# -----------------------------------------------------------------------------
# 19. Build plots for activity
# -----------------------------------------------------------------------------
def _build_plots_for_activity(selected_activity: str, state: dict[str, Any]) -> None:

    """
    Build plots for activity.
    
    Args:
        selected_activity (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)

    subs = _get_subset_list(state)
    if not subs:
        return

    all_vals = []
    global_min_c, global_max_c = None, None

    total_subs = max(1, len(subs))
    for subset_idx, subset in enumerate(subs, start=1):
        vals, _ = _extract_activity_values_for_subset(subset, selected_activity, state)
        all_vals.extend(vals)
        if vals:
            cdict = _count_values(vals)
            mn, mx = min(cdict.values()), max(cdict.values())
            global_min_c = mn if global_min_c is None else min(global_min_c, mn)
            global_max_c = mx if global_max_c is None else max(global_max_c, mx)
        if subset_idx == total_subs or subset_idx % max(1, total_subs // 10) == 0:
            set_loading_screen_progress(state, 1 + (subset_idx / total_subs) * 14)

    if not all_vals:
        dpg.configure_item("global_act_ranges_header", default_value=f"{selected_activity} | no numeric values found")
        return

    scale_mode, low_tick, high_tick, ticks = _get_scale_mode_and_ticks(selected_activity, all_vals, state)
    x_low, x_high = _axis_limits_from_ticks(scale_mode, low_tick, high_tick)
    state["global_ranges_ticks"] = ticks
    state["global_ranges_major_tick_values"] = [float(val) for lab, val in ticks if str(lab).strip()]

    if global_min_c is None:
        global_min_c, global_max_c = 0, 0

    units = "nM" if selected_activity in state.get("nM_activity_types", []) else \
            "%" if selected_activity in state.get("percent_activities", []) else \
            "µg/mL" if selected_activity in state.get("ug/mL_activities", []) else \
            "µM/min" if selected_activity in state.get("uM/min_activities", []) else ""

    dpg.configure_item(
        "global_act_ranges_header",
        default_value=(
            f"Activity: {selected_activity}    "
            f"Activity range: {low_tick:g} ↔ {high_tick:g} {units}    "
            f"Overlaps range: {global_min_c} ↔ {global_max_c}"
        ),
    )
    set_loading_screen_progress(state, 16)

    for ch in dpg.get_item_children("global_act_ranges_plotstack", 1):
        dpg.delete_item(ch)

    plot_h = state["enrich_plot_win_height"] / 2

    tags = {}
    for n, subset in enumerate(subs):
        row_tag = _build_one_subset_plot(
            subset=subset,
            selected_activity=selected_activity,
            scale_mode=scale_mode,
            ticks=ticks,
            x_low=x_low,
            x_high=x_high,
            global_min_c=global_min_c,
            global_max_c=global_max_c,
            plot_h=plot_h,
            parent="global_act_ranges_plotstack",
            first=(n == 0),
            last=(n == len(subs) - 1),
            state=state,
        )
        tags[subset] = row_tag
        rendered = n + 1
        if rendered == total_subs or rendered % max(1, total_subs // 10) == 0:
            set_loading_screen_progress(state, 16 + (rendered / total_subs) * 82)

    state["global_ranges_plot_tags"] = tags
    state["global_ranges_last_activity"] = selected_activity
    state["global_ranges_refresh_colors"] = lambda: _build_plots_for_activity(
        state.get("global_ranges_last_activity", selected_activity),
        state,
    )

    # apply visibility (and also guarantees checkbox->row linkage works now)
    vis = state.get("global_ranges_visible_subsets")
    if vis is None:
        vis = set(subs)
        state["global_ranges_visible_subsets"] = vis

    for subset in subs:
        row_tag = tags.get(subset)
        if row_tag and dpg.does_item_exist(row_tag):
            dpg.configure_item(row_tag, show=(subset in vis))

    _refresh_visible_edge_ticks(state)
    set_loading_screen_progress(state, 99)

    if dpg.does_item_exist("cover_layer") and state["first_loading"] == False:
        dpg.delete_item("cover_layer")
        


# -----------------------------------------------------------------------------
# 20. Show all subsets activity ranges
# -----------------------------------------------------------------------------
def show_all_subsets_activity_ranges(state: dict[str, Any]) -> None:
    """
    Display all subsets activity ranges.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    log_event("Overview", "Drawing 'global activity ranges' plot", indent=1)
    log_settings("Overview", indent=2, activity=_get_selected_activity(state), visible_subsets=len(_get_subset_list(state)), colormap=state.get("colormap_continuous"))
    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)
    _ensure_popup(state)
    set_loading_screen_progress(state, 2)

    sig = _job_signature(state)
    if state.get("global_ranges_job_sig") != sig:
        invalidate_global_ranges_popup_cache(state, hard_ui_reset=True)
        state["global_ranges_job_sig"] = sig
    set_loading_screen_progress(state, 3)

    if not dpg.does_item_exist("global_act_ranges_main"):
        dpg.delete_item("global_act_ranges_popup", children_only=True)
        _build_layout(state)
        _rebuild_subset_checkboxes(state)
    set_loading_screen_progress(state, 4)

    selected_activity = _get_selected_activity(state)
    if not selected_activity or selected_activity == "No activities":
        if dpg.does_item_exist("global_act_ranges_header"):
            dpg.configure_item("global_act_ranges_header", default_value="No activity selected.")
    else:
        if state.get("global_ranges_last_activity") != selected_activity or not state.get("global_ranges_plot_tags"):
            _build_plots_for_activity(selected_activity, state)
            # keep checkboxes consistent if subsets changed
            _rebuild_subset_checkboxes(state)

        # re-apply visibility in case user had toggled before build
        tags = state.get("global_ranges_plot_tags", {})
        vis = state.get("global_ranges_visible_subsets", set(tags.keys()))
        for subset, row_tag in tags.items():
            if row_tag and dpg.does_item_exist(row_tag):
                dpg.configure_item(row_tag, show=(subset in vis))

        _refresh_visible_edge_ticks(state)

    dpg.show_item("global_act_ranges_popup")
    dpg.configure_item("global_act_ranges_popup", collapsed=False)
    dpg.focus_item("global_act_ranges_popup")
    state["global_ranges_refresh_colors"] = lambda: _build_plots_for_activity(
        state.get("global_ranges_last_activity", selected_activity),
        state,
    )
    set_loading_screen_progress(state, 100)


# -----------------------------------------------------------------------------
# 21. Job signature
# -----------------------------------------------------------------------------
def _job_signature(state: dict[str, Any]) -> Any:
    # cheap signature: sorted subset keys + counts
    """
    Execute the job signature routine.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    props = state.get("properties_dict", {})
    subs = [k for k in props.keys() if isinstance(k, str)]
    subs.sort(key=_subset_sort_key)
    return (tuple(subs), len(subs))


# -----------------------------------------------------------------------------
# 22. Invalidate global ranges popup cache
# -----------------------------------------------------------------------------
def invalidate_global_ranges_popup_cache(
    state: dict[str, Any],
    hard_ui_reset: bool = False
) -> None:
    """
    Call this right after loading a new job (i.e., when properties_dict changes).
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
        hard_ui_reset (Any): Parameter accepted by this routine. Defaults to the configured value.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    for k in (
        "global_ranges_last_activity",
        "global_ranges_plot_tags",
        "global_ranges_visible_subsets",
        "global_ranges_cb2subset",
    ):
        state.pop(k, None)

    state["global_ranges_bulk_toggle"] = False

    if hard_ui_reset and dpg.does_item_exist("global_act_ranges_popup"):
        dpg.delete_item("global_act_ranges_popup", children_only=True)
