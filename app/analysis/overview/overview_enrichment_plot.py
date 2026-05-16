"""
=================
overview_enrichment_plot.py

Overview Enrichment Plot Functions
=================
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Update enrich colorbar labels
# 3. Draw global activity bar
# 4. Build enrichment layout
# 5. Update enrichment rgroup

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import re
import math
import dearpygui.dearpygui as dpg
from typing import Any
from app.gui.themes_manager import (
    apply_enrich_plot_theme, 
    get_continuous_colormap_color,
    apply_colormap_theme,
    change_font_type
)
from app.analysis.overview.overview_global_ranges import show_all_subsets_activity_ranges


# -----------------------------------------------------------------------------
# 2. Update enrich colorbar labels
# -----------------------------------------------------------------------------
def _update_enrich_colorbar_labels(state: dict[str, Any]) -> None:
    gmax = state.get("global_activity_max_count")
    gmin = state.get("global_activity_min_count")
    rmax = state.get("rgroup_max_count")
    rmin = state.get("rgroup_min_count")

    max_candidates = [v for v in (gmax, rmax) if v is not None]
    min_candidates = [v for v in (gmin, rmin) if v is not None]

    if not max_candidates or not min_candidates:
        return

    new_max = max(max_candidates)
    new_min = min(min_candidates)

    if dpg.does_item_exist("enrich_colorscale_max_label"):
        dpg.configure_item("enrich_colorscale_max_label", default_value=f"{new_max}")
    if dpg.does_item_exist("enrich_colorscale_min_label"):
        dpg.configure_item("enrich_colorscale_min_label", default_value=f"{new_min}")


# -----------------------------------------------------------------------------
# 3. Draw global activity bar
# -----------------------------------------------------------------------------
def draw_global_activity_bar(id: Any, state: dict[str, Any]) -> None:
    q = id.split("_")
    subset = f"{q[0]}_{q[1]}"

    props = state["properties_dict"]
    selected_activity = state["selected_enrichment_activity"]

    num_pattern = re.compile(r'([<>]=?|=)\s*([0-9]*\.?[0-9]+)')
    values = []

    for mol_id, mol_data in props[subset].items():
        acts = mol_data.get("activities", {})
        for block in acts.values():
            for k, v in block.items():
                sval = str(v)
                if sval.startswith(selected_activity):
                    m = num_pattern.search(sval)
                    if m:
                        try:
                            values.append(float(m.group(2)))
                        except:
                            pass

    if not values:
        return

    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1

    max_count = max(counts.values())
    min_count = min(counts.values())

    state["global_activity_max_count"] = max_count
    state["global_activity_min_count"] = min_count

    _update_enrich_colorbar_labels(state)

    for child in dpg.get_item_children("global_activity_axis_x", 1):
        dpg.delete_item(child)

    for val, cnt in counts.items():
        if max_count > 1:
            t = (cnt - 1) / (max_count - 1)  # 1 -> 0.0, max_count -> 1.0
        else:
            t = 0.0  # all counts == 1 -> use minimum colormap color

        col = get_continuous_colormap_color(t, state)

        sid = dpg.add_line_series(
            x=[val, val],
            y=[0, 1],
            parent="global_activity_axis_x"
        )

        with dpg.theme() as th:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(sid, th)


# -----------------------------------------------------------------------------
# 4. Build enrichment layout
# -----------------------------------------------------------------------------
def build_enrichment_layout(id: Any, state: dict[str, Any]) -> Any:
    state["current_enrichment_id"] = id

    q = id.split("_")
    subset = f"{q[0]}_{q[1]}"

    props_dict = state["properties_dict"]
    bio_dict   = state["bioact_types_dict"]

    nM_types   = state["nM_activity_types"]
    perc_types = state["percent_activities"]
    dimless    = state["dimensionless"]
    ugml_types = state["ug/mL_activities"]
    ummin      = state["uM/min_activities"]

    acts = bio_dict.get(subset, {}).get("bioactivities", [])
    if not acts:
        acts = ["No activities"]

    prev = state.get("selected_enrichment_activity")
    if prev not in acts:
        selected = acts[0]
    else:
        selected = prev
    

    if dpg.does_item_exist("enrichment_activity_choice"):
        selected = dpg.get_value("enrichment_activity_choice")
    else:
        selected = acts[0]

    state["selected_enrichment_activity"] = selected

    numpat = re.compile(r'([<>]=?|=)\s*([0-9]*\.?[0-9]+)')
    values = []
    mol_with_selected_act = set()

    for mol_id, mol_data in props_dict[subset].items():
        acts_dict = mol_data.get("activities", {})
        found_for_this_mol = False
        for blk in acts_dict.values():
            for k, v in blk.items():
                sv = str(v)
                if sv.startswith(selected):
                    m = numpat.search(sv)
                    if m:
                        try:
                            values.append(float(m.group(2)))
                            found_for_this_mol = True
                        except:
                            pass
        if found_for_this_mol:
            mol_with_selected_act.add(mol_id)

    total_mols_subset = len(props_dict[subset])
    num_with_selected = len(mol_with_selected_act)

    state["enrich_subset_total_mols"] = total_mols_subset
    state["enrich_subset_active_mols"] = num_with_selected
    state["rgroup_max_count"] = None
    state["rgroup_min_count"] = None

    if not values or selected == "No activities":
        for tag in ["enrichment_plot", "enrichment_controls_group",
                    "enrichment_colormap_scale", "enrichment_main_group"]:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        return

    min_v = min(values)
    max_v = max(values)

    if selected in nM_types:
        scale_mode = "log"
        pmin = math.floor(math.log10(min_v))
        pmax = math.ceil(math.log10(max_v))
        low_tick  = 10**pmin
        high_tick = 10**pmax
        if high_tick <= low_tick:
            high_tick = low_tick * 10

        tick_values = [10**p for p in range(pmin, pmax+1)]

        def fmt(v: Any) -> Any:
            if v < 1:
                return f"{v*1000:g} pM"
            elif v < 1000:
                return f"{v:g} nM"
            elif v < 1e6:
                return f"{v/1000:g} μM"
            else:
                return f"{v/1e6:g} mM"

        tick_labels = [(fmt(v), v) for v in tick_values]

    elif selected in perc_types:
        scale_mode = "linear"
        low_tick  = min_v
        high_tick = max_v
        tick_values = [low_tick + i*(high_tick-low_tick)/4 for i in range(5)]
        tick_labels = [(f"{v:g} %", v) for v in tick_values]

    elif selected in dimless:
        scale_mode = "linear"
        low_tick  = min_v
        high_tick = max_v
        tick_values = [low_tick + i*(high_tick-low_tick)/4 for i in range(5)]
        tick_labels = [(f"{v:g}", v) for v in tick_values]

    elif selected in ugml_types:
        scale_mode = "log"
        pmin = math.floor(math.log10(min_v))
        pmax = math.ceil(math.log10(max_v))
        low_tick  = 10**pmin
        high_tick = 10**pmax
        tick_values = [10**p for p in range(pmin, pmax+1)]
        tick_labels = [(f"{v:g} µg/mL", v) for v in tick_values]

    elif selected in ummin:
        scale_mode = "linear"
        low_tick  = min_v
        high_tick = max_v
        tick_values = [low_tick + i*(high_tick-low_tick)/4 for i in range(5)]
        tick_labels = [(f"{v:g} µM/min", v) for v in tick_values]

    else:
        scale_mode = "linear"
        low_tick  = min_v
        high_tick = max_v
        tick_values = [low_tick + i*(high_tick-low_tick)/4 for i in range(5)]
        tick_labels = [(f"{v:g}", v) for v in tick_values]

    clean_ticks = []
    for lab, val in tick_labels:
        try:
            v = float(val)
            if not (math.isnan(v) or math.isinf(v)):
                clean_ticks.append((str(lab), v))
        except:
            pass

    if scale_mode == "log":
        all_vals = [v for _, v in tick_labels]
        pmin = math.floor(math.log10(min(all_vals)))
        pmax = math.ceil(math.log10(max(all_vals)))
        for p in range(pmin, pmax):
            base = 10**p
            for m in range(2, 10):
                clean_ticks.append((" ", base*m))

    clean_ticks = tuple(clean_ticks)

    for tag in ["enrichment_plot", "enrichment_controls_group",
                "enrichment_colormap_scale", "enrichment_main_group"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    items_height = state["enrich_plot_win_height"]
    col_plot_h = items_height - state["win_spacer"] * 8
    enr_plot_h = items_height / 2

    with dpg.group(horizontal=True, parent="enrichment_plot_window", tag="enrichment_main_group"):
        with dpg.child_window(width=50, auto_resize_y=True, no_scrollbar=True,
                              horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
            max_label = "1"
            min_label = "1"
            max_text_w = dpg.get_text_size(max_label)[0]
            min_text_w = dpg.get_text_size(min_label)[0]
            center_w = 50

            with dpg.group(horizontal=True):
                dpg.add_spacer(width=max(0, int((center_w - max_text_w) / 2) + 1))
                dpg.add_text(max_label, tag="enrich_colorscale_max_label")
            dpg.add_colormap_scale(
                label="Overlaps",
                tag="enrichment_colormap_scale",
                min_scale=0, max_scale=0,
                height=col_plot_h, mirror=True
            )
            dpg.bind_colormap("enrichment_colormap_scale",
                              state["colormaps"][state["colormap_continuous"]])
            dpg.bind_item_theme("enrichment_colormap_scale",
                                apply_colormap_theme(state))
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=max(0, int((center_w - min_text_w) / 2) + 1))
                dpg.add_text(min_label, tag="enrich_colorscale_min_label")

        with dpg.group():

            base_label = f"{selected} distribution in {subset.replace('subset_', 'Subset ')} ({num_with_selected}/{total_mols_subset} molecules)"

            with dpg.plot(
                tag="global_activity_plot", label=base_label,
                height=enr_plot_h,
                width=-1,
                no_menus=True, no_mouse_pos=True, no_frame=True
            ):
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    tag="global_activity_axis_x",
                    no_gridlines=True,
                    no_highlight=True,
                    no_tick_marks=False,
                    no_side_switch=True,
                    no_menus=True,
                    no_tick_labels=True,
                    scale=dpg.mvPlotScale_Log10 if scale_mode == "log"
                          else dpg.mvPlotScale_Linear
                )
                dpg.add_plot_axis(
                    dpg.mvYAxis, label=f'{subset.replace("subset_", "S")}',
                    tag="global_activity_axis_y",
                    no_gridlines=True, no_highlight=True,
                    no_label=False, no_tick_marks=True,
                    no_side_switch=True, no_menus=True,
                    no_tick_labels=True
                )

            draw_global_activity_bar(id, state)

            with dpg.plot(
                tag="enrichment_plot",
                height=enr_plot_h,
                width=-1,
                no_menus=True, no_mouse_pos=True, no_frame=True
            ):
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    tag="enrich_x_axis",
                    no_gridlines=True,
                    no_highlight=True,
                    no_tick_marks=False,
                    no_side_switch=True,
                    no_menus=True,
                    no_tick_labels=False,
                    opposite=True,
                    scale=dpg.mvPlotScale_Log10 if scale_mode == "log"
                          else dpg.mvPlotScale_Linear
                )

                dpg.add_plot_axis(
                    dpg.mvYAxis, label="R", tag="enrich_y_axis",
                    no_gridlines=True, no_highlight=True, 
                    no_label=False, no_tick_marks=True,
                    no_side_switch=True, no_menus=True,
                    no_tick_labels=True,
                )

            if scale_mode == "log":
                log_low  = math.log10(low_tick)
                log_high = math.log10(high_tick)
                pad = 0.01 * (log_high - log_low)
                low_lim  = 10 ** (log_low  - pad)
                high_lim = 10 ** (log_high + pad)
            else:
                span = high_tick - low_tick
                if span <= 0:
                    span = abs(low_tick) * 0.01 if low_tick != 0 else 1
                pad = 0.01 * span
                low_lim  = low_tick  - pad
                high_lim = high_tick + pad

            dpg.set_axis_limits("global_activity_axis_x", low_lim, high_lim)
            dpg.set_axis_limits("enrich_x_axis", low_lim, high_lim)
            
            dpg.set_axis_ticks("enrich_x_axis", clean_ticks)

            for y_axis in ["global_activity_axis_y", "enrich_y_axis"]:
                dpg.set_axis_limits(y_axis, 0, 1)

            for plot in ["global_activity_plot", "enrichment_plot"]:
                dpg.bind_item_theme(plot, apply_enrich_plot_theme(state))


    with dpg.group(tag="enrichment_controls_group", parent="image_checkboxes_window"):
        dpg.add_separator()
        dpg.add_text("Activity shown in the enrichment plot:")
        dpg.add_combo(
            items=acts,
            default_value=selected,
            width=-1,
            tag="enrichment_activity_choice",
            callback=lambda s, a: build_enrichment_layout(id, state)
        )
        dpg.add_button(
            label="Show all subsets activity ranges",
            width=-1,
            callback=lambda: show_all_subsets_activity_ranges(state),
        )

    state["enrich_low_tick"]  = low_tick
    state["enrich_high_tick"] = high_tick
    state["enrich_ticks"]     = clean_ticks
    state["enrich_scale_mode"] = scale_mode

    last_r = state.get("last_clicked_button_r")
    if last_r and len(q) == 5:
        update_enrichment_Rgroup(last_r, state)

    state["enrichment_refresh_colors"] = lambda: build_enrichment_layout(
        state.get("current_enrichment_id", id),
        state,
    )


# -----------------------------------------------------------------------------
# 5. Update enrichment rgroup
# -----------------------------------------------------------------------------
def update_enrichment_Rgroup(id: Any, state: dict[str, Any]) -> None:

    query = id.split("_")
    subset = f"{query[0]}_{query[1]}"
    molecule = f"{query[2]}_{query[3]}"
    r_group = query[4]

    props = state["properties_dict"]
    selected_activity = dpg.get_value("enrichment_activity_choice")
    num_pattern = re.compile(r'([<>]=?|=)\s*([0-9]*\.?[0-9]+)')

    dpg.configure_item("enrich_y_axis", label=r_group)

    for child in dpg.get_item_children("enrich_x_axis", 1):
        dpg.delete_item(child)

    try:
        clicked_smiles = state["smiles_rgd_dict"][subset][molecule][r_group]
    except KeyError:
        return

    matched_vals = []
    mol_with_selected_act_and_r = set()

    for mol_id, mol_data in state["smiles_rgd_dict"][subset].items():
        if mol_data.get(r_group) != clicked_smiles:
            continue

        acts = props[subset][mol_id].get("activities", {})
        found_for_this_mol = False
        for act_block in acts.values():
            for k, v in act_block.items():
                sval = str(v)
                if sval.startswith(selected_activity):
                    m = num_pattern.search(sval)
                    if m:
                        try:
                            matched_vals.append(float(m.group(2)))
                            found_for_this_mol = True
                        except:
                            pass
        if found_for_this_mol:
            mol_with_selected_act_and_r.add(mol_id)

    if not matched_vals:
        return

    counts = {}
    for v in matched_vals:
        counts[v] = counts.get(v, 0) + 1

    max_count = max(counts.values())
    min_count = min(counts.values())

    state["rgroup_max_count"] = max_count
    state["rgroup_min_count"] = min_count

    _update_enrich_colorbar_labels(state)


    total_mols_subset = state.get("enrich_subset_total_mols", len(props[subset]))
    num_with_selected = state.get("enrich_subset_active_mols", None)
    if num_with_selected is None:
        seen = set()
        for mol_id, mol_data in props[subset].items():
            acts_dict = mol_data.get("activities", {})
            for blk in acts_dict.values():
                for k, v in blk.items():
                    sval = str(v)
                    if sval.startswith(selected_activity):
                        seen.add(mol_id)
                        break
        num_with_selected = len(seen)
        state["enrich_subset_active_mols"] = num_with_selected

    num_with_selected_and_r = len(mol_with_selected_act_and_r)

    new_label = (
        f"{selected_activity} distribution in {subset.replace('subset_', 'Subset ')} for the selected {r_group} group "
        f"({num_with_selected}/{total_mols_subset} molecules; "
        f"{num_with_selected_and_r}/{total_mols_subset} with {r_group})"
    )
    if dpg.does_item_exist("global_activity_plot"):
        dpg.configure_item("global_activity_plot", label=new_label)

    for val, cnt in counts.items():
        if max_count > 1:
            t = (cnt - 1) / (max_count - 1)
        else:
            t = 0.0

        col = get_continuous_colormap_color(t, state)

        sid = dpg.add_line_series(
            x=[val, val], y=[0, 1],
            parent="enrich_x_axis"
        )
        with dpg.theme() as th:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(sid, th)


    stats_1 = (
        f"Molecules in {subset.replace('subset_', 'Subset ')} with {r_group} group and activity type {selected_activity}: "
    )
    stats_2 = (
        f"{num_with_selected_and_r}/{num_with_selected} ({(num_with_selected_and_r/num_with_selected)*100:.1f}%)"
    )
    all_actives = []  # (mol_id, value)

    num_pattern = re.compile(r'([<>]=?|=)\s*([0-9]*\.?[0-9]+)')
    for mol_id, mol_data in props[subset].items():
        acts = mol_data.get("activities", {})
        for blk in acts.values():
            for k, v in blk.items():
                sval = str(v)
                if sval.startswith(selected_activity):
                    m = num_pattern.search(sval)
                    if m:
                        try:
                            all_actives.append((mol_id, float(m.group(2))))
                        except:
                            pass

    if not all_actives:
        return

    all_actives.sort(key=lambda x: x[1])  
    total_active = len(all_actives)

    perc_list = [5, 10, 25]
    enrichment_results = []

    for perc in perc_list:
        topN = max(1, int(total_active * (perc/100)))
        top_subset = all_actives[:topN]

        count_r = 0
        for mol_id, _ in top_subset:
            if state["smiles_rgd_dict"][subset][mol_id].get(r_group) == clicked_smiles:
                count_r += 1

        perc_val = (count_r / topN) * 100 if topN > 0 else 0
        enrichment_results.append((perc, count_r, topN, perc_val))

    for old_tag in ("enrichment_5_label", "enrichment_10_label", "enrichment_25_label"):
        if dpg.does_item_exist(old_tag):
            dpg.delete_item(old_tag)

    with dpg.group(horizontal=True, parent="r_prop_group"):
        dpg.add_text(stats_1)
        change_font_type(dpg.last_item(), "bold", state)
        dpg.add_text(stats_2)

    for perc, rnum, topN, perc_val in enrichment_results:
        with dpg.group(horizontal=True, parent="r_prop_group"):
            dpg.add_text(default_value=f"{r_group} enrichment in top {perc}% active molecules ({selected_activity}):")
            change_font_type(dpg.last_item(), "bold", state)
            dpg.add_text(default_value=f"{rnum}/{topN}   ({perc_val:.1f}%)")
