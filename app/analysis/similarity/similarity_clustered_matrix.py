"""
similarity_clustered_matrix.py

Clustered activity-difference matrix.

Builds a Tanimoto-based clustering of molecules and displays a matrix where
each cell is coloured by the difference in activity between molecule j and i
(Δ = activity_j - activity_i), using IC50-like values parsed from the summary CSV.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Show clustered similarity manager window
# 3. Update clustered similarity activity choices
# 4. Build clustered similarity matrix

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import io
import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
from typing import Any
from PIL import Image as pilImage
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from app.utils.app_logger import log_event, log_settings
from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress
from app.utils.callbacks import register_responsive_image
from app.gui.themes_manager import (
    apply_colormap_theme, 
    apply_plot_theme
)


# -----------------------------------------------------------------------------
# 2. Show clustered similarity manager window
# -----------------------------------------------------------------------------
def show_clustered_similarity_manager_window(state: dict[str, Any]) -> None:

    def _search_clustered_molecule(user_state: dict[str, Any]) -> None:
        """
        Highlight the searched molecule row and column in the clustered matrix.
        """
        try:
            mol_id = int(dpg.get_value("similarity_cluster_subset_lookup_input"))
        except Exception:
            return

        user_state["similarity_cluster_pending_highlight_mol_id"] = mol_id
        build_clustered_similarity_matrix(user_state)

    if not dpg.does_item_exist("clustered_similarity_manager_window"):
        return

    # Clear previous content (if any) to avoid overlapping widgets.
    dpg.delete_item("clustered_similarity_manager_window", children_only=True)
    with dpg.child_window(parent="clustered_similarity_manager_window",
                          no_scrollbar=False,
                          horizontal_scrollbar=False,
                          no_scroll_with_mouse=True,
                          border=False,
                          width=-1,
                          auto_resize_y=True):
        control_w = state.get("plots_manager_combo_width", 220)
        search_input_w = 150
        control_gap = max(6, state["win_spacer"] * 2)

        subsets = list(state.get("bioact_types_dict", {}).keys())
        if not subsets:
            dpg.add_text("No subsets found in bioact_types_dict.", color=(255, 80, 80, 255))
            return

        # Use first subset as default (or any other policy you prefer).
        default_subset = subsets[0]

        with dpg.group(horizontal=True):
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        label="Subset",
                        width=control_w,
                        height_mode=dpg.mvComboHeight_Large,
                        items=subsets,
                        default_value=default_subset,
                        tag="similarity_cluster_subset_choice",
                        callback=_update_clustered_similarity_activity_choices,
                        user_data=state
                    )
                    dpg.add_spacer(width=control_gap)

                    dpg.add_combo(
                        label="Activity type",
                        width=control_w,
                        height_mode=dpg.mvComboHeight_Small,
                        items=[],
                        default_value="",
                        tag="similarity_cluster_activity_choice"
                    )
                    dpg.add_spacer(width=control_gap)

                    dpg.add_checkbox(
                        label="Read undefined",
                        tag="similarity_read_undefined",
                        default_value=False
                    )
                    dpg.add_spacer(width=control_gap)

                    dpg.add_checkbox(
                        label="Include NO activity",
                        tag="similarity_read_inactives",
                        default_value=False
                    )

                _update_clustered_similarity_activity_choices(None, None, state)

                with dpg.group(horizontal=True):
                    dpg.add_text("Similarity threshold (%)")
                    dpg.add_input_int(
                        tag="similarity_cluster_threshold",
                        width=140,
                        min_value=0,
                        max_value=100,
                        step=1,
                        min_clamped=True,
                        max_clamped=True,
                        default_value=85
                    )
                    dpg.add_spacer(width=control_gap)

                    dpg.add_checkbox(
                        label="Δactivity as absolute value",
                        tag="similarity_delta_activity_as_abs",
                        default_value=True
                    )
                    dpg.add_spacer(width=control_gap)

                    dpg.add_text("Search molecule:")
                    dpg.add_input_int(
                        tag="similarity_cluster_subset_lookup_input",
                        width=search_input_w,
                        step=1,
                        default_value=1,
                    )
                    dpg.add_button(
                        label="Search",
                        callback=lambda s, a, u: _search_clustered_molecule(u),
                        user_data=state,
                    )
                    dpg.add_spacer(width=control_gap)

            dpg.add_button(
                label="Build clustered similarity matrix",
                tag="similarity_build_cluster_matrix_button",
                callback=lambda s, a, u: build_clustered_similarity_matrix(u),
                user_data=state
            )


# -----------------------------------------------------------------------------
# 3. Update clustered similarity activity choices
# -----------------------------------------------------------------------------
def _update_clustered_similarity_activity_choices(
    sender: Any,
    app_data: Any,
    state: dict[str, Any]
) -> None:
    subset = dpg.get_value("similarity_cluster_subset_choice") \
        if dpg.does_item_exist("similarity_cluster_subset_choice") else None
    bioact_dict = state.get("bioact_types_dict", {})

    if subset not in bioact_dict:
        dpg.configure_item("similarity_cluster_activity_choice", items=[], default_value="")
        return

    activities = bioact_dict[subset].get("bioactivities", [])
    activities = list(dict.fromkeys(activities))  # keep order, remove duplicates

    if not activities:
        dpg.configure_item("similarity_cluster_activity_choice", items=[], default_value="")
        return

    dpg.configure_item("similarity_cluster_activity_choice",
                       items=activities,
                       default_value=activities[0])


# -----------------------------------------------------------------------------
# 4. Build clustered similarity matrix
# -----------------------------------------------------------------------------
def build_clustered_similarity_matrix(state: dict[str, Any]) -> Any:
    log_event("Similarity", "Drawing clustered similarity matrix", indent=1)
    if not dpg.does_item_exist("clustered_similarity_matrix_window"):
        return

    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)

    # Clear previous plot and couple window content.
    for tag in ["clustered_similarity_mol_couple_window", "clustered_similarity_matrix_window"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=True)
    set_loading_screen_progress(state, 4)

    summary_dir = state.get("summary_dir", "")
    subset = dpg.get_value("similarity_cluster_subset_choice")
    activity_type = dpg.get_value("similarity_cluster_activity_choice")

    read_undef = dpg.get_value("similarity_read_undefined")
    read_inact = dpg.get_value("similarity_read_inactives")

    thr_percent = dpg.get_value("similarity_cluster_threshold")
    thr_percent = max(0, min(100, thr_percent))
    thr = float(thr_percent) / 100.0

    delta_activity_as_abs = dpg.get_value("similarity_delta_activity_as_abs")
    log_settings("Similarity", indent=2, subset=subset, activity=activity_type, include_undefined=read_undef, include_inactives=read_inact, cluster_threshold_percent=thr_percent, absolute_delta=delta_activity_as_abs)
    
    text_color = state["theme"]["Text Color"]

    if not subset or not activity_type:
        with dpg.child_window(parent="clustered_similarity_matrix_window",
                              no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False, 
                              width=-1, height=-1):
            dpg.add_text("No activities in the selected subset", color=(255, 80, 80, 255))
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        return

    csv_file = os.path.join(summary_dir, f"{subset}_summary.csv")
    if not os.path.exists(csv_file):
        with dpg.child_window(parent="clustered_similarity_matrix_window",
                              no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False, 
                              width=-1, height=-1):
            dpg.add_text(f"Summary CSV not found: {csv_file}", color=(255, 80, 80, 255))
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        return

    data = pd.read_csv(csv_file)
    data = data[data["Mol"].notna()]
    set_loading_screen_progress(state, 8)

    if "MolName" in data.columns:
        data["MolName"] = data["MolName"].fillna("")

    if "Mol_sub_ID" not in data.columns:
        # Fallback: create a MolID-like index if missing.
        data["Mol_sub_ID"] = np.arange(1, len(data) + 1)

    if activity_type not in data.columns:
        with dpg.child_window(parent="clustered_similarity_matrix_window",
                              no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False, 
                              width=-1, height=-1):
            dpg.add_text(f"Activity column '{activity_type}' not found in {subset}_summary.csv",
                         color=(255, 80, 80, 255))
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        return


    # -----------------------------------------------------------------------------
    # 4.1. Parse activity string
    # -----------------------------------------------------------------------------
    def parse_activity_string(act_str: Any, read_inequalities: bool = False) -> Any:
        if not isinstance(act_str, str):
            return np.nan

        s = act_str.strip()
        if not s or s.upper() == "N/A":
            return np.nan

        # Identify operator if present
        op = None
        if s.startswith((">=", "<=")):
            op = s[:2]
            num_str = s[2:]
        elif s.startswith((">", "<", "=")):
            op = s[0]
            num_str = s[1:]
        else:
            num_str = s  # pure number

        num_str = num_str.strip()
        if not num_str.isdigit() and not _is_float(num_str):
            return np.nan

        val = float(num_str)

        # If inequalities should NOT be read as exact, skip them
        if op and not read_inequalities:
            return np.nan

        return val


    # -----------------------------------------------------------------------------
    # 4.2. Is float
    # -----------------------------------------------------------------------------
    def _is_float(x: Any) -> Any:
        try:
            float(x)
            return True
        except:
            return False
        
        
    mol_ids = []
    smiles_list = []
    activities = []

    total_rows = max(1, len(data))
    for idx, (_, row) in enumerate(data.iterrows(), start=1):
        smi = row["Mol"]
        if not isinstance(smi, str) or not smi.strip():
            continue

        act_str = row.get(activity_type, "")
        val = parse_activity_string(str(act_str), read_inequalities=read_undef)

        if np.isnan(val):
            if read_inact:
                val = 0.0
            else:
                # Skip molecules with no interpretable value if we don't treat inactives as 0.
                continue

        mol_ids.append(int(row["Mol_sub_ID"]))
        smiles_list.append(smi.strip())
        activities.append(val)
        if idx % max(1, total_rows // 20) == 0 or idx == total_rows:
            set_loading_screen_progress(state, 8 + ((idx / total_rows) * 17))

    if not smiles_list:
        with dpg.child_window(parent="clustered_similarity_matrix_window",
                              no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False, 
                              width=-1, height=-1):
            dpg.add_text("No molecules with valid activity values found.", color=(255, 80, 80, 255))
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        return

    rdkit_mols = []
    fps = []
    kept_ids = []
    kept_acts = []

    total_smiles = max(1, len(smiles_list))
    for idx, (smi, mid, act) in enumerate(zip(smiles_list, mol_ids, activities), start=1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True).GetFingerprint(mol)
        rdkit_mols.append(mol)
        fps.append(fp)
        kept_ids.append(mid)
        kept_acts.append(act)
        if idx % max(1, total_smiles // 20) == 0 or idx == total_smiles:
            set_loading_screen_progress(state, 25 + ((idx / total_smiles) * 15))


    convert_to_log = activity_type in state.get("nM_activity_types", [])

    if convert_to_log:
        new_vals = []
        for v in kept_acts:

            # Inactive: keep at pValue = 0
            if v == 0:
                new_vals.append(0.0)
                continue

            # Normal case: convert nM -> M and compute pIC50
            molar = float(v) * 1e-9
            if molar > 0:
                pval = -np.log10(molar)
            else:
                # extremely unlikely, but fallback
                pval = 0.0
            new_vals.append(pval)

        kept_acts = new_vals


    n = len(fps)
    if n == 0:
        with dpg.child_window(parent="clustered_similarity_matrix_window",
                              no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False, 
                              width=-1, height=-1):
            dpg.add_text("No valid RDKit molecules after parsing SMILES.", color=(255, 80, 80, 255))
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        return

    sim_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            t = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim_matrix[i, j] = t
            sim_matrix[j, i] = t
        if i % max(1, n // 25) == 0 or i == n - 1:
            set_loading_screen_progress(state, 40 + (((i + 1) / max(1, n)) * 25))

    # Build adjacency list for connected components at similarity >= thr.
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= thr:
                adj_list[i].append(j)
                adj_list[j].append(i)

    visited = [False] * n
    clusters = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        comp = []
        visited[start] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for neigh in adj_list[node]:
                if not visited[neigh]:
                    visited[neigh] = True
                    stack.append(neigh)
        clusters.append(comp)
        if start % max(1, n // 20) == 0 or start == n - 1:
            set_loading_screen_progress(state, 65 + (((start + 1) / max(1, n)) * 10))

    # Sort clusters by size (descending).
    clusters.sort(key=len, reverse=True)

    cluster_num_by_node = {}
    for cluster_idx, comp in enumerate(clusters, start=1):
        for node_idx in comp:
            cluster_num_by_node[node_idx] = cluster_idx

    # Inside each cluster, keep molecules ordered by MolID (stable index sort).
    for idx, comp in enumerate(clusters):
        clusters[idx] = sorted(comp, key=lambda k: kept_ids[k])

    order = [idx for comp in clusters for idx in comp]

    ordered_ids = [kept_ids[i] for i in order]
    ordered_acts = np.array([kept_acts[i] for i in order], dtype=float)
    ordered_cluster_nums = [cluster_num_by_node[i] for i in order]
    n_ord = len(ordered_ids)
    highlighted_mol_id = state.pop("similarity_cluster_pending_highlight_mol_id", None)
    set_loading_screen_progress(state, 78)

    def _get_highlight_color() -> tuple[int, int, int, int]:
        """
        Return the first color of the active discrete colormap.
        """
        try:
            colormap = state["plot_colormaps"][state["colormap_discrete"]]
            color = dpg.sample_colormap(colormap, 0.0)
            if len(color) >= 4:
                if max(color[0], color[1], color[2]) <= 1.0:
                    return (
                        int(round(color[0] * 255)),
                        int(round(color[1] * 255)),
                        int(round(color[2] * 255)),
                        255,
                    )
                return (int(color[0]), int(color[1]), int(color[2]), 255)
        except Exception:
            pass
        return tuple(state["theme"]["Title Bar Background"])

    def _refresh_cluster_highlight_theme() -> None:
        """
        Reapply the highlight theme to the existing clustered overlay series.
        """
        if not (
            dpg.does_item_exist("cluster_highlight_row_series")
            or dpg.does_item_exist("cluster_highlight_col_series")
        ):
            return

        highlight_color = _get_highlight_color()
        theme_tag = "cluster_highlight_theme"

        try:
            dpg.delete_item(theme_tag)
        except Exception:
            pass
        try:
            if hasattr(dpg, "does_alias_exist") and dpg.does_alias_exist(theme_tag):
                dpg.remove_alias(theme_tag)
        except Exception:
            pass

        with dpg.theme(tag=theme_tag):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    highlight_color,
                    category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight,
                    3.0,
                    category=dpg.mvThemeCat_Plots
                )

        for item_tag in ("cluster_highlight_row_series", "cluster_highlight_col_series"):
            if dpg.does_item_exist(item_tag):
                dpg.bind_item_theme(item_tag, theme_tag)

    state["similarity_cluster_refresh_highlight"] = _refresh_cluster_highlight_theme

    # activity_j - activity_i -> differences row/col.
    row_vals = ordered_acts.reshape(-1, 1)
    col_vals = ordered_acts.reshape(1, -1)
    if delta_activity_as_abs:
        delta_matrix = np.abs(col_vals - row_vals)  # shape (n, n)
    else:    
        delta_matrix = (col_vals - row_vals)  # shape (n, n)
    max_abs = float(np.max(np.abs(delta_matrix)))
    if max_abs == 0:
        max_abs = 1.0  # evita range nullo

    scale_min = -max_abs if not delta_activity_as_abs else 0.0
    scale_max = max_abs
    delta_axis_label = f"Δ{'p' if convert_to_log else ''}{activity_type}"
    if delta_activity_as_abs:
        delta_axis_label = f"|{delta_axis_label}|"

    def _nice_tick_step(visible_count: int, max_visible_ticks: int) -> int:
        """
        Pick a readable 1-2-5 style tick step.

        Args:
            visible_count (int): Number of matrix indices currently visible.
            max_visible_ticks (int): Approximate number of ticks that fit.

        Returns:
            int: Tick step to use.
        """
        if visible_count <= 1 or max_visible_ticks <= 1:
            return 1

        raw_step = max(1, int(np.ceil(visible_count / max_visible_ticks)))
        magnitude = 1
        while magnitude * 10 < raw_step:
            magnitude *= 10

        for factor in (1, 2, 5, 10):
            candidate = magnitude * factor
            if raw_step <= candidate:
                return candidate
        return magnitude * 10

    def _update_clustered_matrix_ticks() -> None:
        """
        Update clustered matrix ticks based on the currently visible plot region.

        Args:
            None.

        Returns:
            None: This routine updates plot ticks in place.
        """
        if not (
            dpg.does_item_exist("clustered_matrix")
            and dpg.does_item_exist("cluster_delta_x_axis")
            and dpg.does_item_exist("cluster_delta_y_axis")
        ):
            return

        try:
            x_min, x_max = dpg.get_axis_limits("cluster_delta_x_axis")
            y_min, y_max = dpg.get_axis_limits("cluster_delta_y_axis")
        except Exception:
            return

        try:
            plot_w, plot_h = dpg.get_item_rect_size("clustered_matrix")
        except Exception:
            plot_w = plot_h = 0

        plot_w = max(1, int(plot_w or 0))
        plot_h = max(1, int(plot_h or 0))

        vis_x0 = max(0.5, min(n_ord + 0.5, float(x_min)))
        vis_x1 = max(0.5, min(n_ord + 0.5, float(x_max)))
        vis_y0 = max(0.5, min(n_ord + 0.5, float(y_min)))
        vis_y1 = max(0.5, min(n_ord + 0.5, float(y_max)))

        if vis_x1 < vis_x0:
            vis_x0, vis_x1 = vis_x1, vis_x0
        if vis_y1 < vis_y0:
            vis_y0, vis_y1 = vis_y1, vis_y0

        first_x_idx = max(0, int(np.ceil(vis_x0 - 1.0)))
        last_x_idx = min(n_ord - 1, int(np.floor(vis_x1 - 1.0)))
        first_y_idx = max(0, int(np.ceil(vis_y0 - 1.0)))
        last_y_idx = min(n_ord - 1, int(np.floor(vis_y1 - 1.0)))

        visible_x = max(0, last_x_idx - first_x_idx + 1)
        visible_y = max(0, last_y_idx - first_y_idx + 1)

        max_ticks_x = max(1, plot_w // 42)
        max_ticks_y = max(1, plot_h // 24)
        step_x = _nice_tick_step(visible_x, max_ticks_x)
        step_y = _nice_tick_step(visible_y, max_ticks_y)

        x_tick_indices = list(range(first_x_idx, last_x_idx + 1, step_x)) if visible_x else []
        y_tick_indices = list(range(first_y_idx, last_y_idx + 1, step_y)) if visible_y else []

        x_ticks = tuple((str(ordered_ids[i]), i + 1) for i in x_tick_indices)
        y_ticks = tuple((str(ordered_ids[i]), i + 1) for i in y_tick_indices)

        dpg.set_axis_ticks("cluster_delta_x_axis", x_ticks)
        dpg.set_axis_ticks("cluster_delta_y_axis", y_ticks)

    with dpg.child_window(label="Clustered Activity-Difference Matrix",
                          parent="clustered_similarity_matrix_window",
                          no_scrollbar=False,
                          horizontal_scrollbar=False,
                          no_scroll_with_mouse=True,
                          border=False,
                          width=-1,
                          height=-1):

        with dpg.group():

            with dpg.group(horizontal=True):

                dpg.add_colormap_scale(
                    tag="clustered_matrix_colormap_scale",
                    label=delta_axis_label,
                    colormap=state["colormaps"][state["colormap_continuous"]],
                    mirror=True,
                    min_scale=scale_min if not delta_activity_as_abs else 0.0,
                    max_scale=scale_max,
                    height=-1
                )

                with dpg.plot(
                    width=-1,
                    height=-1,
                    tag="clustered_matrix",
                    no_mouse_pos=True,
                    no_menus=True,
                    no_frame=True,
                    no_title=True,
                    equal_aspects=True,
                    zoom_rate=0.05
                ):
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        tag="cluster_delta_x_axis",
                        no_label=True,
                        no_tick_labels=False,
                        no_tick_marks=True,
                        no_gridlines=True
                    )
                    with dpg.plot_axis(
                        dpg.mvYAxis,
                        tag="cluster_delta_y_axis",
                        no_label=False,
                        no_tick_labels=False,
                        no_tick_marks=True,
                        no_gridlines=True
                    ):
                        # Flatten matrix row-major, like in the Tanimoto example.
                        dpg.add_heat_series(
                            delta_matrix[::-1].flatten().tolist(),
                            rows=n_ord,
                            cols=n_ord,
                            format="",
                            tag="cluster_delta_heat_series",
                            bounds_min=(0.5, 0.5),
                            bounds_max=(n_ord + 0.5, n_ord + 0.5),
                            scale_min=scale_min if not delta_activity_as_abs else 0.0,
                            scale_max=scale_max
                        )

                        _update_clustered_matrix_ticks()

                    dpg.add_line_series([0.5, n_ord + 0.5, n_ord + 0.5, 0.5, 0.5], [0.5, 0.5, n_ord + 0.5, n_ord + 0.5, 0.5], tag="cluster_delta_border_series", parent="cluster_delta_y_axis")

                    with dpg.theme() as border_theme:
                        with dpg.theme_component(dpg.mvLineSeries):
                            dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 255), category=dpg.mvThemeCat_Plots)
                    dpg.bind_item_theme("cluster_delta_border_series", border_theme)


                    diag_color = (60, 60, 60, 255)

                    for i in range(n_ord):
                        x0 = i + 0.5
                        x1 = i + 1.5
                        y0 = i + 0.5
                        y1 = i + 1.5

                        dpg.draw_rectangle(
                            pmin=(x0, y0),
                            pmax=(x1, y1),
                            color=diag_color,
                            fill=diag_color,
                            thickness=0,
                            parent="clustered_matrix",
                            tag=f"cluster_diag_rect_{i}"
                        )

                        

                    start_idx = 0

                    for ci, comp in enumerate(clusters):
                        size = len(comp)
                        end_idx = start_idx + size

                        x0 = start_idx + 0.5
                        x1 = end_idx + 0.5
                        y0 = start_idx + 0.5
                        y1 = end_idx + 0.5

                        # Coordinates for square border (clockwise)
                        xs = [x0, x1, x1, x0, x0]
                        ys = [y0, y0, y1, y1, y0]

                        border_tag = f"cluster_box_{ci}"

                        dpg.add_line_series(
                            xs, ys,
                            tag=border_tag,
                            parent="cluster_delta_y_axis"
                        )

                        # Theme: thick black border
                        with dpg.theme() as box_theme:
                            with dpg.theme_component(dpg.mvLineSeries):
                                dpg.add_theme_color(dpg.mvPlotCol_Line, (0, 0, 0, 255), category=dpg.mvThemeCat_Plots)
                                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0, category=dpg.mvThemeCat_Plots)

                        dpg.bind_item_theme(border_tag, box_theme)

                        # Update start of next cluster
                        start_idx = end_idx

                    if isinstance(highlighted_mol_id, int) and highlighted_mol_id in ordered_ids:
                        highlight_color = _get_highlight_color()
                        highlight_index = ordered_ids.index(highlighted_mol_id)
                        x0 = highlight_index + 0.5
                        x1 = highlight_index + 1.5
                        y0 = highlight_index + 0.5
                        y1 = highlight_index + 1.5

                        dpg.add_line_series(
                            [0.5, n_ord + 0.5, n_ord + 0.5, 0.5, 0.5],
                            [y0, y0, y1, y1, y0],
                            tag="cluster_highlight_row_series",
                            parent="cluster_delta_y_axis"
                        )
                        dpg.add_line_series(
                            [x0, x1, x1, x0, x0],
                            [0.5, 0.5, n_ord + 0.5, n_ord + 0.5, 0.5],
                            tag="cluster_highlight_col_series",
                            parent="cluster_delta_y_axis"
                        )

                        _refresh_cluster_highlight_theme()


                dpg.fit_axis_data("cluster_delta_x_axis")
                dpg.fit_axis_data("cluster_delta_y_axis")
                dpg.set_frame_callback(dpg.get_frame_count() + 1, _update_clustered_matrix_ticks)
                dpg.set_frame_callback(dpg.get_frame_count() + 2, _update_clustered_matrix_ticks)

                dpg.bind_colormap("clustered_matrix", state["colormaps"][state["colormap_continuous"]])
                dpg.bind_item_theme("clustered_matrix", apply_plot_theme(state))
                dpg.bind_item_theme("clustered_matrix_colormap_scale", apply_colormap_theme(state))
    set_loading_screen_progress(state, 93)
                


    # Prima rimuovo eventuali handler/texture vecchi
    for tag in [
        "clustered_matrix_click_handler",
        "clustered_mol_x_image_widget",
        "clustered_mol_y_image_widget"
    ]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    render_scale = 1.8
    render_w = int(round(state["similarity_clustered_img_width"] * render_scale))
    render_h = int(round(state["similarity_clustered_img_height"] * render_scale))

    with dpg.child_window(tag="clustered_similarity_mol_couple_inner_window",
                          parent="clustered_similarity_mol_couple_window",
                          no_scrollbar=True,
                          horizontal_scrollbar=False,
                          no_scroll_with_mouse=True,
                          border=False,
                          width=-1,
                          auto_resize_y=True):
        
        dpg.add_text(f"Clusters: {len(clusters)}, molecules: {n_ord}")

        with dpg.group():

            with dpg.group():

                empty = np.ones((render_w * render_h * 4,),
                                 dtype=np.float32)

                if not dpg.does_item_exist("clustered_mol_x_image_texture"):
                    dpg.add_dynamic_texture(render_w,
                                            render_h,
                                            empty,
                                            tag="clustered_mol_x_image_texture",
                                            parent="texture_registry")
                else:
                    dpg.set_value("clustered_mol_x_image_texture", empty)

                dpg.add_image("clustered_mol_x_image_texture",
                              width=state["similarity_clustered_img_width"],
                              height=state["similarity_clustered_img_height"],
                              tag="clustered_mol_x_image_widget",
                              border_color=(0,0,0,0))
                with dpg.tooltip("clustered_mol_x_image_widget", delay=0):
                    dpg.add_text("", tag="clustered_mol_x_tooltip_text")
                register_responsive_image(
                    state,
                    image_tag="clustered_mol_x_image_widget",
                    parent_tag="clustered_similarity_mol_couple_inner_window",
                    aspect_ratio=0.75,
                    tab="clustered_matrix_subtab",
                )

            with dpg.group():

                if not dpg.does_item_exist("clustered_mol_y_image_texture"):
                    dpg.add_dynamic_texture(render_w,
                                            render_h,
                                            empty,
                                            tag="clustered_mol_y_image_texture",
                                            parent="texture_registry")
                else:
                    dpg.set_value("clustered_mol_y_image_texture", empty)

                dpg.add_image("clustered_mol_y_image_texture",
                              width=state["similarity_clustered_img_width"],
                              height=state["similarity_clustered_img_height"],
                              tag="clustered_mol_y_image_widget",
                              border_color=(0,0,0,0))
                with dpg.tooltip("clustered_mol_y_image_widget", delay=0):
                    dpg.add_text("", tag="clustered_mol_y_tooltip_text")
                register_responsive_image(
                    state,
                    image_tag="clustered_mol_y_image_widget",
                    parent_tag="clustered_similarity_mol_couple_inner_window",
                    aspect_ratio=0.75,
                    tab="clustered_matrix_subtab",
                )
    set_loading_screen_progress(state, 98)


    # -----------------------------------------------------------------------------
    # 4.3. Update clustered images
    # -----------------------------------------------------------------------------
    def _update_clustered_images(row_idx: int, col_idx: int) -> None:
        smi_x = smiles_list[order[col_idx]]
        smi_y = smiles_list[order[row_idx]]
        mol_x = Chem.MolFromSmiles(smi_x)
        mol_y = Chem.MolFromSmiles(smi_y)
        if mol_x is None or mol_y is None:
            return

        drawer_x = Draw.MolDraw2DCairo(render_w, render_h)
        opts = drawer_x.drawOptions()
        opts.padding = 0.025
        opts.bondLineWidth = 1
        opts.minFontSize = 1
        opts.legendFontSize = 14
        drawer_x.DrawMolecule(mol_x)
        drawer_x.FinishDrawing()
        img_x = pilImage.open(io.BytesIO(drawer_x.GetDrawingText())).convert("RGBA")
        arr_x = (np.array(img_x) / 255.0).astype(np.float32).flatten()

        drawer_y = Draw.MolDraw2DCairo(render_w, render_h)
        opts2 = drawer_y.drawOptions()
        opts2.padding = 0.025
        opts2.bondLineWidth = 1
        opts2.minFontSize = 1
        opts2.legendFontSize = 14
        drawer_y.DrawMolecule(mol_y)
        drawer_y.FinishDrawing()
        img_y = pilImage.open(io.BytesIO(drawer_y.GetDrawingText())).convert("RGBA")
        arr_y = (np.array(img_y) / 255.0).astype(np.float32).flatten()

        id_x = ordered_ids[col_idx]
        id_y = ordered_ids[row_idx]
        cluster_x = ordered_cluster_nums[col_idx]
        cluster_y = ordered_cluster_nums[row_idx]

        dpg.set_value("clustered_mol_x_image_texture", arr_x)
        dpg.set_value("clustered_mol_y_image_texture", arr_y)

        act_x_linear = activities[order[col_idx]]
        act_y_linear = activities[order[row_idx]]
        dpg.set_value(
            "clustered_mol_x_tooltip_text",
            f"X: Mol {id_x}  |  Cluster {cluster_x}  |  {activity_type} = {act_x_linear:.1f} nM",
        )
        dpg.set_value(
            "clustered_mol_y_tooltip_text",
            f"Y: Mol {id_y}  |  Cluster {cluster_y}  |  {activity_type} = {act_y_linear:.1f} nM",
        )


    # -----------------------------------------------------------------------------
    # 4.4. On clustered heatmap click
    # -----------------------------------------------------------------------------
    def on_clustered_heatmap_click(sender: Any, app_data: Any, user_data: Any) -> None:
        if not dpg.is_item_hovered("clustered_matrix"):
            return
        pos = dpg.get_plot_mouse_pos()
        if not pos:
            return
        x_pos, y_pos = pos

        if x_pos < 0.5 or x_pos > n_ord + 0.5:
            return
        if y_pos < 0.5 or y_pos > n_ord + 0.5:
            return

        col_idx = int(x_pos - 0.5)
        row_idx = int(y_pos - 0.5)
        if 0 <= col_idx < n_ord and 0 <= row_idx < n_ord:
            _update_clustered_images(row_idx, col_idx)


    if dpg.does_item_exist("clustered_matrix_click_handler"):
        dpg.delete_item("clustered_matrix_click_handler")

    dpg.add_mouse_click_handler(
        tag="clustered_matrix_click_handler",
        parent="handler_registry",
        button=dpg.mvMouseButton_Left,
        callback=on_clustered_heatmap_click
    )

    for tag in [
        "clustered_matrix_wheel_handler",
        "clustered_matrix_drag_handler",
        "clustered_matrix_release_handler",
    ]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    dpg.add_mouse_wheel_handler(
        tag="clustered_matrix_wheel_handler",
        parent="handler_registry",
        callback=lambda s, a, u: _update_clustered_matrix_ticks() if dpg.is_item_hovered("clustered_matrix") else None,
    )
    dpg.add_mouse_drag_handler(
        tag="clustered_matrix_drag_handler",
        parent="handler_registry",
        callback=lambda s, a, u: _update_clustered_matrix_ticks() if dpg.is_item_hovered("clustered_matrix") else None,
    )
    dpg.add_mouse_release_handler(
        tag="clustered_matrix_release_handler",
        parent="handler_registry",
        callback=lambda s, a, u: _update_clustered_matrix_ticks() if dpg.does_item_exist("clustered_matrix") else None,
    )


    if n_ord > 0:
        _update_clustered_images(0, 0)

    set_loading_screen_progress(state, 100)

    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")
