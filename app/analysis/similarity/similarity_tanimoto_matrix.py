"""
====================
similarity_tanimoto_matrix.py
====================

Dataset similarity window.

Displays a Tanimoto similarity matrix with fingerprint maps.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Show tanimoto similarity manager window
# 3. Show tanimoto similarity matrix

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
from rdkit.Chem.Draw import SimilarityMaps
from app.utils.app_logger import log_event, log_settings
from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress
from app.utils.callbacks import register_responsive_image
from app.gui.themes_manager import (
    apply_colormap_theme,
    apply_plot_theme
)


# -----------------------------------------------------------------------------
# 2. Show tanimoto similarity manager window
# -----------------------------------------------------------------------------
def show_tanimoto_similarity_manager_window(state: dict[str, Any]) -> None:
    """
    Display the similarity panel with subset and activity selection, pagination,.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    def _search_tanimoto_molecule(user_state: dict[str, Any]) -> None:
        """
        Highlight the searched molecule row and column in the Tanimoto matrix.
        """
        try:
            mol_id = int(dpg.get_value("similarity_tanimoto_search_input"))
        except Exception:
            return

        user_state["similarity_tanimoto_pending_highlight_mol_id"] = mol_id
        show_tanimoto_similarity_matrix(user_state)

    def _toggle_similarity_fp_map(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Redraw the currently selected pair when the map mode changes.
        """
        refresh_cb = user_data.get("refresh_similarity_tanimoto_images")
        if callable(refresh_cb):
            try:
                refresh_cb()
            except Exception:
                pass

    # Build the manager window and wire all controls.
    with dpg.child_window(label="Similarity manager", parent="similarity_manager_window", auto_resize_y=True,
                        no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
        control_w = state.get("plots_manager_combo_width", 220)
        search_input_w = 150
        control_gap = max(6, state["win_spacer"] * 2)

        subsets = list(state["smiles_rgd_dict"].keys())
        with dpg.group(horizontal=True):
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        label="Subset",
                        width=control_w,
                        height_mode=dpg.mvComboHeight_Large,
                        items=subsets,
                        default_value=subsets[0],
                        tag="similarity_subset_choice",
                    )
                    dpg.add_spacer(width=control_gap)
                    dpg.add_checkbox(
                        label="Sort by overall similarity",
                        tag="similarity_cluster_order",
                        default_value=False
                    )
                    dpg.add_spacer(width=control_gap)
                    dpg.add_checkbox(
                        label="Normalize colors to full range (0→1)",
                        tag="similarity_color_01",
                        default_value=False
                    )

                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Draw similarity map",
                        tag="similarity_tanimoto_show_fp_map",
                        default_value=True,
                        callback=_toggle_similarity_fp_map,
                        user_data=state,
                    )
                    dpg.add_spacer(width=control_gap)
                    dpg.add_text("Search molecule:")
                    dpg.add_input_int(
                        tag="similarity_tanimoto_search_input",
                        width=search_input_w,
                        step=1,
                        default_value=1,
                    )
                    dpg.add_button(
                        label="Search",
                        callback=lambda s, a, u: _search_tanimoto_molecule(u),
                        user_data=state,
                    )
                    dpg.add_spacer(width=control_gap)

            dpg.add_button(
                label="Build Tanimoto similarity matrix",
                tag="similarity_build_tanimoto_matrix_button",
                callback=lambda s, a, u: show_tanimoto_similarity_matrix(u),
                user_data=state
            )

# -----------------------------------------------------------------------------
# 3. Show tanimoto similarity matrix
# -----------------------------------------------------------------------------
def show_tanimoto_similarity_matrix(state: dict[str, Any]) -> None:
    """
    Build and display the similarity matrix based on selected subset and activity.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    log_event("Similarity", "Drawing Tanimoto similarity matrix", indent=1)
    render_scale = 1.8
    render_w = int(round(state["similarity_tan_img_width"] * render_scale))
    render_h = int(round(state["similarity_tan_img_height"] * render_scale))

    # Initialise view state, remove stale widgets, read summary CSV, and aggregate activities.
    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)

    for tag in ["tan_matrix_mouse_handler", "mol_y_image_widget", "mol_x_image_widget"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    for tag in ["similarity_tanimoto_window", "similarity_tanimoto_mol_couple_window"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=True)
    set_loading_screen_progress(state, 4)

    summary_dir = state["summary_dir"]
    subset = dpg.get_value("similarity_subset_choice")
    activity = dpg.get_value("similarity_activity_type")
    log_settings("Similarity", indent=2, subset=subset, activity=activity, sort_by_overall_similarity=dpg.get_value("similarity_cluster_order"), normalize_0_1=dpg.get_value("similarity_color_01"))


    # --- Read and aggregate per-molecule data from CSV ---
    csv_file = os.path.join(summary_dir, f"{subset}_summary.csv")
    data = pd.read_csv(csv_file)
    set_loading_screen_progress(state, 8)

    data = data[data["Mol"].notna()]
    if "MolName" in data.columns:
        data["MolName"] = data["MolName"].fillna("")

    fixed_columns = ["MolID", "MolName", "Substructure", "Mol", "logP", "MW", "HBA", "HBD"]
    rgroup_columns = [col for col in data.columns if col.startswith("R")]
    activity_columns = [col for col in data.columns if col not in fixed_columns + rgroup_columns]

    aggregated_rows = []
    grouped_rows = list(data.groupby("MolID", sort=False))
    total_groups = max(1, len(grouped_rows))
    for idx, (mol_id, group) in enumerate(grouped_rows, start=1):
        row = {}
        first_row = group.iloc[0]

        for col in fixed_columns + rgroup_columns:
            if col in group.columns:
                row[col] = first_row[col]

        for act_col in activity_columns:
            values = group[act_col].dropna().astype(str).str.strip()
            values = [v for v in values if v and v.upper() != "N/A"]

            formatted_values = []
            for v in values:
                val = v.strip()

                if any(val.startswith(op) for op in ("<=", ">=", "<", ">")):
                    formatted_values.append(f"{act_col} {val}")
                else:
                    formatted_values.append(f"{act_col} = {val}")

            formatted_values = list(dict.fromkeys(formatted_values))
            row[act_col] = " | ".join(formatted_values) if formatted_values else ""

        aggregated_rows.append(row)
        if idx % max(1, total_groups // 20) == 0 or idx == total_groups:
            set_loading_screen_progress(state, 8 + ((idx / total_groups) * 18))

    df = pd.DataFrame(aggregated_rows)

    if activity != "Any" and activity in df.columns:
        df = df[df[activity].astype(str).str.strip() != ""]

    all_smiles = df["Mol"].tolist()
    all_indices = df.index.tolist()
    displayed_order_ids = [i + 1 for i in range(len(df))]
    set_loading_screen_progress(state, 28)


    # Compute Morgan fingerprints and the full NxN Tanimoto matrix; prepare the heatmap widgets.
    morgan_fp_gen = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fps = []
    total_smiles = max(1, len(all_smiles))
    for idx, smi in enumerate(all_smiles, start=1):
        fps.append(morgan_fp_gen.GetFingerprint(Chem.MolFromSmiles(smi)))
        if idx % max(1, total_smiles // 20) == 0 or idx == total_smiles:
            set_loading_screen_progress(state, 28 + ((idx / total_smiles) * 17))

    n = len(fps)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        if i % max(1, n // 25) == 0 or i == n - 1:
            set_loading_screen_progress(state, 45 + (((i + 1) / max(1, n)) * 35))


    # --- Optional: reorder by "cluster" (sum of similarities) ---
    cluster_order = dpg.get_value("similarity_cluster_order") if dpg.does_item_exist("similarity_cluster_order") else False
    if cluster_order and n > 1:
        # Sum per row (similarity vs ALL others)
        row_sums = similarity_matrix.sum(axis=1)  # shape (n,)
        order = np.argsort(-row_sums, kind="stable")  # descending (more "similar" → more to the left/bottom)

        # Apply order to matrix, smiles, and original indices
        similarity_matrix = similarity_matrix[order][:, order]
        all_smiles = [all_smiles[i] for i in order]
        all_indices = [all_indices[i] for i in order]    
        displayed_order_ids = [displayed_order_ids[i] for i in order]

    state["similarity_tanimoto_displayed_mol_ids"] = displayed_order_ids
    highlighted_mol_id = state.pop("similarity_tanimoto_pending_highlight_mol_id", None)

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

    def _refresh_tanimoto_highlight_theme() -> None:
        """
        Reapply the highlight theme to the existing Tanimoto overlay series.
        """
        if not (
            dpg.does_item_exist("tan_highlight_row_series")
            or dpg.does_item_exist("tan_highlight_col_series")
        ):
            return

        highlight_color = _get_highlight_color()
        theme_tag = "tan_highlight_theme"

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

        for item_tag in ("tan_highlight_row_series", "tan_highlight_col_series"):
            if dpg.does_item_exist(item_tag):
                dpg.bind_item_theme(item_tag, theme_tag)

    state["similarity_tanimoto_refresh_highlight"] = _refresh_tanimoto_highlight_theme
    set_loading_screen_progress(state, 82)


    use_fixed_01 = dpg.get_value("similarity_color_01") if dpg.does_item_exist("similarity_color_01") else False

    if use_fixed_01:
        # --- Fixed scale: 0..1 (old behavior) ---
        vmin, vmax = 0.0, 1.0
        heat_vals = similarity_matrix  # already 0..1
    else:
        # --- Dynamic scale: min..max (new default) ---
        vmin = float(similarity_matrix.min())
        vmax = float(similarity_matrix.max())  # usually 1.0
        den = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
        heat_vals = np.clip((similarity_matrix - vmin) / den, 0.0, 1.0)

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

    def _update_similarity_matrix_ticks() -> None:
        """
        Update custom axis ticks based on the currently visible matrix region.

        Args:
            None.

        Returns:
            None: This routine updates plot ticks in place.
        """
        if not (dpg.does_item_exist("tanimoto_matrix") and dpg.does_item_exist("tan_x_axis") and dpg.does_item_exist("tan_y_axis")):
            return

        try:
            x_min, x_max = dpg.get_axis_limits("tan_x_axis")
            y_min, y_max = dpg.get_axis_limits("tan_y_axis")
        except Exception:
            return

        try:
            plot_w, plot_h = dpg.get_item_rect_size("tanimoto_matrix")
        except Exception:
            plot_w = plot_h = 0

        plot_w = max(1, int(plot_w or 0))
        plot_h = max(1, int(plot_h or 0))

        vis_x0 = max(0.5, min(n + 0.5, float(x_min)))
        vis_x1 = max(0.5, min(n + 0.5, float(x_max)))
        vis_y0 = max(0.5, min(n + 0.5, float(y_min)))
        vis_y1 = max(0.5, min(n + 0.5, float(y_max)))

        if vis_x1 < vis_x0:
            vis_x0, vis_x1 = vis_x1, vis_x0
        if vis_y1 < vis_y0:
            vis_y0, vis_y1 = vis_y1, vis_y0

        first_x_idx = max(0, int(np.ceil(vis_x0 - 1.0)))
        last_x_idx = min(n - 1, int(np.floor(vis_x1 - 1.0)))
        first_y_idx = max(0, int(np.ceil(vis_y0 - 1.0)))
        last_y_idx = min(n - 1, int(np.floor(vis_y1 - 1.0)))

        visible_x = max(0, last_x_idx - first_x_idx + 1)
        visible_y = max(0, last_y_idx - first_y_idx + 1)

        max_ticks_x = max(1, plot_w // 42)
        max_ticks_y = max(1, plot_h // 24)
        step_x = _nice_tick_step(visible_x, max_ticks_x)
        step_y = _nice_tick_step(visible_y, max_ticks_y)

        x_tick_indices = list(range(first_x_idx, last_x_idx + 1, step_x)) if visible_x else []
        y_tick_indices = list(range(first_y_idx, last_y_idx + 1, step_y)) if visible_y else []

        x_ticks = tuple((str(displayed_order_ids[i]), i + 1) for i in x_tick_indices)
        y_ticks = tuple((str(displayed_order_ids[i]), i + 1) for i in y_tick_indices)

        dpg.set_axis_ticks("tan_x_axis", x_ticks)
        dpg.set_axis_ticks("tan_y_axis", y_ticks)

    with dpg.child_window(label="Tanimoto Similarity Matrix", parent="similarity_tanimoto_window",
                        no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
        
        with dpg.group():

            with dpg.group(horizontal=True):

                dpg.add_colormap_scale(
                    tag="tanimoto_matrix_colormap_scale",
                    label="Tanimoto Coefficient",
                    colormap=state["colormaps"][state["colormap_continuous"]],
                    mirror=True,
                    min_scale=vmin,
                    max_scale=vmax,
                    height=-1
                )
                
                with dpg.plot(width=-1, height=-1,
                            tag="tanimoto_matrix", no_mouse_pos=True, no_menus=True, no_frame=True, no_title=True, equal_aspects=True, zoom_rate=0.05):
                    dpg.add_plot_axis(dpg.mvXAxis, tag="tan_x_axis", 
                                    no_label=True, no_tick_labels=False, no_tick_marks=True, no_gridlines=True)
                    with dpg.plot_axis(dpg.mvYAxis, tag="tan_y_axis", 
                                    no_label=True, no_tick_labels=False, no_tick_marks=True, no_gridlines=True):
                        
                        dpg.add_heat_series(
                            heat_vals[::-1].flatten().tolist(),
                            rows=n, cols=n,
                            format="",
                            tag="tan_heat_series",
                            bounds_min=(0.5, 0.5),
                            bounds_max=(n + 0.5, n + 0.5)
                        )

                        _update_similarity_matrix_ticks()

                    dpg.add_line_series([0.5, n + 0.5, n + 0.5, 0.5, 0.5], [0.5, 0.5, n + 0.5, n + 0.5, 0.5], tag="tan_border_series", parent="tan_y_axis")

                    # Optional: make the border clearly visible
                    with dpg.theme() as tan_border_theme:
                        with dpg.theme_component(dpg.mvLineSeries):
                            dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 0), category=dpg.mvThemeCat_Plots)
                    dpg.bind_item_theme("tan_border_series", tan_border_theme)

                    if isinstance(highlighted_mol_id, int) and highlighted_mol_id in displayed_order_ids:
                        highlight_color = _get_highlight_color()
                        highlight_index = displayed_order_ids.index(highlighted_mol_id)
                        x0 = highlight_index + 0.5
                        x1 = highlight_index + 1.5
                        y0 = highlight_index + 0.5
                        y1 = highlight_index + 1.5

                        dpg.add_line_series(
                            [0.5, n + 0.5, n + 0.5, 0.5, 0.5],
                            [y0, y0, y1, y1, y0],
                            tag="tan_highlight_row_series",
                            parent="tan_y_axis"
                        )
                        dpg.add_line_series(
                            [x0, x1, x1, x0, x0],
                            [0.5, 0.5, n + 0.5, n + 0.5, 0.5],
                            tag="tan_highlight_col_series",
                            parent="tan_y_axis"
                        )

                        _refresh_tanimoto_highlight_theme()
                
                dpg.fit_axis_data("tan_x_axis")
                dpg.fit_axis_data("tan_y_axis")
                dpg.set_frame_callback(dpg.get_frame_count() + 1, _update_similarity_matrix_ticks)
                dpg.set_frame_callback(dpg.get_frame_count() + 2, _update_similarity_matrix_ticks)

                dpg.bind_colormap("tanimoto_matrix", state["colormaps"][state["colormap_continuous"]])
                dpg.bind_item_theme("tanimoto_matrix", apply_plot_theme(state))
                dpg.bind_item_theme("tanimoto_matrix_colormap_scale", apply_colormap_theme(state))
        set_loading_screen_progress(state, 94)


        with dpg.child_window(label="Tanimoto Similarity Matrix Mol Couple Window",
                            tag="similarity_tanimoto_mol_couple_inner_window",
                            parent="similarity_tanimoto_mol_couple_window",
                            no_scrollbar=True, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False, auto_resize_y=True):
            
            dpg.add_text("Tanimoto Coefficient, T = None", tag="info_tanimoto")
            
            with dpg.group():    

                with dpg.group():
                    empty_data = np.ones((render_w * render_h * 4,), dtype=np.float32)

                    if not dpg.does_item_exist("similarity_mol_x_image_texture"):
                        dpg.add_dynamic_texture(render_w, render_h, 
                                                empty_data, tag="similarity_mol_x_image_texture", parent="texture_registry")
                    else:
                        dpg.set_value("similarity_mol_x_image_texture", empty_data)

                    dpg.add_image("similarity_mol_x_image_texture", width=state["similarity_tan_img_width"], height=state["similarity_tan_img_height"],
                                tag="similarity_mol_x_image_widget", border_color=(0, 0, 0, 0))
                    with dpg.tooltip("similarity_mol_x_image_widget", delay=0):
                        dpg.add_text("", tag="similarity_mol_x_tooltip_text")
                    register_responsive_image(
                        state,
                        image_tag="similarity_mol_x_image_widget",
                        parent_tag="similarity_tanimoto_mol_couple_inner_window",
                        aspect_ratio=0.75,
                        tab="similarity_matrix_subtab",
                    )

                with dpg.group():
                    if not dpg.does_item_exist("similarity_mol_y_image_texture"):
                        dpg.add_dynamic_texture(render_w, render_h, 
                                                empty_data, tag="similarity_mol_y_image_texture", parent="texture_registry")
                    else:
                        dpg.set_value("similarity_mol_y_image_texture", empty_data)
                    dpg.add_image("similarity_mol_y_image_texture", width=state["similarity_tan_img_width"], height=state["similarity_tan_img_height"],
                                tag="similarity_mol_y_image_widget", border_color=(0, 0, 0, 0))
                    with dpg.tooltip("similarity_mol_y_image_widget", delay=0):
                        dpg.add_text("", tag="similarity_mol_y_tooltip_text")
                    register_responsive_image(
                        state,
                        image_tag="similarity_mol_y_image_widget",
                        parent_tag="similarity_tanimoto_mol_couple_inner_window",
                        aspect_ratio=0.75,
                        tab="similarity_matrix_subtab",
                    )
    set_loading_screen_progress(state, 98)


    # Click handler
    # -----------------------------------------------------------------------------
    # 3.1. Update similarity images
    # -----------------------------------------------------------------------------
    def _update_similarity_images(row_index: int, col_index: int) -> None:
        """
        Update similarity images.
        
        Args:
            row_index (int): Parameter accepted by this routine.
            col_index (int): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        smi_x = all_smiles[col_index]
        smi_y = all_smiles[row_index]
        mol_x = Chem.MolFromSmiles(smi_x)
        mol_y = Chem.MolFromSmiles(smi_y)
        if mol_x is None or mol_y is None:
            return
        show_fp_map = dpg.get_value("similarity_tanimoto_show_fp_map") if dpg.does_item_exist("similarity_tanimoto_show_fp_map") else True

        if show_fp_map:
            drawer_x = Draw.MolDraw2DCairo(render_w, render_h)
            opts_x = drawer_x.drawOptions()
            opts_x.padding = 0.025
            opts_x.bondLineWidth = 1
            opts_x.highlightRadius = 0.5
            opts_x.additionalAtomLabelPadding = 0.1
            opts_x.legendFontSize = 14
            SimilarityMaps.GetSimilarityMapForFingerprintGenerator(
                mol_y,
                mol_x,
                GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
                drawer_x,
                metric=DataStructs.TanimotoSimilarity,
                useCounts=True,
            )
            drawer_x.FinishDrawing()
            img_x = pilImage.open(io.BytesIO(drawer_x.GetDrawingText())).convert("RGBA")

            drawer_y = Draw.MolDraw2DCairo(render_w, render_h)
            opts_y = drawer_y.drawOptions()
            opts_y.padding = 0.025
            opts_y.bondLineWidth = 1
            opts_y.highlightRadius = 0.5
            opts_y.additionalAtomLabelPadding = 0.1
            opts_y.legendFontSize = 14
            SimilarityMaps.GetSimilarityMapForFingerprintGenerator(
                mol_x,
                mol_y,
                GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False),
                drawer_y,
                metric=DataStructs.TanimotoSimilarity,
                useCounts=True,
            )
            drawer_y.FinishDrawing()
            img_y = pilImage.open(io.BytesIO(drawer_y.GetDrawingText())).convert("RGBA")
        else:
            drawer_x = Draw.MolDraw2DCairo(render_w, render_h)
            opts_x = drawer_x.drawOptions()
            opts_x.padding = 0.025
            opts_x.bondLineWidth = 1
            opts_x.minFontSize = 1
            drawer_x.DrawMolecule(mol_x)
            drawer_x.FinishDrawing()
            img_x = pilImage.open(io.BytesIO(drawer_x.GetDrawingText())).convert("RGBA")

            drawer_y = Draw.MolDraw2DCairo(render_w, render_h)
            opts_y = drawer_y.drawOptions()
            opts_y.padding = 0.025
            opts_y.bondLineWidth = 1
            opts_y.minFontSize = 1
            drawer_y.DrawMolecule(mol_y)
            drawer_y.FinishDrawing()
            img_y = pilImage.open(io.BytesIO(drawer_y.GetDrawingText())).convert("RGBA")

        img_arr_x = (np.array(img_x) / 255.0).astype(np.float32).flatten()
        img_arr_y = (np.array(img_y) / 255.0).astype(np.float32).flatten()

        mol_id_x = displayed_order_ids[col_index]
        mol_id_y = displayed_order_ids[row_index]
        state["similarity_tanimoto_last_pair"] = (row_index, col_index)
        dpg.set_value("similarity_mol_x_image_texture", img_arr_x)
        dpg.set_value("similarity_mol_y_image_texture", img_arr_y)
        dpg.set_value("similarity_mol_x_tooltip_text", f"X: Mol {mol_id_x}")
        dpg.set_value("similarity_mol_y_tooltip_text", f"Y: Mol {mol_id_y}")
        tanimoto_value = similarity_matrix[row_index, col_index]
        dpg.set_value("info_tanimoto", f"Tanimoto Coefficient, T = {tanimoto_value:.2f}")

    state["refresh_similarity_tanimoto_images"] = lambda: (
        _update_similarity_images(*state["similarity_tanimoto_last_pair"])
        if state.get("similarity_tanimoto_last_pair") is not None
        else None
    )


    # Click handler
    # -----------------------------------------------------------------------------
    # 3.2. On heatmap click
    # -----------------------------------------------------------------------------
    def on_heatmap_click(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Execute the on heatmap click routine.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
            user_data (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        if not dpg.is_item_hovered("tanimoto_matrix"):
            return
        mouse_pos = dpg.get_plot_mouse_pos()
        if not mouse_pos:
            return

        x_pos, y_pos = mouse_pos
        if x_pos < 0.5 or x_pos > n + 0.5 or y_pos < 0.5 or y_pos > n + 0.5:
            return

        col_index = int(x_pos - 0.5)
        row_index = int(y_pos - 0.5)
        if not (0 <= col_index < n and 0 <= row_index < n):
            return
        _update_similarity_images(row_index, col_index)


    # Register the click handler and bind it to the heatmap plot.
    if dpg.does_item_exist("tanimoto_matrix_click_handler"): 
        dpg.delete_item("tanimoto_matrix_click_handler")

    dpg.add_mouse_click_handler(tag="tanimoto_matrix_click_handler", parent="handler_registry",
                                button=dpg.mvMouseButton_Left, callback=on_heatmap_click)

    for tag in [
        "tanimoto_matrix_wheel_handler",
        "tanimoto_matrix_drag_handler",
        "tanimoto_matrix_release_handler",
    ]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    dpg.add_mouse_wheel_handler(
        tag="tanimoto_matrix_wheel_handler",
        parent="handler_registry",
        callback=lambda s, a, u: _update_similarity_matrix_ticks() if dpg.is_item_hovered("tanimoto_matrix") else None,
    )
    dpg.add_mouse_drag_handler(
        tag="tanimoto_matrix_drag_handler",
        parent="handler_registry",
        callback=lambda s, a, u: _update_similarity_matrix_ticks() if dpg.is_item_hovered("tanimoto_matrix") else None,
    )
    dpg.add_mouse_release_handler(
        tag="tanimoto_matrix_release_handler",
        parent="handler_registry",
        callback=lambda s, a, u: _update_similarity_matrix_ticks() if dpg.does_item_exist("tanimoto_matrix") else None,
    )

    # Simulate first click on bottom-left cell (x=1, y=1) => indices (0,0)
    if n > 0:
        _update_similarity_images(0, 0)

    set_loading_screen_progress(state, 100)

    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")
