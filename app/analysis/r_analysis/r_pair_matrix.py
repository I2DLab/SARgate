"""
===================
rgroups_table.py
===================

R-group decomposition table visualisation.

Generates the interactive table of R-group fragments extracted from molecular
datasets. Displays associated activity values, chemical structures, and summary
statistics, allowing users to explore the relationship between substituents and
bioactivity.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Draw rgroups table

import io
import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
from typing import Any
from PIL import Image as pilImage, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from app.utils.app_logger import log_event, log_settings
from app.utils.callbacks import (
    export_png_popup,
    register_responsive_image
)
from app.gui.themes_manager import (
    get_continuous_colormap_color,
    apply_colormap_theme,
    apply_boxplot_theme
)
from app.gui.loading_win import set_loading_screen_progress


# -----------------------------------------------------------------------------
# 2. Draw rgroups table
# -----------------------------------------------------------------------------
def draw_rgroups_table(
    subset: str,
    activity: str,
    r1_col: Any,
    r2_col: Any,
    data: Any,
    read_undefined: bool,
    state: dict[str, Any],
    activity_mode: str = "Best activity",
) -> Any:
    """
    Draw a 2D R-group combination table (heatmap) against a selected activity.

    Args:
        subset (str): Current subset identifier for labelling.
        activity (str): Target activity column to summarise.
        r1_col (str): Column name for R-group on the X axis (columns).
        r2_col (str): Column name for R-group on the Y axis (rows).
        data (DataFrame): Input dataset including activity and R-group columns.
        read_undefined (bool): Include values with qualifiers (>, <, >=, <=) if True.
        state (dict): Global application state with plotting/theme settings.

    Returns:
        None
    """
    log_event("R-Analysis", "Rendering R-group pair matrix", indent=1)
    log_settings("R-Analysis", indent=2, subset=subset, activity=activity, r1=r1_col, r2=r2_col, include_undefined=read_undefined, rows=len(data), mode=activity_mode)

    for tag in ("heatmap_window", "heatmap_details_window"):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=True)

    # Guard window: ensure a modal popup exists to notify identical R-group selection.
    if not dpg.does_item_exist("same_rgroup_popup"):
        with dpg.window(label="R-Groups Error", tag="same_rgroup_popup", show=False, modal=True, no_close=True, autosize=True):
            dpg.add_text("The compared R-groups cannot be the same")
            dpg.add_button(label="OK", callback=lambda: dpg.configure_item("same_rgroup_popup", show=False))

    # Early exit: prevent comparing the same R-group column against itself.
    if r1_col == r2_col:
        dpg.configure_item("same_rgroup_popup", show=True)
        return

    # Cache image sizes for header/side thumbnails from state.
    heatmap_side_img_width = state["plots_heatmap_img_width"]
    heatmap_side_img_height = state["plots_heatmap_img_height"]
    heatmap_scaffold_render_scale = 1.8
    heatmap_scaffold_render_width = int(round(heatmap_side_img_width * heatmap_scaffold_render_scale))
    heatmap_scaffold_render_height = int(round(heatmap_side_img_height * heatmap_scaffold_render_scale))
    heatmap_group_render_scale = 1.8
    heatmap_group_render_width = int(round(heatmap_side_img_width * heatmap_group_render_scale))
    heatmap_group_render_height = int(round(heatmap_side_img_height * heatmap_group_render_scale))


    # Identify the block of activity columns and validate the requested 'activity'.

    # --- STEP 1.1: Validate target activity column presence ---
    activity_columns = list(data.columns)
    chi4_index = activity_columns.index("Chi4")
    activity_cols = activity_columns[chi4_index + 1:]

    if activity not in activity_cols:
        return

    set_loading_screen_progress(state, 8)

    # -----------------------------------------------------------------------------
    # 2.1. Parse activity value
    # -----------------------------------------------------------------------------
    def parse_activity_value(val: Any) -> Any:
        """
        Parse activity value.
        
        Args:
            val (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        val = str(val).strip()
        if val.startswith((">", "<", ">=", "<=", "=")):
            for sym in [">=", "<=", ">", "<", "="]:
                if val.startswith(sym):
                    try:
                        return float(val.replace(sym, "").strip()), sym
                    except:
                        return None, None
        else:
            try:
                return float(val), None
            except:
                return None, None

    # -----------------------------------------------------------------------------
    # 2.2. Tooltip contrast color
    # -----------------------------------------------------------------------------
    def _tooltip_text_color(bg_rgba: Any) -> tuple[int, int, int, int]:
        """
        Choose black or white text for the tooltip based on background contrast.

        Args:
            bg_rgba (Any): Background RGBA tuple.

        Returns:
            tuple[int, int, int, int]: Contrasting text color.
        """
        r, g, b = bg_rgba[:3]
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        return (20, 20, 20, 255) if luminance > 0.6 else (245, 245, 245, 255)

    # -----------------------------------------------------------------------------
    # 2.3. Build heatmap tooltip theme
    # -----------------------------------------------------------------------------
    def _build_heatmap_tooltip_theme(bg_rgba: Any) -> None:
        """
        Apply a dynamic theme to the heatmap tooltip window using the hovered cell color.

        Args:
            bg_rgba (Any): Background RGBA tuple.

        Returns:
            None: This routine updates the tooltip theme in place.
        """
        text_rgba = _tooltip_text_color(bg_rgba)
        theme_tag = "heatmap_tooltip_theme"
        window_bg_tag = "heatmap_tooltip_theme_window_bg"
        border_tag = "heatmap_tooltip_theme_border"
        text_tag = "heatmap_tooltip_theme_text"

        if not dpg.does_item_exist(theme_tag):
            with dpg.theme(tag=theme_tag):
                with dpg.theme_component(dpg.mvWindowAppItem):
                    dpg.add_theme_color(dpg.mvThemeCol_WindowBg, bg_rgba, tag=window_bg_tag)
                    dpg.add_theme_color(dpg.mvThemeCol_Border, bg_rgba, tag=border_tag)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
                    dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 6)
                with dpg.theme_component(dpg.mvText):
                    dpg.add_theme_color(dpg.mvThemeCol_Text, text_rgba, tag=text_tag)

        dpg.set_value(window_bg_tag, bg_rgba)
        dpg.set_value(border_tag, bg_rgba)
        dpg.set_value(text_tag, text_rgba)

        if dpg.does_item_exist("heatmap_tooltip_window"):
            dpg.bind_item_theme("heatmap_tooltip_window", theme_tag)

    # -----------------------------------------------------------------------------
    # 2.4. Get heatmap cell color
    # -----------------------------------------------------------------------------
    def _get_heatmap_cell_color(value: Any, tooltip: Any, vmin: float, vmax: float) -> tuple[int, int, int, int]:
        """
        Return the displayed RGBA color for a heatmap cell.

        Args:
            value (Any): Cell numeric value.
            tooltip (Any): Tooltip lines associated with the cell.
            vmin (float): Minimum scale value.
            vmax (float): Maximum scale value.

        Returns:
            tuple[int, int, int, int]: RGBA color used for the cell.
        """
        if tooltip and tooltip[0].startswith("NO OCCURRENCES"):
            return (0, 0, 0, 200)
        if tooltip and tooltip[0].startswith("N/A ONLY"):
            return (70, 70, 70, 200)
        if np.isnan(value):
            return (0, 0, 0, 200)

        safe_vmax = vmax if vmax > vmin else vmin + 1.0
        norm_val = (value - vmin) / (safe_vmax - vmin)
        return get_continuous_colormap_color(norm_val, state)

    def _best_raw_activity(values: list[float]) -> float:
        if not values:
            return float("nan")
        if activity in state["ug/mL_activities"]:
            return float(min(values))
        return float(max(values))

    mode_is_best = str(activity_mode or "Best activity").strip() == "Best activity"

    parsed_rows = []
    grouped_data = list(data.groupby("Mol_sub_ID"))
    total_groups = max(1, len(grouped_data))
    for group_idx, (_, group) in enumerate(grouped_data, start=1):
        for _, row in group.iterrows():
            val = row[activity]
            parsed_row = row.copy()
            if pd.isna(val) or val == "":
                parsed_row["__parsed_activity__"] = "N/A"
                parsed_rows.append(parsed_row)
                continue
            num, sym = parse_activity_value(val)
            if num is not None:
                if sym and not read_undefined:
                    continue
                parsed_row["__parsed_activity__"] = num
                parsed_rows.append(parsed_row)
        if group_idx == total_groups or group_idx % 25 == 0:
            set_loading_screen_progress(state, 8 + (group_idx / total_groups) * 22)

    df = pd.DataFrame(parsed_rows)
    set_loading_screen_progress(state, 30)


    # Build ordered sets for R1 (columns) and R2 (rows); clear any previous heatmap resources.

    r1_counts = df[r1_col].value_counts()
    r1_set = r1_counts.index.tolist()
    r2_counts = df[r2_col].value_counts(ascending=True)
    r2_set = r2_counts.index.tolist()

    ncols = len(r1_set)                 # Number of R1 categories (columns)
    nrows = len(r2_set)                 # Number of R2 categories (rows)
    state["heatmap_ncols"] = ncols
    state["heatmap_nrows"] = nrows
    heatmap_matrix = np.full((nrows, ncols), np.nan)
    tooltip_matrix = [[[] for _ in r1_set] for _ in r2_set]
    cell_rgroup_smiles = {}

    # Reset lingering tooltip window, plot, drawlist, handlers, and header textures to avoid duplication.
    # Close any dangling tooltips.
    if dpg.does_item_exist("heatmap_tooltip_window"):
        dpg.delete_item("heatmap_tooltip_window")

    # Remove previous plot/drawlist/handlers if present.
    for tag in ("heatmap_plot", "heatmap_drawlist"):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        

    # Populate numeric matrix and tooltip content per cell, applying pActivity conversion when needed.

    total_cells = max(1, nrows * ncols)
    processed_cells = 0
    for i, r2 in enumerate(r2_set):
        for j, r1 in enumerate(r1_set):
            matches = df[(df[r1_col] == r1) & (df[r2_col] == r2)]
            y_plot = nrows - i - 1
            cell_rgroup_smiles[(j, y_plot)] = (r1, r2)

            if matches.empty:
                tooltip_matrix[i][j] = ["NO OCCURRENCES"]
                continue

            active = matches[matches["__parsed_activity__"].apply(lambda x: isinstance(x, (float, int)))]
            if active.empty:
                tooltip_matrix[i][j] = [f"N/A ONLY\nTotal counts: {len(matches)}"]
                continue

            values = active["__parsed_activity__"].tolist()

            # Convert activities to pScale when in nM units; keep raw otherwise, with units in tooltip.
            if activity in state["nM_activity_types"]:
                values_log = [9 - np.log10(v) if v > 0 else float("nan") for v in values]
                values_log = [v for v in values_log if not np.isnan(v)]
                if not values_log:
                    tooltip_matrix[i][j] = [f"ERROR: non-positive values", f"Total: {len(values)}"]
                    continue
                cell_value = float(max(values_log)) if mode_is_best else float(np.mean(values_log))
                heatmap_matrix[i][j] = cell_value
                stat_label = "Best" if mode_is_best else "Mean"
                tooltip_matrix[i][j] = [
                    f"{stat_label} p{activity}: {cell_value:.2f}",
                    f"Range: {min(values_log):.2f} - {max(values_log):.2f}",
                    f"Total counts: {len(matches)}",
                    f"With activity: {len(values)}",
                    f"N/A: {len(matches) - len(values)}",
                ]
            else:
                cell_value = _best_raw_activity(values) if mode_is_best else float(np.mean(values))
                heatmap_matrix[i][j] = cell_value
                units = (
                    "%" if activity in state["percent_activities"] else
                    "μg/mL" if activity in state["ug/mL_activities"] else
                    "μM/min" if activity in state["uM/min_activities"] else
                    "nM"
                )
                stat_label = "Best" if mode_is_best else "Mean"
                tooltip_matrix[i][j] = [
                    f"{stat_label} {activity}: {cell_value:.2f} {units}",
                    f"Range: {min(values):.2f} - {max(values):.2f} {units}",
                    f"Total counts: {len(matches)}",
                    f"With activity: {len(values)}",
                    f"N/A: {len(matches) - len(values)}",
                ]
            processed_cells += 1
            if processed_cells == total_cells or processed_cells % max(1, total_cells // 40) == 0:
                set_loading_screen_progress(state, 30 + (processed_cells / total_cells) * 20)


    # Convert NaNs/empty to None for draw loop and collect flat list for range stats.

    display_matrix = []
    flat_values = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            val = heatmap_matrix[i][j]
            if tooltip_matrix[i][j] == ["NO OCCURRENCES"] or np.isnan(val):
                row.append(None)
                flat_values.append(float("nan"))
            else:
                row.append(val)
                flat_values.append(val)
        display_matrix.append(row)


    # Compute vmin/vmax robustly, ensuring vmax > vmin to avoid division by zero.

    valid_values = [v for v in flat_values if not np.isnan(v)]
    vmin, vmax = (min(valid_values), max(valid_values)) if valid_values else (0.0, 1.0)
    if vmax == vmin:                      # Robust normalisation in degenerate cases
        vmax = vmin + 1.0
    set_loading_screen_progress(state, 52)


    # Utility helpers: overlay labels on PIL images, convert SMILES to texture data, render R-group images,
    # format activity value strings, and handle click/hover interactions on the heatmap.

    # -----------------------------------------------------------------------------
    # 2.2. Add label on pil
    # -----------------------------------------------------------------------------
    def _load_pil_label_font(size_px: int) -> Any:
        """
        Load a readable PIL font for oversampled labels.

        Args:
            size_px (int): Requested font size in pixels.

        Returns:
            Any: PIL font instance.
        """
        for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(font_name, size_px)
            except Exception:
                continue
        return ImageFont.load_default()

    def _add_label_on_pil(
        img: Any,
        text: str,
        position: str = "top",
        margin: int = 6,
        pad_x: int = 6,
        pad_y: int = 3,
        bg: Any = None,
        fg: Any = (0, 0, 0, 0)
    ) -> Any:
        """
        Draw a centred label at the top or bottom of an RGBA image, black text with optional background.

        Args:
            img (PIL.Image): RGBA image to annotate.
            text (str): Label text to render.
            position (str): "top" or "bottom" placement.
            margin (int): Pixel distance from the edge.
            pad_x (int): Horizontal padding for label background.
            pad_y (int): Vertical padding for label background.
            bg (tuple|None): Optional RGBA background colour; None for no background.
            fg (tuple): Text colour RGBA.

        Returns:
            PIL.Image: Annotated image object (same instance, modified in place).
        """
        if not text:
            return img

        draw = ImageDraw.Draw(img, "RGBA")
        try:
            target_size = max(18, int(round(img.width * 0.065)))
            font = _load_pil_label_font(target_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            while target_size > 12 and (bbox[2] - bbox[0]) > (img.width - 2 * margin):
                target_size -= 1
                font = _load_pil_label_font(target_size)
                bbox = draw.textbbox((0, 0), text, font=font)
        except Exception:
            font = None

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        W, H = img.size

        x = (W - tw) // 2
        y = margin if position == "top" else (H - th - margin)

        # Optional background (only if 'bg' is provided)
        if bg:
            draw.rectangle([x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y], fill=bg)

        draw.text((x, y), text, fill=fg, font=font)
        return img

    # -----------------------------------------------------------------------------
    # 2.3. Smiles to texture data
    # -----------------------------------------------------------------------------
    def _smiles_to_texture_data(
        smiles: str,
        w: int = 256,
        h: int = 256,
        label: str | None = None,
        label_pos: str = "top"
    ) -> Any:
        """
        Return flattened float32 RGBA for a molecule image (with optional label).
        
        Args:
            smiles (str): Parameter accepted by this routine.
            w (int): Parameter accepted by this routine. Defaults to the configured value.
            h (int): Parameter accepted by this routine. Defaults to the configured value.
            label (str): Parameter accepted by this routine. Defaults to the configured value.
            label_pos (Any): Parameter accepted by this routine. Defaults to the configured value.
        
        Returns:
            Any: Value produced by the routine.
        """
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
            opts = drawer.drawOptions()
            opts.clearBackground = True
            opts.padding = 0.05
            opts.bondLineWidth = 1
            opts.minFontSize = 1
            opts.explicitMethyl = False
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()

            img = pilImage.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")

            arr = np.array(img)
            arr[..., 3] = 255
            return (arr / 255.0).flatten().astype(np.float32)
        except Exception:
            return np.zeros((w * h * 4), dtype=np.float32)
        
    # -----------------------------------------------------------------------------
    # 2.4. Render rgroup image
    # -----------------------------------------------------------------------------
    def render_rgroup_image(smiles: str, texture_tag: str, label_text: str | None = None) -> None:
        # Render an R-group thumbnail to a dynamic texture with an optional label.
        """
        Execute the render rgroup image routine.
        
        Args:
            smiles (str): Parameter accepted by this routine.
            texture_tag (Any): Parameter accepted by this routine.
            label_text (Any): Parameter accepted by this routine. Defaults to the configured value.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        img_data = _smiles_to_texture_data(
            smiles,
            heatmap_group_render_width,
            heatmap_group_render_height,
            label=label_text,
            label_pos="top"
        )
        dpg.set_value(texture_tag, img_data)
        tooltip_text_tag = f"{texture_tag}_tooltip_text"
        if dpg.does_item_exist(tooltip_text_tag):
            dpg.set_value(tooltip_text_tag, label_text or "")

    # -----------------------------------------------------------------------------
    # 2.5. Format activity value
    # -----------------------------------------------------------------------------
    def _format_activity_value(row: Any, activity: str, state: dict[str, Any]) -> Any:
        """
        Return a string 'activity = value unit' or 'N/A' if empty/NaN.
        
        Args:
            row (Any): Parameter accepted by this routine.
            activity (str): Parameter accepted by this routine.
            state (dict[str, Any]): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        val = row.get(activity, "")
        if pd.isna(val) or str(val).strip() == "":
            return "N/A"
        unit = (
            "nM" if activity in state["nM_activity_types"] else
            "%" if activity in state["percent_activities"] else
            "μg/mL" if activity in state["ug/mL_activities"] else
            "μM/min" if activity in state["uM/min_activities"] else
            ""
        )
        return f"{activity} = {val} {unit}".strip()

    # -----------------------------------------------------------------------------
    # 2.6. Handle click heatmap
    # -----------------------------------------------------------------------------
    def handle_click_heatmap(sender: Any, app_data: Any, user_data: Any = None) -> Any:
        # Handle left-click on the heatmap area: update selected R1/R2 images and list matching molecules.
        """
        Execute the handle click heatmap routine.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
            user_data (Any): Parameter accepted by this routine. Defaults to the configured value.
        
        Returns:
            Any: Value produced by the routine.
        """
        mouse_pos = dpg.get_mouse_pos()
        plot_min = dpg.get_item_rect_min("heatmap_plot")
        plot_max = dpg.get_item_rect_max("heatmap_plot")
        if not (plot_min[0] <= mouse_pos[0] <= plot_max[0] and plot_min[1] <= mouse_pos[1] <= plot_max[1]):
            return

        x, y = dpg.get_plot_mouse_pos()
        # Also consider header bands; bail if outside main grid + headers.
        if x is None or y is None or x < 0.5 or y < 0.5 or x > ncols + 0.5 or y > nrows + 0.5:
            return

        j, i = int(x - 0.5), int(y - 0.5)
        if state.get("last_clicked_heatmap_cell") == (j, i) or state["current_r_analysis_subtab"] != "r_analysis_table_subtab":
            return
        state["last_clicked_heatmap_cell"] = (j, i)

        if (j, i) not in cell_rgroup_smiles:
            return

        # 1) Update the two R1/R2 images with labels.
        r1_smiles, r2_smiles = cell_rgroup_smiles[(j, i)]
        render_rgroup_image(r1_smiles, "heatmap_group_r1_image_texture", f"{r1_col} Group")
        render_rgroup_image(r2_smiles, "heatmap_group_r2_image_texture", f"{r2_col} Group")

        # 2) Build the molecule list for the (r1, r2) pair with activity strings.
        r1_val, r2_val = r1_smiles, r2_smiles
        matching_rows = df[(df[r1_col] == r1_val) & (df[r2_col] == r2_val)].copy()

        # Optional: order by actives first, then inactives.
        # -----------------------------------------------------------------------------
        # 2.6.1. Key sort
        # -----------------------------------------------------------------------------
        def _key_sort(row: Any) -> Any:
            """
            Execute the key sort routine.
            
            Args:
                row (Any): Parameter accepted by this routine.
            
            Returns:
                Any: Value produced by the routine.
            """
            v = row.get("__parsed_activity__", "N/A")
            return (0, -float(v)) if isinstance(v, (int, float)) else (1, 0.0)
        matching_rows = sorted(matching_rows.to_dict("records"), key=_key_sort)

        # Order matching_rows in ascending order by ID, robust for list or DataFrame inputs.
        if isinstance(matching_rows, pd.DataFrame):
            if "Mol_sub_ID" in matching_rows.columns:
                # Ensure numeric and sort.
                tmp = matching_rows.copy()
                tmp["__mol_id__"] = pd.to_numeric(tmp["Mol_sub_ID"], errors="coerce")
                matching_rows = tmp.sort_values("__mol_id__", ascending=True, kind="mergesort")
            elif "MolID" in matching_rows.columns:
                tmp = matching_rows.copy()
                tmp["__mol_id__"] = pd.to_numeric(tmp["MolID"], errors="coerce")
                matching_rows = tmp.sort_values("__mol_id__", ascending=True, kind="mergesort")
            else:
                # Fallback: keep insertion order.
                pass
            rows_iter = matching_rows.to_dict("records")
        else:
            # matching_rows is already a list of dicts.
            def _id_key(r: Any) -> Any:
                """
                Execute the id key routine.
                
                Args:
                    r (Any): Input accepted by this routine.
                
                Returns:
                    Any: Value returned by the routine.
                """
                v = r.get("Mol_sub_ID", r.get("MolID", None))
                try:
                    return float(v)
                except Exception:
                    return float("inf")
            rows_iter = sorted(matching_rows, key=_id_key, reverse=False)
                
        lines = []
        for row in rows_iter:
            mol_id = int(row["Mol_sub_ID"]) if "Mol_sub_ID" in row and pd.notna(row["Mol_sub_ID"]) else row.get("MolID", "NA")
            act_str = _format_activity_value(row, activity, state)
            lines.append(f"Mol {mol_id}  »  {act_str}")
            
        total = df["Mol_sub_ID"].nunique() if "Mol_sub_ID" in df.columns else len(df)
        header = f"{len(matching_rows)}/{total} Molecules:"
        detail_text = header + ("\n\n" + "\n".join(lines) if lines else "\n\n(No matching molecules)")
        if dpg.does_item_exist("heatmap_molecules_containing_selected_rpair_text"):
            dpg.set_value("heatmap_molecules_containing_selected_rpair_text", detail_text)
                    
    # -----------------------------------------------------------------------------
    # 2.7. Handle hover heatmap
    # -----------------------------------------------------------------------------
    def handle_hover_heatmap(sender: Any, app_data: Any, user_data: Any = None) -> None:
        # Handle mouse move: show/hide tooltip window with cell statistics near the cursor.
        """
        Execute the handle hover heatmap routine.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
            user_data (Any): Parameter accepted by this routine. Defaults to the configured value.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        if state["current_r_analysis_subtab"] != "r_analysis_table_subtab" and dpg.does_item_exist("heatmap_tooltip_window"):
            state["last_hovered_heatmap_cell"] = None
            if dpg.does_item_exist("heatmap_tooltip_window"):
                dpg.delete_item("heatmap_tooltip_window")

        mouse_pos = dpg.get_mouse_pos()
        try:
            plot_min = dpg.get_item_rect_min("heatmap_plot")
            plot_max = dpg.get_item_rect_max("heatmap_plot")
        except Exception:
            plot_min = (0, 0)
            plot_max = (0, 0)

        if not (plot_min[0] <= mouse_pos[0] <= plot_max[0] and plot_min[1] <= mouse_pos[1] <= plot_max[1]):
            state["last_hovered_heatmap_cell"] = None
            if dpg.does_item_exist("heatmap_tooltip_window"):
                dpg.delete_item("heatmap_tooltip_window")
            return

        x, y = dpg.get_plot_mouse_pos()
        # Ignore header bands when outside plot extents.
        if x is None or y is None or x < 0.5 or y < 0.5 or x > ncols + 0.5 or y > nrows + 0.5:
            state["last_hovered_heatmap_cell"] = None
            if dpg.does_item_exist("heatmap_tooltip_window"):
                dpg.delete_item("heatmap_tooltip_window")
            return

        j, i = int(x - 0.5), int(y - 0.5)

        # Keep tooltip position updated while hovering the same cell.
        if state.get("last_hovered_heatmap_cell") == (j, i):
            if dpg.does_item_exist("heatmap_tooltip_window"):
                dpg.set_item_pos("heatmap_tooltip_window", (mouse_pos[0] + 15, mouse_pos[1] + 15))
            return
        else:
            state["last_hovered_heatmap_cell"] = (j, i)
            if dpg.does_item_exist("heatmap_tooltip_window"):
                dpg.delete_item("heatmap_tooltip_window")

        # Create/update tooltip content only when hovering a valid cell in the active tab.
        if (j, i) in cell_rgroup_smiles and state["current_r_analysis_subtab"] == "r_analysis_table_subtab":
            tooltip = tooltip_matrix[nrows - i - 1][j]
            cell_value = heatmap_matrix[nrows - i - 1][j]
            cell_color = _get_heatmap_cell_color(cell_value, tooltip, float(vmin), float(vmax))
            with dpg.window(tag="heatmap_tooltip_window", show=True, no_title_bar=True, no_resize=True,
                            no_move=True, no_scrollbar=True, no_close=True, no_focus_on_appearing=True,
                            autosize=True, min_size=(10, 10), pos=(mouse_pos[0] + 15, mouse_pos[1] + 15)):
                dpg.add_text("\n".join(tooltip))
            _build_heatmap_tooltip_theme(cell_color)
        else:
            state["last_hovered_heatmap_cell"] = None
            if dpg.does_item_exist("heatmap_tooltip_window"):
                dpg.delete_item("heatmap_tooltip_window")


    # Compose the full UI: colormap scale, square-grid plot with headers, interaction handlers, and side panel.
    with dpg.child_window(parent="heatmap_window", width=-1, height=-1,
                          no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):

        with dpg.group(horizontal=True):
            # Left column: colormap scale and the square-aspect heatmap plot.
            with dpg.group(horizontal=True):
                if activity in state["nM_activity_types"]:
                    dpg.add_colormap_scale(tag="heatmap_colormap_scale", colormap=state["colormaps"][state["colormap_continuous"]], label=f"p{activity}",
                                            min_scale=vmin, max_scale=vmax, height=-1, mirror=True, format="%.2f")
                else:
                    units = (
                        "%" if activity in state["percent_activities"] else
                        "μg/mL" if activity in state["ug/mL_activities"] else
                        "μM/min" if activity in state["uM/min_activities"] else
                        "nM"
                    )
                    dpg.add_colormap_scale(tag="heatmap_colormap_scale", colormap=state["colormaps"][state["colormap_continuous"]], label=f"{activity} ({units})",
                                            min_scale=vmin, max_scale=vmax, height=-1, mirror=True, format="%.2f")
                dpg.bind_item_theme("heatmap_colormap_scale", apply_colormap_theme(state))

                # Square cells enforced via equal_aspects=True; hide frame and menus for a clean look.
                with dpg.plot(label=f"{subset.replace('subset_', 'Subset ')}  -  {r1_col} vs {r2_col}", 
                              width=-1, height=-1, tag="heatmap_plot",
                              equal_aspects=True, no_frame=True,
                              no_menus=True, no_mouse_pos=True, zoom_rate=0.05) as plot:

                    dpg.add_plot_axis(dpg.mvXAxis, label=r1_col, tag="heat_x_axis",
                                      no_tick_marks=True, no_highlight=True, no_gridlines=True)
                    dpg.add_plot_axis(dpg.mvYAxis, label=r2_col, tag="heat_y_axis",
                                      no_tick_marks=True, no_highlight=True, no_gridlines=True)

                    # Reserve 1-unit header bands for R1 (top/bottom) and R2 (left/right).
                    HEADER_TOP = 1.0
                    HEADER_LEFT = 1.0

                    def _nice_tick_step(visible_count: int, max_visible_ticks: int) -> int:
                        """
                        Pick a readable 1-2-5 style tick step.

                        Args:
                            visible_count (int): Number of visible matrix indices.
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

                    def _update_heatmap_ticks() -> None:
                        """
                        Update heatmap axis ticks based on the currently visible region.

                        Args:
                            None.

                        Returns:
                            None: This routine updates plot ticks in place.
                        """
                        if not (
                            dpg.does_item_exist("heatmap_plot")
                            and dpg.does_item_exist("heat_x_axis")
                            and dpg.does_item_exist("heat_y_axis")
                        ):
                            return

                        try:
                            x_min, x_max = dpg.get_axis_limits("heat_x_axis")
                            y_min, y_max = dpg.get_axis_limits("heat_y_axis")
                        except Exception:
                            return

                        try:
                            plot_w, plot_h = dpg.get_item_rect_size("heatmap_plot")
                        except Exception:
                            plot_w = plot_h = 0

                        plot_w = max(1, int(plot_w or 0))
                        plot_h = max(1, int(plot_h or 0))

                        vis_x0 = max(0.5, min(ncols + 0.5, float(x_min)))
                        vis_x1 = max(0.5, min(ncols + 0.5, float(x_max)))
                        vis_y0 = max(0.5, min(nrows + 0.5, float(y_min)))
                        vis_y1 = max(0.5, min(nrows + 0.5, float(y_max)))

                        if vis_x1 < vis_x0:
                            vis_x0, vis_x1 = vis_x1, vis_x0
                        if vis_y1 < vis_y0:
                            vis_y0, vis_y1 = vis_y1, vis_y0

                        first_x_idx = max(0, int(np.ceil(vis_x0 - 1.0)))
                        last_x_idx = min(ncols - 1, int(np.floor(vis_x1 - 1.0)))
                        first_y_idx = max(0, int(np.ceil(vis_y0 - 1.0)))
                        last_y_idx = min(nrows - 1, int(np.floor(vis_y1 - 1.0)))

                        visible_x = max(0, last_x_idx - first_x_idx + 1)
                        visible_y = max(0, last_y_idx - first_y_idx + 1)

                        max_ticks_x = max(1, plot_w // 42)
                        max_ticks_y = max(1, plot_h // 24)
                        step_x = _nice_tick_step(visible_x, max_ticks_x)
                        step_y = _nice_tick_step(visible_y, max_ticks_y)

                        x_tick_indices = list(range(first_x_idx, last_x_idx + 1, step_x)) if visible_x else []
                        y_tick_indices = list(range(first_y_idx, last_y_idx + 1, step_y)) if visible_y else []

                        x_ticks = tuple((str(i + 1), i + 1) for i in x_tick_indices)
                        y_ticks = tuple((str(i + 1), i + 1) for i in y_tick_indices)

                        dpg.set_axis_ticks("heat_x_axis", x_ticks)
                        dpg.set_axis_ticks("heat_y_axis", y_ticks)

                    _update_heatmap_ticks()

                    # Header textures registry: create dynamic textures for top/bottom (R1) and left/right (R2).
                    header_tex_size = 256

                    # Column headers (R1)
                    total_headers = max(1, ncols + nrows)
                    processed_headers = 0
                    for i, r1_smiles in enumerate(r1_set):
                        if not dpg.does_item_exist(f"heatmap_r1_tex_{i}"):
                            dpg.add_dynamic_texture(
                                header_tex_size, header_tex_size,
                                _smiles_to_texture_data(r1_smiles, header_tex_size, header_tex_size),
                                tag=f"heatmap_r1_tex_{i}", parent="texture_registry"
                            )
                        else:
                            dpg.set_value(f"heatmap_r1_tex_{i}",
                                          _smiles_to_texture_data(r1_smiles, header_tex_size, header_tex_size))
                        processed_headers += 1
                        if processed_headers == total_headers or processed_headers % max(1, total_headers // 10) == 0:
                            set_loading_screen_progress(state, 54 + (processed_headers / total_headers) * 6)
                    # Row headers (R2)
                    for j, r2_smiles in enumerate(r2_set):
                        if not dpg.does_item_exist(f"heatmap_r2_tex_{j}"):
                            dpg.add_dynamic_texture(
                                header_tex_size, header_tex_size,
                                _smiles_to_texture_data(r2_smiles, header_tex_size, header_tex_size),
                                tag=f"heatmap_r2_tex_{j}", parent="texture_registry"
                            )
                        else:
                            dpg.set_value(f"heatmap_r2_tex_{j}",
                                          _smiles_to_texture_data(r2_smiles, header_tex_size, header_tex_size))
                        processed_headers += 1
                        if processed_headers == total_headers or processed_headers % max(1, total_headers // 10) == 0:
                            set_loading_screen_progress(state, 54 + (processed_headers / total_headers) * 6)

                    # Draw layer inside plot coordinates: grid cells and mirrored header thumbnails.
                    with dpg.draw_layer(parent=plot, tag="heatmap_drawlist"):

                        cell_size = 1.0
                        cell_tags = []  # <-- nuovo: raccolgo i tag per ogni cella

                        # Grid cells: compute colour per cell and draw as unit squares offset by 0.5 to centre ticks.
                        rendered_cells = 0
                        for i in range(nrows):
                            row_tags = []
                            for j in range(ncols):
                                val = heatmap_matrix[i][j]
                                tooltip = tooltip_matrix[i][j]

                                color = _get_heatmap_cell_color(val, tooltip, float(vmin), float(vmax))

                                y_plot = nrows - i - 1
                                x0, y0 = j, y_plot
                                x1, y1 = j + cell_size, y_plot + cell_size

                                cell_tag = f"hm_cell_{i}_{j}"   # <-- tag univoco per cella
                                dpg.draw_rectangle(
                                    pmin=(x0 + 0.5, y0 + 0.5),
                                    pmax=(x1 + 0.5, y1 + 0.5),
                                    fill=color,
                                    thickness=0.0000000000001,
                                    color=(30, 30, 30, 255),
                                    tag=cell_tag
                                )
                                row_tags.append(cell_tag)
                                rendered_cells += 1
                                if rendered_cells == total_cells or rendered_cells % max(1, total_cells // 30) == 0:
                                    set_loading_screen_progress(state, 60 + (rendered_cells / total_cells) * 12)
                            cell_tags.append(row_tags)

                        # salva ciò che serve per il refresh colori
                        state["rg_heatmap_matrix"] = heatmap_matrix
                        state["rg_tooltip_matrix"] = tooltip_matrix
                        state["rg_cell_tags"] = cell_tags
                        state["rg_vmin_vmax"] = (float(vmin), float(vmax))
                        state["rg_nrows_ncols"] = (int(nrows), int(ncols))


                        # Header images (TOP: R1) occupying 1×1 above each column; mirrored also at the BOTTOM.
                        top_y0 = nrows + 0.6
                        top_y1 = nrows + 1.6
                        for i in range(ncols):
                            tex = f"heatmap_r1_tex_{i}"
                            dpg.draw_image(
                                texture_tag=tex,
                                pmin=(i + 0.5, top_y0),
                                pmax=(i + 1.5, top_y1),
                                uv_min=(0, 1),  # vertical flip to match plot coordinates
                                uv_max=(1, 0)
                            )

                        bottom_y0 = -0.6
                        bottom_y1 = 0.4
                        for i in range(ncols):
                            tex = f"heatmap_r1_tex_{i}"
                            dpg.draw_image(
                                texture_tag=tex,
                                pmin=(i + 0.5, bottom_y0),
                                pmax=(i + 1.5, bottom_y1),
                                uv_min=(0, 1),  # vertical flip to match plot coordinates
                                uv_max=(1, 0)
                            )

                        # Header images (LEFT/RIGHT: R2) mirrored on both sides along rows.
                        left_x0 = -0.6
                        left_x1 = 0.4
                        for j in range(nrows):
                            tex = f"heatmap_r2_tex_{j}"
                            y_plot = nrows - j- 1
                            dpg.draw_image(
                                texture_tag=tex,
                                pmin=(left_x0, y_plot + 0.5),
                                pmax=(left_x1, y_plot + 1.5),
                                uv_min=(0, 1),  # vertical flip to match plot coordinates
                                uv_max=(1, 0)
                            )

                        right_x0 = ncols + 0.6
                        right_x1 = ncols + 1.6
                        for j in range(nrows):
                            tex = f"heatmap_r2_tex_{j}"
                            y_plot = nrows - j - 1
                            dpg.draw_image(
                                texture_tag=tex,
                                pmin=(right_x0, y_plot + 0.5),
                                pmax=(right_x1, y_plot + 1.5),
                                uv_min=(0, 1),  # vertical flip to match plot coordinates
                                uv_max=(1, 0)
                            )


                    # Border around the heat grid only (excluding header bands).
                    border_offset = 1.1  # espansione per includere header bands
                    dpg.add_line_series(
                        [0.5 - border_offset, ncols + 0.5 + border_offset, ncols + 0.5 + border_offset, 0.5 - border_offset, 0.5 - border_offset],
                        [0.5 - border_offset, 0.5 - border_offset, nrows + 0.5 + border_offset, nrows + 0.5 + border_offset, 0.5 - border_offset],
                        parent="heat_y_axis", tag="heatmap_border_line_series"
                    )
                with dpg.theme() as border_line_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, (250, 250, 250, 0), category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme("heatmap_border_line_series", border_line_theme)
                dpg.bind_item_theme("heatmap_plot", apply_boxplot_theme(state))
            

            # --- Refresh colours hook: ricolora le celle della heatmap con la colormap attuale ---
            def _refresh_rgroups_heatmap_colours() -> None:
                """
                Refresh rgroups heatmap colours.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                hm = state.get("rg_heatmap_matrix")
                tt = state.get("rg_tooltip_matrix")
                tags = state.get("rg_cell_tags")
                vmin_vmax = state.get("rg_vmin_vmax")
                dims = state.get("rg_nrows_ncols")

                if hm is None or tt is None or tags is None or vmin_vmax is None or dims is None:
                    return

                vmin, vmax = vmin_vmax
                nrows, ncols = dims

                # evita divisione per zero
                if vmax <= vmin:
                    vmax = vmin + 1.0

                for i in range(nrows):
                    for j in range(ncols):
                        cell_tag = tags[i][j]
                        if not dpg.does_item_exist(cell_tag):
                            continue

                        val = hm[i][j]
                        tooltip = tt[i][j]

                        color = _get_heatmap_cell_color(val, tooltip, float(vmin), float(vmax))

                        try:
                            dpg.configure_item(cell_tag, fill=color)
                        except Exception:
                            pass

            # Expose the hook in state so apply_colormap can call it.
            state["rgroups_refresh_colors"] = _refresh_rgroups_heatmap_colours
            state["rgroups_refresh_ticks"] = _update_heatmap_ticks


            # Interaction handlers: mouse click to select cell, mouse move to show tooltip.
            for tag in (
                "heatmap_click_handler",
                "heatmap_mouse_move_handler",
                "heatmap_wheel_handler",
                "heatmap_drag_handler",
                "heatmap_release_handler",
            ):
                if dpg.does_item_exist(tag):
                    dpg.delete_item(tag)
            dpg.add_mouse_click_handler(tag="heatmap_click_handler", parent="handler_registry", callback=handle_click_heatmap)
            dpg.add_mouse_move_handler(tag="heatmap_mouse_move_handler", parent="handler_registry", callback=handle_hover_heatmap)
            dpg.add_mouse_wheel_handler(
                tag="heatmap_wheel_handler",
                parent="handler_registry",
                callback=lambda s, a, u: _update_heatmap_ticks() if dpg.is_item_hovered("heatmap_plot") else None,
            )
            dpg.add_mouse_drag_handler(
                tag="heatmap_drag_handler",
                parent="handler_registry",
                callback=lambda s, a, u: _update_heatmap_ticks() if dpg.is_item_hovered("heatmap_plot") else None,
            )
            dpg.add_mouse_release_handler(
                tag="heatmap_release_handler",
                parent="handler_registry",
                callback=lambda s, a, u: _update_heatmap_ticks() if dpg.does_item_exist("heatmap_plot") else None,
            )
            dpg.set_frame_callback(dpg.get_frame_count() + 1, _update_heatmap_ticks)
            dpg.set_frame_callback(dpg.get_frame_count() + 2, _update_heatmap_ticks)


            # Allocate dynamic textures for scaffold, R1 and R2 thumbnails.
            empty_data = np.zeros((heatmap_scaffold_render_width * heatmap_scaffold_render_height * 4), dtype=np.float32)

            if not dpg.does_item_exist("heatmap_scaffold_image_texture"):
                dpg.add_dynamic_texture(heatmap_scaffold_render_width, heatmap_scaffold_render_height, empty_data, 
                                        tag="heatmap_scaffold_image_texture", parent="texture_registry")
            else:
                dpg.set_value("heatmap_scaffold_image_texture", empty_data)

            if not dpg.does_item_exist("heatmap_group_r1_image_texture") and not dpg.does_item_exist("heatmap_group_r2_image_texture"):
                dpg.add_dynamic_texture(heatmap_group_render_width, heatmap_group_render_height, empty_data, 
                                        tag="heatmap_group_r1_image_texture", parent="texture_registry")
                dpg.add_dynamic_texture(heatmap_group_render_width, heatmap_group_render_height, empty_data, 
                                        tag="heatmap_group_r2_image_texture", parent="texture_registry")
            else:
                dpg.set_value("heatmap_group_r1_image_texture", empty_data)
                dpg.set_value("heatmap_group_r2_image_texture", empty_data)

            # Side details panel: scaffold image (with caption) and two group thumbnails; export popups for each.
            with dpg.child_window(parent="heatmap_details_window", width=-1, height=-1,
                                  no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
                set_loading_screen_progress(state, 73)
                
                dpg.add_image(
                    "heatmap_scaffold_image_texture",
                    tag="heatmap_scaffold_image_widget",
                    width=heatmap_side_img_width,
                    height=heatmap_side_img_height,
                    border_color=(0, 0, 0, 0),
                )
                with dpg.tooltip("heatmap_scaffold_image_widget", delay=0):
                    dpg.add_text(
                        f"{subset.replace('_', ' ').replace('su', 'Su')} - Common Core",
                        tag="heatmap_scaffold_image_tooltip_text",
                    )
                register_responsive_image(
                    state,
                    image_tag="heatmap_scaffold_image_widget",
                    parent_tag="heatmap_details_window",
                    aspect_ratio=0.75,
                    tab="r_analysis_tab"
                )
                export_png_popup("heatmap_scaffold_image_widget", "heatmap_scaffold_image_texture", state)

                # Draw the subset scaffold using the same logic as overview_decomposition.py.
                subset = dpg.get_value("heatmap_subset_choice")
                try:
                    scaffold_mb = (
                        state["molblocks_rgd_dict"][subset]["mol_1"]["Core"]
                        if "mol_1" in state["smiles_rgd_dict"][subset]
                        else state["smiles_rgd_dict"][subset]["mol_2"]["Core"]
                    )
                    try:
                        scaff = Chem.MolFromMolBlock(scaffold_mb, sanitize=False)
                        if scaff is None or scaff.GetNumAtoms() == 0:
                            raise KeyError()
                    except Exception:
                        try:
                            scaff_smiles = (
                                state["smiles_rgd_dict"][subset]["mol_1"]["Core"]
                                if "mol_1" in state["smiles_rgd_dict"][subset]
                                else state["smiles_rgd_dict"][subset]["mol_2"]["Core"]
                            )
                            scaff = Chem.MolFromSmiles(scaff_smiles, sanitize=False)
                            if scaff is None:
                                raise KeyError()
                        except Exception:
                            scaff_smarts = (
                                state["smiles_rgd_dict"][subset]["mol_1"]["Core"]
                                if "mol_1" in state["smiles_rgd_dict"][subset]
                                else state["smiles_rgd_dict"][subset]["mol_2"]["Core"]
                            )
                            scaff = Chem.MolFromSmarts(scaff_smarts)
                    for atom in scaff.GetAtoms():
                        if atom.GetAtomicNum() == 0 and atom.HasProp("molAtomMapNumber"):
                            idx = atom.GetProp("molAtomMapNumber")
                            atom.SetProp("atomLabel", f"R{idx}")
                    Chem.SanitizeMol(
                        scaff,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                    )
                    Chem.AssignStereochemistry(scaff, force=True, cleanIt=True)
                    drawer = rdMolDraw2D.MolDraw2DCairo(
                        heatmap_scaffold_render_width,
                        heatmap_scaffold_render_height,
                    )
                    opts = drawer.drawOptions()
                    opts.padding = 0.025
                    opts.bondLineWidth = 1
                    opts.minFontSize = 1
                    rdMolDraw2D.PrepareAndDrawMolecule(drawer, scaff)
                    drawer.FinishDrawing()
                    img = pilImage.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
                    arr = np.array(img)
                    arr[..., 3] = 255
                    scaff_img_data = (arr / 255.0).flatten().astype(np.float32)
                    dpg.set_value("heatmap_scaffold_image_texture", scaff_img_data)
                except Exception:
                    log_event("R-Analysis", "Warning: could not render scaffold image for heatmap", indent=1, level="WARNING")

                dpg.add_image(
                    "heatmap_group_r1_image_texture",
                    tag="heatmap_group_r1_image_widget",
                    width=heatmap_side_img_width,
                    height=heatmap_side_img_height,
                    border_color=(0, 0, 0, 0),
                )
                with dpg.tooltip("heatmap_group_r1_image_widget", delay=0):
                    dpg.add_text("", tag="heatmap_group_r1_image_texture_tooltip_text")
                register_responsive_image(
                    state,
                    image_tag="heatmap_group_r1_image_widget",
                    parent_tag="heatmap_details_window",
                    aspect_ratio=0.75,
                    tab="r_analysis_tab"
                )
                export_png_popup("heatmap_group_r1_image_widget", "heatmap_group_r1_image_texture", state)

                dpg.add_image(
                    "heatmap_group_r2_image_texture",
                    tag="heatmap_group_r2_image_widget",
                    width=heatmap_side_img_width,
                    height=heatmap_side_img_height,
                    border_color=(0, 0, 0, 0),
                )
                with dpg.tooltip("heatmap_group_r2_image_widget", delay=0):
                    dpg.add_text("", tag="heatmap_group_r2_image_texture_tooltip_text")
                register_responsive_image(
                    state,
                    image_tag="heatmap_group_r2_image_widget",
                    parent_tag="heatmap_details_window",
                    aspect_ratio=0.75,
                    tab="r_analysis_tab"
                )
                export_png_popup("heatmap_group_r2_image_widget", "heatmap_group_r2_image_texture", state)

                # Instructional text: prompt user to click a matrix cell to list matching molecules.
                dpg.add_text(default_value="Click a cell from the matrix\nto get the list of molecules\ncontaining the selected R-groups.", 
                             tag="heatmap_molecules_containing_selected_rpair_text")
                set_loading_screen_progress(state, 78)
