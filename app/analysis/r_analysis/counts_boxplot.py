"""
===============
plot_boxplot.py
===============

Boxplot visualisation for SAR data.

Creates boxplots of bioactivity distributions across R-groups, scaffolds, or
molecular subsets. Used to identify substituent effects and structural trends
in quantitative SAR analysis.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Draw counts boxplot

import re
import io
import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
import math
from typing import Any
from PIL import Image as pilImage
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from app.utils.app_logger import log_event, log_settings
from app.utils.callbacks import (
    export_png_popup,
    register_plot_context_popup,
    register_responsive_image
)
from app.gui.themes_manager import apply_boxplot_theme
from app.gui.loading_win import draw_loading_screen


# -----------------------------------------------------------------------------
# 1.1. Build counts boxplot popup-specific items
# -----------------------------------------------------------------------------
def _build_counts_boxplot_popup_specific_items(state: dict[str, Any]) -> None:
    """
    Add boxplot-specific context popup options and trigger a redraw when changed.
    """
    context_settings = state.setdefault("plot_context_settings", {}).setdefault("counts_boxplot_plot_context", {})
    context_settings.setdefault("show_compound_images", True)
    context_settings.setdefault("hide_non_outliers", True)
    context_settings.setdefault("use_arrow_markers", False)

    def _set_option(option_key: str, value: Any) -> None:
        context_settings[option_key] = bool(value)
        if option_key == "hide_non_outliers" and bool(value):
            context_settings["use_arrow_markers"] = False
        redraw = state.get("counts_boxplot_redraw")
        if callable(redraw):
            redraw()

    dpg.add_checkbox(
        label="Show outliers only",
        default_value=bool(context_settings.get("hide_non_outliers", False)),
        callback=lambda s, a: _set_option("hide_non_outliers", a),
    )
    dpg.add_checkbox(
        label="Use arrow markers",
        default_value=bool(context_settings.get("use_arrow_markers", True)),
        callback=lambda s, a: _set_option("use_arrow_markers", a),
    )
    dpg.add_checkbox(
        label="Show R images",
        default_value=bool(context_settings.get("show_compound_images", True)),
        callback=lambda s, a: _set_option("show_compound_images", a),
    )


# -----------------------------------------------------------------------------
# 2. Draw counts boxplot
# -----------------------------------------------------------------------------
def draw_counts_boxplot(
    r: Any,
    activity: str,
    read_undefined: bool,
    read_inactives: bool,
    min_count: Any,
    data: Any,
    total_molecules: Any,
    state: dict[str, Any]
) -> Any:
    """
    Draw a customised boxplot of a given activity vs R-group using DearPyGui.

    Args:
        subset (str): The identifier of the current subset (used for labelling).
        activity (str): The selected activity column.
        r (str): The selected R-group column.
        data (DataFrame): The dataset containing activity and R-group data.
        read_undefined (bool): Whether to include undefined qualifier values (e.g., '< 10', '> 5').
        read_inactives (bool): Whether to include inactive rows (assigned default low/high values).
        state (dict): Application state dictionary.

    Returns:
        None
    """
    log_event("R-Analysis", "Rendering counts boxplot", indent=1)
    log_settings("R-Analysis", indent=2, rgroup=r, activity=activity, include_undefined=read_undefined, include_inactives=read_inactives, min_count=min_count, molecules=total_molecules)
    draw_loading_screen(state, bg=False)
    plot_context_settings = state.setdefault("plot_context_settings", {}).setdefault("counts_boxplot_plot_context", {})
    hide_non_outliers = bool(plot_context_settings.get("hide_non_outliers", True))
    use_arrow_markers = bool(plot_context_settings.get("use_arrow_markers", False))
    show_compound_images = bool(plot_context_settings.setdefault("show_compound_images", True))

    # -----------------------------------------------------------------------------
    # 2.0. Get box color from plot colormap
    # -----------------------------------------------------------------------------
    def _get_discrete_box_palette() -> list[tuple[int, int, int, int]]:
        """
        Build a compact palette of distinct colors from the active discrete colormap.

        Returns:
            list[tuple[int, int, int, int]]: Ordered RGBA colors sampled from the colormap.
        """
        cache_key = ("counts_boxplot_palette", state.get("colormap_discrete"))
        cached = state.get("_counts_boxplot_palette_cache")
        if isinstance(cached, dict) and cache_key in cached:
            return cached[cache_key]

        palette: list[tuple[int, int, int, int]] = []
        try:
            colormap = state["plot_colormaps"][state["colormap_discrete"]]
            last_color = None
            for step in range(256):
                t = step / 255 if step else 0.0
                color = dpg.sample_colormap(colormap, t)
                if len(color) >= 4:
                    if max(color[0], color[1], color[2]) <= 1.0:
                        rgba = (
                            int(round(color[0] * 255)),
                            int(round(color[1] * 255)),
                            int(round(color[2] * 255)),
                            200,
                        )
                    else:
                        rgba = (int(color[0]), int(color[1]), int(color[2]), 200)
                    if rgba != last_color:
                        palette.append(rgba)
                        last_color = rgba
        except Exception:
            palette = []

        if not palette:
            palette = [
                (0, 82, 165, 200),
                (227, 114, 34, 200),
                (53, 161, 107, 200),
                (196, 61, 61, 200),
                (137, 99, 186, 200),
                (197, 170, 45, 200),
            ]

        if not isinstance(cached, dict):
            cached = {}
            state["_counts_boxplot_palette_cache"] = cached
        cached[cache_key] = palette
        return palette

    def _get_box_color(group_index: int, total_groups: int) -> tuple[int, int, int, int]:
        """
        Pick a box color so consecutive boxes differ even with discrete colormaps.

        Args:
            group_index (int): Zero-based position of the current box.
            total_groups (int): Total number of visible groups.

        Returns:
            tuple[int, int, int, int]: RGBA box fill color.
        """
        palette = _get_discrete_box_palette()
        if not palette:
            return (0, 82, 165, 200)

        palette_len = len(palette)
        if palette_len == 1:
            return palette[0]

        if total_groups <= palette_len:
            idx = int(round(group_index * (palette_len - 1) / max(1, total_groups - 1)))
            return palette[idx]

        stride = max(2, (palette_len // 2) + 1)
        while math.gcd(stride, palette_len) != 1:
            stride += 1
        return palette[(group_index * stride) % palette_len]
    # -----------------------------------------------------------------------------
    # 2.1. Ensure group image texture
    # -----------------------------------------------------------------------------
    def _ensure_group_image_texture(tex_tag: str, smiles: str, px_size: int = 128) -> None:
        """
        Create or update a DPG dynamic texture with the 2D drawing of the fragment 'smiles'.
        
        Args:
            tex_tag (Any): Parameter accepted by this routine.
            smiles (str): Parameter accepted by this routine.
            px_size (Any): Parameter accepted by this routine. Defaults to the configured value.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        mol = None
        if smiles:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                try:
                    # Sanitize while skipping kekulisation to tolerate aromatic edge cases on transparent bg.
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                            Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Exception:
                    mol = None

        if mol is None:
            # Build a fully transparent placeholder when the SMILES cannot be parsed/sanitised.
            img_arr = (np.zeros((px_size, px_size, 4), dtype=np.float32)).flatten()
        else:
            # Render the molecule to a Cairo PNG and convert to RGBA float32 array in [0, 1].
            drawer = rdMolDraw2D.MolDraw2DCairo(px_size, px_size)
            opts = drawer.drawOptions()
            opts.clearBackground = False          # ← keep transparent background (no fill) to preserve alpha
            opts.padding = 0.08
            opts.bondLineWidth = 1
            opts.minFontSize = 1
            # Optional: darker colours on transparent background
            # opts.atomColourPalette[6] = (0.1, 0.1, 0.1)  # almost black carbon
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            png_bytes = drawer.GetDrawingText()
            pil_img = pilImage.open(io.BytesIO(png_bytes)).convert("RGBA")
            img_arr = (np.array(pil_img) / 255.0).astype(np.float32).flatten()

        if not dpg.does_item_exist(tex_tag):
            # Create the texture if it does not exist yet.
            dpg.add_dynamic_texture(px_size, px_size, img_arr, tag=tex_tag, parent="texture_registry")
        else:
            # Update existing texture content.
            dpg.set_value(tex_tag, img_arr)


    # Prepare a clean list of (R-group, numeric activity) pairs honouring undefined/inactive settings.

    # --- STEP 1.1: Extract and filter valid numerical values ---
    # Parse qualifiers (<=, >=, <, >) and read numeric parts; control inclusion via 'read_undefined' and 'read_inactives'.
    valid_rows = []
    for _, row in data.iterrows():
        value = str(row.get(activity, "")).strip()
        match = re.match(r"^(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)", value)  # Read values like '> 10', '<= 5.5', '3.2', etc.

        if match:  # Non-empty, numeric-compatible value
            symbol = match.group(1)
            number = match.group(2)

            # Skip undefined-qualifier values if the user opted out (e.g., '> 10', '<= 3.0').
            if symbol and not read_undefined:
                continue

            num_val = float(number)
            valid_rows.append((row[r], num_val))

        else:  # Empty or unparsable value
            if read_inactives:
                # Assign default low/high numeric value for inactives depending on activity units.
                num_val = 1e9 if activity in state["nM_activity_types"] else 0
                valid_rows.append((row[r], num_val))
            # Otherwise, silently skip the row.
            
    # --- STEP 1.2: Create DataFrame from valid entries ---
    # Build a compact DataFrame with just the R-group and activity columns for plotting.
    df = pd.DataFrame(valid_rows, columns=[r, activity])

    # --- STEP 1.3: Convert to pActivity if applicable ---
    # Convert nM activities to the pX scale: pX = -log10(X [M]).
    if activity in state["nM_activity_types"]:
        df[activity] = -np.log10(df[activity] * 1e-9)
        activity_label = f"p{activity}"
    else:
        activity_label = activity


    # --- STEP 1.4: Filter and sort groups by mean activity / frequency + min_count ---
    # Filter out groups with no actives and those below 'min_count', then sort by mean activity or frequency.
    valid_groups = df.groupby(r).filter(lambda g: (g[activity] > 0).any())
    group_counts = valid_groups.groupby(r)[activity].count()

    if state.get("counts_sort_mode", "frequency") == "activity":
            # Ordina per mean(activity) decrescente
        group_means = valid_groups.groupby(r)[activity].mean()
        group_means = group_means[group_counts.loc[group_means.index] >= min_count]
        idx_map = state.get("counts_index_map", {})
        group_means = group_means.sort_values(
            key=lambda s: s.astype(float),
            ascending=False
        )
        # tie-break: sort ties according to the table index
        group_means = group_means.sort_index(
            key=lambda idx: [idx_map[g] for g in idx]
        )
    elif state["counts_sort_mode"] == "frequency":
        # Sort by frequency descending
        idx_map = state.get("counts_index_map", {})

        sorted_counts = group_counts[group_counts >= min_count].sort_values(ascending=False)

        sorted_counts = sorted_counts.sort_index(
            key=lambda idx: [idx_map[g] for g in idx]
        )

        group_means = sorted_counts

    else:
        # fallback: as 'activity'
        group_means = valid_groups.groupby(r)[activity].mean()
        group_means = group_means[group_counts.loc[group_means.index] >= min_count]
        idx_map = state.get("counts_index_map", {})

        group_means = group_means.sort_values(
            key=lambda s: s.astype(float),
            ascending=False
        )

        # tie-break: sort ties according to the table index
        group_means = group_means.sort_index(
            key=lambda idx: [idx_map[g] for g in idx]
        )

    # Final list of groups in the chosen order
    groups = group_means.index.tolist()
    num_groups = len(groups)

    # Compressed X positions (1..N) for drawing the boxes
    compressed_positions = {g: i + 1 for i, g in enumerate(groups)}
    x_positions = compressed_positions  # if the rest of the code still uses x_positions


    # Build the container window and the horizontal layout with the plot on the left and details on the right.

    # --- STEP 2.1: Horizontal group with plot and detail panel ---
    with dpg.child_window(parent="counts_boxplot_window", 
                          no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False,
                          width=-1, height=-1):

        plot_width = state["plots_boxplot_width"]
        plot_height = state["plots_boxplot_height"]

        # Create the DearPyGui plot, define axes, ticks and palette, then draw boxes, whiskers and points.
        with dpg.plot(tag="counts_boxplot",
                        width=-1, height=-1, 
                        no_menus=True, no_mouse_pos=False, zoom_rate=0.05, crosshairs=True):


            # --- STEP 3.1: Define axes ---
            # Build X as nominal positions 1..N and Y as the activity axis (lock min at 0 for inactives).
            lock_min = True if read_inactives else False
            dpg.add_plot_axis(dpg.mvXAxis, label=r, tag="counts_boxplot_x_axis",
                                no_highlight=True, no_gridlines=True, no_tick_marks=True, tick_format='%.0f')         
            dpg.add_plot_axis(dpg.mvYAxis, label=activity_label, tag="counts_boxplot_y_axis",
                                no_highlight=True, lock_min=lock_min)


            x_real_values = sorted(x_positions.values())   # es. [1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21]

            n = len(x_real_values)
            if n <= 75:
                step = 1
            elif n <= 150:
                step = 2
            elif n <= 300:
                step = 5
            else:
                step = 10

            # counts_index_map contains {SMILES -> table row index}
            idx_map = state.get("counts_index_map", {})
            x_tick_labels = [str(idx_map.get(g, compressed_positions[g])) for g in groups]

            def _nice_tick_step(visible_count: int, max_visible_ticks: int) -> int:
                """
                Pick a readable 1-2-5 style tick step for the X axis.

                Args:
                    visible_count (int): Number of visible groups.
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

            def _refresh_x_ticks() -> None:
                """
                Update custom X-axis ticks based on the current zoomed range.

                Args:
                    None.

                Returns:
                    None: This routine updates plot ticks in place.
                """
                if not (dpg.does_item_exist("counts_boxplot") and dpg.does_item_exist("counts_boxplot_x_axis")):
                    return

                try:
                    x0, x1 = dpg.get_axis_limits("counts_boxplot_x_axis")
                except Exception:
                    return

                try:
                    plot_w, _ = dpg.get_item_rect_size("counts_boxplot")
                except Exception:
                    plot_w = 0

                plot_w = max(1, int(plot_w or 0))
                vis_x0 = max(0.5, min(n + 0.5, float(x0)))
                vis_x1 = max(0.5, min(n + 0.5, float(x1)))
                if vis_x1 < vis_x0:
                    vis_x0, vis_x1 = vis_x1, vis_x0

                first_idx = max(0, int(np.ceil(vis_x0 - 1.0)))
                last_idx = min(n - 1, int(np.floor(vis_x1 - 1.0)))
                visible_count = max(0, last_idx - first_idx + 1)
                max_ticks = max(1, plot_w // 42)
                step = _nice_tick_step(visible_count, max_ticks)

                visible_indices = list(range(first_idx, last_idx + 1, step)) if visible_count else []

                ticks = tuple(
                    (x_tick_labels[i], compressed_positions[groups[i]])
                    for i in visible_indices
                )
                dpg.set_axis_ticks("counts_boxplot_x_axis", ticks)

            _refresh_x_ticks()


                
            whisker_lowers = []
            whisker_uppers = []

            all_x = []
            all_y = []

            box_fill_tags = []

            # --- STEP 3.4: Draw individual boxplots ---
            # Compute quartiles/whiskers and render scatter points and box glyphs for each group.
            for i, g in enumerate(groups):
                group_data = df[df[r] == g][activity].values
                if len(group_data) == 0:
                    continue

                q1 = np.percentile(group_data, 25)
                median = np.percentile(group_data, 50)
                mean_val = np.mean(group_data)
                q3 = np.percentile(group_data, 75)
                iqr = q3 - q1
                lower = max(min(group_data), q1 - 1.5 * iqr)
                upper = min(max(group_data), q3 + 1.5 * iqr)
                x = compressed_positions[g]
                box_color = _get_box_color(i, num_groups)

                whisker_lowers.append(lower)
                whisker_uppers.append(upper)
            
                all_x.append(x)
                all_y.extend([lower, upper])

                # draw one point at a time with different markers
                for y in group_data:
                    is_outlier = bool(y < lower or y > upper)
                    if hide_non_outliers and not is_outlier:
                        continue

                    if hide_non_outliers:
                        marker = dpg.mvPlotMarker_Circle
                        marker_size = 3.0
                        marker_outline = (0, 0, 0, 255)
                        marker_color = (0, 0, 0, 0)
                    elif use_arrow_markers:
                        if y > mean_val:
                            marker = dpg.mvPlotMarker_Up
                            marker_size = 3.0
                            marker_outline = (0, 0, 0, 0)
                            marker_color = box_color
                        elif y < mean_val:
                            marker = dpg.mvPlotMarker_Down
                            marker_size = 3.0
                            marker_outline = (0, 0, 0, 0)
                            marker_color = box_color
                        else:
                            marker = dpg.mvPlotMarker_Circle
                            marker_size = 2.0
                            marker_outline = (60, 60, 60, 255)
                            marker_color = (255, 255, 0, 255)  # yellow for the mean
                    else:
                        marker = dpg.mvPlotMarker_Circle
                        marker_size = 3.0 if is_outlier else 2.5
                        marker_outline = (0, 0, 0, 0) if is_outlier else (60, 60, 60, 120)
                        marker_color = box_color if not np.isclose(y, mean_val) else (255, 255, 0, 255)
                    point_x = x if hide_non_outliers else (x - 0.3)
                    pt_tag = dpg.add_scatter_series(
                        [point_x],
                        [y],
                        parent="counts_boxplot_y_axis",
                        label=""
                    )

                    with dpg.theme() as pt_theme:
                        with dpg.theme_component(dpg.mvScatterSeries):
                            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, marker, category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, marker_size, category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, marker_outline, category=dpg.mvThemeCat_Plots)
                            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, marker_color, category=dpg.mvThemeCat_Plots)

                    dpg.bind_item_theme(pt_tag, pt_theme)
                
                line_color = (55, 55, 55, 255)

                # --- STEP 3.5: Draw box elements ---
                # Render the filled rectangle (Q1–Q3), the median line, whiskers and end caps.
                box_width = 0.2
                cap_width = 0.2
                line_tickness = 0

                box_rect_tag = f"counts_boxplot_box_rect_{i}"
                dpg.draw_rectangle(
                    (x - box_width, q1),
                    (x + box_width, q3),
                    color=line_color,
                    fill=box_color,
                    parent="counts_boxplot",
                    thickness=line_tickness,
                    tag=box_rect_tag,
                )
                box_fill_tags.append(box_rect_tag)

                dpg.draw_line((x - box_width, median), (x + box_width, median), color=line_color, thickness=line_tickness, parent="counts_boxplot")
                dpg.draw_line((x, q1), (x, lower), color=line_color, thickness=line_tickness, parent="counts_boxplot")
                dpg.draw_line((x, q3), (x, upper), color=line_color, thickness=line_tickness, parent="counts_boxplot")
                dpg.draw_line((x - cap_width, lower), (x + cap_width, lower), color=line_color, thickness=line_tickness, parent="counts_boxplot")
                dpg.draw_line((x - cap_width, upper), (x + cap_width, upper), color=line_color, thickness=line_tickness, parent="counts_boxplot")

            # Draw box outlines as a line series to enable double-click selection behaviour on plot items.                
            for i, g in enumerate(groups):
                group_data = df[df[r] == g][activity].values
                if len(group_data) == 0:
                    continue

                q1 = np.percentile(group_data, 25)
                q3 = np.percentile(group_data, 75)
                x = x_positions[g]

                # Outline path (bottom → right → top → left → close) as a single line series.
                outline_x = [
                    x - box_width, x + box_width,  # bottom edge
                    x + box_width, x + box_width,  # right edge
                    x + box_width, x - box_width,  # top edge
                    x - box_width, x - box_width   # left edge (close)
                ]
                outline_y = [
                    q1, q1,        # bottom edge
                    q1, q3,        # right edge
                    q3, q3,        # top edge
                    q3, q1         # left edge (close)
                ]

            # Global outline around all boxes for visual framing
            if groups and whisker_lowers and whisker_uppers:
                x_left  = x_positions[groups[0]]  - 0.5
                x_right = x_positions[groups[-1]] + 0.5

                global_lower = float(np.min(whisker_lowers))
                global_upper = float(np.max(whisker_uppers))
                y_bottom = global_lower - 2.0
                y_top    = global_upper + 2.0

                rect_x = [x_left, x_right, x_right, x_left, x_left]
                rect_y = [y_bottom, y_bottom, y_top, y_top, y_bottom]

                if dpg.does_item_exist("counts_boxplot_frame_outline"):
                    dpg.delete_item("counts_boxplot_frame_outline")

                dpg.add_line_series(
                    rect_x, rect_y,
                    parent="counts_boxplot_y_axis",
                    tag="counts_boxplot_frame_outline"
                )
                with dpg.theme() as boxplot_frame_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 0), category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme("counts_boxplot_frame_outline", boxplot_frame_theme)


            # SAFETY CHECK — no valid points were produced (filters removed everything)
            if not all_x or not all_y:
                dpg.add_text(
                    "No valid groups to display with the current filters.",
                    parent="counts_boxplot_y_axis"
                )
                return

            x_min = min(all_x)
            x_max = max(all_x)
            y_min = min(all_y)
            y_max = max(all_y)
            y_margin = (y_max - y_min) * 0.10 if y_max > y_min else 1.0
            
            # Base limits (data only, no thumbnails considered yet).
            base_x0 = x_min - 0.4
            base_x1 = x_max + 0.3
            if read_inactives:
                base_y0 = 0.0
                base_y1 = y_max + y_margin
            else:
                base_y0 = y_min - y_margin
                base_y1 = y_max + y_margin

            dpg.bind_colormap("counts_boxplot", state["plot_colormaps"][state["colormap_discrete"]])
            dpg.bind_item_theme("counts_boxplot", apply_boxplot_theme(state))
            register_plot_context_popup(
                state,
                context_key="counts_boxplot_plot_context",
                plot_tag="counts_boxplot",
                x_axis_tag="counts_boxplot_x_axis",
                y_axis_tag="counts_boxplot_y_axis",
                theme_kind="boxplot",
                specific_builder=lambda: _build_counts_boxplot_popup_specific_items(state),
            )

            # Width = 2*box_width (box border). Height = width * 3/4 in Y units of the plot.

            def _aspect_height_4_3(width_x_data: Any, x0: Any, x1: Any, y0: Any, y1: Any) -> Any:
                """
                Execute the aspect height 4 3 routine.
                
                Args:
                    width_x_data (Any): Input accepted by this routine.
                    x0 (Any): Input accepted by this routine.
                    x1 (Any): Input accepted by this routine.
                    y0 (Any): Input accepted by this routine.
                    y1 (Any): Input accepted by this routine.
                
                Returns:
                    Any: Value returned by the routine.
                """
                x_span = max(1e-12, (x1 - x0))
                y_span = max(1e-12, (y1 - y0))
                # convert width in X-units to height in Y-units maintaining 4:3 on screen
                return width_x_data * (plot_width / float(plot_height)) * (y_span / float(x_span)) * (4/3)

            def _width_x_to_pixels(width_x_data: float, x0: float, x1: float) -> int:
                """
                Execute the width x to pixels routine.
                
                Args:
                    width_x_data (float): Input accepted by this routine.
                    x0 (float): Input accepted by this routine.
                    x1 (float): Input accepted by this routine.
                
                Returns:
                    int: Value returned by the routine.
                """
                x_span = max(1e-12, (x1 - x0))
                return int(round((width_x_data / x_span) * plot_width))

            state.setdefault("counts_box_img_tex_size", {})

            global_mean = df[activity].mean()
            gap_y_frac = 0.02
            thumb_half_width = 0.47  # almost the whole slot [x-0.5, x+0.5], with a tiny gap between adjacent images
            side_x = 2 * thumb_half_width

            # --- First pass: calculate necessary Y expansion (base limits) ---
            expand_y_min = base_y0
            expand_y_max = base_y1
            img_records = []  # [{"i", "g", "x", "lower", "upper", "median"}...]

            for i, g in enumerate(groups):
                grp_vals = df[df[r] == g][activity].values
                if len(grp_vals) == 0:
                    continue

                q1 = np.percentile(grp_vals, 25)
                median = np.percentile(grp_vals, 50)
                q3 = np.percentile(grp_vals, 75)
                iqr = q3 - q1
                lower = max(np.min(grp_vals), q1 - 1.5 * iqr)
                upper = min(np.max(grp_vals), q3 + 1.5 * iqr)
                x = x_positions[g]

                side_y_base = _aspect_height_4_3(side_x, base_x0, base_x1, base_y0, base_y1)
                gap_y  = (base_y1 - base_y0) * gap_y_frac

                if median > global_mean:
                    y_top    = lower - gap_y
                    y_bottom = y_top - side_y_base
                else:
                    y_bottom = upper + gap_y
                    y_top    = y_bottom + side_y_base

                expand_y_min = min(expand_y_min, y_bottom)
                expand_y_max = max(expand_y_max, y_top)

                img_records.append({
                    "i": i, "g": g, "x": x,
                    "lower": lower, "upper": upper, "median": median
                })

            # --- FINAL limits including thumbnails ---
            final_x0, final_x1 = base_x0, base_x1
            final_y0, final_y1 = expand_y_min, expand_y_max

            # --- Second pass: create/update image_series with 4:3 and FINAL limits ---
            created_series_tags = []
            if show_compound_images:
                for rec in img_records:
                    i = rec["i"]; g = rec["g"]; x = rec["x"]
                    lower = rec["lower"]; upper = rec["upper"]; median = rec["median"]

                    tex_tag    = f"counts_box_img_tex_{r}_{i}"
                    series_tag = f"counts_box_img_series_{r}_{i}"
                    created_series_tags.append(series_tag)

                    side_y = _aspect_height_4_3(side_x, final_x0, final_x1, final_y0, final_y1)
                    gap_y  = (final_y1 - final_y0) * gap_y_frac

                    x0 = x - thumb_half_width
                    x1 = x + thumb_half_width

                    if median > global_mean:
                        y_top    = lower - gap_y
                        y_bottom = y_top - side_y
                    else:
                        y_bottom = upper + gap_y
                        y_top    = y_bottom + side_y

                    bounds_min = (x0, min(y_bottom, y_top))
                    bounds_max = (x1, max(y_bottom, y_top))

                    width_px = _width_x_to_pixels(side_x, final_x0, final_x1)

                    def _px_bounds_for_num_groups(n: int) -> Any:
                        """
                        Execute the px bounds for num groups routine.
                        
                        Args:
                            n (int): Input accepted by this routine.
                        
                        Returns:
                            Any: Value returned by the routine.
                        """
                        if n <= 8:    return (192, 384)
                        if n <= 20:   return (144, 256)
                        if n <= 40:   return (112, 192)
                        if n <= 80:   return (96, 144)
                        if n <= 150:  return (80, 128)
                        return (64, 112)

                    min_px, max_px = _px_bounds_for_num_groups(num_groups)
                    px_size = int(np.clip(width_px, min_px, max_px))

                    prev = state["counts_box_img_tex_size"].get(tex_tag)
                    if prev != px_size and dpg.does_item_exist(tex_tag):
                        dpg.delete_item(tex_tag)
                    state["counts_box_img_tex_size"][tex_tag] = px_size
                    _ensure_group_image_texture(tex_tag, str(g), px_size=px_size)

                    if not dpg.does_item_exist(series_tag):
                        dpg.add_image_series(tex_tag, bounds_min=bounds_min, bounds_max=bounds_max,
                                             parent="counts_boxplot_y_axis", tag=series_tag)
                    else:
                        dpg.configure_item(series_tag, bounds_min=bounds_min, bounds_max=bounds_max)
            else:
                for rec in img_records:
                    series_tag = f"counts_box_img_series_{r}_{rec['i']}"
                    if dpg.does_item_exist(series_tag):
                        dpg.delete_item(series_tag)

            # --- Reflow on zoom/pan/double-click to maintain 4:3 aspect ratio in real time ---

            def _reflow_thumbnails() -> None:
                """
                Execute the reflow thumbnails routine.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                if state["current_r_analysis_subtab"] != "r_analysis_counts_subtab":
                    return

                _refresh_x_ticks()
                
                x0, x1 = dpg.get_axis_limits("counts_boxplot_x_axis")
                y0, y1 = dpg.get_axis_limits("counts_boxplot_y_axis")

                if not show_compound_images:
                    return

                for rec in img_records:
                    i = rec["i"]; g = rec["g"]; x = rec["x"]
                    lower = rec["lower"]; upper = rec["upper"]; median = rec["median"]

                    series_tag = f"counts_box_img_series_{r}_{i}"
                    if not dpg.does_item_exist(series_tag):
                        continue

                    side_y = _aspect_height_4_3(side_x, x0, x1, y0, y1)
                    gap_y  = (y1 - y0) * gap_y_frac

                    x_min = x - thumb_half_width
                    x_max = x + thumb_half_width

                    if median > global_mean:
                        y_top    = lower - gap_y
                        y_bottom = y_top - side_y
                    else:
                        y_bottom = upper + gap_y
                        y_top    = y_bottom + side_y

                    dpg.configure_item(
                        series_tag,
                        bounds_min=(x_min, min(y_bottom, y_top)),
                        bounds_max=(x_max, max(y_bottom, y_top))
                    )


            if dpg.does_item_exist("counts_boxplot_wheel_handler"):
                dpg.delete_item("counts_boxplot_wheel_handler")
            dpg.add_mouse_wheel_handler(
                parent="handler_registry",
                callback=lambda s, a: (_reflow_thumbnails() if dpg.is_item_hovered("counts_boxplot") else None),
                tag="counts_boxplot_wheel_handler"
            )

            if dpg.does_item_exist("counts_boxplot_move_handler"):
                dpg.delete_item("counts_boxplot_move_handler")
            dpg.add_mouse_move_handler(
                parent="handler_registry",
                callback=lambda s, a: (_reflow_thumbnails() if dpg.is_item_hovered("counts_boxplot") else None),
                tag="counts_boxplot_move_handler"
            )          

            if dpg.does_item_exist("counts_boxplot_left_click_handler"):
                dpg.delete_item("counts_boxplot_left_click_handler")
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Left,
                parent="handler_registry",
                callback=lambda s, a: (_reflow_thumbnails() if dpg.is_item_hovered("counts_boxplot") else None),
                tag="counts_boxplot_left_click_handler"
            )            

            if dpg.does_item_exist("counts_boxplot_left_drag_handler"):
                dpg.delete_item("counts_boxplot_left_drag_handler")
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                parent="handler_registry",
                callback=lambda s, a: (_reflow_thumbnails() if dpg.is_item_hovered("counts_boxplot") else None),
                tag="counts_boxplot_left_drag_handler"
            )

            if dpg.does_item_exist("counts_boxplot_right_click_handler"):
                dpg.delete_item("counts_boxplot_right_click_handler")
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right,
                parent="handler_registry",
                callback=lambda s, a: (_reflow_thumbnails() if dpg.is_item_hovered("counts_boxplot") else None),
                tag="counts_boxplot_right_click_handler"
            )                       

            if dpg.does_item_exist("counts_boxplot_right_drag_handler"):
                dpg.delete_item("counts_boxplot_right_drag_handler")
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Right,
                parent="handler_registry",
                callback=lambda s, a: (_reflow_thumbnails() if dpg.is_item_hovered("counts_boxplot") else None),
                tag="counts_boxplot_right_drag_handler"
            )

            _reflow_thumbnails()


    # Allocate (once) the RGBA dynamic texture used to display the selected R-group in the details panel.
    img_size = state["counts_table_img_size"] - state["win_spacer"] * 2
    
    empty_data = np.zeros((img_size * img_size * 4,), dtype=np.float32)
        
    if not dpg.does_item_exist("counts_boxplot_group_r_image_texture"):
        dpg.add_dynamic_texture(img_size, img_size, empty_data, 
                                tag="counts_boxplot_group_r_image_texture", parent="texture_registry")
    else:
        dpg.set_value("counts_boxplot_group_r_image_texture", empty_data)

    # Build the right-hand panel with statistics, current R-group image and PNG export popup.
    with dpg.child_window(parent="counts_boxplot_details_window", width=-1, height=-1,
                            no_scrollbar=True, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
        
        idx_map = state.get("counts_index_map", {})
        max_real_id = max(idx_map[g] for g in groups) if groups else 0
        dpg.add_text(f"Group ID: -/{max_real_id}", tag="counts_boxplot_detail_Box")        
        dpg.add_text(f"Counts (with duplicates): -", tag="counts_boxplot_detail_Counts")
        dpg.add_text("With activity: -", tag="counts_boxplot_detail_Actives")
        dpg.add_text("NO activity: -", tag="counts_boxplot_detail_Inactives")
        dpg.add_image("counts_boxplot_group_r_image_texture", tag="counts_boxplot_group_r_image_widget", border_color=(0, 0, 0, 0))
        register_responsive_image(
            state,
            image_tag="counts_boxplot_group_r_image_widget",
            parent_tag="counts_boxplot_details_window",
            aspect_ratio=1.0,
            tab="r_analysis_tab",
        )
        export_png_popup("counts_boxplot_group_r_image_widget", "counts_boxplot_group_r_image_texture", state)


    # Local utilities for rendering a molecule image and for updating the panel based on mouse position.

    # -----------------------------------------------------------------------------
    # 2.2. Update image for group
    # -----------------------------------------------------------------------------
    def update_image_for_group(texture_tag: str, smiles: str) -> None:
        """
        Update the dynamic image texture with a molecule corresponding to the given SMILES.

        Args:
            texture_tag (str): The tag of the DearPyGui texture to update.
            smiles (str): The SMILES string representing the molecule.

        Returns:
            None
        """
        
        if smiles != "":
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            try:
                # Sanitize while skipping kekulisation to better handle aromatic drawings for thumbnails.
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except:
                return

            # Render the molecule to an RGBA PNG using RDKit's 2D drawer and normalise to [0, 1] float32.
            drawer = rdMolDraw2D.MolDraw2DCairo(img_size, img_size)
            opts = drawer.drawOptions()
            opts.padding = 0.025
            opts.bondLineWidth = 1
            opts.minFontSize = 1
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            png_bytes = drawer.GetDrawingText()
            mol_img = pilImage.open(io.BytesIO(png_bytes)).convert("RGBA")
            mol_arr = (np.array(mol_img) / 255.0).astype(np.float32).flatten()

            # Push the new image data into the target texture so the widget reflects the current selection.
            dpg.set_value(texture_tag, mol_arr)

        else:
            # Provide a transparent placeholder when no SMILES is available.
            empty_data = np.zeros((img_size * img_size * 4,), dtype=np.float32)
            dpg.set_value(texture_tag, empty_data)

    # -----------------------------------------------------------------------------
    # 2.3. Update details for group
    # -----------------------------------------------------------------------------
    def update_details_for_group(group_smiles: str) -> None:
        """
        Refresh the boxplot details panel for a specific R-group.

        Args:
            group_smiles (str): SMILES string identifying the selected R-group.

        Returns:
            None: This routine updates the details widgets in place.
        """
        if not group_smiles:
            return

        if not dpg.does_item_exist("counts_boxplot_detail_Box"):
            return

        group_data = df[df[r] == group_smiles][activity].values
        if len(group_data) == 0:
            return

        inactive_rows = len(df[(df[r] == group_smiles) & (df[activity] == 0)])
        idx_map = state.get("counts_index_map", {})
        real_id = idx_map.get(group_smiles, 0)
        max_real_id = max(idx_map[g] for g in groups) if groups else real_id

        dpg.set_value("counts_boxplot_detail_Box", f"Group ID: {real_id}/{max_real_id}")
        dpg.set_value("counts_boxplot_detail_Counts", f"Counts (with duplicates): {len(group_data)}\n")
        dpg.set_value("counts_boxplot_detail_Actives", f"With activity: {len(group_data) - inactive_rows}")
        dpg.set_value("counts_boxplot_detail_Inactives", f"NO activity: {inactive_rows}")
        update_image_for_group("counts_boxplot_group_r_image_texture", group_smiles)

    state["update_counts_boxplot_details_for_group"] = update_details_for_group
    state["counts_boxplot_box_fill_tags"] = list(box_fill_tags)

    def _refresh_counts_boxplot_colors() -> None:
        """
        Recolor existing box rectangles using the currently applied plot colormap.

        Args:
            None.

        Returns:
            None: This routine updates the existing draw items in place.
        """
        tags = state.get("counts_boxplot_box_fill_tags", [])
        total_tags = len(tags)
        if total_tags == 0:
            return

        for i, tag in enumerate(tags):
            if not dpg.does_item_exist(tag):
                continue
            box_color = _get_box_color(i, total_tags)
            dpg.configure_item(tag, fill=box_color)

    state["counts_boxplot_refresh_colors"] = _refresh_counts_boxplot_colors

    # -----------------------------------------------------------------------------
    # 2.4. Poll mouse and update
    # -----------------------------------------------------------------------------
    def poll_mouse_and_update() -> None:
        """
        Monitor the current mouse position on the plot and update the details panel
        with statistics and the image of the closest hovered R-group.

        Args:
            None

        Returns:
            None
        """
        if state["current_r_analysis_subtab"] != "r_analysis_counts_subtab":
            return

        mouse_x, _ = dpg.get_plot_mouse_pos()
        margin = 0.2
        closest_group = None
        closest_distance = float('inf')
        smiles = ""
        
        # Determine the group whose x position is closest to the mouse (within a small margin).
        for i, g in enumerate(groups):
            x = x_positions[g]
            group_data = df[df[r] == g][activity].values
            if len(group_data) == 0:
                continue
            if (x - margin) <= mouse_x <= (x + margin):
                distance = abs(mouse_x - x)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_group = g
                    group_id = i + 1

        # When a group is in range, compute descriptive statistics and refresh the panel widgets.
        if closest_group is not None:
            g = closest_group
            smiles_row = df[df[r] == g]
            smiles = str(smiles_row[r].iloc[0]) if not smiles_row.empty else ""
            update_details_for_group(smiles)
        _reflow_thumbnails()


        matching_rows = data[data[r] == smiles]

        selected_activity = dpg.get_value("counts_boxplot_activity_type")
        include_inactives = state.get("counts_boxplot_include_inactives", False)
        include_undefined = state.get("counts_boxplot_include_undefined", False)

        chi4_index = list(data.columns).index("Chi4")
        activity_columns = list(data.columns)[chi4_index + 1:]

        # Validate selected activity
        if selected_activity == "No activities" or selected_activity not in activity_columns:
            selected_activity = None

        filtered_rows = []

        for _, row in matching_rows.iterrows():

            # determine inactivity
            if selected_activity:
                val = row[selected_activity]
                is_inactive = pd.isna(val) or str(val).strip() == ""
            else:
                # inactive if ALL activities empty
                is_inactive = all(
                    pd.isna(row[act]) or str(row[act]).strip() == "" for act in activity_columns
                )

            if is_inactive and not include_inactives:
                continue

            # determine undefined (<, <=, >, >=)
            if selected_activity:
                raw_val = str(row[selected_activity]).strip()
                is_undefined = raw_val.startswith("<") or raw_val.startswith(">")
                if is_undefined and not include_undefined:
                    continue

            # require valid activity if not including inactives
            if selected_activity and not include_inactives:
                val = row[selected_activity]
                if pd.isna(val) or str(val).strip() == "":
                    continue

            filtered_rows.append(row)

        total = total_molecules
        n_match = len(filtered_rows)

        lines = []
        for row in filtered_rows:
            mol_id = int(row["Mol_sub_ID"]) if "Mol_sub_ID" in row else row.name
            line = f"Mol {mol_id}"

            # show first activity found
            for act in activity_columns:
                value = row[act]
                if pd.notna(value) and str(value).strip() != "":
                    unit = (
                        "nM" if act in state["nM_activity_types"] else
                        "%" if act in state["percent_activities"] else
                        "μg/mL" if act in state["ug/mL_activities"] else
                        "μM/min" if act in state["uM/min_activities"] else
                        ""
                    )
                    try:
                        line += f"  \u00BB  {act} = {value:.2f} {unit}"
                    except:
                        line += f"  \u00BB  {act} = {value} {unit}"
                    break

            lines.append(line)

        detail_text = f"{n_match}/{total} molecules:\n\n" + "\n".join(lines)
        dpg.set_value("counts_rgroup_details_text", detail_text)


    _reflow_thumbnails()

    
    # Register a left-click handler on the plot to trigger details update based on mouse proximity.
    dpg.add_mouse_click_handler(
        tag="counts_boxplot_click_handler",
        parent="handler_registry",
        callback=lambda s, a: (
            poll_mouse_and_update()
            if dpg.is_item_hovered("counts_boxplot")
            else None
        )
    )


    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")
