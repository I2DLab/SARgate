"""
=================
sal_plot.py
=================

Render layer for the SAR Landscape UI.

This module draws the full DearPyGui view for the SAR Landscape:
- a gradient bar for the colour scale (SALI, linear space);
- the main scatter plot (Tanimoto similarity vs Δp/Δvalue);
- interactive handlers (click on point to update the molecule panels);
- colour-bucket scatter series (efficient rendering of large point clouds);
- the pair of molecule images with export popups;
- the toggle buttons to draw/hide per-atom similarity maps.

All heavy-lifting (data computation, filtering logic, similarity-map generation,
and click handling) is delegated to helpers in `mod.logic.landscape_functions`.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Draw landscape plot

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Any
from app.utils.app_logger import log_event, log_settings
from app.gui.loading_win import set_loading_screen_progress
from app.utils.callbacks import (
    export_png_popup,
    register_plot_context_popup,
    register_responsive_image
)
from app.analysis.similarity.sal_utils import (
    _on_plot_click,
    _update_landscape_scatter,
    _draw_similarity_maps,
    _hide_similarity_maps
)
from app.gui.themes_manager import (
    get_continuous_colormap_color,
    apply_plot_theme,
    apply_colormap_theme,
    apply_infinite_line_theme
)


# -----------------------------------------------------------------------------
# 2. Draw landscape plot
# -----------------------------------------------------------------------------
def draw_landscape_plot(
    state: dict[str, Any],
    fp_choice: Any,
    activity: str,
    xs: Any,
    ys: Any,
    sali_raw: Any,
    lo: Any,
    hi: Any
) -> Any:
    """
    Draw the full SAR Landscape view.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
        fp_choice (Any): Parameter accepted by this routine.
        activity (str): Parameter accepted by this routine.
        xs (Any): Parameter accepted by this routine.
        ys (Any): Parameter accepted by this routine.
        sali_raw (Any): Parameter accepted by this routine.
        lo (Any): Parameter accepted by this routine.
        hi (Any): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    log_event("Similarity", "Rendering SAL plot", indent=1)
    log_settings("Similarity", indent=2, fingerprint=fp_choice, activity=activity, points=len(xs), sali_min=lo, sali_max=hi, colormap=state.get("colormap_continuous"))
    set_loading_screen_progress(state, 84)

    landscape_img_width = state["landscape_img_width"]
    landscape_img_height = round(landscape_img_width / 4 * 3)
    landscape_render_scale = 1.8
    landscape_render_width = int(round(landscape_img_width * landscape_render_scale))
    landscape_render_height = int(round(landscape_img_height * landscape_render_scale))

    for tag in ["landscape_window", "landscape_details_window"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=True)

    if dpg.does_item_exist("tooltip_landscape_drawlist"):
        dpg.delete_item("tooltip_landscape_drawlist")

    # Helper: map a raw SALI value in [lo, hi] to a colormap colour
    # -----------------------------------------------------------------------------
    # 2.1. Color from raw sali
    # -----------------------------------------------------------------------------
    def _color_from_raw_sali(val: float) -> Any:
        """
        Execute the color from raw sali routine.
        
        Args:
            val (float): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        if not np.isfinite(val) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return (128, 128, 128, 200)
        t = float((val - lo) / (hi - lo))
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        return get_continuous_colormap_color(t, state)

    def _tooltip_text_color(bg_rgba: Any) -> tuple[int, int, int, int]:
        """
        Pick black or white text based on the tooltip background contrast.

        Args:
            bg_rgba (Any): Background RGBA tuple.

        Returns:
            tuple[int, int, int, int]: Contrasting text color.
        """
        r, g, b = bg_rgba[:3]
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        return (20, 20, 20, 255) if luminance > 0.6 else (245, 245, 245, 255)

    def _build_tooltip_theme(tag: str, bg_rgba: Any) -> None:
        """
        Draw a compact colored overlay tooltip for the SAL plot.

        Args:
            tag (str): Tooltip prefix tag.
            bg_rgba (Any): Background RGBA tuple.

        Returns:
            None: This routine updates draw items in place.
        """
        text_rgba = _tooltip_text_color(bg_rgba)

        def _show_tooltip(text: str, screen_pos: tuple[int, int]) -> None:
            """
            Render the tooltip overlay contents at the requested screen position.

            Args:
                text (str): Tooltip body text.
                screen_pos (tuple[int, int]): Top-left position in viewport coordinates.

            Returns:
                None: This routine redraws the tooltip overlay.
            """
            drawlist_tag = f"{tag}_drawlist"
            if not dpg.does_item_exist(drawlist_tag):
                with dpg.viewport_drawlist(tag=drawlist_tag, front=True):
                    pass

            dpg.delete_item(drawlist_tag, children_only=True)

            lines = text.splitlines() or [text]
            max_chars = max(len(line) for line in lines) if lines else 0
            padding_x = 8
            padding_y = 6
            line_h = 17
            text_w = max(90, max_chars * 7)
            box_w = text_w + padding_x * 2
            box_h = max(24, len(lines) * line_h + padding_y * 2)
            x0, y0 = int(screen_pos[0]), int(screen_pos[1])
            x1, y1 = x0 + box_w, y0 + box_h

            dpg.draw_rectangle(
                (x0, y0),
                (x1, y1),
                parent=drawlist_tag,
                fill=bg_rgba,
                color=bg_rgba,
                rounding=6,
                thickness=1.0,
            )
            dpg.draw_text(
                (x0 + padding_x, y0 + padding_y),
                text,
                parent=drawlist_tag,
                color=text_rgba,
                size=16,
            )

        def _hide_tooltip() -> None:
            """
            Clear the tooltip overlay contents.

            Args:
                None.

            Returns:
                None: This routine clears the overlay drawlist.
            """
            drawlist_tag = f"{tag}_drawlist"
            if dpg.does_item_exist(drawlist_tag):
                dpg.delete_item(drawlist_tag, children_only=True)

        return _show_tooltip, _hide_tooltip

    with dpg.child_window(parent="landscape_window", width=-1, height=-1,
                        no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):

        with dpg.group(horizontal=True):

            with dpg.child_window(border=False, tag="landscape_gradient_bar_window",
                                  no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True,
                                  auto_resize_x=True,
                                  height=-1):
                dpg.add_colormap_scale(
                    tag="landscape_colormap_scale",
                    colormap=state["colormaps"][state["colormap_continuous"]],
                    label="SALI",
                    min_scale=float(lo),
                    max_scale=float(hi),
                    height=-1,
                    mirror=True,
                    format="%.0f"
                )
                dpg.bind_item_theme("landscape_colormap_scale", apply_colormap_theme(state))

            with dpg.plot(label=f"SAR Landscape: Similarity (Tanimoto on {fp_choice}) vs Δp{activity}",
                          width=-1,
                          height=-1,
                          tag="landscape_plot", no_menus=True, no_mouse_pos=True, zoom_rate=0.05):

                # X axis (Tanimoto similarity)
                dpg.add_plot_axis(dpg.mvXAxis, label="Tanimoto similarity", tag="landscape_x_axis")
                dpg.add_inf_line_series(
                    parent="landscape_x_axis",
                    x=[float(dpg.get_value("landscape_similarity_thresh"))],
                    label="min Similarity",
                    tag="landscape_similarity_thresh_rect",
                    horizontal=False
                )
                dpg.bind_item_theme("landscape_similarity_thresh_rect", apply_infinite_line_theme())

                # Y axis (Δp or Δvalue with units)
                if activity in state["nM_activity_types"]:
                    dpg.add_plot_axis(dpg.mvYAxis, label=f"Δp{activity}", tag="landscape_y_axis")
                else:
                    units = (
                        "%" if activity in state["percent_activities"] else
                        "μg/mL" if activity in state["ug/mL_activities"] else
                        "μM/min" if activity in state["uM/min_activities"] else
                        "nM"
                    )
                    dpg.add_plot_axis(dpg.mvYAxis, label=f"Δ{activity} ({units})", tag="landscape_y_axis")

                # Horizontal threshold (Δ axis)
                dpg.add_inf_line_series(
                    parent="landscape_y_axis",
                    x=[float(dpg.get_value("landscape_delta_thresh"))],
                    tag="landscape_delta_thresh_rect",
                    horizontal=True
                )
                dpg.bind_item_theme("landscape_delta_thresh_rect", apply_infinite_line_theme())

                # X ticks from 0 to 1 with 0.05 spacing
                x_values_to_show = [round(x, 2) for x in np.arange(0, 1.05, 0.05)]
                dpg.set_axis_ticks("landscape_x_axis", tuple((str(x), x) for x in x_values_to_show))

                dpg.bind_item_theme("landscape_plot", apply_plot_theme(state))
                register_plot_context_popup(
                    state,
                    context_key="sal_landscape_plot_context",
                    plot_tag="landscape_plot",
                    x_axis_tag="landscape_x_axis",
                    y_axis_tag="landscape_y_axis",
                    theme_kind="plot",
                )
                set_loading_screen_progress(state, 90)

            # --- Frame: rectangular line around the plot (x: 0→1; y: min/max ± 5%) ---
            # Robust min/max with fallback
            if len(ys):
                y_min = float(np.nanmin(ys))
                y_max = float(np.nanmax(ys))
            else:
                y_min, y_max = 0.0, 1.0

            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                y_low, y_high = -1.0, 1.0
            else:
                margin = 0.05 * (y_max - y_min)  # 5% of total height
                y_low  = 0
                y_high = y_max + margin

            # Single line_series that closes the rectangle: (0,y_low)→(1,y_low)→(1,y_high)→(0,y_high)→(0,y_low)
            dpg.add_line_series(
                x=[0.0, 1.0, 1.0, 0.0, 0.0],
                y=[y_low, y_low, y_high, y_high, y_low],
                parent="landscape_y_axis",
                tag="landscape_frame_outline",
                label="Bounds"
            )

            with dpg.theme() as landscape_frame_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 0), category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme("landscape_frame_outline", landscape_frame_theme)


            if dpg.does_item_exist("landscape_click_handler"):
                dpg.delete_item("landscape_click_handler")
            dpg.add_mouse_click_handler(tag="landscape_click_handler", parent="handler_registry", 
                                        callback=_on_plot_click, user_data=state)


            N_COL = int(state.get("landscape_color_bins", 128))
            N_COL = max(32, min(N_COL, 128))

            HEAVY_POINTS = int(state.get("landscape_heavy_points", 120_000))
            if len(xs) > HEAVY_POINTS:
                N_COL = min(N_COL, 48)

            # Bin edges on raw SALI range [lo, hi]
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo_eff, hi_eff = 0.0, 1.0
            else:
                lo_eff, hi_eff = float(lo), float(hi)

            bin_edges = np.linspace(lo_eff, hi_eff, N_COL + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Colours taken from the active colormap by mapping centers into [0,1]
            bin_colors = [_color_from_raw_sali(c) for c in bin_centers]

            # Marker size (in pixels)
            marker_px = int(state.get("landscape_marker_px", 4))
            state["landscape_marker_px"] = marker_px

            # Build and store bucket themes
            bucket_themes = []
            for col in bin_colors:
                th = dpg.add_theme()
                with dpg.theme_component(dpg.mvScatterSeries, parent=th):
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, float(marker_px), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, col, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill,    col, category=dpg.mvThemeCat_Plots)
                bucket_themes.append(th)
            state["landscape_bucket_themes"] = bucket_themes

            # Create empty series and bind each series to its theme
            color_series_tags = []
            for i in range(N_COL):
                tag = f"landscape_color_series_{i}"
                dpg.add_scatter_series([], [], parent="landscape_y_axis", tag=tag)
                dpg.bind_item_theme(tag, bucket_themes[i])
                color_series_tags.append(tag)

    
            # Keep binning info in state (used for colormap refresh)
            state["landscape_lo_hi"] = (lo_eff, hi_eff)
            state["landscape_bin_centers"] = bin_centers.tolist()
            state["landscape_color_series_tags"] = color_series_tags
            state["landscape_bucket_themes"] = bucket_themes

            def _refresh_landscape_bucket_colours() -> None:
                """
                Rebuild bucket themes using the currently applied colormap and re-bind them to series.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                centers = state.get("landscape_bin_centers")
                series_tags = state.get("landscape_color_series_tags")
                old_themes = state.get("landscape_bucket_themes")

                if not centers or not series_tags:
                    return

                # Compute new RGBA for each bin centre using the current colormap
                new_cols = [get_continuous_colormap_color(float(c - lo_eff) / (hi_eff - lo_eff) if (hi_eff > lo_eff) else 0.0, state)
                            for c in centers]

                # Delete old themes (if any)
                if old_themes:
                    for th in old_themes:
                        if dpg.does_item_exist(th):
                            dpg.delete_item(th)

                # Rebuild themes and bind
                new_themes = []
                marker_px = int(state.get("landscape_marker_px", 4))
                for col in new_cols:
                    th = dpg.add_theme()
                    with dpg.theme_component(dpg.mvScatterSeries, parent=th):
                        dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, float(marker_px), category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, col, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerFill,    col, category=dpg.mvThemeCat_Plots)
                    new_themes.append(th)

                # Bind each series to the corresponding (refreshed) theme
                for i, tag in enumerate(series_tags):
                    if dpg.does_item_exist(tag) and i < len(new_themes):
                        dpg.bind_item_theme(tag, new_themes[i])

                state["landscape_bucket_themes"] = new_themes

            # Expose the refresh function to the outside (apply_colormap will call it)
            state["landscape_refresh_colors"] = _refresh_landscape_bucket_colours

        

            dpg.configure_item(
                "landscape_delta_thresh",
                callback=lambda s, a, u=None: (
                    _update_landscape_scatter(bin_edges, N_COL, state),
                    dpg.configure_item("landscape_delta_thresh_rect", x=[float(a)])
                )
            )

            dpg.configure_item(
                "landscape_similarity_thresh",
                callback=lambda s, a, u=None: (
                    _update_landscape_scatter(bin_edges, N_COL, state),
                    dpg.configure_item("landscape_similarity_thresh_rect", x=[float(a)])
                )
            )
                    
            dpg.configure_item(
                "landscape_sali_index_thresh",
                callback=lambda s, a, u=None: _update_landscape_scatter(bin_edges, N_COL, state)
            )

            # First render based on current thresholds
            _update_landscape_scatter(bin_edges, N_COL, state)
            set_loading_screen_progress(state, 96)

            show_landscape_tooltip, hide_landscape_tooltip = _build_tooltip_theme("tooltip_landscape", (30, 30, 30, 240))

            def update_landscape_tooltip() -> None:
                """
                Show a floating tooltip with pair details when hovering a SAL point.

                Args:
                    None.

                Returns:
                    None: This routine updates the hover popup in place.
                """
                if state.get("current_similarity_subtab") != "landscape_tab":
                    hide_landscape_tooltip()
                    return

                if not dpg.does_item_exist("landscape_plot"):
                    hide_landscape_tooltip()
                    return

                mx_local, my_local = dpg.get_mouse_pos(local=True)
                plot_min = dpg.get_item_rect_min("landscape_plot")
                plot_max = dpg.get_item_rect_max("landscape_plot")
                if (
                    mx_local < plot_min[0] or mx_local > plot_max[0] or
                    my_local < plot_min[1] or my_local > plot_max[1]
                ):
                    hide_landscape_tooltip()
                    return

                vis_idx = state.get("landscape_visible_idx", None)
                vis_xs = state.get("landscape_visible_xs", None)
                vis_ys = state.get("landscape_visible_ys", None)
                if vis_idx is None or vis_xs is None or vis_ys is None or len(vis_idx) == 0:
                    hide_landscape_tooltip()
                    return

                px, py = dpg.get_plot_mouse_pos()
                if not (np.isfinite(px) and np.isfinite(py)):
                    hide_landscape_tooltip()
                    return

                plot_w_px, plot_h_px = dpg.get_item_rect_size("landscape_plot")
                x_min, x_max = dpg.get_axis_limits("landscape_x_axis")
                y_min, y_max = dpg.get_axis_limits("landscape_y_axis")
                x_span = max(1e-12, x_max - x_min)
                y_span = max(1e-12, y_max - y_min)
                marker_px_local = int(state.get("landscape_marker_px", 4))
                radius_px = max(4, int(round(0.8 * marker_px_local)))
                tol_x = radius_px * (x_span / max(1, plot_w_px))
                tol_y = radius_px * (y_span / max(1, plot_h_px))

                dx = (vis_xs - px) / tol_x
                dy = (vis_ys - py) / tol_y
                dist2 = dx * dx + dy * dy
                k = int(np.argmin(dist2))
                if dist2[k] > 1.0:
                    hide_landscape_tooltip()
                    return

                pair_idx = int(vis_idx[k])
                pair_i = state.get("landscape_pair_i")
                pair_j = state.get("landscape_pair_j")
                work = state.get("landscape_work_df")
                if pair_i is None or pair_j is None or work is None:
                    hide_landscape_tooltip()
                    return

                i = int(pair_i[pair_idx])
                j = int(pair_j[pair_idx])
                row_i = work.iloc[i]
                row_j = work.iloc[j]
                delta = ys[pair_idx]
                similarity = xs[pair_idx]
                sali = sali_raw[pair_idx]

                tooltip_text = (
                    f"Mol {row_i['MolID']} - {row_i['Name']}\n"
                    f"Mol {row_j['MolID']} - {row_j['Name']}\n"
                    f"Similarity: {similarity:.2f}\n"
                    f"Δp{activity}: {delta:.2f}\n"
                    f"SALI: {sali:.2f}"
                )
                show_landscape_tooltip, hide_landscape_tooltip_local = _build_tooltip_theme(
                    "tooltip_landscape",
                    _color_from_raw_sali(float(sali)),
                )
                mx, my = dpg.get_mouse_pos(local=False)
                show_landscape_tooltip(tooltip_text, (int(mx) + 14, int(my) + 14))

            if dpg.does_item_exist("landscape_mouse_move_handler"):
                dpg.delete_item("landscape_mouse_move_handler")
            dpg.add_mouse_move_handler(
                tag="landscape_mouse_move_handler",
                parent="handler_registry",
                callback=lambda s, a: update_landscape_tooltip(),
            )
            if dpg.does_item_exist("landscape_mouse_wheel_handler"):
                dpg.delete_item("landscape_mouse_wheel_handler")
            dpg.add_mouse_wheel_handler(
                tag="landscape_mouse_wheel_handler",
                parent="handler_registry",
                callback=lambda s, a: update_landscape_tooltip(),
            )
            if dpg.does_item_exist("landscape_left_drag_tooltip_handler"):
                dpg.delete_item("landscape_left_drag_tooltip_handler")
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                tag="landscape_left_drag_tooltip_handler",
                parent="handler_registry",
                callback=lambda s, a: update_landscape_tooltip(),
            )
            if dpg.does_item_exist("landscape_right_drag_tooltip_handler"):
                dpg.delete_item("landscape_right_drag_tooltip_handler")
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Right,
                tag="landscape_right_drag_tooltip_handler",
                parent="handler_registry",
                callback=lambda s, a: update_landscape_tooltip(),
            )

    with dpg.child_window(parent="landscape_details_window", width=-1, height=-1,
                            no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):


        empty_data = np.zeros((landscape_render_width * landscape_render_height * 4,), dtype=np.float32)
        
        if not dpg.does_item_exist("landscape_mol1_image_texture"):
            dpg.add_dynamic_texture(landscape_render_width, landscape_render_height, empty_data, 
                                    tag="landscape_mol1_image_texture", parent="texture_registry")
        else:
            dpg.set_value("landscape_mol1_image_texture", empty_data)

        if not dpg.does_item_exist("landscape_mol2_image_texture"):
            dpg.add_dynamic_texture(landscape_render_width, landscape_render_height, empty_data, 
                                    tag="landscape_mol2_image_texture", parent="texture_registry")
        else:
            dpg.set_value("landscape_mol2_image_texture", empty_data)


        dpg.add_image("landscape_mol1_image_texture", width=landscape_img_width, height=landscape_img_height,
                        tag="landscape_mol1_image_widget", border_color=(0, 0, 0, 0))
        with dpg.tooltip("landscape_mol1_image_widget", delay=0):
            dpg.add_text("", tag="landscape_mol1_tooltip_text")
        register_responsive_image(
            state,
            image_tag=f"landscape_mol1_image_widget",
            parent_tag="landscape_details_window",
            aspect_ratio=0.75,
            tab="landscape_tab"
        )
        export_png_popup("landscape_mol1_image_widget", "landscape_mol1_image_texture", state)

        with dpg.group(horizontal=True):
            dpg.add_text(f"Δp{activity}= N/A\nSimilarity = N/A\nSALI Index = N/A", tag="landscape_couple_details_text")
            with dpg.group():
                dpg.add_spacer(height=state["win_spacer"])
                dpg.add_button(label="Draw Similarity Map",
                                tag="landscape_similarity_map_button",
                                callback=_draw_similarity_maps, user_data=state)
                dpg.add_button(label="Hide Similarity Map",
                                tag="landscape_similarity_map_hide_button",
                                show=False,
                                callback=_hide_similarity_maps, user_data=state)
                dpg.add_spacer(height=state["win_spacer"])
        
        dpg.add_image("landscape_mol2_image_texture", width=landscape_img_width, height=landscape_img_height,
                        tag="landscape_mol2_image_widget", border_color=(0, 0, 0, 0))
        with dpg.tooltip("landscape_mol2_image_widget", delay=0):
            dpg.add_text("", tag="landscape_mol2_tooltip_text")
        register_responsive_image(
            state,
            image_tag=f"landscape_mol2_image_widget",
            parent_tag="landscape_details_window",
            aspect_ratio=0.75,
            tab="landscape_tab"
        )
        export_png_popup("landscape_mol2_image_widget", "landscape_mol2_image_texture", state)
    set_loading_screen_progress(state, 98)
