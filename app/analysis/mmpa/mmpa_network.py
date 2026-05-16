"""
=====================
mmpa_network.py
=====================

MMPA network visualisation module.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Build mmpa network map
# 3. Render group callback
# 4. Make circular node
# 5. Separate overlaps
# 6. Clear previous rendering
# 7. Compute mcs for component
# 8. Compute highlight sets
# 9. Populate group buttons
# 10. Read activity value
# 11. Ensure ring texture
# 12. Setup hover interaction
# 13. Clear hover overlay
# 14. Get or make line theme
# 15. Get or make img theme
# 16. Build node lods
# 17. Setup lod autoswap
# 18. Mmpa lod maybe swap

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import io
import math
import time
import dearpygui.dearpygui as dpg
import numpy as np
import networkx as nx
from collections import defaultdict
from app.utils.app_logger import log_event, log_settings
from typing import Any
from PIL import Image as pilImage, ImageDraw, ImageOps, ImageChops, ImageFont
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdRGroupDecomposition
from rdkit.Chem.Draw import rdMolDraw2D
from app.gui.loading_win import draw_loading_screen
from app.gui.themes_manager import apply_mmpa_network_theme

MMPA_DETAIL_SCALE_MIN = 0.6
MMPA_DETAIL_SCALE_MAX = 2.4
MMPA_DETAIL_SCALE_DEFAULT = MMPA_DETAIL_SCALE_MIN + 0.25 * (MMPA_DETAIL_SCALE_MAX - MMPA_DETAIL_SCALE_MIN)


def clear_mmpa_network_memory(
    state: dict[str, Any],
    clear_plot: bool = True,
    clear_structure: bool = True,
) -> None:
    """
    Remove every cached/rendered MMPA-network resource from Dear PyGui and state.

    Args:
        state (dict[str, Any]): Shared application state.
        clear_plot (bool): When True, also remove current plot/hover items.
        clear_structure (bool): When True, also drop cached graph/components data.

    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if clear_plot:
        for tag in [
            "mmpa_network_plot",
            "mmpa_plot_handler_registry",
            "mmpa_hover_info_drawlist",
        ]:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

    texture_tags = []
    for key in [
        "mmpa__live_textures",
    ]:
        texture_tags.extend(list(state.get(key, []) or []))

    ring_tex = state.get("mmpa__ring_texture")
    if ring_tex:
        texture_tags.append(ring_tex)
    texture_tags.extend(list((state.get("mmpa__ring_textures", {}) or {}).values()))

    for tex_map_key in ["mmpa__lod_base", "mmpa__lod_high"]:
        tex_map = state.get(tex_map_key, {}) or {}
        for lods in tex_map.values():
            if isinstance(lods, dict):
                texture_tags.extend(list(lods.values()))

    seen = set()
    for tag in texture_tags:
        if tag in seen:
            continue
        seen.add(tag)
        try:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        except Exception:
            pass

    if clear_structure:
        state.pop("mmpa__graph", None)
        state.pop("mmpa__components", None)
    state.pop("mmpa__positions_circle", None)
    state.pop("mmpa__positions_similarity", None)
    state.pop("mmpa__positions_activity", None)
    state.pop("mmpa__min_node_distance", None)
    state.pop("mmpa__texture_render_specs", None)
    state.pop("mmpa__hover_texture_specs", None)
    state.pop("mmpa__lod_style", None)
    state.pop("mmpa__live_textures", None)
    state.pop("mmpa__node_series_map", None)
    state.pop("mmpa__edge_series_map", None)
    state.pop("mmpa__edge_series_order", None)
    state.pop("mmpa__node_radius_units_base", None)
    state.pop("mmpa__node_radius_units", None)
    state.pop("mmpa__hover_items", None)
    state.pop("mmpa__hover_last", None)
    state.pop("mmpa__hover_positions", None)
    state.pop("mmpa__hover_radius", None)
    state.pop("mmpa__hover_axis_y", None)
    state.pop("mmpa__hover_axis_x", None)
    state.pop("mmpa__hover_plot", None)
    state.pop("mmpa__lod_series", None)
    state.pop("mmpa__lod_base", None)
    state.pop("mmpa__lod_high", None)
    state.pop("mmpa__highlighted_nodes", None)
    state.pop("mmpa__hover_throttle_last_t", None)
    state.pop("mmpa__hover_throttle_last_pos", None)
    state.pop("mmpa__hover_adj", None)
    state.pop("mmpa__ring_texture", None)
    state.pop("mmpa__ring_textures", None)
    state.pop("mmpa__hover_info_text", None)
    state.pop("mmpa__hover_render_inputs", None)
    state.pop("mmpa__hover_focus_id", None)
    state.pop("mmpa__component_core", None)
    state.pop("mmpa__component_highlight_sets", None)
    state.pop("mmpa__lod_node_diam_units", None)
    state.pop("mmpa__lod_current", None)
    state.pop("mmpa__current_node_extents", None)
    state.pop("mmpa__current_group_index", None)
    state.pop("_mmpa_lod_maybe_swap", None)
    state.pop("mmpa_detail_scale_quantized", None)
    state.pop("mmpa_network_refresh_colors", None)


def _show_mmpa_hover_info(text: str, screen_pos: tuple[int, int]) -> None:
    tag = "mmpa_hover_info_drawlist"
    if not dpg.does_item_exist(tag):
        with dpg.viewport_drawlist(tag=tag, front=True):
            pass
    dpg.delete_item(tag, children_only=True)
    lines = text.splitlines() or [text]
    max_chars = max(len(line) for line in lines) if lines else 0
    padding_x = 8
    padding_y = 6
    line_h = 17
    text_w = max(110, max_chars * 7)
    box_w = text_w + padding_x * 2
    box_h = max(24, len(lines) * line_h + padding_y * 2)
    x0, y0 = int(screen_pos[0]), int(screen_pos[1])
    x1, y1 = x0 + box_w, y0 + box_h
    dpg.draw_rectangle((x0, y0), (x1, y1), parent=tag, fill=(255, 255, 255, 235), color=(30, 30, 30, 220), rounding=6, thickness=1.0)
    dpg.draw_text((x0 + padding_x, y0 + padding_y), text, parent=tag, color=(20, 20, 20, 255), size=16)


def _hide_mmpa_hover_info() -> None:
    tag = "mmpa_hover_info_drawlist"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag, children_only=True)


def _make_mmpa_hover_label_text(state: dict[str, Any], mol_id: int) -> str:
    mol_name = state.get("mol_names_dict", {}).get(mol_id, "")
    mol_activity = state.get("mol_activity_dict", {}).get(mol_id, "")
    name_string = f"{mol_name}" if str(mol_name) not in ("", "nan", "None") else ""
    act_string = (
        f"{state['mmpa_y_axis_label']}: {float(mol_activity):.2f}"
        if str(mol_activity) not in ("0.0", "nan", "", "None", "N/A", "NA")
        else "INACTIVE"
    )
    parts = [f"Mol {mol_id}"]
    if name_string:
        parts.append(name_string)
    parts.append(act_string)
    return "\n".join(parts).strip()


def _build_mmpa_rgd_parameters(num_mols: int) -> Any:
    params = rdRGroupDecomposition.RGroupDecompositionParameters()
    params.matchingStrategy = rdRGroupDecomposition.RGroupMatching.Greedy
    params.substructMatchParams.numThreads = 0
    params.timeout = -1.0
    params.alignment = rdRGroupDecomposition.RGroupCoreAlignment.NoAlignment
    params.scoreMethod = rdRGroupDecomposition.RGroupScore.Match
    params.removeAllHydrogenRGroups = True
    params.removeAllHydrogenRGroupsAndLabels = True
    params.removeHydrogensPostMatch = True
    params.allowMultipleRGroupsOnUnlabelled = False
    params.allowNonTerminalRGroups = True
    return params


def _load_mmpa_label_font(size_px: int) -> Any:
    size_px = max(12, int(size_px))
    for font_name in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(font_name, size_px)
        except Exception:
            continue
    return ImageFont.load_default()


def _render_mmpa_texture_from_spec(spec: dict[str, Any], detail_scale: float) -> np.ndarray:
    mol = Chem.Mol(spec["mol"])
    highlight = spec.get("highlight")
    highlight_style = spec.get("highlight_style", "hover")
    lod = int(spec["lod"])
    rdw = int(spec["rdw"])
    rdh = int(spec["rdh"])
    legend_fs = int(spec["legend_fs"])
    bond_w = int(spec["bond_w"])
    margin_px = int(spec["margin_px"])
    ss_factor = int(spec.get("ss_factor", 2))

    if highlight_style == "base":
        mcs_color = (1.0, 1.0, 1.0, 1.00)
        var_color = (1.0, 1.0, 1.0, 1.0)
        link_color = (1.0, 1.0, 1.0, 1.00)
    else:
        mcs_color = (0.2, 0.8, 1.0, 1.00)
        var_color = (1.0, 0.25, 0.25, 1.00)
        link_color = (1.0, 0.25, 0.25, 0.00)
    bond_highlight_width_multiplier_base = 0.3
    atom_highlight_radius = 1.2

    font_scale = 0.88 + (float(detail_scale) - 1.0) * 0.05
    bond_scale = 1.0 + (float(detail_scale) - 1.0) * 1.55
    bond_length_scale = max(0.10, 1.0 + (float(detail_scale) - 1.0) * 2.55)
    multiple_bond_offset_scale = 1.0 + (float(detail_scale) - 1.0) * 0.75
    bond_highlight_width_scale = 1.0 - (float(detail_scale) - 1.0) * 0.40
    atom_highlight_scale = 1.0 - (float(detail_scale) - 1.0) * 0.05

    font_px = max(6, int(round(legend_fs * font_scale)))
    bond_px = max(1.0, float(bond_w) * bond_scale)
    bond_len_px = max(34.0, (float(max(rdw, 1)) / 17.0) * bond_length_scale)
    effective_atom_highlight_radius = max(0.04, float(atom_highlight_radius) * atom_highlight_scale)
    effective_bond_highlight_width_multiplier = max(
        0.05,
        min(
            12.0,
            (float(bond_highlight_width_multiplier_base) * bond_highlight_width_scale) / max(0.15, float(bond_scale)),
        ),
    )

    drawer = rdMolDraw2D.MolDraw2DCairo(rdw, rdh)
    opts = drawer.drawOptions()
    opts.padding = min(0.06, 0.024 + max(0.0, float(detail_scale) - 1.0) * 0.015)
    opts.bondLineWidth = bond_px
    if hasattr(opts, "highlightBondWidthMultiplier"):
        try:
            opts.highlightBondWidthMultiplier = effective_bond_highlight_width_multiplier
        except Exception:
            pass
    if hasattr(opts, "atomHighlightsAreCircles"):
        try:
            opts.atomHighlightsAreCircles = True
        except Exception:
            pass
    if hasattr(opts, "standardColoursForHighlightedAtoms"):
        try:
            opts.standardColoursForHighlightedAtoms = (highlight_style == "base")
        except Exception:
            pass
    if hasattr(opts, "multipleBondOffset"):
        try:
            base_offset = float(getattr(opts, "multipleBondOffset"))
            opts.multipleBondOffset = max(0.06, base_offset * multiple_bond_offset_scale)
        except Exception:
            pass
    opts.highlightRadius = effective_atom_highlight_radius
    opts.additionalAtomLabelPadding = 0.18
    if hasattr(opts, "fixedFontSize"):
        try:
            opts.fixedFontSize = font_px
        except Exception:
            pass
    if hasattr(opts, "minFontSize"):
        try:
            opts.minFontSize = font_px
        except Exception:
            pass
    if hasattr(opts, "maxFontSize"):
        try:
            opts.maxFontSize = font_px
        except Exception:
            pass
    if hasattr(opts, "fixedBondLength"):
        try:
            opts.fixedBondLength = bond_len_px
        except Exception:
            pass
    if hasattr(opts, "fixedScale"):
        try:
            opts.fixedScale = -1.0
        except Exception:
            pass
    if hasattr(opts, "drawMolsSameScale"):
        try:
            opts.drawMolsSameScale = False
        except Exception:
            pass
    if hasattr(opts, "clearBackground"):
        try:
            opts.clearBackground = False
        except Exception:
            pass
    if hasattr(opts, "setBackgroundColour"):
        try:
            opts.setBackgroundColour((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass
    elif hasattr(opts, "backgroundColour"):
        try:
            opts.backgroundColour = (1.0, 1.0, 1.0, 0.0)
        except Exception:
            pass

    if highlight is None:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend="")
    else:
        mcs_atoms, mcs_bonds, variable_atoms, variable_bonds, link_bonds = highlight
        if highlight_style == "base":
            all_atoms = list(range(mol.GetNumAtoms()))
            all_bonds = list(range(mol.GetNumBonds()))
            atom_colors = {idx: mcs_color for idx in all_atoms}
            bond_colors = {idx: mcs_color for idx in all_bonds}
            atom_radii = {idx: max(0.08, effective_atom_highlight_radius * 1.45) for idx in all_atoms}
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer,
                mol,
                legend="",
                highlightAtoms=all_atoms,
                highlightBonds=all_bonds,
                highlightAtomColors=atom_colors,
                highlightBondColors=bond_colors,
                highlightAtomRadii=atom_radii,
            )
        else:
            atom_colors = {idx: mcs_color for idx in mcs_atoms}
            atom_colors.update({idx: var_color for idx in variable_atoms})
            # bond_colors = {idx: mcs_color for idx in mcs_bonds}
            # bond_colors.update({idx: link_color for idx in link_bonds})
            # bond_colors.update({idx: var_color for idx in variable_bonds})
            atom_radii = {idx: max(0.06, effective_atom_highlight_radius * 1.10) for idx in variable_atoms}
            atom_radii.update({idx: effective_atom_highlight_radius for idx in mcs_atoms})
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer,
                mol,
                legend="",
                highlightAtoms=(variable_atoms + mcs_atoms),
                # highlightBonds=(variable_bonds + link_bonds + mcs_bonds),
                highlightAtomColors=atom_colors,
                highlightAtomRadii=atom_radii,
            )

    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()
    img = pilImage.open(io.BytesIO(png_data)).convert("RGBA")
    effective_margin = int(round(margin_px / (1.0 + max(0.0, float(bond_length_scale) - 1.0) * 0.18)))
    effective_margin = max(8, effective_margin)
    node_img = make_circular_node(img, lod, margin_px=effective_margin, ss_factor=ss_factor)
    label_text = str(spec.get("label_text", "") or "").strip()
    if label_text:
        node_img = node_img.copy()
        draw = ImageDraw.Draw(node_img, "RGBA")
        label_font_px = max(20, int(round((lod / 17.0) * (0.90 + max(0.0, float(detail_scale) - 1.0) * 0.22))))
        font = _load_mmpa_label_font(label_font_px)
        lines = label_text.splitlines()
        padding_x = max(10, lod // 40)
        padding_y = max(8, lod // 56)
        line_gap = max(3, lod // 180)
        line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        text_w = max((box[2] - box[0]) for box in line_boxes) if line_boxes else 0
        line_h = max((box[3] - box[1]) for box in line_boxes) if line_boxes else 12
        box_h = (line_h * len(lines)) + (line_gap * max(0, len(lines) - 1)) + padding_y * 2
        x0 = max(6, (lod - text_w) // 2 - padding_x)
        y0 = max(6, lod - box_h - max(8, lod // 28))
        x1 = min(lod - 6, x0 + text_w + padding_x * 2)
        y1 = min(lod - 6, y0 + box_h)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=max(8, lod // 40), fill=(255, 255, 255, 232))
        ty = y0 + padding_y
        for line in lines:
            draw.text((x0 + padding_x, ty), line, fill=(15, 15, 15, 255), font=font)
            ty += line_h + line_gap
    return (np.array(node_img) / 255.0).astype(np.float32)


def _quantize_mmpa_detail_scale(scale: float) -> float:
    scale = max(MMPA_DETAIL_SCALE_MIN, min(MMPA_DETAIL_SCALE_MAX, float(scale)))
    return round(scale * 5.0) / 5.0


def _ensure_mmpa_zero_cell_padding_table_theme() -> str:
    tag = "mmpa_zero_cell_padding_table_theme"
    if dpg.does_item_exist(tag):
        return tag
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvTable):
            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 2, 2, category=dpg.mvThemeCat_Core)
    return tag


def _get_mmpa_texture_sizes(n_nodes: int) -> tuple[int, int, int]:
    n_nodes = max(1, int(n_nodes))
    if n_nodes <= 8:
        return 1024, 1024, 2
    if n_nodes <= 16:
        return 640, 768, 2
    if n_nodes <= 28:
        return 448, 640, 1
    return 384, 512, 1


def _make_mmpa_texture_spec(
    mol: Any,
    highlight: Any,
    highlight_style: str,
    lod: int,
    rdw: int,
    rdh: int,
    legend_fs: int,
    bond_w: int,
    margin_px: int,
    ss_factor: int,
) -> dict[str, Any]:
    return {
        "mol": Chem.Mol(mol),
        "highlight": highlight,
        "highlight_style": highlight_style,
        "lod": int(lod),
        "rdw": int(rdw),
        "rdh": int(rdh),
        "legend_fs": max(18, int(legend_fs)),
        "bond_w": max(1, int(round(bond_w))),
        "margin_px": max(16, int(margin_px)),
        "ss_factor": int(ss_factor),
    }


def _ensure_mmpa_hover_texture(state: dict[str, Any], mol_id: int) -> Any:
    tex_map = (state.get("mmpa__lod_high", {}) or {}).get(int(mol_id), {}) or {}
    tex_tag = tex_map.get(1024) or next(iter(tex_map.values()), None)
    if tex_tag and dpg.does_item_exist(tex_tag):
        return tex_tag
    return None


# -----------------------------------------------------------------------------
# 2. Build mmpa network map
# -----------------------------------------------------------------------------
def build_mmpa_network_map(sender: Any, app_data: Any, user_data: Any) -> None:
    log_event("MMPA", "Drawing 'MMPA Network'", indent=1)
    state = user_data
    log_settings("MMPA", indent=2, y_axis_label=state.get("mmpa_y_axis_label"), nodes=len(state.get("mmpa_graph_connections", {})))

    draw_loading_screen(state, bg=False)

    if dpg.does_item_exist("mmpa_network_plot_window"):
        dpg.delete_item("mmpa_network_plot_window", children_only=True)
        with dpg.child_window(
            parent="mmpa_network_plot_window",
            tag="mmpa_network_controls_window",
            width=-1,
            auto_resize_y=True,
            no_scrollbar=True,
            horizontal_scrollbar=False,
            no_scroll_with_mouse=True,
            border=False,
        ):
            with dpg.table(tag="mmpa_network_controls_table", header_row=False, borders_innerH=False, borders_innerV=False, borders_outerH=False, borders_outerV=False, width=-1):
                dpg.add_table_column(init_width_or_weight=1.0)
                dpg.add_table_column(init_width_or_weight=1.0)
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_slider_float(
                            tag="mmpa_layout_balance_slider",
                            width=-1,
                            min_value=0.0,
                            max_value=2.0,
                            default_value=1.0,
                            format="  Circle    ↔    Network    ↔    Activity",
                            callback=lambda s, a: _on_mmpa_layout_balance_change(state, a),
                        )
                    with dpg.table_cell():
                        dpg.add_slider_float(
                            tag="mmpa_edge_length_scale_slider",
                            width=-1,
                            min_value=0.6,
                            max_value=4.0,
                            default_value=float(state.get("mmpa_edge_length_scale", 2.3)),
                            format="Connection Length",
                            callback=lambda s, a: _on_mmpa_edge_length_scale_change(state, a),
                        )
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_slider_float(
                            tag="mmpa_render_scale_slider",
                            width=-1,
                            min_value=0.6,
                            max_value=2.4,
                            default_value=float(state.get("mmpa_render_scale", 2.4)),
                            format="Node Scale",
                            callback=lambda s, a: _on_mmpa_render_scale_change(state, a),
                        )
                    with dpg.table_cell():
                        dpg.add_slider_float(
                            tag="mmpa_detail_scale_slider",
                            width=-1,
                            min_value=MMPA_DETAIL_SCALE_MIN,
                            max_value=MMPA_DETAIL_SCALE_MAX,
                            default_value=float(state.get("mmpa_detail_scale", MMPA_DETAIL_SCALE_DEFAULT)),
                            format="Detail Scale",
                            callback=lambda s, a: _on_mmpa_detail_scale_change(state, a),
                        )
            dpg.bind_item_theme("mmpa_network_controls_table", _ensure_mmpa_zero_cell_padding_table_theme())

        release_tag = "mmpa_detail_scale_release_handler"
        if dpg.does_item_exist(release_tag):
            dpg.delete_item(release_tag)
        dpg.add_mouse_release_handler(
            button=dpg.mvMouseButton_Left,
            parent="handler_registry",
            tag=release_tag,
            callback=lambda s, a: _on_mmpa_detail_scale_release(state),
        )

        with dpg.child_window(
            parent="mmpa_network_plot_window",
            tag="mmpa_network_plot_canvas",
            width=-1,
            height=-1,
            no_scrollbar=False,
            horizontal_scrollbar=False,
            no_scroll_with_mouse=True,
            border=False,
        ):
            pass

    G = nx.Graph()
    connections = state.get("mmpa_graph_connections", {})
    for node, neighbors in connections.items():
        nid = int(node)
        G.add_node(nid)
        for neighbor in neighbors:
            G.add_edge(nid, int(neighbor))

    components = [sorted(list(c)) for c in nx.connected_components(G)]
    components.sort(key=len, reverse=True)
    state["mmpa__graph"] = G
    state["mmpa__components"] = components

    _populate_group_buttons(state)


# -----------------------------------------------------------------------------
# 3. Render group callback
# -----------------------------------------------------------------------------
def _render_group_callback(sender: Any, app_data: Any, user_data: Any) -> None:

    from app.gui.loading_win import set_loading_screen_progress

    payload = user_data
    state = payload["state"]
    g_idx = payload["group_index"]
    state["mmpa__current_group_index"] = int(g_idx)
    G = state["mmpa__graph"]
    components = state["mmpa__components"]
    component_nodes = components[g_idx]

    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)

    _clear_previous_rendering(state)
    set_loading_screen_progress(state, 1)

    # --- area plot ---
    plot_area_w, plot_area_h = dpg.get_item_rect_size("mmpa_network_plot_canvas")
    if not plot_area_w or not plot_area_h:
        vp_w = dpg.get_viewport_client_width()
        vp_h = dpg.get_viewport_client_height()
        sidebar_w = dpg.get_item_rect_size("mmpa_group_sidebar_window")[0] if dpg.does_item_exist("mmpa_group_sidebar_window") else int(vp_w * 0.18)
        controls_h = dpg.get_item_rect_size("mmpa_network_controls_window")[1] if dpg.does_item_exist("mmpa_network_controls_window") else 42
        plot_area_w = max(400, vp_w - sidebar_w - 40)
        plot_area_h = max(300, vp_h - controls_h - 80)

    # --- Texture quality / node diameter ---
    n = max(1, len(component_nodes))
    grid_side = math.sqrt(n)
    base_spacing_factor_px = 1.2
    max_diam_w_px = (plot_area_w / (grid_side * base_spacing_factor_px + 0.5))
    max_diam_h_px = (plot_area_h / (grid_side * base_spacing_factor_px + 0.75))
    estimated_tex_diam_px = int(max(220, min(max_diam_w_px, max_diam_h_px)))
    TEX_DIAMETER = int(max(220, min(1024, estimated_tex_diam_px)))  # px

    # RDKit canvas and style (px)
    if n <= 6:
        RDW, RDH, LEGEND_FS, BW = 1200, 900, 72, 5
    elif n <= 12:
        RDW, RDH, LEGEND_FS, BW = 1000, 800, 62, 4
    elif n <= 24:
        RDW, RDH, LEGEND_FS, BW = 900, 700, 52, 4
    else:
        RDW, RDH, LEGEND_FS, BW = 800, 620, 46, 3
    INNER_MARGIN = max(28, int(TEX_DIAMETER * 0.07))
    set_loading_screen_progress(state, 1)

    # --- Y = actual pValue ---
    ys = {}
    for mid in component_nodes:
        yv = _read_activity_value(state, mid)  # None for 0.0/NaN/inactive
        if yv is not None:
            ys[mid] = yv

    have_y = len(ys) > 0
    if have_y:
        sorted_by_y = sorted(component_nodes, key=lambda m: ys.get(m, -1e9), reverse=True)
        y_vals = [ys[m] for m in sorted_by_y if m in ys]
        y_min = min(y_vals); y_max = max(y_vals)
        if y_min == y_max:
            y_min -= 0.5; y_max += 0.5
    else:
        sorted_by_y = list(component_nodes)
        y_min, y_max = 0.0, 1.0
    y_span = max(1e-6, (y_max - y_min))
    set_loading_screen_progress(state, 1)

    # --- positions: similarity/activity layouts ---
    NODE_SIZE_FRAC = 0.42
    NODE_DIAMETER_UNITS = max(y_span * NODE_SIZE_FRAC, 0.35)
    spacing_factor = min(2.1, 1.10 + 0.018 * float(n))
    NODE_SEP_UNITS = NODE_DIAMETER_UNITS * spacing_factor

    subgraph = G.subgraph(component_nodes).copy()
    similarity_positions: dict[int, tuple[float, float]] = {}
    activity_positions: dict[int, tuple[float, float]] = {}

    if len(component_nodes) == 1:
        only_id = int(component_nodes[0])
        only_y = float(ys.get(only_id, (y_min + y_max) * 0.5))
        similarity_positions[only_id] = (0.0, only_y)
        activity_positions[only_id] = (0.0, only_y)
    else:
        spring_raw = nx.spring_layout(
            subgraph,
            seed=42,
            k=max(0.9, 1.8 / max(1.0, math.sqrt(len(component_nodes)))),
            iterations=300,
            weight=None,
            scale=1.0,
        )

        spring_x = np.array([float(spring_raw[int(mid)][0]) for mid in component_nodes], dtype=float)
        spring_y = np.array([float(spring_raw[int(mid)][1]) for mid in component_nodes], dtype=float)

        if not np.all(np.isfinite(spring_x)) or np.ptp(spring_x) <= 1e-12:
            spring_x = np.linspace(-1.0, 1.0, len(component_nodes), dtype=float)
        if not np.all(np.isfinite(spring_y)) or np.ptp(spring_y) <= 1e-12:
            spring_y = np.linspace(-1.0, 1.0, len(component_nodes), dtype=float)

        x_norm = (spring_x - float(np.mean(spring_x))) / max(1e-12, float(np.max(np.abs(spring_x - np.mean(spring_x)))))
        y_norm = (spring_y - float(np.mean(spring_y))) / max(1e-12, float(np.max(np.abs(spring_y - np.mean(spring_y)))))
        x_half_span = max(NODE_SEP_UNITS * 1.12, NODE_SEP_UNITS * max(1.45, math.sqrt(len(component_nodes)) * 0.92))

        if have_y:
            y_targets = {}
            for i, mid in enumerate(sorted_by_y):
                if mid in ys:
                    y_targets[int(mid)] = float(ys[mid])
                else:
                    frac = 0.0 if len(sorted_by_y) == 1 else i / (len(sorted_by_y) - 1)
                    y_targets[int(mid)] = float(y_min + frac * (y_max - y_min))
        else:
            y_targets = {
                int(mid): float(y_min + ((idx / max(1, len(component_nodes) - 1)) * (y_max - y_min)))
                for idx, mid in enumerate(component_nodes)
            }

        sim_rough = {}
        for idx, mid in enumerate(component_nodes):
            mol_id = int(mid)
            sim_rough[mol_id] = (
                float(x_norm[idx] * x_half_span),
                float(y_min + ((y_norm[idx] + 1.0) * 0.5) * (y_max - y_min)),
            )
        _separate_overlaps(sim_rough, min_dist=NODE_DIAMETER_UNITS * max(1.12, 1.03 + 0.008 * float(n)), iterations=68, step=0.40)
        similarity_positions = {int(mol_id): (float(x), float(y)) for mol_id, (x, y) in sim_rough.items()}

        act_rough = {}
        x_spacing = float(NODE_SEP_UNITS * 0.96)
        start_x = - (len(sorted_by_y) - 1) * 0.5 * x_spacing
        same_y_bucket = defaultdict(list)
        for i, mid in enumerate(sorted_by_y):
            mol_id = int(mid)
            x = start_x + i * x_spacing
            y = float(y_targets[mol_id])
            if have_y and mol_id in ys:
                same_y_bucket[ys[mol_id]].append(mol_id)
            act_rough[mol_id] = (float(x), y)
        if have_y:
            for _, mids in same_y_bucket.items():
                if len(mids) > 1:
                    jitter_amp = 0.24 * NODE_SEP_UNITS
                    step = 0 if len(mids) == 1 else (2 * jitter_amp) / max(1, (len(mids) - 1))
                    start = -jitter_amp
                    for j, mid in enumerate(mids):
                        x0, y0 = act_rough[mid]
                        act_rough[mid] = (x0 + start + j * step, y0)
        _separate_overlaps(act_rough, min_dist=NODE_DIAMETER_UNITS * max(1.15, 1.05 + 0.008 * float(n)), iterations=68, step=0.42)
        max_vertical_drift = NODE_DIAMETER_UNITS * 0.10
        for mol_id, (x, y) in act_rough.items():
            y_target = float(y_targets[mol_id])
            y_adj = y_target + max(-max_vertical_drift, min(max_vertical_drift, float(y) - y_target))
            activity_positions[mol_id] = (float(x), float(y_adj))

    circle_positions = {}
    if component_nodes:
        count = len(component_nodes)
        cx = 0.0
        cy = float((y_min + y_max) * 0.5)
        circle_render_scale = float(max(0.6, min(2.4, float(state.get("mmpa_render_scale", 2.4)))))
        circle_node_radius = (NODE_DIAMETER_UNITS * 0.5) * circle_render_scale * 1.14
        if count <= 1:
            radius = 0.0
        elif count == 2:
            radius = circle_node_radius * 1.15
        else:
            # Minimal circle radius so adjacent nodes fit on the circumference without overlap.
            radius = (circle_node_radius / max(1e-6, math.sin(math.pi / count))) * 1.04
        ordered_circle_nodes = list(sorted_by_y) if sorted_by_y else list(component_nodes)
        for i, mid in enumerate(ordered_circle_nodes):
            angle = (2.0 * math.pi * i / max(1, count)) - (math.pi / 2.0)
            circle_positions[int(mid)] = (
                float(cx + math.cos(angle) * radius),
                float(cy + math.sin(angle) * radius),
            )

    state["mmpa__positions_circle"] = circle_positions
    state["mmpa__positions_similarity"] = similarity_positions
    state["mmpa__positions_activity"] = activity_positions
    state["mmpa__min_node_distance"] = float(NODE_DIAMETER_UNITS * max(1.15, 1.05 + 0.008 * float(n)))
    state["mmpa_edge_length_scale"] = float(max(0.6, min(4.0, float(state.get("mmpa_edge_length_scale", 2.3)))))
    state["mmpa_render_scale"] = float(max(0.6, min(2.4, float(state.get("mmpa_render_scale", 2.4)))))
    positions = _blend_mmpa_positions(state, float(state.get("mmpa_layout_balance", 1.0)))
    set_loading_screen_progress(state, 1)

    component_core, component_highlight_sets = _compute_component_rgd_highlights(component_nodes, state)
    state["mmpa__component_core"] = component_core
    state["mmpa__component_highlight_sets"] = component_highlight_sets

    # --- TEXTURES: adaptive sizes to keep interaction fluid with many nodes ---
    node_lods_base = {}   # {mol_id: {1024: tag}}
    node_lods_high = {}   # {mol_id: {1024: tag}}
    node_series_map = {}  # {mol_id: image_series_id}
    texture_render_specs = {}
    hover_render_inputs = {}
    base_lod_size, hover_lod_size, node_ss_factor = _get_mmpa_texture_sizes(len(component_nodes))

    # save RDKit style (for consistency between base/highlight)
    state["mmpa__lod_style"] = {"RDW": RDW, "RDH": RDH, "LEGEND_FS": LEGEND_FS, "BW": BW, "INNER_MARGIN": INNER_MARGIN}
    state["mmpa__texture_render_specs"] = texture_render_specs
    state["mmpa__hover_render_inputs"] = hover_render_inputs
    state["_mmpa_lod_maybe_swap"] = None
    state["mmpa_detail_scale_quantized"] = _quantize_mmpa_detail_scale(float(state.get("mmpa_detail_scale", MMPA_DETAIL_SCALE_DEFAULT)))
    texture_render_specs["__detail_scale__"] = float(state["mmpa_detail_scale_quantized"])

    live_textures = []
    total_nodes = max(1, len(component_nodes))

    for idx, mol_id in enumerate(component_nodes, start=1):
        smiles = state["mol_smiles_dict"].get(int(mol_id))
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        state.setdefault("mmpa__hover_info_text", {})[int(mol_id)] = _make_mmpa_hover_label_text(state, int(mol_id))
        base_highlight = component_highlight_sets.get(int(mol_id), ([], [], [], [], []))

        # Base
        lod_base = _build_node_lods(
            mol=mol, label_str="",
            rdw=RDW, rdh=RDH, legend_fs=LEGEND_FS, bond_w=BW,
            inner_margin_px=max(12, int(INNER_MARGIN * 0.72)), ss_factor=node_ss_factor, lod_size=base_lod_size,
            highlight=base_highlight,
            highlight_style="base",
            texture_render_specs=texture_render_specs,
        )
        node_lods_base[mol_id] = lod_base
        live_textures.extend(list(lod_base.values()))

        # Hover texture is precomputed once per node using the same component-level core/variable highlighting.
        hover_render_inputs[mol_id] = _make_mmpa_texture_spec(
            mol=mol,
            highlight=base_highlight,
            highlight_style="hover",
            lod=hover_lod_size,
            rdw=RDW,
            rdh=RDH,
            legend_fs=LEGEND_FS,
            bond_w=BW,
            margin_px=max(12, int(INNER_MARGIN * 0.72)),
            ss_factor=node_ss_factor,
        )
        lod_high = _build_node_lods(
            mol=mol, label_str=_make_mmpa_hover_label_text(state, int(mol_id)),
            rdw=RDW, rdh=RDH, legend_fs=LEGEND_FS, bond_w=BW,
            inner_margin_px=max(12, int(INNER_MARGIN * 0.72)), ss_factor=node_ss_factor, lod_size=hover_lod_size,
            highlight=base_highlight,
            highlight_style="hover",
            texture_render_specs=texture_render_specs,
        )
        node_lods_high[mol_id] = lod_high
        live_textures.extend(list(lod_high.values()))
        if idx % max(1, total_nodes // 25) == 0 or idx == total_nodes:
            set_loading_screen_progress(state, 1 + ((idx / total_nodes) * 98))

    state["mmpa__live_textures"] = live_textures
    set_loading_screen_progress(state, 99)

    # --- plot construction ---
    plot_area = "mmpa_network_plot_canvas"
    plot_tag = "mmpa_network_plot"
    axis_x_tag = "mmpa_network_x"
    axis_y_tag = "mmpa_network_y"

    if dpg.does_item_exist(plot_area):
        dpg.delete_item(plot_area, children_only=True)
    else:
        return

    plot_id = dpg.add_plot(
        tag=plot_tag, parent=plot_area, height=-1, width=-1,
        no_menus=True, no_mouse_pos=True, no_frame=True, no_title=True,
        equal_aspects=True
    )

    axis_x = dpg.add_plot_axis(
        dpg.mvXAxis, tag=axis_x_tag, parent=plot_id,
        no_highlight=True, no_tick_labels=True, no_gridlines=True, no_tick_marks=True
    )
    axis_y = dpg.add_plot_axis(
        dpg.mvYAxis, tag=axis_y_tag, parent=plot_id,
        label="",
        no_highlight=True, no_tick_labels=True, no_gridlines=True, no_tick_marks=True
    )
    set_loading_screen_progress(state, 99)

    # anchors to keep a stable viewport while blending layouts
    all_layout_positions = (
        list((state.get("mmpa__positions_circle", {}) or {}).values())
        + list((state.get("mmpa__positions_similarity", {}) or {}).values())
        + list((state.get("mmpa__positions_activity", {}) or {}).values())
    )
    if all_layout_positions:
        x_vals_anchor = [float(p[0]) for p in all_layout_positions]
        y_vals_anchor = [float(p[1]) for p in all_layout_positions]
        x_pad = max(NODE_DIAMETER_UNITS * 1.3, (max(x_vals_anchor) - min(x_vals_anchor)) * 0.05 if len(x_vals_anchor) > 1 else NODE_DIAMETER_UNITS * 1.3)
        y_pad = max(NODE_DIAMETER_UNITS * 1.3, (max(y_vals_anchor) - min(y_vals_anchor)) * 0.08 if len(y_vals_anchor) > 1 else NODE_DIAMETER_UNITS * 1.3)
        x_low_anchor = min(x_vals_anchor) - x_pad
        x_high_anchor = max(x_vals_anchor) + x_pad
        y_low_anchor = min(y_vals_anchor) - y_pad
        y_high_anchor = max(y_vals_anchor) + y_pad
        dpg.add_line_series([x_low_anchor, x_low_anchor], [y_low_anchor, y_low_anchor], parent=axis_y, label="")
        dpg.add_line_series([x_high_anchor, x_high_anchor], [y_high_anchor, y_high_anchor], parent=axis_y, label="")

    # edges
    edge_series_map = {}
    edge_keys_in_order = []
    for u, v in subgraph.edges():
        if u in positions and v in positions:
            x0, y0 = positions[u]; x1, y1 = positions[v]
            sid = dpg.add_line_series([x0, x1], [y0, y1], parent=axis_y, label="", segments=True, loop=True, no_clip=True)
            edge_key = (int(u), int(v))
            edge_series_map[edge_key] = sid
            edge_keys_in_order.append(edge_key)
    state["mmpa__edge_series_order"] = edge_keys_in_order
    state["mmpa__edge_series_map"] = edge_series_map
    _refresh_mmpa_edge_colors(state)
    set_loading_screen_progress(state, 99)

    # nodes (start with LOD 512 base)
    base_r_units = NODE_DIAMETER_UNITS / 2.0
    render_scale = float(state.get("mmpa_render_scale", 2.4))
    r_units = base_r_units * render_scale
    for idx, (mol_id, (x, y)) in enumerate(positions.items(), start=1):
        tex_tag = node_lods_base[mol_id].get(1024) or next(iter(node_lods_base[mol_id].values()))
        sid = dpg.add_image_series(tex_tag, bounds_min=[x - r_units, y - r_units], bounds_max=[x + r_units, y + r_units], parent=axis_y, label="")
        node_series_map[mol_id] = sid
        if idx % max(1, total_nodes // 20) == 0 or idx == total_nodes:
            set_loading_screen_progress(state, 99)

    # fit
    dpg.fit_axis_data(axis_y)
    dpg.fit_axis_data(axis_x)
    set_loading_screen_progress(state, 99)

    # --- hover (instant swap: base<->highlight) ---
    state["mmpa__node_series_map"] = node_series_map
    state["mmpa__edge_series_map"] = edge_series_map
    state["mmpa__node_radius_units_base"] = base_r_units
    state["mmpa__node_radius_units"] = r_units
    state["mmpa_network_refresh_colors"] = lambda: _refresh_mmpa_edge_colors(state)
    _setup_hover_interaction(
        state=state,
        axis_y_tag=axis_y_tag,
        positions=positions,
        edges=list(subgraph.edges()),
        node_radius_units=r_units,
        node_series_map=node_series_map,
        node_lods_base=node_lods_base,
        node_lods_high=node_lods_high
    )
    set_loading_screen_progress(state, 99)

    # --- keep the hover swap logic in the same direct style used by prova.py ---
    _apply_mmpa_layout_positions(state, float(state.get("mmpa_layout_balance", 1.0)))
    set_loading_screen_progress(state, 99)

    dpg.bind_item_theme(plot_tag, apply_mmpa_network_theme())
    set_loading_screen_progress(state, 100)

    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")


# -----------------------------------------------------------------------------
# 4. Make circular node
# -----------------------------------------------------------------------------
def make_circular_node(
    img_rgba: Any,
    diameter: int,
    margin_px: int = 40,
    ss_factor: int = 2
) -> Any:
    """
    Builds a crisp circular RGBA node using supersampling:
    - Renders into a (diameter*ss_factor) canvas, applies a high-res circular mask,
      draws a clean white border, then downsamples with LANCZOS to the target 'diameter'.
    - 'img_rgba' is the RDKit-rendered PNG as PIL RGBA.

    Args:
        img_rgba: PIL RGBA image
        diameter (int): final output diameter in pixels
        margin_px (int): inner padding
        ss_factor (int): supersampling factor (2 or 3 recommended)

    Returns:
        pilImage.Image: RGBA circular node at the requested diameter.
    """
    import math
    D = max(8, int(diameter) * ss_factor)
    M = max(2, int(margin_px) * ss_factor)

    # Hi-res transparent canvas
    canvas = pilImage.new("RGBA", (D, D), (255, 255, 255, 0))

    # Fit molecule image inside margins on hi-res canvas
    target_w = max(1, D - 2 * M)
    target_h = max(1, D - 2 * M)
    fitted = ImageOps.contain(img_rgba, (target_w, target_h), method=pilImage.LANCZOS)
    off_x = (D - fitted.width) // 2
    off_y = (D - fitted.height) // 2

    tmp = pilImage.new("RGBA", (D, D), (255, 255, 255, 0))
    tmp.alpha_composite(fitted, (off_x, off_y))

    # Circular alpha mask (hi-res, antialiased when downsampling)
    mask = pilImage.new("L", (D, D), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, D - 1, D - 1), fill=255)

    # Keep only the pixels that are both inside the circular crop and already
    # opaque in the molecule image. This avoids an opaque disk behind the mol.
    circ = tmp.copy()
    img_alpha = tmp.getchannel("A")
    circ.putalpha(ImageChops.multiply(img_alpha, mask))

    # Downsample to final diameter (antialiased)
    final_img = circ.resize((diameter, diameter), resample=pilImage.LANCZOS)
    arr = np.array(final_img, dtype=np.uint8)
    low_alpha = arr[..., 3] < 40
    arr[low_alpha, 0] = 255
    arr[low_alpha, 1] = 255
    arr[low_alpha, 2] = 255
    arr[low_alpha, 3] = 0
    final_img = pilImage.fromarray(arr, mode="RGBA")
    return final_img


# -----------------------------------------------------------------------------
# 5. Separate overlaps
# -----------------------------------------------------------------------------
def _separate_overlaps(
    points_dict: Any,
    min_dist: Any,
    iterations: int = 25,
    step: float = 0.5
) -> None:
    import math
    keys = list(points_dict.keys())
    for _ in range(iterations):
        moved = False
        for i in range(len(keys)):
            ni = keys[i]
            xi, yi = points_dict[ni]
            for j in range(i + 1, len(keys)):
                nj = keys[j]
                xj, yj = points_dict[nj]
                dx = xj - xi
                dy = yj - yi
                d2 = dx * dx + dy * dy
                if d2 == 0.0:
                    dx, dy = 1e-3, 0.0
                    d = 1e-3
                else:
                    d = math.sqrt(d2)
                if d < min_dist:
                    push = (min_dist - d) * 0.5 * step
                    ux, uy = dx / d, dy / d
                    xj += ux * push
                    yj += uy * push
                    xi -= ux * push
                    yi -= uy * push
                    points_dict[ni] = (xi, yi)
                    points_dict[nj] = (xj, yj)
                    moved = True
        if not moved:
            break


def _separate_overlaps_elliptic(
    points_dict: Any,
    radius_x: float,
    radius_y: float,
    iterations: int = 30,
    step: float = 0.4,
) -> None:
    keys = list(points_dict.keys())
    rx = max(1e-9, float(radius_x))
    ry = max(1e-9, float(radius_y))
    for _ in range(iterations):
        moved = False
        for i in range(len(keys)):
            ni = keys[i]
            xi, yi = points_dict[ni]
            for j in range(i + 1, len(keys)):
                nj = keys[j]
                xj, yj = points_dict[nj]
                dx = float(xj) - float(xi)
                dy = float(yj) - float(yi)
                nxv = dx / (2.0 * rx)
                nyv = dy / (2.0 * ry)
                d = math.sqrt(nxv * nxv + nyv * nyv)
                if d < 1.0:
                    if d <= 1e-9:
                        ux, uy = 1.0, 0.0
                    else:
                        ux, uy = nxv / d, nyv / d
                    push = (1.0 - d) * 0.5 * step
                    xi -= ux * push * 2.0 * rx
                    yi -= uy * push * 2.0 * ry
                    xj += ux * push * 2.0 * rx
                    yj += uy * push * 2.0 * ry
                    points_dict[ni] = (xi, yi)
                    points_dict[nj] = (xj, yj)
                    moved = True
        if not moved:
            break


# -----------------------------------------------------------------------------
# 6. Clear previous rendering
# -----------------------------------------------------------------------------
def _clear_previous_rendering(state: dict[str, Any]) -> None:
    clear_mmpa_network_memory(state, clear_plot=True, clear_structure=False)


def _blend_mmpa_positions(state: dict[str, Any], alpha: float) -> dict[int, tuple[float, float]]:
    alpha = float(max(0.0, min(2.0, alpha)))
    circle_positions = state.get("mmpa__positions_circle", {}) or {}
    similarity_positions = state.get("mmpa__positions_similarity", {}) or {}
    activity_positions = state.get("mmpa__positions_activity", {}) or {}
    if alpha <= 1.0:
        left = circle_positions
        right = similarity_positions
        local_alpha = alpha
    else:
        left = similarity_positions
        right = activity_positions
        local_alpha = alpha - 1.0
    common_nodes = [int(mid) for mid in left.keys() if mid in right]
    blended = {
        int(mid): (
            (1.0 - local_alpha) * float(left[mid][0]) + local_alpha * float(right[mid][0]),
            (1.0 - local_alpha) * float(left[mid][1]) + local_alpha * float(right[mid][1]),
        )
        for mid in common_nodes
    }
    edge_length_scale = float(state.get("mmpa_edge_length_scale", 1.0))
    if blended and abs(edge_length_scale - 1.0) > 1e-9:
        cx = sum(float(p[0]) for p in blended.values()) / len(blended)
        cy = sum(float(p[1]) for p in blended.values()) / len(blended)
        blended = {
            int(mid): (
                cx + (float(pos[0]) - cx) * edge_length_scale,
                cy + (float(pos[1]) - cy) * edge_length_scale,
            )
            for mid, pos in blended.items()
        }
    min_dist = float(state.get("mmpa__min_node_distance", 0.0) or 0.0)
    if blended and min_dist > 0:
        _separate_overlaps(blended, min_dist=min_dist, iterations=18, step=0.32)
    return blended


def _apply_mmpa_layout_positions(state: dict[str, Any], alpha: float) -> None:
    positions = _blend_mmpa_positions(state, float(max(0.0, min(2.0, alpha))))
    if not positions:
        return

    state["mmpa_layout_balance"] = float(max(0.0, min(2.0, alpha)))

    node_series_map = state.get("mmpa__node_series_map", {}) or {}
    edge_series_map = state.get("mmpa__edge_series_map", {}) or {}
    base_r_units = float(state.get("mmpa__node_radius_units_base", state.get("mmpa__node_radius_units", 0.0)) or 0.0)
    render_scale = float(state.get("mmpa_render_scale", 2.4))
    r_units = base_r_units * render_scale
    state["mmpa__node_radius_units"] = r_units
    state["mmpa__hover_radius"] = r_units
    state["mmpa__current_node_extents"] = (float(r_units), float(r_units))
    _separate_overlaps_elliptic(positions, radius_x=r_units * 1.14, radius_y=r_units * 1.14, iterations=36, step=0.42)
    state["mmpa__hover_positions"] = positions

    for mid, series_id in node_series_map.items():
        if not dpg.does_item_exist(series_id) or mid not in positions:
            continue
        x, y = positions[mid]
        try:
            dpg.configure_item(series_id, bounds_min=[x - r_units, y - r_units], bounds_max=[x + r_units, y + r_units])
        except Exception:
            pass

    for (u, v), series_id in edge_series_map.items():
        if not dpg.does_item_exist(series_id) or u not in positions or v not in positions:
            continue
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        try:
            dpg.set_value(series_id, [[x0, x1], [y0, y1]])
        except Exception:
            try:
                dpg.configure_item(series_id, x=[x0, x1], y=[y0, y1])
            except Exception:
                pass

    _clear_hover_overlay(state)
    state["mmpa__hover_last"] = None
    if callable(state.get("_mmpa_lod_maybe_swap")):
        state["_mmpa_lod_maybe_swap"]()


def _on_mmpa_layout_balance_change(state: dict[str, Any], app_data: Any) -> None:
    try:
        alpha = float(app_data)
    except Exception:
        alpha = float(state.get("mmpa_layout_balance", 1.0))
    _apply_mmpa_layout_positions(state, alpha)


def _on_mmpa_edge_length_scale_change(state: dict[str, Any], app_data: Any) -> None:
    try:
        scale = float(app_data)
    except Exception:
        scale = float(state.get("mmpa_edge_length_scale", 2.3))
    state["mmpa_edge_length_scale"] = max(0.6, min(4.0, scale))
    _apply_mmpa_layout_positions(state, float(state.get("mmpa_layout_balance", 1.0)))


def _on_mmpa_render_scale_change(state: dict[str, Any], app_data: Any) -> None:
    try:
        scale = float(app_data)
    except Exception:
        scale = float(state.get("mmpa_render_scale", 2.4))
    state["mmpa_render_scale"] = max(0.6, min(2.4, scale))
    _apply_mmpa_layout_positions(state, float(state.get("mmpa_layout_balance", 1.0)))


def _apply_mmpa_detail_scale(state: dict[str, Any], scale: float) -> None:
    quantized = _quantize_mmpa_detail_scale(scale)
    if state.get("mmpa_detail_scale_quantized") == quantized:
        return

    draw_loading_screen(state, bg=False)
    try:
        state["mmpa_detail_scale_quantized"] = quantized
        specs = state.get("mmpa__texture_render_specs", {}) or {}
        specs["__detail_scale__"] = float(quantized)
        for tag, spec in specs.items():
            if tag == "__detail_scale__":
                continue
            if not dpg.does_item_exist(tag):
                continue
            cache = spec.setdefault("_flat_cache", {})
            flat = cache.get(quantized)
            if flat is None:
                arr = _render_mmpa_texture_from_spec(spec, float(quantized))
                flat = arr.reshape(-1)
                cache[quantized] = flat
                if len(cache) > 3:
                    oldest_key = next(iter(cache))
                    if oldest_key != quantized:
                        cache.pop(oldest_key, None)
            dpg.set_value(tag, flat)
    finally:
        state["mmpa_detail_scale_pending"] = None
        state["mmpa_detail_scale_dirty"] = False
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")


def _on_mmpa_detail_scale_change(state: dict[str, Any], app_data: Any) -> None:
    try:
        scale = float(app_data)
    except Exception:
        scale = float(state.get("mmpa_detail_scale", MMPA_DETAIL_SCALE_DEFAULT))
    state["mmpa_detail_scale"] = max(MMPA_DETAIL_SCALE_MIN, min(MMPA_DETAIL_SCALE_MAX, scale))
    state["mmpa_detail_scale_pending"] = state["mmpa_detail_scale"]
    state["mmpa_detail_scale_dirty"] = True


def _on_mmpa_detail_scale_release(state: dict[str, Any]) -> None:
    if not state.get("mmpa_detail_scale_dirty", False):
        return
    pending = state.get("mmpa_detail_scale_pending", state.get("mmpa_detail_scale", MMPA_DETAIL_SCALE_DEFAULT))
    _apply_mmpa_detail_scale(state, float(pending))


# -----------------------------------------------------------------------------
# 7. Compute mcs for component
# -----------------------------------------------------------------------------
def _compute_mcs_for_mols(mols: list[Any]) -> Any:
    mols = [Chem.Mol(m) for m in mols if m is not None]
    if len(mols) < 2:
        return None

    try:
        params = rdFMCS.MCSParameters()
        if hasattr(params, "AtomTyper"):
            params.AtomTyper = rdFMCS.AtomCompare.CompareElements
        if hasattr(params, "BondTyper"):
            params.BondTyper = rdFMCS.BondCompare.CompareOrderExact
        if hasattr(params, "AtomCompareParameters"):
            params.AtomCompareParameters.MatchValences = True
            params.AtomCompareParameters.MatchChiralTag = True
            params.AtomCompareParameters.CompleteRingsOnly = True
            params.AtomCompareParameters.RingMatchesRingOnly = True
        if hasattr(params, "BondCompareParameters"):
            params.BondCompareParameters.CompleteRingsOnly = True
            params.BondCompareParameters.RingMatchesRingOnly = True
        if hasattr(params, "MaximizeBonds"):
            params.MaximizeBonds = False
        if hasattr(params, "Threshold"):
            params.Threshold = 1.0
        elif hasattr(params, "threshold"):
            params.threshold = 1.0
        mcs_result = rdFMCS.FindMCS(mols, parameters=params)
    except Exception:
        try:
            mcs_result = rdFMCS.FindMCS(
                mols,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                matchValences=True,
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
                threshold=1.0,
            )
        except Exception:
            return None

    try:
        return Chem.MolFromSmarts(mcs_result.smartsString) if mcs_result and mcs_result.smartsString else None
    except Exception:
        return None


def _compute_mcs_for_component(component_nodes: Any, state: dict[str, Any]) -> Any:
    mols = []
    for mol_id in component_nodes:
        smi = state["mol_smiles_dict"].get(int(mol_id))
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)

    return _compute_mcs_for_mols(mols)


def _compute_component_rgd_highlights(component_nodes: Any, state: dict[str, Any]) -> tuple[Any, dict[int, Any]]:
    mol_entries = []
    for mol_id in component_nodes:
        smi = state["mol_smiles_dict"].get(int(mol_id))
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol_entries.append((int(mol_id), mol))

    if not mol_entries:
        return None, {}

    core_mol = _compute_mcs_for_mols([mol for _, mol in mol_entries])
    if core_mol is None:
        empty_map = {
            int(mol_id): ([], [], list(range(mol.GetNumAtoms())), list(range(mol.GetNumBonds())), [])
            for mol_id, mol in mol_entries
        }
        return None, empty_map

    highlight_map: dict[int, Any] = {}
    try:
        rgd_rows, fails = rdRGroupDecomposition.RGroupDecompose(
            [Chem.Mol(core_mol)],
            [Chem.Mol(mol) for _, mol in mol_entries],
            asSmiles=False,
            asRows=True,
            options=_build_mmpa_rgd_parameters(len(mol_entries)),
        )
        fail_set = set(int(i) for i in fails)
        row_idx = 0
        for i, (mol_id, mol) in enumerate(mol_entries):
            if i in fail_set or row_idx >= len(rgd_rows):
                highlight_map[int(mol_id)] = _compute_highlight_sets(Chem.Mol(mol), core_mol)
                continue
            row = rgd_rows[row_idx]
            row_idx += 1
            row_core = row.get("Core") if isinstance(row, dict) else None
            highlight = _compute_highlight_sets(Chem.Mol(mol), row_core if row_core is not None else core_mol)
            if not highlight[0] and core_mol is not None:
                highlight = _compute_highlight_sets(Chem.Mol(mol), core_mol)
            highlight_map[int(mol_id)] = highlight
    except Exception:
        for mol_id, mol in mol_entries:
            highlight_map[int(mol_id)] = _compute_highlight_sets(Chem.Mol(mol), core_mol)

    return core_mol, highlight_map


# -----------------------------------------------------------------------------
# 8. Compute highlight sets
# -----------------------------------------------------------------------------
def _compute_highlight_sets(mol: Any, mcs_mol: Any) -> Any:
    if mcs_mol:
        match_atoms = mol.GetSubstructMatch(mcs_mol)
        mcs_atoms = list(match_atoms) if match_atoms else []
    else:
        mcs_atoms = []

    mcs_bonds = []
    if mcs_atoms:
        mcs_set = set(mcs_atoms)
        for bnd in mol.GetBonds():
            b, e = bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()
            if b in mcs_set and e in mcs_set:
                mcs_bonds.append(bnd.GetIdx())

    mcs_set = set(mcs_atoms)
    variable_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in mcs_set]

    variable_bonds = []
    link_bonds = []
    for bnd in mol.GetBonds():
        b, e = bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()
        in_mcs_b = b in mcs_set
        in_mcs_e = e in mcs_set
        if not in_mcs_b and not in_mcs_e:
            variable_bonds.append(bnd.GetIdx())
        elif in_mcs_b ^ in_mcs_e:
            link_bonds.append(bnd.GetIdx())

    return mcs_atoms, mcs_bonds, variable_atoms, variable_bonds, link_bonds


def _merge_highlight_sets_for_hovered(mol: Any, highlight_sets: list[Any]) -> Any:
    if mol is None:
        return [], [], [], [], []
    if not highlight_sets:
        return [], [], list(range(mol.GetNumAtoms())), list(range(mol.GetNumBonds())), []

    mcs_atoms = sorted({idx for hs in highlight_sets for idx in hs[0]})
    mcs_bonds = sorted({idx for hs in highlight_sets for idx in hs[1]})
    mcs_set = set(mcs_atoms)

    variable_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in mcs_set]
    variable_bonds = []
    link_bonds = []
    for bnd in mol.GetBonds():
        b, e = bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()
        in_mcs_b = b in mcs_set
        in_mcs_e = e in mcs_set
        if not in_mcs_b and not in_mcs_e:
            variable_bonds.append(bnd.GetIdx())
        elif in_mcs_b ^ in_mcs_e:
            link_bonds.append(bnd.GetIdx())

    return mcs_atoms, mcs_bonds, variable_atoms, variable_bonds, link_bonds


# -----------------------------------------------------------------------------
# 9. Populate group buttons
# -----------------------------------------------------------------------------
def _populate_group_buttons(state: dict[str, Any]) -> None:
    container = "mmpa_group_sidebar_window"
    if not dpg.does_item_exist(container):
        return

    dpg.delete_item(container, children_only=True)

    components = state.get("mmpa__components", [])
    if not components:
        dpg.add_text("No connected components found.", parent=container)
        return

    for idx, comp in enumerate(components, start=1):
        size = len(comp)
        label = f"MMPA Group {idx} ({size} molecules)"
        dpg.add_button(
            label=label,
            parent=container,
            callback=_render_group_callback,
            user_data={"state": state, "group_index": idx - 1},
            width=-1
        )
        dpg.add_spacer(height=4, parent=container)  # <— IMPORTANT: parent specified

    # Auto-render an empty plot area
    plot_area = "mmpa_network_plot_canvas" if dpg.does_item_exist("mmpa_network_plot_canvas") else "mmpa_network_plot_window"
    plot_tag = "mmpa_network_plot"
    axis_x_tag = "mmpa_network_x"
    axis_y_tag = "mmpa_network_y"

    plot_id = dpg.add_plot(
        tag=plot_tag, parent=plot_area, height=-1, width=-1,
        no_menus=True, no_mouse_pos=True, no_frame=True, no_title=True,
        equal_aspects=True
    )

    axis_x = dpg.add_plot_axis(
        dpg.mvXAxis, tag=axis_x_tag, parent=plot_id,
        no_highlight=True, no_tick_labels=True, no_gridlines=True, no_tick_marks=True
    )
    axis_y = dpg.add_plot_axis(
        dpg.mvYAxis, tag=axis_y_tag, parent=plot_id,
        label=state.get("mmpa_y_axis_label", "Activity (pValue)"),
        no_highlight=False, no_tick_labels=False, no_gridlines=True, no_tick_marks=False
    )

    dpg.bind_item_theme(plot_tag, apply_mmpa_network_theme())

    


# -----------------------------------------------------------------------------
# 10. Read activity value
# -----------------------------------------------------------------------------
def _read_activity_value(state: dict[str, Any], mol_id: Any) -> Any:
    import math

    # Prefer the authoritative DataFrame
    df = state.get("mmpa_dataframe", None)
    if df is not None and "Mol_ID" in df.columns and "pValue" in df.columns:
        try:
            row = df.loc[df["Mol_ID"].astype(int) == int(mol_id)]
            if len(row) == 1:
                val = float(row["pValue"].values[0])
                if not math.isfinite(val) or val == 0.0:
                    return None
                return val
        except Exception:
            pass  # fallback below

    # Fallback to dict (stringy)
    raw = state.get("mol_activity_dict", {}).get(int(mol_id), None)
    if raw is None:
        return None
    s = str(raw).strip()
    if s in ("", "None", "NONE", "N/A", "NA", "nan", "NaN"):
        return None
    try:
        val = float(s)
        if not math.isfinite(val) or val == 0.0:
            return None
        return val
    except Exception:
        return None


def _get_mmpa_node_half_extents(plot_tag: Any, axis_x_tag: Any, axis_y_tag: Any, y_radius_units: float) -> tuple[float, float]:
    y_radius_units = float(max(1e-9, y_radius_units))
    try:
        plot_w, plot_h = dpg.get_item_rect_size(plot_tag)
        x_min, x_max = dpg.get_axis_limits(axis_x_tag)
        y_min, y_max = dpg.get_axis_limits(axis_y_tag)
        x_span = max(1e-9, float(x_max) - float(x_min))
        y_span = max(1e-9, float(y_max) - float(y_min))
        px_per_unit_x = max(1e-9, float(plot_w) / x_span)
        px_per_unit_y = max(1e-9, float(plot_h) / y_span)
        x_radius_units = y_radius_units * (px_per_unit_y / px_per_unit_x)
        return float(x_radius_units), float(y_radius_units)
    except Exception:
        return float(y_radius_units), float(y_radius_units)
    

# -----------------------------------------------------------------------------
# 11. Ensure ring texture
# -----------------------------------------------------------------------------
def _ensure_ring_texture(
    tag: str = "mmpa__ring_texture",
    size_px: int = 256,
    ring_width_px: int = 6,
    color_rgba: tuple[int, int, int, int] = (255, 220, 80, 128),
) -> Any:
    if dpg.does_item_exist(tag):
        return tag

    import numpy as _np
    from math import hypot

    S = int(size_px)
    R = (S - 2) / 2.0
    r_inner = max(1.0, R - ring_width_px)

    arr = _np.zeros((S, S, 4), dtype=_np.uint8)
    cx = cy = (S - 1) / 2.0
    for y in range(S):
        for x in range(S):
            d = hypot(x - cx, y - cy)
            if r_inner <= d <= R:
                arr[y, x] = color_rgba
            else:
                arr[y, x] = (0, 0, 0, 0)

    arrf = (arr.astype(_np.float32) / 255.0).reshape(-1)

    empty_data = np.ones((0 * 0 * 4,), dtype=np.float32)

    if not dpg.does_item_exist(tag):
        dpg.add_static_texture(S, S, arrf, tag=tag, parent="texture_registry")
    else:
        dpg.set_value(tag, empty_data)

    return tag


def _sample_mmpa_activity_color(state: dict[str, Any], ratio: float, alpha: int = 160) -> tuple[int, int, int, int]:
    try:
        colormap = state["colormaps"][state["colormap_continuous"]]
        rgba = dpg.sample_colormap(colormap, max(0.0, min(1.0, float(ratio))))
        if len(rgba) >= 3:
            if max(rgba[0], rgba[1], rgba[2]) <= 1.0:
                return (
                    int(round(float(rgba[0]) * 255)),
                    int(round(float(rgba[1]) * 255)),
                    int(round(float(rgba[2]) * 255)),
                    int(alpha),
                )
            return (
                int(round(float(rgba[0]))),
                int(round(float(rgba[1]))),
                int(round(float(rgba[2]))),
                int(alpha),
            )
    except Exception:
        pass
    return (255, 220, 80, int(alpha))


# -----------------------------------------------------------------------------
# 12. Setup hover interaction
# -----------------------------------------------------------------------------
def _setup_hover_interaction(
    state: dict[str, Any],
    axis_y_tag: str,
    positions: Any,
    edges: Any,
    node_radius_units: Any,
    node_series_map: Any,
    node_lods_base: Any,
    node_lods_high: Any
) -> Any:
    from collections import defaultdict

    # derive the plot from the parent of the Y axis (more reliable than a fixed name)
    if not dpg.does_item_exist(axis_y_tag):
        return
    plot_tag = dpg.get_item_parent(axis_y_tag)  # axis_y -> plot

    axis_x_tag = None
    try:
        for ch in dpg.get_item_children(plot_tag, 1) or []:
            if dpg.get_item_type(ch) == "mvAppItemType::mvPlotAxis":
                cfg = dpg.get_item_configuration(ch)
                if cfg.get("axis") == dpg.mvXAxis:
                    axis_x_tag = ch
                    break
    except Exception:
        axis_x_tag = None

    state["mmpa__hover_items"] = []
    state["mmpa__hover_last"] = None
    state["mmpa__hover_positions"] = positions
    state["mmpa__hover_radius"] = node_radius_units
    state["mmpa__hover_axis_y"] = axis_y_tag
    state["mmpa__hover_axis_x"] = axis_x_tag
    state["mmpa__hover_plot"] = plot_tag
    state["mmpa__lod_series"] = node_series_map
    state["mmpa__lod_base"] = node_lods_base
    state["mmpa__lod_high"] = node_lods_high
    state["mmpa__highlighted_nodes"] = set()
    state["mmpa__hover_throttle_last_t"] = 0.0
    state["mmpa__hover_throttle_last_pos"] = None
    visible_activity_values = [
        float(v) for nid in positions.keys()
        for v in [_read_activity_value(state, nid)]
        if v is not None and math.isfinite(float(v))
    ]
    if visible_activity_values:
        state["mmpa__hover_activity_min"] = float(min(visible_activity_values))
        state["mmpa__hover_activity_max"] = float(max(visible_activity_values))
    else:
        state["mmpa__hover_activity_min"] = None
        state["mmpa__hover_activity_max"] = None
    state["mmpa__ring_textures"] = {}

    # adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    state["mmpa__hover_adj"] = adj

    # ring texture & themes
    ring_tex = _ensure_ring_texture()
    state["mmpa__ring_texture"] = ring_tex
    state["mmpa__ring_textures"][(255, 220, 80, 128)] = ring_tex
    theme_img  = _get_or_make_img_theme()

    # -----------------------------------------------------------------------------
    # 12.1. Apply highlight for
    # -----------------------------------------------------------------------------
    def _apply_highlight_for(nodes_set: Any) -> None:
        highlighted = set(nodes_set)
        state["mmpa__highlighted_nodes"] = highlighted
        for mid, series_id in (state.get("mmpa__lod_series", {}) or {}).items():
            if not dpg.does_item_exist(series_id):
                continue
            if mid in highlighted:
                tex_tag = _ensure_mmpa_hover_texture(state, int(mid))
            else:
                tex_tag = (state.get("mmpa__lod_base", {}) or {}).get(mid, {}).get(1024)
            if tex_tag and dpg.does_item_exist(tex_tag):
                try:
                    dpg.configure_item(series_id, texture_tag=tex_tag)
                except Exception:
                    pass

    # -----------------------------------------------------------------------------
    # 12.2. Safe is item hovered
    # -----------------------------------------------------------------------------
    def _safe_is_item_hovered(item: Any) -> Any:
        if not item or not dpg.does_item_exist(item):
            return False
        try:
            return dpg.is_item_hovered(item)
        except Exception:
            return False

    # -----------------------------------------------------------------------------
    # 12.3. Safe get plot mouse pos
    # -----------------------------------------------------------------------------
    def _safe_get_plot_mouse_pos(p_tag: str) -> Any:
        try:
            return dpg.get_plot_mouse_pos(p_tag)
        except TypeError:
            try:
                return dpg.get_plot_mouse_pos()
            except Exception:
                return None
        except Exception:
            return None

    # -----------------------------------------------------------------------------
    # 12.4. On mouse move
    # -----------------------------------------------------------------------------
    def _on_mouse_move(sender: Any, app_data: Any) -> None:
        p_tag = state.get("mmpa__hover_plot")
        y_axis = state.get("mmpa__hover_axis_y")

        # If the plot (or the axis) no longer exists, exit silently
        if not p_tag or not dpg.does_item_exist(p_tag) or not y_axis or not dpg.does_item_exist(y_axis):
            return

        # If not hovering the plot, clear overlay and reset highlight
        if not _safe_is_item_hovered(p_tag):
            _clear_hover_overlay(state)
            state["mmpa__hover_focus_id"] = None
            if state["mmpa__highlighted_nodes"]:
                _apply_highlight_for(set())
            state["mmpa__hover_last"] = None
            return

        # Mouse position in data space (may fail while redrawing)
        mp = _safe_get_plot_mouse_pos(p_tag)
        if not mp:
            return
        mx, my = mp

        now = time.monotonic()
        last_t = float(state.get("mmpa__hover_throttle_last_t", 0.0) or 0.0)
        last_pos = state.get("mmpa__hover_throttle_last_pos")
        if last_pos is not None:
            dxp = float(mx) - float(last_pos[0])
            dyp = float(my) - float(last_pos[1])
            if (now - last_t) < 0.016 and (dxp * dxp + dyp * dyp) < 1e-4:
                return
        state["mmpa__hover_throttle_last_t"] = now
        state["mmpa__hover_throttle_last_pos"] = (float(mx), float(my))

        # Hit test
        hit = None
        r = max(1e-9, float(state["mmpa__hover_radius"]))
        for nid, (x, y) in state["mmpa__hover_positions"].items():
            dx = mx - x; dy = my - y
            if dx * dx + dy * dy <= (r * 1.05) ** 2:
                hit = nid; break

        if hit == state["mmpa__hover_last"]:
            return

        _clear_hover_overlay(state)

        if hit is None:
            state["mmpa__hover_focus_id"] = None
            if state["mmpa__highlighted_nodes"]:
                _apply_highlight_for(set())
            state["mmpa__hover_last"] = None
            return

        neighbors = state["mmpa__hover_adj"].get(hit, set())
        ring_nodes = {hit} | neighbors
        state["mmpa__hover_focus_id"] = int(hit)

        # ring around nodes
        rr = float(state["mmpa__hover_radius"]) * 1.18
        activity_min = state.get("mmpa__hover_activity_min")
        activity_max = state.get("mmpa__hover_activity_max")
        for nid in ring_nodes:
            x, y = state["mmpa__hover_positions"][nid]
            if dpg.does_item_exist(y_axis):
                activity_value = _read_activity_value(state, nid)
                if (
                    activity_value is not None
                    and activity_min is not None
                    and activity_max is not None
                    and math.isfinite(float(activity_value))
                ):
                    if abs(float(activity_max) - float(activity_min)) <= 1e-12:
                        ratio = 1.0
                    else:
                        ratio = (float(activity_value) - float(activity_min)) / (float(activity_max) - float(activity_min))
                    ring_color = _sample_mmpa_activity_color(state, ratio, alpha=168)
                else:
                    ring_color = (255, 220, 80, 128)
                ring_texture_map = state.setdefault("mmpa__ring_textures", {})
                ring_tex = ring_texture_map.get(ring_color)
                if not ring_tex or not dpg.does_item_exist(ring_tex):
                    ring_tex = _ensure_ring_texture(
                        tag=f"mmpa__ring_texture_{ring_color[0]}_{ring_color[1]}_{ring_color[2]}_{ring_color[3]}",
                        color_rgba=ring_color,
                    )
                    ring_texture_map[ring_color] = ring_tex
                iid = dpg.add_image_series(ring_tex, [x - rr, y - rr], [x + rr, y + rr], parent=y_axis, label="")
                dpg.bind_item_theme(iid, theme_img)
                state["mmpa__hover_items"].append(iid)

        _apply_highlight_for(ring_nodes)
        state["mmpa__hover_last"] = hit

    # handler registry
    htag = "mmpa_hover_mouse_handler"
    if dpg.does_item_exist(htag):
        dpg.delete_item(htag)
    dpg.add_mouse_move_handler(tag=htag, parent="handler_registry", callback=_on_mouse_move)


# -----------------------------------------------------------------------------
# 13. Clear hover overlay
# -----------------------------------------------------------------------------
def _clear_hover_overlay(state: dict[str, Any]) -> None:
    items = state.get("mmpa__hover_items", [])
    for it in items:
        if dpg.does_item_exist(it):
            dpg.delete_item(it)
    state["mmpa__hover_items"] = []


# -----------------------------------------------------------------------------
# 14. Sample / apply edge colors
# -----------------------------------------------------------------------------
def _sample_mmpa_edge_color(state: dict[str, Any], ratio: float) -> Any:
    """
    Sample the active discrete plot colormap and return an RGBA color in 0-255 space.
    """
    try:
        colormap = state["plot_colormaps"][state["colormap_discrete"]]
        rgba = dpg.sample_colormap(colormap, max(0.0, min(1.0, float(ratio))))
        if len(rgba) >= 4:
            if max(rgba[0], rgba[1], rgba[2], rgba[3]) <= 1.0:
                return (
                    int(round(float(rgba[0]) * 255)),
                    int(round(float(rgba[1]) * 255)),
                    int(round(float(rgba[2]) * 255)),
                    int(round(float(rgba[3]) * 255)),
                )
            return (
                int(round(float(rgba[0]))),
                int(round(float(rgba[1]))),
                int(round(float(rgba[2]))),
                int(round(float(rgba[3]))),
            )
    except Exception:
        pass
    return (90, 90, 90, 255)


def _refresh_mmpa_edge_colors(state: dict[str, Any]) -> None:
    """
    Re-apply sampled discrete-colormap colors to the current network edges.
    """
    edge_series_map = state.get("mmpa__edge_series_map", {}) or {}
    if not edge_series_map:
        return
    edge_keys = list(state.get("mmpa__edge_series_order", []) or edge_series_map.keys())
    total = max(1, len(edge_keys))
    for idx, edge_key in enumerate(edge_keys):
        series_id = edge_series_map.get(edge_key)
        if not series_id or not dpg.does_item_exist(series_id):
            continue
        ratio = 0.0 if total <= 1 else idx / float(total - 1)
        color_rgba = _sample_mmpa_edge_color(state, ratio)
        try:
            dpg.bind_item_theme(series_id, _get_or_make_line_theme(color_rgba=color_rgba, thickness=1))
        except Exception:
            pass


# -----------------------------------------------------------------------------
# 15. Get or make line theme
# -----------------------------------------------------------------------------
def _get_or_make_line_theme(color_rgba: Any = (255, 220, 80, 255), thickness: int = 1) -> Any:
    tag = f"mmpa__line_theme_{color_rgba}_{thickness}"
    if dpg.does_item_exist(tag):
        return tag
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, color_rgba, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, thickness, category=dpg.mvThemeCat_Plots)
    return tag


# -----------------------------------------------------------------------------
# 16. Get or make img theme
# -----------------------------------------------------------------------------
def _get_or_make_img_theme() -> Any:
    """
    Neutral theme for image_series (placeholder, kept for symmetry/extension).
    
    Args:
        None.
    
    Returns:
        Any: Value produced by the routine.
    """
    tag = "mmpa__img_theme_neutral"
    if dpg.does_item_exist(tag):
        return tag
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvImageSeries):
            pass
    return tag


# -----------------------------------------------------------------------------
# 17. Build node lods
# -----------------------------------------------------------------------------
def _build_node_lods(
    mol: Any,
    label_str: str,
    rdw: int,
    rdh: int,
    legend_fs: int,
    bond_w: int,
    inner_margin_px: int,
    ss_factor: int = 2,
    lod_size: int = 1024,
    highlight: Any = None,
    highlight_style: str = "hover",
    texture_render_specs: dict[str, Any] | None = None,
) -> Any:
    lod = int(max(128, lod_size))
    detail_scale = float(_quantize_mmpa_detail_scale(
        texture_render_specs.get("__detail_scale__", MMPA_DETAIL_SCALE_DEFAULT)
        if texture_render_specs is not None
        else MMPA_DETAIL_SCALE_DEFAULT
    ))
    arr = _render_mmpa_texture_from_spec(
        {
            "mol": Chem.Mol(mol),
            "highlight": highlight,
            "highlight_style": highlight_style,
            "label_text": str(label_str or ""),
            "lod": lod,
            "rdw": int(rdw),
            "rdh": int(rdh),
            "legend_fs": max(18, int(legend_fs)),
            "bond_w": max(1, int(round(bond_w))),
            "margin_px": max(16, int(inner_margin_px)),
            "ss_factor": ss_factor,
        },
        detail_scale,
    )

    tag = dpg.generate_uuid()
    dpg.add_dynamic_texture(lod, lod, arr.reshape(-1), tag=tag, parent="texture_registry")
    if texture_render_specs is not None:
        texture_render_specs[tag] = {
            "mol": Chem.Mol(mol),
            "highlight": highlight,
            "highlight_style": highlight_style,
            "label_text": str(label_str or ""),
            "lod": lod,
            "rdw": int(rdw),
            "rdh": int(rdh),
            "legend_fs": max(18, int(legend_fs)),
            "bond_w": max(1, int(round(bond_w))),
            "margin_px": max(16, int(inner_margin_px)),
            "ss_factor": ss_factor,
            "_flat_cache": {detail_scale: arr.reshape(-1)},
        }
    return {1024: tag}


# -----------------------------------------------------------------------------
# 17. Setup lod autoswap
# -----------------------------------------------------------------------------
def _setup_lod_autoswap(
    state: dict[str, Any],
    axis_x_tag: str,
    node_series_map: Any,
    node_lods_base: Any,
    node_lods_high: Any,
    node_diameter_units: Any
) -> Any:
    plot_tag = "mmpa_network_plot"

    state["mmpa__lod_series"] = node_series_map
    state["mmpa__lod_base"] = node_lods_base
    state["mmpa__lod_high"] = node_lods_high
    state["mmpa__lod_node_diam_units"] = node_diameter_units
    state["mmpa__lod_current"] = {}
    state.setdefault("mmpa__highlighted_nodes", set())

    # -----------------------------------------------------------------------------
    # 17.1. Choose lod
    # -----------------------------------------------------------------------------
    def choose_lod(px: Any) -> Any:
        return 1024

    # -----------------------------------------------------------------------------
    # 17.2. Maybe swap
    # -----------------------------------------------------------------------------
    def _maybe_swap() -> None:
        try:
            x_min, x_max = dpg.get_axis_limits(axis_x_tag)
        except Exception:
            return
        if x_min is None or x_max is None or x_max <= x_min:
            return

        pw, _ = dpg.get_item_rect_size(plot_tag)
        if not pw or pw <= 0:
            return

        units_w = (x_max - x_min)
        px_per_unit = pw / units_w
        node_px = state["mmpa__lod_node_diam_units"] * px_per_unit
        target_lod = choose_lod(node_px)

        highlighted = state.get("mmpa__highlighted_nodes", set())
        for mid, series_id in state["mmpa__lod_series"].items():
            cur = state["mmpa__lod_current"].get(mid)
            # choose the correct map
            if mid in highlighted and state["mmpa__lod_high"].get(mid):
                tex_tag = state["mmpa__lod_high"][mid].get(target_lod) or next(iter(state["mmpa__lod_high"][mid].values()))
            else:
                tex_tag = state["mmpa__lod_base"][mid].get(target_lod) or next(iter(state["mmpa__lod_base"][mid].values()))
            if tex_tag and dpg.does_item_exist(series_id):
                if cur != target_lod or dpg.get_item_configuration(series_id).get("texture_tag") != tex_tag:
                    dpg.configure_item(series_id, texture_tag=tex_tag)
                    state["mmpa__lod_current"][mid] = target_lod

    # Expose for use from hover
    state["_mmpa_lod_maybe_swap"] = _maybe_swap

    # Handlers to react to zoom/pan/mouse move
    # -----------------------------------------------------------------------------
    # 17.3. On mouse move
    # -----------------------------------------------------------------------------
    def _on_mouse_move(sender: Any, app_data: Any) -> None:
        """
        Refresh the level-of-detail selection while the mouse moves.

        Args:
            sender (Any): Dear PyGui sender identifier.
            app_data (Any): Dear PyGui callback payload.

        Returns:
            None: This routine updates the rendered network in place.
        """
        _maybe_swap()

    # -----------------------------------------------------------------------------
    # 17.4. On mouse wheel
    # -----------------------------------------------------------------------------
    def _on_mouse_wheel(sender: Any, app_data: Any) -> None:
        """
        Refresh the level-of-detail selection after a mouse-wheel event.

        Args:
            sender (Any): Dear PyGui sender identifier.
            app_data (Any): Dear PyGui callback payload.

        Returns:
            None: This routine updates the rendered network in place.
        """
        _maybe_swap()

    lod_move_tag = "mmpa__lod_mouse_move"
    lod_wheel_tag = "mmpa__lod_mouse_wheel"
    for t in (lod_move_tag, lod_wheel_tag):
        if dpg.does_item_exist(t):
            dpg.delete_item(t)

    dpg.add_mouse_move_handler(tag=lod_move_tag, parent="handler_registry", callback=_on_mouse_move)
    dpg.add_mouse_wheel_handler(tag=lod_wheel_tag, parent="handler_registry", callback=_on_mouse_wheel)

    # Initial evaluation
    _maybe_swap()


# -----------------------------------------------------------------------------
# 18. Mmpa lod maybe swap
# -----------------------------------------------------------------------------
def _mmpa_lod_maybe_swap(state: dict[str, Any]) -> None:
    fn = state.get("_mmpa_lod_maybe_swap")
    if callable(fn):
        fn()
