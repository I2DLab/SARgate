"""
=====================
sal_utils.py
=====================

Logic helpers for the SAR Landscape UI.

This module contains the interactive behaviours and rendering utilities used by
the SAR Landscape view:
- point click handling on the scatter plot to update the molecule panels;
- filtering and bucketed redraw of the scatter series according to thresholds;
- RDKit-based rendering of molecules with legends;
- generation of per-atom similarity maps (RDKit SimilarityMaps) with a legend
  consistent with the “normal” molecule images;
- show/hide logic for the “Draw/Hide Similarity Map” buttons.

These functions are invoked by the GUI layer (`landscape_plot.py`) and operate
over the shared mutable application state (`state`).
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. On plot click
# 3. Get sali thresh linear
# 4. Update landscape scatter
# 5. Render mol with legend
# 6. Simmap fp lambda from choice
# 7. Render similarity map for fp
# 8. Fmt activity display
# 9. Update molecule panel
# 10. Draw similarity maps
# 11. Hide similarity maps

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import io
import math
import textwrap
import dearpygui.dearpygui as dpg
import numpy as np
from typing import Any
from PIL import Image as pilImage, ImageDraw, ImageFont
from rdkit import DataStructs
from rdkit.Chem import rdDepictor
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps


# -----------------------------------------------------------------------------
# 2. On plot click
# -----------------------------------------------------------------------------
def _on_plot_click(sender: Any, app_data: Any, user_data: Any) -> None:
    landscape_x_axis = "landscape_x_axis"
    landscape_y_axis = "landscape_y_axis"
    state = user_data
    activity = state.get("landscape_activity_type", "")
    xs = state.get("landscape_xs", None)
    ys = state.get("landscape_ys", None)
    sali_raw = state.get("landscape_sali_raw", None)
    
    # Left click only
    if isinstance(app_data, dict) and app_data.get("button", -1) != 0:
        return
    # Mouse position in axis coordinates
    px, py = dpg.get_plot_mouse_pos()
    if not (np.isfinite(px) and np.isfinite(py)):
        return

    # --- Tolerance in axis units based on plot size (pixel → data) ---
    plot_w_px, plot_h_px = dpg.get_item_rect_size("landscape_plot")

    # Current axis limits
    x_min, x_max = dpg.get_axis_limits(landscape_x_axis)
    y_min, y_max = dpg.get_axis_limits(landscape_y_axis)
    x_span = max(1e-12, (x_max - x_min))
    y_span = max(1e-12, (y_max - y_min))

    # Clickable radius ≈ half marker + margin (in pixels)
    marker_px = int(state.get("landscape_marker_px", 4))
    radius_px = max(3, int(round(0.6 * marker_px)))   # margin

    # Convert tolerance to axis units
    tol_x = radius_px * (x_span / max(1, plot_w_px))
    tol_y = radius_px * (y_span / max(1, plot_h_px))

    vis_idx = state.get("landscape_visible_idx", None)
    vis_xs  = state.get("landscape_visible_xs", None)
    vis_ys  = state.get("landscape_visible_ys", None)
    if vis_idx is None or vis_xs is None or vis_ys is None or len(vis_idx) == 0:
        return

    # Normalised distance (ellipse of tolerance)
    dx = (vis_xs - px) / tol_x
    dy = (vis_ys - py) / tol_y
    dist2 = dx*dx + dy*dy

    k = int(np.argmin(dist2))
    # Trigger only if inside the tolerance radius (dist^2 <= 1)
    if dist2[k] > 1.0:
        return

    sel_pair_global_idx = int(vis_idx[k])

    # Update molecule panel
    _update_molecule_panel(sel_pair_global_idx, state)

    # Update details text (Δactivity, similarity, SALI)
    da = ys[sel_pair_global_idx]
    sim = xs[sel_pair_global_idx]
    sali = sali_raw[sel_pair_global_idx]
    if activity in state["nM_activity_types"]:
        dpg.set_value("landscape_couple_details_text",
                      f"Δp{activity} = {da:.2f}\n"
                      f"Similarity = {sim:.2f}\n"
                      f"SALI Index = {sali:.2f}")
    else:
        units = "%" if activity in state["percent_activities"] else "μg/mL" if activity in state["ug/mL_activities"] else "μM/min" if activity in state["uM/min_activities"] else ""
        dpg.set_value("landscape_couple_details_text",
                      f"Δ{activity} = {da:.2f} {units}\n"
                      f"Similarity = {sim:.2f}\n"
                      f"SALI Index = {sali:.2f}")

    # Reset similarity-map state and toggle buttons
    state["landscape_showing_simmap"] = False
    if dpg.does_item_exist("landscape_similarity_map_button"):
        dpg.configure_item("landscape_similarity_map_button", show=True)
    if dpg.does_item_exist("landscape_similarity_map_hide_button"):
        dpg.configure_item("landscape_similarity_map_hide_button", show=False)


# -----------------------------------------------------------------------------
# 3. Get sali thresh linear
# -----------------------------------------------------------------------------
def _get_sali_thresh_linear(state: dict[str, Any]) -> float:
    try:
        t = float(dpg.get_value("landscape_sali_index_thresh"))
    except Exception:
        t = 0.0
    t = float(np.clip(t, 0.0, 1.0))

    # Preferred: use explicit min/max if present
    lo_hi = state.get("landscape_sali_minmax", None)
    if lo_hi and len(lo_hi) == 2 and all(np.isfinite(lo_hi)):
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
    else:
        # Fallback: compute from data
        sali_raw = state.get("landscape_sali_raw", None)
        if sali_raw is None or len(sali_raw) == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = float(np.nanmin(sali_raw))
            hi = float(np.nanmax(sali_raw))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0

    # Small overshoot to avoid visual clamp at the very end
    delta = max((hi - lo) * 1e-2, 1e-6)
    return lo + t * ((hi - lo) + delta)


# -----------------------------------------------------------------------------
# 4. Update landscape scatter
# -----------------------------------------------------------------------------
def _update_landscape_scatter(bin_edges: Any, N_COL: Any, state: dict[str, Any]) -> None:
    # if dpg.is_mouse_button_dragging(dpg.mvMouseButton_Left, 1.0):
    #     return
    
    xs = state.get("landscape_xs", None)
    ys = state.get("landscape_ys", None)
    sali_raw = state.get("landscape_sali_raw", None)
    color_series_tags = state.get("landscape_color_series_tags", None)

    if xs is None or ys is None or sali_raw is None or color_series_tags is None:
        return

    # Thresholds from UI
    try:
        thr_x = float(dpg.get_value("landscape_similarity_thresh"))
    except Exception:
        thr_x = 0.0
    try:
        thr_y = float(dpg.get_value("landscape_delta_thresh"))
    except Exception:
        thr_y = 0.0

    sali_thr = _get_sali_thresh_linear(state)  # raw SALI threshold

    # Visibility mask
    mask = (
        (xs >= thr_x) &
        (ys >= thr_y) &
        (sali_raw > sali_thr) &
        np.isfinite(xs) & np.isfinite(ys) & np.isfinite(sali_raw)
    )

    # Save visible subset (for click handler)
    visible_idx = np.nonzero(mask)[0]
    state["landscape_visible_idx"] = visible_idx
    state["landscape_visible_xs"] = xs[visible_idx]
    state["landscape_visible_ys"] = ys[visible_idx]

    xs_f = xs[mask]
    ys_f = ys[mask]
    sali_f = sali_raw[mask]

    # Base (hidden) series for export
    if dpg.does_item_exist("landscape_scatter_series"):
        try:
            dpg.set_value("landscape_scatter_series", [xs_f.tolist(), ys_f.tolist()])
        except Exception:
            pass

    # Clear buckets
    for tag in color_series_tags:
        dpg.set_value(tag, [[], []])

    if len(xs_f) == 0:
        return

    # Bin directly over raw SALI using provided bin_edges
    bins = np.digitize(sali_f, bin_edges, right=False) - 1
    bins = np.clip(bins, 0, N_COL - 1)

    # Group and update series
    for i in range(N_COL):
        sel = (bins == i)
        if not np.any(sel):
            continue
        dpg.set_value(color_series_tags[i], [xs_f[sel].tolist(), ys_f[sel].tolist()])


# -----------------------------------------------------------------------------
# 5. Render mol with legend
# -----------------------------------------------------------------------------
def _render_mol_with_legend(mol: Any, w: Any, h: Any, legend_text: str) -> Any:
    # Empty image fallback if `mol` is None
    if mol is None:
        arr = np.zeros((h, w, 4), dtype=np.float32)
        return arr.ravel().tolist()

    # Compute 2D coordinates for a clean depiction
    rdDepictor.Compute2DCoords(mol)

    # Cairo drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    # Options close to other views for a consistent appearance
    opts.padding = 0.025
    opts.bondLineWidth = 1
    opts.minFontSize = 1
    opts.legendFontSize = 14
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    img = pilImage.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = (np.array(img).astype(np.float32) / 255.0).reshape(-1).tolist()
    return arr


# -----------------------------------------------------------------------------
# 6. Simmap fp lambda from choice
# -----------------------------------------------------------------------------
def _simmap_fp_lambda_from_choice(choice: str) -> Any:
    if choice == "Morgan Fingerprint":
        return GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    elif choice == "RDKit Fingerprint":
        return lambda m, idx: SimilarityMaps.GetRDKitFP(m, atomId=idx)
    elif choice == "Atom Pair Fingerprint":
        return lambda m, idx: SimilarityMaps.GetAtomPairFingerprint(m, atomId=idx)
    elif choice == "Topological Torsion Fingerprint":
        return lambda m, idx: SimilarityMaps.GetTopologicalTorsionFingerprint(m, atomId=idx)
    else:  # Not reliably supported per-atom for: MACCS / Pattern / Layered → fallback to Morgan
        return GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


# -----------------------------------------------------------------------------
# 7. Render similarity map for fp
# -----------------------------------------------------------------------------
def _render_similarity_map_for_fp(
    ref_mol: Any,
    probe_mol: Any,
    w: Any,
    h: Any,
    fp_func: Any,
    legend_text: str
) -> Any:
    if ref_mol is None or probe_mol is None or fp_func is None:
        arr = np.zeros((h, w, 4), dtype=np.float32)
        return arr.ravel().tolist()

    # Ensure robust 2D coordinates for both molecules
    rdDepictor.Compute2DCoords(ref_mol)
    rdDepictor.Compute2DCoords(probe_mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.padding = 0.025
    opts.bondLineWidth = 1
    opts.minFontSize = 1
    # Note: opts.legendFontSize is ignored by SimilarityMaps

    # Draw similarity map (SimilarityMaps does not draw legends itself)
    if callable(fp_func):
        SimilarityMaps.GetSimilarityMapForFingerprint(
            ref_mol,
            probe_mol,
            fp_func,
            drawer,
            metric=DataStructs.TanimotoSimilarity
        )
    else:
        SimilarityMaps.GetSimilarityMapForFingerprintGenerator(
            ref_mol,
            probe_mol,
            fp_func,
            drawer,
            metric=DataStructs.TanimotoSimilarity,
            useCounts=True,
        )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    img = pilImage.open(io.BytesIO(png_bytes)).convert("RGBA")
    return (np.asarray(img, dtype=np.float32) / 255.0).reshape(-1).tolist()
    

# -----------------------------------------------------------------------------
# 8. Fmt activity display
# -----------------------------------------------------------------------------
def _fmt_activity_display(val: Any, label: str, state: dict[str, Any]) -> Any:
    # p-type (use your configured key: "nM_activity_types")
    if label in state.get("nM_activity_types", set()):
        try:
            v = float(val)
            if v > 0:
                pval = 9.0 - math.log10(v)   # keep the existing convention in your app
                return f"p{label} = {pval:.2f}"
            else:
                return f"p{label} = n/a"
        except Exception:
            return f"p{label} = n/a"
    else:
        unit = (
            "%" if label in state.get("percent_activities", set()) else
            "μg/mL" if label in state.get("ug/mL_activities", set()) else
            "μM/min" if label in state.get("uM/min_activities", set()) else
            "nM"
        )
        try:
            v = float(val)
            return f"{label} = {v:.2f} {unit}"
        except Exception:
            return f"{label} = n/a {unit}"
        

# -----------------------------------------------------------------------------
# 9. Update molecule panel
# -----------------------------------------------------------------------------
def _update_molecule_panel(idx_pair: Any, state: dict[str, Any]) -> None:
    landscape_img_width = state["landscape_img_width"]
    landscape_img_height = round(landscape_img_width / 4 * 3)
    landscape_render_scale = 1.8
    landscape_render_width = int(round(landscape_img_width * landscape_render_scale))
    landscape_render_height = int(round(landscape_img_height * landscape_render_scale))
    pair_i = state["landscape_pair_i"]
    pair_j = state["landscape_pair_j"]
    work = state["landscape_work_df"]
    activity = state["landscape_activity_type"]
    i = pair_i[idx_pair]
    j = pair_j[idx_pair]

    mol_i = work["ROMol"].iloc[i]
    mol_j = work["ROMol"].iloc[j]
    id_i  = str(work["MolID"].iloc[i]); name_i = str(work["Name"].iloc[i]); act_i = work["Activity"].iloc[i]
    id_j  = str(work["MolID"].iloc[j]); name_j = str(work["Name"].iloc[j]); act_j = work["Activity"].iloc[j]

    lab1 = f"Mol {id_i}  |  {name_i}  |  {_fmt_activity_display(act_i, activity, state)}"
    lab2 = f"Mol {id_j}  |  {name_j}  |  {_fmt_activity_display(act_j, activity, state)}"

    def _lower_is_better(activity_name: str, app_state: dict[str, Any]) -> bool:
        return activity_name in app_state.get("nM_activity_types", set()) or activity_name in app_state.get("ug/mL_activities", set())

    lower_better = _lower_is_better(activity, state)
    top_is_i = (float(act_i) <= float(act_j)) if lower_better else (float(act_i) >= float(act_j))
    top_mol = mol_i if top_is_i else mol_j
    bottom_mol = mol_j if top_is_i else mol_i
    top_label = lab1 if top_is_i else lab2
    bottom_label = lab2 if top_is_i else lab1

    # Render and push into textures (already created in the image window)
    if dpg.does_item_exist("landscape_mol1_image_texture"):
        tex1 = _render_mol_with_legend(top_mol, landscape_render_width, landscape_render_height, top_label)
        dpg.set_value("landscape_mol1_image_texture", tex1)

    if dpg.does_item_exist("landscape_mol2_image_texture"):
        tex2 = _render_mol_with_legend(bottom_mol, landscape_render_width, landscape_render_height, bottom_label)
        dpg.set_value("landscape_mol2_image_texture", tex2)         
    if dpg.does_item_exist("landscape_mol1_tooltip_text"):
        dpg.set_value("landscape_mol1_tooltip_text", f"Higher activity molecule\n{top_label}" if not lower_better else f"Higher potency molecule\n{top_label}")
    if dpg.does_item_exist("landscape_mol2_tooltip_text"):
        dpg.set_value("landscape_mol2_tooltip_text", f"Lower activity molecule\n{bottom_label}" if not lower_better else f"Lower potency molecule\n{bottom_label}")
                        
    # Cache last selected pair (for similarity-map toggling)
    state["landscape_last_pair_idx"] = int(idx_pair)
    state["landscape_last_pair"] = (int(pair_i[idx_pair]), int(pair_j[idx_pair]))
    state["landscape_last_labels"] = (top_label, bottom_label)
    state["landscape_last_pair_order"] = ("top", "bottom")


# -----------------------------------------------------------------------------
# 10. Draw similarity maps
# -----------------------------------------------------------------------------
def _draw_similarity_maps(sender: Any, app_data: Any, user_data: Any) -> None:
    state = user_data
    last = state.get("landscape_last_pair")
    work = state.get("landscape_work_df")
    activity = state.get("landscape_activity_type", "")
    landscape_img_width = state["landscape_img_width"]
    landscape_img_height = round(landscape_img_width / 4 * 3)
    landscape_render_scale = 1.8
    landscape_render_width = int(round(landscape_img_width * landscape_render_scale))
    landscape_render_height = int(round(landscape_img_height * landscape_render_scale))
    
    if not last:
        return
    i, j = last
    mol_i = work["ROMol"].iloc[i]
    mol_j = work["ROMol"].iloc[j]
    act_i = work["Activity"].iloc[i]
    act_j = work["Activity"].iloc[j]

    # Labels already cached by _update_molecule_panel
    lab1, lab2 = state.get("landscape_last_labels", ("A", "B"))
    lower_better = activity in state.get("nM_activity_types", set()) or activity in state.get("ug/mL_activities", set())
    top_is_i = (float(act_i) <= float(act_j)) if lower_better else (float(act_i) >= float(act_j))
    top_mol = mol_i if top_is_i else mol_j
    bottom_mol = mol_j if top_is_i else mol_i
    top_ref = bottom_mol
    bottom_ref = top_mol

    fp_choice = dpg.get_value("landscape_fingerprint_algorithm_combo")
    fp_func = _simmap_fp_lambda_from_choice(fp_choice)
    if fp_func is None:
        with dpg.window(label="Similarity Map", modal=True, autosize=True, no_resize=True, no_collapse=True):
            dpg.add_text(f"Similarity map not supported for '{fp_choice}'.\nUse Morgan, RDKit, Atom Pair or Topological Torsion.")
            dpg.add_button(label="OK", width=60, callback=lambda: None)
        return

    try:
        texA = _render_similarity_map_for_fp(
            ref_mol=top_ref, probe_mol=top_mol,
            w=landscape_render_width, h=landscape_render_height,
            fp_func=fp_func, legend_text=lab1
        )
        dpg.set_value("landscape_mol1_image_texture", texA)
    except Exception:
        pass

    try:
        texB = _render_similarity_map_for_fp(
            ref_mol=bottom_ref, probe_mol=bottom_mol,
            w=landscape_render_width, h=landscape_render_height,
            fp_func=fp_func, legend_text=lab2
        )
        dpg.set_value("landscape_mol2_image_texture", texB)
    except Exception:
        pass

    state["landscape_showing_simmap"] = True
    if dpg.does_item_exist("landscape_similarity_map_button"):
        dpg.configure_item("landscape_similarity_map_button", show=False)
    if dpg.does_item_exist("landscape_similarity_map_hide_button"):
        dpg.configure_item("landscape_similarity_map_hide_button", show=True)


# -----------------------------------------------------------------------------
# 11. Hide similarity maps
# -----------------------------------------------------------------------------
def _hide_similarity_maps(sender: Any, app_data: Any, user_data: Any) -> None:
    state = user_data
    last_idx = state.get("landscape_last_pair_idx", None)
    if last_idx is not None:
        _update_molecule_panel(int(last_idx), state)

    state["landscape_showing_simmap"] = False
    if dpg.does_item_exist("landscape_similarity_map_button"):
        dpg.configure_item("landscape_similarity_map_button", show=True)
    if dpg.does_item_exist("landscape_similarity_map_hide_button"):
        dpg.configure_item("landscape_similarity_map_hide_button", show=False)
