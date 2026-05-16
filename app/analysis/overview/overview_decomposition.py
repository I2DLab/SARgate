"""
=========================
overview_decomposition.py
=========================

Scaffold/R-group decomposition and interactive visual analytics for SARgate.

This module displays the full decomposition results:
- Builds and caches a scaffold-similarity dendrogram (SciPy) with Dear PyGui
  rendering, live recolouring on threshold changes, and fast cluster relabelling
  without redrawing.
- Provides subset/molecule/R-group browsers with auto-scroll linkage from images
  to list panes, and cluster-aware grouping/sorting (by count, query similarity,
  or hierarchical clustering).
- Renders scalable 2D depictions (RDKit Cairo backend) of scaffolds, molecules,
  and individual R-groups, with right-click PNG export and on-label SVG export.
- Displays property panels per selection (subset/molecule/R) including
  identifiers, descriptors, chemicophysical metrics, topology indices, and
  bioactivity/assay/target summaries.
- Generates per-atom visual explanations: Crippen logP contribution map, Gasteiger
  charge similarity map (with legends), and H-bond donor/acceptor highlights.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Ensure dendro cache
# 3. Get rgba from colorname
# 4. Recolor dendrogram segments
# 5. Fast update clusters from linkage
# 6. Clear clustering ui
# 7. Normalize scaffold
# 8. Sort by scaffold similarity
# 9. Ensure dendro frame theme
# 10. Refresh subset buttons
# 11. On slider threshold change
# 12. Draw scaffold dendrogram
# 13. Show subset choice window
# 14. Show molecule choice window
# 15. Update molecule choice status
# 16. Show r groups choice window
# 17. Update r groups choice status
# 18. Show properties windows
# 19. Update properties windows
# 20. Show activities windows
# 21. Update activities windows
# 22. Autoscroll to button
# 23. Image click callback
# 24. Build align checkbox
# 25. Toggle overview alignment
# 26. Toggle overview counts
# 27. Show results window
# 28. Show log p atomic contribution map
# 29. Show gasteiger atomic contribution map
# 30. Show hba hbd

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import re
import io
import time
import dearpygui.dearpygui as dpg
import webbrowser
import urllib.parse
import numpy as np
from typing import Any
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from PIL import Image as pilImage, ImageDraw, ImageFont
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import Draw, AllChem, rdMolDescriptors, ChemicalFeatures, MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from app.utils.callbacks import (
    color_string_to_rgb255,
    on_button_click,
    export_png_popup,
    export_svg_callback,
    register_plot_context_popup,
    register_responsive_image,
    update_responsive_images
)
from app.gui.themes_manager import (
    apply_inner_child_theme, 
    apply_input_text_theme,
    apply_image_button_theme,
    apply_dendrogram_theme,
    apply_bordered_input_text_theme,
    change_font_type,
    refresh_overrides
)
from app.gui.loading_win import draw_loading_screen
from app.analysis.overview.overview_enrichment_plot import build_enrichment_layout


# --- CACHE FOR DENDROGRAM (reused by slider) ---
# -----------------------------------------------------------------------------
# 2. Ensure dendro cache
# -----------------------------------------------------------------------------
def _ensure_dendro_cache(state: dict[str, Any]) -> Any:
    return state.setdefault("_dpg_dendro_cache", {})


# -----------------------------------------------------------------------------
# 3. Get rgba from colorname
# -----------------------------------------------------------------------------
def _get_rgba_from_colorname(color_str: Any) -> Any:
    try:
        if color_str in ("C0", "b", "blue"):
            return (80, 80, 80, 255)
        r, g, b = color_string_to_rgb255(color_str, fallback=(80, 80, 80))
        return (r, g, b, 255)
    except Exception:
        return (80, 80, 80, 255)


def _load_overview_label_font(size_px: int) -> Any:
    """
    Load a readable PIL font for overview image labels.
    """
    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size_px)
        except Exception:
            continue
    return ImageFont.load_default()


def _add_overview_label_on_pil(
    img: Any,
    text: str,
    position: str = "top",
    margin: int = 6,
    pad_x: int = 6,
    pad_y: int = 3,
    bg: Any = None,
    fg: Any = (0, 0, 0, 255),
) -> Any:
    """
    Draw a centred label at the top or bottom of an RGBA image.
    """
    if not text:
        return img

    draw = ImageDraw.Draw(img, "RGBA")
    try:
        target_size = max(18, int(round(img.width * 0.065)))
        font = _load_overview_label_font(target_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        while target_size > 12 and (bbox[2] - bbox[0]) > (img.width - 2 * margin):
            target_size -= 1
            font = _load_overview_label_font(target_size)
            bbox = draw.textbbox((0, 0), text, font=font)
    except Exception:
        font = None

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = img.size
    x = (w - tw) // 2
    y = margin if position == "top" else (h - th - margin)

    if bg:
        draw.rectangle([x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y], fill=bg)

    draw.text((x, y), text, fill=fg, font=font)
    return img


# -----------------------------------------------------------------------------
# 4. Recolor dendrogram segments
# -----------------------------------------------------------------------------
def _recolor_dendrogram_segments(state: dict[str, Any], threshold: Any) -> None:
    cache = state.get("_dpg_dendro_cache") or {}
    linkage_matrix   = cache.get("linkage")
    labels_input     = cache.get("labels_input")     # the same 'subset_names' passed to scipy
    link_to_seg_ids  = cache.get("link_to_seg_ids")  # [[seg_id, seg_id, seg_id], ...]

    if linkage_matrix is None or labels_input is None or not link_to_seg_ids:
        return  # nothing to do

    # request ONLY colouring from scipy (icoord/dcoord not needed)
    dendro = dendrogram(linkage_matrix, labels=labels_input, no_plot=True, color_threshold=float(threshold))
    color_list = dendro["color_list"]  # one colour per 'link' (3 segments each)

    # apply colour for each link to its 3 segments
    for link_idx, color_name in enumerate(color_list):
        rgba = _get_rgba_from_colorname(color_name)
        for seg_id in link_to_seg_ids[link_idx]:
            if dpg.does_item_exist(seg_id):
                dpg.configure_item(seg_id, color=rgba)


# -----------------------------------------------------------------------------
# 5. Fast update clusters from linkage
# -----------------------------------------------------------------------------
def _fast_update_clusters_from_linkage(state: dict[str, Any], threshold: Any) -> Any:
    cache = state.get("_dpg_dendro_cache") or {}
    linkage_matrix   = cache.get("linkage")
    labels_input     = cache.get("labels_input")     # these were the numeric IDs (strings) passed to scipy
    labels_ordered   = cache.get("labels_ordered")   # ivl (leaf order) from the first build

    if linkage_matrix is None or labels_input is None or labels_ordered is None:
        return None, None, None

    # map numeric ID (string) -> logical subset_name ("subset_7")
    smiles_rgd_dict = state["smiles_rgd_dict"]
    id_to_logical = {str(int(nm.split("_")[1])): nm for nm in smiles_rgd_dict.keys()}

    ordered_logical_names = [id_to_logical[lbl] for lbl in labels_ordered if lbl in id_to_logical]

    raw_labels = fcluster(linkage_matrix, t=float(threshold), criterion='distance')

    # align to labels_input (order used by scipy/linkage)
    name_to_cluster_raw = {id_to_logical[labels_input[i]]: int(raw_labels[i])
                           for i in range(len(labels_input)) if labels_input[i] in id_to_logical}

    # remap 1..K following the dendrogram order
    cluster_remap, next_label, name_to_new = {}, 1, {}
    for nm in ordered_logical_names:
        oc = name_to_cluster_raw[nm]
        if oc not in cluster_remap:
            cluster_remap[oc] = next_label
            next_label += 1
        name_to_new[nm] = cluster_remap[oc]

    original_names = list(smiles_rgd_dict.keys())
    labels = [name_to_new.get(nm, 0) for nm in original_names]
    sorted_subsets = ordered_logical_names[:]
    return sorted_subsets, labels, original_names


# -----------------------------------------------------------------------------
# 6. Clear clustering ui
# -----------------------------------------------------------------------------
def _clear_clustering_ui() -> None:
    # Remove slider and dendrogram if present
    """
    Clear clustering ui.
    
    Args:
        None.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    for tag in ["cluster_threshold_slider", "scaffold_hierarchical_dendrogram", "threshold_line"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)


# -----------------------------------------------------------------------------
# 7. Normalize scaffold
# -----------------------------------------------------------------------------
def _normalize_scaffold(mol: Chem.Mol) -> Chem.Mol:
    if mol is None:
        return None
    m = rdMolStandardize.Cleanup(mol)
    m = rdMolStandardize.TautomerEnumerator().Canonicalize(m)
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except:
        pass
    # set aromaticity consistently again
    Chem.SanitizeMol(
        m,
        Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    )
    Chem.AssignStereochemistry(m, force=True, cleanIt=True)
    return m


# -----------------------------------------------------------------------------
# 8. Sort by scaffold similarity
# -----------------------------------------------------------------------------
def sort_by_scaffold_similarity(
    smiles_rgd_dict: str,
    subset_dir: str,
    threshold: float = 0.2
) -> Any:
    """
    Sorts molecular subsets based on scaffold similarity using hierarchical clustering.

    Args:
        smiles_rgd_dict (dict): Dictionary of SMILES grouped by subset.
        subset_dir (str): Path to the directory containing scaffold SDF files.
        threshold (float, optional): Distance threshold for clustering (0..1).
                                    Example: 0.2 -> all pairs in the cluster have sim ≥ 0.8

    Returns:
        tuple: (ordered subset names, cluster labels (per subset_names original order), original subset names)
    """

    try:
        scaffold_mols = []
        subset_names = []

        # load scaffolds
        for subset_name in smiles_rgd_dict:
            scaffold_id = int(subset_name.split("_")[1])
            scaffold_path = os.path.join(subset_dir, f"scaffold_{scaffold_id}.sdf")
            suppl = Chem.SDMolSupplier(scaffold_path)
            scaffold = next((m for m in suppl if m is not None), None)
            if scaffold:
                scaffold_mols.append(scaffold)
                subset_names.append(subset_name)

        if len(scaffold_mols) < 2:
            return subset_names, [1]*len(subset_names), subset_names

        # sort (mol, name) together by heavy atom count (cosmetic only)
        pairs = sorted(
            zip(scaffold_mols, subset_names),
            key=lambda x: x[0].GetNumHeavyAtoms(),
            reverse=True
        )
        scaffold_mols, subset_names = map(list, zip(*pairs))

        # normalise
        scaffold_mols = [_normalize_scaffold(m) for m in scaffold_mols]

        # fingerprint: Morgan count-based r=3, 4096 bits (more robust)
        gen = GetMorganGenerator(radius=3, fpSize=4096)
        fps = [gen.GetCountFingerprint(m) for m in scaffold_mols]

        # condensed distance (no n x n matrix)
        n = len(fps)
        condensed = np.empty(n*(n-1)//2, dtype=float)
        k = 0
        for i in range(n-1):
            for j in range(i+1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                condensed[k] = 1.0 - sim   # distance = 1 - similarity
                k += 1

        # 'complete' linkage: guarantees max intra-cluster distance <= threshold
        linkage_matrix = linkage(condensed, method='complete')
        raw_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')

        # dendrogram for consistent display order
        dendro = dendrogram(linkage_matrix, labels=subset_names, no_plot=True, color_threshold=threshold)
        ordered_names = dendro["ivl"]

        # map name -> cluster label (raw)
        name_to_cluster = {subset_names[i]: raw_labels[i] for i in range(len(subset_names))}

        # remap labels to have clusters 1..K by order of appearance in the dendrogram
        cluster_remap = {}
        new_label_counter = 1
        name_to_new_label = {}

        for name in ordered_names:
            original_cluster = name_to_cluster[name]
            if original_cluster not in cluster_remap:
                cluster_remap[original_cluster] = new_label_counter
                new_label_counter += 1
            name_to_new_label[name] = cluster_remap[original_cluster]

        # labels aligned to subset_names (original collection order)
        labels = [name_to_new_label[name] for name in subset_names]

        # cluster -> list of names, keeping dendrogram order
        cluster_dict = {}
        for nm in subset_names:
            cluster_dict.setdefault(name_to_new_label[nm], []).append(nm)

        # sort clusters by id, then by dendrogram order
        sorted_clusters = sorted(cluster_dict.items(), key=lambda x: x[0])
        sorted_subsets = []
        for cluster_id, _members in sorted_clusters:
            for name in ordered_names:
                if name_to_new_label[name] == cluster_id:
                    sorted_subsets.append(name)

        return sorted_subsets, labels, subset_names

    except Exception as e:
        log_exception("Overview", "Clustering failed", e, indent=1)
        return list(smiles_rgd_dict.keys()), None, None


# -----------------------------------------------------------------------------
# 9. Ensure dendro frame theme
# -----------------------------------------------------------------------------
def _ensure_dendro_frame_theme(state: dict[str, Any]) -> Any:
    th = state.get("_dendro_frame_theme")
    if th and dpg.does_item_exist(th):
        return th
    # ultra-thin line, subtle grey
    with dpg.theme() as th:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 0.0, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (80, 80, 80, 80), category=dpg.mvThemeCat_Plots)
    state["_dendro_frame_theme"] = th
    return th
     


# -----------------------------------------------------------------------------
# 10. Refresh subset buttons
# -----------------------------------------------------------------------------
def refresh_subset_buttons(state: dict[str, Any]) -> None:

    if dpg.does_item_exist("subset_choice_group"):
        dpg.delete_item("subset_choice_group", children_only=True)
        last_group = None

        dpg.add_spacer(height=state["win_spacer"], parent="subset_choice_group")

        for subset in state["subset_order_query"]:
            group_label = state.get("subset_cluster_groups", {}).get(subset, None)
            if group_label != last_group and group_label is not None:
                dpg.add_text(group_label, bullet=True, parent="subset_choice_group")
                last_group = group_label

            dpg.add_button(
                label=subset.replace("subset_", "Subset "),
                tag=subset,
                width=-1,
                parent="subset_choice_group",
                callback=lambda id=subset: (
                    update_molecule_choice_status(id, state),
                    update_properties_windows(id, state),
                    update_activities_windows(id, state),
                    build_enrichment_layout(id, state),
                    on_button_click(id, state),
                    show_results_window(id, state)
                )
            )              


# -----------------------------------------------------------------------------
# 11. On slider threshold change
# -----------------------------------------------------------------------------
def on_slider_threshold_change(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    # For drag_line, the value is NOT in app_data → read it from the item itself
    new_distance = dpg.get_value(sender)

    if new_distance is None:
        return

    new_distance = float(new_distance)
    new_distance = max(0.0, min(1.0, new_distance))
    state["cluster threshold"] = new_distance

    # Debounce
    now = time.time()
    last_time = state.get("_last_slider_update_time", 0.0)
    if now - last_time < 0.12:
        return
    state["_last_slider_update_time"] = now

    # 1) update title
    if dpg.does_item_exist("scaffold_hierarchical_dendrogram"):
        dpg.configure_item(
            "scaffold_hierarchical_dendrogram",
            label=(
                "Substructures Hierarchical Clustering  |  "
                f"Similarity Threshold = {(1.0 - new_distance) * 100:.0f}%"
            )
        )

    # 2) recolour ONLY the already drawn segments
    _recolor_dendrogram_segments(state, new_distance)

    # 3) quickly update order/labels (reuse linkage)
    fast = _fast_update_clusters_from_linkage(state, new_distance)
    if fast != (None, None, None):
        order, labels, names = fast
        if order:
            state["subset_order_query"] = order
            if labels and names:
                state["subset_cluster_groups"] = {
                    name: f"Cluster {labels[i]}"
                    for i, name in enumerate(names)
                }
            refresh_subset_buttons(state)

    # 4) update combo (UI only)
    if dpg.does_item_exist("overview_subset_sort_choice"):
        dpg.set_value("overview_subset_sort_choice", "Similarity Clustering")


# -----------------------------------------------------------------------------
# 12. Draw scaffold dendrogram
# -----------------------------------------------------------------------------
def draw_scaffold_dendrogram(state: dict[str, Any]) -> Any:
    
    draw_loading_screen(state)

    # -----------------------------------------------------------------------------
    # 12.1. Get color
    # -----------------------------------------------------------------------------
    def get_color(color_str: Any) -> Any:
        try:
            if color_str in ("C0", "b", "blue"):
                return (80, 80, 80, 255)
            return color_string_to_rgb255(color_str, fallback=(80, 80, 80)) + (255,)
        except Exception:
            return (80, 80, 80, 255)
    
    smiles_rgd_dict = state["smiles_rgd_dict"]
    subset_dir = state["subset_dir"]

    subset_names = []
    scaffold_mols = []

    for subset_name in smiles_rgd_dict:
        scaffold_id = int(subset_name.split("_")[1])
        path = os.path.join(subset_dir, f"scaffold_{scaffold_id}.sdf")
        suppl = Chem.SDMolSupplier(path)
        mol = next((m for m in suppl if m is not None), None)
        if mol:
            subset_names.append(str(scaffold_id))
            scaffold_mols.append(mol)

    if len(scaffold_mols) < 2:
        return
    
    # sort (mol, label) together
    pairs = sorted(
        zip(scaffold_mols, subset_names),
        key=lambda x: x[0].GetNumHeavyAtoms(),
        reverse=True
    )
    scaffold_mols, subset_names = map(list, zip(*pairs))

    # normalise
    scaffold_mols = [_normalize_scaffold(m) for m in scaffold_mols]

    # fingerprint count-based r=3, 4096
    gen = GetMorganGenerator(radius=3, fpSize=4096)
    fps = [gen.GetCountFingerprint(m) for m in scaffold_mols]

    # condensed distances
    n = len(fps)
    condensed = np.empty(n*(n-1)//2, dtype=float)
    k = 0
    for i in range(n-1):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            condensed[k] = 1.0 - sim
            k += 1

    # complete linkage (tight clusters)
    linkage_matrix = linkage(condensed, method='complete')

    dendro = dendrogram(linkage_matrix, labels=subset_names, no_plot=True, color_threshold=state["cluster threshold"])
    icoords = dendro["icoord"]
    dcoords = dendro["dcoord"]
    labels = dendro["ivl"]
    colors = dendro["color_list"]

    segments = []
    all_x = []
    all_y = []
    link_to_seg_ids = []  # <<<<< NEW: to recolour 3 segments / link

    with dpg.plot(label=f"Substructures Hierarchical Clustering  |  Similarity Threshold = {(1 - (state['cluster threshold'])) * 100:.0f}%",
                    parent="mol_image_group", tag="scaffold_hierarchical_dendrogram", no_menus=True, no_mouse_pos=True,
                    width=-1, 
                    height=-1) as plot:
        
        dendro_x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Subset", no_highlight=True, no_gridlines=True, no_tick_marks=True)
        dendro_y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Tanimoto Similarity (%)", no_highlight=True, no_gridlines=True, no_tick_marks=True)

        # --- DRAW lines and SAVE ids for fast recolouring ---
        for idx, (x_raw, y_raw) in enumerate(zip(icoords, dcoords)):
            color = get_color(colors[idx])
            this_link_ids = []  # 3 segment ids for this link
            for i in range(3):
                x1 = x_raw[i] / 10
                x2 = x_raw[i+1] / 10
                y1 = y_raw[i]
                y2 = y_raw[i+1]
                seg_id = dpg.draw_line(p1=(x1 + 0.5, y1), p2=(x2 + 0.5, y2), color=color, thickness=0, parent=plot)
                this_link_ids.append(seg_id)
                all_x.extend([x1, x2])
                all_y.extend([y1, y2])
            link_to_seg_ids.append(this_link_ids)

        x_ticks = tuple((lbl, i + 1) for i, lbl in enumerate(labels))
        dpg.set_axis_ticks(dendro_x_axis, x_ticks)

        # dcoord are DISTANCES — convert tick labels to SIMILARITY %
        y_ticks = tuple((f"{((1.0 - y) * 100):.0f}", y) for y in np.linspace(0.0, 1.0, 6))
        dpg.set_axis_ticks(dendro_y_axis, y_ticks)

        x_min, x_max = min(all_x), max(all_x)
        x1 = icoords[0][1] / 10
        x2 = icoords[0][2] / 10
        x_margin = abs((x2 - x1) / 2)

        # --- Thin rectangular frame around the whole dendrogram ---
        # compute data bounds (we already have x_min/x_max/y_min/y_max above)
        frame_xmin = x_min - x_margin
        frame_xmax = x_max + (x_margin * 3)
        frame_ymin = -0.01
        frame_ymax =  1.01

        # if it exists from a previous draw, remove to avoid duplicates
        if dpg.does_item_exist("dendro_frame_series"):
            dpg.delete_item("dendro_frame_series")

        # NB: line_series must be a child of the Y axis
        dpg.add_line_series(
            x=[frame_xmin, frame_xmax, frame_xmax, frame_xmin, frame_xmin],
            y=[frame_ymin, frame_ymin, frame_ymax, frame_ymax, frame_ymin],
            tag="dendro_frame_series",
            parent=dendro_y_axis,
            label=""
        )
        dpg.bind_item_theme("dendro_frame_series", _ensure_dendro_frame_theme(state))

        # Suggestion: leave X limits on auto-fit based on data (frame included)
        # dpg.set_axis_limits(dendro_x_axis, x_min - x_margin, x_max + (x_margin * 3))  # <-- keep commented/disabled

        # Fix Y limits to avoid vertical rescaling:
        dpg.set_axis_limits(dendro_y_axis, frame_ymin, frame_ymax)

        # Fit X after 1 frame (so it also includes the frame)
        dpg.set_frame_callback(1, lambda: (
            dpg.fit_axis_data(dendro_x_axis) if dpg.does_item_exist(dendro_x_axis) else None
        ))

        if dpg.does_item_exist(dendro_x_axis):
            dpg.set_frame_callback(dpg.get_frame_count() + 5, lambda: (
                dpg.set_axis_limits_auto(dendro_x_axis) if dpg.does_item_exist(dendro_x_axis) else None
            ))
        if not dpg.does_item_exist("cluster_threshold_drag"):
            dpg.add_drag_line(
                tag="cluster_threshold_drag",
                default_value=state["cluster threshold"],
                color=(200, 0, 0, 255),
                thickness=2,
                vertical=False,
                parent=plot,
                callback=lambda s, a: on_slider_threshold_change(s, a, state)
            )
            dpg.bind_item_theme("scaffold_hierarchical_dendrogram", apply_dendrogram_theme(state))

    register_plot_context_popup(
        state,
        context_key="overview_dendrogram_plot_context",
        plot_tag="scaffold_hierarchical_dendrogram",
        x_axis_tag=dendro_x_axis,
        y_axis_tag=dendro_y_axis,
        theme_kind="dendrogram",
    )

    # --- CACHE: save everything required for fast slider updates ---
    cache = _ensure_dendro_cache(state)
    cache["linkage"]        = linkage_matrix
    cache["labels_input"]   = subset_names[:]   # labels passed to scipy (string IDs)
    cache["labels_ordered"] = labels[:]         # ivl
    cache["link_to_seg_ids"]= link_to_seg_ids   # [[id,id,id], ...]

    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")


# -----------------------------------------------------------------------------
# 13. Show subset choice window
# -----------------------------------------------------------------------------
def show_subset_choice_window(state: dict[str, Any]) -> Any:
    
    # -----------------------------------------------------------------------------
    # 13.1. Calculate scaffold similarities
    # -----------------------------------------------------------------------------
    def calculate_scaffold_similarities(
        query_smiles: str,
        smiles_rgd_dict: str,
        subset_dir: str
    ) -> Any:
        fp_generators = {
            "Morgan": GetMorganGenerator(radius=2, fpSize=2048),
            "RDKit": GetRDKitFPGenerator(),
            "AtomPairs": GetAtomPairGenerator(),
            "TopologicalTorsions": GetTopologicalTorsionGenerator(),
            "MACCS": None  # Handled separately
        }
        try:
            query_mol = Chem.MolFromSmiles(query_smiles)
            if query_mol is None:
                return None

            query_fps = {
                name: gen.GetFingerprint(query_mol)
                for name, gen in fp_generators.items() if gen is not None
            }
            query_fps["MACCS"] = MACCSkeys.GenMACCSKeys(query_mol)

            all_similarities = {name: [] for name in fp_generators}
            all_similarities["MACCS"] = []

            for subset_name in smiles_rgd_dict:
                scaffold_id = int(subset_name.split("_")[1])
                scaffold_path = os.path.join(subset_dir, f"scaffold_{scaffold_id}.sdf")
                suppl = Chem.SDMolSupplier(scaffold_path)
                scaffold_mol = next((m for m in suppl if m is not None), None)

                if scaffold_mol:
                    for name, query_fp in query_fps.items():
                        if name == "MACCS":
                            scaffold_fp = MACCSkeys.GenMACCSKeys(scaffold_mol)
                        else:
                            scaffold_fp = fp_generators[name].GetFingerprint(scaffold_mol)

                        sim = DataStructs.TanimotoSimilarity(query_fp, scaffold_fp)
                        all_similarities[name].append((subset_name, sim))

            best_method = max(
                all_similarities.items(),
                key=lambda item: max([x[1] for x in item[1]] if item[1] else [0])
            )[0]

            log_event("Overview", f"Best fingerprint method: {best_method}", indent=1)

            best_sorted = sorted(all_similarities[best_method], key=lambda x: x[1], reverse=True)
            return [subset for subset, _ in best_sorted]

        except Exception as e:
            log_exception("Overview", "Query similarity error", e, indent=1)
            return None
                

    # -----------------------------------------------------------------------------
    # 13.2. On subset sort choice
    # -----------------------------------------------------------------------------
    def on_subset_sort_choice(sender: Any, app_data: Any, user_data: Any) -> None:

        show_query_input = app_data == "Query similarity"
        dpg.configure_item("overview_subset_query_input", show=show_query_input)

        if app_data == "Number of molecules":
            # no clustering: clean up dendrogram UI
            _clear_clustering_ui()

            state["subset_order_query"] = state["subset_order_original"]
            state["subset_cluster_groups"] = {}
            refresh_subset_buttons(state)

        elif app_data == "Query similarity":
            # no clustering: clean up dendrogram UI
            _clear_clustering_ui()

            query_smiles = dpg.get_value("overview_subset_query_input").strip()
            if query_smiles:
                order = calculate_scaffold_similarities(
                    query_smiles,
                    state["smiles_rgd_dict"],
                    state["subset_dir"]
                )
                if order:
                    state["subset_order_query"] = order
                    state["subset_cluster_groups"] = {}
                    refresh_subset_buttons(state)

        elif app_data == "Similarity Clustering":
            draw_loading_screen(state)

            # create dendrogram/slider UI
            if not dpg.does_item_exist("scaffold_hierarchical_dendrogram"):

                # perform clustering and DRAW dendrogram/slider
                if "cluster threshold" not in state:
                    state["cluster threshold"] = 0.2  # distance = 0.2 => similarity 80%

                order, labels, names = sort_by_scaffold_similarity(
                    state["smiles_rgd_dict"],
                    state["subset_dir"],
                    threshold=state["cluster threshold"]
                )
                if order:
                    state["subset_order_query"] = order
                    if labels and names:
                        state["subset_cluster_groups"] = {
                            name: f"Cluster {labels[i]}" for i, name in enumerate(names)
                        }
                    refresh_subset_buttons(state)
                    
                    draw_scaffold_dendrogram(state)
                    
            on_button_click("subset_1", state)
            show_results_window("subset_1", state)


    state["subset_order_original"] = list(state["smiles_rgd_dict"].keys())
    state["subset_order_query"] = list(state["smiles_rgd_dict"].keys())

    subset_number = len(state["smiles_rgd_dict"])

    with dpg.group(parent="subset_choice", height=-1):
        dpg.configure_item("subset_choice_column", label=f"Subsets: {subset_number}")

        with dpg.group(tag="subset_sorting_mode_combo_group", parent="image_checkboxes_window"):
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Sort subsets by:")
                dpg.add_combo(items=["Number of molecules", "Query similarity", "Similarity Clustering"],
                                tag="overview_subset_sort_choice",
                                width=-1,
                                default_value="Number of molecules",
                                callback=on_subset_sort_choice)
            
            dpg.add_input_text(hint="Query SMILES", show=False, tag="overview_subset_query_input",
                            width=-1,
                            callback=lambda s, a: on_subset_sort_choice(None, "Query similarity", None))
            
            dpg.bind_item_theme("overview_subset_query_input", apply_bordered_input_text_theme(state))

        with dpg.child_window(tag="subset_choice_group", border=False, width=-1, height=-1,
                                no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False):
            
            dpg.add_spacer(height=state["win_spacer"])

            for subset in state["total_r_groups_dict"].keys():
                button_tag = subset
                dpg.add_button(label=button_tag.replace("subset_","Subset "), tag=button_tag,
                                width=-1, 
                                callback=lambda id=button_tag: (
                                    update_molecule_choice_status(id, state), 
                                    update_properties_windows(id, state), 
                                    update_activities_windows(id, state),
                                    build_enrichment_layout(id, state),
                                    on_button_click(id, state),
                                    show_results_window(id, state)
                                )
                            )
        dpg.bind_item_theme("subset_choice_group", apply_inner_child_theme())

    update_molecule_choice_status("subset_1", state)
    update_properties_windows("subset_1", state)
    update_activities_windows("subset_1", state)
    on_button_click("subset_1", state)
    show_results_window("subset_1", state)
    refresh_overrides(state)


# -----------------------------------------------------------------------------
# 14. Show molecule choice window
# -----------------------------------------------------------------------------
def show_molecule_choice_window(state: dict[str, Any]) -> None:

    console_group_3 = dpg.add_group(parent="molecule_choice", height=-1)
    state["console_group_3"] = console_group_3   

   

# -----------------------------------------------------------------------------
# 15. Update molecule choice status
# -----------------------------------------------------------------------------
def update_molecule_choice_status(parent_subset: str, state: dict[str, Any]) -> None:

    # 5.2.1: Reset molecule and R-group panes
    dpg.delete_item("molecule_choice", children_only=True)
    dpg.delete_item("r_group_choice", children_only=True)
    show_molecule_choice_window(state)
    show_r_groups_choice_window(state)

    # 5.2.2: Header and list of molecules with callbacks
    console_group_3 = state["console_group_3"]
    dpg.configure_item("molecule_choice_column", label=f"Molecules: {len(state['smiles_rgd_dict'][parent_subset])}")

    with dpg.child_window(parent=console_group_3, tag="molecule_choice_group", border=False, width=-1, height=-1,
                          no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False):
        
        dpg.add_spacer(height=state["win_spacer"])

        for molecule in state["smiles_rgd_dict"][parent_subset].keys():
            button_tag = f"{parent_subset}_{molecule}"
            label = molecule.replace("mol_", "Mol ")

            dpg.add_button(label=label, tag=button_tag, width=-1,
                           callback=lambda id=button_tag: (
                               update_properties_windows(id, state), 
                               update_activities_windows(id, state),
                               update_r_groups_choice_status(id, state), 
                               build_enrichment_layout(id, state),
                               on_button_click(id, state),
                               show_results_window(id, state)
                            )
                        )
    dpg.bind_item_theme("molecule_choice_group", apply_inner_child_theme())   
    refresh_overrides(state)


# -----------------------------------------------------------------------------
# 16. Show r groups choice window
# -----------------------------------------------------------------------------
def show_r_groups_choice_window(state: dict[str, Any]) -> None:
    
    console_group_4 = dpg.add_group(parent="r_group_choice", height=-1)
    state["console_group_4"] = console_group_4   


   

# -----------------------------------------------------------------------------
# 17. Update r groups choice status
# -----------------------------------------------------------------------------
def update_r_groups_choice_status(subset_molecule: str, state: dict[str, Any]) -> None:

    # 6.2.1: Reset and parse identifiers
    dpg.delete_item("r_group_choice", children_only=True)
    show_r_groups_choice_window(state)
    subset = "_".join(subset_molecule.split("_")[:2])
    molecule = "_".join(subset_molecule.split("_")[2:])
    r_groups_list = state["smiles_rgd_dict"][subset][molecule]
    total_r_groups = state["total_r_groups_dict"][subset]

    # 6.2.2: Build grid with active R-group buttons (and spacers for missing ones)
    console_group_4 = state["console_group_4"]
    with dpg.child_window(parent=console_group_4, tag="r_group_choice_group", border=False, width=-1, height=-1,
                          no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False):
        
        dpg.add_spacer(height=state["win_spacer"])

        for r_group in total_r_groups:
            button_tag_r = f"{subset_molecule}_{r_group}"
            if r_group in r_groups_list:
                dpg.add_button(label=r_group, tag=button_tag_r, width=-1, 
                               callback=lambda id=button_tag_r: (
                                   update_properties_windows(id, state), 
                                   on_button_click(id, state),
                                   build_enrichment_layout(id, state),
                                   show_results_window(id, state)
                                )
                            )
            else:
                dpg.add_spacer(height=(state["win_spacer"] * 6))
    dpg.bind_item_theme("r_group_choice_group", apply_inner_child_theme())   
    refresh_overrides(state)


# -----------------------------------------------------------------------------
# 18. Show properties windows
# -----------------------------------------------------------------------------
def show_properties_windows(state: dict[str, Any]) -> None:
    console_group = dpg.add_group(parent="properties_window", height=-1)
    state["console_group_5"] = console_group   


# -----------------------------------------------------------------------------
# 19. Update properties windows
# -----------------------------------------------------------------------------
def update_properties_windows(id: Any, state: dict[str, Any]) -> None:

    # 7.2.0: Guarded refresh with loading overlay
    draw_loading_screen(state, bg=False)

    dpg.delete_item("properties_window", children_only=True)
    show_properties_windows(state)
    show_activities_windows(state)

    console_group_5 = state["console_group_5"]
    query = id.split("_")

    smiles_rgd_dict = state["smiles_rgd_dict"]
    total_r_groups_dict = state["total_r_groups_dict"]
    bioact_types_dict = state["bioact_types_dict"]
    r_counts = state["r_counts"]
    props_dict = state["properties_dict"]

    dpg.add_spacer(height=state["win_spacer"], parent=console_group_5)

    # If the query is a subset 
    if len(query) == 2:
        dpg.configure_item("properties_column", label="Subset Properties")
        subset = id
        scaffold_smiles = smiles_rgd_dict[subset].get("mol_1", smiles_rgd_dict[subset].get("mol_2", {})).get("Core", "")
        # First remove group in round brackets, then remove isolated [*:n]
        cleaned_smiles = re.sub(r'\(\[\*\:\d+\]\)', '', scaffold_smiles)  # removes ([*:n])
        cleaned_smiles = re.sub(r'\[\*\:\d+\]', '', cleaned_smiles)  # removes isolated [*:n]

        # 7.2.1.1: Counts and distributions
        mol_number = len(smiles_rgd_dict[subset])
        r_groups_number = len(total_r_groups_dict[subset])

        bioactivities_list = bioact_types_dict[subset]["bioactivities"]
        bioactivity_counts = {bio: 0 for bio in bioactivities_list}
        targets_list = bioact_types_dict[subset]["targets"]
        target_counts = {tgt: 0 for tgt in targets_list}
        organisms_list = bioact_types_dict[subset]["organisms"]
        organism_counts = {org: 0 for org in organisms_list}

        for mol_data in props_dict[subset].values():
            activities = mol_data.get("activities", {})
            
            for activity_entry in activities.values():

                for k, v in activity_entry.items():
                    if k.startswith("Activity") and isinstance(v, str):
                        bio_type = v.split(" ")[0]
                        if bio_type in bioactivity_counts:
                            bioactivity_counts[bio_type] += 1

                for k, v in activity_entry.items():
                    if k.startswith("Target") and isinstance(v, str):
                        if v in target_counts:
                            target_counts[v] += 1

                for k, v in activity_entry.items():
                    if k.startswith("Organism") and isinstance(v, str):
                        if v in organism_counts:
                            organism_counts[v] += 1

        bioactivities_tuple = list(bioactivity_counts.items())
        targets_tuple = list(target_counts.items())
        organism_tuple = list(organism_counts.items())

        # 7.2.1.2: Render subset summary
        with dpg.child_window(parent=console_group_5, border=False, 
                              no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False, 
                              width=-1, height=-1):
            with dpg.group(horizontal=True):
                dpg.add_text("SMILES WITH Rs:")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=scaffold_smiles, width=-1, tag="scaffold_smiles_input",
                                   auto_select_all=True, readonly=True)
            with dpg.group(horizontal=True):
                dpg.add_text("CLEANED SMILES:")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=cleaned_smiles, width=-1, auto_select_all=True, readonly=True)
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                dpg.add_text("MOLECULES:")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_text(mol_number)
            with dpg.group(horizontal=True):
                dpg.add_text("R-GROUPS:")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_text(r_groups_number)

            if bioactivities_tuple:
                dpg.add_spacer(height=20)
                dpg.add_separator(label="ACTIVITIES:")
                for bioact_name, count in bioactivities_tuple:
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"{bioact_name} =")
                        change_font_type(dpg.last_item(), "bold", state)
                        dpg.add_text(f"{count} molecules")

            if targets_tuple:
                dpg.add_spacer(height=20)
                dpg.add_separator(label="TARGETS:")
                for target_name, count in targets_tuple:
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"{target_name} =")
                        change_font_type(dpg.last_item(), "bold", state)
                        dpg.add_text(f"{count} molecules")

            if organism_tuple:
                dpg.add_spacer(height=20)
                dpg.add_separator(label="ORGANISMS:")
                for organism_name, count in organism_tuple:
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"{organism_name} =")
                        change_font_type(dpg.last_item(), "bold", state)
                        dpg.add_text(f"{count} molecules")

    # If the query is a molecule
    elif len(query) == 4:
        dpg.configure_item("properties_column", label="Molecule Properties")
        subset = query[0] + "_" + query[1]
        molecule = query[2] + "_" + query[3]
        props = props_dict[subset][molecule]

        properties = props.get("properties", {})
        activities = props.get("activities", {})

        # 7.2.2.0: Extract identifiers and descriptors
        original_id = properties.get("original_id", "N/A")
        name = properties.get("name", "N/A")
        chembl_id = properties.get("chembl_id", "N/A")
        pubchem_cid = properties.get("pubchem_cid", "N/A")
        formula = properties.get("formula", "N/A")
        smiles = properties.get("smiles", "")

        mw = float(properties.get("molecular_weight", 0))
        logp = float(properties.get("logp", 0))
        hba = properties.get("hba", 0)
        hbd = properties.get("hbd", 0)
        rotbonds = properties.get("RotBonds", 0)
        tpsa = float(properties.get("tpsa", 0))
        molar_refractivity = float(properties.get("molar_refractivity", 0))
        gast_range = float(properties.get("gasteiger_range", 0))
        gast_mean = float(properties.get("gasteiger_mean_abs", 0))
        fraction_csp3 = float(properties.get("fraction_csp3", 0))
        percent_csp3 = fraction_csp3 * 100
        num_rings = int(properties.get("num_rings", 0))
        num_aromatic_rings = int(properties.get("num_aromatic_rings", 0))
        num_aliphatic_rings = int(properties.get("num_aliphatic_rings", 0))
        num_saturated_rings = int(properties.get("num_saturated_rings", 0))
        kappa1 = float(properties.get("kappa1", 0))
        kappa2 = float(properties.get("kappa2", 0))
        kappa3 = float(properties.get("kappa3", 0))
        chi0 = float(properties.get("chi0", 0))
        chi1 = float(properties.get("chi1", 0))
        chi2 = float(properties.get("chi2", 0))
        chi3 = float(properties.get("chi3", 0))
        chi4 = float(properties.get("chi4", 0))


        with dpg.child_window(parent=console_group_5, tag="prop_group", border=False, width=-1, height=-1,
                            no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False):

            # 7.2.2.1.1: Identifiers
            dpg.add_separator(label="IDENTIFIERS")
            
            if name != "N/A" and name != "":
                with dpg.group(horizontal=True):
                    dpg.add_text(f"  Name   :")
                    change_font_type(dpg.last_item(), "bold", state)
                    dpg.add_input_text(default_value=f"{name}", width=-1, auto_select_all=True, readonly=True)
            
            if original_id != "N/A":
                with dpg.group(horizontal=True):
                    dpg.add_text(f"  Input file ID   :")
                    change_font_type(dpg.last_item(), "bold", state)
                    dpg.add_input_text(default_value=f"{original_id}", width=-1, auto_select_all=True, readonly=True)

            if chembl_id != "N/A":
                with dpg.group(horizontal=True):
                    dpg.add_text("  ChEMBL ID   :")
                    change_font_type(dpg.last_item(), "bold", state)    
                    dpg.add_input_text(default_value=chembl_id, width=-1, auto_select_all=True, readonly=True)
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Search on ChEMBL", 
                                callback=lambda: webbrowser.open(f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"))
                    dpg.add_button(label="Search on PubChem", 
                                callback=lambda: webbrowser.open(f"https://pubchem.ncbi.nlm.nih.gov/compound/{chembl_id}"))
            
            if pubchem_cid != "N/A":
                with dpg.group(horizontal=True):
                    dpg.add_text("  PUBCHEM CID   :")
                    change_font_type(dpg.last_item(), "bold", state)
                    dpg.add_input_text(default_value=pubchem_cid, width=-1, auto_select_all=True, readonly=True)

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Search on PubChem",
                                callback=lambda: webbrowser.open(f"https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem_cid}"))

            if formula != "N/A":
                with dpg.group(horizontal=True):
                    dpg.add_text("  Molecular Formula   :")
                    change_font_type(dpg.last_item(), "bold", state)
                    dpg.add_input_text(default_value=formula, width=-1, auto_select_all=True, readonly=True)
            
            with dpg.group(horizontal=True):
                dpg.add_text("  SMILES   :")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=smiles, width=-1, auto_select_all=True, readonly=True)

            dpg.add_spacer(height=20)
            dpg.add_separator(label="DESCRIPTORS")
            
            # 7.2.2.1.2: Rule-of-Five summary (with tooltip)
            with dpg.group(horizontal=True):
                violations_str = properties.get("Lipinsky's RO5 Violations", "None")
                if violations_str.strip() == "None":
                    num_violations = ""
                else:
                    num_violations = f"({len([v.strip() for v in violations_str.split(',') if v.strip()])}) "
                dpg.add_text("  Ro5 Violations   :", tag="ro5_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{num_violations}{violations_str}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("ro5_text"):
                    dpg.add_text("Lipinski's Rule of Five violations")

            dpg.add_spacer(height=20)
            dpg.add_separator(label="CHEMIOPHYSICAL PROPERTIES")

            # 7.2.2.1.3: Selected properties + contribution maps
            with dpg.group(horizontal=True):
                dpg.add_text("  Molecular Weight   =", tag="mw_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{mw:.3f} Da", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("mw_text"):
                    dpg.add_text("Exact molecular weight")

            with dpg.group(horizontal=True):
                dpg.add_text("  LogP   =", tag="logp_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{logp:.3f}", auto_select_all=True, readonly=True)
                with dpg.tooltip("logp_text"):
                    dpg.add_text("Estimated lipophilicity (Crippen's logP)")
                
            dpg.add_button(label="Atomic Contribution to logP", tag="logp_acm",
                        callback=lambda sender: show_logP_atomic_contribution_map(smiles, state))

            with dpg.group(horizontal=True):
                dpg.add_text("  Molar Refractivity   =", tag="mr_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{molar_refractivity:.2f}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("mr_text"):
                    dpg.add_text("Estimated molar refractivity (related to polarizability/volume)")
            
            with dpg.group(horizontal=True):
                dpg.add_text("  TPSA   =", tag="tpsa_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{tpsa:.2f} \u00C5\u00B2", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("tpsa_text"):
                    dpg.add_text("Topological Polar Surface Area (2D estimation of SASA)")

            with dpg.group(horizontal=True):
                dpg.add_text("  G. range   =", tag="gast_range_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{gast_range:.2f}", auto_select_all=True, readonly=True)
                with dpg.tooltip("gast_range_text"):
                    dpg.add_text("Range of Gasteiger partial charges")

            with dpg.group(horizontal=True):
                dpg.add_text("  G. mean   =", tag="gast_mean_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{gast_mean:.2f}", auto_select_all=True, readonly=True)
                with dpg.tooltip("gast_mean_text"):
                    dpg.add_text("Absolute mean of Gasteiger partial charges")

            dpg.add_button(label="Atomic Contribution to Gasteiger Charge", tag="gast_range_acm",
                callback=lambda sender: show_gasteiger_atomic_contribution_map(smiles, state))


            dpg.add_spacer(height=20)
            dpg.add_separator(label="HYDROGEN BONDS")

            dpg.add_button(label="Show HBA/HBD", tag="show_hba_hbd_button",
                            callback=lambda sender: show_hba_hbd(smiles, state))
            with dpg.tooltip("show_hba_hbd_button"):
                dpg.add_text("HBA/HBD visualization is based on SMARTS pattern\nrecognition and may differ from Lipinski count")
                
            with dpg.group(horizontal=True):
                dpg.add_text("  HBA   =", tag="hba_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{hba}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("hba_text"):
                    dpg.add_text("Number of hydrogen bond acceptors")

            with dpg.group(horizontal=True):
                dpg.add_text("  HBD   =", tag="hbd_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{hbd}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("hbd_text"):
                    dpg.add_text("Number of hydrogen bond donors")

            dpg.add_spacer(height=20)
            dpg.add_separator(label="FLEXIBILITY AND GEOMETRY") 

            with dpg.group(horizontal=True):
                dpg.add_text("  Rotatable Bonds   =", tag="rot_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{rotbonds}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("rot_text"):
                    dpg.add_text("Number of rotatable bonds")

            with dpg.group(horizontal=True):
                dpg.add_text("  Fraction Csp\u00B3   =", tag="csp3_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{percent_csp3:.2f} %", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("csp3_text"):
                    dpg.add_text("Fraction (as percentage) of carbon atoms with sp\u00B3 hybridization")

            dpg.add_spacer(height=20)
            dpg.add_separator(label="RINGS")

            with dpg.group(horizontal=True):
                dpg.add_text("  Total   =", tag="rings_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{num_rings}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("rings_text"):
                    dpg.add_text("Total number of rings in the molecule")

            with dpg.group(horizontal=True):
                dpg.add_text("  Aromatic   =", tag="ar_rings_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{num_aromatic_rings}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("ar_rings_text"):
                    dpg.add_text("Number of aromatic rings")

            with dpg.group(horizontal=True):
                dpg.add_text("  Aliphatic   =", tag="aliph_rings_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{num_aliphatic_rings}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("aliph_rings_text"):
                    dpg.add_text("Number of aliphatic rings")

            with dpg.group(horizontal=True):
                dpg.add_text("  Saturated   =", tag="sat_rings_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{num_saturated_rings}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("sat_rings_text"):
                    dpg.add_text("Number of fully saturated rings")

            dpg.add_spacer(height=20)
            dpg.add_separator(label="TOPOLOGICAL INDICES")

            with dpg.group(horizontal=True):
                dpg.add_text("  Kappa 1   =", tag="kappa1_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{kappa1:.3f}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("kappa1_text"):
                    dpg.add_text("Linearity index. Increases with branching and compactness")

            with dpg.group(horizontal=True):
                dpg.add_text("  Kappa 2   =", tag="kappa2_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{kappa2:.3f}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("kappa2_text"):
                    dpg.add_text("Cyclicity index. Increases with number and size of rings")

            with dpg.group(horizontal=True):
                dpg.add_text("  Kappa 3   =", tag="kappa3_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{kappa3:.3f}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("kappa3_text"):
                    dpg.add_text("Complexity index. Sensitive to overall shape and 3D structure")

            with dpg.group(horizontal=True):
                dpg.add_text("  Chi 0   =", tag="chi0_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{chi0:.4f}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("chi0_text"):
                    dpg.add_text("Zero-order index. Counts atom types and connectivity")

            with dpg.group(horizontal=True):
                dpg.add_text("  Chi 1   =", tag="chi1_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{chi1:.4f}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("chi1_text"):
                    dpg.add_text("First-order index. Reflects bond connectivity and branching")

            with dpg.group(horizontal=True):
                dpg.add_text("  Chi 2   =", tag="chi2_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{chi2:.4f}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("chi2_text"):
                    dpg.add_text("Second-order index. Sensitive to long chains and molecular spread")

            with dpg.group(horizontal=True):
                dpg.add_text("  Chi 3   =", tag="chi3_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{chi3:.4f}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("chi3_text"):
                    dpg.add_text("Third-order index. Highlights connectivity patterns and branching")

            with dpg.group(horizontal=True):
                dpg.add_text("  Chi 4   =", tag="chi4_text")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=f"{chi4:.4f}", width=-1, auto_select_all=True, readonly=True)
                with dpg.tooltip("chi4_text"):
                    dpg.add_text("Fourth-order index. Highlights complex connectivity patterns")
                                                    
            # 7.2.2.5: Theme binding for property group
            dpg.bind_item_theme("prop_group", apply_inner_child_theme())


    # If the query is an R-group
    elif len(query) == 5:
        dpg.configure_item("properties_column", label="R-Group Properties")
        subset = query[0] + "_" + query[1]
        molecule = query[2] + "_" + query[3]
        r = query[4]
        r_smiles = smiles_rgd_dict[subset][molecule].get(r, "")
        count = r_counts[subset][r][r_smiles]

        # 7.2.3.1: Render R-group SMILES and count
        with dpg.child_window(parent=console_group_5, border=False, tag="r_prop_group",
                              no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False, 
                              width=-1, height=-1):
            with dpg.group(horizontal=True):
                dpg.add_text("SMILES:")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_input_text(default_value=r_smiles, tag="r_smiles_input",
                                   auto_select_all=True, readonly=True,
                                   width=-1 )
            with dpg.group(horizontal=True):  
                dpg.add_text("Occurrence:")
                change_font_type(dpg.last_item(), "bold", state)
                dpg.add_text(f"{count}/{len(smiles_rgd_dict[subset])} molecules")


    # 7.2.4: Apply input theme for better readability
    dpg.bind_item_theme("properties_window", apply_input_text_theme())       


# -----------------------------------------------------------------------------
# 20. Show activities windows
# -----------------------------------------------------------------------------
def show_activities_windows(state: dict[str, Any]) -> None:
    console_group = dpg.add_group(parent="activities_window", height=-1)
    state["console_group_8"] = console_group   


# -----------------------------------------------------------------------------
# 21. Update activities windows
# -----------------------------------------------------------------------------
def update_activities_windows(id: Any, state: dict[str, Any]) -> Any:

    # 7.2.0: Guarded refresh with loading overlay
    draw_loading_screen(state, bg=False)

    dpg.delete_item("activities_window", children_only=True)
    show_properties_windows(state)
    show_activities_windows(state)
    console_group_8 = state["console_group_8"]
    query = id.split("_")

    props_dict = state["properties_dict"]


    dpg.add_spacer(height=state["win_spacer"], parent=console_group_8)

    # If the query is a molecule
    if len(query) == 4:
        subset = query[0] + "_" + query[1]
        molecule = query[2] + "_" + query[3]
        props = props_dict[subset][molecule]

        activities = props.get("activities", {})

        with dpg.child_window(parent=console_group_8, tag="act_group", border=False, width=-1, height=-1,
                            no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False):

            def go_prev() -> None:
                """
                Navigates to the previous suffix index in the activity view and updates the display.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                # 7.2.2.1.4: Navigate to previous activity block
                current_suffix["index"] = (current_suffix["index"] - 1) % len(suffix_keys)
                update_activity_view()

            def go_next() -> None:
                """
                Navigates to the next suffix index in the activity view and updates the display.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                # 7.2.2.1.5: Navigate to next activity block
                current_suffix["index"] = (current_suffix["index"] + 1) % len(suffix_keys)
                update_activity_view()

            suffix_keys = sorted(activities.keys(), key=lambda x: int(x.split("_")[-1]))
            current_suffix = {"index": 0}
            single_activity = len(suffix_keys) in (0, 1)

            # 7.2.2.2: Header and navigator
            dpg.add_separator(label="ACTIVITY")
            if not single_activity:
                with dpg.group(horizontal=True):
                    dpg.add_button(arrow=True, direction=0, callback=go_prev)
                    dpg.add_text(f"Activity {current_suffix['index']+1}/{len(suffix_keys)}", tag="suffix_counter")
                    change_font_type(dpg.last_item(), "bold", state)
                    dpg.add_button(arrow=True, direction=1, callback=go_next)

            # 7.2.2.3: Activity detail block
            def strip_suffix(label: str) -> Any:
                """
                Removes numeric suffix from a label string if present (e.g., 'Mol_1' → 'Mol').

                Args:
                    label (str): Input label string.

                Returns:
                    str: Label without numeric suffix.
                """
                return parts[0] if len(parts := label.rsplit("_", 1)) == 2 and parts[1].isdigit() else label

            def format_label(label: str) -> Any:
                """
                Formats the label by stripping numeric suffix and replacing underscores with spaces.

                Args:
                    label (str): Input label string.

                Returns:
                    str: Formatted label.
                """
                return strip_suffix(label).replace("_", " ")

            def update_activity_view() -> None:
                """
                Updates the activity view display by clearing and redrawing.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """

                dpg.delete_item("activity_block", children_only=True)

                if not activities or not suffix_keys:
                    with dpg.group(horizontal=True, parent="activity_block"):
                        dpg.add_text("No activity data available.")
                    if not single_activity:
                        dpg.set_value("suffix_counter", "Activity 0/0")
                    return

                suffix = suffix_keys[current_suffix["index"]]
                block = activities[suffix]

                preferred = [
                    "Activity", "pValue",
                    "Assay", "Assay_Description", "Assay_Type", "BAO_Label",
                    "Assay_ChEMBL_ID", "Assay_PubChem_AID", "BAO_Format",
                    "Action_Type", "Action_Description",
                    "Organism", "Target", "Target_ChEMBL_ID", "Target_Uniprot_ID",
                    "Max_Phase", "Comment"
                ]

                ordered = []
                rest = []

                for k in block:
                    base = strip_suffix(k)
                    if base in preferred:
                        ordered.append(k)
                    else:
                        rest.append(k)

                ordered = [k for pref in preferred for k in ordered if strip_suffix(k) == pref] + rest

                section_map = {
                    "ASSAY": {
                        "keys": {
                            "Assay", "Assay_Description", "Assay_Type",
                            "BAO_Label", "Assay_ChEMBL_ID",
                            "Assay_PubChem_AID", "BAO_Format"
                        },
                        "inserted": False
                    },
                    "TARGET PROTEIN AND ORGANISM": {
                        "keys": {
                            "Organism", "Target",
                            "Target_ChEMBL_ID", "Target_Uniprot_ID"
                        },
                        "inserted": False
                    }
                }

                assay_chembl_ID = next(
                    (v for k, v in block.items() if strip_suffix(k) == "Assay_ChEMBL_ID"), "N/A"
                )
                assay_pubchem_AID = next(
                    (v for k, v in block.items() if strip_suffix(k) == "Assay_PubChem_AID"), "N/A"
                )
                BAO_Format = next(
                    (v for k, v in block.items() if strip_suffix(k) == "BAO_Format"), "N/A"
                )
                target_ID = next(
                    (v for k, v in block.items() if strip_suffix(k) == "Target_ChEMBL_ID"), "N/A"
                )

                for label in ordered:
                    clean = format_label(label)
                    val = block[label]
                    base = strip_suffix(label)

                    # Insert labeled separator if needed
                    for section_name, section in section_map.items():
                        if base in section["keys"] and not section["inserted"]:
                            dpg.add_spacer(height=20, parent="activity_block")
                            dpg.add_separator(label=section_name, parent="activity_block")
                            section["inserted"] = True
                            break


                    if base == "Assay_ChEMBL_ID":
                        with dpg.group(horizontal=True, parent="activity_block"):
                            dpg.add_text(f"{clean}  :")
                            change_font_type(dpg.last_item(), "bold", state)
                            dpg.add_input_text(default_value=assay_chembl_ID, width=-1,
                                        auto_select_all=True, readonly=True)

                        with dpg.group(horizontal=True, parent="activity_block"): 
                            dpg.add_button(
                                label="Search assay on ChEMBL",
                                callback=lambda: webbrowser.open(
                                    f"https://www.ebi.ac.uk/chembl/explore/assay/{assay_chembl_ID}"
                                )
                            )
                            dpg.add_button(
                                label="Search assay on PubChem",
                                callback=lambda: webbrowser.open(
                                    f"https://pubchem.ncbi.nlm.nih.gov/target/{assay_chembl_ID}"
                                )
                            )

                    elif base == "Assay_PubChem_AID":
                        with dpg.group(horizontal=True, parent="activity_block"):
                            dpg.add_text(f"{clean}  :")
                            change_font_type(dpg.last_item(), "bold", state)
                            dpg.add_input_text(default_value=assay_pubchem_AID, width=-1,
                                            auto_select_all=True, readonly=True)
                            
                        dpg.add_button(
                            label="Search assay on PubChem", parent="activity_block",
                            callback=lambda: webbrowser.open(
                                f"https://pubchem.ncbi.nlm.nih.gov/target/{assay_pubchem_AID}"
                            )
                        )
 
                    elif base == "BAO_Format":
                        with dpg.group(horizontal=True, parent="activity_block"):
                            dpg.add_text(f"{clean}  :")
                            change_font_type(dpg.last_item(), "bold", state)
                            dpg.add_input_text(default_value=BAO_Format, width=-1,
                                            auto_select_all=True, readonly=True)
                            
                        iri = f"http://www.bioassayontology.org/bao#{BAO_Format}"
                        encoded_iri = urllib.parse.quote(iri, safe="")
                        bioportal_url = (
                            "https://bioportal.bioontology.org/ontologies/BAO"
                            f"?p=classes&conceptid={encoded_iri}"
                        )
                        dpg.add_button(
                            label="Search assay on BioPortal", parent="activity_block",
                            callback=lambda: webbrowser.open(bioportal_url)
                        )

                    elif base == "Target_ChEMBL_ID":
                        with dpg.group(horizontal=True, parent="activity_block"):
                            dpg.add_text(f"{clean}  :")
                            change_font_type(dpg.last_item(), "bold", state)
                            dpg.add_input_text(default_value=target_ID, width=-1,
                                            auto_select_all=True, readonly=True)
                        
                        with dpg.group(horizontal=True, parent="activity_block"): 
                            dpg.add_button(
                                label="Search target on ChEMBL",
                                callback=lambda: webbrowser.open(
                                    f"https://www.ebi.ac.uk/chembl/explore/target/{target_ID}"
                                )
                            )
                            dpg.add_button(
                                label="Search target on PubChem",
                                callback=lambda: webbrowser.open(
                                    f"https://pubchem.ncbi.nlm.nih.gov/target/{target_ID}"
                                )
                            )

                    elif base == "Target_Uniprot_ID":
                        with dpg.group(horizontal=True, parent="activity_block"):
                            dpg.add_text(f"{clean}  :")
                            change_font_type(dpg.last_item(), "bold", state)
                            dpg.add_input_text(default_value=val, width=-1,
                                            auto_select_all=True, readonly=True)
                        
                        with dpg.group(horizontal=True, parent="activity_block"): 
                            dpg.add_button(
                                label="Search target on UniProt",
                                callback=lambda: webbrowser.open(
                                    f"https://www.uniprot.org/uniprot/{val}"
                                )
                            )
                            dpg.add_button(
                                label="Search target on PubChem",
                                callback=lambda: webbrowser.open(
                                    f"https://pubchem.ncbi.nlm.nih.gov/target/{val}"
                                )
                            )

                    else:
                        # Activity formatting (Unicode operators)
                        if base == "Activity" and isinstance(val, str):
                            val = val.replace(">=", "≥").replace("<=", "≤")
                        with dpg.group(horizontal=True, parent="activity_block"):
                            dpg.add_text(f"{clean}  :")
                            change_font_type(dpg.last_item(), "bold", state)
                            dpg.add_input_text(default_value=val, width=-1,
                                            auto_select_all=True, readonly=True)
                            

                if not single_activity:
                    dpg.set_value(
                        "suffix_counter",
                        f"Activity {current_suffix['index'] + 1}/{len(suffix_keys)}"
                    )
                    
            # 7.2.2.4.7: First draw
            with dpg.group(tag="activity_block"):
                update_activity_view()

            # 7.2.2.5: Theme binding for property group
            dpg.bind_item_theme("act_group", apply_inner_child_theme())


    # 7.2.4: Apply input theme for better readability
    dpg.bind_item_theme("activities_window", apply_input_text_theme())   


# -----------------------------------------------------------------------------
# 22. Autoscroll to button
# -----------------------------------------------------------------------------
def autoscroll_to_button(sender: Any, app_data: Any, user_data: Any) -> None:
    button_id = user_data

    if "scaff_img" in sender:
        container = "subset_choice_group"
    elif "mol_img" in sender:
        container = "molecule_choice_group"
    elif "rgroup_img" in sender:
        container = "r_group_choice_group"
        
    button_y = dpg.get_item_state(button_id)["pos"][1]
    dpg.set_y_scroll(container, button_y)
    dpg.set_y_scroll("overview_tab_child", 0)


# -----------------------------------------------------------------------------
# 23. Image click callback
# -----------------------------------------------------------------------------
def image_click_callback(sender: Any, app_data: Any, user_data: Any) -> None:
    smiles_str, state, label_tag, subset = user_data

    # call autoscroll
    autoscroll_to_button(sender, app_data, subset)

    # call export svg
    export_svg_callback(sender, app_data, (smiles_str, state, label_tag))


def _bind_compact_image_group_theme(group_tag: str) -> None:
    """
    Apply a compact spacing theme to the overview image row group.

    Args:
        group_tag (str): Tag of the horizontal Dear PyGui group that contains
            the structure images.

    Returns:
        None: This helper creates and binds a reusable compact group theme.
    """
    theme_tag = "overview_compact_image_group_theme"
    if not dpg.does_item_exist(theme_tag):
        with dpg.theme(tag=theme_tag):
            with dpg.theme_component(dpg.mvGroup):
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 2, 0, category=dpg.mvThemeCat_Core)
    if dpg.does_item_exist(group_tag):
        dpg.bind_item_theme(group_tag, theme_tag)


# -----------------------------------------------------------------------------
# 24. Build align checkbox
# -----------------------------------------------------------------------------
def build_align_checkbox(state: dict[str, Any]) -> None:

    if not dpg.does_item_exist("overview_align_checkbox"):
        with dpg.group(parent="image_checkboxes_window", before="subset_sorting_mode_combo_group", horizontal=True):
            dpg.add_checkbox(
                label="Align to molecule",
                tag="overview_align_checkbox",
                default_value=state.get("overview_align", False),
                callback=toggle_overview_alignment,
                user_data=state
            )

            dpg.add_checkbox(
                label="Show counts",
                tag="overview_show_counts_checkbox",
                default_value=state.get("overview_show_counts", True),
                callback=toggle_overview_counts,
                user_data=state
            )


# -----------------------------------------------------------------------------
# 25. Toggle overview alignment
# -----------------------------------------------------------------------------
def toggle_overview_alignment(sender: Any, app_data: Any, user_data: Any) -> None:
    state = user_data
    state["overview_align"] = bool(app_data)
    last_id = state.get("overview_last_id")
    if last_id:
        show_results_window(last_id, state)


# -----------------------------------------------------------------------------
# 26. Toggle overview counts
# -----------------------------------------------------------------------------
def toggle_overview_counts(sender: Any, app_data: Any, user_data: Any) -> None:
    state = user_data
    state["overview_show_counts"] = bool(app_data)

    last_id = state.get("overview_last_id")
    if last_id:
        show_results_window(last_id, state)


# -----------------------------------------------------------------------------
# 27. Show results window
# -----------------------------------------------------------------------------
def show_results_window(id: Any, state: dict[str, Any]) -> None:

    img_width = state["overview_img_width"]
    img_height = int(img_width * 0.75)
    overview_render_scale = 1.6
    render_width = int(round(img_width * overview_render_scale))
    render_height = int(round(img_height * overview_render_scale))
    query = id.split("_")

    # remember last shown id
    state["overview_last_id"] = id
    aligned = state.get("overview_align", False)

    if len(query) == 2:  # Subset_n
        subset = id

        if aligned:
            # --- MolBlock → SMILES → SMARTS ---
            try:
                subset_dict = state["molblocks_rgd_dict"].get(subset, {})
                scaffold_mb = None
                for _, mol_data in subset_dict.items():
                    if "Core" in mol_data:
                        scaffold_mb = mol_data["Core"]
                        break
                scaffold = Chem.MolFromMolBlock(scaffold_mb, sanitize=False)
                if scaffold is None or scaffold.GetNumAtoms() == 0:
                    raise KeyError()
            except:
                try:
                    subset_dict = state["smiles_rgd_dict"].get(subset, {})
                    scaffold_smi = None
                    for _, mol_data in subset_dict.items():
                        if "Core" in mol_data:
                            scaffold_smi = mol_data["Core"]
                            break
                    scaffold = Chem.MolFromSmiles(scaffold_smi, sanitize=False)
                    if scaffold is None:
                        raise KeyError()
                except:
                    scaffold_sma = state["smiles_rgd_dict"][subset].get("Core", "")
                    scaffold = Chem.MolFromSmarts(scaffold_sma)
        else:
            # --- SMILES → SMARTS → MolBlock ---
            try:
                subset_dict = state["smiles_rgd_dict"].get(subset, {})
                scaffold_smi = None
                for _, mol_data in subset_dict.items():
                    if "Core" in mol_data:
                        scaffold_smi = mol_data["Core"]
                        break
                scaffold = Chem.MolFromSmiles(scaffold_smi, sanitize=False)
                if scaffold is None:
                    raise KeyError()
            except:
                try:
                    scaffold_sma = state["smiles_rgd_dict"][subset].get("Core", "")
                    scaffold = Chem.MolFromSmarts(scaffold_sma)
                    if scaffold is None:
                        raise KeyError()
                except:
                    subset_dict = state["molblocks_rgd_dict"].get(subset, {})
                    scaffold_mb = None
                    for _, mol_data in subset_dict.items():
                        if "Core" in mol_data:
                            scaffold_mb = mol_data["Core"]
                            break
                    scaffold = Chem.MolFromMolBlock(scaffold_mb, sanitize=False)

        try:
            for atom in scaffold.GetAtoms():
                if atom.GetAtomicNum() == 0 and atom.HasProp("molAtomMapNumber"):
                    idx = atom.GetProp("molAtomMapNumber")
                    atom.SetProp("atomLabel", f"R{idx}")
        except:
            pass

        Chem.SanitizeMol(scaffold, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.AssignStereochemistry(scaffold, force=True, cleanIt=True)

        drawer = rdMolDraw2D.MolDraw2DCairo(render_width, render_height)
        opts = drawer.drawOptions()
        opts.padding = 0.025
        opts.bondLineWidth = 1
        opts.minFontSize = 1
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, scaffold)
        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        scaffold_img = pilImage.open(io.BytesIO(png_data)).convert("RGBA")
        scaffold_data = (np.array(scaffold_img) / 255.0).flatten().astype(np.float32)

        width, height = scaffold_img.size
        scaff_texture_id_1 = "overview_scaffold_texture_1"
        if not dpg.does_item_exist(scaff_texture_id_1):
            dpg.add_dynamic_texture(width, height, scaffold_data, tag=scaff_texture_id_1, parent="texture_registry")
        else:
            dpg.set_value(scaff_texture_id_1, scaffold_data)

        if dpg.does_item_exist("mol_image"):
            dpg.delete_item("mol_image", children_only=True)

        
        build_align_checkbox(state)
        
        with dpg.group(parent="mol_image", horizontal=True, tag="mol_image_group"):
            scaffold_smi = Chem.MolToSmiles(scaffold, isomericSmiles=True)

            combo_data = (scaffold_smi, state, f"{subset}_svg_{state['img_popup_counter']}", subset)
            dpg.add_image_button(
                scaff_texture_id_1,
                tag="scaff_img",
                width=img_width,
                height=img_height,
                background_color=(0, 0, 0, 255),
                callback=image_click_callback,
                user_data=combo_data
            )
            dpg.bind_item_theme("scaff_img", apply_image_button_theme(state))
            with dpg.tooltip("scaff_img", delay=0):
                dpg.add_text(f"{subset.replace('_',' ').replace('su','Su')} - Common Core")
            register_responsive_image(
                state,
                image_tag="scaff_img",
                parent_tag="mol_image",
                aspect_ratio=0.75,
                tab="overview_tab",
            )
            export_png_popup("scaff_img", scaff_texture_id_1, state)


        if dpg.get_value("overview_subset_sort_choice") == "Similarity Clustering" and not dpg.does_item_exist("scaffold_hierarchical_dendrogram"):
            draw_scaffold_dendrogram(state)

    elif len(query) == 4:  # Subset_n_Mol_m
        subset = f"{query[0]}_{query[1]}"
        molecule = f"{query[2]}_{query[3]}"

        # --- Scaffold ---
        if aligned:
            # MolBlock → SMILES → SMARTS
            try:
                scaffold_mb = state["molblocks_rgd_dict"][subset][molecule]["Core"]
                scaffold = Chem.MolFromMolBlock(scaffold_mb, sanitize=False)
                if scaffold is None or scaffold.GetNumAtoms() == 0:
                    raise KeyError()
            except:
                try:
                    scaffold_smi = state["smiles_rgd_dict"][subset][molecule]["Core"]
                    scaffold = Chem.MolFromSmiles(scaffold_smi, sanitize=False)
                    if scaffold is None:
                        raise KeyError()
                except:
                    scaffold_sma = state["smiles_rgd_dict"][subset][molecule]["Core"]
                    scaffold = Chem.MolFromSmarts(scaffold_sma)
        else:
            # SMILES → SMARTS → MolBlock
            try:
                scaffold_smi = state["smiles_rgd_dict"][subset][molecule]["Core"]
                scaffold = Chem.MolFromSmiles(scaffold_smi, sanitize=False)
                if scaffold is None:
                    raise KeyError()
            except:
                try:
                    scaffold_sma = state["smiles_rgd_dict"][subset][molecule]["Core"]
                    scaffold = Chem.MolFromSmarts(scaffold_sma)
                    if scaffold is None:
                        raise KeyError()
                except:
                    scaffold_mb = state["molblocks_rgd_dict"][subset][molecule]["Core"]
                    scaffold = Chem.MolFromMolBlock(scaffold_mb, sanitize=False)

        try:
            smiles_dict = state["smiles_rgd_dict"][subset]
            r_counts = state["r_counts"][subset]

            for atom in scaffold.GetAtoms():
                if atom.GetAtomicNum() == 0 and atom.HasProp("molAtomMapNumber"):
                    idx = atom.GetProp("molAtomMapNumber")
                    r_name = f"R{idx}"

                    # R-group SMILES for this molecule.
                    r_smiles = smiles_dict[molecule].get(r_name, "")

                    # Frequency in the active subset.
                    count = r_counts[r_name].get(r_smiles, 0)

                    if state.get("overview_show_counts", False):
                        atom.SetProp("atomLabel", f"{r_name}({count})")
                    else:
                        atom.SetProp("atomLabel", r_name)
        except:
            pass

        Chem.SanitizeMol(scaffold, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.AssignStereochemistry(scaffold, force=True, cleanIt=True)

        drawer = rdMolDraw2D.MolDraw2DCairo(render_width, render_height)
        opts = drawer.drawOptions()
        opts.padding = 0.025
        opts.bondLineWidth = 1
        opts.minFontSize = 1
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, scaffold)
        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        scaffold_img = pilImage.open(io.BytesIO(png_data)).convert("RGBA")
        scaff_data = (np.array(scaffold_img) / 255.0).flatten().astype(np.float32)

        # --- Molecule ---
        if aligned:
            try:
                mol_mb = state["molblocks_rgd_dict"][subset][molecule]["Mol"]
                mol = Chem.MolFromMolBlock(mol_mb)
                if mol is None:
                    raise KeyError()
            except:
                try:
                    mol_smi = state["smiles_rgd_dict"][subset][molecule]["Mol"]
                    mol = Chem.MolFromSmiles(mol_smi, sanitize=False)
                    if mol is None:
                        raise KeyError()
                except:
                    mol_sma = state["smiles_rgd_dict"][subset][molecule]["Mol"]
                    mol = Chem.MolFromSmarts(mol_sma)
        else:
            try:
                mol_smi = state["smiles_rgd_dict"][subset][molecule]["Mol"]
                mol = Chem.MolFromSmiles(mol_smi, sanitize=False)
                if mol is None:
                    raise KeyError()
            except:
                try:
                    mol_sma = state["smiles_rgd_dict"][subset][molecule]["Mol"]
                    mol = Chem.MolFromSmarts(mol_sma)
                    if mol is None:
                        raise KeyError()
                except:
                    mol_mb = state["molblocks_rgd_dict"][subset][molecule]["Mol"]
                    mol = Chem.MolFromMolBlock(mol_mb)


        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

        drawer = rdMolDraw2D.MolDraw2DCairo(render_width, render_height)
        opts = drawer.drawOptions()
        opts.padding = 0.025
        opts.bondLineWidth = 1
        opts.minFontSize = 1
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        mol_img = pilImage.open(io.BytesIO(png_data)).convert("RGBA")
        mol_data = (np.array(mol_img) / 255.0).flatten().astype(np.float32)

        width, height = scaffold_img.size
        scaff_texture_id_2 = "overview_scaffold_texture_2"
        if not dpg.does_item_exist(scaff_texture_id_2):
            dpg.add_dynamic_texture(width, height, scaff_data, tag=scaff_texture_id_2, parent="texture_registry")
        else:
            dpg.set_value(scaff_texture_id_2, scaff_data)

        mol_texture_id_1 = "overview_molecule_texture_1"
        if not dpg.does_item_exist(mol_texture_id_1):
            dpg.add_dynamic_texture(width, height, mol_data, tag=mol_texture_id_1, parent="texture_registry")
        else:
            dpg.set_value(mol_texture_id_1, mol_data)

        if dpg.does_item_exist("mol_image"):
            dpg.delete_item("mol_image", children_only=True)


        build_align_checkbox(state)

        with dpg.group(parent="mol_image", horizontal=True, tag="mol_image_group"):
            scaffold_smi = Chem.MolToSmiles(scaffold, isomericSmiles=True)
            mol_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            _bind_compact_image_group_theme("mol_image_group")

            combo_data = (scaffold_smi, state, f"{subset}_svg_{state['img_popup_counter']}", subset)
            dpg.add_image_button(
                scaff_texture_id_2, tag="scaff_img",
                width=img_width, height=img_height,
                background_color=(0, 0, 0, 255),
                callback=image_click_callback, user_data=combo_data
            )
            dpg.bind_item_theme("scaff_img", apply_image_button_theme(state))
            with dpg.tooltip("scaff_img", delay=0):
                dpg.add_text(f"{subset.replace('_',' ').replace('su','Su')} - Common Core")
            register_responsive_image(
                state,
                image_tag="scaff_img",
                parent_tag="mol_image",
                aspect_ratio=0.75,
                tab="overview_tab",
            )
            export_png_popup("scaff_img", scaff_texture_id_2, state)

            combo_data = (mol_smi, state, f"{subset}_{molecule}_svg_{state['img_popup_counter']}", f"{subset}_{molecule}")
            dpg.add_image_button(
                mol_texture_id_1, tag="mol_img",
                width=img_width, height=img_height,
                background_color=(0, 0, 0, 255),
                callback=image_click_callback, user_data=combo_data
            )
            dpg.bind_item_theme("mol_img", apply_image_button_theme(state))
            with dpg.tooltip("mol_img", delay=0):
                dpg.add_text(f"{molecule.replace('_',' ').replace('mol','Molecule')}")
            register_responsive_image(
                state,
                image_tag="mol_img",
                parent_tag="mol_image",
                aspect_ratio=0.75,
                tab="overview_tab",
            )
            export_png_popup("mol_img", mol_texture_id_1, state)


    elif len(query) == 5:  # Subset_n_Mol_m_Rr
        subset = f"{query[0]}_{query[1]}"
        molecule = f"{query[2]}_{query[3]}"
        r = query[4]
        
        smiles_rgd_dict = state["smiles_rgd_dict"]


        # --- Scaffold ---
        if aligned:
            try:
                scaffold_mb = state["molblocks_rgd_dict"][subset][molecule]["Core"]
                scaffold = Chem.MolFromMolBlock(scaffold_mb, sanitize=False)
                if scaffold is None or scaffold.GetNumAtoms() == 0:
                    raise KeyError()
            except:
                try:
                    scaffold_smi = state["smiles_rgd_dict"][subset][molecule]["Core"]
                    scaffold = Chem.MolFromSmiles(scaffold_smi, sanitize=False)
                    if scaffold is None:
                        raise KeyError()
                except:
                    scaffold_sma = state["smiles_rgd_dict"][subset][molecule]["Core"]
                    scaffold = Chem.MolFromSmarts(scaffold_sma)
        else:
            try:
                scaffold_smi = state["smiles_rgd_dict"][subset][molecule]["Core"]
                scaffold = Chem.MolFromSmiles(scaffold_smi, sanitize=False)
                if scaffold is None:
                    raise KeyError()
            except:
                try:
                    scaffold_sma = state["smiles_rgd_dict"][subset][molecule]["Core"]
                    scaffold = Chem.MolFromSmarts(scaffold_sma)
                    if scaffold is None:
                        raise KeyError()
                except:
                    scaffold_mb = state["molblocks_rgd_dict"][subset][molecule]["Core"]
                    scaffold = Chem.MolFromMolBlock(scaffold_mb, sanitize=False)

        try:
            smiles_dict = smiles_rgd_dict[subset]
            r_counts = state["r_counts"][subset]

            for atom in scaffold.GetAtoms():
                if atom.GetAtomicNum() == 0 and atom.HasProp("molAtomMapNumber"):
                    idx = atom.GetProp("molAtomMapNumber")
                    r_name = f"R{idx}"

                    r_smiles = smiles_dict[molecule].get(r_name, "")

                    count = r_counts[r_name].get(r_smiles, 0)

                    if state.get("overview_show_counts", False):
                        atom.SetProp("atomLabel", f"{r_name}({count})")
                    else:
                        atom.SetProp("atomLabel", r_name)
        except:
            pass

        Chem.SanitizeMol(scaffold, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.AssignStereochemistry(scaffold, force=True, cleanIt=True)

        drawer = rdMolDraw2D.MolDraw2DCairo(render_width, render_height)
        opts = drawer.drawOptions()
        opts.padding = 0.025
        opts.bondLineWidth = 1
        opts.minFontSize = 1
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, scaffold)
        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        scaffold_img = pilImage.open(io.BytesIO(png_data)).convert("RGBA")
        scaff_data = (np.array(scaffold_img) / 255.0).flatten().astype(np.float32)

        # --- Molecule ---
        if aligned:
            try:
                mol_mb = state["molblocks_rgd_dict"][subset][molecule]["Mol"]
                mol = Chem.MolFromMolBlock(mol_mb)
                if mol is None:
                    raise KeyError()
            except:
                try:
                    mol_smi = state["smiles_rgd_dict"][subset][molecule]["Mol"]
                    mol = Chem.MolFromSmiles(mol_smi, sanitize=False)
                    if mol is None:
                        raise KeyError()
                except:
                    mol_sma = state["smiles_rgd_dict"][subset][molecule]["Mol"]
                    mol = Chem.MolFromSmarts(mol_sma)
        else:
            try:
                mol_smi = state["smiles_rgd_dict"][subset][molecule]["Mol"]
                mol = Chem.MolFromSmiles(mol_smi, sanitize=False)
                if mol is None:
                    raise KeyError()
            except:
                try:
                    mol_sma = state["smiles_rgd_dict"][subset][molecule]["Mol"]
                    mol = Chem.MolFromSmarts(mol_sma)
                    if mol is None:
                        raise KeyError()
                except:
                    mol_mb = state["molblocks_rgd_dict"][subset][molecule]["Mol"]
                    mol = Chem.MolFromMolBlock(mol_mb)


        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

        drawer = rdMolDraw2D.MolDraw2DCairo(render_width, render_height)
        opts = drawer.drawOptions()
        opts.padding = 0.025
        opts.bondLineWidth = 1
        opts.minFontSize = 1
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        mol_img = pilImage.open(io.BytesIO(png_data)).convert("RGBA")
        mol_data = (np.array(mol_img) / 255.0).flatten().astype(np.float32)

        # --- R-group ---
        if aligned:
            try:
                r_mb = state["molblocks_rgd_dict"][subset][molecule][r]
                r_group = Chem.MolFromMolBlock(r_mb, sanitize=False)
                if r_group is None:
                    raise KeyError()
            except:
                try:
                    r_smi = state["smiles_rgd_dict"][subset][molecule][r]
                    r_group = Chem.MolFromSmiles(r_smi, sanitize=False)
                    if r_group is None:
                        raise KeyError()
                except:
                    r_sma = state["smiles_rgd_dict"][subset][molecule][r]
                    r_group = Chem.MolFromSmarts(r_sma)
        else:
            try:
                r_smi = state["smiles_rgd_dict"][subset][molecule][r]
                r_group = Chem.MolFromSmiles(r_smi, sanitize=False)
                if r_group is None:
                    raise KeyError()
            except:
                try:
                    r_sma = state["smiles_rgd_dict"][subset][molecule][r]
                    r_group = Chem.MolFromSmarts(r_sma)
                    if r_group is None:
                        raise KeyError()
                except:
                    r_mb = state["molblocks_rgd_dict"][subset][molecule][r]
                    r_group = Chem.MolFromMolBlock(r_mb, sanitize=False)

        try:
            for atom in r_group.GetAtoms():
                if atom.GetAtomicNum() == 0 and atom.HasProp("molAtomMapNumber"):
                    idx = atom.GetProp("molAtomMapNumber")
                    atom.SetProp("atomLabel", f"*{idx}")
        except:
            pass


        r_counts = state["r_counts"]
        r_smiles = smiles_rgd_dict[subset][molecule].get(r, "")
        count = r_counts[subset][r][r_smiles]
    
        Chem.SanitizeMol(r_group, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.AssignStereochemistry(r_group, force=True, cleanIt=True)

        drawer = rdMolDraw2D.MolDraw2DCairo(render_width, render_height)
        opts = drawer.drawOptions()
        opts.padding = 0.025
        opts.bondLineWidth = 1
        opts.minFontSize = 1
        opts.explicitMethyl = True
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, r_group)
        
        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        r_img = pilImage.open(io.BytesIO(png_data)).convert("RGBA")
        r_data = (np.array(r_img) / 255.0).flatten().astype(np.float32)

        width, height = scaffold_img.size

        scaff_texture_id_3 = "overview_scaffold_texture_3"
        if not dpg.does_item_exist(scaff_texture_id_3):
            dpg.add_dynamic_texture(width, height, scaff_data, tag=scaff_texture_id_3, parent="texture_registry")
        else:
            dpg.set_value(scaff_texture_id_3, scaff_data)

        mol_texture_id_2 = "overview_molecule_texture_2"
        if not dpg.does_item_exist(mol_texture_id_2):
            mol_texture_id_2 = dpg.add_dynamic_texture(width, height, mol_data, tag=mol_texture_id_2, parent="texture_registry")
        else:
            dpg.set_value(mol_texture_id_2, mol_data)

        r_texture_id_1 = "overview_rgroup_texture_1"
        if not dpg.does_item_exist(r_texture_id_1):
            r_texture_id_1 = dpg.add_dynamic_texture(width, height, r_data, tag=r_texture_id_1, parent="texture_registry")
        else:
            dpg.set_value(r_texture_id_1, r_data)

        if dpg.does_item_exist("mol_image"):
            dpg.delete_item("mol_image", children_only=True)


        build_align_checkbox(state)

        with dpg.group(parent="mol_image", horizontal=True, tag="mol_image_group"):
            # Scaffold
            scaffold_smi = Chem.MolToSmiles(scaffold, isomericSmiles=True)
            mol_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            r_smi = Chem.MolToSmiles(r_group, isomericSmiles=True)
            _bind_compact_image_group_theme("mol_image_group")

            combo_data = (scaffold_smi, state, f"{subset}_svg_{state['img_popup_counter']}", subset)
            dpg.add_image_button(
                scaff_texture_id_3, tag="scaff_img",
                width=img_width, height=img_height,
                background_color=(0, 0, 0, 255),
                callback=image_click_callback, user_data=combo_data
            )
            dpg.bind_item_theme("scaff_img", apply_image_button_theme(state))
            with dpg.tooltip("scaff_img", delay=0):
                dpg.add_text(f"{subset.replace('_',' ').replace('su','Su')} - Common Core")
            register_responsive_image(
                state,
                image_tag="scaff_img",
                parent_tag="mol_image",
                aspect_ratio=0.75,
                tab="overview_tab",
            )
            export_png_popup("scaff_img", scaff_texture_id_3, state)

            # Molecule
            combo_data = (mol_smi, state, f"{subset}_{molecule}_svg_{state['img_popup_counter']}", f"{subset}_{molecule}")
            dpg.add_image_button(
                mol_texture_id_2, tag="mol_img",
                width=img_width, height=img_height,
                background_color=(0, 0, 0, 255),
                callback=image_click_callback, user_data=combo_data
            )
            dpg.bind_item_theme("mol_img", apply_image_button_theme(state))
            with dpg.tooltip("mol_img", delay=0):
                dpg.add_text(f"{molecule.replace('_',' ').replace('mol','Molecule')}")
            register_responsive_image(
                state,
                image_tag="mol_img",
                parent_tag="mol_image",
                aspect_ratio=0.75,
                tab="overview_tab",
            )
            export_png_popup("mol_img", mol_texture_id_2, state)

            # R-group
            combo_data = (r_smi, state, f"{subset}_{molecule}_{r}_svg_{state['img_popup_counter']}", f"{subset}_{molecule}_{r}")
            dpg.add_image_button(
                r_texture_id_1, tag="rgroup_img",
                width=img_width, height=img_height,
                background_color=(0, 0, 0, 255),
                callback=image_click_callback, user_data=combo_data
            )
            dpg.bind_item_theme("rgroup_img", apply_image_button_theme(state))
            with dpg.tooltip("rgroup_img", delay=0):
                dpg.add_text(f"{r} - occurrence: {count}/{len(smiles_rgd_dict[subset])} molecules")
            register_responsive_image(
                state,
                image_tag="rgroup_img",
                parent_tag="mol_image",
                aspect_ratio=0.75,
                tab="overview_tab",
            )
            export_png_popup("rgroup_img", r_texture_id_1, state)

    
    update_responsive_images(state)

    # 9.1.5: Remove loading cover if present
    if dpg.does_item_exist("cover_layer") and state["first_loading"] == False:
        dpg.delete_item("cover_layer")

   

# -----------------------------------------------------------------------------
# 28. Show log p atomic contribution map
# -----------------------------------------------------------------------------
def show_logP_atomic_contribution_map(smiles: str, state: dict[str, Any]) -> None:
    
    # 10.1.1: Replace existing molecule image in 'mol_image' window
    for tag in ["gasteiger_legend_text", "logp_legend_text"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=False)

    # 10.1.2: Generate atomic contributions for logP
    mol = Chem.MolFromSmiles(smiles)
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    weights = [x for x, _ in contribs]

    # 10.1.3: Draw similarity map to a Cairo image
    d2d = Draw.MolDraw2DCairo(state["overview_img_width"], state["overview_img_height"])
    _ = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, draw2d=d2d,
                                                   colorMap='seismic', contourLines=10)
    d2d.FinishDrawing()
    img_bytes = d2d.GetDrawingText()

    # 10.1.4: Convert image to texture data
    img = pilImage.open(io.BytesIO(img_bytes)).convert("RGBA")
    img_array = np.array(img) / 255.0
    img_data = img_array.flatten().astype(np.float32)
    width, height = img.size

    # 10.1.5: Add texture
    if not dpg.does_item_exist("mol_acm_texture"):
        dpg.add_dynamic_texture(width, height, img_data, tag="mol_acm_texture", parent="texture_registry")
    else:
        dpg.set_value("mol_acm_texture", img_data)

    # 10.1.6: Load into DearPyGui as a texture
    dpg.configure_item("mol_img", texture_tag="mol_acm_texture")
    
    # 10.1.8: Right-click export popup
    export_png_popup("mol_img", "mol_acm_texture", state)


   

# -----------------------------------------------------------------------------
# 29. Show gasteiger atomic contribution map
# -----------------------------------------------------------------------------
def show_gasteiger_atomic_contribution_map(smiles: str, state: dict[str, Any]) -> None:
    # 11.1.1: Replace existing molecule image in 'mol_image' window
    for tag in ["gasteiger_legend_text", "logp_legend_text"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=False)

    # 11.1.2: Compute Gasteiger charges
    mol = Chem.MolFromSmiles(smiles)
    AllChem.ComputeGasteigerCharges(mol)

    try:
        charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
    except:
        log_event("Overview", "Failed to extract Gasteiger charges.", indent=1, level="ERROR")
        return

    # 11.1.3: Draw similarity map based on charges
    d2d = Draw.MolDraw2DCairo(state["overview_img_width"], state["overview_img_height"])
    _ = SimilarityMaps.GetSimilarityMapFromWeights(mol, charges, draw2d=d2d,
                                                   colorMap='bwr', contourLines=10)
    d2d.FinishDrawing()
    img_bytes = d2d.GetDrawingText()

    # 11.1.4: Convert to RGBA texture
    img = pilImage.open(io.BytesIO(img_bytes)).convert("RGBA")
    img_array = np.array(img) / 255.0
    img_data = img_array.flatten().astype(np.float32)
    width, height = img.size

    # 11.1.5: Add texture

    if not dpg.does_item_exist("mol_acm_texture"):
        dpg.add_dynamic_texture(width, height, img_data, tag="mol_acm_texture", parent="texture_registry")
    else:
        dpg.set_value("mol_acm_texture", img_data)

    # 11.1.6: Load as texture
    dpg.configure_item("mol_img", texture_tag="mol_acm_texture")

    # 11.1.8: Export popup
    export_png_popup("mol_img", "mol_acm_texture", state)


# -----------------------------------------------------------------------------
# 30. Show hba hbd
# -----------------------------------------------------------------------------
def show_hba_hbd(smiles: str, state: dict[str, Any]) -> None:

    # 12.1.1: Clean previous image and legend
    for tag in ["gasteiger_legend_text", "logp_legend_text"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=False)

    # 12.1.2: Load molecule and feature definitions
    mol = Chem.MolFromSmiles(smiles)
    fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)

    # 12.1.3: Identify donor and acceptor atoms
    features = factory.GetFeaturesForMol(mol)
    donor_atoms = set(f.GetAtomIds()[0] for f in features if f.GetFamily() == 'Donor')
    acceptor_atoms = set(f.GetAtomIds()[0] for f in features if f.GetFamily() == 'Acceptor')

    # Add match for S as acceptor
    sulfur_acceptor = Chem.MolFromSmarts("[#16]")
    matches_sulfur = mol.GetSubstructMatches(sulfur_acceptor)
    sulfur_atoms = {idx[0] for idx in matches_sulfur}

    acceptor_atoms.update(sulfur_atoms)

    both_atoms = donor_atoms & acceptor_atoms
    only_donor = donor_atoms - both_atoms
    only_acceptor = acceptor_atoms - both_atoms

    atom_colors = {}

    for idx in only_donor:
        atom_colors[idx] = (0.3, 0.3, 1.0, 0.4)  # red
    for idx in only_acceptor:
        atom_colors[idx] = (1.0, 0.3, 0.3, 0.4)  # blue
    for idx in both_atoms:
        atom_colors[idx] = (1.0, 0.3, 1.0, 0.4)  # magenta
        
    # 12.1.5: Draw molecule with highlights
    d2d = Draw.MolDraw2DCairo(state["overview_img_width"], state["overview_img_height"])
    rdMolDraw2D.PrepareAndDrawMolecule(
        d2d,
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightAtomRadii={i: 0.5 for i in atom_colors}
    )
    d2d.FinishDrawing()
    img_bytes = d2d.GetDrawingText()

    # 12.1.6: Convert to DearPyGui texture
    img = pilImage.open(io.BytesIO(img_bytes)).convert("RGBA")
    img_array = np.array(img) / 255.0
    img_data = img_array.flatten().astype(np.float32)
    width, height = img.size

    if not dpg.does_item_exist("mol_acm_texture"):
        dpg.add_dynamic_texture(width, height, img_data, tag="mol_acm_texture", parent="texture_registry")
    else:
        dpg.set_value("mol_acm_texture", img_data)

    dpg.configure_item("mol_img", texture_tag="mol_acm_texture")

    # 12.1.7: Right-click export popup
    export_png_popup("mol_img", "mol_acm_texture", state)
from app.utils.app_logger import log_event, log_exception
