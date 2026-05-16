"""
==============
callbacks.py
==============

Global callback definitions for GUI interaction.

Contains event-driven functions linked to Dear PyGui elements, such as button
clicks, checkbox updates, combo selections, and tab switches. Ensures responsive
user interaction and consistent state synchronisation throughout the interface.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Update checkbox state
# 3. On button click
# 4. Open manual
# 5. Open contextual help
# 6. Change tab
# 7. Change chemspace subtab
# 8. Change overview subtab
# 9. Change similarity subtab
# 10. Change r analysis subtab
# 11. Close current job
# 12. Show library table confirm popup
# 13. Confirm build library table
# 14. Skip build library table
# 15. Export png popup
# 16. Export png callback
# 17. Export svg callback
# 18. Append to log
# 19. Rgba tuple to string
# 20. Toggle fullscreen with check
# 21. Register responsive image
# 22. Update responsive images

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import time
import dearpygui.dearpygui as dpg
import webbrowser
import numpy as np
from typing import Any, Callable
from PIL import Image as pilImage
from rdkit import Chem
from rdkit.Chem import Draw
from app.gui.event_log import show_event_log_window
from app.gui.themes_manager import (
    apply_bordered_input_text_theme,
)
from app.analysis.mmpa.mmpa_network import clear_mmpa_network_memory


# -----------------------------------------------------------------------------
# 2. Update checkbox state
# -----------------------------------------------------------------------------
def update_checkbox_state(sender: Any, app_data: Any, user_data: Any) -> Any:
    """
    Update the application state dictionary when a checkbox is toggled.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        user_data (Any): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """

    # -----------------------------------------------------------------------------
    # 2.1. Validate job name
    # -----------------------------------------------------------------------------
    def validate_job_name(name: str) -> Any:
        """
        Sanitize the job name by removing forbidden characters and
        normalising obviously problematic input.

        Args:
            name (str): Raw job name entered by the user.

        Returns:
            str: Cleaned job name (may be empty if only invalid content is provided).
        """
        forbidden = '<>:"/\\|?*'
        name = name.strip()
        if not name:
            return ""

        # Remove forbidden characters
        cleaned = "".join(c for c in name if c not in forbidden)

        # Remove control characters
        cleaned = "".join(c for c in cleaned if ord(c) >= 32)

        # Remove trailing dot (Windows cannot handle it reliably)
        cleaned = cleaned.rstrip(".")

        return cleaned

    # Resolve the logical checkbox key and the shared state dictionary.
    if isinstance(user_data, tuple):
        checkbox_key, state = user_data
    else:
        checkbox_key = sender if isinstance(sender, str) else "Unknown Checkbox"
        state = user_data

    # Apply validation ONLY for "Job name"
    if checkbox_key == "Job name":
        value = validate_job_name(app_data)
    else:
        value = app_data

    state["checkbox_states"][checkbox_key] = value

    # Keep a compact echo of UI interactions for debugging runs.
    print(f"{checkbox_key} is now {app_data}")


# -----------------------------------------------------------------------------
# 3. On button click
# -----------------------------------------------------------------------------
def on_button_click(button_tag: str, state: dict[str, Any]) -> None:
    """
    Handle button highlighting for the last clicked button in each overview subsection.
    
    Args:
        button_tag (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    # -----------------------------------------------------------------------------
    # 3.1. Change button color
    # -----------------------------------------------------------------------------
    def change_button_color(button_tag: str, last_clicked_button: Any) -> None:
        """
        Update the visual theme of a button to reflect its selection state.
        
        Args:
            button_tag (Any): Parameter accepted by this routine.
            last_clicked_button (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        if last_clicked_button and dpg.does_item_exist(last_clicked_button):
            dpg.bind_item_theme(last_clicked_button, "overview_choice_button_theme")

        if dpg.does_item_exist(button_tag):
            dpg.bind_item_theme(button_tag, "overview_choice_button_active_theme")

    # Infer the sub-area from the tag structure and store the last-clicked reference.
    if not isinstance(button_tag, str):
        return

    if button_tag.startswith("subset"):
        parts = button_tag.split("_")

        # subset_X
        if len(parts) == 2:
            last = state.get("last_clicked_button_sub")
            change_button_color(button_tag, last)
            state["last_clicked_button_sub"] = button_tag
            state["last_clicked_window"] = "subsets"

        # subset_X_mol_Y
        elif len(parts) == 4:
            last = state.get("last_clicked_button_mol")
            change_button_color(button_tag, last)
            state["last_clicked_button_mol"] = button_tag
            state["last_clicked_window"] = "molecules"

        # subset_X_mol_Y_r_Z
        elif len(parts) == 5:
            last = state.get("last_clicked_button_r")
            change_button_color(button_tag, last)
            state["last_clicked_button_r"] = button_tag
            state["last_clicked_window"] = "r_groups"


# -----------------------------------------------------------------------------
# 4. Open manual
# -----------------------------------------------------------------------------
def open_manual(state: dict[str, Any]) -> None:
    
    """
    Open manual.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    manual_dir = os.path.join(state["Home_dir_path"], "assets", "manual")

    page = "1_introduction.html"
    path = os.path.join(manual_dir, page)

    if os.path.exists(path):
        webbrowser.open(f"file://{path}")


# -----------------------------------------------------------------------------
# 5. Open contextual help
# -----------------------------------------------------------------------------
def open_contextual_help(state: dict[str, Any]) -> None:
    
    """
    Open contextual help.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    manual_dir = os.path.join(state["Home_dir_path"], "assets", "manual")

    page = state["manual_sections"].get(state.get("current_tab"), "1_introduction.html")
    path = os.path.join(manual_dir, page)

    if os.path.exists(path):
        webbrowser.open(f"file://{path}")


# -----------------------------------------------------------------------------
# 6. Change tab
# -----------------------------------------------------------------------------
def activate_main_tab(selected_tab: str, state: dict[str, Any]) -> None:
    """
    Show only the child window associated with the selected top-level area.

    Args:
        selected_tab (str): Logical tab identifier to activate.
        state (dict[str, Any]): Shared application state.

    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if not isinstance(selected_tab, str) or not selected_tab:
        selected_tab = state.get("current_tab", "input_tab")

    for tab_child in [
        "input_tab_child",
        "analysis_tab_child",
        "overview_tab_child",
        "r_analysis_tab_child",
        "similarity_tab_child",
        "stereo_tab_child",
        "mmpa_tab_child",
        "chemspace_tab_child",
        "prediction_tab_child",
        "utilities_tab_child",
        "slith_tab_child",
    ]:
        if dpg.does_item_exist(tab_child):
            dpg.configure_item(tab_child, show=tab_child.startswith(selected_tab))

    if selected_tab == "slith_tab":
        state["slith_ON"] = True
    else:
        state["slith_ON"] = False
        if not state.get("slith_is_paused", True):
            state["slith_is_paused"] = True
            if dpg.does_item_exist("pause_game_button"):
                dpg.set_item_label("pause_game_button", "Resume")

    state["current_tab"] = selected_tab
    request_responsive_image_update(state, frames=4)
    if callable(state.get("refresh_top_nav_selection")):
        try:
            state["refresh_top_nav_selection"]()
        except Exception:
            pass


def change_tab(state: dict[str, Any]) -> None:
    """
    Switch the visible GUI elements based on the selected main tab.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    selected_value = dpg.get_value("tab_bar")
    selected_tab = dpg.get_item_alias(selected_value) if selected_value is not None else None
    locked_text_tabs = state.get("locked_text_tabs", set())
    if isinstance(selected_tab, str) and selected_tab in locked_text_tabs:
        fallback_tab = state.get("current_tab", "input_tab")
        if dpg.does_item_exist("tab_bar") and dpg.does_item_exist(fallback_tab):
            dpg.set_value("tab_bar", fallback_tab)
        if callable(state.get("refresh_text_tab_themes")):
            try:
                state["refresh_text_tab_themes"]()
            except Exception:
                pass
        return
    activate_main_tab(selected_tab, state)


# -----------------------------------------------------------------------------
# 7. Change chemspace subtab
# -----------------------------------------------------------------------------
def change_chemspace_subtab(state: dict[str, Any]) -> None:
    """
    Update the current chemical space subtab selection in the shared state.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    selected_chemspace_subtab = dpg.get_item_alias(dpg.get_value("chemspace_tab_bar"))
    state["current_chemspace_subtab"] = selected_chemspace_subtab
    request_responsive_image_update(state, frames=4)


# -----------------------------------------------------------------------------
# 8. Change overview subtab
# -----------------------------------------------------------------------------
def change_overview_subtab(state: dict[str, Any]) -> None:
    """
    Update the current overview subtab selection in the shared state.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    selected_overview_subtab = dpg.get_item_alias(dpg.get_value("overview_tab_bar"))
    state["current_overview_subtab"] = selected_overview_subtab
    request_responsive_image_update(state, frames=4)


# -----------------------------------------------------------------------------
# 9. Change similarity subtab
# -----------------------------------------------------------------------------
def change_similarity_subtab(state: dict[str, Any]) -> None:
    """
    Update the current similarity subtab selection in the shared state.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    selected_similarity_subtab = dpg.get_item_alias(dpg.get_value("similarity_tab_bar"))
    state["current_similarity_subtab"] = selected_similarity_subtab
    request_responsive_image_update(state, frames=4)


# -----------------------------------------------------------------------------
# 10. Change r analysis subtab
# -----------------------------------------------------------------------------
def change_r_analysis_subtab(state: dict[str, Any]) -> None:
    """
    Update the current r_analysis subtab selection in the shared state.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    selected_r_analysis_subtab = dpg.get_item_alias(dpg.get_value("r_analysis_tab_bar"))
    state["current_r_analysis_subtab"] = selected_r_analysis_subtab
    request_responsive_image_update(state, frames=4)


# -----------------------------------------------------------------------------
# 11. Close current job
# -----------------------------------------------------------------------------
def close_current_job(state: dict[str, Any]) -> None:
    """
    Close all windows and handlers from the current job and reset.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    clear_mmpa_network_memory(state, clear_plot=True)

    # These are single-instance items shared across several windows.
    for item in [
        "r1_label",
        "r2_label",
        "scatter_layer",
        "gradient_bar",
        "tooltip_pca",
        "tooltip_text_pca",
        "tooltip_umap",
        "tooltip_text_umap",
        "tooltip_tsne",
        "tooltip_text_tsne",
        "tooltip_scatter",
        "tooltip_text_scatter",
        "details_panel",
        # Image Widgets
        "landscape_mol1_image_widget",
        "landscape_mol2_image_widget",
        "mol_y_image_widget",
        "mol_x_image_widget",
        "r1_image_widget",
        "r2_image_widget",
        "descriptors_mol_image_widget",
        "pca_mol_image_widget",
        "umap_molecule_image_widget",
        "tsne_molecule_image_widget",
        # Handlers
        "tanimoto_matrix_click_handler",
        "clustered_matrix_click_handler",
        "landscape_click_handler",
        "counts_boxplot_click_handler",
        "counts_boxplot_move_handler",
        "counts_boxplot_wheel_handler",
        "counts_boxplot_left_click_handler",
        "counts_boxplot_left_drag_handler",
        "counts_boxplot_right_click_handler",
        "counts_boxplot_right_drag_handler",
        "heatmap_click_handler",
        "heatmap_mouse_move_handler",
        "descriptors_click_handler",
        "descriptors_mouse_move_handler",
        "pca_click_handler",
        "pca_mouse_move_handler",
        "umap_plot_handler_registry",
        "tsne_plot_handler_registry",
        "umap_error_window",
        "tsne_error_window",
    ]:
        if dpg.does_item_exist(item):
            dpg.delete_item(item)

    old_series = state.pop("landscape_color_series_tags", [])
    for tag in old_series:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    old_themes = state.pop("landscape_bucket_themes", [])
    for th in old_themes:
        try:
            if dpg.does_item_exist(th):
                dpg.delete_item(th)
        except Exception:
            # Themes may already be gone depending on shutdown order.
            pass

    # Keep the main window containers but remove their children so a new job
    # can repopulate them cleanly.
    for item in [
        # Analysis tab windows
        "library_preparation_window",
        "scaffold_analysis_window",
        "rga_window",
        # Overview decomposition windows
        "subset_choice",
        "molecule_choice",
        "r_group_choice",
        "mol_image",
        # Overview enrichment windows
        "properties_window",
        "activities_window",
        "image_checkboxes_window",
        "enrichment_plot_window",
        # Overview table windows
        "overview_table_manager",
        "overview_table_window",
        # Similarity tab windows
        "similarity_manager_window",
        "similarity_tanimoto_window",
        "similarity_tanimoto_mol_couple_window",
        "clustered_similarity_manager_window",
        "clustered_similarity_matrix_window",
        "clustered_similarity_mol_couple_window",
        # R-Counts tab windows
        "counts_selection_window",
        "counts_scaffold_img_window",
        "counts_boxplot_window",
        "counts_rgroup_details_window",
        "counts_boxplot_details_window",
        "counts_table_window",
        "counts_boxplot_manager_window",
        # Stereo tab windows
        "isomers_manager_window",
        "isomers_images_main_window",
        # MMPA tab windows
        "mmpa_window",
        "mmpa_global_plot_window",
        "mmpa_subset_plot_window",
        "mmpa_table_window",
        "mmpa_images_window",
        "mmpa_group_sidebar_window",
        "mmpa_network_plot_window",
        # Plot managers windows
        "heatmap_manager_window",
        "descriptors_manager_window",
        "dendrogram_manager_window",
        "pca_manager_window",
        "umap_manager_window",
        "tsne_manager_window",
        "landscape_manager_window",
        # Prediction windows
        "prediction_manager_host",
        "prediction_window",
        "prediction_manager_window",
        "prediction_manager_controls",
        "prediction_output_host",
        "prediction_results_window",
        "prediction_plot_window",
        "prediction_details_window",
        # Heatmap windows
        "heatmap_window",
        "heatmap_details_window",
        # Descriptors windows
        "descriptors_window",
        "descriptors_details_window",
        "dendrogram_window",
        "dendrogram_details_window",
        "chemspace_dendro_move_handler",
        "chemspace_dendro_wheel_handler",
        "chemspace_dendro_left_drag_handler",
        "chemspace_dendro_right_drag_handler",
        "chemspace_dendro_left_click_handler",
        # pca windows
        "pca_window",
        "pca_details_window",
        # umap windows
        "umap_window",
        "umap_details_window",
        "umap_gradient_bar_window",
        # tsne windows
        "tsne_window",
        "tsne_details_window",
        "tsne_gradient_bar_window",
        # Landscape windows
        "landscape_window",
        "landscape_details_window",
    ]:
        if dpg.does_item_exist(item):
            dpg.delete_item(item, children_only=True)

    # Reinitialise sequential step indices and message buffers.
    state["prep_log"] = []
    state["scaff_log"] = []
    state["rga_log"] = []
    state["STEP_prep"] = 1
    state["STEP_scaff"] = 1
    state["STEP_rga"] = 1
    state["prediction_results"] = []
    state["prediction_results_map"] = {}
    state["prediction_plot_points"] = []
    state["prediction_selected_record_key"] = None
    state["prediction_status_message"] = ""
    state["prediction_metrics"] = {}
    state.pop("umap_embedding", None)
    state.pop("umap_projection_model", None)
    state.pop("umap_projection_fp_algorithm", None)
    state.pop("umap_refresh_colors", None)
    state.pop("umap_relayout_colormap_scale", None)
    state.pop("umap_current_coloring", None)
    state.pop("tsne_embedding", None)
    state.pop("tsne_projection_model", None)
    state.pop("tsne_projection_fp_algorithm", None)
    state.pop("tsne_refresh_colors", None)
    state.pop("tsne_relayout_colormap_scale", None)
    state.pop("tsne_current_coloring", None)
    state.pop("chemspace_dendrogram_refresh_colors", None)
    state.pop("chemspace_dendrogram_refresh_highlight", None)
    state.pop("mmpa_network_refresh_colors", None)
    state.pop("_chemspace_dendro_cache", None)
    state.pop("chemspace_cluster_threshold", None)
    state.pop("chemspace_dendrogram_selected_subset", None)
    state.pop("chemspace_dendrogram_highlight_subset", None)


# -----------------------------------------------------------------------------
# 12. Show library table confirm popup
# -----------------------------------------------------------------------------
def show_library_table_confirm_popup(state: dict[str, Any]) -> None:
    """
    Ask the user whether to build the library summary table, which can be slow.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if dpg.does_item_exist("library_table_confirm_popup"):
        dpg.delete_item("library_table_confirm_popup")

    with dpg.window(
        tag="library_table_confirm_popup",
        label="Build Library Table?",
        modal=True,
        no_resize=True,
        no_collapse=True,
        autosize=True,
        on_close=lambda: skip_build_library_table(state),
    ):
        dpg.add_text(
            "Do you want to generate the library summary table?\n\n"
            "This may take some time on large datasets"
        )
        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Yes",
                callback=lambda: confirm_build_library_table(state),
            )
            dpg.add_button(
                label="No",
                callback=lambda: skip_build_library_table(state),
            )


# -----------------------------------------------------------------------------
# 13. Confirm build library table
# -----------------------------------------------------------------------------
def confirm_build_library_table(state: dict[str, Any]) -> None:
    """
    User accepted: build the library table and show the window.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if dpg.does_item_exist("library_table_confirm_popup"):
        dpg.delete_item("library_table_confirm_popup")

    try:
        if dpg.does_item_exist("library_table_window_inner"):
            dpg.delete_item("library_table_window_inner")
        dpg.configure_item("library_overview_table", show=True)
        from app.lmm.lmm_library_table import show_library_summary_table

        show_library_summary_table(state, state["selected_file_path"])
    except Exception as e:
        log_exception("INPUT", "Error showing library summary table", e, indent=1)
        if dpg.does_item_exist("library_overview_table"):
            dpg.configure_item("library_overview_table", show=False)


# -----------------------------------------------------------------------------
# 14. Skip build library table
# -----------------------------------------------------------------------------
def skip_build_library_table(state: dict[str, Any]) -> None:
    """
    User declined: do not build the table, just close the confirmation popup.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if dpg.does_item_exist("library_table_confirm_popup"):
        dpg.delete_item("library_table_confirm_popup")

    # Ensure the library overview window remains hidden when the user skips.
    if dpg.does_item_exist("library_overview_table"):
        dpg.configure_item("library_overview_table", show=False)


# -----------------------------------------------------------------------------
# 15. Export png popup
# -----------------------------------------------------------------------------
def export_png_popup(tag: str, texture_id: str, state: dict[str, Any]) -> None:
    """
    Create a right-click popup menu attached to an image widget that allows exporting the.
    
    Args:
        tag (str): Parameter accepted by this routine.
        texture_id (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    # Provide a lightweight UI with filename input and an export action.
    with dpg.popup(
        tag,
        mousebutton=dpg.mvMouseButton_Right,
        no_move=True,
        min_size=(10, 10),
    ) as popup_tag:
        state["img_popup_counter"] += 1
        popup_id = state["img_popup_counter"]
        input_tag = f"export_filename_input_{popup_id}"
        dpg.add_text("Enter filename:")
        dpg.add_input_text(
            label="",
            tag=input_tag,
            default_value="export.png",
            width=85,
        )
        dpg.bind_item_theme(input_tag, apply_bordered_input_text_theme(state))
        dpg.add_button(
            label="Export image",
            callback=export_png_callback,
            width=85,
            user_data=(texture_id, input_tag, popup_tag, state),
        )


# -----------------------------------------------------------------------------
# 16. Export png callback
# -----------------------------------------------------------------------------
def export_png_callback(sender: Any, app_data: Any, user_data: Any) -> None:
    """
    Export a DearPyGui texture to a PNG image file at 300 DPI.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        user_data (Any): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    texture_tag, input_tag, popup_tag, state = user_data

    # Read filename from the dedicated input field and ensure .png extension.
    filename = dpg.get_value(input_tag) or "export.png"
    filename = filename.strip()
    if not filename.lower().endswith(".png"):
        filename += ".png"

    # Convert the normalised float RGBA to uint8 and build a PIL image.
    tex_info = dpg.get_item_configuration(texture_tag)
    tex_data = dpg.get_value(texture_tag)
    width = tex_info["width"]
    height = tex_info["height"]

    array = (np.array(tex_data).reshape((height, width, 4)) * 255).astype(np.uint8)
    img = pilImage.fromarray(array, mode="RGBA")

    # Ensure the output directory exists, then write the PNG.
    output_path = os.path.join(state["image_dir"], filename)
    os.makedirs(state["image_dir"], exist_ok=True)
    img.save(output_path, dpi=(300, 300))
    print(f"✅ Saved as {filename} at 300 DPI")

    # Remove the popup menu once the export has completed.
    if dpg.does_item_exist(popup_tag):
        dpg.delete_item(popup_tag)


# -----------------------------------------------------------------------------
# 17. Export svg callback
# -----------------------------------------------------------------------------
def export_svg_callback(sender: Any, app_data: Any, user_data: Any) -> None:
    """
    Export a molecule (given as SMILES) to a true vector SVG file.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        user_data (Any): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    # Validate SMILES and generate a 2D depiction for vector drawing.
    smiles_str, state, label_tag = user_data
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        log_event("INPUT", f"Failed to parse SMILES: {smiles_str}", indent=1, level="ERROR")
        return

    Chem.rdDepictor.Compute2DCoords(mol)

    inch_size = 2.0  # 2 inches per side
    dpi = 300
    px_size = int(inch_size * dpi)

    drawer = Draw.MolDraw2DSVG(px_size, px_size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_data = drawer.GetDrawingText()

    # Normalise the label to a filesystem-friendly SVG name.
    filename = label_tag.split("_svg_")[0]
    output_path = os.path.join(state["image_dir"], f"{filename}.svg")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_data)

    print(f"✅ SVG saved as: {output_path}")


# -----------------------------------------------------------------------------
# 18. Plot context popup helpers
# -----------------------------------------------------------------------------
def _dismiss_window_popup_on_outside_click(popup_tag: str, state: dict[str, Any], context_key: str) -> None:
    """
    Close a custom DearPyGui window-popup when the user clicks outside it.
    """
    if not dpg.does_item_exist(popup_tag):
        return
    opened_at = float(state.get(f"{context_key}_popup_opened_at", 0.0) or 0.0)
    if opened_at and (time.monotonic() - opened_at) <= 0.15:
        return
    try:
        px, py = dpg.get_mouse_pos(local=False)
        pmin = dpg.get_item_rect_min(popup_tag)
        pmax = dpg.get_item_rect_max(popup_tag)
    except Exception:
        return
    inside_popup = pmin[0] <= px <= pmax[0] and pmin[1] <= py <= pmax[1]
    if not inside_popup:
        dpg.delete_item(popup_tag)


def add_chemspace_plot_specific_popup_controls(
    *,
    state: dict[str, Any],
    point_size_state_key: str,
    point_size_combo_tag: str,
    point_size_callback: Callable[[Any], None],
    mcs_timeout_state_key: str,
    mcs_timeout_combo_tag: str,
    mcs_timeout_callback: Callable[[Any], None],
    mcs_features_state_key: str,
    mcs_features_checkbox_tag: str,
    mcs_features_callback: Callable[[Any], None],
    input_tag: str,
    draw_callback: Callable[[], None],
    delete_callback: Callable[[], None],
) -> None:
    """
    Add the ChemSpace-specific controls currently used by PCA/UMAP/t-SNE plots.
    """
    dpg.add_combo(
        tag=point_size_combo_tag,
        label="Point size",
        items=["Small", "Medium", "Large"],
        default_value=str(state.get(point_size_state_key, "Medium")),
        width=160,
        callback=lambda s, a: point_size_callback(a),
    )
    dpg.add_combo(
        tag=mcs_timeout_combo_tag,
        label="MCS Timeout",
        items=["10s", "30s", "60s", "120s", "300s", "Unlimited"],
        default_value=str(state.get(mcs_timeout_state_key, "10s")),
        width=160,
        callback=lambda s, a: mcs_timeout_callback(a),
    )
    dpg.add_checkbox(
        tag=mcs_features_checkbox_tag,
        label="Show MCS annotations",
        default_value=bool(state.get(mcs_features_state_key, True)),
        callback=lambda s, a: mcs_features_callback(a),
    )
    dpg.add_separator()
    dpg.add_input_text(tag=input_tag, width=340, hint="Paste a SMILES string")
    with dpg.group(horizontal=True):
        dpg.add_button(label="Draw", callback=lambda: draw_callback())
        dpg.add_button(label="Delete all", callback=lambda: delete_callback())


def _build_plot_context_theme(
    state: dict[str, Any],
    *,
    theme_kind: str,
    show_grid: bool,
    white_background: bool,
) -> Any:
    """
    Build a plot theme variant used by the shared right-click context popup.
    """
    theme_dict = state["themes"][state["theme_name"]]
    if theme_kind == "dendrogram":
        base_bg = theme_dict["Main Background"]
    elif theme_kind == "boxplot":
        base_bg = (255, 255, 255, 255)
    else:
        base_bg = theme_dict["Plot Background"]

    bg_rgba = (255, 255, 255, 255) if white_background else base_bg
    text_rgba = theme_dict["Text Color"]

    if white_background:
        grid_rgba = (140, 140, 140, 120) if show_grid else (0, 0, 0, 0)
    elif theme_kind == "boxplot":
        grid_rgba = (60, 60, 60, 128) if show_grid else (0, 0, 0, 0)
    else:
        grid_rgba = (
            int(text_rgba[0]),
            int(text_rgba[1]),
            int(text_rgba[2]),
            80,
        ) if show_grid else (0, 0, 0, 0)

    with dpg.theme() as plot_theme:
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, bg_rgba, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, grid_rgba, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 0, 0)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotBorderSize, 0)
            dpg.add_theme_color(dpg.mvPlotCol_LegendText, text_rgba, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, theme_dict["Secondary Background"], category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder, theme_dict["Border Color"], category=dpg.mvThemeCat_Plots)
            if theme_kind == "boxplot":
                dpg.add_theme_color(dpg.mvPlotCol_InlayText, (0, 0, 0, 255), category=dpg.mvThemeCat_Plots)
    return plot_theme


def register_plot_context_popup(
    state: dict[str, Any],
    *,
    context_key: str,
    plot_tag: str,
    x_axis_tag: str,
    y_axis_tag: str,
    theme_kind: str = "plot",
    specific_builder: Callable[[], None] | None = None,
) -> None:
    """
    Register a shared right-click popup for a plot, opening only on a quick
    single right click and not after a right-drag zoom rectangle.
    """
    if not dpg.does_item_exist(plot_tag):
        return

    popup_tag = f"{context_key}_plot_context_popup"
    down_state_key = f"{context_key}_right_click_pending"
    drag_time_key = f"{context_key}_right_click_down_time"
    drag_pos_key = f"{context_key}_right_click_down_pos"
    drag_limits_key = f"{context_key}_right_click_down_limits"
    settings_store = state.setdefault("plot_context_settings", {})

    def _infer_defaults() -> dict[str, bool]:
        plot_cfg = {}
        try:
            plot_cfg = dpg.get_item_configuration(plot_tag)
        except Exception:
            plot_cfg = {}
        show_coordinates = not bool(plot_cfg.get("no_mouse_pos", True))
        show_crosshairs = bool(plot_cfg.get("crosshairs", False))
        show_grid = False
        grid_default_axes = (y_axis_tag,) if theme_kind == "boxplot" else (x_axis_tag, y_axis_tag)
        for axis_tag in grid_default_axes:
            if not dpg.does_item_exist(axis_tag):
                continue
            try:
                axis_cfg = dpg.get_item_configuration(axis_tag)
                if not bool(axis_cfg.get("no_gridlines", False)):
                    show_grid = True
                    break
            except Exception:
                pass
        return {
            "show_grid": show_grid,
            "show_crosshairs": show_crosshairs,
            "show_coordinates": show_coordinates,
            "white_background": False,
        }

    existing_context_settings = settings_store.get(context_key)
    context_settings = existing_context_settings
    if not isinstance(context_settings, dict):
        context_settings = _infer_defaults()
        settings_store[context_key] = context_settings

    def _apply_context_visuals() -> None:
        if not dpg.does_item_exist(plot_tag):
            return
        show_grid = bool(context_settings.get("show_grid", False))
        show_crosshairs = bool(context_settings.get("show_crosshairs", False))
        show_coordinates = bool(context_settings.get("show_coordinates", False))
        white_background = bool(context_settings.get("white_background", False))
        try:
            dpg.configure_item(plot_tag, crosshairs=show_crosshairs, no_mouse_pos=not show_coordinates)
        except Exception:
            pass
        if theme_kind == "boxplot":
            if dpg.does_item_exist(x_axis_tag):
                try:
                    dpg.configure_item(x_axis_tag, no_gridlines=True)
                except Exception:
                    pass
            if dpg.does_item_exist(y_axis_tag):
                try:
                    dpg.configure_item(y_axis_tag, no_gridlines=not show_grid)
                except Exception:
                    pass
        else:
            for axis_tag in (x_axis_tag, y_axis_tag):
                if dpg.does_item_exist(axis_tag):
                    try:
                        dpg.configure_item(axis_tag, no_gridlines=not show_grid)
                    except Exception:
                        pass
        theme_state_key = f"{context_key}_plot_context_theme_tag"
        old_theme = state.get(theme_state_key)
        if old_theme and dpg.does_item_exist(old_theme):
            dpg.delete_item(old_theme)
        new_theme = _build_plot_context_theme(
            state,
            theme_kind=theme_kind,
            show_grid=show_grid,
            white_background=white_background,
        )
        state[theme_state_key] = new_theme
        dpg.bind_item_theme(plot_tag, new_theme)

    def _set_bool_option(option_key: str, value: Any) -> None:
        context_settings[option_key] = bool(value)
        _apply_context_visuals()

    def _open_context_popup() -> None:
        if not dpg.is_item_hovered(plot_tag):
            return
        if dpg.does_item_exist(popup_tag):
            dpg.delete_item(popup_tag)
        mx, my = dpg.get_mouse_pos(local=False)
        state[f"{context_key}_popup_opened_at"] = time.monotonic()
        with dpg.window(
            tag=popup_tag,
            no_title_bar=True,
            show=True,
            no_resize=True,
            no_scrollbar=True,
            no_collapse=True,
            autosize=True,
            pos=(int(mx) + 14, int(my) + 14),
        ):
            dpg.add_checkbox(
                label="Show grid",
                default_value=bool(context_settings.get("show_grid", False)),
                callback=lambda s, a: _set_bool_option("show_grid", a),
            )
            dpg.add_checkbox(
                label="Show crosshairs",
                default_value=bool(context_settings.get("show_crosshairs", False)),
                callback=lambda s, a: _set_bool_option("show_crosshairs", a),
            )
            dpg.add_checkbox(
                label="Show coordinates",
                default_value=bool(context_settings.get("show_coordinates", False)),
                callback=lambda s, a: _set_bool_option("show_coordinates", a),
            )
            dpg.add_checkbox(
                label="White background",
                default_value=bool(context_settings.get("white_background", False)),
                callback=lambda s, a: _set_bool_option("white_background", a),
            )
            if callable(specific_builder):
                dpg.add_separator()
                specific_builder()

    def _on_right_mouse_down() -> None:
        state[down_state_key] = bool(dpg.is_item_hovered(plot_tag))
        state[drag_time_key] = time.monotonic()
        state[drag_pos_key] = tuple(dpg.get_mouse_pos(local=False))
        try:
            state[drag_limits_key] = (
                tuple(dpg.get_axis_limits(x_axis_tag)),
                tuple(dpg.get_axis_limits(y_axis_tag)),
            )
        except Exception:
            state[drag_limits_key] = None

    def _on_right_mouse_release() -> None:
        down_time = float(state.get(drag_time_key, 0.0) or 0.0)
        down_pos = state.get(drag_pos_key)
        up_pos = tuple(dpg.get_mouse_pos(local=False))
        duration = time.monotonic() - down_time if down_time else 999.0
        down_limits = state.get(drag_limits_key)
        moved = True
        if isinstance(down_pos, (tuple, list)) and len(down_pos) >= 2:
            dx = float(up_pos[0]) - float(down_pos[0])
            dy = float(up_pos[1]) - float(down_pos[1])
            moved = (dx * dx + dy * dy) > (6.0 * 6.0)
        zoomed = False
        if isinstance(down_limits, (tuple, list)) and len(down_limits) == 2:
            try:
                cur_x = tuple(dpg.get_axis_limits(x_axis_tag))
                cur_y = tuple(dpg.get_axis_limits(y_axis_tag))
                dx0 = abs(float(cur_x[0]) - float(down_limits[0][0]))
                dx1 = abs(float(cur_x[1]) - float(down_limits[0][1]))
                dy0 = abs(float(cur_y[0]) - float(down_limits[1][0]))
                dy1 = abs(float(cur_y[1]) - float(down_limits[1][1]))
                zoomed = max(dx0, dx1, dy0, dy1) > 1e-9
            except Exception:
                zoomed = False
        if state.get(down_state_key) and not moved and not zoomed and duration <= 0.25:
            _open_context_popup()
        state[down_state_key] = False
        state[drag_time_key] = 0.0
        state[drag_pos_key] = None
        state[drag_limits_key] = None

    down_handler_tag = f"{context_key}_plot_context_right_down_handler"
    release_handler_tag = f"{context_key}_plot_context_right_release_handler"
    dismiss_handler_tag = f"{context_key}_plot_context_dismiss_handler"
    for tag in (down_handler_tag, release_handler_tag, dismiss_handler_tag):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    dpg.add_mouse_down_handler(
        button=dpg.mvMouseButton_Right,
        parent="handler_registry",
        tag=down_handler_tag,
        callback=lambda s, a: _on_right_mouse_down(),
    )
    dpg.add_mouse_release_handler(
        button=dpg.mvMouseButton_Right,
        parent="handler_registry",
        tag=release_handler_tag,
        callback=lambda s, a: _on_right_mouse_release(),
    )
    dpg.add_mouse_click_handler(
        parent="handler_registry",
        tag=dismiss_handler_tag,
        callback=lambda s, a: _dismiss_window_popup_on_outside_click(popup_tag, state, context_key),
    )

    if isinstance(existing_context_settings, dict) or state.get(f"{context_key}_plot_context_theme_tag"):
        _apply_context_visuals()


# -----------------------------------------------------------------------------
# 19. Append to log
# -----------------------------------------------------------------------------
def append_to_log(state: dict[str, Any], message: str) -> None:
    """
    Append a message to the log.txt file located in the report directory,.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
        message (Any): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    def _emit_lmm_line(line: str) -> None:
        if line.strip():
            print(f"[LMM] {line}")
        else:
            print("")

    for line in str(message).splitlines():
        _emit_lmm_line(line)

    # Use [HH:MM:SS] prefix unless the message is a visual separator line.
    current_time = time.strftime("%H:%M:%S", time.localtime())
    report_dir = str(state.get("report_dir", "") or "").strip()
    if not report_dir:
        work_dir = str(state.get("work_dir", "") or "").strip()
        if work_dir:
            report_dir = os.path.join(work_dir, "reports")
            state["report_dir"] = report_dir
    if not report_dir:
        return
    os.makedirs(report_dir, exist_ok=True)
    log_path = os.path.join(report_dir, "log.txt")

    # Force UTF-8 encoding to support special characters on all platforms.
    with open(log_path, "a", encoding="utf-8", errors="strict") as f:
        for line in str(message).splitlines():
            if "========================================" in line:
                f.write(f"[{current_time}] [LMM] {line}\n")
            elif line.strip():
                f.write(f"[{current_time}] [LMM] {line}\n")
            else:
                f.write("\n")


# -----------------------------------------------------------------------------
# 19. Rgba tuple to string
# -----------------------------------------------------------------------------
def rgba_tuple_to_string(rgba_tuple: Any) -> Any:
    """
    Convert an (R, G, B, A) tuple with 0–255 channels into a CSS rgba() string
    with alpha in [0, 1].

    Args:
        rgba_tuple (tuple[int, int, int, int]): Channels as 0–255 integers (R, G, B, A).

    Returns:
        str: CSS-formatted rgba string, e.g., 'rgba(12, 34, 56, 0.75)'.
    """
    # Keep colour channels as integers; convert alpha to a two-decimal fraction.
    r, g, b, a = rgba_tuple
    return f"rgba({r}, {g}, {b}, {a / 255:.2f})"


# -----------------------------------------------------------------------------
# 20. Parse color string to rgb tuple
# -----------------------------------------------------------------------------
_MATPLOTLIB_TAB10_RGB = {
    "C0": (31, 119, 180),
    "C1": (255, 127, 14),
    "C2": (44, 160, 44),
    "C3": (214, 39, 40),
    "C4": (148, 103, 189),
    "C5": (140, 86, 75),
    "C6": (227, 119, 194),
    "C7": (127, 127, 127),
    "C8": (188, 189, 34),
    "C9": (23, 190, 207),
}

_NAMED_COLOR_RGB = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "brown": (165, 42, 42),
    "pink": (255, 192, 203),
}


def color_string_to_rgb255(color_value: Any, fallback: tuple[int, int, int] = (80, 80, 80)) -> tuple[int, int, int]:
    """
    Convert a small set of plot-cycle/CSS-like color values into 0-255 RGB.

    Supported formats:
    - plot-cycle ids like ``C0`` ... ``C9``
    - named colors like ``blue`` or ``red``
    - hex strings ``#rgb`` / ``#rrggbb``
    - tuples/lists with either 0..1 floats or 0..255 channels
    """
    if color_value is None:
        return fallback

    if isinstance(color_value, (tuple, list)) and len(color_value) >= 3:
        channels = list(color_value[:3])
        if all(isinstance(channel, (int, float)) for channel in channels):
            if max(float(channel) for channel in channels) <= 1.0:
                return tuple(int(round(float(channel) * 255)) for channel in channels)
            return tuple(max(0, min(255, int(round(float(channel))))) for channel in channels)

    if not isinstance(color_value, str):
        return fallback

    color_str = color_value.strip()
    if not color_str:
        return fallback

    if color_str in _MATPLOTLIB_TAB10_RGB:
        return _MATPLOTLIB_TAB10_RGB[color_str]

    lowered = color_str.lower()
    if lowered in _NAMED_COLOR_RGB:
        return _NAMED_COLOR_RGB[lowered]

    if color_str.startswith("#"):
        hex_value = color_str[1:]
        if len(hex_value) == 3:
            try:
                return tuple(int(ch * 2, 16) for ch in hex_value)
            except ValueError:
                return fallback
        if len(hex_value) == 6:
            try:
                return tuple(int(hex_value[idx:idx + 2], 16) for idx in (0, 2, 4))
            except ValueError:
                return fallback

    return fallback


# -----------------------------------------------------------------------------
# 21. Toggle fullscreen with check
# -----------------------------------------------------------------------------
def toggle_fullscreen_with_check(state: dict[str, Any]) -> None:
    """
    Toggle the viewport fullscreen mode and keep the corresponding menu item.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    current = state.get("is_fullscreen", False)
    state["is_fullscreen"] = not current

    dpg.toggle_viewport_fullscreen()
    request_responsive_image_update(state, frames=10)

    if dpg.does_item_exist("fullscreen_menu_item"):
        dpg.configure_item("fullscreen_menu_item", check=state["is_fullscreen"])


# -----------------------------------------------------------------------------
# 22. Register responsive image
# -----------------------------------------------------------------------------
def register_responsive_image(
    state: dict[str, Any],
    image_tag: str,
    parent_tag: str,
    aspect_ratio: float = 0.75,
    parent_of_parent: Any = None,
    default_pop_h: Any = None,
    tab: Any = None
) -> None:
    """
    Register an image to auto-resize based on its parent visible width.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
        image_tag (Any): Parameter accepted by this routine.
        parent_tag (Any): Parameter accepted by this routine.
        aspect_ratio (Any): Parameter accepted by this routine. Defaults to the configured value.
        parent_of_parent (Any): Parameter accepted by this routine. Defaults to the configured value.
        default_pop_h (Any): Parameter accepted by this routine. Defaults to the configured value.
        tab (Any): Parameter accepted by this routine. Defaults to the configured value.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    state.setdefault("responsive_images", {})
    state["responsive_images"][image_tag] = {
        "parent": parent_tag,
        "parent_of_parent": parent_of_parent,
        "default_pop_h": default_pop_h,
        "aspect": float(aspect_ratio),
        "tab": tab,
    }
    state.setdefault("responsive_image_layout_signatures", {}).pop(image_tag, None)
    state.setdefault("responsive_image_applied_sizes", {}).pop(image_tag, None)
    _bind_responsive_image_resize_handlers(state, image_tag)
    request_responsive_image_update(state, frames=4)


# -----------------------------------------------------------------------------
# 23. Update responsive images
# -----------------------------------------------------------------------------
def request_responsive_image_update(state: dict[str, Any], frames: int = 2) -> None:
    """
    Request responsive image refreshes for the next few polling ticks.

    Args:
        state (dict[str, Any]): Shared application state.
        frames (int, optional): Number of upcoming polling ticks that should
            force a refresh. Defaults to `2`.

    Returns:
        None: This routine updates state in place.
    """
    state["_responsive_image_force_ticks"] = max(
        int(state.get("_responsive_image_force_ticks", 0) or 0),
        int(frames),
    )


def _tag_to_safe_suffix(item_tag: Any) -> str:
    """
    Convert an arbitrary Dear PyGui tag to a stable handler suffix.

    Args:
        item_tag (Any): Dear PyGui item tag.

    Returns:
        str: Safe tag suffix.
    """
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(item_tag))


def _bind_responsive_image_resize_handlers(state: dict[str, Any], image_tag: str) -> None:
    """
    Bind resize handlers to the container chain used by one responsive image.

    Args:
        state (dict[str, Any]): Shared application state.
        image_tag (str): Registered responsive image tag.

    Returns:
        None: This routine creates Dear PyGui item handlers when possible.
    """
    info = state.get("responsive_images", {}).get(image_tag)
    if not info:
        return

    watched_items = _responsive_image_watched_items(image_tag, info, include_image=False)

    def _on_responsive_item_resize(sender: Any, app_data: Any, user_data: Any) -> None:
        request_responsive_image_update(user_data, frames=4)
        poll_responsive_image_layout_changes(user_data)

    for item_tag in watched_items:
        handler_tag = f"responsive_image_resize_handler_{_tag_to_safe_suffix(item_tag)}"
        try:
            if not dpg.does_item_exist(handler_tag):
                with dpg.item_handler_registry(tag=handler_tag):
                    dpg.add_item_resize_handler(
                        callback=_on_responsive_item_resize,
                        user_data=state,
                    )
            dpg.bind_item_handler_registry(item_tag, handler_tag)
        except Exception:
            pass


def _responsive_image_watched_items(
    image_tag: Any,
    info: dict[str, Any],
    include_image: bool = True,
) -> list[Any]:
    """
    Build the set of layout items that can affect one responsive image.

    Args:
        image_tag (Any): Responsive image tag.
        info (dict[str, Any]): Registration metadata for the image.
        include_image (bool, optional): Include the image itself in the list.
            Defaults to `True`.

    Returns:
        list[Any]: Existing item tags to watch.
    """
    watched_items = []

    def _add_item(item_tag: Any) -> None:
        if item_tag and item_tag not in watched_items and dpg.does_item_exist(item_tag):
            watched_items.append(item_tag)

    if include_image:
        _add_item(image_tag)

    for item_tag in (info.get("parent"), info.get("parent_of_parent")):
        _add_item(item_tag)

    for start_tag in (image_tag, info.get("parent")):
        current_tag = start_tag
        for _ in range(8):
            if not current_tag or not dpg.does_item_exist(current_tag):
                break
            try:
                current_tag = dpg.get_item_parent(current_tag)
            except Exception:
                break
            _add_item(current_tag)

    return watched_items


def _measure_item_size(item_tag: Any) -> tuple[int, int]:
    """
    Read the current rendered size of a Dear PyGui item.

    Args:
        item_tag (Any): Dear PyGui item tag.

    Returns:
        tuple[int, int]: Width and height, or `(0, 0)` if unavailable.
    """
    if not item_tag or not dpg.does_item_exist(item_tag):
        return 0, 0
    try:
        w, h = dpg.get_item_rect_size(item_tag)
        if w or h:
            return int(w or 0), int(h or 0)
    except Exception:
        pass
    width = 0
    height = 0
    try:
        width = int(dpg.get_item_width(item_tag) or 0)
    except Exception:
        pass
    try:
        height = int(dpg.get_item_height(item_tag) or 0)
    except Exception:
        pass
    return width, height


def _responsive_image_matches_context(state: dict[str, Any], tab: Any) -> bool:
    """
    Check whether a responsive image belongs to the currently visible context.

    Args:
        state (dict[str, Any]): Shared application state.
        tab (Any): Registered tab or subtab identifier for the image.

    Returns:
        bool: `True` when the image should be updated in the active UI context.
    """
    current_tab = state.get("current_tab")
    if current_tab == "similarity_tab":
        return tab == state.get("current_similarity_subtab")
    if current_tab == "chemspace_tab":
        return tab == state.get("current_chemspace_subtab")
    return tab == current_tab


def poll_responsive_image_layout_changes(state: dict[str, Any]) -> None:
    """
    Refresh responsive images only when their layout inputs actually change.

    The poller watches the rendered size of each registered image and of the
    containers used to compute its responsive width. This catches viewport,
    child-window, table-column, splitter, and nested-container resizes without
    tying image updates to mouse movement.

    Args:
        state (dict[str, Any]): Shared application state.

    Returns:
        None: This routine updates responsive images when needed.
    """
    registry = state.get("responsive_images", {})
    if not registry:
        return

    signatures = state.setdefault("responsive_image_layout_signatures", {})
    context_signature = (
        state.get("current_tab"),
        state.get("current_similarity_subtab"),
        state.get("current_chemspace_subtab"),
        int(dpg.get_viewport_client_width() or dpg.get_viewport_width() or 0),
        int(dpg.get_viewport_client_height() or dpg.get_viewport_height() or 0),
    )
    layout_changed = False

    for img_tag, info in list(registry.items()):
        tab = info.get("tab", None)
        if not _responsive_image_matches_context(state, tab):
            signature = (context_signature, "inactive", tab)
        else:
            signature = (
                context_signature,
                tab,
                tuple(
                    (item_tag, *_measure_item_size(item_tag))
                    for item_tag in _responsive_image_watched_items(img_tag, info)
                ),
            )

        if signatures.get(img_tag) != signature:
            signatures[img_tag] = signature
            layout_changed = True

    force_ticks = int(state.get("_responsive_image_force_ticks", 0) or 0)
    if force_ticks > 0:
        state["_responsive_image_force_ticks"] = force_ticks - 1
        layout_changed = True

    fallback_tick = int(state.get("_responsive_image_fallback_tick", 0) or 0) + 1
    if fallback_tick >= 4:
        fallback_tick = 0
        layout_changed = True
    state["_responsive_image_fallback_tick"] = fallback_tick

    if layout_changed:
        update_responsive_images(state)

    reflow_boxplot = state.get("counts_boxplot_reflow_thumbnails")
    if callable(reflow_boxplot) and state.get("current_r_analysis_subtab") == "r_analysis_counts_subtab":
        try:
            plot_signature = (
                _measure_item_size("counts_boxplot"),
                _measure_item_size("counts_boxplot_window"),
                tuple(dpg.get_axis_limits("counts_boxplot_x_axis")),
                tuple(dpg.get_axis_limits("counts_boxplot_y_axis")),
            )
        except Exception:
            plot_signature = None
        if plot_signature and state.get("_counts_boxplot_layout_signature") != plot_signature:
            state["_counts_boxplot_layout_signature"] = plot_signature
            try:
                reflow_boxplot()
            except Exception:
                pass


def update_responsive_images(state: dict[str, Any]) -> None:
    """
    Recompute sizes for all registered responsive images.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    if callable(state.get("refresh_top_nav_layout")):
        try:
            state["refresh_top_nav_layout"]()
        except Exception:
            pass

    try:
        if (
            state.get("current_tab") == "chemspace_tab"
            and state.get("current_chemspace_subtab") == "pca_tab"
            and callable(state.get("pca_relayout_colormap_scale"))
        ):
            state["pca_relayout_colormap_scale"]()
    except Exception:
        pass

    registry = state.get("responsive_images", {})
    if not registry:
        return

    for img_tag, info in list(registry.items()):
        
        parent_tag = info["parent"]
        parent_of_parent = info["parent_of_parent"]
        aspect = info.get("aspect", 0.75)
        tab = info.get("tab", None)
            
        if not _responsive_image_matches_context(state, tab):
            continue

        if not (dpg.does_item_exist(img_tag) and dpg.does_item_exist(parent_tag)):
            continue

        parent_w, _ = _measure_item_size(parent_tag)
        if parent_w <= 0 and parent_of_parent:
            parent_w, _ = _measure_item_size(parent_of_parent)
        if parent_w <= 0:
            try:
                container = dpg.get_item_parent(img_tag)
            except Exception:
                container = None
            if container:
                parent_w, _ = _measure_item_size(container)
        if parent_w <= 0:
            continue

        effective_parent_w = parent_w

        safety_margin = 6

        if parent_tag == "mol_image":
            new_w = (parent_w - state["win_spacer"] * 8) / 3
            safety_margin = 5

        elif parent_tag.startswith("overview_table_texture_"):
            new_w = parent_w - state["win_spacer"] * 2

        elif parent_tag.startswith("rgroup_image_texture_"):
            new_w = parent_w - state["win_spacer"] * 2

        elif parent_tag.startswith("similarity_"):
            new_w = parent_w - state["win_spacer"] * 3

        elif parent_tag.startswith("clustered_"):
            new_w = parent_w - state["win_spacer"] * 3

        elif parent_tag.startswith("stereo_texture_"):
            new_w = parent_w - state["win_spacer"] * 2
            safety_margin = 12

        elif parent_tag == "mmpa_images_window":
            new_w = (parent_w - state["win_spacer"] * 3) / 4

        elif parent_tag.startswith("counts_rgroup_texture_") or parent_tag.startswith(
            "counts_boxplot_group_r_image_texture"
        ):
            new_w = parent_w - state["win_spacer"] * 2
            safety_margin = 12

        elif img_tag.startswith("heatmap_") and parent_tag == "heatmap_details_window":
            new_w = parent_w - state["win_spacer"] * 2
            # Keep the responsive image inside the visible child-window area
            # when the vertical scrollbar is present.
            safety_margin = 24

        elif img_tag.startswith("similarity_mol_") and parent_tag == "similarity_tanimoto_mol_couple_window":
            # This child window uses a vertical scrollbar; reserve its width so
            # the image stops before the scrollbar instead of ending underneath it.
            effective_parent_w = max(1, parent_w - 24)
            new_w = effective_parent_w - state["win_spacer"] * 2
            safety_margin = 16

        elif img_tag.startswith("clustered_mol_") and parent_tag == "clustered_similarity_mol_couple_window":
            # Same adjustment as the plain Tanimoto matrix: use the practical
            # content width of the scrollable child, not the full outer width.
            effective_parent_w = max(1, parent_w - 24)
            new_w = effective_parent_w - state["win_spacer"] * 2
            safety_margin = 16

        elif img_tag.startswith("landscape_mol") and parent_tag == "landscape_details_window":
            new_w = parent_w - state["win_spacer"] * 2
            safety_margin = 24
        else:
            new_w = parent_w - state["win_spacer"] * 2

        new_w = max(1, int(new_w) - safety_margin)
        new_h = max(1, int(new_w * aspect))
        applied_sizes = state.setdefault("responsive_image_applied_sizes", {})
        if applied_sizes.get(img_tag) == (new_w, new_h):
            continue
        dpg.configure_item(img_tag, width=new_w, height=new_h)
        applied_sizes[img_tag] = (new_w, new_h)
from app.utils.app_logger import log_event, log_exception
