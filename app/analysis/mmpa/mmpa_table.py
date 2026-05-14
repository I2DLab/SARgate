"""
===============
mmpa_table.py
===============

MMPA transformation table visualisation.

Displays interactive tables of matched molecular pairs, highlighting R-group
transformations, ΔActivity values, and molecular images. Enables selection and
export of key transformations for SAR interpretation.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Draw mmpa table

import re
import io
import math
import dearpygui.dearpygui as dpg
import numpy as np
from typing import Any
from PIL import Image as pilImage
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from app.utils.app_logger import log_event, log_settings, log_exception, log_traceback
from app.utils.callbacks import (
    export_png_popup,
    register_responsive_image
)
from app.gui.loading_win import set_loading_screen_progress
from app.gui.themes_manager import apply_bordered_input_text_theme


# -----------------------------------------------------------------------------
# 2. Draw mmpa table
# -----------------------------------------------------------------------------
def draw_mmpa_table(activity: str, table_data: Any, state: dict[str, Any]) -> Any:
    """
    Build and display the MMPA results table, plus the dynamic image panel for.
    
    Args:
        activity (str): Parameter accepted by this routine.
        table_data (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    log_event("MMPA", f"Rendering MMPA table for activity '{activity}'", indent=1)
    log_settings("MMPA", indent=2, activity=activity, transformations=len(table_data) if hasattr(table_data, "__len__") else None)
    dpg.configure_item("mmpa_table_cont", show=True)
    set_loading_screen_progress(state, 97)
    state.setdefault("mmpa_table_page", 1)
    state.setdefault("mmpa_table_max_per_page", 25)
    state.setdefault("mmpa_filter_query", "")
    state.setdefault("mmpa_selected_transform", None)
    img_width = state["mmpa_img_width"]
    img_height = state["mmpa_img_height"]
    render_scale = 1.8
    render_width = int(round(img_width * render_scale))
    render_height = int(round(img_height * render_scale))
    pair_nav_button_size = state["win_spacer"] * 5

    def _position_mmpa_pair_buttons() -> None:
        """
        Place the prev/next pair buttons between the third and fourth images.
        """
        required = [
            "molA_image_widget",
            "molB_image_widget",
            "mmpa_prev_pair_button",
            "mmpa_next_pair_button",
        ]
        if not all(dpg.does_item_exist(tag) for tag in required):
            return
        try:
            mol_a_x, mol_a_y = dpg.get_item_pos("molA_image_widget")
            mol_b_x, _ = dpg.get_item_pos("molB_image_widget")
            mol_a_w, _ = dpg.get_item_rect_size("molA_image_widget")
            seam_center_x = (mol_a_x + mol_a_w + mol_b_x) / 2.0
            top_y = mol_a_y + max(4, int(state["win_spacer"] * 2))
            gap = max(4, int(state["win_spacer"] * 0.75))
            dpg.set_item_pos(
                "mmpa_prev_pair_button",
                (int(seam_center_x - pair_nav_button_size - (gap / 2)), int(top_y)),
            )
            dpg.set_item_pos(
                "mmpa_next_pair_button",
                (int(seam_center_x + (gap / 2)), int(top_y)),
            )
        except Exception:
            pass

    # -----------------------------------------------------------------------------
    # 2.1. Update mmpa images
    # -----------------------------------------------------------------------------
    def update_mmpa_images(sender: Any, app_data: Any, user_data: Any) -> Any:
        """
        Updates the molecular and R-group images for a given transformation based on current MMPA selection.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
            user_data (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        # Parse the current transformation, pick the active pair, and draw four panels (R1, R2, MolA, MolB).
        try:
            df = state["mmpa_dataframe"]

            # user_data contains transformation "r1 » r2"
            r_smiles = user_data.split(" \u00BB ")
            r_smiles1 = r_smiles[0].strip()
            r_smiles2 = r_smiles[1].strip()

            mol1_smi, mol2_smi = None, None

            # Locate the selected transformation and prepare paging controls
            for (r, t), vals in state["mmpa_table_data_raw"].items():
                if t == user_data:
                    if state.get("mmpa_current_transform_name") != user_data:
                        state["mmpa_current_vals"] = vals
                        state["mmpa_current_pair_index"] = 0
                        state["mmpa_current_transform_name"] = user_data

                    if dpg.does_item_exist("mmpa_prev_pair_button"):
                        dpg.configure_item("mmpa_prev_pair_button", show=(len(vals) > 1))
                    if dpg.does_item_exist("mmpa_next_pair_button"):
                        dpg.configure_item("mmpa_next_pair_button", show=(len(vals) > 1))
                    break

            vals = state.get("mmpa_current_vals", [])
            idx = state.get("mmpa_current_pair_index", 0)
            if not vals or idx >= len(vals):
                return

            pair_index = state.get("mmpa_current_pair_index", 0)
            _, name1, name2, _, _, _ = vals[pair_index % len(vals)]


            def extract_index(name: str) -> Any:
                """
                Extract a numerical index from the end of a molecule name string.

                Args:
                    name (str): Molecule name, potentially containing an ID at the end (e.g., 'MolName (123)').

                Returns:
                    int | None: Extracted integer index if found, otherwise None.
                """
                match = re.search(r'\(?(\d+)\)?$', name)
                if match:
                    return int(match.group(1))
                return None

            id1 = extract_index(name1)
            id2 = extract_index(name2)

            if id1 is None or id2 is None:
                print("[MMPA] Could not extract molecule index.")
                return

            # Resolve SMILES from Mol_IDs
            mol1_smi = df.loc[df["Mol_ID"] == id1, "Mol"].values
            if len(mol1_smi) > 0:
                mol1_smi = mol1_smi[0]
            else:
                print(f"[MMPA] Warning: Mol_ID {id1} not found in DataFrame")
                return
            
            mol2_smi = df.loc[df["Mol_ID"] == id2, "Mol"].values
            if len(mol2_smi) > 0:
                mol2_smi = mol2_smi[0]
            else:
                print(f"[MMPA] Warning: Mol_ID {id2} not found in DataFrame")
                return

            mol1 = Chem.MolFromSmiles(mol1_smi, sanitize=False)
            mol2 = Chem.MolFromSmiles(mol2_smi, sanitize=False)

            # Helper to read activity by Mol_ID, returning "N/A" if missing
            def get_activity_by_id(df: Any, mol_id: Any, col: Any) -> Any:
                """
                Return activity by id.
                
                Args:
                    df (pd.DataFrame): Input accepted by this routine.
                    mol_id (Any): Input accepted by this routine.
                    col (Any): Input accepted by this routine.
                
                Returns:
                    Any: Value returned by the routine.
                """
                row = df.loc[df["Mol_ID"] == mol_id]
                if not row.empty and col in row:
                    return row[col].values[0]
                return "N/A"

            # Determine unit for the activity label
            unit = "%" if activity in state["percent_activities"] else "μg/mL" if activity in state["ug/mL_activities"] else "μM/min" if activity in state["uM/min_activities"] else ""

            # Build activity strings (pValue for nM metrics, numeric otherwise)
            if activity in state["nM_activity_types"]:
                mol_1_act = get_activity_by_id(df, id1, "pValue")
                mol_2_act = get_activity_by_id(df, id2, "pValue")
                act_str_1 = f"p{activity} = {round(mol_1_act, 2) if mol_1_act != 'N/A' else 'N/A'} {unit}"
                act_str_2 = f"p{activity} = {round(mol_2_act, 2) if mol_2_act != 'N/A' else 'N/A'} {unit}"
            else:
                mol_1_act = get_activity_by_id(df, id1, "_activity_numeric")
                mol_2_act = get_activity_by_id(df, id2, "_activity_numeric")
                act_str_1 = f'{activity} = {mol_1_act} {unit}' if mol_1_act != "N/A" else f'{activity} = N/A'
                act_str_2 = f'{activity} = {mol_2_act} {unit}' if mol_2_act != "N/A" else f'{activity} = N/A'

            if mol1 is None or mol2 is None:
                log_event("MMPA", "Invalid SMILES during rendering.", indent=1, level="ERROR")
                return

            # Prepare 2D coordinates for drawing
            Chem.SanitizeMol(mol1, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            Chem.SanitizeMol(mol2, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            rdDepictor.Compute2DCoords(mol1)
            rdDepictor.Compute2DCoords(mol2)


            def render_and_set_texture(mol: Any, texture_tag: str, label_text: str) -> None:
                """
                Render a molecule with RDKit MolDraw2D (Cairo backend) and set it as a DearPyGui dynamic texture.
                
                Args:
                    mol (Chem.Mol): Input accepted by this routine.
                    texture_tag (Any): Input accepted by this routine.
                    label_text (Any): Input accepted by this routine.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                if not dpg.does_item_exist(texture_tag):
                    return

                drawer = rdMolDraw2D.MolDraw2DCairo(render_width, render_height)
                opts = drawer.drawOptions()
                opts.padding = 0.025
                opts.bondLineWidth = 1
                opts.minFontSize = 1

                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                drawer.FinishDrawing()

                png_bytes = drawer.GetDrawingText()
                mol_img = pilImage.open(io.BytesIO(png_bytes)).convert("RGBA")
                mol_arr = (np.array(mol_img) / 255.0).astype(np.float32).flatten()

                # Push pixel buffer into the dynamic texture
                dpg.set_value(texture_tag, mol_arr)
                tooltip_map = {
                    "r1_image_texture": "r1_image_tooltip_text",
                    "r2_image_texture": "r2_image_tooltip_text",
                    "molA_image_texture": "molA_image_tooltip_text",
                    "molB_image_texture": "molB_image_tooltip_text",
                }
                tooltip_tag = tooltip_map.get(texture_tag)
                if tooltip_tag and dpg.does_item_exist(tooltip_tag):
                    dpg.set_value(tooltip_tag, label_text)
                
            # Draw R-group fragments and matched molecules
            render_and_set_texture(Chem.MolFromSmiles(r_smiles1, sanitize=False), "r1_image_texture", f"From:  {r_smiles1}")
            render_and_set_texture(Chem.MolFromSmiles(r_smiles2, sanitize=False), "r2_image_texture", f"To:  {r_smiles2}")
            render_and_set_texture(mol1, "molA_image_texture", f"From:  Mol {id1} | {act_str_1}")
            render_and_set_texture(mol2, "molB_image_texture", f"To:  Mol {id2} | {act_str_2}")
            dpg.set_frame_callback(dpg.get_frame_count() + 1, lambda: _position_mmpa_pair_buttons())

        except Exception as e:
            log_exception("MMPA", "Error updating molecule images", e, indent=1)
            log_traceback("MMPA", indent=2)
        


    # -----------------------------------------------------------------------------
    # 2.2. Mmpa prev pair callback
    # -----------------------------------------------------------------------------
    def mmpa_prev_pair_callback(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Callback to display the previous molecular pair for the currently selected transformation.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
            user_data (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        vals = state.get("mmpa_current_vals", [])
        if not vals:
            return
        state["mmpa_current_pair_index"] = (state["mmpa_current_pair_index"] - 1) % len(vals)
        update_mmpa_images(None, None, state["mmpa_current_transform_name"])


    # -----------------------------------------------------------------------------
    # 2.3. Mmpa next pair callback
    # -----------------------------------------------------------------------------
    def mmpa_next_pair_callback(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Callback to display the next molecular pair for the currently selected transformation.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
            user_data (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        vals = state.get("mmpa_current_vals", [])
        if not vals:
            return
        state["mmpa_current_pair_index"] = (state["mmpa_current_pair_index"] + 1) % len(vals)
        update_mmpa_images(None, None, state["mmpa_current_transform_name"])


    # -----------------------------------------------------------------------------
    # 2.4. Sort callback
    # -----------------------------------------------------------------------------
    def sort_callback(sender: Any, sort_specs: Any) -> None:
        """
        Sorts the MMPA table based on the selected column.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            sort_specs (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        if not sort_specs:
            return

        column_id, direction = sort_specs[0]
        column_idx = dpg.get_item_label(column_id)

        idx_map = {
            "ID": 0,
            "R": 1,
            "Transformation": 2,
            "Molecule Pairs": 3,
            f"Mean Δ{'p' if is_log_scale else ''}{activity}": 4,
            "Std Dev": 5,
            "Count": 6,
        }
        col_idx = idx_map.get(column_idx, None)
        if col_idx is None:
            return

        sorted_rows = sorted(
            [(row_data, row_data[col_idx]) for row_data in state["mmpa_table_data"]],
            key=lambda x: x[1],
            reverse=(direction < 0)
        )
        state["mmpa_table_data"] = [row_data for row_data, _ in sorted_rows]
        state["mmpa_table_page"] = 1
        rebuild_mmpa_table_rows()
    
        
    def _mmpa_filtered_rows() -> list[Any]:
        """
        Return table rows after applying the current search filter.
        """
        query = str(state.get("mmpa_filter_query", "") or "").strip().lower()
        if not query:
            return list(state.get("mmpa_table_data", []))
        filtered_rows = []
        for row_data in state.get("mmpa_table_data", []):
            haystack = str(row_data[3]).lower()
            if query in haystack:
                filtered_rows.append(row_data)
        return filtered_rows

    def _mmpa_total_pages() -> int:
        """
        Compute total pages using the current filter and rows-per-page setting.
        """
        max_per_page = max(1, int(state.get("mmpa_table_max_per_page", 25)))
        return max(1, math.ceil(len(_mmpa_filtered_rows()) / max_per_page))

    def _sync_mmpa_page_widgets() -> None:
        """
        Update page widgets to match current pagination state.
        """
        total_pages = _mmpa_total_pages()
        state["mmpa_table_page"] = max(1, min(int(state.get("mmpa_table_page", 1)), total_pages))
        if dpg.does_item_exist("mmpa_tbl_page_number"):
            dpg.set_value("mmpa_tbl_page_number", str(int(state.get("mmpa_table_page", 1))))
        if dpg.does_item_exist("mmpa_tbl_total_pages"):
            dpg.set_value("mmpa_tbl_total_pages", f"/ {total_pages}")

    def rebuild_mmpa_table_rows() -> None:
        """
        Rebuild only the visible MMPA table rows for the current page.
        """
        if not dpg.does_item_exist("mmpa_table"):
            return
        filtered_rows = _mmpa_filtered_rows()
        total_pages = _mmpa_total_pages()
        state["mmpa_table_page"] = max(1, min(int(state.get("mmpa_table_page", 1)), total_pages))
        max_per_page = max(1, int(state.get("mmpa_table_max_per_page", 25)))
        page = int(state.get("mmpa_table_page", 1))
        start = (page - 1) * max_per_page
        end = start + max_per_page
        page_rows = filtered_rows[start:end]

        for row_id in dpg.get_item_children("mmpa_table", 1) or []:
            dpg.delete_item(row_id)

        state["mmpa_row_ids"] = []

        selected_transform = state.get("mmpa_selected_transform")
        first_selectable_info = None
        selected_row_found = False

        for row_data in page_rows:
            transformation_value = row_data[2]
            row_tag = dpg.generate_uuid()
            row_cell_tags = []

            with dpg.table_row(parent="mmpa_table", tag=row_tag):
                for col_idx, cell in enumerate(row_data):
                    with dpg.table_cell():
                        text_value = str(cell)
                        if col_idx == 4:
                            clean = text_value.replace("*", "").replace("#", "")
                            if "*" in text_value:
                                color = (200, 50, 0, 255)
                            elif "#" in text_value:
                                color = (180, 150, 0, 255)
                            else:
                                color = None
                            display_text = clean
                        else:
                            color = None
                            display_text = text_value

                        selectable_tag = f"{row_tag}_col_{col_idx}"
                        row_cell_tags.append(selectable_tag)
                        dpg.add_selectable(
                            label=display_text,
                            tag=selectable_tag,
                            default_value=(transformation_value == selected_transform),
                            callback=make_callback(transformation_value, row_cell_tags, selected_row_ref),
                            span_columns=True
                        )

                        if color:
                            dpg.bind_item_theme(selectable_tag, make_text_color_theme(color))
                        else:
                            dpg.bind_item_theme(selectable_tag, selectable_theme_tag)

            if first_selectable_info is None:
                first_selectable_info = (transformation_value, list(row_cell_tags))
            if transformation_value == selected_transform:
                selected_row_ref["current"] = list(row_cell_tags)
                selected_row_found = True
            state["mmpa_row_ids"].append(row_tag)

        _sync_mmpa_page_widgets()
        if not selected_row_found and first_selectable_info is not None:
            transf, row_tags = first_selectable_info
            state["mmpa_selected_transform"] = transf
            on_row_click(transf, row_tags, selected_row_ref)

    def prev_mmpa_page_callback(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Go to the previous MMPA table page.
        """
        if int(state.get("mmpa_table_page", 1)) <= 1:
            return
        state["mmpa_table_page"] = int(state.get("mmpa_table_page", 1)) - 1
        rebuild_mmpa_table_rows()

    def next_mmpa_page_callback(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Go to the next MMPA table page.
        """
        total_pages = _mmpa_total_pages()
        if int(state.get("mmpa_table_page", 1)) >= total_pages:
            return
        state["mmpa_table_page"] = int(state.get("mmpa_table_page", 1)) + 1
        rebuild_mmpa_table_rows()

    def jump_to_mmpa_page_callback(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Jump to a specific MMPA table page.
        """
        try:
            state["mmpa_table_page"] = int(str(app_data).strip())
        except Exception:
            _sync_mmpa_page_widgets()
            return
        rebuild_mmpa_table_rows()

    def update_mmpa_rows_per_page(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Update the number of rows shown per page.
        """
        try:
            new_value = max(1, int(app_data))
        except Exception:
            new_value = 25
        if new_value == int(state.get("mmpa_table_max_per_page", 25)):
            return
        state["mmpa_table_max_per_page"] = new_value
        state["mmpa_table_page"] = 1
        rebuild_mmpa_table_rows()

    def update_mmpa_filter(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Filter rows and reset pagination to the first page.
        """
        state["mmpa_filter_query"] = str(app_data or "")
        state["mmpa_table_page"] = 1
        rebuild_mmpa_table_rows()

    is_log_scale = activity in state["nM_activity_types"]

    with dpg.child_window(parent="mmpa_table_window", width=-1, height=-1,
                          no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):

        # --- Controls row: network switch + filter box ---
        with dpg.group(horizontal=True):

            # Toggle back to the table
            with dpg.group(tag="show_mmpa_table_button_group", show=False):
                dpg.add_button(label="Show MMPA Table", callback= lambda: (
                                dpg.configure_item("mmpa_table", show=True),
                                dpg.configure_item("show_mmpa_network_button_group", show=True),
                                dpg.configure_item("show_mmpa_table_button_group", show=False),
                                dpg.configure_item("mmpa_network_plot", show=False)
                                )
                )

            dpg.add_button(
                arrow=True,
                direction=dpg.mvDir_Left,
                tag="mmpa_tbl_prev_page",
                callback=prev_mmpa_page_callback,
            )
            dpg.add_input_text(
                tag="mmpa_tbl_page_number",
                width=80,
                default_value=str(int(state.get("mmpa_table_page", 1))),
                on_enter=True,
                callback=jump_to_mmpa_page_callback,
            )
            dpg.add_text("/ 1", tag="mmpa_tbl_total_pages")
            dpg.add_button(
                arrow=True,
                direction=dpg.mvDir_Right,
                tag="mmpa_tbl_next_page",
                callback=next_mmpa_page_callback,
            )
            dpg.add_spacer(width=state["win_spacer"])
            dpg.add_combo(
                items=["10", "25", "50", "100"],
                default_value=str(int(state.get("mmpa_table_max_per_page", 25))),
                width=80,
                tag="mmpa_tbl_rows_per_page",
                callback=update_mmpa_rows_per_page,
            )
            dpg.add_spacer(width=state["win_spacer"])
            # Text filter for the "Molecule Pairs" column
            dpg.add_input_text(hint="Search by molecule ID or name ...", tag="mmpa_filter_input", 
                               auto_select_all=True, width=-1,
                               default_value=str(state.get("mmpa_filter_query", "")),
                               callback=update_mmpa_filter)
            dpg.bind_item_theme("mmpa_filter_input", apply_bordered_input_text_theme(state))
        # --- Main table ---
        with dpg.table(tag="mmpa_table", header_row=True,
                       no_host_extendX=True, no_host_extendY=True,
                       borders_innerH=True, borders_outerH=True, 
                       borders_innerV=True, borders_outerV=True, freeze_rows=1, scrollY=True,
                       row_background=True, resizable=True, sortable=True, sort_tristate=True, callback=sort_callback):

            # STEP 1.6.1: Create table columns with tooltips
            dpg.add_table_column(label="ID", tag="mmpa_col_0", init_width_or_weight=3)
            dpg.add_table_column(label="R", tag="mmpa_col_1", init_width_or_weight=2)
            dpg.add_table_column(label="Transformation", tag="mmpa_col_2", init_width_or_weight=15)
            dpg.add_table_column(label="Molecule Pairs", tag="mmpa_col_3", init_width_or_weight=58)
            dpg.add_table_column(label=f"Mean Δ{'p' if is_log_scale else ''}{activity}", tag="mmpa_col_4", init_width_or_weight=15)
            dpg.add_table_column(label="StDev", tag="mmpa_col_5", init_width_or_weight=4)
            dpg.add_table_column(label="Freq.", tag="mmpa_col_6", init_width_or_weight=3)   
            

            # STEP 1.6.2: Row selection callbacks and helper themes
            def on_row_click(
                transformation_value: Any,
                row_selectable_tags: Any,
                selected_ref: Any
            ) -> None:
                """
                Execute the on row click routine.
                
                Args:
                    transformation_value (Any): Input accepted by this routine.
                    row_selectable_tags (Any): Input accepted by this routine.
                    selected_ref (Any): Input accepted by this routine.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                update_mmpa_images(None, None, transformation_value)
                state["mmpa_selected_transform"] = transformation_value
                previous_tags = selected_ref.get("current") or []
                for prev_tag in previous_tags:
                    if dpg.does_item_exist(prev_tag):
                        dpg.set_value(prev_tag, False)
                for tag in row_selectable_tags:
                    if dpg.does_item_exist(tag):
                        dpg.set_value(tag, True)
                selected_ref["current"] = list(row_selectable_tags)

            def make_callback(transf: Any, row_tags: Any, ref: Any) -> Any:
                """
                Execute the make callback routine.
                
                Args:
                    transf (Any): Input accepted by this routine.
                    row_tags (Any): Input accepted by this routine.
                    ref (Any): Input accepted by this routine.
                
                Returns:
                    Any: Value returned by the routine.
                """
                return lambda s, a: on_row_click(transf, row_tags, ref)

            def make_text_color_theme(color: Any) -> Any:
                """
                Execute the make text color theme routine.
                
                Args:
                    color (Any): Input accepted by this routine.
                
                Returns:
                    Any: Value returned by the routine.
                """
                tag = f"theme_{color}"
                if not dpg.does_item_exist(tag):
                    with dpg.theme(tag=tag):
                        with dpg.theme_component(dpg.mvSelectable):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, color)
                            dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, 0.0, 0.5, category=dpg.mvThemeCat_Core)
                return tag

            selectable_theme_tag = "mmpa_table_selectable_theme"
            if not dpg.does_item_exist(selectable_theme_tag):
                with dpg.theme(tag=selectable_theme_tag):
                    with dpg.theme_component(dpg.mvSelectable):
                        dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, 0.0, 0.5, category=dpg.mvThemeCat_Core)
            set_loading_screen_progress(state, 97.4)

            selected_row_ref = {"current": None}
            with dpg.theme() as table_theme:
                with dpg.theme_component(dpg.mvTable):
                    pass              
            dpg.bind_item_theme("mmpa_table", table_theme)


        # STEP 1.6.4: Initial sort on Mean Δp{activity} (desc)
        try:
            # Get internal column id for the Mean Δ column
            col_ids = dpg.get_item_children("mmpa_table", 0)  # slot 0 = columns
            mean_col_id = next((cid for cid in col_ids if dpg.get_item_alias(cid) == "mmpa_col_4"), None)
            if mean_col_id is not None:
                # direction=1 -> descending per tua logica nel sort_callback
                sort_callback(None, [(mean_col_id, -1)])
        except Exception:
            pass
    empty_data = np.zeros((render_height * render_width * 4,), dtype=np.float32)

    if not dpg.does_item_exist("r1_image_texture") and not dpg.does_item_exist("r2_image_texture") and not dpg.does_item_exist("molA_image_texture") and not dpg.does_item_exist("molB_image_texture"):
        dpg.add_dynamic_texture(render_width, render_height, empty_data, tag="r1_image_texture", parent="texture_registry")
        dpg.add_dynamic_texture(render_width, render_height, empty_data, tag="r2_image_texture", parent="texture_registry")
        dpg.add_dynamic_texture(render_width, render_height, empty_data, tag="molA_image_texture", parent="texture_registry")
        dpg.add_dynamic_texture(render_width, render_height, empty_data, tag="molB_image_texture", parent="texture_registry")
    else:
        dpg.set_value("r1_image_texture", empty_data)
        dpg.set_value("r2_image_texture", empty_data)
        dpg.set_value("molA_image_texture", empty_data)
        dpg.set_value("molB_image_texture", empty_data)
    rebuild_mmpa_table_rows()
    set_loading_screen_progress(state, 99.1)
    set_loading_screen_progress(state, 99.5)


    with dpg.child_window(parent="mmpa_images_window", width=-1, auto_resize_y=True,
                        no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
        with dpg.group(horizontal=True):
            # R-group images
            dpg.add_image("r1_image_texture", width=img_width, height=img_height, tag="r1_image_widget", border_color=(0, 0, 0, 0))
            with dpg.tooltip("r1_image_widget", delay=0):
                dpg.add_text("", tag="r1_image_tooltip_text")
            register_responsive_image(
                state,
                image_tag="r1_image_widget",
                parent_tag="mmpa_images_window",
                aspect_ratio=0.75,
                tab="mmpa_tab"
            )
            export_png_popup("r1_image_widget", "r1_image_texture", state)

            dpg.add_image("r2_image_texture", width=img_width, height=img_height, tag="r2_image_widget", border_color=(0, 0, 0, 0))
            with dpg.tooltip("r2_image_widget", delay=0):
                dpg.add_text("", tag="r2_image_tooltip_text")
            register_responsive_image(
                state,
                image_tag="r2_image_widget",
                parent_tag="mmpa_images_window",
                aspect_ratio=0.75,
                tab="mmpa_tab"
            )
            export_png_popup("r2_image_widget", "r2_image_texture", state)

            # Molecule images
            dpg.add_image("molA_image_texture", width=img_width, height=img_height, tag="molA_image_widget", border_color=(0, 0, 0, 0))
            with dpg.tooltip("molA_image_widget", delay=0):
                dpg.add_text("", tag="molA_image_tooltip_text")
            register_responsive_image(
                state,
                image_tag="molA_image_widget",
                parent_tag="mmpa_images_window",
                aspect_ratio=0.75,
                tab="mmpa_tab"
            )
            export_png_popup("molA_image_widget", "molA_image_texture", state)

            dpg.add_image("molB_image_texture", width=img_width, height=img_height, tag="molB_image_widget", border_color=(0, 0, 0, 0))
            with dpg.tooltip("molB_image_widget", delay=0):
                dpg.add_text("", tag="molB_image_tooltip_text")
            register_responsive_image(
                state,
                image_tag="molB_image_widget",
                parent_tag="mmpa_images_window",
                aspect_ratio=0.75,
                tab="mmpa_tab"
            )
            export_png_popup("molB_image_widget", "molB_image_texture", state)

        # Navigation arrows for multi-pair transformations
        dpg.add_button(arrow=True, direction=0, tag="mmpa_prev_pair_button", show=False,
                        width=pair_nav_button_size,
                        height=pair_nav_button_size,
                        pos=((img_width * 3) - (state["win_spacer"]) * 2, state["win_spacer"] * 2),
                        callback=mmpa_prev_pair_callback, user_data="mmpa_current_transform_name")
        dpg.add_button(arrow=True, direction=1, tag="mmpa_next_pair_button", show=False,
                        width=pair_nav_button_size,
                        height=pair_nav_button_size,
                        pos=((img_width * 3) + (state["win_spacer"] * 5), state["win_spacer"] * 2),
                        callback=mmpa_next_pair_callback, user_data="mmpa_current_transform_name")
        dpg.set_frame_callback(dpg.get_frame_count() + 1, lambda: _position_mmpa_pair_buttons())
        

        # STEP 1.6.5: Auto-select first row
        try:
            rows = dpg.get_item_children("mmpa_table", 1) or []
            if rows:
                first_row = rows[0]
                cells = dpg.get_item_children(first_row, 1)
                if len(cells) >= 3:
                    # With the new leading ID column, Transformation is now in column 2.
                    c2_children = dpg.get_item_children(cells[2], 1)
                    transf_label = dpg.get_item_configuration(c2_children[0]).get("label", "") if c2_children else ""
                    c2_alias = dpg.get_item_alias(c2_children[0]) if c2_children else ""
                    if c2_alias and transf_label:
                        on_row_click(transf_label, c2_alias, selected_row_ref)
        except:
            pass
    set_loading_screen_progress(state, 100)
