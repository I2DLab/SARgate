"""
=================
overview_table.py

Molecule overview table for subset/activity selection and pagination.
=================
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Normalize to list
# 3. Show overview table window
# 4. Manage overview table
# 5. Show overview table
# 6. Build overview table
# 7. Prev overview table page callback
# 8. Next overview table page callback
# 9. Jump to mol id callback

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import re
import math
import time
import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
from typing import Any
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress
from app.analysis.overview.overview_decomposition import (
    autoscroll_to_button,
    update_molecule_choice_status,
    update_properties_windows,
    update_activities_windows,
    show_results_window,
    update_r_groups_choice_status
)
from app.utils.callbacks import (
    export_png_popup,
    change_tab,
    change_overview_subtab,
    on_button_click,
    register_responsive_image
)
from app.gui.themes_manager import (
    apply_input_text_theme, 
    apply_image_button_theme,
    apply_bordered_input_text_theme
)


# -----------------------------------------------------------------------------
# 2. Normalize to list
# -----------------------------------------------------------------------------
def _normalize_to_list(cell: Any) -> Any:
    """
    Execute the normalize to list routine.
    
    Args:
        cell (Any): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    try:
        if cell != cell:
            return []
    except Exception:
        pass

    if isinstance(cell, str):
        parts = [p.strip() for p in cell.split("|")]
        return [p for p in parts if p]

    try:
        iter(cell)
        if isinstance(cell, dict):
            return [str(cell)]
        if isinstance(cell, (bytes, bytearray)):
            return [cell.decode(errors="ignore")]
        return [str(x).strip() for x in cell if str(x).strip()]
    except Exception:
        return [str(cell).strip()] if str(cell).strip() else []
    


# -----------------------------------------------------------------------------
# 3. Show overview table window
# -----------------------------------------------------------------------------
def show_overview_table_window(state: dict[str, Any]) -> None:
    """
    Display the overview table panel with subset, activity, pagination, and search bar.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    # Update activity options when subset changes and (re)build the overview table.
    # -----------------------------------------------------------------------------
    # 3.1. Update overview table options
    # -----------------------------------------------------------------------------
    def update_overview_table_options(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Update the activity type combo and rebuild the overview table when subset changes.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
            user_data (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """

        subset = app_data
        bioact_types_dict = user_data["bioact_types_dict"]

        activities = bioact_types_dict.get(subset, {}).get("bioactivities", [])
        activity_items = ["Any"] + activities
        dpg.configure_item("overview_table_activity_type", items=activity_items, enabled=True)
        dpg.configure_item("overview_table_column", label=f"{subset.replace('subset_', 'Subset ')} Overview Table")
        dpg.set_value("overview_table_activity_type", "Any")

        manage_overview_table(user_data)

        
    manager_combo_width = state.get("overview_table_manager_combo_width", state.get("similarity_manager_combo_width", 180))

    # Build the manager window and wire all controls.
    with dpg.child_window(label="Overview table manager", parent="overview_table_manager", auto_resize_y=True,
                        no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):

        with dpg.group(horizontal=True):
            subsets = list(state["smiles_rgd_dict"].keys())
            default_subset = subsets[0] if subsets else "subset_1"

            # Subset selector (drives available activity types).
            dpg.add_text("Subset:")
            dpg.add_combo(width=manager_combo_width, 
                        height_mode=dpg.mvComboHeight_Large, items=subsets,
                        default_value=default_subset, tag="overview_table_subset_choice",
                        callback=update_overview_table_options, user_data=state)

            dpg.add_spacer(width=state["win_spacer"] * 4)

            # Activity selector (default to 'Any').
            bioact_types_dict = state["bioact_types_dict"]
            initial_activities = bioact_types_dict.get(default_subset, {}).get("bioactivities", [])
            activity_items = ["Any"] + initial_activities

            dpg.add_text("Activity type:")
            dpg.add_combo(width=manager_combo_width, 
                        height_mode=dpg.mvComboHeight_Largest, items=activity_items,
                        default_value="Any", tag="overview_table_activity_type",
                        callback=lambda s, a, u: manage_overview_table(u), user_data=state)

            dpg.add_spacer(width=state["win_spacer"] * 4)
            
            # Pagination controls and a jump-to box.
            dpg.add_button(arrow=True, direction=0, tag="overview_table_prev_page",
                        callback=prev_overview_table_page_callback, user_data=state)

            dpg.add_text("1/1", tag="overview_table_page_number")

            dpg.add_button(arrow=True, direction=1, tag="overview_table_next_page",
                        callback=next_overview_table_page_callback, user_data=state)

            dpg.add_spacer(width=state["win_spacer"] * 4)

            dpg.add_input_text(label="", hint="Search molecule by name or ID number",
                            on_enter=True, callback=jump_to_mol_id_callback, user_data=state, 
                            tag="overview_table_mol_jump", width=-1)
            dpg.bind_item_theme("overview_table_mol_jump", apply_bordered_input_text_theme(state))

            

# -----------------------------------------------------------------------------
# 4. Manage overview table
# -----------------------------------------------------------------------------
def manage_overview_table(state: dict[str, Any]) -> None:
    """
    Reset pagination and display the first page of the overview table.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    # Establish page size and start at page 1, then build the table.
    state["overview_table_page"] = 1
    state["overview_table_max_per_page"] = 12

    show_overview_table(state)


# -----------------------------------------------------------------------------
# 5. Show overview table
# -----------------------------------------------------------------------------
def show_overview_table(state: dict[str, Any]) -> None:
    """
    Build and display the overview table based on selected subset and activity.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    # Initialise view state, remove stale widgets, read summary CSV, and aggregate activities.

    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)

    if dpg.does_item_exist("overview_table_window"):
        dpg.delete_item("overview_table_window", children_only=True)

    summary_dir = state["summary_dir"]
    subset = dpg.get_value("overview_table_subset_choice")
    activity = dpg.get_value("overview_table_activity_type")

    # --- Read and aggregate per-molecule data from CSV ---
    csv_file = os.path.join(summary_dir, f"{subset}_summary.csv")
    set_loading_screen_progress(state, 4)
    data = pd.read_csv(csv_file)
    set_loading_screen_progress(state, 8)

    data = data[data["Mol"].notna()]
    if "MolName" in data.columns:
        data["MolName"] = data["MolName"].fillna("")

    fixed_columns = ["MolID", "MolName", "Substructure", "Mol", "logP", "MW", "HBA", "HBD"]
    rgroup_columns = [col for col in data.columns if col.startswith("R")]
    activity_columns = [col for col in data.columns if col not in fixed_columns + rgroup_columns]

    aggregated_rows = []
    grouped_data = list(data.groupby("MolID", sort=False))
    total_groups = max(1, len(grouped_data))
    for group_idx, (mol_id, group) in enumerate(grouped_data, start=1):
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
        if group_idx == total_groups or group_idx % max(1, total_groups // 25 or 1) == 0:
            set_loading_screen_progress(state, 8 + (group_idx / total_groups) * 22)

    df = pd.DataFrame(aggregated_rows)
    set_loading_screen_progress(state, 30)

    if activity != "Any" and activity in df.columns:
        df = df[df[activity].astype(str).str.strip() != ""]
    set_loading_screen_progress(state, 34)

    state["overview_table_df"] = df
    state["overview_table_full_smiles"] = df["Mol"].tolist()
    state["overview_table_full_indices"] = df.index.tolist()

    page = state["overview_table_page"]
    max_per_page = state["overview_table_max_per_page"]

    all_smiles = df["Mol"].tolist()
    all_indices = df.index.tolist()

    start = (page - 1) * max_per_page
    end = page * max_per_page

    smiles_list = all_smiles[start:end]
    true_indices = all_indices[start:end]

    state["overview_table_smiles_list"] = smiles_list
    state["overview_table_true_indices"] = true_indices
    set_loading_screen_progress(state, 40)


    # --- Choose the table layout based on the matrix toggle and build the table ---
    columns = 4
    smiles_list = state["overview_table_smiles_list"]
    true_indices = state["overview_table_true_indices"]
    img_width = state.get("overview_tbl_img_width", state.get("similarity_tbl_img_width"))
    df = state["overview_table_df"]
    width = state.get("overview_tbl_win_width", state.get("similarity_tbl_win_width"))
    height = state.get("overview_tbl_win_height", state.get("similarity_tbl_win_height"))
    pos = (state["win_spacer"], state.get("overview_tbl_win_y", state.get("similarity_tbl_win_y", 0)))

    build_overview_table(state, columns, smiles_list, true_indices, df, img_width, width, height, pos)

    # Update page number label and toggle matrix visibility accordingly.
    total_pages = math.ceil(len(df) / max_per_page)
    dpg.set_value("overview_table_page_number", f"Page {page}/{total_pages}")
    set_loading_screen_progress(state, 100)

    # Remove any loading overlay once rendering is complete.
    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")


# -----------------------------------------------------------------------------
# 6. Build table
# -----------------------------------------------------------------------------
def build_overview_table(
    state: dict[str, Any],
    columns: Any,
    smiles_list: str,
    true_indices: Any,
    df: Any,
    img_width: Any,
    width: Any,
    height: Any,
    pos: Any
) -> Any:
    """
    Build and display the molecule table with images, SMILES, names, and activity values.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
        columns (Any): Parameter accepted by this routine.
        smiles_list (Any): Parameter accepted by this routine.
        true_indices (Any): Parameter accepted by this routine.
        df (pd.DataFrame): Parameter accepted by this routine.
        img_width (Any): Parameter accepted by this routine.
        width (int): Parameter accepted by this routine.
        height (int): Parameter accepted by this routine.
        pos (Any): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """

    # Compute a 4:3 image height and declare a converter to normalised float32 RGBA arrays.
    tbl_img_width = img_width
    tbl_img_height = round(tbl_img_width / 4 * 3)

    # -----------------------------------------------------------------------------
    # 6.1. Pil image to dpg array
    # -----------------------------------------------------------------------------
    def pil_image_to_dpg_array(pil_img: Any) -> Any:
        """
        Convert a PIL image to a normalised float32 RGBA array for DearPyGui.

        Args:
            pil_img (PIL.Image): Image to convert.

        Returns:
            np.ndarray: Normalised RGBA image array.
        """
        
        img_rgba = pil_img.convert("RGBA")
        img_array = np.array(img_rgba).astype(np.float32) / 255.0
        return img_array

    # On molecule image click, switch tab, update selection, and autoscroll to relevant widgets.
    # -----------------------------------------------------------------------------
    # 6.2. Show mol in overview tab
    # -----------------------------------------------------------------------------
    def show_mol_in_overview_tab(sender: Any, app_data: Any, user_data: Any) -> None:
        """
        Callback to display a molecule in the overview tab when an image button is clicked.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
            user_data (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        subset, mol = user_data
        dpg.set_value("tab_bar", "overview_tab")
        change_tab(state)
        dpg.set_value("overview_tab_bar", "overview_decomposition_subtab")
        change_overview_subtab(state)

        update_molecule_choice_status(subset, state)
        on_button_click(subset, state)
        update_r_groups_choice_status(f"{subset}_mol_{mol}", state)
        update_properties_windows(f"{subset}_mol_{mol}", state)
        update_activities_windows(f"{subset}_mol_{mol}", state)
        on_button_click(f"{subset}_mol_{mol}", state)
        show_results_window(f"{subset}_mol_{mol}", state)

        autoscroll_to_button(sender="scaff_img", app_data=None, user_data=(subset))
        time.sleep(0.1)  # Small delay to ensure proper rendering before next scroll
        autoscroll_to_button(sender="mol_img", app_data=None, user_data=f"{subset}_mol_{mol}")


    # Generate RDKit 2D depictions for the given SMILES and register them as dynamic textures.
    overview_table_image_list = []
    total_smiles = max(1, len(smiles_list))
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rdDepictor.Compute2DCoords(mol)
        img_pil = Draw.MolToImage(mol, size=(tbl_img_width, tbl_img_height))
        overview_table_image_list.append((smi, img_pil))
        if (i + 1) == total_smiles or (i + 1) % max(1, total_smiles // 8) == 0:
            set_loading_screen_progress(state, 40 + ((i + 1) / total_smiles) * 28)


    total_images = max(1, len(overview_table_image_list))
    for idx, (smi, img_pil) in enumerate(overview_table_image_list):
        img_array = pil_image_to_dpg_array(img_pil)

        if not dpg.does_item_exist(f"overview_table_texture_{idx}"):
            dpg.add_dynamic_texture(img_pil.width, img_pil.height, img_array, tag=f"overview_table_texture_{idx}", parent="texture_registry")
        else:
            dpg.set_value(f"overview_table_texture_{idx}", img_array)
        if (idx + 1) == total_images or (idx + 1) % max(1, total_images // 8) == 0:
            set_loading_screen_progress(state, 68 + ((idx + 1) / total_images) * 16)

    # Build a border-separated fixed-fit table with per-cell info and a clickable image button.
    with dpg.child_window(label="Overview table", parent="overview_table_window", auto_resize_y=True,
                        no_scrollbar=True, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
        
        with dpg.table(tag="overview_table",
                    header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchSame, scrollY=False,
                    borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
            columns_per_row = columns
            for _ in range(columns_per_row):
                dpg.add_table_column()

            subset = dpg.get_value("overview_table_subset_choice")
            bioact_types_dict = state["bioact_types_dict"]
            activity_names = bioact_types_dict.get(subset, {}).get("bioactivities", [])

            for i in range(0, len(overview_table_image_list), columns_per_row):
                default_h = tbl_img_height + (state["win_spacer"] * 10)
                state["default_overview_table_row_height"] = default_h
                with dpg.table_row(tag=f"overview_table_row_{i // columns_per_row}", height=default_h):
                    for j in range(columns_per_row):
                        if i + j < len(overview_table_image_list):
                            smi, _ = overview_table_image_list[i + j]
                            tex_tag = f"overview_table_texture_{i + j}"
                            with dpg.table_cell():
                                real_idx = true_indices[i + j]
                                row = df.loc[real_idx]

                                # Show MolID and optional MolName in a read-only input for easy copy.
                                if "MolName" in df.columns and row["MolName"] != "":
                                    mol_name = row["MolName"]
                                    dpg.add_input_text(default_value=f"Mol {real_idx + 1} - {mol_name}", readonly=True, width=-1)
                                else:
                                    dpg.add_input_text(default_value=f"Mol {real_idx + 1}", readonly=True, width=-1)

                                # Assemble activity display lines with inferred units and operators.
                                activity_texts = []

                                for act in activity_names:
                                    vals = row.get(act, [])
                                    if isinstance(vals, str):
                                        vals = [v.strip() for v in vals.split(" | ") if v.strip()]

                                    vals_list = [x for x in _normalize_to_list(vals) if x]
                                    for v in vals_list:
                                        v_clean = v.replace(f"{act} = ", "").replace(f"{act} ", "").strip()

                                        if act in state["nM_activity_types"]:
                                            unit = "nM"
                                        elif act in state["percent_activities"]:
                                            unit = "%"
                                        elif act in state["ug/mL_activities"]:
                                            unit = "μg/mL"
                                        elif act in state["uM/min_activities"]:
                                            unit = "μM/min"
                                        else:
                                            unit = ""

                                        match = re.match(r"^(<=|>=|<|>)\s*(\d+(\.\d+)?)$", v_clean)
                                        if match:
                                            relation = match.group(1)
                                            if relation == "<=":
                                                relation = "≤"
                                            elif relation == ">=":
                                                relation = "≥"
                                            number = match.group(2)
                                            activity_texts.append(f"{act} {relation} {number} {unit}".strip())
                                        else:
                                            activity_texts.append(f"{act} = {v_clean} {unit}".strip())

                                if activity_texts:
                                    dpg.add_input_text(default_value=" | ".join(activity_texts), width=-1, readonly=True)
                                else:
                                    dpg.add_text("No activity", color=(150,150,150,255))
                                    
                                # Provide the SMILES string as read-only for quick copy/paste.
                                dpg.add_input_text(
                                    default_value=smi,
                                    readonly=True,
                                    width=state.get("overview_tbl_img_width", state.get("similarity_tbl_img_width", 200))
                                )

                                # Clickable image (opens the corresponding items in the Overview tab).
                                with dpg.child_window(tag=f"{tex_tag}_cell", auto_resize_y=True,
                                                    no_scrollbar=True, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
                                    dpg.add_image_button(texture_tag=tex_tag,
                                        tag=f"{tex_tag}_tag",
                                        width=tbl_img_width - (state["win_spacer"] * 2),
                                        height=tbl_img_height  - (state["win_spacer"] * 2),
                                        # frame_padding=5, 
                                        background_color=(0, 0, 0, 255),
                                        callback=show_mol_in_overview_tab,
                                        user_data=(subset, real_idx + 1)) 
                                register_responsive_image(
                                    state,
                                    image_tag=f"{tex_tag}_tag",
                                    parent_tag=f"{tex_tag}_cell",
                                    aspect_ratio=0.75,
                                    tab="overview_tab",
                                )
                                export_png_popup(f"{tex_tag}_tag", tex_tag, state)

                                dpg.bind_item_theme(f"{tex_tag}_tag", apply_image_button_theme(state))
                               
                        else:
                            dpg.add_table_cell()
                rendered_cards = min(i + columns_per_row, len(overview_table_image_list))
                if rendered_cards == len(overview_table_image_list) or rendered_cards % max(1, len(overview_table_image_list) // 4) == 0:
                    set_loading_screen_progress(state, 84 + (rendered_cards / max(1, len(overview_table_image_list))) * 14)

    dpg.bind_item_theme("overview_table_window", apply_input_text_theme())
    set_loading_screen_progress(state, 99)


# -----------------------------------------------------------------------------
# 7. Prev overview page callback
# -----------------------------------------------------------------------------
def prev_overview_table_page_callback(sender: Any, app_data: Any, user_data: Any) -> None:
    """
    Callback for navigating to the previous page in the overview table.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        user_data (Any): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    # Decrement page or wrap around to the last page, then rebuild the table.
    page = user_data["overview_table_page"]
    total = len(user_data["overview_table_df"])
    max_per_page = user_data["overview_table_max_per_page"]
    total_pages = math.ceil(total / max_per_page)

    if page > 1:
        user_data["overview_table_page"] -= 1
    else:
        user_data["overview_table_page"] = total_pages

    show_overview_table(user_data)


# -----------------------------------------------------------------------------
# 8. Next overview page callback
# -----------------------------------------------------------------------------
def next_overview_table_page_callback(sender: Any, app_data: Any, user_data: Any) -> None:
    """
    Callback for navigating to the next page in the overview table.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        user_data (Any): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
            
    # Increment page or wrap back to the first page, then rebuild the table.
    page = user_data["overview_table_page"]
    total = len(user_data["overview_table_df"])
    max_per_page = user_data["overview_table_max_per_page"]
    total_pages = math.ceil(total / max_per_page)

    if page < total_pages:
        user_data["overview_table_page"] += 1
    else:
        user_data["overview_table_page"] = 1

    show_overview_table(user_data)


# -----------------------------------------------------------------------------
# 9. Jump to mol id callback
# -----------------------------------------------------------------------------
def jump_to_mol_id_callback(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    """
    Callback to jump directly to the page containing a molecule by its ID or name.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    # Compute target page and rebuild the table; log if not found.
    query = str(app_data).strip()

    df = state["overview_table_df"]
    max_per_page = state["overview_table_max_per_page"]

    if query.isdigit():
        mol_id = int(query)
        if 1 <= mol_id <= len(df):
            page = math.ceil(mol_id / max_per_page)
            state["overview_table_page"] = page
            show_overview_table(state)
        else:
            print(f"[Mol ID {mol_id}] not found")
        return

    if "MolName" in df.columns:
        lower_query = query.lower()
        for idx, name in enumerate(df["MolName"]):
            if isinstance(name, str) and lower_query in name.lower():
                page = math.ceil((idx + 1) / max_per_page)
                state["overview_table_page"] = page
                show_overview_table(state)
                return

    print(f"[Mol Name '{query}'] not found")
