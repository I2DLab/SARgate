"""
============
utilities.py
============

Auxiliary tools and utility window.

Contains optional interface tools for quick conversions, dataset inspection,
or debugging tasks within SARgate. Hidden by default, can be toggled from the
main menu for power users and developers.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Show utilities window

import os
import csv
import re
import time
import dearpygui.dearpygui as dpg
import requests
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
from collections import Counter
from typing import Any
from rdkit import Chem
from rdkit.Chem import Draw
from app.utils.app_logger import log_event, log_settings
from app.gui.loading_win import draw_loading_screen
from app.utils.native_dialogs import open_directory_dialog, open_file_dialog
from app.gui.themes_manager import apply_bordered_input_text_theme


# -----------------------------------------------------------------------------
# 2. Show utilities window
# -----------------------------------------------------------------------------
def show_utilities_window(state: dict[str, Any], log_on_open: bool = True) -> Any:
    """
    Displays the Utilities GUI panel and all its tools.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    if log_on_open:
        log_event("Utilities", "Opening utilities window", indent=1)
        log_settings("Utilities", indent=2, input_dir=state.get("input_dir"), output_dir=state.get("output_dir"))

    # Create the main utilities group and append to the parent child window.

    with dpg.group(parent="utils_window"):

        # --- Spacer before next section ---
        dpg.add_spacer(height=state["win_spacer"])

        # Collapsible section to convert molecule strings (SMILES/SMARTS/InChI/MolBlock), draw the molecule, and export to another format.
        with dpg.tree_node(label="Mol Draw and String Converter", tag="molstring_converter_tree_node", default_open=False):
            dpg.add_text("Convert a molecule string (SMILES, SMARTS, InChI or MolBlock) to another format and draw the molecule image")
            
            def convert_molstring(sender: Any, app_data: Any, user_data: Any) -> None:
                """
                Live converter callback:.
                
                Args:
                    sender (Any): Input accepted by this routine.
                    app_data (Any): Input accepted by this routine.
                    user_data (Any): Input accepted by this routine.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                molstring_input = dpg.get_value("converter_molstring_input")
                mol = None

                # Try SMILES
                try:
                    mol = Chem.MolFromSmiles(molstring_input)
                except Exception:
                    pass

                # Try SMARTS
                if mol is None:
                    try:
                        mol = Chem.MolFromSmarts(molstring_input)
                    except Exception:
                        pass

                # Try InChI
                if mol is None:
                    try:
                        mol = Chem.MolFromInchi(molstring_input)
                    except Exception:
                        pass

                # Try MolBlock
                if mol is None:
                    try:
                        mol = Chem.MolFromMolBlock(molstring_input)
                    except Exception:
                        pass
                
                # If parsed, draw image and convert to selected output format; otherwise, prepare a blank texture and error text.
                if mol is not None:
                    mol_img = Draw.MolToImage(mol, size=(img_width, img_height)).convert("RGBA")
                    data = (np.array(mol_img) / 255.0).flatten().astype(np.float32)
                    output_format = dpg.get_value("molstring_converter_combo")
                    if output_format == "SMILES":
                        molstring_output = Chem.MolToSmiles(mol)
                    elif output_format == "SMARTS":
                        molstring_output = Chem.MolToSmarts(mol)
                    elif output_format == "InChI":
                        molstring_output = Chem.MolToInchi(mol)
                    elif output_format == "MolBlock":
                        molstring_output = Chem.MolToMolBlock(mol)
                    dpg.set_value("converter_molstring_output", molstring_output)
                else:
                    data = np.ones((img_height, img_width, 4), dtype=np.float32)
                    dpg.set_value("converter_molstring_output", "Invalid molecule string")

                texture_tag = dpg.add_static_texture(img_width, img_height, data, parent="texture_registry")
                dpg.configure_item("mol_from_string_image", texture_tag=texture_tag)

            # --- STEP 2.1: INITIALISE IMAGE TEXTURE PLACEHOLDERS ---
            # Prepare canvas size, a blank texture, and register it.
            img_width = 400
            img_height = int(img_width * 0.75)

            texture_tag = "mol_from_string_texture"
            blank_image = np.ones((img_height, img_width, 4), dtype=np.float32)
            dpg.add_static_texture(img_width, img_height, blank_image.flatten(), 
                                   tag=texture_tag, parent="texture_registry")

            # --- STEP 2.2: LAYOUT: INPUT, IMAGE, OUTPUT FORMAT AND OUTPUT TEXT ---
            # Three side-by-side groups: (A) input text, (B) rendered image, (C) output settings and converted string.
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Molecule String Input")
                    dpg.add_text("Type here the SMILES, SMARTS, InChI or MolBlock string:")
                    dpg.add_input_text(tag="converter_molstring_input", hint="SMILES, SMARTS, InChI or MolBlock",
                                    width=img_width, height=img_height, multiline=True,
                                    callback=convert_molstring)
                    dpg.bind_item_theme("converter_molstring_input", apply_bordered_input_text_theme(state))

                with dpg.group():
                    dpg.add_text("")
                    dpg.add_text("Molecule Image")
                    dpg.add_image(texture_tag, width=img_width, height=img_height,
                                tag="mol_from_string_image", border_color=(0, 0, 0, 255))

                with dpg.group():
                    dpg.add_text("Convert to:")
                    dpg.add_combo(items=["SMILES", "SMARTS", "InChI", "MolBlock"], default_value="SMILES",
                                  width=img_width, tag="molstring_converter_combo",
                                  callback=convert_molstring)
                    dpg.add_input_text(default_value="", tag="converter_molstring_output", readonly=True, auto_select_all=True,
                                    width=img_width, height=img_height, multiline=True)
                    dpg.bind_item_theme("converter_molstring_output", apply_bordered_input_text_theme(state))

        # --- Spacer and separator before next tool --- 
        dpg.add_spacer(height=state["win_spacer"])
        dpg.add_separator()

        # Collapsible section: select a folder, pick multiple SDFs, and merge them into a single SDF.
        with dpg.tree_node(label="Merge SDFs", tag="merge_SDFs_tree_node", default_open=False):
            dpg.add_text("Merge two or more SDF files into a single output SDF output (e.g., subsets of interest for further analysis)")

            def select_merge_folder_callback(folder: str) -> None:
                """
                Refresh the merge table after selecting a source folder.
                
                Args:
                    folder (str): Directory that contains the SDF files to
                        display in the merge table.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                if os.path.isdir(folder):
                    sdf_files = sorted((f for f in os.listdir(folder) if f.endswith(".sdf")),key=sort_keys)

                    # Update state and UI labels/table.
                    state["merge_sdf_folder"] = folder
                    state["merge_sdf_selected"] = set()
                    dpg.set_value("merge_sdf_label", f"Selected Folder:\n{folder}")
                    dpg.delete_item("merge_sdf_table", children_only=True)

                    # Create fixed 5-column grid.
                    for _ in range(5):
                        dpg.add_table_column(parent="merge_sdf_table", init_width_or_weight=20)

                    # Populate rows with selectable filenames.
                    for row_start in range(0, len(sdf_files), 5):
                        with dpg.table_row(parent="merge_sdf_table"):
                            for i in range(5):
                                idx = row_start + i
                                if idx < len(sdf_files):
                                    filename = sdf_files[idx]
                                    tag = f"merge_sdf_selectable_{idx}"
                                    dpg.add_selectable(label=filename, tag=tag,
                                                       callback=toggle_sdf_selection, user_data=filename)
                                else:
                                    dpg.add_text("")

            def prompt_merge_folder_selection() -> None:
                """
                Open the native folder picker for the merge-SDF tool.

                Args:
                    None.

                Returns:
                    None: This routine updates the merge source folder in place.
                """
                selected_folder = open_directory_dialog(
                    "Select the Folder Containing SDF Files to Merge",
                    state.get("merge_sdf_folder", state["input_dir"]),
                )
                if selected_folder:
                    select_merge_folder_callback(selected_folder)
                                    
            def toggle_sdf_selection(sender: Any, app_data: Any, user_data: Any) -> None:
                """
                Toggle the inclusion of a given SDF filename in the merge set and print selection summary to stdout.
                
                Args:
                    sender (Any): Input accepted by this routine.
                    app_data (Any): Input accepted by this routine.
                    user_data (Any): Input accepted by this routine.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                filename = user_data
                if filename in state["merge_sdf_selected"]:
                    state["merge_sdf_selected"].remove(filename)
                else:
                    state["merge_sdf_selected"].add(filename)
                print(f"Selected SDFs: {len(state['merge_sdf_selected'])}")
                for sdf in state["merge_sdf_selected"]:
                    print(f"    {sdf}")

            def merge_selected_sdfs_callback() -> None:
                """
                Execute the merge:.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                folder = state.get("merge_sdf_folder", state["input_dir"])
                selected = state.get("merge_sdf_selected", set())
                output_name = dpg.get_value("merge_sdf_output_name").strip()
                output_path = os.path.join(state["input_dir"], output_name)

                # Ensure an output name was provided.
                if output_name == "":
                    dpg.configure_item("insert_name_popup", show=True)
                    return
                
                # Ensure extension is .sdf.
                if not output_name.endswith(".sdf"):
                    output_name += ".sdf"

                # Ensure at least two files selected.
                if len(selected) < 2:
                    dpg.configure_item("merge_too_few_popup", show=True)
                    return

                # Avoid overwriting existing files.
                if os.path.exists(output_path):
                    dpg.configure_item("merge_exists_popup", show=True)
                    return

                # Show loading layer and perform the merge.
                draw_loading_screen(state, bg=False)

                writer = Chem.SDWriter(output_path)
                for sdf_file in selected:
                    file_path = os.path.join(folder, sdf_file)
                    suppl = Chem.SDMolSupplier(file_path)
                    for mol in suppl:
                        if mol:
                            writer.write(mol)
                writer.close()

                # Remove cover if any.
                if dpg.does_item_exist("cover_layer"):
                    dpg.delete_item("cover_layer")
                
            def close_merge_popup() -> None:
                """
                Close the 'file exists' pop-up for the merge section.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                dpg.configure_item("merge_exists_popup", show=False)

            def sort_keys(filename: str) -> Any:
                """
                Natural-sort helper: split strings on digits to ensure numeric order for embedded numbers.
                
                Args:
                    filename (str): Input accepted by this routine.
                
                Returns:
                    Any: Value returned by the routine.
                """
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

            # --- STEP 3.1: MERGE SDFs TOOLBAR (SELECT FOLDER, NAME, RUN) ---
            with dpg.group(horizontal=True):
                dpg.add_button(label="Select Folder", callback=prompt_merge_folder_selection)
                dpg.add_input_text(tag="merge_sdf_output_name", hint="Output SDF name...", width=200)
                dpg.add_button(label="Merge Selected", callback=merge_selected_sdfs_callback)
                dpg.bind_item_theme("merge_sdf_output_name", apply_bordered_input_text_theme(state))

            # --- STEP 3.2: INITIAL TABLE POPULATION (DEFAULT: INPUT DIR) ---
            state["merge_sdf_folder"] = state["input_dir"]
            dpg.add_text(f"Selected Folder:\n{state['input_dir']}", tag="merge_sdf_label", wrap=600)

            with dpg.group(horizontal=False, tag="merge_sdf_container"):
                state["merge_sdf_folder"] = state["input_dir"]
                state["merge_sdf_selected"] = set()
                sdf_files = sorted((f for f in os.listdir(state["input_dir"]) if f.endswith(".sdf")),key=sort_keys)

                with dpg.table(tag="merge_sdf_table", header_row=False,
                            resizable=False, policy=dpg.mvTable_SizingStretchProp,
                            borders_innerH=False, borders_innerV=False,
                            borders_outerH=False, borders_outerV=False,
                            row_background=False):
                    # Fixed 5-column grid for listings.
                    for _ in range(5):
                        dpg.add_table_column(init_width_or_weight=20)
                    
                    # Fill the table with selectable entries.
                    for row_start in range(0, len(sdf_files), 5):
                        with dpg.table_row(parent="merge_sdf_table"):
                            for i in range(5):
                                idx = row_start + i
                                if idx < len(sdf_files):
                                    filename = sdf_files[idx]
                                    tag = f"merge_sdf_selectable_{idx}"
                                    dpg.add_selectable(label=filename, tag=tag, callback=toggle_sdf_selection, user_data=filename)
                                else:
                                    dpg.add_text("")

            # --- STEP 3.3: POP-UPS ---
            # Conflicting name pop-up.
            with dpg.window(label="File Exists", modal=True, show=False, tag="merge_exists_popup",
                            no_title_bar=False, no_move=True, no_resize=True, no_close=True):
                dpg.add_text("A file with this name already exists in the input folder.")
                dpg.add_button(label="OK", callback=close_merge_popup)

            # Too few files pop-up.
            with dpg.window(label="Too Few Files", modal=True, show=False, tag="merge_too_few_popup",
                            no_title_bar=False, no_move=True, no_resize=True, no_close=True):
                dpg.add_text("Please select at least two SDF files to merge.")
                dpg.add_button(label="OK", callback=lambda: dpg.configure_item("merge_too_few_popup", show=False))

            # Missing output name pop-up.
            with dpg.window(label="Invalid Name", modal=True, show=False, tag="insert_name_popup",
                            no_title_bar=False, no_move=True, no_resize=True, no_close=True):
                dpg.add_text("Please insert a name for the output SDF file.")
                dpg.add_button(label="OK", callback=lambda: dpg.configure_item("insert_name_popup", show=False))

        # --- Spacer and separator before next tool ---
        dpg.add_spacer(height=state["win_spacer"])
        dpg.add_separator()

        # Collapsible section: compare two or more SDFs by SMILES; show totals, uniques, and common unique molecules.
        with dpg.tree_node(label="Compare SDFs", tag="compare_SDFs_tree_node", default_open=False):
            dpg.add_text("Compare two or more SDF files to count common molecules")

            # --- STEP 4.1: INITIALISE STATE FOR COMPARISON ---
            state["compare_sdf_files"] = []

            def sdf_to_smiles_set(sdf_path: str) -> Any:
                """
                Load an SDF file and produce:.
                
                Args:
                    sdf_path (str): Input accepted by this routine.
                
                Returns:
                    Any: Value returned by the routine.
                """
                suppl = Chem.SDMolSupplier(sdf_path)
                smiles_set = set()
                smiles_list = []
                for mol in suppl:
                    if mol is not None:
                        smi = Chem.MolToSmiles(mol)
                        smiles_set.add(smi)
                        smiles_list.append(smi)
                return smiles_set, smiles_list

            def compare_sdf_files_callback() -> None:
                """
                Run the comparison across the currently selected SDF files:.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                draw_loading_screen(state, bg=False)

                files = [f for f in state["compare_sdf_files"] if f is not None]
                results = []
                counters = []

                # Compute per-file stats (total and unique) and store per-file counters.
                for sdf_path in files:
                    smi_set, smi_list = sdf_to_smiles_set(sdf_path)
                    counters.append(Counter(smi_list))
                    results.append((len(smi_list), len(smi_set)))

                # Update per-file totals in UI.
                for i, (total, unique) in enumerate(results):
                    dpg.set_value(f"sdf_total_{i}", str(total))
                    dpg.set_value(f"sdf_unique_{i}", str(unique))

                # Compute intersection of unique SMILES across selected files.
                all_sets = [set(counter.keys()) for counter in counters]

                common_unique = set.intersection(*all_sets) if all_sets else set()
                common_total = sum(min(counter[smi] for counter in counters) for smi in common_unique)
                state["compare_common_smiles"] = sorted(common_unique)

                # Update the summary cells for common molecules.
                dpg.set_value("common_total", str(common_total))
                dpg.set_value("common_unique", str(len(common_unique)))

                # Close overlay if present.
                if dpg.does_item_exist("cover_layer"):
                    dpg.delete_item("cover_layer")

            def show_common_smiles_popup() -> None:
                """
                Display a pop-up listing all common unique SMILES across selected SDFs.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                smiles_list = state.get("compare_common_smiles", [])
                text = "\n".join(smiles_list) if smiles_list else "No common molecules found."

                if dpg.does_item_exist("common_smiles_popup"):
                    dpg.delete_item("common_smiles_popup")

                with dpg.window(label="Common Molecules SMILES (Unique)", tag="common_smiles_popup", modal=True, autosize=True):
                    dpg.add_input_text(multiline=True, readonly=True, width=state["file_dialog_width"], height=state["file_dialog_height"], default_value=text)
                    dpg.bind_item_theme(dpg.last_item(), apply_bordered_input_text_theme(state)) 

            def show_dialog_callback(sender: Any, app_data: Any, user_data: Any) -> None:
                """
                Open the native file dialog for the SDF corresponding to the row index.
                
                Args:
                    sender (Any): Input accepted by this routine.
                    app_data (Any): Input accepted by this routine.
                    user_data (Any): Input accepted by this routine.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                selected_path = open_file_dialog(
                    title="Select an SDF File to Compare",
                    default_path=state["input_dir"],
                    file_types=[("SDF files", "*.sdf"), ("All files", "*.*")],
                )
                if selected_path:
                    select_sdf_callback(selected_path, user_data)

            def select_sdf_callback(path: str, user_data: Any) -> None:
                """
                Update the comparison table after selecting one SDF file.
                
                Args:
                    path (str): Absolute path of the selected SDF file.
                    user_data (Any): Input accepted by this routine.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                idx = user_data["index"]
                filename = os.path.basename(path) 

                if idx == len(state["compare_sdf_files"]):
                    state["compare_sdf_files"].append(path)

                    # Recreate the common row to keep it at the bottom.
                    dpg.delete_item("common_row")

                    with dpg.table_row(parent="compare_sdf_table"):
                        dpg.add_button(label=f"SDF File {idx + 1}", tag=f"sdf_button_{idx}",
                                    callback=show_dialog_callback, user_data={"index": idx})
                        dpg.add_text(f"Dataset {idx + 1}: {filename}", tag=f"sdf_path_text_{idx}")
                        dpg.add_text("", tag=f"sdf_total_{idx}")
                        dpg.add_text("", tag=f"sdf_unique_{idx}")

                    with dpg.table_row(parent="compare_sdf_table", tag="common_row"):
                        dpg.add_text("Common Molecules:")
                        dpg.add_text("")
                        dpg.add_text("", tag="common_total")
                        dpg.add_text("", tag="common_unique")

                    # Prepare the next index.
                    next_idx = idx + 1

                else:
                    state["compare_sdf_files"][idx] = path
                    dpg.set_value(f"sdf_path_text_{idx}", f"Dataset {idx + 1}: {filename}")

                # If all currently present slots are filled, add a new empty row and prepare its dialog.
                if None not in state["compare_sdf_files"] and len(state["compare_sdf_files"]) >= 2:
                    new_idx = len(state["compare_sdf_files"])

                    state["compare_sdf_files"].append(None)

                    # Recreate the common row to place it at the bottom.
                    dpg.delete_item("common_row")

                    with dpg.table_row(parent="compare_sdf_table"):
                        dpg.add_button(label=f"SDF File {new_idx + 1}", tag=f"sdf_button_{new_idx}",
                                    callback=show_dialog_callback, user_data={"index": new_idx})
                        dpg.add_text(f"Dataset {new_idx + 1}: None", tag=f"sdf_path_text_{new_idx}")
                        dpg.add_text("", tag="sdf_total_{new_idx}")
                        dpg.add_text("", tag="sdf_unique_{new_idx}")

                    with dpg.table_row(parent="compare_sdf_table", tag="common_row"):
                        dpg.add_text("Common Molecules:")
                        dpg.add_text("")
                        dpg.add_text("", tag="common_total")
                        dpg.add_text("", tag="common_unique")


            def delete_all_sdf_rows() -> None:
                """
                Reset the comparison table:.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                state["compare_sdf_files"] = []
                
                state["compare_sdf_index"] = 2

                # Recreate the comparison table from scratch.
                if dpg.does_item_exist("compare_sdf_table"):
                    dpg.delete_item("compare_sdf_table")

                with dpg.table(tag="compare_sdf_table", parent="compare_SDFs_tree_node", borders_innerH=True, borders_innerV=True,
                            borders_outerH=True, borders_outerV=True, policy=dpg.mvTable_SizingStretchProp):

                    dpg.add_table_column(tag="compare_sdf_button_col")
                    dpg.add_table_column(label="File Name")
                    dpg.add_table_column(label="All Molecules")
                    dpg.add_table_column(label="Unique Molecules")

                    with dpg.table_row():
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Compare SDF Files", callback=compare_sdf_files_callback)
                            dpg.add_button(label="Show Common SMILES", callback=show_common_smiles_popup)
                            dpg.add_button(label="Reset All", callback=delete_all_sdf_rows)
                        dpg.add_text("")
                        dpg.add_text("")
                        dpg.add_text("")

                    for i in range(2):
                        state["compare_sdf_files"].append(None)
                        with dpg.table_row():
                            dpg.add_button(label=f"SDF File {i + 1}", tag=f"sdf_button_{i}",
                                        callback=show_dialog_callback, user_data={"index": i})
                            dpg.add_text("Dataset {}: None".format(i + 1), tag=f"sdf_path_text_{i}")
                            dpg.add_text("", tag=f"sdf_total_{i}")
                            dpg.add_text("", tag=f"sdf_unique_{i}")

                    with dpg.table_row(tag="common_row"):
                        dpg.add_text("Common Molecules:")
                        dpg.add_text("")
                        dpg.add_text("", tag="common_total")
                        dpg.add_text("", tag="common_unique")

            # --- STEP 4.2: BUILD INITIAL COMPARISON TABLE (TWO ROWS + COMMON SUMMARY) ---
            with dpg.table(tag="compare_sdf_table", borders_innerH=True, borders_innerV=True,
                           borders_outerH=True, borders_outerV=True, policy=dpg.mvTable_SizingStretchProp):

                dpg.add_table_column(tag="compare_sdf_button_col")
                dpg.add_table_column(label="File Name")
                dpg.add_table_column(label="All Molecules")
                dpg.add_table_column(label="Unique Molecules")

                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Compare SDF", callback=compare_sdf_files_callback)
                        dpg.add_button(label="Show Common SMILES", callback=show_common_smiles_popup)
                        dpg.add_button(label="Reset All", callback=delete_all_sdf_rows)
                    dpg.add_text("")
                    dpg.add_text("")
                    dpg.add_text("")

                for i in range(2):
                    state["compare_sdf_files"].append(None)
                    with dpg.table_row():
                        dpg.add_button(label=f"SDF File {i + 1}", tag=f"sdf_button_{i}",
                                       callback=show_dialog_callback, user_data={"index": i})
                        dpg.add_text("Dataset {}: None".format(i + 1), tag=f"sdf_path_text_{i}")
                        dpg.add_text("", tag=f"sdf_total_{i}")
                        dpg.add_text("", tag=f"sdf_unique_{i}")

                with dpg.table_row(tag="common_row"):
                    dpg.add_text("Common Molecules:")
                    dpg.add_text("")
                    dpg.add_text("", tag="common_total")
                    dpg.add_text("", tag="common_unique")

        # --- Spacer and separator before next tool ---
        dpg.add_spacer(height=state["win_spacer"])
        dpg.add_separator()

        # Collapsible section: convert SDF to CSV or CSV to SDF, with auto-detection of CSV delimiter and activity property handling.
        with dpg.tree_node(label="SDF <-> CSV", tag="utilities_tree_node", default_open=False):
            dpg.add_text("Convert SDF files to CSV format, or vice versa")

            # --- STEP 5.1: INITIALISE CONVERTER STATE ---
            state["converter_format"] = "SDF to CSV"
            state["converter_input_path"] = ""

            def select_input_file_callback(input_path: str) -> None:
                """
                Update the selected input path and reflect the file name in the UI.
                
                Args:
                    input_path (str): Absolute path to the selected source file.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                state["converter_input_path"] = input_path
                dpg.set_value("converter_input_name", os.path.basename(input_path))

            def open_converter_input_dialog() -> None:
                """
                Open the native input-file dialog for the current converter mode.

                Args:
                    None.

                Returns:
                    None: This routine updates the converter input path in place.
                """
                file_types = [("All files", "*.*")]
                if state["converter_format"] == "SDF to CSV":
                    file_types = [("SDF files", "*.sdf"), ("All files", "*.*")]
                elif state["converter_format"] == "CSV to SDF":
                    file_types = [("CSV files", "*.csv"), ("All files", "*.*")]

                selected_path = open_file_dialog(
                    title="Select Input File for Conversion",
                    default_path=state["input_dir"],
                    file_types=file_types,
                )
                if selected_path:
                    select_input_file_callback(selected_path)

            def converter_format_changed(sender: Any, app_data: Any, user_data: Any) -> None:
                """
                Handle format mode change:.
                
                Args:
                    sender (Any): Input accepted by this routine.
                    app_data (Any): Input accepted by this routine.
                    user_data (Any): Input accepted by this routine.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                state["converter_format"] = app_data
                state["converter_input_path"] = ""
                dpg.set_value("converter_input_name", "")

            def run_conversion_callback() -> Any:
                """
                Run the conversion based on the current mode:.
                
                Args:
                    None.
                
                Returns:
                    Any: Value returned by the routine.
                """
                input_path = state["converter_input_path"]
                if not input_path:
                    print("No input file selected.")
                    dpg.configure_item("converter_no_input_popup", show=True)
                    return

                output_name = dpg.get_value("converter_output_name")
                if not output_name:
                    print("No output name provided.")
                    dpg.configure_item("converter_no_filename_popup", show=True)
                    return

                if state["converter_format"] == "SDF to CSV":
                    # Ensure CSV extension and path.
                    if not output_name.endswith(".csv"):
                        output_name += ".csv"
                    output_path = os.path.join(state["input_dir"], output_name)

                    # Avoid overwriting an existing file.
                    if os.path.exists(output_path):
                        dpg.configure_item("converter_file_exists_popup", show=True)
                        return

                    # Convert SDF to a DataFrame with SMILES and properties.
                    draw_loading_screen(state, bg=False)
                    suppl = Chem.SDMolSupplier(input_path)
                    data = []

                    for mol in suppl:
                        if mol is None:
                            continue
                        row = {"SMILES": Chem.MolToSmiles(mol)}
                        props = mol.GetPropsAsDict()

                        for key, val in props.items():
                            # Handle Activity properties (Activity, Activity_1, ...): split into Type, Relation, Value, Units.
                            if key.startswith("Activity"):
                                try:
                                    parts = str(val).strip().split()
                                    if len(parts) >= 4:
                                        act_type = parts[0]
                                        act_relation = parts[1].strip("'\"")  # Remove single/double quotes
                                        act_value = parts[2]
                                        act_units = parts[3]

                                        row[act_type] = act_value
                                        row[f"{act_type}_relation"] = act_relation
                                        row[f"{act_type}_units"] = act_units
                                except Exception:
                                    continue
                            else:
                                row[key] = val

                        data.append(row)

                    # If no valid molecules, report and stop.
                    if not data:
                        dpg.set_value("conversion_status_text", "No valid molecules found.")
                        return

                    df = pd.DataFrame(data)

                    # Remove any residual Activity columns (Activity, Activity_1, ...) if present.
                    df = df.drop(columns=[col for col in df.columns if col.startswith("Activity")], errors="ignore")

                    # Save CSV.
                    df.to_csv(output_path, index=False, sep=";")

                    # Close overlay if present.
                    if dpg.does_item_exist("cover_layer"):
                        dpg.delete_item("cover_layer")

                elif state["converter_format"] == "CSV to SDF":
                    # Ensure SDF extension and path.
                    if not output_name.endswith(".sdf"):
                        output_name += ".sdf"
                    output_path = os.path.join(state["input_dir"], output_name)

                    # Avoid overwriting an existing file.
                    if os.path.exists(output_path):
                        dpg.configure_item("converter_file_exists_popup", show=True)
                        return

                    # --- Detect CSV delimiter with csv.Sniffer (fallback to comma) ---
                    with open(input_path, 'r') as f:
                        sniffer = csv.Sniffer()
                        sample = f.read(2048)
                        f.seek(0)
                        try:
                            dialect = sniffer.sniff(sample)
                            sep = dialect.delimiter
                        except csv.Error:
                            sep = ','  # fallback
                    df = pd.read_csv(input_path, sep=sep)

                    # Normalise column names by trimming surrounding whitespace.
                    df.columns = [col.strip() for col in df.columns]

                    # Locate SMILES column (case-insensitive exact match).
                    smiles_col = next((col for col in df.columns if col.strip().lower() == "smiles"), None)
                    if not smiles_col:
                        dpg.configure_item("converter_no_smiles_column_popup", show=True)
                        return

                    # Check whether the standard activity columns are present to optionally merge them into one 'Activity' string.
                    activity_cols = ["Standard Type", "Standard Relation", "Standard Value", "Standard Units"]
                    activity_cols_found = all(col in df.columns for col in activity_cols)

                    if activity_cols_found:
                        def merge_activity(row: Any) -> Any:
                            # Remove any quotes or stray whitespace around the relation
                            """
                            Execute the merge activity routine.
                            
                            Args:
                                row (Any): Input accepted by this routine.
                            
                            Returns:
                                Any: Value returned by the routine.
                            """
                            relation_clean = str(row["Standard Relation"]).strip().strip('"').strip("'")
                            return f"{row['Standard Type']} {relation_clean} {row['Standard Value']} {row['Standard Units']}"
                        df["Activity"] = df.apply(merge_activity, axis=1)
                        exclude_cols = set(activity_cols)  # Skip them when saving as properties
                    else:
                        exclude_cols = set()

                    # Stream rows into an SDWriter, mapping non-empty columns as SD properties.
                    draw_loading_screen(state, bg=False)
                    writer = Chem.SDWriter(output_path)
                    for _, row in df.iterrows():
                        try:
                            mol = Chem.MolFromSmiles(row[smiles_col])
                            if mol is None:
                                continue
                            for col in df.columns:
                                if col == smiles_col:
                                    continue
                                if col in exclude_cols:
                                    continue
                                value = row[col]
                                if pd.isna(value):
                                    continue
                                mol.SetProp(str(col), str(value))
                            writer.write(mol)
                        except Exception:
                            continue
                    writer.close()

                    # Close overlay if present.
                    if dpg.does_item_exist("cover_layer"):
                        dpg.delete_item("cover_layer")

            # --- STEP 5.2: POP-UPS FOR CONVERTER (ERRORS/VALIDATION) ---
            with dpg.window(label="Output File Name Not Valid", tag="converter_file_exists_popup", modal=True, show=False,
                            no_close=True, no_resize=True, no_move=True, autosize=True):
                dpg.add_text("A file with that name already exists")
                dpg.add_spacer(height=state["win_spacer"])
                dpg.add_button(label="OK", callback=lambda: dpg.configure_item("converter_file_exists_popup", show=False))

            with dpg.window(label="Output File Name Not Valid", tag="converter_no_filename_popup", modal=True, show=False,
                            no_close=True, no_resize=True, no_move=True, autosize=True):
                dpg.add_text("Any provided output file name")
                dpg.add_spacer(height=state["win_spacer"])
                dpg.add_button(label="OK", callback=lambda: dpg.configure_item("converter_no_filename_popup", show=False))

            with dpg.window(label="Missing Input File", tag="converter_no_input_popup", modal=True, show=False,
                            no_close=True, no_resize=True, no_move=True, autosize=True):
                dpg.add_text("Any Input File Selected")
                dpg.add_spacer(height=state["win_spacer"])
                dpg.add_button(label="OK", callback=lambda: dpg.configure_item("converter_no_input_popup", show=False))

            with dpg.window(label="Missing SMILES Column", tag="converter_no_smiles_column_popup", modal=True, show=False,
                            no_close=True, no_resize=True, no_move=True, autosize=True):
                dpg.add_text("Any SMILES column found in the selected CSV file")
                dpg.add_spacer(height=state["win_spacer"])
                dpg.add_button(label="OK", callback=lambda: dpg.configure_item("converter_no_smiles_column_popup", show=False))

            # --- STEP 5.3: CONVERTER CONTROL BAR (MODE, INPUT, OUTPUT, RUN) ---
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Conversion:")
                    dpg.add_combo(["SDF to CSV", "CSV to SDF"], default_value="SDF to CSV",
                                callback=converter_format_changed, width=140)
                with dpg.group():
                    dpg.add_text("Input File:")
                    dpg.add_button(label="Select File", callback=open_converter_input_dialog)
                with dpg.group():
                    dpg.add_text("Selected File Name:")
                    dpg.add_input_text(tag="converter_input_name", default_value="", width=200, readonly=True)
                    dpg.bind_item_theme("converter_input_name", apply_bordered_input_text_theme(state))
                with dpg.group():
                    dpg.add_text("Output Name:")
                    dpg.add_input_text(tag="converter_output_name", hint="Output name", width=200)
                    dpg.bind_item_theme("converter_output_name", apply_bordered_input_text_theme(state))
                with dpg.group():
                    dpg.add_text("")
                    dpg.add_button(label="Convert", callback=run_conversion_callback)

        # --- Spacer and separator before next tool ---
        dpg.add_spacer(height=state["win_spacer"])
        dpg.add_separator()

        # Collapsible section: fetch activities for a target via ChEMBL API (paginated), group by document year, show counts.
        with dpg.tree_node(label="ChEMBL Entries by Year", tag="chembl_by_year_tree_node", default_open=False):
            dpg.add_text("Count ChEMBL entries by year for a specific target")

            def fetch_chembl_entries_by_year_callback() -> None:
                """
                Retrieve all activity entries for a given ChEMBL Target ID using the ChEMBL REST API:.
                
                Args:
                    None.
                
                Returns:
                    None: This routine performs in-place updates or side effects only.
                """
                chembl_id = dpg.get_value("chembl_target_id_input").strip().upper()
                if not chembl_id.startswith("CHEMBL"):
                    dpg.set_value("chembl_entries_status", "Please enter a valid ChEMBL Target ID.")
                    return

                base_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={chembl_id}"
                limit = 1000
                offset = 0
                all_molecules = []
                dpg.set_value("chembl_entries_status", f"Fetching entries for {chembl_id}...")

                try:
                    draw_loading_screen(state, bg=False)
                    while True:
                        url = f"{base_url}&limit={limit}&offset={offset}"
                        response = requests.get(url)
                        if not response.ok:
                            dpg.set_value("chembl_entries_status", f"Error fetching data: {response.status_code}")
                            return
                        data = response.json()
                        all_molecules += data["activities"]
                        if offset + limit >= data["page_meta"]["total_count"]:
                            break
                        offset += limit
                        time.sleep(0.1)
                    if dpg.does_item_exist("cover_layer"):
                        dpg.delete_item("cover_layer")

                except Exception as e:
                    dpg.set_value("chembl_entries_status", f"Connection failed: {str(e)}")
                    return

                # Extract years, preserving 'N/A' for entries without a valid integer year.
                years = []
                for entry in all_molecules:
                    year = entry.get("document_year")
                    if isinstance(year, int):
                        years.append(year)
                    else:
                        years.append("N/A")

                year_counts = Counter(years)

                # Recreate the year table with sorted integer years and an 'N/A' row if present.
                if dpg.does_item_exist("chembl_year_table"):
                    dpg.delete_item("chembl_year_table")

                with dpg.table(parent="chembl_table_group", tag="chembl_year_table",
                            borders_innerH=True, borders_innerV=True,
                            borders_outerH=True, borders_outerV=True,
                            policy=dpg.mvTable_SizingStretchProp):
                    dpg.add_table_column(label="Year")
                    dpg.add_table_column(label="Entries")

                    valid_years = sorted([year for year in year_counts if isinstance(year, int)])
                    for year in valid_years:
                        with dpg.table_row():
                            dpg.add_text(str(year))
                            dpg.add_text(str(year_counts[year]))

                    if "N/A" in year_counts:
                        with dpg.table_row():
                            dpg.add_text("N/A")
                            dpg.add_text(str(year_counts["N/A"]))

                dpg.set_value("chembl_entries_status", f"Found {sum(year_counts.values())} entries for {chembl_id}.")
                
            # --- STEP 6.1: INPUTS AND ACTION BUTTON ---
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="chembl_target_id_input", width=400, hint="Insert ChEMBL Target ID (e.g. CHEMBL1234)")
                dpg.add_button(label="Fetch", callback=fetch_chembl_entries_by_year_callback)
                dpg.bind_item_theme("chembl_target_id_input", apply_bordered_input_text_theme(state))

            # --- STEP 6.2: STATUS + RESULTS TABLE CONTAINER ---
            dpg.add_spacer(height=state["win_spacer"])
            dpg.add_text("", tag="chembl_entries_status")
            dpg.add_spacer(height=state["win_spacer"])

            with dpg.group(tag="chembl_table_group"):
                pass
