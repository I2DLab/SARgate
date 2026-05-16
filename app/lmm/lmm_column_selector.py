"""
===================
lmm_column_selector.py
===================

Dynamic CSV/TSV/XLSX column selector window.

Builds a Dear PyGui popup that lets the user choose which columns represent
SMILES, activity values, and other relevant data fields in a loaded table file.
Supports both 'ChEMBL-type' and 'single-column' modes, with automatic separator
detection for CSV/TXT e gestione robusta dei file Excel (XLSX).
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Detect separator
# 4. Toggle log flag
# 5. Update log flag
# 6. Show csv column selector
# 7. Confirm csv column selection
# 8. Click cancel button

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
from typing import Any
from app.utils.callbacks import show_library_table_confirm_popup
from app.lmm.lmm_file_reader import _read_excel_robust


# -----------------------------------------------------------------------------
# 2. Detect separator
# -----------------------------------------------------------------------------
def detect_separator(file_path: str) -> Any:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".xlsx":
        return None

    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.readline()

    separators = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t")}
    max_sep = max(separators, key=separators.get)
    if separators[max_sep] > 0:
        return max_sep
    raise ValueError("❌ No valid separator found (only ',', ';', or tab).")


# -----------------------------------------------------------------------------
# 4. Toggle log flag
# -----------------------------------------------------------------------------
def toggle_log_flag(col: Any, state: dict[str, Any]) -> None:
    col_tag = f"chk_{col}"
    log_tag = f"log_flag_{col}"
    is_selected = dpg.get_value(col_tag)
    if dpg.does_item_exist(log_tag):
        dpg.configure_item(log_tag, show=is_selected)

    if is_selected:
        if col not in [c for c, _ in state["activity_columns"]]:
            scale = "log" if dpg.get_value(log_tag) else "lin"
            state["activity_columns"].append((col, scale))
    else:
        state["activity_columns"] = [item for item in state["activity_columns"] if item[0] != col]


# -----------------------------------------------------------------------------
# 5. Update log flag
# -----------------------------------------------------------------------------
def update_log_flag(col: Any, state: dict[str, Any]) -> None:
    for idx, (c, _) in enumerate(state["activity_columns"]):
        if c == col:
            state["activity_columns"][idx] = (col, "log" if dpg.get_value(f"log_flag_{col}") else "lin")
            break


# ------------------------------------------------------------------------------
# Main popup
# ------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 6. Show csv column selector
# -----------------------------------------------------------------------------
def show_csv_column_selector(state: dict[str, Any], file_path: str) -> Any:

    # --- Helpers -------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # 6.1. Is likely logarithmic
    # -----------------------------------------------------------------------------
    def is_likely_logarithmic(values: Any) -> Any:
        try:
            numeric_vals = pd.to_numeric(values.dropna())
            if numeric_vals.empty:
                return False
            return numeric_vals.between(3, 12).mean() > 0.7
        except Exception:
            return False

    # -----------------------------------------------------------------------------
    # 6.2. Update activity checkboxes
    # -----------------------------------------------------------------------------
    def update_activity_checkboxes() -> None:
        """
        Rebuild activity checkboxes according to the selected mode.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        activity_defaults = [
            "IC50", "EC50", "GI50", "Ki", "Kd",
            "pIC50", "pEC50", "pGI50", "pKi", "pKd",
            "Inhibition", "Activation", "Activity"
        ]
        if "log_flags" not in state:
            state["log_flags"] = {}
        if dpg.does_item_exist("csv_activity_group"):
            dpg.delete_item("csv_activity_group", children_only=True)

        # Single-column mode only
        if dpg.get_value("single_checkbox") is True:
            for col in state["csv_columns"]:
                col_tag = f"chk_{col}"
                log_tag = f"log_flag_{col}"
                default_checked = col.strip().lower() in [x.lower() for x in activity_defaults]
                default_log = bool(is_likely_logarithmic(df[col])) if col in df.columns else False

                with dpg.group(horizontal=True, parent="csv_activity_group"):
                    with dpg.group(horizontal=True, width=170):
                        dpg.add_checkbox(tag=col_tag, default_value=default_checked)
                        dpg.add_text(col, wrap=170)
                    dpg.add_spacer(width=10)
                    with dpg.group():
                        dpg.add_checkbox(label="is log scale?", tag=log_tag,
                                         default_value=default_log, show=False)

                dpg.set_item_callback(log_tag, lambda s, a, c=col: update_log_flag(c, state))
                dpg.set_item_callback(col_tag, lambda s, a, c=col: toggle_log_flag(c, state))

                if default_checked:
                    dpg.show_item(log_tag)
                    state["log_flags"][col] = default_log

    # -----------------------------------------------------------------------------
    # 6.3. Set activity mode
    # -----------------------------------------------------------------------------
    def set_activity_mode(mode: str, state: dict[str, Any]) -> None:
        if mode == "":
            dpg.set_value("no_activity_checkbox", True)
            dpg.set_value("chembl_checkbox", False)
            dpg.set_value("single_checkbox", False)
            dpg.set_value("activity_choice_text", "")
            state["activity_mode"] = ""
        elif mode == "single":
            dpg.set_value("no_activity_checkbox", False)
            dpg.set_value("single_checkbox", True)
            dpg.set_value("chembl_checkbox", False)
            dpg.set_value("activity_choice_text", "Select activity columns:\n ")
            state["activity_mode"] = "single"
        elif mode == "chembl":
            dpg.set_value("no_activity_checkbox", False)
            dpg.set_value("single_checkbox", False)
            dpg.set_value("chembl_checkbox", True)
            dpg.set_value("activity_choice_text",
                          "Activity Format:\nStandard Type, Standard Relation,\nStandard Value, Standard Unit:")
            state["activity_mode"] = "chembl"
        update_activity_checkboxes()

    # --- Init state slots ----------------------------------------------------
    for var in [
        "smiles_column", "molname_column", "chembl_id_column", "pubchem_cid_column", "activity_mode",
        "chembl_assay_id_column", "pubchem_aid_column", "assay_desc_column", "assay_type_column",
        "bao_format_column", "bao_label_column",
        "target_chembl_id_column", "target_uniprot_id_column", "target_protein_column",
        "target_organism_column", "action_type_column", "max_phase_column", "comment_column"
    ]:
        state[var] = ""
    state["activity_columns"] = []

    # --- Load CSV/TXT or Excel and extract columns ---
    ext = os.path.splitext(file_path)[1].lower()

    # -----------------------------------------------------------------------------
    # 6.4. Show excel fix popup
    # -----------------------------------------------------------------------------
    def _show_excel_fix_popup(msg: str) -> None:
        if dpg.does_item_exist("excel_fix_popup"):
            dpg.delete_item("excel_fix_popup")
        with dpg.window(
            tag="excel_fix_popup",
            modal=True,
            no_resize=True,
            no_collapse=True,
            no_move=False,
            autosize=False,
            width=760,
            height=420,
            label="Cannot read this Excel file"
        ):
            dpg.add_text(
                "This workbook could not be read automatically.\n"
                "Try one of these fixes:",
                wrap=700,
            )
            dpg.add_spacer(height=10)
            with dpg.child_window(width=-1, height=300, border=True):
                dpg.add_text(
                    "  1) Open it in Excel or LibreOffice\n"
                    "  2) Copy the visible table and Paste Values into a new sheet\n"
                    "  3) Save again as .xlsx, then reload it\n"
                    "  4) If needed, export the sheet as CSV and import that file\n",
                    wrap=700,
                )
                dpg.add_spacer(height=8)
                dpg.add_separator()
                dpg.add_spacer(height=8)
                dpg.add_text("Details:", wrap=700)
                dpg.add_text(str(msg), wrap=700)
            dpg.add_spacer(height=12)
            with dpg.group(horizontal=True):
                dpg.add_button(label="OK", width=80, callback=lambda: dpg.delete_item("excel_fix_popup"))

    try:
        if ext == ".xlsx":
            # Use the robust reader; if no data found, raises ValueError with clear message
            df = _read_excel_robust(file_path)
        else:
            sep = detect_separator(file_path)
            df = pd.read_csv(file_path, sep=sep, engine="python", encoding="utf-8-sig")

    except ValueError as exc:
        # Typical case: "Workbook contains no tabular data..."
        _show_excel_fix_popup(str(exc))
        return
    except Exception as exc:
        # Any other unexpected error
        _show_excel_fix_popup(f"Unexpected error: {exc}")
        return

    state["csv_columns"] = df.columns.tolist()


    # --- UI: popup and tabs --------------------------------------------------
    if dpg.does_item_exist("csv_column_popup"):
        dpg.delete_item("csv_column_popup")

    with dpg.window(label="CSV/XLSX Column Selector", tag="csv_column_popup", autosize=True, modal=True,
                    no_collapse=True, no_close=True, no_resize=True, pos=(330, 50)):

        with dpg.child_window(tag="csv_column_scroll_area", width=-1, height=440, autosize_x=True,
                              horizontal_scrollbar=False, no_scrollbar=True, border=False):

            with dpg.tab_bar():

                # Molecule tab
                with dpg.tab(label="Molecule"):
                    dpg.add_text("Smiles")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["smiles_column"],
                                  tag="smiles_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("Molecule Name:")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["molname_column"],
                                  tag="molname_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("Molecule ChEMBL ID:")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["chembl_id_column"],
                                  tag="chembl_id_combo", width=200)
                    dpg.add_text("Molecule PubChem CID:")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["pubchem_cid_column"],
                                  tag="pubchem_cid_combo", width=200)

                # Activity tab (auto mode)
                chembl_cols = {"Standard Type", "Standard Relation", "Standard Value", "Standard Units"}
                pubchem_cols = {"Activity_Type", "Activity_Qualifier", "Activity_Value"}
                available_lower = {col.strip().lower() for col in state["csv_columns"]}

                if all(col.lower() in available_lower for col in chembl_cols):
                    # ChEMBL “true”
                    state["activity_mode"] = "chembl"
                    state["activity_source"] = "chembl"
                    state["activity_fixed_unit"] = None

                    default_check_no_activity = False
                    default_check_single = False
                    default_check_chembl = True
                    activity_choice_text = "Activity Format:\nStandard Type, Standard Relation,\nStandard Value, Standard Unit:"

                elif all(col.lower() in available_lower for col in pubchem_cols):
                    # PubChem: 3 columns + implicit unit uM
                    state["activity_mode"] = "chembl"
                    state["activity_source"] = "pubchem"
                    state["activity_fixed_unit"] = "uM"

                    default_check_no_activity = False
                    default_check_single = False
                    default_check_chembl = True
                    activity_choice_text = "Activity Format:\nActivity_Type, Activity_Qualifier,\nActivity_Value (uM):"

                elif any(act in available_lower for act in
                        ["ic50", "ec50", "gi50", "ki", "kd",
                        "pic50", "pec50", "pgi50", "pki", "pkd",
                        "inhibition", "activation", "activity"]):
                    # Single-column mode
                    state["activity_mode"] = "single"
                    state["activity_source"] = "single"
                    state["activity_fixed_unit"] = None  # depends on the column itself or undefined

                    default_check_no_activity = False
                    default_check_single = True
                    default_check_chembl = False
                    activity_choice_text = "Select activity columns:\n "

                else:
                    # No activity found
                    state["activity_mode"] = ""
                    state["activity_source"] = None
                    state["activity_fixed_unit"] = None

                    default_check_no_activity = True
                    default_check_single = False
                    default_check_chembl = False
                    activity_choice_text = "No activity column found"

                with dpg.tab(label="Activity"):
                    with dpg.child_window(tag="activity_tab_area", width=-1, height=440, autosize_x=True,
                                          horizontal_scrollbar=False, border=False):
                        dpg.add_text("Select activity format")
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(label="No activity data", tag="no_activity_checkbox",
                                             default_value=default_check_no_activity,
                                             callback=lambda s, a: set_activity_mode("", state))
                            dpg.add_checkbox(label="Single-column", tag="single_checkbox",
                                             default_value=default_check_single,
                                             callback=lambda s, a: set_activity_mode("single", state))
                            dpg.add_checkbox(label="ChEMBL-style", tag="chembl_checkbox",
                                             default_value=default_check_chembl,
                                             callback=lambda s, a: set_activity_mode("chembl", state))
                        dpg.add_spacer(height=10)
                        dpg.add_text(activity_choice_text, tag="activity_choice_text")
                        with dpg.group(tag="csv_activity_group"):
                            pass

                # Assay tab
                with dpg.tab(label="Assay"):
                    dpg.add_text("Assay ChEMBL ID")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["chembl_assay_id_column"],
                                  tag="chembl_assay_id_combo", width=200)
                    dpg.add_text("Assay PubChem AID")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["pubchem_aid_column"],
                                  tag="pubchem_aid_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("Assay Description")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["assay_desc_column"],
                                  tag="assay_desc_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("Assay Type")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["assay_type_column"],
                                  tag="assay_type_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("BAO Format ID")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["bao_format_column"],
                                  tag="bao_format_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("BAO Label")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["bao_label_column"],
                                  tag="bao_label_combo", width=200)

                # Target tab
                with dpg.tab(label="Target"):
                    dpg.add_text("Target ChEMBL ID")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["target_chembl_id_column"],
                                  tag="target_chembl_id_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("Target Uniprot Accession ID")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["target_uniprot_id_column"],
                                  tag="target_uniprot_id_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("Target Name")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["target_protein_column"],
                                  tag="target_protein_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("Target Organism")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["target_organism_column"],
                                  tag="target_organism_combo", width=200)

                # Other tab
                with dpg.tab(label="Other"):
                    dpg.add_text("Action Type")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["action_type_column"],
                                  tag="action_type_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("Molecule Max Phase")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["max_phase_column"],
                                  tag="max_phase_combo", width=200)
                    dpg.add_spacer(height=10)
                    dpg.add_text("Comment")
                    dpg.add_combo([""] + state["csv_columns"], default_value=state["comment_column"],
                                  tag="comment_combo", width=200)

            # Auto-fill sensible defaults
            combo_defaults = {
                "smiles_combo": ["Smiles", "SMILES", "Molecule Smiles", "PUBCHEM_EXT_DATASOURCE_SMILES"],
                "molname_combo": ["Molecule Name", "Compound Name", "Molecule_Name", "Compound_Name"],
                "chembl_id_combo": ["Molecule ChEMBL ID"],
                "pubchem_cid_combo": ["Molecule PubChem CID", "Compound CID", "Compound_CID"],
                "chembl_assay_id_combo": ["Assay ChEMBL ID"],
                "pubchem_aid_combo": ["Assay PubChem AID", "BioAssay_AID"],
                "assay_desc_combo": ["Assay Description", "BioAssay_Name"],
                "assay_type_combo": ["Assay Type"],
                "bao_format_combo": ["BAO Format ID"],
                "bao_label_combo": ["BAO Label"],
                "target_chembl_id_combo": ["Target ChEMBL ID"],
                "target_uniprot_id_combo": ["Target Uniprot ID"],
                "target_protein_combo": ["Target Name", "Target_Name"],
                "target_organism_combo": ["Target Organism"],
                "action_type_combo": ["Action Type"],
                "max_phase_combo": ["Molecule Max Phase"],
                "comment_combo": ["Comment"]
            }
            for tag, labels in combo_defaults.items():
                for col in state["csv_columns"]:
                    for label in labels:
                        if col.strip().lower() == label.strip().lower():
                            dpg.set_value(tag, col)
                            break

        # Init checkboxes UI
        update_activity_checkboxes()

        # Footer buttons
        dpg.add_spacer(height=5)
        dpg.add_separator()
        dpg.add_spacer(height=8)
        with dpg.group(horizontal=True, indent=250):
            dpg.add_button(label="Cancel", callback=lambda: click_cancel_button(state))
            dpg.add_button(label="Confirm", callback=lambda: confirm_csv_column_selection(state))


# -----------------------------------------------------------------------------
# 7. Confirm csv column selection
# -----------------------------------------------------------------------------
def confirm_csv_column_selection(state: dict[str, Any]) -> None:
    for combo in [
        "smiles_combo", "molname_combo", "chembl_id_combo", "pubchem_cid_combo",
        "chembl_assay_id_combo", "pubchem_aid_combo", "assay_desc_combo", "assay_type_combo",
        "bao_format_combo", "bao_label_combo",
        "target_chembl_id_combo", "target_uniprot_id_combo", "target_protein_combo",
        "target_organism_combo", "action_type_combo", "max_phase_combo", "comment_combo"
    ]:
        state[combo.replace("_combo", "_column")] = dpg.get_value(combo)

    if state["smiles_column"] == "":
        return
    
    if state["activity_mode"] == "chembl":
        # Determine which source is used:
        # - ChEMBL true: columns Standard Type/Relation/Value/Units
        # - PubChem: Activity_Type / Activity_Qualifier / Activity_Value (+ fixed unit uM)
        src = state.get("activity_source", "chembl")

        if src == "chembl":
            # Legacy mode: the rest of the code already knows that for ChEMBL
            # it must use Standard Type, Standard Relation, Standard Value, Standard Units.
            state["activity_columns"] = [
                ("Standard Type", "lin"),
                ("Standard Relation", "lin"),
                ("Standard Value", "lin"),
                ("Standard Units", "lin"),
            ]
            state["activity_fixed_unit"] = None

        elif src == "pubchem":
            # PubChem: 3 columns + implicit unit uM
            state["activity_columns"] = [
                ("Activity_Type", "lin"),
                ("Activity_Qualifier", "lin"),
                ("Activity_Value", "lin"),
                ("uM", "lin"),  # piece 4: unit *constant*, not a column
            ]
            state["activity_fixed_unit"] = "uM"

        else:
            # fallback safety
            state["activity_columns"] = [("Standard", "lin")]


    elif state["activity_mode"] == "single":
        results = []
        for col in state.get("csv_columns", []):
            col_tag = f"chk_{col}"
            log_tag = f"log_flag_{col}"
            if dpg.does_item_exist(col_tag) and dpg.get_value(col_tag):
                is_log = dpg.get_value(log_tag) if dpg.does_item_exist(log_tag) else False
                results.append((col, "log" if is_log else "lin"))
        if not results:
            return
        state["activity_columns"] = results

    # Close the column selector popup
    if dpg.does_item_exist("csv_column_popup"):
        dpg.delete_item("csv_column_popup", children_only=False)

    # Create the confirmation popup ON THE NEXT FRAME
    # -----------------------------------------------------------------------------
    # 7.1. Show confirm
    # -----------------------------------------------------------------------------
    def _show_confirm(_: Any = None) -> None:
        show_library_table_confirm_popup(state)

    dpg.set_frame_callback(dpg.get_frame_count() + 1, _show_confirm)


# -----------------------------------------------------------------------------
# 8. Click cancel button
# -----------------------------------------------------------------------------
def click_cancel_button(state: dict[str, Any]) -> None:
    dpg.set_value("file_name_text", "No selected file")
    if dpg.does_item_exist("library_table_window_inner"):
        dpg.delete_item("library_table_window_inner")
        dpg.configure_item("library_overview_table", show=False)
    state["selected_file_path"] = ""
    state["selected_file_name"] = ""
    dpg.delete_item("csv_column_popup")
