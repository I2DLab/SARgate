"""
=========================
lmm_read_local.py
=========================

Local file import and validation module.

Handles the reading of molecular datasets from 
local files in SDF, CSV, TSV, XLSX, SMI or TXT format.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Read sdf
# 3. Create from csv
# 4. Create from smi or txt

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import re
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
from typing import Any
from rdkit import Chem
from rdkit.Chem import SDWriter, AllChem
from app.lmm.lmm_activity_curation import (
    activity_curation,
    calculate_pvalue,
    classify_assay
)
from app.utils.callbacks import append_to_log
from app.lmm.lmm_gui import update_library_preparation_status
from app.lmm.lmm_abort import confirm_cancellation
from app.lmm.lmm_file_reader import (
    _read_text_robust,
    _read_excel_robust
)


# -----------------------------------------------------------------------------
# 2. Read sdf
# -----------------------------------------------------------------------------
def read_sdf(state: dict[str, Any]) -> None:
    """
    Read and preprocess an input SDF file, ensuring UTF-8 encoding and curating.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    input_path = state["selected_file_path"]                        # Path to the original input file
    input_name = state["selected_file_name"]                        # Filename of the original input file
    subset_dir = state["subset_dir"]                                # Directory to store intermediate SDF files
    checkbox_states = state["checkbox_states"]                      # User options from GUI checkboxes

    utf8_sdf = os.path.join(subset_dir, f"{input_name[:-4]}_utf8.sdf")  # Temporary UTF-8 encoded file path
    with open(input_path, "rb") as f:
        raw_data = f.read()                                         # Read raw bytes from original file
    decoded_data = raw_data.decode("latin-1")                       # Decode using latin-1
    with open(utf8_sdf, "w", encoding="utf-8") as f:
        f.write(decoded_data)                                       # Save as UTF-8 encoded file

    suppl = Chem.SDMolSupplier(utf8_sdf)                            # Create RDKit supplier from UTF-8 file
    converted_sdf = os.path.join(subset_dir, input_name)
    writer = Chem.SDWriter(converted_sdf)                           # Writer for the curated SDF
    activity_pattern = re.compile(r"^\s*(.*?)\s*([<>]=?|=)\s*([-+]?\d*\.?\d+)\s*(.*?)\s*$")

    for mol in suppl:
        if mol is None:
            continue  # Skip invalid molecules

        try:
            if mol.HasProp("Activity"):
                activity_value = mol.GetProp("Activity")                # Get raw activity string
                match = activity_pattern.match(activity_value)
                if not match:
                    raise ValueError(f"could not parse activity string: {activity_value!r}")

                bioactivity_type = match.group(1).strip()
                bioactivity_relation = match.group(2).strip()
                bioactivity_value = match.group(3).strip()
                bioactivity_units = match.group(4).strip()

                if (checkbox_states["Enable ambiguous activities"] == False) and bioactivity_relation in ["<", "<=", ">=", ">"]:
                    bioactivity = ""
                    log_activity = ""

                else:
                    bioactivity_type, bioactivity_value, bioactivity_units = activity_curation(
                        bioactivity_type, bioactivity_value, bioactivity_units
                    )
                    bioactivity = f"{bioactivity_type} {bioactivity_relation} {bioactivity_value} {bioactivity_units}"

                    if bioactivity_type in state["nM_activity_types"] and bioactivity_units == "nM":
                        pvalue = calculate_pvalue(bioactivity_type, bioactivity_value, bioactivity_units)

                        # Invert the relational operator for log-transformed values
                        inverted_relation = (
                            ">" if bioactivity_relation == "<" else
                            "<" if bioactivity_relation == ">" else
                            ">=" if bioactivity_relation == "<=" else
                            "<=" if bioactivity_relation == ">=" else
                            "="
                        )
                        log_activity = f"p{bioactivity_type} {inverted_relation} {pvalue}"
                    else:
                        log_activity = ""

                if bioactivity != "":
                    mol.SetProp("Activity", bioactivity)
                    if log_activity:
                        mol.SetProp("pValue", log_activity)
                    elif mol.HasProp("pValue"):
                        mol.ClearProp("pValue")
                else:
                    mol.ClearProp("Activity")
                    if mol.HasProp("pValue"):
                        mol.ClearProp("pValue")

        except Exception as e:
            append_to_log(state, f"Error processing activity for molecule {mol.GetProp('_Name')}: {e}. Skipping the activity curation for this molecule")
            mol.ClearProp("Activity")  # Clear activity if error occurs
            if mol.HasProp("pValue"):
                mol.ClearProp("pValue")
        writer.write(mol)

    writer.close()

    state["output_sdf"] = converted_sdf                             # Save output path in state
    add_recent_file = state.get("add_recent_file")
    if callable(add_recent_file):
        add_recent_file(converted_sdf)

    append_to_log(state, f"Molecules saved in {converted_sdf}")
    try:
        os.remove(utf8_sdf)                                             # Fails on Windows
    except:
        pass


# -----------------------------------------------------------------------------
# 3. Create from csv
# -----------------------------------------------------------------------------
def create_from_csv(state: dict[str, Any]) -> Any:
    """
    Convert a CSV/TSV/XLSX file into an SDF, extracting SMILES and activity data and.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """

    update_library_preparation_status("CONVERTING FILE TO SDF", state, separator=True, step_id=True)

    subset_dir = state["subset_dir"]
    input_name = state["selected_file_name"]
    checkbox_states = state["checkbox_states"]
    csv_file = state["selected_file_path"]
    smiles_col = state.get("smiles_column")
    activity_mode = state.get("activity_mode")
    activity_columns = state.get("activity_columns", [])
    activity_source = state.get("activity_source", "chembl")
    fixed_unit = state.get("activity_fixed_unit", None)

    output_sdf = os.path.join(subset_dir, f"{os.path.splitext(input_name)[0]}.sdf")

    if not smiles_col:
        append_to_log(state, "❌ No SMILES column selected for the input table file.")
        update_library_preparation_status(
            "   Error: no SMILES column selected for the input table file",
            state,
            separator=True,
        )
        confirm_cancellation(state)
        return

    ext = os.path.splitext(csv_file)[1].lower()
    if ext == ".xlsx":
        df = _read_excel_robust(csv_file)
    else:
        df = _read_text_robust(csv_file)

    name_col = state.get("molname_column") or next(
        (col for col in df.columns if str(col).lower() in ("name", "molecule name")), None
    )

    writer = SDWriter(output_sdf)
    molecule_count = 0

    def get_activity_unit(activity_type: str) -> str:
        """
        Return the display unit associated with an activity label.
        """
        if activity_type in state["nM_activity_types"]:
            return "nM"
        if activity_type in state["percent_activities"]:
            return "%"
        if activity_type in state["ug/mL_activities"]:
            return "μg/mL"
        if activity_type in state["uM/min_activities"]:
            return "μM/min"
        return ""

    # -----------------------------------------------------------------------------
    # 3.1. Add common properties
    # -----------------------------------------------------------------------------
    def add_common_properties(mol: Any, row: Any, smiles: str) -> None:
        """
        Set IDs, assay, target and misc metadata onto the molecule.
        
        Args:
            mol (Chem.Mol): Parameter accepted by this routine.
            row (Any): Parameter accepted by this routine.
            smiles (str): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        mol.SetProp("SMILES", str(smiles))
        if name_col and pd.notna(row.get(name_col)):
            mol.SetProp("_Name", str(row[name_col]))

        chembl_col = state.get("chembl_id_column")
        if chembl_col and pd.notna(row.get(chembl_col)):
            mol.SetProp("ChEMBL_ID", str(row[chembl_col]).strip())

        pubchem_col = state.get("pubchem_cid_column")
        if pubchem_col and pd.notna(row.get(pubchem_col)):
            mol.SetProp("PubChem_CID", str(row[pubchem_col]).strip())

        assay_desc_col = state.get("assay_desc_column")
        if assay_desc_col and pd.notna(row.get(assay_desc_col)):
            mol.SetProp("Assay_Description", str(row[assay_desc_col]).strip())

        assay_id_col = state.get("chembl_assay_id_column") or state.get("assay_id_column")
        if assay_id_col and pd.notna(row.get(assay_id_col)):
            mol.SetProp("Assay_ChEMBL_ID", str(row[assay_id_col]).strip())

        assay_type_col = state.get("assay_type_column")
        if assay_type_col and pd.notna(row.get(assay_type_col)):
            mol.SetProp("Assay_Type", str(row[assay_type_col]).strip())

        bao_format_col = state.get("bao_format_column")
        if bao_format_col and pd.notna(row.get(bao_format_col)):
            mol.SetProp("BAO_Format", str(row[bao_format_col]).strip())

        bao_label_col = state.get("bao_label_column")
        if bao_label_col and pd.notna(row.get(bao_label_col)):
            mol.SetProp("BAO_Label", str(row[bao_label_col]).strip())

        if bao_label_col and assay_type_col and pd.notna(row.get(bao_label_col)) and pd.notna(row.get(assay_type_col)):
            assay = classify_assay(str(row[bao_label_col]).strip(), str(row[assay_type_col]).strip())
            mol.SetProp("Assay", assay)

        organism_col = state.get("target_organism_column")
        if organism_col and pd.notna(row.get(organism_col)):
            mol.SetProp("Organism", str(row[organism_col]).strip())

        target_prot_col = state.get("target_protein_column")
        if target_prot_col and pd.notna(row.get(target_prot_col)):
            mol.SetProp("Target", str(row[target_prot_col]).strip())

        target_id_col = state.get("target_chembl_id_column") or state.get("target_id_column")
        if target_id_col and pd.notna(row.get(target_id_col)):
            mol.SetProp("Target_ChEMBL_ID", str(row[target_id_col]).strip())

        action_type_col = state.get("action_type_column")
        if action_type_col and pd.notna(row.get(action_type_col)):
            mol.SetProp("Action_Type", str(row[action_type_col]).strip())

        phase_col = state.get("max_phase_column")
        if phase_col and pd.notna(row.get(phase_col)):
            mol.SetProp("Max_Phase", str(row[phase_col]).strip())

        comment_col = state.get("comment_column")
        if comment_col and pd.notna(row.get(comment_col)):
            mol.SetProp("Comment", str(row[comment_col]).strip())

    columns = list(df.columns)
    total_rows = len(df)
    progress_stride = 100

    for idx, row_values in enumerate(df.itertuples(index=False, name=None), start=1):
        if state.get("abort_analysis", False):
            confirm_cancellation(state)
            return

        if idx == 1 or idx % progress_stride == 0 or idx == total_rows:
            update_library_preparation_status(
                f"Mol {idx}/{total_rows}",
                state,
                separator=False,
                step_id=False,
                temp=True,
            )

        row = dict(zip(columns, row_values))

        smiles = row.get(smiles_col)
        if pd.isna(smiles):
            append_to_log(state, f"⚠️ Skipping row {idx}: missing SMILES")
            continue

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            append_to_log(state, f"⚠️ Skipping invalid SMILES at row {idx}: {smiles}")
            continue

        found_activity = False

        # --- Single-column activity mode ---
        if activity_mode == "single":
            valid_acts = []
            for col, mode in activity_columns:
                if col in row and pd.notna(row[col]):
                    try:
                        if float(row[col]) == 0:
                            continue
                    except Exception:
                        pass
                    valid_acts.append((col, row[col], mode))

            def bioactivity_string(col: Any, val: Any, mode: str) -> Any:
                """
                Execute the bioactivity string routine.
                
                Args:
                    col (Any): Input accepted by this routine.
                    val (Any): Input accepted by this routine.
                    mode (Any): Input accepted by this routine.
                
                Returns:
                    Any: Value returned by the routine.
                """
                try:
                    activity_type = str(col).strip()
                    numeric_value = float(val)

                    if mode == "log" and activity_type in state["dimensionless"]:
                        display_type = activity_type
                        display_value = numeric_value
                    elif mode == "log":
                        display_type = activity_type.lstrip("p")
                        display_value = 10 ** (-numeric_value) * 1e9
                    else:
                        display_type = activity_type
                        display_value = numeric_value

                    unit = get_activity_unit(display_type)
                    if unit:
                        return f"{display_type} = {display_value:.2f} {unit}"
                    return f"{display_type} = {display_value:.2f}"
                except Exception:
                    activity_type = str(col).strip()
                    if mode == "log" and activity_type in state["dimensionless"]:
                        display_type = activity_type
                    elif mode == "log":
                        display_type = activity_type.lstrip("p")
                    else:
                        display_type = activity_type

                    unit = get_activity_unit(display_type)
                    if unit:
                        return f"{display_type} = {val} {unit}"
                    return f"{display_type} = {val}"

            for col, val, mode in valid_acts:
                mol_copy = Chem.Mol(mol)
                add_common_properties(mol_copy, row, smiles)
                mol_copy.SetProp("Activity", bioactivity_string(col, val, mode))

                col_linear = col.lstrip("p") if mode == "log" else col
                if col_linear in state["nM_activity_types"]:
                    try:
                        lin_val = float(val) if mode == "lin" else 10 ** (-float(val)) * 1e9
                        p_val = calculate_pvalue(col_linear, lin_val, "nM")
                        mol_copy.SetProp("pValue", f"p{col_linear} = {p_val}")
                    except Exception as e:
                        append_to_log(state, f"⚠️ Error calculating pValue for {col} at row {idx}: {e}")

                writer.write(mol_copy)
                molecule_count += 1
                found_activity = True

        # --- ChEMBL-like / PubChem activity mode ---
        elif activity_mode == "chembl":
            def clean_field(v: Any) -> Any:
                """
                Execute the clean field routine.
                
                Args:
                    v (Any): Input accepted by this routine.
                
                Returns:
                    Any: Value returned by the routine.
                """
                s = str(v) if not isinstance(v, str) else v
                return re.sub(r"\s+", "", s.strip())

            if activity_source == "chembl":
                keys = ["Standard Type", "Standard Relation", "Standard Value", "Standard Units"]
                if all(k in row and pd.notna(row[k]) for k in keys):
                    std_type = clean_field(row["Standard Type"])
                    relation = clean_field(str(row["Standard Relation"]).replace('"', '').replace("'", ''))
                    std_value = clean_field(row["Standard Value"])
                    std_unit = clean_field(row["Standard Units"])

                    if std_unit.lower() in ["ug mL-1:", "ug.mL-1", "ug ml-1", "ug.ml-1"]:
                        std_unit = "ug/mL"
                    if std_unit in ["None", None]:
                        std_unit = "N/A"

                    mol_copy = Chem.Mol(mol)
                    add_common_properties(mol_copy, row, smiles)

                    if (checkbox_states["Enable ambiguous activities"] is False) and relation in ["<", "<=", ">=", ">"]:
                        pass
                    else:
                        std_type, std_value, std_unit = activity_curation(std_type, std_value, std_unit)
                        mol_copy.SetProp("Activity", f"{std_type} {relation} {std_value} {std_unit}")

                        if std_type in state["nM_activity_types"]:
                            inv = (">" if relation == "<" else "<" if relation == ">" else
                                   ">=" if relation == "<=" else "<=" if relation == ">=" else "=")
                            pV = calculate_pvalue(std_type, std_value, std_unit)
                            mol_copy.SetProp("pValue", f"p{std_type} {inv} {pV}")

                    writer.write(mol_copy)
                    molecule_count += 1
                    found_activity = True

            elif activity_source == "pubchem":
                keys = ["Activity_Type", "Activity_Qualifier", "Activity_Value"]
                if all(k in row and pd.notna(row[k]) for k in keys):
                    std_type = clean_field(row["Activity_Type"])
                    relation = clean_field(str(row["Activity_Qualifier"]).replace('"', '').replace("'", ''))
                    std_value = clean_field(row["Activity_Value"])
                    std_unit = fixed_unit or "uM"

                    mol_copy = Chem.Mol(mol)
                    add_common_properties(mol_copy, row, smiles)

                    if (checkbox_states["Enable ambiguous activities"] is False) and relation in ["<", "<=", ">=", ">"]:
                        pass
                    else:
                        std_type, std_value, std_unit = activity_curation(std_type, std_value, std_unit)
                        mol_copy.SetProp("Activity", f"{std_type} {relation} {std_value} {std_unit}")

                        if std_type in state["nM_activity_types"]:
                            inv = (">" if relation == "<" else "<" if relation == ">" else
                                   ">=" if relation == "<=" else "<=" if relation == ">=" else "=")
                            pV = calculate_pvalue(std_type, std_value, std_unit)
                            mol_copy.SetProp("pValue", f"p{std_type} {inv} {pV}")

                    writer.write(mol_copy)
                    molecule_count += 1
                    found_activity = True

        # --- No activity found → write bare molecule ---
        if not found_activity:
            mol_copy = Chem.Mol(mol)
            add_common_properties(mol_copy, row, smiles)
            writer.write(mol_copy)
            molecule_count += 1

    writer.close()

    if molecule_count == 0:
        try:
            os.remove(output_sdf)
        except Exception:
            pass
        append_to_log(state, "❌ No valid molecules were written to the converted SDF file.")
        update_library_preparation_status(
            "   Error: no valid molecules found in the selected table file",
            state,
            separator=True,
        )
        confirm_cancellation(state)
        return

    state["output_sdf"] = output_sdf
    add_recent_file = state.get("add_recent_file")
    if callable(add_recent_file):
        add_recent_file(output_sdf)
    append_to_log(
        state,
        f"SDF file '{os.path.basename(output_sdf)}' created with {molecule_count} molecules and saved in the 'sdf_files' folder"
    )


# -----------------------------------------------------------------------------
# 4. Create from smi or txt
# -----------------------------------------------------------------------------
def create_from_smi_or_txt(state: dict[str, Any]) -> None:
    """
    Create an SDF dataset from a .smi or .txt file containing SMILES strings.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    subset_dir = state["subset_dir"]
    input_name = state["selected_file_name"]
    input_smi_or_txt = state["selected_file_path"]
    output_sdf = os.path.join(subset_dir, f"{input_name[:-4]}.sdf")

    extension = os.path.splitext(input_smi_or_txt)[1].lower()
    suppl = []

    if extension == ".smi":
        with open(input_smi_or_txt, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else f"Mol_{i}"
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol.SetProp("_Name", name)
                AllChem.Compute2DCoords(mol)
                suppl.append(mol)

    elif extension == ".txt":
        with open(input_smi_or_txt, "r") as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        for i, smiles in enumerate(smiles_list, 1):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol.SetProp("_Name", f"Mol_{i}")
                AllChem.Compute2DCoords(mol)
                suppl.append(mol)

    if suppl:
        writer = Chem.SDWriter(output_sdf)
        for mol in suppl:
            writer.write(mol)
        writer.close()

        state["output_sdf"] = output_sdf  # Save path for later use
        add_recent_file = state.get("add_recent_file")
        if callable(add_recent_file):
            add_recent_file(output_sdf)

        append_to_log(state, f"SDF file '{input_name[:-4]}.sdf' created with {len(suppl)} molecules and saved in the 'sdf_files' folder")
    else:
        append_to_log(state, "❌ Any molecule found")
        confirm_cancellation(state)
