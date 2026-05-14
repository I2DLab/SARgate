"""
=============================
lmm_finalize.py
=============================

Analysis result export and report generation.

Saves R-group decomposition outputs, scaffold images, and computed statistics
to CSV and PDF files. Ensures organised storage within the results directory
and starts GUI tabs for further analysis.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Save analysis tab windows text
# 3. Save properties dictionaries
# 4. Open overview tab

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import time
import json
import traceback
import dearpygui.dearpygui as dpg
from collections import defaultdict, OrderedDict
from typing import Any
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Crippen, Lipinski
from app.utils.callbacks import change_tab, append_to_log
from app.analysis.similarity.similarity_tanimoto_matrix import (
    show_tanimoto_similarity_manager_window,
)
from app.analysis.similarity.similarity_clustered_matrix import show_clustered_similarity_manager_window
from app.analysis.overview.overview_decomposition import (
    show_subset_choice_window,
    show_molecule_choice_window,
    show_r_groups_choice_window,
    show_properties_windows,
    show_activities_windows,
)
from app.analysis.overview.overview_enrichment_plot import build_enrichment_layout
from app.analysis.overview.overview_table import show_overview_table_window
from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress
from app.analysis.mmpa.mmpa_precompute import run_mmpa_delta_precompute
from app.analysis.r_analysis.counts import show_counts_selection_window
from app.analysis.stereo.stereo_manager import show_isomers_window
from app.analysis.mmpa.mmpa_manager import show_mmpa_window
from app.analysis.prediction.prediction_manager import show_prediction_window
from app.analysis.r_analysis.r_pair_matrix_manager import show_r_pair_matrix_window
from app.analysis.similarity.sal_manager import show_sal_window
from app.analysis.chemspace.chemspace_manager_common import (
    show_dendrogram_window,
    show_descriptors_window,
    show_pca_window,
    show_umap_window,
    show_tsne_window,
)
from app.analysis.tools.sar_notes import (
    notes_popup_window,
    refresh_notes_after_data_change
)
from app.gui.themes_manager import refresh_overrides
from app.utils.app_logger import log_event, log_settings, log_exception, log_traceback


# -----------------------------------------------------------------------------
# 2. Save analysis tab windows text
# -----------------------------------------------------------------------------
def save_analysis_tab_windows_text(state: dict[str, Any]) -> Any:
    """
    Save the texts shown in the three consoles into <work_dir>/results.sof.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    json_path = os.path.join(state["work_dir"], "results.sof")
    if not os.path.exists(json_path):
        json_path = os.path.join(state["work_dir"], "results.srf")

    # read existing JSON (if any)
    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            data = {}

    # build payloads from state logs (default empty lists)
    prep = [{"text": e.get("text", ""), "separator": bool(e.get("separator", False))}
            for e in state.get("prep_log", [])]
    scaff = [{"text": e.get("text", ""), "separator": bool(e.get("separator", False))}
             for e in state.get("scaff_log", [])]
    rga = [{"text": e.get("text", ""), "separator": bool(e.get("separator", False))}
           for e in state.get("rga_log", [])]

    # write/merge
    data["library_preparation_window_text"] = prep
    data["scaffold_analysis_window_text"] = scaff
    data["rga_window_text"] = rga

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return True, f"Saved console texts → {json_path} " \
                 f"(prep={len(prep)}, scaff={len(scaff)}, rga={len(rga)})"
    


# -----------------------------------------------------------------------------
# 3. Save properties dictionaries
# -----------------------------------------------------------------------------
def save_properties_dictionaries(state: dict[str, Any]) -> None:
    """
    Extract and persist molecular properties and activity metadata for downstream analysis.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    state["file_name"] = state["selected_file_name"][:-4]                
    checkbox_states = state["checkbox_states"]                            
    scaffold_dict = state["scaffold_dict"]                                

    dpg.set_value("rgd_progress_bar", 0.90)
    dpg.configure_item("rgd_progress_bar", overlay=f"Precomputing MMPA Deltas – Progress: 90%")
    # Remove scaffold entries whose subset size is below the user-defined threshold.
    for scaffold in list(scaffold_dict.keys()):
        if len(scaffold_dict[scaffold]) < checkbox_states["Filtering threshold"]:
            scaffold_dict.pop(scaffold)

  
    def _build_mol_entry(mol: Any) -> dict[str, Any]:
        mol_entry = {"properties": {}, "activities": {}}
        mol_h = Chem.AddHs(mol, addCoords=True)

        AllChem.ComputeGasteigerCharges(mol)
        charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
        gasteiger_range = max(charges) - min(charges) if charges else 0.0
        gasteiger_mean_abs = sum(abs(q) for q in charges) / len(charges) if charges else 0.0

        mol_entry["properties"] = {
            "original_id": mol.GetIntProp("Mol_ID") if mol.HasProp("Mol_ID") else -1,
            "name": mol.GetProp("_Name") if mol.HasProp("_Name") else "N/A",
            "chembl_id": mol.GetProp("ChEMBL_ID") if mol.HasProp("ChEMBL_ID") else "N/A",
            "pubchem_cid": mol.GetProp("PubChem_CID") if mol.HasProp("PubChem_CID") else "N/A",
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "smiles": mol.GetProp("Smiles") if mol.HasProp("Smiles") and mol.GetProp("Smiles") != "N/A" else Chem.MolToSmiles(mol),
            "molecular_weight": rdMolDescriptors.CalcExactMolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "molar_refractivity": Crippen.MolMR(mol),
            "gasteiger_range": gasteiger_range,
            "gasteiger_mean_abs": gasteiger_mean_abs,
            "tpsa": rdMolDescriptors.CalcTPSA(mol_h),
            "hba": Lipinski.NumHAcceptors(mol_h),
            "hbd": Lipinski.NumHDonors(mol_h),
            "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol_h),
            "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "num_aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "num_saturated_rings": rdMolDescriptors.CalcNumSaturatedRings(mol),
            "kappa1": rdMolDescriptors.CalcKappa1(mol),
            "kappa2": rdMolDescriptors.CalcKappa2(mol),
            "kappa3": rdMolDescriptors.CalcKappa3(mol),
            "chi0": rdMolDescriptors.CalcChi0n(mol),
            "chi1": rdMolDescriptors.CalcChi1n(mol),
            "chi2": rdMolDescriptors.CalcChi2n(mol),
            "chi3": rdMolDescriptors.CalcChi3n(mol),
            "chi4": rdMolDescriptors.CalcChi4n(mol),
        }

        mw = mol_entry["properties"]["molecular_weight"]
        logp = mol_entry["properties"]["logp"]
        hba = mol_entry["properties"]["hba"]
        hbd = mol_entry["properties"]["hbd"]
        rotbonds = mol_entry["properties"]["RotBonds"]
        violations = []
        if mw > 500: violations.append("MW")
        if logp > 5: violations.append("logP")
        if hba > 10: violations.append("HBA")
        if hbd > 5: violations.append("HBD")
        if rotbonds > 10: violations.append("Rotatable Bonds")
        mol_entry["properties"]["Lipinsky's RO5 Violations"] = ", ".join(violations) if violations else "None"

        activity_data = defaultdict(dict)
        shared_props = {}
        activity_prefixes = [
            "Activity", "pValue", "Action_Type", "Action_Description",
            "Assay_Description", "Assay_ChEMBL_ID", "Assay_PubChem_AID", "Assay_Type", "BAO_Label", "BAO_Format",
            "Assay", "Comment", "Target", "Target_ChEMBL_ID", "Target_Uniprot_ID", "Organism", "Max_Phase"
        ]

        for prop_name in mol.GetPropNames():
            val = mol.GetProp(prop_name)
            matched = False
            for prefix in activity_prefixes:
                if prop_name == prefix:
                    shared_props[prop_name] = val
                    matched = True
                    break
                elif prop_name.startswith(prefix + "_"):
                    suffix = prop_name.rsplit("_", 1)[-1]
                    if suffix.isdigit():
                        key = f"activity_{suffix}"
                        activity_data[key][prop_name] = val
                        matched = True
                        break
            if matched:
                continue

        if shared_props and not activity_data:
            activity_data["activity_1"] = shared_props.copy()
        elif shared_props:
            for activity in activity_data.values():
                activity.update(shared_props)

        preferred_order = [
            "Activity", "pValue",
            "Assay", "Assay_Type", "Assay_Description", "Assay_ChEMBL_ID", "Assay_PubChem_AID", "BAO_Label", "BAO_Format",
            "Comment", "Target", "Target_ChEMBL_ID", "Target_Uniprot_ID","Organism", "Action_Type", "Action_Description", "Max_Phase"
        ]

        ordered_activities = {}
        for key, act in activity_data.items():
            ordered = OrderedDict()
            for prefix in preferred_order:
                for k in act:
                    if k.startswith(prefix):
                        ordered[k] = act[k]
            for k in act:
                if k not in ordered:
                    ordered[k] = act[k]
            ordered_activities[key] = ordered

        mol_entry["activities"] = ordered_activities
        return mol_entry

    # Prepare the containers that will hold all per-subset/per-molecule properties and activities.
    properties_dict = {}
    properties_dict_full = {"dataset_full": {}}
    scaff_list = list(scaffold_dict.keys())

  
    for sid, scaffold in enumerate(scaff_list, 1):
        properties_dict[f"subset_{sid}"] = {}                              

        for id, mol in enumerate(scaffold_dict[scaffold], 1):
            properties_dict[f"subset_{sid}"][f"mol_{id}"] = _build_mol_entry(mol)

    prepared_sdf = str(state.get("prepared_sdf", "") or "")
    if prepared_sdf and os.path.isfile(prepared_sdf):
        supplier_full = Chem.SDMolSupplier(prepared_sdf)
        for mol_id, mol in enumerate(supplier_full, 1):
            if mol is None:
                continue
            properties_dict_full["dataset_full"][f"mol_{mol_id}"] = _build_mol_entry(mol)
    else:
        log_event("OVERVIEW", "Prepared dataset not available while building properties_dict_full", indent=1, level="WARNING")

  
    # Persist dictionaries to JSON and launch ΔpValue histogram precomputation.
    state["properties_dict"] = properties_dict
    state["properties_dict_full"] = properties_dict_full
    work_dir = state["work_dir"]
    data = {
        "file_name": state["file_name"],
        "molblocks_rgd_dict": state["molblocks_rgd_dict"],
        "smiles_rgd_dict": state["smiles_rgd_dict"],
        "bioact_types_dict": state["bioact_types_dict"],
        "properties_dict": properties_dict,
        "properties_dict_full": properties_dict_full,
        "total_r_groups_dict": state["total_r_groups_dict"],
        "r_counts": state["r_counts"],
        "settings": checkbox_states
    }

    with open(os.path.join(work_dir, "results.sof"), "w") as f:
        json.dump(data, f, indent=4)

    add_recent_file = state.get("add_recent_file")
    if callable(add_recent_file):
        add_recent_file(work_dir)

    append_to_log(state, f"Properties dictionaries saved as 'results.sof' in the job folder")

    append_to_log(state, "MMPA ΔActivity precomputation started ...")

    run_mmpa_delta_precompute(
        state,
        include_undefined=True,
        include_inactive=False,
        min_delta=0.0,
        bins=50
    )
    append_to_log(state, "MMPA ΔActivity precomputation completed. ΔpValue distribution reports saved in the 'reports' folder")

    save_analysis_tab_windows_text(state)

    dpg.set_value("rgd_progress_bar", 1.0)
    dpg.configure_item("rgd_progress_bar", overlay="Progress: 100%")

# -----------------------------------------------------------------------------
# 4. Open overview tab
# -----------------------------------------------------------------------------
def open_overview_tab(state: dict[str, Any]) -> None:
    """
    Build and initialise all GUI tabs related to overview outputs and auxiliary analyses.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    for tab in ["analysis_tab", "overview_tab", "r_analysis_tab", "similarity_tab",
                "stereo_tab", "mmpa_tab", "chemspace_tab", "prediction_tab"]:
        if dpg.does_item_exist(tab):
            if not dpg.is_item_shown(tab):
                dpg.show_item(tab)
    locked_text_tabs = state.get("locked_text_tabs")
    if isinstance(locked_text_tabs, set):
        for tab in ["analysis_tab", "overview_tab", "r_analysis_tab", "similarity_tab",
                    "stereo_tab", "mmpa_tab", "chemspace_tab", "prediction_tab"]:
            locked_text_tabs.discard(tab)
    locked_text_tab_buttons = state.get("locked_text_tab_buttons")
    if isinstance(locked_text_tab_buttons, set):
        locked_text_tab_buttons.discard("notes_text_button")
    refresh_text_tab_themes = state.get("refresh_text_tab_themes")
    if callable(refresh_text_tab_themes):
        try:
            refresh_text_tab_themes()
        except Exception:
            pass
    for button in [
        "analysis_nav_button",
        "overview_nav_button",
        "r_analysis_nav_button",
        "similarity_nav_button",
        "stereo_nav_button",
        "mmpa_nav_button",
        "chemspace_nav_button",
        "prediction_nav_button",
        "notes_nav_button",
    ]:
        if dpg.does_item_exist(button):
            if not dpg.is_item_shown(button):
                dpg.show_item(button)
            top_nav_button_enabled = state.get("top_nav_button_enabled")
            if isinstance(top_nav_button_enabled, dict):
                top_nav_button_enabled[button] = True
    set_loading_screen_progress(state, 2)

    time.sleep(0.2)
    if state["current_tab"] != "slith_tab":
        draw_loading_screen(state)
        set_loading_screen_progress(state, 4)
    state["first_loading"] = True


    log_event("Overview", "Opening 'OVERVIEW' section'", indent=1)

    try: 
        show_subset_choice_window(state)                             
        set_loading_screen_progress(state, 8)
        show_molecule_choice_window(state)                               
        set_loading_screen_progress(state, 12)
        show_r_groups_choice_window(state)                               
        set_loading_screen_progress(state, 16)
        show_properties_windows(state)                                  
        set_loading_screen_progress(state, 20)
        show_activities_windows(state)                                  
        set_loading_screen_progress(state, 24)
        build_enrichment_layout("subset_1", state)                 
        set_loading_screen_progress(state, 28)
        show_overview_table_window(state)                              
        set_loading_screen_progress(state, 32)
        # manage_similarity_table(state)                                       
    except Exception as e:
        log_exception("Overview", "Error in creating OVERVIEW tab", e, indent=1)
        log_traceback("Overview", indent=2)

    log_event("R-Analysis", "Opening 'R-ANALYSIS' section", indent=1)

    try:
        dpg.configure_item("counts_table", show=False)
        show_counts_selection_window(state)
        set_loading_screen_progress(state, 40)
        show_r_pair_matrix_window(state)
        set_loading_screen_progress(state, 48)
    except Exception as e:
        log_exception("R-Analysis", "Error in creating R-ANALYSIS tab", e, indent=1)
        log_traceback("R-Analysis", indent=2)

    log_event("R-Analysis", "Opening 'SIMILARITY' section", indent=1)
    try:
        show_sal_window(state)
        set_loading_screen_progress(state, 56)
        show_tanimoto_similarity_manager_window(state)
        set_loading_screen_progress(state, 64)
        show_clustered_similarity_manager_window(state)
        set_loading_screen_progress(state, 72)
    except Exception as e:
        log_exception("Similarity", "Error in creating SIMILARITY tab", e, indent=1)
        log_traceback("Similarity", indent=2)

    log_event("Stereo", "Opening 'STEREO' section", indent=1)

    try:
        show_isomers_window(state)
        set_loading_screen_progress(state, 78)
    except Exception as e:
        log_exception("Stereo", "Error in creating STEREO tab", e, indent=1)
        log_traceback("Stereo", indent=2)

    log_event("MMPA", "Opening 'MMPA' section", indent=1)

    try:
        dpg.configure_item("mmpa_table_cont", show=False)
        show_mmpa_window(state)
        set_loading_screen_progress(state, 84)
    except Exception as e:
        log_exception("MMPA", "Error in creating MMPA tab", e, indent=1)
        log_traceback("MMPA", indent=2)
    
    log_event("CHEMSPACE", "Opening 'CHEMSPACE' section", indent=1)

    try:
        show_descriptors_window(state)
        set_loading_screen_progress(state, 88)
        show_dendrogram_window(state)
        set_loading_screen_progress(state, 90)
        show_pca_window(state)
        set_loading_screen_progress(state, 92)
        show_umap_window(state)
        set_loading_screen_progress(state, 94)
        show_tsne_window(state)
        set_loading_screen_progress(state, 96)
    except Exception as e:
        log_exception("ChemSpace", "Error in creating CHEMICAL SPACE tab", e, indent=1)
        log_traceback("ChemSpace", indent=2)

    log_event("Prediction", "Opening 'PREDICTION' section", indent=1)

    try:
        show_prediction_window(state)
        set_loading_screen_progress(state, 98)
    except Exception as e:
        log_exception("Prediction", "Error in creating PREDICTION tab", e, indent=1)
        log_traceback("Prediction", indent=2)
  
    log_event("Notes", "Opening 'SAR NOTES' section", indent=1)

    try:
        notes_popup_window(state)
        refresh_notes_after_data_change(state)
        set_loading_screen_progress(state, 100)
    except Exception as e:
        log_exception("Notes", "Error in refreshing NOTES", e, indent=1)
        log_traceback("Notes", indent=2)

    if dpg.does_item_exist("notes_nav_button"):
        dpg.show_item("notes_nav_button")
        dpg.configure_item("notes_nav_button", enabled=True)
        refresh_overrides(state)

        
    if state["current_tab"] != "slith_tab":
        dpg.set_value("tab_bar", "overview_tab")
        change_tab(state)
        dpg.set_y_scroll("main_window", 0)
    state["first_loading"] = False

    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")

    dpg.set_viewport_title(f"SARgate - {os.path.basename(state['work_dir'])}")

 
