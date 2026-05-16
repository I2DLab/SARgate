"""
=================
lmm_rgd.py
=================

R-group decomposition and mapping module.

Executes the R-group decomposition on scaffold-based molecular subsets. 
Identifies substituents, maps attachment points, and stores R-group
structures, enabling downstream frequency and SAR analyses.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. R groups decomposition

import os
import csv
import re
import glob
import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
from collections import Counter
from typing import Any
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor, AllChem, rdRGroupDecomposition, rdMolDescriptors, Crippen, Lipinski
from app.lmm.lmm_gui import update_rga_status
from app.lmm.lmm_abort import confirm_cancellation
from app.utils.callbacks import append_to_log
from app.gui.themes_manager import apply_progress_bar_theme


# -----------------------------------------------------------------------------
# 1.1. Build rgd parameters
# -----------------------------------------------------------------------------
def _build_rgd_parameters(num_mols: int) -> Any:
    """
    Build RDKit R-group decomposition parameters using the conservative defaults.

    Returns:
        Any: Configured RGroupDecompositionParameters instance.
    """
    params = rdRGroupDecomposition.RGroupDecompositionParameters()
    # Matching strategy options
    params.matchingStrategy = rdRGroupDecomposition.RGroupMatching.Greedy
    params.substructMatchParams.numThreads = 0
    # GreedyChunk options
    params.chunkSize = 5
    params.timeout = -1.0
    # Core alignment and scoring options
    params.alignment = rdRGroupDecomposition.RGroupCoreAlignment.NoAlignment
    params.scoreMethod = rdRGroupDecomposition.RGroupScore.Match
    # Hydrogen handling options
    params.removeAllHydrogenRGroups = True ## Remove R-groups that are just hydrogens
    params.removeAllHydrogenRGroupsAndLabels = True ## Remove R-groups that are just hydrogens and their labels
    params.removeHydrogensPostMatch = True ## Remove hydrogens after matching to avoid affecting R-group definitions
    # R-group labeling options
    params.allowMultipleRGroupsOnUnlabelled = False # Allow multiple R-groups to be assigned to the same core attachment point
    params.allowNonTerminalRGroups = True ## Allow R-groups on non-terminal coreatoms
    return params


# -----------------------------------------------------------------------------
# 2. R groups decomposition
# -----------------------------------------------------------------------------
def r_groups_decomposition(state: dict[str, Any]) -> Any:

    scaffold_dict = state["scaffold_dict"] 
    total_subsets = state["scaffold_id"]                    
    subset_dir = state["subset_dir"]                     
    summary_dir = state["summary_dir"]                      

    state["molblocks_rgd_dict"] = {}                       
    state["smiles_rgd_dict"] = {}                           
    state["bioact_types_dict"] = {}                         
    state["r_groups_dict"] = {}                             
    state["total_r_groups_dict"] = {}                       
    state["r_counts"] = {}                                  

    dpg.add_progress_bar(
        tag="rgd_progress_bar",
        parent="rga_window",
        default_value=0.0,
        overlay="0%",
        width=-1,
    )

    # Step 1.1: Initialize progress bar to 0%
    total_mols = sum(len(v) for v in scaffold_dict.values())
    processed_mols = 0

    dpg.set_value("rgd_progress_bar", 0.0)
    dpg.configure_item("rgd_progress_bar", overlay="0%")
    dpg.configure_item("rgd_progress_bar", overlay=f"Decomposing subset 1/{total_subsets} – Progress: 0%")
    dpg.bind_item_theme("rgd_progress_bar", apply_progress_bar_theme(state))

    update_rga_status(f"Decomposing each subset", state, step_id=True)  

    def _parse_core_mol(core_smi: str) -> Any:
        core = Chem.MolFromSmiles(core_smi)
        if core is not None:
            return core
        core = Chem.MolFromSmiles(core_smi, sanitize=False)
        if core is None:
            return None
        try:
            Chem.SanitizeMol(
                core,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
        except Exception:
            pass
        return core

    for sub_id, subset in enumerate(scaffold_dict.values(), start=1):

        # Check for abort signal
        if state.get("abort_analysis", False):
            confirm_cancellation(state)
            return
        
        # Only process unfiltered subsets (those with a number of molecules over the selected threshold)
        if sub_id <= total_subsets:
            append_to_log(state, f"Working on subset {sub_id}: {len(subset)} molecules")  # Log current subset info
            update_rga_status(f"   Subset {sub_id}: decomposing {len(subset)} molecules ...", state, temp=True)  # UI feedback


            mols = [mol for mol in Chem.SDMolSupplier(os.path.join(subset_dir, f'subset_{sub_id}.sdf')) if mol is not None]  
            core_smi = list(scaffold_dict.keys())[sub_id - 1]                                                                
            core = _parse_core_mol(core_smi)
            if core is None:
                append_to_log(state, f"     ❌ Subset {sub_id} skipped: scaffold core could not be parsed")
                update_rga_status(f"   Subset {sub_id}: core parsing failed, skipped", state, temp=True)
                processed_mols += len(subset)
                progress = 0.90 * (processed_mols / total_mols if total_mols else 0.0)
                dpg.set_value("rgd_progress_bar", progress)
                next_subset = min(sub_id + 1, total_subsets)
                dpg.configure_item("rgd_progress_bar", overlay=f"Decomposing subset {next_subset}/{total_subsets} – Progress: {int(progress * 100)}%")
                continue
            
            for mol in mols:
                rdDepictor.Compute2DCoords(mol)                                                                             
            rdDepictor.SetPreferCoordGen(True)                                                                         
            rdDepictor.Compute2DCoords(core)                                                                                


            mols_h = [Chem.AddHs(mol, addCoords=True) for mol in mols]                                                       
            for mol in mols_h:
                for atom in mol.GetAtoms():
                    atom.SetIntProp("SourceAtomIdx", atom.GetIdx())                                                         


            rgd_params = _build_rgd_parameters(len(mols_h))
            RDLogger.DisableLog("rdApp.warning")
            try:
                rgd, fails = rdRGroupDecomposition.RGroupDecompose(
                    [core],
                    mols_h,
                    asSmiles=False,
                    asRows=True,
                    options=rgd_params,
                )
            finally:
                RDLogger.EnableLog("rdApp.warning")

            state["molblocks_rgd_dict"][f"subset_{sub_id}"] = {}                                                             
            state["smiles_rgd_dict"][f"subset_{sub_id}"] = {}                                                             
            state["smiles_fails_dict"][f"subset_{sub_id}"] = {}                                                             
            skipped = 0                                                                                                       


            lbls = set(key for dict in rgd for key in dict)                                                                  
            lbls.discard("Core")                                                                                             
            lbls = sorted(lbls, key=lambda x: int(x[1:]) if x[1:].isdigit() else x)                                         
            state["total_r_groups_dict"][f"subset_{sub_id}"] = lbls                                                          

            for row in rgd:
                for key in ['Core'] + lbls:
                    mol = row.get(key)
                    if mol is None:
                        continue
                    for atom in mol.GetAtoms():
                        if atom.GetAtomicNum() == 0:
                            # Clear RDKit default labels before exporting MolBlocks
                            atom.ClearProp("_MolFileRLabel")
                            atom.ClearProp("dummyLabel")


            def clean_v3000_rgroups(molblock: str) -> str:
                """
                Remove unsupported 'RGROUPS=(1 X)' tags from R# atoms in V3000 MolBlock.

                Args:
                    molblock (str): V3000 MolBlock content.

                Returns:
                    str: Cleaned MolBlock with RGROUPS tags stripped from R# atoms.
                """
                cleaned_lines = []
                for line in molblock.splitlines():
                    if "R#" in line and "RGROUPS=" in line:
                        line = line.split("RGROUPS=")[0].strip()
                    cleaned_lines.append(line)
                return "\n".join(cleaned_lines)

            for i, mol in enumerate(mols):
                if state.get("abort_analysis", False):
                    confirm_cancellation(state)
                    return

                mol_key = f"mol_{i+1}"
                subset_key = f"subset_{sub_id}"

                if i in fails or mol.HasProp("Scaffold_placeholder"):
                    skipped += 1                                                                                            
                else:
                    molblock = Chem.MolToMolBlock(mol, forceV3000=True)                                                   
                    molblock = clean_v3000_rgroups(molblock)                                                           
                    state["molblocks_rgd_dict"][subset_key][mol_key] = {"Mol": molblock}                                  

                    if not mol.HasProp("Scaffold_placeholder"):
                        mol_dict = {}
                        for key, value in rgd[i - skipped].items():                                                          
                            if value is not None:
                                try:
                                    block = Chem.MolToMolBlock(value, forceV3000=True)                                       
                                    block = clean_v3000_rgroups(block)                                                     
                                    mol_dict[key] = block
                                except Exception:
                                    mol_dict[key] = None
                        state["molblocks_rgd_dict"][subset_key][mol_key].update(mol_dict)                                    
                        
            update_rga_status(f"   Subset {sub_id}: {len(subset)} molecules - {len(lbls)} R-groups\n", state)                    
            append_to_log(state, f"     Subset {sub_id} decomposed: {len(subset)} molecules - {len(lbls)} R-groups")          


            lbls = set(key for dict in rgd for key in dict)                                                                 
            lbls.remove('Core')
            lbls = sorted(lbls, key=lambda x: int(x[1:]))                                                                      
            state["total_r_groups_dict"][f"subset_{sub_id}"] = lbls                                                         

            for row in rgd:
                for key in ['Core'] + lbls:
                    try:
                        mol = row[key]
                        if mol is not None:
                            for atom in mol.GetAtoms():
                                if atom.HasProp("dummyLabel"):
                                    atom.SetProp("dummyLabel", "R")                                                          
                    except:
                        continue


            for i, mol in enumerate(mols):
                if state.get("abort_analysis", False):
                    confirm_cancellation(state)
                    return

                if i in fails or mol.HasProp("Scaffold_placeholder"):
                    state["smiles_fails_dict"][f"subset_{sub_id}"][f"mol_{i+1}"] = Chem.MolToSmiles(mol)                     
                    skipped += 1
                else:
                    state["smiles_rgd_dict"][f"subset_{sub_id}"][f"mol_{i+1}"] = {                                            
                        "Mol": Chem.MolToSmiles(mol)
                    }
                    if not mol.HasProp("Scaffold_placeholder"):
                        state["smiles_rgd_dict"][f"subset_{sub_id}"][f"mol_{i+1}"].update({                                    
                            key: Chem.MolToSmiles(value)
                            for key, value in rgd[i - skipped].items()
                            if value is not None
                        })
                    

            bioactivity_counter = Counter()                                                                                    
            targets_set = set()                                                                                               
            organisms_set = set()                                                                                             
            state["bioact_types_dict"][f"subset_{sub_id}"] = {}                                                                

            for mol in mols:
                if state.get("abort_analysis", False):
                    confirm_cancellation(state)
                    return
                if mol is not None:
                    # Scan properties labelled "Activity*" and count activity types
                    bioactivity_props = [prop for prop in mol.GetPropNames() if prop.startswith("Activity")]
                    for prop in bioactivity_props:
                        bioactivity_data = mol.GetProp(prop).split(" ")
                        bioact_type = bioactivity_data[0].strip()
                        bioactivity_counter[bioact_type] += 1

                    # Target fields: Target, Target_1, ... (exclude IDs like Target_ChEMBL_ID)
                    target_props = [prop for prop in mol.GetPropNames() if re.fullmatch(rf"{re.escape('Target')}(_\d+)?", prop)]
                    for prop in target_props:
                        targets_set.add(mol.GetProp(prop))

                    # Organism fields: Organism, Organism_1, ...
                    organism_props = [prop for prop in mol.GetPropNames() if re.fullmatch(r"Organism(_\d+)?", prop)]
                    for prop in organism_props:
                        organisms_set.add(mol.GetProp(prop))

            ordered_bioactivities = [item[0] for item in bioactivity_counter.most_common()]                                   
            state["bioact_types_dict"][f"subset_{sub_id}"]["bioactivities"] = ordered_bioactivities                          
            state["bioact_types_dict"][f"subset_{sub_id}"]["targets"] = list(targets_set)                                     
            state["bioact_types_dict"][f"subset_{sub_id}"]["organisms"] = list(organisms_set)                                


            lbls = set(key for dict in rgd for key in dict)
            lbls.remove('Core')
            lbls = sorted(lbls, key=lambda x: int(x[1:]))                                                                     
            state["total_r_groups_dict"][f"subset_{sub_id}"] = lbls

            subset_r_smiles = {}                                                                                               
            skipped = 0
            for i, _ in enumerate(mols_h):
                subset_r_smiles[f"mol_{i+1}"] = []
                if i not in fails and not mol.HasProp("Scaffold_placeholder"):                                                
                    for r_group in lbls:
                        r_smi = Chem.MolToSmiles(rgd[i - skipped][r_group]) if r_group in rgd[i - skipped] else ""            
                        subset_r_smiles[f"mol_{i+1}"].append(r_smi)
                else:
                    skipped += 1
                    subset_r_smiles[f"mol_{i+1}"].extend(["" for _ in lbls])                                                

            r_counters = {i: Counter() for i in range(len(lbls))}                                                              
            for mol_r_list in subset_r_smiles.values():
                for i, smiles in enumerate(mol_r_list):
                    r_counters[i].update([smiles])
            r_counts_dict = {f"R{i + 1}": dict(counter) for i, counter in r_counters.items()}                             
            state["r_counts"][f"subset_{sub_id}"] = r_counts_dict                                                              


            scaffold_smiles = core_smi                                                                                         
            activity_csv_full = []                                                                                            
            skipped = 0
            sub_id_counter = 1                                                                                            
            for mol_id, mol in enumerate(mols):
                if state.get("abort_analysis", False):
                    confirm_cancellation(state)
                    return
                if mol.HasProp("Scaffold_placeholder"):
                    continue                                                                                                   

                real_mol_id = mol.GetIntProp("Mol_ID") if mol.HasProp("Mol_ID") else -1                                       
                mol_h = Chem.AddHs(mol, addCoords=True)                                                                      

                # Calculate Gasteiger charges (range and mean absolute value)
                AllChem.ComputeGasteigerCharges(mol)
                charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
                gasteiger_range = max(charges) - min(charges)
                gasteiger_mean_abs = sum(abs(q) for q in charges) / len(charges)

                base_entry = {
                    "Mol": Chem.MolToSmiles(mol),                                                                              
                    "MolID": real_mol_id,                                                                                      
                    "Subset": sub_id,                                                                                          
                    "Mol_sub_ID": sub_id_counter,                                                                              
                    "MolName": mol.GetProp("_Name") if mol.HasProp("_Name") else mol.GetProp("Name") if mol.HasProp("Name") else "",
                    "Formula": rdMolDescriptors.CalcMolFormula(mol),                                                            
                    "Substructure": scaffold_smiles,                                                                            
                    "logP": Crippen.MolLogP(mol),                                                                              
                    "GasteigerRange" : gasteiger_range,                                                                        
                    "GasteigerMeanAbs": gasteiger_mean_abs,                                                                     
                    "MW": rdMolDescriptors.CalcExactMolWt(mol),                                                                 
                    "HBA": Lipinski.NumHAcceptors(mol_h),                                                                       
                    "HBD": Lipinski.NumHDonors(mol_h),                                                                          
                    "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol_h),                                                  
                    "TPSA": rdMolDescriptors.CalcTPSA(mol_h),                                                                   
                    "MolarRefractivity": Crippen.MolMR(mol),                                                                    
                    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),                                                    
                    "NumRings": rdMolDescriptors.CalcNumRings(mol),                                                             
                    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),                                             
                    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),                                           
                    "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),                                           
                    "Kappa1": rdMolDescriptors.CalcKappa1(mol),                                                                 
                    "Kappa2": rdMolDescriptors.CalcKappa2(mol),                                                                 
                    "Kappa3": rdMolDescriptors.CalcKappa3(mol),                                                                 
                    "Chi0": rdMolDescriptors.CalcChi0n(mol),                                                                   
                    "Chi1": rdMolDescriptors.CalcChi1n(mol),                                                                   
                    "Chi2": rdMolDescriptors.CalcChi2n(mol),                                                                   
                    "Chi3": rdMolDescriptors.CalcChi3n(mol),                                                                   
                    "Chi4": rdMolDescriptors.CalcChi4n(mol),                                                                   
                }

                descriptors = ["logP", "MW", "HBA", "HBD", "RotBonds", "TPSA", "MolarRefractivity", "GasteigerRange", "GasteigerMeanAbs",
                               "fraction_csp3", "NumRings", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
                               "Kappa1", "Kappa2", "Kappa3", "Chi0", "Chi1", "Chi2", "Chi3", "Chi4"]

                old_dict = rgd[mol_id - skipped]                                                                               
                for lbl in lbls:
                    base_entry[lbl] = Chem.MolToSmiles(old_dict.get(lbl)) if lbl in old_dict else ""                         

                bioactivity_props = [prop for prop in mol.GetPropNames() if prop == "Activity" or prop.startswith("Activity" + "_")]

                values_per_type = {btype: [] for btype in ordered_bioactivities}                                             
                for prop in bioactivity_props:
                    if mol.HasProp(prop):
                        parts = mol.GetProp(prop).split(" ")
                        if len(parts) >= 2:
                            btype = parts[0].strip()
                            relation = parts[1].strip()
                            bval = parts[2] if len(parts) > 2 else ""
                            if btype in values_per_type:
                                if relation == "=":
                                    values_per_type[btype].append(bval)                                                       
                                elif relation in {">", "<", ">=", "<="}:
                                    values_per_type[btype].append(f"{relation}{bval}")                                        

                # Emit one or multiple rows depending on how many activity values exist for the molecule
                if all(len(v) == 0 for v in values_per_type.values()):
                    entry = base_entry.copy()
                    for btype in ordered_bioactivities:
                        entry[btype] = ""
                    activity_csv_full.append(entry)
                else:
                    for btype in ordered_bioactivities:
                        for val in values_per_type[btype]:
                            entry = base_entry.copy()
                            for bt in ordered_bioactivities:
                                entry[bt] = val if bt == btype else ""
                            activity_csv_full.append(entry)

                sub_id_counter += 1                                                                                            

            csv_file = os.path.join(summary_dir, f"subset_{sub_id}_summary.csv")                                               
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["Subset", "MolID", "Mol_sub_ID", "MolName", "Substructure", "Mol", "Formula"] + lbls + descriptors + ordered_bioactivities
                )
                writer.writeheader()                                                                                         
                for activity_row in activity_csv_full:
                    writer.writerow(activity_row)                                                                             

        # Update progress bar 
        subset_size = len(subset)
        processed_mols += subset_size

        progress = 0.90 * (processed_mols / total_mols if total_mols else 0.0)
        dpg.set_value("rgd_progress_bar", progress)
        dpg.configure_item("rgd_progress_bar", overlay=f"Decomposing subset {sub_id + 1}/{total_subsets} – Progress: {int(progress * 100)}%")
        

    append_to_log(state, "R-groups decomposition completed")

    if dpg.does_item_exist("temp_message_2"):
        dpg.delete_item("temp_message_2")                                                                                      


    all_bioactivities = set()                                                                                                  
    all_targets = set()
    all_organisms = set()

    for subset_dict in state["bioact_types_dict"].values():
        all_bioactivities.update(subset_dict.get("bioactivities", []))
        all_targets.update(subset_dict.get("targets", []))
        all_organisms.update(subset_dict.get("organisms", []))

    state["bioact_types_dict"]["Dataset"] = {
        "bioactivities": sorted(all_bioactivities),
        "targets": sorted(all_targets),
        "organisms": sorted(all_organisms)
    }                                                                                                                          


    combined_csv_path = os.path.join(summary_dir, "Dataset_summary.csv")                                                      

    subset_csv_files = sorted(                                                                                               
        glob.glob(os.path.join(summary_dir, "subset_*_summary.csv")),
        key=lambda x: int(os.path.basename(x).split("_")[1])
    )

    combined_rows = []                                                                                                         
    all_columns = set()                                                                                                       


    max_r_index = 0
    for rgroup_list in state["total_r_groups_dict"].values():
        for r in rgroup_list:
            match = re.match(r"R(\d+)", r)
            if match:
                idx = int(match.group(1))
                if idx > max_r_index:
                    max_r_index = idx
    lbls = [f"R{i}" for i in range(1, max_r_index + 1)]                                                                        


    for file_path in subset_csv_files:
        df = pd.read_csv(file_path)                                                                                           
        combined_rows.append(df)
        all_columns.update(df.columns)                                                                                        


    descriptors = ["logP", "MW", "HBA", "HBD", "RotBonds", "TPSA", "MolarRefractivity", "GasteigerRange", "GasteigerMeanAbs",
                   "fraction_csp3", "NumRings", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
                   "Kappa1", "Kappa2", "Kappa3", "Chi0", "Chi1", "Chi2", "Chi3", "Chi4"]                                     
    preferred_order = ["Subset", "MolID", "Mol_sub_ID", "MolName", "Substructure", "Mol", "Formula"] + lbls + descriptors    
    remaining_cols = sorted(all_columns - set(preferred_order))                                                                 
    final_column_order = preferred_order + remaining_cols                                                                       


    combined_df = pd.concat(combined_rows, ignore_index=True).reindex(columns=final_column_order)                          
    combined_df.to_csv(combined_csv_path, index=False)                                                                         
    
    append_to_log(state, "CSV summary reports saved in the 'summary' folder")                                                  
    update_rga_status(f"Decomposition completed for all subsets", state, separator=True)
