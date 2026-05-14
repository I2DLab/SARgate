"""
========================
lmm_preparation.py
========================

Library preparation and pre-processing module.

Handles the import and cleaning of input molecular datasets (SDF or CSV/TSV/XLSX files),
including salt removal, solvent stripping, duplicate filtering, and scaffold
initialisation. Prepares the dataset for subsequent structural and activity
analyses within SARgate.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Library preparation

import os
import re
from collections import defaultdict, Counter
from typing import Any
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from app.lmm.lmm_gui import (
    update_library_preparation_status,
    show_pie_chart
)
from app.lmm.lmm_abort import confirm_cancellation
from app.utils.callbacks import append_to_log
from app.lmm.lmm_activity_curation import calculate_pvalue


# -----------------------------------------------------------------------------
# 2. Library preparation
# -----------------------------------------------------------------------------
def library_preparation(state: dict[str, Any]) -> Any:
    """
    Prepare the molecular library for downstream analysis.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """

    checkbox_states = state["checkbox_states"]
    input_name = state["selected_file_name"]
    subset_dir = state["subset_dir"]
    input_sdf = state["output_sdf"]

    unprep_suppl = Chem.SDMolSupplier(input_sdf)
    suppl = []
    duplicates_mode = " ".join(str(checkbox_states.get("Duplicates handling", "")).split())
    activity_block_prefixes = [
        "Activity", "pValue", "Action_Type", "Action_Description",
        "Assay_Description", "Assay_ChEMBL_ID", "Assay_PubChem_AID", "Assay_Type", "BAO_Label", "BAO_Format",
        "Assay", "Comment", "Target", "Target_ChEMBL_ID", "Target_Uniprot_ID", "Organism", "Max_Phase",
    ]
    activity_metadata_prefixes = [prefix for prefix in activity_block_prefixes if prefix not in {"Activity", "pValue"}]

    def _prop_suffix(prop_name: str, prefix: str) -> str | None:
        """
        Return the numeric suffix for a grouped property, or "" for unsuffixed items.
        """
        if prop_name == prefix:
            return ""
        if prop_name.startswith(prefix + "_"):
            suffix = prop_name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                return suffix
        return None

    def _prune_activity_blocks(mol: Any, allowed_suffixes: set[str]) -> Any:
        """
        Keep only grouped activity-related properties belonging to the allowed suffixes.
        """
        if mol is None:
            return None
        mol_copy = Chem.Mol(mol)
        for prop_name in list(mol_copy.GetPropNames()):
            matched_suffix = None
            for prefix in activity_block_prefixes:
                suffix = _prop_suffix(prop_name, prefix)
                if suffix is not None:
                    matched_suffix = suffix
                    break
            if matched_suffix is not None and matched_suffix not in allowed_suffixes:
                mol_copy.ClearProp(prop_name)
        return mol_copy

    def _is_activity_grouped_property(prop_name: str) -> bool:
        """
        Return True if the property belongs to an activity-related block.
        """
        for prefix in activity_block_prefixes:
            if _prop_suffix(prop_name, prefix) is not None:
                return True
        return False

    def _collect_consensus_non_group_props(props_list: list[dict[str, Any]]) -> dict[str, str]:
        """
        Keep only non-grouped properties that are identical across all entries.
        """
        if not props_list:
            return {}
        all_keys = set().union(*[props.keys() for props in props_list])
        consensus: dict[str, str] = {}
        for key in all_keys:
            if _is_activity_grouped_property(key):
                continue
            values = []
            for props in props_list:
                val = props.get(key)
                if val is None:
                    continue
                sval = str(val).strip()
                if sval:
                    values.append(sval)
            if values and len(set(values)) == 1:
                consensus[key] = values[0]
        return consensus

    def _extract_activity_blocks_from_props(props: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract normalized activity blocks from one molecule property dictionary.
        """
        shared_group_props: dict[str, str] = {}
        suffixed_group_props: dict[str, dict[str, str]] = defaultdict(dict)
        activity_suffixes: set[str] = set()

        for key, val in props.items():
            if val is None:
                continue
            sval = str(val).strip()
            if not sval:
                continue
            matched_prefix = None
            matched_suffix = None
            for prefix in activity_block_prefixes:
                suffix = _prop_suffix(key, prefix)
                if suffix is not None:
                    matched_prefix = prefix
                    matched_suffix = suffix
                    break
            if matched_prefix is None:
                continue
            if matched_suffix == "":
                shared_group_props[key] = sval
                if matched_prefix in {"Activity", "pValue"}:
                    activity_suffixes.add("")
            else:
                suffixed_group_props[matched_suffix][key] = sval
                if matched_prefix in {"Activity", "pValue"}:
                    activity_suffixes.add(matched_suffix)

        suffixes: list[str] = []
        if "" in activity_suffixes:
            suffixes.append("")
        numeric_suffixes = sorted([s for s in activity_suffixes if s != ""], key=lambda s: int(s))
        suffixes.extend(numeric_suffixes)

        blocks: list[dict[str, Any]] = []
        for suffix in suffixes:
            block_props = {}
            for key, value in shared_group_props.items():
                if _prop_suffix(key, "Activity") == "" or _prop_suffix(key, "pValue") == "":
                    if suffix == "":
                        block_props[key] = value
                else:
                    block_props[key] = value
            if suffix != "":
                block_props.update(suffixed_group_props.get(suffix, {}))
            if suffix == "" and "" in suffixed_group_props:
                block_props.update(suffixed_group_props.get("", {}))
            activity_value = block_props.get("Activity" if suffix == "" else f"Activity_{suffix}")
            if not activity_value and "Activity" in block_props:
                activity_value = block_props["Activity"]
            if activity_value:
                blocks.append({
                    "suffix": suffix,
                    "props": block_props,
                    "activity": activity_value,
                })
        return blocks

    def _extract_activity_type(activity_text: Any) -> str | None:
        """
        Extract the activity type from an activity string such as
        'IC50 = 120 nM' or 'My Activity Type > 3.4 uM'.
        """
        if activity_text is None:
            return None
        text = str(activity_text).strip()
        if not text:
            return None
        match = re.match(r'^\s*(.+?)\s*([<>]=?|=)\s*[-+]?\d*\.?\d+', text)
        if match:
            return match.group(1).strip()
        parts = text.split()
        return parts[0].strip() if parts else None


    ## === STEP 1a: REMOVE DUPLICATES AND AGGREGATE PROPERTIES ===
    if duplicates_mode == "Keep one entry with multiple activities":
        update_library_preparation_status("REMOVING DUPLICATES", state, step_id=True)
        merged = defaultdict(list)
        append_to_log(state, f"{len(unprep_suppl)} molecules before merging identical structures")
        update_library_preparation_status(f"   {len(unprep_suppl)} molecule{'s' if len(unprep_suppl) != 1 else ''} in the library\n", state)

        # Group by canonical SMILES
        for mol in unprep_suppl:
            if mol is None:
                continue
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            props = mol.GetPropsAsDict()
            merged[can_smi].append((mol, props))

        suppl = []
        for mol_list in merged.values():
            ref_mol = Chem.Mol(mol_list[0][0])
            for key in list(ref_mol.GetPropNames()):
                ref_mol.ClearProp(key)

            props_list = [props for _mol, props in mol_list]
            consensus_props = _collect_consensus_non_group_props(props_list)
            for key, value in consensus_props.items():
                ref_mol.SetProp(key, value)

            flattened_blocks: list[dict[str, Any]] = []
            for _mol, props in mol_list:
                flattened_blocks.extend(_extract_activity_blocks_from_props(props))

            for idx, block in enumerate(flattened_blocks, start=1):
                suffix = "" if len(flattened_blocks) == 1 else f"_{idx}"
                for prefix in activity_block_prefixes:
                    source_key = prefix if block["suffix"] == "" else f"{prefix}_{block['suffix']}"
                    if source_key in block["props"]:
                        ref_mol.SetProp(f"{prefix}{suffix}", str(block["props"][source_key]))
                    elif block["suffix"] != "" and prefix in block["props"] and prefix in activity_metadata_prefixes:
                        ref_mol.SetProp(f"{prefix}{suffix}", str(block["props"][prefix]))

            suppl.append(ref_mol)

            
        append_to_log(state, f"{len(suppl)} molecules after merging identical structures")
        if checkbox_states["Input source"] != "Database":
            append_to_log(state, f"The cleaned dataset {input_name} contains {len(suppl)} compounds.")


    ## === STEP 1b: REMOVE DUPLICATES AND KEEP BEST ACTIVITY PER TYPE (WITH MATCHED pValue) ===
    elif duplicates_mode == "Keep one entry with the best activity":
        update_library_preparation_status("REMOVING DUPLICATES (BEST ACTIVITY)", state, step_id=True)
        append_to_log(state, f"{len(unprep_suppl)} molecules before merging identical structures")
        update_library_preparation_status(f"   {len(unprep_suppl)} molecule{'s' if len(unprep_suppl) != 1 else ''} in the library\n", state)

        # Regex for Activity/pValue keys and values
        act_key_re = re.compile(r'^Activity(?:_(\d+))?$')                 # Activity or Activity_2
        pval_key_re = re.compile(r'^(pValue|p_Value)(?:_(\d+))?$')         # pValue / p_Value, with or without suffix
        act_val_re = re.compile(r'^\s*(.+?)\s*([<>]=?|=)\s*([0-9]+(?:\.[0-9]+)?)\s*([^\s]+)\s*$')  # "EC50 = 150 nM"
        pval_val_re = re.compile(r'^\s*([^\s]+)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$')                     # "pEC50 = 6.8239"

        def parse_activity(s: Any) -> Any:
            """
            Parse activity.
            
            Args:
                s (Any): Input accepted by this routine.
            
            Returns:
                Any: Value returned by the routine.
            """
            m = act_val_re.match(str(s))
            if not m:
                return None
            typ, rel, val, unit = m.group(1), m.group(2), float(m.group(3)), m.group(4)
            return typ, rel, val, unit

        def parse_pvalue(s: Any) -> Any:
            """
            Parse pvalue.
            
            Args:
                s (Any): Input accepted by this routine.
            
            Returns:
                Any: Value returned by the routine.
            """
            m = pval_val_re.match(str(s))
            if not m:
                return None
            plabel, pnum = m.group(1), float(m.group(2))
            return plabel, pnum

        # Activity sets: smaller is better / bigger is better
        smaller_better = set(state.get("nM_activity_types", [])) \
            | set(state.get("ug/mL_activities", [])) \
            | set(state.get("uM/min_activities", []))
        bigger_better  = set(state.get("dimensionless", [])) \
            | set(state.get("percent_activities", []))

        merged = defaultdict(list)  # can_smi -> [mol, ...]
        for mol in unprep_suppl:
            if mol is None:
                continue
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            merged[can_smi].append(mol)

        suppl = []
        for mol_list in merged.values():
            ref_mol = Chem.Mol(mol_list[0])

            all_props_list = [m.GetPropsAsDict() for m in mol_list]
            for key in list(ref_mol.GetPropNames()):
                ref_mol.ClearProp(key)
            consensus_props = _collect_consensus_non_group_props(all_props_list)
            for key, value in consensus_props.items():
                ref_mol.SetProp(key, value)

            by_type = defaultdict(list)  # typ -> list of entries with origin block props

            for m in mol_list:
                props = m.GetPropsAsDict()
                for block in _extract_activity_blocks_from_props(props):
                    parsed = parse_activity(block["activity"])
                    if not parsed:
                        continue
                    typ, rel, num, unit = parsed
                    pval_key = "pValue" if block["suffix"] == "" else f"pValue_{block['suffix']}"
                    pnum = None
                    if pval_key in block["props"]:
                        parsed_pval = parse_pvalue(block["props"][pval_key])
                        if parsed_pval:
                            _plabel, pnum = parsed_pval

                    by_type[typ].append({
                        "rel": rel,
                        "val": num,
                        "unit": unit,
                        "pnum": pnum,
                        "block_props": block["props"],
                    })

            # Choose best record per activity type
            chosen_by_type = []  # list of dicts: {"typ","rel","val","unit","pnum","props"}
            for typ, items in by_type.items():
                prefer_small = (typ in smaller_better) or (typ not in smaller_better and typ not in bigger_better)
                chosen = min(items, key=lambda x: x["val"]) if prefer_small else max(items, key=lambda x: x["val"])
                chosen_by_type.append((typ, chosen))

            # Stable order: keep deterministic (alphabetical by type)
            chosen_by_type.sort(key=lambda t: t[0])

            num_types = len(chosen_by_type)

            for i, (typ, ch) in enumerate(chosen_by_type, start=1):
                rel, num, unit, pnum, block_props = ch["rel"], ch["val"], ch["unit"], ch["pnum"], ch["block_props"]

                # Activity string
                act_str = f"{typ} {rel} {num:.6g} {unit}"

                # pValue string (only for nM activities)
                if typ in state.get("nM_activity_types", []):
                    if pnum is None:
                        try:
                            pcalc = calculate_pvalue(typ, num, unit)
                            pval_str = f"p{typ} = {float(pcalc):.4f}"
                        except Exception:
                            pval_str = None
                    else:
                        pval_str = f"p{typ} = {float(pnum):.4f}"
                else:
                    pval_str = None

                # Decide suffixing
                suf = "" if num_types == 1 else f"_{i}"

                # Set Activity / pValue
                ref_mol.SetProp(f"Activity{suf}", act_str)
                if pval_str:
                    ref_mol.SetProp(f"pValue{suf}", pval_str)

                # Copy only the metadata associated with the winning activity block
                for prefix in activity_metadata_prefixes:
                    source_key = prefix
                    if source_key not in block_props:
                        continue
                    out_k = prefix if num_types == 1 else f"{prefix}_{i}"
                    ref_mol.SetProp(out_k, str(block_props[source_key]).strip())
            suppl.append(ref_mol)

        append_to_log(state, f"{len(suppl)} molecules after merging identical structures")
        if checkbox_states["Input source"] != "Database":
            append_to_log(state, f"The cleaned dataset {input_name} contains {len(suppl)} compounds.")
                                    

    ## === STEP 1c: REMOVE DUPLICATES AND KEEP AVERAGE ACTIVITY PER TYPE (RECOMPUTE pValue) ===
    elif duplicates_mode == "Keep one entry with average activities":
        update_library_preparation_status("REMOVING DUPLICATES (AVERAGE ACTIVITIES)", state, step_id=True)
        append_to_log(state, f"{len(unprep_suppl)} molecules before merging identical structures")
        update_library_preparation_status(f"   {len(unprep_suppl)} molecule{'s' if len(unprep_suppl) != 1 else ''} in the library\n", state)

        act_key_re = re.compile(r'^Activity(?:_(\d+))?$')
        pval_key_re = re.compile(r'^(pValue|p_Value)(?:_(\d+))?$')
        act_val_re = re.compile(r'^\s*(.+?)\s*([<>]=?|=)\s*([0-9]+(?:\.[0-9]+)?)\s*([^\s]+)\s*$')
        def parse_activity(s: Any) -> Any:
            """
            Parse activity.
            
            Args:
                s (Any): Input accepted by this routine.
            
            Returns:
                Any: Value returned by the routine.
            """
            m = act_val_re.match(str(s))
            if not m:
                return None
            typ, rel, val, unit = m.group(1), m.group(2), float(m.group(3)), m.group(4)
            return typ, rel, val, unit

        merged = defaultdict(list)
        for mol in unprep_suppl:
            if mol is None:
                continue
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            merged[can_smi].append(mol)

        suppl = []
        for mol_list in merged.values():
            ref_mol = Chem.Mol(mol_list[0])
            all_props_list = [m.GetPropsAsDict() for m in mol_list]
            for key in list(ref_mol.GetPropNames()):
                ref_mol.ClearProp(key)
            consensus_props = _collect_consensus_non_group_props(all_props_list)
            for key, value in consensus_props.items():
                ref_mol.SetProp(key, value)

            by_type_vals = defaultdict(list)   # type -> [(val, unit), ...]
            by_type_unit = {}                  # type -> unit (assumed consistent)
            by_type_blocks = defaultdict(list) # type -> [block_props, ...]

            for props in all_props_list:
                for block in _extract_activity_blocks_from_props(props):
                    parsed = parse_activity(block["activity"])
                    if not parsed:
                        continue
                    typ, _rel, num, unit = parsed
                    by_type_vals[typ].append((num, unit))
                    by_type_unit.setdefault(typ, unit)
                    by_type_blocks[typ].append(block["props"])

            final_acts = []   # list of (act_str, typ, mean_val, unit)
            final_pvals = []  # list of pval_str or None

            for typ in sorted(by_type_vals.keys()):
                items = by_type_vals[typ]
                if not items:
                    continue
                # Arithmetic mean of values (units assumed consistent)
                mean_val = sum(v for v, _u in items) / len(items)
                unit = by_type_unit[typ]
                act_str = f"{typ} = {mean_val:.6g} {unit}"
                final_acts.append((act_str, typ, mean_val, unit))

                # pValue: if loggable, recompute from the averaged value
                if typ in state.get("nM_activity_types", []):
                    try:
                        pcalc = calculate_pvalue(typ, mean_val, unit)
                    except Exception:
                        pcalc = ""
                    pval_str = f"p{typ} = {float(pcalc):.4f}" if (pcalc not in ("", None)) else None
                else:
                    pval_str = None

                final_pvals.append(pval_str)

            num_acts = len(final_acts)
            if num_acts == 1:
                (act_str, typ, _val, _unit), pval_str = final_acts[0], final_pvals[0]
                ref_mol.SetProp("Activity", act_str)
                if pval_str:
                    ref_mol.SetProp("pValue", pval_str)
                block_props_list = by_type_blocks.get(typ, [])
                for prefix in activity_metadata_prefixes:
                    values = []
                    for block_props in block_props_list:
                        sval = str(block_props.get(prefix, "")).strip()
                        if sval:
                            values.append(sval)
                    if values and len(values) == len(block_props_list) and len(set(values)) == 1:
                        ref_mol.SetProp(prefix, values[0])
            else:
                for i, ((act_str, typ, _val, _unit), pval_str) in enumerate(zip(final_acts, final_pvals), start=1):
                    ref_mol.SetProp(f"Activity_{i}", act_str)
                    if pval_str:
                        ref_mol.SetProp(f"pValue_{i}", pval_str)
                    block_props_list = by_type_blocks.get(typ, [])
                    for prefix in activity_metadata_prefixes:
                        values = []
                        for block_props in block_props_list:
                            sval = str(block_props.get(prefix, "")).strip()
                            if sval:
                                values.append(sval)
                        if values and len(values) == len(block_props_list) and len(set(values)) == 1:
                            ref_mol.SetProp(f"{prefix}_{i}", values[0])

            suppl.append(ref_mol)

        append_to_log(state, f"{len(suppl)} molecules after merging identical structures")
        if checkbox_states["Input source"] != "Database":
            append_to_log(state, f"The cleaned dataset {input_name} contains {len(suppl)} compounds.")
                                    
                        
    ## === STEP 1d: KEEP DUPLICATES SEPARATED (NO MERGE) ===
    elif duplicates_mode == "Keep duplicates separated":
        update_library_preparation_status("KEEPING DUPLICATES SEPARATED", state, step_id=True)
        # No aggregation: shallow-copy valid molecules
        suppl = [Chem.Mol(m) for m in unprep_suppl if m is not None]
        append_to_log(state, f"{len(suppl)} molecules (duplicates kept separated)")
        if checkbox_states["Input source"] != "Database":
            append_to_log(state, f"The cleaned dataset {input_name} contains {len(suppl)} compounds.")

    else:
        append_to_log(
            state,
            f"❌ Unknown duplicates handling mode: {checkbox_states.get('Duplicates handling')}. "
            "Library preparation aborted.",
        )
        update_library_preparation_status(
            "   Error: unknown duplicates handling mode",
            state,
            separator=True,
        )
        confirm_cancellation(state)
        return

    if not suppl:
        append_to_log(state, "❌ No valid molecules available after input parsing/preparation.")
        update_library_preparation_status(
            "   Error: no valid molecules available after input parsing/preparation",
            state,
            separator=True,
        )
        confirm_cancellation(state)
        return


    prepared_mols = []
    try:
        for idx, mol in enumerate(suppl, start=1):
            if state.get("abort_analysis", False):
                confirm_cancellation(state)
                return
            
            if mol is None:
                continue

            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if not frags:
                continue

            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            mol.SetIntProp("Mol_ID", idx)
            prepared_mols.append(mol)
    except Exception as e:
        log_exception("LMM", "Error during molecule preparation", e, indent=1)
        append_to_log(state, f"Error during molecule preparation: {e}")
        update_library_preparation_status("   Error during molecule preparation - aborting analysis", state, separator=True)
        state["abort_analysis"] = True
        confirm_cancellation(state)
        return

    prepared_sdf = os.path.join(subset_dir, f"{input_name[:-4]}_prepared.sdf")
    with Chem.SDWriter(prepared_sdf) as writer:
        for mol in prepared_mols:
            if state.get("abort_analysis", False):
                confirm_cancellation(state)
                return
            
            writer.write(mol)
        writer.close()

    suppl = Chem.SDMolSupplier(prepared_sdf)
    state["prepared_sdf"] = prepared_sdf
    append_to_log(state, f"The prepared dataset contains {len(suppl)} compounds")
    update_library_preparation_status(f"   {len(suppl)} unique molecule{'s' if len(suppl) != 1 else ''} after the preparation", state, separator=True)


    filter_mode = state["checkbox_states"].get("Filter by structure similarity", "No filters")
    struct_target = state["checkbox_states"].get("Structure for which to calculate similarity", "Entire molecule")
    scaffold_input = state["checkbox_states"].get("structure_similarity_input", "")    # <--- NESSUN strip()
    threshold = state["checkbox_states"].get("Input structure similarity threshold", 0)

    # Skip entirely if disabled or empty input
    if filter_mode != "No filters" and scaffold_input.strip():
        update_library_preparation_status("FILTERING BY STRUCTURE SIMILARITY", state, step_id=True)
        append_to_log(state, f"Structure similarity enabled ({filter_mode}, {struct_target}), threshold = {threshold}%")
        print(f"\n[StructSim] Mode={filter_mode}, Target={struct_target}, Threshold={threshold}%")

        parsing_attempts = [
            ("SMILES", Chem.MolFromSmiles),
            ("SMARTS", Chem.MolFromSmarts),
            ("InChI", Chem.MolFromInchi),
            ("MolBlock", lambda s: Chem.MolFromMolBlock(s, sanitize=False)),  # <--- sanitize=False !!  
        ]

        scaffold_mol = None
        for fmt, func in parsing_attempts:
            try:
                mol = func(scaffold_input)
                if mol is not None:
                    scaffold_mol = mol
                    append_to_log(state, f"[StructSim] Input parsed as {fmt}")
                    print(f"[StructSim] Input parsed as {fmt}")
                    break
            except Exception:
                continue

        if scaffold_mol is None:
            msg = "[StructSim] ERROR: Invalid input structure. Skipping similarity filter."
            print(msg)
            append_to_log(state, msg)
            update_library_preparation_status("   Invalid input structure - skipping similarity filtering", state, separator=True)
            return

        # sanitize
        try:
            Chem.SanitizeMol(scaffold_mol)
        except Exception as e:
            print(f"[StructSim] Warning: SanitizeMol failed but MolBlock may still be usable: {e}")

        def murcko_generalized(m: Any) -> Any:
            """
            Execute the murcko generalized routine.
            
            Args:
                m (Any): Input accepted by this routine.
            
            Returns:
                Any: Value returned by the routine.
            """
            try:
                core = MurckoScaffold.GetScaffoldForMol(m)
                if core is None:
                    return None
                rw = Chem.RWMol(core)
                for a in rw.GetAtoms():
                    a.SetAtomicNum(6)   # carbon topology only
                return rw.GetMol()
            except Exception:
                return None

        if "generalized" in filter_mode.lower():
            scaffold_proc = murcko_generalized(scaffold_mol)
        else:
            if struct_target == "Molecule's Murcko scaffold":
                try:
                    scaffold_proc = MurckoScaffold.GetScaffoldForMol(scaffold_mol)
                except Exception:
                    scaffold_proc = None
            else:
                scaffold_proc = scaffold_mol

        if scaffold_proc is None:
            msg = "[StructSim] ERROR: Failed to prepare input structure."
            print(msg)
            append_to_log(state, msg)
            update_library_preparation_status("   Failed to process input structure - skipping filter", state, separator=True)
            return

        fp_scaffold = Chem.RDKFingerprint(scaffold_proc)

        filtered = []
        kept = 0

        for mol in suppl:
            try:
                if mol is None:
                    continue

                # --- choose molecule representation
                if struct_target == "Entire molecule":
                    base = mol
                else:
                    try:
                        base = MurckoScaffold.GetScaffoldForMol(mol)
                    except Exception:
                        base = None

                if base is None:
                    continue

                # generalize if needed
                if "generalized" in filter_mode.lower():
                    base = murcko_generalized(base)

                if base is None:
                    continue

                fp = Chem.RDKFingerprint(base)
                sim = DataStructs.TanimotoSimilarity(fp_scaffold, fp) * 100.0

                if sim >= threshold:
                    filtered.append(mol)
                    kept += 1

            except Exception as e:
                print(f"[StructSim] Molecule skipped due to error: {e}")
                append_to_log(state, f"Skipped one molecule during similarity filtering: {e}")
                continue

        print(f"[StructSim] Retained {kept}/{len(suppl)} molecules after filter.")
        append_to_log(state, f"Retained {kept}/{len(suppl)} molecules after structure similarity filtering.")
        update_library_preparation_status(
            f"   {kept} molecule{'s' if kept != 1 else ''} retained after structure similarity filtering",
            state, separator=True
        )

        suppl = filtered

        if len(suppl) == 0:
            msg = "[StructSim] WARNING: No molecules left after structure similarity filtering. Aborting analysis."
            print(msg)
            append_to_log(state, msg)
            update_library_preparation_status("   No molecules left after structure similarity filtering - aborting analysis", state, separator=True)
            state["abort_analysis"] = True
            confirm_cancellation(state)
            return


    update_library_preparation_status("READING THE PROPERTIES", state, step_id=True)

    target_counter = Counter()
    for mol in suppl:
        if state.get("abort_analysis", False):
            confirm_cancellation(state)
            return
        
        if mol is None:
            continue

        for prop_name in mol.GetPropNames():
            if prop_name.startswith("Target") and "ChEMBL_ID" not in prop_name and "Uniprot_ID" not in prop_name:
                target = mol.GetProp(prop_name)
                target_counter.update([target])

    sorted_targets = sorted(target_counter.items(), key=lambda x: x[1], reverse=True)
    total_targets = sum(target_counter.values())

    append_to_log(state, f"The dataset contains {total_targets} target annotations ({len(sorted_targets)} target{'s' if len(sorted_targets) != 1 else ''})")
    for target, value in sorted_targets:
        append_to_log(state, f"     {target} : {value}")

    if sorted_targets:
        update_library_preparation_status(f"   {total_targets} target annotations ({len(sorted_targets)} target{'s' if len(sorted_targets) != 1 else ''})\n", state)
    else:
        update_library_preparation_status("   No targets found in the dataset\n", state, separator=True)

    # Draw pie chart of target categories
    if sorted_targets:
        show_pie_chart(sorted_targets, "target_pie", state)


    activity_counter = Counter()
    for mol in suppl:
        if state.get("abort_analysis", False):
            confirm_cancellation(state)
            return
        
        if mol is None:
            continue

        for prop_name in mol.GetPropNames():
            if prop_name.startswith("Activity"):
                activity = mol.GetProp(prop_name)
                activity_type = _extract_activity_type(activity)
                if activity_type:
                    activity_counter.update([activity_type])

    sorted_activities = sorted(activity_counter.items(), key=lambda x: x[1], reverse=True)
    total_activities = sum(activity_counter.values())

    append_to_log(state, f"The dataset contains {total_activities} activity annotations ({len(sorted_activities)} activity type{'' if len(sorted_activities) == 1 else 's'})")
    for activity, value in sorted_activities:
        append_to_log(state, f"     {activity if activity else 'Undefined'} : {value}")

    if sorted_activities:
        update_library_preparation_status(f"   {total_activities} activity annotations ({len(sorted_activities)} activity type{'s' if len(sorted_activities) != 1 else ''})\n", state)
    else:
        update_library_preparation_status("   No activities found in the dataset\n", state, separator=True)

    # Draw pie chart of activity categories
    if sorted_activities:
        show_pie_chart(sorted_activities, "activity_pie", state)


    assay_counter = Counter()
    for mol in suppl:
        if state.get("abort_analysis", False):
            confirm_cancellation(state)
            return
        
        if mol is None:
            continue

        for prop_name in mol.GetPropNames():
            if prop_name.startswith("Assay") and "ChEMBL_ID" not in prop_name and "PubChem_AID" not in prop_name and "Description" not in prop_name and "Type" not in prop_name:
                assay = mol.GetProp(prop_name)
                assay_counter.update([assay])

    total_assays = sum(assay_counter.values())
    sorted_assays = sorted(assay_counter.items(), key=lambda x: x[1], reverse=True)

    append_to_log(state, f"The dataset contains {total_assays} assay annotations ({len(sorted_assays)} assay type{'s' if len(sorted_assays) != 1 else ''})")
    for assay, value in sorted_assays:
        append_to_log(state, f"     {assay} : {value}")

    if sorted_assays:
        update_library_preparation_status(f"   {total_assays} assay annotations ({len(sorted_assays)} assay type{'s' if len(sorted_assays) != 1 else ''})\n", state)
    else:
        update_library_preparation_status("   No assays found in the dataset\n", state, separator=True)

    # Draw pie chart of assay categories
    if sorted_assays:
        show_pie_chart(sorted_assays, "assay_pie", state)


    # -- Filter by Target --
    if checkbox_states["Filter by target"]:
        update_library_preparation_status("FILTERING MOLECULES BY TARGET", state, step_id=True)
        query1 = checkbox_states["Target query 1"]
        query2 = checkbox_states["Target query 2"]
        filtered_suppl = []
        for mol in suppl:
            if mol is None:
                continue
            matching_suffixes = set()
            for prop in mol.GetPropNames():
                suffix = _prop_suffix(prop, "Target")
                if suffix is None or "ChEMBL_ID" in prop or "Uniprot_ID" in prop:
                    continue
                target_values = [item.strip() for item in mol.GetProp(prop).split(",")]
                if (
                    (query1 is not None and query1 in target_values) or
                    (query2 is not None and query2 in target_values)
                ):
                    matching_suffixes.add(suffix)
            if matching_suffixes:
                filtered_suppl.append(_prune_activity_blocks(mol, matching_suffixes))
        suppl = filtered_suppl
        append_to_log(state, f"Filtering by target: the prepared dataset contains {len(suppl)} molecules matching the target query")
        update_library_preparation_status(f"   {len(suppl)} molecules matching the target query", state, separator=True)

        if len(suppl) == 0:
            msg = "WARNING: No molecules left after target filtering. Aborting analysis."
            print(msg)
            append_to_log(state, msg)
            update_library_preparation_status("   No molecules left after target filtering - aborting analysis", state, separator=True)
            state["abort_analysis"] = True
            confirm_cancellation(state)
            return
        

    # -- Filter by Activity --
    if checkbox_states["Filter by activity"]:
        update_library_preparation_status("FILTERING MOLECULES BY ACTIVITY", state, step_id=True)
        query1 = checkbox_states["Activity query 1"]
        query2 = checkbox_states["Activity query 2"]
        filtered_suppl = []
        for mol in suppl:
            if mol is None:
                continue
            matching_suffixes = set()
            for prop in mol.GetPropNames():
                suffix = _prop_suffix(prop, "Activity")
                if suffix is None:
                    continue
                activity_type = _extract_activity_type(mol.GetProp(prop))
                if (
                    activity_type is not None and
                    (
                        (query1 is not None and query1 == activity_type) or
                        (query2 is not None and query2 == activity_type)
                    )
                ):
                    matching_suffixes.add(suffix)
            if matching_suffixes:
                filtered_suppl.append(_prune_activity_blocks(mol, matching_suffixes))
        suppl = filtered_suppl
        append_to_log(state, f"Filtering by activity: the prepared dataset contains {len(suppl)} molecules matching the activity query")
        update_library_preparation_status(f"   {len(suppl)} molecules matching the activity query\n", state, separator=True)

        if len(suppl) == 0:
            msg = "WARNING: No molecules left after activity filtering. Aborting analysis."
            print(msg)
            append_to_log(state, msg)
            update_library_preparation_status("   No molecules left after activity filtering - aborting analysis", state, separator=True)
            state["abort_analysis"] = True
            confirm_cancellation(state)
            return
        

    # -- Filter by Assay --
    if checkbox_states.get("Filter by assay", False):
        update_library_preparation_status("FILTERING MOLECULES BY ASSAY", state, step_id=True)
        query1 = checkbox_states.get("Assay query 1", "")
        query2 = checkbox_states.get("Assay query 2", "")
        suppl = [
            mol for mol in suppl if mol and any(
                prop.startswith("Assay") and
                "ChEMBL_ID" not in prop and
                "PubChem_AID" not in prop and
                "Description" not in prop and
                "Type" not in prop and (
                    (query1 is not None and query1 in mol.GetProp(prop)) or
                    (query2 is not None and query2 in mol.GetProp(prop))
                )
                for prop in mol.GetPropNames()
            )
        ]
        append_to_log(state, f"Filtering by assay: the prepared dataset contains {len(suppl)} molecules matching the assay query")
        update_library_preparation_status(f"   {len(suppl)} molecules matching the assay query", state, separator=True)

        if len(suppl) == 0:
            msg = "WARNING: No molecules left after assay filtering. Aborting analysis."
            print(msg)
            append_to_log(state, msg)
            update_library_preparation_status("   No molecules left after assay filtering - aborting analysis", state, separator=True)
            state["abort_analysis"] = True
            confirm_cancellation(state)
            return


    state["supplier"] = suppl
from app.utils.app_logger import log_event, log_exception
