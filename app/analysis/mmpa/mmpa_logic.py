"""
==================
mmpa_logic.py
==================

Core logic for Matched Molecular Pairs Analysis (MMPA).

Implements the detection of matched molecular pairs based on structural
differences, calculates ΔActivity values, and prepares the transformation
table used by the MMPA visualisation modules.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Run mmpa analysis

import os
import re
import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
from collections import defaultdict
from typing import Any
from app.utils.app_logger import log_event, log_settings, log_event as _log_event, log_exception
from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress
from app.analysis.mmpa.mmpa_network import build_mmpa_network_map
from app.analysis.mmpa.mmpa_table import draw_mmpa_table


# -----------------------------------------------------------------------------
# 2. Run mmpa analysis
# -----------------------------------------------------------------------------
def run_mmpa_analysis(state: dict[str, Any]) -> Any:
    log_event("MMPA", "Running matched molecular pairs analysis", indent=1)
    # Show light loading overlay
    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)

    for item in dpg.get_all_items():
        alias = dpg.get_item_alias(item)
        if isinstance(alias, str) and alias.startswith("mmpa_hover_handler"):
            dpg.delete_item(item)


    summary_dir = state["summary_dir"]
    subset = dpg.get_value("mmpa_subset_choice")
    activity = dpg.get_value("mmpa_activity_type")
    delta_thresh = dpg.get_value("mmpa_delta_thresh")
    read_undefined = dpg.get_value("mmpa_include_undefined_choice")
    read_inactive = dpg.get_value("mmpa_include_inactive_choice")
    log_settings("MMPA", indent=2, subset=subset, activity=activity, include_undefined=read_undefined, include_inactive=read_inactive, min_delta=delta_thresh)
    
    
    csv_path = os.path.join(summary_dir, f"{subset}_summary.csv")
    if not os.path.exists(csv_path):
        log_event("MMPA", f"CSV file not found: {csv_path}", indent=1, level="ERROR")
        return

    df = pd.read_csv(csv_path)
    set_loading_screen_progress(state, 1.2)
    if activity not in df.columns:
        log_event("MMPA", f"Activity column '{activity}' not found in dataset.", indent=1, level="ERROR")
        return

    df = df[df["Mol"].notna()].copy()
    set_loading_screen_progress(state, 1.4)

    if "Mol_sub_ID" in df.columns:
        df = df.rename(columns={"Mol_sub_ID": "Mol_ID"})
    else:
        df["Mol_ID"] = range(1, len(df) + 1)

    r_group_cols = [c for c in df.columns if re.fullmatch(r"R\d+", c)]
    df = df[["Mol_ID", "Mol", "MolName", activity] + r_group_cols].copy()

    activity_raw = df[activity].astype(str).str.strip()

    mask_defined = activity_raw.str.match(r"^=?\s*\d+(\.\d+)?$", na=False)
    mask_undefined = activity_raw.str.match(r"^[<>]=?\s*\d+(\.\d+)?$", na=False) if read_undefined else pd.Series(False, index=df.index)
    mask_inactive = activity_raw.isin(["", "N/A", "NA", "None", "NONE"]) | df[activity].isna() if read_inactive else pd.Series(False, index=df.index)

    df = df[mask_defined | mask_undefined | mask_inactive].copy()
    set_loading_screen_progress(state, 1.6)

    undefined_mask = activity_raw.str.match(r"^[<>]=?\s*\d+(\.\d+)?$", na=False)
    df["Undefined"] = undefined_mask.loc[df.index]  # aligned with filtered df


    # --- Helper: extract numeric part from activity strings ---
    # -----------------------------------------------------------------------------
    # 2.1. Extract value
    # -----------------------------------------------------------------------------
    def extract_value(s: Any) -> Any:
        """
        Extracts the numeric value from an activity string, ignoring relational operators.

        Args:
            s (str): Activity value string (e.g., '= 10', '< 5.5', '>100').

        Returns:
            float or None: Extracted numeric value as float if found, else None.
        """
        match = re.search(r"([<>=]{1,2})?\s*(\d+\.?\d*)", str(s))
        return float(match.group(2)) if match else None

    df["_activity_numeric"] = df[activity].astype(str).str.strip().apply(extract_value)
    df["pValue"] = np.nan

    # If read_inactive = True assign 0.0 to inactive molecules
    if read_inactive:
        inactive_mask = activity_raw.isin(["", "N/A", "NA", "None", "NONE"]) | df[activity].isna()
        df.loc[inactive_mask, "_activity_numeric"] = 0.0
        df.loc[inactive_mask, "pValue"] = 0.0

    # Calculate pValue based on activity type
    if activity in state["nM_activity_types"]:
        df.loc[df["pValue"].isna(), "pValue"] = df["_activity_numeric"].apply(lambda x: -np.log10(x * 1e-9) if x and x > 0 else 0.0)
    else:
        df.loc[df["pValue"].isna(), "pValue"] = df["_activity_numeric"]
    set_loading_screen_progress(state, 1.8)

    # Y axis label for the network map
    if activity in state["nM_activity_types"]:
        state["mmpa_y_axis_label"] = f"p{activity}"
    else:
        state["mmpa_y_axis_label"] = activity
        
    r_group_cols = [col for col in df.columns if col.startswith("R")]

    agg_funcs = {
        "Mol": "first",                # Keep first SMILES string (should be the same)
        "MolName": "first",            # Keep first name
        "pValue": "mean",              # Average activity
        "Undefined": "any",            # True if any row had Undefined = True
    }

    # Add R-group columns to aggregation as first (assuming consistency per Mol_ID)
    agg_funcs.update({col: "first" for col in r_group_cols})

    df = df.groupby("Mol_ID", as_index=False).agg(agg_funcs)
    if r_group_cols:
        df[r_group_cols] = df[r_group_cols].fillna("").astype(str)

    state["mmpa_dataframe"] = df
    set_loading_screen_progress(state, 2)
    
    # Prepare graph connections store
    mmpa_graph_connections = defaultdict(set)  # dict: Mol_ID -> set of linked Mol_IDs


    # -----------------------------------------------------------------------------
    # 2.2. Format name
    # -----------------------------------------------------------------------------
    def format_name(name: str, mol_id: Any) -> Any:
        """
        Formats the molecule name by combining name and ID, or using only ID if name is missing.

        Args:
            name (str): Molecule name (can be empty or None).
            mol_id (str or int): Molecule identifier.

        Returns:
            str: Formatted label (e.g., 'Aspirin (123)' or 'Mol 123').
        """
        if pd.notna(name) and name:
            return f"{name} ({mol_id})"
        else:
            return f"Mol {mol_id}"
        
    
    results = defaultdict(list)
    total_rows = len(df)
    progress_step = max(1, total_rows // 40) if total_rows else 1
    for i in range(total_rows):
        if i % progress_step == 0 or i == total_rows - 1:
            loop_progress = 2 + (((i + 1) / max(1, total_rows)) * 88)
            set_loading_screen_progress(state, loop_progress)
        for j in range(i + 1, len(df)):
            row1, row2 = df.iloc[i], df.iloc[j]
            val1, val2 = row1["pValue"], row2["pValue"]
            if pd.isna(val1) or pd.isna(val2):
                continue

            delta = abs(val1 - val2)
            if delta < delta_thresh:
                continue

            diffs = [(col, row1[col], row2[col]) for col in r_group_cols if row1[col] != row2[col]]

            if len(diffs) == 1:
                col, r1, r2 = diffs[0]

            elif len(diffs) == 2:
                (col1, frag1_a, frag1_b), (col2, frag2_a, frag2_b) = diffs

                # Check if the fragments are the same (even if swapped)
                if frag1_a == frag2_a and frag1_b == frag2_b or frag1_a == frag2_b and frag1_b == frag2_a:

                    # Define a function to count dummy atoms
                    def count_dummies(smi: Any) -> Any:
                        return smi.count("[*:") >= 2

                    # Apply check to both fragments
                    if count_dummies(frag1_a) and count_dummies(frag1_b):
                        # Consider only the first R group
                        col, r1, r2 = col1, frag1_a, frag1_b
                    else:
                        continue  # Fragments not valid
                else:
                    continue  # Fragments differ in content
                
            elif len(diffs) > 2:
                continue  # More than two R-groups differ, skip this pair
            
            id1, id2 = row1["Mol_ID"], row2["Mol_ID"]

            # Direction: always from lower to higher pValue; record graph connectivity
            try:
                if val1 < val2:
                    n1 = format_name(row1["MolName"], id1)
                    n2 = format_name(row2["MolName"], id2)
                    transform = f"{r1} \u00BB {r2}"
                    mmpa_graph_connections[id1].add(id2)
                    mmpa_graph_connections[id2].add(id1)
                    has_undef = row1.get("Undefined", False) or row2.get("Undefined", False)
                    results[(col, transform)].append((val2 - val1, n1, n2, val1, val2, has_undef))
                else:
                    n1 = format_name(row2["MolName"], id2) 
                    n2 = format_name(row1["MolName"], id1)
                    transform = f"{r2} \u00BB {r1}"
                    mmpa_graph_connections[id1].add(id2)
                    mmpa_graph_connections[id2].add(id1)
                    has_undef = row1.get("Undefined", False) or row2.get("Undefined", False)
                    results[(col, transform)].append((val1 - val2, n1, n2, val1, val2, has_undef))
            except:
                if dpg.does_item_exist("cover_layer"):
                    dpg.delete_item("cover_layer")
                return

    state["mmpa_table_data_raw"] = results
    table_data = []
    set_loading_screen_progress(state, 90)

    total_groups = len(results)
    for group_index, ((r, transform), values) in enumerate(results.items(), start=1):
        group_with_inactive = []
        group_active_only = []

        for v in values:
            p1, p2 = v[3], v[4]
            if p1 == 0.0 or p2 == 0.0:
                group_with_inactive.append(v)
            else:
                group_active_only.append(v)

        # Compose a table row for a group of pairs (inactive vs active-only)
        def build_row(value_list: Any, is_inactive: bool) -> Any:
            """
            Builds a table row from a list of MMPA value tuples.

            Args:
                value_list (list): List of tuples (delta, mol1, mol2, val1, val2, has_undef).
                is_inactive (bool): Whether the row corresponds to inactive molecules.

            Returns:
                list or None: Row data for the MMPA table, or None if input list is empty.
            """
            if not value_list:
                return None
            deltas = [v[0] for v in value_list]
            undef_flags = [v[5] for v in value_list]
            mol_pairs = [f"{v[1]} <> {v[2]}" for v in value_list]
            # mol_pairs = [f"Mol {v[1].split('(')[-1].strip(')')} <> Mol {v[2].split('(')[-1].strip(')')}" for v in value_list]
            mean_delta = round(np.mean(deltas), 4)
            std_delta = round(np.std(deltas), 2)
            count = len(value_list)
            suffix = ""
            if is_inactive:
                suffix += "*"
            if any(undef_flags):
                suffix += "#"
            mean_delta_str = f"{mean_delta}{suffix}"
            return [r, transform, ", ".join(mol_pairs), mean_delta_str, std_delta, count]

        row_inactive = build_row(group_with_inactive, is_inactive=True)
        row_active = build_row(group_active_only, is_inactive=False)

        if row_inactive:
            table_data.append(row_inactive)
        if row_active:
            table_data.append(row_active)
        group_progress = 90 + ((group_index / max(1, total_groups)) * 2)
        set_loading_screen_progress(state, group_progress)
            
    def _mean_delta_sort_value(row: Any) -> float:
        """
        Extract the numeric mean-delta value from a table row for stable default ranking.
        """
        try:
            return float(str(row[3]).replace("*", "").replace("#", ""))
        except Exception:
            return float("-inf")

    table_data = sorted(table_data, key=_mean_delta_sort_value, reverse=True)
    table_data = [[idx, *row] for idx, row in enumerate(table_data, start=1)]

    state["mmpa_table_data"] = table_data
    state["mmpa_row_ids"] = []

    state["mmpa_graph_connections"] = dict(mmpa_graph_connections)
    set_loading_screen_progress(state, 92.2)
    
    mol_smiles_dict = {}
    for _, row in df.iterrows():
        mol_id = int(row["Mol_ID"])
        smiles = str(row["Mol"]).strip()
        if smiles:
            mol_smiles_dict[mol_id] = smiles
    state["mol_smiles_dict"] = mol_smiles_dict
    set_loading_screen_progress(state, 93.2)

    mol_names_dict = {}
    for _, row in df.iterrows():
        mol_id = int(row["Mol_ID"])
        name = str(row["MolName"]).strip()
        if name:
            mol_names_dict[mol_id] = name
    state["mol_names_dict"] = mol_names_dict
    set_loading_screen_progress(state, 94.2)

    mol_activity_dict = {}
    for _, row in df.iterrows():
        mol_id = int(row["Mol_ID"])
        activity_val = str(row["pValue"]).strip()
        if activity_val:
            mol_activity_dict[mol_id] = float(row["pValue"])
    state["mol_activity_dict"] = mol_activity_dict
    set_loading_screen_progress(state, 95)


    set_loading_screen_progress(state, 96)
    build_mmpa_network_map(sender=None, app_data=None, user_data=state)
    set_loading_screen_progress(state, 97)
    draw_mmpa_table(activity, table_data, state)
