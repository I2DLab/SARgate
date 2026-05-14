"""
=================
stereo_logic.py
=================

Stereo analysis logic.
"""

import os
import re
from collections import defaultdict
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
from rdkit import Chem
from rdkit.Chem import inchi

from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress
from app.analysis.stereo.stereo_table import update_isomer_images
from app.utils.app_logger import log_event, log_settings


def run_isomers_analysis(state: dict[str, Any]) -> Any:
    """
    Perform analysis to identify and categorise groups of stereoisomers only.
    """
    log_event("Stereo", "Running stereoisomer analysis", indent=1)
    log_settings("Stereo", indent=2, subset=dpg.get_value("isomers_subset_choice"))
    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)

    for tag in ["stereo_sets_group_list", "isomers_groups_selector_window", "isomers_images_main_window"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=True)
    set_loading_screen_progress(state, 3)

    def extract_activities_by_id(df: Any) -> Any:
        chi4_idx = list(df.columns).index("Chi4")
        activity_columns = df.columns[chi4_idx + 1:]
        mol_activities = defaultdict(list)

        for _, row in df.iterrows():
            mol_id = row[ID_column]
            for activity in activity_columns:
                raw_value = str(row[activity]).strip()
                if raw_value and raw_value != "nan":
                    match = re.match(r"^([<>]=?|=)?\s*([\d.]+)", raw_value)
                    if match:
                        operator = match.group(1) if match.group(1) else "="
                        value = match.group(2)
                        units = "nM" if activity in state["nM_activity_types"] else ""
                        formatted = f"{activity} {operator} {value} {units}"
                        mol_activities[mol_id].append(formatted)

        df_unique = df.drop_duplicates(subset=ID_column).copy()
        df_unique["Activities"] = df_unique[ID_column].map(lambda mid: " | ".join(mol_activities[mid]))
        mol_activity_strings = dict(zip(df_unique[ID_column], df_unique["Activities"]))
        return mol_activity_strings, df_unique

    summary_dir = state["summary_dir"]
    subset = dpg.get_value("isomers_subset_choice")
    csv_path = os.path.join(summary_dir, f"{subset}_summary.csv")
    ID_column = "MolID" if subset == "Dataset" else "Mol_sub_ID"
    state["isomers_id_col"] = ID_column

    df = pd.read_csv(csv_path)
    state["isomers_df_all_mols"] = df.copy()
    set_loading_screen_progress(state, 10)

    mol_activity_strings, df_unique = extract_activities_by_id(df)
    state["isomers_activities"] = mol_activity_strings
    state["isomers_df_unique"] = df_unique
    set_loading_screen_progress(state, 22)

    mol_dict = {}
    total_unique = max(1, len(df_unique))
    for idx, (_, row) in enumerate(df_unique.iterrows(), start=1):
        mol_id = row[ID_column]
        smiles = row["Mol"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol_dict[mol_id] = mol
        if idx % max(1, total_unique // 25) == 0 or idx == total_unique:
            set_loading_screen_progress(state, 22 + ((idx / total_unique) * 33))

    inchi_root_map = {}
    total_mols = max(1, len(mol_dict))
    for idx, (mid, mol) in enumerate(mol_dict.items(), start=1):
        ik = inchi.MolToInchiKey(mol)
        root = ik.split("-")[0]
        inchi_root_map.setdefault(root, []).append(mid)
        if idx % max(1, total_mols // 25) == 0 or idx == total_mols:
            set_loading_screen_progress(state, 55 + ((idx / total_mols) * 40))

    stereo_groups = [grp for grp in inchi_root_map.values() if len(grp) > 1]
    set_loading_screen_progress(state, 95)

    delta_stereo = {}
    gold_stereo = {}
    expanded_stereo = {}
    total_groups = max(1, len(stereo_groups))
    for idx, mol_ids in enumerate(stereo_groups, start=1):
        sorted_ids = sorted(mol_ids, key=lambda mid: get_activity_nm(mid, state))
        delta = get_activity_str(sorted_ids, state)
        gold = get_activity_fold_ratio(sorted_ids, state)
        delta_stereo[sorted_ids[0]] = delta
        gold_stereo[sorted_ids[0]] = gold
        for mol_id in sorted_ids:
            expanded_stereo[mol_id] = sorted_ids
        if idx % max(1, total_groups // 20) == 0 or idx == total_groups:
            set_loading_screen_progress(state, 95 + ((idx / total_groups) * 1.5))

    sorted_stereo = dict(sorted(expanded_stereo.items()))
    state["expanded_stereo"] = dict(expanded_stereo)
    state["collapsed_stereo"] = {sorted_ids[0]: sorted_ids for sorted_ids in set(tuple(v) for v in expanded_stereo.values())}
    state["delta_stereo"] = delta_stereo
    state["gold_stereo"] = gold_stereo
    set_loading_screen_progress(state, 97)

    sort_mode = dpg.get_value("stereisomers_sorting_mode_combo") or "Numeric order"
    combo_stereo_groups = build_combo_from_groups(
        sorted_stereo,
        delta_dict=delta_stereo,
        gold_dict=gold_stereo,
        sort_mode=sort_mode,
    )
    set_loading_screen_progress(state, 98)

    if dpg.does_item_exist("stereo_groups_choice"):
        dpg.configure_item(
            "stereo_groups_choice",
            items=combo_stereo_groups,
            default_value=combo_stereo_groups[0] if combo_stereo_groups else None,
            user_data=state,
        )
    set_loading_screen_progress(state, 98.5)

    with dpg.table(tag="stereo_images_table", parent="isomers_images_main_window",
                   header_row=False, width=-1, height=-1,
                   resizable=False, context_menu_in_body=True,
                   borders_innerH=False, borders_outerH=False,
                   borders_innerV=False, borders_outerV=False,
                   policy=dpg.mvTable_SizingStretchSame):
        for idx in range(4):
            dpg.add_table_column(
                label="STEREOISOMERS" if idx == 0 else "",
                tag=f"col_stereo_{idx+1}",
                init_width_or_weight=25
            )
    set_loading_screen_progress(state, 99)

    update_isomer_images(None, None, state)

    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")


def get_activity_nm(mol_id: Any, state: dict[str, Any]) -> Any:
    """
    Return the first available activity value in nM for a molecule ID, or +inf if not present.
    """
    acts = state["isomers_activities"].get(mol_id, "")
    for s in acts.split("|"):
        if "nM" in s:
            match = re.search(r"([<>]=?|=)?\s*([\d.]+)\s*nM", s.strip())
            if match:
                return float(match.group(2))
    return float("inf")


def get_activity_str(sorted_ids: Any, state: dict[str, Any]) -> Any:
    """
    Compute the DeltaValue for a sorted list of molecule IDs, defined as (max_nM - min_nM).
    """
    def find_last(ids: Any) -> Any:
        for mol_id in reversed(ids):
            val = get_activity_nm(mol_id, state)
            if not np.isinf(val):
                return val
        return 0

    def find_first(ids: Any) -> Any:
        for mol_id in ids:
            val = get_activity_nm(mol_id, state)
            if not np.isinf(val):
                return val
        return 0

    delta = find_last(sorted_ids) - find_first(sorted_ids)
    return delta


def get_activity_fold_ratio(sorted_ids: Any, state: dict[str, Any]) -> float:
    """
    Compute the activity fold-change as max_nM / min_nM for the available values.
    """
    valid_values = [get_activity_nm(mol_id, state) for mol_id in sorted_ids]
    valid_values = [float(v) for v in valid_values if not np.isinf(v) and float(v) > 0]
    if len(valid_values) < 2:
        return 0.0
    return max(valid_values) / min(valid_values)


def build_combo_from_groups(
    groups: Any,
    delta_dict: Any = None,
    gold_dict: Any = None,
    sort_mode: str = "Numeric order",
) -> Any:
    """
    Build human-readable combo labels from groups mapping using the selected order.
    """
    def rep_id_of(item: Any) -> Any:
        mol_id, group = item
        return group[0] if isinstance(group, (list, tuple)) and group else mol_id

    if sort_mode == "ΔActivity" and delta_dict:
        sorted_items = sorted(
            groups.items(),
            key=lambda x: delta_dict.get(rep_id_of(x), 0),
            reverse=True
        )
    elif sort_mode == "Activity fold-change" and gold_dict:
        sorted_items = sorted(
            groups.items(),
            key=lambda x: gold_dict.get(rep_id_of(x), 0),
            reverse=True,
        )
    else:
        sorted_items = sorted(groups.items())

    labels = []
    for mol_id, group in sorted_items:
        rep_id = rep_id_of((mol_id, group))
        dv = delta_dict.get(rep_id, 0) if delta_dict else 0
        gv = gold_dict.get(rep_id, 0) if gold_dict else 0
        labels.append(
            f"Mol {mol_id}: {group} | Δ = {dv:.2f} | Fold = {gv:.2f}x"
        )
    return labels
