"""
=================
sal_logic.py
=================

Computation and orchestration layer for the SAR Landscape.

This module:
- reads current UI selections (subset, activity, options);
- prepares/cleans the working dataset for the chosen activity;
- generates RDKit molecules and fingerprints;
- computes pairwise similarities (Tanimoto) and Δp (log10 difference);
- computes SALI values and linear integer display bounds;
- stores all intermediate artefacts in `state`;
- delegates the actual plotting to `mod.gui.landscape_plot.draw_landscape_plot`.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Run landscape analysis

import os
import re
import math
import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
from typing import Any
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from app.utils.app_logger import log_event, log_settings
from app.gui.loading_win import set_loading_screen_progress
from app.analysis.similarity.sal_plot import draw_landscape_plot


# -----------------------------------------------------------------------------
# 2. Run landscape analysis
# -----------------------------------------------------------------------------
def run_landscape_analysis(state: dict[str, Any]) -> Any:
    log_event("Similarity", "Computing Structure-Activity Landscape data", indent=1)

    subset = dpg.get_value("landscape_subset_choice")
    activity = dpg.get_value("landscape_activity_type")
    state["landscape_activity_type"] = activity
    read_undefined = dpg.get_value("landscape_include_undefined_choice")
    read_inactives = dpg.get_value("landscape_include_inactive_choice")
    log_settings("Similarity", indent=2, subset=subset, activity=activity, fingerprint=dpg.get_value("landscape_fingerprint_algorithm_combo"), include_undefined=read_undefined, include_inactives=read_inactives, delta_thresh=dpg.get_value("landscape_delta_thresh"), similarity_thresh=dpg.get_value("landscape_similarity_thresh"), sali_thresh=dpg.get_value("landscape_sali_index_thresh"))

    old_series = state.pop("landscape_color_series_tags", [])
    for tag in old_series:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    old_themes = state.pop("landscape_bucket_themes", [])
    for th in old_themes:
        try:
            if dpg.does_item_exist(th):
                dpg.delete_item(th)
        except Exception:
            pass

    for tag in [
        "landscape_plot_main_window", "landscape_plot_handlers",
        "landscape_mol1_image_widget", "landscape_mol2_image_widget"
    ]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    set_loading_screen_progress(state, 4)

    summary_dir = state["summary_dir"]
    csv_file = os.path.join(summary_dir, f"{subset}_summary.csv")
    data = pd.read_csv(csv_file)
    set_loading_screen_progress(state, 8)

    # -----------------------------------------------------------------------------
    # 2.1. Parse activity series
    # -----------------------------------------------------------------------------
    def _parse_activity_series(
        df: Any,
        activity_col: str,
        read_undefined: bool,
        read_inactives: bool
    ) -> Any:
        s = df[activity_col].copy()
        if read_undefined:
            s = s.astype(str)
            s = s.apply(
                lambda x: re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)[0]
                if re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x) else np.nan
            )
        s = pd.to_numeric(s, errors="coerce")
        if read_inactives:
            s = s.fillna(0.0)
        return s

    required_cols = ["Mol_sub_ID", "MolName", "Mol"]
    for col in required_cols + [activity]:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_file}")

    work = pd.DataFrame({
        "MolID":  data["Mol_sub_ID"],
        "Name":   data["MolName"].fillna("N/A").replace("", "N/A"),
        "SMILES": data["Mol"]
    })
    work["Activity"] = _parse_activity_series(data, activity, read_undefined, read_inactives)
    set_loading_screen_progress(state, 16)

    # Valid rows
    work = work[work["SMILES"].notna() & (work["SMILES"].astype(str) != "")]
    if not read_inactives:
        work = work[work["Activity"].notna()]

    if len(work) < 2:
        with dpg.window(label="SAR Landscape Analysis", modal=True, autosize=True, no_resize=True, no_collapse=True):
            dpg.add_text("Not enough valid molecules to build pairwise plot (need at least 2).")
            dpg.add_button(label="OK", width=75, height=25, callback=lambda: None)
        return

    fp_choice = dpg.get_value("landscape_fingerprint_algorithm_combo")

    # -----------------------------------------------------------------------------
    # 2.2. Mol from smiles
    # -----------------------------------------------------------------------------
    def _mol_from_smiles(smi: Any) -> Any:
        try:
            return Chem.MolFromSmiles(smi)
        except Exception:
            return None

    # -----------------------------------------------------------------------------
    # 2.3. Fingerprint
    # -----------------------------------------------------------------------------
    def _fingerprint(mol: Any, choice: str) -> Any:
        if mol is None:
            return None
        morgan_fp_gen = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
        try:
            if choice == "Morgan Fingerprint":
                return morgan_fp_gen.GetFingerprint(mol)
            elif choice == "RDKit Fingerprint":
                return Chem.RDKFingerprint(mol)
            elif choice == "Atom Pair Fingerprint":
                return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
            elif choice == "MACCS Keys":
                return MACCSkeys.GenMACCSKeys(mol)
            elif choice == "Topological Torsion Fingerprint":
                return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
            elif choice == "Pattern Fingerprint":
                return Chem.PatternFingerprint(mol)
            elif choice == "Layered Fingerprint":
                return Chem.LayeredFingerprint(mol)
            else:
                return morgan_fp_gen.GetFingerprint(mol)
        except Exception:
            return None

    work["ROMol"] = work["SMILES"].apply(_mol_from_smiles)
    work["FP"]    = work["ROMol"].apply(lambda m: _fingerprint(m, fp_choice))
    work = work[work["FP"].notna()]
    set_loading_screen_progress(state, 28)

    state["landscape_work_df"] = work
    if len(work) < 2:
        with dpg.window(label="SAR Landscape Analysis", modal=True, autosize=True, no_resize=True, no_collapse=True):
            dpg.add_text("Not enough valid fingerprints to build pairwise plot (need at least 2).")
            dpg.add_button(label="OK", width=75, height=25, callback=lambda: None)
        return

    act = work["Activity"].copy()
    positive = act[act > 0]
    if len(positive) == 0:
        with dpg.window(label="SAR Landscape Analysis", modal=True, autosize=True, no_resize=True, no_collapse=True):
            dpg.add_text("All activities are zero/undefined; cannot compute ΔpValue.")
            dpg.add_button(label="OK", width=75, height=25, callback=lambda: None)
        return

    epsilon = float(positive.min()) * 0.5
    epsilon = epsilon if epsilon > 0 else 1e-12
    act = act.fillna(0.0)
    act = act.where(act > 0, other=epsilon)
    work["_act_eps"] = act

    # -----------------------------------------------------------------------------
    # 2.4. Delta p
    # -----------------------------------------------------------------------------
    def _delta_p(a: Any, b: Any) -> Any:
        return float(abs(math.log10(b) - math.log10(a)))

    fps = list(work["FP"])
    mol_ids = work["MolID"].astype(str).tolist()
    n = len(fps)

    xs = []
    pair_i = []
    pair_j = []
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for off, sim in enumerate(sims, start=1):
            j = i + off
            if mol_ids[i] == mol_ids[j]:
                continue
            sim = min(float(sim), 0.95)  # <-- cap similarity to 0.95 to avoid high disproportionate SALI values
            pair_i.append(i)
            pair_j.append(j)
            xs.append(sim)
        if i % max(1, n // 25) == 0 or i == n - 2:
            set_loading_screen_progress(state, 28 + (((i + 1) / max(1, n - 1)) * 50))

    xs = np.asarray(xs, dtype=float)
    pair_i = np.asarray(pair_i, dtype=int)
    pair_j = np.asarray(pair_j, dtype=int)

    if xs.size == 0:
        with dpg.window(label="SAR Landscape Analysis", modal=True, autosize=True, no_resize=True, no_collapse=True):
            dpg.add_text("No valid molecular pairs after filtering duplicates (same Mol_sub_ID).")
            dpg.add_button(label="OK", width=75, height=25, callback=lambda: None)
        return

    state["landscape_pair_i"] = pair_i
    state["landscape_pair_j"] = pair_j

    pos_vals = work["_act_eps"][work["_act_eps"] > 0]
    if len(pos_vals) == 0:
        with dpg.window(label="SAR Landscape Analysis", modal=True, autosize=True, no_resize=True, no_collapse=True):
            dpg.add_text("No positive activity values to compute Δp.")
            dpg.add_button(label="OK", width=75, height=25, callback=lambda: None)
        return

    # Recompute Δp only for saved pairs
    acts = work["_act_eps"].to_numpy(dtype=float, copy=True)
    ys = np.array([_delta_p(acts[i], acts[j]) for i, j in zip(pair_i, pair_j)], dtype=float)

    # SALI in linear space
    s_clip = 0.99999
    denom = 1.0 - np.minimum(xs, s_clip)
    denom = np.maximum(denom, 1e-12)
    sali_raw = ys / denom

    # --- SALI bounds ---
    sali_min = float(np.nanmin(sali_raw))
    sali_max = float(np.nanmax(sali_raw))

    lo = sali_min if np.isfinite(sali_min) else 0.0
    hi = sali_max if np.isfinite(sali_max) else 1.0
    if hi <= lo:
        hi = lo + 1.0

    # Persist only what the other modules actually use
    state["landscape_xs"] = xs
    state["landscape_ys"] = ys
    state["landscape_sali_raw"] = sali_raw
    state["landscape_sali_minmax"] = (lo, hi)
    set_loading_screen_progress(state, 82)

    draw_landscape_plot(state, fp_choice, activity, xs, ys, sali_raw, lo, hi)
