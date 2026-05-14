"""
======================
mmpa_precompute.py
======================

Precomputation of MMPA transformations.

Performs a preliminary analysis of molecular pairs and their 
activity differences, needed to build the MMPA distribution plots.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Run mmpa delta precompute
# 3. Run mmpa delta precompute single activity
# 4. Load bioactivity types from json
# 5. List subset files
# 6. Prepare activity dataframe
# 7. Aggregate by mol id for mmpa
# 8. Compute all pair deltas
# 9. Make stats
# 10. Sanitize

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import re
import json
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
from collections import defaultdict
from typing import Any
from app.utils.app_logger import log_event, log_settings, log_exception
from app.lmm.lmm_gui import update_rga_status


# -----------------------------------------------------------------------------
# 2. Run mmpa delta precompute
# -----------------------------------------------------------------------------
def run_mmpa_delta_precompute(
    state: dict[str, Any],
    include_undefined: bool = True,
    include_inactive: bool = False,
    min_delta: float = 0.0,
    bins: int = 50
) -> None:
    """
    Pre-compute the distribution of ΔpValue MMPA for ALL activities in the dataset.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
        include_undefined (Any): Parameter accepted by this routine. Defaults to the configured value.
        include_inactive (Any): Parameter accepted by this routine. Defaults to the configured value.
        min_delta (Any): Parameter accepted by this routine. Defaults to the configured value.
        bins (Any): Parameter accepted by this routine. Defaults to the configured value.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    log_event("LMM", "Precomputing MMPA delta-activity distributions", indent=1)
    log_settings("LMM", indent=2, include_undefined=include_undefined, include_inactive=include_inactive, min_delta=min_delta, bins=bins)
    update_rga_status(f"MMPA Δactivity precomputation started ...", state, step_id=True)
    work_dir = state.get("work_dir", "")

    activities = _load_bioactivity_types_from_json(work_dir)
    if not activities:
        log_event("LMM", "No activities found in results.sof; aborted.", indent=2, level="ERROR")
        update_rga_status(f"   No activities found", state)
        return

    if "mmpa_delta_hist" not in state or not isinstance(state["mmpa_delta_hist"], dict):
        state["mmpa_delta_hist"] = {}

    subset_files = _list_subset_files(state["summary_dir"])
    if not subset_files:
        log_event("LMM", f"No '*_summary.csv' found in: {state['summary_dir']}", indent=2, level="ERROR")
        update_rga_status(f"   No subset summary files found", state)
        return

    subset_frames: dict[str, Any] = {}
    for subset_name, csv_path in subset_files:
        try:
            subset_frames[subset_name] = pd.read_csv(csv_path)
        except Exception as e:
            log_exception("LMM", f"Cannot read {csv_path}", e, indent=2)

    for act in activities:
        update_rga_status(f"   MMPA Δ{act} precomputation started", state, temp=True)
        try:
            run_mmpa_delta_precompute_single_activity(
                state=state,
                activity=act,
                subset_frames=subset_frames,
                include_undefined=include_undefined,
                include_inactive=include_inactive,
                min_delta=min_delta,
                bins=bins
            )
            update_rga_status(f"   MMPA Δ{act} precomputation completed", state)
        except Exception as e:
            log_exception("LMM", f"Error on activity '{act}' during MMPA Δ-precomputation", e, indent=2)
            update_rga_status(f"   MMPA Δ{act} precomputation error", state)

 


# -----------------------------------------------------------------------------
# 3. Run mmpa delta precompute single activity
# -----------------------------------------------------------------------------
def run_mmpa_delta_precompute_single_activity(
    state: dict[str, Any],
    activity: str,
    subset_frames: dict[str, Any] | None = None,
    include_undefined: bool = True,
    include_inactive: bool = False,
    min_delta: float = 0.0,
    bins: int = 50
) -> None:
    """
    Execute ΔpValue pre-computation for ONE activity across all subsets.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
        activity (str): Parameter accepted by this routine.
        include_undefined (Any): Parameter accepted by this routine. Defaults to the configured value.
        include_inactive (Any): Parameter accepted by this routine. Defaults to the configured value.
        min_delta (Any): Parameter accepted by this routine. Defaults to the configured value.
        bins (Any): Parameter accepted by this routine. Defaults to the configured value.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    log_event("LMM", f"Processing activity '{activity}' for MMPA precomputation", indent=2)
    log_settings("LMM", indent=3, activity=activity, include_undefined=include_undefined, include_inactive=include_inactive, min_delta=min_delta, bins=bins)
    summary_dir = state["summary_dir"]
    report_dir = state["report_dir"]
    mmpa_dir = os.path.join(report_dir, "mmpa_delta_distributions")
    nm_types = set(state.get("nM_activity_types", []))

    # Output for this activity
    safe_act = _sanitize(activity)
    os.makedirs(mmpa_dir, exist_ok=True)
    out_csv = os.path.join(mmpa_dir, f"mmpa_delta_distribution_{safe_act}.csv")

    subset_files = _list_subset_files(summary_dir)
    if not subset_files:
        log_event("LMM", f"No '*_summary.csv' found in: {summary_dir}", indent=2, level="ERROR")
        return

    global_deltas = []
    by_subset = defaultdict(list)
    pair_types_by_subset = defaultdict(list)

    for subset_name, csv_path in subset_files:
        if subset_frames is not None and subset_name in subset_frames:
            df0 = subset_frames[subset_name]
        else:
            try:
                df0 = pd.read_csv(csv_path)
            except Exception as e:
                log_exception("LMM", f"Cannot read {csv_path}", e, indent=2)
                continue

        # Skip subsets without the activity column
        if activity not in df0.columns:
            continue

        df = _prepare_activity_dataframe(df0, activity, nm_types,
                                         include_undefined=include_undefined,
                                         include_inactive=include_inactive)
        if df.empty:
            continue

        df_agg, r_cols = _aggregate_by_mol_id_for_mmpa(df)
        if len(df_agg) < 2:
            continue

        deltas, kinds = _compute_all_pair_deltas(df_agg, r_cols, min_delta=min_delta)
        if not deltas:
            continue

        global_deltas.extend(deltas)
        by_subset[subset_name].extend(deltas)
        pair_types_by_subset[subset_name].extend(kinds)

    if global_deltas:
        rows = []
        for s_name in by_subset.keys():
            ds = by_subset[s_name]
            ks = pair_types_by_subset[s_name]
            for d, k in zip(ds, ks):
                rows.append((s_name, d, k))
        # GLOBAL row
        for d in global_deltas:
            rows.append(("GLOBAL", d, "Mixed"))

        df_out = pd.DataFrame(rows, columns=["Subset", "Delta", "PairType"])
        df_out.to_csv(out_csv, index=False)
                            
    stats_global = _make_stats(global_deltas)
    stats_by_subset = {s: _make_stats(ds) for s, ds in by_subset.items()}

    if "mmpa_delta_hist" not in state or not isinstance(state["mmpa_delta_hist"], dict):
        state["mmpa_delta_hist"] = {}
    state["mmpa_delta_hist"][activity] = {
        "global_deltas": global_deltas,
        "by_subset": dict(by_subset),
        "stats": {"GLOBAL": stats_global, **{s: st for s, st in stats_by_subset.items()}}
    }


# -----------------------------------------------------------------------------
# 4. Load bioactivity types from json
# -----------------------------------------------------------------------------
def _load_bioactivity_types_from_json(work_dir: str) -> Any:
    """
    Read 'output_dir/work_dir/results.sof' and return the activity list.
    Priority: bioact_types_dict['Dataset']['bioactivities'].
    If missing or empty, merge the bioactivities of the individual subsets.

    Args:
        work_dir (str): Base work directory.

    Returns:
        list[str]: Sorted list of unique activity names.
    """
    json_path = os.path.join(work_dir, "results.sof")
    if not os.path.isfile(json_path):
        json_path = os.path.join(work_dir, "results.srf")
    if not os.path.isfile(json_path):
        print(f"[MMPA-Δ] JSON not found: {json_path}")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[MMPA-Δ] Error reading JSON: {e}")
        return []

    bio = data.get("bioact_types_dict", {})
    # 1) Dataset
    ds = bio.get("Dataset", {})
    acts = ds.get("bioactivities", []) or []
    acts = [a for a in acts if isinstance(a, str) and a.strip()]
    if acts:
        return sorted(set(acts), key=str)

    # 2) Fallback: union of subsets
    found = set()
    for k, v in bio.items():
        if not isinstance(v, dict):
            continue
        arr = v.get("bioactivities", []) or []
        for a in arr:
            if isinstance(a, str) and a.strip():
                found.add(a)
    return sorted(found, key=str)


# -----------------------------------------------------------------------------
# 5. List subset files
# -----------------------------------------------------------------------------
def _list_subset_files(summary_dir: str) -> Any:
    """
    Return a list of (subset_name, csv_path) for all '<subset>_summary.csv',
    EXCLUDING 'Dataset_summary.csv'.

    Args:
        summary_dir (str): Directory containing the per-subset summary files.

    Returns:
        list[tuple[str, str]]: Pairs (subset_name, csv_path).
    """
    files = []
    for fn in os.listdir(summary_dir):
        if not fn.endswith("_summary.csv"):
            continue
        subset_name = fn.replace("_summary.csv", "")
        if subset_name.lower() == "dataset":
            # Do not include "Dataset_summary.csv"
            continue
        files.append((subset_name, os.path.join(summary_dir, fn)))
    files.sort(key=lambda x: x[0])
    return files


# -----------------------------------------------------------------------------
# 6. Prepare activity dataframe
# -----------------------------------------------------------------------------
def _prepare_activity_dataframe(
    df0: Any,
    activity: str,
    nm_types: Any,
    include_undefined: bool = True,
    include_inactive: bool = False
) -> Any:
    """
    Apply the same filtering and pValue rules as in run_mmpa_analysis.

    Args:
        df0 (pd.DataFrame): Original subset DataFrame.
        activity (str): Activity column to use.
        nm_types (set[str]): Activities converted to pValue with -log10(nM*1e-9).
        include_undefined (bool): If True includes '<', '>' etc.
        include_inactive (bool): If True treats empty/N.A. as pValue 0.0.

    Returns:
        pd.DataFrame: DataFrame with standardised columns and pValue computed.
    """
    df = df0.copy()
    if "Mol" not in df.columns:
        return pd.DataFrame()
    df = df[df["Mol"].notna()].copy()

    if "Mol_sub_ID" in df.columns:
        df = df.rename(columns={"Mol_sub_ID": "Mol_ID"})
    elif "Mol_ID" not in df.columns:
        df["Mol_ID"] = range(1, len(df) + 1)

    r_cols = [c for c in df.columns if re.fullmatch(r"R\d+", c)]
    keep = ["Mol_ID", "Mol", "MolName", activity] + r_cols
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    activity_raw = df[activity].astype(str).str.strip()
    mask_defined = activity_raw.str.match(r"^=?\s*\d+(\.\d+)?$", na=False)
    mask_undefined = activity_raw.str.match(r"^[<>]=?\s*\d+(\.\d+)?$", na=False) if include_undefined else pd.Series(False, index=df.index)
    mask_inactive = (activity_raw.isin(["", "N/A", "NA", "None", "NONE"]) | df[activity].isna()) if include_inactive else pd.Series(False, index=df.index)

    df = df[mask_defined | mask_undefined | mask_inactive].copy()

    undefined_mask = activity_raw.str.match(r"^[<>]=?\s*\d+(\.\d+)?$", na=False)
    df["Undefined"] = undefined_mask.loc[df.index]

    extracted = df[activity].astype(str).str.strip().str.extract(r"([<>=]{1,2})?\s*(\d+\.?\d*)")
    df["_activity_numeric"] = pd.to_numeric(extracted[1], errors="coerce")
    df["pValue"] = np.nan

    if include_inactive:
        inactive_mask = (activity_raw.isin(["", "N/A", "NA", "None", "NONE"]) | df[activity].isna())
        df.loc[inactive_mask, "_activity_numeric"] = 0.0
        df.loc[inactive_mask, "pValue"] = 0.0

    if activity in nm_types:
        numeric_vals = pd.to_numeric(df["_activity_numeric"], errors="coerce").to_numpy(dtype=float)
        pvals = np.where(np.isfinite(numeric_vals) & (numeric_vals > 0), -np.log10(numeric_vals * 1e-9), 0.0)
        df.loc[df["pValue"].isna(), "pValue"] = pvals[df["pValue"].isna().to_numpy()]
    else:
        df.loc[df["pValue"].isna(), "pValue"] = df["_activity_numeric"]

    return df


# -----------------------------------------------------------------------------
# 7. Aggregate by mol id for mmpa
# -----------------------------------------------------------------------------
def _aggregate_by_mol_id_for_mmpa(df: Any) -> Any:
    """
    Collapse duplicated rows by Mol_ID:
    - pValue: mean
    - Mol, MolName: first
    - Undefined: any
    - R*: first

    Args:
        df (pd.DataFrame): DataFrame after activity parsing.

    Returns:
        tuple[pd.DataFrame, list[str]]: (aggregated df, list of R* columns)
    """
    r_cols = [c for c in df.columns if c.startswith("R")]

    agg_funcs = {
        "Mol": "first",
        "MolName": "first",
        "pValue": "mean",
        "Undefined": "any",
    }
    agg_funcs.update({c: "first" for c in r_cols})

    df_agg = df.groupby("Mol_ID", as_index=False).agg(agg_funcs)
    return df_agg, r_cols


# -----------------------------------------------------------------------------
# 8. Compute all pair deltas
# -----------------------------------------------------------------------------
def _compute_all_pair_deltas(df_agg: Any, r_cols: Any, min_delta: float = 0.0) -> Any:
    """
    Compute ΔpValue for all pairs (i<j) that differ in exactly 1 R,
    or in 2 R positions when the two fragments coincide (possibly swapped) and contain two dummy atoms.

    Args:
        df_agg (pd.DataFrame): DataFrame aggregated by Mol_ID.
        r_cols (list[str]): R* columns.
        min_delta (float): Discard ΔpValue < min_delta.

    Returns:
        tuple[list[float], list[str]]: (list of ΔpValues, list of tags 'Active'/'Inactive*'/'HasUndefined#')
    """
    # -----------------------------------------------------------------------------
    # 8.1. Count two dummies
    # -----------------------------------------------------------------------------
    def _count_two_dummies(smi: Any) -> Any:
        """
        Execute the count two dummies routine.
        
        Args:
            smi (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        return smi.count("[*:") >= 2

    n = len(df_agg)
    deltas = []
    kinds = []
    if n < 2:
        return deltas, kinds

    pvalues = pd.to_numeric(df_agg["pValue"], errors="coerce").to_numpy(dtype=float)
    undefineds = df_agg["Undefined"].fillna(False).astype(bool).to_numpy()
    r_mats = [df_agg[c].fillna("").astype(str).to_numpy() for c in r_cols]

    for i in range(n - 1):
        v1 = pvalues[i]
        if not np.isfinite(v1):
            continue
        for j in range(i + 1, n):
            try:
                v2 = pvalues[j]
                if not np.isfinite(v2):
                    continue

                diff_idx = []
                for col_idx, _ in enumerate(r_cols):
                    if r_mats[col_idx][i] != r_mats[col_idx][j]:
                        diff_idx.append(col_idx)
                        if len(diff_idx) > 2:
                            break

                if len(diff_idx) == 1:
                    pass
                elif len(diff_idx) == 2:
                    idx1, idx2 = diff_idx
                    a1, b1 = r_mats[idx1][i], r_mats[idx1][j]
                    a2, b2 = r_mats[idx2][i], r_mats[idx2][j]
                    same_pair = (a1 == a2 and b1 == b2) or (a1 == b2 and b1 == a2)
                    if not same_pair:
                        continue
                    if not (_count_two_dummies(a1) and _count_two_dummies(b1)):
                        continue
                else:
                    continue

                d = abs(v1 - v2)
                if d < min_delta:
                    continue

                has_undef = bool(undefineds[i] or undefineds[j])
                if v1 == 0.0 or v2 == 0.0:
                    kind = "Inactive*"
                elif has_undef:
                    kind = "HasUndefined#"
                else:
                    kind = "Active"

                deltas.append(float(d))
                kinds.append(kind)
            except Exception:
                continue
            
    return deltas, kinds


# -----------------------------------------------------------------------------
# 9. Make stats
# -----------------------------------------------------------------------------
def _make_stats(values: Any) -> Any:
    """
    Compute basic descriptive statistics over a list of floats.

    Args:
        values (list[float]): Values to summarise.

    Returns:
        dict: n, mean, median, std, min, max, p90, p95, p99 (or None if empty list).
    """
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None, "p90": None, "p95": None, "p99": None}

    arr = np.array(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


# -----------------------------------------------------------------------------
# 10. Sanitize
# -----------------------------------------------------------------------------
def _sanitize(s: Any) -> Any:
    """
    Make a string safe for filenames.

    Args:
        s (str or None): Input string.

    Returns:
        str or None: Sanitised string, or None if input is None/empty.
    """
    if not s:
        return None
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
