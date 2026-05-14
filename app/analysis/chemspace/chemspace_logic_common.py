"""
========================
chemspace_logic_common.py
========================

Shared preprocessing utilities for ChemSpace dimensionality reducers.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator, GetTopologicalTorsionGenerator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from app.gui.loading_win import set_loading_screen_progress
from app.utils.app_logger import log_event, log_settings


def _run_with_smoothed_progress(
    *,
    state: dict[str, Any],
    progress_callback: Callable[[float], None],
    progress_start: float,
    progress_end: float,
    target: Callable[[], Any],
    ramp_seconds: float = 1.8,
    poll_seconds: float = 0.08,
    headroom: float = 0.6,
) -> Any:
    result_box: dict[str, Any] = {"result": None, "error": None}

    def _worker() -> None:
        try:
            result_box["result"] = target()
        except Exception as exc:
            result_box["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    start_time = time.monotonic()
    max_visible = max(progress_start, progress_end - max(0.1, headroom))
    progress_callback(progress_start)

    while worker.is_alive():
        elapsed = max(0.0, time.monotonic() - start_time)
        frac = 1.0 - np.exp(-elapsed / max(0.25, ramp_seconds))
        current = min(max_visible, progress_start + ((progress_end - progress_start) * frac))
        progress_callback(current)
        time.sleep(max(0.02, poll_seconds))

    worker.join()
    if result_box["error"] is not None:
        raise result_box["error"]
    progress_callback(progress_end)
    return result_box["result"]


def _parse_activity_value(raw_value: Any, include_undefined: bool) -> float | None:
    if pd.isna(raw_value):
        return None
    text = str(raw_value).strip()
    if not text:
        return None

    had_operator = False
    for op in ("<=", ">=", "<", ">"):
        if text.startswith(op):
            had_operator = True
            text = text[len(op):].strip()
            break

    if had_operator and not include_undefined:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def build_fingerprint_array(mol: Any, fp_algorithm: str) -> np.ndarray | None:
    if mol is None:
        return None

    if fp_algorithm == "Morgan Fingerprint":
        fp = GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
    elif fp_algorithm == "RDKit Fingerprint":
        fp = GetRDKitFPGenerator(fpSize=2048).GetFingerprint(mol)
    elif fp_algorithm == "Atom Pair Fingerprint":
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
    elif fp_algorithm == "MACCS Keys":
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif fp_algorithm == "Topological Torsion Fingerprint":
        fp = GetTopologicalTorsionGenerator(fpSize=1024).GetFingerprint(mol)
    elif fp_algorithm == "Pattern Fingerprint":
        fp = Chem.PatternFingerprint(mol)
    elif fp_algorithm == "Layered Fingerprint":
        fp = Chem.LayeredFingerprint(mol, fpSize=2048)
    else:
        raise ValueError(f"Unsupported fingerprint algorithm: {fp_algorithm}")

    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(float)


def prepare_reducer_inputs(
    *,
    activity: str,
    data: pd.DataFrame,
    fp_algorithm: str,
    include_undefined: bool,
    state: dict[str, Any],
    progress_callback: Callable[[float], None] | None = None,
    progress_start: float = 0.0,
    progress_end: float = 100.0,
) -> tuple[np.ndarray, list[dict[str, Any]], list[float], str]:
    """
    Convert a summary dataframe into fingerprint arrays, metadata, and activity values.
    """
    working_df = data.copy()
    activity_label = "None"

    if activity != "No activities":
        working_df[activity] = working_df[activity].apply(
            lambda value: _parse_activity_value(value, include_undefined)
        )
        working_df = working_df[working_df[activity].notna() & (working_df[activity] != 0)]

        if activity in state["nM_activity_types"]:
            working_df[activity] = -np.log10(working_df[activity].replace(0, np.nan) * 1e-9)
            working_df = working_df[working_df[activity].notna()]
            activity_label = f"p{activity}"
        else:
            activity_label = activity

    fingerprints: list[np.ndarray] = []
    activity_values: list[float] = []
    molecule_data: list[dict[str, Any]] = []
    total_rows = max(1, len(working_df))

    for row_idx, (_, row) in enumerate(working_df.iterrows(), start=1):
        mol = Chem.MolFromSmiles(str(row.get("Mol", "")))
        fp_array = build_fingerprint_array(mol, fp_algorithm)
        if fp_array is None:
            if progress_callback is not None:
                progress_callback(progress_start + ((row_idx / total_rows) * (progress_end - progress_start)))
            continue

        fingerprints.append(fp_array)
        activity_value = float(row[activity]) if activity_label != "None" else 0.0
        activity_values.append(activity_value)
        molecule_data.append(
            {
                "Mol_ID": int(row["Mol_sub_ID"]),
                "Name": "" if pd.isna(row.get("MolName", "N/A")) else row.get("MolName", "N/A"),
                "SMILES": row.get("Mol", ""),
                "Subset": int(row.get("Subset", -1)),
                activity_label: activity_value if activity_label != "None" else "N/A",
            }
        )
        if progress_callback is not None:
            progress_callback(progress_start + ((row_idx / total_rows) * (progress_end - progress_start)))

    if not fingerprints:
        return np.empty((0, 0)), [], [], activity_label

    return np.asarray(fingerprints, dtype=float), molecule_data, activity_values, activity_label


def compute_embedding_clusters(
    embedding: np.ndarray,
    progress_callback: Callable[[float], None] | None = None,
    progress_start: float = 0.0,
    progress_end: float = 100.0,
) -> np.ndarray:
    """
    Compute cluster labels for a 2D embedding using the same spirit as PCA:
    try multiple K values and keep the one with the best silhouette score.
    """
    if embedding is None or len(embedding) < 3:
        return np.zeros((len(embedding) if embedding is not None else 0,), dtype=int)

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ModuleNotFoundError:
        return np.zeros((len(embedding),), dtype=int)

    max_k = min(15, len(embedding))
    candidate_ks = list(range(2, max_k))
    if not candidate_ks:
        return np.zeros((len(embedding),), dtype=int)

    best_score = -1.0
    best_k = 2
    total_candidates = max(1, len(candidate_ks))
    for idx, k in enumerate(candidate_ks, start=1):
        model = KMeans(n_clusters=k, random_state=0).fit(embedding)
        if len(set(model.labels_)) < 2:
            if progress_callback is not None:
                progress_callback(progress_start + ((idx / total_candidates) * (progress_end - progress_start)))
            continue
        score = silhouette_score(embedding, model.labels_)
        if score > best_score:
            best_score = score
            best_k = k
        if progress_callback is not None:
            progress_callback(progress_start + ((idx / total_candidates) * (progress_end - progress_start)))

    if progress_callback is not None:
        progress_callback(progress_end)
    return KMeans(n_clusters=best_k, random_state=0).fit(embedding).labels_.astype(int)


def perform_pca_projection(
    activity: str,
    data: pd.DataFrame,
    subset: str,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Shared PCA backend: prepare molecular fingerprints, fit PCA, cluster the
    weighted coordinates, and return all payload needed by 2D/3D renderers.
    """
    log_event("PCA", "Preparing PCA coordinates and clustering inputs", indent=1)
    set_loading_screen_progress(state, 10)

    fp_algorithm = dpg.get_value("pca_fingerprint_algorithm_combo")
    include_undefined = dpg.get_value("pca_include_undefined_choice")
    dimension = dpg.get_value("pca_dimension_combo")
    state["pca_shown_cluster_idx"] = 0
    log_settings(
        "PCA",
        indent=2,
        subset=subset,
        activity=activity,
        dimension=dimension,
        fingerprint=fp_algorithm,
        include_undefined=include_undefined,
        molecules=len(data),
    )

    working_df = data.copy()
    activity_label = "None"
    if activity != "No activities":
        working_df[activity] = working_df[activity].apply(
            lambda value: _parse_activity_value(value, include_undefined)
        )
        working_df = working_df[working_df[activity].notna() & (working_df[activity] != 0)]
        if activity in state["nM_activity_types"]:
            working_df[activity] = -np.log10(working_df[activity].replace(0, np.nan) * 1e-9)
            working_df = working_df[working_df[activity].notna()]
            activity_label = f"p{activity}"
        else:
            activity_label = activity
    set_loading_screen_progress(state, 18)

    filtered_mols: list[Any] = []
    fingerprints: list[np.ndarray] = []
    activity_values: list[float] = []
    molecule_data: list[dict[str, Any]] = []

    total_rows = max(1, len(working_df))
    for row_idx, (_, row) in enumerate(working_df.iterrows(), start=1):
        mol = Chem.MolFromSmiles(str(row.get("Mol", "")))
        fp_array = build_fingerprint_array(mol, fp_algorithm)
        if fp_array is None:
            continue

        filtered_mols.append(mol)
        fingerprints.append(fp_array)
        activity_value = float(row[activity]) if activity_label != "None" else 0.0
        activity_values.append(activity_value)
        molecule_data.append(
            {
                "Mol_ID": int(row["Mol_sub_ID"]),
                "Name": "" if pd.isna(row.get("MolName", "N/A")) else row.get("MolName", "N/A"),
                activity_label: activity_value if activity_label != "None" else "N/A",
                "Subset": int(row.get("Subset", -1)),
                "Cluster": 0,
                "Cl. Mean Act": "N/A",
            }
        )
        if row_idx == total_rows or row_idx == 1 or row_idx % 10 == 0:
            set_loading_screen_progress(state, 18 + ((row_idx / total_rows) * 50))

    if len(filtered_mols) < 2:
        set_loading_screen_progress(state, 100)
        return None

    fingerprints_array = np.array(fingerprints, dtype=float)
    set_loading_screen_progress(state, 69)

    pca_model = PCA(n_components=3)
    x_pca = pca_model.fit_transform(fingerprints_array)
    variance_ratio = pca_model.explained_variance_ratio_ * 100
    set_loading_screen_progress(state, 78)

    pca_axis_signs = np.ones(3, dtype=float)
    ref = fingerprints_array[:, 0]
    if np.std(x_pca[:, 0]) > 1e-8 and np.std(ref) > 1e-8:
        if np.corrcoef(x_pca[:, 0], ref)[0, 1] < 0:
            x_pca[:, 0] *= -1
            pca_axis_signs[0] = -1.0
    if np.std(x_pca[:, 1]) > 1e-8 and np.std(ref) > 1e-8:
        if np.corrcoef(x_pca[:, 1], ref)[0, 1] < 0:
            x_pca[:, 1] *= -1
            pca_axis_signs[1] = -1.0

    state["pca_projection_model"] = pca_model
    state["pca_projection_signs"] = tuple(float(v) for v in pca_axis_signs)
    state["pca_projection_fp_algorithm"] = fp_algorithm
    set_loading_screen_progress(state, 82)

    weights = (variance_ratio / 100) ** 0.5
    weighted_coords = x_pca * weights

    best_score = -1.0
    best_k = 2
    max_k = min(15, len(filtered_mols))
    candidate_ks = list(range(2, max_k))
    total_candidates = max(1, len(candidate_ks))
    if not candidate_ks:
        set_loading_screen_progress(state, 90)

    for idx, k in enumerate(candidate_ks, start=1):
        model = KMeans(n_clusters=k, random_state=0).fit(weighted_coords)
        if len(set(model.labels_)) < 2:
            if idx == total_candidates or idx == 1 or idx % 2 == 0:
                set_loading_screen_progress(state, 82 + ((idx / total_candidates) * 8))
            continue
        score = silhouette_score(weighted_coords, model.labels_)
        if score > best_score:
            best_score = score
            best_k = k
        if idx == total_candidates or idx == 1 or idx % 2 == 0:
            set_loading_screen_progress(state, 82 + ((idx / total_candidates) * 8))

    labels = KMeans(n_clusters=best_k, random_state=0).fit(weighted_coords).labels_
    set_loading_screen_progress(state, 92)

    csv_df = pd.DataFrame(molecule_data)
    csv_df["cluster"] = labels
    activity_cols = [col for col in csv_df.columns if col in state["activity_types"]]
    activity_col = activity_cols[0] if activity_cols else None
    cluster_means = None

    if activity_col:
        cluster_activity = csv_df[[activity_col, "cluster"]].copy().replace("N/A", np.nan)
        cluster_activity[activity_col] = pd.to_numeric(cluster_activity[activity_col], errors="coerce")
        cluster_means = cluster_activity.groupby("cluster")[activity_col].mean()

        if activity_col in state["dimensionless"] or activity_col in state["percent_activities"]:
            sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()
        else:
            sorted_clusters = cluster_means.sort_values().index.tolist()

        cluster_map = {old: new for new, old in enumerate(sorted_clusters)}
        labels = np.array([cluster_map[old] for old in labels])

    set_loading_screen_progress(state, 94)
    total_entries = max(1, len(molecule_data))
    for i, entry in enumerate(molecule_data):
        entry["Cluster"] = int(labels[i]) + 1
        entry["Cl. Mean Act"] = cluster_means.get(int(labels[i]), "N/A") if cluster_means is not None else "N/A"
        if (i + 1) == total_entries or (i + 1) == 1 or (i + 1) % 10 == 0:
            set_loading_screen_progress(state, 94 + (((i + 1) / total_entries) * 3))

    return {
        "subset": subset,
        "x_pca": x_pca,
        "variance_ratio": variance_ratio,
        "filtered_mols": filtered_mols,
        "molecule_data": molecule_data,
        "activity": activity,
        "activity_label": activity_label,
        "labels": labels,
        "activity_values": activity_values,
        "fp_algorithm": fp_algorithm,
        "best_k": best_k,
        "dimension": dimension,
    }


def perform_umap(activity: str, data: pd.DataFrame, subset: str, state: dict[str, Any]) -> None:
    last_progress = {"value": None}

    def _progress(value: float) -> None:
        rounded = round(float(value), 1)
        if last_progress["value"] == rounded:
            return
        last_progress["value"] = rounded
        set_loading_screen_progress(state, rounded)

    log_event("UMAP", "Preparing UMAP coordinates", indent=1)
    fp_algorithm = dpg.get_value("umap_fingerprint_algorithm_combo")
    include_undefined = dpg.get_value("umap_include_undefined_choice")
    dimension = dpg.get_value("umap_dimension_combo")
    n_neighbors = int(dpg.get_value("umap_neighbors_combo"))
    min_dist = float(dpg.get_value("umap_min_dist_combo"))
    metric = dpg.get_value("umap_metric_combo")
    log_settings(
        "UMAP",
        indent=2,
        subset=subset,
        activity=activity,
        fingerprint=fp_algorithm,
        dimension=dimension,
        include_undefined=include_undefined,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        molecules=len(data),
    )

    fingerprints, molecule_data, activity_values, activity_label = prepare_reducer_inputs(
        activity=activity,
        data=data,
        fp_algorithm=fp_algorithm,
        include_undefined=include_undefined,
        state=state,
        progress_callback=_progress,
        progress_start=0.4,
        progress_end=2.0,
    )
    if len(molecule_data) < 2:
        set_loading_screen_progress(state, 100)
        return

    try:
        import umap
    except ModuleNotFoundError as exc:
        raise RuntimeError("UMAP is not available in the current environment. Install 'umap-learn' to enable this ChemSpace view.") from exc

    reducer = umap.UMAP(
        n_components=2 if dimension == "2D" else 3,
        n_neighbors=max(2, min(n_neighbors, len(molecule_data) - 1)),
        min_dist=min_dist,
        metric=metric,
        init="random",
        random_state=42,
    )
    embedding = _run_with_smoothed_progress(
        state=state,
        progress_callback=_progress,
        progress_start=2.0,
        progress_end=95.0,
        target=lambda: reducer.fit_transform(fingerprints),
        ramp_seconds=1.9,
        poll_seconds=0.08,
        headroom=0.8,
    )
    cluster_labels = compute_embedding_clusters(
        embedding,
        progress_callback=_progress,
        progress_start=95.0,
        progress_end=98.2,
    )
    for idx, label in enumerate(cluster_labels):
        molecule_data[idx]["Cluster"] = int(label) + 1
    state["umap_embedding"] = embedding
    state["umap_projection_model"] = reducer
    state["umap_projection_fp_algorithm"] = fp_algorithm
    state["umap_projection_fingerprints"] = np.asarray(fingerprints, dtype=float)
    _progress(98.8)

    from app.analysis.chemspace.chemspace_plot_common import draw_umap_plot_2d, draw_umap_plot_3d

    if dimension == "2D":
        draw_umap_plot_2d(
            subset=subset,
            embedding=embedding,
            molecule_data=molecule_data,
            activity_label=activity_label,
            activity_values=activity_values,
            fp_algorithm=fp_algorithm,
            cluster_labels=cluster_labels,
            state=state,
        )
    else:
        draw_umap_plot_3d(
            subset=subset,
            embedding=embedding,
            molecule_data=molecule_data,
            activity_label=activity_label,
            activity_values=activity_values,
            fp_algorithm=fp_algorithm,
            cluster_labels=cluster_labels,
            state=state,
        )
    _progress(99.4)


def perform_tsne(activity: str, data: pd.DataFrame, subset: str, state: dict[str, Any]) -> None:
    last_progress = {"value": None}

    def _progress(value: float) -> None:
        rounded = round(float(value), 1)
        if last_progress["value"] == rounded:
            return
        last_progress["value"] = rounded
        set_loading_screen_progress(state, rounded)

    log_event("TSNE", "Preparing t-SNE coordinates", indent=1)
    fp_algorithm = dpg.get_value("tsne_fingerprint_algorithm_combo")
    include_undefined = dpg.get_value("tsne_include_undefined_choice")
    dimension = dpg.get_value("tsne_dimension_combo")
    perplexity = float(dpg.get_value("tsne_perplexity_combo"))
    learning_rate_raw = dpg.get_value("tsne_learning_rate_combo")
    n_iter = int(dpg.get_value("tsne_iterations_combo"))
    metric = dpg.get_value("tsne_metric_combo")
    log_settings(
        "TSNE",
        indent=2,
        subset=subset,
        activity=activity,
        fingerprint=fp_algorithm,
        dimension=dimension,
        include_undefined=include_undefined,
        perplexity=perplexity,
        learning_rate=learning_rate_raw,
        iterations=n_iter,
        metric=metric,
        molecules=len(data),
    )

    fingerprints, molecule_data, activity_values, activity_label = prepare_reducer_inputs(
        activity=activity,
        data=data,
        fp_algorithm=fp_algorithm,
        include_undefined=include_undefined,
        state=state,
        progress_callback=_progress,
        progress_start=0.4,
        progress_end=2.0,
    )
    if len(molecule_data) < 2:
        set_loading_screen_progress(state, 100)
        return

    try:
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise RuntimeError("t-SNE is not available in the current environment because scikit-learn could not be imported.") from exc

    effective_perplexity = min(perplexity, max(2.0, len(molecule_data) - 1.0))
    if effective_perplexity >= len(molecule_data):
        effective_perplexity = max(1.0, len(molecule_data) - 1.0)
    learning_rate: str | float = learning_rate_raw if learning_rate_raw == "auto" else float(learning_rate_raw)

    reducer = TSNE(
        n_components=2 if dimension == "2D" else 3,
        perplexity=effective_perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        metric=metric,
        init="pca",
        random_state=42,
    )
    embedding = _run_with_smoothed_progress(
        state=state,
        progress_callback=_progress,
        progress_start=2.0,
        progress_end=95.0,
        target=lambda: reducer.fit_transform(fingerprints),
        ramp_seconds=2.1,
        poll_seconds=0.08,
        headroom=0.8,
    )
    cluster_labels = compute_embedding_clusters(
        embedding,
        progress_callback=_progress,
        progress_start=95.0,
        progress_end=98.2,
    )
    for idx, label in enumerate(cluster_labels):
        molecule_data[idx]["Cluster"] = int(label) + 1
    state["tsne_embedding"] = embedding
    state["tsne_projection_model"] = reducer
    state["tsne_projection_fp_algorithm"] = fp_algorithm
    state["tsne_projection_fingerprints"] = np.asarray(fingerprints, dtype=float)
    _progress(98.8)

    from app.analysis.chemspace.chemspace_plot_common import draw_tsne_plot_2d, draw_tsne_plot_3d

    if dimension == "2D":
        draw_tsne_plot_2d(
            subset=subset,
            embedding=embedding,
            molecule_data=molecule_data,
            activity_label=activity_label,
            activity_values=activity_values,
            fp_algorithm=fp_algorithm,
            cluster_labels=cluster_labels,
            state=state,
        )
    else:
        draw_tsne_plot_3d(
            subset=subset,
            embedding=embedding,
            molecule_data=molecule_data,
            activity_label=activity_label,
            activity_values=activity_values,
            fp_algorithm=fp_algorithm,
            cluster_labels=cluster_labels,
            state=state,
        )
    _progress(99.4)
