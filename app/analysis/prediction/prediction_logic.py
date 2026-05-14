"""
===================
prediction_logic.py
===================

ML/QSAR computation logic for the Prediction tab.
"""

from __future__ import annotations

import math
import os
import pickle
import re
import uuid
from datetime import datetime
from typing import Any
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import KFold

from app.gui.loading_win import set_loading_screen_progress
from app.utils.app_logger import log_event, log_settings


class PredictionAborted(Exception):
    """
    Cooperative cancellation raised when the user aborts Prediction.
    """


def prediction_feature_items() -> list[str]:
    """
    Return the feature sets exposed by the Prediction manager.
    """
    return [
        "Morgan (1024 bits)",
        "Morgan + RDKit descriptors",
        "Morgan (2048 bits) + RDKit descriptors",
    ]


def prediction_model_items() -> list[str]:
    """
    Return the model choices exposed by the Prediction manager.
    """
    return [
        "Auto (fast CV)",
        "Auto (extended CV)",
        "Random Forest",
        "Extra Trees",
        "HistGradient Boosting",
        "SVR",
        "MLP",
    ]


def prediction_available_model_items() -> list[str]:
    """
    Return the models that can actually run in the current environment.
    """
    return prediction_model_items()


def _abort_prediction(state: dict[str, Any]) -> None:
    """
    Mark the running prediction workflow as aborted.
    """
    state["_prediction_abort_requested"] = True


def _raise_if_prediction_aborted(state: dict[str, Any]) -> None:
    """
    Interrupt the workflow as soon as the user requests an abort.
    """
    if state.get("_prediction_abort_requested"):
        raise PredictionAborted("Prediction aborted by user.")


def _set_prediction_progress(state: dict[str, Any], progress: float) -> None:
    """
    Update the shared progress value and the global loading overlay.
    """
    clamped = max(0.0, min(100.0, float(progress)))
    state["_prediction_progress_value"] = clamped
    set_loading_screen_progress(state, clamped)


def _normalize_prediction_scope(scope: str) -> str:
    """
    Normalize legacy and current scope labels.
    """
    normalized = str(scope or "Dataset (prepared)").strip()
    if normalized == "Dataset":
        return "Dataset (prepared)"
    return normalized


def collect_prediction_activities(state: dict[str, Any], scope: str = "Dataset") -> list[str]:
    """
    Collect the bioactivity labels available for the selected prediction scope.
    """
    bioact_types_dict = state.get("bioact_types_dict", {})
    if not isinstance(bioact_types_dict, dict):
        return []

    normalized_scope = _normalize_prediction_scope(scope)
    bioactivity_scope = "Dataset" if normalized_scope in {"Dataset (full)", "Dataset (prepared)"} else normalized_scope
    scoped_bioactivities = bioact_types_dict.get(bioactivity_scope, {}).get("bioactivities", [])
    if not isinstance(scoped_bioactivities, list):
        return []

    seen: set[str] = set()
    ordered: list[str] = []
    for activity in scoped_bioactivities:
        if isinstance(activity, str) and activity and activity not in seen:
            seen.add(activity)
            ordered.append(activity)
    return ordered


def prediction_scope_items(state: dict[str, Any]) -> list[str]:
    """
    Build the available training/prediction scopes.
    """
    subset_keys = [
        key for key in (state.get("properties_dict", {}) or {}).keys()
        if isinstance(key, str) and key.startswith("subset_")
    ]
    try:
        subset_keys = sorted(subset_keys, key=lambda s: int(s.split("_")[-1]))
    except Exception:
        subset_keys = sorted(subset_keys)
    return ["Dataset (prepared)", "Dataset (full)"] + subset_keys


def prediction_target_label(activity_name: str, state: dict[str, Any]) -> str:
    """
    Return the display label for the selected target.
    """
    if activity_name in state.get("nM_activity_types", []):
        return f"p{activity_name}"
    return activity_name


def _scope_to_subsets(scope: str, state: dict[str, Any]) -> list[str]:
    """
    Resolve the selected scope into concrete subset keys.
    """
    normalized_scope = _normalize_prediction_scope(scope)
    if normalized_scope == "Dataset (full)":
        return list((state.get("properties_dict_full", {}) or {}).keys())
    if normalized_scope == "Dataset (prepared)":
        return [s for s in prediction_scope_items(state) if isinstance(s, str) and s.startswith("subset_")]
    return [normalized_scope] if normalized_scope in (state.get("properties_dict", {}) or {}) else []


def _parse_activity_measure(
    value_text: Any,
    expected_prefix: str,
) -> tuple[str, float] | None:
    """
    Parse a stored activity string such as 'IC50 > 300 nM' or 'pIC50 < 6.2'.
    """
    if not isinstance(value_text, str):
        return None

    pattern = rf"^\s*{re.escape(expected_prefix)}\s*([<>]=?|=)\s*([-+]?\d*\.?\d+)"
    match = re.match(pattern, value_text.strip())
    if not match:
        return None
    relation = match.group(1).strip()
    numeric_value = float(match.group(2))
    return relation, numeric_value


def _extract_selected_activity(
    mol_entry: dict[str, Any],
    activity_name: str,
    include_undefined: bool,
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract the selected activity from one molecule entry.
    """
    activities_map = mol_entry.get("activities", {}) or {}
    target_uses_pvalue = activity_name in state.get("nM_activity_types", [])
    preferred_prefix = f"p{activity_name}" if target_uses_pvalue else activity_name

    first_undefined_payload: dict[str, Any] | None = None

    for activity_block in activities_map.values():
        if not isinstance(activity_block, dict):
            continue

        parsed_primary = None
        raw_primary = None
        parsed_fallback = None
        raw_fallback = None

        for key, raw_value in activity_block.items():
            if not isinstance(raw_value, str):
                continue
            if key.startswith("pValue"):
                parsed = _parse_activity_measure(raw_value, f"p{activity_name}")
                if parsed is not None:
                    parsed_primary = parsed
                    raw_primary = raw_value.strip()
                    break
            elif key.startswith("Activity"):
                parsed = _parse_activity_measure(raw_value, activity_name)
                if parsed is not None:
                    parsed_fallback = parsed
                    raw_fallback = raw_value.strip()

        relation = None
        numeric_value = None
        raw_value_text = None
        source_scale = preferred_prefix

        if target_uses_pvalue and parsed_primary is not None:
            relation, numeric_value = parsed_primary
            raw_value_text = raw_primary
        elif parsed_fallback is not None:
            relation, numeric_value = parsed_fallback
            raw_value_text = raw_fallback
            if target_uses_pvalue and numeric_value is not None and numeric_value > 0:
                numeric_value = -math.log10(numeric_value * 1e-9)
                source_scale = f"p{activity_name}"
        else:
            continue

        if relation is None or numeric_value is None:
            continue

        is_undefined = relation != "="
        payload = {
            "available": True,
            "relation": relation,
            "value": float(numeric_value),
            "raw_value": raw_value_text,
            "is_undefined": bool(is_undefined),
            "scale_label": source_scale,
        }

        if not is_undefined:
            return payload
        if first_undefined_payload is None:
            first_undefined_payload = payload

    if include_undefined and first_undefined_payload is not None:
        return first_undefined_payload

    return {
        "available": False,
        "relation": None,
        "value": None,
        "raw_value": None,
        "is_undefined": False,
        "scale_label": preferred_prefix,
    }


def _build_feature_vector(
    smiles: str,
    feature_mode: str,
    mol_properties: dict[str, Any],
    state: dict[str, Any],
) -> np.ndarray | None:
    """
    Build the feature vector used by the regression model.
    """
    smiles = (smiles or "").strip()
    if not smiles:
        return None

    cache = state.setdefault("prediction_feature_cache", {})
    cache_key = (smiles, feature_mode)
    if cache_key in cache:
        return cache[cache_key]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp_size = 2048 if "2048" in feature_mode else 1024
    fp = GetMorganGenerator(radius=2, fpSize=fp_size, includeChirality=True).GetFingerprint(mol)
    fp_arr = np.zeros((fp.GetNumBits(),), dtype=float)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    if "RDKit descriptors" in feature_mode:
        descriptor_names = [
            "molecular_weight",
            "logp",
            "molar_refractivity",
            "gasteiger_range",
            "gasteiger_mean_abs",
            "tpsa",
            "hba",
            "hbd",
            "RotBonds",
            "fraction_csp3",
            "num_rings",
            "num_aromatic_rings",
            "num_aliphatic_rings",
            "num_saturated_rings",
            "kappa1",
            "kappa2",
            "kappa3",
            "chi0",
            "chi1",
            "chi2",
            "chi3",
            "chi4",
        ]

        descriptors = []
        for name in descriptor_names:
            value = mol_properties.get(name, 0.0)
            try:
                numeric_value = float(value)
                descriptors.append(numeric_value if math.isfinite(numeric_value) else 0.0)
            except Exception:
                descriptors.append(0.0)
        vec = np.concatenate([fp_arr, np.asarray(descriptors, dtype=float)])
    else:
        vec = fp_arr

    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    cache[cache_key] = vec
    return vec


def _collect_prediction_records_from_properties(
    scope: str,
    activity_name: str,
    feature_mode: str,
    include_undefined: bool,
    state: dict[str, Any],
    progress_callback: Any = None,
) -> list[dict[str, Any]]:
    """
    Assemble one normalized record per molecule from the prepared dataset state.
    """
    normalized_scope = _normalize_prediction_scope(scope)
    properties_dict = (
        state.get("properties_dict_full", {}) if normalized_scope == "Dataset (full)"
        else state.get("properties_dict", {})
    ) or {}

    def _entry_dedupe_key(props: dict[str, Any]) -> str | None:
        mol_id = props.get("original_id", -1)
        try:
            mol_id_int = int(mol_id)
        except Exception:
            mol_id_int = -1
        if mol_id_int >= 0:
            return f"mol_id:{mol_id_int}"
        smiles = str(props.get("smiles", "") or "").strip()
        if smiles:
            return f"smiles:{smiles}"
        return None
    subsets = _scope_to_subsets(scope, state)

    records: list[dict[str, Any]] = []
    total_molecules = 0
    for subset in subsets:
        subset_payload = properties_dict.get(subset, {}) or {}
        total_molecules += sum(1 for _, mol_entry in subset_payload.items() if isinstance(mol_entry, dict))

    processed = 0
    for subset in subsets:
        _raise_if_prediction_aborted(state)
        subset_payload = properties_dict.get(subset, {}) or {}
        for mol_key, mol_entry in subset_payload.items():
            _raise_if_prediction_aborted(state)
            if not isinstance(mol_entry, dict):
                continue

            processed += 1
            if callable(progress_callback) and total_molecules > 0:
                if processed == total_molecules or processed == 1 or processed % 10 == 0:
                    progress_callback(processed / total_molecules)

            props = mol_entry.get("properties", {}) or {}
            smiles = str(props.get("smiles", "") or "").strip()
            if not smiles:
                continue

            features = _build_feature_vector(smiles, feature_mode, props, state)
            if features is None:
                continue

            measure = _extract_selected_activity(mol_entry, activity_name, include_undefined, state)
            mol_id = props.get("original_id", -1)
            try:
                mol_id_int = int(mol_id)
            except Exception:
                mol_id_int = -1

            records.append({
                "record_key": f"{subset}:{mol_key}",
                "subset": subset,
                "subset_memberships": [subset],
                "mol_key": mol_key,
                "mol_id": mol_id_int,
                "name": str(props.get("name", "N/A")),
                "smiles": smiles,
                "features": features,
                "activity_available": bool(measure["available"]),
                "is_undefined": bool(measure["is_undefined"]),
                "relation": measure["relation"],
                "real_value": measure["value"],
                "real_raw": measure["raw_value"],
                "scale_label": measure["scale_label"],
            })

    if normalized_scope == "Dataset (prepared)":
        aggregated_records: dict[str, dict[str, Any]] = {}

        def _merge_prepared_duplicates(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
            memberships = list(existing.get("subset_memberships", []))
            for subset_name in candidate.get("subset_memberships", []):
                if subset_name not in memberships:
                    memberships.append(subset_name)
            existing["subset_memberships"] = memberships
            existing["subset"] = memberships[0] if len(memberships) == 1 else "Multiple subsets"

            existing_has_activity = bool(existing.get("activity_available")) and existing.get("real_value") is not None
            candidate_has_activity = bool(candidate.get("activity_available")) and candidate.get("real_value") is not None

            if candidate_has_activity and not existing_has_activity:
                merged = dict(candidate)
                merged["subset_memberships"] = memberships
                merged["subset"] = memberships[0] if len(memberships) == 1 else "Multiple subsets"
                return merged
            if not candidate_has_activity:
                return existing
            if not existing_has_activity:
                return existing

            existing_defined = not bool(existing.get("is_undefined"))
            candidate_defined = not bool(candidate.get("is_undefined"))

            if existing_defined and not candidate_defined:
                return existing
            if candidate_defined and not existing_defined:
                merged = dict(candidate)
                merged["subset_memberships"] = memberships
                merged["subset"] = memberships[0] if len(memberships) == 1 else "Multiple subsets"
                return merged

            avg_value = float(np.mean([float(existing["real_value"]), float(candidate["real_value"])]))
            existing["activity_available"] = True
            existing["real_value"] = avg_value
            existing["real_raw"] = None
            existing["relation"] = "="
            existing["is_undefined"] = False
            return existing

        for record in records:
            dedupe_key = _entry_dedupe_key(
                {
                    "original_id": record.get("mol_id", -1),
                    "smiles": record.get("smiles", ""),
                }
            )
            if not dedupe_key:
                dedupe_key = f"record:{record.get('record_key')}"
            if dedupe_key in aggregated_records:
                aggregated_records[dedupe_key] = _merge_prepared_duplicates(aggregated_records[dedupe_key], record)
            else:
                aggregated_records[dedupe_key] = dict(record)
        records = list(aggregated_records.values())

    records.sort(
        key=lambda record: (
            0 if isinstance(record.get("mol_id"), int) and int(record.get("mol_id")) >= 0 else 1,
            int(record.get("mol_id")) if isinstance(record.get("mol_id"), int) else 10**12,
            str(record.get("smiles", "") or ""),
            str(record.get("name", "") or ""),
            str(record.get("subset", "") or ""),
            str(record.get("record_key", "") or ""),
        )
    )
    return records
def _collect_prediction_records(
    scope: str,
    activity_name: str,
    feature_mode: str,
    include_undefined: bool,
    state: dict[str, Any],
    progress_callback: Any = None,
) -> list[dict[str, Any]]:
    """
    Assemble one normalized record per molecule for the prediction workflow.
    """
    normalized_scope = _normalize_prediction_scope(scope)
    return _collect_prediction_records_from_properties(
        scope=normalized_scope,
        activity_name=activity_name,
        feature_mode=feature_mode,
        include_undefined=include_undefined,
        state=state,
        progress_callback=progress_callback,
    )


def _build_regressor(model_name: str, training_size: int | None = None) -> Any:
    """
    Instantiate the selected regression model.
    """
    if model_name == "HistGradient Boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        effective_training_size = max(1, int(training_size or 0))
        min_samples_leaf = max(2, min(20, int(round(effective_training_size * 0.1)) or 2))
        max_leaf_nodes = max(15, min(63, effective_training_size * 2))

        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=0.1,
            random_state=42,
        )
    if model_name == "Extra Trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=700,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        )
    if model_name == "SVR":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        return make_pipeline(StandardScaler(), SVR(C=6.0, epsilon=0.15, kernel="rbf"))
    if model_name == "MLP":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                max_iter=900,
                random_state=42,
            ),
        )
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=700,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )


def _r2_score_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the coefficient of determination from predictions.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _prediction_model_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute comparable model-quality metrics on the same CV predictions.
    """
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    r2 = _r2_score_from_predictions(y_true, y_pred)
    target_sd = float(np.std(y_true, ddof=0))
    rmse_over_sd = float(rmse / target_sd) if target_sd > 1e-12 else float("inf")
    mae_over_sd = float(mae / target_sd) if target_sd > 1e-12 else float("inf")
    composite_score = float(r2 - (0.5 * rmse_over_sd) - (0.25 * mae_over_sd))
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "target_sd": target_sd,
        "rmse_over_sd": rmse_over_sd,
        "mae_over_sd": mae_over_sd,
        "composite_score": composite_score,
    }


def _prediction_error_scale(activity_name: str, state: dict[str, Any]) -> float:
    """
    Return a fixed absolute error scale so plot colors keep the same meaning
    across different analyses.
    """
    if activity_name in state.get("nM_activity_types", []):
        return 0.5
    return 30.0


def _prediction_high_match_threshold(activity_name: str, state: dict[str, Any]) -> float:
    """
    Return the fixed threshold used to count highly matching predictions.
    """
    if activity_name in state.get("nM_activity_types", []):
        return 0.1
    return 10.0


def _prediction_quality_score(abs_delta: float, activity_name: str, state: dict[str, Any]) -> float:
    """
    Map an absolute prediction error to a stable [0, 1] score.
    1.0 means perfect agreement, 0.0 means poor agreement.
    """
    error_scale = max(1e-12, float(_prediction_error_scale(activity_name, state)))
    return max(0.0, min(1.0, 1.0 - (float(abs_delta) / error_scale)))


def _load_saved_prediction_model_if_needed(state: dict[str, Any]) -> Any | None:
    """
    Return the currently fitted model, loading it from disk if necessary.
    """
    model = state.get("prediction_current_model")
    if model is not None:
        return model

    selected_model_name = str(
        state.get("prediction_selected_model_name")
        or state.get("prediction_active_model_view")
        or state.get("prediction_model_name")
        or ""
    )
    model_views = state.get("prediction_model_views", {}) or {}
    if selected_model_name and isinstance(model_views.get(selected_model_name), dict):
        model = model_views[selected_model_name].get("final_model")
        if model is not None:
            state["prediction_current_model"] = model
            return model
    return None


def _store_prediction_base_state(
    state: dict[str, Any],
    results: list[dict[str, Any]],
    results_map: dict[str, dict[str, Any]],
    plot_points: list[dict[str, Any]],
    metrics: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
) -> None:
    """
    Persist the original prediction output so external compounds can be added
    without overwriting the baseline results or the model comparison table.
    """
    base_results = [dict(record, row_origin="original") for record in results]
    base_map = {
        str(record.get("record_key")): base_record
        for record, base_record in zip(results, base_results)
        if record.get("record_key") is not None
    }
    base_plot_keys = {
        str(point.get("record_key")) for point in plot_points if point.get("record_key") is not None
    }
    base_plot_points = [
        base_map[str(record.get("record_key"))]
        for record in results
        if str(record.get("record_key")) in base_plot_keys
    ]
    state["prediction_base_results"] = base_results
    state["prediction_base_results_map"] = base_map
    state["prediction_base_plot_points"] = base_plot_points
    state["prediction_base_metrics"] = dict(metrics)
    state["prediction_base_model_comparison"] = [dict(row) for row in comparison_rows]
    state["prediction_external_results"] = []
    state["prediction_external_results_map"] = {}
    state["prediction_external_plot_points"] = []
    state["prediction_origin_filter"] = "All compounds"


def _rebuild_prediction_merged_state(state: dict[str, Any]) -> None:
    """
    Merge original and external prediction rows into the visible prediction state
    while keeping the original metrics/comparison untouched.
    """
    base_results = [dict(record) for record in (state.get("prediction_base_results", []) or [])]
    external_results = [dict(record) for record in (state.get("prediction_external_results", []) or [])]
    merged_results = base_results + external_results
    merged_map: dict[str, dict[str, Any]] = {}
    for record in merged_results:
        record_key = record.get("record_key")
        if record_key is not None:
            merged_map[str(record_key)] = record

    plot_keys = {
        str(point.get("record_key"))
        for point in (state.get("prediction_base_plot_points", []) or []) + (state.get("prediction_external_plot_points", []) or [])
        if point.get("record_key") is not None
    }
    merged_plot_points = [
        merged_map[str(record.get("record_key"))]
        for record in merged_results
        if str(record.get("record_key")) in plot_keys
    ]

    state["prediction_results"] = merged_results
    state["prediction_results_map"] = merged_map
    state["prediction_plot_points"] = merged_plot_points
    if merged_results:
        existing_selected = str(state.get("prediction_selected_record_key") or "")
        if existing_selected not in merged_map:
            state["prediction_selected_record_key"] = merged_results[0]["record_key"]
    else:
        state["prediction_selected_record_key"] = None


def _build_prediction_output_bundle(
    *,
    train_records: list[dict[str, Any]],
    missing_records: list[dict[str, Any]],
    cv_predictions: np.ndarray,
    final_model: Any,
    activity_name: str,
    dataset_size: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the full visible Prediction output for one fitted model.
    """
    visible_results: list[dict[str, Any]] = []
    results_map: dict[str, dict[str, Any]] = {}
    plot_points: list[dict[str, Any]] = []

    final_train_predictions = final_model.predict(np.vstack([r["features"] for r in train_records]))
    for idx, record in enumerate(train_records):
        cv_pred = float(cv_predictions[idx])
        final_pred = float(final_train_predictions[idx])
        abs_delta = abs(cv_pred - float(record["real_value"]))
        enriched = dict(record)
        enriched["predicted_value"] = cv_pred
        enriched["cv_predicted_value"] = cv_pred
        enriched["final_predicted_value"] = final_pred
        enriched["status"] = "Cross-validated"
        enriched["quality_score"] = _prediction_quality_score(abs_delta, activity_name, state)
        enriched["absolute_delta"] = abs_delta
        visible_results.append(enriched)
        results_map[str(enriched["record_key"])] = enriched
        plot_points.append(enriched)

    if missing_records:
        X_missing = np.vstack([r["features"] for r in missing_records])
        missing_predictions = final_model.predict(X_missing)
        for idx, record in enumerate(missing_records):
            final_pred = float(missing_predictions[idx])
            enriched = dict(record)
            enriched["predicted_value"] = final_pred
            enriched["cv_predicted_value"] = None
            enriched["final_predicted_value"] = final_pred
            enriched["status"] = "Predicted"
            enriched["quality_score"] = None
            enriched["absolute_delta"] = None
            visible_results.append(enriched)
            results_map[str(enriched["record_key"])] = enriched

    visible_results.sort(
        key=lambda r: (
            r.get("subset", ""),
            int(r.get("mol_id", -1)) if isinstance(r.get("mol_id", -1), int) else -1,
        )
    )

    summary = _prediction_model_summary(
        np.asarray([float(r["real_value"]) for r in train_records], dtype=float),
        cv_predictions,
    )
    high_match_threshold = _prediction_high_match_threshold(activity_name, state)
    high_match_count = int(
        np.sum(
            np.abs(
                cv_predictions - np.asarray([float(r["real_value"]) for r in train_records], dtype=float)
            ) <= max(high_match_threshold, 1e-9)
        )
    )

    return {
        "results": visible_results,
        "results_map": results_map,
        "plot_points": plot_points,
        "metrics": {
            "dataset_size": dataset_size,
            "training_size": len(train_records),
            "predicted_size": len(missing_records),
            "mae": summary["mae"],
            "rmse": summary["rmse"],
            "r2": summary["r2"],
            "target_sd": summary["target_sd"],
            "rmse_over_sd": summary["rmse_over_sd"],
            "mae_over_sd": summary["mae_over_sd"],
            "composite_score": summary["composite_score"],
            "high_match_count": high_match_count,
            "high_match_threshold": high_match_threshold,
        },
        "final_model": final_model,
    }


def activate_prediction_model_view(model_name: str, state: dict[str, Any]) -> bool:
    """
    Activate one stored model-output bundle produced by Prediction auto mode.
    """
    model_views = state.get("prediction_model_views", {}) or {}
    bundle = model_views.get(model_name)
    if not isinstance(bundle, dict):
        return False

    state["prediction_results"] = [dict(record) for record in bundle.get("results", [])]
    state["prediction_results_map"] = {
        str(key): dict(value) for key, value in (bundle.get("results_map", {}) or {}).items()
    }
    state["prediction_plot_points"] = [
        state["prediction_results_map"][str(record.get("record_key"))]
        for record in bundle.get("plot_points", [])
        if str(record.get("record_key")) in state["prediction_results_map"]
    ]
    state["prediction_metrics"] = dict(bundle.get("metrics", {}) or {})
    state["prediction_current_model"] = bundle.get("final_model")
    state["prediction_active_model_view"] = model_name
    state["prediction_selected_model_name"] = model_name
    state["prediction_model_name"] = model_name
    state["prediction_selected_record_key"] = (
        state["prediction_results"][0]["record_key"] if state.get("prediction_results") else None
    )
    state["prediction_table_order"] = [
        str(record.get("record_key"))
        for record in state.get("prediction_results", [])
        if record.get("record_key") is not None
    ]
    return True


def run_prediction_on_external_records(
    state: dict[str, Any],
    external_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Apply the current fitted prediction model to external compounds and append
    them to the visible prediction results.
    """
    try:
        model = _load_saved_prediction_model_if_needed(state)
        if model is None:
            return {"ok": False, "message": "No fitted prediction model is available."}

        model_metadata = state.get("prediction_model_metadata", {}) or {}
        activity_name = str(
            model_metadata.get("activity_name")
            or state.get("prediction_activity_name", "")
            or ""
        ).strip()
        feature_mode = str(
            model_metadata.get("feature_mode")
            or state.get("prediction_feature_mode", "Morgan (1024 bits)")
            or "Morgan (1024 bits)"
        )
        if not activity_name:
            return {"ok": False, "message": "No prediction activity is currently selected."}

        existing_external = list(state.get("prediction_external_results", []) or [])
        next_external_index = len(existing_external) + 1

        processed_records: list[dict[str, Any]] = []
        features_batch: list[np.ndarray] = []
        predictable_indices: list[int] = []

        for idx, entry in enumerate(external_entries, start=next_external_index):
            smiles = str(entry.get("smiles", "") or "").strip()
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            canonical_smiles = Chem.MolToSmiles(mol)
            from rdkit.Chem import AllChem, Crippen, Lipinski, rdMolDescriptors

            mol_h = Chem.AddHs(mol, addCoords=True)
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for atom in mol.GetAtoms():
                try:
                    charges.append(atom.GetDoubleProp("_GasteigerCharge"))
                except Exception:
                    continue
            gasteiger_range = max(charges) - min(charges) if charges else 0.0
            gasteiger_mean_abs = sum(abs(q) for q in charges) / len(charges) if charges else 0.0
            props = {
                "smiles": canonical_smiles,
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
            features = _build_feature_vector(canonical_smiles, feature_mode, props, state)
            if features is None:
                continue

            real_value = entry.get("real_value")
            record = {
                "record_key": f"external:{idx}:{uuid.uuid4().hex[:8]}",
                "subset": "External",
                "subset_memberships": ["External"],
                "row_origin": "external",
                "mol_key": f"external_{idx}",
                "mol_id": idx,
                "name": str(entry.get("name", "") or f"External {idx}"),
                "smiles": canonical_smiles,
                "features": features,
                "activity_available": real_value is not None,
                "is_undefined": False,
                "relation": entry.get("relation") if real_value is not None else None,
                "real_value": float(real_value) if real_value is not None else None,
                "real_raw": entry.get("real_raw"),
                "scale_label": prediction_target_label(activity_name, state),
            }
            processed_records.append(record)
            features_batch.append(features)
            predictable_indices.append(len(processed_records) - 1)

        if not processed_records:
            return {"ok": False, "message": "No valid external compounds were provided."}

        X_ext = np.vstack(features_batch)
        y_pred = model.predict(X_ext)
        unique_predictions = {round(float(value), 12) for value in y_pred}
        if len(processed_records) > 1 and len(unique_predictions) == 1:
            log_event(
                "Prediction",
                (
                    "All external compounds received the same predicted value. "
                    "This usually means the fitted model is behaving as a near-constant predictor for this feature space."
                ),
                indent=2,
                level="WARNING",
            )

        original_prediction_by_smiles: dict[str, dict[str, Any]] = {}
        for base_record in state.get("prediction_base_results", []) or []:
            smiles_key = str(base_record.get("smiles", "") or "").strip()
            if smiles_key:
                original_prediction_by_smiles.setdefault(smiles_key, dict(base_record))

        results_map: dict[str, dict[str, Any]] = {}
        plot_points: list[dict[str, Any]] = []
        for idx, pred_val in zip(predictable_indices, y_pred):
            record = processed_records[idx]
            smiles_key = str(record.get("smiles", "") or "").strip()
            matching_original = original_prediction_by_smiles.get(smiles_key)
            final_pred_val = float(
                matching_original.get("final_predicted_value", matching_original.get("predicted_value"))
            ) if isinstance(matching_original, dict) and isinstance(
                matching_original.get("final_predicted_value", matching_original.get("predicted_value")),
                (int, float),
            ) else float(pred_val)
            cv_pred_val = None
            if isinstance(matching_original, dict) and isinstance(
                matching_original.get("cv_predicted_value", matching_original.get("predicted_value")),
                (int, float),
            ):
                cv_pred_val = float(matching_original.get("cv_predicted_value", matching_original.get("predicted_value")))
            record["cv_predicted_value"] = cv_pred_val
            record["final_predicted_value"] = final_pred_val
            record["predicted_value"] = final_pred_val
            if record.get("real_value") is not None:
                abs_delta = abs(final_pred_val - float(record["real_value"]))
                record["quality_score"] = _prediction_quality_score(abs_delta, activity_name, state)
                record["absolute_delta"] = abs_delta
                record["status"] = "External compared"
                plot_points.append(record)
            else:
                record["quality_score"] = None
                record["absolute_delta"] = None
                record["status"] = "External predicted"
            results_map[str(record["record_key"])] = record

        y_true = np.asarray([float(r["real_value"]) for r in plot_points], dtype=float) if plot_points else np.asarray([], dtype=float)
        y_pred_known = np.asarray([float(r["predicted_value"]) for r in plot_points], dtype=float) if plot_points else np.asarray([], dtype=float)
        summary = _prediction_model_summary(y_true, y_pred_known) if len(y_true) else {
            "r2": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "target_sd": 0.0,
            "rmse_over_sd": 0.0,
            "mae_over_sd": 0.0,
            "composite_score": 0.0,
        }
        high_match_threshold = _prediction_high_match_threshold(activity_name, state)
        high_match_count = int(np.sum(np.abs(y_pred_known - y_true) <= max(high_match_threshold, 1e-9))) if len(y_true) else 0

        if not state.get("prediction_base_results"):
            current_results = list(state.get("prediction_results", []) or [])
            current_map = dict(state.get("prediction_results_map", {}) or {})
            current_plot = list(state.get("prediction_plot_points", []) or [])
            current_metrics = dict(state.get("prediction_metrics", {}) or {})
            current_comparison = list(state.get("prediction_model_comparison", []) or [])
            if current_results:
                _store_prediction_base_state(
                    state,
                    current_results,
                    current_map,
                    current_plot,
                    current_metrics,
                    current_comparison,
                )

        existing_external_map = dict(state.get("prediction_external_results_map", {}) or {})
        existing_external_plot = list(state.get("prediction_external_plot_points", []) or [])
        existing_external.extend(processed_records)
        existing_external_map.update(results_map)
        existing_external_plot.extend(plot_points)
        state["prediction_external_results"] = existing_external
        state["prediction_external_results_map"] = existing_external_map
        state["prediction_external_plot_points"] = existing_external_plot
        _rebuild_prediction_merged_state(state)
        state["prediction_selected_record_key"] = processed_records[0]["record_key"] if processed_records else state.get("prediction_selected_record_key")
        state["prediction_status_message"] = ""
        state["prediction_filter_query"] = ""
        state["prediction_table_page"] = 1
        state["prediction_sort_spec"] = None
        state["prediction_table_order"] = [
            str(record.get("record_key")) for record in state.get("prediction_results", []) if record.get("record_key") is not None
        ]
        state["prediction_origin_filter"] = "All compounds"
        state["prediction_external_metrics"] = {
            "dataset_size": len(processed_records),
            "training_size": len(plot_points),
            "predicted_size": sum(1 for r in processed_records if r.get("real_value") is None),
            "mae": summary["mae"],
            "rmse": summary["rmse"],
            "r2": summary["r2"],
            "target_sd": summary["target_sd"],
            "rmse_over_sd": summary["rmse_over_sd"],
            "mae_over_sd": summary["mae_over_sd"],
            "composite_score": summary["composite_score"],
            "high_match_count": high_match_count,
            "high_match_threshold": high_match_threshold,
        }
        return {"ok": True, "message": ""}
    except Exception as exc:
        return {"ok": False, "message": f"External prediction failed: {exc}"}


def _cross_validated_predictions(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: KFold,
    progress_callback: Any = None,
    state: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Run fold-by-fold cross-validation predictions with progress updates.
    """
    from sklearn.base import clone

    predictions = np.zeros(len(y_train), dtype=float)
    splits = list(cv.split(X_train, y_train))
    total_folds = max(1, len(splits))

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        if state is not None:
            _raise_if_prediction_aborted(state)
        fold_start = (fold_idx - 1) / total_folds
        fold_end = fold_idx / total_folds
        fit_end = fold_start + ((fold_end - fold_start) * 0.9)
        fold_model = _fit_model_with_progress(
            clone(model),
            X_train[train_idx],
            y_train[train_idx],
            progress_callback=progress_callback,
            progress_start=fold_start,
            progress_end=fit_end,
        )
        predictions[test_idx] = fold_model.predict(X_train[test_idx])
        if callable(progress_callback):
            progress_callback(fold_end)
    return predictions


def _prediction_auto_candidates(requested_model_name: str) -> list[str]:
    """
    Resolve the candidate models for one of the automatic CV modes.
    """
    if requested_model_name == "Auto (fast CV)":
        return ["Extra Trees", "Random Forest"]
    if requested_model_name == "Auto (extended CV)":
        return [
            "Extra Trees",
            "Random Forest",
            "HistGradient Boosting",
            "SVR",
            "MLP",
        ]
    return []


def _prediction_safe_filename_part(value: Any) -> str:
    """
    Convert one metadata value into a filesystem-friendly filename fragment.
    """
    text = str(value or "NA").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text or "NA"


def _prediction_job_name(state: dict[str, Any]) -> str:
    """
    Return the current job name used in exported prediction artifacts.
    """
    work_dir = str(state.get("work_dir", "") or "").strip()
    if work_dir:
        return os.path.basename(work_dir.rstrip(os.sep)) or "NA"
    return "NA"


def _prediction_session_dir(state: dict[str, Any]) -> str | None:
    """
    Return the directory used to persist reloadable prediction sessions.
    """
    output_dir = str(
        state.get("predictions_dir")
        or (state.get("settings", {}) or {}).get("predictions_directory", "")
        or ""
    ).strip()
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "data", "predictions")
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_prediction_session_artifact(
    state: dict[str, Any],
    *,
    requested_model_name: str,
    resolved_model_name: str,
    scope: str,
    activity_name: str,
    feature_mode: str,
    include_undefined: bool,
    training_size: int,
    dataset_size: int,
) -> str | None:
    """
    Persist the whole Prediction session so it can be reloaded later with all
    model views and fitted models intact.
    """
    session_dir = _prediction_session_dir(state)
    if not session_dir:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = (
        f"predsession__"
        f"{timestamp}__"
        f"{_prediction_safe_filename_part(_prediction_job_name(state))}__"
        f"{_prediction_safe_filename_part(activity_name)}__"
        f"{_prediction_safe_filename_part(scope)}__"
        f"{_prediction_safe_filename_part(requested_model_name)}__"
        f"{_prediction_safe_filename_part(feature_mode)}__"
        f"undef-{int(bool(include_undefined))}__"
        f"train-{int(training_size)}__"
        f"data-{int(dataset_size)}__"
        f".pkl"
    )
    output_path = os.path.join(session_dir, filename)

    payload = {
        "metadata": {
            "requested_model_name": requested_model_name,
            "resolved_model_name": resolved_model_name,
            "scope": scope,
            "activity_name": activity_name,
            "target_label": state.get("prediction_target_label"),
            "feature_mode": feature_mode,
            "include_undefined": bool(include_undefined),
            "training_size": int(training_size),
            "dataset_size": int(dataset_size),
            "job_name": _prediction_job_name(state),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "work_dir": str(state.get("work_dir", "") or "").strip(),
        },
        "prediction_model_comparison": list(state.get("prediction_model_comparison", []) or []),
        "prediction_model_views": dict(state.get("prediction_model_views", {}) or {}),
        "prediction_model_metadata": dict(state.get("prediction_model_metadata", {}) or {}),
        "prediction_active_model_view": state.get("prediction_active_model_view"),
        "prediction_selected_model_name": state.get("prediction_selected_model_name"),
        "prediction_model_name": state.get("prediction_model_name"),
        "prediction_results": list(state.get("prediction_results", []) or []),
        "prediction_results_map": dict(state.get("prediction_results_map", {}) or {}),
        "prediction_plot_points": list(state.get("prediction_plot_points", []) or []),
        "prediction_metrics": dict(state.get("prediction_metrics", {}) or {}),
        "prediction_selected_record_key": state.get("prediction_selected_record_key"),
        "prediction_target_label": state.get("prediction_target_label"),
        "prediction_activity_name": state.get("prediction_activity_name"),
        "prediction_scope": state.get("prediction_scope"),
        "prediction_feature_mode": state.get("prediction_feature_mode"),
        "prediction_include_undefined": state.get("prediction_include_undefined"),
        "prediction_display_linear": bool(state.get("prediction_display_linear", False)),
        "prediction_table_order": list(state.get("prediction_table_order", []) or []),
        "prediction_base_results": list(state.get("prediction_base_results", []) or []),
        "prediction_base_results_map": dict(state.get("prediction_base_results_map", {}) or {}),
        "prediction_base_plot_points": list(state.get("prediction_base_plot_points", []) or []),
        "prediction_base_metrics": dict(state.get("prediction_base_metrics", {}) or {}),
        "prediction_base_model_comparison": list(state.get("prediction_base_model_comparison", []) or []),
        "prediction_external_results": list(state.get("prediction_external_results", []) or []),
        "prediction_external_results_map": dict(state.get("prediction_external_results_map", {}) or {}),
        "prediction_external_plot_points": list(state.get("prediction_external_plot_points", []) or []),
        "prediction_origin_filter": state.get("prediction_origin_filter", "All compounds"),
    }

    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path


def _fit_model_with_progress(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    progress_callback: Any = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
    state: dict[str, Any] | None = None,
) -> Any:
    """
    Fit a model while emitting the most granular progress updates available.
    """
    if callable(progress_callback):
        progress_callback(progress_start)

    if model.__class__.__name__ in {"RandomForestRegressor", "ExtraTreesRegressor"}:
        total_estimators = max(1, int(getattr(model, "n_estimators", 100)))
        chunk_size = max(1, min(5, total_estimators // 40 or 1))
        model.set_params(warm_start=True, n_estimators=0)
        built_estimators = 0
        while built_estimators < total_estimators:
            if state is not None:
                _raise_if_prediction_aborted(state)
            built_estimators = min(total_estimators, built_estimators + chunk_size)
            model.set_params(n_estimators=built_estimators)
            model.fit(X_train, y_train)
            if callable(progress_callback):
                ratio = built_estimators / total_estimators
                progress_callback(progress_start + ((progress_end - progress_start) * ratio))
        return model

    if state is not None:
        _raise_if_prediction_aborted(state)
    if callable(progress_callback):
        progress_callback(progress_start + ((progress_end - progress_start) * 0.25))
    model.fit(X_train, y_train)
    if state is not None:
        _raise_if_prediction_aborted(state)
    if callable(progress_callback):
        progress_callback(progress_end)
    return model


def run_prediction_analysis(state: dict[str, Any], options: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Train the selected regression model and populate prediction state.
    """
    try:
        state["_prediction_abort_requested"] = False
        _set_prediction_progress(state, 0)
        options = options or {}
        activity_name = str(options.get("activity_name") or "").strip()
        scope = _normalize_prediction_scope(str(options.get("scope") or "Dataset (prepared)"))
        model_name = str(options.get("model_name") or "Random Forest")
        feature_mode = str(options.get("feature_mode") or "Morgan (1024 bits)")
        include_undefined = bool(options.get("include_undefined"))
        target_label = prediction_target_label(activity_name, state)
        records = _collect_prediction_records(
            scope=scope,
            activity_name=activity_name,
            feature_mode=feature_mode,
            include_undefined=include_undefined,
            state=state,
            progress_callback=lambda ratio: _set_prediction_progress(state, 1 + (ratio * 3)),
        )
        _set_prediction_progress(state, 4)
        _raise_if_prediction_aborted(state)

        train_records = [r for r in records if r.get("activity_available") and r.get("real_value") is not None]
        missing_records = [r for r in records if not r.get("activity_available")]

        log_event("Prediction", "Running prediction workflow", indent=1)
        log_settings(
            "Prediction",
            indent=2,
            scope=scope,
            activity=activity_name,
            target_label=target_label,
            model=model_name,
            features=feature_mode,
            include_undefined=include_undefined,
            molecules=len(records),
            training=len(train_records),
            missing=len(missing_records),
        )

        state["prediction_target_label"] = target_label
        state["prediction_activity_name"] = activity_name
        state["prediction_scope"] = scope
        requested_model_name = model_name
        state["prediction_model_name"] = requested_model_name
        state["prediction_feature_mode"] = feature_mode
        state["prediction_include_undefined"] = include_undefined
        state["prediction_results"] = []
        state["prediction_results_map"] = {}
        state["prediction_plot_points"] = []
        state["prediction_selected_record_key"] = None
        state["prediction_status_message"] = ""
        state["prediction_metrics"] = {}
        state["prediction_model_comparison"] = []
        state["prediction_model_views"] = {}
        state["prediction_active_model_view"] = None
        state["prediction_selected_model_name"] = None

        if len(train_records) < 3:
            state["prediction_status_message"] = (
                "Not enough molecules with known activity to train a regression model.\n"
                "At least 3 labeled molecules are required."
            )
            _set_prediction_progress(state, 100)
            return {"ok": False, "message": state["prediction_status_message"]}

        X_train = np.vstack([r["features"] for r in train_records])
        y_train = np.asarray([float(r["real_value"]) for r in train_records], dtype=float)
        _set_prediction_progress(state, 5)

        n_splits = max(2, min(5, len(train_records)))
        from sklearn.model_selection import KFold

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        resolved_model_name = requested_model_name
        if requested_model_name in {"Auto (fast CV)", "Auto (extended CV)"}:
            candidate_names = _prediction_auto_candidates(requested_model_name)
            best_candidate_name = candidate_names[0]
            best_candidate_score = float("-inf")
            best_cv_predictions: np.ndarray | None = None
            comparison_rows: list[dict[str, Any]] = []
            candidate_views: dict[str, dict[str, Any]] = {}

            log_event("Prediction", "Auto model selection enabled (composite score)", indent=2)
            for candidate_idx, candidate_name in enumerate(candidate_names):
                try:
                    candidate_model = _build_regressor(candidate_name, training_size=len(train_records))
                except Exception as exc:
                    log_event(
                        "Prediction",
                        f"Skipping {candidate_name}: {exc}",
                        indent=3,
                        level="WARNING",
                    )
                    continue
                start_progress = 5 + ((candidate_idx / len(candidate_names)) * 85)
                end_progress = 5 + (((candidate_idx + 1) / len(candidate_names)) * 85)
                cv_end_progress = start_progress + ((end_progress - start_progress) * 0.82)
                candidate_cv_predictions = _cross_validated_predictions(
                    model=candidate_model,
                    X_train=X_train,
                    y_train=y_train,
                    cv=cv,
                    progress_callback=lambda ratio, sp=start_progress, ep=cv_end_progress: _set_prediction_progress(
                        state, sp + (ratio * (ep - sp))
                    ),
                    state=state,
                )
                candidate_summary = _prediction_model_summary(y_train, candidate_cv_predictions)
                comparison_row = {"model": candidate_name, **candidate_summary}
                comparison_rows.append(comparison_row)
                log_event(
                    "Prediction",
                    (
                        f"{candidate_name}: score = {candidate_summary['composite_score']:.3f}, "
                        f"R² = {candidate_summary['r2']:.3f}, "
                        f"RMSE/SD = {candidate_summary['rmse_over_sd']:.3f}, "
                        f"MAE/SD = {candidate_summary['mae_over_sd']:.3f}"
                    ),
                    indent=3,
                )
                candidate_final_model = _fit_model_with_progress(
                    candidate_model,
                    X_train,
                    y_train,
                    progress_callback=lambda ratio, sp=cv_end_progress, ep=end_progress: _set_prediction_progress(
                        state, sp + (ratio * (ep - sp))
                    ),
                    progress_start=0.0,
                    progress_end=1.0,
                    state=state,
                )
                candidate_views[candidate_name] = _build_prediction_output_bundle(
                    train_records=train_records,
                    missing_records=missing_records,
                    cv_predictions=candidate_cv_predictions,
                    final_model=candidate_final_model,
                    activity_name=activity_name,
                    dataset_size=len(records),
                    state=state,
                )
                if candidate_summary["composite_score"] > best_candidate_score:
                    best_candidate_score = candidate_summary["composite_score"]
                    best_candidate_name = candidate_name
                    best_cv_predictions = candidate_cv_predictions

            if best_cv_predictions is None:
                state["prediction_status_message"] = (
                    "No prediction model could be evaluated in the selected auto mode.\n"
                    "Check whether optional ML libraries are installed."
                )
                _set_prediction_progress(state, 100)
                return {"ok": False, "message": state["prediction_status_message"]}

            resolved_model_name = best_candidate_name
            cv_predictions = best_cv_predictions if best_cv_predictions is not None else np.zeros_like(y_train)
            for row in comparison_rows:
                row["selected"] = (row.get("model") == resolved_model_name)
            state["prediction_model_name"] = resolved_model_name
            state["prediction_model_comparison"] = comparison_rows
            state["prediction_model_views"] = candidate_views
            log_event("Prediction", f"Selected model: {resolved_model_name}", indent=2, level="SUCCESS")
        else:
            model = _build_regressor(requested_model_name, training_size=len(train_records))
            cv_predictions = _cross_validated_predictions(
                model=model,
                X_train=X_train,
                y_train=y_train,
                cv=cv,
                progress_callback=lambda ratio: _set_prediction_progress(state, 5 + (ratio * 85)),
                state=state,
            )
            state["prediction_model_comparison"] = [{"model": requested_model_name, "selected": True, **_prediction_model_summary(y_train, cv_predictions)}]
        _set_prediction_progress(state, 90)
        _raise_if_prediction_aborted(state)

        from sklearn.base import clone

        if requested_model_name in {"Auto (fast CV)", "Auto (extended CV)"}:
            final_model = ((state.get("prediction_model_views", {}) or {}).get(resolved_model_name, {}) or {}).get("final_model")
            if final_model is None:
                return {"ok": False, "message": "The selected auto model could not be restored."}
        else:
            final_model = _fit_model_with_progress(
                clone(model),
                X_train,
                y_train,
                progress_callback=lambda ratio: _set_prediction_progress(state, 90 + (ratio * 8)),
                progress_start=0.0,
                progress_end=1.0,
                state=state,
            )
        state["prediction_current_model"] = final_model
        state["prediction_model_metadata"] = {
            "requested_model_name": requested_model_name,
            "resolved_model_name": resolved_model_name,
            "scope": scope,
            "activity_name": activity_name,
            "target_label": target_label,
            "feature_mode": feature_mode,
            "include_undefined": bool(include_undefined),
            "training_size": int(len(train_records)),
            "dataset_size": int(len(records)),
        }
        _set_prediction_progress(state, 98)
        _raise_if_prediction_aborted(state)

        selected_summary = _prediction_model_summary(y_train, cv_predictions)
        _set_prediction_progress(state, 98.4)
        if requested_model_name in {"Auto (fast CV)", "Auto (extended CV)"}:
            activate_prediction_model_view(resolved_model_name, state)
        else:
            manual_bundle = _build_prediction_output_bundle(
                train_records=train_records,
                missing_records=missing_records,
                cv_predictions=cv_predictions,
                final_model=final_model,
                activity_name=activity_name,
                dataset_size=len(records),
                state=state,
            )
            state["prediction_model_views"] = {requested_model_name: manual_bundle}
            activate_prediction_model_view(requested_model_name, state)
        _store_prediction_base_state(
            state,
            list(state.get("prediction_results", []) or []),
            dict(state.get("prediction_results_map", {}) or {}),
            list(state.get("prediction_plot_points", []) or []),
            state["prediction_metrics"],
            list(state.get("prediction_model_comparison", []) or []),
        )
        try:
            saved_session_path = _save_prediction_session_artifact(
                state,
                requested_model_name=requested_model_name,
                resolved_model_name=resolved_model_name,
                scope=scope,
                activity_name=activity_name,
                feature_mode=feature_mode,
                include_undefined=include_undefined,
                training_size=len(train_records),
                dataset_size=len(records),
            )
            if saved_session_path:
                state["prediction_session_path"] = saved_session_path
                log_event("Prediction", f"Saved prediction session: {saved_session_path}", indent=2, level="SUCCESS")
        except Exception as exc:
            log_event("Prediction", f"Could not save prediction session: {exc}", indent=2, level="WARNING")
        _set_prediction_progress(state, 99.9)
        _set_prediction_progress(state, 100)
        return {"ok": True, "message": state["prediction_status_message"]}
    except PredictionAborted as e:
        state["prediction_status_message"] = str(e)
        log_event("Prediction", "Prediction aborted by user.", indent=1)
        return {"ok": False, "message": state["prediction_status_message"]}
    except Exception as e:
        state["prediction_status_message"] = f"Prediction failed: {e}"
        _set_prediction_progress(state, 100)
        return {"ok": False, "message": state["prediction_status_message"]}
    finally:
        state["_prediction_abort_requested"] = False
