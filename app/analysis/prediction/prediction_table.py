"""
===================
prediction_table.py
===================

GUI output widgets for the Prediction tab: results table, regression plot,
and selected molecule details.
"""

import io
import math
import csv
import os
import pickle
import re
from datetime import datetime
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image as pilImage
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from app.gui.themes_manager import (
    apply_plot_theme,
    get_continuous_colormap_color,
)
from app.gui.loading_win import draw_loading_screen
from app.utils.callbacks import register_plot_context_popup, register_responsive_image
from app.utils.app_logger import log_event, log_exception, log_traceback
from app.utils.native_dialogs import open_file_dialog, save_file_dialog
from app.analysis.prediction.prediction_logic import (
    run_prediction_on_external_records,
    _store_prediction_base_state,
    activate_prediction_model_view,
)


def _discrete_plot_color(state: dict[str, Any], ratio: float) -> tuple[int, int, int, int]:
    """
    Sample the active discrete plot colormap.
    """
    rgba = dpg.sample_colormap(state["plot_colormaps"][state["colormap_discrete"]], max(0.0, min(1.0, float(ratio))))
    if max(rgba[0], rgba[1], rgba[2]) <= 1.0:
        return (
            int(round(rgba[0] * 255)),
            int(round(rgba[1] * 255)),
            int(round(rgba[2] * 255)),
            int(round((rgba[3] if len(rgba) > 3 else 1.0) * 255)),
        )
    return (
        int(round(rgba[0])),
        int(round(rgba[1])),
        int(round(rgba[2])),
        int(round(rgba[3] if len(rgba) > 3 else 255)),
    )


def _prediction_display_color(record: dict[str, Any], state: dict[str, Any]) -> tuple[int, int, int, int]:
    """
    Return the color used both in the table and in the plot.
    """
    quality = _prediction_display_quality_score(record, state)
    if isinstance(quality, (int, float)):
        return get_continuous_colormap_color(float(quality), state)
    return _discrete_plot_color(state, 0.0)


def _prediction_selected_model_label(state: dict[str, Any]) -> str:
    """
    Return the compact selected-model label for the summary line.
    """
    model_name = str(state.get("prediction_model_name", "N/A") or "N/A")
    if "->" in model_name:
        return model_name.split("->", 1)[1].strip()
    return model_name


def _ensure_prediction_model_comparison_theme() -> str:
    """
    Create a compact table theme for the model-comparison table.
    """
    tag = "prediction_model_comparison_table_theme"
    if dpg.does_item_exist(tag):
        return tag
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvTable):
            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 5, 5, category=dpg.mvThemeCat_Core)
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 0, category=dpg.mvThemeCat_Core)
    return tag


def _add_prediction_header_tooltip(parent_tag: str, text: str) -> None:
    """
    Attach a compact explanatory tooltip to one table header item.
    """
    if not dpg.does_item_exist(parent_tag):
        return
    with dpg.tooltip(parent_tag):
        dpg.add_text(text, wrap=420)


def _prediction_is_loggable(state: dict[str, Any]) -> bool:
    activity_name = str(state.get("prediction_activity_name", "") or "")
    return activity_name in state.get("nM_activity_types", [])


def _prediction_display_linear(state: dict[str, Any]) -> bool:
    return bool(state.get("prediction_display_linear")) and _prediction_is_loggable(state)


def _prediction_display_target_label(state: dict[str, Any]) -> str:
    activity_name = str(state.get("prediction_activity_name", "") or "")
    if not activity_name:
        return str(state.get("prediction_target_label", "Activity"))
    return activity_name if _prediction_display_linear(state) else str(state.get("prediction_target_label", activity_name))


def _prediction_results_target_label(state: dict[str, Any]) -> str:
    """
    Return the label shown in results/export headers, including units only in
    linear mode for nM-based activities.
    """
    base_label = _prediction_display_target_label(state)
    if _prediction_display_linear(state) and _prediction_is_loggable(state):
        return f"{base_label} (nM)"
    return base_label


def _prediction_uses_decimal_linear_format(state: dict[str, Any]) -> bool:
    return _prediction_display_linear(state) or not _prediction_is_loggable(state)


def _prediction_to_display_value(value: float | None, state: dict[str, Any]) -> float | None:
    if value is None:
        return None
    if not _prediction_display_linear(state):
        return float(value)
    return float(10 ** (9.0 - float(value)))


def _ensure_prediction_display_cache(record: dict[str, Any]) -> None:
    """
    Cache both log and linear display values on the record so switching view
    only changes labels, not ordering semantics, and avoids repeated
    conversions during table rebuilds.
    """
    if record.get("_prediction_display_cache_ready"):
        return

    real_value = record.get("real_value")
    pred_value = record.get("predicted_value")

    real_log = None if real_value is None else float(real_value)
    pred_log = None if pred_value is None else float(pred_value)
    delta_log = None
    if real_log is not None and pred_log is not None:
        delta_log = abs(pred_log - real_log)

    real_linear = None if real_log is None else float(10 ** (9.0 - real_log))
    pred_linear = None if pred_log is None else float(10 ** (9.0 - pred_log))
    delta_linear = None
    if real_linear is not None and pred_linear is not None:
        delta_linear = abs(pred_linear - real_linear)

    record["_prediction_real_display_log"] = real_log
    record["_prediction_pred_display_log"] = pred_log
    record["_prediction_delta_display_log"] = delta_log
    record["_prediction_real_display_linear"] = real_linear
    record["_prediction_pred_display_linear"] = pred_linear
    record["_prediction_delta_display_linear"] = delta_linear
    record["_prediction_display_cache_ready"] = True


def _prediction_record_display_value(record: dict[str, Any], kind: str, state: dict[str, Any]) -> float | None:
    _ensure_prediction_display_cache(record)
    if _prediction_display_linear(state) and _prediction_is_loggable(state):
        return record.get(f"_prediction_{kind}_display_linear")
    return record.get(f"_prediction_{kind}_display_log")


def _prediction_display_value_from_internal(value: float | None, state: dict[str, Any]) -> float | None:
    if value is None:
        return None
    return _prediction_to_display_value(float(value), state)


def _prediction_format_display_value(value: float | None, state: dict[str, Any]) -> str:
    if value is None:
        return "-"
    if _prediction_uses_decimal_linear_format(state):
        return f"{float(value):.1f}"
    return f"{float(value):.3f}"


def _format_prediction_value(value: float | None, state: dict[str, Any]) -> str:
    if value is None:
        return "-"
    display_value = _prediction_to_display_value(value, state)
    if display_value is None:
        return "-"
    if _prediction_uses_decimal_linear_format(state):
        return f"{display_value:.1f}"
    return f"{display_value:.3f}"


def _prediction_display_delta(real_value: float | None, pred_value: float | None, state: dict[str, Any]) -> float | None:
    if real_value is None or pred_value is None:
        return None
    real_disp = _prediction_to_display_value(float(real_value), state)
    pred_disp = _prediction_to_display_value(float(pred_value), state)
    if real_disp is None or pred_disp is None:
        return None
    return abs(float(pred_disp) - float(real_disp))


def _prediction_display_quality_score(record: dict[str, Any], state: dict[str, Any]) -> float | None:
    """
    Return the stable quality score used to color the delta label.
    The color meaning must stay identical between log and linear views.
    """
    quality = record.get("quality_score")
    if quality is None:
        quality = _prediction_quality_score_from_delta(record.get("absolute_delta"), state)
    return float(quality) if isinstance(quality, (int, float)) else None


def _format_prediction_delta(real_value: float | None, pred_value: float | None, state: dict[str, Any]) -> str:
    display_delta = _prediction_display_delta(real_value, pred_value, state)
    if display_delta is None:
        return "-"
    if _prediction_uses_decimal_linear_format(state):
        return f"{display_delta:.1f}"
    return f"{display_delta:.3f}"


def _prediction_records_in_current_table_order(state: dict[str, Any]) -> list[dict[str, Any]]:
    records = list(state.get("prediction_results", []))
    record_map = {str(record.get("record_key")): record for record in records if record.get("record_key") is not None}
    order = list(state.get("prediction_table_order", []))
    ordered_records = [record_map[key] for key in order if key in record_map]
    if len(ordered_records) != len(records):
        return records
    return ordered_records


def _ensure_prediction_pagination_state(state: dict[str, Any]) -> None:
    state.setdefault("prediction_table_page", 1)
    state.setdefault("prediction_table_max_per_page", 25)
    state.setdefault("prediction_origin_filter", "All compounds")


def _sync_prediction_table_order(state: dict[str, Any]) -> None:
    records = list(state.get("prediction_results", []))
    current_keys = [str(record.get("record_key")) for record in records if record.get("record_key") is not None]
    existing_order = list(state.get("prediction_table_order", []))
    if set(existing_order) != set(current_keys) or len(existing_order) != len(current_keys):
        state["prediction_table_order"] = current_keys


def _prediction_total_pages(state: dict[str, Any]) -> int:
    target_label = _prediction_display_target_label(state)
    header_target_label = _prediction_results_target_label(state)
    records = _get_prediction_visible_records(state, target_label)
    max_per_page = max(1, int(state.get("prediction_table_max_per_page", 25)))
    return max(1, math.ceil(len(records) / max_per_page))


def _set_prediction_page_label(state: dict[str, Any]) -> None:
    if dpg.does_item_exist("prediction_tbl_page_number"):
        dpg.set_value("prediction_tbl_page_number", str(int(state.get("prediction_table_page", 1))))


def _clamp_prediction_page(state: dict[str, Any]) -> None:
    total_pages = _prediction_total_pages(state)
    state["prediction_table_page"] = max(1, min(int(state.get("prediction_table_page", 1)), total_pages))


def prev_prediction_page_callback(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    if state.get("_prediction_table_building"):
        return
    _ensure_prediction_pagination_state(state)
    page = int(state.get("prediction_table_page", 1))
    if page <= 1:
        return
    state["prediction_table_page"] = page - 1
    _refresh_prediction_table_with_loading(state)


def next_prediction_page_callback(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    if state.get("_prediction_table_building"):
        return
    _ensure_prediction_pagination_state(state)
    total_pages = _prediction_total_pages(state)
    page = int(state.get("prediction_table_page", 1))
    if page >= total_pages:
        return
    state["prediction_table_page"] = page + 1
    _refresh_prediction_table_with_loading(state)


def jump_to_prediction_page_callback(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    if state.get("_prediction_table_building"):
        return
    _ensure_prediction_pagination_state(state)
    try:
        requested_page = int(str(app_data).strip())
    except Exception:
        _set_prediction_page_label(state)
        return
    if requested_page == int(state.get("prediction_table_page", 1)):
        _set_prediction_page_label(state)
        return
    state["prediction_table_page"] = requested_page
    _clamp_prediction_page(state)
    _refresh_prediction_table_with_loading(state)


def update_prediction_rows_per_page(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    if state.get("_prediction_table_building"):
        return
    _ensure_prediction_pagination_state(state)
    try:
        new_value = max(1, int(app_data))
    except Exception:
        new_value = 25
    if new_value == int(state.get("prediction_table_max_per_page", 25)):
        return
    state["prediction_table_max_per_page"] = new_value
    state["prediction_table_page"] = 1
    _refresh_prediction_table_with_loading(state)


def update_prediction_display_scale(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    if state.get("_prediction_table_building"):
        return
    new_value = not bool(app_data)
    if new_value == bool(state.get("prediction_display_linear")):
        return
    state["prediction_display_linear"] = new_value
    _refresh_prediction_output_with_loading(state)


def _refresh_prediction_table_with_loading(state: dict[str, Any]) -> None:
    """
    Show the loading overlay while the prediction table is being rebuilt.
    """
    draw_loading_screen(state, bg=False)
    try:
        if dpg.does_item_exist("prediction_results_table"):
            _rebuild_prediction_results_table_rows(state)
        else:
            render_prediction_results_table(state)
    finally:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")


def _schedule_prediction_table_refresh_with_loading(state: dict[str, Any]) -> None:
    """
    Refresh the prediction table on the next frame with the loading overlay.
    """
    dpg.set_frame_callback(
        dpg.get_frame_count() + 1,
        lambda: _refresh_prediction_table_with_loading(state),
    )


def _refresh_prediction_output_with_loading(state: dict[str, Any]) -> None:
    """
    Show the loading overlay while prediction output is being rebuilt.
    """
    draw_loading_screen(state, bg=False)
    try:
        render_prediction_output(state)
    finally:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")


def _prediction_internal_value_from_display(value: float | None, state: dict[str, Any]) -> float | None:
    if value is None:
        return None
    if _prediction_display_linear(state) and _prediction_is_loggable(state):
        if float(value) <= 0:
            return None
        return float(-math.log10(float(value) * 1e-9))
    return float(value)


def _prediction_default_csv_dir(state: dict[str, Any] | None = None) -> str:
    """
    Return the default directory used by Prediction CSV load/save dialogs.
    """
    if state is not None:
        configured_dir = str(
            state.get("predictions_dir")
            or (state.get("settings", {}) or {}).get("predictions_directory", "")
            or ""
        ).strip()
        if configured_dir:
            configured_dir = os.path.expanduser(configured_dir)
            os.makedirs(configured_dir, exist_ok=True)
            return configured_dir

    preferred_dir = os.path.join(os.getcwd(), "data", "predictions")
    return preferred_dir if os.path.isdir(preferred_dir) else os.getcwd()


def _prediction_session_default_dir(state: dict[str, Any]) -> str:
    """
    Return the default directory used to load full prediction sessions.
    """
    prediction_dir = _prediction_default_csv_dir(state)
    if prediction_dir and os.path.isdir(prediction_dir):
        return prediction_dir

    work_dir = str(state.get("work_dir", "") or "").strip()
    if work_dir:
        preferred_dir = os.path.join(work_dir, "prediction")
        if os.path.isdir(preferred_dir):
            return preferred_dir
    return prediction_dir


def _parse_prediction_csv_activity_label(headers: list[str]) -> tuple[str, str]:
    for header in headers:
        if isinstance(header, str) and header.startswith("Real "):
            display_label = header[5:].strip()
            activity_name = display_label[1:] if display_label.startswith("p") else display_label
            return activity_name, display_label
    return "Activity", "Activity"


def _read_prediction_csv_with_metadata(selected_path: str) -> tuple[list[str], list[dict[str, str]], dict[str, str], list[dict[str, str]]]:
    """
    Read a prediction CSV export, supporting metadata and model-comparison blocks.
    """
    with open(selected_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return [], [], {}, []

    comparison_header_idx = None
    header_idx = None
    for idx, row in enumerate(rows):
        if row and str(row[0]).strip() == "Model" and len(row) > 1 and str(row[1]).strip() == "Score":
            comparison_header_idx = idx
        if row and str(row[0]).strip() == "Mol":
            header_idx = idx
            break

    if header_idx is None:
        return [], [], {}, []

    metadata: dict[str, str] = {}
    metadata_end_idx = comparison_header_idx if comparison_header_idx is not None else header_idx
    for row in rows[:metadata_end_idx]:
        if not row:
            continue
        key = str(row[0]).strip()
        value = str(row[1]).strip() if len(row) > 1 else ""
        if key:
            metadata[key] = value

    comparison_rows: list[dict[str, str]] = []
    if comparison_header_idx is not None and comparison_header_idx < header_idx:
        comparison_headers = [str(cell).strip() for cell in rows[comparison_header_idx]]
        for row in rows[comparison_header_idx + 1:header_idx]:
            if not any(str(cell).strip() for cell in row):
                continue
            padded = list(row) + [""] * max(0, len(comparison_headers) - len(row))
            comparison_rows.append({comparison_headers[i]: str(padded[i]).strip() for i in range(len(comparison_headers))})

    headers = [str(cell) for cell in rows[header_idx]]
    data_rows: list[dict[str, str]] = []
    for row in rows[header_idx + 1:]:
        if not any(str(cell).strip() for cell in row):
            continue
        padded = list(row) + [""] * max(0, len(headers) - len(row))
        data_rows.append({headers[i]: str(padded[i]) for i in range(len(headers))})

    return headers, data_rows, metadata, comparison_rows


def _get_prediction_export_metadata(state: dict[str, Any]) -> list[tuple[str, str]]:
    """
    Build the metadata rows written before the prediction table header.
    """
    scope_value = str(state.get("prediction_scope") or state.get("prediction_selected_scope") or "Dataset")
    activity_value = str(state.get("prediction_activity_name") or state.get("prediction_selected_activity") or "")
    features_value = str(state.get("prediction_feature_mode") or "")
    model_value = str(
        state.get("prediction_selected_model_name")
        or state.get("prediction_active_model_view")
        or state.get("prediction_model_name")
        or ""
    )
    session_path = str(state.get("prediction_session_path", "") or "").strip()
    session_name = os.path.basename(session_path) if session_path else ""
    include_undefined = bool(state.get("prediction_include_undefined", False))
    work_dir = str(state.get("work_dir") or "").strip()
    dataset_name = str(
        state.get("prediction_loaded_csv_name")
        or state.get("prediction_dataset_name")
        or (os.path.basename(work_dir.rstrip(os.sep)) if work_dir else "")
        or "N/A"
    )

    if dpg.does_item_exist("prediction_scope_choice"):
        scope_value = str(dpg.get_value("prediction_scope_choice") or scope_value)
    if dpg.does_item_exist("prediction_activity_choice"):
        activity_value = str(dpg.get_value("prediction_activity_choice") or activity_value)
    if dpg.does_item_exist("prediction_feature_choice"):
        features_value = str(dpg.get_value("prediction_feature_choice") or features_value)
    if dpg.does_item_exist("prediction_include_undefined_choice"):
        include_undefined = bool(dpg.get_value("prediction_include_undefined_choice"))

    return [
        ("Dataset", dataset_name),
        ("Exported at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Training scope", scope_value),
        ("Activity type", activity_value),
        ("Features", features_value),
        ("Model", model_value),
        ("Session file", session_name),
        ("Read undefined", "True" if include_undefined else "False"),
    ]


def _prediction_export_table_default_name(state: dict[str, Any]) -> str:
    """
    Build the default filename for exporting the currently displayed
    prediction table.
    """
    work_dir = str(state.get("work_dir", "") or "").strip()
    job_name = (
        str(state.get("prediction_dataset_name") or "").strip()
        or (os.path.basename(work_dir.rstrip(os.sep)) if work_dir else "")
        or "NA"
    )
    activity_name = str(
        state.get("prediction_activity_name")
        or state.get("prediction_selected_activity")
        or "Activity"
    ).strip() or "Activity"
    model_name = str(
        state.get("prediction_selected_model_name")
        or state.get("prediction_active_model_view")
        or state.get("prediction_model_name")
        or "Model"
    ).strip() or "Model"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_name = (
        f"predictiontable__{timestamp}__"
        f"{job_name}__"
        f"{activity_name}__"
        f"{model_name}.csv"
    )
    default_name = re.sub(r"[^A-Za-z0-9._-]+", "-", default_name)
    default_name = re.sub(r"-{2,}", "-", default_name)
    return default_name


def _apply_loaded_prediction_metadata(metadata: dict[str, str], state: dict[str, Any]) -> None:
    """
    Sync loaded export metadata back into state and visible controls when available.
    """
    if not metadata:
        return

    dataset_name = str(metadata.get("Dataset", "") or "").strip()
    if dataset_name:
        state["prediction_dataset_name"] = dataset_name

    scope_value = str(metadata.get("Training scope", "") or "").strip()
    activity_value = str(metadata.get("Activity type", "") or "").strip()
    features_value = str(metadata.get("Features", "") or "").strip()
    model_value = str(metadata.get("Model", "") or "").strip()
    session_file_value = str(metadata.get("Session file", "") or "").strip()
    include_undefined_text = str(metadata.get("Read undefined", "") or "").strip().lower()
    include_undefined = include_undefined_text in {"true", "1", "yes", "y"}

    if scope_value:
        state["prediction_scope"] = scope_value
        state["prediction_selected_scope"] = scope_value
        if dpg.does_item_exist("prediction_scope_choice"):
            try:
                dpg.set_value("prediction_scope_choice", scope_value)
            except Exception:
                pass

    if activity_value:
        state["prediction_activity_name"] = activity_value
        state["prediction_selected_activity"] = activity_value
        if dpg.does_item_exist("prediction_activity_choice"):
            try:
                dpg.set_value("prediction_activity_choice", activity_value)
            except Exception:
                pass

    if features_value:
        state["prediction_feature_mode"] = features_value
        if dpg.does_item_exist("prediction_feature_choice"):
            try:
                dpg.set_value("prediction_feature_choice", features_value)
            except Exception:
                pass

    if model_value:
        state["prediction_model_name"] = model_value
        if dpg.does_item_exist("prediction_model_choice"):
            try:
                dpg.set_value("prediction_model_choice", model_value)
            except Exception:
                pass

    if session_file_value:
        state["prediction_saved_session_name"] = session_file_value
        work_dir = str(state.get("work_dir", "") or "").strip()
        if work_dir:
            candidate_path = os.path.join(work_dir, "prediction", session_file_value)
            if os.path.isfile(candidate_path):
                state["prediction_session_path"] = candidate_path

    state["prediction_include_undefined"] = include_undefined
    if dpg.does_item_exist("prediction_include_undefined_choice"):
        try:
            dpg.set_value("prediction_include_undefined_choice", include_undefined)
        except Exception:
            pass


def _parse_loaded_prediction_model_comparison(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Convert the CSV model-comparison block back into numeric state rows.
    """
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        model_name = str(row.get("Model", "") or "").strip()
        if not model_name:
            continue
        parsed_rows.append(
            {
                "model": model_name,
                "composite_score": float(str(row.get("Score", "0") or "0")),
                "r2": float(str(row.get("R²", "0") or "0")),
                "mae": float(str(row.get("MAE", "0") or "0")),
                "rmse": float(str(row.get("RMSE", "0") or "0")),
                "mae_over_sd": float(str(row.get("MAE/SD", "0") or "0")),
                "rmse_over_sd": float(str(row.get("RMSE/SD", "0") or "0")),
            }
        )
    return parsed_rows


def _parse_prediction_csv_real_cell(value: str) -> tuple[str | None, float | None]:
    text = str(value or "").strip()
    if not text or text == "-":
        return None, None

    prefixes = ("≥ ", "≤ ", "> ", "< ", "= ")
    relation_map = {"≥ ": ">=", "≤ ": "<=", "> ": ">", "< ": "<", "= ": "="}
    for prefix in prefixes:
        if text.startswith(prefix):
            numeric_text = text[len(prefix):].strip()
            try:
                return relation_map[prefix], float(numeric_text)
            except Exception:
                return relation_map[prefix], None

    try:
        return "=", float(text)
    except Exception:
        return "=", None


def _prediction_high_match_threshold_from_state(state: dict[str, Any]) -> float:
    return 0.1 if _prediction_is_loggable(state) else 10.0


def _prediction_quality_score_from_delta(abs_delta: float | None, state: dict[str, Any]) -> float | None:
    if abs_delta is None:
        return None
    error_scale = 0.5 if _prediction_is_loggable(state) else 30.0
    return max(0.0, min(1.0, 1.0 - (float(abs_delta) / max(error_scale, 1e-12))))


def _load_prediction_session_file(selected_path: str, state: dict[str, Any]) -> None:
    """
    Load a full prediction session from a PKL file.
    """
    with open(selected_path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Unsupported prediction session format.")

    metadata = dict(payload.get("metadata", {}) or {})
    if metadata:
        state["prediction_model_metadata"] = metadata
        if metadata.get("job_name"):
            state["prediction_dataset_name"] = metadata["job_name"]
        if metadata.get("scope"):
            state["prediction_scope"] = metadata["scope"]
            state["prediction_selected_scope"] = metadata["scope"]
        if metadata.get("activity_name"):
            state["prediction_activity_name"] = metadata["activity_name"]
            state["prediction_selected_activity"] = metadata["activity_name"]
        if metadata.get("target_label"):
            state["prediction_target_label"] = metadata["target_label"]
        if metadata.get("feature_mode"):
            state["prediction_feature_mode"] = metadata["feature_mode"]
        state["prediction_include_undefined"] = bool(metadata.get("include_undefined", False))

    for key in [
        "prediction_model_comparison",
        "prediction_model_views",
        "prediction_results",
        "prediction_results_map",
        "prediction_plot_points",
        "prediction_metrics",
        "prediction_selected_record_key",
        "prediction_target_label",
        "prediction_activity_name",
        "prediction_scope",
        "prediction_feature_mode",
        "prediction_include_undefined",
        "prediction_display_linear",
        "prediction_table_order",
        "prediction_base_results",
        "prediction_base_results_map",
        "prediction_base_plot_points",
        "prediction_base_metrics",
        "prediction_base_model_comparison",
        "prediction_external_results",
        "prediction_external_results_map",
        "prediction_external_plot_points",
        "prediction_origin_filter",
        "prediction_active_model_view",
        "prediction_selected_model_name",
        "prediction_model_name",
    ]:
        if key in payload:
            state[key] = payload.get(key)

    state["prediction_session_path"] = selected_path
    state["prediction_loaded_session_name"] = os.path.basename(selected_path)
    state["prediction_loaded_csv_name"] = ""
    state["prediction_status_message"] = ""
    state["prediction_current_model"] = None

    selected_model_name = str(
        state.get("prediction_selected_model_name")
        or state.get("prediction_active_model_view")
        or state.get("prediction_model_name")
        or ""
    )
    if selected_model_name:
        activate_prediction_model_view(selected_model_name, state)

    if dpg.does_item_exist("prediction_scope_choice") and state.get("prediction_scope"):
        try:
            dpg.set_value("prediction_scope_choice", state["prediction_scope"])
        except Exception:
            pass
    if dpg.does_item_exist("prediction_activity_choice") and state.get("prediction_activity_name"):
        try:
            dpg.set_value("prediction_activity_choice", state["prediction_activity_name"])
        except Exception:
            pass
    if dpg.does_item_exist("prediction_feature_choice") and state.get("prediction_feature_mode"):
        try:
            dpg.set_value("prediction_feature_choice", state["prediction_feature_mode"])
        except Exception:
            pass
    if dpg.does_item_exist("prediction_model_choice") and state.get("prediction_model_name"):
        try:
            dpg.set_value("prediction_model_choice", state["prediction_model_name"])
        except Exception:
            pass
    if dpg.does_item_exist("prediction_include_undefined_choice"):
        try:
            dpg.set_value("prediction_include_undefined_choice", bool(state.get("prediction_include_undefined", False)))
        except Exception:
            pass


def _prediction_external_default_dir(state: dict[str, Any]) -> str:
    input_dir = str(state.get("input_dir", "") or "").strip()
    if input_dir and os.path.isdir(input_dir):
        return input_dir
    selected_file_path = str(state.get("selected_file_path", "") or "").strip()
    if selected_file_path:
        selected_dir = os.path.dirname(selected_file_path)
        if selected_dir and os.path.isdir(selected_dir):
            return selected_dir
    work_dir = str(state.get("work_dir", "") or "").strip()
    if work_dir and os.path.isdir(work_dir):
        return work_dir
    return os.getcwd()


def _prediction_external_parse_real_value(value: Any, state: dict[str, Any], source_label: str | None = None) -> tuple[float | None, str | None]:
    text = str(value or "").strip()
    if not text or text == "-":
        return None, None

    relation = "="
    prefixes = ("≥ ", "≤ ", "> ", "< ", "= ")
    relation_map = {"≥ ": ">=", "≤ ": "<=", "> ": ">", "< ": "<", "= ": "="}
    for prefix in prefixes:
        if text.startswith(prefix):
            relation = relation_map[prefix]
            text = text[len(prefix):].strip()
            break
    try:
        numeric_value = float(text)
    except Exception:
        return None, None

    activity_name = str(state.get("prediction_activity_name", "") or "")
    target_label = str(state.get("prediction_target_label", activity_name) or activity_name)
    label = str(source_label or activity_name or "").strip()
    label_lower = label.lower()
    activity_lower = activity_name.lower()
    target_lower = target_label.lower()
    # Manual input and generic "activity" columns are always interpreted in the
    # original activity scale (e.g. IC50), then converted to the internal
    # logarithmic representation when needed. Only an explicit pActivity-like
    # source header should be treated as already logarithmic.
    if _prediction_is_loggable(state) and label_lower in {
        activity_lower,
        "activity",
        "real activity",
        "",
    }:
        if numeric_value <= 0:
            return None, None
        return float(-math.log10(numeric_value * 1e-9)), relation
    if _prediction_is_loggable(state) and label_lower == target_lower:
        return numeric_value, relation
    return numeric_value, relation


def _read_external_prediction_csv_entries(selected_path: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    activity_name = str(state.get("prediction_activity_name", "") or "")
    target_label = str(state.get("prediction_target_label", activity_name) or activity_name)

    with open(selected_path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        headers = reader.fieldnames or []
        lowered = {str(h).strip().lower(): h for h in headers if h}
        smiles_key = lowered.get("smiles")
        name_key = lowered.get("name") or lowered.get("mol name") or lowered.get("compound") or lowered.get("molecule")
        real_key = None
        for candidate in [activity_name, target_label, "real activity", "activity"]:
            if candidate and candidate.lower() in lowered:
                real_key = lowered[candidate.lower()]
                break

        entries: list[dict[str, Any]] = []
        for row in reader:
            smiles = str(row.get(smiles_key, "") or "").strip() if smiles_key else ""
            if not smiles:
                continue
            raw_real = row.get(real_key) if real_key else None
            real_value, relation = _prediction_external_parse_real_value(raw_real, state, source_label=real_key)
            entries.append(
                {
                    "name": str(row.get(name_key, "") or "").strip() if name_key else "",
                    "smiles": smiles,
                    "real_value": real_value,
                    "real_raw": str(raw_real).strip() if raw_real not in (None, "") else None,
                    "relation": relation,
                }
            )
    return entries


def _ensure_prediction_external_state(state: dict[str, Any]) -> None:
    state.setdefault("prediction_external_entries", [])


def _prediction_external_collect_widget_entries(state: dict[str, Any]) -> list[dict[str, Any]]:
    _ensure_prediction_external_state(state)
    entries: list[dict[str, Any]] = []
    for idx, _entry in enumerate(state.get("prediction_external_entries", [])):
        name_tag = f"prediction_external_name_{idx}"
        smiles_tag = f"prediction_external_smiles_{idx}"
        real_tag = f"prediction_external_real_{idx}"
        if not dpg.does_item_exist(smiles_tag):
            continue
        name = str(dpg.get_value(name_tag) or "").strip() if dpg.does_item_exist(name_tag) else ""
        smiles = str(dpg.get_value(smiles_tag) or "").strip()
        real_text = str(dpg.get_value(real_tag) or "").strip() if dpg.does_item_exist(real_tag) else ""
        real_value, relation = _prediction_external_parse_real_value(real_text, state)
        entries.append(
            {
                "name": name,
                "smiles": smiles,
                "real_value": real_value,
                "real_raw": real_text if real_text else None,
                "relation": relation,
            }
        )
    state["prediction_external_entries"] = entries
    return entries


def _prediction_external_set_entries(entries: list[dict[str, Any]], state: dict[str, Any]) -> None:
    state["prediction_external_entries"] = [
        {
            "name": str(entry.get("name", "") or ""),
            "smiles": str(entry.get("smiles", "") or ""),
            "real_value": entry.get("real_value"),
            "real_raw": entry.get("real_raw"),
            "relation": entry.get("relation"),
        }
        for entry in entries
    ]


def _rebuild_prediction_external_rows(state: dict[str, Any]) -> None:
    if not dpg.does_item_exist("prediction_external_entries_table"):
        return
    _ensure_prediction_external_state(state)
    for row_id in dpg.get_item_children("prediction_external_entries_table", 1) or []:
        dpg.delete_item(row_id)
    activity_label = str(state.get("prediction_activity_name", "") or _prediction_display_target_label(state))

    def _remove_row(row_idx: int) -> None:
        current_entries = _prediction_external_collect_widget_entries(state)
        if 0 <= row_idx < len(current_entries):
            current_entries.pop(row_idx)
        _prediction_external_set_entries(current_entries, state)
        _rebuild_prediction_external_rows(state)

    for idx, entry in enumerate(state.get("prediction_external_entries", [])):
        with dpg.table_row(parent="prediction_external_entries_table"):
            with dpg.table_cell():
                dpg.add_input_text(
                    tag=f"prediction_external_smiles_{idx}",
                    default_value=str(entry.get("smiles", "") or ""),
                    width=-1,
                )
            with dpg.table_cell():
                dpg.add_input_text(
                    tag=f"prediction_external_name_{idx}",
                    default_value=str(entry.get("name", "") or ""),
                    width=-1,
                    hint="optional",
                )
            with dpg.table_cell():
                real_default = str(entry.get("real_raw", "") or "")
                if not real_default and entry.get("real_value") is not None:
                    display_val = _prediction_to_display_value(float(entry["real_value"]), state)
                    if display_val is not None:
                        real_default = f"{display_val:.1f}" if _prediction_uses_decimal_linear_format(state) else f"{display_val:.3f}"
                dpg.add_input_text(
                    tag=f"prediction_external_real_{idx}",
                    default_value=real_default,
                    width=-1,
                    hint=f"{activity_label} (optional)",
                )
            with dpg.table_cell():
                dpg.add_button(
                    label="Remove",
                    user_data=idx,
                    callback=lambda s, a, u: _remove_row(u),
                )


def _open_prediction_external_window(state: dict[str, Any]) -> None:
    _ensure_prediction_external_state(state)
    if not state["prediction_external_entries"]:
        state["prediction_external_entries"] = [{"name": "", "smiles": "", "real_value": None, "real_raw": None, "relation": None}]

    if dpg.does_item_exist("prediction_external_window"):
        dpg.delete_item("prediction_external_window")

    def _add_row() -> None:
        current_entries = _prediction_external_collect_widget_entries(state)
        current_entries.append({"name": "", "smiles": "", "real_value": None, "real_raw": None, "relation": None})
        _prediction_external_set_entries(current_entries, state)
        _rebuild_prediction_external_rows(state)

    def _clear_rows() -> None:
        state["prediction_external_entries"] = [{"name": "", "smiles": "", "real_value": None, "real_raw": None, "relation": None}]
        _rebuild_prediction_external_rows(state)

    def _load_csv() -> None:
        selected_path = open_file_dialog(
            title="Load external prediction CSV",
            default_path=_prediction_external_default_dir(state),
            file_types=[("CSV files", "*.csv")],
        )
        if not selected_path:
            return
        entries = _read_external_prediction_csv_entries(selected_path, state)
        if entries:
            _prediction_external_set_entries(entries, state)
            _rebuild_prediction_external_rows(state)

    def _load_sdf() -> None:
        selected_path = open_file_dialog(
            title="Load external prediction SDF",
            default_path=_prediction_external_default_dir(state),
            file_types=[("SDF files", "*.sdf")],
        )
        if not selected_path:
            return
        entries: list[dict[str, Any]] = []
        activity_name = str(state.get("prediction_activity_name", "") or "")
        target_label = str(state.get("prediction_target_label", activity_name) or activity_name)
        supplier = Chem.SDMolSupplier(selected_path)
        for mol in supplier:
            if mol is None:
                continue
            smiles = mol.GetProp("Smiles") if mol.HasProp("Smiles") else Chem.MolToSmiles(mol)
            raw_real = None
            source_label = None
            for candidate in [activity_name, target_label, "Real activity", "Activity"]:
                if mol.HasProp(candidate):
                    raw_real = mol.GetProp(candidate)
                    source_label = candidate
                    break
            real_value, relation = _prediction_external_parse_real_value(raw_real, state, source_label=source_label)
            entries.append(
                {
                    "name": mol.GetProp("_Name") if mol.HasProp("_Name") else "",
                    "smiles": smiles,
                    "real_value": real_value,
                    "real_raw": str(raw_real).strip() if raw_real not in (None, "") else None,
                    "relation": relation,
                }
            )
        if entries:
            _prediction_external_set_entries(entries, state)
            _rebuild_prediction_external_rows(state)

    def _run_external_prediction() -> None:
        entries = _prediction_external_collect_widget_entries(state)
        draw_loading_screen(state, bg=False)
        try:
            result = run_prediction_on_external_records(state, entries)
            if not result.get("ok"):
                state["prediction_status_message"] = str(result.get("message") or "External prediction failed.")
            render_prediction_output(state)
            if result.get("ok") and dpg.does_item_exist("prediction_external_window"):
                dpg.delete_item("prediction_external_window")
        finally:
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")

    with dpg.window(
        label="Predict External Compounds",
        tag="prediction_external_window",
        modal=True,
        width=1000,
        height=560,
        no_resize=False,
        no_collapse=True,
    ):
        dpg.add_text("Insert compounds manually or import them from CSV/SDF. SMILES is required; real activity is optional.")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Add row", callback=lambda: _add_row())
            dpg.add_button(label="Load CSV", callback=lambda: _load_csv())
            dpg.add_button(label="Load SDF", callback=lambda: _load_sdf())
            dpg.add_button(label="Clear", callback=lambda: _clear_rows())
            dpg.add_spacer(width=20)
            dpg.add_button(label="Run external prediction", callback=lambda: _run_external_prediction())
        with dpg.child_window(width=-1, height=-1, border=False):
            with dpg.table(
                tag="prediction_external_entries_table",
                header_row=True,
                resizable=True,
                policy=dpg.mvTable_SizingStretchProp,
                borders_innerH=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_outerV=True,
                row_background=True,
                scrollX=True,
            ):
                dpg.add_table_column(label="SMILES")
                dpg.add_table_column(label="Name (optional)")
                dpg.add_table_column(label=f"Real {str(state.get('prediction_activity_name', '') or _prediction_display_target_label(state))} (optional)")
                dpg.add_table_column(label="")
        _rebuild_prediction_external_rows(state)


def load_prediction_results_table(state: dict[str, Any]) -> None:
    """
    Load either a full prediction session PKL or an exported prediction CSV.
    """
    default_dir = _prediction_session_default_dir(state)
    selected_path = open_file_dialog(
        title="Load prediction",
        default_path=default_dir,
        file_types=[("Prediction sessions", "*.pkl"), ("CSV files", "*.csv")],
    )
    if not selected_path:
        return

    draw_loading_screen(state, bg=False)
    try:
        if selected_path.lower().endswith(".pkl"):
            _load_prediction_session_file(selected_path, state)
            render_prediction_output(state)
            return

        state["prediction_session_path"] = ""
        state["prediction_model_views"] = {}
        state["prediction_active_model_view"] = None
        state["prediction_selected_model_name"] = None
        state["prediction_current_model"] = None
        state["prediction_loaded_csv_name"] = os.path.basename(selected_path)
        headers, rows, metadata, comparison_rows = _read_prediction_csv_with_metadata(selected_path)

        if not rows or not headers:
            state["prediction_results"] = []
            state["prediction_results_map"] = {}
            state["prediction_plot_points"] = []
            state["prediction_metrics"] = {}
            state["prediction_status_message"] = "Loaded prediction table is empty."
            render_prediction_output(state)
            return

        _apply_loaded_prediction_metadata(metadata, state)
        state["prediction_model_comparison"] = _parse_loaded_prediction_model_comparison(comparison_rows)
        activity_name, display_label = _parse_prediction_csv_activity_label(headers)
        state["prediction_activity_name"] = activity_name
        if activity_name in state.get("nM_activity_types", []):
            state["prediction_target_label"] = f"p{activity_name}"
            state["prediction_display_linear"] = display_label == activity_name
        else:
            state["prediction_target_label"] = activity_name
            state["prediction_display_linear"] = False

        real_col = f"Real {display_label} (nM)" if f"Real {display_label} (nM)" in headers else f"Real {display_label}"
        cv_pred_col = (
            f"CV Pred {display_label} (nM)" if f"CV Pred {display_label} (nM)" in headers else (
                f"CV Pred {display_label}" if f"CV Pred {display_label}" in headers else None
            )
        )
        final_pred_col = (
            f"Fit Pred {display_label} (nM)" if f"Fit Pred {display_label} (nM)" in headers else (
                f"Fit Pred {display_label}" if f"Fit Pred {display_label}" in headers else (
                    f"Final Pred {display_label}" if f"Final Pred {display_label}" in headers else None
                )
            )
        )
        pred_col = f"Pred {display_label}" if f"Pred {display_label}" in headers else None
        smiles_col = "SMILES" if "SMILES" in headers else None

        results: list[dict[str, Any]] = []
        results_map: dict[str, dict[str, Any]] = {}
        plot_points: list[dict[str, Any]] = []

        for idx, row in enumerate(rows, start=1):
            relation, real_display_value = _parse_prediction_csv_real_cell(row.get(real_col, "-"))
            cv_pred_text = str(row.get(cv_pred_col or "", "-") if cv_pred_col else "-").strip()
            final_pred_text = str(row.get(final_pred_col or "", "-") if final_pred_col else "-").strip()
            legacy_pred_text = str(row.get(pred_col or "", "-") if pred_col else "-").strip()

            cv_pred_display_value = None if cv_pred_text == "-" else float(cv_pred_text)
            final_pred_display_value = None if final_pred_text == "-" else float(final_pred_text)
            if final_pred_display_value is None and legacy_pred_text != "-":
                final_pred_display_value = float(legacy_pred_text)
            if cv_pred_display_value is None and legacy_pred_text != "-":
                cv_pred_display_value = float(legacy_pred_text)

            real_value = _prediction_internal_value_from_display(real_display_value, state)
            cv_pred_value = _prediction_internal_value_from_display(cv_pred_display_value, state)
            final_pred_value = _prediction_internal_value_from_display(final_pred_display_value, state)
            pred_value = cv_pred_value if cv_pred_value is not None else final_pred_value
            abs_delta = abs(pred_value - real_value) if real_value is not None and pred_value is not None else None
            quality = _prediction_quality_score_from_delta(abs_delta, state)

            subset_text = str(row.get("Subset", "N/A") or "N/A").strip()
            subset_value = subset_text.replace("Subset ", "subset_") if subset_text.startswith("Subset ") else subset_text
            mol_text = str(row.get("Mol", "N/A") or "N/A").strip()
            try:
                mol_id = int(mol_text)
            except Exception:
                mol_id = -1

            record = {
                "record_key": f"loaded:{idx}",
                "subset": subset_value,
                "mol_key": f"loaded_{idx}",
                "mol_id": mol_id,
                "name": str(row.get("Name", "N/A") or "N/A"),
                "smiles": str(row.get(smiles_col, "")).strip() if smiles_col else "",
                "features": None,
                "activity_available": real_value is not None,
                "is_undefined": relation not in (None, "="),
                "relation": relation,
                "real_value": real_value,
                "real_raw": row.get(real_col, "-"),
                "scale_label": state["prediction_target_label"],
                "predicted_value": pred_value,
                "cv_predicted_value": cv_pred_value,
                "final_predicted_value": final_pred_value,
                "status": str(row.get("Status", "Unknown") or "Unknown"),
                "quality_score": quality,
                "absolute_delta": abs_delta,
            }
            results.append(record)
            results_map[record["record_key"]] = record
            if real_value is not None and pred_value is not None:
                plot_points.append(record)

        y_true = np.asarray([float(r["real_value"]) for r in plot_points], dtype=float) if plot_points else np.asarray([], dtype=float)
        y_pred = np.asarray([float(r["predicted_value"]) for r in plot_points], dtype=float) if plot_points else np.asarray([], dtype=float)
        mae = float(np.mean(np.abs(y_pred - y_true))) if len(y_true) else 0.0
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if len(y_true) else 0.0
        ss_res = float(np.sum((y_true - y_pred) ** 2)) if len(y_true) else 0.0
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) if len(y_true) else 0.0
        r2 = float(1.0 - ss_res / ss_tot) if len(y_true) and ss_tot > 0 else 0.0
        high_match_threshold = _prediction_high_match_threshold_from_state(state)
        high_match_count = int(np.sum(np.abs(y_pred - y_true) <= max(high_match_threshold, 1e-9))) if len(y_true) else 0
        state["prediction_results"] = results
        state["prediction_results_map"] = results_map
        state["prediction_plot_points"] = plot_points
        state["prediction_selected_record_key"] = results[0]["record_key"] if results else None
        state["prediction_status_message"] = ""
        state["prediction_metrics"] = {
            "dataset_size": len(results),
            "training_size": len(plot_points),
            "predicted_size": sum(1 for r in results if r.get("real_value") is None),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "high_match_count": high_match_count,
            "high_match_threshold": high_match_threshold,
        }
        _store_prediction_base_state(
            state,
            results,
            results_map,
            plot_points,
            state["prediction_metrics"],
            list(state.get("prediction_model_comparison", []) or []),
        )
        render_prediction_output(state)
    finally:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")


def _export_prediction_results_table(state: dict[str, Any]) -> None:
    records = _get_prediction_visible_records(state, _prediction_display_target_label(state))
    if not records:
        return

    activity_label = _prediction_display_target_label(state)
    header_activity_label = _prediction_results_target_label(state)
    default_dir = _prediction_default_csv_dir(state)
    default_name = _prediction_export_table_default_name(state)
    selected_path = save_file_dialog(
        title="Export prediction table",
        default_path=default_dir,
        default_name=default_name,
        file_types=[("CSV files", "*.csv")],
    )
    if not selected_path:
        return

    if not selected_path.lower().endswith(".csv"):
        selected_path = f"{selected_path}.csv"

    try:
        with open(selected_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for key, value in _get_prediction_export_metadata(state):
                writer.writerow([key, value])
            writer.writerow([])
            comparison_rows = sorted(
                list(state.get("prediction_model_comparison", []) or []),
                key=lambda row: float(row.get("composite_score", float("-inf"))),
                reverse=True,
            )
            if comparison_rows:
                writer.writerow(["Model", "Score", "R²", "MAE", "RMSE", "MAE/SD", "RMSE/SD"])
                for row in comparison_rows:
                    writer.writerow([
                        row.get("model", ""),
                        f"{float(row.get('composite_score', 0.0)):.6f}",
                        f"{float(row.get('r2', 0.0)):.6f}",
                        f"{float(row.get('mae', 0.0)):.6f}",
                        f"{float(row.get('rmse', 0.0)):.6f}",
                        f"{float(row.get('mae_over_sd', 0.0)):.6f}",
                        f"{float(row.get('rmse_over_sd', 0.0)):.6f}",
                    ])
                writer.writerow([])
            writer.writerow([
                "Mol",
                "Name",
                "SMILES",
                "Subset",
                f"Real {header_activity_label}",
                f"CV Pred {header_activity_label}",
                f"|ΔCV {header_activity_label}|",
                f"Fit Pred {header_activity_label}",
                f"|ΔFit {header_activity_label}|",
                "Status",
            ])
            for record in records:
                relation = record.get("relation") or "="
                relation = "≤" if relation == "<=" else "≥" if relation == ">=" else relation
                relation_prefix = "" if relation == "=" else f"{relation} "
                real_text = "-"
                if record.get("real_value") is not None:
                    real_text = f"{relation_prefix}{_format_prediction_value(float(record['real_value']), state)}"
                cv_pred_text = (
                    _format_prediction_value(float(record["cv_predicted_value"]), state)
                    if record.get("cv_predicted_value") is not None
                    else "-"
                )
                final_pred_text = (
                    _format_prediction_value(float(record["final_predicted_value"]), state)
                    if record.get("final_predicted_value") is not None
                    else "-"
                )
                cv_delta_text = _format_prediction_delta(record.get("real_value"), record.get("cv_predicted_value"), state)
                final_delta_text = _format_prediction_delta(record.get("real_value"), record.get("final_predicted_value"), state)
                writer.writerow([
                    record.get("mol_id", "N/A"),
                    record.get("name", "N/A"),
                    record.get("smiles", ""),
                    str(record.get("subset", "N/A")).replace("subset_", "Subset "),
                    real_text,
                    cv_pred_text,
                    cv_delta_text,
                    final_pred_text,
                    final_delta_text,
                    record.get("status", "Unknown"),
                ])
        log_event("Prediction", f"Prediction CSV exported: {selected_path}", indent=1, level="SUCCESS")
    except Exception as e:
        log_exception("Prediction", "Error exporting prediction CSV", e, indent=1)
        log_traceback("Prediction", indent=2)


def _blank_prediction_texture(width: int, height: int) -> np.ndarray:
    """
    Return a white RGBA texture placeholder.
    """
    arr = np.ones((height, width, 4), dtype=np.float32)
    arr[..., 3] = 1.0
    return arr.flatten()


def _make_prediction_selectable_theme(
    color: tuple[int, int, int, int] | None = None,
    active: bool = False,
) -> str:
    """
    Create/reuse a selectable theme for prediction rows.
    """
    if color is None:
        tag = "prediction_row_selectable_theme_active" if active else "prediction_row_selectable_theme"
    else:
        tag = f"prediction_row_selectable_theme_{color[0]}_{color[1]}_{color[2]}_{color[3]}_{1 if active else 0}"

    if dpg.does_item_exist(tag):
        return tag

    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvSelectable):
            if color is not None:
                dpg.add_theme_color(dpg.mvThemeCol_Text, color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, 0.0, 0.5, category=dpg.mvThemeCat_Core)
    return tag


def _ensure_prediction_details_widgets(state: dict[str, Any]) -> None:
    """
    Create the image/details widgets if missing.
    """
    if dpg.does_item_exist("prediction_details_window") and not dpg.does_item_exist("prediction_molecule_image_texture"):
        img_w = int(state["prediction_img_width"])
        img_h = int(round(img_w * 0.75))
        render_scale = 1.8
        render_w = int(round(img_w * render_scale))
        render_h = int(round(img_h * render_scale))
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                render_w,
                render_h,
                _blank_prediction_texture(render_w, render_h),
                tag="prediction_molecule_image_texture",
            )

    if dpg.does_item_exist("prediction_details_window") and not dpg.does_item_exist("prediction_details_group"):
        with dpg.child_window(
            parent="prediction_details_window",
            tag="prediction_details_group",
            width=-1,
            height=-1,
            border=False,
            no_scrollbar=False,
            horizontal_scrollbar=False,
        ):
            with dpg.group(horizontal=False):
                with dpg.child_window(
                    tag="prediction_molecule_image_cell",
                    width=-1,
                    border=False,
                    auto_resize_y=True,
                    no_scrollbar=True,
                    horizontal_scrollbar=False,
                    no_scroll_with_mouse=True,
                ):
                    dpg.add_image(
                        "prediction_molecule_image_texture",
                        tag="prediction_molecule_image_widget",
                        width=state["prediction_img_width"],
                        height=round(state["prediction_img_width"] * 0.75),
                    )
                    register_responsive_image(
                        state,
                        image_tag="prediction_molecule_image_widget",
                        parent_tag="prediction_molecule_image_cell",
                        aspect_ratio=0.75,
                        tab="prediction_tab",
                    )

                dpg.add_spacer(height=max(2, int(state["win_spacer"] * 0.35)))
                dpg.add_text("Name: -", tag="prediction_details_name_text", wrap=600)
                dpg.add_text("Real activity: -", tag="prediction_details_real_text", wrap=600)
                dpg.add_text("CV predicted activity: -", tag="prediction_details_cv_pred_text", wrap=600)
                dpg.add_text("Fit predicted activity: -", tag="prediction_details_final_pred_text", wrap=600)
                dpg.add_text("Subset: -", tag="prediction_details_subset_text", wrap=600)


def update_prediction_details(record: dict[str, Any] | None, state: dict[str, Any]) -> None:
    """
    Update the bottom-right details panel.
    """
    _ensure_prediction_details_widgets(state)

    if record is None:
        if dpg.does_item_exist("prediction_molecule_image_texture"):
            img_w = int(state["prediction_img_width"])
            img_h = int(round(img_w * 0.75))
            render_scale = 1.8
            render_w = int(round(img_w * render_scale))
            render_h = int(round(img_h * render_scale))
            dpg.set_value("prediction_molecule_image_texture", _blank_prediction_texture(render_w, render_h))
        if dpg.does_item_exist("prediction_details_name_text"):
            dpg.set_value("prediction_details_name_text", "Name: -")
            dpg.set_value("prediction_details_real_text", "Real activity: -")
            dpg.set_value("prediction_details_cv_pred_text", "CV predicted activity: -")
            dpg.set_value("prediction_details_final_pred_text", "Fit predicted activity: -")
            dpg.set_value("prediction_details_subset_text", "Subset: -")
        return

    _ensure_prediction_display_cache(record)

    mol = Chem.MolFromSmiles(record.get("smiles", ""))
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(record.get("smiles", ""), sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            mol = None
    if mol is not None and dpg.does_item_exist("prediction_molecule_image_texture"):
        rdDepictor.Compute2DCoords(mol)
        render_scale = 1.8
        width = int(round(int(state["prediction_img_width"]) * render_scale))
        height = int(round(round(int(state["prediction_img_width"]) * 0.75) * render_scale))
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.padding = 0.025
        opts.bondLineWidth = 1
        opts.minFontSize = 1
        opts.legendFontSize = 14
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        img = pilImage.open(io.BytesIO(png_bytes)).convert("RGBA")
        arr = (np.array(img) / 255.0).astype(np.float32).flatten()
        dpg.set_value("prediction_molecule_image_texture", arr)

    target_label = _prediction_display_target_label(state)
    real_text = "-"
    if record.get("real_value") is not None:
        relation = record.get("relation") or "="
        relation = "≤" if relation == "<=" else "≥" if relation == ">=" else relation
        real_display = _prediction_record_display_value(record, "real", state)
        if real_display is not None:
            formatted = _prediction_format_display_value(real_display, state)
            real_text = f"{target_label} {relation} {formatted}"
    elif record.get("real_raw"):
        real_text = str(record["real_raw"])

    cv_pred_text = "-"
    if record.get("cv_predicted_value") is not None:
        cv_pred_display = _prediction_display_value_from_internal(record.get("cv_predicted_value"), state)
        if cv_pred_display is not None:
            formatted = _prediction_format_display_value(cv_pred_display, state)
            cv_pred_text = f"{target_label} = {formatted}"

    final_pred_text = "-"
    if record.get("final_predicted_value") is not None:
        final_pred_display = _prediction_display_value_from_internal(record.get("final_predicted_value"), state)
        if final_pred_display is not None:
            formatted = _prediction_format_display_value(final_pred_display, state)
            final_pred_text = f"{target_label} = {formatted}"

    dpg.set_value("prediction_details_name_text", f"Name: {record.get('name', 'N/A')}  |  Mol {record.get('mol_id', 'N/A')}")
    dpg.set_value("prediction_details_real_text", f"Real activity: {real_text}")
    dpg.set_value("prediction_details_cv_pred_text", f"CV predicted activity: {cv_pred_text}")
    dpg.set_value("prediction_details_final_pred_text", f"Fit predicted activity: {final_pred_text}")
    dpg.set_value("prediction_details_subset_text", f"Subset: {record.get('subset', 'N/A')}")


def select_prediction_record(record_key: str | None, state: dict[str, Any], refresh_table: bool = True) -> None:
    """
    Select one prediction record and refresh dependent widgets.
    """
    if state.get("prediction_selected_record_key") == record_key:
        records_map = state.get("prediction_results_map", {})
        record = records_map.get(record_key) if isinstance(records_map, dict) else None
        update_prediction_details(record, state)
        return

    state["prediction_selected_record_key"] = record_key
    records_map = state.get("prediction_results_map", {})
    record = records_map.get(record_key) if isinstance(records_map, dict) else None
    update_prediction_details(record, state)
    if refresh_table:
        if dpg.does_item_exist("prediction_results_table"):
            _rebuild_prediction_results_table_rows(state)
        else:
            render_prediction_results_table(state)


def _prediction_sort_key(value: Any) -> tuple[int, Any]:
    """
    Normalise table values for stable mixed-type sorting.
    """
    if value is None:
        return (0, 0.0)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value))
    text = str(value)
    if text.strip() == "-":
        return (0, 0.0)
    try:
        return (0, float(text))
    except Exception:
        return (1, text.lower())


def _prediction_record_sort_value(record: dict[str, Any], data_key: str, state: dict[str, Any]) -> Any:
    """
    Extract a sortable value from one prediction record.
    """
    if data_key == "real_sort_value":
        if record.get("real_value") is None:
            return 0.0
        return float(record["real_value"])
    if data_key == "cv_pred_sort_value":
        if record.get("cv_predicted_value") is None:
            return 0.0
        return float(record["cv_predicted_value"])
    if data_key == "final_pred_sort_value":
        if record.get("final_predicted_value") is None:
            return 0.0
        return float(record["final_predicted_value"])
    if data_key == "cv_delta_sort_value":
        delta = None
        if record.get("real_value") is not None and record.get("cv_predicted_value") is not None:
            delta = abs(float(record["cv_predicted_value"]) - float(record["real_value"]))
        return 0.0 if delta is None else delta
    if data_key == "final_delta_sort_value":
        delta = None
        if record.get("real_value") is not None and record.get("final_predicted_value") is not None:
            delta = abs(float(record["final_predicted_value"]) - float(record["real_value"]))
        return 0.0 if delta is None else delta
    return record.get(data_key)


def _get_prediction_filtered_records(records: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Filter prediction records by source, molecule name or SMILES.
    """
    origin_filter = str(state.get("prediction_origin_filter", "All compounds") or "All compounds")
    if origin_filter == "Original prediction set":
        records = [record for record in records if str(record.get("row_origin", "original")) != "external"]
    elif origin_filter == "Added compounds":
        records = [record for record in records if str(record.get("row_origin", "")) == "external"]

    query = str(state.get("prediction_filter_query", "") or "").strip().lower()
    if not query:
        return records

    filtered_records: list[dict[str, Any]] = []
    for record in records:
        mol_id = str(record.get("mol_id", "") or "").lower()
        name = str(record.get("name", "") or "").lower()
        smiles = str(record.get("smiles", "") or "").lower()
        if query in mol_id or query in name or query in smiles:
            filtered_records.append(record)
    return filtered_records


def _get_prediction_sort_map(target_label: str) -> dict[str, str]:
    return {
        "Mol": "mol_id",
        "Name": "name",
        f"Real {target_label}": "real_sort_value",
        f"CV Pred {target_label}": "cv_pred_sort_value",
        f"|ΔCV {target_label}|": "cv_delta_sort_value",
        f"Fit Pred {target_label}": "final_pred_sort_value",
        f"|ΔFit {target_label}|": "final_delta_sort_value",
        "Status": "status",
    }


def _get_prediction_ordered_records(state: dict[str, Any], target_label: str) -> list[dict[str, Any]]:
    _ensure_prediction_pagination_state(state)
    _sync_prediction_table_order(state)
    records = _prediction_records_in_current_table_order(state)
    sort_map = _get_prediction_sort_map(target_label)
    current_sort = state.get("prediction_sort_spec")
    if isinstance(current_sort, tuple) and len(current_sort) == 2:
        sort_label, sort_direction = current_sort
        data_key = sort_map.get(sort_label)
        if data_key is not None:
            records = sorted(
                records,
                key=lambda record: _prediction_sort_key(_prediction_record_sort_value(record, data_key, state)),
                reverse=(sort_direction < 0),
            )
            state["prediction_table_order"] = [
                str(record.get("record_key")) for record in records if record.get("record_key") is not None
            ]
    return records


def _get_prediction_visible_records(state: dict[str, Any], target_label: str) -> list[dict[str, Any]]:
    """
    Return ordered and search-filtered prediction records.
    """
    ordered_records = _get_prediction_ordered_records(state, target_label)
    return _get_prediction_filtered_records(ordered_records, state)


def _rebuild_prediction_results_table_rows(state: dict[str, Any]) -> None:
    """
    Rebuild only the rows of the prediction results table, preserving the widget itself.
    """
    if not dpg.does_item_exist("prediction_results_table"):
        return

    target_label = _prediction_display_target_label(state)
    records = _get_prediction_visible_records(state, target_label)
    total_pages = max(1, math.ceil(len(records) / max(1, int(state.get("prediction_table_max_per_page", 25)))))
    state["prediction_table_page"] = max(1, min(int(state.get("prediction_table_page", 1)), total_pages))
    selected_key = state.get("prediction_selected_record_key")
    max_per_page = max(1, int(state.get("prediction_table_max_per_page", 25)))
    page = int(state.get("prediction_table_page", 1))
    start = (page - 1) * max_per_page
    end = start + max_per_page
    page_records = records[start:end]

    for row_id in dpg.get_item_children("prediction_results_table", 1) or []:
        dpg.delete_item(row_id)

    state["prediction_table_data"] = []
    state["prediction_row_ids"] = []
    state["prediction_row_record_map"] = {}

    def _on_row_click(row_key: str) -> None:
        if state.get("_prediction_table_building"):
            return
        select_prediction_record(row_key, state, refresh_table=True)

    for record in page_records:
        _ensure_prediction_display_cache(record)
        row_key = record.get("record_key")
        row_active = row_key == selected_key
        base_theme = _make_prediction_selectable_theme(active=row_active)
        cv_delta_value = None
        final_delta_value = None
        if record.get("real_value") is not None and record.get("cv_predicted_value") is not None:
            cv_delta_value = abs(float(record["cv_predicted_value"]) - float(record["real_value"]))
        if record.get("real_value") is not None and record.get("final_predicted_value") is not None:
            final_delta_value = abs(float(record["final_predicted_value"]) - float(record["real_value"]))
        cv_delta_quality = (
            _prediction_quality_score_from_delta(cv_delta_value, state)
            if cv_delta_value is not None else None
        )
        final_delta_quality = (
            _prediction_quality_score_from_delta(final_delta_value, state)
            if final_delta_value is not None else None
        )
        delta_theme = (
            _make_prediction_selectable_theme(
                get_continuous_colormap_color(float(cv_delta_quality), state),
                active=row_active,
            )
            if cv_delta_quality is not None
            else base_theme
        )
        final_delta_theme = (
            _make_prediction_selectable_theme(
                get_continuous_colormap_color(float(final_delta_quality), state),
                active=row_active,
            )
            if final_delta_quality is not None
            else base_theme
        )

        real_text = "-"
        if record.get("real_value") is not None:
            relation = record.get("relation") or "="
            relation = "≤" if relation == "<=" else "≥" if relation == ">=" else relation
            relation_prefix = "" if relation == "=" else f"{relation} "
            real_display = _prediction_record_display_value(record, "real", state)
            if real_display is not None:
                real_text = f"{relation_prefix}{_prediction_format_display_value(real_display, state)}"

        cv_pred_text = _prediction_format_display_value(
            _prediction_display_value_from_internal(record.get("cv_predicted_value"), state),
            state,
        )
        final_pred_text = _prediction_format_display_value(
            _prediction_display_value_from_internal(record.get("final_predicted_value"), state),
            state,
        )
        cv_delta_text = _format_prediction_delta(
            record.get("real_value"),
            record.get("cv_predicted_value"),
            state,
        )
        final_delta_text = _format_prediction_delta(
            record.get("real_value"),
            record.get("final_predicted_value"),
            state,
        )

        row_values = [
            str(record.get("mol_id", "N/A")),
            str(record.get("name", "N/A")),
            real_text,
            cv_pred_text,
            cv_delta_text,
            final_pred_text,
            final_delta_text,
            str(record.get("status", "Unknown")),
        ]

        sortable_record = dict(record)
        sortable_record["real_sort_value"] = float(record["real_value"]) if record.get("real_value") is not None else None
        sortable_record["cv_pred_sort_value"] = float(record["cv_predicted_value"]) if record.get("cv_predicted_value") is not None else None
        sortable_record["final_pred_sort_value"] = float(record["final_predicted_value"]) if record.get("final_predicted_value") is not None else None
        sortable_record["cv_delta_sort_value"] = cv_delta_value if cv_delta_value is not None else 0.0
        sortable_record["final_delta_sort_value"] = final_delta_value if final_delta_value is not None else 0.0

        row_tag = dpg.generate_uuid()
        with dpg.table_row(parent="prediction_results_table", tag=row_tag):
            for idx, value in enumerate(row_values):
                with dpg.table_cell():
                    tag = f"prediction_row_{str(row_key).replace(':', '_')}_{idx}"
                    dpg.add_selectable(
                        label=value,
                        tag=tag,
                        default_value=row_active,
                        span_columns=True,
                        callback=lambda s, a, u: _on_row_click(u),
                        user_data=row_key,
                    )
                    dpg.bind_item_theme(
                        tag,
                        delta_theme if idx == 4 else final_delta_theme if idx == 6 else base_theme,
                    )
        state["prediction_row_ids"].append(row_tag)
        state["prediction_table_data"].append(sortable_record)
        state["prediction_row_record_map"][row_tag] = record

    _set_prediction_page_label(state)
    if dpg.does_item_exist("prediction_tbl_total_pages"):
        dpg.set_value("prediction_tbl_total_pages", f"/ {total_pages}")


def update_prediction_filter_query(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    """
    Filter the prediction table by molecule name or SMILES.
    """
    if state.get("_prediction_table_building"):
        return
    query = str(app_data or "")
    if query == str(state.get("prediction_filter_query", "")):
        return
    state["prediction_filter_query"] = query
    state["prediction_table_page"] = 1
    _refresh_prediction_table_with_loading(state)


def update_prediction_origin_filter(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    """
    Filter prediction records by their origin.
    """
    if state.get("_prediction_table_building"):
        return
    new_value = str(app_data or "All compounds")
    if new_value == str(state.get("prediction_origin_filter", "All compounds")):
        return
    state["prediction_origin_filter"] = new_value
    state["prediction_table_page"] = 1
    _refresh_prediction_table_with_loading(state)


def select_prediction_model_view(model_name: str, state: dict[str, Any]) -> None:
    """
    Activate one model from the comparison table and rebuild the displayed
    prediction outputs.
    """
    state["prediction_selected_model_name"] = model_name
    if not activate_prediction_model_view(model_name, state):
        render_prediction_model_comparison(state)
        return
    state["prediction_filter_query"] = ""
    state["prediction_table_page"] = 1
    state["prediction_sort_spec"] = None
    _sync_prediction_table_order(state)
    _store_prediction_base_state(
        state,
        list(state.get("prediction_results", []) or []),
        dict(state.get("prediction_results_map", {}) or {}),
        list(state.get("prediction_plot_points", []) or []),
        dict(state.get("prediction_metrics", {}) or {}),
        list(state.get("prediction_model_comparison", []) or []),
    )
    render_prediction_model_comparison(state)
    render_prediction_results_table(state, rebuild_comparison=False)
    render_prediction_plot(state)
    selected_key = state.get("prediction_selected_record_key")
    selected_record = (state.get("prediction_results_map", {}) or {}).get(selected_key)
    update_prediction_details(selected_record, state)


def _apply_prediction_model_row_selection(state: dict[str, Any]) -> None:
    """
    Force the comparison-table row selection so it behaves like a single-choice
    table selection instead of independent checkbox-like selectables.
    """
    selected_model_name = str(
        state.get("prediction_selected_model_name")
        or state.get("prediction_active_model_view")
        or state.get("prediction_model_name")
        or ""
    )
    row_widgets = state.get("prediction_model_row_widgets", {}) or {}
    for model_name, widget_tags in row_widgets.items():
        row_active = str(model_name) == selected_model_name
        theme = _make_prediction_selectable_theme(active=row_active)
        for tag in widget_tags:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, row_active)
                dpg.bind_item_theme(tag, theme)


def _on_prediction_model_row_click(sender: Any, app_data: Any, user_data: Any) -> None:
    """
    Handle clicks on the comparison table rows.
    """
    payload = user_data if isinstance(user_data, tuple) and len(user_data) == 2 else None
    if payload is None:
        return
    model_name, state = payload
    model_name = str(model_name or "")
    if not model_name:
        return

    state["prediction_selected_model_name"] = model_name
    if not activate_prediction_model_view(model_name, state):
        render_prediction_model_comparison(state)
        return

    state["prediction_filter_query"] = ""
    state["prediction_table_page"] = 1
    state["prediction_sort_spec"] = None
    _sync_prediction_table_order(state)
    _store_prediction_base_state(
        state,
        list(state.get("prediction_results", []) or []),
        dict(state.get("prediction_results_map", {}) or {}),
        list(state.get("prediction_plot_points", []) or []),
        dict(state.get("prediction_metrics", {}) or {}),
        list(state.get("prediction_model_comparison", []) or []),
    )
    render_prediction_model_comparison(state)
    render_prediction_results_table(state, rebuild_comparison=False)
    render_prediction_plot(state)
    selected_key = state.get("prediction_selected_record_key")
    selected_record = (state.get("prediction_results_map", {}) or {}).get(selected_key)
    update_prediction_details(selected_record, state)
    dpg.set_frame_callback(
        dpg.get_frame_count() + 1,
        lambda: _apply_prediction_model_row_selection(state),
    )


def render_prediction_model_comparison(state: dict[str, Any]) -> None:
    """
    Render the model-comparison area independently from the results table.
    """
    if not dpg.does_item_exist("prediction_model_comparison_window"):
        return

    dpg.delete_item("prediction_model_comparison_window", children_only=True)
    metrics = state.get("prediction_metrics", {}) or {}
    comparison_rows = list(state.get("prediction_model_comparison", []) or [])
    if not comparison_rows:
        return

    comparison_rows = sorted(
        comparison_rows,
        key=lambda row: float(row.get("composite_score", float("-inf"))),
        reverse=True,
    )
    dataset_size = metrics.get("dataset_size", 0)
    training_size = metrics.get("training_size", 0)
    high_match_count = metrics.get("high_match_count", 0)
    selected_model_name = str(
        state.get("prediction_selected_model_name")
        or state.get("prediction_active_model_view")
        or state.get("prediction_model_name")
        or ""
    )
    row_widgets: dict[str, list[str]] = {}

    with dpg.child_window(
        parent="prediction_model_comparison_window",
        tag="prediction_model_comparison_container",
        width=-1,
        auto_resize_y=True,
        border=False,
        no_scrollbar=True,
        horizontal_scrollbar=False,
        no_scroll_with_mouse=True,
    ):
        selected_model = _prediction_selected_model_label(state)
        with dpg.group(horizontal=True):
            dpg.add_text(
                (
                    f"Selected model: {selected_model}    "
                    f"Dataset size: {dataset_size}    "
                    f"Training set: {training_size}    "
                    f"High match molecules: {high_match_count}"
                ),
                wrap=1100,
            )
            dpg.add_button(
                label="Predict external compounds",
                tag="prediction_external_button",
                callback=lambda: _open_prediction_external_window(state),
            )
        with dpg.table(
            tag="prediction_model_comparison_table",
            header_row=True,
            resizable=True,
            policy=dpg.mvTable_SizingStretchProp,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            row_background=True,
            scrollX=True,
        ):
            dpg.bind_item_theme("prediction_model_comparison_table", _ensure_prediction_model_comparison_theme())
            dpg.add_table_column(label="Model", tag="prediction_cmp_col_model")
            dpg.add_table_column(label="Score", tag="prediction_cmp_col_score")
            dpg.add_table_column(label="R²", tag="prediction_cmp_col_r2")
            dpg.add_table_column(label="MAE", tag="prediction_cmp_col_mae")
            dpg.add_table_column(label="RMSE", tag="prediction_cmp_col_rmse")
            dpg.add_table_column(label="MAE/SD", tag="prediction_cmp_col_mae_sd")
            dpg.add_table_column(label="RMSE/SD", tag="prediction_cmp_col_rmse_sd")
            _add_prediction_header_tooltip("prediction_cmp_col_model", "Regression model evaluated in Prediction.")
            _add_prediction_header_tooltip("prediction_cmp_col_score", "Composite score used to rank candidate models. Higher is better.")
            _add_prediction_header_tooltip("prediction_cmp_col_r2", "Cross-validated R². Measures how well the model generalizes to unseen labeled molecules.")
            _add_prediction_header_tooltip("prediction_cmp_col_mae", "Cross-validated Mean Absolute Error in the current activity scale. Lower is better.")
            _add_prediction_header_tooltip("prediction_cmp_col_rmse", "Cross-validated Root Mean Squared Error. Penalizes large errors more strongly than MAE.")
            _add_prediction_header_tooltip("prediction_cmp_col_mae_sd", "MAE normalized by the standard deviation of the target values. Lower is better.")
            _add_prediction_header_tooltip("prediction_cmp_col_rmse_sd", "RMSE normalized by the standard deviation of the target values. Lower is better.")
            for row in comparison_rows:
                model_name = str(row.get("model", "Model"))
                row_active = model_name == selected_model_name
                active_theme = _make_prediction_selectable_theme(active=row_active)
                row_tags: list[str] = []
                with dpg.table_row():
                    for idx, value in enumerate([
                        model_name,
                        f"{float(row.get('composite_score', 0.0)):.3f}",
                        f"{float(row.get('r2', 0.0)):.3f}",
                        f"{float(row.get('mae', 0.0)):.3f}",
                        f"{float(row.get('rmse', 0.0)):.3f}",
                        f"{float(row.get('mae_over_sd', 0.0)):.3f}",
                        f"{float(row.get('rmse_over_sd', 0.0)):.3f}",
                    ]):
                        with dpg.table_cell():
                            tag = f"prediction_model_cmp_{re.sub(r'[^A-Za-z0-9_]+', '_', model_name)}_{idx}"
                            dpg.add_selectable(
                                label=value,
                                tag=tag,
                                default_value=row_active,
                                span_columns=True,
                                callback=_on_prediction_model_row_click,
                                user_data=(model_name, state),
                            )
                            dpg.bind_item_theme(tag, active_theme)
                            row_tags.append(tag)
                row_widgets[model_name] = row_tags

    state["prediction_model_row_widgets"] = row_widgets
    dpg.set_frame_callback(
        dpg.get_frame_count() + 1,
        lambda: _apply_prediction_model_row_selection(state),
    )


def render_prediction_results_table(state: dict[str, Any], rebuild_comparison: bool = True) -> None:
    """
    Render the left-side results table.
    """
    if not dpg.does_item_exist("prediction_results_window"):
        return

    state["_prediction_table_building"] = True
    try:
        if rebuild_comparison and dpg.does_item_exist("prediction_model_comparison_window"):
            dpg.delete_item("prediction_model_comparison_window", children_only=True)
        dpg.delete_item("prediction_results_window", children_only=True)

        _ensure_prediction_pagination_state(state)
        state.setdefault("prediction_filter_query", "")
        _sync_prediction_table_order(state)
        _clamp_prediction_page(state)
        target_label = _prediction_display_target_label(state)
        header_target_label = _prediction_results_target_label(state)
        sort_map = _get_prediction_sort_map(header_target_label)
        metrics = state.get("prediction_metrics", {}) or {}
        total_pages = _prediction_total_pages(state)
        max_per_page = max(1, int(state.get("prediction_table_max_per_page", 25)))
        page = int(state.get("prediction_table_page", 1))

        if metrics:
            dataset_size = metrics.get("dataset_size", 0)
            training_size = metrics.get("training_size", 0)
            predicted_size = metrics.get("predicted_size", 0)
            mae = metrics.get("mae", 0.0)
            rmse = metrics.get("rmse", 0.0)
            r2 = metrics.get("r2", 0.0)
            rmse_over_sd = metrics.get("rmse_over_sd", 0.0)
            mae_over_sd = metrics.get("mae_over_sd", 0.0)
            composite_score = metrics.get("composite_score", 0.0)
            high_match_count = metrics.get("high_match_count", 0)
            if rebuild_comparison:
                render_prediction_model_comparison(state)
            with dpg.group(horizontal=True, parent="prediction_results_window"):
                dpg.add_button(
                    label="Export",
                    tag="prediction_export_table_output_button",
                    callback=lambda: _export_prediction_results_table(state),
                )
                if _prediction_is_loggable(state):
                    dpg.add_checkbox(
                        label="Log scale",
                        tag="prediction_display_linear_checkbox",
                        default_value=not _prediction_display_linear(state),
                        callback=update_prediction_display_scale,
                        user_data=state,
                    )
                dpg.add_spacer(width=state["win_spacer"])
                dpg.add_button(
                    arrow=True,
                    direction=dpg.mvDir_Left,
                    tag="prediction_tbl_prev_page",
                    callback=prev_prediction_page_callback,
                    user_data=state,
                )
                dpg.add_input_text(
                    tag="prediction_tbl_page_number",
                    width=80,
                    default_value=str(page),
                    on_enter=True,
                    callback=jump_to_prediction_page_callback,
                    user_data=state,
                )
                dpg.add_text(f"/ {total_pages}", tag="prediction_tbl_total_pages")
                dpg.add_button(
                    arrow=True,
                    direction=dpg.mvDir_Right,
                    tag="prediction_tbl_next_page",
                    callback=next_prediction_page_callback,
                    user_data=state,
                )
                dpg.add_spacer(width=state["win_spacer"])
                dpg.add_combo(
                    items=["10", "25", "50", "100"],
                    default_value=str(max_per_page),
                    width=80,
                    tag="prediction_tbl_rows_per_page",
                    callback=update_prediction_rows_per_page,
                    user_data=state,
                )
                dpg.add_spacer(width=state["win_spacer"])
                dpg.add_combo(
                    items=["All compounds", "Original prediction set", "Added compounds"],
                    default_value=str(state.get("prediction_origin_filter", "All compounds")),
                    width=100,
                    tag="prediction_tbl_origin_filter",
                    callback=update_prediction_origin_filter,
                    user_data=state,
                )
                dpg.add_spacer(width=state["win_spacer"])
                dpg.add_input_text(
                    hint="Search mol ID, name or SMILES...",
                    width=-1,
                    tag="prediction_tbl_filter_input",
                    default_value=str(state.get("prediction_filter_query", "")),
                    callback=update_prediction_filter_query,
                    user_data=state,
                )

        def _sort_prediction_table(sender: Any, sort_specs: Any) -> None:
            """
            Sort the prediction table according to the selected column.
            """
            if not sort_specs or state.get("_prediction_table_building"):
                return
            if dpg.get_frame_count() <= int(state.get("_prediction_sort_suppress_until_frame", -1)):
                return
            column_id, direction = sort_specs[0]
            column_label = dpg.get_item_label(column_id)
            if column_label not in sort_map:
                return

            data_key = sort_map[column_label]
            state["prediction_results"] = sorted(
                list(state.get("prediction_results", [])),
                key=lambda record: _prediction_sort_key(_prediction_record_sort_value(record, data_key, state)),
                reverse=(direction < 0),
            )
            state["prediction_table_order"] = [
                str(record.get("record_key")) for record in state["prediction_results"] if record.get("record_key") is not None
            ]
            state["prediction_sort_spec"] = (column_label, direction)
            state["prediction_table_page"] = 1
            _refresh_prediction_table_with_loading(state)

        with dpg.table(
            parent="prediction_results_window",
            tag="prediction_results_table",
            header_row=True,
            width=-1,
            height=-1,
            resizable=True,
            sortable=True,
            sort_tristate=False,
            callback=_sort_prediction_table,
            policy=dpg.mvTable_SizingStretchProp,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            row_background=True,
            freeze_rows=1,
            scrollY=True,
            scrollX=True,
        ):
            dpg.add_table_column(label="Mol", tag="prediction_results_col_mol")
            dpg.add_table_column(label="Name", tag="prediction_results_col_name")
            dpg.add_table_column(label=f"Real {header_target_label}", tag="prediction_results_col_real")
            dpg.add_table_column(label=f"CV Pred {header_target_label}", tag="prediction_results_col_cv_pred")
            dpg.add_table_column(label=f"|ΔCV {header_target_label}|", tag="prediction_results_col_cv_delta")
            dpg.add_table_column(label=f"Fit Pred {header_target_label}", tag="prediction_results_col_fit_pred")
            dpg.add_table_column(label=f"|ΔFit {header_target_label}|", tag="prediction_results_col_fit_delta")
            dpg.add_table_column(label="Status", tag="prediction_results_col_status")
        _add_prediction_header_tooltip("prediction_results_col_mol", "Molecule identifier shown in the current prediction table.")
        _add_prediction_header_tooltip("prediction_results_col_name", "Molecule name, when available.")
        _add_prediction_header_tooltip("prediction_results_col_real", "Experimental activity value, if known. This is the reference used to compute deltas.")
        _add_prediction_header_tooltip("prediction_results_col_cv_pred", "Cross-validated prediction available for labeled molecules. It is used to assess validation performance because each value comes from a fold where that molecule was left out of training.")
        _add_prediction_header_tooltip("prediction_results_col_cv_delta", "Absolute difference between Real and CV Pred. Smaller values indicate better cross-validated performance on labeled molecules.")
        _add_prediction_header_tooltip("prediction_results_col_fit_pred", "Prediction from the model fitted on the full labeled training set. This is the estimate used for unknown compounds and for newly added compounds.")
        _add_prediction_header_tooltip("prediction_results_col_fit_delta", "Absolute difference between Real and Fit Pred. It shows how the fully fitted model reproduces labeled compounds, but it is not the validation metric used to rank models.")
        _add_prediction_header_tooltip("prediction_results_col_status", "Prediction status: original cross-validated molecule, unlabeled predicted molecule, or externally added compound.")
        _rebuild_prediction_results_table_rows(state)
    finally:
        state["_prediction_table_building"] = False
        state["_prediction_sort_suppress_until_frame"] = dpg.get_frame_count() + 2


def render_prediction_plot(state: dict[str, Any]) -> None:
    """
    Render the regression/evaluation plot.
    """
    if not dpg.does_item_exist("prediction_plot_window"):
        return

    dpg.delete_item("prediction_plot_window", children_only=True)

    points = list(state.get("prediction_plot_points", []))
    target_label = _prediction_display_target_label(state)
    discrete_a = _discrete_plot_color(state, 0.0)
    discrete_b = _discrete_plot_color(state, 1.0)

    old_bucket_themes = state.get("prediction_plot_bucket_themes", [])
    if isinstance(old_bucket_themes, list):
        for theme_tag in old_bucket_themes:
            if dpg.does_item_exist(theme_tag):
                dpg.delete_item(theme_tag)
    state["prediction_plot_bucket_themes"] = []

    with dpg.plot(
        parent="prediction_plot_window",
        tag="prediction_plot",
        label=f"{target_label} regression",
        width=-1,
        height=-1,
        no_menus=True,
        no_mouse_pos=True,
        equal_aspects=False,
    ):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label=f"Real {target_label}", tag="prediction_x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label=f"CV Predicted {target_label}", tag="prediction_y_axis")
        register_plot_context_popup(
            state,
            context_key="prediction_plot_context",
            plot_tag="prediction_plot",
            x_axis_tag="prediction_x_axis",
            y_axis_tag="prediction_y_axis",
            theme_kind="plot",
        )

        if not points:
            dpg.bind_item_theme("prediction_plot", apply_plot_theme(state))
            return

        for point in points:
            _ensure_prediction_display_cache(point)

        real_values = np.asarray([float(_prediction_record_display_value(p, "real", state)) for p in points], dtype=float)
        pred_values = np.asarray([float(_prediction_record_display_value(p, "pred", state)) for p in points], dtype=float)

        lo = float(min(np.min(real_values), np.min(pred_values)))
        hi = float(max(np.max(real_values), np.max(pred_values)))
        pad = max(0.1, (hi - lo) * 0.08)
        lo -= pad
        hi += pad

        dpg.add_line_series([lo, hi], [lo, hi], parent="prediction_y_axis", tag="prediction_diagonal_line", label="Ideal")
        with dpg.theme() as diag_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, discrete_a, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme("prediction_diagonal_line", diag_theme)

        if len(points) >= 2:
            slope, intercept = np.polyfit(real_values, pred_values, 1)
            x_line = np.linspace(lo, hi, 64)
            y_line = slope * x_line + intercept
            dpg.add_line_series(list(x_line), list(y_line), parent="prediction_y_axis", tag="prediction_regression_line", label="Regression")
            with dpg.theme() as reg_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, discrete_b, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0, category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme("prediction_regression_line", reg_theme)

        bucket_count = int(state.get("prediction_plot_color_bins", 64))
        bucket_count = max(16, min(bucket_count, 96))
        if len(points) > 10_000:
            bucket_count = min(bucket_count, 48)

        bucket_themes: list[Any] = []
        bucket_tags: list[str] = []
        for idx in range(bucket_count):
            ratio = idx / max(1, bucket_count - 1)
            point_color = get_continuous_colormap_color(ratio, state)
            theme_tag = dpg.add_theme()
            with dpg.theme_component(dpg.mvScatterSeries, parent=theme_tag):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, point_color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, point_color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3.0, category=dpg.mvThemeCat_Plots)
            bucket_themes.append(theme_tag)

            series_tag = f"prediction_point_bucket_{idx}"
            dpg.add_scatter_series([], [], parent="prediction_y_axis", tag=series_tag, label="")
            dpg.bind_item_theme(series_tag, theme_tag)
            bucket_tags.append(series_tag)

        unknown_tag = "prediction_point_bucket_unknown"
        dpg.add_scatter_series([], [], parent="prediction_y_axis", tag=unknown_tag, label="")
        with dpg.theme() as unknown_theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, discrete_a, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, discrete_a, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3.0, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(unknown_tag, unknown_theme)

        bucket_xs = [[] for _ in range(bucket_count)]
        bucket_ys = [[] for _ in range(bucket_count)]
        unknown_xs: list[float] = []
        unknown_ys: list[float] = []

        for point in points:
            x_val = float(_prediction_record_display_value(point, "real", state))
            y_val = float(_prediction_record_display_value(point, "pred", state))
            quality = point.get("quality_score")
            if isinstance(quality, (int, float)):
                bucket_idx = int(round(max(0.0, min(1.0, float(quality))) * (bucket_count - 1)))
                bucket_xs[bucket_idx].append(x_val)
                bucket_ys[bucket_idx].append(y_val)
            else:
                unknown_xs.append(x_val)
                unknown_ys.append(y_val)

        for idx, series_tag in enumerate(bucket_tags):
            dpg.set_value(series_tag, [bucket_xs[idx], bucket_ys[idx]])
        dpg.set_value(unknown_tag, [unknown_xs, unknown_ys])
        state["prediction_plot_bucket_themes"] = bucket_themes + [unknown_theme]

        dpg.add_line_series(
            [lo, hi, hi, lo, lo],
            [lo, lo, hi, hi, lo],
            parent="prediction_y_axis",
            tag="prediction_fit_box",
        )
        with dpg.theme() as fit_box_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 0), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 0.0, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme("prediction_fit_box", fit_box_theme)
        dpg.bind_item_theme("prediction_plot", apply_plot_theme(state))
        dpg.fit_axis_data("prediction_x_axis")
        dpg.fit_axis_data("prediction_y_axis")


def _ensure_prediction_plot_handler(state: dict[str, Any]) -> None:
    """
    Add the left-click plot selection handler.
    """
    def _on_prediction_plot_click() -> None:
        if state.get("current_tab") != "prediction_tab":
            return
        if not dpg.does_item_exist("prediction_plot") or not dpg.is_item_hovered("prediction_plot"):
            return

        try:
            mouse_x, mouse_y = dpg.get_plot_mouse_pos()
            x_lower, x_upper = dpg.get_axis_limits("prediction_x_axis")
            y_lower, y_upper = dpg.get_axis_limits("prediction_y_axis")
        except Exception:
            return

        x_tol = max(1e-9, (x_upper - x_lower) * 0.03)
        y_tol = max(1e-9, (y_upper - y_lower) * 0.03)

        nearest_key = None
        best_d2 = 1e300
        for point in state.get("prediction_plot_points", []):
            dx = float(_prediction_to_display_value(float(point["real_value"]), state)) - mouse_x
            dy = float(_prediction_to_display_value(float(point["predicted_value"]), state)) - mouse_y
            if abs(dx) < x_tol and abs(dy) < y_tol:
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    nearest_key = point["record_key"]

        if nearest_key is not None:
            select_prediction_record(nearest_key, state)

    if dpg.does_item_exist("prediction_plot_click_handler"):
        dpg.delete_item("prediction_plot_click_handler")
    dpg.add_mouse_click_handler(
        button=dpg.mvMouseButton_Left,
        parent="handler_registry",
        tag="prediction_plot_click_handler",
        callback=lambda s, a: _on_prediction_plot_click(),
    )


def _create_prediction_output_windows(state: dict[str, Any]) -> None:
    """
    Create the lower resizable output area for Prediction.
    """
    with dpg.table(
        parent="prediction_output_host",
        tag="prediction_layout_table",
        header_row=False,
        width=-1,
        height=-1,
        resizable=True,
        context_menu_in_body=False,
        borders_innerH=False,
        borders_outerH=False,
        borders_innerV=False,
        borders_outerV=False,
        policy=dpg.mvTable_SizingStretchProp,
    ):
        dpg.add_table_column(label="Prediction Table", init_width_or_weight=state["prediction_tbl_width"], width_stretch=True)
        dpg.add_table_column(label="Prediction Right Column", init_width_or_weight=state["prediction_img_win_width"], width_stretch=True)

        with dpg.table_row():
            with dpg.child_window(
                tag="prediction_left_column_window",
                width=-1,
                height=-1,
                border=False,
                no_scrollbar=False,
                horizontal_scrollbar=False,
                no_scroll_with_mouse=False,
            ):
                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=-1,
                    resizable=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    dpg.add_table_column(label="Prediction Left Column", width_stretch=True)

                    with dpg.table_row():
                        with dpg.child_window(
                            tag="prediction_model_comparison_window",
                            width=-1,
                            auto_resize_y=True,
                            border=True,
                            no_scrollbar=True,
                            horizontal_scrollbar=False,
                            no_scroll_with_mouse=True,
                        ):
                            pass

                    with dpg.table_row():
                        with dpg.child_window(
                            tag="prediction_results_window",
                            width=-1,
                            height=-1,
                            border=True,
                            no_scrollbar=False,
                            horizontal_scrollbar=True,
                            no_scroll_with_mouse=False,
                        ):
                            pass

            with dpg.child_window(
                tag="prediction_right_column_window",
                width=-1,
                height=-1,
                border=False,
                no_scrollbar=False,
                horizontal_scrollbar=False,
                no_scroll_with_mouse=False,
            ):
                with dpg.table(
                    header_row=False,
                    width=-1,
                    height=-1,
                    resizable=True,
                    borders_innerH=False,
                    borders_outerH=False,
                    borders_innerV=False,
                    borders_outerV=False,
                    policy=dpg.mvTable_SizingStretchProp,
                ):
                    dpg.add_table_column(label="Prediction Right Column", width_stretch=True)

                    with dpg.table_row():
                        with dpg.child_window(
                            tag="prediction_plot_window",
                            width=-1,
                            height=state["prediction_plot_height"],
                            border=True,
                            no_scrollbar=False,
                            horizontal_scrollbar=False,
                            no_scroll_with_mouse=True,
                        ):
                            pass

                    with dpg.table_row():
                        with dpg.child_window(
                            tag="prediction_details_window",
                            width=-1,
                            height=-1,
                            border=True,
                            no_scrollbar=False,
                            horizontal_scrollbar=False,
                            no_scroll_with_mouse=False,
                        ):
                            pass

    _ensure_prediction_details_widgets(state)
    _ensure_prediction_plot_handler(state)


def build_prediction_output_layout(state: dict[str, Any]) -> None:
    """
    Keep the Prediction output host empty until an analysis is run.
    """
    if dpg.does_item_exist("prediction_output_host"):
        dpg.delete_item("prediction_output_host", children_only=True)
    state["prediction_refresh_colors"] = refresh_prediction_colors


def render_prediction_output(state: dict[str, Any]) -> None:
    """
    Refresh all Prediction output widgets from the current state.
    """
    records = list(state.get("prediction_results", []))
    status_message = state.get("prediction_status_message", "")
    if not records and not status_message:
        if dpg.does_item_exist("prediction_output_host"):
            dpg.delete_item("prediction_output_host", children_only=True)
        return

    if not dpg.does_item_exist("prediction_layout_table"):
        _create_prediction_output_windows(state)

    render_prediction_results_table(state)
    render_prediction_plot(state)

    selected_key = state.get("prediction_selected_record_key")
    if selected_key and selected_key in (state.get("prediction_results_map", {}) or {}):
        select_prediction_record(selected_key, state, refresh_table=False)
        return

    points = list(state.get("prediction_plot_points", []))
    default_selection = records[0]["record_key"] if records else (points[0]["record_key"] if points else None)
    select_prediction_record(default_selection, state, refresh_table=False)


def schedule_prediction_output_render(state: dict[str, Any]) -> None:
    """
    Rebuild the Prediction output in a few consecutive GUI frames instead of in
    one large burst. This is gentler on Dear PyGui/native backends and helps
    avoid instability on larger datasets.
    """
    if state.get("_prediction_render_scheduled"):
        return
    state["_prediction_render_scheduled"] = True

    def _finish() -> None:
        state["_prediction_render_scheduled"] = False

    def _stage_selection() -> None:
        try:
            records = list(state.get("prediction_results", []))
            selected_key = state.get("prediction_selected_record_key")
            if selected_key and selected_key in (state.get("prediction_results_map", {}) or {}):
                select_prediction_record(selected_key, state, refresh_table=False)
                return
            points = list(state.get("prediction_plot_points", []))
            default_selection = records[0]["record_key"] if records else (points[0]["record_key"] if points else None)
            select_prediction_record(default_selection, state, refresh_table=False)
        except Exception as exc:
            log_exception("Prediction", "Error while finalizing Prediction GUI render", exc, indent=1)
            log_traceback("Prediction", indent=2)
        finally:
            _finish()

    def _stage_plot() -> None:
        try:
            render_prediction_plot(state)
            dpg.set_frame_callback(dpg.get_frame_count() + 1, lambda: _stage_selection())
        except Exception as exc:
            log_exception("Prediction", "Error while drawing Prediction plot", exc, indent=1)
            log_traceback("Prediction", indent=2)
            _finish()

    def _stage_table() -> None:
        try:
            if not dpg.does_item_exist("prediction_layout_table"):
                _create_prediction_output_windows(state)
            render_prediction_results_table(state)
            dpg.set_frame_callback(dpg.get_frame_count() + 1, lambda: _stage_plot())
        except Exception as exc:
            log_exception("Prediction", "Error while building Prediction table", exc, indent=1)
            log_traceback("Prediction", indent=2)
            _finish()

    _stage_table()


def refresh_prediction_colors(state: dict[str, Any]) -> None:
    """
    Rebuild table and plot so their colors follow the current colormaps.
    """
    if state.get("current_tab") != "prediction_tab" and not dpg.does_item_exist("prediction_plot"):
        return
    render_prediction_output(state)
