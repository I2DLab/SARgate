"""
=====================
prediction_manager.py
=====================

Prediction manager bar for the Prediction tab.
"""

from typing import Any

import dearpygui.dearpygui as dpg

from app.gui.loading_win import draw_loading_screen
from app.analysis.prediction.prediction_logic import (
    collect_prediction_activities,
    prediction_feature_items,
    prediction_model_items,
    prediction_scope_items,
    run_prediction_analysis,
)
from app.analysis.prediction.prediction_table import (
    build_prediction_output_layout,
    load_prediction_results_table,
    render_prediction_output,
    schedule_prediction_output_render,
)


def prediction_scope_tooltips() -> dict[str, str]:
    """
    Human-readable explanations for prediction scopes.
    """
    return {
        "Dataset (prepared)": "Use the prepared dataset: filters, duplicate handling, and preparation choices already applied.",
        "Dataset (full)": "Use the full input library (not prepared), including molecules filtered out of the final prepared dataset. Molecules without the selected activity are predicted only.",
    }


def prediction_feature_tooltips() -> dict[str, str]:
    """
    Human-readable explanations for feature sets.
    """
    return {
        "Morgan (1024 bits)": "Circular Morgan fingerprint with 1024 bits. Fastest option and often a good baseline.",
        "Morgan + RDKit descriptors": "Morgan 1024-bit fingerprint plus curated RDKit physicochemical descriptors.",
        "Morgan (2048 bits) + RDKit descriptors": "Richer Morgan 2048-bit fingerprint plus curated RDKit descriptors. Usually the strongest general-purpose feature set, but a bit slower.",
    }


def prediction_model_tooltips() -> dict[str, str]:
    """
    Human-readable explanations for model choices.
    """
    return {
        "Auto (fast CV)": "Test only Extra Trees and Random Forest with cross-validation, then keep the best one. Best default when you want speed and robustness.",
        "Auto (extended CV)": "Test all supported sklearn models with cross-validation, then keep the best one.",
        "Random Forest": "Robust tree ensemble with good generalization on curated QSAR datasets.",
        "Extra Trees": "More randomized tree ensemble. Often very strong and slightly more flexible than Random Forest on fingerprint-based features.",
        "HistGradient Boosting": "Gradient boosting model for tabular data. Can be strong, but not always better than tree ensembles on these datasets.",
        "SVR": "Kernel support vector regression. Can model nonlinear structure well, but usually becomes slower on large datasets.",
        "MLP": "Feed-forward neural network on the engineered feature vectors. More sensitive to tuning and often less stable than tree ensembles.",
    }


def show_prediction_window(state: dict[str, Any]) -> None:
    """
    Build the Prediction tab scaffold and its manager bar.
    """

    if dpg.does_item_exist("prediction_manager_host"):
        dpg.delete_item("prediction_manager_host", children_only=True)
    if dpg.does_item_exist("prediction_output_host"):
        dpg.delete_item("prediction_output_host", children_only=True)

    scopes = prediction_scope_items(state)
    scope_tooltips = prediction_scope_tooltips()
    feature_items = prediction_feature_items()
    feature_tooltips = prediction_feature_tooltips()
    model_items = prediction_model_items()
    model_tooltips = prediction_model_tooltips()
    default_scope = "Dataset (prepared)" if "Dataset (prepared)" in scopes else (scopes[0] if scopes else "Dataset (prepared)")
    activities = collect_prediction_activities(state, default_scope)
    default_activity = activities[0] if activities else "No activities found"

    state["prediction_available_activities"] = activities
    state["prediction_selected_activity"] = default_activity
    state["prediction_selected_scope"] = default_scope
    state.setdefault("prediction_display_linear", False)
    state.setdefault("prediction_feature_cache", {})

    def _run_prediction() -> None:
        options = {
            "activity_name": str(dpg.get_value("prediction_activity_choice") or "").strip(),
            "scope": str(dpg.get_value("prediction_scope_choice") or "Dataset (prepared)"),
            "model_name": str(dpg.get_value("prediction_model_choice") or "Random Forest"),
            "feature_mode": str(dpg.get_value("prediction_feature_choice") or "Morgan (1024 bits)"),
            "include_undefined": bool(dpg.get_value("prediction_include_undefined_choice")),
        }
        state["_prediction_progress_value"] = 0.0
        state["_prediction_render_scheduled"] = False

        for item_tag in ["run_prediction_button", "load_prediction_button"]:
            if dpg.does_item_exist(item_tag):
                dpg.configure_item(item_tag, enabled=False)

        draw_loading_screen(state, bg=True)
        try:
            run_prediction_analysis(state, options=options)
            schedule_prediction_output_render(state)
        finally:
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")
            for item_tag in ["run_prediction_button", "load_prediction_button"]:
                if dpg.does_item_exist(item_tag):
                    dpg.configure_item(item_tag, enabled=True)

    def update_prediction_options(sender: Any, app_data: Any, user_data: Any) -> None:
        selected_scope = app_data
        scope_activities = collect_prediction_activities(state, str(selected_scope))
        state["prediction_selected_scope"] = str(selected_scope)
        state["prediction_available_activities"] = scope_activities
        if scope_activities:
            dpg.configure_item("prediction_activity_choice", items=scope_activities, enabled=True)
            dpg.set_value("prediction_activity_choice", scope_activities[0])
            state["prediction_selected_activity"] = scope_activities[0]
            if dpg.does_item_exist("run_prediction_button"):
                dpg.configure_item("run_prediction_button", enabled=True)
        else:
            dpg.configure_item("prediction_activity_choice", items=["No activities found"], enabled=False)
            dpg.set_value("prediction_activity_choice", "No activities found")
            state["prediction_selected_activity"] = "No activities found"
            if dpg.does_item_exist("run_prediction_button"):
                dpg.configure_item("run_prediction_button", enabled=False)

    def _add_static_tooltip_lines(parent_tag: str, title: str, lines: list[str]) -> None:
        with dpg.tooltip(parent_tag):
            dpg.add_text(title)
            dpg.add_separator()
            for line in lines:
                dpg.add_text(line, wrap=560)

    scope_tooltip_lines = [
        f"Dataset (prepared): {scope_tooltips.get('Dataset (prepared)', '')}",
        f"Dataset (full): {scope_tooltips.get('Dataset (full)', '')}",
        "Subsets: run the prediction on a single subset instead of the whole dataset.",
    ]
    feature_tooltip_lines = [f"{item}: {feature_tooltips.get(item, item)}" for item in feature_items]
    model_tooltip_lines = [f"{item}: {model_tooltips.get(item, item)}" for item in model_items]

    with dpg.child_window(
        parent="prediction_manager_host",
        tag="prediction_manager_window",
        width=-1,
        auto_resize_y=True,
        border=False,
        no_scrollbar=True,
        horizontal_scrollbar=False,
        no_scroll_with_mouse=True,
    ):
        with dpg.group(horizontal=True):
            with dpg.group(tag="prediction_manager_controls"):
                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=True, tag="prediction_scope_group"):
                        dpg.add_text("Training scope:")
                        dpg.add_combo(
                            width=state["prediction_manager_combo_width"],
                            items=scopes,
                            default_value=default_scope,
                            tag="prediction_scope_choice",
                            callback=update_prediction_options,
                            user_data=state,
                        )
                    _add_static_tooltip_lines("prediction_scope_group", "Training scope", scope_tooltip_lines)

                    dpg.add_spacer(width=state["prediction_manager_combo_spacer"])

                    dpg.add_text("Activity type:")
                    if activities:
                        dpg.add_combo(
                            width=state["prediction_manager_combo_width"],
                            items=activities,
                            default_value=default_activity,
                            tag="prediction_activity_choice",
                        )
                    else:
                        dpg.add_combo(
                            width=state["prediction_manager_combo_width"],
                            items=["No activities found"],
                            default_value="No activities found",
                            tag="prediction_activity_choice",
                            enabled=False,
                            no_arrow_button=True,
                        )

                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=True, tag="prediction_feature_group"):
                        dpg.add_text("Features:")
                        dpg.add_combo(
                            width=state["prediction_manager_combo_width"],
                            items=feature_items,
                            default_value="Morgan (2048 bits) + RDKit descriptors",
                            tag="prediction_feature_choice",
                        )
                    _add_static_tooltip_lines("prediction_feature_group", "Features", feature_tooltip_lines)

                    dpg.add_spacer(width=state["prediction_manager_combo_spacer"])

                    with dpg.group(horizontal=True, tag="prediction_model_group"):
                        dpg.add_text("Model:")
                        dpg.add_combo(
                            width=state["prediction_manager_combo_width"],
                            items=model_items,
                            default_value="Random Forest",
                            tag="prediction_model_choice",
                        )
                    _add_static_tooltip_lines("prediction_model_group", "Model", model_tooltip_lines)

                    dpg.add_spacer(width=state["prediction_manager_combo_spacer"])

                    dpg.add_checkbox(
                        label="Read undefined",
                        tag="prediction_include_undefined_choice",
                        default_value=False,
                    )
                    with dpg.tooltip("prediction_include_undefined_choice"):
                        dpg.add_text(
                            "Include activity values with qualifiers such as <, <=, >, >=.\n"
                            "They are converted to their numeric part and used during training.",
                        )
            
            dpg.add_spacer(width=state["prediction_manager_combo_spacer"])

            with dpg.group(horizontal=False):
                dpg.add_button(
                    label="Load Prediction",
                    tag="load_prediction_button",
                    callback=lambda: load_prediction_results_table(state),
                )
                dpg.add_button(
                    label="Run Prediction",
                    tag="run_prediction_button",
                    callback=_run_prediction,
                    enabled=bool(activities),
                )

    build_prediction_output_layout(state)
    render_prediction_output(state)
