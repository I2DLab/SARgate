"""
==========================
chemspace_manager_common.py
==========================

Unified managers for ChemSpace views.
"""

from __future__ import annotations

import os
from typing import Any

import dearpygui.dearpygui as dpg
import pandas as pd

from app.analysis.chemspace.chemspace_logic_common import perform_tsne, perform_umap
from app.analysis.chemspace.chemspace_plot_common import (
    draw_chemspace_dendrogram,
    draw_descriptors_2d,
    draw_descriptors_3d,
    draw_pca_plot,
)
from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress

pd.set_option("future.no_silent_downcasting", True)


def _subset_items(state: dict[str, Any]) -> list[str]:
    subsets = list(state["smiles_rgd_dict"].keys())
    if "Dataset" not in subsets:
        subsets = ["Dataset"] + subsets
    return subsets


def _activity_items(state: dict[str, Any], subset: str) -> list[str]:
    activities = state["bioact_types_dict"].get(subset, {}).get("bioactivities", [])
    if "No activities" not in activities:
        activities = ["No activities"] + activities
    return activities or ["No activities"]


def _draw_plot_with_loading(state: dict[str, Any], subset: str, activity: str, runner: Any) -> None:
    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)
    csv_file = os.path.join(state["summary_dir"], f"{subset}_summary.csv")
    data = pd.read_csv(csv_file)
    set_loading_screen_progress(state, 2)
    runner(activity, data, subset, state)
    set_loading_screen_progress(state, 100)
    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")


def _descriptor_items() -> list[str]:
    return [
        "MW", "logP", "HBA", "HBD", "RotBonds", "TPSA", "MolarRefractivity", "fraction_csp3",
        "NumRings", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
        "Kappa1", "Kappa2", "Kappa3", "Chi0", "Chi1", "Chi2", "Chi3", "Chi4",
    ]


def show_pca_window(state: dict[str, Any]) -> None:
    plot_manager_tag = "pca_manager_window"
    control_gap = max(6, state["win_spacer"] * 2)

    def update_pca_options(sender: Any, app_data: Any, user_data: Any) -> None:
        subset = app_data
        activities = _activity_items(user_data, subset)
        if activities and set(activities) != {"No activities"}:
            dpg.configure_item("pca_activity_type", items=activities, enabled=True, no_arrow_button=False)
            dpg.set_value("pca_activity_type", activities[1] if activities[0] == "No activities" and len(activities) > 1 else activities[0])
            dpg.show_item("pca_draw_plot_button")
        else:
            dpg.configure_item("pca_activity_type", items=["No activities"], enabled=False, no_arrow_button=True)
            dpg.set_value("pca_activity_type", "No activities")
            dpg.hide_item("pca_draw_plot_button")

    with dpg.child_window(parent=plot_manager_tag, width=-1, auto_resize_y=True, no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False, border=False):
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=True):
                subsets = _subset_items(state)
                default_subset = subsets[0] if subsets else "Dataset"
                activities = _activity_items(state, default_subset)

                with dpg.group(horizontal=True, tag="pca_subset_choice_group"):
                    dpg.add_text("Subset:")
                    dpg.add_combo(width=state["plots_manager_combo_width"], items=subsets, height_mode=dpg.mvComboHeight_Large, default_value=default_subset, tag="pca_subset_choice", callback=update_pca_options, user_data=state)
                    dpg.add_spacer(width=control_gap)

                with dpg.group(horizontal=True, tag="pca_activity_type_group"):
                    dpg.add_text("Activity:")
                    dpg.add_combo(width=state["plots_manager_combo_width"], height_mode=dpg.mvComboHeight_Largest, items=activities, default_value=activities[1] if len(activities) > 1 else activities[0], tag="pca_activity_type", enabled=bool(activities))
                    dpg.add_spacer(width=control_gap)

                with dpg.group(horizontal=True, tag="pca_dimension_combo_group"):
                    dpg.add_text("2D/3D:")
                    dpg.add_combo(height_mode=dpg.mvComboHeight_Largest, width=state["plots_manager_combo_width"], items=["2D", "3D"], default_value="2D", tag="pca_dimension_combo")
                    dpg.add_spacer(width=control_gap)

                with dpg.group(horizontal=True, tag="pca_fingerprint_algorithm_combo_group"):
                    dpg.add_text("Fingerprint:")
                    dpg.add_combo(
                        tag="pca_fingerprint_algorithm_combo",
                        height_mode=dpg.mvComboHeight_Largest,
                        width=state["plots_manager_combo_width"],
                        items=[
                            "Morgan Fingerprint", "RDKit Fingerprint", "Atom Pair Fingerprint",
                            "MACCS Keys", "Topological Torsion Fingerprint",
                            "Pattern Fingerprint", "Layered Fingerprint",
                        ],
                        default_value="Morgan Fingerprint",
                    )
                    dpg.add_spacer(width=control_gap)

                with dpg.group(horizontal=True, tag="pca_include_undefined_choice_group"):
                    dpg.add_checkbox(label="Include undefined", tag="pca_include_undefined_choice", default_value=False)
                    with dpg.tooltip("pca_include_undefined_choice"):
                        dpg.add_text("Include molecules with undefined activity values (<, ≤, ≥, >).\nUndefined values will be treated as exact ones (=)")
                    dpg.add_spacer(width=control_gap)

            dpg.add_button(label="Draw Plot", tag="pca_draw_plot_button", callback=lambda: try_draw_pca(state), show=bool(activities))


def show_descriptors_window(state: dict[str, Any]) -> None:
    """
    Build the manager controls for descriptors plots.
    """
    plot_manager_tag = "descriptors_manager_window"
    control_gap = max(6, state["win_spacer"] * 2)
    descriptors = _descriptor_items()

    def update_descriptors_options(sender: Any, app_data: Any, user_data: Any) -> None:
        subset = app_data
        activities = user_data["bioact_types_dict"][subset]["bioactivities"]
        full_descriptors = descriptors + activities
        dpg.configure_item("descriptors_axis_x_combo", items=full_descriptors)
        dpg.set_value("descriptors_axis_x_combo", full_descriptors[0])
        dpg.configure_item("descriptors_axis_y_combo", items=full_descriptors)
        dpg.set_value("descriptors_axis_y_combo", full_descriptors[1])
        dpg.configure_item("descriptors_axis_z_combo", items=full_descriptors)
        dpg.set_value("descriptors_axis_z_combo", full_descriptors[2])
        dpg.configure_item("descriptors_color_combo", items=full_descriptors)
        dpg.set_value("descriptors_color_combo", activities[0] if activities else full_descriptors[3])
        dpg.show_item("descriptors_draw_plot_button")

    with dpg.child_window(
        parent=plot_manager_tag,
        width=-1,
        auto_resize_y=True,
        no_scrollbar=False,
        horizontal_scrollbar=False,
        no_scroll_with_mouse=False,
        border=False,
    ):
        with dpg.group(horizontal=True):
            with dpg.group():
                subsets = _subset_items(state)
                default_subset = subsets[0] if subsets else "Dataset"
                activities = state["bioact_types_dict"].get(default_subset, {}).get("bioactivities", [])
                full_descriptors = descriptors + activities

                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=True, tag="descriptors_subset_choice_group"):
                        dpg.add_text("Subset:")
                        dpg.add_combo(
                            width=state["plots_manager_combo_width"],
                            items=subsets,
                            height_mode=dpg.mvComboHeight_Large,
                            default_value=default_subset,
                            tag="descriptors_subset_choice",
                            callback=update_descriptors_options,
                            user_data=state,
                        )
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="descriptors_dimension_combo_group"):
                        dpg.add_text("2D/3D:")
                        dpg.add_combo(
                            height_mode=dpg.mvComboHeight_Largest,
                            width=state["plots_manager_combo_width"],
                            items=["2D", "3D"],
                            default_value="2D",
                            tag="descriptors_dimension_combo",
                        )
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="descriptors_include_undefined_choice_group"):
                        dpg.add_checkbox(label="Include undefined", tag="descriptors_include_undefined_choice", default_value=False)
                        with dpg.tooltip("descriptors_include_undefined_choice"):
                            dpg.add_text("Include molecules with undefined activity values (<, ≤, ≥, >).\nUndefined values will be treated as exact ones (=)")

                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=True, tag="descriptors_axis_x_combo_group"):
                        dpg.add_text("X:")
                        dpg.add_combo(
                            height_mode=dpg.mvComboHeight_Largest,
                            width=state["plots_manager_combo_width"],
                            items=full_descriptors,
                            default_value=full_descriptors[0],
                            tag="descriptors_axis_x_combo",
                        )
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="descriptors_axis_y_combo_group"):
                        dpg.add_text("Y:")
                        dpg.add_combo(
                            height_mode=dpg.mvComboHeight_Largest,
                            width=state["plots_manager_combo_width"],
                            items=full_descriptors,
                            default_value=full_descriptors[1],
                            tag="descriptors_axis_y_combo",
                        )
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="descriptors_axis_z_combo_group"):
                        dpg.add_text("Z (3D only):")
                        dpg.add_combo(
                            height_mode=dpg.mvComboHeight_Largest,
                            width=state["plots_manager_combo_width"],
                            items=full_descriptors,
                            default_value=full_descriptors[2],
                            tag="descriptors_axis_z_combo",
                        )
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="descriptors_color_combo_group"):
                        dpg.add_text("Colorscale (3D only):")
                        dpg.add_combo(
                            height_mode=dpg.mvComboHeight_Largest,
                            width=state["plots_manager_combo_width"],
                            items=full_descriptors,
                            default_value=full_descriptors[-1],
                            tag="descriptors_color_combo",
                        )

            dpg.add_spacer(width=max(12, state["win_spacer"] * 2))
            with dpg.group():
                dpg.add_button(label="Draw Plot", tag="descriptors_draw_plot_button", callback=lambda: try_draw_descriptors(state))


def show_dendrogram_window(state: dict[str, Any]) -> None:
    subsets = _subset_items(state)

    def _dendrogram_safe_tag(text: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in text)

    def _refresh_dendrogram_subset_panel() -> None:
        panel_tag = "dendrogram_subset_panel"
        if not dpg.does_item_exist(panel_tag):
            return
        dpg.delete_item(panel_tag, children_only=True)
        visible = state.setdefault("chemspace_dendrogram_visible_subsets", set(subsets))
        state["chemspace_dendrogram_cb2subset"] = {}
        state["chemspace_dendrogram_bulk_toggle"] = False
        subset_count = len(subsets)
        if "chemspace_dendrogram_highlight_subset" not in state:
            state["chemspace_dendrogram_highlight_subset"] = 0
        else:
            try:
                state["chemspace_dendrogram_highlight_subset"] = max(
                    0,
                    min(int(state.get("chemspace_dendrogram_highlight_subset", 0)), subset_count),
                )
            except Exception:
                state["chemspace_dendrogram_highlight_subset"] = 0

        def _refresh_highlight() -> None:
            if callable(state.get("chemspace_dendrogram_refresh_highlight")):
                try:
                    state["chemspace_dendrogram_refresh_highlight"]()
                except Exception:
                    pass

        def _on_highlight_subset_change(sender: Any, app_data: Any) -> None:
            try:
                state["chemspace_dendrogram_highlight_subset"] = max(0, min(int(app_data), subset_count))
            except Exception:
                state["chemspace_dendrogram_highlight_subset"] = 0
            _refresh_highlight()

        def _toggle_one(sender: Any, app_data: Any) -> None:
            if state.get("chemspace_dendrogram_bulk_toggle", False):
                return
            subset = state.get("chemspace_dendrogram_cb2subset", {}).get(sender)
            if not subset:
                return
            if app_data:
                visible.add(subset)
            else:
                visible.discard(subset)
            if dpg.does_item_exist("chemspace_dendrogram_select_all"):
                dpg.set_value("chemspace_dendrogram_select_all", bool(subsets) and all(s in visible for s in subsets))

        def _toggle_all(select_all: bool) -> None:
            visible.clear()
            if select_all:
                visible.update(subsets)
            state["chemspace_dendrogram_bulk_toggle"] = True
            try:
                for subset in subsets:
                    cb = f"chemspace_dendrogram_cb__{_dendrogram_safe_tag(subset)}"
                    if dpg.does_item_exist(cb):
                        dpg.set_value(cb, bool(select_all))
            finally:
                state["chemspace_dendrogram_bulk_toggle"] = False

        dpg.add_button(label="Draw Plot", tag="dendrogram_draw_plot_button", parent=panel_tag, width=-1, callback=lambda: try_draw_dendrogram(state))
        dpg.add_spacer(height=max(4, state["win_spacer"]), parent=panel_tag)
        dpg.add_text("Highlight subset:", parent=panel_tag)
        dpg.add_input_int(
            tag="chemspace_dendrogram_highlight_input",
            parent=panel_tag,
            width=-1,
            default_value=int(state.get("chemspace_dendrogram_highlight_subset", 0)),
            min_value=0,
            max_value=subset_count,
            min_clamped=True,
            max_clamped=True,
            step=1,
            callback=_on_highlight_subset_change,
        )
        dpg.add_spacer(height=max(4, state["win_spacer"]), parent=panel_tag)

        dpg.add_text("Subsets to display:", parent=panel_tag)
        dpg.add_checkbox(
            label="Select all",
            tag="chemspace_dendrogram_select_all",
            default_value=bool(subsets) and all(s in visible for s in subsets),
            parent=panel_tag,
            callback=lambda s, a: _toggle_all(bool(a)),
        )
        dpg.add_separator(parent=panel_tag)
        with dpg.child_window(
            parent=panel_tag,
            width=-1,
            height=-1,
            tag="chemspace_dendrogram_subset_controls",
            border=False,
            no_scrollbar=False,
            horizontal_scrollbar=False,
            no_scroll_with_mouse=False,
        ):
            pass
        if dpg.does_item_exist("manager_panel_theme"):
            dpg.bind_item_theme("chemspace_dendrogram_subset_controls", "manager_panel_theme")
        for subset in subsets:
            cb_tag = f"chemspace_dendrogram_cb__{_dendrogram_safe_tag(subset)}"
            state["chemspace_dendrogram_cb2subset"][cb_tag] = subset
            dpg.add_checkbox(
                label=subset.replace("subset_", "Subset "),
                tag=cb_tag,
                default_value=(subset in visible),
                parent="chemspace_dendrogram_subset_controls",
                callback=_toggle_one,
            )
    _refresh_dendrogram_subset_panel()


def try_draw_dendrogram(state: dict[str, Any]) -> None:
    state["current_chemspace_subtab"] = "dendrogram_tab"
    for tag in [
        "dendrogram_window",
        "chemspace_scaffold_hierarchical_dendrogram",
        "chemspace_cluster_threshold_drag",
        "chemspace_dendro_frame_series",
    ]:
        if dpg.does_item_exist(tag):
            try:
                dpg.delete_item(tag, children_only=True)
            except Exception:
                dpg.delete_item(tag)
    try:
        draw_loading_screen(state, bg=False)
        set_loading_screen_progress(state, 1)
        draw_chemspace_dendrogram(state)
        set_loading_screen_progress(state, 100)
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        with dpg.window(label="Dendrogram Error", tag="chemspace_dendrogram_error_window", modal=False, no_resize=True, no_collapse=True, autosize=True):
            dpg.add_text(f"An error occurred during the dendrogram plot generation:\n\n{e}")
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("chemspace_dendrogram_error_window"))


def try_draw_descriptors(state: dict[str, Any]) -> None:
    """
    Run descriptors plotting with cleanup and error handling.
    """
    if dpg.get_value("descriptors_dimension_combo") == "2D":
        for tag in [
            "descriptors_window",
            "descriptors_details_window",
            "descriptors_mol_image_widget",
            "descriptors_click_handler",
            "descriptors_mouse_move_handler",
            "tooltip_scatter",
            "tooltip_text_scatter",
        ]:
            if dpg.does_item_exist(tag):
                try:
                    dpg.delete_item(tag, children_only=True)
                except Exception:
                    dpg.delete_item(tag)

    try:
        draw_descriptors(state)
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        with dpg.window(label="Plot Error", tag="descriptors_error_window", modal=False, no_resize=True, no_collapse=True, autosize=True):
            dpg.add_text(f"An error occurred during the descriptors plot generation:\n\n{e}")
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("descriptors_error_window"))


def draw_descriptors(state: dict[str, Any]) -> None:
    """
    Draw the descriptors plot from current UI selections.
    """
    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)
    subset = dpg.get_value("descriptors_subset_choice")
    read_undefined = dpg.get_value("descriptors_include_undefined_choice")
    csv_file = os.path.join(state["summary_dir"], f"{subset}_summary.csv")
    set_loading_screen_progress(state, 3)
    data = pd.read_csv(csv_file)
    set_loading_screen_progress(state, 5)

    if dpg.get_value("descriptors_dimension_combo") == "2D":
        draw_descriptors_2d(subset, data, read_undefined, state)
        set_loading_screen_progress(state, 98)
        dpg.fit_axis_data("descriptors_x_axis")
        dpg.fit_axis_data("descriptors_y_axis")
    else:
        draw_descriptors_3d(data, read_undefined, state)
        set_loading_screen_progress(state, 99)

    set_loading_screen_progress(state, 100)
    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")


def try_draw_pca(state: dict[str, Any]) -> None:
    if dpg.get_value("pca_dimension_combo") == "2D":
        for tag in [
            "pca_window",
            "pca_details_window",
            "pca_mol_image_widget",
            "pca_scatter_layer",
            "tooltip_pca",
            "tooltip_text_pca",
            "pca_click_handler",
            "pca_mouse_move_handler",
        ]:
            if dpg.does_item_exist(tag):
                try:
                    dpg.delete_item(tag, children_only=True)
                except Exception:
                    dpg.delete_item(tag)
    try:
        subset = dpg.get_value("pca_subset_choice")
        activity = dpg.get_value("pca_activity_type")
        _draw_plot_with_loading(state, subset, activity, draw_pca_plot)
        set_loading_screen_progress(state, 98)
        if dpg.get_value("pca_dimension_combo") == "2D":
            dpg.fit_axis_data("pca_x_axis")
            dpg.fit_axis_data("pca_y_axis")
        set_loading_screen_progress(state, 99)
        set_loading_screen_progress(state, 100)
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        with dpg.window(label="Plot Error", tag="pca_error_window", modal=False, no_resize=True, no_collapse=True, autosize=True):
            dpg.add_text(f"An error occurred during the PCA plot generation:\n\n{e}")
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("pca_error_window"))


def show_umap_window(state: dict[str, Any]) -> None:
    plot_manager_tag = "umap_manager_window"
    control_gap = max(6, state["win_spacer"] * 2)

    def update_umap_options(sender: Any, app_data: Any, user_data: Any) -> None:
        subset = app_data
        activities = _activity_items(user_data, subset)
        if activities and set(activities) != {"No activities"}:
            dpg.configure_item("umap_activity_type", items=activities, enabled=True, no_arrow_button=False)
            dpg.set_value("umap_activity_type", activities[1] if activities[0] == "No activities" and len(activities) > 1 else activities[0])
            dpg.show_item("umap_draw_plot_button")
        else:
            dpg.configure_item("umap_activity_type", items=["No activities"], enabled=False, no_arrow_button=True)
            dpg.set_value("umap_activity_type", "No activities")
            dpg.hide_item("umap_draw_plot_button")

    with dpg.child_window(parent=plot_manager_tag, width=-1, auto_resize_y=True, no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False, border=False):
        with dpg.group(horizontal=True):
            subsets = _subset_items(state)
            default_subset = subsets[0] if subsets else "Dataset"
            activities = _activity_items(state, default_subset)
            default_activity = activities[1] if len(activities) > 1 and activities[0] == "No activities" else activities[0]

            with dpg.group():
                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=True, tag="umap_subset_choice_group"):
                        dpg.add_text("Subset:")
                        dpg.add_combo(width=state["plots_manager_combo_width"], height_mode=dpg.mvComboHeight_Large, items=subsets, default_value=default_subset, tag="umap_subset_choice", callback=update_umap_options, user_data=state)
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="umap_activity_type_group"):
                        dpg.add_text("Activity:")
                        dpg.add_combo(width=state["plots_manager_combo_width"], height_mode=dpg.mvComboHeight_Large, items=activities, default_value=default_activity, tag="umap_activity_type", enabled=bool(activities))
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="umap_dimension_combo_group"):
                        dpg.add_text("2D/3D:")
                        dpg.add_combo(tag="umap_dimension_combo", height_mode=dpg.mvComboHeight_Large, width=state["plots_manager_combo_width"], items=["2D", "3D"], default_value="2D")
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="umap_fingerprint_algorithm_combo_group"):
                        dpg.add_text("Fingerprint:")
                        dpg.add_combo(tag="umap_fingerprint_algorithm_combo", height_mode=dpg.mvComboHeight_Large, width=state["plots_manager_combo_width"], items=["Morgan Fingerprint", "RDKit Fingerprint", "Atom Pair Fingerprint", "MACCS Keys", "Topological Torsion Fingerprint", "Pattern Fingerprint", "Layered Fingerprint"], default_value="Morgan Fingerprint")
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="umap_include_undefined_choice_group"):
                        dpg.add_checkbox(label="Include undefined", tag="umap_include_undefined_choice", default_value=False)
                        with dpg.tooltip("umap_include_undefined_choice"):
                            dpg.add_text("Include molecules with undefined activity values (<, <=, >=, >).\nUndefined values are converted to exact values for the projection.")

                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=True, tag="umap_neighbors_combo_group"):
                        dpg.add_text("Neighbors:")
                        dpg.add_combo(tag="umap_neighbors_combo", height_mode=dpg.mvComboHeight_Large, width=110, items=["5", "10", "15", "25", "50"], default_value="5")
                        with dpg.tooltip("umap_neighbors_combo"):
                            dpg.add_text("Number of neighboring molecules used to learn the local manifold.\nLower values emphasize local clusters; higher values preserve broader structure.")
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="umap_min_dist_combo_group"):
                        dpg.add_text("Min dist:")
                        dpg.add_combo(tag="umap_min_dist_combo", height_mode=dpg.mvComboHeight_Large, width=110, items=["0.0", "0.1", "0.25", "0.5", "0.8"], default_value="0.8")
                        with dpg.tooltip("umap_min_dist_combo"):
                            dpg.add_text("Controls how tightly UMAP packs points together.\nLower values create denser clusters; higher values leave more space between neighborhoods.")
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="umap_metric_combo_group"):
                        dpg.add_text("Metric:")
                        dpg.add_combo(tag="umap_metric_combo", height_mode=dpg.mvComboHeight_Large, width=140, items=["euclidean", "manhattan", "cosine", "jaccard"], default_value="jaccard")
                        with dpg.tooltip("umap_metric_combo"):
                            dpg.add_text("Distance metric used to compare molecular fingerprints before the UMAP projection.\nJaccard is usually the most natural choice for binary fingerprints.")

            dpg.add_spacer(width=max(12, state["win_spacer"] * 2))
            with dpg.group():
                dpg.add_button(label="Draw Plot", tag="umap_draw_plot_button", callback=lambda: try_draw_umap(state), show=bool(activities))


def try_draw_umap(state: dict[str, Any]) -> None:
    if dpg.get_value("umap_dimension_combo") == "2D":
        for tag in ["umap_window", "umap_details_window", "umap_gradient_bar_window", "umap_plot_handler_registry"]:
            if dpg.does_item_exist(tag):
                try:
                    dpg.delete_item(tag, children_only=True)
                except Exception:
                    dpg.delete_item(tag)
    try:
        subset = dpg.get_value("umap_subset_choice")
        activity = dpg.get_value("umap_activity_type")
        _draw_plot_with_loading(state, subset, activity, perform_umap)
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        with dpg.window(label="UMAP Error", tag="umap_error_window", modal=False, no_resize=True, no_collapse=True, autosize=True):
            dpg.add_text(f"An error occurred during the UMAP plot generation:\n\n{e}")
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("umap_error_window"))


def show_tsne_window(state: dict[str, Any]) -> None:
    plot_manager_tag = "tsne_manager_window"
    control_gap = max(6, state["win_spacer"] * 2)

    def update_tsne_options(sender: Any, app_data: Any, user_data: Any) -> None:
        subset = app_data
        activities = _activity_items(user_data, subset)
        if activities and set(activities) != {"No activities"}:
            dpg.configure_item("tsne_activity_type", items=activities, enabled=True, no_arrow_button=False)
            dpg.set_value("tsne_activity_type", activities[1] if activities[0] == "No activities" and len(activities) > 1 else activities[0])
            dpg.show_item("tsne_draw_plot_button")
        else:
            dpg.configure_item("tsne_activity_type", items=["No activities"], enabled=False, no_arrow_button=True)
            dpg.set_value("tsne_activity_type", "No activities")
            dpg.hide_item("tsne_draw_plot_button")

    with dpg.child_window(parent=plot_manager_tag, width=-1, auto_resize_y=True, no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=False, border=False):
        with dpg.group(horizontal=True):
            subsets = _subset_items(state)
            default_subset = subsets[0] if subsets else "Dataset"
            activities = _activity_items(state, default_subset)
            default_activity = activities[1] if len(activities) > 1 and activities[0] == "No activities" else activities[0]

            with dpg.group():
                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=True, tag="tsne_subset_choice_group"):
                        dpg.add_text("Subset:")
                        dpg.add_combo(width=state["plots_manager_combo_width"], height_mode=dpg.mvComboHeight_Large, items=subsets, default_value=default_subset, tag="tsne_subset_choice", callback=update_tsne_options, user_data=state)
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="tsne_activity_type_group"):
                        dpg.add_text("Activity:")
                        dpg.add_combo(width=state["plots_manager_combo_width"], height_mode=dpg.mvComboHeight_Large, items=activities, default_value=default_activity, tag="tsne_activity_type", enabled=bool(activities))
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="tsne_dimension_combo_group"):
                        dpg.add_text("2D/3D:")
                        dpg.add_combo(tag="tsne_dimension_combo", height_mode=dpg.mvComboHeight_Large, width=state["plots_manager_combo_width"], items=["2D", "3D"], default_value="2D")
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="tsne_fingerprint_algorithm_combo_group"):
                        dpg.add_text("Fingerprint:")
                        dpg.add_combo(tag="tsne_fingerprint_algorithm_combo", height_mode=dpg.mvComboHeight_Large, width=state["plots_manager_combo_width"], items=["Morgan Fingerprint", "RDKit Fingerprint", "Atom Pair Fingerprint", "MACCS Keys", "Topological Torsion Fingerprint", "Pattern Fingerprint", "Layered Fingerprint"], default_value="Morgan Fingerprint")
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="tsne_include_undefined_choice_group"):
                        dpg.add_checkbox(label="Include undefined", tag="tsne_include_undefined_choice", default_value=False)
                        with dpg.tooltip("tsne_include_undefined_choice"):
                            dpg.add_text("Include molecules with undefined activity values (<, <=, >=, >).\nUndefined values are converted to exact values for the projection.")

                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=True, tag="tsne_perplexity_combo_group"):
                        dpg.add_text("Perplexity:")
                        dpg.add_combo(tag="tsne_perplexity_combo", height_mode=dpg.mvComboHeight_Large, width=110, items=["5", "10", "20", "30", "40"], default_value="40")
                        with dpg.tooltip("tsne_perplexity_combo"):
                            dpg.add_text("Approximate effective neighborhood size used by t-SNE.\nSmaller values emphasize fine local structure; larger values smooth the embedding.")
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="tsne_learning_rate_combo_group"):
                        dpg.add_text("Learning rate:")
                        dpg.add_combo(tag="tsne_learning_rate_combo", height_mode=dpg.mvComboHeight_Large, width=120, items=["auto", "50", "100", "200", "500"], default_value="auto")
                        with dpg.tooltip("tsne_learning_rate_combo"):
                            dpg.add_text("Optimization step size for the t-SNE embedding.\nThe default 'auto' is usually the safest starting point.")
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="tsne_iterations_combo_group"):
                        dpg.add_text("Iterations:")
                        dpg.add_combo(tag="tsne_iterations_combo", height_mode=dpg.mvComboHeight_Large, width=110, items=["1000", "2000", "4000"], default_value="4000")
                        with dpg.tooltip("tsne_iterations_combo"):
                            dpg.add_text("Number of optimization steps.\nMore iterations can improve convergence, but also make the plot slower to compute.")
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="tsne_metric_combo_group"):
                        dpg.add_text("Metric:")
                        dpg.add_combo(tag="tsne_metric_combo", height_mode=dpg.mvComboHeight_Large, width=140, items=["cosine", "euclidean", "manhattan"], default_value="cosine")
                        with dpg.tooltip("tsne_metric_combo"):
                            dpg.add_text("Distance metric used before t-SNE compresses the molecular fingerprint space into 2D.")

            dpg.add_spacer(width=max(12, state["win_spacer"] * 2))
            with dpg.group():
                dpg.add_button(label="Draw Plot", tag="tsne_draw_plot_button", callback=lambda: try_draw_tsne(state), show=bool(activities))


def try_draw_tsne(state: dict[str, Any]) -> None:
    if dpg.get_value("tsne_dimension_combo") == "2D":
        for tag in ["tsne_window", "tsne_details_window", "tsne_gradient_bar_window", "tsne_plot_handler_registry"]:
            if dpg.does_item_exist(tag):
                try:
                    dpg.delete_item(tag, children_only=True)
                except Exception:
                    dpg.delete_item(tag)
    try:
        subset = dpg.get_value("tsne_subset_choice")
        activity = dpg.get_value("tsne_activity_type")
        _draw_plot_with_loading(state, subset, activity, perform_tsne)
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        with dpg.window(label="t-SNE Error", tag="tsne_error_window", modal=False, no_resize=True, no_collapse=True, autosize=True):
            dpg.add_text(f"An error occurred during the t-SNE plot generation:\n\n{e}")
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("tsne_error_window"))
