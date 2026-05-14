"""
==============
sal_manager.py
==============

Manager for the Structure-Activity Landscape plot.
"""

from typing import Any

import dearpygui.dearpygui as dpg

from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress
from app.utils.app_logger import log_event
from app.analysis.similarity.sal_logic import run_landscape_analysis


def show_sal_window(state: dict[str, Any]) -> None:
    """
    Build the manager controls for the SAL plot.
    """
    plot_manager_tag = "landscape_manager_window"
    control_gap = max(6, state["win_spacer"] * 2)

    def update_sal_options(sender: Any, app_data: Any, user_data: Any) -> None:
        subset = app_data
        activities = user_data["bioact_types_dict"][subset]["bioactivities"]
        activities = [a for a in activities if a != "No activities"]
        if activities:
            dpg.configure_item("landscape_activity_type", items=activities, enabled=True, no_arrow_button=False)
            dpg.set_value("landscape_activity_type", activities[0])
            dpg.show_item("landscape_draw_plot_button")
            update_sal_delta_thresh(None, activities[0], user_data)
        else:
            dpg.configure_item("landscape_activity_type", items=["No activities"], enabled=False, no_arrow_button=True)
            dpg.set_value("landscape_activity_type", "No activities")
            dpg.hide_item("landscape_draw_plot_button")

    def update_sal_delta_thresh(sender: Any, app_data: Any, user_data: Any) -> None:
        selected_activity = app_data
        tag = "landscape_delta_thresh"
        if selected_activity not in user_data["nM_activity_types"] and selected_activity not in user_data["dimensionless"]:
            dpg.set_item_label(tag, "Min ΔValue")
            dpg.configure_item(tag, max_value=100, format="%.0f")
            dpg.set_value(tag, 0)
        else:
            dpg.set_item_label(tag, "Min ΔpValue")
            dpg.configure_item(tag, max_value=15.0, format="%.1f")
            dpg.set_value(tag, 0.0)

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
                with dpg.group(horizontal=True):
                    subsets = [s for s in state["smiles_rgd_dict"].keys() if s != "Dataset"]
                    default_subset = subsets[0] if subsets else "subset_1"
                    activities = [a for a in state["bioact_types_dict"].get(default_subset, {}).get("bioactivities", []) if a != "No activities"]

                    with dpg.group(horizontal=True, tag="landscape_subset_choice_group"):
                        dpg.add_text("Subset:")
                        dpg.add_combo(
                            width=state["plots_manager_combo_width"],
                            items=subsets,
                            height_mode=dpg.mvComboHeight_Large,
                            default_value=default_subset,
                            tag="landscape_subset_choice",
                            callback=update_sal_options,
                            user_data=state,
                        )
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="landscape_activity_type_group"):
                        dpg.add_text("Activity:")
                        dpg.add_combo(
                            width=state["plots_manager_combo_width"],
                            height_mode=dpg.mvComboHeight_Largest,
                            items=activities if activities else ["No activities"],
                            default_value=activities[0] if activities else "No activities",
                            tag="landscape_activity_type",
                            enabled=bool(activities),
                            callback=update_sal_delta_thresh,
                            user_data=state,
                        )
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="landscape_fingerprint_algorithm_combo_group"):
                        dpg.add_text("Fingerprint:")
                        dpg.add_combo(
                            tag="landscape_fingerprint_algorithm_combo",
                            height_mode=dpg.mvComboHeight_Largest,
                            width=state["plots_manager_combo_width"],
                            items=[
                                "Morgan Fingerprint", "RDKit Fingerprint", "Atom Pair Fingerprint",
                                "MACCS Keys", "Topological Torsion Fingerprint",
                                "Pattern Fingerprint", "Layered Fingerprint"
                            ],
                            default_value="Morgan Fingerprint",
                        )
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="landscape_include_undefined_choice_group"):
                        dpg.add_checkbox(label="Include undefined", tag="landscape_include_undefined_choice", default_value=False)
                        with dpg.tooltip("landscape_include_undefined_choice"):
                            dpg.add_text("Include molecules with undefined activity values (<, ≤, ≥, >).\nUndefined values will be treated as exact ones (=)")
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="landscape_include_inactives_choice_group"):
                        dpg.add_checkbox(label="Include NO activity", tag="landscape_include_inactives_choice", default_value=False)
                        with dpg.tooltip("landscape_include_inactives_choice"):
                            dpg.add_text("Include molecules lacking activity values.\nTo those molecules, the activity value will be set to 0.\n")

                with dpg.group(horizontal=True, tag="landscape_second_row_group"):
                    with dpg.group(horizontal=True, tag="landscape_delta_thresh_group"):
                        dpg.add_text("Min ΔpValue:")
                        dpg.add_slider_float(
                            width=state["landscape_manager_combo_width"] * 1.5,
                            tag="landscape_delta_thresh",
                            min_value=0.0,
                            max_value=15.0,
                            default_value=0.0,
                            format="%.1f",
                        )
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="landscape_similarity_thresh_group"):
                        dpg.add_text("Min Similarity:")
                        dpg.add_slider_float(
                            width=state["landscape_manager_combo_width"] * 1.5,
                            tag="landscape_similarity_thresh",
                            min_value=0.00,
                            max_value=1.00,
                            default_value=0.00,
                            format="%.2f",
                        )
                        dpg.add_spacer(width=control_gap)

                    with dpg.group(horizontal=True, tag="landscape_sali_index_thresh_group"):
                        dpg.add_text("Min SALI:")
                        dpg.add_colormap_slider(
                            width=state["landscape_manager_combo_width"] * 1.5,
                            tag="landscape_sali_index_thresh",
                            default_value=0,
                        )
                        dpg.bind_colormap("landscape_sali_index_thresh", state["colormaps"][state["colormap_continuous"]])
                        dpg.add_spacer(width=control_gap)

            dpg.add_button(
                label="Draw Plot",
                tag="landscape_draw_plot_button",
                callback=lambda: try_draw_sal(state),
                show=bool(activities),
            )


def try_draw_sal(state: dict[str, Any]) -> None:
    """
    Run the SAL drawing with cleanup and error handling.
    """
    for tag in [
        "landscape_plot_handlers",
        "landscape_click_handler",
        "landscape_mol1_image_widget",
        "landscape_mol2_image_widget",
    ]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

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

    for item in dpg.get_all_items():
        alias = dpg.get_item_alias(item)
        if isinstance(alias, str) and alias.startswith("landscape_hover_handler"):
            dpg.delete_item(item)

    try:
        draw_sal(state)
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        with dpg.window(label="Plot Error", tag="landscape_error_window", modal=False, no_resize=True, no_collapse=True, autosize=True):
            dpg.add_text(f"An error occurred during the SAL plot generation:\n\n{e}")
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("landscape_error_window"))


def draw_sal(state: dict[str, Any]) -> None:
    """
    Draw the SAL plot from current UI selections.
    """
    log_event("Similarity", "Drawing SAL plot", indent=1)
    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)
    run_landscape_analysis(state)
    set_loading_screen_progress(state, 99)
    dpg.fit_axis_data("landscape_x_axis")
    dpg.fit_axis_data("landscape_y_axis")
    set_loading_screen_progress(state, 100)
    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")
