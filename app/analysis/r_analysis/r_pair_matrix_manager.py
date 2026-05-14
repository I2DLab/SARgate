"""
=========================
r_pair_matrix_manager.py
=========================

Manager for the R-Pair Matrix plot.
"""

import os
from typing import Any

import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)

from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress
from app.analysis.r_analysis.r_pair_matrix import draw_rgroups_table


def show_r_pair_matrix_window(state: dict[str, Any]) -> None:
    """
    Build the manager controls for the R-Pair Matrix plot.
    """
    plot_manager_tag = "heatmap_manager_window"
    control_gap = max(6, state["win_spacer"] * 2)

    def update_r_pair_matrix_options(sender: Any, app_data: Any, user_data: Any) -> None:
        subset = app_data
        bioact_types_dict = user_data["bioact_types_dict"]
        activities = bioact_types_dict[subset]["bioactivities"]
        activities = [a for a in activities if a != "No activities"]

        if activities:
            dpg.configure_item("heatmap_activity_type", items=activities, enabled=True, no_arrow_button=False)
            dpg.set_value("heatmap_activity_type", activities[0])
        else:
            dpg.configure_item("heatmap_activity_type", items=["No activities"], enabled=False, no_arrow_button=True)
            dpg.set_value("heatmap_activity_type", "No activities")

        r_groups = user_data["total_r_groups_dict"][subset]
        dpg.configure_item("heatmap_rgroup_choice", items=r_groups)
        dpg.set_value("heatmap_rgroup_choice", r_groups[0])
        dpg.configure_item("heatmap_rgroup_2_choice", items=r_groups)
        dpg.set_value("heatmap_rgroup_2_choice", r_groups[1] if len(r_groups) > 1 else r_groups[0])

        dpg.show_item("heatmap_draw_plot_button")

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
            with dpg.group(horizontal=True):
                subsets = [s for s in state["smiles_rgd_dict"].keys() if s != "Dataset"]
                default_subset = subsets[0] if subsets else "subset_1"
                activities = state["bioact_types_dict"].get(default_subset, {}).get("bioactivities", [])
                activities = [a for a in activities if a != "No activities"]
                r_groups = state["total_r_groups_dict"].get(default_subset, [])

                with dpg.group(horizontal=True, tag="heatmap_subset_choice_group"):
                    dpg.add_text("Subset:")
                    dpg.add_combo(
                        width=state["plots_manager_combo_width"],
                        items=subsets,
                        height_mode=dpg.mvComboHeight_Large,
                        default_value=default_subset,
                        tag="heatmap_subset_choice",
                        callback=update_r_pair_matrix_options,
                        user_data=state,
                    )
                    dpg.add_spacer(width=control_gap)

                with dpg.group(horizontal=True, tag="heatmap_activity_type_group"):
                    dpg.add_text("Activity:")
                    dpg.add_combo(
                        width=state["plots_manager_combo_width"],
                        height_mode=dpg.mvComboHeight_Largest,
                        items=activities if activities else ["No activities"],
                        default_value=activities[0] if activities else "No activities",
                        tag="heatmap_activity_type",
                        enabled=bool(activities),
                    )
                    dpg.add_spacer(width=control_gap)

                with dpg.group(horizontal=True, tag="heatmap_activity_mode_group"):
                    dpg.add_text("Cell color:")
                    dpg.add_combo(
                        tag="heatmap_activity_mode_choice",
                        width=state["plots_manager_combo_width"],
                        height_mode=dpg.mvComboHeight_Largest,
                        items=["Best activity", "Mean activity"],
                        default_value="Best activity",
                    )
                    dpg.add_spacer(width=control_gap)

                with dpg.group(horizontal=True, tag="heatmap_rgroup_choice_group"):
                    dpg.add_text("R-Group:")
                    dpg.add_combo(
                        tag="heatmap_rgroup_choice",
                        width=state["plots_manager_combo_width"],
                        height_mode=dpg.mvComboHeight_Largest,
                        items=r_groups,
                        default_value=r_groups[0] if r_groups else "",
                    )
                    dpg.add_spacer(width=control_gap)

                with dpg.group(horizontal=True, tag="heatmap_rgroup_2_choice_group"):
                    dpg.add_text("R-Group 2:")
                    dpg.add_combo(
                        tag="heatmap_rgroup_2_choice",
                        width=state["plots_manager_combo_width"],
                        height_mode=dpg.mvComboHeight_Largest,
                        items=r_groups,
                        default_value=r_groups[1] if len(r_groups) > 1 else (r_groups[0] if r_groups else ""),
                    )
                    dpg.add_spacer(width=control_gap)

            dpg.add_button(
                label="Draw Plot",
                tag="heatmap_draw_plot_button",
                callback=lambda: try_draw_r_pair_matrix(state),
                show=bool(activities),
            )


def try_draw_r_pair_matrix(state: dict[str, Any]) -> None:
    """
    Run the R-Pair Matrix drawing with cleanup and error handling.
    """
    plot_type = "heatmap"
    for tag in [
        "heatmap_window",
        "heatmap_details_window",
        "heatmap_scaffold_image_widget",
        "heatmap_group_r1_image_widget",
        "heatmap_group_r2_image_widget",
        "heatmap_tooltip_text",
        "heatmap_tooltip_window",
        "heatmap_click_handler",
        "heatmap_mouse_move_handler",
    ]:
        if dpg.does_item_exist(tag):
            try:
                dpg.delete_item(tag, children_only=True)
            except Exception:
                dpg.delete_item(tag)

    try:
        draw_r_pair_matrix(state)
    except Exception as e:
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        with dpg.window(
            label="Plot Error",
            tag="heatmap_error_window",
            modal=False,
            no_resize=True,
            no_collapse=True,
            autosize=True,
        ):
            dpg.add_text(f"An error occurred during the R-Pair Matrix generation:\n\n{e}")
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("heatmap_error_window"))


def draw_r_pair_matrix(state: dict[str, Any]) -> None:
    """
    Draw the R-Pair Matrix plot from current UI selections.
    """
    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)
    subset = dpg.get_value("heatmap_subset_choice")
    activity = dpg.get_value("heatmap_activity_type")
    activity_mode = dpg.get_value("heatmap_activity_mode_choice") if dpg.does_item_exist("heatmap_activity_mode_choice") else "Best activity"
    r1 = dpg.get_value("heatmap_rgroup_choice")
    r2 = dpg.get_value("heatmap_rgroup_2_choice")
    read_undefined = dpg.get_value("heatmap_include_undefined_choice") if dpg.does_item_exist("heatmap_include_undefined_choice") else False

    csv_file = os.path.join(state["summary_dir"], f"{subset}_summary.csv")
    set_loading_screen_progress(state, 3)
    data = pd.read_csv(csv_file)
    set_loading_screen_progress(state, 5)
    draw_rgroups_table(subset, activity, r1, r2, data, read_undefined, state, activity_mode=str(activity_mode))

    if state["heatmap_ncols"] > state["heatmap_nrows"]:
        dpg.fit_axis_data("heat_y_axis")
        dpg.fit_axis_data("heat_x_axis")
    else:
        dpg.fit_axis_data("heat_x_axis")
        dpg.fit_axis_data("heat_y_axis")

    set_loading_screen_progress(state, 100)
    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")
