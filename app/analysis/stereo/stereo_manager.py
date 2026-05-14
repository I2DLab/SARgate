"""
=================
stereo_manager.py
=================

Stereo manager GUI.
"""

from typing import Any

import dearpygui.dearpygui as dpg

from app.analysis.stereo.stereo_logic import build_combo_from_groups, run_isomers_analysis
from app.analysis.stereo.stereo_table import update_isomer_images, bind_stereo_table_theme


def show_isomers_window(state: dict[str, Any]) -> None:
    """
    Create and display the Isomers Manager window with subset selection and a button to run the analysis.
    """
    state["stereo_update_group_combos"] = update_isomers_group_combos

    with dpg.child_window(label="Isomers manager window", parent="isomers_manager_window", auto_resize_y=True,
                          no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
        with dpg.group(horizontal=True):
            subsets = ["Dataset"] + list(state["smiles_rgd_dict"].keys())
            dpg.add_combo(label="", tag="isomers_subset_choice",
                          width=state["isomers_manager_combo_width"],
                          height_mode=dpg.mvComboHeight_Large, items=subsets,
                          default_value=subsets[0])

            dpg.add_spacer(width=state["win_spacer"] * 6)

            dpg.add_text("Order:")
            dpg.add_combo(
                label="",
                tag="stereisomers_sorting_mode_combo",
                width=state["isomers_manager_combo_width"],
                height_mode=dpg.mvComboHeight_Large,
                items=["Numeric order", "ΔActivity", "Activity fold-change"],
                default_value="Numeric order",
                callback=lambda: state["stereo_update_group_combos"](state),
            )

            dpg.add_spacer(width=state["win_spacer"] * 6)

            dpg.add_combo(
                label="",
                tag="stereo_groups_choice",
                width=state["isomers_manager_combo_width"] * 4,
                height_mode=dpg.mvComboHeight_Large,
                items=[],
                default_value=None,
                callback=update_isomer_images,
                user_data=state,
            )

            dpg.add_spacer(width=state["win_spacer"] * 6)

            dpg.add_button(label="Show Isomers", tag="show_isomers_button",
                           callback=lambda: try_run_isomers_analysis(state))

def try_run_isomers_analysis(state: dict[str, Any]) -> None:
    """
    Wrapper to run the isomers analysis with error handling.
    """
    try:
        run_isomers_analysis(state)
        bind_stereo_table_theme()
    except Exception as e:
        for tag in ["stereo_sets_group_list", "isomers_groups_selector_window", "isomers_images_main_window"]:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag, children_only=True)

        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")

        with dpg.window(label="Isomers Analysis Error", tag="isomers_error_window", modal=True, no_resize=True,
                        no_collapse=True, autosize=True, on_close=dpg.delete_item("isomers_error_window")):
            dpg.add_text(f"An error occurred during the isomers analysis:\n\n{e}")
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("isomers_error_window"))


def update_isomers_group_combos(state: dict[str, Any]) -> None:
    """
    Update the combo box for stereoisomer group selection.
    """
    sort_mode = dpg.get_value("stereisomers_sorting_mode_combo")
    sort_mode = sort_mode or "Numeric order"
    stereo_dict = state["expanded_stereo"]
    combo_stereo_groups = build_combo_from_groups(
        stereo_dict,
        delta_dict=state.get("delta_stereo"),
        gold_dict=state.get("gold_stereo"),
        sort_mode=sort_mode,
    )
    dpg.configure_item(
        "stereo_groups_choice",
        items=combo_stereo_groups,
        default_value=combo_stereo_groups[0] if combo_stereo_groups else None
    )
