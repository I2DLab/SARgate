"""
=================
stereo_table.py
=================

Stereo analysis output rendering.
"""

import io
import time
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image as pilImage
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from app.gui.loading_win import set_loading_screen_progress
from app.gui.themes_manager import apply_input_text_theme
from app.analysis.overview.overview_decomposition import (
    autoscroll_to_button,
    show_results_window,
    update_activities_windows,
    update_molecule_choice_status,
    update_properties_windows,
    update_r_groups_choice_status,
)
from app.utils.callbacks import (
    change_overview_subtab,
    change_tab,
    export_png_popup,
    on_button_click,
    register_responsive_image,
)


def show_mol_in_overview_tab(sender: Any, app_data: Any, user_data: Any) -> None:
    """
    Callback to display a molecule in the Overview tab when an image button is clicked.
    """
    if dpg.get_value("isomers_subset_choice") != "Dataset":
        subset, mol, state = user_data
        dpg.set_value("tab_bar", "overview_tab")
        change_tab(state)
        dpg.set_value("overview_tab_bar", "overview_decomposition_subtab")
        change_overview_subtab(state)

        update_molecule_choice_status(f"subset_{subset}", state)
        on_button_click(f"subset_{subset}", state)
        update_r_groups_choice_status(f"subset_{subset}_mol_{mol}", state)
        update_properties_windows(f"subset_{subset}_mol_{mol}", state)
        update_activities_windows(f"subset_{subset}_mol_{mol}", state)
        on_button_click(f"subset_{subset}_mol_{mol}", state)
        show_results_window(f"subset_{subset}_mol_{mol}", state)

        autoscroll_to_button(sender="scaff_img", app_data=None, user_data=f"subset_{subset}")
        time.sleep(0.1)
        autoscroll_to_button(sender="mol_img", app_data=None, user_data=f"subset_{subset}_mol_{mol}")


def update_isomer_images(sender: Any, app_data: Any, user_data: Any) -> None:
    """
    Render and update the stereoisomer images grid according to the selected group.
    """
    state = user_data
    set_loading_screen_progress(state, 99)

    ID_column = state["isomers_id_col"]
    combo_tag = "stereo_groups_choice"
    table_tag = "stereo_images_table"
    expanded_key = "expanded_stereo"
    texture_prefix = "stereo_texture"

    selected_label = dpg.get_value(combo_tag)
    if not selected_label or "Mol" not in selected_label:
        return

    mol_id_str = selected_label.split(":")[0].replace("Mol", "").strip()
    mol_id = int(mol_id_str)
    current_dict = state[expanded_key]
    mol_ids = current_dict[mol_id]

    children = dpg.get_item_children(table_tag, 1)
    if children:
        for row in children:
            dpg.delete_item(row)
    set_loading_screen_progress(state, 99.2)

    df_all = state["isomers_df_all_mols"]
    img_width = int(state["isomers_table_img_width"])
    img_height = int(state["isomers_table_img_height"])
    render_scale = 1.8
    render_width = int(round(img_width * render_scale))
    render_height = int(round(img_height * render_scale))

    for row_start in range(0, len(mol_ids), 4):
        default_h = img_height + state["win_spacer"] * 10
        state["default_stereo_table_row_height"] = default_h
        with dpg.table_row(parent=table_tag, tag=f"stereo_table_row_{row_start//4}", height=default_h):
            for k in range(4):
                if row_start + k >= len(mol_ids):
                    dpg.add_spacer(width=img_width, height=10)
                    continue

                mid = mol_ids[row_start + k]
                smiles = df_all.loc[df_all[ID_column] == mid, "Mol"].values[0]
                mol_name = df_all.loc[df_all[ID_column] == mid, "MolName"].values[0]
                subset = df_all.loc[df_all[ID_column] == mid, "Subset"].values[0]
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    dpg.add_spacer(width=img_width, height=10)
                    continue

                drawer = rdMolDraw2D.MolDraw2DCairo(render_width, render_height)
                opts = drawer.drawOptions()
                opts.clearBackground = True
                opts.padding = 0.03
                opts.bondLineWidth = 1
                opts.minFontSize = 1
                opts.explicitMethyl = False
                opts.addStereoAnnotation = True
                opts.minFontSize = 10
                opts.maxFontSize = 14
                opts.annotationFontScale = 1.25

                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                drawer.FinishDrawing()

                img = pilImage.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
                tex_tag = f"{texture_prefix}_{mid}"
                img_array = np.array(img.convert("RGBA")).astype(np.float32) / 255.0

                if not dpg.does_item_exist(tex_tag):
                    dpg.add_dynamic_texture(img.width, img.height, img_array, tag=tex_tag, parent="texture_registry")
                else:
                    dpg.set_value(tex_tag, img_array)

                act = state["isomers_activities"].get(mid, "N/A")
                subset_str = f"Subset {subset} - " if dpg.get_value("isomers_subset_choice") == "Dataset" else ""
                name_str = f"{subset_str}Mol {mid} - {mol_name}"
                with dpg.group():
                    dpg.add_text(default_value=name_str)
                    dpg.add_text(default_value=f"{act}")
                    with dpg.child_window(tag=f"{tex_tag}_cell", auto_resize_y=True,
                                          no_scrollbar=True, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
                        dpg.add_image_button(
                            tex_tag,
                            width=img_width - (state["win_spacer"] * 2) - 4,
                            height=img_height - (state["win_spacer"] * 2) - 4,
                            tag=f"{tex_tag}_tag",
                            background_color=(0, 0, 0, 255),
                            callback=show_mol_in_overview_tab,
                            user_data=(subset, mid, state)
                        )
                        register_responsive_image(
                            state,
                            image_tag=f"{tex_tag}_tag",
                            parent_tag=f"{tex_tag}_cell",
                            aspect_ratio=0.75,
                            tab="stereo_tab",
                        )
                        export_png_popup(f"{tex_tag}_tag", tex_tag, state)
        total_ids = max(1, len(mol_ids))
        processed = min(row_start + 4, len(mol_ids))
        set_loading_screen_progress(state, 99.2 + ((processed / total_ids) * 0.8))


def bind_stereo_table_theme() -> None:
    """
    Apply the standard input-text theme to the stereo images table if present.
    """
    if dpg.does_item_exist("stereo_images_table"):
        dpg.bind_item_theme("stereo_images_table", apply_input_text_theme())
