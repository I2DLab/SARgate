"""
=========
counts.py
=========

R-group frequency count and analysis.

Computes and displays the occurrence of each R-group within all molecular subsets.
Provides exportable tables and visual plots to evaluate substituent prevalence
and their contribution to biological activity.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Show counts selection window
# 3. Change counts sort order
# 4. Show counts table window
# 5. Extract features

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import io
import math
import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
from datetime import datetime
from collections import Counter
from typing import Any
from PIL import Image as pilImage, ImageDraw, ImageFont
from app.utils.app_logger import log_event, log_settings
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from rdkit import Chem, RDConfig
from rdkit.Chem import Draw, rdDepictor, ChemicalFeatures
from rdkit.Chem.Draw import rdMolDraw2D
from app.utils.callbacks import (
    export_png_popup,
    register_responsive_image,
    update_responsive_images
)
from app.gui.themes_manager import (
    apply_image_button_theme,
)
from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress
from app.analysis.r_analysis.counts_boxplot import draw_counts_boxplot
from app.utils.native_dialogs import save_file_dialog


# -----------------------------------------------------------------------------
# 2. Show counts selection window
# -----------------------------------------------------------------------------
def show_counts_selection_window(state: dict[str, Any]) -> None:

    # Default sorting mode
    state.setdefault("counts_sort_mode", "frequency")

    # Small helper that refreshes the R-group combo based on the chosen subset.
    # -----------------------------------------------------------------------------
    # 2.1. Update rgroup options
    # -----------------------------------------------------------------------------
    def update_rgroup_options(sender: Any, app_data: Any, user_data: Any) -> None:

        # Pull the selected subset and resolve its available R-groups
        selected_subset = app_data
        rgroup_dict = user_data["total_r_groups_dict"]
        r_groups = rgroup_dict[selected_subset]

        # Enable and seed the R-group choice with the correct items and default
        dpg.configure_item("counts_rgroup_choice", items=r_groups)
        dpg.set_value("counts_rgroup_choice", r_groups[0])

    # Build the container that hosts subset/R-group selectors and the action button.
    with dpg.child_window(parent="counts_selection_window", auto_resize_y=True,
                          no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):

        with dpg.group(horizontal=True):
            # Subset combo box: populate from state and pick a sensible default
            subsets = list(state["total_r_groups_dict"].keys())
            default_subset = subsets[0] if subsets else "No subset"
            dpg.add_text("Subset:")
            dpg.add_combo(width=state["counts_selection_combo_width"], 
                            height_mode=dpg.mvComboHeight_Large, items=subsets,
                            default_value=default_subset, tag="counts_subset_choice",
                            callback=update_rgroup_options, user_data=state)

            dpg.add_spacer(width=state["win_spacer"] * 4)

            # Initial R-group combo box: seeded from the default subset
            initial_rgroups = state["total_r_groups_dict"].get(default_subset, [])
            dpg.add_text("R-group:")
            dpg.add_combo(width=state["counts_selection_combo_width"],
                            height_mode=dpg.mvComboHeight_Largest, items=initial_rgroups,
                            default_value=initial_rgroups[0], tag="counts_rgroup_choice")

            dpg.add_spacer(width=state["win_spacer"] * 4)

            dpg.add_button(label="Show Counts", tag="show_counts_button",  
                            callback=lambda: show_counts_table_window(state))

            dpg.add_spacer(width=state["win_spacer"] * 4)

            dpg.add_button(label="Save PDF Report", tag="save_counts_pdf_button",
                        callback=lambda: None)  # Placeholder; actual callback set later
            
            

# -----------------------------------------------------------------------------
# 3. Change counts sort order
# -----------------------------------------------------------------------------
def change_counts_sort_order(state: dict[str, Any]) -> None:
    mode = state.get("counts_sort_mode", "frequency")
    state["counts_sort_mode"] = "activity" if mode == "frequency" else "frequency"
    # Rebuild the window with the new ordering
    show_counts_table_window(state)
    update_responsive_images(state)


# -----------------------------------------------------------------------------
# 4. Show counts table window
# -----------------------------------------------------------------------------
def show_counts_table_window(state: dict[str, Any]) -> Any:
    log_event("R-Analysis", "Drawing counts table", indent=1)
    log_settings("R-Analysis", indent=2, subset=dpg.get_value("counts_subset_choice"), rgroup=dpg.get_value("counts_rgroup_choice"), sort_mode=state.get("counts_sort_mode", "frequency"))
    dpg.configure_item("counts_table", show=True)

    # Lightweight overlay to indicate processing while UI is being built
    draw_loading_screen(state, bg=False)
    set_loading_screen_progress(state, 1)

    # Parse the CSV, filter invalid entries and build occurrence counts.
    # -----------------------------------------------------------------------------
    # 4.1. Read rgroup data
    # -----------------------------------------------------------------------------
    def read_rgroup_data(csv_path: str, r_group: Any) -> Any:
        """
        Reads the R-group column from CSV, removes invalid entries, and counts occurrences.

        Args:
            csv_path (str): Path to the CSV summary file.
            r_group (str): Column name for the R-group.

        Returns:
            tuple: (List of (SMILES, count), total number of valid molecules)
        """
        df = pd.read_csv(csv_path)
        valid_smiles = df[r_group].dropna()
        valid_smiles = valid_smiles[valid_smiles != "Failed"]
        smiles_list = valid_smiles.tolist()
        total_molecules = len(smiles_list)
        counter = Counter(smiles_list)
        sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        return sorted_counts, total_molecules

    # Convert each SMILES to an RDKit Mol, depict, and rasterise to PIL.
    # -----------------------------------------------------------------------------
    # 4.2. Generate rgroup images
    # -----------------------------------------------------------------------------
    def generate_rgroup_images(smiles_counts: str) -> Any:
        """
        Converts R-group SMILES into PIL images.

        Args:
            smiles_counts (list): List of tuples (SMILES, count).

        Returns:
            list: List of tuples (SMILES, count, PIL Image).
        """

        rgroup_image_list = []
        for smiles, count in smiles_counts:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            if mol is None:
                log_event("R-Analysis", f"Error in SMILES: {smiles}", indent=1, level="WARNING")
                continue
            rdDepictor.Compute2DCoords(mol)
            img_pil = Draw.MolToImage(mol, size=(img_size, img_size))
            rgroup_image_list.append((smiles, count, img_pil))
        return rgroup_image_list

    # Prepare float RGBA arrays suitable for DearPyGui dynamic textures.
    # -----------------------------------------------------------------------------
    # 4.3. Pil image to dpg array
    # -----------------------------------------------------------------------------
    def pil_image_to_dpg_array(pil_img: Any) -> Any:
        """
        Converts a PIL image to a DearPyGui-compatible RGBA float array.

        Args:
            pil_img (PIL.Image): PIL image to convert.

        Returns:
            np.ndarray: Float array for dynamic texture.
        """
        
        img_rgba = pil_img.convert("RGBA")
        img_array = np.array(img_rgba).astype(np.float32) / 255.0
        return img_array

    def _load_pil_label_font(size_px: int) -> Any:
        """
        Load a readable PIL font for oversampled labels.

        Args:
            size_px (int): Requested font size in pixels.

        Returns:
            Any: PIL font instance.
        """
        for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(font_name, size_px)
            except Exception:
                continue
        return ImageFont.load_default()

    def _add_label_on_pil(
        img: Any,
        text: str,
        position: str = "top",
        margin: int = 6,
        pad_x: int = 6,
        pad_y: int = 3,
        bg: Any = None,
        fg: Any = (0, 0, 0, 255),
    ) -> Any:
        """
        Draw a centred label at the top or bottom of an RGBA image.

        Args:
            img (PIL.Image): RGBA image to annotate.
            text (str): Label text to render.
            position (str): "top" or "bottom" placement.
            margin (int): Pixel distance from the edge.
            pad_x (int): Horizontal padding for label background.
            pad_y (int): Vertical padding for label background.
            bg (tuple|None): Optional RGBA background colour.
            fg (tuple): Text colour RGBA.

        Returns:
            PIL.Image: Annotated image object (same instance, modified in place).
        """
        if not text:
            return img

        draw = ImageDraw.Draw(img, "RGBA")
        try:
            target_size = max(18, int(round(img.width * 0.065)))
            font = _load_pil_label_font(target_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            while target_size > 12 and (bbox[2] - bbox[0]) > (img.width - 2 * margin):
                target_size -= 1
                font = _load_pil_label_font(target_size)
                bbox = draw.textbbox((0, 0), text, font=font)
        except Exception:
            font = None

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        w, h = img.size
        x = (w - tw) // 2
        y = margin if position == "top" else (h - th - margin)

        if bg:
            draw.rectangle([x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y], fill=bg)

        draw.text((x, y), text, fill=fg, font=font)
        return img


    # Compose a paginated PDF with scaffold, R-group thumbnails, frequencies and SMILES.
    # -----------------------------------------------------------------------------
    # 4.4. Save rgroup report
    # -----------------------------------------------------------------------------
    def save_rgroup_report() -> None:
        """
        Generates and saves a PDF report with scaffold and R-group images, frequencies, SMILES,.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """

        sort_label = "mean activity" if state.get("counts_sort_mode") == "activity" else "frequency"
        default_name = f"Report_R-counts_{subset}_{r_group}_sortedby{sort_label.replace(' ', '')}.pdf"
        pdf_path = save_file_dialog(
            title="Save R-counts PDF report",
            default_path=report_dir,
            default_name=default_name,
            file_types=[("PDF files", "*.pdf")],
        )
        if not pdf_path:
            return
        if not pdf_path.lower().endswith(".pdf"):
            pdf_path = f"{pdf_path}.pdf"

        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        margin = 40
        y_position = height - margin

        # ---- Header ----
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y_position, f"R-Group Counts Report - {subset.replace('subset_', 'Subset ')} - {r_group}")
        y_position -= 15

        project_name = state.get("file_name", "Unnamed Project")
        timestamp = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(margin, y_position, f"Project: {project_name}   |   Saved on: {timestamp}")
        y_position -= 30

        # ---- Scaffold ----
        c.setFont("Helvetica", 10)
        c.drawString(margin, y_position, f"{subset.replace('subset_', 'Subset ')} common substructure:")
        y_position -= 20

        img_buffer = io.BytesIO()
        scaffold_image_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        scaffold_reader = ImageReader(img_buffer)
        scaff_width, scaff_height = scaffold_image_pil.size
        scaff_scale = 120 / scaff_height
        c.drawImage(scaffold_reader, margin, y_position - 120, width=scaff_width * scaff_scale, height=scaff_height * scaff_scale)
        y_position -= 150


        # ---- Activity choice for the PDF (first available) ----
        act_col = dpg.get_value("counts_boxplot_activity_type")
        if act_col == "No activities" or act_col not in csv_df.columns:
            act_col = None       
        act_unit_global = _activity_unit(act_col) if act_col else ""

        c.setFont("Helvetica-Oblique", 10)
        c.drawString(margin, y_position, f"{r_group} groups sorted by {sort_label}:")
        y_position -= 20

        # ---- R-groups listing ----
        for idx, (smi, count, pil_img) in enumerate(rgroup_image_list):
            # New page if needed
            if y_position < margin + 90:
                c.showPage()
                y_position = height - margin
                c.setFont("Helvetica", 10)

            # Image
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_reader = ImageReader(img_buffer)
            img_width, img_height = pil_img.size
            img_scale = 80 / img_height
            c.drawImage(img_reader, margin, y_position - 80, width=img_width * img_scale, height=img_height * img_scale)

            # Text block to the right
            text_x = margin + img_width * img_scale + 10
            c.setFont("Helvetica", 10)
            c.drawString(text_x, y_position - 20, f"SMILES: {smi}")
            c.drawString(text_x, y_position - 40, f"Frequency: {count}/{total_molecules}")

            # Mean activity (if available)
            if act_col:
                metric_val, metric_label, metric_unit, is_p = _mean_metric_for_smi(smi, act_col)
                if metric_val is not None:
                    unit_str = "" if is_p else (f" {metric_unit}" if metric_unit else "")
                    c.drawString(text_x, y_position - 60, f"{metric_label}: {metric_val:.2f}{unit_str}")
                else:
                    c.drawString(text_x, y_position - 60, f"{'Mean p'+act_col if act_col in state.get('nM_activity_types', []) else 'Mean '+act_col}: N/A")
                y_text_block = 70
            else:
                y_text_block = 50

            y_position -= max(90, y_text_block)

        c.save()
        print(f"PDF report saved as {pdf_path}")


    # On image interaction, list matching molecule IDs and their first available activity.
    # -----------------------------------------------------------------------------
    # 4.5. Update rgroup details
    # -----------------------------------------------------------------------------
    def update_rgroup_details(sender: Any, app_data: Any, user_data: Any) -> None:

        hovered_smi, state = user_data

        selected_activity = dpg.get_value("counts_boxplot_activity_type")
        include_inactives = state.get("counts_boxplot_include_inactives", False)
        include_undefined = state.get("counts_boxplot_include_undefined", False)

        # extract matching rows
        matching_rows = csv_df[csv_df[r_group] == hovered_smi]

        # define activity columns after Chi4
        chi4_index = list(csv_df.columns).index("Chi4")
        activity_columns = list(csv_df.columns)[chi4_index + 1:]

        # If no valid activity is selected
        if selected_activity == "No activities" or selected_activity not in activity_columns:
            selected_activity = None

        filtered_rows = []

        for _, row in matching_rows.iterrows():

            # Determine molecule activity status
            value = None
            if selected_activity:
                value = row[selected_activity]

            # inactive if selected_activity exists but row has no value
            is_inactive = (
                (selected_activity and (pd.isna(value) or str(value).strip() == ""))
                or (not selected_activity and all(
                    pd.isna(row[act]) or str(row[act]).strip() == "" for act in activity_columns
                ))
            )

            if is_inactive and not include_inactives:
                continue  # reject this molecule

            raw_val = None if selected_activity is None else str(value).strip()

            is_undefined = (
                raw_val is not None and
                (raw_val.startswith("<") or raw_val.startswith(">"))
            )

            if is_undefined and not include_undefined:
                continue  # reject undefined values

            if selected_activity and not include_inactives:
                # selected activity must exist and not be empty
                if pd.isna(value) or str(value).strip() == "":
                    continue

            filtered_rows.append(row)


        # Build output text
        total = total_molecules
        n_match = len(filtered_rows)

        lines = []
        for row in filtered_rows:
            mol_id = int(row["Mol_sub_ID"]) if "Mol_sub_ID" in row else row.name
            line = f"Mol {mol_id}"

            # find first available activity for display
            for activity in activity_columns:
                val = row[activity]
                if pd.notna(val) and str(val).strip() != "":
                    unit = (
                        "nM" if activity in state["nM_activity_types"] else
                        "%" if activity in state["percent_activities"] else
                        "μg/mL" if activity in state["ug/mL_activities"] else
                        "μM/min" if activity in state["uM/min_activities"] else
                        ""
                    )
                    try:
                        line += f"  »  {activity} = {val:.2f} {unit}"
                    except:
                        line += f"  »  {activity} = {val} {unit}"
                    break

            lines.append(line)

        detail_text = f"{n_match}/{total} molecules:\n\n" + "\n".join(lines)
        dpg.set_value("counts_rgroup_details_text", detail_text)

        update_boxplot_details = state.get("update_counts_boxplot_details_for_group")
        if callable(update_boxplot_details):
            try:
                update_boxplot_details(hovered_smi)
            except Exception:
                pass


    # Pull directories, current subset/R-group and sizing; read counts and images.
    summary_dir = state["summary_dir"]
    molblocks_rgd_dict = state["molblocks_rgd_dict"]
    smiles_rgd_dict = state["smiles_rgd_dict"]
    report_dir = state["report_dir"]
    
    subset = dpg.get_value("counts_subset_choice")
    r_group = dpg.get_value("counts_rgroup_choice")
    csv_path = os.path.join(summary_dir, f"{subset}_summary.csv")
    img_size = state["counts_table_img_size"]
    sort_mode = state.get("counts_sort_mode", "frequency")

    set_loading_screen_progress(state, 4)
    smiles_counts, total_molecules = read_rgroup_data(csv_path, r_group)
    set_loading_screen_progress(state, 10)
    rgroup_image_list = generate_rgroup_images(smiles_counts)
    set_loading_screen_progress(state, 32)

    state["counts_order"] = [smi for (smi, _, _) in rgroup_image_list]
    state["counts_index_map"] = {smi: idx+1 for idx, (smi, _, _) in enumerate(rgroup_image_list)}


    # Build the scaffold image using the same logic as overview_decomposition.py.
    scaffold_mb = molblocks_rgd_dict[subset]["mol_1"]["Core"] if "mol_1" in smiles_rgd_dict[subset] else smiles_rgd_dict[subset]["mol_2"]["Core"]
    try:
        scaffold_mol = Chem.MolFromMolBlock(scaffold_mb, sanitize=False)
        if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
            raise KeyError()
    except:
        try:
            scaffold_smi = smiles_rgd_dict[subset]["mol_1"]["Core"] if "mol_1" in smiles_rgd_dict[subset] else smiles_rgd_dict[subset]["mol_2"]["Core"]
            scaffold_mol = Chem.MolFromSmiles(scaffold_smi, sanitize=False)
            if scaffold_mol is None:
                raise KeyError()
        except:
            scaffold_sma = smiles_rgd_dict[subset]["mol_1"]["Core"] if "mol_1" in smiles_rgd_dict[subset] else smiles_rgd_dict[subset]["mol_2"]["Core"]
            scaffold_mol = Chem.MolFromSmarts(scaffold_sma)
    counts_scaffold_display_width = state["counts_scaff_img_width"]
    counts_scaffold_display_height = state["counts_scaff_img_height"]
    counts_scaffold_render_scale = 1.2
    counts_scaffold_render_width = int(round(counts_scaffold_display_width * counts_scaffold_render_scale))
    counts_scaffold_render_height = int(round(counts_scaffold_display_height * counts_scaffold_render_scale))

    scaffold_image_pil = pilImage.new(
        "RGBA",
        (counts_scaffold_render_width, counts_scaffold_render_height),
        (255, 255, 255, 255),
    )
    try:
        for atom in scaffold_mol.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.HasProp("molAtomMapNumber"):
                idx = atom.GetProp("molAtomMapNumber")
                atom.SetProp("atomLabel", f"R{idx}")
        Chem.SanitizeMol(
            scaffold_mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        Chem.AssignStereochemistry(scaffold_mol, force=True, cleanIt=True)
        drawer = rdMolDraw2D.MolDraw2DCairo(
            counts_scaffold_render_width,
            counts_scaffold_render_height,
        )
        opts = drawer.drawOptions()
        opts.padding = 0.025
        opts.bondLineWidth = 1
        opts.minFontSize = 1
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, scaffold_mol)
        drawer.FinishDrawing()
        scaffold_image_pil = pilImage.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    except Exception:
        try:
            scaffold_image_pil = Draw.MolToImage(
                scaffold_mol,
                size=(counts_scaffold_render_width, counts_scaffold_render_height),
            ).convert("RGBA")
        except Exception:
            log_event(
                "R-Analysis",
                "Warning: could not render scaffold image for counts",
                indent=1,
                level="WARNING",
            )
    set_loading_screen_progress(state, 38)

    # Load the full summary to enable plotting and detail lookups.
    csv_df = pd.read_csv(csv_path)
    set_loading_screen_progress(state, 42)

    # Determine available activity columns (same logic used later for plotting)
    activity_cols = [col for col in csv_df.columns if col in state["activity_types"]]
    if not activity_cols:
        state["counts_sort_mode"] = "frequency"
        act_col = None

    # Helper: mean activity for a given SMILES on a chosen activity column
    # -----------------------------------------------------------------------------
    # 4.6. Mean activity for smi
    # -----------------------------------------------------------------------------
    def _mean_activity_for_smi(smi: Any, activity_col: str) -> Any:
        rows = csv_df[csv_df[r_group] == smi]
        vals = pd.to_numeric(rows[activity_col], errors="coerce").dropna()
        if vals.empty:
            return None
        # If 'nM' activity, convert to p (higher is better); otherwise use mean as-is
        if activity_col in state.get("nM_activity_types", []):
            vals = vals[vals > 0]
            if vals.empty:
                return None
            pvals = 9 - np.log10(vals)
            return float(pvals.mean())
        else:
            return float(vals.mean())

    # -----------------------------------------------------------------------------
    # 4.7. Activity unit
    # -----------------------------------------------------------------------------
    def _activity_unit(activity_col: str) -> Any:
        if activity_col in state.get("nM_activity_types", []):
            return "nM"
        if activity_col in state.get("percent_activities", []):
            return "%"
        if activity_col in state.get("μg/mL_activities", []):
            return "μg/mL"
        if activity_col in state.get("μM/min_activities", []):
            return "μM/min"
        return ""


    # -----------------------------------------------------------------------------
    # 4.8. Mean metric for smi
    # -----------------------------------------------------------------------------
    def _mean_metric_for_smi(smi: Any, activity_col: str) -> Any:
        if activity_col is None:
            return None, None, None, False
        rows = csv_df[csv_df[r_group] == smi]
        vals = pd.to_numeric(rows[activity_col], errors="coerce").dropna()
        if vals.empty:
            return None, None, None, False

        if activity_col in state.get("nM_activity_types", []):
            vals = vals[vals > 0]
            if vals.empty:
                return None, None, None, False
            pvals = 9 - np.log10(vals)
            return float(pvals.mean()), f"Mean p{activity_col}", "", True
        else:
            unit = _activity_unit(activity_col)
            return float(vals.mean()), f"Mean {activity_col}", unit, False
        


    # Reorder by activity metric if requested (fallback to frequency order otherwise)
    act_col = dpg.get_value("counts_boxplot_activity_type")
    if act_col == "No activities" or act_col not in csv_df.columns:
        act_col = None
    metric_map = {}

    if sort_mode == "activity" and act_col is not None:
        for smi, _cnt, _img in rgroup_image_list:
            metric_val, _, _, _ = _mean_metric_for_smi(smi, act_col)
            metric_map[smi] = metric_val

        rgroup_image_list.sort(
            key=lambda t: metric_map.get(t[0]) if metric_map.get(t[0]) is not None else -1e18,
            reverse=True
        )
    else:
        for smi, _cnt, _img in rgroup_image_list:
            metric_val, _, _, _ = _mean_metric_for_smi(smi, act_col) if act_col else (None, None, None, False)
            metric_map[smi] = metric_val

    state["counts_order"] = [smi for (smi, _, _) in rgroup_image_list]
    state["counts_index_map"] = {smi: idx + 1 for idx, (smi, _, _) in enumerate(rgroup_image_list)}
    set_loading_screen_progress(state, 48)


    # Ensure a clean texture registry and register scaffold/R-group textures.
    for tag in ["counts_scaffold_img_window", "counts_plot_window", "counts_rgroup_details_window", "counts_boxplot_details_window",
                "counts_boxplot_manager_window", "counts_table_window"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag, children_only=True)

    # Step 4.1: Add scaffold texture
    scaffold_array = pil_image_to_dpg_array(scaffold_image_pil)
    set_loading_screen_progress(state, 52)

    if not dpg.does_item_exist("scaffold_texture"):
        dpg.add_dynamic_texture(scaffold_image_pil.width, scaffold_image_pil.height, scaffold_array, 
                            tag="scaffold_texture", parent="texture_registry")
    else:
        dpg.set_value("scaffold_texture", scaffold_array)

    # Step 4.2: Add R-group textures
    total_rgroups = max(1, len(rgroup_image_list))
    for idx, (smi, count, pil_img) in enumerate(rgroup_image_list):
        img_array = pil_image_to_dpg_array(pil_img)
        if not dpg.does_item_exist(f"counts_rgroup_texture_{idx}"):
            dpg.add_dynamic_texture(pil_img.width, pil_img.height, img_array, 
                                tag=f"counts_rgroup_texture_{idx}", parent="texture_registry")
        else:
            dpg.set_value(f"counts_rgroup_texture_{idx}", img_array)
        if (idx + 1) == total_rgroups or (idx + 1) % max(1, total_rgroups // 20) == 0:
            set_loading_screen_progress(state, 52 + ((idx + 1) / total_rgroups) * 12)
    # Assemble the upper info pane with the scaffold preview, plot, and details box.
    with dpg.child_window(parent="counts_scaffold_img_window", width=-1, height=-1,
                          no_scrollbar=True, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):

        dpg.add_image(
            "scaffold_texture",
            tag="counts_scaff_img",
            width=counts_scaffold_display_width,
            height=counts_scaffold_display_height,
        )
        with dpg.tooltip("counts_scaff_img", delay=0):
            dpg.add_text(f"{subset.replace('_', ' ').replace('su', 'Su')} - Common Core")
        register_responsive_image(
            state,
            image_tag="counts_scaff_img",
            parent_tag="counts_scaffold_img_window",
            aspect_ratio=0.75,
            parent_of_parent="counts_details_table_row",
            default_pop_h=state["default_counts_details_table_row_height"],
            tab="r_analysis_tab",
        )
        export_png_popup("counts_scaff_img", "scaffold_texture", state)
    set_loading_screen_progress(state, 66)


    # -----------------------------------------------------------------------------
    # 4.9. Prepare counts boxplot
    # -----------------------------------------------------------------------------
    def prepare_counts_boxplot(sender: Any) -> None:
        if sender == "activity_type":
            activity = dpg.get_value("counts_boxplot_activity_type")
            state["counts_last_activity"] = activity

            if activity == "No activities":
                # Hide label and checkbox when no activity exists
                if dpg.does_item_exist("sort_counts_label"):
                    dpg.configure_item("sort_counts_label", show=False)
                if dpg.does_item_exist("sort_counts_checkbox"):
                    dpg.configure_item("sort_counts_checkbox", show=False)
            else:
                # Update label text and show controls
                sort_text = f"Sort by Mean {activity}:"
                if dpg.does_item_exist("sort_counts_label"):
                    dpg.configure_item(
                        "sort_counts_label",
                        default_value=sort_text,
                        show=True,
                    )
                if dpg.does_item_exist("sort_counts_checkbox"):
                    dpg.configure_item("sort_counts_checkbox", show=True)

                # Redraw the table and boxplot if activity sorting is active.
                if state.get("counts_sort_mode") == "activity":
                    show_counts_table_window(state)
                    update_responsive_images(state)

        elif sender == "include_undefined":
            state["counts_boxplot_include_undefined"] = bool(
                dpg.get_value("counts_boxplot_include_undefined_choice")
            )

        elif sender == "include_inactives":
            state["counts_boxplot_include_inactives"] = bool(
                dpg.get_value("counts_boxplot_include_inactives_choice")
            )

        elif sender == "min_count":
            state["counts_boxplot_min_count"] = dpg.get_value("counts_boxplot_min_count_slider")


        for tag in [f"counts_boxplot_window", f"counts_boxplot_details_window"]:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag, children_only=True)

        for tag in ["counts_boxplot_click_handler"]:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
                
        activity = dpg.get_value(f"counts_boxplot_activity_type")
        include_undefined = dpg.get_value(f"counts_boxplot_include_undefined_choice")
        include_inactives = dpg.get_value(f"counts_boxplot_include_inactives_choice")
        min_count = dpg.get_value(f"counts_boxplot_min_count_slider")
        
        draw_counts_boxplot(r_group, activity, include_undefined, include_inactives, min_count, csv_df, total_molecules, state)
        dpg.fit_axis_data("counts_boxplot_x_axis")
        dpg.fit_axis_data("counts_boxplot_y_axis")

    state["counts_boxplot_redraw"] = lambda: prepare_counts_boxplot("plot_context")


    with dpg.child_window(parent="counts_boxplot_manager_window", width=-1, auto_resize_y=True,
                            no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
    
        with dpg.group(horizontal=True, tag="counts_boxplot_settings_group"):
            with dpg.group(horizontal=True, tag=f"counts_boxplot_activity_type_group"):
                bioact_types_dict = state["bioact_types_dict"]
                activities = bioact_types_dict.get(subset, {}).get("bioactivities", [])

                last_act = state.get("counts_last_activity")

                if not activities:
                    last_act = "No activities"
                elif last_act not in activities:
                    last_act = activities[0]

                dpg.add_text("Activity type:")
                dpg.add_combo(
                    width=state["plots_manager_combo_width"],
                    height_mode=dpg.mvComboHeight_Largest,
                    items=activities if activities else ["No activities"],
                    default_value=last_act,
                    tag="counts_boxplot_activity_type",
                    enabled=bool(activities),
                    callback=lambda: prepare_counts_boxplot("activity_type")
                )

                dpg.set_value("counts_boxplot_activity_type", last_act)
                                
            dpg.add_spacer(width=state["win_spacer"] * 4)

            activity_for_sort = dpg.get_value("counts_boxplot_activity_type")

            # Hide the sorting controls when no activity column exists.
            if activity_for_sort == "No activities":
                # Keep hidden items available so theme binding remains stable.
                dpg.add_text("", tag="sort_counts_label", show=False)
                dpg.add_checkbox(tag="sort_counts_checkbox", show=False)
            else:
                sort_label = f"Sort by Mean {activity_for_sort}:"
                dpg.add_text(sort_label, tag="sort_counts_label")

                dpg.add_checkbox(
                    tag="sort_counts_checkbox",
                    default_value=(state.get("counts_sort_mode") == "activity"),
                    callback=lambda: change_counts_sort_order(state)
                )

            dpg.add_spacer(width=state["win_spacer"] * 4)

            with dpg.group(horizontal=True, tag=f"counts_boxplot_include_undefined_choice_group"):
                dpg.add_text("Include undefined:")
                dpg.add_checkbox(tag=f"counts_boxplot_include_undefined_choice", 
                                 default_value=state.get("counts_boxplot_include_undefined", False),
                                 callback=lambda: prepare_counts_boxplot("include_undefined"))
                with dpg.tooltip(f"counts_boxplot_include_undefined_choice_group"):
                    dpg.add_text("Include molecules with undefined activity values (<, ≤, ≥, >).\n"
                                "Undefined values will be treated as exact ones (=)")
            
            dpg.add_spacer(width=state["win_spacer"] * 4)
            
            with dpg.group(horizontal=True, tag=f"counts_boxplot_include_inactives_choice_group"):
                dpg.add_text("Include NO activity:")
                dpg.add_checkbox(tag=f"counts_boxplot_include_inactives_choice", 
                                 default_value=state.get("counts_boxplot_include_inactives", False),
                                 callback=lambda: prepare_counts_boxplot("include_inactives"))
                with dpg.tooltip(f"counts_boxplot_include_inactives_choice_group"):
                    dpg.add_text("Include molecules lacking activity values.\n"
                                "To those molecules, the activity value will be set to 0.\n")

            dpg.add_spacer(width=state["win_spacer"] * 4)

            with dpg.group(horizontal=True, tag=f"counts_boxplot_min_count_group"):
                dpg.add_text("Min. Count Filter:")
                dpg.add_input_int(tag=f"counts_boxplot_min_count_slider", width=150,
                                    min_value=1, max_value=total_molecules, min_clamped=True, max_clamped=True, default_value=2,
                                    callback=lambda s, a, u: prepare_counts_boxplot("min_count"))
                with dpg.tooltip(f"counts_boxplot_min_count_group"):
                    dpg.add_text("Filter boxes in the boxplot by minimum molecules count.\n")
                                
    prepare_counts_boxplot("")
    set_loading_screen_progress(state, 82)

    # A small scroll-free area for listing molecules and a Save PDF Report button.
    with dpg.child_window(parent="counts_rgroup_details_window", width=-1, height=-1, border=False,
                            no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True):
        dpg.add_text(default_value="Click an R-group\nimage to see details", tag="counts_rgroup_details_text")
    
    dpg.configure_item("save_counts_pdf_button", callback=lambda: save_rgroup_report())


    # Page size
    PAGE_SIZE = state.get("counts_page_size", 20)

    # Row height constant used by responsive system
    state["default_counts_table_row_height"] = img_size

    # Prepare list of activity columns ordered by population
    activity_cols = [col for col in csv_df.columns if col in state["activity_types"]]
    if not activity_cols:
        # Spegni sort_by_activity se attivo
        state["counts_sort_mode"] = "frequency"

        # Assicurati che la combo e la checkbox non vengano usate
        act_col = None

    activity_counts = {}
    for act in activity_cols:
        vals = pd.to_numeric(csv_df[act], errors="coerce").dropna()
        activity_counts[act] = len(vals)

    sorted_acts = sorted(activity_cols, key=lambda x: activity_counts[x], reverse=True)

    # Ensure pagination state exists
    state.setdefault("counts_current_page", 0)
    total_items = len(rgroup_image_list)
    total_pages = max(1, math.ceil(total_items / PAGE_SIZE))
    state["counts_total_items"] = total_items
    state["counts_total_pages"] = total_pages


    # Utility: compute min/max/mean stats for all activities for one SMILES
    # -----------------------------------------------------------------------------
    # 4.10. Compute stats for smi
    # -----------------------------------------------------------------------------
    def _compute_stats_for_smi(smi: Any) -> Any:
        matching = csv_df[csv_df[r_group] == smi]
        results_min = []
        results_max = []
        results_mean = []

        for act in sorted_acts:
            vals = pd.to_numeric(matching[act], errors="coerce").dropna()

            if act in state.get("nM_activity_types", []):
                vals = vals[vals > 0]
                if not vals.empty:
                    mn, mx, md = vals.min(), vals.max(), vals.mean()
                else:
                    mn = mx = md = None
                unit = "nM"
            else:
                mn = vals.min() if not vals.empty else None
                mx = vals.max() if not vals.empty else None
                md = vals.mean() if not vals.empty else None
                unit = (
                    "nM" if act in state["nM_activity_types"] else
                    "%" if act in state["percent_activities"] else
                    "μg/mL" if act in state["ug/mL_activities"] else
                    "μM/min" if act in state["uM/min_activities"] else
                    ""
                )

            def fmt(v: Any) -> Any:
                return f"{v:.1f}" if v is not None else "N/A"

            if not vals.empty:
                results_min.append(f"{act}: {fmt(mn)}{(' ' + unit) if unit else ''}")
                results_max.append(f"{act}: {fmt(mx)}{(' ' + unit) if unit else ''}")
                results_mean.append(f"{act}: {fmt(md)}{(' ' + unit) if unit else ''}")

        return results_min, results_max, results_mean


    # Utility: draw exactly one row
    # -----------------------------------------------------------------------------
    # 4.11. Draw single row
    # -----------------------------------------------------------------------------
    def _draw_single_row(idx: int, smi: Any, count: int, pil_img: Any) -> None:
        row_tag = f"counts_table_row_{idx}"
        with dpg.table_row(tag=row_tag):

            with dpg.table_cell():
                dpg.add_text(f"{idx + 1}")

            img_texture_tag = f"counts_rgroup_texture_{idx}"
            with dpg.table_cell():
                with dpg.child_window(
                    tag=f"{img_texture_tag}_cell",
                    no_scrollbar=True,
                    horizontal_scrollbar=False,
                    no_scroll_with_mouse=True,
                    border=False,
                    auto_resize_y=True
                ):
                    dpg.add_image_button(
                        texture_tag=img_texture_tag,
                        tag=f"{img_texture_tag}_tag",
                        width=img_size - state["win_spacer"]*2 - 4,
                        height=img_size - state["win_spacer"]*2 - 4,
                        background_color=(0,0,0,255),
                        callback=update_rgroup_details,
                        user_data=(smi, state)
                    )

                register_responsive_image(
                    state,
                    image_tag=f"{img_texture_tag}_tag",
                    parent_tag=f"{img_texture_tag}_cell",
                    aspect_ratio=1.0,
                    parent_of_parent=row_tag,
                    default_pop_h=state["default_counts_table_row_height"],
                    tab="r_analysis_tab",
                )
                export_png_popup(f"{img_texture_tag}_tag", img_texture_tag, state)
                dpg.bind_item_theme(f"{img_texture_tag}_tag", apply_image_button_theme(state))

            with dpg.table_cell():
                matching = csv_df[csv_df[r_group] == smi]
                unique_ids = set(matching["Mol_sub_ID"]) if "Mol_sub_ID" in matching.columns else set(matching.index)
                dpg.add_text(f"{len(unique_ids)} / {total_molecules}")

            with dpg.table_cell():
                mol_feat = Chem.MolFromSmiles(smi)
                feats = extract_features(mol_feat)
                labels = sorted({lbl for (lbl, atoms) in feats})
                dpg.add_text(", ".join(labels) if labels else "N/A")

            stats_min, stats_max, stats_mean = _compute_stats_for_smi(smi)

            with dpg.table_cell():
                dpg.add_text("\n".join(stats_min))

            with dpg.table_cell():
                dpg.add_text("\n".join(stats_max))

            with dpg.table_cell():
                dpg.add_text("\n".join(stats_mean))


    # Utility: rebuild table for current page
    # -----------------------------------------------------------------------------
    # 4.12. Rebuild counts table
    # -----------------------------------------------------------------------------
    def _rebuild_counts_table() -> None:
        if dpg.does_item_exist("counts_table_table"):
            dpg.delete_item("counts_table_table")

        with dpg.table(
            tag="counts_table_table",
            parent="counts_table_window",
            header_row=True,
            resizable=True,
            scrollX=True,
            scrollY=True,
            row_background=True,
            freeze_rows=1,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True
        ):
            # Headers
            dpg.add_table_column(label="ID", init_width_or_weight=2)
            dpg.add_table_column(label="Image", init_width_or_weight=17)
            dpg.add_table_column(label="Count (no duplicates)", init_width_or_weight=12)
            dpg.add_table_column(label="Features", init_width_or_weight=14)
            dpg.add_table_column(label="Activity min", init_width_or_weight=15)
            dpg.add_table_column(label="Activity max", init_width_or_weight=15)
            dpg.add_table_column(label="Activity mean", init_width_or_weight=15)

            # Page slicing
            page = state["counts_current_page"]
            start = page * PAGE_SIZE
            end = min(start + PAGE_SIZE, total_items)

            visible_count = max(1, end - start)
            for idx in range(start, end):
                smi, count, pil_img = rgroup_image_list[idx]
                _draw_single_row(idx, smi, count, pil_img)
                rendered = idx - start + 1
                if rendered == visible_count or rendered % max(1, visible_count // 10) == 0:
                    set_loading_screen_progress(state, 82 + (rendered / visible_count) * 16)


    # Utility: go to previous page
    # -----------------------------------------------------------------------------
    # 4.13. Prev page
    # -----------------------------------------------------------------------------
    def _prev_page() -> None:
        if state["counts_current_page"] > 0:
            state["counts_current_page"] -= 1
            draw_loading_screen(state, bg=False)
            _rebuild_counts_table()
            update_responsive_images(state)
            dpg.set_value("counts_page_label", f"Page {state['counts_current_page']+1} / {total_pages}")
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")

    # Utility: go to next page
    # -----------------------------------------------------------------------------
    # 4.14. Next page
    # -----------------------------------------------------------------------------
    def _next_page() -> None:
        if state["counts_current_page"] < state["counts_total_pages"] - 1:
            state["counts_current_page"] += 1
            draw_loading_screen(state, bg=False)
            _rebuild_counts_table()
            update_responsive_images(state)
            dpg.set_value("counts_page_label", f"Page {state['counts_current_page']+1} / {total_pages}")
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")


    with dpg.group(horizontal=True, parent="counts_boxplot_settings_group", before="sort_counts_label"):
        dpg.add_button(arrow=True, direction=0, callback=lambda: _prev_page())
        dpg.add_text(f"Page {state['counts_current_page']+1} / {total_pages}", tag="counts_page_label")
        dpg.add_button(arrow=True, direction=1, callback=lambda: _next_page())
        dpg.add_spacer(width=state["win_spacer"] * 3)

    # Build first page
    _rebuild_counts_table()
    set_loading_screen_progress(state, 99)
    update_responsive_images(state)


    update_responsive_images(state)

    set_loading_screen_progress(state, 100)
    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")


# -----------------------------------------------------------------------------
# 5. Extract features
# -----------------------------------------------------------------------------
def extract_features(mol: Any) -> Any:
    """
    Extracts the features from a molecule, using RDKit's feature definitions and additional rules.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
        list: List of tuples (feature_label, atom_ids) where atom_ids can be None for generalised features.
    """
    # Defensive check for invalid molecules
    if mol is None:
        return [("N/A", None)]
    
    # Load RDKit base feature definitions and map verbose names to concise labels
    fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)

    feature_abbrev = {
        "Hydrogen": "Hydrogen",
        "Acceptor": "H-Bond Acceptor",
        "Donor": "H-Bond Donor",
        "Hydrophobe": "Hydrophobic",
        "Halogen": "Halogen",
        "Aromatic": "Aromatic",
        "NegIonizable": "Negative Ionizable",
        "PosIonizable": "Positive Ionizable",
        "Steric": "Steric Bulk",
        "LumpedHydrophobe": "Lumped Hydrophobe",
        "ZnBinder": "Zn Binder",
    }

    # Heuristics for 2-atom fragments like [*]-H, [*]-C, [*]-X as generic features.
    if mol.GetNumAtoms() == 2:
        symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        if "*" in symbols:
            other = (symbols - {"*"}).pop()
            if other == "H":
                return [("Hydrogen", None)]
            if other == "C":
                return [("Hydrophobic", None)]
            if other in {"F", "Cl", "Br", "I"}:
                return [("Halogen", None)]

    # Handle small halogen-only (±C/H/*) fragments as 'Halogen' features.
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    halogens = {"F", "Cl", "Br", "I"}
    if mol.GetNumAtoms() <= 6:
        halogen_count = sum(1 for s in atom_symbols if s in halogens)
        if halogen_count >= 1 and all(s in halogens.union({"C", "H", "*"}) for s in atom_symbols):
            return [("Halogen", None)]

    # Identify strong acceptor motifs from simple triple bonds (C#N, C#C).
    extra_features = []
    for bond in mol.GetBonds():
        if (bond.GetBondType() == Chem.BondType.TRIPLE and
            bond.GetBeginAtom().GetSymbol() in {"C", "N"} and
            bond.GetEndAtom().GetSymbol() in {"C", "N"}):
            extra_features.append(("H-Bond Acceptor", None))
            break

    # Compute RDKit features and group them by their atom ID tuples.
    features = factory.GetFeaturesForMol(mol)
    feature_names = [f.GetFamily() for f in features]
    abbrev_feats = [feature_abbrev.get(f, f) for f in feature_names]

    is_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    is_bulky = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]) >= 5

    result = []

    atomid_to_feats = {}
    for abbr, f_obj in zip(abbrev_feats, features):
        atom_ids = tuple(f_obj.GetAtomIds())
        atomid_to_feats.setdefault(atom_ids, []).append(abbr)

    # Prefer meaningful combos; otherwise adjust by aromatic/bulky context.
    for atom_ids, feats in atomid_to_feats.items():
        feats_set = set(feats)

        # Default combinations
        if "Aromatic" in feats_set and "H-Bond Acceptor" in feats_set and "Negative Ionizable" in feats_set:
            label = "Aromatic\nH-Bond Acceptor\nNegative Ionizable"
        elif "Aromatic" in feats_set and "H-Bond Donor" in feats_set and "Positive Ionizable" in feats_set:
            label = "Aromatic\nH-Bond Donor\nPositive Ionizable"
        elif "Aromatic" in feats_set and "H-Bond Acceptor" in feats_set:
            label = "Aromatic\nH-Bond Acceptor"
        elif "Aromatic" in feats_set and "H-Bond Donor" in feats_set:
            label = "Aromatic\nH-Bond Donor"
        elif "Aromatic" in feats_set and "Halogen" in feats_set:
            label = "Aromatic\nHalogen"
        elif "Aromatic" in feats_set and "Hydrophobic" in feats_set:
            label = "Aromatic\nHydrophobic"
        elif "Steric Bulk" in feats_set and "Hydrophobic" in feats_set:
            label = "Steric Bulk\nHydrophobic"
        elif "H-Bond Acceptor" in feats_set and "H-Bond Donor" in feats_set:
            label = "H-Bond Acceptor\nH-Bond Donor"
        elif "H-Bond Acceptor" in feats_set and "Negative Ionizable" in feats_set:
            label = "H-Bond Acceptor\nNegative Ionizable"
        elif "H-Bond Donor" in feats_set and "Positive Ionizable" in feats_set:
            label = "H-Bond Donor\nPositive Ionizable"
        elif "H-Bond Acceptor" in feats_set and "Zn Binder" in feats_set:
            label = "H-Bond Acceptor\nZn Binder"
        elif "H-Bond Donor" in feats_set and "Zn Binder" in feats_set:
            label = "H-Bond Donor\nZn Binder"
        else:
            # Fallback: take the first feature and add context if aromatic/bulky.
            label = feats[0]
            if is_aromatic and label != "Aromatic":
                label = f"Aromatic\n{label}"
            elif is_bulky and not is_aromatic and label != "Steric Bulk":
                label = f"Steric Bulk\n{label}"

        result.append((label, list(atom_ids)))

    # If no localised features were found, apply aromatic/bulky-only labels.
    if is_aromatic and not result:
        result.append(("Aromatic", None))
    elif is_bulky and not result:
        result.append(("Steric Bulk", None))

    # Append any extra rule-based features and return.
    result.extend(extra_features)
    return result
