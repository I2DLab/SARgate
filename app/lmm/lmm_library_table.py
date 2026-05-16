"""
==================
lmm_library_table.py
==================

Library summary table visualization.

Displays all molecules loaded into SARgate with their metadata.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Show library summary table

import os
import dearpygui.dearpygui as dpg
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
from typing import Any
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from app.gui.loading_win import draw_loading_screen
from app.gui.themes_manager import (
    apply_inner_child_theme,
    apply_input_text_theme
)
from app.lmm.lmm_file_reader import _read_excel_robust
from collections import OrderedDict
import io
from PIL import Image as pilImage


# -----------------------------------------------------------------------------
# 2. Show library summary table
# -----------------------------------------------------------------------------
def show_library_summary_table(state: dict[str, Any], selected_file_path: str) -> Any:
    if dpg.does_item_exist("library_table_window_inner"):
        return

    draw_loading_screen(state, bg=False)

    filename = os.path.basename(selected_file_path)
    ext = os.path.splitext(selected_file_path)[1].lower()

    if ext == ".sdf":
        suppl = Chem.SDMolSupplier(selected_file_path)
        mols = [mol for mol in suppl if mol is not None]
        if not mols:
            print(f"[Library] No valid molecules found in {filename}.")
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")
            return

        names = [mol.GetProp("_Name") if mol.HasProp("_Name") else f"Mol_{i+1}" for i, mol in enumerate(mols)]
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        df = pd.DataFrame({"ROMol": mols, "Molecule Name": names, "Smiles": smiles})

    elif ext in [".csv", ".tsv", ".xlsx"]:

        def _detect_sep_quick(path: str) -> Any:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                line = f.readline()
            counts = {"\t": line.count("\t"), ",": line.count(","), ";": line.count(";")}
            return max(counts, key=counts.get)

        try:
            if ext == ".xlsx":
                df = _read_excel_robust(selected_file_path)
            else:
                sep = _detect_sep_quick(selected_file_path)
                df = pd.read_csv(selected_file_path, sep=sep, engine="python", encoding="utf-8-sig")
        except Exception as exc:
            print(f"[Library] Error reading table file: {exc}")
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")
            return

        smiles_col = state.get("smiles_column")
        if not smiles_col or smiles_col not in df.columns:
            print(f"[Library] Missing SMILES column '{smiles_col}' in {selected_file_path}.")
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")
            return

        # Normalise columns
        df["Smiles"] = df[smiles_col].astype(str)

        # Build ROMol
        mols = [Chem.MolFromSmiles(smi) for smi in df["Smiles"]]
        df["ROMol"] = mols

        if "Molecule Name" not in df.columns:
            df["Molecule Name"] = [f"Mol_{i+1}" for i in range(len(df))]


    elif ext in [".smi", ".txt"]:
        try:
            with open(selected_file_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            smiles, names = [], []
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) == 1:
                    smiles.append(parts[0]); names.append(f"Mol_{i+1}")
                else:
                    smiles.append(parts[0]); names.append(" ".join(parts[1:]))
            df = pd.DataFrame({"Smiles": smiles, "Molecule Name": names})
        except Exception:
            print(f"[Library] Failed to read SMILES file: {filename}")
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")
            return
        mols = [Chem.MolFromSmiles(smi) for smi in df["Smiles"]]
        df["ROMol"] = mols

    else:
        print(f"[Library] Unsupported file type: {ext}")
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        return

    df = df[df["ROMol"].notna()].reset_index(drop=True)
    mols = df["ROMol"].tolist()
    if not mols:
        print(f"[Library] No valid molecules found in {filename}.")
        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
        return

    # Create a lightweight LRU texture cache that builds textures on demand for visible rows only.
    # -----------------------------------------------------------------------------
    # 2.1. Texture cache
    # -----------------------------------------------------------------------------
    class _TextureCache:
        """
        LRU cache for DearPyGui textures backed by on-demand RDKit 2D drawings.
        It keeps GPU memory bounded and avoids pre-rendering all molecules.
        """
        def __init__(self, capacity: int, img_w: int, img_h: int) -> None:
            self.capacity = max(10, int(capacity))
            self.img_w = img_w
            self.img_h = img_h
            self._store = OrderedDict()   # idx -> tag

        def _make_texture(self, idx: int, mol: Any) -> Any:
            # Compute 2D coords (once per mol) and draw to Cairo.
            try:
                rdDepictor.Compute2DCoords(mol, clearConfs=True)
            except Exception:
                pass

            drawer = rdMolDraw2D.MolDraw2DCairo(self.img_w, self.img_h)
            opts = drawer.drawOptions()
            opts.padding = 0.025
            opts.bondLineWidth = 1
            opts.minFontSize = 1
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            png_bytes = drawer.GetDrawingText()

            mol_img = pilImage.open(io.BytesIO(png_bytes)).convert("RGBA")
            mol_arr = (np.array(mol_img) / 255.0).astype(np.float32).flatten()

            tex_tag = f"library_table_mol_texture_{idx}"
            # Ensure unique tag per idx; if it existed and was evicted, recreate.

            if not dpg.does_item_exist(tex_tag):
                dpg.add_dynamic_texture(self.img_w, self.img_h, mol_arr, tag=tex_tag, parent="texture_registry")
            else:
                dpg.set_value(tex_tag, mol_arr)
                
            return tex_tag


        def get(self, idx: int, mol: Any) -> Any:
            # Return existing texture or create a new one; maintain LRU order.
            if idx in self._store:
                tag = self._store.pop(idx)
                self._store[idx] = tag
                return tag
            tag = self._make_texture(idx, mol)
            self._store[idx] = tag
            # Evict if over capacity.
            if len(self._store) > self.capacity:
                old_idx, old_tag = self._store.popitem(last=False)
                if dpg.does_item_exist(old_tag):
                    dpg.delete_item(old_tag)
            return tag

        def clear(self) -> None:
            # Delete all GPU textures managed by this cache.
            """
            Clear the requested state.
            
            Args:
                None.
            
            Returns:
                None: This routine updates state or performs side effects in place.
            """
            for _, tag in list(self._store.items()):
                if dpg.does_item_exist(tag):
                    dpg.delete_item(tag)
            self._store.clear()

    img_width  = state["library_table_img_width"]
    img_height = state["library_table_img_height"]
    PAGE_SIZE  = 50  # keep configurable if needed

    # Initialize cache sized to a few pages (to make paging snappy).
    tex_cache = _TextureCache(capacity=PAGE_SIZE * 3, img_w=img_width, img_h=img_height)
    state["_lib_tex_cache"] = tex_cache

    # Prepare a placeholder texture column (values filled lazily in renderer).
    df["__texture__"] = None

    all_columns = df.columns.tolist()
    priority = ["__texture__", "Molecule Name", "Smiles"]
    rest = [col for col in all_columns if col not in priority]
    df = df[priority + rest]

    state["_lib_df"] = df
    state["_lib_total"] = len(df)
    state["_lib_indices"] = list(range(len(df)))
    state["_lib_page"] = 0
    state["_lib_page_size"] = PAGE_SIZE

    # -----------------------------------------------------------------------------
    # 2.2. Filter table
    # -----------------------------------------------------------------------------
    def filter_table() -> Any:
        """
        Create a callback that performs a global text search over the dataset and refreshes the view.
        
        Args:
            None.
        
        Returns:
            Any: Value produced by the routine.
        """
        # -----------------------------------------------------------------------------
        # 2.2.1. Filter callback
        # -----------------------------------------------------------------------------
        def _filter_callback() -> None:
            query = dpg.get_value("library_table_filter_input").strip().lower()
            df_all = state["_lib_df"]

            if not query:
                state["_lib_indices"] = list(range(len(df_all)))
            else:
                # Exclude non-textual columns from the search for speed and correctness.
                cols_to_search = [c for c in df_all.columns if c not in ("__texture__", "ROMol")]
                # Vectorized contains across selected columns.
                mask = df_all[cols_to_search].astype(str).apply(lambda s: s.str.lower().str.contains(query, na=False))
                matched = mask.any(axis=1)
                state["_lib_indices"] = list(np.flatnonzero(matched.values))

            state["_lib_page"] = 0
            render_table_rows()
            update_pager_labels()
        return _filter_callback

    # -----------------------------------------------------------------------------
    # 2.3. Render table rows
    # -----------------------------------------------------------------------------
    def render_table_rows() -> None:
        """
        Rebuild the table rows limited to the current page.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        table_tag = "library_table"
        df_all = state["_lib_df"]
        indices = state["_lib_indices"]
        page = state["_lib_page"]
        page_size = state["_lib_page_size"]

        # Clear existing rows.
        children = dpg.get_item_children(table_tag, 1)
        if children:
            for row in children:
                dpg.delete_item(row)

        start = page * page_size
        end = min(start + page_size, len(indices))

        for pos in range(start, end):
            i = indices[pos]
            row = df_all.iloc[i]
            with dpg.table_row(parent=table_tag):
                # ID column (stable global index + 1)
                dpg.add_text(str(i + 1))
                for col in df_all.columns:
                    if col == "__texture__":
                        # Lazy fetch/generate texture for this row.
                        mol = df_all.at[i, "ROMol"]
                        tex_tag = tex_cache.get(i, mol)
                        dpg.add_image(tex_tag, width=img_width, height=img_height)
                    elif col == "ROMol":
                        # Skip raw molecule object column in the UI (too verbose).
                        continue
                    else:
                        # Lightweight, non-editable cell.
                        dpg.add_text("" if pd.isna(row[col]) else str(row[col]))

    # -----------------------------------------------------------------------------
    # 2.4. Update pager labels
    # -----------------------------------------------------------------------------
    def update_pager_labels() -> None:
        """
        Update the top toolbar counters and current page information.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        total = state["_lib_total"]
        vis = len(state["_lib_indices"])
        page = state["_lib_page"]
        page_size = state["_lib_page_size"]
        pages = max(1, (vis + page_size - 1) // page_size)

        dpg.set_value("library_table_count_text", f"Molecules: {vis} (of {total})")
        dpg.set_value("library_table_page_text", f"Page {min(page+1, pages)}/{pages}")
        dpg.configure_item("library_table_last_btn", label=f"Page {pages}")

    # -----------------------------------------------------------------------------
    # 2.5. Go first page
    # -----------------------------------------------------------------------------
    def go_first_page() -> None:
        state["_lib_page"] = 0
        render_table_rows(); update_pager_labels()

    # -----------------------------------------------------------------------------
    # 2.6. Go prev page
    # -----------------------------------------------------------------------------
    def go_prev_page() -> None:
        if state["_lib_page"] > 0:
            state["_lib_page"] -= 1
            render_table_rows(); update_pager_labels()

    # -----------------------------------------------------------------------------
    # 2.7. Go next page
    # -----------------------------------------------------------------------------
    def go_next_page() -> None:
        vis = len(state["_lib_indices"])
        pages = max(1, (vis + state["_lib_page_size"] - 1) // state["_lib_page_size"])
        if state["_lib_page"] < pages - 1:
            state["_lib_page"] += 1
            render_table_rows(); update_pager_labels()

    # -----------------------------------------------------------------------------
    # 2.8. Go last page
    # -----------------------------------------------------------------------------
    def go_last_page() -> None:
        vis = len(state["_lib_indices"])
        state["_lib_page"] = max(0, (vis + state["_lib_page_size"] - 1) // state["_lib_page_size"] - 1)
        render_table_rows(); update_pager_labels()

    # -----------------------------------------------------------------------------
    # 2.9. Clear search
    # -----------------------------------------------------------------------------
    def clear_search() -> None:
        """
        Clear search.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        dpg.set_value("library_table_filter_input", "")
        state["_lib_indices"] = list(range(len(state["_lib_df"])))
        state["_lib_page"] = 0
        render_table_rows(); update_pager_labels()


    with dpg.child_window(parent="library_table_window", tag="library_table_window_inner",
                          no_scrollbar=False, horizontal_scrollbar=True, no_scroll_with_mouse=False, border=False):

        with dpg.group(horizontal=True):
            dpg.add_text("", tag="library_table_count_text")
            dpg.add_spacer(width=state["win_spacer"]*2)

            dpg.add_input_text(hint="Search...", tag="library_table_filter_input",
                               auto_select_all=True, width=220)
            dpg.add_button(label="Search", callback=filter_table())
            dpg.add_button(label="Reset",  callback=clear_search)

            dpg.add_spacer(width=state["win_spacer"]*2)
            dpg.add_button(label="Page 1", callback=go_first_page)
            dpg.add_button(arrow=True, direction=0, callback=go_prev_page)
            dpg.add_text("", tag="library_table_page_text")
            dpg.add_button(arrow=True, direction=1, callback=go_next_page)
            dpg.add_button(label="Last Page", tag="library_table_last_btn", callback=go_last_page)

        dpg.add_separator()

        with dpg.child_window(autosize_x=True, autosize_y=True, horizontal_scrollbar=True, border=False,
                              tag="library_table_child_window"):
            with dpg.table(tag="library_table", header_row=True, resizable=True,
                           borders_innerH=True, borders_innerV=True,
                           borders_outerH=True, borders_outerV=True,
                           reorderable=True, scrollX=True, scrollY=True,
                           policy=dpg.mvTable_SizingStretchProp, freeze_rows=1):


                id_weight = 1
                mol_weight = 20

                dpg.add_table_column(label="ID", no_reorder=True, init_width_or_weight=id_weight)
                
                # Create columns for the reordered dataframe; skip ROMol in the header.

                for col in df.columns:
                    if col == "ROMol":
                        continue
                    elif col == "__texture__":
                        dpg.add_table_column(label="Molecule", no_reorder=True, init_width_or_weight=mol_weight)
                    else:
                        dpg.add_table_column(label=col, init_width_or_weight=(100 - id_weight - mol_weight) / len(df.columns))

                # Initial page render.
                render_table_rows()

            dpg.bind_item_theme("library_table", apply_input_text_theme())
        dpg.bind_item_theme("library_table_child_window", apply_inner_child_theme())

    update_pager_labels()

    if dpg.does_item_exist("cover_layer"):
        dpg.delete_item("cover_layer")
