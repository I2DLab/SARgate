"""
=======
notes.py
=======

Substructure-notepad module for SARgate GUI

Notes directly on the scaffold at R-group anchor positions:
- Invisible triggers placed on top of RDKit anchors [*:n] (aligned with drawing padding).
- On click, a small top-level DearPyGui window opens with a text input (movable).
- Notes are per-subset (including general notes) and autosaved to JSON in report_dir.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Define module configuration and shared state
# 3. Hex to rgba
# 4. Rgba f to u8
# 5. Build subset cores index
# 6. Refresh notes after data change
# 7. Ensure notes store
# 8. Load notes from disk
# 9. Save notes to disk
# 10. Smiles to texture arr
# 11. Get subsets
# 12. Notes popup window
# 13. Refresh notes subsets list
# 14. Load notes for subset
# 15. Extract rgroup mapnums
# 16. Core anchor norms
# 17. Place r triggers
# 18. Cb general notes changed
# 19. Cb r input changed
# 20. On notes edit
# 21. Ensure notes global handlers
# 22. Cb notes drag start
# 23. Cb notes global mouse move
# 24. Cb notes global mouse up

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import re
import io
import json
import dearpygui.dearpygui as dpg
import numpy as np
from collections import Counter
from typing import Any
from PIL import Image as pilImage
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D  
from app.utils.app_logger import log_exception


# -----------------------------------------------------------------------------
# 2. Define module configuration and shared state
# -----------------------------------------------------------------------------

_RG_MAPNUM_RE = re.compile(r"\[\*\:(\d+)\]")
NOTES_DEFAULT_PAD_FRAC = 0.25

# -----------------------------------------------------------------------------
# 3. Hex to rgba
# -----------------------------------------------------------------------------
def _hex_to_rgba(hex_str: Any, a: float = 1.0) -> Any:
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b, a)

PALETTE_RN = [
    _hex_to_rgba("#1f77b4"), _hex_to_rgba("#ff7f0e"), _hex_to_rgba("#2ca02c"), _hex_to_rgba("#d62728"), 
    _hex_to_rgba("#bcbd22"), _hex_to_rgba("#17becf"), _hex_to_rgba("#8c564b"), _hex_to_rgba("#9467bd"),
    _hex_to_rgba("#ff9896"), _hex_to_rgba("#c49c94"), _hex_to_rgba("#f7b6d2"), _hex_to_rgba("#e377c2"),
    _hex_to_rgba("#dbdb8d"), _hex_to_rgba("#9edae5"), _hex_to_rgba("#aec7e8"), _hex_to_rgba("#ffbb78"),
    _hex_to_rgba("#98df8a"), _hex_to_rgba("#c5b0d5"), _hex_to_rgba("#c7c7c7"), _hex_to_rgba("#7f7f7f"), 
]

# -----------------------------------------------------------------------------
# 4. Rgba f to u8
# -----------------------------------------------------------------------------
def _rgba_f_to_u8(r: Any, g: Any, b: Any, a: float = 1.0) -> Any:
    return (int(r*255), int(g*255), int(b*255), int(a*255))


# -----------------------------------------------------------------------------
# 5. Build subset cores index
# -----------------------------------------------------------------------------
def build_subset_cores_index(state: dict[str, Any]) -> Any:
    srgd = state.get("smiles_rgd_dict", {}) or {}
    index = {}
    for subset, mols in srgd.items():
        ctr = Counter()
        if isinstance(mols, dict):
            for _, rec in mols.items():
                if isinstance(rec, dict):
                    core = rec.get("Core")
                    if core:
                        ctr[core] += 1
        ranked = [c for c, _ in ctr.most_common()]
        index[subset] = {
            "counts": dict(ctr),
            "ranked": ranked,
            "most_common": (ranked[0] if ranked else None)
        }
    state["subset_cores_index"] = index
    return index


# -----------------------------------------------------------------------------
# 6. Refresh notes after data change
# -----------------------------------------------------------------------------
def refresh_notes_after_data_change(state: dict[str, Any]) -> None:
    build_subset_cores_index(state)
    load_notes_from_disk(state)
    if dpg.does_item_exist("notes_popup"):
        current = state.get("notes_current_subset")
        _refresh_notes_subsets_list(state)
        if current and current in (state.get("subset_cores_index") or {}):
            _load_notes_for_subset(current, state)


# -----------------------------------------------------------------------------
# 7. Ensure notes store
# -----------------------------------------------------------------------------
def _ensure_notes_store(state: dict[str, Any]) -> Any:
    if "notes_store" not in state:
        state["notes_store"] = {}  # { subset: { "R1": "...", "R2": "...", "__general__": "..." } }
    report_dir = state.get("report_dir") or ""
    if report_dir and os.path.isdir(report_dir):
        state["notes_file"] = os.path.join(report_dir, "notes.snf")
    else:
        base = os.path.dirname(state.get("settings_file", os.path.join("assets", "config", "settings.ssf")))
        state["notes_file"] = os.path.join(base, "notes.snf")
    return state["notes_store"], state["notes_file"]


# -----------------------------------------------------------------------------
# 8. Load notes from disk
# -----------------------------------------------------------------------------
def load_notes_from_disk(state: dict[str, Any]) -> None:
    store, path = _ensure_notes_store(state)
    candidate_paths = [path]
    if path.lower().endswith(".snf"):
        candidate_paths.append(f"{os.path.splitext(path)[0]}.json")
    try:
        for candidate_path in candidate_paths:
            if not os.path.isfile(candidate_path):
                continue
            with open(candidate_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                state["notes_store"] = data
                return
    except Exception as e:
        log_exception("Notes", "Notes load failed", e, indent=1)
    state["notes_store"] = store


# -----------------------------------------------------------------------------
# 9. Save notes to disk
# -----------------------------------------------------------------------------
def save_notes_to_disk(state: dict[str, Any]) -> None:
    store, path = _ensure_notes_store(state)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(store, f, indent=2)
    except Exception as e:
        log_exception("Notes", "Notes save failed", e, indent=1)


# -----------------------------------------------------------------------------
# 10. Smiles to texture arr
# -----------------------------------------------------------------------------
def _smiles_to_texture_arr(
    smi_or_smarts: str,
    w: Any,
    h: Any,
    pad_frac: Any = NOTES_DEFAULT_PAD_FRAC
) -> Any:
    # --- parse mol ---
    mol = None
    if isinstance(smi_or_smarts, str) and smi_or_smarts:
        mol = Chem.MolFromSmiles(smi_or_smarts)
        if mol is None:
            mol = Chem.MolFromSmarts(smi_or_smarts)
    if mol is None:
        raise ValueError("Invalid SMILES/SMARTS")

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.padding = float(pad_frac)
    opts.bondLineWidth = 1
    opts.highlightBondWidthMultiplier = 12

    # --- Mapped atoms (R-anchors) + label 'Rn' ---
    mapped = []
    for a in mol.GetAtoms():
        amap = a.GetAtomMapNum()
        if amap > 0:
            mapped.append((a.GetIdx(), amap))
            opts.atomLabels[a.GetIdx()] = f"R{amap}"
    mapped.sort(key=lambda t: t[1])  # R1, R2, R3, ...

    # --- Highlight atoms with fixed palette ---
    highlight_atoms = []
    highlight_cols = {}
    atom_color = {}  # aidx -> (r,g,b,a)
    if mapped:
        for i, (aidx, _amap) in enumerate(mapped):
            r, g, b, a = PALETTE_RN[i % len(PALETTE_RN)]
            highlight_atoms.append(aidx)
            color = (r, g, b, a)
            highlight_cols[aidx] = color
            atom_color[aidx] = color

    # --- Bonds "toward the R": those with EXACTLY one mapped end ---
    highlight_bonds = []
    highlight_bond_cols = {}
    if atom_color:
        for bond in mol.GetBonds():
            b = bond.GetBeginAtomIdx()
            e = bond.GetEndAtomIdx()
            b_m = b in atom_color
            e_m = e in atom_color
            if b_m ^ e_m:  # XOR: exactly one end mapped
                bid = bond.GetIdx()
                highlight_bonds.append(bid)
                # Color = color of the mapped atom
                col = atom_color[b] if b_m else atom_color[e]
                # RDKit accepts (r,g,b) or (r,g,b,a). If your build doesn't support alpha, remove it:
                highlight_bond_cols[bid] = col  # or col[:3]

    # --- draw ---
    rdMolDraw2D.PrepareMolForDrawing(mol)
    if highlight_atoms or highlight_bonds:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_cols,
            highlightBonds=highlight_bonds,
            highlightBondColors=highlight_bond_cols,
            highlightAtomRadii={a: 0.40 for a in highlight_atoms},
        )
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()

    img = pilImage.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    arr = (np.asarray(img) / 255.0).astype(np.float32)
    return arr.flatten()


# -----------------------------------------------------------------------------
# 11. Get subsets
# -----------------------------------------------------------------------------
def _get_subsets(state: dict[str, Any]) -> Any:
    srgd = state.get("smiles_rgd_dict", {}) or {}
    return [str(k) for k in srgd.keys()]


# -----------------------------------------------------------------------------
# 12. Notes popup window
# -----------------------------------------------------------------------------
def notes_popup_window(state: dict[str, Any]) -> None:
    build_subset_cores_index(state)
    load_notes_from_disk(state)
    _ensure_notes_global_handlers(state)

    IMG_W = int(state.get("notes_img_width"))
    IMG_H = int(state.get("notes_img_height"))

    if not dpg.does_item_exist("notes_popup"):
        with dpg.window(label="SAR Notes", show=False, tag="notes_popup",
                        width=IMG_W + 155, height=775,
                        no_scrollbar=False, horizontal_scrollbar=True, no_scroll_with_mouse=False, no_resize=False):
            with dpg.group(horizontal=True):
                # LEFT: subsets list
                with dpg.child_window(tag="notes_left_panel", width=120, height=-1, border=False,
                                      no_scroll_with_mouse=False, no_scrollbar=False):
                    dpg.add_text("Subsets")
                    dpg.add_separator()
                    dpg.add_child_window(tag="notes_subsets_list", width=-1, height=-1, border=False)

                # RIGHT: scaffold + general notes
                with dpg.child_window(tag="notes_mid_panel", auto_resize_x=True, auto_resize_y=True, border=False,
                                      no_scroll_with_mouse=False, no_scrollbar=False):
                    dpg.add_dynamic_texture(IMG_W, IMG_H, [1.0] * (IMG_W * IMG_H * 4), 
                                            tag="notes_scaffold_tex", parent="texture_registry")
                    dpg.add_image("notes_scaffold_tex", tag="notes_scaffold_img",
                                  width=IMG_W, height=IMG_H, border_color=(0, 0, 0, 0))

                    dpg.add_collapsing_header(label="General notes", tag="notes_collapsing_header",
                                              default_open=True, closable=False, leaf=True, bullet=False)
                    dpg.add_input_text(tag="notes_general_input",
                                       width=IMG_W, height=75, 
                                       multiline=True,
                                       tab_input=True,
                                       default_value="")
                    with dpg.theme() as general_notes_input_theme:
                        with dpg.theme_component(dpg.mvInputText):
                            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,      (255, 255, 255, 255))
                            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1.0)
                            dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0, 255))
                    dpg.bind_item_theme("notes_general_input", general_notes_input_theme)

        # init tracking for move-follow + z-order
        state["notes_popup_last_pos"] = dpg.get_item_pos("notes_popup")
        state["notes_open_windows"] = {}
    _refresh_notes_subsets_list(state)


# -----------------------------------------------------------------------------
# 13. Refresh notes subsets list
# -----------------------------------------------------------------------------
def _refresh_notes_subsets_list(state: dict[str, Any]) -> None:
    if dpg.does_item_exist("notes_subsets_list"):
        dpg.delete_item("notes_subsets_list", children_only=True)

    for subset in _get_subsets(state):
        subset_str = str(subset)
        dpg.add_button(
            label=subset_str.replace("subset_", "Subset "),
            parent="notes_subsets_list",
            user_data=subset_str,
            callback=lambda sender, app_data, user_data: _load_notes_for_subset(user_data, state),
            width=-1, height=24
        )


# -----------------------------------------------------------------------------
# 14. Load notes for subset
# -----------------------------------------------------------------------------
def _load_notes_for_subset(subset: str, state: dict[str, Any]) -> None:
    subset = str(subset)
    state["notes_current_subset"] = subset
    store, _ = _ensure_notes_store(state)
    subset_notes = store.setdefault(subset, {})

    # General notes (restore + autosave)
    if dpg.does_item_exist("notes_general_input"):
        dpg.set_value("notes_general_input", str(subset_notes.get("__general__", "")))
        dpg.configure_item(
            "notes_general_input",
            callback=_cb_general_notes_changed,
            user_data={"subset": subset, "state": state}
        )

    # Scaffold image (hide anchor labels)
    IMG_W = int(state.get("notes_img_width"))
    IMG_H = int(state.get("notes_img_height"))

    pad_frac = float(state.get("notes_draw_padding", NOTES_DEFAULT_PAD_FRAC))
    idx = state.get("subset_cores_index", {})
    core_smarts = (idx.get(subset) or {}).get("most_common")

    if not core_smarts:
        try:
            core_smarts = state["smiles_rgd_dict"][subset]["mol_1"]["Core"]
        except Exception:
            core_smarts = (state.get("smiles_rgd_dict", {}).get(subset, {}).get("mol_2", {}) or {}).get("Core", None)

    tex = _smiles_to_texture_arr(core_smarts, IMG_W, IMG_H, pad_frac=pad_frac) \
          if core_smarts else np.ones((IMG_W * IMG_H * 4,), dtype=np.float32)
    dpg.set_value("notes_scaffold_tex", tex)

    # Place invisible triggers aligned to the R anchors/labels
    _place_r_triggers(subset, core_smarts, state)


# -----------------------------------------------------------------------------
# 15. Extract rgroup mapnums
# -----------------------------------------------------------------------------
def _extract_rgroup_mapnums(subset: str, state: dict[str, Any]) -> Any:
    srgd = state.get("smiles_rgd_dict", {}) or {}
    rec = srgd.get(subset) or {}
    mol_block = rec.get("mol_1") or rec.get("mol_2") or {}

    out = {}
    for k, v in mol_block.items():
        if not isinstance(k, str):
            continue
        if not k or k[0] not in ("R", "r"):
            # skip non-R keys (e.g. 'Core')
            continue

        # Standard case: 'R3' -> 3
        suf = k[1:]
        if suf.isdigit():
            out[k] = int(suf)
            continue

        # Fallback ONLY for keys starting with R/r but without a digit after
        if isinstance(v, str):
            m = _RG_MAPNUM_RE.search(v)
            if m:
                out[k] = int(m.group(1))

    return out


# -----------------------------------------------------------------------------
# 16. Core anchor norms
# -----------------------------------------------------------------------------
def _core_anchor_norms(core_smarts: str) -> Any:
    out = {}
    if not core_smarts:
        return out

    patt = Chem.MolFromSmarts(core_smarts)
    if patt is None:
        patt = Chem.MolFromSmiles(core_smarts)
        if patt is None:
            return out

    rdDepictor.Compute2DCoords(patt)
    conf = patt.GetConformer()
    xs, ys = [], []
    for a in patt.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        xs.append(p.x)
        ys.append(p.y)

    if not xs or not ys:
        return out

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0

    for a in patt.GetAtoms():
        amap = a.GetAtomMapNum()
        if amap <= 0:
            continue
        p = conf.GetAtomPosition(a.GetIdx())
        xn = (p.x - xmin) / dx
        yn = (p.y - ymin) / dy
        yn = 1.0 - yn  # invert for image coordinates
        out[int(amap)] = (float(xn), float(yn))
    return out


# -----------------------------------------------------------------------------
# 17. Place r triggers
# -----------------------------------------------------------------------------
def _place_r_triggers(subset: str, core_smarts: str, state: dict[str, Any]) -> Any:

    # Local helper: convert float RGBA [0..1] to uint8 [0..255]
    # -----------------------------------------------------------------------------
    # 17.1. Rgba f to u8
    # -----------------------------------------------------------------------------
    def _rgba_f_to_u8(r: Any, g: Any, b: Any, a: float = 1.0) -> Any:
        return (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

    subset = str(subset)
    IMG_W = int(state.get("notes_img_width"))
    IMG_H = int(state.get("notes_img_height"))

    # Ensure overlay group exists (inside notes_mid_panel)
    if not dpg.does_item_exist("notes_overlay_group"):
        dpg.add_group(tag="notes_overlay_group", parent="notes_mid_panel")
    else:
        dpg.delete_item("notes_overlay_group", children_only=True)

    anchors = _core_anchor_norms(core_smarts)          # {mapnum: (xn, yn)} ONLY those present in the core
    r_to_map = _extract_rgroup_mapnums(subset, state)  # {'R1':1, 'R2':2, ...}
    img_pos = dpg.get_item_pos("notes_scaffold_img")

    # Same padding used for drawing to align correctly
    pad_frac = float(state.get("notes_draw_padding", NOTES_DEFAULT_PAD_FRAC))
    pad_x = int(IMG_W * pad_frac)
    pad_y = int(IMG_H * pad_frac)
    inner_w = IMG_W - 2 * pad_x
    inner_h = IMG_H - 2 * pad_y

    present_amaps = sorted(anchors.keys())
    amap_to_color = {am: PALETTE_RN[i % len(PALETTE_RN)] for i, am in enumerate(present_amaps)}
    state.setdefault("notes_color_maps", {})[subset] = amap_to_color  # optional: useful for legends/tooltips

    all_hdr_themes = state.setdefault("notes_hdr_themes", {})  # {subset: {amap: theme_tag}}
    # if themes exist for this subset, delete them
    if subset in all_hdr_themes:
        for _am, _tag in list(all_hdr_themes[subset].items()):
            if dpg.does_item_exist(_tag):
                dpg.delete_item(_tag)
        all_hdr_themes[subset].clear()
    else:
        all_hdr_themes[subset] = {}

    # create new themes for each PRESENT amap
    for am in present_amaps:
        r_f, g_f, b_f, _ = amap_to_color[am]
        hdr_col_base   = _rgba_f_to_u8(r_f, g_f, b_f, 0.60)
        hdr_col_hover  = _rgba_f_to_u8(r_f, g_f, b_f, 0.75)
        hdr_col_active = _rgba_f_to_u8(r_f, g_f, b_f, 0.90)

        theme_tag = f"notes_hdr_theme_{subset}_am{am}"  # unique for subset+amap
        with dpg.theme(tag=theme_tag):
            with dpg.theme_component(dpg.mvCollapsingHeader):
                dpg.add_theme_color(dpg.mvThemeCol_Header,        hdr_col_base)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, hdr_col_hover)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,  hdr_col_active)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,      (255, 255, 255, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1.0)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0, 255))
        all_hdr_themes[subset][am] = theme_tag

    # --- create a child for each declared R but ONLY if the anchor is present in the image ---
    for r_label, amap in r_to_map.items():
        if amap not in anchors:
            # R present in data but absent on drawn core → skip to avoid mismatched colors
            continue

        xn, yn = anchors[amap]
        # Anchor centre in absolute coordinates
        cx = int(img_pos[0] + pad_x + xn * inner_w)
        cy = int(img_pos[1] + pad_y + yn * inner_h)

        win_tag   = f"notes_win_{subset}_{r_label}"
        hdr_tag   = f"{win_tag}_hdr"
        input_tag = f"notes_input_{subset}_{r_label}"

        # initial compact dimensions; the child stays close to the anchor
        pw, ph = 100, 60
        px = int(cx)
        py = int(cy)

        with dpg.child_window(
            tag=win_tag,
            parent="notes_overlay_group",
            pos=(px, py),
            width=pw,
            height=ph,
            no_scrollbar=False,
            horizontal_scrollbar=True,
            border=False,
            resizable_x=True,
            resizable_y=True
        ):
            # Collapsing header = draggable grip + content toggle
            dpg.add_collapsing_header(
                tag=hdr_tag,
                label=r_label,
                default_open=False,
                closable=False,
                open_on_arrow=True,
                open_on_double_click=False
            )

            # Handler: mouse down on collapsing header = start drag
            reg_tag = f"{hdr_tag}_handlers"
            if not dpg.does_item_exist(reg_tag):
                with dpg.item_handler_registry(tag=reg_tag):
                    dpg.add_item_clicked_handler(
                        callback=_cb_notes_drag_start,
                        user_data={"win_tag": win_tag, "state": state}
                    )
            dpg.bind_item_handler_registry(hdr_tag, reg_tag)

            # Writable content (shown when the header is open)
            store, _ = _ensure_notes_store(state)
            subset_notes = store.setdefault(subset, {})
            default_val = str(subset_notes.get(r_label, ""))

            with dpg.group(parent=hdr_tag):
                dpg.add_input_text(
                    tag=input_tag,
                    multiline=True,
                    tab_input=True,
                    width=-1,
                    height=-1,
                    default_value=default_val,
                    callback=_cb_r_input_changed,
                    user_data={"subset": subset, "r_label": r_label, "state": state}
                )

        # BIND header theme (per-subset and per-amap), outside the child context
        theme_tag = all_hdr_themes[subset].get(amap)
        if theme_tag and dpg.does_item_exist(theme_tag):
            dpg.bind_item_theme(hdr_tag, theme_tag)

        # Theme for the child (transparent, white input)
        with dpg.theme() as notes_child_no_frame_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (0, 0, 0, 0))
            with dpg.theme_component(dpg.mvInputText):
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255, 255))
        dpg.bind_item_theme(win_tag, notes_child_no_frame_theme)

        # Register offset for possible popup follow
        popup_pos = dpg.get_item_pos("notes_popup")
        dx = px - popup_pos[0]
        dy = py - popup_pos[1]
        state["notes_open_windows"][win_tag] = {"subset": subset, "r": r_label, "offset": (dx, dy)}


# -----------------------------------------------------------------------------
# 18. Cb general notes changed
# -----------------------------------------------------------------------------
def _cb_general_notes_changed(sender: Any, app_data: Any, user_data: Any) -> None:
    subset = user_data.get("subset")
    state  = user_data.get("state")
    _on_notes_edit(subset, "__general__", app_data, state)


# -----------------------------------------------------------------------------
# 19. Cb r input changed
# -----------------------------------------------------------------------------
def _cb_r_input_changed(sender: Any, app_data: Any, user_data: Any) -> None:
    _on_notes_edit(user_data["subset"], user_data["r_label"], app_data, user_data["state"])


# -----------------------------------------------------------------------------
# 20. On notes edit
# -----------------------------------------------------------------------------
def _on_notes_edit(subset: str, field: Any, value: Any, state: dict[str, Any]) -> None:
    store, _ = _ensure_notes_store(state)
    subset_notes = store.setdefault(subset, {})
    subset_notes[field] = value
    try:
        save_notes_to_disk(state)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 21. Ensure notes global handlers
# -----------------------------------------------------------------------------
def _ensure_notes_global_handlers(state: dict[str, Any]) -> None:
    dpg.add_mouse_move_handler(parent="handler_registry", callback=_cb_notes_global_mouse_move, user_data=state)
    dpg.add_mouse_release_handler(parent="handler_registry", callback=_cb_notes_global_mouse_up, user_data=state)


# -----------------------------------------------------------------------------
# 22. Cb notes drag start
# -----------------------------------------------------------------------------
def _cb_notes_drag_start(sender: Any, app_data: Any, user_data: Any) -> None:
    state = user_data["state"]
    win_tag = user_data["win_tag"]
    mx, my = dpg.get_mouse_pos(local=False)
    wx, wy = dpg.get_item_pos(win_tag)
    state["notes_drag"] = {"win_tag": win_tag, "dx": mx - wx, "dy": my - wy, "active": True}


# -----------------------------------------------------------------------------
# 23. Cb notes global mouse move
# -----------------------------------------------------------------------------
def _cb_notes_global_mouse_move(sender: Any, app_data: Any, user_data: Any) -> None:
    state = user_data
    drag = state.get("notes_drag") or {}
    if not drag.get("active"):
        return
    mx, my = dpg.get_mouse_pos(local=False)
    nx = int(mx - drag["dx"])
    ny = int(my - drag["dy"])
    try:
        dpg.set_item_pos(drag["win_tag"], (nx, ny))
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 24. Cb notes global mouse up
# -----------------------------------------------------------------------------
def _cb_notes_global_mouse_up(sender: Any, app_data: Any, user_data: Any) -> None:
    state = user_data
    drag = state.get("notes_drag")
    if drag:
        drag["active"] = False
        
