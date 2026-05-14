"""
========================
chemspace_plot_common.py
========================

Shared plot utilities for ChemSpace reducers.
"""

import io
import math
import os
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image as pilImage, ImageDraw, ImageFont
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import MACCSkeys, rdDepictor, rdMolDescriptors, rdFMCS, ChemicalFeatures
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator, GetTopologicalTorsionGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from app.gui.loading_win import draw_loading_screen, set_loading_screen_progress
from app.gui.themes_manager import (
    apply_dendrogram_theme,
    apply_colormap_theme,
    apply_plot_theme,
    get_continuous_colormap_color,
)
from app.utils.app_logger import log_event, log_settings
from app.utils.callbacks import (
    add_chemspace_plot_specific_popup_controls,
    color_string_to_rgb255,
    export_png_popup,
    register_plot_context_popup,
    register_responsive_image,
    rgba_tuple_to_string,
    update_responsive_images,
)
from app.utils.resource_paths import open_html_safely, resource_path


def _chemspace_html_output_path(state: dict[str, Any], filename: str) -> Path:
    """
    Return a persistent output path for browser-opened ChemSpace HTML files.

    Args:
        state (dict[str, Any]): Shared application state.
        filename (str): HTML filename to create.

    Returns:
        Path: Writable path outside volatile system temporary directories.
    """
    html_dir = resource_path("app", "analysis", "chemspace", "html")
    html_dir.mkdir(parents=True, exist_ok=True)
    return html_dir / filename


def _tooltip_text_color(bg_rgba: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r, g, b = bg_rgba[:3]
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return (20, 20, 20, 255) if luminance > 0.6 else (245, 245, 245, 255)


def _build_tooltip_theme(tag: str, bg_rgba: tuple[int, int, int, int]) -> tuple[Any, Any]:
    text_rgba = _tooltip_text_color(bg_rgba)

    def _show_tooltip(text: str, screen_pos: tuple[int, int]) -> None:
        drawlist_tag = f"{tag}_drawlist"
        if not dpg.does_item_exist(drawlist_tag):
            with dpg.viewport_drawlist(tag=drawlist_tag, front=True):
                pass

        dpg.delete_item(drawlist_tag, children_only=True)
        lines = text.splitlines() or [text]
        max_chars = max(len(line) for line in lines) if lines else 0
        padding_x = 8
        padding_y = 6
        line_h = 17
        text_w = max(90, max_chars * 7)
        box_w = text_w + padding_x * 2
        box_h = max(24, len(lines) * line_h + padding_y * 2)
        x0, y0 = int(screen_pos[0]), int(screen_pos[1])
        x1, y1 = x0 + box_w, y0 + box_h

        dpg.draw_rectangle((x0, y0), (x1, y1), parent=drawlist_tag, fill=bg_rgba, color=bg_rgba, rounding=6, thickness=1.0)
        dpg.draw_text((x0 + padding_x, y0 + padding_y), text, parent=drawlist_tag, color=text_rgba, size=16)

    def _hide_tooltip() -> None:
        drawlist_tag = f"{tag}_drawlist"
        if dpg.does_item_exist(drawlist_tag):
            dpg.delete_item(drawlist_tag, children_only=True)

    return _show_tooltip, _hide_tooltip


def _sample_discrete_plot_color(state: dict[str, Any], ratio: float) -> tuple[int, int, int, int]:
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


def _discrete_plot_cycle_size(state: dict[str, Any]) -> int:
    active_defs = _get_active_discrete_colormap(state)
    return max(
        1,
        int(
            (state.get("plot_colormap_sizes", {}) or {}).get(
                state["colormap_discrete"],
                len(active_defs),
            )
        ),
    )


def _cycle_discrete_plot_color(state: dict[str, Any], index: int) -> tuple[int, int, int, int]:
    cycle_size = _discrete_plot_cycle_size(state)
    wrapped_idx = int(index) % cycle_size
    ratio = 0.5 if cycle_size == 1 else wrapped_idx / (cycle_size - 1)
    return _sample_discrete_plot_color(state, ratio)


def _make_scatter_theme(tag: str, color: tuple[int, int, int, int]) -> str:
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Fill, color, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 0.0, category=dpg.mvThemeCat_Plots)
    return tag


def _point_size_mode_to_marker_size(mode: str) -> float:
    mode = str(mode or "Medium").strip().lower()
    if mode == "small":
        return 2.0
    if mode == "large":
        return 5.0
    return 3.5


def _point_size_mode_to_inserted_marker_size(mode: str) -> float:
    mode = str(mode or "Medium").strip().lower()
    if mode == "small":
        return 8.0
    if mode == "large":
        return 12.0
    return 10.0


def _normalize_point_size_mode(mode: Any) -> str:
    mode_str = str(mode or "Medium").strip().lower()
    if mode_str == "small":
        return "Small"
    if mode_str == "large":
        return "Large"
    return "Medium"


def _save_point_size_setting(state: dict[str, Any], settings_key: str, mode: Any) -> str:
    normalized = _normalize_point_size_mode(mode)
    state[settings_key] = normalized
    settings_dict = state.get("settings")
    if isinstance(settings_dict, dict):
        settings_dict[settings_key] = normalized
        settings_file = state.get("settings_file")
        if settings_file:
            try:
                with open(settings_file, "w", encoding="utf-8") as f:
                    json.dump(settings_dict, f, indent=4, ensure_ascii=False)
            except Exception:
                pass
    return normalized


def _normalize_mcs_timeout_mode(value: Any) -> str:
    value_str = str(value or "10s").strip()
    allowed = {"10s", "30s", "60s", "120s", "300s", "Unlimited"}
    return value_str if value_str in allowed else "10s"


def _save_mcs_timeout_setting(state: dict[str, Any], settings_key: str, value: Any) -> str:
    normalized = _normalize_mcs_timeout_mode(value)
    state[settings_key] = normalized
    settings_dict = state.get("settings")
    if isinstance(settings_dict, dict):
        settings_dict[settings_key] = normalized
        settings_file = state.get("settings_file")
        if settings_file:
            try:
                with open(settings_file, "w", encoding="utf-8") as f:
                    json.dump(settings_dict, f, indent=4, ensure_ascii=False)
            except Exception:
                pass
    return normalized


def _normalize_mcs_features_mode(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"false", "0", "no", "off"}:
        return False
    return True


def _save_mcs_features_setting(state: dict[str, Any], settings_key: str, value: Any) -> bool:
    normalized = _normalize_mcs_features_mode(value)
    state[settings_key] = normalized
    settings_dict = state.get("settings")
    if isinstance(settings_dict, dict):
        settings_dict[settings_key] = normalized
        settings_file = state.get("settings_file")
        if settings_file:
            try:
                with open(settings_file, "w", encoding="utf-8") as f:
                    json.dump(settings_dict, f, indent=4, ensure_ascii=False)
            except Exception:
                pass
    return normalized


def _open_insert_smiles_popup(
    *,
    plot_tag: str,
    popup_tag: str,
    state: dict[str, Any],
    point_size_state_key: str,
    point_size_combo_tag: str,
    point_size_callback: Any,
    mcs_timeout_state_key: str,
    mcs_timeout_combo_tag: str,
    mcs_timeout_callback: Any,
    mcs_features_state_key: str,
    mcs_features_checkbox_tag: str,
    mcs_features_callback: Any,
    input_tag: str,
    draw_callback: Any,
    delete_callback: Any,
) -> None:
    if not dpg.is_item_hovered(plot_tag):
        return
    if dpg.does_item_exist(popup_tag):
        dpg.delete_item(popup_tag)
    mx, my = dpg.get_mouse_pos(local=False)
    with dpg.window(
        tag=popup_tag,
        no_title_bar=True,
        show=True,
        no_resize=True,
        no_scrollbar=True,
        no_collapse=True,
        autosize=True,
        pos=(int(mx) + 14, int(my) + 14),
    ):
        dpg.add_combo(
            tag=point_size_combo_tag,
            label="Point size",
            items=["Small", "Medium", "Large"],
            default_value=str(state.get(point_size_state_key, "Medium")),
            width=160,
            callback=lambda s, a: point_size_callback(a),
        )
        dpg.add_combo(
            tag=mcs_timeout_combo_tag,
            label="MCS Timeout",
            items=["10s", "30s", "60s", "120s", "300s", "Unlimited"],
            default_value=str(state.get(mcs_timeout_state_key, "10s")),
            width=160,
            callback=lambda s, a: mcs_timeout_callback(a),
        )
        dpg.add_checkbox(
            tag=mcs_features_checkbox_tag,
            label="Show MCS annotations",
            default_value=bool(state.get(mcs_features_state_key, True)),
            callback=lambda s, a: mcs_features_callback(a),
        )
        dpg.add_separator()
        dpg.add_input_text(tag=input_tag, width=340, hint="Paste a SMILES string")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Draw", callback=lambda: draw_callback())
            dpg.add_button(label="Delete all", callback=lambda: delete_callback())


def _dismiss_insert_smiles_popup_on_outside_click(popup_tag: str) -> None:
    if not dpg.does_item_exist(popup_tag):
        return
    try:
        px, py = dpg.get_mouse_pos(local=False)
        pmin = dpg.get_item_rect_min(popup_tag)
        pmax = dpg.get_item_rect_max(popup_tag)
    except Exception:
        return
    inside_popup = pmin[0] <= px <= pmax[0] and pmin[1] <= py <= pmax[1]
    if not inside_popup:
        dpg.delete_item(popup_tag)


def _selection_overlay_tag(prefix: str) -> str:
    return f"{prefix}_selection_overlay"


def _clear_selection_overlay(prefix: str) -> None:
    tag = _selection_overlay_tag(prefix)
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)


def _draw_selection_overlay(prefix: str, start_pos: tuple[float, float], end_pos: tuple[float, float]) -> None:
    tag = _selection_overlay_tag(prefix)
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    x0, y0 = start_pos
    x1, y1 = end_pos
    with dpg.viewport_drawlist(tag=tag, front=True):
        dpg.draw_rectangle(
            (x0, y0),
            (x1, y1),
            color=(255, 120, 40, 220),
            fill=(255, 170, 60, 40),
            thickness=1.5,
        )


def _clamp_screen_pos_to_plot(plot_tag: str, screen_pos: tuple[float, float]) -> tuple[float, float]:
    try:
        pmin = dpg.get_item_rect_min(plot_tag)
        pmax = dpg.get_item_rect_max(plot_tag)
        return (
            min(max(float(screen_pos[0]), float(pmin[0])), float(pmax[0])),
            min(max(float(screen_pos[1]), float(pmin[1])), float(pmax[1])),
        )
    except Exception:
        return float(screen_pos[0]), float(screen_pos[1])


def _clamp_plot_pos_to_axes(x_axis_tag: str, y_axis_tag: str, plot_pos: tuple[float, float] | list[float]) -> tuple[float, float] | None:
    if not isinstance(plot_pos, (tuple, list)) or len(plot_pos) < 2:
        return None
    try:
        x_limits = dpg.get_axis_limits(x_axis_tag)
        y_limits = dpg.get_axis_limits(y_axis_tag)
    except Exception:
        return float(plot_pos[0]), float(plot_pos[1])
    return (
        min(max(float(plot_pos[0]), float(x_limits[0])), float(x_limits[1])),
        min(max(float(plot_pos[1]), float(y_limits[0])), float(y_limits[1])),
    )


def _mcs_parse_smiles_cached(smiles: str) -> tuple[str, Chem.Mol] | None:
    smiles = str(smiles or "").strip()
    if not smiles:
        return None
    cache = getattr(_mcs_parse_smiles_cached, "_cache", None)
    if cache is None:
        cache = {}
        _mcs_parse_smiles_cached._cache = cache
    cached = cache.get(smiles)
    if cached is not None:
        return cached
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                )
            except Exception:
                pass
    if mol is None:
        return None
    try:
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        canonical = smiles
    cached_value = (canonical, mol)
    cache[smiles] = cached_value
    return cached_value


def _unique_mols_from_smiles_list(smiles_list: list[str]) -> tuple[list[str], list[Chem.Mol]]:
    unique_smiles: list[str] = []
    unique_mols: list[Chem.Mol] = []
    seen: set[str] = set()
    for smiles in smiles_list:
        parsed = _mcs_parse_smiles_cached(smiles)
        if parsed is None:
            continue
        canonical, mol = parsed
        if canonical in seen:
            continue
        seen.add(canonical)
        unique_smiles.append(canonical)
        unique_mols.append(Chem.Mol(mol))
    order = sorted(
        range(len(unique_mols)),
        key=lambda i: (
            int(unique_mols[i].GetNumHeavyAtoms()),
            int(unique_mols[i].GetNumBonds()),
            int(unique_mols[i].GetRingInfo().NumRings()) if unique_mols[i].GetRingInfo() is not None else 0,
            len(unique_smiles[i]),
        ),
    )
    unique_smiles = [unique_smiles[i] for i in order]
    unique_mols = [unique_mols[i] for i in order]
    return unique_smiles, unique_mols


def _largest_fragment(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return mol
    if not frags:
        return mol
    return max(frags, key=lambda frag: (frag.GetNumHeavyAtoms(), frag.GetNumAtoms()))


def _prune_incomplete_ring_mcs(mcs_mol: Chem.Mol | None, mols: list[Chem.Mol]) -> Chem.Mol | None:
    if mcs_mol is None or not mols or mcs_mol.GetNumAtoms() == 0:
        return mcs_mol

    matches: list[tuple[int, ...]] = []
    for mol in mols:
        match = mol.GetSubstructMatch(mcs_mol)
        if not match:
            return mcs_mol
        matches.append(tuple(int(idx) for idx in match))

    atoms_to_remove: list[int] = []
    for atom in mcs_mol.GetAtoms():
        atom_idx = int(atom.GetIdx())
        if atom.IsInRing():
            continue
        try:
            if all(mol.GetAtomWithIdx(match[atom_idx]).IsInRing() for mol, match in zip(mols, matches)):
                atoms_to_remove.append(atom_idx)
        except Exception:
            continue

    if not atoms_to_remove:
        return mcs_mol

    rw = Chem.RWMol(mcs_mol)
    for atom_idx in sorted(set(atoms_to_remove), reverse=True):
        try:
            rw.RemoveAtom(int(atom_idx))
        except Exception:
            continue

    pruned = rw.GetMol()
    try:
        Chem.SanitizeMol(pruned, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        pass
    return _largest_fragment(pruned)


def _mcs_result_from_smiles_list(
    smiles_list: list[str],
    timeout_setting: Any = "10s",
    include_features: bool = True,
) -> dict[str, Any]:
    unique_smiles, mols = _unique_mols_from_smiles_list(smiles_list)
    timeout_value = _normalize_mcs_timeout_mode(timeout_setting)
    result: dict[str, Any] = {
        "mol": None,
        "interrupted": False,
        "timeout_setting": timeout_value,
        "feature_annotations": {},
    }
    if not mols:
        return result
    if len(mols) == 1:
        result["mol"] = mols[0]
        return result
    params = rdFMCS.MCSParameters()
    timeout_seconds = None if timeout_value == "Unlimited" else int(timeout_value.rstrip("s"))
    if timeout_seconds is not None:
        try:
            params.Timeout = timeout_seconds
        except Exception:
            try:
                params.timeout = timeout_seconds
            except Exception:
                pass
    try:
        if hasattr(params, "AtomTyper"):
            params.AtomTyper = rdFMCS.AtomCompare.CompareElements
        if hasattr(params, "BondTyper"):
            params.BondTyper = rdFMCS.BondCompare.CompareOrder
        if hasattr(params, "AtomCompareParameters"):
            params.AtomCompareParameters.RingMatchesRingOnly = True
        if hasattr(params, "BondCompareParameters"):
            params.BondCompareParameters.RingMatchesRingOnly = True
        if hasattr(params, "Threshold"):
            params.Threshold = 1.0
        elif hasattr(params, "threshold"):
            params.threshold = 1.0
        mcs = rdFMCS.FindMCS(mols, parameters=params)
    except Exception:
        fallback_kwargs = dict(
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            threshold=1.0,
        )
        if timeout_seconds is not None:
            fallback_kwargs["timeout"] = timeout_seconds
        mcs = rdFMCS.FindMCS(mols, **fallback_kwargs)
    result["interrupted"] = bool(getattr(mcs, "canceled", False))
    smarts = str(getattr(mcs, "smartsString", "") or "").strip()
    if not smarts:
        return result
    mcs_mol_from_smarts = Chem.MolFromSmarts(smarts)
    if mcs_mol_from_smarts is None or mcs_mol_from_smarts.GetNumAtoms() == 0:
        return result
    mcs_mol = None
    try:
        mcs_smiles = Chem.MolToSmiles(mcs_mol_from_smarts, isomericSmiles=True, canonical=True)
        mcs_mol = Chem.MolFromSmiles(mcs_smiles, sanitize=True)
        if mcs_mol is None:
            mcs_mol = Chem.MolFromSmiles(mcs_smiles, sanitize=False)
            if mcs_mol is not None:
                try:
                    Chem.SanitizeMol(
                        mcs_mol,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                    )
                except Exception:
                    pass
    except Exception:
        mcs_mol = None
    result["mol"] = _prune_incomplete_ring_mcs(
        mcs_mol if mcs_mol is not None else Chem.Mol(mcs_mol_from_smarts),
        mols,
    )
    if include_features:
        result["feature_annotations"] = _mcs_feature_annotations_from_smiles_list(unique_smiles, result["mol"])
    return result


def _mcs_mol_from_smiles_list(
    smiles_list: list[str],
    timeout_setting: Any = "10s",
    include_features: bool = True,
) -> Chem.Mol | None:
    return _mcs_result_from_smiles_list(smiles_list, timeout_setting, include_features=include_features).get("mol")


def _mcs_feature_factory() -> Any:
    cache = getattr(_mcs_feature_factory, "_cache", None)
    if cache is not None:
        return cache
    try:
        factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
    except Exception:
        factory = None
    _mcs_feature_factory._cache = factory
    return factory


def _collect_non_mcs_fragment_atoms(mol: Chem.Mol, start_atom_idx: int, matched_atoms: set[int]) -> set[int]:
    stack = [int(start_atom_idx)]
    seen: set[int] = set()
    while stack:
        atom_idx = int(stack.pop())
        if atom_idx in seen or atom_idx in matched_atoms:
            continue
        seen.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = int(neighbor.GetIdx())
            if neighbor_idx not in seen and neighbor_idx not in matched_atoms:
                stack.append(neighbor_idx)
    return seen


def _direct_attachment_signature(mol: Chem.Mol, atom_idx: int, matched_atoms: set[int]) -> tuple[str, int, int]:
    atom = mol.GetAtomWithIdx(int(atom_idx))
    external_neighbors = [
        int(neighbor.GetIdx())
        for neighbor in atom.GetNeighbors()
        if int(neighbor.GetIdx()) not in matched_atoms
    ]
    if len(external_neighbors) > 1:
        fragment_sizes = [
            len(_collect_non_mcs_fragment_atoms(mol, neighbor_idx, matched_atoms))
            for neighbor_idx in external_neighbors
        ]
        return ("multi", len(external_neighbors), max(fragment_sizes) if fragment_sizes else 0)
    if len(external_neighbors) == 1:
        neighbor_atom = mol.GetAtomWithIdx(external_neighbors[0])
        fragment_size = len(_collect_non_mcs_fragment_atoms(mol, external_neighbors[0], matched_atoms))
        atom_code = int(neighbor_atom.GetAtomicNum())
        if neighbor_atom.GetIsAromatic():
            atom_code += 1000
        elif neighbor_atom.IsInRing():
            atom_code += 2000
        return (f"atom:{atom_code}", 1, fragment_size)
    try:
        direct_h_count = int(atom.GetTotalNumHs(includeNeighbors=False))
    except Exception:
        try:
            direct_h_count = int(atom.GetNumImplicitHs())
        except Exception:
            direct_h_count = 0
    if direct_h_count == 1:
        return ("H", 1, 1)
    return ("none", 0, 0)


def _fragment_feature_labels(mol: Chem.Mol, atom_indices: set[int]) -> set[str]:
    if not atom_indices:
        return set()
    labels: set[str] = set()
    atoms = [mol.GetAtomWithIdx(int(idx)) for idx in sorted(atom_indices)]

    if any(atom.GetAtomicNum() in {9, 17, 35, 53, 85} for atom in atoms):
        labels.add("X")
    if any(atom.IsInRing() for atom in atoms):
        labels.add("Ring")
    if any(atom.GetIsAromatic() for atom in atoms):
        labels.add("Ar")

    feature_factory = _mcs_feature_factory()
    if feature_factory is not None:
        family_map = {
            "Donor": "HBD",
            "Acceptor": "HBA",
            "PosIonizable": "Pos",
            "NegIonizable": "Neg",
            "Hydrophobe": "HPh",
            "LumpedHydrophobe": "HPh",
            "Aromatic": "Ar",
        }
        try:
            for feature in feature_factory.GetFeaturesForMol(mol):
                feature_label = family_map.get(str(feature.GetFamily() or ""))
                if feature_label and atom_indices.intersection(set(int(i) for i in feature.GetAtomIds())):
                    labels.add(feature_label)
        except Exception:
            pass
    if "Ar" in labels and "Ring" in labels:
        labels.discard("Ring")
    return labels


def _collect_common_mcs_feature_annotations(mols: list[Chem.Mol], mcs_mol: Chem.Mol | None) -> dict[int, list[str]]:
    if mcs_mol is None or len(mols) < 2:
        return {}

    matches: list[tuple[int, ...]] = []
    for mol in mols:
        match = mol.GetSubstructMatch(mcs_mol)
        if not match:
            return {}
        matches.append(tuple(int(idx) for idx in match))

    per_mol_annotations: list[dict[int, set[str]]] = []
    per_mol_direct_attachment_signatures: list[dict[int, tuple[str, int, int]]] = []
    for mol, match in zip(mols, matches):
        matched_atoms = set(match)
        mol_annotations: dict[int, set[str]] = {mcs_idx: set() for mcs_idx in range(len(match))}
        mol_attachment_signatures: dict[int, tuple[str, int, int]] = {mcs_idx: ("none", 0, 0) for mcs_idx in range(len(match))}
        for mcs_idx, atom_idx in enumerate(match):
            atom = mol.GetAtomWithIdx(atom_idx)
            mol_attachment_signatures[mcs_idx] = _direct_attachment_signature(mol, atom_idx, matched_atoms)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = int(neighbor.GetIdx())
                if neighbor_idx in matched_atoms:
                    continue
                fragment_atoms = _collect_non_mcs_fragment_atoms(mol, neighbor_idx, matched_atoms)
                mol_annotations[mcs_idx].update(_fragment_feature_labels(mol, fragment_atoms))
        per_mol_annotations.append(mol_annotations)
        per_mol_direct_attachment_signatures.append(mol_attachment_signatures)

    priority = {"X": 0, "Ar": 1, "Ring": 2, "HBA": 3, "HBD": 4, "Pos": 5, "Neg": 6, "HPh": 7, "A": 8}
    common_annotations: dict[int, list[str]] = {}
    pending_r_atoms: list[int] = []
    for mcs_idx in range(mcs_mol.GetNumAtoms()):
        label_sets = [ann.get(mcs_idx, set()) for ann in per_mol_annotations]
        if not label_sets or any(not labels for labels in label_sets):
            common = set()
        else:
            common = set.intersection(*label_sets)
        if not common:
            signatures = [sig_map.get(mcs_idx, ("none", 0, 0)) for sig_map in per_mol_direct_attachment_signatures]
            if not signatures or len(set(signatures)) == 1:
                continue
            if all(int(sig[1]) == 1 for sig in signatures):
                direct_atom_labels = {str(sig[0]) for sig in signatures}
                all_single_atom_variants = all(int(sig[2]) <= 1 for sig in signatures)
                if all_single_atom_variants and len(direct_atom_labels) > 1:
                    common = {"A"}
                else:
                    pending_r_atoms.append(mcs_idx)
                    continue
            elif any(int(sig[1]) > 1 for sig in signatures):
                pending_r_atoms.append(mcs_idx)
                continue
            else:
                common = {"A"}
        if "Ar" in common and "Ring" in common:
            common.discard("Ring")
        if not common:
            continue
        common_annotations[mcs_idx] = sorted(common, key=lambda label: (priority.get(label, 99), label))[:3]
    for mcs_idx in sorted(pending_r_atoms):
        common_annotations[mcs_idx] = ["R"]
    return common_annotations


def _mcs_feature_annotations_from_smiles_list(smiles_list: list[str], mcs_mol: Chem.Mol | None) -> dict[int, list[str]]:
    _, mols = _unique_mols_from_smiles_list(smiles_list)
    return _collect_common_mcs_feature_annotations(mols, mcs_mol)


def _format_mcs_selection_header(selected_count: int, interrupted: bool, timeout_setting: str) -> str:
    if not selected_count:
        return "MCS: no molecules selected"
    suffix = f" (interrupted at {timeout_setting})" if interrupted and timeout_setting != "Unlimited" else ""
    return f"MCS of {selected_count} molecules{suffix}"


def _summarize_selection_values(records: list[dict[str, Any]], key: str, label: str) -> str:
    values = sorted({str(r.get(key, "N/A")).strip() for r in records if str(r.get(key, "N/A")).strip()})
    visible = ", ".join(values[:4]) + (" ..." if len(values) > 4 else "")
    return f"{label}: {visible if visible else '-'}"


def _format_numeric_value(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value == 0:
        return "0"
    if abs_value >= 1000:
        return f"{float(value):,.0f}".replace(",", " ")
    if abs_value >= 100:
        return f"{float(value):.1f}"
    if abs_value >= 1:
        return f"{float(value):.2f}"
    if abs_value >= 0.01:
        return f"{float(value):.3f}"
    if abs_value >= 0.001:
        return f"{float(value):.4f}"
    if abs_value >= 0.0001:
        return f"{float(value):.5f}"
    return f"{float(value):.2f}"


def _average_activity_line(records: list[dict[str, Any]], activity_label: str, state: dict[str, Any]) -> str:
    if activity_label == "None":
        return "Average activity: -"
    numeric_values: list[float] = []
    for record in records:
        raw_value = record.get("activity_text", "N/A")
        try:
            numeric_values.append(float(raw_value))
        except Exception:
            continue
    if not numeric_values:
        return f"Average {activity_label} = -"
    display_label = activity_label
    averaged_value = float(np.mean(numeric_values))
    if activity_label.startswith("p") and activity_label[1:] in state["nM_activity_types"]:
        display_label = activity_label[1:]
        averaged_value = float(np.mean([10 ** (9.0 - value) for value in numeric_values]))
    unit = _descriptor_unit(display_label, state)
    unit_suffix = f" {unit}" if unit else ""
    return f"Average {display_label} = {_format_numeric_value(averaged_value)}{unit_suffix}"


def _load_feature_annotation_font(image_width: int) -> Any:
    font_size = max(12, int(round(image_width * 0.048)))
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _set_texture_from_mol(
    texture_tag: str,
    mol: Chem.Mol | None,
    image_width: int,
    transparent: bool = False,
    feature_annotations: dict[int, list[str]] | None = None,
) -> None:
    if not dpg.does_item_exist(texture_tag):
        return
    render_scale = 1.8
    width = int(round(image_width * render_scale))
    height = int(round(image_width * 0.75 * render_scale))
    if mol is None:
        dpg.set_value(texture_tag, _transparent_blank_texture(width, height) if transparent else _blank_texture(width, height))
        return
    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:
        pass
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    if feature_annotations:
        max_chars = max(
            (len("/".join(labels)) for labels in feature_annotations.values() if labels),
            default=0,
        )
        opts.padding = min(0.18, 0.065 + max_chars * 0.008)
    else:
        opts.padding = 0.025
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    annotation_positions: dict[int, tuple[float, float]] = {}
    if feature_annotations:
        for atom_idx in feature_annotations.keys():
            try:
                pt = drawer.GetDrawCoords(int(atom_idx))
                annotation_positions[int(atom_idx)] = (float(pt.x), float(pt.y))
            except Exception:
                continue
    drawer.FinishDrawing()
    img = pilImage.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    if feature_annotations and annotation_positions:
        aa_scale = 2
        overlay = pilImage.new("RGBA", (width * aa_scale, height * aa_scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = _load_feature_annotation_font(width * aa_scale)
        for ann_idx, atom_idx in enumerate(sorted(annotation_positions.keys())):
            labels = feature_annotations.get(atom_idx, [])
            if not labels:
                continue
            label_text = "/".join(labels)
            x_px, y_px = annotation_positions[atom_idx]
            x_px *= aa_scale
            y_px *= aa_scale

            anchor_atom = mol.GetAtomWithIdx(int(atom_idx))
            neighbor_points: list[tuple[float, float]] = []
            for neighbor in anchor_atom.GetNeighbors():
                neighbor_idx = int(neighbor.GetIdx())
                if neighbor_idx not in annotation_positions:
                    try:
                        npt = drawer.GetDrawCoords(neighbor_idx)
                        neighbor_points.append((float(npt.x) * aa_scale, float(npt.y) * aa_scale))
                    except Exception:
                        continue
                else:
                    npt = annotation_positions[neighbor_idx]
                    neighbor_points.append((npt[0] * aa_scale, npt[1] * aa_scale))

            if neighbor_points:
                avg_nx = float(np.mean([pt[0] for pt in neighbor_points]))
                avg_ny = float(np.mean([pt[1] for pt in neighbor_points]))
                vx = x_px - avg_nx
                vy = y_px - avg_ny
            else:
                angle = -0.85 + (ann_idx % 5) * 0.45
                vx = math.cos(angle)
                vy = math.sin(angle)

            norm = math.hypot(vx, vy)
            if norm <= 1e-6:
                vx, vy, norm = 0.0, -1.0, 1.0
            ux = vx / norm
            uy = vy / norm

            stub_len = max(24.0, width * aa_scale * 0.055)
            label_gap = max(10.0, width * aa_scale * 0.018)
            line_x1 = x_px + ux * stub_len
            line_y1 = y_px + uy * stub_len

            label_cx = line_x1 + ux * label_gap
            label_cy = line_y1 + uy * label_gap

            bbox = draw.textbbox(
                (label_cx, label_cy),
                label_text,
                font=font,
                anchor="mm",
                stroke_width=max(1, aa_scale),
            )
            if bbox:
                dx = 0.0
                dy = 0.0
                if bbox[0] < 4.0:
                    dx = 4.0 - bbox[0]
                elif bbox[2] > float(width * aa_scale - 4):
                    dx = float(width * aa_scale - 4) - bbox[2]
                if bbox[1] < 4.0:
                    dy = 4.0 - bbox[1]
                elif bbox[3] > float(height * aa_scale - 4):
                    dy = float(height * aa_scale - 4) - bbox[3]
                label_cx += dx
                label_cy += dy

            line_width = max(3, int(round(width * aa_scale * 0.0055)))
            draw.line(
                [(x_px, y_px), (line_x1, line_y1)],
                fill=(55, 55, 55, 200),
                width=line_width,
            )
            endpoint_r = max(2, int(round(line_width * 0.42)))
            draw.ellipse(
                [(line_x1 - endpoint_r, line_y1 - endpoint_r), (line_x1 + endpoint_r, line_y1 + endpoint_r)],
                fill=(55, 55, 55, 210),
            )
            draw.text(
                (label_cx, label_cy),
                label_text,
                font=font,
                anchor="mm",
                fill=(25, 25, 25, 255),
                stroke_width=max(1, aa_scale),
                stroke_fill=(255, 255, 255, 220),
            )
        overlay = overlay.resize((width, height), pilImage.Resampling.LANCZOS)
        img = pilImage.alpha_composite(img, overlay)
    arr = (np.array(img) / 255.0).astype(np.float32).flatten()
    dpg.set_value(texture_tag, arr)


def _ensure_discrete_colormap(tag: str, colors_rgba: list[tuple[int, int, int, int]]) -> str:
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    return dpg.add_colormap(colors_rgba, True, tag=tag, parent="colormap_registry")


def _visible_discrete_label_indices(total_labels: int, available_height: int, min_label_gap: int = 24) -> list[int]:
    if total_labels <= 1 or available_height <= 0:
        return [0]

    max_visible_labels = max(1, available_height // max(1, min_label_gap))
    raw_step = max(1, int(np.ceil(total_labels / max_visible_labels)))

    magnitude = 1
    while magnitude * 10 < raw_step:
        magnitude *= 10

    for factor in (1, 2, 5, 10):
        candidate = magnitude * factor
        if raw_step <= candidate:
            step = candidate
            break
    else:
        step = magnitude * 10

    indices = list(range(0, total_labels, step))
    return indices or [0]


def _blank_texture(width: int, height: int) -> np.ndarray:
    arr = np.ones((height, width, 4), dtype=np.float32)
    arr[..., 3] = 1.0
    return arr.flatten()


def _transparent_blank_texture(width: int, height: int) -> np.ndarray:
    return np.zeros((height * width * 4,), dtype=np.float32)


def _ensure_details_widgets(prefix: str, details_window_tag: str, image_width: int, state: dict[str, Any]) -> None:
    texture_tag = f"{prefix}_molecule_image_texture"
    img_h = round(image_width * 0.75)
    render_scale = 1.8
    render_w = int(round(image_width * render_scale))
    render_h = int(round(img_h * render_scale))
    if dpg.does_item_exist(details_window_tag) and not dpg.does_item_exist(texture_tag):
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                render_w,
                render_h,
                _blank_texture(render_w, render_h),
                tag=texture_tag,
            )

    group_tag = f"{prefix}_details_group"
    if dpg.does_item_exist(details_window_tag) and not dpg.does_item_exist(group_tag):
        with dpg.child_window(
            parent=details_window_tag,
            tag=group_tag,
            width=-1,
            height=-1,
            border=False,
            no_scrollbar=False,
            horizontal_scrollbar=False,
        ):
            widget_tag = f"{prefix}_molecule_image_widget"
            dpg.add_image(
                texture_tag,
                tag=widget_tag,
                width=image_width,
                height=img_h,
                border_color=(0, 0, 0, 0),
            )
            register_responsive_image(
                state,
                image_tag=widget_tag,
                parent_tag=details_window_tag,
                aspect_ratio=0.75,
                tab=f"{prefix}_tab",
            )
            dpg.add_text("", tag=f"{prefix}_details_name_text", wrap=600)
            dpg.add_text("Subset: -", tag=f"{prefix}_details_subset_text", wrap=600)
            dpg.add_text("Activity: -", tag=f"{prefix}_details_activity_text", wrap=600)
            dpg.add_text("Coordinates: -", tag=f"{prefix}_details_coords_text", wrap=600)


def update_reducer_details(
    prefix: str,
    record: dict[str, Any] | None,
    details_window_tag: str,
    image_width: int,
    activity_label: str,
    state: dict[str, Any],
) -> None:
    _ensure_details_widgets(prefix, details_window_tag, image_width, state)
    texture_tag = f"{prefix}_molecule_image_texture"

    if record is None:
        if dpg.does_item_exist(texture_tag):
            if prefix in {"umap", "tsne"}:
                render_scale = 1.8
                render_w = int(round(image_width * render_scale))
                render_h = int(round(round(image_width * 0.75) * render_scale))
                dpg.set_value(texture_tag, _transparent_blank_texture(render_w, render_h))
            else:
                render_scale = 1.8
                render_w = int(round(image_width * render_scale))
                render_h = int(round(round(image_width * 0.75) * render_scale))
                dpg.set_value(texture_tag, _blank_texture(render_w, render_h))
        for tag, text in [
            (f"{prefix}_details_name_text", "Name: -"),
            (f"{prefix}_details_subset_text", "Subset: -"),
            (f"{prefix}_details_activity_text", "Activity: -"),
            (f"{prefix}_details_coords_text", "Coordinates: -"),
        ]:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, text)
        return

    mol = Chem.MolFromSmiles(record.get("smiles", ""))
    if mol is not None and dpg.does_item_exist(texture_tag):
        rdDepictor.Compute2DCoords(mol)
        render_scale = 1.8
        width = int(round(image_width * render_scale))
        height = int(round(round(image_width * 0.75) * render_scale))
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.padding = 0.025
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        img = pilImage.open(io.BytesIO(png_bytes)).convert("RGBA")
        arr = (np.array(img) / 255.0).astype(np.float32).flatten()
        dpg.set_value(texture_tag, arr)

    try:
        activity_text = f"{float(record.get('activity_text', 'nan')):.2f}"
    except Exception:
        activity_text = str(record.get("activity_text", "N/A"))

    dpg.set_value(f"{prefix}_details_name_text", f"Mol {record.get('mol_id', 'N/A')}  |  Name: {record.get('name', 'N/A')}")
    dpg.set_value(f"{prefix}_details_subset_text", f"Subset: {record.get('subset_text', 'N/A')}")
    dpg.set_value(
        f"{prefix}_details_activity_text",
        f"Activity: {activity_label} = {activity_text}" if activity_label != "None" and record.get("activity_text") not in (None, "", "N/A") else "Activity: N/A",
    )
    dpg.set_value(
        f"{prefix}_details_coords_text",
        f"Coordinates: ({record.get('x', 0.0):.3f}, {record.get('y', 0.0):.3f})",
    )


def _update_reducer_details_mcs(
    prefix: str,
    records: list[dict[str, Any]],
    details_window_tag: str,
    image_width: int,
    activity_label: str,
    state: dict[str, Any],
) -> None:
    _ensure_details_widgets(prefix, details_window_tag, image_width, state)
    texture_tag = f"{prefix}_molecule_image_texture"
    transparent = prefix in {"umap", "tsne"}
    if not records:
        _set_texture_from_mol(texture_tag, None, image_width, transparent=transparent)
        if dpg.does_item_exist(f"{prefix}_details_name_text"):
            dpg.set_value(f"{prefix}_details_name_text", "MCS: no molecules selected")
        if dpg.does_item_exist(f"{prefix}_details_subset_text"):
            dpg.set_value(f"{prefix}_details_subset_text", "Subset: -")
        if dpg.does_item_exist(f"{prefix}_details_activity_text"):
            dpg.set_value(f"{prefix}_details_activity_text", "Activity: -")
        if dpg.does_item_exist(f"{prefix}_details_coords_text"):
            dpg.set_value(f"{prefix}_details_coords_text", "Coordinates: -")
        return

    show_features = bool(state.get(f"{prefix}_mcs_features", True))
    mcs_result = _mcs_result_from_smiles_list(
        [str(r.get("smiles", "")) for r in records],
        state.get(f"{prefix}_mcs_timeout", "10s"),
        include_features=show_features,
    )
    mcs_mol = mcs_result.get("mol")
    feature_annotations = dict(mcs_result.get("feature_annotations", {}) or {})
    _set_texture_from_mol(
        texture_tag,
        mcs_mol,
        image_width,
        transparent=transparent,
        feature_annotations=feature_annotations,
    )
    atom_count = mcs_mol.GetNumAtoms() if mcs_mol is not None else 0
    if dpg.does_item_exist(f"{prefix}_details_name_text"):
        dpg.set_value(
            f"{prefix}_details_name_text",
            _format_mcs_selection_header(
                len(records),
                bool(mcs_result.get("interrupted")),
                str(mcs_result.get("timeout_setting", "10s")),
            ),
        )
    if dpg.does_item_exist(f"{prefix}_details_subset_text"):
        dpg.set_value(
            f"{prefix}_details_subset_text",
            _summarize_selection_values(records, "subset_text", "Subset")
            + "\n"
            + _summarize_selection_values(records, "cluster_text", "Cluster"),
        )
    if dpg.does_item_exist(f"{prefix}_details_activity_text"):
        if atom_count:
            feature_count = int(sum(len(v) for v in feature_annotations.values()))
            suffix = f"  |  Feature labels: {feature_count}" if show_features and feature_count else ""
            dpg.set_value(f"{prefix}_details_activity_text", f"MCS atoms: {atom_count}{suffix}")
        else:
            dpg.set_value(f"{prefix}_details_activity_text", "MCS not found")
    if dpg.does_item_exist(f"{prefix}_details_coords_text"):
        dpg.set_value(f"{prefix}_details_coords_text", _average_activity_line(records, activity_label, state))


def _project_reducer_inserted_smiles(prefix: str, smiles: str, state: dict[str, Any]) -> tuple[float, float] | None:
    from app.analysis.chemspace.chemspace_logic_common import build_fingerprint_array

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None

    fp_algorithm = state.get(f"{prefix}_projection_fp_algorithm")
    if not fp_algorithm:
        return None

    fp_array = build_fingerprint_array(mol, fp_algorithm)
    if fp_array is None:
        return None

    reducer = state.get(f"{prefix}_projection_model")
    if prefix == "umap" and reducer is not None and hasattr(reducer, "transform"):
        try:
            coords = np.asarray(reducer.transform(np.asarray(fp_array, dtype=float).reshape(1, -1))[0], dtype=float)
            if coords.size >= 2 and np.all(np.isfinite(coords[:2])):
                return float(coords[0]), float(coords[1])
        except Exception:
            pass

    train_fps = np.asarray(state.get(f"{prefix}_projection_fingerprints", []), dtype=float)
    train_embedding = np.asarray(state.get(f"{prefix}_embedding", []), dtype=float)
    if train_fps.ndim != 2 or train_embedding.ndim != 2 or len(train_fps) == 0 or len(train_fps) != len(train_embedding):
        return None

    query = np.asarray(fp_array, dtype=float)
    intersections = np.minimum(train_fps, query).sum(axis=1)
    unions = np.maximum(train_fps, query).sum(axis=1)
    similarities = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)

    k = min(8, len(train_fps))
    top_idx = np.argsort(-similarities)[:k]
    if len(top_idx) == 0:
        return None

    weights = np.asarray(similarities[top_idx], dtype=float)
    if not np.any(weights > 0):
        distances = np.linalg.norm(train_fps[top_idx] - query, axis=1)
        weights = 1.0 / np.maximum(distances, 1e-9)
    weights = np.asarray(weights, dtype=float)
    if not np.any(np.isfinite(weights)) or float(np.sum(weights)) <= 0:
        weights = np.ones(len(top_idx), dtype=float)
    weights = weights / np.sum(weights)

    coords = np.average(train_embedding[top_idx, :2], axis=0, weights=weights)
    if coords.size < 2 or not np.all(np.isfinite(coords[:2])):
        return None
    return float(coords[0]), float(coords[1])


def draw_reducer_plot_2d(
    *,
    prefix: str,
    log_section: str,
    subset: str,
    embedding: np.ndarray,
    molecule_data: list[dict[str, Any]],
    activity_label: str,
    activity_values: list[float],
    cluster_labels: np.ndarray,
    fp_algorithm: str,
    plot_window_tag: str,
    details_window_tag: str,
    title: str,
    x_label: str,
    y_label: str,
    state: dict[str, Any],
) -> None:
    for tag in [
        f"{prefix}_plot",
        f"{prefix}_gradient_bar_window",
        f"{prefix}_plot_handler_registry",
    ]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    dpg.delete_item(plot_window_tag, children_only=True)
    dpg.delete_item(details_window_tag, children_only=True)

    image_width = int(state["plots_pca_img_width"])
    _ensure_details_widgets(prefix, details_window_tag, image_width, state)
    cluster_labels = np.asarray(cluster_labels if cluster_labels is not None else np.zeros(len(molecule_data), dtype=int), dtype=int)

    xs = embedding[:, 0].astype(float).tolist()
    ys = embedding[:, 1].astype(float).tolist()
    act_min = float(min(activity_values)) if activity_values else 0.0
    act_max = float(max(activity_values)) if activity_values else 1.0
    if act_max <= act_min:
        act_max = act_min + 1e-9
    state[f"{prefix}_plot_points"] = []
    for idx, mol_info in enumerate(molecule_data):
        state[f"{prefix}_plot_points"].append(
            {
                "x": xs[idx],
                "y": ys[idx],
                "smiles": mol_info.get("SMILES", ""),
                "name": mol_info.get("Name", "N/A"),
                "mol_id": mol_info.get("Mol_ID", idx + 1),
                "subset_text": str(mol_info.get("Subset", "N/A")).replace("subset_", "Subset "),
                "activity_text": mol_info.get(activity_label, "N/A") if activity_label != "None" else "N/A",
                "cluster_text": int(cluster_labels[idx]) + 1 if len(cluster_labels) > idx else "N/A",
            }
        )

    inserted_context_key = (
        subset,
        activity_label,
        fp_algorithm,
        int(embedding.shape[1]) if getattr(embedding, "ndim", 0) >= 2 else 2,
    )
    if state.get(f"{prefix}_inserted_points_context") != inserted_context_key:
        state[f"{prefix}_inserted_points"] = []
        state[f"{prefix}_inserted_points_context"] = inserted_context_key

    current_coloring = activity_label if activity_label != "None" and activity_values else "subset"
    state[f"{prefix}_current_coloring"] = current_coloring

    def _color_rgba_from_colormap(colormap_tag: Any, ratio: float) -> tuple[int, int, int, int]:
        rgba = dpg.sample_colormap(colormap_tag, max(0.0, min(1.0, float(ratio))))
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

    def _build_cluster_color_map() -> tuple[list[int], dict[int, tuple[int, int, int, int]]]:
        unique_clusters = sorted(set(int(c) for c in cluster_labels)) or [0]
        color_map = {
            cid: _cycle_discrete_plot_color(state, idx)
            for idx, cid in enumerate(unique_clusters)
        }
        return unique_clusters, color_map

    def _build_subset_color_map() -> tuple[list[int], dict[int, tuple[int, int, int, int]]]:
        subset_ids = sorted({int(m.get("Subset", -1)) for m in molecule_data if int(m.get("Subset", -1)) >= 0}) or [-1]
        color_map = {
            sid: _cycle_discrete_plot_color(state, idx)
            for idx, sid in enumerate(subset_ids)
        }
        return subset_ids, color_map

    def _cluster_color(cluster_id: int) -> tuple[int, int, int, int]:
        _, color_map = _build_cluster_color_map()
        return color_map.get(cluster_id, _cycle_discrete_plot_color(state, 0))

    def _subset_color(subset_id: int) -> tuple[int, int, int, int]:
        _, color_map = _build_subset_color_map()
        return color_map.get(subset_id, _cycle_discrete_plot_color(state, 0))

    def _recreate_color_scale(mode: str) -> None:
        if dpg.does_item_exist(f"{prefix}_gradient_bar_window"):
            dpg.delete_item(f"{prefix}_gradient_bar_window", children_only=True)
        else:
            return

        text_color = state["themes"][state["theme_name"]]["Text Color"]
        border_color = state["themes"][state["theme_name"]]["Border Color"]

        def _draw_discrete_scale(
            *,
            label: str,
            labels: list[str],
            colors: list[tuple[int, int, int, int]],
            labels_drawlist_tag: str,
            host_tag: str,
            bar_drawlist_tag: str,
        ) -> None:
            dpg.add_text(label)
            with dpg.group(horizontal=True):
                label_w = max(34, int(state.get("win_spacer", 8) * 4))
                bar_w = max(20, int(state.get("win_spacer", 8) * 2))
                with dpg.drawlist(width=label_w, height=1, tag=labels_drawlist_tag):
                    pass
                with dpg.child_window(
                    tag=host_tag,
                    width=bar_w,
                    height=-1,
                    border=False,
                    no_scrollbar=True,
                    no_scroll_with_mouse=True,
                ):
                    with dpg.drawlist(width=bar_w, height=1, tag=bar_drawlist_tag):
                        pass

            def _relayout() -> None:
                if not (
                    dpg.does_item_exist(host_tag)
                    and dpg.does_item_exist(labels_drawlist_tag)
                    and dpg.does_item_exist(bar_drawlist_tag)
                ):
                    return

                try:
                    _, bar_h = dpg.get_item_rect_size(host_tag)
                except Exception:
                    return

                draw_h = int(bar_h or 0)
                if draw_h <= 1:
                    return

                label_w = max(34, int(state.get("win_spacer", 8) * 4))
                bar_w = max(20, int(state.get("win_spacer", 8) * 2))
                n_labels = max(1, len(labels))
                step_h = draw_h / n_labels
                visible_indices = set(_visible_discrete_label_indices(n_labels, draw_h))

                dpg.configure_item(labels_drawlist_tag, width=label_w, height=draw_h)
                dpg.configure_item(bar_drawlist_tag, width=bar_w, height=draw_h)
                dpg.delete_item(labels_drawlist_tag, children_only=True)
                dpg.delete_item(bar_drawlist_tag, children_only=True)

                for idx in range(n_labels):
                    seg_y0 = draw_h - ((idx + 1) * step_h)
                    seg_y1 = draw_h - (idx * step_h)
                    color = colors[idx]
                    dpg.draw_rectangle(
                        (0, seg_y0),
                        (bar_w, seg_y1),
                        fill=color,
                        color=color,
                        thickness=1.0,
                        parent=bar_drawlist_tag,
                    )

                dpg.draw_rectangle(
                    (0, 0),
                    (bar_w, draw_h),
                    fill=(0, 0, 0, 0),
                    color=border_color,
                    thickness=1.0,
                    parent=bar_drawlist_tag,
                )

                for idx, lbl in enumerate(labels):
                    if idx not in visible_indices:
                        continue
                    center_y = draw_h - ((idx + 0.5) * step_h)
                    text_y = center_y - 8
                    dpg.draw_text(
                        (2, text_y),
                        str(lbl),
                        color=text_color,
                        size=16,
                        parent=labels_drawlist_tag,
                    )
                    dpg.draw_line(
                        (0, center_y),
                        (bar_w / 2, center_y),
                        color=text_color,
                        thickness=1.0,
                        parent=bar_drawlist_tag,
                    )

            state[f"{prefix}_relayout_colormap_scale"] = _relayout
            dpg.set_frame_callback(dpg.get_frame_count() + 1, _relayout)

        with dpg.group(parent=f"{prefix}_gradient_bar_window"):
            dpg.add_colormap_button(
                tag=f"{prefix}_activity_colormap_button",
                label="Activity",
                width=-1,
                indent=3,
                callback=lambda: _set_color_mode(activity_label if activity_label != "None" and activity_values else "subset"),
            )
            dpg.add_colormap_button(
                tag=f"{prefix}_subset_colormap_button",
                label="Subsets",
                width=-1,
                indent=3,
                callback=lambda: _set_color_mode("subset"),
            )
            dpg.add_colormap_button(
                tag=f"{prefix}_cluster_colormap_button",
                label="Clusters",
                width=-1,
                indent=3,
                callback=lambda: _set_color_mode("cluster"),
            )

            unique_clusters, cluster_color_map = _build_cluster_color_map()
            cluster_colors = [cluster_color_map[cluster_id] for cluster_id in unique_clusters] or [(160, 160, 160, 255)]
            cluster_cm_tag = _ensure_discrete_colormap(f"{prefix}_colormap_clusters", cluster_colors)
            dpg.bind_colormap(f"{prefix}_cluster_colormap_button", cluster_cm_tag)

            subset_ids, subset_color_map = _build_subset_color_map()
            subset_colors = [subset_color_map[subset_id] for subset_id in subset_ids] or [(160, 160, 160, 255)]
            subset_cm_tag = _ensure_discrete_colormap(f"{prefix}_colormap_subsets", subset_colors)
            dpg.bind_colormap(f"{prefix}_subset_colormap_button", subset_cm_tag)
            dpg.bind_colormap(f"{prefix}_activity_colormap_button", state["colormaps"][state["colormap_continuous"]])

            if mode == "cluster":
                labels = [str(int(c) + 1) for c in unique_clusters] or ["1"]
                _draw_discrete_scale(
                    label="Clusters",
                    labels=labels,
                    colors=cluster_colors,
                    labels_drawlist_tag=f"{prefix}_cluster_labels_drawlist",
                    host_tag=f"{prefix}_cluster_scale_host",
                    bar_drawlist_tag=f"{prefix}_cluster_bar_drawlist",
                )
            elif mode == "subset":
                subset_labels = [str(int(sid)) for sid in subset_ids] or ["1"]
                _draw_discrete_scale(
                    label="Subsets",
                    labels=subset_labels,
                    colors=subset_colors,
                    labels_drawlist_tag=f"{prefix}_subset_labels_drawlist",
                    host_tag=f"{prefix}_subset_scale_host",
                    bar_drawlist_tag=f"{prefix}_subset_bar_drawlist",
                )
            else:
                dpg.add_colormap_scale(
                    tag=f"{prefix}_colormap_scale",
                    label=activity_label,
                    colormap=state["colormaps"][state["colormap_continuous"]],
                    min_scale=act_min,
                    max_scale=act_max,
                    height=-1,
                    mirror=True,
                    format="%.2f",
                )
                dpg.bind_item_theme(f"{prefix}_colormap_scale", apply_colormap_theme(state))
                state[f"{prefix}_relayout_colormap_scale"] = lambda: None

    created_series: list[str] = []
    inserted_series: list[str] = []
    theme_cache: dict[Any, Any] = {}

    def _clear_series() -> None:
        nonlocal created_series
        for series_tag in created_series:
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.delete_item(series_tag)
                except Exception:
                    pass
        created_series = []

    def _clear_inserted_series() -> None:
        nonlocal inserted_series
        for series_tag in inserted_series:
            if dpg.does_item_exist(series_tag):
                try:
                    dpg.delete_item(series_tag)
                except Exception:
                    pass
        inserted_series = []

    def _set_point_size_mode(app_data: Any) -> None:
        settings_key = f"{prefix}_point_size" if prefix in {"umap", "tsne"} else f"{prefix}_point_size_mode"
        state[f"{prefix}_point_size_mode"] = _save_point_size_setting(state, settings_key, app_data)
        state[f"{prefix}_refresh_colors"]()

    def _get_series_theme(fill_rgba: tuple[int, int, int, int]) -> Any:
        marker_size = _point_size_mode_to_marker_size(state.get(f"{prefix}_point_size_mode", "Medium"))
        key = (tuple(fill_rgba), marker_size)
        th = theme_cache.get(key)
        if th is not None and dpg.does_item_exist(th):
            return th
        with dpg.theme() as th:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, fill_rgba, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, fill_rgba, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, marker_size, category=dpg.mvThemeCat_Plots)
        theme_cache[key] = th
        return th

    def _get_inserted_point_color() -> tuple[int, int, int, int]:
        return _sample_discrete_plot_color(state, 0.0)

    def _get_inserted_theme() -> Any:
        color = _get_inserted_point_color()
        marker_size = _point_size_mode_to_inserted_marker_size(state.get(f"{prefix}_point_size_mode", "Medium"))
        key = ("inserted_core", tuple(color), marker_size)
        th = theme_cache.get(key)
        if th is not None and dpg.does_item_exist(th):
            return th
        with dpg.theme() as th:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Cross, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, marker_size, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerWeight, 5.0, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots)
        theme_cache[key] = th
        return th

    def _draw_inserted_points() -> None:
        _clear_inserted_series()
        points = list(state.get(f"{prefix}_inserted_points", []))
        if not points:
            return
        inserted_sid = dpg.add_scatter_series(
            [float(p["x"]) for p in points],
            [float(p["y"]) for p in points],
            parent=f"{prefix}_y_axis",
            label="",
        )
        dpg.bind_item_theme(inserted_sid, _get_inserted_theme())
        inserted_series.append(inserted_sid)

    def _render_series(mode: str) -> None:
        _clear_series()
        if mode == "cluster":
            grouped: dict[int, tuple[list[float], list[float]]] = {}
            for idx, cluster_id in enumerate(cluster_labels):
                grouped.setdefault(int(cluster_id), ([], []))
                grouped[int(cluster_id)][0].append(xs[idx])
                grouped[int(cluster_id)][1].append(ys[idx])
            for cluster_id, (gx, gy) in grouped.items():
                series_tag = f"{prefix}_series_cluster_{cluster_id}"
                dpg.add_scatter_series(gx, gy, parent=f"{prefix}_y_axis", tag=series_tag, label=f"Cluster {cluster_id + 1}")
                dpg.bind_item_theme(series_tag, _get_series_theme(_cluster_color(cluster_id)))
                created_series.append(series_tag)
        elif mode == "subset":
            grouped: dict[int, tuple[list[float], list[float]]] = {}
            for idx, mol_info in enumerate(molecule_data):
                subset_id = int(mol_info.get("Subset", -1))
                grouped.setdefault(subset_id, ([], []))
                grouped[subset_id][0].append(xs[idx])
                grouped[subset_id][1].append(ys[idx])
            for subset_id, (gx, gy) in grouped.items():
                series_tag = f"{prefix}_series_subset_{subset_id}"
                dpg.add_scatter_series(gx, gy, parent=f"{prefix}_y_axis", tag=series_tag, label=f"Subset {subset_id}")
                dpg.bind_item_theme(series_tag, _get_series_theme(_subset_color(subset_id)))
                created_series.append(series_tag)
        else:
            values = np.asarray(activity_values, dtype=float)
            bucket_count = 24
            bucket_x: list[list[float]] = [[] for _ in range(bucket_count)]
            bucket_y: list[list[float]] = [[] for _ in range(bucket_count)]
            for x, y, value in zip(xs, ys, values):
                norm = (float(value) - act_min) / (act_max - act_min)
                bucket_idx = min(bucket_count - 1, max(0, int(norm * (bucket_count - 1))))
                bucket_x[bucket_idx].append(float(x))
                bucket_y[bucket_idx].append(float(y))
            for bucket_idx in range(bucket_count):
                if not bucket_x[bucket_idx]:
                    continue
                ratio = bucket_idx / max(1, bucket_count - 1)
                color = _color_rgba_from_colormap(state["colormaps"][state["colormap_continuous"]], ratio)
                series_tag = f"{prefix}_series_activity_{bucket_idx}"
                dpg.add_scatter_series(bucket_x[bucket_idx], bucket_y[bucket_idx], parent=f"{prefix}_y_axis", tag=series_tag, label=f"bin_{bucket_idx}")
                dpg.bind_item_theme(series_tag, _get_series_theme(color))
                created_series.append(series_tag)
        dpg.fit_axis_data(f"{prefix}_x_axis")
        dpg.fit_axis_data(f"{prefix}_y_axis")
        _draw_inserted_points()

    def _set_color_mode(mode: str) -> None:
        nonlocal current_coloring
        current_coloring = mode
        state[f"{prefix}_current_coloring"] = mode
        _recreate_color_scale(mode)
        _render_series(mode)

    with dpg.child_window(
        parent=plot_window_tag,
        width=-1,
        height=-1,
        no_scrollbar=False,
        horizontal_scrollbar=False,
        no_scroll_with_mouse=True,
        border=False,
    ):
        with dpg.group(horizontal=True, tag=f"{prefix}_plot_group"):
            with dpg.child_window(
                border=False,
                tag=f"{prefix}_gradient_bar_window",
                no_scrollbar=True,
                horizontal_scrollbar=False,
                no_scroll_with_mouse=True,
                width=max(110, int(state.get("win_spacer", 8) * 10)),
                height=-1,
            ):
                pass

            with dpg.plot(
                label=title,
                tag=f"{prefix}_plot",
                width=-1,
                height=-1,
                equal_aspects=False,
                no_menus=True,
                no_box_select=False,
                no_mouse_pos=True,
                zoom_rate=0.05,
            ):
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label=x_label,
                    tag=f"{prefix}_x_axis",
                    no_tick_marks=True,
                    no_tick_labels=True,
                    no_highlight=True,
                )
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    label=y_label,
                    tag=f"{prefix}_y_axis",
                    no_tick_marks=True,
                    no_tick_labels=True,
                    no_highlight=True,
                )

                x_min, x_max = float(np.nanmin(embedding[:, 0])), float(np.nanmax(embedding[:, 0]))
                y_min, y_max = float(np.nanmin(embedding[:, 1])), float(np.nanmax(embedding[:, 1]))
                if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                    x_min, x_max = -1.0, 1.0
                if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                    y_min, y_max = -1.0, 1.0
                margin_x = 0.05 * (x_max - x_min)
                margin_y = 0.05 * (y_max - y_min)
                x_low, x_high = x_min - margin_x, x_max + margin_x
                y_low, y_high = y_min - margin_y, y_max + margin_y
                dpg.add_line_series(
                    x=[x_low, x_high, x_high, x_low, x_low],
                    y=[y_low, y_low, y_high, y_high, y_low],
                    parent=f"{prefix}_y_axis",
                    tag=f"{prefix}_frame_outline",
                    label="Bounds",
                )

                with dpg.theme() as reducer_frame_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 0), category=dpg.mvThemeCat_Plots)

                dpg.bind_item_theme(f"{prefix}_plot", apply_plot_theme(state))
                dpg.bind_item_theme(f"{prefix}_frame_outline", reducer_frame_theme)
        _set_color_mode(current_coloring)

    def _point_color(point_idx: int) -> tuple[int, int, int, int]:
        if current_coloring == "cluster":
            return _cluster_color(int(cluster_labels[point_idx]) if len(cluster_labels) > point_idx else 0)
        if current_coloring == "subset":
            return _subset_color(int(molecule_data[point_idx].get("Subset", -1)))
        value = float(activity_values[point_idx]) if len(activity_values) > point_idx else act_min
        ratio = max(0.0, min(1.0, (value - act_min) / (act_max - act_min)))
        return _color_rgba_from_colormap(state["colormaps"][state["colormap_continuous"]], ratio)

    def _update_tooltip() -> None:
        if state.get("current_chemspace_subtab") != f"{prefix}_tab":
            hide_tooltip()
            return
        if not dpg.does_item_exist(f"{prefix}_plot") or not dpg.is_item_hovered(f"{prefix}_plot"):
            hide_tooltip()
            return

        mouse_pos = dpg.get_plot_mouse_pos()
        plot_w_px, plot_h_px = dpg.get_item_rect_size(f"{prefix}_plot")
        x_lower, x_upper = dpg.get_axis_limits(f"{prefix}_x_axis")
        y_lower, y_upper = dpg.get_axis_limits(f"{prefix}_y_axis")
        x_range = max(1e-12, x_upper - x_lower)
        y_range = max(1e-12, y_upper - y_lower)
        radius_px = 5
        tol_x = radius_px * (x_range / max(1, plot_w_px))
        tol_y = radius_px * (y_range / max(1, plot_h_px))

        nearest_idx = None
        nearest_inserted_idx = None
        best_d2 = 1e300
        for idx, point in enumerate(state.get(f"{prefix}_plot_points", [])):
            dx = float(point["x"]) - mouse_pos[0]
            dy = float(point["y"]) - mouse_pos[1]
            if abs(dx) < tol_x and abs(dy) < tol_y:
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    nearest_idx = idx

        for idx, point in enumerate(state.get(f"{prefix}_inserted_points", [])):
            dx = float(point["x"]) - mouse_pos[0]
            dy = float(point["y"]) - mouse_pos[1]
            if abs(dx) < tol_x and abs(dy) < tol_y:
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    nearest_idx = None
                    nearest_inserted_idx = idx

        if nearest_inserted_idx is not None:
            point = state[f"{prefix}_inserted_points"][nearest_inserted_idx]
            tooltip_text = (
                f"Inserted mol {point['inserted_id']}\n"
                f"{x_label}: {float(point['x']):.2f}\n"
                f"{y_label}: {float(point['y']):.2f}"
            )
            show_local, _hide_local = _build_tooltip_theme(f"tooltip_{prefix}", _get_inserted_point_color())
            mx, my = dpg.get_mouse_pos(local=False)
            show_local(tooltip_text, (int(mx) + 14, int(my) + 14))
            return

        if nearest_idx is None:
            hide_tooltip()
            return

        point = state[f"{prefix}_plot_points"][nearest_idx]
        try:
            point_activity = f"{float(point.get('activity_text', 'nan')):.2f}"
        except Exception:
            point_activity = str(point.get("activity_text", "N/A"))
        tooltip_text = (
            f"Mol_ID: {point.get('mol_id', 'N/A')}\n"
            f"Name: {point.get('name', 'N/A')}\n"
            f"{activity_label if activity_label != 'None' else 'Activity'}: {point_activity}\n"
            f"{x_label}: {float(point['x']):.2f}\n"
            f"{y_label}: {float(point['y']):.2f}\n"
            f"Subset: {point.get('subset_text', 'N/A')}\n"
            f"Cluster: {point.get('cluster_text', 'N/A')}"
        )
        show_local, _hide_local = _build_tooltip_theme(f"tooltip_{prefix}", _point_color(nearest_idx))
        mx, my = dpg.get_mouse_pos(local=False)
        show_local(tooltip_text, (int(mx) + 14, int(my) + 14))

    show_tooltip, hide_tooltip = _build_tooltip_theme(f"tooltip_{prefix}", (30, 30, 30, 240))

    def _on_click() -> None:
        if state.get(f"{prefix}_mcs_selection_just_finished"):
            state[f"{prefix}_mcs_selection_just_finished"] = False
            return
        current_subtab = state.get("current_chemspace_subtab")
        expected_subtab = f"{prefix}_tab"
        if current_subtab != expected_subtab:
            return
        if not dpg.does_item_exist(f"{prefix}_plot") or not dpg.is_item_hovered(f"{prefix}_plot"):
            return
        mouse_pos = dpg.get_plot_mouse_pos()
        if not mouse_pos:
            return
        mx, my = float(mouse_pos[0]), float(mouse_pos[1])
        points = list(state.get(f"{prefix}_plot_points", []) or [])
        if not points:
            return
        xs_local = [float(p["x"]) for p in points]
        ys_local = [float(p["y"]) for p in points]
        span_x = max(1e-9, max(xs_local) - min(xs_local))
        span_y = max(1e-9, max(ys_local) - min(ys_local))
        best_idx = None
        best_inserted_idx = None
        best_dist = None
        for idx, point in enumerate(points):
            dx = (float(point["x"]) - mx) / span_x
            dy = (float(point["y"]) - my) / span_y
            dist = (dx * dx + dy * dy) ** 0.5
            if best_dist is None or dist < best_dist:
                best_idx = idx
                best_dist = dist
        inserted_points = list(state.get(f"{prefix}_inserted_points", []) or [])
        for idx, point in enumerate(inserted_points):
            dx = (float(point["x"]) - mx) / span_x
            dy = (float(point["y"]) - my) / span_y
            dist = (dx * dx + dy * dy) ** 0.5
            if best_dist is None or dist < best_dist:
                best_inserted_idx = idx
                best_idx = None
                best_dist = dist
        if best_idx is None or best_dist is None or best_dist > 0.06:
            if best_inserted_idx is None or best_dist is None or best_dist > 0.06:
                return
        if best_inserted_idx is not None:
            update_reducer_details(prefix, inserted_points[best_inserted_idx], details_window_tag, image_width, activity_label, state)
            return
        update_reducer_details(prefix, points[best_idx], details_window_tag, image_width, activity_label, state)

    def _draw_inserted_smiles_from_input() -> None:
        input_tag = f"{prefix}_insert_smiles_input"
        if not dpg.does_item_exist(input_tag):
            return
        smiles = str(dpg.get_value(input_tag) or "").strip()
        if not smiles:
            return
        coords = _project_reducer_inserted_smiles(prefix, smiles, state)
        if coords is None:
            log_event(log_section, "Failed to project inserted SMILES", indent=2)
            log_settings(log_section, indent=3, smiles=smiles, fingerprint=fp_algorithm)
            return
        inserted_points = state.setdefault(f"{prefix}_inserted_points", [])
        inserted_points.append(
            {
                "inserted_id": len(inserted_points) + 1,
                "smiles": smiles,
                "x": float(coords[0]),
                "y": float(coords[1]),
                "name": "Inserted molecule",
                "mol_id": "-",
                "subset_text": "Inserted",
                "activity_text": "N/A",
                "cluster_text": "N/A",
            }
        )
        log_event(log_section, "Projected external molecule into plot space", indent=2)
        log_settings(log_section, indent=3, inserted_id=len(inserted_points), x=f"{coords[0]:.3f}", y=f"{coords[1]:.3f}")
        state[f"{prefix}_refresh_colors"]()

    def _delete_all_inserted_smiles() -> None:
        state[f"{prefix}_inserted_points"] = []
        log_event(log_section, "Removed all inserted molecules", indent=2)
        state[f"{prefix}_refresh_colors"]()

    def _set_mcs_timeout_mode(app_data: Any) -> None:
        state[f"{prefix}_mcs_timeout"] = _save_mcs_timeout_setting(state, f"{prefix}_mcs_timeout", app_data)

    def _set_mcs_features_mode(app_data: Any) -> None:
        state[f"{prefix}_mcs_features"] = _save_mcs_features_setting(state, f"{prefix}_mcs_features", app_data)

    def _begin_mcs_selection() -> None:
        if state.get(f"{prefix}_mcs_selection_in_progress"):
            return
        if state.get("current_chemspace_subtab") != f"{prefix}_tab":
            return
        if not dpg.does_item_exist(f"{prefix}_plot") or not dpg.is_item_hovered(f"{prefix}_plot"):
            return
        plot_mouse_raw = dpg.get_plot_mouse_pos()
        screen_mouse_raw = tuple(dpg.get_mouse_pos(local=False))
        plot_mouse = _clamp_plot_pos_to_axes(f"{prefix}_x_axis", f"{prefix}_y_axis", plot_mouse_raw)
        screen_mouse = _clamp_screen_pos_to_plot(f"{prefix}_plot", screen_mouse_raw)
        if not plot_mouse:
            return
        state[f"{prefix}_mcs_selection_in_progress"] = True
        state[f"{prefix}_mcs_start_plot_pos"] = (float(plot_mouse[0]), float(plot_mouse[1]))
        state[f"{prefix}_mcs_start_screen_pos"] = (float(screen_mouse[0]), float(screen_mouse[1]))
        _draw_selection_overlay(prefix, state[f"{prefix}_mcs_start_screen_pos"], state[f"{prefix}_mcs_start_screen_pos"])

    def _update_mcs_selection_drag() -> None:
        if not state.get(f"{prefix}_mcs_selection_in_progress"):
            return
        start_screen = state.get(f"{prefix}_mcs_start_screen_pos")
        if not isinstance(start_screen, (tuple, list)) or len(start_screen) < 2:
            return
        current_screen = _clamp_screen_pos_to_plot(f"{prefix}_plot", tuple(dpg.get_mouse_pos(local=False)))
        _draw_selection_overlay(prefix, (float(start_screen[0]), float(start_screen[1])), (float(current_screen[0]), float(current_screen[1])))

    def _finish_mcs_selection() -> None:
        if not state.get(f"{prefix}_mcs_selection_in_progress"):
            return
        start_plot = state.get(f"{prefix}_mcs_start_plot_pos")
        end_plot = _clamp_plot_pos_to_axes(f"{prefix}_x_axis", f"{prefix}_y_axis", dpg.get_plot_mouse_pos())
        _clear_selection_overlay(prefix)
        state[f"{prefix}_mcs_selection_in_progress"] = False
        state[f"{prefix}_mcs_selection_just_finished"] = True
        if not isinstance(start_plot, (tuple, list)) or len(start_plot) < 2 or not end_plot:
            return
        x0, y0 = float(start_plot[0]), float(start_plot[1])
        x1, y1 = float(end_plot[0]), float(end_plot[1])
        if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
            return
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        selected_records = [
            point for point in list(state.get(f"{prefix}_plot_points", []) or [])
            if xmin <= float(point["x"]) <= xmax and ymin <= float(point["y"]) <= ymax
        ]
        draw_loading_screen(state, bg=False)
        try:
            _update_reducer_details_mcs(prefix, selected_records, details_window_tag, image_width, activity_label, state)
        finally:
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")

    if dpg.does_item_exist(f"{prefix}_plot_handler_registry"):
        dpg.delete_item(f"{prefix}_plot_handler_registry")
    with dpg.handler_registry(tag=f"{prefix}_plot_handler_registry"):
        dpg.add_mouse_click_handler(callback=lambda: _on_click())
        dpg.add_mouse_move_handler(callback=lambda: _update_tooltip())
        dpg.add_mouse_wheel_handler(callback=lambda: _update_tooltip())
        dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Middle, callback=lambda: _begin_mcs_selection())
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=lambda: (_update_mcs_selection_drag(), _update_tooltip()))
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Middle, callback=lambda: _finish_mcs_selection())

    def _build_plot_specific_popup_items() -> None:
        add_chemspace_plot_specific_popup_controls(
            state=state,
            point_size_state_key=f"{prefix}_point_size_mode",
            point_size_combo_tag=f"{prefix}_point_size_combo",
            point_size_callback=_set_point_size_mode,
            mcs_timeout_state_key=f"{prefix}_mcs_timeout",
            mcs_timeout_combo_tag=f"{prefix}_mcs_timeout_combo",
            mcs_timeout_callback=_set_mcs_timeout_mode,
            mcs_features_state_key=f"{prefix}_mcs_features",
            mcs_features_checkbox_tag=f"{prefix}_mcs_features_checkbox",
            mcs_features_callback=_set_mcs_features_mode,
            input_tag=f"{prefix}_insert_smiles_input",
            draw_callback=_draw_inserted_smiles_from_input,
            delete_callback=_delete_all_inserted_smiles,
        )

    register_plot_context_popup(
        state,
        context_key=f"{prefix}_plot_context",
        plot_tag=f"{prefix}_plot",
        x_axis_tag=f"{prefix}_x_axis",
        y_axis_tag=f"{prefix}_y_axis",
        theme_kind="plot",
        specific_builder=_build_plot_specific_popup_items,
    )

    state[f"{prefix}_refresh_colors"] = lambda: (
        _set_color_mode(state.get(f"{prefix}_current_coloring", current_coloring)),
        state.get(f"{prefix}_relayout_colormap_scale", lambda: None)(),
    )
    update_reducer_details(prefix, None, details_window_tag, image_width, activity_label, state)


def draw_umap_plot_2d(
    subset: str,
    embedding: np.ndarray,
    molecule_data: list[dict[str, Any]],
    activity_label: str,
    activity_values: list[float],
    fp_algorithm: str,
    cluster_labels: np.ndarray,
    state: dict[str, Any],
) -> None:
    draw_reducer_plot_2d(
        prefix="umap",
        log_section="UMAP",
        subset=subset,
        embedding=embedding,
        molecule_data=molecule_data,
        activity_label=activity_label,
        activity_values=activity_values,
        cluster_labels=cluster_labels,
        fp_algorithm=fp_algorithm,
        plot_window_tag="umap_window",
        details_window_tag="umap_details_window",
        title=f"{subset.replace('subset_', 'Subset ')} | {fp_algorithm}-Based UMAP",
        x_label="UMAP 1",
        y_label="UMAP 2",
        state=state,
    )


def draw_tsne_plot_2d(
    subset: str,
    embedding: np.ndarray,
    molecule_data: list[dict[str, Any]],
    activity_label: str,
    activity_values: list[float],
    fp_algorithm: str,
    cluster_labels: np.ndarray,
    state: dict[str, Any],
) -> None:
    draw_reducer_plot_2d(
        prefix="tsne",
        log_section="TSNE",
        subset=subset,
        embedding=embedding,
        molecule_data=molecule_data,
        activity_label=activity_label,
        activity_values=activity_values,
        cluster_labels=cluster_labels,
        fp_algorithm=fp_algorithm,
        plot_window_tag="tsne_window",
        details_window_tag="tsne_details_window",
        title=f"{subset.replace('subset_', 'Subset ')} | {fp_algorithm}-Based t-SNE",
        x_label="t-SNE 1",
        y_label="t-SNE 2",
        state=state,
    )


def _apply_embedding_jitter_3d(coords: np.ndarray, base_scale: float = 0.005, precision: int = 5) -> np.ndarray:
    def find_duplicates(arr: np.ndarray) -> list[list[int]]:
        rounded_coords = np.round(arr, decimals=precision)
        coord_map: dict[tuple[float, float, float], list[int]] = defaultdict(list)
        for i, coord in enumerate(rounded_coords):
            coord_map[tuple(coord)].append(i)
        return [group for group in coord_map.values() if len(group) > 1]

    def is_discrete(values: np.ndarray) -> bool:
        return bool(np.all(np.equal(np.mod(values, 1), 0)))

    arr = np.array(coords, dtype=float, copy=True)
    x_discrete = is_discrete(arr[:, 0])
    y_discrete = is_discrete(arr[:, 1])
    z_discrete = is_discrete(arr[:, 2])
    scale_x = base_scale * (np.ptp(arr[:, 0]) if not x_discrete else 1.0)
    scale_y = base_scale * (np.ptp(arr[:, 1]) if not y_discrete else 1.0)
    scale_z = base_scale * (np.ptp(arr[:, 2]) if not z_discrete else 1.0)
    for group in find_duplicates(arr):
        for idx in group:
            if not x_discrete:
                arr[idx, 0] += np.random.normal(0, scale_x)
            if not y_discrete:
                arr[idx, 1] += np.random.normal(0, scale_y)
            if not z_discrete:
                arr[idx, 2] += np.random.normal(0, scale_z)
    return arr


def draw_embedding_plot_3d_common(
    *,
    prefix: str,
    subset: str,
    embedding: np.ndarray,
    molecule_data: list[dict[str, Any]],
    activity_label: str,
    activity_values: list[float],
    axis_labels: tuple[str, str, str],
    title_prefix: str,
    html_template_name: str,
    state: dict[str, Any],
) -> None:
    set_loading_screen_progress(state, 97)
    text_color = rgba_tuple_to_string(state["themes"][state["theme_name"]]["Text Color"])

    custom_tooltips = []
    for idx, m in enumerate(molecule_data):
        lines = [
            f"Mol {m['Mol_ID']}",
            f"Name: {m['Name']}",
            f"Subset: {m['Subset']}",
            f"Cluster: {m.get('Cluster', 'N/A')}",
        ]
        if activity_label != "None":
            value = m.get(activity_label, "N/A")
            try:
                value = f"{float(value):.2f}"
            except Exception:
                value = str(value)
            lines.append(f"{activity_label}: {value}")
        custom_tooltips.append("<br>".join(lines))
    set_loading_screen_progress(state, 97.3)

    min_act, max_act = min(activity_values), max(activity_values)
    continuous_colors = _get_active_continuous_colormap(state)
    plotly_continuous_colorscale = _plotly_colorscale_from_rgba(continuous_colors)
    rgba_colors = []
    for val in activity_values:
        r, g, b = _interpolate_continuous_rgb(val, min_act, max_act, continuous_colors)
        rgba_colors.append(f"rgba({r},{g},{b},1.0)")

    coords = _apply_embedding_jitter_3d(np.asarray(embedding[:, :3], dtype=float))
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    set_loading_screen_progress(state, 97.6)

    subset_values = [int(m["Subset"]) for m in molecule_data]
    cluster_values = [int(m.get("Cluster", 0)) for m in molecule_data]
    discrete_palette = _get_active_discrete_colormap(state)
    css_activity_gradient = "linear-gradient(to top, " + ", ".join(
        f"rgb({color[0]}, {color[1]}, {color[2]})" for color in continuous_colors
    ) + ")"

    trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        showlegend=False,
        name="",
        marker=dict(
            size=4,
            opacity=1.0,
            color=rgba_colors,
            showscale=True,
            colorscale=plotly_continuous_colorscale,
            colorbar=dict(title=activity_label),
        ),
        text=custom_tooltips,
        hovertemplate="%{text}<extra></extra>",
    )

    x_label, y_label, z_label = axis_labels
    fig = go.Figure(data=[trace])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=20),
        scene=dict(
            xaxis=dict(title=dict(text=x_label, font=dict(color=text_color)), showticklabels=False, showspikes=False, gridcolor="lightgray", backgroundcolor="rgba(80,80,80,255)"),
            yaxis=dict(title=dict(text=y_label, font=dict(color=text_color)), showticklabels=False, showspikes=False, gridcolor="lightgray", backgroundcolor="rgba(80,80,80,255)"),
            zaxis=dict(title=dict(text=z_label, font=dict(color=text_color)), showticklabels=False, showspikes=False, gridcolor="lightgray", backgroundcolor="rgba(80,80,80,255)"),
            camera=dict(eye=dict(x=0.01, y=0.1, z=1.99), up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), projection=dict(type="perspective")),
            dragmode="orbit",
        ),
        paper_bgcolor=rgba_tuple_to_string(state["themes"][state["theme_name"]]["Main Background"]),
    )
    set_loading_screen_progress(state, 98.0)

    html_path = _chemspace_html_output_path(state, f"{subset}_3D_{prefix}.html")
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    set_loading_screen_progress(state, 98.6)

    html_template_path = resource_path("app", "analysis", "chemspace", "html", html_template_name)
    with open(html_template_path, "r", encoding="utf-8") as f:
        custom_template = f.read()
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    custom_buttons_html = (
        custom_template
        .replace("__js_activity__", json.dumps(activity_values))
        .replace("__js_subset__", json.dumps(subset_values))
        .replace("__js_cluster__", json.dumps(cluster_values))
        .replace("__js_discrete_palette__", json.dumps(discrete_palette))
        .replace("__js_subset_ids__", json.dumps(sorted(set(subset_values))))
        .replace("__js_cluster_ids__", json.dumps(sorted(set(cluster_values))))
        .replace("__js_tooltips__", json.dumps(custom_tooltips))
        .replace("__activity_label__", json.dumps(activity_label))
        .replace("__js_textcolor__", json.dumps(text_color))
        .replace("__js_continuous_colorscale__", json.dumps(plotly_continuous_colorscale))
        .replace("__css_activity_gradient__", css_activity_gradient)
    )

    html += "\n<!-- Custom Controls Injected -->\n" + custom_buttons_html
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    set_loading_screen_progress(state, 99.0)
    open_html_safely(html_path)


def draw_pca_plot(activity: str, data: Any, subset: str, state: dict[str, Any]) -> None:
    from app.analysis.chemspace.chemspace_logic_common import perform_pca_projection

    payload = perform_pca_projection(activity, data, subset, state)
    if payload is None:
        return

    if payload["dimension"] == "2D":
        draw_pca_plot_2d_common(
            subset=payload["subset"],
            x_pca=payload["x_pca"],
            variance_ratio=payload["variance_ratio"],
            filtered_mols=payload["filtered_mols"],
            molecule_data=payload["molecule_data"],
            activity=payload["activity"],
            activity_label=payload["activity_label"],
            labels=payload["labels"],
            activity_values=payload["activity_values"],
            fp_algorithm=payload["fp_algorithm"],
            best_k=payload["best_k"],
            state=state,
        )
    else:
        draw_pca_plot_3d_common(
            subset=payload["subset"],
            x_pca=payload["x_pca"],
            variance_ratio=payload["variance_ratio"],
            molecule_data=payload["molecule_data"],
            activity_label=payload["activity_label"],
            activity_values=payload["activity_values"],
            state=state,
        )


def _get_active_continuous_colormap(state: dict[str, Any]) -> list[list[int]]:
    defs = state.get("continuous_colormap_defs", {}) or {}
    name = state.get("colormap_continuous", "")
    colors = defs.get(name)
    if isinstance(colors, list) and len(colors) >= 2:
        return colors
    for fallback in defs.values():
        if isinstance(fallback, list) and len(fallback) >= 2:
            return fallback
    return [
        [255, 0, 0, 255],
        [255, 165, 0, 255],
        [255, 255, 0, 255],
        [50, 205, 50, 255],
        [0, 100, 0, 255],
    ]


def _interpolate_continuous_rgb(value: float, min_value: float, max_value: float, colors: list[list[int]]) -> tuple[int, int, int]:
    if max_value == min_value:
        norm = 0.5
    else:
        norm = (value - min_value) / (max_value - min_value)
        norm = max(0.0, min(1.0, norm))

    if len(colors) == 1:
        r, g, b = colors[0][:3]
        return int(r), int(g), int(b)

    scaled = norm * (len(colors) - 1)
    index = int(scaled)
    t = scaled - index
    if index >= len(colors) - 1:
        r, g, b = colors[-1][:3]
        return int(r), int(g), int(b)

    r1, g1, b1 = colors[index][:3]
    r2, g2, b2 = colors[index + 1][:3]
    r = int(round(r1 + (r2 - r1) * t))
    g = int(round(g1 + (g2 - g1) * t))
    b = int(round(b1 + (b2 - b1) * t))
    return r, g, b


def _plotly_colorscale_from_rgba(colors: list[list[int]]) -> list[list[Any]]:
    if len(colors) == 1:
        r, g, b = colors[0][:3]
        return [[0.0, f"rgb({r},{g},{b})"], [1.0, f"rgb({r},{g},{b})"]]
    steps = len(colors) - 1
    return [[idx / steps, f"rgb({color[0]},{color[1]},{color[2]})"] for idx, color in enumerate(colors)]


def _get_active_discrete_colormap(state: dict[str, Any]) -> list[list[int]]:
    defs = state.get("discrete_colormap_defs", {}) or {}
    name = state.get("colormap_discrete", "")
    colors = defs.get(name)
    if isinstance(colors, list) and colors:
        return colors
    for fallback in defs.values():
        if isinstance(fallback, list) and fallback:
            return fallback
    return [
        [26, 152, 171, 255],
        [200, 40, 90, 255],
        [255, 196, 61, 255],
        [92, 184, 92, 255],
        [122, 88, 193, 255],
        [232, 125, 49, 255],
        [70, 129, 218, 255],
        [214, 78, 154, 255],
        [157, 194, 62, 255],
        [255, 152, 202, 255],
    ]


def draw_pca_plot_3d_common(
    subset: str,
    x_pca: np.ndarray,
    variance_ratio: np.ndarray,
    molecule_data: list[dict[str, Any]],
    activity_label: str,
    activity_values: list[float],
    state: dict[str, Any],
) -> None:
    def find_duplicates(xs: Any, ys: Any, zs: Any, precision: int = 5) -> Any:
        rounded_coords = np.round(np.column_stack((xs, ys, zs)), decimals=precision)
        coord_map = defaultdict(list)
        for i, coord in enumerate(rounded_coords):
            coord_map[tuple(coord)].append(i)
        return [group for group in coord_map.values() if len(group) > 1]

    def is_discrete(values: Any) -> Any:
        return np.all(np.equal(np.mod(values, 1), 0))

    def apply_jitter_to_duplicates(xs: Any, ys: Any, zs: Any, base_scale: float = 0.005, precision: int = 5) -> Any:
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        x_discrete = is_discrete(xs)
        y_discrete = is_discrete(ys)
        z_discrete = is_discrete(zs)
        range_x = np.ptp(xs) if not x_discrete else 1.0
        range_y = np.ptp(ys) if not y_discrete else 1.0
        range_z = np.ptp(zs) if not z_discrete else 1.0
        scale_x = base_scale * range_x
        scale_y = base_scale * range_y
        scale_z = base_scale * range_z
        duplicates = find_duplicates(xs, ys, zs, precision)
        for group in duplicates:
            for idx in group:
                if not x_discrete:
                    xs[idx] += np.random.normal(0, scale_x)
                if not y_discrete:
                    ys[idx] += np.random.normal(0, scale_y)
                if not z_discrete:
                    zs[idx] += np.random.normal(0, scale_z)
        return xs, ys, zs

    set_loading_screen_progress(state, 97)
    text_color = rgba_tuple_to_string(state["themes"][state["theme_name"]]["Text Color"])

    custom_tooltips = []
    for m in molecule_data:
        if activity_label == "None":
            custom_tooltips.append(f"Mol {m['Mol_ID']}<br>Name: {m['Name']}<br>Subset: {m['Subset']}<br>Cluster: {m['Cluster']}")
        else:
            if isinstance(m["Cl. Mean Act"], (float, int)):
                custom_tooltips.append(
                    f"Mol {m['Mol_ID']}<br>Name: {m['Name']}<br>Subset: {m['Subset']}<br>Cluster: {m['Cluster']}<br>Cl. Mean Act: {float(m['Cl. Mean Act']):.2f}<br>{activity_label}: {float(m[activity_label]):.2f}"
                )
            else:
                custom_tooltips.append(
                    f"Mol {m['Mol_ID']}<br>Name: {m['Name']}<br>Subset: {m['Subset']}<br>Cluster: {m['Cluster']}<br>Cl. Mean Act: {m['Cl. Mean Act']}<br>{activity_label}: {float(m[activity_label]):.2f}"
                )
    set_loading_screen_progress(state, 97.3)

    min_act, max_act = min(activity_values), max(activity_values)
    continuous_colors = _get_active_continuous_colormap(state)
    plotly_continuous_colorscale = _plotly_colorscale_from_rgba(continuous_colors)
    rgba_colors = []
    for val in activity_values:
        r, g, b = _interpolate_continuous_rgb(val, min_act, max_act, continuous_colors)
        rgba_colors.append(f"rgba({r},{g},{b},1.0)")

    xs, ys, zs = x_pca[:, 0], x_pca[:, 1], x_pca[:, 2]
    xs, ys, zs = apply_jitter_to_duplicates(xs, ys, zs)
    set_loading_screen_progress(state, 97.6)

    subset_values = [int(m["Subset"]) for m in molecule_data]
    cluster_values = [int(m["Cluster"]) for m in molecule_data]
    discrete_palette = _get_active_discrete_colormap(state)
    css_activity_gradient = "linear-gradient(to top, " + ", ".join(
        f"rgb({color[0]}, {color[1]}, {color[2]})" for color in continuous_colors
    ) + ")"

    trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        showlegend=False,
        name="",
        marker=dict(
            size=4,
            opacity=1.0,
            color=rgba_colors,
            showscale=True,
            colorscale=plotly_continuous_colorscale,
            colorbar=dict(title=activity_label),
        ),
        text=custom_tooltips,
        hovertemplate="%{text}<extra></extra>",
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=20),
        scene=dict(
            xaxis=dict(title=dict(text=f"PC1 ({variance_ratio[0]:.1f}%)", font=dict(color=text_color)), showticklabels=False, showspikes=False, gridcolor="lightgray", backgroundcolor="rgba(80,80,80,255)"),
            yaxis=dict(title=dict(text=f"PC2 ({variance_ratio[1]:.1f}%)", font=dict(color=text_color)), showticklabels=False, showspikes=False, gridcolor="lightgray", backgroundcolor="rgba(80,80,80,255)"),
            zaxis=dict(title=dict(text=f"PC3 ({variance_ratio[2]:.1f}%)", font=dict(color=text_color)), showticklabels=False, showspikes=False, gridcolor="lightgray", backgroundcolor="rgba(80,80,80,255)"),
            camera=dict(eye=dict(x=0.01, y=0.1, z=1.99), up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), projection=dict(type="perspective")),
            dragmode="orbit",
        ),
        paper_bgcolor=rgba_tuple_to_string(state["themes"][state["theme_name"]]["Main Background"]),
    )
    set_loading_screen_progress(state, 98.0)

    html_path = _chemspace_html_output_path(state, f"{subset}_3D_pca.html")
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    set_loading_screen_progress(state, 98.6)

    html_template_path = resource_path("app", "analysis", "chemspace", "html", "3d_pca.html")
    with open(html_template_path, "r", encoding="utf-8") as f:
        custom_template = f.read()
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    custom_buttons_html = (
        custom_template
        .replace("__js_activity__", json.dumps(activity_values))
        .replace("__js_subset__", json.dumps(subset_values))
        .replace("__js_cluster__", json.dumps(cluster_values))
        .replace("__js_discrete_palette__", json.dumps(discrete_palette))
        .replace("__js_subset_ids__", json.dumps(sorted(set(subset_values))))
        .replace("__js_cluster_ids__", json.dumps(sorted(set(cluster_values))))
        .replace("__js_tooltips__", json.dumps(custom_tooltips))
        .replace("__activity_label__", json.dumps(activity_label))
        .replace("__js_textcolor__", json.dumps(text_color))
        .replace("__js_continuous_colorscale__", json.dumps(plotly_continuous_colorscale))
        .replace("__css_activity_gradient__", css_activity_gradient)
    )

    html += "\n<!-- Custom Controls Injected -->\n" + custom_buttons_html
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    set_loading_screen_progress(state, 99.0)
    open_html_safely(html_path)


def draw_umap_plot_3d(
    *,
    subset: str,
    embedding: np.ndarray,
    molecule_data: list[dict[str, Any]],
    activity_label: str,
    activity_values: list[float],
    fp_algorithm: str,
    cluster_labels: np.ndarray,
    state: dict[str, Any],
) -> None:
    log_event("UMAP", "Drawing 'umap' plot", indent=1)
    log_settings(
        "UMAP",
        indent=2,
        subset=subset,
        activity=activity_label,
        fingerprint=fp_algorithm,
        molecules=len(molecule_data),
        colormap_continuous=state.get("colormap_continuous"),
        colormap_discrete=state.get("colormap_discrete"),
        dimension="3D",
    )
    draw_embedding_plot_3d_common(
        prefix="umap",
        subset=subset,
        embedding=embedding,
        molecule_data=molecule_data,
        activity_label=activity_label,
        activity_values=activity_values,
        axis_labels=("UMAP1", "UMAP2", "UMAP3"),
        title_prefix="UMAP",
        html_template_name="3d_umap.html",
        state=state,
    )


def draw_tsne_plot_3d(
    *,
    subset: str,
    embedding: np.ndarray,
    molecule_data: list[dict[str, Any]],
    activity_label: str,
    activity_values: list[float],
    fp_algorithm: str,
    cluster_labels: np.ndarray,
    state: dict[str, Any],
) -> None:
    log_event("TSNE", "Drawing 't-SNE' plot", indent=1)
    log_settings(
        "TSNE",
        indent=2,
        subset=subset,
        activity=activity_label,
        fingerprint=fp_algorithm,
        molecules=len(molecule_data),
        colormap_continuous=state.get("colormap_continuous"),
        colormap_discrete=state.get("colormap_discrete"),
        dimension="3D",
    )
    draw_embedding_plot_3d_common(
        prefix="tsne",
        subset=subset,
        embedding=embedding,
        molecule_data=molecule_data,
        activity_label=activity_label,
        activity_values=activity_values,
        axis_labels=("t-SNE1", "t-SNE2", "t-SNE3"),
        title_prefix="t-SNE",
        html_template_name="3d_tsne.html",
        state=state,
    )


def _ensure_chemspace_dendro_cache(state: dict[str, Any]) -> dict[str, Any]:
    return state.setdefault("_chemspace_dendro_cache", {})


def _normalize_scaffold_for_dendrogram(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None:
        return None
    m = rdMolStandardize.Cleanup(mol)
    m = rdMolStandardize.TautomerEnumerator().Canonicalize(m)
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:
        pass
    Chem.SanitizeMol(
        m,
        Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
    )
    Chem.AssignStereochemistry(m, force=True, cleanIt=True)
    return m


def _dendro_rgba_from_colorname(color_str: Any) -> tuple[int, int, int, int]:
    try:
        if color_str in ("C0", "b", "blue"):
            return (80, 80, 80, 255)
        r, g, b = color_string_to_rgb255(color_str, fallback=(80, 80, 80))
        return (r, g, b, 255)
    except Exception:
        return (80, 80, 80, 255)


def _chemspace_dendrogram_color_map(
    state: dict[str, Any],
    color_list: list[Any],
) -> dict[Any, tuple[int, int, int, int]]:
    color_map: dict[Any, tuple[int, int, int, int]] = {}
    discrete_idx = 0
    for color_name in color_list:
        if color_name in color_map:
            continue
        if color_name in ("C0", "b", "blue"):
            color_map[color_name] = (80, 80, 80, 255)
            continue
        color_map[color_name] = _cycle_discrete_plot_color(state, discrete_idx)
        discrete_idx += 1
    return color_map


def _chemspace_dendrogram_cluster_map(
    state: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, int], list[str]]:
    cache = state.get("_chemspace_dendro_cache") or {}
    linkage_matrix = cache.get("linkage")
    labels_input = cache.get("labels_input")
    labels_ordered = cache.get("labels_ordered")
    if linkage_matrix is None or labels_input is None or labels_ordered is None:
        return {}, []

    id_to_logical = {str(int(nm.split("_")[1])): nm for nm in state["smiles_rgd_dict"].keys()}
    ordered_logical_names = [id_to_logical[lbl] for lbl in labels_ordered if lbl in id_to_logical]
    raw_labels = fcluster(linkage_matrix, t=float(threshold), criterion="distance")
    name_to_cluster_raw = {
        id_to_logical[labels_input[i]]: int(raw_labels[i])
        for i in range(len(labels_input))
        if labels_input[i] in id_to_logical
    }

    cluster_remap: dict[int, int] = {}
    next_label = 1
    name_to_cluster: dict[str, int] = {}
    for nm in ordered_logical_names:
        raw_cluster = name_to_cluster_raw.get(nm)
        if raw_cluster is None:
            continue
        if raw_cluster not in cluster_remap:
            cluster_remap[raw_cluster] = next_label
            next_label += 1
        name_to_cluster[nm] = cluster_remap[raw_cluster]
    return name_to_cluster, ordered_logical_names


def _reset_dendrogram_details(state: dict[str, Any]) -> None:
    img_w = int(state["plots_pca_img_width"])
    img_h = round(img_w * 0.75)
    empty = np.zeros((img_w * img_h * 4,), dtype=np.float32)
    if dpg.does_item_exist("dendrogram_mol_image_texture"):
        dpg.set_value("dendrogram_mol_image_texture", empty)
    for tag, text in [
        ("dendrogram_details_name_text", "-"),
        ("dendrogram_details_coords_text", "Subset molecules: -"),
        ("dendrogram_details_subset_text", "Cluster: -"),
        ("dendrogram_details_activity_text", "Cluster size: -"),
    ]:
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, text)


def _update_dendrogram_details_for_subset(state: dict[str, Any], subset_name: str) -> None:
    _ensure_dendrogram_details_widgets(state)
    scaffold_id = int(subset_name.split("_")[1])
    mol = None

    try:
        subset_dict = state["smiles_rgd_dict"].get(subset_name, {})
        scaffold_smi = None
        for _, mol_data in subset_dict.items():
            if isinstance(mol_data, dict) and "Core" in mol_data:
                scaffold_smi = mol_data["Core"]
                break
        if scaffold_smi:
            mol = Chem.MolFromSmiles(scaffold_smi, sanitize=False)
            if mol is None:
                raise ValueError()
    except Exception:
        mol = None

    if mol is None:
        try:
            scaffold_sma = state["smiles_rgd_dict"][subset_name].get("Core", "")
            mol = Chem.MolFromSmarts(scaffold_sma)
            if mol is None:
                raise ValueError()
        except Exception:
            mol = None

    if mol is None:
        try:
            subset_dict = state["molblocks_rgd_dict"].get(subset_name, {})
            scaffold_mb = None
            for _, mol_data in subset_dict.items():
                if isinstance(mol_data, dict) and "Core" in mol_data:
                    scaffold_mb = mol_data["Core"]
                    break
            if scaffold_mb:
                mol = Chem.MolFromMolBlock(scaffold_mb, sanitize=False)
                if mol is None or mol.GetNumAtoms() == 0:
                    raise ValueError()
        except Exception:
            mol = None

    if mol is None:
        _reset_dendrogram_details(state)
        return

    try:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.HasProp("molAtomMapNumber"):
                idx = atom.GetProp("molAtomMapNumber")
                atom.SetProp("atomLabel", f"R{idx}")
    except Exception:
        pass

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    except Exception:
        pass

    img_w = int(state["plots_pca_img_width"])
    img_h = round(img_w * 0.75)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(img_w, img_h)
    opts = drawer.drawOptions()
    opts.padding = 0.025
    opts.bondLineWidth = 1
    opts.minFontSize = 1
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=f"{subset_name.replace('_', ' ').replace('su', 'Su')} - Common Core",
    )
    drawer.FinishDrawing()
    img = pilImage.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    arr = (np.array(img) / 255.0).astype(np.float32).flatten()
    if dpg.does_item_exist("dendrogram_mol_image_texture"):
        dpg.set_value("dendrogram_mol_image_texture", arr)

    threshold = float(state.get("chemspace_cluster_threshold", state.get("cluster threshold", 0.2)))
    name_to_cluster, _ = _chemspace_dendrogram_cluster_map(state, threshold)
    cluster_id = name_to_cluster.get(subset_name)
    cluster_text = str(cluster_id) if cluster_id is not None else "-"
    cluster_size = sum(1 for _, cid in name_to_cluster.items() if cid == cluster_id) if cluster_id is not None else 0
    subset_entry = (state.get("smiles_rgd_dict") or {}).get(subset_name, {})
    subset_molecule_count = len(subset_entry)

    if dpg.does_item_exist("dendrogram_details_name_text"):
        dpg.set_value("dendrogram_details_name_text", f"{subset_name.replace('_', ' ').title()}")
    if dpg.does_item_exist("dendrogram_details_subset_text"):
        dpg.set_value("dendrogram_details_subset_text", f"Cluster: {cluster_text}")
    if dpg.does_item_exist("dendrogram_details_activity_text"):
        dpg.set_value(
            "dendrogram_details_activity_text",
            f"Cluster size: {cluster_size}",
        )
    if dpg.does_item_exist("dendrogram_details_coords_text"):
        dpg.set_value(
            "dendrogram_details_coords_text",
            f"Subset molecules: {subset_molecule_count}",
        )


def _ensure_chemspace_dendro_frame_theme(state: dict[str, Any]) -> Any:
    th = state.get("_chemspace_dendro_frame_theme")
    if th and dpg.does_item_exist(th):
        return th
    with dpg.theme() as th:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 0.0, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (80, 80, 80, 80), category=dpg.mvThemeCat_Plots)
    state["_chemspace_dendro_frame_theme"] = th
    return th


def _visible_dendrogram_tick_indices(total_labels: int) -> list[int]:
    if total_labels <= 16:
        return list(range(total_labels))
    max_visible = 16
    step = max(1, int(np.ceil(total_labels / max_visible)))
    indices = list(range(0, total_labels, step))
    if (total_labels - 1) not in indices:
        indices.append(total_labels - 1)
    return indices


def _ensure_dendrogram_details_widgets(state: dict[str, Any]) -> None:
    if not dpg.does_item_exist("dendrogram_details_window"):
        return
    img_w = int(state["plots_pca_img_width"])
    img_h = round(img_w * 0.75)
    empty = np.zeros((img_w * img_h * 4,), dtype=np.float32)
    if not dpg.does_item_exist("dendrogram_mol_image_texture"):
        dpg.add_dynamic_texture(img_w, img_h, empty, tag="dendrogram_mol_image_texture", parent="texture_registry")
    else:
        dpg.set_value("dendrogram_mol_image_texture", empty)
    if dpg.does_item_exist("dendrogram_details_window"):
        dpg.delete_item("dendrogram_details_window", children_only=True)
        dpg.add_image(
            "dendrogram_mol_image_texture",
            width=img_w,
            height=img_h,
            tag="dendrogram_mol_image_widget",
            parent="dendrogram_details_window",
            border_color=(0, 0, 0, 0),
        )
        register_responsive_image(
            state,
            image_tag="dendrogram_mol_image_widget",
            parent_tag="dendrogram_details_window",
            aspect_ratio=0.75,
            tab="dendrogram_tab",
        )
        dpg.add_text("-", tag="dendrogram_details_name_text", parent="dendrogram_details_window")
        dpg.add_text("Subset molecules: -", tag="dendrogram_details_coords_text", parent="dendrogram_details_window")
        dpg.add_text("Cluster: -", tag="dendrogram_details_subset_text", parent="dendrogram_details_window")
        dpg.add_text("Cluster size: -", tag="dendrogram_details_activity_text", parent="dendrogram_details_window")
        update_responsive_images(state)
        for offset in (1, 3, 6):
            dpg.set_frame_callback(
                dpg.get_frame_count() + offset,
                lambda *_: update_responsive_images(state),
            )


def _recolor_chemspace_dendrogram_segments(state: dict[str, Any], threshold: float) -> None:
    cache = state.get("_chemspace_dendro_cache") or {}
    linkage_matrix = cache.get("linkage")
    labels_input = cache.get("labels_input")
    link_to_seg_ids = cache.get("link_to_seg_ids")
    if linkage_matrix is None or labels_input is None or not link_to_seg_ids:
        return
    dendro = dendrogram(linkage_matrix, labels=labels_input, no_plot=True, color_threshold=float(threshold))
    color_map = _chemspace_dendrogram_color_map(state, dendro["color_list"])
    for link_idx, color_name in enumerate(dendro["color_list"]):
        rgba = color_map.get(color_name, _dendro_rgba_from_colorname(color_name))
        for seg_id in link_to_seg_ids[link_idx]:
            if dpg.does_item_exist(seg_id):
                dpg.configure_item(seg_id, color=rgba)


def _refresh_chemspace_dendrogram_highlight(state: dict[str, Any]) -> None:
    cache = state.get("_chemspace_dendro_cache") or {}
    labels_ordered = cache.get("labels_ordered") or []
    frame_ymin = cache.get("frame_ymin", -0.01)
    frame_ymax = cache.get("frame_ymax", 1.01)
    line_tag = "chemspace_dendro_highlight_series"

    if not dpg.does_item_exist("chemspace_dendro_y_axis"):
        return

    try:
        subset_number = int(state.get("chemspace_dendrogram_highlight_subset", 0) or 0)
    except Exception:
        subset_number = 0

    if dpg.does_item_exist(line_tag):
        dpg.delete_item(line_tag)

    if subset_number <= 0:
        return

    try:
        target_label = str(int(subset_number))
    except Exception:
        return
    if target_label not in labels_ordered:
        return

    x_value = labels_ordered.index(target_label) + 1
    dpg.add_line_series(
        x=[x_value, x_value],
        y=[frame_ymin, frame_ymax],
        tag=line_tag,
        parent="chemspace_dendro_y_axis",
        label="",
    )
    if not dpg.does_item_exist("chemspace_dendro_highlight_theme"):
        with dpg.theme(tag="chemspace_dendro_highlight_theme"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (220, 40, 40, 255), category=dpg.mvThemeCat_Plots)
    dpg.bind_item_theme(line_tag, "chemspace_dendro_highlight_theme")


def on_chemspace_dendrogram_threshold_change(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    new_distance = dpg.get_value(sender)
    if new_distance is None:
        return
    new_distance = max(0.0, min(1.0, float(new_distance)))
    state["chemspace_cluster_threshold"] = new_distance
    if dpg.does_item_exist("chemspace_scaffold_hierarchical_dendrogram"):
        dpg.configure_item(
            "chemspace_scaffold_hierarchical_dendrogram",
            label=f"Substructures Hierarchical Clustering  |  Similarity Threshold = {(1.0 - new_distance) * 100:.0f}%"
        )
    _recolor_chemspace_dendrogram_segments(state, new_distance)
    _refresh_chemspace_dendrogram_highlight(state)
    selected_subset = state.get("chemspace_dendrogram_selected_subset")
    if selected_subset:
        _update_dendrogram_details_for_subset(state, selected_subset)


def draw_chemspace_dendrogram(state: dict[str, Any]) -> None:
    set_loading_screen_progress(state, 10)
    _ensure_dendrogram_details_widgets(state)
    _reset_dendrogram_details(state)
    threshold = float(state.get("chemspace_cluster_threshold", state.get("cluster threshold", 0.2)))
    state["chemspace_cluster_threshold"] = threshold
    visible_subsets = state.get("chemspace_dendrogram_visible_subsets")
    all_subsets = list(state["smiles_rgd_dict"].keys())
    if visible_subsets is None:
        visible_subsets = set(all_subsets)
        state["chemspace_dendrogram_visible_subsets"] = visible_subsets
    smiles_rgd_dict = {k: v for k, v in state["smiles_rgd_dict"].items() if k in visible_subsets}
    subset_dir = state["subset_dir"]
    subset_names: list[str] = []
    scaffold_mols: list[Chem.Mol] = []

    for subset_name in smiles_rgd_dict:
        scaffold_id = int(subset_name.split("_")[1])
        path = os.path.join(subset_dir, f"scaffold_{scaffold_id}.sdf")
        suppl = Chem.SDMolSupplier(path)
        mol = next((m for m in suppl if m is not None), None)
        if mol:
            subset_names.append(str(scaffold_id))
            scaffold_mols.append(mol)

    if len(scaffold_mols) < 2:
        if dpg.does_item_exist("dendrogram_window"):
            dpg.delete_item("dendrogram_window", children_only=True)
            dpg.add_text("Select at least two subsets to draw the dendrogram.", parent="dendrogram_window")
        set_loading_screen_progress(state, 100)
        return
    set_loading_screen_progress(state, 25)

    pairs = sorted(zip(scaffold_mols, subset_names), key=lambda x: x[0].GetNumHeavyAtoms(), reverse=True)
    scaffold_mols, subset_names = map(list, zip(*pairs))
    scaffold_mols = [_normalize_scaffold_for_dendrogram(m) for m in scaffold_mols]
    gen = GetMorganGenerator(radius=3, fpSize=4096)
    fps = [gen.GetCountFingerprint(m) for m in scaffold_mols]
    n = len(fps)
    condensed = np.empty(n * (n - 1) // 2, dtype=float)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            condensed[k] = 1.0 - sim
            k += 1
    set_loading_screen_progress(state, 55)

    linkage_matrix = linkage(condensed, method="complete")
    dendro = dendrogram(linkage_matrix, labels=subset_names, no_plot=True, color_threshold=threshold)
    icoords = dendro["icoord"]
    dcoords = dendro["dcoord"]
    labels = dendro["ivl"]
    colors = dendro["color_list"]
    color_map = _chemspace_dendrogram_color_map(state, colors)
    set_loading_screen_progress(state, 72)

    dpg.delete_item("dendrogram_window", children_only=True)
    segments = []
    all_x = []
    all_y = []
    link_to_seg_ids = []
    with dpg.child_window(parent="dendrogram_window", width=-1, height=-1, no_scrollbar=False, horizontal_scrollbar=True, no_scroll_with_mouse=True, border=False):
        with dpg.plot(
            label=f"Substructures Hierarchical Clustering  |  Similarity Threshold = {(1 - threshold) * 100:.0f}%",
            tag="chemspace_scaffold_hierarchical_dendrogram",
            no_menus=True,
            no_mouse_pos=True,
            width=-1,
            height=-1,
        ) as plot:
            dendro_x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Subset", tag="chemspace_dendro_x_axis", no_highlight=True, no_gridlines=True, no_tick_marks=True)
            dendro_y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Tanimoto Similarity (%)", tag="chemspace_dendro_y_axis", no_highlight=True, no_gridlines=True, no_tick_marks=True)

            for idx, (x_raw, y_raw) in enumerate(zip(icoords, dcoords)):
                color = color_map.get(colors[idx], _dendro_rgba_from_colorname(colors[idx]))
                this_link_ids = []
                for i in range(3):
                    x1 = x_raw[i] / 10
                    x2 = x_raw[i + 1] / 10
                    y1 = y_raw[i]
                    y2 = y_raw[i + 1]
                    seg_id = dpg.draw_line(p1=(x1 + 0.5, y1), p2=(x2 + 0.5, y2), color=color, thickness=0, parent=plot)
                    this_link_ids.append(seg_id)
                    all_x.extend([x1, x2])
                    all_y.extend([y1, y2])
                link_to_seg_ids.append(this_link_ids)

            def _nice_tick_step(total_visible: int, max_ticks: int) -> int:
                if total_visible <= 0:
                    return 1
                raw_step = max(1, int(np.ceil(total_visible / max(1, max_ticks))))
                magnitude = 1
                while magnitude * 10 < raw_step:
                    magnitude *= 10
                for factor in (1, 2, 5, 10):
                    candidate = magnitude * factor
                    if raw_step <= candidate:
                        return candidate
                return magnitude * 10

            def _refresh_dendrogram_x_ticks() -> None:
                if not (dpg.does_item_exist("chemspace_scaffold_hierarchical_dendrogram") and dpg.does_item_exist("chemspace_dendro_x_axis")):
                    return
                try:
                    x0, x1 = dpg.get_axis_limits("chemspace_dendro_x_axis")
                except Exception:
                    return
                try:
                    plot_w, _ = dpg.get_item_rect_size("chemspace_scaffold_hierarchical_dendrogram")
                except Exception:
                    plot_w = 0
                plot_w = max(1, int(plot_w or 0))
                vis_x0 = max(0.5, min(len(labels) + 0.5, float(x0)))
                vis_x1 = max(0.5, min(len(labels) + 0.5, float(x1)))
                if vis_x1 < vis_x0:
                    vis_x0, vis_x1 = vis_x1, vis_x0
                first_idx = max(0, int(np.ceil(vis_x0 - 1.0)))
                last_idx = min(len(labels) - 1, int(np.floor(vis_x1 - 1.0)))
                visible_count = max(0, last_idx - first_idx + 1)
                max_ticks = max(1, plot_w // 42)
                step = _nice_tick_step(visible_count, max_ticks)
                visible_indices = list(range(first_idx, last_idx + 1, step)) if visible_count else []
                ticks = tuple((labels[i], i + 1) for i in visible_indices)
                dpg.set_axis_ticks("chemspace_dendro_x_axis", ticks)

            _refresh_dendrogram_x_ticks()
            dpg.set_axis_ticks(dendro_y_axis, tuple((f"{((1.0 - y) * 100):.0f}", y) for y in np.linspace(0.0, 1.0, 6)))

            x_min, x_max = min(all_x), max(all_x)
            x1 = icoords[0][1] / 10
            x2 = icoords[0][2] / 10
            x_margin = abs((x2 - x1) / 2)
            frame_xmin = x_min - x_margin
            frame_xmax = x_max + (x_margin * 3)
            frame_ymin = -0.01
            frame_ymax = 1.01
            dpg.add_line_series(
                x=[frame_xmin, frame_xmax, frame_xmax, frame_xmin, frame_xmin],
                y=[frame_ymin, frame_ymin, frame_ymax, frame_ymax, frame_ymin],
                tag="chemspace_dendro_frame_series",
                parent=dendro_y_axis,
                label="",
            )
            dpg.bind_item_theme("chemspace_dendro_frame_series", _ensure_chemspace_dendro_frame_theme(state))
            dpg.set_axis_limits(dendro_y_axis, frame_ymin, frame_ymax)
            dpg.set_frame_callback(
                dpg.get_frame_count() + 1,
                lambda: (dpg.fit_axis_data(dendro_x_axis), _refresh_dendrogram_x_ticks(), _refresh_chemspace_dendrogram_highlight(state))
                if dpg.does_item_exist(dendro_x_axis) else None
            )

            dpg.add_drag_line(
                tag="chemspace_cluster_threshold_drag",
                default_value=threshold,
                color=(200, 0, 0, 255),
                thickness=2,
                vertical=False,
                parent=plot,
                callback=lambda s, a: on_chemspace_dendrogram_threshold_change(s, a, state),
            )
            dpg.bind_item_theme("chemspace_scaffold_hierarchical_dendrogram", apply_dendrogram_theme(state))

    label_to_subset = {str(lbl): f"subset_{int(lbl)}" for lbl in labels}

    def _select_clicked_dendrogram_subset() -> None:
        if not dpg.is_item_hovered("chemspace_scaffold_hierarchical_dendrogram"):
            return
        try:
            mouse_x, _ = dpg.get_plot_mouse_pos()
        except Exception:
            return
        if mouse_x is None:
            return
        closest_idx = int(np.argmin([abs(float(mouse_x) - (idx + 1)) for idx in range(len(labels))]))
        clicked_label = labels[closest_idx]
        subset_name = label_to_subset.get(str(clicked_label))
        if subset_name is None:
            return
        state["chemspace_dendrogram_selected_subset"] = subset_name
        _update_dendrogram_details_for_subset(state, subset_name)

    for handler_tag in [
        "chemspace_dendro_move_handler",
        "chemspace_dendro_wheel_handler",
        "chemspace_dendro_left_drag_handler",
        "chemspace_dendro_right_drag_handler",
        "chemspace_dendro_left_click_handler",
    ]:
        if dpg.does_item_exist(handler_tag):
            dpg.delete_item(handler_tag)

    dpg.add_mouse_move_handler(
        parent="handler_registry",
        tag="chemspace_dendro_move_handler",
        callback=lambda s, a: (_refresh_dendrogram_x_ticks() if dpg.is_item_hovered("chemspace_scaffold_hierarchical_dendrogram") else None),
    )
    dpg.add_mouse_wheel_handler(
        parent="handler_registry",
        tag="chemspace_dendro_wheel_handler",
        callback=lambda s, a: (_refresh_dendrogram_x_ticks() if dpg.is_item_hovered("chemspace_scaffold_hierarchical_dendrogram") else None),
    )
    dpg.add_mouse_drag_handler(
        button=dpg.mvMouseButton_Left,
        parent="handler_registry",
        tag="chemspace_dendro_left_drag_handler",
        callback=lambda s, a: (_refresh_dendrogram_x_ticks() if dpg.is_item_hovered("chemspace_scaffold_hierarchical_dendrogram") else None),
    )
    dpg.add_mouse_drag_handler(
        button=dpg.mvMouseButton_Right,
        parent="handler_registry",
        tag="chemspace_dendro_right_drag_handler",
        callback=lambda s, a: (_refresh_dendrogram_x_ticks() if dpg.is_item_hovered("chemspace_scaffold_hierarchical_dendrogram") else None),
    )
    dpg.add_mouse_click_handler(
        button=dpg.mvMouseButton_Left,
        parent="handler_registry",
        tag="chemspace_dendro_left_click_handler",
        callback=lambda s, a: ((_refresh_dendrogram_x_ticks(), _select_clicked_dendrogram_subset()) if dpg.is_item_hovered("chemspace_scaffold_hierarchical_dendrogram") else None),
    )

    cache = _ensure_chemspace_dendro_cache(state)
    cache["linkage"] = linkage_matrix
    cache["labels_input"] = subset_names[:]
    cache["labels_ordered"] = labels[:]
    cache["link_to_seg_ids"] = link_to_seg_ids
    cache["label_to_subset"] = label_to_subset
    cache["frame_ymin"] = frame_ymin
    cache["frame_ymax"] = frame_ymax
    state["chemspace_dendrogram_refresh_colors"] = lambda: _recolor_chemspace_dendrogram_segments(
        state,
        float(state.get("chemspace_cluster_threshold", threshold)),
    )
    state["chemspace_dendrogram_refresh_highlight"] = lambda: _refresh_chemspace_dendrogram_highlight(state)
    set_loading_screen_progress(state, 98)


def draw_pca_plot_2d_common(
    subset: str,
    x_pca: np.ndarray,
    variance_ratio: np.ndarray,
    filtered_mols: list[Any],
    molecule_data: list[dict[str, Any]],
    activity: str,
    activity_label: str,
    labels: np.ndarray,
    activity_values: list[float],
    fp_algorithm: str,
    best_k: int,
    state: dict[str, Any],
) -> None:
    log_event("PCA", "Drawing 'pca' plot", indent=1)
    log_settings(
        "PCA",
        indent=2,
        subset=subset,
        activity=activity_label,
        fingerprint=fp_algorithm,
        clusters=best_k,
        molecules=len(filtered_mols),
        colormap_continuous=state.get("colormap_continuous"),
        colormap_discrete=state.get("colormap_discrete"),
    )
    set_loading_screen_progress(state, 97)

    for tag in [
        "pca_plot",
        "pca_gradient_bar_window",
        "pca_colormap_scale",
        "pca_plot_2d",
        "pca_colormap_scale_wrap",
        "pca_colormap_scale_labels_drawlist",
        "pca_colormap_scale_host",
        "pca_colormap_scale_bar_drawlist",
    ]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    pca_img_width = state["plots_pca_img_width"]
    pca_img_height = round(pca_img_width / 4 * 3)
    pca_context_key = (
        subset,
        activity_label,
        fp_algorithm,
        tuple(int(m.get("Mol_ID")) for m in molecule_data),
    )
    if state.get("pca_inserted_points_context") != pca_context_key:
        state["pca_inserted_points"] = []
        state["pca_inserted_points_context"] = pca_context_key

    state["pca_plot_points"] = []
    for idx, mol_info in enumerate(molecule_data):
        smiles = ""
        try:
            if idx < len(filtered_mols) and filtered_mols[idx] is not None:
                smiles = Chem.MolToSmiles(filtered_mols[idx])
        except Exception:
            smiles = ""
        state["pca_plot_points"].append(
            {
                "x": float(x_pca[idx, 0]),
                "y": float(x_pca[idx, 1]),
                "smiles": smiles,
                "name": mol_info.get("Name", "N/A"),
                "mol_id": mol_info.get("Mol_ID", idx + 1),
                "subset_text": str(mol_info.get("Subset", "N/A")).replace("subset_", "Subset "),
                "activity_text": mol_info.get(activity_label, "N/A") if activity_label != "None" else "N/A",
                "cluster_text": int(labels[idx]) + 1 if labels is not None and len(labels) > idx else "N/A",
            }
        )

    subset_ids_sorted = sorted(set(int(m.get("Subset", -1)) for m in molecule_data if int(m.get("Subset", -1)) >= 0))
    subset_index_map = {sid: idx for idx, sid in enumerate(subset_ids_sorted)}
    cluster_ids_sorted = sorted(set(int(c) for c in labels)) if labels is not None else []
    cluster_index_map = {cid: idx for idx, cid in enumerate(cluster_ids_sorted)}
    act_min = float(np.nanmin(activity_values)) if len(activity_values) else 0.0
    act_max = float(np.nanmax(activity_values)) if len(activity_values) else 1.0
    current_coloring = activity_label

    def _sample_plot_colormap(norm_val: float) -> tuple[int, int, int, int]:
        rgba = dpg.sample_colormap(state["plot_colormaps"][state["colormap_discrete"]], max(0.0, min(1.0, float(norm_val))))
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

    def _current_pca_marker_px() -> int:
        return int(round(_point_size_mode_to_marker_size(state.get("pca_point_size_mode", "Medium"))))

    def _ratio_from_activity(v: Any) -> float:
        if not np.isfinite(v) or act_max == act_min:
            return 0.5
        return float((v - act_min) / (act_max - act_min))

    def _get_inserted_point_color() -> tuple[int, int, int, int]:
        return _sample_plot_colormap(0.0)

    def get_color_cluster(cluster_id: Any) -> tuple[int, int, int, int]:
        if cluster_id not in cluster_index_map:
            return (150, 150, 150, 255)
        return _cycle_discrete_plot_color(state, cluster_index_map[cluster_id])

    def get_color_subset(subset_id: Any) -> tuple[int, int, int, int]:
        if subset_id not in subset_index_map:
            return (150, 150, 150, 255)
        return _cycle_discrete_plot_color(state, subset_index_map[subset_id])

    def _compute_pca_fp_array(smiles: str) -> np.ndarray | None:
        mol = Chem.MolFromSmiles((smiles or "").strip())
        if mol is None:
            return None
        if fp_algorithm == "Morgan Fingerprint":
            fp = GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
        elif fp_algorithm == "RDKit Fingerprint":
            fp = GetRDKitFPGenerator(fpSize=2048).GetFingerprint(mol)
        elif fp_algorithm == "Atom Pair Fingerprint":
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
        elif fp_algorithm == "MACCS Keys":
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif fp_algorithm == "Topological Torsion Fingerprint":
            fp = GetTopologicalTorsionGenerator(fpSize=1024).GetFingerprint(mol)
        elif fp_algorithm == "Pattern Fingerprint":
            fp = Chem.PatternFingerprint(mol)
        elif fp_algorithm == "Layered Fingerprint":
            fp = Chem.LayeredFingerprint(mol, fpSize=2048)
        else:
            return None
        try:
            n_bits = int(fp.GetNumBits())
        except Exception:
            n_bits = int(len(fp))
        arr = np.zeros((n_bits,), dtype=float)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _project_inserted_smiles(smiles: str) -> tuple[float, float] | None:
        pca_model = state.get("pca_projection_model")
        if pca_model is None:
            return None
        fp_arr = _compute_pca_fp_array(smiles)
        if fp_arr is None:
            return None
        coords = pca_model.transform(fp_arr.reshape(1, -1))[0]
        signs = np.asarray(state.get("pca_projection_signs", (1.0, 1.0, 1.0)), dtype=float)
        if signs.size >= 2:
            coords = coords * signs[: coords.shape[0]]
        return float(coords[0]), float(coords[1])

    def _recreate_colormap_scale(parent_tag: str, label: str, colormap: Any, min_v: Any, max_v: Any, fmt: str, discrete_labels: list[str] | None = None) -> None:
        text_color = state["themes"][state["theme_name"]]["Text Color"]
        border_color = state["themes"][state["theme_name"]]["Border Color"]
        for tag in ["pca_colormap_scale_wrap", "pca_colormap_scale_labels_drawlist", "pca_colormap_scale_host", "pca_colormap_scale_bar_drawlist", "pca_colormap_scale"]:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        state["_pca_colormap_scale_layout_cache"] = None

        def _relayout_discrete_colormap_scale(expected_labels: list[str] | None = None, expected_colormap: Any = None) -> None:
            if not expected_labels:
                return
            if not (dpg.does_item_exist("pca_colormap_scale_host") and dpg.does_item_exist("pca_colormap_scale_labels_drawlist") and dpg.does_item_exist("pca_colormap_scale_bar_drawlist")):
                return
            try:
                _, bar_h = dpg.get_item_rect_size("pca_colormap_scale_host")
            except Exception:
                return
            draw_h = int(bar_h or 0)
            if draw_h <= 1:
                return
            label_w = max(34, int(state.get("win_spacer", 8) * 4))
            bar_w = max(20, int(state.get("win_spacer", 8) * 2))
            n_labels = max(1, len(expected_labels))
            step_h = draw_h / n_labels
            visible_indices = set(_visible_discrete_label_indices(n_labels, draw_h))
            dpg.configure_item("pca_colormap_scale_labels_drawlist", width=label_w, height=draw_h)
            dpg.configure_item("pca_colormap_scale_bar_drawlist", width=bar_w, height=draw_h)
            dpg.delete_item("pca_colormap_scale_labels_drawlist", children_only=True)
            dpg.delete_item("pca_colormap_scale_bar_drawlist", children_only=True)
            for idx in range(n_labels):
                seg_y0 = draw_h - ((idx + 1) * step_h)
                seg_y1 = draw_h - (idx * step_h)
                t = 0.5 if n_labels == 1 else idx / (n_labels - 1)
                seg_color = _sample_plot_colormap(t) if expected_colormap is None else dpg.sample_colormap(expected_colormap, t)
                if max(seg_color[0], seg_color[1], seg_color[2]) <= 1.0:
                    seg_color = (
                        int(round(seg_color[0] * 255)),
                        int(round(seg_color[1] * 255)),
                        int(round(seg_color[2] * 255)),
                        int(round((seg_color[3] if len(seg_color) > 3 else 1.0) * 255)),
                    )
                else:
                    seg_color = (
                        int(round(seg_color[0])),
                        int(round(seg_color[1])),
                        int(round(seg_color[2])),
                        int(round(seg_color[3] if len(seg_color) > 3 else 255)),
                    )
                dpg.draw_rectangle((0, seg_y0), (bar_w, seg_y1), fill=seg_color, color=seg_color, thickness=1.0, parent="pca_colormap_scale_bar_drawlist")
                center_y = draw_h - ((idx + 0.5) * step_h)
                if idx in visible_indices:
                    dpg.draw_text((2, center_y - 8), str(expected_labels[idx]), color=text_color, size=16, parent="pca_colormap_scale_labels_drawlist")
                    dpg.draw_line((0, center_y), (bar_w / 2, center_y), color=text_color, thickness=1.0, parent="pca_colormap_scale_bar_drawlist")
            dpg.draw_rectangle((0, 0), (bar_w, draw_h), fill=(0, 0, 0, 0), color=border_color, thickness=1.0, parent="pca_colormap_scale_bar_drawlist")

        with dpg.group(parent=parent_tag, tag="pca_colormap_scale_wrap"):
            if discrete_labels:
                dpg.add_text(label)
                with dpg.group(horizontal=True):
                    label_w = max(34, int(state.get("win_spacer", 8) * 4))
                    with dpg.drawlist(width=label_w, height=1, tag="pca_colormap_scale_labels_drawlist"):
                        pass
                    with dpg.child_window(tag="pca_colormap_scale_host", width=max(20, int(state.get("win_spacer", 8) * 2)), height=-1, border=False, no_scrollbar=True, no_scroll_with_mouse=True):
                        with dpg.drawlist(width=max(20, int(state.get("win_spacer", 8) * 2)), height=1, tag="pca_colormap_scale_bar_drawlist"):
                            pass
                state["pca_relayout_colormap_scale"] = lambda: _relayout_discrete_colormap_scale(discrete_labels, colormap)
                dpg.set_frame_callback(dpg.get_frame_count() + 1, lambda: _relayout_discrete_colormap_scale(discrete_labels, colormap))
            else:
                dpg.add_colormap_scale(tag="pca_colormap_scale", label=label, colormap=colormap, min_scale=min_v, max_scale=max_v, height=-1, mirror=True, format=fmt)
                dpg.bind_item_theme("pca_colormap_scale", apply_colormap_theme(state))
                state["pca_relayout_colormap_scale"] = lambda: None

    with dpg.child_window(parent="pca_window", width=-1, height=-1, no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
        with dpg.group(horizontal=True, tag="pca_plot"):
            with dpg.child_window(border=False, tag="pca_gradient_bar_window", no_scrollbar=True, horizontal_scrollbar=False, no_scroll_with_mouse=True, auto_resize_x=True, height=-1):
                dpg.add_colormap_button(tag="pca_activity_colormap_button", label="Activity", width=-1, indent=3, callback=lambda: set_color_mode(activity_label))
                dpg.add_colormap_button(tag="pca_subset_colormap_button", label="Subset", width=-1, indent=3, callback=lambda: set_color_mode("subset"))
                dpg.add_colormap_button(tag="pca_cluster_colormap_button", label="Cluster", width=-1, indent=3, callback=lambda: set_color_mode("cluster"))
                _recreate_colormap_scale("pca_gradient_bar_window", activity_label, state["colormaps"][state["colormap_continuous"]], act_min, act_max, "%.1f")

                used_clusters = sorted(set(int(c) for c in labels)) if labels is not None else []
                cluster_colors = [get_color_cluster(c) for c in used_clusters] or [(160, 160, 160, 255)]
                dpg.bind_colormap("pca_cluster_colormap_button", _ensure_discrete_colormap("pca_colormap_clusters", cluster_colors))
                subset_colors = [get_color_subset(sid) for sid in subset_ids_sorted] or [(160, 160, 160, 255)]
                dpg.bind_colormap("pca_subset_colormap_button", _ensure_discrete_colormap("pca_colormap_subsets", subset_colors))
                dpg.bind_colormap("pca_activity_colormap_button", state["colormaps"][state["colormap_continuous"]])

            with dpg.plot(label=f"{subset.replace('subset_', 'Subset ')} | {fp_algorithm}-Based PCA", tag="pca_plot_2d", width=-1, height=-1, no_menus=True, no_mouse_pos=True, equal_aspects=False, zoom_rate=0.05):
                dpg.add_plot_axis(dpg.mvXAxis, label=f"PC1 ({variance_ratio[0]:.1f}%)", tag="pca_x_axis", no_tick_marks=True, no_tick_labels=True, no_highlight=True)
                dpg.add_plot_axis(dpg.mvYAxis, label=f"PC2 ({variance_ratio[1]:.1f}%)", tag="pca_y_axis", no_tick_marks=True, no_tick_labels=True, no_highlight=True)
                if x_pca is not None and len(x_pca) and x_pca.shape[1] >= 2:
                    x_vals = x_pca[:, 0]
                    y_vals = x_pca[:, 1]
                    x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
                    y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
                else:
                    x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0
                if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                    x_min, x_max = -1.0, 1.0
                if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                    y_min, y_max = -1.0, 1.0
                margin_x = 0.05 * (x_max - x_min)
                margin_y = 0.05 * (y_max - y_min)
                x_low, x_high = x_min - margin_x, x_max + margin_x
                y_low, y_high = y_min - margin_y, y_max + margin_y
                dpg.add_line_series(x=[x_low, x_high, x_high, x_low, x_low], y=[y_low, y_low, y_high, y_high, y_low], parent="pca_y_axis", tag="pca_frame_outline", label="Bounds")
                with dpg.theme() as pca_frame_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 0), category=dpg.mvThemeCat_Plots)
                set_loading_screen_progress(state, 97.4)

            dpg.bind_item_theme("pca_plot_2d", apply_plot_theme(state))
            dpg.bind_item_theme("pca_frame_outline", pca_frame_theme)
            show_pca_tooltip, hide_pca_tooltip = _build_tooltip_theme("tooltip_pca", (30, 30, 30, 240))
            series_parent = "pca_y_axis"
            created_series: list[str] = []
            inserted_series: list[str] = []
            theme_cache: dict[Any, Any] = {}

            def _clear_pca_series() -> None:
                nonlocal created_series
                for sid in created_series:
                    if dpg.does_item_exist(sid):
                        dpg.delete_item(sid)
                created_series = []

            def _clear_inserted_series() -> None:
                nonlocal inserted_series
                for sid in inserted_series:
                    if dpg.does_item_exist(sid):
                        dpg.delete_item(sid)
                inserted_series = []

            def _get_series_theme(fill_rgba: Any) -> Any:
                marker_size = _point_size_mode_to_marker_size(state.get("pca_point_size_mode", "Medium"))
                key = (tuple(fill_rgba), round(float(marker_size), 3))
                th = theme_cache.get(key)
                if th is not None and dpg.does_item_exist(th):
                    return th
                with dpg.theme() as _th:
                    with dpg.theme_component(dpg.mvScatterSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, fill_rgba, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, fill_rgba, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, marker_size, category=dpg.mvThemeCat_Plots)
                theme_cache[key] = _th
                return _th

            def _inserted_core_theme() -> Any:
                color = _sample_plot_colormap(0.0)
                marker_size = _point_size_mode_to_inserted_marker_size(state.get("pca_point_size_mode", "Medium"))
                key = ("inserted_core", tuple(color), round(float(marker_size), 3))
                th = theme_cache.get(key)
                if th is not None and dpg.does_item_exist(th):
                    return th
                with dpg.theme() as _th:
                    with dpg.theme_component(dpg.mvScatterSeries):
                        dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Cross, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, marker_size, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerWeight, 5.0, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots)
                theme_cache[key] = _th
                return _th

            def _draw_inserted_points() -> None:
                _clear_inserted_series()
                points = list(state.get("pca_inserted_points", []))
                if not points:
                    return
                xs = [float(p["x"]) for p in points]
                ys = [float(p["y"]) for p in points]
                inserted_sid = dpg.add_scatter_series(xs, ys, parent=series_parent, label="")
                dpg.bind_item_theme(inserted_sid, _inserted_core_theme())
                inserted_series.append(inserted_sid)

            def _build_activity_series() -> None:
                _clear_pca_series()
                vals = np.asarray(activity_values, dtype=float)
                n_col = int(state.get("pca_color_bins", 128))
                ratios = np.array([_ratio_from_activity(v) for v in vals], dtype=float)
                edges = np.linspace(0.0, 1.0, n_col + 1)
                idx = np.clip(np.digitize(ratios, edges) - 1, 0, n_col - 1)
                xs_bins = [list() for _ in range(n_col)]
                ys_bins = [list() for _ in range(n_col)]
                for i, (x, y) in enumerate(x_pca[:, :2]):
                    b = int(idx[i])
                    xs_bins[b].append(float(x))
                    ys_bins[b].append(float(y))
                bin_centers = np.linspace(0.0, 1.0, n_col)
                bin_colors = [get_continuous_colormap_color(c, state) for c in bin_centers]
                for i in range(n_col):
                    if not xs_bins[i]:
                        continue
                    sid = dpg.add_scatter_series(xs_bins[i], ys_bins[i], parent=series_parent, label="")
                    try:
                        dpg.configure_item(sid, size=_current_pca_marker_px())
                    except Exception:
                        pass
                    dpg.bind_item_theme(sid, _get_series_theme(tuple(bin_colors[i])))
                    created_series.append(sid)
                set_loading_screen_progress(state, 96)

            def _build_subset_series() -> None:
                _clear_pca_series()
                groups: dict[int, list[list[float]]] = {}
                for (x, y), sid_ in zip(x_pca[:, :2], [int(m.get("Subset", -1)) for m in molecule_data]):
                    groups.setdefault(sid_, [[], []])
                    groups[sid_][0].append(float(x))
                    groups[sid_][1].append(float(y))
                for sid_, (xs_, ys_) in groups.items():
                    sid = dpg.add_scatter_series(xs_, ys_, parent=series_parent, label=f"S{sid_}")
                    try:
                        dpg.configure_item(sid, size=_current_pca_marker_px())
                    except Exception:
                        pass
                    dpg.bind_item_theme(sid, _get_series_theme(get_color_subset(sid_)))
                    created_series.append(sid)

            def _build_cluster_series() -> None:
                _clear_pca_series()
                if labels is None:
                    _build_activity_series()
                    return
                groups: dict[int, list[list[float]]] = {}
                for (x, y), cid in zip(x_pca[:, :2], labels):
                    cid = int(cid)
                    groups.setdefault(cid, [[], []])
                    groups[cid][0].append(float(x))
                    groups[cid][1].append(float(y))
                for cid, (xs_, ys_) in groups.items():
                    sid = dpg.add_scatter_series(xs_, ys_, parent=series_parent, label=f"C{cid + 1}")
                    try:
                        dpg.configure_item(sid, size=_current_pca_marker_px())
                    except Exception:
                        pass
                    dpg.bind_item_theme(sid, _get_series_theme(get_color_cluster(cid)))
                    created_series.append(sid)

            def set_color_mode(mode: str) -> None:
                nonlocal current_coloring
                current_coloring = mode
                if mode == activity_label:
                    _recreate_colormap_scale("pca_gradient_bar_window", activity_label, state["colormaps"][state["colormap_continuous"]], act_min, act_max, "%.1f")
                    _build_activity_series()
                elif mode == "subset":
                    colors = [get_color_subset(sid) for sid in subset_ids_sorted] or [(160, 160, 160, 255)]
                    cm_tag = _ensure_discrete_colormap("pca_colormap_subsets", colors)
                    dpg.bind_colormap("pca_subset_colormap_button", cm_tag)
                    _recreate_colormap_scale("pca_gradient_bar_window", "Subsets", cm_tag, 1, max(1, len(colors)), "%.0f", discrete_labels=[str(i) for i in range(1, len(colors) + 1)])
                    _build_subset_series()
                elif mode == "cluster":
                    used_clusters = sorted(set(int(c) for c in labels)) if labels is not None else []
                    cluster_colors = [get_color_cluster(c) for c in used_clusters] or [(160, 160, 160, 255)]
                    cm_tag = _ensure_discrete_colormap("pca_colormap_clusters", cluster_colors)
                    dpg.bind_colormap("pca_cluster_colormap_button", cm_tag)
                    _recreate_colormap_scale("pca_gradient_bar_window", "Clusters", cm_tag, 1, max(1, len(cluster_colors)), "%.0f", discrete_labels=[str(i) for i in range(1, len(cluster_colors) + 1)])
                    _build_cluster_series()
                else:
                    _build_activity_series()
                _draw_inserted_points()

            _build_activity_series()
            set_loading_screen_progress(state, 97.8)
            state["pca_refresh_colors"] = lambda: set_color_mode(current_coloring)

    with dpg.child_window(parent="pca_details_window", width=-1, height=-1, no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
        pca_render_scale = 1.8
        pca_render_w = int(round(pca_img_width * pca_render_scale))
        pca_render_h = int(round(pca_img_height * pca_render_scale))
        empty_data = np.zeros((pca_render_w * pca_render_h * 4,), dtype=np.float32)
        if not dpg.does_item_exist("pca_mol_image_texture"):
            dpg.add_dynamic_texture(pca_render_w, pca_render_h, empty_data, tag="pca_mol_image_texture", parent="texture_registry")
        else:
            dpg.set_value("pca_mol_image_texture", empty_data)
        dpg.add_image("pca_mol_image_texture", width=pca_img_width, height=pca_img_height, tag="pca_mol_image_widget", border_color=(0, 0, 0, 0))
        register_responsive_image(state, image_tag="pca_mol_image_widget", parent_tag="pca_details_window", aspect_ratio=0.75, tab="pca_tab")
        export_png_popup("pca_mol_image_widget", "pca_mol_image_texture", state)
        dpg.add_text("", tag="pca_details_name_text")
        dpg.add_text("", tag="pca_details_subset_text")
        dpg.add_text("", tag="pca_details_activity_text")
        dpg.add_text("", tag="pca_details_coords_text")
        set_loading_screen_progress(state, 98)

        def _set_pca_details_lines(name_line: str = "", subset_line: str = "", activity_line: str = "", coords_line: str = "") -> None:
            dpg.set_value("pca_details_name_text", name_line)
            dpg.set_value("pca_details_subset_text", subset_line)
            dpg.set_value("pca_details_activity_text", activity_line)
            dpg.set_value("pca_details_coords_text", coords_line)

        def _pca_point_color(point_idx: int) -> tuple[int, int, int, int]:
            if current_coloring == "subset":
                return get_color_subset(int(molecule_data[point_idx].get("Subset", -1)))
            if current_coloring == "cluster":
                cluster_id = int(labels[point_idx]) if labels is not None else -1
                return get_color_cluster(cluster_id if cluster_id >= 0 else 0)
            value = float(activity_values[point_idx]) if len(activity_values) > point_idx else 0.0
            return get_continuous_colormap_color(_ratio_from_activity(value), state)

        def update_pca_tooltip(active_label: str) -> None:
            if state["current_chemspace_subtab"] != "pca_tab":
                hide_pca_tooltip()
                return
            if not dpg.does_item_exist("pca_plot_2d"):
                hide_pca_tooltip()
                return
            mx_local, my_local = dpg.get_mouse_pos(local=True)
            plot_min = dpg.get_item_rect_min("pca_plot_2d")
            plot_max = dpg.get_item_rect_max("pca_plot_2d")
            if mx_local < plot_min[0] or mx_local > plot_max[0] or my_local < plot_min[1] or my_local > plot_max[1]:
                hide_pca_tooltip()
                return
            mouse_pos = dpg.get_plot_mouse_pos()
            plot_w_px, plot_h_px = dpg.get_item_rect_size("pca_plot_2d")
            x_lower, x_upper = dpg.get_axis_limits("pca_x_axis")
            y_lower, y_upper = dpg.get_axis_limits("pca_y_axis")
            x_range = max(1e-12, x_upper - x_lower)
            y_range = max(1e-12, y_upper - y_lower)
            radius_px = max(4, int(round(0.8 * _current_pca_marker_px())))
            tol_x = radius_px * (x_range / max(1, plot_w_px))
            tol_y = radius_px * (y_range / max(1, plot_h_px))
            nearest_idx = None
            nearest_inserted_idx = None
            best_d2 = 1e300
            for i, (x, y) in enumerate(x_pca[:, :2]):
                dx = x - mouse_pos[0]
                dy = y - mouse_pos[1]
                if abs(dx) < tol_x and abs(dy) < tol_y:
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best_d2 = d2
                        nearest_idx = i
            inserted_points = list(state.get("pca_inserted_points", []))
            for i, point in enumerate(inserted_points):
                dx = float(point["x"]) - mouse_pos[0]
                dy = float(point["y"]) - mouse_pos[1]
                if abs(dx) < tol_x and abs(dy) < tol_y:
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best_d2 = d2
                        nearest_idx = None
                        nearest_inserted_idx = i
            if nearest_inserted_idx is not None:
                point = inserted_points[nearest_inserted_idx]
                tooltip_text = f"Inserted mol {point['inserted_id']}\nPC1: {float(point['x']):.2f}\nPC2: {float(point['y']):.2f}"
                show_local, _ = _build_tooltip_theme("tooltip_pca", _get_inserted_point_color())
                mx, my = dpg.get_mouse_pos(local=False)
                show_local(tooltip_text, (int(mx) + 14, int(my) + 14))
            elif nearest_idx is not None:
                val = molecule_data[nearest_idx].get(active_label, "N/A")
                try:
                    activity_text = f"{active_label}: {float(val):.2f}"
                except Exception:
                    activity_text = f"{active_label}: {val}"
                tooltip_text = (
                    f"Mol_ID: {molecule_data[nearest_idx]['Mol_ID']}\n"
                    f"Name: {molecule_data[nearest_idx].get('Name', 'N/A')}\n"
                    f"{activity_text}\n"
                    f"PC1: {x_pca[nearest_idx, 0]:.2f}\n"
                    f"PC2: {x_pca[nearest_idx, 1]:.2f}\n"
                    f"Subset: {molecule_data[nearest_idx].get('Subset', 'N/A')}\n"
                    f"Cluster: {labels[nearest_idx] + 1 if labels is not None else 'N/A'}"
                )
                show_local, _ = _build_tooltip_theme("tooltip_pca", _pca_point_color(nearest_idx))
                mx, my = dpg.get_mouse_pos(local=False)
                show_local(tooltip_text, (int(mx) + 14, int(my) + 14))
            else:
                hide_pca_tooltip()

        def update_pca_image(active_label: str) -> None:
            if state["current_chemspace_subtab"] != "pca_tab":
                return
            mouse_pos = dpg.get_plot_mouse_pos()
            x_lower, x_upper = dpg.get_axis_limits("pca_x_axis")
            y_lower, y_upper = dpg.get_axis_limits("pca_y_axis")
            tol_x = 0.005 * (x_upper - x_lower)
            tol_y = 0.005 * (y_upper - y_lower)
            nearest_idx = None
            nearest_inserted_idx = None
            best_d2 = 1e300
            for i, (x, y) in enumerate(x_pca[:, :2]):
                dx = x - mouse_pos[0]
                dy = y - mouse_pos[1]
                if abs(dx) < tol_x and abs(dy) < tol_y:
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best_d2 = d2
                        nearest_idx = i
            inserted_points = list(state.get("pca_inserted_points", []))
            for i, point in enumerate(inserted_points):
                dx = float(point["x"]) - mouse_pos[0]
                dy = float(point["y"]) - mouse_pos[1]
                if abs(dx) < tol_x and abs(dy) < tol_y:
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best_d2 = d2
                        nearest_idx = None
                        nearest_inserted_idx = i
            if nearest_inserted_idx is not None:
                point = inserted_points[nearest_inserted_idx]
                mol = Chem.MolFromSmiles(str(point["smiles"]))
                if mol is None:
                    return
                rdDepictor.Compute2DCoords(mol)
                drawer = rdMolDraw2D.MolDraw2DCairo(pca_render_w, pca_render_h)
                opts = drawer.drawOptions()
                opts.padding = 0.025
                opts.bondLineWidth = 1
                opts.minFontSize = 1
                opts.legendFontSize = 14
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                drawer.FinishDrawing()
                mol_arr = (np.array(pilImage.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")) / 255.0).astype(np.float32).flatten()
                dpg.set_value("pca_mol_image_texture", mol_arr)
                _set_pca_details_lines(name_line=f"Inserted mol {point['inserted_id']}", coords_line=f"PC1 = {float(point['x']):.2f}  |  PC2 = {float(point['y']):.2f}")
            elif nearest_idx is not None:
                mol = filtered_mols[nearest_idx]
                rdDepictor.Compute2DCoords(mol)
                drawer = rdMolDraw2D.MolDraw2DCairo(pca_render_w, pca_render_h)
                opts = drawer.drawOptions()
                opts.padding = 0.025
                opts.bondLineWidth = 1
                opts.minFontSize = 1
                opts.legendFontSize = 14
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                drawer.FinishDrawing()
                mol_arr = (np.array(pilImage.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")) / 255.0).astype(np.float32).flatten()
                dpg.set_value("pca_mol_image_texture", mol_arr)
                mol_id = molecule_data[nearest_idx]["Mol_ID"]
                mol_name = molecule_data[nearest_idx].get("Name", "N/A")
                val = molecule_data[nearest_idx].get(active_label, "N/A")
                try:
                    activity_val = f"{float(val):.2f}"
                except Exception:
                    activity_val = str(val)
                sub_id = molecule_data[nearest_idx].get("Subset", "N/A")
                _set_pca_details_lines(
                    name_line=f"Mol {mol_id}  |  Name: {mol_name}",
                    subset_line=f"Subset {sub_id}",
                    activity_line=f"{active_label} = {activity_val}",
                    coords_line=f"PC1 = {x_pca[nearest_idx, 0]:.2f}  |  PC2 = {x_pca[nearest_idx, 1]:.2f}",
                )

    for tag in [
        "pca_click_handler",
        "pca_mouse_move_handler",
        "pca_mouse_wheel_handler",
        "pca_left_drag_handler",
        "pca_middle_drag_handler",
        "pca_mcs_middle_down_handler",
        "pca_mcs_middle_release_handler",
        "pca_right_drag_handler",
        "pca_right_down_handler",
        "pca_right_release_handler",
        "pca_insert_popup_dismiss_handler",
    ]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    dpg.add_mouse_click_handler(
        button=dpg.mvMouseButton_Left,
        parent="handler_registry",
        tag="pca_click_handler",
        callback=lambda s, a: (
            state.__setitem__("pca_mcs_selection_just_finished", False)
            if state.get("pca_mcs_selection_just_finished")
            else (update_pca_image(activity_label) if dpg.is_item_hovered("pca_plot_2d") else None)
        ),
    )
    dpg.add_mouse_move_handler(parent="handler_registry", tag="pca_mouse_move_handler", callback=lambda s, a: update_pca_tooltip(activity_label))
    dpg.add_mouse_wheel_handler(parent="handler_registry", tag="pca_mouse_wheel_handler", callback=lambda s, a: update_pca_tooltip(activity_label))
    dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Middle, parent="handler_registry", tag="pca_mcs_middle_down_handler", callback=lambda s, a: _begin_pca_mcs_selection())
    dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, parent="handler_registry", tag="pca_middle_drag_handler", callback=lambda s, a: (_update_pca_mcs_selection_drag(), update_pca_tooltip(activity_label)))
    dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Middle, parent="handler_registry", tag="pca_mcs_middle_release_handler", callback=lambda s, a: _finish_pca_mcs_selection())
    dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, parent="handler_registry", tag="pca_right_drag_handler", callback=lambda s, a: (state.__setitem__("pca_right_click_dragged", True), update_pca_tooltip(activity_label)))

    def _draw_inserted_smiles_from_input() -> None:
        if not dpg.does_item_exist("pca_insert_smiles_input"):
            return
        smiles = str(dpg.get_value("pca_insert_smiles_input") or "").strip()
        if not smiles:
            return
        coords = _project_inserted_smiles(smiles)
        if coords is None:
            log_event("PCA", "Failed to project inserted SMILES", indent=2)
            log_settings("PCA", indent=3, smiles=smiles, fingerprint=fp_algorithm)
            return
        inserted_points = state.setdefault("pca_inserted_points", [])
        inserted_points.append({"inserted_id": len(inserted_points) + 1, "smiles": smiles, "x": float(coords[0]), "y": float(coords[1])})
        log_event("PCA", "Projected external molecule into PCA space", indent=2)
        log_settings("PCA", indent=3, inserted_id=len(inserted_points), pc1=f"{coords[0]:.3f}", pc2=f"{coords[1]:.3f}")
        state["pca_refresh_colors"]()

    def _delete_all_inserted_smiles() -> None:
        state["pca_inserted_points"] = []
        log_event("PCA", "Removed all inserted PCA molecules", indent=2)
        state["pca_refresh_colors"]()

    def _begin_pca_mcs_selection() -> None:
        if state.get("pca_mcs_selection_in_progress"):
            return
        if state.get("current_chemspace_subtab") != "pca_tab":
            return
        if not dpg.does_item_exist("pca_plot_2d") or not dpg.is_item_hovered("pca_plot_2d"):
            return
        plot_mouse_raw = dpg.get_plot_mouse_pos()
        screen_mouse_raw = tuple(dpg.get_mouse_pos(local=False))
        plot_mouse = _clamp_plot_pos_to_axes("pca_x_axis", "pca_y_axis", plot_mouse_raw)
        screen_mouse = _clamp_screen_pos_to_plot("pca_plot_2d", screen_mouse_raw)
        if not plot_mouse:
            return
        state["pca_mcs_selection_in_progress"] = True
        state["pca_mcs_start_plot_pos"] = (float(plot_mouse[0]), float(plot_mouse[1]))
        state["pca_mcs_start_screen_pos"] = (float(screen_mouse[0]), float(screen_mouse[1]))
        _draw_selection_overlay("pca", state["pca_mcs_start_screen_pos"], state["pca_mcs_start_screen_pos"])

    def _update_pca_mcs_selection_drag() -> None:
        if not state.get("pca_mcs_selection_in_progress"):
            return
        start_screen = state.get("pca_mcs_start_screen_pos")
        if not isinstance(start_screen, (tuple, list)) or len(start_screen) < 2:
            return
        current_screen = _clamp_screen_pos_to_plot("pca_plot_2d", tuple(dpg.get_mouse_pos(local=False)))
        _draw_selection_overlay("pca", (float(start_screen[0]), float(start_screen[1])), (float(current_screen[0]), float(current_screen[1])))

    def _finish_pca_mcs_selection() -> None:
        if not state.get("pca_mcs_selection_in_progress"):
            return
        start_plot = state.get("pca_mcs_start_plot_pos")
        end_plot = _clamp_plot_pos_to_axes("pca_x_axis", "pca_y_axis", dpg.get_plot_mouse_pos())
        _clear_selection_overlay("pca")
        state["pca_mcs_selection_in_progress"] = False
        state["pca_mcs_selection_just_finished"] = True
        if not isinstance(start_plot, (tuple, list)) or len(start_plot) < 2 or not end_plot:
            return
        x0, y0 = float(start_plot[0]), float(start_plot[1])
        x1, y1 = float(end_plot[0]), float(end_plot[1])
        if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
            return
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        selected_records = [
            point for point in list(state.get("pca_plot_points", []) or [])
            if xmin <= float(point["x"]) <= xmax and ymin <= float(point["y"]) <= ymax
        ]
        draw_loading_screen(state, bg=False)
        try:
            show_features = bool(state.get("pca_mcs_features", True))
            mcs_result = _mcs_result_from_smiles_list(
                [str(r.get("smiles", "")) for r in selected_records],
                state.get("pca_mcs_timeout", "10s"),
                include_features=show_features,
            )
            mcs_mol = mcs_result.get("mol")
            feature_annotations = dict(mcs_result.get("feature_annotations", {}) or {})
            _set_texture_from_mol(
                "pca_mol_image_texture",
                mcs_mol,
                pca_img_width,
                transparent=False,
                feature_annotations=feature_annotations,
            )
            _set_pca_details_lines(
                name_line=_format_mcs_selection_header(
                    len(selected_records),
                    bool(mcs_result.get("interrupted")),
                    str(mcs_result.get("timeout_setting", "10s")),
                ),
                subset_line=_summarize_selection_values(selected_records, "subset_text", "Subset")
                + "\n"
                + _summarize_selection_values(selected_records, "cluster_text", "Cluster"),
                activity_line=(
                    f"MCS atoms: {mcs_mol.GetNumAtoms()}"
                    + (f"  |  Feature labels: {int(sum(len(v) for v in feature_annotations.values()))}" if show_features and feature_annotations else "")
                ) if mcs_mol is not None else "MCS not found",
                coords_line=_average_activity_line(selected_records, activity_label, state),
            )
        finally:
            if dpg.does_item_exist("cover_layer"):
                dpg.delete_item("cover_layer")

    def _set_pca_point_size_mode(app_data: Any) -> None:
        state["pca_point_size_mode"] = _save_point_size_setting(state, "pca_point_size", app_data)
        state["pca_refresh_colors"]()

    def _set_pca_mcs_timeout_mode(app_data: Any) -> None:
        state["pca_mcs_timeout"] = _save_mcs_timeout_setting(state, "pca_mcs_timeout", app_data)

    def _set_pca_mcs_features_mode(app_data: Any) -> None:
        state["pca_mcs_features"] = _save_mcs_features_setting(state, "pca_mcs_features", app_data)

    def _build_pca_popup_specific_items() -> None:
        add_chemspace_plot_specific_popup_controls(
            state=state,
            point_size_state_key="pca_point_size_mode",
            point_size_combo_tag="pca_point_size_combo",
            point_size_callback=_set_pca_point_size_mode,
            mcs_timeout_state_key="pca_mcs_timeout",
            mcs_timeout_combo_tag="pca_mcs_timeout_combo",
            mcs_timeout_callback=_set_pca_mcs_timeout_mode,
            mcs_features_state_key="pca_mcs_features",
            mcs_features_checkbox_tag="pca_mcs_features_checkbox",
            mcs_features_callback=_set_pca_mcs_features_mode,
            input_tag="pca_insert_smiles_input",
            draw_callback=_draw_inserted_smiles_from_input,
            delete_callback=_delete_all_inserted_smiles,
        )

    register_plot_context_popup(
        state,
        context_key="pca_plot_context",
        plot_tag="pca_plot_2d",
        x_axis_tag="pca_x_axis",
        y_axis_tag="pca_y_axis",
        theme_kind="plot",
        specific_builder=_build_pca_popup_specific_items,
    )
    state["pca_refresh_colors"]()


def _descriptor_items() -> list[str]:
    return [
        "MW", "logP", "HBA", "HBD", "RotBonds", "TPSA", "MolarRefractivity",
        "fraction_csp3", "NumRings", "NumAromaticRings", "NumAliphaticRings",
        "NumSaturatedRings", "Kappa1", "Kappa2", "Kappa3",
        "Chi0", "Chi1", "Chi2", "Chi3", "Chi4",
    ]


def _parse_descriptor_activity_value(val: Any, read_undefined: bool) -> Any:
    if pd.isna(val):
        return None
    val = str(val).strip()
    if not val:
        return None
    if read_undefined:
        for op in ["<=", ">=", "<", ">"]:
            if val.startswith(op):
                val = val[len(op):].strip()
                break
    else:
        if any(val.startswith(op) for op in ["<=", ">=", "<", ">"]):
            return None
    try:
        return float(val)
    except ValueError:
        return None


def _descriptor_unit(label: str, state: dict[str, Any], descriptors: list[str] | None = None) -> str:
    descriptors = descriptors or _descriptor_items()
    if label in descriptors:
        return "Da" if label == "MW" else "Å²" if label == "TPSA" else ""
    if label.startswith("p") and label[1:] in state["nM_activity_types"]:
        return ""
    if label in state["nM_activity_types"]:
        return "nM"
    if label in state["percent_activities"]:
        return "%"
    if label in state["ug/mL_activities"]:
        return "μg/mL"
    if label in state["uM/min_activities"]:
        return "μM/min"
    return ""


def _get_active_continuous_colormap_rgba(state: dict[str, Any]) -> list[list[int]]:
    defs = state.get("continuous_colormap_defs", {}) or {}
    name = state.get("colormap_continuous", "")
    colors = defs.get(name)
    if isinstance(colors, list) and len(colors) >= 2:
        return colors
    for fallback in defs.values():
        if isinstance(fallback, list) and len(fallback) >= 2:
            return fallback
    return [
        [255, 0, 0, 255],
        [255, 165, 0, 255],
        [255, 255, 0, 255],
        [50, 205, 50, 255],
        [0, 100, 0, 255],
    ]


def _interpolate_continuous_rgb(value: float, min_value: float, max_value: float, colors: list[list[int]]) -> tuple[int, int, int]:
    if max_value == min_value:
        norm = 0.5
    else:
        norm = (value - min_value) / (max_value - min_value)
        norm = max(0.0, min(1.0, norm))
    if len(colors) == 1:
        r, g, b = colors[0][:3]
        return int(r), int(g), int(b)
    scaled = norm * (len(colors) - 1)
    index = int(scaled)
    t = scaled - index
    if index >= len(colors) - 1:
        r, g, b = colors[-1][:3]
        return int(r), int(g), int(b)
    r1, g1, b1 = colors[index][:3]
    r2, g2, b2 = colors[index + 1][:3]
    return (
        int(round(r1 + (r2 - r1) * t)),
        int(round(g1 + (g2 - g1) * t)),
        int(round(b1 + (b2 - b1) * t)),
    )


def _plotly_colorscale_from_rgba(colors: list[list[int]]) -> list[list[Any]]:
    if len(colors) == 1:
        r, g, b = colors[0][:3]
        return [[0.0, f"rgb({r},{g},{b})"], [1.0, f"rgb({r},{g},{b})"]]
    steps = len(colors) - 1
    return [[idx / steps, f"rgb({color[0]},{color[1]},{color[2]})"] for idx, color in enumerate(colors)]


def draw_descriptors_2d(subset: str, data: Any, read_undefined: bool, state: dict[str, Any]) -> Any:
    log_event("ChemSpace", "Drawing 'descriptors' plot", indent=1)
    set_loading_screen_progress(state, 10)

    descriptors = _descriptor_items()
    x_col = dpg.get_value("descriptors_axis_x_combo")
    y_col = dpg.get_value("descriptors_axis_y_combo")
    log_settings("ChemSpace", indent=2, subset=subset, x=x_col, y=y_col, dimension="2D", include_undefined=read_undefined, molecules=len(data))

    for col in [x_col, y_col]:
        if col not in descriptors:
            data[col] = data[col].apply(lambda v: _parse_descriptor_activity_value(v, read_undefined))
            if col in state["nM_activity_types"]:
                new_col = f"p{col}"
                data[new_col] = -np.log10(data[col].replace(0, np.nan) * 1e-9)
                if col == x_col:
                    x_col = new_col
                else:
                    y_col = new_col
    set_loading_screen_progress(state, 22)

    x_label = x_col
    y_label = y_col
    df = data[data[x_col].notna() & data[y_col].notna()]
    set_loading_screen_progress(state, 32)

    molecule_data = [{"Mol_ID": int(row["Mol_sub_ID"]), "MolName": row["MolName"] if pd.notna(row["MolName"]) else "", "Subset": int(row["Subset"])} for _, row in df.iterrows()]
    set_loading_screen_progress(state, 40)

    plot_width = state["plots_main_win_width"]
    plot_height = state["plots_main_win_height"]

    with dpg.child_window(parent="descriptors_window", width=-1, height=-1, no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
        with dpg.plot(label=f"{subset.replace('subset_', 'Subset ')} | {x_label} vs {y_label}", tag="descriptors_2d", no_menus=True, no_mouse_pos=True, equal_aspects=False, zoom_rate=0.05, width=-1, height=-1):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast, horizontal=False)
            dpg.add_plot_axis(dpg.mvXAxis, label=x_label, tag="descriptors_x_axis", no_highlight=True)
            dpg.add_plot_axis(dpg.mvYAxis, label=y_label, no_tick_marks=True, tag="descriptors_y_axis", no_highlight=True)

            x_data = df[x_label].tolist()
            y_data = df[y_label].tolist()
            x_arr = np.asarray(x_data, dtype=float)
            y_arr = np.asarray(y_data, dtype=float)
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_arr, y_arr = x_arr[mask], y_arr[mask]
            uniq_x = np.unique(x_arr)
            use_bins = uniq_x.size > 50
            xs, ys, ws = [], [], []
            if not use_bins:
                xs = np.sort(uniq_x)
                ys = np.array([y_arr[x_arr == xv].mean() for xv in xs], dtype=float)
                ws = np.array([int((x_arr == xv).sum()) for xv in xs], dtype=float)
            else:
                n = x_arr.size
                q25, q75 = np.percentile(x_arr, [25, 75])
                iqr = max(q75 - q25, 1e-12)
                bin_w = 2.0 * iqr / (n ** (1 / 3))
                if not np.isfinite(bin_w) or bin_w <= 0:
                    bin_w = (x_arr.max() - x_arr.min()) / 20.0 if x_arr.max() > x_arr.min() else 1.0
                nbins = int(np.clip(np.ceil((x_arr.max() - x_arr.min()) / bin_w), 8, 120))
                edges = np.linspace(x_arr.min(), x_arr.max(), nbins + 1)
                bin_id = np.digitize(x_arr, edges, right=False) - 1
                bin_id = np.clip(bin_id, 0, nbins - 1)
                for b in range(nbins):
                    m = bin_id == b
                    if not m.any():
                        continue
                    xs.append(float(x_arr[m].mean()))
                    ys.append(float(y_arr[m].mean()))
                    ws.append(float(m.sum()))
                xs = np.asarray(xs, dtype=float)
                ys = np.asarray(ys, dtype=float)
                ws = np.asarray(ws, dtype=float)
            order = np.argsort(xs, kind="stable")
            xs, ys, ws = xs[order], ys[order], ws[order]
            if xs.size:
                ux, inv = np.unique(xs, return_inverse=True)
                if ux.size != xs.size:
                    y_acc = np.zeros(ux.size, dtype=float)
                    w_acc = np.zeros(ux.size, dtype=float)
                    for i, g in enumerate(inv):
                        y_acc[g] += ys[i] * ws[i]
                        w_acc[g] += ws[i]
                    ys = y_acc / np.maximum(w_acc, 1.0)
                    ws = w_acc
                    xs = ux
            descriptors_scatter_series = dpg.add_scatter_series(x_data, y_data, parent="descriptors_y_axis", label="Data Points")
            with dpg.tooltip(descriptors_scatter_series, tag="tooltip_scatter"):
                dpg.add_text("", tag="tooltip_text_scatter")
            if len(x_data) and len(y_data):
                x_min, x_max = float(np.nanmin(x_data)), float(np.nanmax(x_data))
                y_min, y_max = float(np.nanmin(y_data)), float(np.nanmax(y_data))
            else:
                x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
            if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                x_min, x_max = 0.0, 1.0
            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                y_min, y_max = 0.0, 1.0
            margin_x = 0.05 * (x_max - x_min)
            margin_y = 0.05 * (y_max - y_min)
            x_low, x_high = x_min - margin_x, x_max + margin_x
            y_low, y_high = y_min - margin_y, y_max + margin_y
            dpg.add_line_series(x=[x_low, x_high, x_high, x_low, x_low], y=[y_low, y_low, y_high, y_high, y_low], parent="descriptors_y_axis", tag="descriptors_frame_outline", label="")
            if dpg.does_item_exist("descriptors_trend_line"):
                dpg.delete_item("descriptors_trend_line")
            if xs is not None and ys is not None and len(xs) >= 2:
                dpg.add_line_series(xs.tolist(), ys.tolist(), parent="descriptors_y_axis", tag="descriptors_trend_line", label="Mean trend")
            set_loading_screen_progress(state, 78)
            with dpg.theme() as descriptors_frame_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 0), category=dpg.mvThemeCat_Plots)
            dpg.bind_colormap("descriptors_2d", state["plot_colormaps"][state["colormap_discrete"]])
            dpg.bind_item_theme("descriptors_2d", apply_plot_theme(state))
            dpg.bind_item_theme("descriptors_frame_outline", descriptors_frame_theme)
            register_plot_context_popup(
                state,
                context_key="descriptors_plot_context",
                plot_tag="descriptors_2d",
                x_axis_tag="descriptors_x_axis",
                y_axis_tag="descriptors_y_axis",
                theme_kind="plot",
            )
            set_loading_screen_progress(state, 86)

    with dpg.child_window(parent="descriptors_details_window", width=-1, height=-1, no_scrollbar=False, horizontal_scrollbar=False, no_scroll_with_mouse=True, border=False):
        scatter_img_width = state["plots_descriptors_img_width"]
        scatter_img_height = round(scatter_img_width / 4 * 3)
        scatter_render_scale = 1.8
        scatter_render_w = int(round(scatter_img_width * scatter_render_scale))
        scatter_render_h = int(round(scatter_img_height * scatter_render_scale))
        empty_data = np.zeros((scatter_render_w * scatter_render_h * 4,), dtype=np.float32)
        if not dpg.does_item_exist("descriptors_mol_image_texture"):
            dpg.add_dynamic_texture(scatter_render_w, scatter_render_h, empty_data, tag="descriptors_mol_image_texture", parent="texture_registry")
        else:
            dpg.set_value("descriptors_mol_image_texture", empty_data)
        dpg.add_image("descriptors_mol_image_texture", width=scatter_img_width, height=scatter_img_height, tag="descriptors_mol_image_widget", border_color=(0, 0, 0, 0))
        register_responsive_image(state, image_tag="descriptors_mol_image_widget", parent_tag="descriptors_details_window", aspect_ratio=0.75, tab="descriptors_tab")
        export_png_popup("descriptors_mol_image_widget", "descriptors_mol_image_texture", state)
        dpg.add_text("", tag="descriptors_mol_label")
        set_loading_screen_progress(state, 94)

    def _nearest_idx() -> Any:
        mouse_pos = dpg.get_plot_mouse_pos()
        x_lower, x_upper = dpg.get_axis_limits("descriptors_x_axis")
        y_lower, y_upper = dpg.get_axis_limits("descriptors_y_axis")
        x_range = x_upper - x_lower or 1
        y_range = y_upper - y_lower or 1
        plot_width_cm = 24.6
        plot_height_cm = plot_width_cm * (plot_height / plot_width)
        tooltip_distance_x = 0.2 / (plot_width_cm / x_range)
        tooltip_distance_y = 0.2 / (plot_height_cm / y_range)
        nearest_idx = None
        min_dist = float("inf")
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            dist_x = abs(x - mouse_pos[0])
            dist_y = abs(y - mouse_pos[1])
            if dist_x < tooltip_distance_x and dist_y < tooltip_distance_y:
                dist = np.hypot(dist_x, dist_y)
                if dist < min_dist:
                    nearest_idx = i
                    min_dist = dist
        return nearest_idx

    def update_scatter_tooltip() -> Any:
        if state["current_chemspace_subtab"] != "descriptors_tab":
            return
        nearest_idx = _nearest_idx()
        if nearest_idx is not None:
            x_unit = _descriptor_unit(x_label, state, descriptors)
            y_unit = _descriptor_unit(y_label, state, descriptors)
            dpg.set_value("tooltip_text_scatter", f"ID: {molecule_data[nearest_idx]['Mol_ID']}\nName: {molecule_data[nearest_idx]['MolName']}\nSubset: {molecule_data[nearest_idx]['Subset']}\n{x_label}: {x_data[nearest_idx]:.2f} {x_unit}\n{y_label}: {y_data[nearest_idx]:.2f} {y_unit}")
            dpg.show_item("tooltip_scatter")
        else:
            dpg.hide_item("tooltip_scatter")

    def update_scatter_image() -> Any:
        if state["current_chemspace_subtab"] != "descriptors_tab":
            return
        nearest_idx = _nearest_idx()
        if nearest_idx is None:
            return
        x_unit = _descriptor_unit(x_label, state, descriptors)
        y_unit = _descriptor_unit(y_label, state, descriptors)
        smiles = df["Mol"].iloc[nearest_idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            drawer = rdMolDraw2D.MolDraw2DCairo(scatter_render_w, scatter_render_h)
            opts = drawer.drawOptions()
            opts.padding = 0.025
            opts.bondLineWidth = 1
            opts.minFontSize = 1
            opts.legendFontSize = 14
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            png_bytes = drawer.GetDrawingText()
            mol_img = pilImage.open(io.BytesIO(png_bytes)).convert("RGBA")
            mol_arr = (np.array(mol_img) / 255.0).astype(np.float32).flatten()
            dpg.set_value("descriptors_mol_image_texture", mol_arr)
            dpg.set_value("descriptors_mol_label", f"Mol {molecule_data[nearest_idx]['Mol_ID']}  |  {x_label}: {x_data[nearest_idx]:.2f} {x_unit}  |  {y_label}: {y_data[nearest_idx]:.2f} {y_unit}")

    if dpg.does_item_exist("descriptors_click_handler"):
        dpg.delete_item("descriptors_click_handler")
    if dpg.does_item_exist("descriptors_mouse_move_handler"):
        dpg.delete_item("descriptors_mouse_move_handler")
    dpg.add_mouse_click_handler(parent="handler_registry", tag="descriptors_click_handler", callback=lambda s, a: update_scatter_image())
    dpg.add_mouse_move_handler(parent="handler_registry", tag="descriptors_mouse_move_handler", callback=lambda s, a: update_scatter_tooltip())


def draw_descriptors_3d(data: Any, read_undefined: bool, state: dict[str, Any]) -> Any:
    log_event("ChemSpace", "Drawing 'descriptors' plot", indent=1)
    set_loading_screen_progress(state, 10)
    x_col = dpg.get_value("descriptors_axis_x_combo")
    y_col = dpg.get_value("descriptors_axis_y_combo")
    z_col = dpg.get_value("descriptors_axis_z_combo")
    colorscale = dpg.get_value("descriptors_color_combo")
    subset = dpg.get_value("descriptors_subset_choice")
    log_settings("ChemSpace", indent=2, subset=subset, x=x_col, y=y_col, z=z_col, color=colorscale, dimension="3D", include_undefined=read_undefined, molecules=len(data), colormap=state.get("colormap_continuous"))
    set_loading_screen_progress(state, 15)

    descriptors = _descriptor_items()
    labels = {}
    for col in [x_col, y_col, z_col, colorscale]:
        if col not in descriptors:
            data[col] = data[col].apply(lambda v: _parse_descriptor_activity_value(v, read_undefined))
            if col in state["nM_activity_types"]:
                new_col = f"p{col}"
                data[new_col] = -np.log10(data[col].replace(0, np.nan) * 1e-9)
                labels[col] = new_col
            else:
                labels[col] = col
        else:
            labels[col] = col
    set_loading_screen_progress(state, 28)

    x_col, y_col, z_col, colorscale = labels[x_col], labels[y_col], labels[z_col], labels[colorscale]
    df = data[data[x_col].notna() & data[y_col].notna() & data[z_col].notna() & data[colorscale].notna()]
    set_loading_screen_progress(state, 40)
    x_unit = _descriptor_unit(x_col, state, descriptors)
    y_unit = _descriptor_unit(y_col, state, descriptors)
    z_unit = _descriptor_unit(z_col, state, descriptors)
    colorscale_unit = _descriptor_unit(colorscale, state, descriptors)
    molecule_data = [{
        "Mol_ID": int(row["Mol_sub_ID"]),
        "MolName": row["MolName"] if pd.notna(row["MolName"]) else "",
        "Subset": int(row["Subset"]),
        x_col: f"{row[x_col]:.2f} {x_unit}",
        y_col: f"{row[y_col]:.2f} {y_unit}",
        z_col: f"{row[z_col]:.2f} {z_unit}",
        colorscale: f"{row[colorscale]:.2f} {colorscale_unit}"
    } for _, row in df.iterrows()]
    custom_tooltips = [f"Mol {m['Mol_ID']}<br>Name: {m['MolName']}<br>Subset: {m['Subset']}<br>{x_col}: {m[x_col]}<br>{y_col}: {m[y_col]}<br>{z_col}: {m[z_col]}<br>{colorscale}: {m[colorscale]}" for m in molecule_data]
    set_loading_screen_progress(state, 52)

    text_color = rgba_tuple_to_string(state["themes"][state["theme_name"]]["Text Color"])
    colorscale_vals = df[colorscale].to_numpy()
    min_colorscale, max_colorscale = colorscale_vals.min(), colorscale_vals.max()
    continuous_colors = _get_active_continuous_colormap_rgba(state)
    plotly_continuous_colorscale = _plotly_colorscale_from_rgba(continuous_colors)
    rgba_colors = [f"rgba({r},{g},{b},1.0)" for r, g, b in [_interpolate_continuous_rgb(val, min_colorscale, max_colorscale, continuous_colors) for val in colorscale_vals]]
    set_loading_screen_progress(state, 64)

    def find_duplicates(x: Any, y: Any, z: Any, precision: int = 5) -> Any:
        rounded_coords = np.round(np.column_stack((x, y, z)), decimals=precision)
        coord_map = defaultdict(list)
        for i, coord in enumerate(rounded_coords):
            coord_map[tuple(coord)].append(i)
        return [group for group in coord_map.values() if len(group) > 1]

    def is_discrete(values: Any) -> Any:
        return np.all(np.equal(np.mod(values, 1), 0))

    def apply_jitter_to_duplicates(x: Any, y: Any, z: Any, base_scale: float = 0.005, precision: int = 5) -> Any:
        x, y, z = np.array(x), np.array(y), np.array(z)
        x_discrete = is_discrete(x)
        y_discrete = is_discrete(y)
        z_discrete = is_discrete(z)
        range_x = np.ptp(x) if not x_discrete else 1.0
        range_y = np.ptp(y) if not y_discrete else 1.0
        range_z = np.ptp(z) if not z_discrete else 1.0
        duplicates = find_duplicates(x, y, z, precision)
        for group in duplicates:
            for idx in group:
                if not x_discrete:
                    x[idx] += np.random.normal(0, base_scale * range_x)
                if not y_discrete:
                    y[idx] += np.random.normal(0, base_scale * range_y)
                if not z_discrete:
                    z[idx] += np.random.normal(0, base_scale * range_z)
        return x, y, z

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    z = df[z_col].to_numpy()
    z_min, z_max = z.min(), z.max()
    x, y, z = apply_jitter_to_duplicates(x, y, z)
    set_loading_screen_progress(state, 74)

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode="markers", showlegend=False, name="",
        marker=dict(size=4, opacity=1.0, color=rgba_colors, showscale=True, colorscale=plotly_continuous_colorscale, colorbar=dict(title=colorscale)),
        text=custom_tooltips, hovertemplate="%{text}<extra></extra>"
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=20),
        scene=dict(
            xaxis=dict(title=dict(text=x_col, font=dict(color=text_color)), showticklabels=True, tickfont=dict(color=text_color), showspikes=True, gridcolor="lightgray", backgroundcolor="rgba(80,80,80,255)"),
            yaxis=dict(title=dict(text=y_col, font=dict(color=text_color)), showticklabels=True, tickfont=dict(color=text_color), showspikes=True, gridcolor="lightgray", backgroundcolor="rgba(80,80,80,255)"),
            zaxis=dict(title=dict(text=z_col, font=dict(color=text_color)), range=[z_max, z_min], showticklabels=True, tickfont=dict(color=text_color), showspikes=True, gridcolor="lightgray", backgroundcolor="rgba(80,80,80,255)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), projection=dict(type="perspective")),
            dragmode="orbit"
        ),
        paper_bgcolor=rgba_tuple_to_string(state["themes"][state["theme_name"]]["Main Background"])
    )
    set_loading_screen_progress(state, 84)

    html_path = _chemspace_html_output_path(state, f"{subset}_3D_descriptors.html")
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    set_loading_screen_progress(state, 90)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    template_path = resource_path("app", "analysis", "chemspace", "html", "3d_descriptors.html")
    with open(template_path, "r", encoding="utf-8") as f:
        custom_template = f.read()
    custom_buttons_html = (
        custom_template
        .replace("__js_xlabel__", json.dumps(x_col))
        .replace("__js_ylabel__", json.dumps(y_col))
        .replace("__js_zlabel__", json.dumps(z_col))
        .replace("__js_colorscale_label__", json.dumps(colorscale))
        .replace("__js_x__", json.dumps(df[x_col].tolist()))
        .replace("__js_y__", json.dumps(df[y_col].tolist()))
        .replace("__js_z__", json.dumps(df[z_col].tolist()))
        .replace("__js_colorscale__", json.dumps(df[colorscale].tolist()))
        .replace("__js_tooltips__", json.dumps(custom_tooltips))
        .replace("__js_textcolor__", json.dumps(text_color))
        .replace("__js_continuous_colorscale__", json.dumps(plotly_continuous_colorscale))
        .replace("__css_activity_gradient__", "linear-gradient(to top, " + ", ".join(f"rgb({color[0]}, {color[1]}, {color[2]})" for color in continuous_colors) + ")")
    )
    html += "\n<!-- Custom Controls Injected -->\n" + custom_buttons_html
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    set_loading_screen_progress(state, 96)
    open_html_safely(html_path)
