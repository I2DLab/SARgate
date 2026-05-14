"""
==================
molecule_sketcher.py
==================

Standalone 2D molecule editor for SARgate.

This version is intentionally built on tkinter for stability in the current
environment, but its interaction model is closer to a conventional chemistry
editor: atom palette, bond palette, ring palette, drag-to-draw bonds, fused
ring placement, geometry-aware growth, and valence warnings.
"""

from __future__ import annotations

import copy
import io
import itertools
import json
import math
import os
import sys
import tkinter as tk
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageTk
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import GetPeriodicTable
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.error")


CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 780
PREVIEW_WIDTH = 380
PREVIEW_HEIGHT = 280
STANDARD_BOND_LENGTH = 1.45
ATOM_HIT_RADIUS = 0.34
BOND_HIT_TOLERANCE = 0.20
DRAG_THRESHOLD_PX = 5
MINI_TOOL_BUTTON_SIZE = 40
TOOLBAR_GROUP_SPACING = 8
TOOLBAR_GROUP_TOP_PAD = 3
ATOM_LABEL_BOND_GAP_PX = 5.0
DOUBLE_TRIPLE_BOND_SPACING = 6.0
CARBON_HIDE_ELEMENTS = {"C"}

COMMON_ATOM_BUTTONS = [
    "H", "C", "N", "O", "F",
    "P", "S", "Cl", "Br", "I",
]

PERIODIC_LAYOUT = [
    ("H", 0, 0), ("He", 0, 17),
    ("Li", 1, 0), ("Be", 1, 1), ("B", 1, 12), ("C", 1, 13), ("N", 1, 14), ("O", 1, 15), ("F", 1, 16), ("Ne", 1, 17),
    ("Na", 2, 0), ("Mg", 2, 1), ("Al", 2, 12), ("Si", 2, 13), ("P", 2, 14), ("S", 2, 15), ("Cl", 2, 16), ("Ar", 2, 17),
    ("K", 3, 0), ("Ca", 3, 1), ("Sc", 3, 2), ("Ti", 3, 3), ("V", 3, 4), ("Cr", 3, 5), ("Mn", 3, 6), ("Fe", 3, 7), ("Co", 3, 8), ("Ni", 3, 9), ("Cu", 3, 10), ("Zn", 3, 11), ("Ga", 3, 12), ("Ge", 3, 13), ("As", 3, 14), ("Se", 3, 15), ("Br", 3, 16), ("Kr", 3, 17),
    ("Rb", 4, 0), ("Sr", 4, 1), ("Y", 4, 2), ("Zr", 4, 3), ("Nb", 4, 4), ("Mo", 4, 5), ("Tc", 4, 6), ("Ru", 4, 7), ("Rh", 4, 8), ("Pd", 4, 9), ("Ag", 4, 10), ("Cd", 4, 11), ("In", 4, 12), ("Sn", 4, 13), ("Sb", 4, 14), ("Te", 4, 15), ("I", 4, 16), ("Xe", 4, 17),
    ("Cs", 5, 0), ("Ba", 5, 1), ("La-Lu", 5, 2), ("Hf", 5, 3), ("Ta", 5, 4), ("W", 5, 5), ("Re", 5, 6), ("Os", 5, 7), ("Ir", 5, 8), ("Pt", 5, 9), ("Au", 5, 10), ("Hg", 5, 11), ("Tl", 5, 12), ("Pb", 5, 13), ("Bi", 5, 14), ("Po", 5, 15), ("At", 5, 16), ("Rn", 5, 17),
    ("Fr", 6, 0), ("Ra", 6, 1), ("Ac-Lr", 6, 2), ("Rf", 6, 3), ("Db", 6, 4), ("Sg", 6, 5), ("Bh", 6, 6), ("Hs", 6, 7), ("Mt", 6, 8), ("Ds", 6, 9), ("Rg", 6, 10), ("Cn", 6, 11), ("Nh", 6, 12), ("Fl", 6, 13), ("Mc", 6, 14), ("Lv", 6, 15), ("Ts", 6, 16), ("Og", 6, 17),
    ("La", 7, 2), ("Ce", 7, 3), ("Pr", 7, 4), ("Nd", 7, 5), ("Pm", 7, 6), ("Sm", 7, 7), ("Eu", 7, 8), ("Gd", 7, 9), ("Tb", 7, 10), ("Dy", 7, 11), ("Ho", 7, 12), ("Er", 7, 13), ("Tm", 7, 14), ("Yb", 7, 15), ("Lu", 7, 16),
    ("Ac", 8, 2), ("Th", 8, 3), ("Pa", 8, 4), ("U", 8, 5), ("Np", 8, 6), ("Pu", 8, 7), ("Am", 8, 8), ("Cm", 8, 9), ("Bk", 8, 10), ("Cf", 8, 11), ("Es", 8, 12), ("Fm", 8, 13), ("Md", 8, 14), ("No", 8, 15), ("Lr", 8, 16),
]

BOND_STYLE_DEFS = [
    ("single", "Single"),
    ("double", "Double"),
    ("triple", "Triple"),
    ("wedge", "Wedge"),
    ("hashed", "Hashed"),
    ("chain", "Chain"),
]

ACTION_GLYPHS = {
    "undo": "↩",
    "redo": "↪",
    "select": "↖",
    "rotate": "↻",
    "erase": "⌫",
    "hflip": "⇄",
    "vflip": "⇅",
    "stereocenters": "@",
    "save": "💾",
}

RING_TEMPLATES = [
    ("benzene", "⌬", 6, True),
    ("cyclohexane", "⬡", 6, False),
    ("cyclopentane", "⬠", 5, False),
    ("cyclopropane", "△", 3, False),
    ("cyclobutane", "□", 4, False),
    ("cycloheptane", "7", 7, False),
]

_PERIODIC_TABLE = GetPeriodicTable()


def _build_max_valence_map() -> dict[str, float]:
    max_valence: dict[str, float] = {}
    for atomic_num in range(1, 119):
        symbol = _PERIODIC_TABLE.GetElementSymbol(atomic_num)
        try:
            allowed_valences = list(_PERIODIC_TABLE.GetValenceList(atomic_num))
        except Exception:
            allowed_valences = []
        finite_valences = [float(v) for v in allowed_valences if int(v) >= 0]
        if finite_valences:
            max_valence[symbol] = max(finite_valences)
        else:
            try:
                default_valence = int(_PERIODIC_TABLE.GetDefaultValence(atomic_num))
            except Exception:
                default_valence = -1
            if default_valence >= 0:
                max_valence[symbol] = float(default_valence)
    return max_valence


MAX_VALENCE = _build_max_valence_map()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FONT_ASSETS_DIR = PROJECT_ROOT / "assets" / "fonts"


def _load_font_with_fallback(font_names: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # First pass: local project fonts.
    for font_name in font_names:
        local_path = FONT_ASSETS_DIR / font_name
        try:
            return ImageFont.truetype(str(local_path), size)
        except Exception:
            continue
    # Second pass: system-resolved fonts.
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()
ATOM_COLORS = {
    "H": "#8c8f96",
    "He": "#d9ffff",
    "Li": "#cc80ff",
    "Be": "#c2ff00",
    "B": "#ffb5b5",
    "C": "#000000",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "F": "#90e050",
    "Ne": "#b3e3f5",
    "Na": "#ab5cf2",
    "Mg": "#8aff00",
    "Al": "#bfa6a6",
    "Si": "#f0c8a0",
    "P": "#ff8000",
    "S": "#ffff30",
    "Cl": "#1ff01f",
    "Ar": "#80d1e3",
    "K": "#8f40d4",
    "Ca": "#3dff00",
    "Sc": "#e6e6e6",
    "Ti": "#bfc2c7",
    "V": "#a6a6ab",
    "Cr": "#8a99c7",
    "Mn": "#9c7ac7",
    "Fe": "#e06633",
    "Co": "#f090a0",
    "Ni": "#50d050",
    "Cu": "#c88033",
    "Zn": "#7d80b0",
    "Ga": "#c28f8f",
    "Ge": "#668f8f",
    "As": "#bd80e3",
    "Se": "#ffa100",
    "Br": "#a62929",
    "Kr": "#5cb8d1",
    "Rb": "#702eb0",
    "Sr": "#00ff00",
    "Y": "#94ffff",
    "Zr": "#94e0e0",
    "Nb": "#73c2c9",
    "Mo": "#54b5b5",
    "Tc": "#3b9e9e",
    "Ru": "#248f8f",
    "Rh": "#0a7d8c",
    "Pd": "#006985",
    "Ag": "#c0c0c0",
    "Cd": "#ffd98f",
    "In": "#a67573",
    "Sn": "#668080",
    "Sb": "#9e63b5",
    "Te": "#d47a00",
    "I": "#940094",
    "Xe": "#429eb0",
    "Cs": "#57178f",
    "Ba": "#00c900",
    "La": "#70d4ff",
    "Ce": "#d6d68f",
    "Pr": "#d9ffc7",
    "Nd": "#c7ffc7",
    "Pm": "#a3ffc7",
    "Sm": "#8fffc7",
    "Eu": "#61ffc7",
    "Gd": "#45ffc7",
    "Tb": "#30ffc7",
    "Dy": "#1fffc7",
    "Ho": "#00ff9c",
    "Er": "#00e675",
    "Tm": "#00d452",
    "Yb": "#00bf38",
    "Lu": "#00ab24",
    "Hf": "#4dc2ff",
    "Ta": "#4da6ff",
    "W": "#2194d6",
    "Re": "#267dab",
    "Os": "#266696",
    "Ir": "#175487",
    "Pt": "#d0d0e0",
    "Au": "#ffd123",
    "Hg": "#b8b8d0",
    "Tl": "#a6544d",
    "Pb": "#575961",
    "Bi": "#9e4fb5",
    "Po": "#ab5c00",
    "At": "#754f45",
    "Rn": "#428296",
    "Fr": "#420066",
    "Ra": "#007d00",
    "Ac": "#70abfa",
    "Th": "#00baff",
    "Pa": "#00a1ff",
    "U": "#008fff",
    "Np": "#0080ff",
    "Pu": "#006bff",
    "Am": "#545cf2",
    "Cm": "#785ce3",
    "Bk": "#8a4fe3",
    "Cf": "#a136d4",
    "Es": "#b31fd4",
    "Fm": "#b31fba",
    "Md": "#b30da6",
    "No": "#bd0d87",
    "Lr": "#c70066",
    "Rf": "#cc0059",
    "Db": "#d1004f",
    "Sg": "#d90045",
    "Bh": "#e00038",
    "Hs": "#e6002e",
    "Mt": "#eb0026",
    "Ds": "#f0001f",
    "Rg": "#f5001a",
    "Cn": "#fa0014",
    "Nh": "#ff000f",
    "Fl": "#ff1a1a",
    "Mc": "#ff3333",
    "Lv": "#ff4d4d",
    "Ts": "#ff6666",
    "Og": "#ff8080",
}

PERIODIC_CLASS_COLORS = {
    "alkali_metal": "#f6b8b8",
    "alkaline_earth_metal": "#f7c9ab",
    "transition_metal": "#f3e1a6",
    "post_transition_metal": "#d9ebb1",
    "metalloid": "#b9e5de",
    "nonmetal": "#bfe8c8",
    "halogen": "#b9e8f1",
    "noble_gas": "#bfd3f6",
    "lanthanide": "#d8c7f5",
    "actinide": "#f3c2de",
    "unknown": "#cbd5e1",
}

def _env_hex(name: str, fallback: str) -> str:
    value = str(os.environ.get(name, fallback) or fallback).strip()
    if len(value) == 7 and value.startswith("#"):
        return value
    return fallback


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _hex_to_rgb(color: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(color, str):
        return fallback
    value = color.strip()
    if len(value) != 7 or not value.startswith("#"):
        return fallback
    try:
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
    except Exception:
        return fallback


def _mix_hex(color_a: str, color_b: str, ratio: float = 0.5) -> str:
    ra, ga, ba = _hex_to_rgb(color_a, (0, 0, 0))
    rb, gb, bb = _hex_to_rgb(color_b, (255, 255, 255))
    ratio = max(0.0, min(1.0, float(ratio)))
    r = int(round(ra * (1.0 - ratio) + rb * ratio))
    g = int(round(ga * (1.0 - ratio) + gb * ratio))
    b = int(round(ba * (1.0 - ratio) + bb * ratio))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def _contrast_text(bg_hex: str) -> str:
    r, g, b = _hex_to_rgb(bg_hex, (255, 255, 255))
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return "#111111" if luminance > 160 else "#ffffff"


def _periodic_class(symbol: str) -> str:
    if symbol in {"Li", "Na", "K", "Rb", "Cs", "Fr"}:
        return "alkali_metal"
    if symbol in {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}:
        return "alkaline_earth_metal"
    if symbol in {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    }:
        return "transition_metal"
    if symbol in {"Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi", "Nh", "Fl", "Mc", "Lv"}:
        return "post_transition_metal"
    if symbol in {"B", "Si", "Ge", "As", "Sb", "Te", "Po"}:
        return "metalloid"
    if symbol in {"H", "C", "N", "O", "P", "S", "Se"}:
        return "nonmetal"
    if symbol in {"F", "Cl", "Br", "I", "At", "Ts"}:
        return "halogen"
    if symbol in {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"}:
        return "noble_gas"
    if symbol in {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}:
        return "lanthanide"
    if symbol in {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}:
        return "actinide"
    return "unknown"


BG = _env_hex("SARGATE_SKETCHER_BG", "#f5f7fb")
GRID = _env_hex("SARGATE_SKETCHER_GRID", "#e3e8f0")
BOND = "#303846"
AROMATIC = BOND
SELECTED = "#f59e0b"
GUIDE = "#4f46e5"
TEXT = _env_hex("SARGATE_SKETCHER_TEXT", "#1f2937")
PANEL = _env_hex("SARGATE_SKETCHER_PANEL", "#ffffff")
PANEL_EDGE = _env_hex("SARGATE_SKETCHER_BORDER", "#d7dee8")
RED = "#dc2626"
BLUE = "#2563eb"


@dataclass
class EditorAtom:
    id: int
    element: str
    x: float
    y: float
    charge: int = 0
    aromatic: bool = False
    label: str = ""


@dataclass
class EditorBond:
    id: int
    a1: int
    a2: int
    style: str = "single"


class MoleculeSketcherApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("SARgate Molecule Sketcher")
        self.root.geometry("1620x940")
        self.root.minsize(1400, 860)
        self.root.configure(bg=BG)
        self.root.lift()
        try:
            self.root.attributes("-topmost", True)
            self.root.after(350, lambda: self.root.attributes("-topmost", False))
        except Exception:
            pass
        self.root.after(10, self.root.focus_force)

        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.theme_sync_path = os.environ.get("SARGATE_SKETCHER_THEME_FILE", "").strip()
        self._theme_sync_mtime: float | None = None
        self._last_focus_nonce = -1
        self._theme_payload: dict[str, Any] = {}
        self.button_bg = PANEL
        self.button_hover = _mix_hex(PANEL, TEXT, 0.10)
        self.button_active = _mix_hex(PANEL, TEXT, 0.18)
        self.button_selected_bg = SELECTED
        self.button_selected_text = _contrast_text(SELECTED)
        self.window_border_size = 1
        self.frame_border_size = 1
        self.window_rounding = 8
        self.frame_rounding = 6
        self.tab_rounding = 5

        self.atoms: list[EditorAtom] = []
        self.bonds: list[EditorBond] = []
        self.next_atom_id = 1
        self.next_bond_id = 1
        self._atom_index: dict[int, EditorAtom] = {}
        self._bond_index: dict[int, EditorBond] = {}

        self.mode: str | None = "atom"
        self.selected_atom_symbol = "C"
        self.selected_bond_style = "single"
        self.selected_ring_template = "benzene"
        self.current_charge = 0
        self._stored_control_atom_symbol: str | None = None
        self._stored_charge_atom_symbol: str | None = None
        self._show_bond_button_selection = False
        self.show_stereocenters = False

        self.selected_atom_id: int | None = None
        self.selected_bond_id: int | None = None
        self.selected_atom_ids: set[int] = set()
        self.selected_bond_ids: set[int] = set()
        self.drag_atom_id: int | None = None
        self.drag_group_ids: set[int] = set()
        self.drag_group_origin: dict[int, tuple[float, float]] = {}
        self.drag_offset_world = (0.0, 0.0)
        self.drag_preview_delta_world = (0.0, 0.0)
        self.drag_bridge_bond_ids: set[int] = set()
        self.drag_start_screen = (0.0, 0.0)
        self.drag_last_screen = (0.0, 0.0)
        self.pending_draw_start_atom_id: int | None = None
        self.pending_draw_start_pos: tuple[float, float] | None = None
        self.pending_draw_mode: str | None = None
        self.chain_drag_side_sign: float | None = None
        self.preview_world_pos: tuple[float, float] | None = None
        self.preview_world_path: list[tuple[float, float]] = []
        self.pending_ring_active = False
        self.preview_ring_points: list[tuple[float, float]] = []
        self.preview_ring_bonds: list[tuple[int, int]] = []
        self.preview_ring_commit: dict[str, Any] | None = None
        self.is_panning = False
        self.selection_rect_start: tuple[float, float] | None = None
        self.selection_rect_end: tuple[float, float] | None = None
        self.erase_rect_start: tuple[float, float] | None = None
        self.erase_rect_end: tuple[float, float] | None = None
        self.erase_hit_atom_id: int | None = None
        self.erase_hit_bond_id: int | None = None
        self.preview_rect_atom_ids: set[int] = set()
        self.preview_rect_bond_ids: set[int] = set()
        self.preview_fuse_atom_ids: set[int] = set()
        self.preview_fuse_bond_ids: set[int] = set()
        self.hover_atom_id: int | None = None
        self.hover_bond_id: int | None = None
        self.mouse_screen_pos: tuple[float, float] | None = None
        self.pan_anchor = (0.0, 0.0)
        self.pan_origin = (0.0, 0.0)
        self.right_click_anchor = (0.0, 0.0)
        self.right_click_root = (0.0, 0.0)
        self.right_click_dragged = False
        self.left_button_down = False
        self.is_rotating = False
        self.rotation_anchor_angle = 0.0
        self.rotation_origin_atoms: dict[int, tuple[float, float]] = {}
        self.rotation_center = (0.0, 0.0)
        self.total_rotation_deg = 0.0
        self.last_canvas_click_world = (0.0, 0.0)

        self.scale = 31.0
        self.offset_x = CANVAS_WIDTH / 2
        self.offset_y = CANVAS_HEIGHT / 2

        self.history: list[dict[str, Any]] = []
        self.history_index = -1

        self.status_var = tk.StringVar(value="Ready")
        self.smiles_var = tk.StringVar(value="")
        self.output_base_var = tk.StringVar(value=str(Path.cwd() / "molecule_sketcher_output"))
        self.current_smiles_var = tk.StringVar(value="")
        self.charge_var = tk.StringVar(value="0")
        self.carbons_display_var = tk.StringVar(value="None")
        self.hydrogens_display_var = tk.StringVar(value="Hetero")
        self.colored_atoms_var = tk.BooleanVar(value=True)
        self.periodic_color_mode_var = tk.StringVar(value="class")

        self.canvas: tk.Canvas | None = None
        self.preview_image_tk: ImageTk.PhotoImage | None = None
        if sys.platform.startswith("win"):
            self._aa_render_scale_idle = 2.5
            self._aa_render_scale_interactive = 2.5
        else:
            self._aa_render_scale_idle = 2.5
            self._aa_render_scale_interactive = 2.5
        self._aa_render_scale = self._aa_render_scale_idle
        self._active_aa_render_scale = self._aa_render_scale_idle
        self._aa_render_image: Image.Image | None = None
        self._aa_render_draw: ImageDraw.ImageDraw | None = None
        self._pil_font_cache: dict[tuple[str, int, str], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self.molblock_text: tk.Text | None = None
        self._atom_main_font: tkfont.Font | None = None
        self._atom_side_font: tkfont.Font | None = None

        self.atom_button_widgets: dict[str, tk.Label] = {}
        self.bond_button_widgets: dict[str, tk.Label] = {}
        self.ring_button_widgets: dict[str, tk.Label] = {}
        self.mode_button_widgets: dict[str, tk.Label] = {}
        self.structure_button_widgets: dict[str, tk.Label] = {}
        self.aux_action_button_widgets: dict[str, tk.Label] = {}
        self.ring_button_images: dict[str, ImageTk.PhotoImage] = {}
        self.action_button_images: dict[str, ImageTk.PhotoImage] = {}
        self.element_button_images: dict[str, ImageTk.PhotoImage] = {}
        self.periodic_atom_buttons: list[tuple[str, tk.Label]] = []
        self.periodic_popup_labels: list[tk.Label] = []
        self.periodic_popup_window: tk.Toplevel | None = None
        self.styled_entries: list[tk.Entry] = []
        self.styled_option_menus: list[tk.Menubutton] = []
        self.smiles_label_widget: tk.Label | None = None
        self.smiles_display_frame: tk.Frame | None = None
        self.smiles_display_label: tk.Label | None = None
        self.copy_smiles_button: tk.Label | None = None
        self.charge_label_widget: tk.Label | None = None
        self.charge_display_frame: tk.Frame | None = None
        self.charge_display_label: tk.Label | None = None
        self.charge_popup_menu: tk.Menu | None = None
        self.charge_button_widgets: dict[str, tk.Label] = {}
        self.carbons_label_widget: tk.Label | None = None
        self.carbons_display_frame: tk.Frame | None = None
        self.carbons_display_label: tk.Label | None = None
        self.carbons_popup_menu: tk.Menu | None = None
        self.hydrogens_label_widget: tk.Label | None = None
        self.hydrogens_display_frame: tk.Frame | None = None
        self.hydrogens_display_label: tk.Label | None = None
        self.hydrogens_popup_menu: tk.Menu | None = None
        self.colored_atoms_checkbutton: tk.Checkbutton | None = None
        self.save_image_button: tk.Label | None = None
        self.visibility_button_widgets: dict[str, tk.Label] = {}
        self.canvas_context_menu: tk.Menu | None = None
        self.square_button_images: dict[tuple[int, str], tk.PhotoImage] = {}
        self._tooltip_window: tk.Toplevel | None = None
        self._tooltip_label: tk.Label | None = None
        self._canvas_render_pending = False
        self._canvas_render_after_id: str | None = None
        self._atom_key_buffer = ""
        self._atom_key_after_id: str | None = None

        self._build_ui()
        self._bind_events()
        self.apply_runtime_theme(self._load_theme_payload())
        self._poll_theme_sync()
        self._push_history("Initial state")
        self.render_all()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="SAR.TFrame", padding=12)
        outer.pack(fill="both", expand=True)

        toolbar = ttk.Frame(outer, style="SAR.TFrame")
        toolbar.pack(fill="x", pady=(0, 4))

        canvas_host = ttk.Frame(outer, style="SAR.TFrame")
        canvas_host.pack(fill="both", expand=True)

        self._build_top_toolbar(toolbar)
        self._build_canvas(canvas_host)

    def _build_top_toolbar(self, parent: ttk.Frame) -> None:
        left_cluster = tk.Frame(parent, bg=PANEL)
        left_cluster.grid(row=0, column=0, sticky="nw", padx=(0, 0), pady=0)
        left_cluster.grid_columnconfigure(0, weight=1)

        left_top_row = tk.Frame(left_cluster, bg=PANEL)
        left_top_row.grid(row=0, column=0, sticky="nw", padx=0, pady=0)
        structure_group = tk.Frame(left_top_row, bg=PANEL)
        structure_group.pack(side="left", anchor="nw")
        self._build_structure_buttons(structure_group)

        mode_group = tk.Frame(left_top_row, bg=PANEL)
        mode_group.pack(side="left", anchor="nw", padx=(TOOLBAR_GROUP_SPACING, 0))
        self._build_mode_buttons(mode_group)

        smiles_group = tk.Frame(left_cluster, bg=PANEL)
        smiles_group.grid(row=1, column=0, sticky="new", padx=0, pady=(0, 0))
        smiles_group.grid_configure(ipady=0)
        self._build_smiles_display(smiles_group)

        aux_visibility_group = tk.Frame(parent, bg=PANEL)
        aux_visibility_group.grid(row=0, column=1, sticky="nw", padx=(TOOLBAR_GROUP_SPACING, 0), pady=(TOOLBAR_GROUP_TOP_PAD, 0))
        self._build_aux_action_buttons(aux_visibility_group)
        self._build_visibility_group(aux_visibility_group)

        atoms_group = tk.Frame(parent, bg=PANEL)
        atoms_group.grid(row=0, column=2, sticky="nw", padx=(TOOLBAR_GROUP_SPACING, 0), pady=(TOOLBAR_GROUP_TOP_PAD, 0))
        self._build_atom_buttons(atoms_group)

        misc_group = tk.Frame(parent, bg=PANEL)
        misc_group.grid(row=0, column=3, sticky="nw", padx=(TOOLBAR_GROUP_SPACING, 0), pady=(TOOLBAR_GROUP_TOP_PAD, 0))
        self._build_charge_periodic_group(misc_group)

        bonds_rings_group = tk.Frame(parent, bg=PANEL)
        bonds_rings_group.grid(row=0, column=4, sticky="nw", padx=(TOOLBAR_GROUP_SPACING, 0), pady=(TOOLBAR_GROUP_TOP_PAD, 0))
        self._build_bond_buttons(bonds_rings_group)
        self._build_ring_buttons(bonds_rings_group)

    def _square_button_image(self, size: int, bg: str | None = None) -> tk.PhotoImage:
        fill = bg or self.button_bg
        cache_key = (size, fill)
        cached = self.square_button_images.get(cache_key)
        if cached is not None:
            return cached
        image = tk.PhotoImage(width=size, height=size)
        image.put(fill, to=(0, 0, size, size))
        self.square_button_images[cache_key] = image
        return image

    def _sync_square_button_backing(self, btn: tk.Label, bg: str) -> None:
        if not bool(getattr(btn, "_sargate_square_backing", False)):
            return
        size = getattr(btn, "_sargate_square_size", None)
        if not size:
            return
        img = self._square_button_image(int(size), bg)
        btn.configure(image=img, bg=bg)

    def _make_toggle_button(self, parent: tk.Widget, text: str, command: Any, width: int = 8, square_size: int | None = None) -> tk.Label:
        button_kwargs: dict[str, Any] = {
            "text": text,
            "relief": tk.RAISED,
            "bd": max(1, self.frame_border_size),
            "bg": self.button_bg,
            "fg": TEXT,
            "font": ("Arial", 10, "bold"),
            "highlightbackground": PANEL_EDGE,
            "highlightcolor": PANEL_EDGE,
            "highlightthickness": 1,
            "cursor": "hand2",
            "anchor": "center",
            "justify": "center",
            "padx": 0,
            "pady": 0,
        }
        if square_size is None:
            button_kwargs["width"] = max(width, 1)
        else:
            square_img = self._square_button_image(square_size, self.button_bg)
            button_kwargs.update(
                image=square_img,
                compound="center",
                width=square_size,
                height=square_size,
                padx=0,
                pady=0,
                font=("Arial", 12, "bold"),
            )
        btn = tk.Label(parent, **button_kwargs)
        if square_size is not None:
            setattr(btn, "_sargate_square_size", square_size)
            setattr(btn, "_sargate_square_backing", bool(text))
        btn.bind("<Button-1>", lambda _event: command())
        btn.bind("<Enter>", lambda _event, widget=btn: self._set_button_hover(widget, True))
        btn.bind("<Leave>", lambda _event, widget=btn: self._set_button_hover(widget, False))
        return btn

    def _make_element_icon(self, symbol: str, size: int = 40) -> ImageTk.PhotoImage:
        cache_key = f"{symbol}|{size}|{TEXT}"
        cached = self.element_button_images.get(cache_key)
        if cached is not None:
            return cached
        aa = 4
        hi = size * aa
        image = Image.new("RGBA", (hi, hi), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        text_color = _hex_to_rgb(TEXT, (49, 56, 70)) + (255,)
        if len(symbol) == 1:
            font_size = 90
        elif len(symbol) == 2:
            font_size = 74
        else:
            font_size = 50
        font = _load_font_with_fallback(["Arial.ttf", "arial.ttf"], font_size)
        draw.text((hi / 2, hi / 2), symbol, fill=text_color, font=font, anchor="mm")
        out = ImageTk.PhotoImage(image.resize((size, size), Image.Resampling.LANCZOS))
        self.element_button_images[cache_key] = out
        return out

    def _configure_element_button_face(self, btn: tk.Label, symbol: str, size: int = 40) -> None:
        icon = self._make_element_icon(symbol, size)
        btn.configure(text="", image=icon, compound="center")
        setattr(btn, "_sargate_element_symbol", symbol)
        setattr(btn, "_sargate_element_icon", icon)
        setattr(btn, "_sargate_square_backing", False)

    def _style_entry_widget(self, entry: tk.Entry) -> None:
        entry.configure(
            bg=GRID,
            fg=TEXT,
            insertbackground=TEXT,
            readonlybackground=GRID,
            disabledbackground=GRID,
            disabledforeground=TEXT,
            selectbackground=self.button_hover,
            selectforeground=TEXT,
            highlightbackground=PANEL_EDGE,
            highlightcolor=PANEL_EDGE,
            highlightthickness=1,
            relief="solid",
            bd=max(1, self.frame_border_size),
        )
        try:
            entry.configure(state="readonly")
        except Exception:
            pass

    def _style_charge_menu_widget(self, option_menu: tk.Menubutton) -> None:
        option_menu.configure(
            bg=GRID,
            fg=TEXT,
            activebackground=self.button_hover,
            activeforeground=TEXT,
            disabledforeground=TEXT,
            highlightbackground=PANEL_EDGE,
            highlightcolor=PANEL_EDGE,
            highlightthickness=1,
            relief=tk.RAISED,
            bd=max(1, self.frame_border_size),
            width=6,
            font=("Arial", 10),
        )
        try:
            menu_name = option_menu.cget("menu")
            if menu_name:
                menu_widget = option_menu.nametowidget(menu_name)
                menu_widget.configure(
                    bg=GRID,
                    fg=TEXT,
                    activebackground=self.button_hover,
                    activeforeground=TEXT,
                    selectcolor=GRID,
                    relief=tk.FLAT,
                    bd=0,
                )
        except Exception:
            pass

    def _style_readonly_display_frame(self, frame: tk.Frame, label: tk.Widget) -> None:
        frame.configure(
            bg=GRID,
            highlightbackground=PANEL_EDGE,
            highlightcolor=PANEL_EDGE,
            highlightthickness=0,
            bd=max(1, self.frame_border_size),
            relief="solid",
        )
        label.configure(
            bg=GRID,
            fg=TEXT,
            anchor="w",
            justify="left",
            font=("Arial", 10),
        )

    def _style_charge_display(self, frame: tk.Frame, label: tk.Label) -> None:
        frame.configure(
            bg=GRID,
            highlightbackground=PANEL_EDGE,
            highlightcolor=PANEL_EDGE,
            highlightthickness=0,
            bd=max(1, self.frame_border_size),
            relief="solid",
        )
        label.configure(
            bg=GRID,
            fg=TEXT,
            anchor="center",
            justify="center",
            font=("Arial", 13, "bold"),
        )

    def _style_choice_display(self, frame: tk.Frame, label: tk.Label) -> None:
        frame.configure(
            bg=GRID,
            highlightbackground=PANEL_EDGE,
            highlightcolor=PANEL_EDGE,
            highlightthickness=0,
            bd=max(1, self.frame_border_size),
            relief="solid",
        )
        label.configure(
            bg=GRID,
            fg=TEXT,
            anchor="w",
            justify="left",
            font=("Arial", 10),
        )

    def _style_plain_toolbar_label(self, label: tk.Label) -> None:
        label.configure(
            bg=PANEL,
            fg=TEXT,
            bd=0,
            highlightthickness=0,
            highlightbackground=PANEL,
            highlightcolor=PANEL,
            relief="flat",
            font=("Arial", 10),
        )

    def _periodic_popup_bg(self, symbol: str) -> str:
        if self.periodic_color_mode_var.get().strip().lower() == "cpk":
            return ATOM_COLORS.get(symbol, PERIODIC_CLASS_COLORS["unknown"])
        return PERIODIC_CLASS_COLORS.get(_periodic_class(symbol), PERIODIC_CLASS_COLORS["unknown"])

    def _configure_periodic_popup_button_face(self, btn: tk.Label, symbol: str) -> None:
        bg = self._periodic_popup_bg(symbol)
        fg = _contrast_text(bg)
        font_size = 11 if len(symbol) == 1 else 10
        square_size = int(getattr(btn, "_sargate_square_size", 40) or 40)
        aa = 4
        hi = square_size * aa
        image = Image.new("RGBA", (hi, hi), _hex_to_rgb(bg, (203, 213, 225)) + (255,))
        draw = ImageDraw.Draw(image)
        font = _load_font_with_fallback(
            ["Arial Bold.ttf", "arialbd.ttf", "Arial.ttf", "arial.ttf"],
            46 if len(symbol) == 1 else 40,
        )
        draw.text(
            (hi / 2, hi / 2),
            symbol,
            fill=_hex_to_rgb(fg, (17, 17, 17)) + (255,),
            font=font,
            anchor="mm",
        )
        backing = ImageTk.PhotoImage(image.resize((square_size, square_size), Image.Resampling.LANCZOS))
        btn.configure(
            text="",
            image=backing,
            compound="center",
            width=square_size,
            height=square_size,
            bg=bg,
            fg=fg,
            font=("Arial", font_size, "bold"),
            anchor="center",
            justify="center",
            relief=tk.RAISED,
            bd=max(1, self.frame_border_size),
            highlightbackground=PANEL_EDGE,
            highlightcolor=PANEL_EDGE,
            highlightthickness=1,
        )
        setattr(btn, "_sargate_square_backing", False)
        setattr(btn, "_sargate_periodic_symbol", symbol)
        setattr(btn, "_sargate_periodic_icon", backing)
        setattr(btn, "_sargate_periodic_bg", bg)
        setattr(btn, "_sargate_periodic_fg", fg)

    def _configure_periodic_popup_label_face(self, label: tk.Label, symbol: str) -> None:
        bg = self._periodic_popup_bg(symbol)
        fg = _contrast_text(bg)
        label.configure(
            text=symbol,
            bg=bg,
            fg=fg,
            width=6,
            height=2,
            relief=tk.RAISED,
            bd=max(1, self.frame_border_size),
            highlightbackground=PANEL_EDGE,
            highlightcolor=PANEL_EDGE,
            highlightthickness=1,
            font=("Arial", 9, "bold"),
        )
        setattr(label, "_sargate_periodic_symbol", symbol)

    def _set_periodic_popup_hover(self, btn: tk.Label, hovering: bool) -> None:
        base_bg = str(getattr(btn, "_sargate_periodic_bg", self.button_bg))
        base_fg = str(getattr(btn, "_sargate_periodic_fg", TEXT))
        target_bg = _mix_hex(base_bg, "#ffffff" if base_fg == "#111111" else "#000000", 0.10) if hovering else base_bg
        target_fg = _contrast_text(target_bg)
        symbol = str(getattr(btn, "_sargate_periodic_symbol", "") or "")
        if symbol:
            square_size = int(getattr(btn, "_sargate_square_size", 40) or 40)
            aa = 4
            hi = square_size * aa
            image = Image.new("RGBA", (hi, hi), _hex_to_rgb(target_bg, (203, 213, 225)) + (255,))
            draw = ImageDraw.Draw(image)
            font = _load_font_with_fallback(
                ["Arial Bold.ttf", "arialbd.ttf", "Arial.ttf", "arial.ttf"],
                46 if len(symbol) == 1 else 40,
            )
            draw.text(
                (hi / 2, hi / 2),
                symbol,
                fill=_hex_to_rgb(target_fg, (17, 17, 17)) + (255,),
                font=font,
                anchor="mm",
            )
            icon = ImageTk.PhotoImage(image.resize((square_size, square_size), Image.Resampling.LANCZOS))
            setattr(btn, "_sargate_periodic_icon", icon)
            btn.configure(image=icon, bg=target_bg, fg=target_fg)
            return
        btn.configure(bg=target_bg, fg=target_fg)

    def _refresh_periodic_popup_colors(self) -> None:
        live_buttons: list[tuple[str, tk.Label]] = []
        for symbol, btn in self.periodic_atom_buttons:
            try:
                btn.winfo_exists()
                self._configure_periodic_popup_button_face(btn, symbol)
                live_buttons.append((symbol, btn))
            except Exception:
                pass
        self.periodic_atom_buttons = live_buttons
        live_labels: list[tk.Label] = []
        for label in self.periodic_popup_labels:
            try:
                label.winfo_exists()
                symbol = str(getattr(label, "_sargate_periodic_symbol", label.cget("text")) or "")
                if symbol:
                    self._configure_periodic_popup_label_face(label, symbol)
                live_labels.append(label)
            except Exception:
                pass
        self.periodic_popup_labels = live_labels

    def _charge_suffix_text(self, charge: int) -> str:
        if charge == 0:
            return ""
        sign = "+" if charge > 0 else "-"
        magnitude = abs(int(charge))
        if magnitude == 1:
            return sign
        return f"{magnitude}{sign}"

    def _open_charge_popup(self, widget: tk.Widget) -> None:
        if self.charge_popup_menu is None:
            return
        try:
            x = widget.winfo_rootx()
            y = widget.winfo_rooty() + widget.winfo_height()
            self.charge_popup_menu.tk_popup(x, y)
        finally:
            try:
                self.charge_popup_menu.grab_release()
            except Exception:
                pass

    def _open_popup_menu(self, widget: tk.Widget, menu: tk.Menu | None) -> None:
        if menu is None:
            return
        try:
            x = widget.winfo_rootx()
            y = widget.winfo_rooty() + widget.winfo_height()
            menu.tk_popup(x, y)
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass

    def _bind_tooltip(self, widget: tk.Widget, text: str) -> None:
        setattr(widget, "_sargate_tooltip_text", text)
        if bool(getattr(widget, "_sargate_tooltip_bound", False)):
            return
        widget.bind("<Enter>", lambda _event, w=widget: self._show_tooltip(w, str(getattr(w, "_sargate_tooltip_text", ""))), add="+")
        widget.bind("<Leave>", lambda _event: self._hide_tooltip(), add="+")
        widget.bind("<Button-1>", lambda _event: self._hide_tooltip(), add="+")
        setattr(widget, "_sargate_tooltip_bound", True)

    def _show_tooltip(self, widget: tk.Widget, text: str) -> None:
        self._hide_tooltip()
        tooltip = tk.Toplevel(self.root)
        tooltip.overrideredirect(True)
        tooltip.attributes("-topmost", True)
        label = tk.Label(
            tooltip,
            text=text,
            bg=PANEL,
            fg=TEXT,
            relief="solid",
            bd=1,
            padx=6,
            pady=4,
            font=("Arial", 10),
            highlightbackground=PANEL_EDGE,
            highlightcolor=PANEL_EDGE,
            highlightthickness=1,
        )
        label.pack()
        x = widget.winfo_rootx() + (widget.winfo_width() // 2) + 10
        y = widget.winfo_rooty() + widget.winfo_height() + 8
        tooltip.geometry(f"+{x}+{y}")
        self._tooltip_window = tooltip
        self._tooltip_label = label

    def _hide_tooltip(self) -> None:
        if self._tooltip_window is not None:
            try:
                self._tooltip_window.destroy()
            except Exception:
                pass
        self._tooltip_window = None
        self._tooltip_label = None

    def _clipboard_has_text(self) -> bool:
        try:
            text = self.root.clipboard_get()
        except Exception:
            return False
        return bool(str(text).strip())

    def _show_canvas_context_menu(self, x_root: int, y_root: int) -> None:
        if self.canvas_context_menu is None:
            self.canvas_context_menu = tk.Menu(self.root, tearoff=False)
        menu = self.canvas_context_menu
        menu.delete(0, "end")
        has_selection = bool(self.selected_atom_ids or self.selected_bond_ids)
        menu.add_command(
            label="Copy selection",
            command=lambda: self.on_copy_selection(),
            state=(tk.NORMAL if has_selection else tk.DISABLED),
        )
        menu.add_command(
            label="Paste",
            command=lambda: self.on_canvas_paste(),
            state=(tk.NORMAL if self._clipboard_has_text() else tk.DISABLED),
        )
        menu.add_command(
            label="Erase selection",
            command=self.delete_selected,
            state=(tk.NORMAL if has_selection else tk.DISABLED),
        )
        try:
            menu.tk_popup(x_root, y_root)
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass

    def _set_button_hover(self, btn: tk.Label, hovering: bool) -> None:
        is_selected = bool(getattr(btn, "_sargate_selected", False))
        if is_selected:
            self._sync_square_button_backing(btn, self.button_selected_bg)
            btn.configure(bg=self.button_selected_bg, fg=self.button_selected_text)
            return
        target_bg = self.button_hover if hovering else self.button_bg
        self._sync_square_button_backing(btn, target_bg)
        btn.configure(bg=target_bg, fg=TEXT)

    def _set_button_selected(self, btn: tk.Label, selected: bool) -> None:
        setattr(btn, "_sargate_selected", selected)
        if selected:
            self._sync_square_button_backing(btn, self.button_selected_bg)
            btn.configure(
                bg=self.button_selected_bg,
                fg=self.button_selected_text,
                relief=tk.SUNKEN,
                bd=max(1, self.frame_border_size),
                highlightbackground=PANEL_EDGE,
                highlightcolor=PANEL_EDGE,
                highlightthickness=1,
            )
        else:
            self._sync_square_button_backing(btn, self.button_bg)
            btn.configure(
                bg=self.button_bg,
                fg=TEXT,
                relief=tk.RAISED,
                bd=max(1, self.frame_border_size),
                highlightbackground=PANEL_EDGE,
                highlightcolor=PANEL_EDGE,
                highlightthickness=1,
            )

    def _make_ring_icon(self, key: str) -> ImageTk.PhotoImage:
        size = 40
        aa = 4
        hi = size * aa
        image = Image.new("RGBA", (hi, hi), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        cx = cy = hi / 2
        if key == "cyclopropane":
            sides = 3
            radius = 13 * aa
        elif key == "cyclobutane":
            sides = 4
            radius = 13 * aa
        elif key == "cyclopentane":
            sides = 5
            radius = 15 * aa
        elif key == "cycloheptane":
            sides = 7
            radius = 14 * aa
        else:
            sides = 6
            radius = 16 * aa
        start = -math.pi / 2
        pts = [
            (
                cx + radius * math.cos(start + 2 * math.pi * i / sides),
                cy + radius * math.sin(start + 2 * math.pi * i / sides),
            )
            for i in range(sides)
        ]
        line_color = _hex_to_rgb(TEXT, (49, 56, 70)) + (255,)
        dbl_color = line_color
        for i in range(sides):
            draw.line((pts[i], pts[(i + 1) % sides]), fill=line_color, width=3 * aa)
        if key == "benzene":
            for i in (0, 2, 4):
                self._draw_parallel_icon_bond(draw, pts[i], pts[(i + 1) % sides], dbl_color, aa)
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _make_bond_icon(self, key: str) -> ImageTk.PhotoImage:
        size = 40
        aa = 4
        hi = size * aa
        image = Image.new("RGBA", (hi, hi), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        line_color = _hex_to_rgb(TEXT, (49, 56, 70)) + (255,)
        cx = hi / 2
        cy = hi / 2
        left = 28
        right = hi - 28
        if key == "single":
            draw.line((left, cy, right, cy), fill=line_color, width=10)
        elif key == "double":
            draw.line((left, cy - 14, right, cy - 14), fill=line_color, width=8)
            draw.line((left, cy + 14, right, cy + 14), fill=line_color, width=8)
        elif key == "triple":
            draw.line((left, cy - 20, right, cy - 20), fill=line_color, width=7)
            draw.line((left, cy, right, cy), fill=line_color, width=7)
            draw.line((left, cy + 20, right, cy + 20), fill=line_color, width=7)
        elif key == "wedge":
            draw.polygon([(left, cy - 22), (left, cy + 22), (right, cy)], fill=line_color)
        elif key == "hashed":
            tip_x = right
            base_x = left + 8
            for i in range(6):
                t = i / 5.0
                x = base_x + (tip_x - base_x) * t
                half_h = 24 * (1.0 - t) + 2
                draw.line((x, cy - half_h, x, cy + half_h), fill=line_color, width=4)
        elif key == "chain":
            pts = [
                (left, cy + 18),
                (left + 34, cy - 4),
                (left + 68, cy + 18),
                (left + 102, cy - 4),
            ]
            draw.line(pts, fill=line_color, width=8, joint="curve")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _make_action_icon(self, key: str, bg: str | None = None, size: int = 40) -> ImageTk.PhotoImage:
        base_size = 40
        aa = 4
        hi = base_size * aa
        if bg is None:
            image = Image.new("RGBA", (hi, hi), (255, 255, 255, 0))
        else:
            bg_rgb = _hex_to_rgb(bg, (229, 231, 235))
            image = Image.new("RGBA", (hi, hi), bg_rgb + (255,))
        draw = ImageDraw.Draw(image)
        line_color = _hex_to_rgb(TEXT, (49, 56, 70)) + (255,)
        accent_color = _hex_to_rgb(TEXT, (49, 56, 70)) + (255,)
        cx = hi / 2
        cy = hi / 2
        if key == "save":
            outer = [(44, 34), (100, 34), (116, 50), (116, 126), (44, 126)]
            draw.line(outer + [outer[0]], fill=line_color, width=8, joint="curve")
            draw.line((100, 34, 100, 56, 116, 56), fill=line_color, width=8)
            draw.rectangle((58, 44, 88, 66), outline=line_color, width=6)
            draw.rectangle((58, 82, 102, 112), outline=line_color, width=6)
            image = image.resize((size, size), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)
        if key == "copy":
            font = _load_font_with_fallback(["Arial Bold.ttf", "arialbd.ttf", "Arial.ttf", "arial.ttf"], 42)
            draw.text((cx, cy), "Copy", fill=line_color, font=font, anchor="mm")
            image = image.resize((size, size), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)
        glyph = ACTION_GLYPHS.get(key)
        if glyph is not None:
            font = _load_font_with_fallback(
                ["Arial-Unicode.ttf", "Arial Unicode.ttf", "Arial.ttf", "arial.ttf"],
                118,
            )
            draw.text(
                (cx, cy - 6),
                glyph,
                fill=line_color,
                font=font,
                anchor="mm",
                stroke_width=3,
                stroke_fill=line_color,
            )
            image = image.resize((size, size), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)
        if key == "clean2d":
            # Brand new icon: magic wand + sparkle
            draw.line((52, 108, 108, 52), fill=line_color, width=8)
            draw.line((46, 114, 58, 102), fill=line_color, width=8)
            sx, sy = 108, 52
            draw.line((sx - 18, sy, sx + 18, sy), fill=line_color, width=5)
            draw.line((sx, sy - 18, sx, sy + 18), fill=line_color, width=5)
            draw.line((sx - 12, sy - 12, sx + 12, sy + 12), fill=line_color, width=4)
            draw.line((sx - 12, sy + 12, sx + 12, sy - 12), fill=line_color, width=4)
        elif key == "center":
            draw.ellipse((42, 42, hi - 42, hi - 42), outline=line_color, width=8)
            draw.line((cx, 26, cx, 58), fill=line_color, width=8)
            draw.line((cx, hi - 58, cx, hi - 26), fill=line_color, width=8)
            draw.line((26, cy, 58, cy), fill=line_color, width=8)
            draw.line((hi - 58, cy, hi - 26, cy), fill=line_color, width=8)
            draw.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill=line_color)
        elif key == "clear":
            draw.line((42, 42, hi - 42, hi - 42), fill=line_color, width=10)
            draw.line((42, hi - 42, hi - 42, 42), fill=line_color, width=10)
        elif key == "undo":
            pts = [(122, 56), (88, 56), (68, 74), (54, 88), (34, 88)]
            draw.line(pts, fill=line_color, width=9, joint="curve")
            self._draw_icon_arrow_head(draw, pts[-1], (-1.0, 0.0), line_color, 9, head_len=17, head_half_width=10)
        elif key == "redo":
            pts = [(38, 56), (72, 56), (92, 74), (106, 88), (126, 88)]
            draw.line(pts, fill=line_color, width=9, joint="curve")
            self._draw_icon_arrow_head(draw, pts[-1], (1.0, 0.0), line_color, 9, head_len=17, head_half_width=10)
        elif key == "select":
            cursor = [
                (50, 36),
                (96, 84),
                (77, 88),
                (95, 124),
                (80, 132),
                (63, 95),
                (44, 111),
            ]
            draw.polygon(cursor, fill=line_color)
        elif key == "erase":
            body = [(52, 100), (76, 64), (114, 64), (90, 100)]
            draw.polygon(body, outline=line_color, fill=None, width=8)
            draw.line((72, 64, 94, 100), fill=line_color, width=6)
            draw.line((46, 110, 120, 110), fill=line_color, width=7)
            draw.line((104, 106, 116, 94), fill=line_color, width=5)
        elif key == "erase_selection":
            body = [(56, 98), (78, 66), (112, 66), (90, 98)]
            draw.polygon(body, outline=line_color, fill=None, width=8)
            draw.line((72, 66, 92, 98), fill=line_color, width=5)
            draw.line((48, 44, 70, 44), fill=line_color, width=5)
            draw.line((48, 44, 48, 66), fill=line_color, width=5)
            draw.line((120, 44, 98, 44), fill=line_color, width=5)
            draw.line((120, 44, 120, 66), fill=line_color, width=5)
            draw.line((48, 120, 70, 120), fill=line_color, width=5)
            draw.line((48, 98, 48, 120), fill=line_color, width=5)
            draw.line((120, 120, 98, 120), fill=line_color, width=5)
            draw.line((120, 98, 120, 120), fill=line_color, width=5)
        elif key == "pan":
            draw.line((cx, 36, cx, hi - 36), fill=line_color, width=8)
            draw.line((36, cy, hi - 36, cy), fill=line_color, width=8)
            draw.polygon([(cx, 18), (cx - 14, 42), (cx + 14, 42)], fill=accent_color)
            draw.polygon([(cx, hi - 18), (cx - 14, hi - 42), (cx + 14, hi - 42)], fill=accent_color)
            draw.polygon([(18, cy), (42, cy - 14), (42, cy + 14)], fill=accent_color)
            draw.polygon([(hi - 18, cy), (hi - 42, cy - 14), (hi - 42, cy + 14)], fill=accent_color)
        elif key == "rotate":
            self._draw_icon_arc_with_arrow(
                draw,
                (34, 34, hi - 34, hi - 34),
                40,
                325,
                line_color,
                8,
                head_len=18,
                head_half_width=10,
            )
        elif key == "colored_labels":
            font = _load_font_with_fallback(["Arial Bold.ttf", "arialbd.ttf", "Arial.ttf", "arial.ttf"], 54)
            draw.text((16, cy), "C", fill=(49, 56, 70, 255), font=font, anchor="lm")
            draw.text((80, cy), "N", fill=(29, 78, 216, 255), font=font, anchor="mm")
            draw.text((144, cy), "O", fill=(220, 38, 38, 255), font=font, anchor="rm")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _configure_action_button_face(self, btn: tk.Label, key: str, storage_key: str, size: int = 40) -> None:
        icon = self._make_action_icon(key, None, size=size)
        self.action_button_images[storage_key] = icon
        btn.configure(
            text="",
            image=icon,
            compound="center",
            width=size,
            height=size,
            relief=tk.RAISED,
            bd=max(1, self.frame_border_size),
            highlightbackground=PANEL_EDGE,
            highlightcolor=PANEL_EDGE,
            highlightthickness=1,
        )

    def _draw_parallel_icon_bond(self, draw: ImageDraw.ImageDraw, p1: tuple[float, float], p2: tuple[float, float], color: tuple[int, int, int, int], aa_scale: int = 1) -> None:
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        nx = -dy / length
        ny = dx / length
        inset = 4.5 * aa_scale
        draw.line(
            ((x1 + nx * inset, y1 + ny * inset), (x2 + nx * inset, y2 + ny * inset)),
            fill=color,
            width=max(2, 2 * aa_scale),
        )

    def _draw_icon_arrow_head(
        self,
        draw: ImageDraw.ImageDraw,
        tip: tuple[float, float],
        direction: tuple[float, float],
        color: tuple[int, int, int, int],
        line_width: int,
        head_len: float = 16.0,
        head_half_width: float = 10.0,
    ) -> None:
        dx, dy = direction
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux
        tx, ty = tip
        p1 = (tx - ux * head_len + px * head_half_width, ty - uy * head_len + py * head_half_width)
        p2 = (tx - ux * head_len - px * head_half_width, ty - uy * head_len - py * head_half_width)
        draw.line((tip, p1), fill=color, width=line_width)
        draw.line((tip, p2), fill=color, width=line_width)

    def _draw_icon_arc_with_arrow(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: tuple[float, float, float, float],
        start_deg: float,
        end_deg: float,
        color: tuple[int, int, int, int],
        line_width: int,
        head_len: float = 16.0,
        head_half_width: float = 10.0,
    ) -> None:
        draw.arc(bbox, start=start_deg, end=end_deg, fill=color, width=line_width)
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx = (x1 - x0) / 2.0
        ry = (y1 - y0) / 2.0
        theta = math.radians(end_deg)
        tip = (cx + rx * math.cos(theta), cy + ry * math.sin(theta))
        tangent = (-rx * math.sin(theta), ry * math.cos(theta))
        if end_deg < start_deg:
            tangent = (-tangent[0], -tangent[1])
        self._draw_icon_arrow_head(draw, tip, tangent, color, line_width, head_len=head_len, head_half_width=head_half_width)

    def _build_structure_buttons(self, parent: tk.Widget) -> None:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(anchor="w")
        for label, callback, icon_key, widget_key in [
            ("Save image", self.save_image_dialog, "save", "save"),
            ("Clear", self.clear_all, "clear", "clear"),
            ("Undo", self.undo, "undo", "undo"),
            ("Redo", self.redo, "redo", "redo"),
        ]:
            btn = self._make_toggle_button(row, "", callback, square_size=40)
            self._configure_action_button_face(btn, icon_key, f"structure_{widget_key}", size=40)
            btn.pack(side="left", padx=0, pady=0)
            self.structure_button_widgets[widget_key] = btn
            self._bind_tooltip(btn, label)

    def _build_smiles_display(self, parent: tk.Widget) -> None:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=(0, 0), anchor="nw")
        smiles_text_label = tk.Label(row, text="SMILES", bg=PANEL, fg=TEXT, font=("Arial", 10), bd=0, highlightthickness=0)
        smiles_text_label.pack(side="left", padx=(0, 8))
        self.smiles_label_widget = smiles_text_label
        smiles_box = tk.Frame(row, height=39)
        smiles_box.pack(side="left", fill="x", expand=True)
        smiles_box.pack_propagate(False)
        smiles_label = tk.Label(smiles_box, textvariable=self.current_smiles_var, padx=6, pady=0)
        smiles_label.pack(fill="both", expand=True)
        self.smiles_display_frame = smiles_box
        self.smiles_display_label = smiles_label
        self._style_readonly_display_frame(smiles_box, smiles_label)
        copy_btn = self._make_toggle_button(row, "", self.copy_current_smiles, square_size=40)
        self._configure_action_button_face(copy_btn, "copy", "smiles_copy", size=40)
        copy_btn.configure(bg=self.button_bg, fg=TEXT)
        copy_btn.pack(side="left", padx=(8, 0), pady=0)
        self.copy_smiles_button = copy_btn
        self._bind_tooltip(copy_btn, "Copy SMILES")

    def _build_mode_buttons(self, parent: tk.Widget) -> None:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(anchor="w")
        for label, mode, icon_key in [
            ("Select", "select", "select"),
            ("Erase", "erase", "erase"),
            ("Pan", "pan", "pan"),
            ("Rotate", "rotate", "rotate"),
        ]:
            btn = self._make_toggle_button(row, "", lambda m=mode: self.toggle_mode(m), square_size=40)
            self._configure_action_button_face(btn, icon_key, f"mode_{mode}", size=40)
            btn.pack(side="left", padx=0, pady=0)
            self.mode_button_widgets[mode] = btn
            self._bind_tooltip(btn, label)

    def _build_aux_action_buttons(self, parent: tk.Widget) -> None:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(anchor="w", pady=0)
        for label, callback, icon_key, widget_key in [
            ("Clean 2D", self.clean_2d, "clean2d", "clean2d"),
            ("Horizontal flip", self.flip_selection_horizontal, "hflip", "hflip"),
            ("Vertical flip", self.flip_selection_vertical, "vflip", "vflip"),
            ("Show stereocenters", self.toggle_show_stereocenters, "stereocenters", "stereocenters"),
        ]:
            btn = self._make_toggle_button(row, "", callback, square_size=MINI_TOOL_BUTTON_SIZE)
            self._configure_action_button_face(btn, icon_key, f"aux_{widget_key}", size=MINI_TOOL_BUTTON_SIZE)
            btn.pack(side="left", padx=0, pady=0)
            self.aux_action_button_widgets[widget_key] = btn
            self._bind_tooltip(btn, label)

    def _build_atom_buttons(self, parent: tk.Widget) -> None:
        grid = tk.Frame(parent, bg=PANEL)
        grid.pack(anchor="w")
        for idx, symbol in enumerate(COMMON_ATOM_BUTTONS):
            btn = self._make_toggle_button(grid, symbol, lambda s=symbol: self.select_atom_symbol(s), square_size=40)
            self._configure_element_button_face(btn, symbol, 40)
            btn.grid(row=idx // 5, column=idx % 5, padx=0, pady=(0, 2) if idx // 5 == 0 else (0, 0))
            self.atom_button_widgets[symbol] = btn

    def _build_charge_periodic_group(self, parent: tk.Widget) -> None:
        top = tk.Frame(parent, bg=PANEL)
        top.pack(anchor="w", pady=(0, 0))
        button_inner_size = 40
        button_outer_size = button_inner_size + (2 * max(1, self.frame_border_size)) + 2
        periodic_total_width = button_outer_size * 3
        periodic_box = tk.Frame(top, width=periodic_total_width, height=button_outer_size, bg=PANEL)
        periodic_box.pack(side="left", anchor="w")
        periodic_box.pack_propagate(False)
        periodic_button = tk.Label(
            periodic_box,
            text="Periodic table",
            relief=tk.RAISED,
            bd=max(1, self.frame_border_size),
            bg=self.button_bg,
            fg=TEXT,
            font=("Arial", 9, "bold"),
            highlightbackground=PANEL_EDGE,
            highlightcolor=PANEL_EDGE,
            highlightthickness=1,
            cursor="hand2",
            padx=0,
            pady=0,
            anchor="center",
            justify="center",
        )
        periodic_button.pack(fill="both", expand=True)
        periodic_button.bind("<Button-1>", lambda _e: self.open_periodic_table_popup())
        periodic_button.bind("<Enter>", lambda _e, widget=periodic_button: self._set_button_hover(widget, True))
        periodic_button.bind("<Leave>", lambda _e, widget=periodic_button: self._set_button_hover(widget, False))

        misc_row = tk.Frame(parent, bg=PANEL)
        misc_row.pack(anchor="w", pady=(0, 0))
        r_btn = self._make_toggle_button(misc_row, "R", lambda: self.select_atom_symbol("R"), square_size=40)
        self._configure_element_button_face(r_btn, "R", 40)
        r_btn.pack(side="left", padx=(0, 0), pady=0)
        self.atom_button_widgets["R"] = r_btn
        self._bind_tooltip(r_btn, "R-group")
        self.charge_label_widget = None
        self.charge_display_frame = None
        self.charge_display_label = None
        self.charge_popup_menu = None

        minus_btn = self._make_toggle_button(misc_row, "–", lambda: self._adjust_charge(-1), square_size=40)
        self._configure_element_button_face(minus_btn, "–", 40)
        minus_btn.pack(side="left", padx=(0, 0), pady=0)
        self.charge_button_widgets["minus"] = minus_btn
        self._bind_tooltip(minus_btn, "Decrease charge")

        plus_btn = self._make_toggle_button(misc_row, "+", lambda: self._adjust_charge(+1), square_size=40)
        self._configure_element_button_face(plus_btn, "+", 40)
        plus_btn.pack(side="left", padx=(0, 0), pady=0)
        self.charge_button_widgets["plus"] = plus_btn
        self._bind_tooltip(plus_btn, "Increase charge")

    def _build_bond_buttons(self, parent: tk.Widget) -> None:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(anchor="w")
        for style_key, label in BOND_STYLE_DEFS:
            bond_img = self._make_bond_icon(style_key)
            self.ring_button_images[f"bond_{style_key}"] = bond_img
            btn = self._make_toggle_button(row, "", lambda s=style_key: self.select_bond_style(s), square_size=40)
            btn.configure(image=bond_img, compound="center")
            btn.pack(side="left", padx=0, pady=0)
            self.bond_button_widgets[style_key] = btn
            self._bind_tooltip(btn, label)

    def _build_ring_buttons(self, parent: tk.Widget) -> None:
        row = tk.Frame(parent, bg=BG)
        row.pack(anchor="w", pady=(0, 0))
        ring_tooltips = {
            "cyclopropane": "Cyclopropane",
            "cyclobutane": "Cyclobutane",
            "cyclohexane": "Cyclohexane",
            "benzene": "Benzene",
            "cyclopentane": "Cyclopentane",
            "cycloheptane": "Cycloheptane",
        }
        for key, _icon, _size, _aromatic in RING_TEMPLATES:
            ring_img = self._make_ring_icon(key)
            self.ring_button_images[key] = ring_img
            btn = tk.Label(
                row,
                image=ring_img,
                width=40,
                height=40,
                relief=tk.RAISED,
                bd=max(1, self.frame_border_size),
                bg=self.button_bg,
                highlightbackground=PANEL_EDGE,
                highlightcolor=PANEL_EDGE,
                highlightthickness=1,
                cursor="hand2",
            )
            btn.bind("<Button-1>", lambda _event, k=key: self.select_ring_template(k))
            btn.bind("<Enter>", lambda _event, widget=btn: self._set_button_hover(widget, True))
            btn.bind("<Leave>", lambda _event, widget=btn: self._set_button_hover(widget, False))
            btn.pack(side="left", padx=0, pady=0)
            self.ring_button_widgets[key] = btn
            self._bind_tooltip(btn, ring_tooltips.get(key, key))

    def _build_visibility_group(self, parent: tk.Widget) -> None:
        button_size = MINI_TOOL_BUTTON_SIZE
        row = tk.Frame(parent, bg=PANEL)
        row.pack(anchor="w", pady=0)

        center_btn = self._make_toggle_button(row, "", self.center_view, square_size=button_size)
        self._configure_action_button_face(center_btn, "center", "visibility_center", size=button_size)
        center_btn.pack(side="left", padx=0, pady=0)
        self.visibility_button_widgets["center"] = center_btn
        self._bind_tooltip(center_btn, "Center")

        c_btn = self._make_toggle_button(row, "C", self.cycle_carbon_labels, square_size=button_size)
        self._configure_element_button_face(c_btn, "C", button_size)
        c_btn.pack(side="left", padx=0, pady=0)
        self.visibility_button_widgets["carbons"] = c_btn
        self._bind_tooltip(c_btn, self._carbon_labels_tooltip())

        h_btn = self._make_toggle_button(row, "H", self.cycle_hydrogen_labels, square_size=button_size)
        self._configure_element_button_face(h_btn, "H", button_size)
        h_btn.pack(side="left", padx=0, pady=0)
        self.visibility_button_widgets["hydrogens"] = h_btn
        self._bind_tooltip(h_btn, self._hydrogen_labels_tooltip())

        colored_btn = self._make_toggle_button(row, "", self.toggle_colored_atoms, square_size=button_size)
        self._configure_action_button_face(colored_btn, "colored_labels", "visibility_colored", size=button_size)
        colored_btn.pack(side="left", padx=0, pady=0)
        self.visibility_button_widgets["colored"] = colored_btn
        self._bind_tooltip(colored_btn, self._colored_atoms_tooltip())

    def _build_mode_box(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent, style="SAR.TFrame", padding=0)
        box.pack(anchor="w", pady=(0, 10))
        row = tk.Frame(box, bg=PANEL)
        row.pack(anchor="w")
        for label, mode, icon_key in [
            ("Select", "select", "select"),
            ("Erase", "erase", "erase"),
            ("Pan", "pan", "pan"),
            ("Rotate", "rotate", "rotate"),
        ]:
            btn = self._make_toggle_button(row, "", lambda m=mode: self.toggle_mode(m), square_size=40)
            self._configure_action_button_face(btn, icon_key, f"mode_{mode}")
            btn.pack(side="left", padx=0, pady=0)
            self.mode_button_widgets[mode] = btn
            self._bind_tooltip(btn, label)

    def _build_atoms_box(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent, style="SAR.TFrame", padding=0)
        box.pack(anchor="w", pady=(0, 10))

        grid = tk.Frame(box, bg=PANEL)
        grid.pack(anchor="w")
        for idx, symbol in enumerate(COMMON_ATOM_BUTTONS):
            btn = self._make_toggle_button(grid, symbol, lambda s=symbol: self.select_atom_symbol(s), square_size=40)
            self._configure_element_button_face(btn, symbol, 40)
            btn.grid(row=idx // 5, column=idx % 5, padx=0, pady=0)
            self.atom_button_widgets[symbol] = btn

        misc_row = tk.Frame(box, bg=GRID)
        misc_row.pack(anchor="w", pady=(8, 0))
        charge_label = tk.Label(misc_row, text="Charge", bg=PANEL, fg=TEXT, font=("Arial", 10))
        charge_label.pack(side="left")
        charge_box = tk.Frame(misc_row)
        charge_box.pack(side="left", padx=(8, 10))
        charge_value_label = tk.Label(charge_box, textvariable=self.charge_var, cursor="hand2", width=6, padx=6, pady=2)
        charge_value_label.pack(fill="both", expand=True)
        self.charge_display_frame = charge_box
        self.charge_display_label = charge_value_label
        charge_popup = tk.Menu(charge_box, tearoff=False)
        for charge_option in ["-3", "-2", "-1", "0", "+1", "+2", "+3"]:
            charge_popup.add_command(
                label=charge_option,
                command=lambda value=charge_option: (self.charge_var.set(value), self._set_charge_from_ui()),
            )
        self.charge_popup_menu = charge_popup
        charge_box.bind("<Button-1>", lambda _e, w=charge_box: self._open_charge_popup(w))
        charge_value_label.bind("<Button-1>", lambda _e, w=charge_box: self._open_charge_popup(w))
        self._style_charge_display(charge_box, charge_value_label)
        ttk.Button(misc_row, text="Periodic table...", command=self.open_periodic_table_popup, style="SAR.TButton").pack(side="left")

    def _build_bonds_box(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent, style="SAR.TFrame", padding=0)
        box.pack(anchor="w", pady=(0, 10))

        row = tk.Frame(box, bg=PANEL)
        row.pack(anchor="w")
        for style_key, label in BOND_STYLE_DEFS:
            bond_img = self._make_bond_icon(style_key)
            self.ring_button_images[f"bond_{style_key}"] = bond_img
            btn = self._make_toggle_button(row, "", lambda s=style_key: self.select_bond_style(s), square_size=40)
            btn.configure(image=bond_img, compound="center")
            btn.pack(side="left", padx=0, pady=0)
            self.bond_button_widgets[style_key] = btn
            self._bind_tooltip(btn, label)

    def _build_rings_box(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent, style="SAR.TFrame", padding=0)
        box.pack(anchor="w", pady=(0, 10))
        row = tk.Frame(box, bg=PANEL)
        row.pack(anchor="w")
        ring_tooltips = {
            "cyclopropane": "Cyclopropane",
            "cyclobutane": "Cyclobutane",
            "cyclohexane": "Cyclohexane",
            "benzene": "Benzene",
            "cyclopentane": "Cyclopentane",
            "cycloheptane": "Cycloheptane",
        }
        for key, _icon, _size, _aromatic in RING_TEMPLATES:
            ring_img = self._make_ring_icon(key)
            self.ring_button_images[key] = ring_img
            btn = tk.Label(
                row,
                image=ring_img,
                width=40,
                height=40,
                relief=tk.RAISED,
                bd=max(1, self.frame_border_size),
                bg=self.button_bg,
                highlightbackground=PANEL_EDGE,
                highlightcolor=PANEL_EDGE,
                highlightthickness=1,
                cursor="hand2",
            )
            btn.bind("<Button-1>", lambda _event, k=key: self.select_ring_template(k))
            btn.bind("<Enter>", lambda _event, widget=btn: self._set_button_hover(widget, True))
            btn.bind("<Leave>", lambda _event, widget=btn: self._set_button_hover(widget, False))
            btn.pack(side="left", padx=0, pady=0)
            self.ring_button_widgets[key] = btn
            self._bind_tooltip(btn, ring_tooltips.get(key, key))
    def _build_structure_box(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent, style="SAR.TFrame", padding=0)
        box.pack(anchor="w", pady=(0, 10))
        row = tk.Frame(box, bg=PANEL)
        row.pack(anchor="w")
        for label, callback, icon_key, widget_key in [
            ("Clean 2D", self.clean_2d, "clean2d", "clean2d"),
            ("Center", self.center_view, "center", "center"),
            ("Clear", self.clear_all, "clear", "clear"),
            ("Undo", self.undo, "undo", "undo"),
            ("Redo", self.redo, "redo", "redo"),
        ]:
            btn = self._make_toggle_button(row, "", callback, square_size=40)
            self._configure_action_button_face(btn, icon_key, f"structure_{widget_key}")
            btn.pack(side="left", padx=0, pady=0)
            self.structure_button_widgets[widget_key] = btn
            self._bind_tooltip(btn, label)
        ttk.Label(box, text="SMILES", style="SAR.TLabel").pack(anchor="w", pady=(8, 0))
        smiles_box = tk.Frame(box)
        smiles_box.pack(fill="x", pady=(2, 0))
        smiles_label = tk.Label(smiles_box, textvariable=self.current_smiles_var, padx=6, pady=4)
        smiles_label.pack(fill="x", expand=True)
        self.smiles_display_frame = smiles_box
        self.smiles_display_label = smiles_label
        self._style_readonly_display_frame(smiles_box, smiles_label)

    def _build_canvas(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent, style="SAR.TFrame", padding=10)
        box.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            box,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground=PANEL_EDGE,
        )
        setattr(self.canvas, "_sargate_chem_canvas", True)
        self.canvas.pack(fill="both", expand=True)

    def _bind_events(self) -> None:
        assert self.canvas is not None
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<Button-3>", self.on_right_down)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_up)
        self.canvas.bind("<Button-2>", self.on_right_down)
        self.canvas.bind("<B2-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_right_up)
        self.canvas.bind("<Control-Button-1>", self.on_canvas_context_menu)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self.on_mousewheel_linux)
        self.canvas.bind("<Command-v>", self.on_canvas_paste)
        self.canvas.bind("<Control-v>", self.on_canvas_paste)
        self.root.bind("<Command-c>", self.on_copy_selection)
        self.root.bind("<Control-c>", self.on_copy_selection)
        self.root.bind("<Command-x>", self.on_cut_selection)
        self.root.bind("<Control-x>", self.on_cut_selection)
        self.root.bind("<Command-a>", self.select_all_canvas)
        self.root.bind("<Control-a>", self.select_all_canvas)
        self.root.bind("<Delete>", lambda _e: self.delete_selected())
        self.root.bind("<BackSpace>", lambda _e: self.delete_selected())
        self.root.bind("<Escape>", lambda _e: self.clear_selection())
        self.root.bind("<Command-z>", lambda _e: self.undo())
        self.root.bind("<Command-y>", lambda _e: self.redo())
        self.root.bind("<Command-Z>", lambda _e: self.redo())
        self.root.bind("<Control-z>", lambda _e: self.undo())
        self.root.bind("<Control-y>", lambda _e: self.redo())
        self.root.bind("<Control-Z>", lambda _e: self.redo())
        self.root.bind("<KeyPress>", self._on_keypress)

        self._refresh_atom_buttons()
        self._refresh_bond_buttons()
        self._refresh_ring_buttons()
        self._refresh_mode_buttons()

    def _load_theme_payload(self) -> dict[str, Any]:
        payload = {
            "main_bg": BG,
            "panel_bg": PANEL,
            "text": TEXT,
            "border": PANEL_EDGE,
            "border_shadow": _mix_hex(PANEL_EDGE, "#000000", 0.15),
            "frame_bg": GRID,
            "title_bar_bg": SELECTED,
            "menu_bar_bg": PANEL,
            "tabs_color": _mix_hex(PANEL, TEXT, 0.08),
            "tabs_hovered": _mix_hex(PANEL, TEXT, 0.14),
            "tabs_active": SELECTED,
            "button_color": _mix_hex(PANEL, TEXT, 0.08),
            "button_hovered": _mix_hex(PANEL, TEXT, 0.14),
            "button_active": _mix_hex(PANEL, TEXT, 0.22),
            "checkmark_color": SELECTED,
            "slider_grab": SELECTED,
            "frame_border_size": 1,
            "window_rounding": 8,
            "frame_rounding": 6,
            "tab_rounding": 5,
            "focus_nonce": 0,
        }
        if not self.theme_sync_path:
            return payload
        try:
            with open(self.theme_sync_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                payload.update(loaded)
        except Exception:
            pass
        return payload

    def apply_runtime_theme(self, payload: dict[str, Any]) -> None:
        global BG, GRID, TEXT, PANEL, PANEL_EDGE

        self._theme_payload = dict(payload)
        BG = str(payload.get("main_bg") or BG)
        PANEL = str(payload.get("panel_bg") or PANEL)
        TEXT = str(payload.get("text") or TEXT)
        PANEL_EDGE = str(payload.get("border") or PANEL_EDGE)
        GRID = str(payload.get("frame_bg") or GRID)

        self.button_bg = str(payload.get("button_color") or PANEL)
        self.button_hover = str(payload.get("button_hovered") or self.button_bg)
        self.button_active = str(payload.get("button_active") or self.button_hover)
        self.button_selected_bg = str(payload.get("tabs_active") or self.button_active)
        self.button_selected_text = _contrast_text(self.button_selected_bg)
        self.window_border_size = max(1, _safe_int(payload.get("frame_border_size"), 1))
        self.frame_border_size = max(1, _safe_int(payload.get("frame_border_size"), 1))
        self.window_rounding = max(0, _safe_int(payload.get("window_rounding"), 8))
        self.frame_rounding = max(0, _safe_int(payload.get("frame_rounding"), 6))
        self.tab_rounding = max(0, _safe_int(payload.get("tab_rounding"), 5))

        self.root.configure(bg=BG, highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE, bd=self.window_border_size)
        self.style.configure("SAR.TFrame", background=PANEL)
        self.style.configure("SAR.TLabelframe", background=PANEL, foreground=TEXT, bordercolor=PANEL_EDGE, relief="solid", borderwidth=self.frame_border_size)
        self.style.configure("SAR.TLabelframe.Label", background=PANEL, foreground=TEXT)
        self.style.configure("SAR.TLabel", background=PANEL, foreground=TEXT)
        self.style.configure(
            "SAR.TButton",
            background=self.button_bg,
            foreground=TEXT,
            bordercolor=PANEL_EDGE,
            darkcolor=self.button_bg,
            lightcolor=self.button_bg,
            focusthickness=0,
            focuscolor=self.button_bg,
            relief="flat",
            borderwidth=max(1, self.frame_border_size),
            padding=max(6, self.frame_rounding + 2),
        )
        self.style.map(
            "SAR.TButton",
            background=[("active", self.button_hover), ("pressed", self.button_active)],
            foreground=[("active", TEXT), ("pressed", TEXT)],
            bordercolor=[("active", PANEL_EDGE), ("pressed", PANEL_EDGE)],
        )
        if self.canvas is not None:
            self.canvas.configure(bg="#ffffff", highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE, bd=self.frame_border_size)
        if self.molblock_text is not None:
            self.molblock_text.configure(
                bg=GRID,
                fg=TEXT,
                insertbackground=TEXT,
                highlightbackground=PANEL_EDGE,
                highlightcolor=PANEL_EDGE,
                relief="solid",
                borderwidth=self.frame_border_size,
            )
        for entry in self.styled_entries:
            try:
                self._style_entry_widget(entry)
            except Exception:
                pass
        for option_menu in self.styled_option_menus:
            try:
                self._style_charge_menu_widget(option_menu)
            except Exception:
                pass
        self._apply_palette_recursive(self.root)
        if self.smiles_display_frame is not None and self.smiles_display_label is not None:
            try:
                self._style_readonly_display_frame(self.smiles_display_frame, self.smiles_display_label)
            except Exception:
                pass
        if self.smiles_label_widget is not None:
            try:
                self._style_plain_toolbar_label(self.smiles_label_widget)
            except Exception:
                pass
        if self.charge_display_frame is not None and self.charge_display_label is not None:
            try:
                self._style_charge_display(self.charge_display_frame, self.charge_display_label)
            except Exception:
                pass
        if self.charge_label_widget is not None:
            try:
                self._style_plain_toolbar_label(self.charge_label_widget)
            except Exception:
                pass
        for key, btn in self.ring_button_widgets.items():
            ring_img = self._make_ring_icon(key)
            self.ring_button_images[key] = ring_img
            btn.configure(image=ring_img, bg=self.button_bg, bd=max(1, self.frame_border_size), highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
        for key, btn in self.bond_button_widgets.items():
            bond_img = self._make_bond_icon(key)
            self.ring_button_images[f"bond_{key}"] = bond_img
            btn.configure(image=bond_img, bg=self.button_bg, bd=max(1, self.frame_border_size), highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
        for key, btn in self.mode_button_widgets.items():
            self._configure_action_button_face(btn, key, f"mode_{key}", size=40)
            btn.configure(bg=self.button_bg, bd=max(1, self.frame_border_size), highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
        for key, btn in self.structure_button_widgets.items():
            self._configure_action_button_face(btn, key, f"structure_{key}", size=40)
            btn.configure(bg=self.button_bg, bd=max(1, self.frame_border_size), highlightthickness=0, highlightbackground=self.button_bg, highlightcolor=self.button_bg)
        for key, btn in self.aux_action_button_widgets.items():
            self._configure_action_button_face(btn, key, f"aux_{key}", size=MINI_TOOL_BUTTON_SIZE)
            btn.configure(bg=self.button_bg, bd=max(1, self.frame_border_size), highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
        if self.copy_smiles_button is not None:
            self._configure_action_button_face(self.copy_smiles_button, "copy", "smiles_copy", size=40)
            self.copy_smiles_button.configure(
                bg=self.button_bg,
                fg=TEXT,
                bd=max(1, self.frame_border_size),
                highlightbackground=PANEL_EDGE,
                highlightcolor=PANEL_EDGE,
                highlightthickness=1,
            )
        for symbol, btn in self.atom_button_widgets.items():
            self._configure_element_button_face(btn, symbol, 40)
            btn.configure(bg=self.button_bg, bd=max(1, self.frame_border_size), highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
        for key, btn in self.charge_button_widgets.items():
            charge_symbol = "–" if key == "minus" else "+"
            self._configure_element_button_face(btn, charge_symbol, 40)
            btn.configure(bg=self.button_bg, bd=max(1, self.frame_border_size), highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
        visibility_center_btn = self.visibility_button_widgets.get("center")
        if visibility_center_btn is not None:
            self._configure_action_button_face(visibility_center_btn, "center", "visibility_center", size=MINI_TOOL_BUTTON_SIZE)
            visibility_center_btn.configure(bg=self.button_bg, bd=max(1, self.frame_border_size), highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
        visibility_carbons_btn = self.visibility_button_widgets.get("carbons")
        if visibility_carbons_btn is not None:
            self._configure_element_button_face(visibility_carbons_btn, "C", MINI_TOOL_BUTTON_SIZE)
            visibility_carbons_btn.configure(bg=self.button_bg, bd=max(1, self.frame_border_size), highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
        visibility_hydrogens_btn = self.visibility_button_widgets.get("hydrogens")
        if visibility_hydrogens_btn is not None:
            self._configure_element_button_face(visibility_hydrogens_btn, "H", MINI_TOOL_BUTTON_SIZE)
            visibility_hydrogens_btn.configure(bg=self.button_bg, bd=max(1, self.frame_border_size), highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
        visibility_colored_btn = self.visibility_button_widgets.get("colored")
        if visibility_colored_btn is not None:
            self._configure_action_button_face(visibility_colored_btn, "colored_labels", "visibility_colored", size=MINI_TOOL_BUTTON_SIZE)
            visibility_colored_btn.configure(bg=self.button_bg, bd=max(1, self.frame_border_size), highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
        self._refresh_periodic_popup_colors()

        self._refresh_atom_buttons()
        self._refresh_bond_buttons()
        self._refresh_ring_buttons()
        self._refresh_mode_buttons()
        self._refresh_charge_buttons()
        self._refresh_aux_action_buttons()
        self._refresh_visibility_buttons()
        self.render_all()

    def _focus_window(self) -> None:
        self.root.deiconify()
        self.root.lift()
        try:
            self.root.attributes("-topmost", True)
            self.root.after(250, lambda: self.root.attributes("-topmost", False))
        except Exception:
            pass
        try:
            self.root.focus_force()
        except Exception:
            pass

    def _poll_theme_sync(self) -> None:
        if self.theme_sync_path:
            try:
                mtime = os.path.getmtime(self.theme_sync_path)
            except OSError:
                mtime = None
            if mtime is not None and mtime != self._theme_sync_mtime:
                self._theme_sync_mtime = mtime
                payload = self._load_theme_payload()
                self.apply_runtime_theme(payload)
                focus_nonce = _safe_int(payload.get("focus_nonce"), 0)
                if focus_nonce != self._last_focus_nonce:
                    self._last_focus_nonce = focus_nonce
                    self._focus_window()
        self.root.after(350, self._poll_theme_sync)

    def _apply_palette_recursive(self, widget: tk.Misc) -> None:
        for child in widget.winfo_children():
            try:
                if isinstance(child, tk.Canvas):
                    if bool(getattr(child, "_sargate_chem_canvas", False)):
                        child.configure(bg="#ffffff", highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
                    else:
                        child.configure(bg=PANEL, highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
                elif isinstance(child, tk.Text):
                    child.configure(bg=GRID, fg=TEXT, insertbackground=TEXT, highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE)
                elif isinstance(child, tk.Label):
                    child.configure(
                        bg=self.button_bg,
                        fg=TEXT,
                        highlightbackground=PANEL_EDGE,
                        highlightcolor=PANEL_EDGE,
                        highlightthickness=1,
                        bd=max(1, self.frame_border_size),
                    )
                elif isinstance(child, tk.Toplevel):
                    child.configure(bg=BG, highlightbackground=PANEL_EDGE, highlightcolor=PANEL_EDGE, bd=self.window_border_size)
                elif isinstance(child, (tk.Frame, tk.LabelFrame)):
                    child.configure(
                        bg=PANEL,
                        highlightbackground=PANEL_EDGE,
                        highlightcolor=PANEL_EDGE,
                        bd=max(1, self.frame_border_size),
                    )
            except Exception:
                pass
            self._apply_palette_recursive(child)

    # ------------------------------------------------------------------
    # Toolbar state
    # ------------------------------------------------------------------
    def _refresh_atom_buttons(self) -> None:
        control_mode_active = self.mode in {"select", "erase", "pan", "rotate"} or self.current_charge != 0
        for symbol, btn in self.atom_button_widgets.items():
            self._set_button_selected(btn, (not control_mode_active) and symbol == self.selected_atom_symbol)

    def _refresh_bond_buttons(self) -> None:
        control_mode_active = self.mode in {"select", "erase", "pan", "rotate"} or self.current_charge != 0
        for style, btn in self.bond_button_widgets.items():
            self._set_button_selected(
                btn,
                (not control_mode_active) and self._show_bond_button_selection and style == self.selected_bond_style,
            )

    def _refresh_ring_buttons(self) -> None:
        control_mode_active = self.mode in {"select", "erase", "pan", "rotate"} or self.current_charge != 0
        for key, btn in self.ring_button_widgets.items():
            self._set_button_selected(btn, (not control_mode_active) and self.mode == "ring" and key == self.selected_ring_template)

    def _refresh_mode_buttons(self) -> None:
        for mode, btn in self.mode_button_widgets.items():
            self._set_button_selected(btn, mode == self.mode)

    def _refresh_charge_buttons(self) -> None:
        plus_btn = self.charge_button_widgets.get("plus")
        minus_btn = self.charge_button_widgets.get("minus")
        if plus_btn is not None:
            self._set_button_selected(plus_btn, self.current_charge > 0)
        if minus_btn is not None:
            self._set_button_selected(minus_btn, self.current_charge < 0)

    def _refresh_aux_action_buttons(self) -> None:
        for key, btn in self.aux_action_button_widgets.items():
            self._set_button_selected(btn, key == "stereocenters" and self.show_stereocenters)

    def _refresh_visibility_buttons(self) -> None:
        colored_btn = self.visibility_button_widgets.get("colored")
        if colored_btn is not None:
            self._set_button_selected(colored_btn, bool(self.colored_atoms_var.get()))

    def _set_display_option(self, variable: tk.StringVar, value: str) -> None:
        variable.set(value)
        self.render_all()

    def _carbon_labels_tooltip(self) -> str:
        return f"C labels: {self.carbons_display_var.get().strip()}"

    def _hydrogen_labels_tooltip(self) -> str:
        return f"H labels: {self.hydrogens_display_var.get().strip()}"

    def _colored_atoms_tooltip(self) -> str:
        return f"Colored atoms: {'On' if self.colored_atoms_var.get() else 'Off'}"

    def _cycle_display_option(self, variable: tk.StringVar, options: list[str]) -> None:
        current = variable.get().strip()
        try:
            idx = options.index(current)
        except ValueError:
            idx = -1
        variable.set(options[(idx + 1) % len(options)])
        self.render_all()

    def cycle_carbon_labels(self) -> None:
        self._cycle_display_option(self.carbons_display_var, ["None", "Terminal", "All"])
        btn = self.visibility_button_widgets.get("carbons")
        if btn is not None:
            self._bind_tooltip(btn, self._carbon_labels_tooltip())

    def cycle_hydrogen_labels(self) -> None:
        self._cycle_display_option(self.hydrogens_display_var, ["Hetero and terminal", "Hetero", "All", "None"])
        btn = self.visibility_button_widgets.get("hydrogens")
        if btn is not None:
            self._bind_tooltip(btn, self._hydrogen_labels_tooltip())

    def toggle_colored_atoms(self) -> None:
        self.colored_atoms_var.set(not self.colored_atoms_var.get())
        self._refresh_visibility_buttons()
        btn = self.visibility_button_widgets.get("colored")
        if btn is not None:
            self._bind_tooltip(btn, self._colored_atoms_tooltip())
        self.render_all()

    def select_atom_symbol(self, symbol: str) -> None:
        self.current_charge = 0
        self.charge_var.set("0")
        self._stored_charge_atom_symbol = None
        self.selected_atom_symbol = symbol
        self._show_bond_button_selection = False
        self.set_mode("atom")
        self._refresh_atom_buttons()
        self._refresh_bond_buttons()
        self._refresh_ring_buttons()
        self._refresh_charge_buttons()

    def select_bond_style(self, style: str) -> None:
        self.current_charge = 0
        self.charge_var.set("0")
        self._stored_charge_atom_symbol = None
        if style == "chain":
            self.selected_atom_symbol = "C"
        self.selected_bond_style = style
        self._show_bond_button_selection = True
        self.set_mode("bond")
        self._refresh_atom_buttons()
        self._refresh_bond_buttons()
        self._refresh_ring_buttons()
        self._refresh_charge_buttons()

    def select_ring_template(self, ring_key: str) -> None:
        self.current_charge = 0
        self.charge_var.set("0")
        self._stored_charge_atom_symbol = None
        self.selected_ring_template = ring_key
        self.selected_atom_symbol = "C"
        self._show_bond_button_selection = False
        self.set_mode("ring")
        self._refresh_atom_buttons()
        self._refresh_bond_buttons()
        self._refresh_ring_buttons()

    def toggle_mode(self, mode: str) -> None:
        if self.mode == mode:
            restored_symbol = self._stored_control_atom_symbol or self.selected_atom_symbol
            self._stored_control_atom_symbol = None
            self.set_mode(None)
            self.select_atom_symbol(restored_symbol)
            return
        if mode in {"select", "erase", "pan", "rotate"}:
            if self.current_charge != 0:
                self.current_charge = 0
                self.charge_var.set("0")
                self._stored_charge_atom_symbol = None
            self._stored_control_atom_symbol = self.selected_atom_symbol
        self.set_mode(mode)

    def set_mode(self, mode: str | None) -> None:
        self.mode = mode
        self.pending_draw_start_atom_id = None
        self.pending_draw_start_pos = None
        self.pending_draw_mode = None
        self.chain_drag_side_sign = None
        self.preview_world_pos = None
        self.preview_world_path = []
        self.pending_ring_active = False
        self.preview_ring_points = []
        self.preview_ring_bonds = []
        self.preview_ring_commit = None
        self.is_rotating = False
        self.selection_rect_start = None
        self.selection_rect_end = None
        self.erase_rect_start = None
        self.erase_rect_end = None
        self.erase_hit_atom_id = None
        self.erase_hit_bond_id = None
        self.preview_rect_atom_ids = set()
        self.preview_rect_bond_ids = set()
        self.hover_atom_id = None
        self.hover_bond_id = None
        self._refresh_atom_buttons()
        self._refresh_bond_buttons()
        self._refresh_ring_buttons()
        self._refresh_mode_buttons()
        self._refresh_charge_buttons()
        self._update_ring_hover_preview()
        self.set_status(f"Mode set to {mode}" if mode is not None else "Mode cleared")
        self.render_all()

    def _update_ring_hover_preview(self) -> None:
        if self.mode != "ring" or self.canvas is None or self.mouse_screen_pos is None:
            self.pending_ring_active = False
            self.preview_ring_points = []
            self.preview_ring_bonds = []
            self.preview_ring_commit = None
            return
        wx, wy = self.screen_to_world(*self.mouse_screen_pos)
        self.pending_ring_active = True
        self.preview_ring_points, self.preview_ring_bonds, self.preview_ring_commit = self._ring_preview_for_cursor(wx, wy)

    def _set_charge_from_ui(self) -> None:
        try:
            self.current_charge = int(self.charge_var.get())
        except Exception:
            self.current_charge = 0
        self._refresh_charge_buttons()

    def _adjust_charge(self, delta: int) -> None:
        target_charge = 1 if delta > 0 else -1
        if self.current_charge == target_charge:
            restored_symbol = self._stored_charge_atom_symbol or self.selected_atom_symbol
            self.current_charge = 0
            self.charge_var.set("0")
            self._stored_charge_atom_symbol = None
            self.select_atom_symbol(restored_symbol)
            return
        if self.mode in {"select", "erase", "pan", "rotate", "ring"}:
            self.set_mode(None)
        if self.current_charge == 0:
            self._stored_charge_atom_symbol = self.selected_atom_symbol
        self.current_charge = target_charge
        if self.current_charge > 0:
            self.charge_var.set(f"+{self.current_charge}")
        else:
            self.charge_var.set(str(self.current_charge))
        self._show_bond_button_selection = False
        self._refresh_charge_buttons()
        self._refresh_atom_buttons()
        self._refresh_bond_buttons()
        self._refresh_ring_buttons()
        self.set_status(f"Charge set to {self.charge_var.get()}")
        self.render_all()
        self.render_all()

    def _on_atom_shortcut(self, symbol: str) -> str:
        self.select_atom_symbol(symbol)
        return "break"

    def _flush_atom_key_buffer(self) -> None:
        self._atom_key_after_id = None
        token = self._atom_key_buffer.lower()
        self._atom_key_buffer = ""
        mapping = {
            "h": "H",
            "c": "C",
            "n": "N",
            "o": "O",
            "p": "P",
            "r": "R",
            "s": "S",
            "f": "F",
            "i": "I",
            "cl": "Cl",
            "br": "Br",
        }
        symbol = mapping.get(token)
        if symbol is not None:
            self.select_atom_symbol(symbol)

    def _on_keypress(self, event: tk.Event[tk.Misc]) -> str | None:
        if event.state & 0x4 or event.state & 0x8:
            return None
        char = (event.keysym or event.char or "").lower()
        if char not in {"h", "c", "n", "o", "p", "s", "f", "i", "b", "l", "r"}:
            return None
        if self._atom_key_after_id is not None:
            try:
                self.root.after_cancel(self._atom_key_after_id)
            except Exception:
                pass
            self._atom_key_after_id = None
        candidate = (self._atom_key_buffer + char).lower()
        valid_tokens = {"h", "c", "n", "o", "p", "r", "s", "f", "i", "cl", "br", "b"}
        valid_prefixes = {"c", "b"}
        if candidate in valid_tokens:
            if candidate in valid_prefixes:
                self._atom_key_buffer = candidate
                self._atom_key_after_id = self.root.after(220, self._flush_atom_key_buffer)
            else:
                self._atom_key_buffer = candidate
                self._flush_atom_key_buffer()
            return "break"
        if candidate in {"cl", "br"}:
            self._atom_key_buffer = candidate
            self._flush_atom_key_buffer()
            return "break"
        if char in valid_tokens or char in valid_prefixes:
            self._atom_key_buffer = char
            if char in valid_prefixes:
                self._atom_key_after_id = self.root.after(220, self._flush_atom_key_buffer)
            else:
                self._flush_atom_key_buffer()
            return "break"
        self._atom_key_buffer = ""
        return None

    def open_periodic_table_popup(self) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("Periodic Table")
        popup.transient(self.root)
        popup.resizable(False, False)
        self.periodic_popup_window = popup
        popup.protocol("WM_DELETE_WINDOW", lambda p=popup: (setattr(self, "periodic_popup_window", None), p.destroy()))
        self.periodic_atom_buttons = []
        self.periodic_popup_labels = []
        holder = tk.Frame(popup, bg=PANEL, padx=10, pady=10)
        holder.pack(fill="both", expand=True)
        controls = tk.Frame(holder, bg=PANEL)
        controls.grid(row=0, column=0, columnspan=18, sticky="ew", pady=(0, 8))
        tk.Label(controls, text="Color by:", bg=PANEL, fg=TEXT, font=("Arial", 10, "bold")).pack(side="left")
        tk.Radiobutton(
            controls,
            text="CPK",
            value="cpk",
            variable=self.periodic_color_mode_var,
            command=self._refresh_periodic_popup_colors,
            bg=PANEL,
            fg=TEXT,
            selectcolor=PANEL,
            activebackground=PANEL,
            activeforeground=TEXT,
            highlightthickness=0,
            bd=0,
            font=("Arial", 10),
        ).pack(side="left", padx=(8, 0))
        tk.Radiobutton(
            controls,
            text="Class",
            value="class",
            variable=self.periodic_color_mode_var,
            command=self._refresh_periodic_popup_colors,
            bg=PANEL,
            fg=TEXT,
            selectcolor=PANEL,
            activebackground=PANEL,
            activeforeground=TEXT,
            highlightthickness=0,
            bd=0,
            font=("Arial", 10),
        ).pack(side="left", padx=(8, 0))
        for symbol, row, col in PERIODIC_LAYOUT:
            grid_row = row + 1
            if symbol in {"La-Lu", "Ac-Lr"}:
                spacer = tk.Frame(
                    holder,
                    bg=PANEL,
                    width=40,
                    height=40,
                    bd=0,
                    highlightthickness=0,
                )
                spacer.grid(row=grid_row, column=col, padx=2, pady=2)
                spacer.grid_propagate(False)
            else:
                btn = self._make_toggle_button(holder, "", lambda s=symbol: self._select_atom_from_popup(s, popup), square_size=40)
                self._configure_periodic_popup_button_face(btn, symbol)
                btn.bind("<Enter>", lambda _event, widget=btn: self._set_periodic_popup_hover(widget, True))
                btn.bind("<Leave>", lambda _event, widget=btn: self._set_periodic_popup_hover(widget, False))
                btn.grid(row=grid_row, column=col, padx=2, pady=2)
                self.periodic_atom_buttons.append((symbol, btn))

    def _select_atom_from_popup(self, symbol: str, popup: tk.Toplevel) -> None:
        self.select_atom_symbol(symbol)
        popup.destroy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def current_element(self) -> str:
        return self.selected_atom_symbol

    def _r_group_index(self, label: str) -> int | None:
        if not isinstance(label, str):
            return None
        text = label.strip()
        if len(text) < 2 or not text.upper().startswith("R"):
            return None
        suffix = text[1:]
        if not suffix.isdigit():
            return None
        try:
            return int(suffix)
        except Exception:
            return None

    def _next_r_group_label(self) -> str:
        max_index = 0
        for atom in self.atoms:
            idx = self._r_group_index(atom.label)
            if idx is not None and idx > max_index:
                max_index = idx
        return f"R{max_index + 1}"

    def _renumber_r_group_labels(self) -> None:
        r_atoms: list[tuple[int, EditorAtom]] = []
        for atom in self.atoms:
            idx = self._r_group_index(atom.label)
            if idx is not None:
                r_atoms.append((idx, atom))
        if not r_atoms:
            return
        r_atoms.sort(key=lambda item: (item[0], item[1].id))
        for new_idx, (_old_idx, atom) in enumerate(r_atoms, start=1):
            atom.label = f"R{new_idx}"

    def current_bond_style_def(self) -> tuple[float, bool]:
        if self.selected_bond_style == "double":
            return 2.0, False
        if self.selected_bond_style == "triple":
            return 3.0, False
        if self.selected_bond_style == "aromatic":
            return 1.5, True
        return 1.0, False

    def _effective_draw_bond_style(self) -> str:
        return "single" if self.selected_bond_style == "chain" else self.selected_bond_style

    def _clicked_existing_bond_target_style(self, current_style: str) -> str:
        selected = self.selected_bond_style
        linear_cycle = ["single", "double", "triple"]
        if selected in linear_cycle:
            if selected != "single":
                return selected
            if current_style in linear_cycle:
                return linear_cycle[(linear_cycle.index(current_style) + 1) % len(linear_cycle)]
            return selected
        if selected in {"wedge", "hashed"}:
            return selected
        return selected

    def bond_order_value(self, style: str) -> float:
        if style == "double":
            return 2.0
        if style == "triple":
            return 3.0
        if style == "aromatic":
            return 1.5
        return 1.0

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self.offset_x + x * self.scale, self.offset_y - y * self.scale

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx - self.offset_x) / self.scale, (self.offset_y - sy) / self.scale

    def _canvas_size(self) -> tuple[float, float]:
        if self.canvas is None:
            return float(CANVAS_WIDTH), float(CANVAS_HEIGHT)
        self.root.update_idletasks()
        return float(max(1, self.canvas.winfo_width())), float(max(1, self.canvas.winfo_height()))

    def _invalidate_indices(self) -> None:
        self._atom_index = {}
        self._bond_index = {}

    def _ensure_indices(self) -> None:
        if len(self._atom_index) != len(self.atoms):
            self._atom_index = {atom.id: atom for atom in self.atoms}
        if len(self._bond_index) != len(self.bonds):
            self._bond_index = {bond.id: bond for bond in self.bonds}

    def _atom_by_id(self, atom_id: int | None) -> EditorAtom | None:
        if atom_id is None:
            return None
        self._ensure_indices()
        return self._atom_index.get(atom_id)

    def _bond_by_id(self, bond_id: int | None) -> EditorBond | None:
        if bond_id is None:
            return None
        self._ensure_indices()
        return self._bond_index.get(bond_id)

    def _selected_fragment_atom_ids(self) -> set[int]:
        atom_ids = set(self.selected_atom_ids)
        for bond_id in self.selected_bond_ids:
            bond = self._bond_by_id(bond_id)
            if bond is None:
                continue
            atom_ids.add(bond.a1)
            atom_ids.add(bond.a2)
        return atom_ids

    def _closest_atom_in_group(self, atom_ids: set[int], wx: float, wy: float) -> int | None:
        best_atom_id: int | None = None
        best_dist = float("inf")
        for atom_id in atom_ids:
            atom = self._atom_by_id(atom_id)
            if atom is None:
                continue
            dist = math.dist((atom.x, atom.y), (wx, wy))
            if dist < best_dist:
                best_dist = dist
                best_atom_id = atom_id
        return best_atom_id

    def _atom_degree(self, atom_id: int) -> int:
        return sum(1 for bond in self.bonds if bond.a1 == atom_id or bond.a2 == atom_id)

    def _single_neighbor_atom(self, atom_id: int) -> EditorAtom | None:
        neighbor_ids = self._neighbor_ids(atom_id)
        if len(neighbor_ids) != 1:
            return None
        return self._atom_by_id(neighbor_ids[0])

    def _is_hetero_atom(self, atom: EditorAtom | None) -> bool:
        if atom is None:
            return False
        return atom.element not in {"C", "H", "*"}

    def _should_show_carbon_label(self, atom: EditorAtom) -> bool:
        if self._atom_degree(atom.id) == 0:
            return True
        mode = self.carbons_display_var.get().strip()
        if mode == "All":
            return True
        if mode == "Terminal":
            return self._atom_degree(atom.id) <= 1
        return False

    def _should_show_hydrogen_label(self, atom: EditorAtom) -> bool:
        mode = self.hydrogens_display_var.get().strip()
        if mode == "All":
            return True
        if mode == "None":
            return False
        neighbor = self._single_neighbor_atom(atom.id)
        if mode == "Hetero":
            return self._is_hetero_atom(neighbor)
        if mode == "Hetero and terminal":
            if self._is_hetero_atom(neighbor):
                return True
            if neighbor is None:
                return False
            return self._atom_degree(neighbor.id) <= 1
        return False

    def _hydrogen_count_by_atom_id(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        mol = self.to_rdkit_mol()
        if mol is not None:
            for idx, atom in enumerate(self.atoms):
                if idx >= mol.GetNumAtoms():
                    break
                try:
                    rd_atom = mol.GetAtomWithIdx(idx)
                    rd_count = int(rd_atom.GetTotalNumHs())
                    if atom.element == "N" and atom.charge > 0:
                        # Keep protonated amines visually saturated at tetravalent N.
                        # RDKit can drop one implicit H for higher formal positive charges,
                        # but for the sketcher we want NH3^2+ rather than NH2^2+ on a
                        # terminal primary amine until the user goes beyond that pattern.
                        explicit_valence = self._atom_valence(atom.id)
                        expected = max(0, int(round(4.0 - explicit_valence)))
                        counts[atom.id] = max(rd_count, expected)
                    else:
                        counts[atom.id] = rd_count
                except Exception:
                    counts[atom.id] = 0
            return counts
        for atom in self.atoms:
            if atom.element == "H":
                counts[atom.id] = 0
                continue
            base_valence = MAX_VALENCE.get(atom.element)
            if base_valence is None:
                counts[atom.id] = 0
                continue
            effective_valence = max(0.0, base_valence + atom.charge)
            counts[atom.id] = max(0, int(round(effective_valence - self._atom_valence(atom.id))))
        return counts

    def _should_show_hydrogens_on_atom(self, atom: EditorAtom, hydrogen_count: int) -> bool:
        if hydrogen_count <= 0:
            return False
        if atom.element in {"F", "Cl", "Br", "I"}:
            return False
        mode = self.hydrogens_display_var.get().strip()
        if mode == "None":
            return False
        if mode == "All":
            return atom.element != "H"
        if mode == "Hetero":
            return self._is_hetero_atom(atom)
        if mode == "Hetero and terminal":
            return self._is_hetero_atom(atom) or self._atom_degree(atom.id) <= 1
        return False

    def _hydrogen_suffix_text(self, hydrogen_count: int) -> str:
        if hydrogen_count <= 0:
            return ""
        if hydrogen_count == 1:
            return "H"
        return f"H{hydrogen_count}"

    def _ensure_atom_fonts(self) -> None:
        if self._atom_main_font is None:
            self._atom_main_font = tkfont.Font(family="Arial", size=16, weight="bold")
        if self._atom_side_font is None:
            self._atom_side_font = tkfont.Font(family="Arial", size=13, weight="bold")

    def _atom_label_layout(
        self,
        base_label: str,
        hydrogen_count: int,
        show_hydrogens: bool,
        charge: int,
    ) -> tuple[str, str, str, float, float, float]:
        self._ensure_atom_fonts()
        assert self._atom_main_font is not None
        assert self._atom_side_font is not None
        main_text = base_label + ("H" if show_hydrogens else "")
        sub_text = str(hydrogen_count) if show_hydrogens and hydrogen_count > 1 else ""
        super_text = self._charge_suffix_text(charge)
        main_w = float(self._atom_main_font.measure(main_text))
        side_w = 0.0
        if sub_text or super_text:
            side_w = float(
                max(
                    self._atom_side_font.measure(sub_text) if sub_text else 0,
                    self._atom_side_font.measure(super_text) if super_text else 0,
                )
            )
        total_w = main_w + side_w
        return main_text, sub_text, super_text, main_w, side_w, total_w

    def _draw_atom_label(
        self,
        sx: float,
        sy: float,
        base_label: str,
        hydrogen_count: int,
        show_hydrogens: bool,
        charge: int,
        color: Any,
    ) -> None:
        main_text, sub_text, super_text, main_w, side_w, total_w = self._atom_label_layout(
            base_label,
            hydrogen_count,
            show_hydrogens,
            charge,
        )
        rx = max(12.0, total_w / 2.0 + 4.0)
        self._aa_ellipse((sx - rx, sy - 12, sx + rx, sy + 12), fill="#ffffff", outline="#ffffff", width=1)
        self._ensure_atom_fonts()
        assert self._atom_main_font is not None
        assert self._atom_side_font is not None
        left_x = sx - total_w / 2.0
        self._aa_text((left_x, sy), main_text, color, self._atom_main_font, anchor="w")
        if side_w > 0.0:
            stack_x = left_x + main_w + side_w / 2.0
            if sub_text:
                self._aa_text((stack_x, sy + 4), sub_text, color, self._atom_side_font, anchor="c")
            if super_text:
                self._aa_text((stack_x, sy - 5), super_text, color, self._atom_side_font, anchor="c")

    def _neighbor_ids(self, atom_id: int) -> list[int]:
        out: list[int] = []
        for bond in self.bonds:
            if bond.a1 == atom_id:
                out.append(bond.a2)
            elif bond.a2 == atom_id:
                out.append(bond.a1)
        return out

    def _shortest_path_excluding_bond(self, start_id: int, end_id: int, excluded_bond_id: int) -> list[int] | None:
        queue: list[list[int]] = [[start_id]]
        seen = {start_id}
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == end_id:
                return path
            for bond in self.bonds:
                if bond.id == excluded_bond_id:
                    continue
                if bond.a1 == node:
                    nxt = bond.a2
                elif bond.a2 == node:
                    nxt = bond.a1
                else:
                    continue
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append([*path, nxt])
        return None

    def _distance_to_segment(self, px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        dx = bx - ax
        dy = by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return math.dist((px, py), (ax, ay))
        t = ((px - ax) * dx + (py - ay) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.dist((px, py), (proj_x, proj_y))

    def _rotate(self, vx: float, vy: float, angle: float) -> tuple[float, float]:
        ca = math.cos(angle)
        sa = math.sin(angle)
        return vx * ca - vy * sa, vx * sa + vy * ca

    def _angle_delta(self, a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _mid_angle(self, a: float, b: float) -> float:
        return a + self._angle_delta(b, a) / 2.0

    def _normalize(self, vx: float, vy: float) -> tuple[float, float]:
        norm = math.hypot(vx, vy)
        if norm < 1e-9:
            return 1.0, 0.0
        return vx / norm, vy / norm

    def _preferred_growth_angles(self, atom_id: int) -> list[float]:
        atom = self._atom_by_id(atom_id)
        if atom is None:
            return [0.0]
        neighbor_angles: list[float] = []
        for neighbor_id in self._neighbor_ids(atom_id):
            neighbor = self._atom_by_id(neighbor_id)
            if neighbor is None:
                continue
            neighbor_angles.append(math.atan2(neighbor.y - atom.y, neighbor.x - atom.x))
        if not neighbor_angles:
            return [0.0, math.radians(120), math.radians(240)]
        if len(neighbor_angles) == 1:
            base = neighbor_angles[0]
            return [base + math.radians(120), base - math.radians(120), base + math.pi]
        if len(neighbor_angles) == 2:
            vx = sum(math.cos(ang) for ang in neighbor_angles)
            vy = sum(math.sin(ang) for ang in neighbor_angles)
            if math.hypot(vx, vy) > 1e-6:
                outward = math.atan2(-vy, -vx)
                return [outward, outward + math.radians(60), outward - math.radians(60)]
        candidates = [math.radians(60 * i) for i in range(6)]
        scored = []
        for cand in candidates:
            mindiff = min(abs(math.atan2(math.sin(cand - ang), math.cos(cand - ang))) for ang in neighbor_angles)
            scored.append((mindiff, cand))
        scored.sort(reverse=True)
        return [cand for _score, cand in scored]

    def _bond_angles(self, atom_id: int) -> list[tuple[float, float]]:
        atom = self._atom_by_id(atom_id)
        if atom is None:
            return []
        angles: list[tuple[float, float]] = []
        for bond in self.bonds:
            if bond.a1 == atom_id or bond.a2 == atom_id:
                other = self._atom_by_id(bond.a2 if bond.a1 == atom_id else bond.a1)
                if other is None:
                    continue
                ang = math.atan2(other.y - atom.y, other.x - atom.x)
                angles.append((ang, self.bond_order_value(bond.style)))
        return angles

    def _preferred_growth_angles_for_style(self, atom_id: int, style: str = "single") -> list[float]:
        bond_info = self._bond_angles(atom_id)
        if not bond_info:
            if style in {"double", "triple"}:
                return [0.0, math.pi]
            return self._preferred_growth_angles(atom_id)
        target_order = self.bond_order_value(style)
        angles = [ang for ang, _order in bond_info]
        max_existing_order = max(order for _ang, order in bond_info)
        linear = (
            style == "triple"
            or (target_order >= 2.0 and len(bond_info) == 1 and max_existing_order >= 2.0)
            or (max_existing_order >= 3.0)
        )
        if linear:
            if len(angles) == 1:
                return [angles[0] + math.pi]
            avg_x = sum(math.cos(ang) for ang in angles)
            avg_y = sum(math.sin(ang) for ang in angles)
            return [math.atan2(-avg_y, -avg_x)]
        return self._preferred_growth_angles(atom_id)

    def _assign_target_angles(
        self,
        neighbor_ids: list[int],
        target_angles: list[float],
        center: EditorAtom,
    ) -> list[tuple[int, float]]:
        if not neighbor_ids or not target_angles:
            return []
        current_angles: dict[int, float] = {}
        for neighbor_id in neighbor_ids:
            neighbor = self._atom_by_id(neighbor_id)
            if neighbor is None:
                continue
            current_angles[neighbor_id] = math.atan2(neighbor.y - center.y, neighbor.x - center.x)
        valid_neighbor_ids = [neighbor_id for neighbor_id in neighbor_ids if neighbor_id in current_angles]
        if not valid_neighbor_ids:
            return []
        n = min(len(valid_neighbor_ids), len(target_angles))
        valid_neighbor_ids = valid_neighbor_ids[:n]
        target_angles = target_angles[:n]
        best_cost = float("inf")
        best_assignment: list[tuple[int, float]] = []
        for perm in itertools.permutations(target_angles, n):
            cost = 0.0
            for neighbor_id, target_angle in zip(valid_neighbor_ids, perm):
                cost += abs(self._angle_delta(current_angles[neighbor_id], target_angle))
            if cost < best_cost:
                best_cost = cost
                best_assignment = list(zip(valid_neighbor_ids, perm))
        return best_assignment

    def _chain_continuation_leaf_id(self, center_id: int, chain_root_id: int, leaf_ids: list[int]) -> int | None:
        center = self._atom_by_id(center_id)
        chain_root = self._atom_by_id(chain_root_id)
        if center is None or chain_root is None or not leaf_ids:
            return None
        root_parent_ids = [nid for nid in self._neighbor_ids(chain_root_id) if nid != center_id]
        expected_angle: float | None = None
        if len(root_parent_ids) == 1:
            root_parent = self._atom_by_id(root_parent_ids[0])
            if root_parent is not None:
                prev_in = math.atan2(chain_root.y - root_parent.y, chain_root.x - root_parent.x)
                prev_out = math.atan2(center.y - chain_root.y, center.x - chain_root.x)
                prev_turn = self._angle_delta(prev_out, prev_in)
                if abs(prev_turn) > 1e-6:
                    base = math.atan2(chain_root.y - center.y, chain_root.x - center.x)
                    candidates = [base + math.radians(120.0), base - math.radians(120.0)]
                    opposite_candidates: list[tuple[float, float]] = []
                    for cand in candidates:
                        turn = self._angle_delta(cand, prev_out)
                        if prev_turn > 0 and turn < 0:
                            opposite_candidates.append((abs(turn), cand))
                        elif prev_turn < 0 and turn > 0:
                            opposite_candidates.append((abs(turn), cand))
                    if opposite_candidates:
                        opposite_candidates.sort(reverse=True)
                        expected_angle = opposite_candidates[0][1]
        best_leaf_id: int | None = None
        best_score: float | None = None
        for leaf_id in leaf_ids:
            leaf = self._atom_by_id(leaf_id)
            if leaf is None:
                continue
            leaf_angle = math.atan2(leaf.y - center.y, leaf.x - center.x)
            if expected_angle is not None:
                score = abs(self._angle_delta(leaf_angle, expected_angle))
            else:
                axis_angle = math.atan2(chain_root.y - center.y, chain_root.x - center.x) + math.pi
                score = abs(self._angle_delta(leaf_angle, axis_angle))
            if best_score is None or score < best_score:
                best_score = score
                best_leaf_id = leaf_id
        return best_leaf_id

    def _relax_primary_neighbors_after_growth(self, center_id: int, new_atom_id: int, previous_neighbor_ids: list[int] | None = None) -> None:
        center = self._atom_by_id(center_id)
        new_atom = self._atom_by_id(new_atom_id)
        if center is None or new_atom is None:
            return
        neighbor_ids = self._neighbor_ids(center_id)
        if len(neighbor_ids) < 2:
            return
        leaf_neighbor_ids = [nid for nid in neighbor_ids if self._atom_degree(nid) == 1]
        if new_atom_id not in leaf_neighbor_ids:
            return
        non_leaf_neighbor_ids = [nid for nid in neighbor_ids if nid not in leaf_neighbor_ids]
        if len(neighbor_ids) == 2 and len(leaf_neighbor_ids) == 2:
            return

        angle_by_neighbor: dict[int, float] = {}
        for neighbor_id in neighbor_ids:
            neighbor = self._atom_by_id(neighbor_id)
            if neighbor is None:
                continue
            angle_by_neighbor[neighbor_id] = math.atan2(neighbor.y - center.y, neighbor.x - center.x)

        target_angles: list[float] = []

        if previous_neighbor_ids is not None and len(previous_neighbor_ids) == 3 and len(neighbor_ids) == 4:
            previous_leaf_ids = [nid for nid in previous_neighbor_ids if self._atom_degree(nid) == 1 and nid in angle_by_neighbor]
            previous_non_leaf_ids = [nid for nid in previous_neighbor_ids if nid not in previous_leaf_ids and nid in angle_by_neighbor]
            if len(previous_non_leaf_ids) == 1 and len(previous_leaf_ids) == 2 and new_atom_id in angle_by_neighbor:
                chain_root_id = previous_non_leaf_ids[0]
                continuation_primary_id = self._chain_continuation_leaf_id(center_id, chain_root_id, previous_leaf_ids)
                if continuation_primary_id is None:
                    continuation_primary_id = previous_leaf_ids[0]
                branch_primary_id = next(nid for nid in previous_leaf_ids if nid != continuation_primary_id)
                chain_root = self._atom_by_id(chain_root_id)
                continuation_atom = self._atom_by_id(continuation_primary_id)
                if chain_root is None or continuation_atom is None:
                    return
                axis_angle = math.atan2(
                    continuation_atom.y - chain_root.y,
                    continuation_atom.x - chain_root.x,
                )
                branch_current_angle = angle_by_neighbor[branch_primary_id]
                branch_side_sign = 1.0 if self._angle_delta(branch_current_angle, axis_angle) >= 0 else -1.0
                perpendicular = axis_angle + branch_side_sign * math.radians(90.0)
                candidate_sets = [[
                    perpendicular - branch_side_sign * math.radians(30.0),
                    perpendicular + branch_side_sign * math.radians(30.0),
                ]]
                best_assignments: list[tuple[int, float]] = []
                best_cost = float("inf")
                for target_set in candidate_sets:
                    assignments = self._assign_target_angles([branch_primary_id, new_atom_id], target_set, center)
                    branch_target = next((angle for nid, angle in assignments if nid == branch_primary_id), None)
                    if branch_target is None:
                        continue
                    cost = abs(self._angle_delta(branch_current_angle, branch_target))
                    if cost < best_cost:
                        best_cost = cost
                        best_assignments = assignments
                for neighbor_id, target_angle in best_assignments:
                    neighbor = self._atom_by_id(neighbor_id)
                    if neighbor is None:
                        continue
                    neighbor.x = center.x + STANDARD_BOND_LENGTH * math.cos(target_angle)
                    neighbor.y = center.y + STANDARD_BOND_LENGTH * math.sin(target_angle)
                return
            if len(previous_non_leaf_ids) == 2 and len(previous_leaf_ids) == 1 and new_atom_id in angle_by_neighbor:
                chain_a = self._atom_by_id(previous_non_leaf_ids[0])
                chain_b = self._atom_by_id(previous_non_leaf_ids[1])
                branch_atom = self._atom_by_id(previous_leaf_ids[0])
                if chain_a is None or chain_b is None or branch_atom is None:
                    return
                branch_primary_id = previous_leaf_ids[0]
                axis_angle = math.atan2(chain_b.y - chain_a.y, chain_b.x - chain_a.x)
                branch_current_angle = angle_by_neighbor[branch_primary_id]
                branch_side_sign = 1.0 if self._angle_delta(branch_current_angle, axis_angle) >= 0 else -1.0
                perpendicular = axis_angle + branch_side_sign * math.radians(90.0)
                target_set = [
                    perpendicular - branch_side_sign * math.radians(30.0),
                    perpendicular + branch_side_sign * math.radians(30.0),
                ]
                assignments = self._assign_target_angles([branch_primary_id, new_atom_id], target_set, center)
                for neighbor_id, target_angle in assignments:
                    neighbor = self._atom_by_id(neighbor_id)
                    if neighbor is None:
                        continue
                    neighbor.x = center.x + STANDARD_BOND_LENGTH * math.cos(target_angle)
                    neighbor.y = center.y + STANDARD_BOND_LENGTH * math.sin(target_angle)
                return

        if len(neighbor_ids) == 4 and len(leaf_neighbor_ids) >= 2 and len(angle_by_neighbor) == 4:
            best_pair: tuple[int, int] | None = None
            best_sep = -1.0
            neighbor_list = list(angle_by_neighbor.keys())
            for i, first_id in enumerate(neighbor_list):
                for second_id in neighbor_list[i + 1:]:
                    sep = abs(self._angle_delta(angle_by_neighbor[first_id], angle_by_neighbor[second_id]))
                    sep = min(sep, 2.0 * math.pi - sep)
                    if sep > best_sep:
                        best_sep = sep
                        best_pair = (first_id, second_id)
            if best_pair is not None:
                movable_leaf_ids = [nid for nid in leaf_neighbor_ids if nid not in best_pair]
                if len(movable_leaf_ids) == 2:
                    fixed_a = angle_by_neighbor[best_pair[0]]
                    fixed_b = angle_by_neighbor[best_pair[1]]
                    if abs(best_sep - math.pi) < math.radians(25.0):
                        base = fixed_a + math.pi / 2.0
                    else:
                        base = self._mid_angle(fixed_a, fixed_b)
                    spread = math.radians(30.0)
                    assignments = self._assign_target_angles(
                        movable_leaf_ids,
                        [base - spread, base + spread],
                        center,
                    )
                    for neighbor_id, target_angle in assignments:
                        neighbor = self._atom_by_id(neighbor_id)
                        if neighbor is None:
                            continue
                        neighbor.x = center.x + STANDARD_BOND_LENGTH * math.cos(target_angle)
                        neighbor.y = center.y + STANDARD_BOND_LENGTH * math.sin(target_angle)
                    return

    def _chain_continuation_angle(self, atom_id: int) -> float | None:
        atom = self._atom_by_id(atom_id)
        if atom is None:
            return None
        neighbor_ids = self._neighbor_ids(atom_id)
        if len(neighbor_ids) != 1:
            return None
        parent_id = neighbor_ids[0]
        parent = self._atom_by_id(parent_id)
        if parent is None:
            return None
        grandparent_ids = [nid for nid in self._neighbor_ids(parent_id) if nid != atom_id]
        if len(grandparent_ids) != 1:
            return None
        grandparent = self._atom_by_id(grandparent_ids[0])
        if grandparent is None:
            return None

        prev_in = math.atan2(parent.y - grandparent.y, parent.x - grandparent.x)
        prev_out = math.atan2(atom.y - parent.y, atom.x - parent.x)
        prev_turn = math.atan2(math.sin(prev_out - prev_in), math.cos(prev_out - prev_in))
        if abs(prev_turn) < 1e-6:
            return None

        base = math.atan2(parent.y - atom.y, parent.x - atom.x)
        incoming = math.atan2(atom.y - parent.y, atom.x - parent.x)
        candidates = [base + math.radians(120), base - math.radians(120)]

        opposite_candidates = []
        for cand in candidates:
            turn = math.atan2(math.sin(cand - incoming), math.cos(cand - incoming))
            if prev_turn > 0 and turn < 0:
                opposite_candidates.append((abs(turn), cand))
            elif prev_turn < 0 and turn > 0:
                opposite_candidates.append((abs(turn), cand))
        if opposite_candidates:
            opposite_candidates.sort(reverse=True)
            return opposite_candidates[0][1]
        return None

    def _snap_angle_from_atom(self, atom_id: int, wx: float, wy: float) -> tuple[float, float]:
        atom = self._atom_by_id(atom_id)
        if atom is None:
            return wx, wy
        pref_angles = self._preferred_growth_angles(atom_id)
        drag_angle = math.atan2(wy - atom.y, wx - atom.x)
        best = min(
            pref_angles,
            key=lambda ang: abs(math.atan2(math.sin(drag_angle - ang), math.cos(drag_angle - ang))),
        )
        return atom.x + STANDARD_BOND_LENGTH * math.cos(best), atom.y + STANDARD_BOND_LENGTH * math.sin(best)

    def _default_growth_position(self, atom_id: int, style: str = "single") -> tuple[float, float]:
        atom = self._atom_by_id(atom_id)
        if atom is None:
            return 0.0, 0.0
        angle = self._chain_continuation_angle(atom_id) if style == "single" else None
        if angle is None:
            angle = self._preferred_growth_angles_for_style(atom_id, style)[0]
        return atom.x + STANDARD_BOND_LENGTH * math.cos(angle), atom.y + STANDARD_BOND_LENGTH * math.sin(angle)

    def _preview_attachment_angle_text(self) -> str | None:
        start_atom_id = self.pending_draw_start_atom_id
        if start_atom_id is None or self.preview_world_pos is None:
            return None
        start_atom = self._atom_by_id(start_atom_id)
        if start_atom is None:
            return None
        preview_angle = math.atan2(
            self.preview_world_pos[1] - start_atom.y,
            self.preview_world_pos[0] - start_atom.x,
        )
        best_delta_deg: float | None = None
        for neighbor_id in self._neighbor_ids(start_atom_id):
            neighbor = self._atom_by_id(neighbor_id)
            if neighbor is None:
                continue
            neighbor_angle = math.atan2(neighbor.y - start_atom.y, neighbor.x - start_atom.x)
            delta_deg = abs(math.degrees(self._angle_delta(preview_angle, neighbor_angle)))
            if best_delta_deg is None or delta_deg < best_delta_deg:
                best_delta_deg = delta_deg
        if best_delta_deg is None:
            return None
        return f"{int(round(best_delta_deg))}°"

    def _chain_axis_angle_text(self) -> str | None:
        if self.pending_draw_start_pos is None or self.mouse_screen_pos is None:
            return None
        x1, y1 = self.world_to_screen(*self.pending_draw_start_pos)
        x2, y2 = self.mouse_screen_pos
        dx = x2 - x1
        dy = y1 - y2
        if math.hypot(dx, dy) < 1e-6:
            return None
        angle_deg = math.degrees(math.atan2(dy, dx))
        carbon_count = max(2, len(self.preview_world_path) + 1)
        return f"{carbon_count}C , {int(round(angle_deg))}°"

    def _free_bond_drag_position(self, atom_id: int, wx: float, wy: float) -> tuple[float, float]:
        atom = self._atom_by_id(atom_id)
        if atom is None:
            return wx, wy
        dx = wx - atom.x
        dy = wy - atom.y
        ndx, ndy = self._normalize(dx, dy)
        return atom.x + STANDARD_BOND_LENGTH * ndx, atom.y + STANDARD_BOND_LENGTH * ndy

    def _chain_path_from_drag(self, atom_id: int | None, wx: float, wy: float, start_pos: tuple[float, float] | None = None) -> list[tuple[float, float]]:
        atom = self._atom_by_id(atom_id) if atom_id is not None else None
        if atom is not None:
            start_x, start_y = atom.x, atom.y
        elif start_pos is not None:
            start_x, start_y = start_pos
        else:
            return []
        dx = wx - start_x
        dy = wy - start_y
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            return [(start_x + STANDARD_BOND_LENGTH, start_y)]
        axis_angle = math.atan2(dy, dx)
        if self.chain_drag_side_sign is None:
            if atom is not None:
                default_point = self._default_growth_position(atom_id)
                base_angle = math.atan2(default_point[1] - start_y, default_point[0] - start_x)
                delta = math.atan2(math.sin(axis_angle - base_angle), math.cos(axis_angle - base_angle))
                self.chain_drag_side_sign = 1.0 if delta >= 0 else -1.0
            else:
                self.chain_drag_side_sign = 1.0 if dy >= 0 else -1.0
        sign = self.chain_drag_side_sign
        projection_step = STANDARD_BOND_LENGTH * math.cos(math.radians(30.0))
        second_atom_threshold = 2.0 * projection_step
        if distance < second_atom_threshold:
            return [(wx, wy)]
        segments = max(2, int(distance // projection_step))
        first_angle = axis_angle + sign * math.radians(30.0)
        second_angle = axis_angle - sign * math.radians(30.0)
        points: list[tuple[float, float]] = []
        cx, cy = start_x, start_y
        for i in range(segments):
            angle = first_angle if i % 2 == 0 else second_angle
            cx += STANDARD_BOND_LENGTH * math.cos(angle)
            cy += STANDARD_BOND_LENGTH * math.sin(angle)
            points.append((cx, cy))
        return points

    def _regular_polygon_points_from_bottom(self, anchor_x: float, anchor_y: float, size: int) -> list[tuple[float, float]]:
        radius = STANDARD_BOND_LENGTH / (2.0 * math.sin(math.pi / size))
        center_x = anchor_x
        center_y = anchor_y + radius
        start_angle = -math.pi / 2
        return [
            (
                center_x + radius * math.cos(start_angle + 2 * math.pi * i / size),
                center_y + radius * math.sin(start_angle + 2 * math.pi * i / size),
            )
            for i in range(size)
        ]

    def _preview_fusion_bond_from_atom(self, atom_id: int, wx: float, wy: float) -> int | None:
        atom = self._atom_by_id(atom_id)
        if atom is None:
            return None
        neighbor_ids = self._neighbor_ids(atom_id)
        if not neighbor_ids:
            return None
        drag_angle = math.atan2(wy - atom.y, wx - atom.x)
        best_neighbor = min(
            neighbor_ids,
            key=lambda nid: abs(
                math.atan2(
                    math.sin(drag_angle - math.atan2((self._atom_by_id(nid).y - atom.y), (self._atom_by_id(nid).x - atom.x))),
                    math.cos(drag_angle - math.atan2((self._atom_by_id(nid).y - atom.y), (self._atom_by_id(nid).x - atom.x))),
                )
            ),
        )
        for bond in self.bonds:
            if {bond.a1, bond.a2} == {atom_id, best_neighbor}:
                return bond.id
        return None

    def _ring_preview_for_cursor(self, wx: float, wy: float) -> tuple[list[tuple[float, float]], list[tuple[int, int]], dict[str, Any] | None]:
        atom_id, bond_id = self._pick_hit_targets(wx, wy)
        if bond_id is not None:
            points, edges = self._fused_ring_geometry(bond_id, (wx, wy))
            return points, edges, {"type": "bond", "bond_id": bond_id, "click_pos": (wx, wy)}
        if atom_id is not None:
            atom = self._atom_by_id(atom_id)
            if atom is not None:
                _key, _icon, size, _aromatic = self._ring_template_def()
                points = self._regular_polygon_points_from_bottom(atom.x, atom.y, size)
                desired = math.atan2(wy - atom.y, wx - atom.x)
                delta = desired - math.pi / 2
                rotated = [points[0]]
                for px, py in points[1:]:
                    rx, ry = self._rotate(px - atom.x, py - atom.y, delta)
                    rotated.append((atom.x + rx, atom.y + ry))
                edges = [(i, (i + 1) % size) for i in range(size)]
                return rotated, edges, {"type": "atom", "atom_id": atom_id, "cursor_pos": (wx, wy)}
        _key, _icon, size, _aromatic = self._ring_template_def()
        points = self._regular_polygon_points_from_bottom(wx, wy, size)
        edges = [(i, (i + 1) % size) for i in range(size)]
        return points, edges, {"type": "free", "anchor": (wx, wy)}

    def _find_atom_hit(self, wx: float, wy: float) -> tuple[int | None, float | None]:
        best_atom = None
        best_dist_px = 16.0
        tx, ty = self.world_to_screen(wx, wy)
        for atom in self.atoms:
            sx, sy = self.world_to_screen(atom.x, atom.y)
            dist_px = math.dist((tx, ty), (sx, sy))
            if dist_px <= best_dist_px:
                best_dist_px = dist_px
                best_atom = atom.id
        return best_atom, (best_dist_px if best_atom is not None else None)

    def _find_bond_hit(self, wx: float, wy: float) -> tuple[int | None, float | None]:
        best_bond = None
        best_dist_px = 24.0
        tx, ty = self.world_to_screen(wx, wy)
        for bond in self.bonds:
            a1 = self._atom_by_id(bond.a1)
            a2 = self._atom_by_id(bond.a2)
            if not a1 or not a2:
                continue
            sx1, sy1 = self.world_to_screen(a1.x, a1.y)
            sx2, sy2 = self.world_to_screen(a2.x, a2.y)
            dist_px = self._distance_to_segment(tx, ty, sx1, sy1, sx2, sy2)
            if dist_px <= best_dist_px:
                best_dist_px = dist_px
                best_bond = bond.id
        return best_bond, (best_dist_px if best_bond is not None else None)

    def _pick_hit_targets(self, wx: float, wy: float) -> tuple[int | None, int | None]:
        atom_id, atom_dist = self._find_atom_hit(wx, wy)
        bond_id, bond_dist = self._find_bond_hit(wx, wy)
        # Treat the inner core of the atom as atom-only. Outside that core,
        # let a bond hit win so bond clicks still work near atom endpoints.
        atom_core_px = 10.0
        if atom_id is not None and atom_dist is not None and atom_dist <= atom_core_px:
            return atom_id, None
        if bond_id is not None:
            return None, bond_id
        return atom_id, None

    def find_atom_at(self, wx: float, wy: float) -> int | None:
        return self._find_atom_hit(wx, wy)[0]

    def find_bond_at(self, wx: float, wy: float) -> int | None:
        return self._find_bond_hit(wx, wy)[0]

    def clear_selection(self) -> None:
        self.selected_atom_id = None
        self.selected_bond_id = None
        self.selected_atom_ids = set()
        self.selected_bond_ids = set()
        self.pending_draw_start_atom_id = None
        self.pending_draw_start_pos = None
        self.pending_draw_mode = None
        self.preview_world_pos = None
        self.preview_world_path = []
        self.pending_ring_active = False
        self.preview_ring_points = []
        self.preview_ring_bonds = []
        self.preview_ring_commit = None
        self.selection_rect_start = None
        self.selection_rect_end = None
        self.erase_rect_start = None
        self.erase_rect_end = None
        self.erase_hit_atom_id = None
        self.erase_hit_bond_id = None
        self.preview_rect_atom_ids = set()
        self.preview_rect_bond_ids = set()
        self.render_all()

    def select_all_canvas(self, _event: tk.Event[tk.Misc] | None = None) -> str:
        self.selected_atom_id = None
        self.selected_bond_id = None
        self.selected_atom_ids = {atom.id for atom in self.atoms}
        self.selected_bond_ids = {bond.id for bond in self.bonds}
        self.selection_rect_start = None
        self.selection_rect_end = None
        self.preview_rect_atom_ids = set()
        self.preview_rect_bond_ids = set()
        self.render_all()
        return "break"

    def _flip_target_atom_ids(self) -> set[int]:
        atom_ids = self._selected_fragment_atom_ids()
        if atom_ids:
            return atom_ids
        return {atom.id for atom in self.atoms}

    def _flip_target_bonds(self, atom_ids: set[int]) -> list[EditorBond]:
        return [bond for bond in self.bonds if bond.a1 in atom_ids or bond.a2 in atom_ids]

    def _flip_selection(self, horizontal: bool) -> None:
        atom_ids = self._flip_target_atom_ids()
        if not atom_ids:
            self.set_status("Nothing to flip")
            return
        atoms = [self._atom_by_id(atom_id) for atom_id in atom_ids]
        atoms = [atom for atom in atoms if atom is not None]
        if not atoms:
            self.set_status("Nothing to flip")
            return
        if horizontal:
            axis = (min(atom.x for atom in atoms) + max(atom.x for atom in atoms)) / 2.0
            for atom in atoms:
                atom.x = (2.0 * axis) - atom.x
        else:
            axis = (min(atom.y for atom in atoms) + max(atom.y for atom in atoms)) / 2.0
            for atom in atoms:
                atom.y = (2.0 * axis) - atom.y
        for bond in self._flip_target_bonds(atom_ids):
            if bond.style == "wedge":
                bond.style = "hashed"
            elif bond.style == "hashed":
                bond.style = "wedge"
        self._push_history("Horizontal flip" if horizontal else "Vertical flip")
        self.render_all()

    def flip_selection_horizontal(self) -> None:
        self._flip_selection(horizontal=True)

    def flip_selection_vertical(self) -> None:
        self._flip_selection(horizontal=False)

    def erase_selection_action(self) -> None:
        if not (self.selected_atom_ids or self.selected_bond_ids):
            self.set_status("Nothing selected")
            return
        self.delete_selected()

    def toggle_show_stereocenters(self) -> None:
        self.show_stereocenters = not self.show_stereocenters
        self._refresh_aux_action_buttons()
        self.set_status("Show stereocenters on" if self.show_stereocenters else "Show stereocenters off")
        self.render_all()

    def _update_preview_rect_selection(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        sx0, sx1 = sorted((start[0], end[0]))
        sy0, sy1 = sorted((start[1], end[1]))
        self.preview_rect_atom_ids = set()
        for atom in self.atoms:
            ax, ay = self.world_to_screen(atom.x, atom.y)
            if sx0 <= ax <= sx1 and sy0 <= ay <= sy1:
                self.preview_rect_atom_ids.add(atom.id)
        self.preview_rect_bond_ids = {
            bond.id
            for bond in self.bonds
            if bond.a1 in self.preview_rect_atom_ids and bond.a2 in self.preview_rect_atom_ids
        }

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def _snapshot(self) -> dict[str, Any]:
        return {
            "atoms": [asdict(atom) for atom in self.atoms],
            "bonds": [asdict(bond) for bond in self.bonds],
            "next_atom_id": self.next_atom_id,
            "next_bond_id": self.next_bond_id,
        }

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.atoms = [
            EditorAtom(
                id=atom["id"],
                element=atom["element"],
                x=atom["x"],
                y=atom["y"],
                charge=atom.get("charge", 0),
                aromatic=atom.get("aromatic", False),
                label=atom.get("label", ""),
            )
            for atom in snapshot["atoms"]
        ]
        self.bonds = [EditorBond(**bond) for bond in snapshot["bonds"]]
        self._invalidate_indices()
        self.next_atom_id = snapshot["next_atom_id"]
        self.next_bond_id = snapshot["next_bond_id"]
        self.selected_atom_id = None
        self.selected_bond_id = None
        self.selected_atom_ids = set()
        self.selected_bond_ids = set()
        self.pending_draw_start_atom_id = None
        self.pending_draw_start_pos = None
        self.pending_draw_mode = None
        self.preview_world_pos = None
        self.preview_world_path = []
        self.preview_fuse_atom_ids = set()
        self.preview_fuse_bond_ids = set()
        self.drag_atom_id = None
        self.drag_group_ids = set()
        self.drag_group_origin = {}
        self.drag_preview_delta_world = (0.0, 0.0)

    def _push_history(self, label: str) -> None:
        snapshot = self._snapshot()
        if self.history_index >= 0 and self.history[self.history_index] == snapshot:
            return
        self.history = self.history[: self.history_index + 1]
        self.history.append(copy.deepcopy(snapshot))
        self.history_index += 1
        self.set_status(label)

    def undo(self) -> None:
        if self.history_index <= 0:
            self.set_status("Nothing to undo")
            return
        self.history_index -= 1
        self._restore_snapshot(self.history[self.history_index])
        self.set_status("Undo")
        self.render_all()

    def redo(self) -> None:
        if self.history_index >= len(self.history) - 1:
            self.set_status("Nothing to redo")
            return
        self.history_index += 1
        self._restore_snapshot(self.history[self.history_index])
        self.set_status("Redo")
        self.render_all()

    # ------------------------------------------------------------------
    # Structure editing
    # ------------------------------------------------------------------
    def add_atom(
        self,
        x: float,
        y: float,
        *,
        element: str | None = None,
        charge: int | None = None,
        aromatic: bool = False,
        label: str = "",
    ) -> int:
        requested_element = (element or self.current_element()).strip() or "C"
        actual_element = requested_element
        actual_label = label
        if requested_element == "R":
            actual_element = "*"
            actual_label = label or self._next_r_group_label()
        atom = EditorAtom(
            id=self.next_atom_id,
            element=actual_element,
            x=x,
            y=y,
            charge=0 if charge is None else charge,
            aromatic=aromatic,
            label=actual_label,
        )
        self.atoms.append(atom)
        self._atom_index[atom.id] = atom
        self.next_atom_id += 1
        return atom.id

    def _reuse_or_add_atom(
        self,
        x: float,
        y: float,
        *,
        element: str | None = None,
        charge: int | None = None,
        aromatic: bool = False,
        label: str = "",
        tolerance: float = 0.28,
    ) -> int:
        for atom in self.atoms:
            if math.dist((atom.x, atom.y), (x, y)) <= tolerance:
                return atom.id
        return self.add_atom(x, y, element=element, charge=charge, aromatic=aromatic, label=label)

    def add_or_update_bond(self, a1: int, a2: int, *, style: str | None = None) -> int:
        if a1 == a2:
            return -1
        style = style or self.selected_bond_style
        for bond in self.bonds:
            if {bond.a1, bond.a2} == {a1, a2}:
                bond.style = style
                return bond.id
        bond = EditorBond(id=self.next_bond_id, a1=a1, a2=a2, style=style)
        self.bonds.append(bond)
        self._bond_index[bond.id] = bond
        self.next_bond_id += 1
        return bond.id

    def remove_atom(self, atom_id: int) -> None:
        self.atoms = [atom for atom in self.atoms if atom.id != atom_id]
        self.bonds = [bond for bond in self.bonds if bond.a1 != atom_id and bond.a2 != atom_id]
        self._invalidate_indices()
        self._renumber_r_group_labels()

    def remove_bond(self, bond_id: int) -> None:
        self.bonds = [bond for bond in self.bonds if bond.id != bond_id]
        self._invalidate_indices()

    def _bond_exists_between(self, a1: int, a2: int) -> bool:
        return any({bond.a1, bond.a2} == {a1, a2} for bond in self.bonds)

    def _find_existing_atom_near(
        self,
        x: float,
        y: float,
        *,
        tolerance: float = 0.28,
        exclude_ids: set[int] | None = None,
    ) -> int | None:
        excluded = exclude_ids or set()
        best_atom_id: int | None = None
        best_dist = tolerance
        for atom in self.atoms:
            if atom.id in excluded:
                continue
            dist = math.dist((atom.x, atom.y), (x, y))
            if dist <= best_dist:
                best_dist = dist
                best_atom_id = atom.id
        return best_atom_id

    def _dummy_atom_label_from_rdkit(self, atom: Chem.Atom) -> str:
        if atom.GetSymbol() != "*":
            return ""
        atom_map = int(atom.GetAtomMapNum() or 0)
        isotope = int(atom.GetIsotope() or 0)
        if atom_map > 0:
            return f"R{atom_map}"
        if isotope > 0:
            return f"R{isotope}"
        return ""

    def _atom_drag_world_pos(self, atom: EditorAtom) -> tuple[float, float]:
        if self.drag_atom_id is not None and atom.id in self.drag_group_ids:
            dx, dy = self.drag_preview_delta_world
            return atom.x + dx, atom.y + dy
        return atom.x, atom.y

    def _merge_atom_into(self, source_id: int, target_id: int) -> bool:
        if source_id == target_id:
            return False
        source = self._atom_by_id(source_id)
        target = self._atom_by_id(target_id)
        if source is None or target is None:
            return False
        changed = False
        bonds_snapshot = list(self.bonds)
        for bond in bonds_snapshot:
            if bond.a1 != source_id and bond.a2 != source_id:
                continue
            other_id = bond.a2 if bond.a1 == source_id else bond.a1
            if other_id == target_id:
                self.remove_bond(bond.id)
                changed = True
                continue
            if not self._bond_exists_between(target_id, other_id):
                self.add_or_update_bond(target_id, other_id, style=bond.style)
                changed = True
            self.remove_bond(bond.id)
            changed = True
        self.remove_atom(source_id)
        changed = True
        if self.selected_atom_id == source_id:
            self.selected_atom_id = target_id
        if source_id in self.selected_atom_ids:
            self.selected_atom_ids.discard(source_id)
            self.selected_atom_ids.add(target_id)
        return changed

    def _fuse_moved_selection(self, moved_atom_ids: set[int]) -> bool:
        if not moved_atom_ids:
            return False
        changed = False
        bond_tolerance = 0.24
        atom_tolerance = 0.22
        moved_bonds = [bond for bond in self.bonds if bond.a1 in moved_atom_ids and bond.a2 in moved_atom_ids]
        other_bonds = [bond for bond in self.bonds if not (bond.a1 in moved_atom_ids and bond.a2 in moved_atom_ids)]

        for moved_bond in list(moved_bonds):
            a1 = self._atom_by_id(moved_bond.a1)
            a2 = self._atom_by_id(moved_bond.a2)
            if a1 is None or a2 is None:
                continue
            ma1 = self._atom_drag_world_pos(a1)
            ma2 = self._atom_drag_world_pos(a2)
            for other_bond in list(other_bonds):
                b1 = self._atom_by_id(other_bond.a1)
                b2 = self._atom_by_id(other_bond.a2)
                if b1 is None or b2 is None:
                    continue
                same = math.dist(ma1, (b1.x, b1.y)) <= bond_tolerance and math.dist(ma2, (b2.x, b2.y)) <= bond_tolerance
                swapped = math.dist(ma1, (b2.x, b2.y)) <= bond_tolerance and math.dist(ma2, (b1.x, b1.y)) <= bond_tolerance
                if not same and not swapped:
                    continue
                source_a = other_bond.a1 if same else other_bond.a2
                source_b = other_bond.a2 if same else other_bond.a1
                changed = self._merge_atom_into(source_a, moved_bond.a1) or changed
                changed = self._merge_atom_into(source_b, moved_bond.a2) or changed
                break

        moved_atoms = [atom for atom in self.atoms if atom.id in moved_atom_ids]
        other_atoms = [atom for atom in self.atoms if atom.id not in moved_atom_ids]
        for moved_atom in moved_atoms:
            moved_pos = self._atom_drag_world_pos(moved_atom)
            for other_atom in other_atoms:
                if math.dist(moved_pos, (other_atom.x, other_atom.y)) <= atom_tolerance:
                    changed = self._merge_atom_into(other_atom.id, moved_atom.id) or changed
                    break
        return changed

    def _compute_fuse_preview(self, moved_atom_ids: set[int]) -> tuple[set[int], set[int]]:
        preview_atoms: set[int] = set()
        preview_bonds: set[int] = set()
        if not moved_atom_ids:
            return preview_atoms, preview_bonds
        bond_tolerance = 0.24
        atom_tolerance = 0.22
        moved_bonds = [bond for bond in self.bonds if bond.a1 in moved_atom_ids and bond.a2 in moved_atom_ids]
        other_bonds = [bond for bond in self.bonds if not (bond.a1 in moved_atom_ids and bond.a2 in moved_atom_ids)]
        for moved_bond in moved_bonds:
            a1 = self._atom_by_id(moved_bond.a1)
            a2 = self._atom_by_id(moved_bond.a2)
            if a1 is None or a2 is None:
                continue
            ma1 = self._atom_drag_world_pos(a1)
            ma2 = self._atom_drag_world_pos(a2)
            for other_bond in other_bonds:
                b1 = self._atom_by_id(other_bond.a1)
                b2 = self._atom_by_id(other_bond.a2)
                if b1 is None or b2 is None:
                    continue
                same = math.dist(ma1, (b1.x, b1.y)) <= bond_tolerance and math.dist(ma2, (b2.x, b2.y)) <= bond_tolerance
                swapped = math.dist(ma1, (b2.x, b2.y)) <= bond_tolerance and math.dist(ma2, (b1.x, b1.y)) <= bond_tolerance
                if same or swapped:
                    preview_bonds.add(moved_bond.id)
                    preview_bonds.add(other_bond.id)
                    preview_atoms.update({moved_bond.a1, moved_bond.a2, other_bond.a1, other_bond.a2})
        for moved_atom in self.atoms:
            if moved_atom.id not in moved_atom_ids:
                continue
            moved_pos = self._atom_drag_world_pos(moved_atom)
            for other_atom in self.atoms:
                if other_atom.id in moved_atom_ids:
                    continue
                if math.dist(moved_pos, (other_atom.x, other_atom.y)) <= atom_tolerance:
                    preview_atoms.add(moved_atom.id)
                    preview_atoms.add(other_atom.id)
        return preview_atoms, preview_bonds

    def delete_selected(self) -> None:
        if self.selected_atom_ids or self.selected_bond_ids:
            atom_ids = set(self.selected_atom_ids)
            bond_ids = set(self.selected_bond_ids)
            self.atoms = [atom for atom in self.atoms if atom.id not in atom_ids]
            self.bonds = [
                bond for bond in self.bonds
                if bond.id not in bond_ids and bond.a1 not in atom_ids and bond.a2 not in atom_ids
            ]
            self._renumber_r_group_labels()
            self.selected_atom_id = None
            self.selected_bond_id = None
            self.selected_atom_ids = set()
            self.selected_bond_ids = set()
            self._push_history("Selection deleted")
            self.render_all()
            return
        if self.selected_atom_id is not None:
            self.remove_atom(self.selected_atom_id)
            self.selected_atom_id = None
            self._push_history("Atom deleted")
            self.render_all()
            return
        if self.selected_bond_id is not None:
            self.remove_bond(self.selected_bond_id)
            self.selected_bond_id = None
            self._push_history("Bond deleted")
            self.render_all()
            return
        self.set_status("Nothing selected")

    def clear_all(self) -> None:
        self.atoms.clear()
        self.bonds.clear()
        self.next_atom_id = 1
        self.next_bond_id = 1
        self.selected_atom_id = None
        self.selected_bond_id = None
        self.pending_draw_start_atom_id = None
        self.preview_world_pos = None
        self._push_history("Structure cleared")
        self.render_all()

    def _ring_template_def(self, key: str | None = None) -> tuple[str, str, int, bool]:
        key = key or self.selected_ring_template
        for ring in RING_TEMPLATES:
            if ring[0] == key:
                return ring
        return RING_TEMPLATES[0]

    def _ring_bond_styles(self, size: int, aromatic: bool) -> list[str]:
        if not aromatic:
            return ["single"] * size
        if size == 6:
            return ["double" if i % 2 == 0 else "single" for i in range(size)]
        if size == 5:
            return ["double", "single", "double", "single", "single"]
        return ["single"] * size

    def _ring_bond_styles_for_fusion(self, size: int, aromatic: bool, shared_bond_style: str) -> list[str]:
        styles = self._ring_bond_styles(size, aromatic)
        if not aromatic:
            return styles
        if shared_bond_style == "double" and styles:
            if size == 6:
                return ["single" if i % 2 == 0 else "double" for i in range(size)]
            if size == 5:
                return ["single", "double", "single", "double", "single"]
        return styles

    def _rebalance_aromatic_component(self, seed_atom_ids: set[int]) -> bool:
        aromatic_atom_ids = {atom.id for atom in self.atoms if atom.aromatic}
        component_atom_ids = set(seed_atom_ids) & aromatic_atom_ids
        if not component_atom_ids:
            return False

        changed = False
        pending = list(component_atom_ids)
        while pending:
            atom_id = pending.pop()
            for bond in self.bonds:
                if atom_id not in {bond.a1, bond.a2}:
                    continue
                other_id = bond.a2 if bond.a1 == atom_id else bond.a1
                if other_id in component_atom_ids or other_id not in aromatic_atom_ids:
                    continue
                component_atom_ids.add(other_id)
                pending.append(other_id)

        aromatic_bonds = [
            bond for bond in self.bonds
            if bond.a1 in component_atom_ids and bond.a2 in component_atom_ids and bond.style not in {"wedge", "hashed", "triple"}
        ]
        if not aromatic_bonds:
            return False

        rw = Chem.RWMol()
        atom_map: dict[int, int] = {}
        for atom_id in sorted(component_atom_ids):
            atom = self._atom_by_id(atom_id)
            if atom is None:
                continue
            rd_atom = Chem.Atom(atom.element)
            rd_atom.SetFormalCharge(atom.charge)
            rd_atom.SetIsAromatic(True)
            atom_map[atom_id] = rw.AddAtom(rd_atom)

        for bond in aromatic_bonds:
            a1 = atom_map.get(bond.a1)
            a2 = atom_map.get(bond.a2)
            if a1 is None or a2 is None:
                continue
            rw.AddBond(a1, a2, Chem.BondType.AROMATIC)
            rd_bond = rw.GetBondBetweenAtoms(a1, a2)
            if rd_bond is not None:
                rd_bond.SetIsAromatic(True)

        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            return False

        local_to_editor = {local_idx: atom_id for atom_id, local_idx in atom_map.items()}
        for rd_bond in mol.GetBonds():
            editor_a1 = local_to_editor.get(rd_bond.GetBeginAtomIdx())
            editor_a2 = local_to_editor.get(rd_bond.GetEndAtomIdx())
            if editor_a1 is None or editor_a2 is None:
                continue
            editor_bond = next((b for b in self.bonds if {b.a1, b.a2} == {editor_a1, editor_a2}), None)
            if editor_bond is None or editor_bond.style in {"wedge", "hashed", "triple"}:
                continue
            new_style = "double" if rd_bond.GetBondType() == Chem.BondType.DOUBLE else "single"
            if editor_bond.style != new_style:
                editor_bond.style = new_style
                changed = True
        return changed

    def insert_ring_at_point(self, wx: float, wy: float) -> None:
        _key, _icon, size, aromatic = self._ring_template_def()
        atom_ids: list[int] = []
        for px, py in self._regular_polygon_points_from_bottom(wx, wy, size):
            atom_ids.append(self._reuse_or_add_atom(px, py, element="C", aromatic=aromatic))
        styles = self._ring_bond_styles(size, aromatic)
        for i in range(size):
            self.add_or_update_bond(atom_ids[i], atom_ids[(i + 1) % size], style=styles[i])
        self.selected_atom_id = None
        self._push_history(f"{self.selected_ring_template} inserted")
        self.render_all()

    def insert_ring_on_atom(self, atom_id: int, cursor_pos: tuple[float, float]) -> None:
        atom = self._atom_by_id(atom_id)
        if atom is None:
            return
        _key, _icon, size, aromatic = self._ring_template_def()
        points = self._regular_polygon_points_from_bottom(atom.x, atom.y, size)
        desired = math.atan2(cursor_pos[1] - atom.y, cursor_pos[0] - atom.x)
        delta = desired - math.pi / 2
        rotated = [points[0]]
        for px, py in points[1:]:
            rx, ry = self._rotate(px - atom.x, py - atom.y, delta)
            rotated.append((atom.x + rx, atom.y + ry))
        atom_ids = [atom_id]
        for px, py in rotated[1:]:
            atom_ids.append(self._reuse_or_add_atom(px, py, element="C", aromatic=aromatic))
        styles = self._ring_bond_styles(size, aromatic)
        for i in range(size):
            self.add_or_update_bond(atom_ids[i], atom_ids[(i + 1) % size], style=styles[i])
        self.selected_atom_id = None
        self._push_history(f"{self.selected_ring_template} attached")
        self.render_all()

    def _fused_ring_geometry(self, bond_id: int, click_pos: tuple[float, float]) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
        bond = self._bond_by_id(bond_id)
        if bond is None:
            return [], []
        a1 = self._atom_by_id(bond.a1)
        a2 = self._atom_by_id(bond.a2)
        if not a1 or not a2:
            return [], []
        _key, _icon, size, aromatic = self._ring_template_def()
        vx = a2.x - a1.x
        vy = a2.y - a1.y
        length = math.hypot(vx, vy)
        if length < 1e-9:
            return [], []
        ux, uy = vx / length, vy / length
        vx, vy = ux * STANDARD_BOND_LENGTH, uy * STANDARD_BOND_LENGTH
        sign = 1.0
        cross = vx * (click_pos[1] - a1.y) - vy * (click_pos[0] - a1.x)
        if cross < 0:
            sign = -1.0
        ext_angle = sign * (2 * math.pi / size)
        vec_x, vec_y = vx, vy
        curr_x, curr_y = a2.x, a2.y
        points: list[tuple[float, float]] = [(a1.x, a1.y), (a2.x, a2.y)]
        for _ in range(size - 2):
            vec_x, vec_y = self._rotate(vec_x, vec_y, ext_angle)
            curr_x += vec_x
            curr_y += vec_y
            points.append((curr_x, curr_y))
        edges = [(i, i + 1) for i in range(len(points) - 1)]
        edges.append((len(points) - 1, 0))
        return points, edges

    def insert_fused_ring_on_bond(self, bond_id: int, click_pos: tuple[float, float]) -> None:
        bond = self._bond_by_id(bond_id)
        if bond is None:
            return
        _key, _icon, _size, aromatic = self._ring_template_def()
        points, _edges = self._fused_ring_geometry(bond_id, click_pos)
        if len(points) < 3:
            return
        new_atom_ids: list[int] = []
        for px, py in points[2:]:
            new_atom_ids.append(self._reuse_or_add_atom(px, py, element="C", aromatic=aromatic))
        chain = [bond.a1, bond.a2, *new_atom_ids]
        styles = self._ring_bond_styles_for_fusion(len(chain), aromatic, bond.style)
        for i in range(len(chain) - 1):
            self.add_or_update_bond(chain[i], chain[i + 1], style=styles[i])
        self.add_or_update_bond(chain[-1], bond.a1, style=styles[-1])
        if aromatic:
            self._rebalance_aromatic_component(set(chain))
        self.selected_atom_id = None
        self._push_history(f"{self.selected_ring_template} fused")
        self.render_all()

    # ------------------------------------------------------------------
    # RDKit
    # ------------------------------------------------------------------
    def to_rdkit_mol(self) -> Chem.Mol | None:
        if not self.atoms:
            return None
        rw = Chem.RWMol()
        atom_map: dict[int, int] = {}
        for atom in self.atoms:
            rd_atom = Chem.Atom(atom.element)
            rd_atom.SetFormalCharge(atom.charge)
            rd_atom.SetIsAromatic(atom.aromatic)
            if atom.element == "*" and atom.label.upper().startswith("R"):
                try:
                    rd_atom.SetAtomMapNum(int(atom.label[1:]))
                except Exception:
                    pass
            atom_map[atom.id] = rw.AddAtom(rd_atom)
        for bond in self.bonds:
            if bond.style == "double":
                bond_type = Chem.BondType.DOUBLE
            elif bond.style == "triple":
                bond_type = Chem.BondType.TRIPLE
            elif bond.style == "aromatic":
                bond_type = Chem.BondType.AROMATIC
            else:
                bond_type = Chem.BondType.SINGLE
            rw.AddBond(atom_map[bond.a1], atom_map[bond.a2], bond_type)
            rd_bond = rw.GetBondBetweenAtoms(atom_map[bond.a1], atom_map[bond.a2])
            if rd_bond is not None:
                if bond.style == "aromatic":
                    rd_bond.SetIsAromatic(True)
                elif bond.style == "wedge":
                    rd_bond.SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif bond.style == "hashed":
                    rd_bond.SetBondDir(Chem.BondDir.BEGINDASH)

        mol = rw.GetMol()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom in self.atoms:
            conf.SetAtomPosition(atom_map[atom.id], Point3D(float(atom.x), float(atom.y), 0.0))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                return None
        try:
            Chem.AssignChiralTypesFromBondDirs(mol, replaceExistingTags=True)
        except Exception:
            pass
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        except Exception:
            pass
        return mol

    def load_from_rdkit_mol(self, mol: Chem.Mol) -> None:
        mol = self._prepare_import_mol_for_editor(mol)
        if mol is None:
            return
        conf = mol.GetConformer()
        self.atoms.clear()
        self.bonds.clear()
        self.next_atom_id = 1
        self.next_bond_id = 1
        rd_to_editor: dict[int, int] = {}
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            rd_to_editor[atom.GetIdx()] = self.add_atom(
                float(pos.x),
                float(pos.y),
                element=atom.GetSymbol(),
                charge=atom.GetFormalCharge(),
                aromatic=atom.GetIsAromatic(),
                label=self._dummy_atom_label_from_rdkit(atom),
            )
        for bond in mol.GetBonds():
            if bond.GetIsAromatic():
                style = "aromatic"
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                style = "double"
            elif bond.GetBondType() == Chem.BondType.TRIPLE:
                style = "triple"
            else:
                style = "single"
            if bond.GetBondDir() == Chem.BondDir.BEGINWEDGE:
                style = "wedge"
            elif bond.GetBondDir() == Chem.BondDir.BEGINDASH:
                style = "hashed"
            self.add_or_update_bond(rd_to_editor[bond.GetBeginAtomIdx()], rd_to_editor[bond.GetEndAtomIdx()], style=style)
        self.selected_atom_id = None
        self.selected_bond_id = None
        self.pending_draw_start_atom_id = None
        self.preview_world_pos = None
        self._push_history("Structure loaded")
        self.center_view()
        self.render_all()

    def append_from_rdkit_mol(self, mol: Chem.Mol, anchor: tuple[float, float] | None = None) -> None:
        mol = self._prepare_import_mol_for_editor(mol)
        if mol is None:
            return
        conf = mol.GetConformer()
        coords: list[tuple[float, float]] = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append((float(pos.x), float(pos.y)))
        if not coords:
            return
        if anchor is None:
            anchor = self.last_canvas_click_world
        cx = sum(x for x, _y in coords) / len(coords)
        cy = sum(y for _x, y in coords) / len(coords)
        dx = anchor[0] - cx
        dy = anchor[1] - cy

        rd_to_editor: dict[int, int] = {}
        new_atom_ids: set[int] = set()
        new_bond_ids: set[int] = set()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            editor_id = self.add_atom(
                float(pos.x) + dx,
                float(pos.y) + dy,
                element=atom.GetSymbol(),
                charge=atom.GetFormalCharge(),
                aromatic=atom.GetIsAromatic(),
                label=self._dummy_atom_label_from_rdkit(atom),
            )
            rd_to_editor[atom.GetIdx()] = editor_id
            new_atom_ids.add(editor_id)
        for bond in mol.GetBonds():
            if bond.GetIsAromatic():
                style = "aromatic"
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                style = "double"
            elif bond.GetBondType() == Chem.BondType.TRIPLE:
                style = "triple"
            else:
                style = "single"
            if bond.GetBondDir() == Chem.BondDir.BEGINWEDGE:
                style = "wedge"
            elif bond.GetBondDir() == Chem.BondDir.BEGINDASH:
                style = "hashed"
            bond_id = self.add_or_update_bond(
                rd_to_editor[bond.GetBeginAtomIdx()],
                rd_to_editor[bond.GetEndAtomIdx()],
                style=style,
            )
            if bond_id >= 0:
                new_bond_ids.add(bond_id)
        self.selected_atom_ids = new_atom_ids
        self.selected_bond_ids = new_bond_ids
        self.selected_atom_id = next(iter(new_atom_ids), None)
        self.selected_bond_id = next(iter(new_bond_ids), None) if not new_atom_ids else None
        self._push_history("Structure pasted")
        self.render_all()

    def _prepare_import_mol_for_editor(self, mol: Chem.Mol | None) -> Chem.Mol | None:
        if mol is None:
            return None
        prepared = Chem.Mol(mol)
        try:
            Chem.Kekulize(prepared, clearAromaticFlags=True)
        except Exception:
            pass
        try:
            Chem.AssignStereochemistry(prepared, cleanIt=True, force=True)
        except Exception:
            pass
        if prepared.GetNumConformers() == 0:
            try:
                AllChem.Compute2DCoords(prepared)
            except Exception:
                return prepared
        try:
            Chem.WedgeMolBonds(prepared, prepared.GetConformer())
        except Exception:
            pass
        return prepared

    def clean_2d(self) -> None:
        mol = self.to_rdkit_mol()
        if mol is None:
            self.set_status("Cannot clean layout: invalid structure")
            return
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        for idx, atom in enumerate(self.atoms):
            pos = conf.GetAtomPosition(idx)
            atom.x = float(pos.x)
            atom.y = float(pos.y)
        self._push_history("2D coordinates optimized")
        self.center_view()
        self.render_all()

    # ------------------------------------------------------------------
    # Import / export
    # ------------------------------------------------------------------
    def load_smiles_from_input(self) -> None:
        text = self.smiles_var.get().strip()
        if not text:
            self.set_status("No SMILES provided")
            return
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            self.set_status("Invalid SMILES")
            return
        AllChem.Compute2DCoords(mol)
        self.load_from_rdkit_mol(mol)

    def load_molblock_from_input(self) -> None:
        assert self.molblock_text is not None
        text = self.molblock_text.get("1.0", "end").strip()
        if not text:
            self.set_status("No MolBlock provided")
            return
        mol = Chem.MolFromMolBlock(text, sanitize=True, removeHs=False)
        if mol is None:
            self.set_status("Invalid MolBlock")
            return
        self.load_from_rdkit_mol(mol)

    def _mol_from_pasted_text(self, text: str) -> Chem.Mol | None:
        raw = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        pasted = raw.strip()
        if not pasted:
            return None
        molblock_candidate = raw
        if "$$$$" in molblock_candidate:
            molblock_candidate = molblock_candidate.split("$$$$", 1)[0]
        if "M  END" in molblock_candidate:
            molblock_candidate = molblock_candidate.split("M  END", 1)[0] + "M  END\n"
        mol: Chem.Mol | None = None
        if pasted.startswith("InChI="):
            try:
                mol = Chem.MolFromInchi(pasted)
            except Exception:
                mol = None
        if mol is None and ("\n" in raw or "\r" in raw):
            try:
                mol = Chem.MolFromMolBlock(molblock_candidate, sanitize=True, removeHs=False)
            except Exception:
                mol = None
            if mol is None:
                try:
                    mol = Chem.MolFromMolBlock(molblock_candidate, sanitize=False, removeHs=False, strictParsing=False)
                    if mol is not None:
                        try:
                            Chem.SanitizeMol(mol)
                        except Exception:
                            try:
                                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                            except Exception:
                                mol = None
                except Exception:
                    mol = None
        if mol is None:
            try:
                mol = Chem.MolFromSmiles(pasted)
            except Exception:
                mol = None
        if mol is None:
            try:
                mol = Chem.MolFromSmarts(pasted)
            except Exception:
                mol = None
        return mol

    def _selected_subgraph_to_rdkit_mol(self) -> Chem.Mol | None:
        atom_ids = set(self.selected_atom_ids)
        bond_ids = set(self.selected_bond_ids)
        if bond_ids:
            for bond in self.bonds:
                if bond.id in bond_ids:
                    atom_ids.add(bond.a1)
                    atom_ids.add(bond.a2)
        if not atom_ids:
            return None
        rw = Chem.RWMol()
        atom_map: dict[int, int] = {}
        for atom in self.atoms:
            if atom.id not in atom_ids:
                continue
            rd_atom = Chem.Atom(atom.element)
            rd_atom.SetFormalCharge(atom.charge)
            rd_atom.SetIsAromatic(atom.aromatic)
            if atom.element == "*" and atom.label.upper().startswith("R"):
                try:
                    rd_atom.SetAtomMapNum(int(atom.label[1:]))
                except Exception:
                    pass
            atom_map[atom.id] = rw.AddAtom(rd_atom)
        included_bonds = [
            bond for bond in self.bonds
            if bond.a1 in atom_ids and bond.a2 in atom_ids and (not bond_ids or bond.id in bond_ids or {bond.a1, bond.a2}.issubset(atom_ids))
        ]
        for bond in included_bonds:
            if bond.style == "double":
                bond_type = Chem.BondType.DOUBLE
            elif bond.style == "triple":
                bond_type = Chem.BondType.TRIPLE
            elif bond.style == "aromatic":
                bond_type = Chem.BondType.AROMATIC
            else:
                bond_type = Chem.BondType.SINGLE
            rw.AddBond(atom_map[bond.a1], atom_map[bond.a2], bond_type)
            rd_bond = rw.GetBondBetweenAtoms(atom_map[bond.a1], atom_map[bond.a2])
            if rd_bond is not None:
                if bond.style == "aromatic":
                    rd_bond.SetIsAromatic(True)
                elif bond.style == "wedge":
                    rd_bond.SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif bond.style == "hashed":
                    rd_bond.SetBondDir(Chem.BondDir.BEGINDASH)
        mol = rw.GetMol()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom in self.atoms:
            if atom.id not in atom_ids:
                continue
            conf.SetAtomPosition(atom_map[atom.id], Point3D(float(atom.x), float(atom.y), 0.0))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                return None
        try:
            Chem.AssignChiralTypesFromBondDirs(mol, replaceExistingTags=True)
        except Exception:
            pass
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        except Exception:
            pass
        return mol

    def _show_invalid_paste_popup(self) -> None:
        try:
            messagebox.showerror(
                "Invalid structure",
                "The clipboard content is not a valid SMILES, SMARTS, InChI, or MolBlock.",
                parent=self.root,
            )
        except Exception:
            pass

    def _copy_text_to_clipboard(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        try:
            self.root.update()
        except Exception:
            pass

    def on_canvas_paste(self, _event: tk.Event[tk.Misc] | None = None) -> str:
        try:
            pasted = self.root.clipboard_get()
        except Exception:
            pasted = ""
        mol = self._mol_from_pasted_text(pasted)
        if mol is None:
            self.set_status("Invalid pasted structure")
            self._show_invalid_paste_popup()
            return "break"
        try:
            if mol.GetNumConformers() == 0:
                AllChem.Compute2DCoords(mol)
        except Exception:
            pass
        self.append_from_rdkit_mol(mol, anchor=self.last_canvas_click_world)
        self.mode = "select"
        self._refresh_mode_buttons()
        self.set_status("Structure pasted from clipboard")
        self.render_all()
        return "break"

    def on_copy_selection(self, _event: tk.Event[tk.Misc] | None = None) -> str | None:
        mol = self._selected_subgraph_to_rdkit_mol()
        if mol is None:
            return None
        try:
            molblock = Chem.MolToMolBlock(mol)
        except Exception:
            return None
        self._copy_text_to_clipboard(molblock)
        self.set_status("Selection copied as MolBlock")
        return "break"

    def on_cut_selection(self, _event: tk.Event[tk.Misc] | None = None) -> str | None:
        copied = self.on_copy_selection()
        if copied != "break":
            return None
        self.delete_selected()
        self.set_status("Selection cut as MolBlock")
        return "break"

    def current_smiles(self) -> str:
        mol = self.to_rdkit_mol()
        if mol is None:
            return ""
        try:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            return ""

    def copy_current_smiles(self) -> None:
        smiles = self.current_smiles()
        if not smiles:
            self.set_status("No valid structure to copy")
            return
        self._copy_text_to_clipboard(smiles)
        self.set_status("SMILES copied to clipboard")

    def _build_image_export_mol(self) -> Chem.Mol | None:
        mol = self.to_rdkit_mol()
        if mol is None:
            return None
        try:
            molblock = Chem.MolToMolBlock(mol)
        except Exception:
            return None
        export_mol = None
        for sanitize, strict_parsing in ((False, False), (True, False), (False, True), (True, True)):
            try:
                export_mol = Chem.MolFromMolBlock(
                    molblock,
                    sanitize=sanitize,
                    removeHs=False,
                    strictParsing=strict_parsing,
                )
            except Exception:
                export_mol = None
            if export_mol is not None:
                break
        if export_mol is None:
            return None
        for idx, atom in enumerate(self.atoms):
            if idx >= export_mol.GetNumAtoms():
                break
            rd_atom = export_mol.GetAtomWithIdx(idx)
            if atom.label:
                rd_atom.SetProp("atomLabel", atom.label)
            elif rd_atom.HasProp("atomLabel"):
                rd_atom.ClearProp("atomLabel")
        return export_mol

    def _configure_image_export_sketcher(self, sketcher: rdMolDraw2D.MolDraw2DBase) -> None:
        opts = sketcher.drawOptions()
        opts.prepareMolsBeforeDrawing = False
        if self.colored_atoms_var.get():
            palette: dict[int, tuple[float, float, float]] = {}
            for symbol, hex_color in ATOM_COLORS.items():
                if symbol == "*":
                    continue
                try:
                    atomic_num = _PERIODIC_TABLE.GetAtomicNumber(symbol)
                except Exception:
                    atomic_num = 0
                if atomic_num <= 0:
                    continue
                hex_value = hex_color.lstrip("#")
                if len(hex_value) != 6:
                    continue
                try:
                    red = int(hex_value[0:2], 16) / 255.0
                    green = int(hex_value[2:4], 16) / 255.0
                    blue = int(hex_value[4:6], 16) / 255.0
                except Exception:
                    continue
                palette[atomic_num] = (red, green, blue)
            if palette:
                opts.useBWAtomPalette()
                opts.updateAtomPalette(palette)
        else:
            opts.useBWAtomPalette()

    def save_image_dialog(self) -> None:
        smiles = self.current_smiles().strip()
        mol = self._build_image_export_mol()
        if not smiles or mol is None:
            messagebox.showerror("Save image", "Cannot save image: the current structure is not valid.")
            self.set_status("Cannot save image: invalid structure")
            return
        path = filedialog.asksaveasfilename(
            title="Save image",
            initialdir=str(Path.cwd()),
            initialfile="molecule.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("SVG image", "*.svg")],
        )
        if not path:
            return
        out_path = Path(path)
        ext = out_path.suffix.lower()
        if ext not in {".png", ".svg"}:
            out_path = out_path.with_suffix(".png")
            ext = ".png"
        try:
            if ext == ".svg":
                sketcher = rdMolDraw2D.MolDraw2DSVG(1400, 1000)
                self._configure_image_export_sketcher(sketcher)
                sketcher.DrawMolecule(mol)
                sketcher.FinishDrawing()
                out_path.write_text(sketcher.GetDrawingText(), encoding="utf-8")
            else:
                sketcher = rdMolDraw2D.MolDraw2DCairo(1400, 1000)
                self._configure_image_export_sketcher(sketcher)
                sketcher.DrawMolecule(mol)
                sketcher.FinishDrawing()
                out_path.write_bytes(sketcher.GetDrawingText())
            self.set_status(f"Image saved to {out_path}")
        except Exception as exc:
            messagebox.showerror("Save image", f"Failed to save image.\n\n{exc}")
            self.set_status("Failed to save image")

    def choose_output_base(self) -> None:
        initial = Path(self.output_base_var.get()).parent
        path = filedialog.asksaveasfilename(
            title="Choose output base name",
            initialdir=str(initial if initial.exists() else Path.cwd()),
            initialfile=Path(self.output_base_var.get()).name,
            defaultextension=".png",
            filetypes=[("All files", "*.*")],
        )
        if path:
            self.output_base_var.set(str(Path(path).with_suffix("")))

    def save_png(self) -> None:
        mol = self.to_rdkit_mol()
        if mol is None:
            self.set_status("Cannot export PNG: invalid structure")
            return
        path = Path(self.output_base_var.get()).with_suffix(".png")
        sketcher = rdMolDraw2D.MolDraw2DCairo(1400, 1000)
        rdMolDraw2D.PrepareAndDrawMolecule(sketcher, mol)
        sketcher.FinishDrawing()
        path.write_bytes(sketcher.GetDrawingText())
        self.set_status(f"PNG saved to {path}")

    def save_sdf(self) -> None:
        mol = self.to_rdkit_mol()
        if mol is None:
            self.set_status("Cannot export SDF: invalid structure")
            return
        path = Path(self.output_base_var.get()).with_suffix(".sdf")
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()
        self.set_status(f"SDF saved to {path}")

    def save_mol(self) -> None:
        mol = self.to_rdkit_mol()
        if mol is None:
            self.set_status("Cannot export MOL: invalid structure")
            return
        path = Path(self.output_base_var.get()).with_suffix(".mol")
        path.write_text(Chem.MolToMolBlock(mol), encoding="utf-8")
        self.set_status(f"MOL saved to {path}")

    # ------------------------------------------------------------------
    # Geometry / drawing logic
    # ------------------------------------------------------------------
    def center_view(self) -> None:
        canvas_width, canvas_height = self._canvas_size()
        if not self.atoms:
            self.scale = 31.0
            self.offset_x = canvas_width / 2
            self.offset_y = canvas_height / 2
            self.render_all()
            return
        xs = [a.x for a in self.atoms]
        ys = [a.y for a in self.atoms]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = max(0.8, (max_x - min_x) * 0.08)
        pad_y = max(0.8, (max_y - min_y) * 0.10)
        min_x -= pad_x
        max_x += pad_x
        min_y -= pad_y
        max_y += pad_y
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        scale_x = (canvas_width * 0.90) / span_x
        scale_y = (canvas_height * 0.90) / span_y
        self.scale = max(20.0, min(4000.0, min(scale_x, scale_y)))
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        self.offset_x = (canvas_width / 2.0) - center_x * self.scale
        self.offset_y = (canvas_height / 2.0) + center_y * self.scale
        self.render_all()

    def render_all(self) -> None:
        self.render_canvas()
        if self.drag_atom_id is None and not self.is_rotating and not self.is_panning:
            self.current_smiles_var.set(self.current_smiles())

    def request_canvas_render(self) -> None:
        if self._canvas_render_pending:
            return
        self._canvas_render_pending = True
        self._canvas_render_after_id = self.root.after(16, self._flush_canvas_render)

    def _current_render_aa_scale(self) -> int:
        interactive = (
            self.left_button_down
            or self.right_click_dragged
            or self.drag_atom_id is not None
            or self.is_rotating
            or self.is_panning
            or self.selection_rect_start is not None
            or self.erase_rect_start is not None
            or self.pending_draw_start_pos is not None
            or self.pending_ring_active
        )
        return max(1, int(self._aa_render_scale_interactive if interactive else self._aa_render_scale_idle))

    def _cursor_overlay_active(self) -> bool:
        control_mode_active = self.mode in {"select", "erase", "pan", "rotate"} or self.current_charge != 0
        if self.mode == "ring":
            return True
        if control_mode_active and self.mouse_screen_pos is not None:
            return True
        if self.current_charge != 0 and self.mouse_screen_pos is not None:
            return True
        return bool(
            not control_mode_active
            and self.mode != "ring"
            and self.selected_atom_symbol
            and self.mouse_screen_pos is not None
        )

    def _flush_canvas_render(self) -> None:
        self._canvas_render_pending = False
        self._canvas_render_after_id = None
        self.render_canvas()

    def render_canvas(self) -> None:
        self._canvas_render_pending = False
        self._canvas_render_after_id = None
        assert self.canvas is not None
        canvas_width = max(1, int(self.canvas.winfo_width() or self.canvas.winfo_reqwidth() or CANVAS_WIDTH))
        canvas_height = max(1, int(self.canvas.winfo_height() or self.canvas.winfo_reqheight() or CANVAS_HEIGHT))
        aa = self._current_render_aa_scale()
        self._active_aa_render_scale = aa
        self._aa_render_image = Image.new("RGBA", (canvas_width * aa, canvas_height * aa), (255, 255, 255, 255))
        self._aa_render_draw = ImageDraw.Draw(self._aa_render_image)
        self.preview_fuse_atom_ids = set()
        self.preview_fuse_bond_ids = set()
        if self.drag_atom_id is not None:
            self._draw_drag_fuse_overlay()
        self.canvas.delete("all")
        self.draw_bonds()
        self.draw_atoms()
        self.draw_guides()
        assert self._aa_render_image is not None
        final_image = self._aa_render_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.preview_image_tk = ImageTk.PhotoImage(final_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.preview_image_tk)
        self._aa_render_draw = None
        self._aa_render_image = None

    def _pil_font_for_spec(self, spec: Any, aa_scale: int | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        scale = max(1, int(aa_scale or self._active_aa_render_scale or self._aa_render_scale))
        family = "Arial"
        size = 12
        weight = "normal"
        if isinstance(spec, tkfont.Font):
            try:
                family = str(spec.cget("family") or family)
                size = abs(int(spec.cget("size") or size))
                weight = str(spec.cget("weight") or weight)
            except Exception:
                pass
        elif isinstance(spec, tuple) and len(spec) >= 2:
            family = str(spec[0] or family)
            try:
                size = abs(int(spec[1]))
            except Exception:
                size = 12
            rest = [str(v).lower() for v in spec[2:]]
            if "bold" in rest:
                weight = "bold"
        cache_key = (family.lower(), max(1, int(round(size * scale))), weight.lower())
        cached = self._pil_font_cache.get(cache_key)
        if cached is not None:
            return cached
        font_size = cache_key[1]
        candidates: list[str] = []
        is_bold = weight.lower() == "bold"
        if family.lower() == "arial":
            candidates.extend(["Arial Bold.ttf", "arialbd.ttf"] if is_bold else ["Arial.ttf", "arial.ttf"])
        candidates.extend(["DejaVuSans-Bold.ttf"] if is_bold else ["DejaVuSans.ttf"])
        for font_name in candidates:
            local_font = FONT_ASSETS_DIR / font_name
            for font_candidate in (str(local_font), font_name):
                try:
                    font = ImageFont.truetype(font_candidate, font_size)
                    self._pil_font_cache[cache_key] = font
                    return font
                except Exception:
                    continue
        fallback = ImageFont.load_default()
        self._pil_font_cache[cache_key] = fallback
        return fallback

    def _aa_line(self, p1: tuple[float, float], p2: tuple[float, float], fill: Any, width: float = 1.0, dash: tuple[int, ...] | None = None) -> None:
        if self._aa_render_draw is None:
            return
        aa = max(1, int(self._active_aa_render_scale or self._aa_render_scale))
        x1, y1 = p1[0] * aa, p1[1] * aa
        x2, y2 = p2[0] * aa, p2[1] * aa
        line_width = max(1, int(round(width * aa)))
        if dash:
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-6:
                return
            ux, uy = dx / length, dy / length
            pattern = [max(1.0, float(v) * aa) for v in dash]
            dist = 0.0
            pattern_idx = 0
            draw_on = True
            while dist < length:
                seg_len = min(pattern[pattern_idx % len(pattern)], length - dist)
                if draw_on:
                    sx = x1 + ux * dist
                    sy = y1 + uy * dist
                    ex = x1 + ux * (dist + seg_len)
                    ey = y1 + uy * (dist + seg_len)
                    self._aa_render_draw.line((sx, sy, ex, ey), fill=fill, width=line_width)
                dist += seg_len
                pattern_idx += 1
                draw_on = not draw_on
        else:
            self._aa_render_draw.line((x1, y1, x2, y2), fill=fill, width=line_width)
            radius = line_width / 2.0
            self._aa_render_draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=fill, outline=fill)
            self._aa_render_draw.ellipse((x2 - radius, y2 - radius, x2 + radius, y2 + radius), fill=fill, outline=fill)

    def _aa_polygon(self, points: list[float], fill: Any | None = None, outline: Any | None = None, width: float = 1.0) -> None:
        if self._aa_render_draw is None:
            return
        aa = max(1, int(self._active_aa_render_scale or self._aa_render_scale))
        scaled: list[float] = []
        for idx, value in enumerate(points):
            scaled.append(float(value) * aa)
        self._aa_render_draw.polygon(scaled, fill=fill, outline=outline)
        if outline is not None and width > 1:
            pts = [(scaled[i], scaled[i + 1]) for i in range(0, len(scaled), 2)]
            for i in range(len(pts)):
                self._aa_line(pts[i], pts[(i + 1) % len(pts)], outline, width=width)

    def _aa_ellipse(self, bbox: tuple[float, float, float, float], outline: Any | None = None, fill: Any | None = None, width: float = 1.0, dash: tuple[int, ...] | None = None) -> None:
        if self._aa_render_draw is None:
            return
        aa = max(1, int(self._active_aa_render_scale or self._aa_render_scale))
        x0, y0, x1, y1 = [float(v) * aa for v in bbox]
        if fill is not None:
            self._aa_render_draw.ellipse((x0, y0, x1, y1), fill=fill, outline=None)
        if outline is None:
            return
        if dash:
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            rx = abs(x1 - x0) / 2.0
            ry = abs(y1 - y0) / 2.0
            points: list[tuple[float, float]] = []
            steps = 96
            for step in range(steps + 1):
                t = (2.0 * math.pi * step) / steps
                points.append((cx + rx * math.cos(t), cy + ry * math.sin(t)))
            pattern = [max(1.0, float(v) * aa) for v in dash]
            pat_idx = 0
            remaining = pattern[0]
            draw_on = True
            for i in range(len(points) - 1):
                p_start = points[i]
                p_end = points[i + 1]
                seg_len = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
                if draw_on:
                    self._aa_render_draw.line((p_start[0], p_start[1], p_end[0], p_end[1]), fill=outline, width=max(1, int(round(width * aa))))
                remaining -= seg_len
                if remaining <= 0:
                    pat_idx = (pat_idx + 1) % len(pattern)
                    remaining = pattern[pat_idx]
                    draw_on = not draw_on
        else:
            for offset in range(max(1, int(round(width * aa)))):
                self._aa_render_draw.ellipse((x0 + offset, y0 + offset, x1 - offset, y1 - offset), outline=outline)

    def _aa_rectangle(self, bbox: tuple[float, float, float, float], outline: Any | None = None, fill: Any | None = None, width: float = 1.0, dash: tuple[int, ...] | None = None) -> None:
        if self._aa_render_draw is None:
            return
        x0, y0, x1, y1 = bbox
        if fill is not None:
            aa = max(1, int(self._active_aa_render_scale or self._aa_render_scale))
            self._aa_render_draw.rectangle((x0 * aa, y0 * aa, x1 * aa, y1 * aa), fill=fill)
        if outline is None:
            return
        self._aa_line((x0, y0), (x1, y0), outline, width=width, dash=dash)
        self._aa_line((x1, y0), (x1, y1), outline, width=width, dash=dash)
        self._aa_line((x1, y1), (x0, y1), outline, width=width, dash=dash)
        self._aa_line((x0, y1), (x0, y0), outline, width=width, dash=dash)

    def _aa_text(self, pos: tuple[float, float], text: str, fill: Any, font: Any, anchor: str = "center") -> None:
        if self._aa_render_draw is None or not text:
            return
        aa = max(1, int(self._active_aa_render_scale or self._aa_render_scale))
        pil_font = self._pil_font_for_spec(font, aa_scale=aa)
        anchor_map = {
            "center": "mm",
            "c": "mm",
            "w": "lm",
            "e": "rm",
            "s": "ms",
            "n": "mt",
        }
        self._aa_render_draw.text((pos[0] * aa, pos[1] * aa), text=text, fill=fill, font=pil_font, anchor=anchor_map.get(anchor, "mm"))

    def _clear_drag_fuse_overlay(self) -> None:
        self.preview_fuse_atom_ids = set()
        self.preview_fuse_bond_ids = set()

    def _clear_drag_bridge_overlay(self) -> None:
        return

    def _drag_bridge_bonds(self, moved_atom_ids: set[int]) -> list[EditorBond]:
        if not moved_atom_ids:
            return []
        return [
            bond
            for bond in self.bonds
            if (bond.a1 in moved_atom_ids) ^ (bond.a2 in moved_atom_ids)
        ]

    def _hide_drag_bridge_bonds(self) -> None:
        moved_ids = set(self.drag_group_ids) or ({self.drag_atom_id} if self.drag_atom_id is not None else set())
        self.drag_bridge_bond_ids = {bond.id for bond in self._drag_bridge_bonds(moved_ids)}

    def _draw_drag_bridge_overlay(self) -> None:
        return

    def _draw_drag_fuse_overlay(self) -> None:
        moved_ids = set(self.drag_group_ids) or ({self.drag_atom_id} if self.drag_atom_id is not None else set())
        if not moved_ids:
            self.preview_fuse_atom_ids = set()
            self.preview_fuse_bond_ids = set()
            return
        preview_atoms, preview_bonds = self._compute_fuse_preview(moved_ids)
        self.preview_fuse_atom_ids = preview_atoms
        self.preview_fuse_bond_ids = preview_bonds

    def draw_grid(self) -> None:
        return

    def draw_bonds(self) -> None:
        for bond in self.bonds:
            a1 = self._atom_by_id(bond.a1)
            a2 = self._atom_by_id(bond.a2)
            if not a1 or not a2:
                continue
            p1 = self.world_to_screen(*self._atom_drag_world_pos(a1))
            p2 = self.world_to_screen(*self._atom_drag_world_pos(a2))
            bond_tags = [f"bond_{bond.id}"]
            if bond.id == self.selected_bond_id or bond.id in self.selected_bond_ids:
                bond_tags.append("drag_selected")
            tags: tuple[str, ...] = tuple(bond_tags)
            color = GUIDE if bond.id in self.preview_fuse_bond_ids else (SELECTED if (bond.id == self.selected_bond_id or bond.id in self.selected_bond_ids or bond.id in self.preview_rect_bond_ids or bond.id == self.hover_bond_id) else (AROMATIC if bond.style == "aromatic" else BOND))
            self._draw_bond_geometry(p1, p2, bond.style, color, bond, tags=tags)

    def _atom_label_disk_radius_px(self, atom: EditorAtom | None) -> float:
        if atom is None:
            return 0.0
        hydrogen_counts = self._hydrogen_count_by_atom_id()
        hydrogen_count = hydrogen_counts.get(atom.id, 0)
        show_hydrogens = self._should_show_hydrogens_on_atom(atom, hydrogen_count)
        if self.should_show_atom_label(atom) or show_hydrogens:
            if atom.label:
                base_label = atom.label
            elif atom.element in CARBON_HIDE_ELEMENTS and show_hydrogens:
                base_label = "C"
            else:
                base_label = atom.element
            _, _, _, _, _, total_width = self._atom_label_layout(
                base_label,
                hydrogen_count if show_hydrogens else 0,
                show_hydrogens,
                atom.charge,
            )
            return max(15.0, total_width / 2.0 + 4.0)
        return 0.0

    def _bond_inner_side_sign(self, bond: EditorBond) -> float:
        a1 = self._atom_by_id(bond.a1)
        a2 = self._atom_by_id(bond.a2)
        if not a1 or not a2:
            return 1.0
        mx = (a1.x + a2.x) / 2.0
        my = (a1.y + a2.y) / 2.0
        path = self._shortest_path_excluding_bond(bond.a1, bond.a2, bond.id)
        cx = 0.0
        cy = 0.0
        count = 0
        if path is not None and len(path) >= 3:
            for atom_id in path[1:-1]:
                atom = self._atom_by_id(atom_id)
                if atom is None:
                    continue
                cx += atom.x
                cy += atom.y
                count += 1
        if count == 0:
            cx = 0.0
            cy = 0.0
            count = 0
            for atom_id in (bond.a1, bond.a2):
                atom = self._atom_by_id(atom_id)
                if atom is None:
                    continue
                for neighbor_id in self._neighbor_ids(atom_id):
                    if neighbor_id in {bond.a1, bond.a2}:
                        continue
                    neighbor = self._atom_by_id(neighbor_id)
                    if neighbor is None:
                        continue
                    cx += neighbor.x
                    cy += neighbor.y
                    count += 1
        if count == 0:
            return 1.0
        cx /= count
        cy /= count
        vx = a2.x - a1.x
        vy = a2.y - a1.y
        nx = -vy
        ny = vx
        dot = (cx - mx) * nx + (cy - my) * ny
        return -1.0 if dot >= 0 else 1.0

    def _bond_is_in_ring(self, bond: EditorBond | None) -> bool:
        if bond is None:
            return False
        path = self._shortest_path_excluding_bond(bond.a1, bond.a2, bond.id)
        return bool(path is not None and len(path) >= 3)

    def _draw_bond_geometry(self, p1: tuple[float, float], p2: tuple[float, float], style: str, color: str, bond: EditorBond | None = None, tags: tuple[str, ...] = ()) -> None:
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return
        if bond is not None:
            a1 = self._atom_by_id(bond.a1)
            a2 = self._atom_by_id(bond.a2)
            start_inset = self._atom_label_disk_radius_px(a1)
            end_inset = self._atom_label_disk_radius_px(a2)
            if start_inset > 0.0 or end_inset > 0.0:
                if start_inset > 0.0:
                    start_inset += ATOM_LABEL_BOND_GAP_PX
                if end_inset > 0.0:
                    end_inset += ATOM_LABEL_BOND_GAP_PX
                ux = dx / length
                uy = dy / length
                # If the on-screen bond becomes shorter than the two label disks,
                # prefer hiding the covered segment entirely instead of letting it
                # peek through inside the white atom background.
                max_inset = max(0.0, (length - 1.0) / 2.0)
                start_inset = min(start_inset, max_inset)
                end_inset = min(end_inset, max_inset)
                x1 += ux * start_inset
                y1 += uy * start_inset
                x2 -= ux * end_inset
                y2 -= uy * end_inset
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length < 1e-9:
                    return
        nx = -dy / length
        ny = dx / length
        bond_in_ring = self._bond_is_in_ring(bond)
        if style == "double":
            if bond_in_ring:
                self._aa_line((x1, y1), (x2, y2), color, width=3)
                sign = self._bond_inner_side_sign(bond) if bond is not None else 1.0
                delta = DOUBLE_TRIPLE_BOND_SPACING * sign
                inset = min(10.0, max(6.0, length * 0.14))
                ux = dx / length
                uy = dy / length
                self._aa_line(
                    (x1 + ux * inset + nx * delta, y1 + uy * inset + ny * delta),
                    (x2 - ux * inset + nx * delta, y2 - uy * inset + ny * delta),
                    color,
                    width=3,
                )
            else:
                delta = DOUBLE_TRIPLE_BOND_SPACING / 2.0
                self._aa_line((x1 + nx * delta, y1 + ny * delta), (x2 + nx * delta, y2 + ny * delta), color, width=3)
                self._aa_line((x1 - nx * delta, y1 - ny * delta), (x2 - nx * delta, y2 - ny * delta), color, width=3)
        elif style == "triple":
            self._aa_line((x1, y1), (x2, y2), color, width=3)
            for delta in (-DOUBLE_TRIPLE_BOND_SPACING, DOUBLE_TRIPLE_BOND_SPACING):
                self._aa_line(
                    (x1 + nx * delta, y1 + ny * delta),
                    (x2 + nx * delta, y2 + ny * delta),
                    color,
                    width=3,
                )
        elif style == "aromatic":
            self._aa_line((x1, y1), (x2, y2), color, width=3)
            dash_count = max(4, int(length // 14))
            for i in range(dash_count):
                t1 = i / dash_count
                t2 = min(1.0, t1 + 0.5 / dash_count)
                sx = x1 + dx * t1
                sy = y1 + dy * t1
                ex = x1 + dx * t2
                ey = y1 + dy * t2
                self._aa_line((sx + nx * 6, sy + ny * 6), (ex + nx * 6, ey + ny * 6), color, width=2)
        elif style == "wedge":
            poly = [x1 - nx * 2, y1 - ny * 2, x1 + nx * 2, y1 + ny * 2, x2 + nx * 10, y2 + ny * 10, x2 - nx * 10, y2 - ny * 10]
            self._aa_polygon(poly, fill=color, outline=color, width=1)
        elif style == "hashed":
            steps = 7
            for i in range(steps):
                t = (i + 1) / steps
                cx = x1 + dx * t
                cy = y1 + dy * t
                width = 2 + i * 1.3
                self._aa_line((cx - nx * width, cy - ny * width), (cx + nx * width, cy + ny * width), color, width=1)
        else:
            self._aa_line((x1, y1), (x2, y2), color, width=3)

    def _atom_valence(self, atom_id: int) -> float:
        total = 0.0
        for bond in self.bonds:
            if bond.a1 == atom_id or bond.a2 == atom_id:
                total += self.bond_order_value(bond.style)
        return total

    def _atom_has_valence_issue(self, atom: EditorAtom) -> bool:
        max_val = MAX_VALENCE.get(atom.element)
        if max_val is None:
            return False
        effective_max_val = max(0.0, max_val + atom.charge)
        return self._atom_valence(atom.id) > effective_max_val + 0.05

    def _stereocenter_info(self) -> dict[int, str]:
        if not self.show_stereocenters:
            return {}
        mol = self.to_rdkit_mol()
        if mol is None:
            return {}
        try:
            AllChem.Compute2DCoords(mol, clearConfs=False)
        except Exception:
            pass
        try:
            Chem.AssignChiralTypesFromBondDirs(mol, replaceExistingTags=True)
        except Exception:
            pass
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        except Exception:
            pass
        try:
            Chem.rdCIPLabeler.AssignCIPLabels(mol)
        except Exception:
            pass
        try:
            Chem.AssignStereochemistry(mol, cleanIt=False, force=True, flagPossibleStereoCenters=True)
        except Exception:
            pass
        try:
            chiral_centers = Chem.FindMolChiralCenters(
                mol,
                force=True,
                includeUnassigned=True,
                includeCIP=True,
                useLegacyImplementation=False,
            )
        except Exception:
            return {}
        info: dict[int, str] = {}
        for idx, label in chiral_centers:
            if not (0 <= idx < len(self.atoms)):
                continue
            cip_label = "?"
            try:
                atom = mol.GetAtomWithIdx(idx)
                if atom.HasProp("_CIPCode"):
                    cip_label = str(atom.GetProp("_CIPCode") or "?").strip().upper() or "?"
                else:
                    raw_label = str(label or "?").strip().upper()
                    if raw_label in {"R", "S"}:
                        cip_label = raw_label
            except Exception:
                raw_label = str(label or "?").strip().upper()
                if raw_label in {"R", "S"}:
                    cip_label = raw_label
            info[self.atoms[idx].id] = cip_label
        return info

    def should_show_atom_label(self, atom: EditorAtom) -> bool:
        if atom.label:
            return True
        if atom.charge != 0:
            return True
        if atom.element == "H":
            return True
        if atom.element in CARBON_HIDE_ELEMENTS:
            return self._should_show_carbon_label(atom)
        if atom.element not in CARBON_HIDE_ELEMENTS:
            return True
        return self._atom_degree(atom.id) == 0

    def draw_atoms(self) -> None:
        hydrogen_counts = self._hydrogen_count_by_atom_id()
        stereocenter_info = self._stereocenter_info()
        for atom in self.atoms:
            sx, sy = self.world_to_screen(*self._atom_drag_world_pos(atom))
            tags: tuple[str, ...] = ("drag_selected",) if (atom.id == self.selected_atom_id or atom.id in self.selected_atom_ids) else ()
            if atom.id in self.preview_fuse_atom_ids:
                self._aa_ellipse((sx - 18, sy - 18, sx + 18, sy + 18), outline=GUIDE, width=3, dash=(4, 2))
            if atom.id == self.selected_atom_id or atom.id in self.selected_atom_ids or atom.id in self.preview_rect_atom_ids or atom.id == self.hover_atom_id:
                outline = RED if self.mode == "erase" and atom.id in self.preview_rect_atom_ids else SELECTED
                self._aa_ellipse((sx - 16, sy - 16, sx + 16, sy + 16), outline=outline, width=3)
            if self._atom_has_valence_issue(atom):
                self._aa_ellipse((sx - 20, sy - 20, sx + 20, sy + 20), outline=RED, width=2)
            stereo_label = stereocenter_info.get(atom.id)
            if stereo_label is not None:
                self._aa_ellipse((sx - 20, sy - 20, sx + 20, sy + 20), outline=BLUE, width=2)
            color = ATOM_COLORS.get(atom.element, "#000000") if self.colored_atoms_var.get() else "#000000"
            hydrogen_count = hydrogen_counts.get(atom.id, 0)
            show_hydrogens = self._should_show_hydrogens_on_atom(atom, hydrogen_count)
            if self.should_show_atom_label(atom) or show_hydrogens:
                if atom.label:
                    base_label = atom.label
                elif atom.element in CARBON_HIDE_ELEMENTS and show_hydrogens:
                    base_label = "C"
                else:
                    base_label = atom.element
                self._draw_atom_label(
                    sx,
                    sy,
                    base_label,
                    hydrogen_count if show_hydrogens else 0,
                    show_hydrogens,
                    atom.charge,
                    color,
                )
            if stereo_label is not None:
                label_x = sx + 18
                label_y = sy - 16
                self._aa_ellipse((label_x - 8, label_y - 7, label_x + 8, label_y + 7), fill="#ffffff", outline="#ffffff", width=1)
                self._aa_text((label_x, label_y), stereo_label.upper(), BLUE, ("Arial", 10, "bold"), anchor="center")

    def draw_guides(self) -> None:
        control_mode_active = self.mode in {"select", "erase", "pan", "rotate"} or self.current_charge != 0
        bond_hover_active = self.hover_bond_id is not None
        mode_cursor_labels = {
            "select": "Select",
            "erase": "Erase",
            "pan": "Pan",
            "rotate": "Rotate",
        }
        if self.mode in mode_cursor_labels and self.mouse_screen_pos is not None and not self.left_button_down:
            mx, my = self.mouse_screen_pos
            self._aa_text((mx, my - 8), mode_cursor_labels[self.mode], GUIDE, ("Arial", 10, "bold"), anchor="s")
        if self.current_charge != 0 and self.mouse_screen_pos is not None and not bond_hover_active:
            mx, my = self.mouse_screen_pos
            charge_text = "+" if self.current_charge > 0 else "–"
            self._aa_text((mx, my), charge_text, GUIDE, ("Arial", 18, "bold"), anchor="center")
        elif not control_mode_active and self.mode != "ring" and self.selected_atom_symbol and self.mouse_screen_pos is not None and not bond_hover_active:
            if self.preview_world_pos is not None:
                mx, my = self.world_to_screen(*self.preview_world_pos)
            else:
                mx, my = self.mouse_screen_pos
            preview_label = self._next_r_group_label() if self.selected_atom_symbol == "R" else self.selected_atom_symbol
            self._aa_text((mx, my), preview_label, GUIDE, ("Arial", 16, "bold"), anchor="center")
        if self.pending_ring_active and self.preview_ring_points:
            screen_points = [self.world_to_screen(px, py) for px, py in self.preview_ring_points]
            for a_idx, b_idx in self.preview_ring_bonds:
                x1, y1 = screen_points[a_idx]
                x2, y2 = screen_points[b_idx]
                self._aa_line((x1, y1), (x2, y2), GUIDE, width=2, dash=(4, 3))
            for sx, sy in screen_points:
                self._aa_ellipse((sx - 4, sy - 4, sx + 4, sy + 4), outline=GUIDE, width=1)
            return
        if self.erase_rect_start is not None and self.erase_rect_end is not None:
            x0, y0 = self.erase_rect_start
            x1, y1 = self.erase_rect_end
            self._aa_rectangle((x0, y0, x1, y1), outline=RED, width=2, dash=(4, 3))
            return
        if self.pending_draw_start_pos is None:
            if self.selection_rect_start is not None and self.selection_rect_end is not None:
                x0, y0 = self.selection_rect_start
                x1, y1 = self.selection_rect_end
                self._aa_rectangle((x0, y0, x1, y1), outline=GUIDE, width=2, dash=(4, 3))
            elif self.mode == "rotate" and self.is_rotating and self.mouse_screen_pos is not None:
                cx, cy = self.world_to_screen(*self.rotation_center)
                current_angle = math.atan2(self.mouse_screen_pos[1] - cy, self.mouse_screen_pos[0] - cx)
                delta_deg = math.degrees(self.rotation_anchor_angle - current_angle)
                mx, my = self.mouse_screen_pos
                self._aa_text((mx, my - 18), f"{delta_deg:+.1f}°", GUIDE, ("Arial", 11, "bold"), anchor="s")
            return
        if self.selected_bond_style == "chain" and self.preview_world_path:
            points = [self.pending_draw_start_pos, *self.preview_world_path]
            for i in range(len(points) - 1):
                x1, y1 = self.world_to_screen(*points[i])
                x2, y2 = self.world_to_screen(*points[i + 1])
                self._aa_line((x1, y1), (x2, y2), GUIDE, width=2, dash=(4, 3))
            lx, ly = self.world_to_screen(*points[-1])
            self._aa_ellipse((lx - 12, ly - 12, lx + 12, ly + 12), outline=GUIDE, width=2)
            angle_text = self._chain_axis_angle_text()
            if angle_text is not None:
                self._aa_text((lx, ly - 22), angle_text, GUIDE, ("Arial", 11, "bold"), anchor="s")
            return
        if self.preview_world_pos is None:
            return
        x1, y1 = self.world_to_screen(*self.pending_draw_start_pos)
        x2, y2 = self.world_to_screen(*self.preview_world_pos)
        self._aa_line((x1, y1), (x2, y2), GUIDE, width=2)
        self._aa_ellipse((x2 - 12, y2 - 12, x2 + 12, y2 + 12), outline=GUIDE, width=2)
        angle_text = self._preview_attachment_angle_text()
        if angle_text is not None:
            self._aa_text((x2, y2 - 22), angle_text, GUIDE, ("Arial", 11, "bold"), anchor="s")

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------
    def on_canvas_context_menu(self, event: tk.Event[tk.Misc]) -> str:
        try:
            assert self.canvas is not None
            self.canvas.focus_set()
        except Exception:
            pass
        self.mouse_screen_pos = (event.x, event.y)
        wx, wy = self.screen_to_world(event.x, event.y)
        self.last_canvas_click_world = (wx, wy)
        self._hide_tooltip()
        self._show_canvas_context_menu(event.x_root, event.y_root)
        return "break"

    def on_right_down(self, event: tk.Event[tk.Misc]) -> str:
        try:
            assert self.canvas is not None
            self.canvas.focus_set()
        except Exception:
            pass
        self.mouse_screen_pos = (event.x, event.y)
        wx, wy = self.screen_to_world(event.x, event.y)
        self.last_canvas_click_world = (wx, wy)
        self.right_click_anchor = (event.x, event.y)
        self.right_click_root = (event.x_root, event.y_root)
        self.right_click_dragged = False
        self.is_panning = False
        self.pan_anchor = (event.x, event.y)
        self.pan_origin = (self.offset_x, self.offset_y)
        return "break"

    def on_right_drag(self, event: tk.Event[tk.Misc]) -> str:
        self.mouse_screen_pos = (event.x, event.y)
        moved_px = math.dist(self.right_click_anchor, (event.x, event.y))
        if moved_px >= DRAG_THRESHOLD_PX:
            self.right_click_dragged = True
            self.is_panning = True
            self.offset_x = self.pan_origin[0] + (event.x - self.pan_anchor[0])
            self.offset_y = self.pan_origin[1] + (event.y - self.pan_anchor[1])
            self.request_canvas_render()
        return "break"

    def on_right_up(self, event: tk.Event[tk.Misc]) -> str:
        self.mouse_screen_pos = (event.x, event.y)
        moved_px = math.dist(self.right_click_anchor, (event.x, event.y))
        was_drag = self.right_click_dragged or moved_px >= DRAG_THRESHOLD_PX
        self.is_panning = False
        self.right_click_dragged = False
        if not was_drag:
            wx, wy = self.screen_to_world(event.x, event.y)
            self.last_canvas_click_world = (wx, wy)
            self._hide_tooltip()
            self._show_canvas_context_menu(event.x_root, event.y_root)
        return "break"

    def on_left_down(self, event: tk.Event[tk.Misc]) -> None:
        try:
            assert self.canvas is not None
            self.canvas.focus_set()
        except Exception:
            pass
        self.left_button_down = True
        self.mouse_screen_pos = (event.x, event.y)
        wx, wy = self.screen_to_world(event.x, event.y)
        self.last_canvas_click_world = (wx, wy)
        atom_id, bond_id = self._pick_hit_targets(wx, wy)
        self.drag_start_screen = (event.x, event.y)

        if self.current_charge != 0:
            if atom_id is not None:
                atom = self._atom_by_id(atom_id)
                if atom is not None:
                    atom.charge = max(-3, min(3, atom.charge + self.current_charge))
                    self._push_history("Atom charge updated")
                    self.render_all()
            return

        if self.mode == "select":
            self.selection_rect_start = None
            self.selection_rect_end = None
            self.preview_fuse_atom_ids = set()
            self.preview_fuse_bond_ids = set()
            self.selected_atom_id = atom_id
            self.selected_bond_id = None if atom_id is not None else bond_id
            if atom_id is not None:
                if atom_id not in self.selected_atom_ids:
                    self.selected_atom_ids = {atom_id}
                    self.selected_bond_ids = set()
                atom = self._atom_by_id(atom_id)
                if atom is not None:
                    if self.selected_atom_ids:
                        self.drag_group_ids = set(self.selected_atom_ids)
                        self.drag_group_origin = {aid: (self._atom_by_id(aid).x, self._atom_by_id(aid).y) for aid in self.drag_group_ids if self._atom_by_id(aid) is not None}
                        self.drag_atom_id = atom_id
                        self.drag_offset_world = (atom.x - wx, atom.y - wy)
                        self.drag_last_screen = (event.x, event.y)
            elif bond_id is not None:
                self.selected_atom_ids = set()
                self.selected_bond_ids = {bond_id}
            else:
                self.selected_atom_ids = set()
                self.selected_bond_ids = set()
                self.selection_rect_start = (event.x, event.y)
                self.selection_rect_end = (event.x, event.y)
            self.render_all()
            return

        if self.mode == "erase":
            self.erase_rect_start = (event.x, event.y)
            self.erase_rect_end = (event.x, event.y)
            self.erase_hit_atom_id = atom_id
            self.erase_hit_bond_id = bond_id
            self.preview_rect_atom_ids = set()
            self.preview_rect_bond_ids = set()
            self.render_all()
            return

        if self.mode == "atom":
            if atom_id is not None:
                start_atom = self._atom_by_id(atom_id)
                if start_atom is not None:
                    self.pending_draw_start_atom_id = atom_id
                    self.pending_draw_start_pos = (start_atom.x, start_atom.y)
                    self.pending_draw_mode = "atom_from_atom"
                    self.preview_world_pos = None
                    self.preview_world_path = []
                    self.selected_atom_id = atom_id
                    self.selected_bond_id = None
                    self.render_all()
                    return
            self.pending_draw_start_atom_id = None
            self.pending_draw_start_pos = (wx, wy)
            self.pending_draw_mode = "atom_free"
            self.preview_world_pos = (wx + STANDARD_BOND_LENGTH, wy)
            self.preview_world_path = []
            self.selected_atom_id = None
            self.selected_bond_id = None
            self.render_all()
            return

        if self.mode == "bond":
            if bond_id is not None and atom_id is None and self.selected_bond_style != "chain":
                bond = self._bond_by_id(bond_id)
                if bond is not None:
                    if self.selected_bond_style in {"wedge", "hashed"}:
                        if bond.style == self.selected_bond_style:
                            bond.a1, bond.a2 = bond.a2, bond.a1
                        else:
                            bond.style = self.selected_bond_style
                    else:
                        bond.style = self._clicked_existing_bond_target_style(bond.style)
                    self.selected_bond_id = None
                    self.selected_bond_ids.discard(bond_id)
                    self.selected_atom_id = None
                    self._push_history(f"Bond changed to {bond.style}")
                    self.render_all()
                    return
            if atom_id is not None:
                start_atom = self._atom_by_id(atom_id)
                if start_atom is not None:
                    self.pending_draw_start_atom_id = atom_id
                    self.pending_draw_start_pos = (start_atom.x, start_atom.y)
                    self.pending_draw_mode = "bond_from_atom"
                    self.chain_drag_side_sign = None
                    if self.selected_bond_style == "chain":
                        self.preview_world_path = [self._default_growth_position(atom_id, "single")]
                        self.preview_world_pos = self.preview_world_path[-1]
                    else:
                        self.preview_world_pos = self._default_growth_position(atom_id, self.selected_bond_style)
                        self.preview_world_path = []
                    self.selected_atom_id = atom_id
                    self.selected_bond_id = None
                    self.render_all()
                    return
            self.pending_draw_start_atom_id = None
            self.pending_draw_start_pos = (wx, wy)
            self.pending_draw_mode = "bond_free"
            self.chain_drag_side_sign = None
            if self.selected_bond_style == "chain":
                self.preview_world_path = [(wx + STANDARD_BOND_LENGTH, wy)]
                self.preview_world_pos = self.preview_world_path[-1]
            else:
                self.preview_world_path = []
                self.preview_world_pos = (wx + STANDARD_BOND_LENGTH, wy)
            self.selected_atom_id = None
            self.selected_bond_id = None
            self.render_all()
            return

        if self.mode == "ring":
            self._update_ring_hover_preview()
            self.render_all()
            return

        if self.mode == "pan":
            moved_atom_ids = self._selected_fragment_atom_ids()
            if moved_atom_ids:
                anchor_atom_id = self._closest_atom_in_group(moved_atom_ids, wx, wy)
                anchor_atom = self._atom_by_id(anchor_atom_id) if anchor_atom_id is not None else None
                if anchor_atom is not None:
                    self.drag_group_ids = set(moved_atom_ids)
                    self.drag_group_origin = {
                        aid: (self._atom_by_id(aid).x, self._atom_by_id(aid).y)
                        for aid in self.drag_group_ids
                        if self._atom_by_id(aid) is not None
                    }
                    self.drag_atom_id = anchor_atom_id
                    self.drag_offset_world = (anchor_atom.x - wx, anchor_atom.y - wy)
                    self.drag_last_screen = (event.x, event.y)
                    self.is_panning = False
                    self.render_all()
                    return
            self.is_panning = True
            self.pan_anchor = (event.x, event.y)
            self.pan_origin = (self.offset_x, self.offset_y)

        if self.mode == "rotate":
            target_atoms = [a for a in self.atoms if a.id in self.selected_atom_ids] or list(self.atoms)
            if not target_atoms:
                return
            self.is_rotating = True
            xs = [a.x for a in target_atoms]
            ys = [a.y for a in target_atoms]
            self.rotation_center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
            cx, cy = self.world_to_screen(*self.rotation_center)
            self.rotation_anchor_angle = math.atan2(event.y - cy, event.x - cx)
            self.rotation_origin_atoms = {atom.id: (atom.x, atom.y) for atom in target_atoms}

    def on_left_drag(self, event: tk.Event[tk.Misc]) -> None:
        self.mouse_screen_pos = (event.x, event.y)
        wx, wy = self.screen_to_world(event.x, event.y)
        if self.mode == "pan" and self.is_panning and self.drag_atom_id is None:
            self.offset_x = self.pan_origin[0] + (event.x - self.pan_anchor[0])
            self.offset_y = self.pan_origin[1] + (event.y - self.pan_anchor[1])
            self.request_canvas_render()
            return

        if self.mode == "rotate" and self.is_rotating:
            cx, cy = self.world_to_screen(*self.rotation_center)
            current_angle = math.atan2(event.y - cy, event.x - cx)
            delta = self.rotation_anchor_angle - current_angle
            for atom_id, (ox, oy) in self.rotation_origin_atoms.items():
                atom = self._atom_by_id(atom_id)
                if atom is None:
                    continue
                rx, ry = ox - self.rotation_center[0], oy - self.rotation_center[1]
                nx, ny = self._rotate(rx, ry, delta)
                atom.x = self.rotation_center[0] + nx
                atom.y = self.rotation_center[1] + ny
            self.request_canvas_render()
            return

        if self.drag_atom_id is not None:
            anchor = self._atom_by_id(self.drag_atom_id)
            if anchor is not None:
                dx = (wx + self.drag_offset_world[0]) - anchor.x
                dy = (wy + self.drag_offset_world[1]) - anchor.y
                self.drag_preview_delta_world = (dx, dy)
                self._hide_drag_bridge_bonds()
                self._draw_drag_bridge_overlay()
                self._draw_drag_fuse_overlay()
                self.drag_last_screen = (event.x, event.y)
                self.request_canvas_render()
            return

        if self.mode == "ring":
            self._update_ring_hover_preview()
            self.request_canvas_render()
            return

        if self.erase_rect_start is not None and self.mode == "erase":
            self.erase_rect_end = (event.x, event.y)
            self._update_preview_rect_selection(self.erase_rect_start, self.erase_rect_end)
            self.request_canvas_render()
            return

        if self.selection_rect_start is not None and self.mode == "select":
            self.selection_rect_end = (event.x, event.y)
            self._update_preview_rect_selection(self.selection_rect_start, self.selection_rect_end)
            self.request_canvas_render()
            return

        if self.pending_draw_start_pos is not None:
            target_atom_id = self.find_atom_at(wx, wy)
            if self.pending_draw_mode == "bond_free":
                if self.selected_bond_style == "chain":
                    self.preview_world_path = self._chain_path_from_drag(None, wx, wy, self.pending_draw_start_pos)
                    self.preview_world_pos = self.preview_world_path[-1] if self.preview_world_path else None
                elif target_atom_id is not None:
                    target_atom = self._atom_by_id(target_atom_id)
                    if target_atom is not None:
                        self.preview_world_pos = (target_atom.x, target_atom.y)
                else:
                    sx, sy = self.pending_draw_start_pos
                    dx = wx - sx
                    dy = wy - sy
                    ndx, ndy = self._normalize(dx, dy)
                    self.preview_world_pos = (sx + STANDARD_BOND_LENGTH * ndx, sy + STANDARD_BOND_LENGTH * ndy)
            elif self.pending_draw_mode == "atom_from_atom":
                moved_px = math.dist(self.drag_start_screen, (event.x, event.y))
                if target_atom_id is not None and target_atom_id != self.pending_draw_start_atom_id:
                    target_atom = self._atom_by_id(target_atom_id)
                    if target_atom is not None:
                        self.preview_world_pos = (target_atom.x, target_atom.y)
                else:
                    if moved_px >= DRAG_THRESHOLD_PX:
                        self.preview_world_pos = self._free_bond_drag_position(self.pending_draw_start_atom_id or -1, wx, wy)
                    else:
                        self.preview_world_pos = None
            elif self.pending_draw_mode == "atom_free":
                sx, sy = self.pending_draw_start_pos
                dx = wx - sx
                dy = wy - sy
                ndx, ndy = self._normalize(dx, dy)
                self.preview_world_pos = (sx + STANDARD_BOND_LENGTH * ndx, sy + STANDARD_BOND_LENGTH * ndy)
            elif self.selected_bond_style == "chain":
                self.preview_world_path = self._chain_path_from_drag(self.pending_draw_start_atom_id, wx, wy)
                self.preview_world_pos = self.preview_world_path[-1] if self.preview_world_path else None
            elif target_atom_id is not None and target_atom_id != self.pending_draw_start_atom_id:
                target_atom = self._atom_by_id(target_atom_id)
                if target_atom is not None:
                    self.preview_world_pos = (target_atom.x, target_atom.y)
            else:
                self.preview_world_pos = self._free_bond_drag_position(self.pending_draw_start_atom_id, wx, wy)
            self.request_canvas_render()

    def on_left_up(self, event: tk.Event[tk.Misc]) -> None:
        self.left_button_down = False
        moved_px = math.dist(self.drag_start_screen, (event.x, event.y))

        if self.mode == "rotate" and self.is_rotating:
            self.is_rotating = False
            cx, cy = self.world_to_screen(*self.rotation_center)
            final_angle = math.atan2(event.y - cy, event.x - cx)
            delta_deg = math.degrees(self.rotation_anchor_angle - final_angle)
            self.total_rotation_deg += delta_deg
            self._push_history(f"Rotated to {self.total_rotation_deg:.1f}°")
            self.render_all()
            return

        if self.drag_atom_id is not None:
            dx, dy = self.drag_preview_delta_world
            if dx or dy:
                for aid in self.drag_group_ids or ({self.drag_atom_id} if self.drag_atom_id is not None else set()):
                    atom = self._atom_by_id(aid)
                    if atom is None:
                        continue
                    ox, oy = self.drag_group_origin.get(aid, (atom.x, atom.y))
                    atom.x = ox + dx
                    atom.y = oy + dy
            moved_ids = set(self.drag_group_ids) or ({self.drag_atom_id} if self.drag_atom_id is not None else set())
            self._clear_drag_fuse_overlay()
            self._clear_drag_bridge_overlay()
            self.drag_atom_id = None
            self.drag_group_ids = set()
            self.drag_group_origin = {}
            self.drag_preview_delta_world = (0.0, 0.0)
            self.drag_bridge_bond_ids = set()
            fused = self._fuse_moved_selection(moved_ids)
            self.preview_fuse_atom_ids = set()
            self.preview_fuse_bond_ids = set()
            self._push_history("Selection fused" if fused else "Atom moved")
            self.render_all()
            return

        if self.pending_ring_active:
            commit = dict(self.preview_ring_commit or {})
            self.pending_ring_active = False
            self.preview_ring_points = []
            self.preview_ring_bonds = []
            self.preview_ring_commit = None
            if commit.get("type") == "bond":
                self.insert_fused_ring_on_bond(int(commit["bond_id"]), tuple(commit["click_pos"]))
            elif commit.get("type") == "atom":
                self.insert_ring_on_atom(int(commit["atom_id"]), tuple(commit["cursor_pos"]))
            elif commit.get("type") == "free":
                anchor = tuple(commit.get("anchor", (0.0, 0.0)))
                self.insert_ring_at_point(float(anchor[0]), float(anchor[1]))
            else:
                self.render_all()
            return

        if self.erase_rect_start is not None and self.mode == "erase":
            x0, y0 = self.erase_rect_start
            x1, y1 = self.erase_rect_end or self.erase_rect_start
            self.erase_rect_start = None
            self.erase_rect_end = None
            if moved_px < DRAG_THRESHOLD_PX:
                if self.erase_hit_atom_id is not None:
                    self.remove_atom(self.erase_hit_atom_id)
                    self._push_history("Atom erased")
                elif self.erase_hit_bond_id is not None:
                    self.remove_bond(self.erase_hit_bond_id)
                    self._push_history("Bond erased")
                self.erase_hit_atom_id = None
                self.erase_hit_bond_id = None
                self.preview_rect_atom_ids = set()
                self.preview_rect_bond_ids = set()
                self.render_all()
                return
            sx0, sx1 = min(x0, x1), max(x0, x1)
            sy0, sy1 = min(y0, y1), max(y0, y1)
            atom_ids = {
                atom.id
                for atom in self.atoms
                if sx0 <= self.world_to_screen(atom.x, atom.y)[0] <= sx1
                and sy0 <= self.world_to_screen(atom.x, atom.y)[1] <= sy1
            }
            bond_ids = set()
            for bond in self.bonds:
                a1 = self._atom_by_id(bond.a1)
                a2 = self._atom_by_id(bond.a2)
                if not a1 or not a2:
                    continue
                p1 = self.world_to_screen(a1.x, a1.y)
                p2 = self.world_to_screen(a2.x, a2.y)
                if (sx0 <= p1[0] <= sx1 and sy0 <= p1[1] <= sy1) or (sx0 <= p2[0] <= sx1 and sy0 <= p2[1] <= sy1):
                    bond_ids.add(bond.id)
            if atom_ids or bond_ids:
                self.atoms = [atom for atom in self.atoms if atom.id not in atom_ids]
                self.bonds = [
                    bond for bond in self.bonds
                    if bond.id not in bond_ids and bond.a1 not in atom_ids and bond.a2 not in atom_ids
                ]
                self._renumber_r_group_labels()
                self._push_history("Area erased")
            self.erase_hit_atom_id = None
            self.erase_hit_bond_id = None
            self.preview_rect_atom_ids = set()
            self.preview_rect_bond_ids = set()
            self.render_all()
            return

        if self.selection_rect_start is not None and self.mode == "select":
            x0, y0 = self.selection_rect_start
            x1, y1 = self.selection_rect_end or self.selection_rect_start
            sx0, sx1 = min(x0, x1), max(x0, x1)
            sy0, sy1 = min(y0, y1), max(y0, y1)
            selected_atoms: set[int] = set()
            for atom in self.atoms:
                ax, ay = self.world_to_screen(atom.x, atom.y)
                if sx0 <= ax <= sx1 and sy0 <= ay <= sy1:
                    selected_atoms.add(atom.id)
            selected_bonds: set[int] = set()
            for bond in self.bonds:
                if bond.a1 in selected_atoms and bond.a2 in selected_atoms:
                    selected_bonds.add(bond.id)
            self.selected_atom_ids = selected_atoms
            self.selected_bond_ids = selected_bonds
            self.selected_atom_id = next(iter(selected_atoms), None)
            self.selected_bond_id = next(iter(selected_bonds), None) if not selected_atoms else None
            self.selection_rect_start = None
            self.selection_rect_end = None
            self.preview_rect_atom_ids = set()
            self.preview_rect_bond_ids = set()
            self.render_all()
            return

        if self.pending_draw_start_pos is not None and self.mode in {"bond", "atom"}:
            start_atom_id = self.pending_draw_start_atom_id
            end_wx, end_wy = self.screen_to_world(event.x, event.y)
            target_atom_id = self.find_atom_at(end_wx, end_wy)
            created = False

            if self.pending_draw_mode == "atom_from_atom" and start_atom_id is not None:
                if moved_px < DRAG_THRESHOLD_PX:
                    target_replace_id = target_atom_id if target_atom_id is not None else start_atom_id
                    target_atom = self._atom_by_id(target_replace_id)
                    if target_atom is not None:
                        if self.current_element() == "R":
                            target_atom.element = "*"
                            target_atom.label = self._next_r_group_label()
                        else:
                            target_atom.element = self.current_element()
                            target_atom.label = ""
                        target_atom.charge = self.current_charge
                        self.selected_atom_id = target_replace_id
                        created = True
                else:
                    if target_atom_id is not None and target_atom_id != start_atom_id:
                        self.add_or_update_bond(start_atom_id, target_atom_id, style=self._effective_draw_bond_style())
                        self.selected_atom_id = target_atom_id
                    else:
                        new_x, new_y = self._free_bond_drag_position(start_atom_id, end_wx, end_wy)
                        new_atom_id = self.add_atom(new_x, new_y, element=self.current_element(), charge=self.current_charge)
                        self.add_or_update_bond(start_atom_id, new_atom_id, style=self._effective_draw_bond_style())
                        self.selected_atom_id = new_atom_id
                    created = True
            elif self.pending_draw_mode == "atom_free":
                sx, sy = self.pending_draw_start_pos
                if moved_px < DRAG_THRESHOLD_PX:
                    new_atom_id = self.add_atom(sx, sy, element=self.current_element(), charge=self.current_charge)
                    self.selected_atom_id = new_atom_id
                else:
                    ex, ey = self.preview_world_pos if self.preview_world_pos is not None else (sx + STANDARD_BOND_LENGTH, sy)
                    start_new_id = self.add_atom(sx, sy, element=self.current_element(), charge=self.current_charge)
                    end_new_id = self.add_atom(ex, ey, element=self.current_element(), charge=self.current_charge)
                    self.add_or_update_bond(start_new_id, end_new_id, style=self._effective_draw_bond_style())
                    self.selected_atom_id = end_new_id
                created = True
            elif self.pending_draw_mode == "bond_free":
                sx, sy = self.pending_draw_start_pos
                if self.selected_bond_style == "chain":
                    start_new_id = self.add_atom(sx, sy, element=self.current_element())
                    chain_points = self.preview_world_path[:] if self.preview_world_path else [(sx + STANDARD_BOND_LENGTH, sy)]
                    current_id = start_new_id
                    for px, py in chain_points:
                        new_id = self.add_atom(px, py)
                        self.add_or_update_bond(current_id, new_id, style="single")
                        current_id = new_id
                    self.selected_atom_id = current_id
                else:
                    start_new_id = self.add_atom(sx, sy, element=self.current_element())
                    if target_atom_id is not None:
                        self.add_or_update_bond(start_new_id, target_atom_id)
                        self.selected_atom_id = target_atom_id
                    else:
                        if moved_px < DRAG_THRESHOLD_PX:
                            ex, ey = sx + STANDARD_BOND_LENGTH, sy
                        else:
                            ex, ey = self.preview_world_pos if self.preview_world_pos is not None else (sx + STANDARD_BOND_LENGTH, sy)
                        end_new_id = self.add_atom(ex, ey, element=self.current_element())
                        self.add_or_update_bond(start_new_id, end_new_id)
                        self.selected_atom_id = end_new_id
                created = True
            elif self.selected_bond_style == "chain" and start_atom_id is not None:
                chain_points = self.preview_world_path[:] if self.preview_world_path else [self._default_growth_position(start_atom_id)]
                current_id = start_atom_id
                for px, py in chain_points:
                    new_id = self.add_atom(px, py)
                    self.add_or_update_bond(current_id, new_id, style="single")
                    current_id = new_id
                self.selected_atom_id = current_id
                created = True
            elif start_atom_id is not None and target_atom_id is not None and target_atom_id != start_atom_id:
                self.add_or_update_bond(start_atom_id, target_atom_id)
                self.selected_atom_id = target_atom_id
                created = True
            else:
                previous_neighbor_ids = self._neighbor_ids(start_atom_id)[:] if start_atom_id is not None else []
                if start_atom_id is not None and moved_px < DRAG_THRESHOLD_PX:
                    new_x, new_y = self._default_growth_position(start_atom_id, self.selected_bond_style)
                    closure_atom_id = self._find_existing_atom_near(
                        new_x,
                        new_y,
                        exclude_ids={start_atom_id},
                    )
                    if closure_atom_id is not None:
                        self.add_or_update_bond(start_atom_id, closure_atom_id)
                        self.selected_atom_id = closure_atom_id
                        created = True
                        self.pending_draw_start_atom_id = None
                        self.pending_draw_start_pos = None
                        self.pending_draw_mode = None
                        self.preview_world_pos = None
                        self.preview_world_path = []
                        self.selected_atom_id = None
                        self.selected_bond_id = None
                        self._push_history("Structure updated")
                        self.render_all()
                        self.is_panning = False
                        return
                else:
                    if start_atom_id is not None:
                        new_x, new_y = self._free_bond_drag_position(start_atom_id, end_wx, end_wy)
                    else:
                        new_x, new_y = self.preview_world_pos if self.preview_world_pos is not None else (end_wx, end_wy)
                new_atom_id = self.add_atom(new_x, new_y)
                if start_atom_id is not None:
                    self.add_or_update_bond(start_atom_id, new_atom_id, style=self._effective_draw_bond_style())
                    if moved_px < DRAG_THRESHOLD_PX and self._effective_draw_bond_style() == "single":
                        self._relax_primary_neighbors_after_growth(start_atom_id, new_atom_id, previous_neighbor_ids)
                self.selected_atom_id = new_atom_id
                created = True

            self.pending_draw_start_atom_id = None
            self.pending_draw_start_pos = None
            self.pending_draw_mode = None
            self.chain_drag_side_sign = None
            self.preview_world_pos = None
            self.preview_world_path = []
            if created:
                self.selected_atom_id = None
                self.selected_bond_id = None
                self._push_history("Structure updated")
            self.render_all()

        self.is_panning = False

    def on_mouse_move(self, event: tk.Event[tk.Misc]) -> None:
        if self.canvas is None:
            return
        previous_mouse_pos = self.mouse_screen_pos
        self.mouse_screen_pos = (event.x, event.y)
        if self.mode == "ring":
            self._update_ring_hover_preview()
            self.request_canvas_render()
            return
        if self.drag_atom_id is not None or self.is_rotating or self.is_panning or self.pending_ring_active or self.pending_draw_start_pos is not None:
            self.request_canvas_render()
            return
        wx, wy = self.screen_to_world(event.x, event.y)
        atom_id, bond_id = self._pick_hit_targets(wx, wy)
        if atom_id != self.hover_atom_id or bond_id != self.hover_bond_id:
            self.hover_atom_id = atom_id
            self.hover_bond_id = bond_id
            self.request_canvas_render()
            return
        if self._cursor_overlay_active() and previous_mouse_pos != self.mouse_screen_pos:
            self.request_canvas_render()

    def on_mouse_leave(self, _event: tk.Event[tk.Misc]) -> None:
        self.mouse_screen_pos = None
        if self.mode == "ring":
            self._update_ring_hover_preview()
        if self.hover_atom_id is None and self.hover_bond_id is None:
            if self.mode == "ring":
                self.request_canvas_render()
            return
        self.hover_atom_id = None
        self.hover_bond_id = None
        self.request_canvas_render()

    def on_middle_down(self, event: tk.Event[tk.Misc]) -> None:
        self.is_panning = True
        self.pan_anchor = (event.x, event.y)
        self.pan_origin = (self.offset_x, self.offset_y)

    def on_middle_drag(self, event: tk.Event[tk.Misc]) -> None:
        if not self.is_panning:
            return
        self.offset_x = self.pan_origin[0] + (event.x - self.pan_anchor[0])
        self.offset_y = self.pan_origin[1] + (event.y - self.pan_anchor[1])
        self.render_all()

    def on_middle_up(self, _event: tk.Event[tk.Misc]) -> None:
        self.is_panning = False

    def on_mousewheel(self, event: tk.Event[tk.Misc]) -> None:
        self.zoom_at(event.x, event.y, 1.12 if event.delta > 0 else 1 / 1.12)

    def on_mousewheel_linux(self, event: tk.Event[tk.Misc]) -> None:
        if event.num == 4:
            self.zoom_at(event.x, event.y, 1.12)
        elif event.num == 5:
            self.zoom_at(event.x, event.y, 1 / 1.12)

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        world_before = self.screen_to_world(sx, sy)
        self.scale = max(4.0, min(5000.0, self.scale * factor))
        self.offset_x = sx - world_before[0] * self.scale
        self.offset_y = sy + world_before[1] * self.scale
        self.render_all()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = MoleculeSketcherApp()
    app.run()


if __name__ == "__main__":
    main()
