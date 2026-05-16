"""
===========
launcher.py
===========

Lightweight startup launcher for SARgate with splash screen.

This script displays a Dear PyGui splash window while synchronously pre-loading
all major Python modules (NumPy, pandas, SciPy, scikit-learn, RDKit, Plotly,
etc.) before the main GUI starts. It ensures DPI-aware rendering on all systems,
auto-detects asset paths for both development and PyInstaller bundles, and
finally launches `app/main.py` as the application core.

The final handoff to the main GUI always happens in a fresh child process. This
avoids native instability caused by tearing down the splash Dear PyGui context
and then bootstrapping the full application inside the same interpreter
session, while also reducing the visible gap between the splash shutdown and
the main window startup.
"""

#   1.1: _bundle_base_dir() -> robust base directory for bundled assets
#   1.2: _get_screen_size() -> detect primary screen size across platforms
#
#
#
#   4.1: Enable Windows Per-Monitor DPI awareness
#   4.2: Resolve bundled image path
#   4.3: Load splash image (Pillow first, then DearPyGui fallback)
#   4.4: Create viewport and theming
#   4.5: Layout: image, progress bar, centred text
#   4.6: Centre viewport and draw first frame
#
#
#
#
#   8.1: Ensure base dir on sys.path
#   8.2: Build splash and preload synchronously
#   8.3: Launch main module in a fresh child process when possible
#   8.4: Wait for the main-process readiness signal
#   8.5: Exit with error if preload failed


# === IMPORTS (lightweight) ===
import os, sys, time, importlib, runpy, subprocess, shlex, tempfile, json, ctypes, ctypes.util, multiprocessing, urllib.parse
from collections.abc import Sequence
import dearpygui.dearpygui as dpg


MAIN_SKIP_SPLASH_ENV = "SARGATE_SKIP_SPLASH"
MAIN_READY_FILE_ENV = "SARGATE_READY_FILE"
MAIN_STARTUP_FILES_ENV = "SARGATE_STARTUP_FILES"
SUPPORTED_INPUT_EXTENSIONS = {".sdf", ".csv", ".tsv", ".xlsx", ".smi", ".txt"}


# ---------- Helpers cross-platform ----------
def _bundle_base_dir() -> str:
    """Return the 'base dir' to load bundled assets (splash_img.png, etc.)
    robustly for:
      - PyInstaller onedir/onefile (uses sys._MEIPASS if present)
      - Linux/macOS/Windows onedir: dist/SARgate/_internal next to the executable
      - macOS .app: Contents/Resources
      - Development: folder of this file
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return meipass

    exe = os.path.abspath(sys.executable)
    exe_dir = os.path.dirname(exe)

    internal = os.path.join(exe_dir, "_internal")
    if os.path.isdir(internal):
        return internal

    if sys.platform == "darwin":
        # e.g. /path/App.app/Contents/MacOS/SARgate  -> Resources
        parts = exe_dir.split(os.sep)
        if ".app" in " ".join(parts):
            try:
                app_root = exe_dir[:exe_dir.index(".app") + 4]  # include ".app"
                resources = os.path.join(app_root, "Contents", "Resources")
                if os.path.isdir(resources):
                    return resources
            except Exception:
                pass

    return os.path.dirname(os.path.abspath(__file__))


def _get_screen_size():
    """Determine the primary screen size with multiple fallbacks.

    Order of attempts:
      1) screeninfo (if installed)
      2) Windows via ctypes
      3) Linux via xrandr (if DISPLAY is set)
      4) Cross-platform tkinter
      5) Final fallback: 1920x1080
    """
    try:
        from screeninfo import get_monitors
        mons = get_monitors()
        if mons:
            m = next((mm for mm in mons if getattr(mm, "is_primary", True)), mons[0])
            return m.width, m.height
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            pass

    if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        try:
            out = subprocess.check_output(shlex.split("xrandr --current"), stderr=subprocess.DEVNULL, text=True)
            # look for "*" which marks the current mode of the primary output
            for line in out.splitlines():
                if "*" in line:
                    # e.g.: "   1920x1080     60.00*+  59.94"
                    parts = line.strip().split()
                    if "x" in parts[0]:
                        w, h = parts[0].split("x")
                        return int(w), int(h)
        except Exception:
            pass

    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        w = root.winfo_screenwidth(); h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def _apply_macos_viewport_rounding(radius: float) -> None:
    """
    Apply native rounded corners to the macOS splash viewport.

    Args:
        radius (float): Corner radius in screen points.

    Returns:
        None: This helper updates the native Cocoa window in place when possible.
    """
    if sys.platform != "darwin":
        return

    try:
        objc_path = ctypes.util.find_library("objc")
        if not objc_path:
            return

        objc = ctypes.cdll.LoadLibrary(objc_path)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_msgSend.restype = ctypes.c_void_p

        def _sel(name: str) -> ctypes.c_void_p:
            return objc.sel_registerName(name.encode("utf-8"))

        def _msg(
            obj: int,
            selector: str,
            *args: object,
            restype: object = ctypes.c_void_p,
            argtypes: Sequence[object] | None = None,
        ) -> object:
            objc.objc_msgSend.restype = restype
            objc.objc_msgSend.argtypes = argtypes or [ctypes.c_void_p, ctypes.c_void_p]
            return objc.objc_msgSend(obj, _sel(selector), *args)

        ns_app = _msg(objc.objc_getClass(b"NSApplication"), "sharedApplication")
        window = _msg(ns_app, "mainWindow")
        if not window:
            window = _msg(ns_app, "keyWindow")
        if not window:
            windows = _msg(ns_app, "windows")
            if windows:
                count = _msg(windows, "count", restype=ctypes.c_ulonglong)
                if count:
                    window = _msg(
                        windows,
                        "objectAtIndex:",
                        0,
                        restype=ctypes.c_void_p,
                        argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong],
                    )
        if not window:
            return

        clear_color = _msg(objc.objc_getClass(b"NSColor"), "clearColor")
        _msg(
            window,
            "setOpaque:",
            False,
            restype=None,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool],
        )
        _msg(
            window,
            "setBackgroundColor:",
            clear_color,
            restype=None,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        )

        content_view = _msg(window, "contentView")
        if not content_view:
            return
        for view in (content_view, _msg(content_view, "superview")):
            if not view:
                continue
            _msg(
                view,
                "setWantsLayer:",
                True,
                restype=None,
                argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool],
            )
            layer = _msg(view, "layer")
            if not layer:
                continue
            _msg(
                layer,
                "setMasksToBounds:",
                True,
                restype=None,
                argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool],
            )
            _msg(
                layer,
                "setCornerRadius:",
                float(radius),
                restype=None,
                argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double],
            )
    except Exception:
        pass


def _apply_windows_viewport_rounding(radius: float, width: int, height: int) -> None:
    """
    Apply native rounded corners to the splash viewport on Windows.

    Args:
        radius (float): Corner radius in logical pixels.
        width (int): Viewport width.
        height (int): Viewport height.

    Returns:
        None: The native window region is updated in place when possible.
    """
    if sys.platform != "win32":
        return

    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        hwnd = user32.FindWindowW(None, "SARgate")
        if not hwnd:
            hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return

        diameter = max(2, int(radius * 2))
        region = gdi32.CreateRoundRectRgn(0, 0, int(width) + 1, int(height) + 1, diameter, diameter)
        if not region:
            return
        user32.SetWindowRgn(hwnd, region, True)
    except Exception:
        pass


def _apply_linux_viewport_rounding(radius: float, width: int, height: int) -> None:
    """
    Apply best-effort rounded corners to the splash viewport on X11 Linux.

    Args:
        radius (float): Corner radius in pixels.
        width (int): Viewport width.
        height (int): Viewport height.

    Returns:
        None: The window bounding region is updated in place when possible.
    """
    if not sys.platform.startswith("linux") or not os.environ.get("DISPLAY"):
        return

    class XRectangle(ctypes.Structure):
        _fields_ = [
            ("x", ctypes.c_short),
            ("y", ctypes.c_short),
            ("width", ctypes.c_ushort),
            ("height", ctypes.c_ushort),
        ]

    try:
        x11_path = ctypes.util.find_library("X11")
        xext_path = ctypes.util.find_library("Xext")
        if not x11_path or not xext_path:
            return

        x11 = ctypes.cdll.LoadLibrary(x11_path)
        xext = ctypes.cdll.LoadLibrary(xext_path)

        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x11.XInternAtom.restype = ctypes.c_ulong
        x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
        x11.XGetWindowProperty.restype = ctypes.c_int
        x11.XGetWindowProperty.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long, ctypes.c_long,
            ctypes.c_bool, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_void_p),
        ]
        x11.XFree.restype = ctypes.c_int
        x11.XFree.argtypes = [ctypes.c_void_p]
        x11.XCreateRegion.restype = ctypes.c_void_p
        x11.XUnionRectWithRegion.restype = ctypes.c_int
        x11.XUnionRectWithRegion.argtypes = [ctypes.POINTER(XRectangle), ctypes.c_void_p, ctypes.c_void_p]
        x11.XDestroyRegion.restype = ctypes.c_int
        x11.XDestroyRegion.argtypes = [ctypes.c_void_p]
        x11.XFlush.restype = ctypes.c_int
        x11.XFlush.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.restype = ctypes.c_int
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        xext.XShapeCombineRegion.restype = None
        xext.XShapeCombineRegion.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

        display = x11.XOpenDisplay(None)
        if not display:
            return

        root = x11.XDefaultRootWindow(display)
        atom_client_list = x11.XInternAtom(display, b"_NET_CLIENT_LIST", False)
        atom_wm_pid = x11.XInternAtom(display, b"_NET_WM_PID", False)
        atom_cardinal = x11.XInternAtom(display, b"CARDINAL", False)

        actual_type = ctypes.c_ulong()
        actual_format = ctypes.c_int()
        nitems = ctypes.c_ulong()
        bytes_after = ctypes.c_ulong()
        prop = ctypes.c_void_p()

        window_id = None
        if x11.XGetWindowProperty(
            display, root, atom_client_list, 0, 4096, False, 0,
            ctypes.byref(actual_type), ctypes.byref(actual_format),
            ctypes.byref(nitems), ctypes.byref(bytes_after), ctypes.byref(prop)
        ) == 0 and prop.value:
            windows = ctypes.cast(prop, ctypes.POINTER(ctypes.c_ulong))
            for i in range(int(nitems.value)):
                candidate = windows[i]
                pid_type = ctypes.c_ulong()
                pid_format = ctypes.c_int()
                pid_items = ctypes.c_ulong()
                pid_after = ctypes.c_ulong()
                pid_prop = ctypes.c_void_p()
                ok = x11.XGetWindowProperty(
                    display, candidate, atom_wm_pid, 0, 1, False, atom_cardinal,
                    ctypes.byref(pid_type), ctypes.byref(pid_format),
                    ctypes.byref(pid_items), ctypes.byref(pid_after), ctypes.byref(pid_prop)
                ) == 0
                if ok and pid_prop.value:
                    pid_ptr = ctypes.cast(pid_prop, ctypes.POINTER(ctypes.c_ulong))
                    if int(pid_ptr[0]) == os.getpid():
                        window_id = candidate
                        x11.XFree(pid_prop)
                        break
                    x11.XFree(pid_prop)
            x11.XFree(prop)

        if not window_id:
            x11.XCloseDisplay(display)
            return

        region = x11.XCreateRegion()
        if not region:
            x11.XCloseDisplay(display)
            return

        r = max(4, int(radius))
        rects = [
            XRectangle(r, 0, max(1, int(width) - 2 * r), int(height)),
            XRectangle(0, r, int(width), max(1, int(height) - 2 * r)),
            XRectangle(int(r * 0.35), int(r * 0.35), max(1, int(width) - int(r * 0.7) * 2), max(1, int(height) - int(r * 0.7) * 2)),
        ]
        for rect in rects:
            x11.XUnionRectWithRegion(ctypes.byref(rect), region, region)

        ShapeBounding = 0
        ShapeSet = 0
        xext.XShapeCombineRegion(display, window_id, ShapeBounding, 0, 0, region, ShapeSet)
        x11.XFlush(display)
        x11.XDestroyRegion(region)
        x11.XCloseDisplay(display)
    except Exception:
        pass


def _apply_native_viewport_rounding(radius: float, width: int, height: int) -> None:
    """
    Apply rounded corners to the splash viewport using platform-native APIs.

    Args:
        radius (float): Corner radius in screen pixels/points.
        width (int): Viewport width.
        height (int): Viewport height.

    Returns:
        None: Platform-specific helpers update the splash window in place.
    """
    if sys.platform == "darwin":
        _apply_macos_viewport_rounding(radius)
    elif sys.platform == "win32":
        _apply_windows_viewport_rounding(radius, width, height)
    elif sys.platform.startswith("linux"):
        _apply_linux_viewport_rounding(radius, width, height)


# === CONFIG ===
APP_MODULE = "app.main"   # application entry module
WIDTH, HEIGHT = 720, 405
X_POS, Y_POS = 400, 300  # initial position (overridden in build_splash)

STEPS = [
    ("Importing NumPy",                   ["numpy"]),
    ("Importing pandas",                  ["pandas"]),
    ("Importing SciPy",                   ["scipy.cluster.hierarchy"]),
    ("Importing scikit-learn",            ["sklearn.decomposition", "sklearn.cluster"]),
    ("Importing plotting libs",           ["plotly.graph_objects"]),
    ("Importing graph/image/pdf",         ["networkx", "PIL.Image", "reportlab.pdfgen", "reportlab.lib.pagesizes"]),
    ("Importing web/utils...",            ["requests", "webbrowser", "urllib.parse", "screeninfo"]),
    ("Importing RDKit...",                ["rdkit", "rdkit.Chem", "rdkit.Chem.AllChem", "rdkit.Chem.Draw",
                                           "rdkit.Chem.Descriptors", "rdkit.Chem.rdFingerprintGenerator", "rdkit.Chem.MolStandardize"]),
]


# === SHARED STATE (only for UI) ===
state = {
    "progress": 0.0,
    "message": "Starting...",
}


# === BUILD SPLASH UI ===
def build_splash():
    """Create a crisp, consistent 720x405 splash on macOS/Windows/Linux.

    Behaviour:
      - Load splash_img.png (from the bundle, wherever it lives)
      - Resize to 720x405
      - Force alpha=1.0
      - Create DearPyGui float texture [0..1] and draw 1:1
      - On Windows, enable Per-Monitor DPI Awareness
    """
    def _enable_windows_per_monitor_dpi_v2():
        if sys.platform != "win32":
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    TARGET_W, TARGET_H = 720, 405
    GAP_ABOVE_BAR, GAP_BELOW_BAR = 10, 10
    BAR_H, MARGIN_X = 18, 10

    _enable_windows_per_monitor_dpi_v2()
    dpg.create_context()

    base_dir = _bundle_base_dir()
    img_path = os.path.join(base_dir, "assets", "icons", "splash_img.png")

    tex_tag, loaded = None, False
    try:
        from PIL import Image
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        im = Image.open(img_path).convert("RGBA")
        if im.size != (TARGET_W, TARGET_H):
            im = im.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        r, g, b, a = im.split()
        alpha_full = Image.new("L", im.size, color=255)
        im = Image.merge("RGBA", (r, g, b, alpha_full))
        raw = im.tobytes()
        inv255 = 1.0 / 255.0
        rgba_float = [byte * inv255 for byte in raw]
        with dpg.texture_registry(show=False):
            tex_tag = "splash_bg_tex"
            dpg.add_static_texture(TARGET_W, TARGET_H, rgba_float, tag=tex_tag)
        loaded = True
    except Exception as e:
        print(f"[Splash] Pillow failed ({type(e).__name__}: {e}). Trying dpg.load_image fallback.")
        if os.path.exists(img_path):
            try:
                w, h, c, data = dpg.load_image(img_path)
                if c == 3:
                    data_fixed = []
                    for i in range(0, len(data), 3):
                        data_fixed.extend((data[i], data[i+1], data[i+2], 1.0))
                    data = data_fixed
                else:
                    for i in range(3, len(data), 4):
                        data[i] = 1.0
                with dpg.texture_registry(show=False):
                    tex_tag = "splash_bg_tex"
                    dpg.add_static_texture(w, h, data, tag=tex_tag)
                loaded = True
            except Exception as ee:
                print(f"[Splash] dpg.load_image failed ({type(ee).__name__}: {ee}). No image.")
                tex_tag = None

    WIN_W = TARGET_W
    WIN_H = TARGET_H + GAP_ABOVE_BAR + BAR_H + GAP_BELOW_BAR
    BAR_W = WIN_W - 2 * MARGIN_X
    BAR_X, BAR_Y = MARGIN_X, TARGET_H + GAP_ABOVE_BAR

    state["_BAR_W"] = BAR_W
    state["_BAR_H"] = BAR_H
    state["_DRAW_PAD"] = 8

    dpg.create_viewport(title="SARgate", resizable=False, decorated=False, width=WIN_W, height=WIN_H)

    with dpg.theme(tag="splash_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)

    with dpg.window(tag="splash_window",
                    no_title_bar=True, no_move=True, no_resize=True, no_collapse=True,
                    no_background=False, no_scrollbar=True,
                    width=WIN_W, height=WIN_H, pos=[0, 0], modal=True):
        dpg.bind_item_theme("splash_window", "splash_theme")
        if tex_tag and loaded:
            dpg.add_image(tex_tag)
        elif tex_tag:
            dpg.add_image(tex_tag, width=TARGET_W, height=TARGET_H)
        else:
            dpg.add_spacer(width=TARGET_W, height=TARGET_H)

        dpg.add_progress_bar(tag="splash_bar", default_value=0.0,
                             width=BAR_W, height=BAR_H, pos=(BAR_X, BAR_Y))
        with dpg.theme() as progress_bar_theme:
            with dpg.theme_component(dpg.mvProgressBar):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (86, 149, 175, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (22, 26, 32, 220))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 7)
        dpg.bind_item_theme("splash_bar", progress_bar_theme)
    
        with dpg.drawlist(width=BAR_W, height=BAR_H + state["_DRAW_PAD"],
                          pos=(BAR_X, BAR_Y - state["_DRAW_PAD"] // 2)):
            dpg.draw_text((BAR_W // 2, (BAR_H + state["_DRAW_PAD"]) // 2),
                          "", tag="splash_text", size=14,
                          color=(255, 255, 255, 255))

    

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.render_dearpygui_frame()
    dpg.render_dearpygui_frame()
    dpg.render_dearpygui_frame()
    try:
        sw, sh = _get_screen_size()
        vx = max(0, (sw - WIN_W) // 2)
        vy = max(0, (sh - WIN_H) // 2)
        dpg.set_primary_window("splash_window", True)
        dpg.set_viewport_pos([vx, vy])
    except Exception:
        pass

    dpg.render_dearpygui_frame()



def ui_update(message: str, progress: float):
    """Update the progress bar and centred status text, then force a repaint.

    Args:
        message (str): Status message to display.
        progress (float): Progress value in [0.0, 1.0].
    """
    msg = str(message).replace("\n", " ")
    state["message"] = msg
    state["progress"] = max(0.0, min(1.0, progress))

    dpg.set_value("splash_bar", state["progress"])

    BAR_W = state.get("_BAR_W", WIDTH - 20)
    BAR_H = state.get("_BAR_H", 24)
    DRAW_PAD = state.get("_DRAW_PAD", 8)

    tw, th = dpg.get_text_size(state["message"])
    tx = (BAR_W - tw) // 2
    ty = ((BAR_H + DRAW_PAD) - th) // 2 + 5



    dpg.configure_item("splash_text", text=state["message"], pos=(tx, ty))

    dpg.render_dearpygui_frame()
    dpg.render_dearpygui_frame()



def preload_sync():
    """Perform synchronous imports with UI progress updates (no threads).

    Returns:
        tuple[bool, Exception|None]: (success, exception if any)
    """
    total = sum(len(mods) for _, mods in STEPS)
    done = 0

    tot_steps = sum(len(mods) for _, mods in STEPS)
    percent_per_step = 1.0 / max(tot_steps, 1)
    percent = 0.0

    try:
        for label, mods in STEPS:
            ui_update(label, done / total if total else 0.0)
            for m in mods:
                percent += percent_per_step
                ui_update(f"{percent:.0%} | {label}  ->  {m}", done / total if total else 0.0)
                importlib.import_module(m)
                done += 1
                ui_update(label, done / total)

        ui_update("Ready", 1.0)
        time.sleep(1)
        return True, None

    except Exception as e:
        ui_update(f"Error while importing:\n{type(e).__name__}: {e}", done / total if total else 0.0)
        time.sleep(1)
        return False, e



def cleanup_splash():
    """Close and destroy the Dear PyGui context used by the splash."""
    try:
        if dpg.does_item_exist("splash_window"):
            dpg.delete_item("splash_window")
    except Exception:
        pass

    try:
        dpg.destroy_context()
    except Exception:
        pass


def _coerce_startup_file_path(value: str) -> str:
    """
    Normalize a file argument received from macOS, Windows, Linux, or PyInstaller.
    """
    raw = str(value or "").strip().strip("\x00")
    if not raw or raw.startswith("-psn_"):
        return ""
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]
    if raw.startswith("file://"):
        parsed = urllib.parse.urlparse(raw)
        raw = urllib.parse.unquote(parsed.path or "")
    else:
        raw = urllib.parse.unquote(raw)
    return os.path.abspath(os.path.expanduser(raw)) if raw else ""


def _startup_input_files(args: Sequence[str]) -> list[str]:
    """
    Return existing input files passed to SARgate by the operating system.
    """
    input_files: list[str] = []
    for arg in args:
        path = _coerce_startup_file_path(str(arg))
        ext = os.path.splitext(path)[1].lower()
        if ext in SUPPORTED_INPUT_EXTENSIONS and os.path.isfile(path):
            input_files.append(path)
    return input_files


def _launch_main_fresh(
    base_dir: str,
    startup_files: Sequence[str] = (),
    raw_startup_args: Sequence[str] = (),
) -> subprocess.Popen[str]:
    """
    Launch the main GUI in a fresh child process.

    Args:
        base_dir (str): Project root directory that contains `app/main.py` during
            source-based runs.

    Returns:
        subprocess.Popen[str]: Child process handle for the launched app.
    """
    ready_fd, ready_path = tempfile.mkstemp(prefix="sargate_ready_", suffix=".flag")
    os.close(ready_fd)
    try:
        os.unlink(ready_path)
    except FileNotFoundError:
        pass

    child_env = os.environ.copy()
    child_env[MAIN_READY_FILE_ENV] = ready_path
    child_env[MAIN_SKIP_SPLASH_ENV] = "1"
    child_env[MAIN_STARTUP_FILES_ENV] = json.dumps([*raw_startup_args, *startup_files])

    if getattr(sys, "frozen", False):
        command = [sys.executable, *startup_files]
        child_cwd = os.path.dirname(sys.executable)
    else:
        main_path = os.path.join(base_dir, "app", "main.py")
        if not os.path.exists(main_path):
            raise FileNotFoundError(main_path)
        command = [sys.executable, main_path, *startup_files]
        child_cwd = base_dir

    process = subprocess.Popen(command, cwd=child_cwd, env=child_env)
    process._sargate_ready_path = ready_path  # type: ignore[attr-defined]
    return process


def _wait_for_main_ready(ready_path: str, timeout_s: float = 6.0) -> None:
    """
    Keep repainting the splash until the main process reports readiness.

    Args:
        ready_path (str): Temporary file path touched by `app/main.py` when the
            main interface is ready to take over.
        timeout_s (float, optional): Maximum wait time before the splash is
            dismissed anyway. Defaults to `6.0`.

    Returns:
        None: This helper keeps the splash responsive during handoff.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(ready_path):
            break
        dpg.render_dearpygui_frame()
        time.sleep(0.01)



# === MAIN (launcher) ===
if __name__ == "__main__":
    multiprocessing.freeze_support()
    raw_startup_args = list(sys.argv[1:])
    startup_files = _startup_input_files(raw_startup_args)

    if len(sys.argv) >= 2 and sys.argv[1] == "--sketcher-helper":
        from app.analysis.tools.molecule_sketcher import main as sketcher_main

        sketcher_main()
        sys.exit(0)

    if len(sys.argv) >= 3 and sys.argv[1] == "--tk-helper":
        from app.utils.native_dialogs import _run_tk_helper_main

        sys.exit(_run_tk_helper_main(json.loads(sys.argv[2])))

    if os.environ.get(MAIN_SKIP_SPLASH_ENV) == "1":
        runpy.run_module("app.main", run_name="__main__")
        sys.exit(0)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    build_splash()
    ok, err = preload_sync()

    if ok:
        try:
            ui_update("Launching SARgate...", 1.0)
            child = _launch_main_fresh(base_dir, startup_files, raw_startup_args)
            ready_path = getattr(child, "_sargate_ready_path", "")
            if ready_path:
                _wait_for_main_ready(ready_path)
                try:
                    os.unlink(ready_path)
                except FileNotFoundError:
                    pass
            cleanup_splash()
            sys.exit(0)
        except Exception:
            cleanup_splash()
            raise
    else:
        cleanup_splash()
        sys.exit(1)
