"""
=================
viewport_size.py
=================

Viewport sizing helpers for the SARgate interface.

This module detects the available screen size, creates and stabilizes the Dear
PyGui viewport, and keeps the main window aligned with the live viewport
geometry during resize events. The stored design reference dimensions are used
to keep layout calculations consistent across the application.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Capture screen metrics
# 3. Create and stabilize the viewport
# 4. Create and synchronize the main window

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import sys
import dearpygui.dearpygui as dpg
from typing import Any
from screeninfo import get_monitors
from app.utils.callbacks import update_responsive_images


# -----------------------------------------------------------------------------
# 2. Capture screen metrics
# -----------------------------------------------------------------------------
def get_screen_size(state: dict[str, Any]) -> None:
    """
    Detect the primary monitor size and store derived layout metrics.

    Args:
        state (dict[str, Any]): Shared application state updated with screen
            dimensions and the base window spacer value.

    Returns:
        None: This function updates the provided state dictionary in place.
    """
    try:
        if get_monitors is not None:
            mon = get_monitors()[0]
            sw, sh = int(mon.width), int(mon.height)
        else:
            sw, sh = 1920, 1080
    except Exception:
        sw, sh = 1920, 1080

    # --- STEP 2.1: Store screen dimensions ---
    state["screen_width"] = sw
    state["screen_height"] = sh

    # --- STEP 2.2: Compute the base window spacer ---
    state["win_spacer"] = int(sw / 288)


# -----------------------------------------------------------------------------
# 3. Create and stabilize the viewport
# -----------------------------------------------------------------------------
def setup_viewport(state: dict[str, Any], home_dir: str, title: str = "SARgate") -> None:
    """
    Create the Dear PyGui viewport and wait for stable client dimensions.

    Args:
        state (dict[str, Any]): Shared application state updated with viewport
            readiness flags and design reference dimensions.
        home_dir (str): Project root directory used to resolve viewport icon
            assets.
        title (str, optional): Window title shown in the viewport. Defaults to
            `"SARgate"`.

    Returns:
        None: This function creates the viewport and updates state in place.
    """

    # -----------------------------------------------------------------------------
    # 3.1. Start viewport stable watch
    # -----------------------------------------------------------------------------
    def _start_viewport_stable_watch(state: dict[str, Any], n_same_frames: int = 10) -> None:
        """
        Monitor viewport size until the maximized client area becomes stable.

        Args:
            state (dict[str, Any]): Shared application state updated with the
                latest viewport metrics.
            n_same_frames (int, optional): Number of identical consecutive
                frames required before the viewport size is considered stable.
                Defaults to `10`.

        Returns:
            None: This helper schedules frame callbacks and updates state in
            place.
        """
        state["_stable_counter"] = 0
        state["_last_vw"] = None
        state["_last_vh"] = None

        # -----------------------------------------------------------------------------
        # 3.1.1. Probe
        # -----------------------------------------------------------------------------
        def _probe(_: Any = None) -> None:
            """
            Sample the viewport size and freeze the design reference when ready.

            Args:
                _ (Any, optional): Unused frame-callback payload accepted for
                    Dear PyGui compatibility.

            Returns:
                None: This callback updates viewport state in place.
            """
            vw = dpg.get_viewport_client_width() or dpg.get_viewport_width()
            vh = dpg.get_viewport_client_height() or dpg.get_viewport_height()

            # --- STEP 3.1.1: Compare with the previous frame ---
            if state["_last_vw"] is None:
                state["_last_vw"], state["_last_vh"] = vw, vh
                dpg.set_frame_callback(dpg.get_frame_count() + 1, _probe)
                return

            if vw == state["_last_vw"] and vh == state["_last_vh"]:
                # --- STEP 3.1.2: Count consecutive identical frames ---
                state["_stable_counter"] += 1
            else:
                state["_stable_counter"] = 0
                state["_last_vw"], state["_last_vh"] = vw, vh

            # --- STEP 3.1.3: Freeze design reference dimensions when stable ---
            if state["_stable_counter"] >= n_same_frames:
                state["design_ref_width"] = int(vw)
                state["design_ref_height"] = int(vh)
                state["_viewport_ready"] = True

                # --- STEP 3.1.4: Align the main window if it already exists ---
                if dpg.does_item_exist("main_window"):
                    dpg.configure_item("main_window", width=int(vw), height=int(vh))
                return

            dpg.set_frame_callback(dpg.get_frame_count() + 1, _probe)

        dpg.set_frame_callback(dpg.get_frame_count() + 1, _probe)

    starting_dim = (1280, 800)
    icon_path = os.path.join(home_dir, "assets", "icons", "SARgate_icon.ico")
    viewport_kwargs = {
        "title": title,
        "width": starting_dim[0],
        "height": starting_dim[1],
        "resizable": True,
        "clear_color": state.get("theme", (0, 0, 0, 255)),
        "x_pos": 50,
        "y_pos": 0,
    }
    if sys.platform == "win32" and os.path.exists(icon_path):
        viewport_kwargs["small_icon"] = icon_path
        viewport_kwargs["large_icon"] = icon_path

    # --- STEP 3.1: Create the viewport ---
    dpg.create_viewport(**viewport_kwargs)

    # --- STEP 3.2: Set up and show the viewport ---
    dpg.setup_dearpygui()
    dpg.configure_app()
    dpg.show_viewport()

    # --- STEP 3.3: Maximize the viewport ---
    dpg.maximize_viewport()

    # --- STEP 3.4: Initialize viewport stability flags ---
    state["_viewport_ready"] = False
    state["_stable_counter"] = 0
    state["_last_vw"] = None
    state["_last_vh"] = None

    # --- STEP 3.5: Start the viewport stability watcher ---
    _start_viewport_stable_watch(state, n_same_frames=10)


# -----------------------------------------------------------------------------
# 4. Create and synchronize the main window
# -----------------------------------------------------------------------------
def setup_main_window(state: dict[str, Any]) -> None:
    """
    Create the main application window and keep it aligned to the viewport.

    Args:
        state (dict[str, Any]): Shared application state updated with main
            window geometry and resize-related metadata.

    Returns:
        None: This function creates UI items and updates state in place.
    """
    live_vw = dpg.get_viewport_client_width() or dpg.get_viewport_width()
    live_vh = dpg.get_viewport_client_height() or dpg.get_viewport_height()

    dpg.set_viewport_min_width(int(live_vw / 2))
    dpg.set_viewport_min_height(int(live_vh / 2))

    # --- STEP 4.1: Use a provisional design reference when needed ---
    if not state.get("_viewport_ready"):
        state["design_ref_width"] = int(live_vw)
        state["design_ref_height"] = int(live_vh)

    # --- STEP 4.2: Compute the main window geometry ---
    vw = int(state["design_ref_width"])
    vh = int(state["design_ref_height"])
    win_w = max(1, vw)
    win_h = max(1, vh)

    # --- STEP 4.3: Create the main window ---
    dpg.add_window(
        label="Main Window",
        width=vw,
        height=vh,
        pos=(0, 0),
        tag="main_window",
        no_title_bar=True,
        menubar=False,
        no_move=True,
        no_resize=True,
        no_close=True,
        no_scrollbar=False,
        horizontal_scrollbar=True,
        no_scroll_with_mouse=False
    )

    with dpg.theme() as main_win_theme:
        with dpg.theme_component(dpg.mvWindowAppItem):
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
    dpg.bind_item_theme("main_window", main_win_theme)

    # -----------------------------------------------------------------------------
    # 4.1. On viewport resize
    # -----------------------------------------------------------------------------
    def _on_viewport_resize() -> None:
        """
        Resize the main window to match the live viewport dimensions.

        Args:
            None.

        Returns:
            None: This callback updates existing UI items in place.
        """
        vw = dpg.get_viewport_client_width() or dpg.get_viewport_width()
        vh = dpg.get_viewport_client_height() or dpg.get_viewport_height()

        # --- STEP 4.4.1: Update the main window geometry ---
        dpg.configure_item("main_window", width=int(vw), height=int(vh))
        try:
            if callable(state.get("redraw_main_frame_overlay")):
                state["redraw_main_frame_overlay"](state)
        except Exception:
            pass
        try:
            if callable(state.get("refresh_loading_screen_layout")):
                state["refresh_loading_screen_layout"](state)
        except Exception:
            pass

        # -----------------------------------------------------------------------------
        # 4.1.1. Resize cb
        # -----------------------------------------------------------------------------
        def _resize_cb(sender: Any, app_data: Any, user_data: Any) -> None:
            """
            Refresh responsive images after the viewport resize settles.

            Args:
                sender (Any): Dear PyGui callback sender. Included for callback
                    compatibility.
                app_data (Any): Dear PyGui callback payload. Included for
                    callback compatibility.
                user_data (dict[str, Any]): Shared application state passed to
                    the image refresh callback.

            Returns:
                None: This callback updates responsive UI assets in place.
            """
            # --- STEP 4.4.2: Refresh responsive images ---
            update_responsive_images(user_data)

        dpg.set_frame_callback(dpg.get_frame_count() + 1, _resize_cb, user_data=state)
        dpg.set_frame_callback(dpg.get_frame_count() + 8, _resize_cb, user_data=state)

    dpg.set_frame_callback(dpg.get_frame_count() + 2, _on_viewport_resize)
    dpg.set_viewport_resize_callback(_on_viewport_resize)

    # --- STEP 4.5: Persist the initial main window geometry ---
    state["main_win_width"] = win_w
    state["main_win_height"] = win_h
    state["main_win_y"] = 0
