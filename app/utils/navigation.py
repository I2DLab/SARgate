"""
=============
navigation.py
=============

Keyboard navigation and event handler utilities.

Registers global key bindings for tab switching, theme toggling, and focus
management within the SARgate interface. Used to provide consistent keyboard
shortcuts across different windows and interactive panels.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Key handler callback
# 3. Setup key handlers
# 4. Navigate buttons

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import sys
import os
import time
import dearpygui.dearpygui as dpg
from typing import Any
from app.utils.callbacks import (
    on_button_click, 
    change_tab,
    activate_main_tab,
    open_contextual_help
)
from app.analysis.overview.overview_decomposition import (
    update_molecule_choice_status,
    update_r_groups_choice_status,
    update_properties_windows,
    update_activities_windows,
    show_results_window
)
from app.analysis.overview.overview_enrichment_plot import (
    build_enrichment_layout,
    update_enrichment_Rgroup
)

pynput_keyboard = None


def _get_pynput_keyboard() -> Any:
    """
    Import pynput lazily for the optional global shortcut listener.

    Args:
        None.

    Returns:
        Any: The pynput keyboard module when available, otherwise `None`.
    """
    global pynput_keyboard
    if pynput_keyboard is not None:
        return pynput_keyboard

    if sys.platform in {"darwin", "win32"}:
        return None

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return None

    try:
        from pynput import keyboard
    except Exception:
        return None

    pynput_keyboard = keyboard
    return pynput_keyboard


# -----------------------------------------------------------------------------
# 2. Key handler callback
# -----------------------------------------------------------------------------
def _is_slith_shortcut_pressed() -> bool:
    """
    Check whether the Slith global shortcut is currently pressed.

    Args:
        None.

    Returns:
        bool: `True` when the user is pressing Shift+S together with Ctrl or
        Command/Super, otherwise `False`.
    """
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    ctrl_pressed = (
        dpg.is_key_down(dpg.mvKey_ModCtrl)
        or dpg.is_key_down(dpg.mvKey_LControl)
        or dpg.is_key_down(dpg.mvKey_RControl)
    )
    cmd_pressed = (
        dpg.is_key_down(dpg.mvKey_ModSuper)
        or dpg.is_key_down(dpg.mvKey_LWin)
        or dpg.is_key_down(dpg.mvKey_RWin)
    )
    return shift_pressed and (ctrl_pressed or cmd_pressed) and dpg.is_key_down(dpg.mvKey_S)


def _is_listener_shortcut_pressed(state: dict[str, Any]) -> bool:
    """
    Check whether the background global listener currently sees the Slith combo.

    Args:
        state (dict[str, Any]): Shared application state containing the latest
            listener key-state snapshot.

    Returns:
        bool: `True` when the listener reports Shift+S with Ctrl/Cmd.
    """
    pressed_keys = set(state.get("_slith_listener_pressed_keys", ()))
    return "shift" in pressed_keys and "s" in pressed_keys and ("ctrl" in pressed_keys or "cmd" in pressed_keys)


# -----------------------------------------------------------------------------
# 3. Toggle Slith Visibility
# -----------------------------------------------------------------------------
def _toggle_slith_visibility(state: dict[str, Any]) -> None:
    """
    Open or close the Slith tab while preserving the previous tab state.

    Args:
        state (dict[str, Any]): Shared application state used to track the
            current tab and Slith visibility.

    Returns:
        None: This helper updates the UI and state in place.
    """
    if not state["slith_ON"]:
        if dpg.does_item_exist("slith_tab") and not dpg.is_item_shown("slith_tab"):
            dpg.configure_item("slith_tab", show=True)
        if dpg.does_item_exist("slith_nav_button") and not dpg.is_item_shown("slith_nav_button"):
            dpg.configure_item("slith_nav_button", show=True)
        if dpg.does_item_exist("slith_nav_button_tooltip") and not dpg.is_item_shown("slith_nav_button_tooltip"):
            dpg.configure_item("slith_nav_button_tooltip", show=True)
        if dpg.does_item_exist("slith_nav_button"):
            top_nav_button_enabled = state.get("top_nav_button_enabled")
            if isinstance(top_nav_button_enabled, dict):
                top_nav_button_enabled["slith_nav_button"] = True
        if dpg.does_item_exist("slith_tab_child"):
            dpg.configure_item("slith_tab_child", show=True)

        if dpg.does_item_exist("cover_layer"):
            state["_slith_restore_cover_layer"] = dpg.is_item_shown("cover_layer")
            dpg.hide_item("cover_layer")
        else:
            state["_slith_restore_cover_layer"] = False

        state["slith_ON"] = True
        state["last_tab"] = state.get("current_tab", "input_tab")
        dpg.set_value("tab_bar", "slith_tab")
        activate_main_tab("slith_tab", state)
        if dpg.does_item_exist("slith_main_window"):
            try:
                dpg.focus_item("slith_main_window")
            except Exception:
                pass
        return

    if state.get("_slith_restore_cover_layer", False) and dpg.does_item_exist("cover_layer"):
        dpg.show_item("cover_layer")
    state["_slith_restore_cover_layer"] = False

    state["slith_ON"] = False
    last_tab = state.get("last_tab", "input_tab")
    if isinstance(last_tab, str):
        try:
            dpg.set_value("tab_bar", last_tab)
        except Exception:
            pass
        activate_main_tab(last_tab, state)
    else:
        change_tab(state)

    if not state["slith_is_paused"]:
        state["slith_is_paused"] = True
        if dpg.does_item_exist("pause_game_button"):
            dpg.set_item_label("pause_game_button", "Resume")

    if dpg.does_item_exist("slith_tab"):
        dpg.configure_item("slith_tab", show=False)
    if dpg.does_item_exist("slith_nav_button"):
        dpg.configure_item("slith_nav_button", show=False)
    if dpg.does_item_exist("slith_nav_button_tooltip"):
        dpg.configure_item("slith_nav_button_tooltip", show=False)
    if dpg.does_item_exist("slith_nav_button"):
        top_nav_button_enabled = state.get("top_nav_button_enabled")
        if isinstance(top_nav_button_enabled, dict):
            top_nav_button_enabled["slith_nav_button"] = False


# -----------------------------------------------------------------------------
# 4. Handle Slith Shortcut State
# -----------------------------------------------------------------------------
def _request_toggle_slith(state: dict[str, Any]) -> None:
    """
    Toggle Slith visibility with a small debounce shared by all shortcut paths.

    Args:
        state (dict[str, Any]): Shared application state used to keep shortcut
            debounce metadata.

    Returns:
        None: This helper toggles Slith in place when the debounce window has
        expired.
    """
    now = time.monotonic()
    last_toggle = state.get("_slith_shortcut_last_toggle", 0.0)
    if (now - last_toggle) < 0.35:
        return
    state["_slith_shortcut_last_toggle"] = now
    _toggle_slith_visibility(state)


# -----------------------------------------------------------------------------
# 5. Handle Slith Shortcut State
# -----------------------------------------------------------------------------
def _handle_slith_shortcut_state(state: dict[str, Any]) -> None:
    """
    Process the current Slith shortcut state using rising-edge detection.

    Args:
        state (dict[str, Any]): Shared application state used to track whether
            the shortcut was already pressed on the previous check.

    Returns:
        None: This helper toggles Slith visibility in place when needed.
    """
    pressed = _is_slith_shortcut_pressed() or _is_listener_shortcut_pressed(state)
    if pressed and not state.get("_slith_shortcut_was_down", False):
        _request_toggle_slith(state)
    state["_slith_shortcut_was_down"] = pressed


# -----------------------------------------------------------------------------
# 6. Start Slith Shortcut Listener
# -----------------------------------------------------------------------------
def _start_slith_shortcut_listener(state: dict[str, Any]) -> None:
    """
    Start a global keyboard listener for the Slith shortcut when available.

    Args:
        state (dict[str, Any]): Shared application state updated with shortcut
            request flags and listener metadata.

    Returns:
        None: This helper starts the background listener once.
    """
    if sys.platform in {"darwin", "win32"}:
        return

    keyboard = _get_pynput_keyboard()
    if keyboard is None or state.get("_slith_shortcut_listener_started", False):
        return

    canonical = keyboard.Listener.canonical

    ctrl_hotkey = keyboard.HotKey(
        keyboard.HotKey.parse("<ctrl>+<shift>+s"),
        lambda: state.__setitem__("_slith_shortcut_requested", True),
    )
    cmd_hotkey = keyboard.HotKey(
        keyboard.HotKey.parse("<cmd>+<shift>+s"),
        lambda: state.__setitem__("_slith_shortcut_requested", True),
    )

    def _safe_canonical(key: Any) -> Any:
        """
        Canonicalize a pynput key when possible, falling back to the raw key.

        Args:
            key (Any): Raw pynput key event.

        Returns:
            Any: Canonicalized key object when available.
        """
        try:
            return canonical(key)
        except Exception:
            return key

    def _on_press(key: Any) -> None:
        key = _safe_canonical(key)
        ctrl_hotkey.press(key)
        cmd_hotkey.press(key)

    def _on_release(key: Any) -> None:
        key = _safe_canonical(key)
        ctrl_hotkey.release(key)
        cmd_hotkey.release(key)

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.daemon = True
    listener.start()
    state["_slith_shortcut_listener"] = listener
    state["_slith_shortcut_listener_started"] = True


# -----------------------------------------------------------------------------
# 7. Key handler callback
# -----------------------------------------------------------------------------
def key_handler_callback(sender: Any, app_data: Any, state: dict[str, Any]) -> None:
    """
    Handle the opening/closing of the Slith window via ALT + SHIFT + S.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    if dpg.is_key_down(dpg.mvKey_F1):
        open_contextual_help(state)

    _handle_slith_shortcut_state(state)


# -----------------------------------------------------------------------------
# 8. Poll Slith Shortcut
# -----------------------------------------------------------------------------
def _poll_slith_shortcut(state: dict[str, Any]) -> None:
    """
    Poll the Slith shortcut state so it still works under blocking overlays.

    Args:
        state (dict[str, Any]): Shared application state.

    Returns:
        None: This helper polls keyboard state and reschedules itself.
    """
    if state.get("_slith_shortcut_requested", False):
        state["_slith_shortcut_requested"] = False
        _request_toggle_slith(state)

    _handle_slith_shortcut_state(state)

# -----------------------------------------------------------------------------
# 9. Setup key handlers
# -----------------------------------------------------------------------------
def setup_key_handlers(state: dict[str, Any]) -> None:
    """
    Register key-press handlers for navigation and global shortcuts.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """
    state["_slith_shortcut_was_down"] = False
    state["_slith_shortcut_requested"] = False
    state["_slith_shortcut_last_toggle"] = 0.0
    state["_slith_listener_pressed_keys"] = ()
    _start_slith_shortcut_listener(state)
    dpg.add_key_press_handler(parent="handler_registry", callback=lambda s, a: key_handler_callback(s, a, state))
    dpg.add_key_press_handler(dpg.mvKey_Up, parent="handler_registry", callback=lambda: navigate_buttons(None, None, "up", state))
    dpg.add_key_press_handler(dpg.mvKey_Down, parent="handler_registry", callback=lambda: navigate_buttons(None, None, "down", state))

    def _poll_callback(sender: Any = None, app_data: Any = None) -> None:
        """
        Poll the Slith shortcut state and queue the next frame callback.

        Args:
            sender (Any, optional): Dear PyGui callback sender.
            app_data (Any, optional): Dear PyGui callback payload.

        Returns:
            None: This callback keeps the shortcut bridge alive.
        """
        _poll_slith_shortcut(state)
        dpg.set_frame_callback(dpg.get_frame_count() + 1, _poll_callback)

    dpg.set_frame_callback(dpg.get_frame_count() + 1, _poll_callback)


# -----------------------------------------------------------------------------
# 10. Navigate buttons
# -----------------------------------------------------------------------------
def navigate_buttons(sender: Any, app_data: Any, direction: str, state: dict[str, Any]) -> None:
    """
    Handle button navigation in the Overview tab using Up/Down arrow keys.
    
    Args:
        sender (Any): Parameter accepted by this routine.
        app_data (Any): Parameter accepted by this routine.
        direction (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    selected_tab = dpg.get_item_alias(dpg.get_value("tab_bar"))
    if selected_tab != "overview_tab":
        return

    window_data = {
        "subsets": {
            "group": "subset_choice_group",
            "scroll": "subset_choice",
            "last_button_key": "last_clicked_button_sub",
            "update_funcs": [update_molecule_choice_status],
        },
        "molecules": {
            "group": "molecule_choice_group",
            "scroll": "molecule_choice",
            "last_button_key": "last_clicked_button_mol",
            "update_funcs": [update_r_groups_choice_status],
        },
        "r_groups": {
            "group": "r_group_choice_group",
            "scroll": "r_group_choice",
            "last_button_key": "last_clicked_button_r",
            "update_funcs": [],
        },
    }

    last_clicked = state.get("last_clicked_window")
    if last_clicked not in window_data:
        return

    group = window_data[last_clicked]["group"]
    scroll = window_data[last_clicked]["scroll"]
    last_button_key = window_data[last_clicked]["last_button_key"]
    update_funcs = window_data[last_clicked]["update_funcs"]

    buttons = [
        dpg.get_item_alias(item)
        for item in dpg.get_item_children(group, 1)
        if dpg.get_item_type(item) == "mvAppItemType::mvButton"
    ]
    focused_button = state.get(last_button_key)
    if not focused_button or focused_button not in buttons:
        return

    current_index = buttons.index(focused_button)
    if direction == "up" and current_index > 0:
        target_button = buttons[current_index - 1]
        button_y = dpg.get_item_pos(target_button)[1]
        dpg.set_y_scroll(group, button_y)
    elif direction == "down" and current_index < len(buttons) - 1:
        target_button = buttons[current_index + 1]
        button_y = dpg.get_item_pos(target_button)[1]
        dpg.set_y_scroll(group, button_y)
    else:
        return

    for func in update_funcs:
        func(target_button, state)
    update_properties_windows(target_button, state)
    update_activities_windows(target_button, state)
    on_button_click(target_button, state)
    show_results_window(target_button, state)
    
    if last_clicked == "subsets":
        build_enrichment_layout(target_button, state)
    elif last_clicked == "r_groups":
        update_enrichment_Rgroup(target_button, state)


    button_y = dpg.get_item_pos(target_button)[1]
    window_y = dpg.get_item_pos(scroll)[1]
    scroll_y = dpg.get_y_scroll(scroll)
    window_height = dpg.get_item_height(scroll)
    relative_y = button_y - window_y - scroll_y

    if relative_y < 0:
        dpg.set_y_scroll(scroll, scroll_y + relative_y - 5)
    elif relative_y + 25 > window_height:
        dpg.set_y_scroll(scroll, scroll_y + (relative_y + 25 - window_height) + 5)

    state[last_button_key] = target_button
