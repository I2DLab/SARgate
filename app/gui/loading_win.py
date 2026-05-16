"""
==============
loading_win.py
==============

Loading splash screen manager.

Displays a minimal progress window while SARgate 
initialises or processes heavy computations.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Draw loading screen

import os

import dearpygui.dearpygui as dpg
from typing import Any


# -----------------------------------------------------------------------------
# 1. Helpers
# -----------------------------------------------------------------------------
def _get_loading_overlay_state(state: dict[str, Any], context: str) -> dict[str, Any]:
    """
    Return the per-context overlay state container.
    """
    overlays = state.setdefault("_loading_overlays", {})
    return overlays.setdefault(context, {})


def _get_loading_tags(context: str) -> dict[str, str]:
    """
    Build the DPG tags used by one loading overlay context.
    """
    if context == "global":
        return {
            "overlay": "cover_layer",
            "background": "loading_overlay_background",
            "indicator": "win_loading_indicator",
            "drawlist": "loading_overlay_drawlist",
        }
    return {
        "overlay": f"{context}_cover_layer",
        "background": f"{context}_loading_overlay_background",
        "indicator": f"{context}_win_loading_indicator",
        "drawlist": f"{context}_loading_overlay_drawlist",
    }


def _get_loading_indicator_extent(radius: float, circle_count: int) -> float:
    """
    Estimate the square footprint occupied by the loading indicator.

    Args:
        radius (float): Radius of each small circle composing the indicator.
        circle_count (int): Number of circles in the indicator.

    Returns:
        float: Estimated width/height of the indicator footprint.
    """
    return radius * max(8.0, min(16.0, circle_count * 0.8))


def _get_loading_text_size(text: str, size: int) -> tuple[float, float]:
    """
    Measure loading text size, with a safe Dear PyGui fallback.

    Args:
        text (str): Text to measure.
        size (int): Requested font size.

    Returns:
        tuple[float, float]: Text width and height.
    """
    try:
        return dpg.get_text_size(text, size=size)
    except Exception:
        return (len(text) * (size * 0.55), size * 1.2)


def _ensure_loading_cover_theme() -> None:
    """
    Create the loading overlay theme with zero padding once.

    Returns:
        None: Theme is created only if missing.
    """
    if dpg.does_item_exist("loading_cover_zero_padding_theme"):
        return

    with dpg.theme(tag="loading_cover_zero_padding_theme"):
        with dpg.theme_component(dpg.mvWindowAppItem):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0, category=dpg.mvThemeCat_Core)


def _ensure_loading_background_texture(state: dict[str, Any]) -> str | None:
    """
    Load the background texture once and cache its size in state.
    """
    texture_tag = "loading_splash_bg_texture"
    if dpg.does_item_exist(texture_tag):
        return texture_tag

    image_path = os.path.join(os.getcwd(), "assets", "icons", "splash_bg.png")
    if not os.path.isfile(image_path):
        return None

    try:
        width, height, channels, data = dpg.load_image(image_path)
    except Exception:
        return None

    state["_loading_background_size"] = (int(width), int(height))
    registry_tag = "loading_splash_texture_registry"
    if not dpg.does_item_exist(registry_tag):
        with dpg.texture_registry(tag=registry_tag):
            dpg.add_static_texture(width=width, height=height, default_value=data, tag=texture_tag)
    else:
        dpg.add_static_texture(width=width, height=height, default_value=data, tag=texture_tag, parent=registry_tag)
    return texture_tag


def _schedule_loading_screen_layout_refresh(state: dict[str, Any], context: str = "global") -> None:
    """
    Keep the loading overlay centered while it exists.

    Args:
        state (dict[str, Any]): Shared application state.

    Returns:
        None: Registers the next frame callback if needed.
    """
    overlay_tag = _get_loading_tags(context)["overlay"]
    if not dpg.does_item_exist(overlay_tag):
        return

    def _tick() -> None:
        if not dpg.does_item_exist(overlay_tag):
            return
        try:
            refresh_loading_screen_layout(state, context=context)
        except Exception:
            pass
        if dpg.does_item_exist(overlay_tag):
            _schedule_loading_screen_layout_refresh(state, context=context)

    dpg.set_frame_callback(dpg.get_frame_count() + 1, _tick)


def refresh_loading_screen_layout(state: dict[str, Any], context: str = "global") -> None:
    """
    Recenter the loading overlay contents inside the live viewport.

    Args:
        state (dict[str, Any]): Shared application state.

    Returns:
        None: Existing UI items are updated in place.
    """
    overlay_state = _get_loading_overlay_state(state, context)
    tags = _get_loading_tags(context)
    overlay_tag = tags["overlay"]
    if not dpg.does_item_exist(overlay_tag):
        return

    parent_tag = overlay_state.get("parent")
    is_local_overlay = bool(parent_tag)
    if is_local_overlay and dpg.does_item_exist(parent_tag):
        try:
            width, height = dpg.get_item_rect_size(parent_tag)
        except Exception:
            width, height = (0, 0)
    else:
        width = dpg.get_viewport_client_width() or dpg.get_viewport_width()
        height = dpg.get_viewport_client_height() or dpg.get_viewport_height()

    if width <= 0 or height <= 0:
        return

    radius = float(overlay_state.get("radius", state["analysis_init_load_radius"] * 1.8))
    circle_count = int(overlay_state.get("circle_count", 16))
    text = str(overlay_state.get("text", ""))
    text_size = int(overlay_state.get("text_size", max(16, round(radius * 2.0))))

    indicator_extent = _get_loading_indicator_extent(radius, circle_count)
    indicator_x = (width - indicator_extent) / 2.0
    indicator_y = (height - indicator_extent) / 2.0

    overlay_cfg = (0, 0, int(width), int(height), bool(is_local_overlay))
    if overlay_state.get("_last_overlay_cfg") != overlay_cfg:
        if is_local_overlay:
            dpg.configure_item(overlay_tag, width=int(width), height=int(height))
        else:
            dpg.configure_item(overlay_tag, pos=(0, 0), width=int(width), height=int(height))
        overlay_state["_last_overlay_cfg"] = overlay_cfg

    if dpg.does_item_exist(tags["background"]):
        background_cfg = (0, 0, int(width), int(height))
        if overlay_state.get("_last_background_cfg") != background_cfg:
            dpg.configure_item(
                tags["background"],
                pos=(0, 0),
                width=max(1, int(width)),
                height=max(1, int(height)),
            )
            overlay_state["_last_background_cfg"] = background_cfg

    if dpg.does_item_exist(tags["indicator"]):
        indicator_cfg = (round(indicator_x, 2), round(indicator_y, 2), round(radius, 2), int(circle_count))
        if overlay_state.get("_last_indicator_cfg") != indicator_cfg:
            dpg.configure_item(
                tags["indicator"],
                pos=(indicator_x, indicator_y),
                radius=radius,
                circle_count=circle_count,
            )
            overlay_state["_last_indicator_cfg"] = indicator_cfg

    if dpg.does_item_exist(tags["drawlist"]):
        text_w, text_h = _get_loading_text_size(text, text_size)
        drawlist_cfg = (0, 0, int(width), int(height))
        if overlay_state.get("_last_drawlist_cfg") != drawlist_cfg:
            dpg.configure_item(
                tags["drawlist"],
                pos=(0, 0),
                width=max(1, int(width)),
                height=max(1, int(height)),
            )
            overlay_state["_last_drawlist_cfg"] = drawlist_cfg
        dpg.delete_item(tags["drawlist"], children_only=True)
        center_x = indicator_x + (indicator_extent / 2.0)
        center_y = indicator_y + (indicator_extent / 2.0)
        if dpg.does_item_exist(tags["indicator"]):
            try:
                indicator_w, indicator_h = dpg.get_item_rect_size(tags["indicator"])
                if indicator_w > 0 and indicator_h > 0:
                    center_x = indicator_x + (indicator_w / 2.0)
                    center_y = indicator_y + (indicator_h / 2.0)
            except Exception:
                pass
        text_x = center_x - (text_w / 2.0)
        text_y = center_y - (text_h / 2.0)
        dpg.draw_text(
            (text_x, text_y),
            text,
            size=text_size,
            color=overlay_state.get(
                "text_color",
                state["themes"][state["theme_name"]]["Button Active"],
            ),
            parent=tags["drawlist"],
        )


def refresh_all_loading_screen_layouts(state: dict[str, Any]) -> None:
    """
    Refresh every active loading overlay layout.
    """
    overlays = state.get("_loading_overlays", {})
    if not isinstance(overlays, dict) or not overlays:
        refresh_loading_screen_layout(state)
        return
    for context in list(overlays.keys()):
        try:
            refresh_loading_screen_layout(state, context=str(context))
        except Exception:
            pass


def set_loading_screen_text(state: dict[str, Any], text: str, context: str = "global") -> None:
    """
    Update the text shown inside the loading indicator.

    Args:
        state (dict[str, Any]): Shared application state.
        text (str): Text displayed inside the loading indicator.

    Returns:
        None: The loading overlay is updated in place when present.
    """
    _get_loading_overlay_state(state, context)["text"] = str(text)
    try:
        refresh_loading_screen_layout(state, context=context)
    except Exception:
        pass
    try:
        dpg.split_frame()
    except Exception:
        pass


def set_loading_screen_progress(state: dict[str, Any], progress: float, context: str = "global") -> None:
    """
    Update the loading indicator percentage.

    Args:
        state (dict[str, Any]): Shared application state.
        progress (float): Progress in the [0, 100] interval.

    Returns:
        None: The loading overlay text is updated in place.
    """
    clamped = max(0.0, min(100.0, float(progress)))
    new_text = f"{int(round(clamped))}%"
    current_text = _get_loading_overlay_state(state, context).get("text", "")
    if current_text == new_text:
        return
    set_loading_screen_text(state, new_text, context=context)


def delete_loading_screen(state: dict[str, Any], context: str = "global") -> None:
    """
    Remove one loading overlay context if it exists.
    """
    overlay_tag = _get_loading_tags(context)["overlay"]
    if dpg.does_item_exist(overlay_tag):
        dpg.delete_item(overlay_tag)
    overlays = state.get("_loading_overlays", {})
    if isinstance(overlays, dict):
        overlays.pop(context, None)


# -----------------------------------------------------------------------------
# 2. Draw loading screen
# -----------------------------------------------------------------------------
def draw_loading_screen(
    state: dict[str, Any],
    bg: bool = True,
    parent: str | None = None,
    context: str = "global",
) -> None:

    # Prevent multiple overlays; if one exists, do nothing and exit.
    tags = _get_loading_tags(context)
    overlay_state = _get_loading_overlay_state(state, context)
    state["refresh_loading_screen_layout"] = refresh_all_loading_screen_layouts
    overlay_state["parent"] = parent
    overlay_state["text"] = ""

    if not dpg.does_item_exist(tags["overlay"]):

        # Create a full-viewport, modal window acting as a blocking cover layer.

        # Compute the drawing area based on main window dimensions.
        if parent and dpg.does_item_exist(parent):
            width, height = (-1, -1)
        else:
            width, height = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()

        base_container_kwargs = dict(
            tag=tags["overlay"],
            pos=(0, 0),
            width=width if width == -1 else max(1, int(width)),
            height=height if height == -1 else max(1, int(height)),
            no_scrollbar=True,
            horizontal_scrollbar=False,
            no_scroll_with_mouse=True,
        )

        if parent and dpg.does_item_exist(parent):
            container = dpg.child_window(parent=parent, border=False, **base_container_kwargs)
        else:
            container = dpg.window(
                no_title_bar=True,
                no_move=True,
                no_resize=True,
                no_collapse=True,
                modal=False,
                no_background=not bg,
                **base_container_kwargs,
            )

        with container:
            _ensure_loading_cover_theme()
            dpg.bind_item_theme(tags["overlay"], "loading_cover_zero_padding_theme")
            
            # Place a large spinner near the centre; size scales with `analysis_init_load_radius`.
            theme_dict = state["themes"][state["theme_name"]]
            radius = state["analysis_init_load_radius"] * 1.8
            circle_count = 14
            text_size = max(16, round(radius * 2.0))

            overlay_state.pop("_last_overlay_cfg", None)
            overlay_state.pop("_last_background_cfg", None)
            overlay_state.pop("_last_indicator_cfg", None)
            overlay_state.pop("_last_drawlist_cfg", None)
            overlay_state["radius"] = radius
            overlay_state["circle_count"] = circle_count
            overlay_state["text"] = str(overlay_state.get("text", ""))
            overlay_state["text_size"] = text_size
            overlay_state["text_color"] = theme_dict["Button Active"]
            background_texture = _ensure_loading_background_texture(state) if bg else None
            if background_texture is not None:
                dpg.add_image(
                    background_texture,
                    tag=tags["background"],
                    width=max(1, int(width if width != -1 else 1)),
                    height=max(1, int(height if height != -1 else 1)),
                    pos=(0, 0),
                )
            dpg.add_loading_indicator(
                tag=tags["indicator"],
                speed=1.6,
                style=0,
                color=theme_dict["Button Active"],
                secondary_color=theme_dict["Table Header"],
                circle_count=circle_count,
                radius=radius,
                pos=(0, 0),
            )
            draw_w = max(1, int(_get_loading_indicator_extent(radius, circle_count)))
            draw_h = draw_w
            dpg.add_drawlist(width=draw_w, height=draw_h, tag=tags["drawlist"])

        refresh_loading_screen_layout(state, context=context)
        _schedule_loading_screen_layout_refresh(state, context=context)
        try:
            dpg.split_frame()
        except Exception:
            pass
    else:
        refresh_loading_screen_layout(state, context=context)
        try:
            dpg.split_frame()
        except Exception:
            pass
