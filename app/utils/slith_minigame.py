"""
=================
slith_minigame.py
=================

Alkane Snake mini-game on a hex edge grid.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Open slith window

import os
import math
import time
import json
import random
import dearpygui.dearpygui as dpg
from typing import Any
from app.gui.themes_manager import apply_inner_child_theme


# -----------------------------------------------------------------------------
# 2. Open slith window
# -----------------------------------------------------------------------------
def open_slith_window(state: dict[str, Any]) -> Any:
    """
    Create the Slith mini-game UI and start the loop.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """

    global text_color, canvas_width, canvas_height, HEX_SIZE, ROWS, COLS
    global start_q, start_r, start_dir, floating_messages, is_game_over
    global start_time, pause_start_time, paused_duration
    global snake_segments, turn_flag, auto_turn, forced_vertical, direction
    global food, last_update_time, speed
    global fatty_acids, slith_record, has_shown_new_record, last_food_pos
    global effects_time, last_effects_tick
    global ORIGIN_X, ORIGIN_Y, GRID_MINX, GRID_MINY, GRID_MAXX, GRID_MAXY
    global _slith_static_dirty, _frame_buffer_span, _last_processed_frame, _last_render_time, _idle_render_interval
    global food_eaten, pending_growth  # chemical and geometric growth

    text_color = state["theme"]["Text Color"]
    ROWS = COLS = 9

    # placeholders before late sizing
    HEX_SIZE = int(state.get("slith_hexagon_size", 18))
    canvas_width = int(state.get("slith_canvas_width", 900))
    canvas_height = int(state.get("slith_canvas_height", 560))

    # -----------------------------------------------------------------------------
    # 2.1. Compute sizes
    # -----------------------------------------------------------------------------
    def _compute_sizes(parent_w: Any, parent_h: Any) -> Any:
        """
        Given parent size, return HEX_SIZE, field_w, field_h, inner_x, inner_y, controls_w, controls_h, controls_x, controls_y.
        
        Args:
            parent_w (Any): Parameter accepted by this routine.
            parent_h (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        controls_h = 72
        gap = 8
        avail_h = max(1, parent_h - controls_h - gap)
        hex_w_lim = parent_w / (math.sqrt(3) * COLS)
        hex_h_lim = avail_h / (1.5 * (ROWS - 1) + 2)
        hs = int(max(1, math.floor(min(hex_w_lim, hex_h_lim))))
        field_w = int(math.ceil(math.sqrt(3) * hs * COLS))
        field_h = int(math.ceil(hs * (1.5 * (ROWS - 1) + 2)))
        inner_x = max(0, (parent_w - field_w) // 2)
        inner_y = 0
        controls_w = max(field_w, 360)
        controls_x = max(0, (parent_w - controls_w) // 2)
        controls_y = field_h + gap
        return hs, field_w, field_h, inner_x, inner_y, controls_w, controls_h, controls_x, controls_y

    # -----------------------------------------------------------------------------
    # 2.2. Apply sizes
    # -----------------------------------------------------------------------------
    def _apply_sizes(parent_w: Any, parent_h: Any) -> None:
        """
        Apply computed sizes to items and state; recompute centring and redraw.
        
        Args:
            parent_w (Any): Parameter accepted by this routine.
            parent_h (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        global HEX_SIZE, canvas_width, canvas_height, _slith_static_dirty
        hs, field_w, field_h, inner_x, inner_y, ctrl_w, ctrl_h, ctrl_x, ctrl_y = _compute_sizes(parent_w, parent_h)
        HEX_SIZE, canvas_width, canvas_height = hs, field_w, field_h

        # persist
        state["slith_main_win_width"]  = int(parent_w)
        state["slith_main_win_height"] = int(parent_h)
        state["slith_hexagon_size"]    = int(HEX_SIZE)
        state["slith_canvas_width"]    = int(canvas_width)
        state["slith_canvas_height"]   = int(canvas_height)
        state["slith_canvas_x"]        = int(inner_x)
        state["slith_canvas_y"]        = int(inner_y)

        # apply to UI
        if dpg.does_item_exist("slith_inner_window"):
            dpg.configure_item("slith_inner_window", width=canvas_width, height=canvas_height, pos=(inner_x, inner_y))
        if dpg.does_item_exist("canvas"):
            dpg.configure_item("canvas", width=canvas_width, height=canvas_height)
        if dpg.does_item_exist("slith_controls_window"):
            dpg.configure_item("slith_controls_window", width=ctrl_w, height=ctrl_h, pos=(ctrl_x, ctrl_y))

        _compute_grid_bbox()
        _slith_static_dirty = True
        draw_board()
        draw_food()

    snake_segments = []
    direction = 0
    turn_flag = None
    auto_turn = "right"
    forced_vertical = None
    speed = 0.6
    start_time = time.monotonic()
    start_q, start_r = 5, 5
    start_dir = 0
    floating_messages = []
    is_game_over = False
    pause_start_time = 0.0
    paused_duration = 0.0
    last_food_pos = (0.0, 0.0)
    effects_time = 0.0
    last_effects_tick = time.monotonic()
    last_update_time = time.monotonic()
    _slith_static_dirty = True
    _frame_buffer_span = 4
    _last_processed_frame = -1
    _last_render_time = time.monotonic()
    _idle_render_interval = 1.0 / 30.0
    food_eaten = 0          # 0 → 4C; each food → +2C
    pending_growth = 0      # segments to add (each move uses 1 if >0)

    ORIGIN_X = ORIGIN_Y = 0.0
    GRID_MINX = GRID_MINY = 0.0
    GRID_MAXX = float(canvas_width)
    GRID_MAXY = float(canvas_height)

    settings_path = state.get("settings_file", os.path.join("assets", "config", "settings.ssf"))

    # -----------------------------------------------------------------------------
    # 2.3. Read slith record
    # -----------------------------------------------------------------------------
    def read_slith_record() -> Any:
        # default = 4 carbons (butyric acid) instead of 3
        """
        Execute the read slith record routine.
        
        Args:
            None.
        
        Returns:
            Any: Value produced by the routine.
        """
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
            except Exception:
                return 4
            return int(settings.get("slith_record", 4))
        return 4

    # -----------------------------------------------------------------------------
    # 2.4. Save slith record
    # -----------------------------------------------------------------------------
    def save_slith_record(new_record: Any) -> None:
        """
        Save slith record.
        
        Args:
            new_record (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        settings = {}
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
            except Exception:
                settings = {}
        settings["slith_record"] = int(new_record)
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)

    slith_record = read_slith_record()
    has_shown_new_record = False

    common_names = {
        4:  "butyric acid",
        6:  "caproic acid",
        8:  "caprylic acid",
        10: "capric acid",
        12: "lauric acid",
        14: "myristic acid",
        16: "palmitic acid",
        18: "stearic acid",
        20: "arachidic acid",
        22: "behenic acid",
        24: "lignoceric acid",
        26: "cerotic acid",
        28: "montanic acid",
        30: "melissic acid",
        32: "lacceroic acid",
    }

    prefixes = [
        "", "", "", "prop", "but", "pent", "hex", "hept", "oct", "non", "dec",
        "undec", "dodec", "tridec", "tetradec", "pentadec", "hexadec",
        "heptadec", "octadec", "nonadec", "eicos", "heneicos", "docos",
        "tricos", "tetracos", "pentacos", "hexacos", "heptacos",
        "octacos", "nonacos", "triacont", "hentriacont", "dotriacont",
        "tritriacont", "tetratriacont", "pentatriacont", "hexatriacont",
        "heptatriacont", "octatriacont", "nonatriacont", "tetracont",
        "hentetracont", "dotetracont", "tritetracont", "tetratetracont",
        "pentatetracont", "hexatetracont", "heptatetracont", "octatetracont",
        "nonatetracont", "pentacont", "henpentacont", "dopentacont",
        "tripentacont", "tetrapentacont", "pentapentacont",
        "hexapentacont", "heptapentacont", "octapentacont",
        "nonapentacont", "hexacont", "henhexacont", "dohexacont",
        "trihexacont", "tetrahexacont", "pentahexacont", "hexahexacont",
        "heptahexacont", "octahexacont", "nonahexacont", "heptacont",
        "henheptacont", "doheptacont", "triheptacont", "tetraheptacont",
        "pentaheptacont", "hexaheptacont", "heptaheptacont",
        "octaheptacont", "nonaheptacont", "octacont", "henoctacont",
        "dooctacont", "trioctacont", "tetraoctacont", "pentaoctacont",
        "hexaoctacont", "heptaoctacont", "octaoctacont", "nonaoctacont",
        "enneacont", "henenneacont", "doenneacont", "trienneacont",
        "tetraenneacont", "pentaenneacont", "hexaenneacont",
        "heptaenneacont", "octaenneacont", "nonaenneacont", "hect"
    ]

    # -----------------------------------------------------------------------------
    # 2.5. Fatty acid name
    # -----------------------------------------------------------------------------
    def fatty_acid_name(n: int) -> Any:
        """
        Execute the fatty acid name routine.
        
        Args:
            n (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        if n < len(prefixes):
            base = prefixes[n] + "anoic acid"
        else:
            base = f"C{n}H{2*n}O2"
        if n in common_names:
            return f"{base} ({common_names[n]})"
        return base

    fatty_acids = [fatty_acid_name(n) for n in range(0, 570)]

    # -----------------------------------------------------------------------------
    # 2.6. Hex to pixel raw
    # -----------------------------------------------------------------------------
    def _hex_to_pixel_raw(q: Any, r: Any) -> Any:
        """
        Execute the hex to pixel raw routine.
        
        Args:
            q (Any): Parameter accepted by this routine.
            r (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        x = HEX_SIZE * math.sqrt(3) * (q + 0.5 * (r % 2))
        y = HEX_SIZE * 1.5 * r
        return x, y

    # -----------------------------------------------------------------------------
    # 2.7. Hex corner raw
    # -----------------------------------------------------------------------------
    def hex_corner_raw(center_pt: Any, size: Any, i: int) -> Any:
        """
        Execute the hex corner raw routine.
        
        Args:
            center_pt (Any): Parameter accepted by this routine.
            size (Any): Parameter accepted by this routine.
            i (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        a = math.radians(60 * i - 30)
        return center_pt[0] + size * math.cos(a), center_pt[1] + size * math.sin(a)

    # -----------------------------------------------------------------------------
    # 2.8. Compute grid bbox
    # -----------------------------------------------------------------------------
    def _compute_grid_bbox() -> None:
        """
        Center the 9x9 grid in the canvas and cache bounds.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        global ORIGIN_X, ORIGIN_Y, GRID_MINX, GRID_MINY, GRID_MAXX, GRID_MAXY
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")
        for r in range(ROWS):
            max_q = COLS if r % 2 == 0 else COLS - 1
            for q in range(max_q):
                cx, cy = _hex_to_pixel_raw(q, r)
                for i in range(6):
                    x, y = hex_corner_raw((cx, cy), HEX_SIZE, i)
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        grid_w = max(1.0, max_x - min_x)
        grid_h = max(1.0, max_y - min_y)
        ORIGIN_X = (canvas_width - grid_w) / 2.0 - min_x
        ORIGIN_Y = (canvas_height - grid_h) / 2.0 - min_y
        GRID_MINX = min_x + ORIGIN_X
        GRID_MAXX = max_x + ORIGIN_X
        GRID_MINY = min_y + ORIGIN_Y
        GRID_MAXY = max_y + ORIGIN_Y

    # -----------------------------------------------------------------------------
    # 2.9. Hex to pixel
    # -----------------------------------------------------------------------------
    def hex_to_pixel(q: Any, r: Any) -> Any:
        """
        Execute the hex to pixel routine.
        
        Args:
            q (Any): Parameter accepted by this routine.
            r (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        x, y = _hex_to_pixel_raw(q, r)
        return x + ORIGIN_X, y + ORIGIN_Y

    # -----------------------------------------------------------------------------
    # 2.10. Hex corner
    # -----------------------------------------------------------------------------
    def hex_corner(center_pt: Any, size: Any, i: int) -> Any:
        """
        Execute the hex corner routine.
        
        Args:
            center_pt (Any): Parameter accepted by this routine.
            size (Any): Parameter accepted by this routine.
            i (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        a = math.radians(60 * i - 30)
        return center_pt[0] + size * math.cos(a), center_pt[1] + size * math.sin(a)

    _compute_grid_bbox()

    # -----------------------------------------------------------------------------
    # 2.11. On key
    # -----------------------------------------------------------------------------
    def on_key(sender: Any, app_data: Any) -> None:
        """
        Execute the on key routine.
        
        Args:
            sender (Any): Parameter accepted by this routine.
            app_data (Any): Parameter accepted by this routine.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        global turn_flag
        if app_data == dpg.mvKey_Left:
            turn_flag = "left"
        elif app_data == dpg.mvKey_Right:
            turn_flag = "right"

    # -----------------------------------------------------------------------------
    # 2.12. Schedule next frames
    # -----------------------------------------------------------------------------
    def _schedule_next_frames() -> None:
        """
        Execute the schedule next frames routine.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        if not dpg.does_item_exist("canvas"):
            return
        base = dpg.get_frame_count()
        for k in range(1, _frame_buffer_span + 1):
            dpg.set_frame_callback(base + k, _on_frame)

    # -----------------------------------------------------------------------------
    # 2.13. On frame
    # -----------------------------------------------------------------------------
    def _on_frame(
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
    ) -> None:
        """
        Execute the on frame routine.
        
        Args:
            sender (Any, optional): Dear PyGui callback sender. Included for
                callback compatibility.
            app_data (Any, optional): Dear PyGui callback payload. Included for
                callback compatibility.
            user_data (Any, optional): Dear PyGui callback user data. Included
                for callback compatibility.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        global last_update_time, effects_time, last_effects_tick, _last_processed_frame, _last_render_time
        if not dpg.does_item_exist("canvas"):
            return

        current_frame = dpg.get_frame_count()
        if current_frame == _last_processed_frame:
            _schedule_next_frames()
            return
        _last_processed_frame = current_frame

        now = time.monotonic()
        effects_time += (now - last_effects_tick)
        last_effects_tick = now

        if not state.get("slith_is_paused", False) and (now - last_update_time) >= speed:
            move_snake()
            last_update_time = now
            _last_render_time = now
        elif (now - _last_render_time) >= _idle_render_interval:
            draw_board()
            draw_food()
            _last_render_time = now

        _schedule_next_frames()

    # -----------------------------------------------------------------------------
    # 2.14. Move snake
    # -----------------------------------------------------------------------------
    def move_snake() -> Any:
        """
        Execute the move snake routine.
        
        Args:
            None.
        
        Returns:
            Any: Value produced by the routine.
        """
        global direction, turn_flag, food, speed, auto_turn, food_eaten, pending_growth, _last_render_time
        head_seg = snake_segments[0]
        _, p2_head = head_seg
        cx, cy = p2_head

        if turn_flag == "right":
            direction = (direction + 1) % 6
            attempted_turn = "right"
            turn_flag = None
        elif turn_flag == "left":
            direction = (direction - 1) % 6
            attempted_turn = "left"
            turn_flag = None
        else:
            if auto_turn == "right":
                direction = (direction + 1) % 6
                attempted_turn = "right"
                auto_turn = "left"
            else:
                direction = (direction - 1) % 6
                attempted_turn = "left"
                auto_turn = "right"

        # -----------------------------------------------------------------------------
        # 2.14.1. Try step
        # -----------------------------------------------------------------------------
        def try_step(dir_idx: str) -> Any:
            """
            Execute the try step routine.
            
            Args:
                dir_idx (Any): Parameter accepted by this routine.
            
            Returns:
                Any: Value produced by the routine.
            """
            a = math.radians(60 * dir_idx - 30)
            dx, dy = HEX_SIZE * math.cos(a), HEX_SIZE * math.sin(a)
            next_p = (cx + dx, cy + dy)
            for r in range(ROWS):
                max_q = COLS if r % 2 == 0 else COLS - 1
                for q in range(max_q):
                    center_pt = hex_to_pixel(q, r)
                    corners = [hex_corner(center_pt, HEX_SIZE, i) for i in range(6)]
                    for i in range(6):
                        a0, b0 = corners[i], corners[(i + 1) % 6]
                        if (
                            (math.isclose(a0[0], cx, abs_tol=1e-1) and math.isclose(a0[1], cy, abs_tol=1e-1) and
                             math.isclose(b0[0], next_p[0], abs_tol=1e-1) and math.isclose(b0[1], next_p[1], abs_tol=1e-1))
                            or
                            (math.isclose(b0[0], cx, abs_tol=1e-1) and math.isclose(b0[1], cy, abs_tol=1e-1) and
                             math.isclose(a0[0], next_p[0], abs_tol=1e-1) and math.isclose(a0[1], next_p[1], abs_tol=1e-1))
                        ):
                            return True, next_p
            return False, next_p

        ok, next_p = try_step(direction)
        if not ok:
            opposite_turn = (direction - 2) % 6 if attempted_turn == "right" else (direction + 2) % 6
            ok, next_p = try_step(opposite_turn)
            if ok:
                direction = opposite_turn

        if not ok:
            draw_board()
            draw_food()
            return

        # add new head segment
        snake_segments.insert(0, ((cx, cy), next_p))

        fx, fy = food
        if math.hypot(next_p[0] - fx, next_p[1] - fy) < 5:
            # snake eats: +2 carbons (chemically) and +2 segments (geometrically)
            food = spawn_food()
            food_eaten += 1
            pending_growth += 2
            floating_messages.append({
                "atom": "CO",
                "charge": "2",
                "atom_pos": (fx - 20, fy - 35),
                "charge_pos": (fx - 3, fy - 30),
                "start_tick": effects_time
            })
            speed = max(0.1, speed * 0.95)
        # growth handling: if pending_growth > 0, do not remove tail
        if pending_growth > 0:
            pending_growth -= 1
        else:
            snake_segments.pop()

        draw_board()
        draw_food()
        _last_render_time = time.monotonic()

        # self-collision check
        for i in range(1, len(snake_segments) - 1):
            seg = snake_segments[i]
            if (math.isclose(next_p[0], seg[0][0], abs_tol=1e-1) and math.isclose(next_p[1], seg[0][1], abs_tol=1e-1)) or \
               (math.isclose(next_p[0], seg[1][0], abs_tol=1e-1) and math.isclose(next_p[1], seg[1][1], abs_tol=1e-1)):
                game_over()
                return

    # -----------------------------------------------------------------------------
    # 2.15. Draw board
    # -----------------------------------------------------------------------------
    def draw_board() -> None:
        """
        Draw board.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        global has_shown_new_record, _slith_static_dirty
        if not dpg.does_item_exist("canvas"):
            return
        if _slith_static_dirty and dpg.does_item_exist("slith_static_layer"):
            dpg.delete_item("slith_static_layer", children_only=True)
            for r in range(ROWS):
                max_q = COLS if r % 2 == 0 else COLS - 1
                for q in range(max_q):
                    cx, cy = hex_to_pixel(q, r)
                    pts = [hex_corner((cx, cy), HEX_SIZE, i) for i in range(6)]
                    dpg.draw_polygon(
                        points=pts,
                        color=(255, 255, 255, 10),
                        fill=(150, 150, 150, 255),
                        thickness=7,
                        parent="slith_static_layer",
                    )
            _slith_static_dirty = False

        if dpg.does_item_exist("slith_dynamic_layer"):
            dpg.delete_item("slith_dynamic_layer", children_only=True)
            parent_tag = "slith_dynamic_layer"
        else:
            parent_tag = "canvas"

        # snake
        radius = max(2, int(HEX_SIZE * 0.12))
        for i, (sp1, sp2) in enumerate(snake_segments):
            # outer white
            dpg.draw_line(p1=sp1, p2=sp2, color=(255, 255, 255, 255),
                          thickness=8 if i == 0 else 6, parent=parent_tag)
            # middle black
            dpg.draw_line(p1=sp1, p2=sp2, color=(0, 0, 0, 255),
                          thickness=8 if i == 0 else 5, parent=parent_tag)
            # inner colored
            dpg.draw_line(
                p1=sp1, p2=sp2,
                color=(255, 0, 0, 255) if i == 0 else (255, 255, 0, 255),
                thickness=6 if i == 0 else 3,
                parent=parent_tag
            )

            # carbon atoms at segment ends
            dpg.draw_circle(
                center=sp1,
                radius=radius,
                color=(0, 0, 0, 0),
                fill=(0, 0, 0, 30),
                parent=parent_tag,
            )
            dpg.draw_circle(
                center=sp2,
                radius=radius,
                color=(0, 0, 0, 0),
                fill=(0, 0, 0, 30),
                parent=parent_tag,
            )

            if i == 0:
                # sp2 is the head end
                x1, y1 = sp1
                x2, y2 = sp2
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy) or 1.0
                # unit vector along bond
                ux, uy = dx / length, dy / length
                # unit normal (perpendicolare)
                nx, ny = -uy, ux

                # distanza dal carbonio e separazione fra gli ossigeni
                offset_along = radius * 1.4
                offset_side = radius * 0.9

                # punto base vicino alla testa
                base_x = x2 + ux * offset_along
                base_y = y2 + uy * offset_along

                # due O affiancati
                o1 = (base_x + nx * offset_side, base_y + ny * offset_side)
                o2 = (base_x - nx * offset_side, base_y - ny * offset_side)

                for ox, oy in (o1, o2):
                    dpg.draw_circle(
                        center=(ox, oy),
                        radius=radius,
                        color=(255, 0, 0, 0),
                        fill=(255, 0, 0, 30),
                        parent=parent_tag,
                    )
        # chemical length: start at 4C, +2 per food
        length = 4 + 2 * food_eaten
        dpg.set_value("length_text", f"Length: {length}")
        name = fatty_acids[length] if length < len(fatty_acids) else f"C{length}H{2*length}O2"
        dpg.set_value("name_text", f"{name}")

        # high score logic in carbons
        if length > slith_record:
            save_slith_record(length)
            dpg.set_value("record_text", f"High Score: {length}")
            if not has_shown_new_record:
                fx, fy = last_food_pos
                floating_messages.append({
                    "atom": "NEW HIGH SCORE!",
                    "charge": "",
                    "atom_pos": (fx - 60, fy - HEX_SIZE - 20),
                    "charge_pos": (0, 0),
                    "start_tick": effects_time
                })
                has_shown_new_record = True

                
    # -----------------------------------------------------------------------------
    # 2.16. Draw food
    # -----------------------------------------------------------------------------
    def draw_food() -> Any:
        """
        Draw food.
        
        Args:
            None.
        
        Returns:
            Any: Value produced by the routine.
        """
        global last_food_pos
        if not dpg.does_item_exist("canvas"):
            return

        parent_tag = "slith_dynamic_layer" if dpg.does_item_exist("slith_dynamic_layer") else "canvas"

        fx, fy = food
        last_food_pos = (fx, fy)
        pulse = HEX_SIZE / 1.8 + 2.0 * math.sin(6.0 * effects_time)
        dpg.draw_circle(center=(fx, fy), radius=pulse, color=(255, 0, 0), fill=(255, 255, 0, 235), parent=parent_tag)

        dpg.draw_text([fx - 15.75, fy - 4.5], "Mal-CoA", color=(0, 0, 0), size=9, parent=parent_tag)

        # floating messages
        for msg in floating_messages[:]:
            elapsed = effects_time - msg["start_tick"]
            if elapsed < 2.0:
                dy = -20 * elapsed
                alpha = int(255 * (1.0 - elapsed))
                col = (255, 0, 0, alpha)
                dpg.draw_text((msg["atom_pos"][0],   msg["atom_pos"][1] + dy),   msg["atom"],   color=col, size=14, parent=parent_tag)
                if msg["charge"]:
                    dpg.draw_text((msg["charge_pos"][0], msg["charge_pos"][1] + dy), msg["charge"], color=col, size=10, parent=parent_tag)
            else:
                floating_messages.remove(msg)

        if is_game_over:
            length = 4 + 2 * food_eaten
            line1 = "GAME OVER"
            line2 = f"score: {length}"
            line3 = " NEW HIGH SCORE!" if has_shown_new_record else None
            cx = (GRID_MINX + GRID_MAXX) / 2.0
            cy = (GRID_MINY + GRID_MAXY) / 2.0

            def _txt_sz(t: Any, s: Any) -> Any:
                """
                Execute the txt sz routine.
                
                Args:
                    t (Any): Input accepted by this routine.
                    s (Any): Input accepted by this routine.
                
                Returns:
                    Any: Value returned by the routine.
                """
                try:
                    return dpg.get_text_size(t, size=s)
                except Exception:
                    return (len(t) * (s * 0.55), s * 1.2)

            s1, s2, s3 = 24, 20, 20
            w1, h1 = _txt_sz(line1, s1)
            w2, h2 = _txt_sz(line2, s2)
            w3, h3 = _txt_sz(line3, s3) if line3 else (0, 0)
            y1 = cy - (h1 + 6)
            y2 = cy + 6
            y3 = y2 + h2 + 4

            dpg.draw_text((cx - w1/2.0, y1), line1, color=(200, 0, 0), size=s1, parent=parent_tag)
            dpg.draw_text((cx - w2/2.0, y2), line2, color=(200, 0, 0), size=s2, parent=parent_tag)
            if line3:
                dpg.draw_text((cx - w3/2.0, y3), line3, color=(255, 80, 80), size=s3, parent=parent_tag)

    # -----------------------------------------------------------------------------
    # 2.17. Spawn food
    # -----------------------------------------------------------------------------
    def spawn_food() -> Any:
        """
        Execute the spawn food routine.
        
        Args:
            None.
        
        Returns:
            Any: Value produced by the routine.
        """
        while True:
            r = random.randint(0, ROWS - 1)
            max_q = COLS if r % 2 == 0 else COLS - 1
            q = random.randint(0, max_q - 1)
            center_pt = hex_to_pixel(q, r)
            i = random.randint(0, 5)
            corner = hex_corner(center_pt, HEX_SIZE, i)
            if all(not (math.isclose(corner[0], seg[1][0], abs_tol=1e-1) and
                        math.isclose(corner[1], seg[1][1], abs_tol=1e-1)) for seg in snake_segments):
                return corner

    # -----------------------------------------------------------------------------
    # 2.18. Pause game
    # -----------------------------------------------------------------------------
    def pause_game() -> None:
        """
        Execute the pause game routine.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        global pause_start_time, paused_duration
        if is_game_over:
            return
        if not state.get("slith_is_paused", False):
            state["slith_is_paused"] = True
            dpg.set_item_label("pause_game_button", "Resume")
            pause_start_time = time.monotonic()
        else:
            paused_duration += time.monotonic() - pause_start_time
            state["slith_is_paused"] = False
            dpg.set_item_label("pause_game_button", "Pause")

    # -----------------------------------------------------------------------------
    # 2.19. Start game
    # -----------------------------------------------------------------------------
    def start_game() -> None:
        """
        Execute the start game routine.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        global snake_segments, direction, turn_flag, auto_turn, forced_vertical, speed, start_time
        global is_game_over, food, has_shown_new_record, slith_record, last_update_time, _last_render_time
        global food_eaten, pending_growth

        snake_segments.clear()
        turn_flag = None
        auto_turn = "right"
        forced_vertical = None
        speed = 0.6
        state["slith_is_paused"] = False
        dpg.set_item_label("pause_game_button", "Pause")
        is_game_over = False
        start_time = time.monotonic()
        last_update_time = time.monotonic()
        food_eaten = 0
        pending_growth = 0

        center_pt = hex_to_pixel(start_q, start_r)
        # 4 consecutive corners around the hex for 3 initial segments (4 C)
        i0 = (start_dir - 1) % 6
        i1 = start_dir
        i2 = (start_dir + 1) % 6
        i3 = (start_dir + 2) % 6
        p0 = hex_corner(center_pt, HEX_SIZE, i0)
        p1 = hex_corner(center_pt, HEX_SIZE, i1)
        p2 = hex_corner(center_pt, HEX_SIZE, i2)
        p3 = hex_corner(center_pt, HEX_SIZE, i3)

        snake_segments.append((p2, p3))  # head segment
        snake_segments.append((p1, p2))
        snake_segments.append((p0, p1))  # tail segment
        direction = i3  # moving from p2 to p3

        has_shown_new_record = False
        slith_record = read_slith_record()

        food = spawn_food()
        draw_board()
        draw_food()
        _last_render_time = time.monotonic()

    # -----------------------------------------------------------------------------
    # 2.20. Game over
    # -----------------------------------------------------------------------------
    def game_over() -> None:
        """
        Execute the game over routine.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        global is_game_over
        state["slith_is_paused"] = True
        is_game_over = True

    with dpg.child_window(
        tag="slith_inner_window",
        parent="slith_main_window",
        no_scrollbar=True,
        horizontal_scrollbar=False,
        no_scroll_with_mouse=True,
        border=False,
        width=canvas_width,
        height=canvas_height,
        pos=(0, 0),
    ):
        with dpg.drawlist(tag="canvas", width=canvas_width, height=canvas_height):
            with dpg.draw_layer(tag="slith_static_layer"):
                pass
            with dpg.draw_layer(tag="slith_dynamic_layer"):
                pass

    PAD_BACKDROP = 8
    GAP_BELOW = 8

    controls_x = int(GRID_MINX - PAD_BACKDROP)
    controls_w = int((GRID_MAXX - GRID_MINX) + 2 * PAD_BACKDROP)
    controls_y = int(GRID_MAXY + PAD_BACKDROP + GAP_BELOW)

    if dpg.does_item_exist("slith_controls_window"):
        dpg.configure_item(
            "slith_controls_window",
            pos=(controls_x, controls_y),
            width=controls_w,
            height=72,
        )

    with dpg.child_window(
        tag="slith_controls_window",
        parent="slith_main_window",
        no_scrollbar=True,
        horizontal_scrollbar=False,
        no_scroll_with_mouse=True,
        border=False,
        pos=(controls_x, controls_y),
        width=controls_w,
        height=72,
    ):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Pause", tag="pause_game_button", width=80, height=30, callback=pause_game)
            dpg.add_button(label="Restart", tag="restart_game_button", width=80, height=30, callback=start_game)
            dpg.add_text("Length: 4", tag="length_text", color=text_color)
            dpg.add_text("butanoic acid (butyric acid)", tag="name_text", color=text_color)
            dpg.add_text(f"High Score: {slith_record}", tag="record_text", color=text_color)

    try:
        apply_inner_child_theme("slith_inner_window")
        apply_inner_child_theme("slith_controls_window")
    except Exception:
        pass

    start_r = ROWS // 2
    start_q = COLS // 2
    center_pt = hex_to_pixel(start_q, start_r)
    start_dir = 0

    # 3 segmenti anche alla prima apertura (coerente con start_game)
    i0 = (start_dir - 1) % 6
    i1 = start_dir
    i2 = (start_dir + 1) % 6
    i3 = (start_dir + 2) % 6
    p0 = hex_corner(center_pt, HEX_SIZE, i0)
    p1 = hex_corner(center_pt, HEX_SIZE, i1)
    p2 = hex_corner(center_pt, HEX_SIZE, i2)
    p3 = hex_corner(center_pt, HEX_SIZE, i3)

    snake_segments.clear()
    snake_segments.append((p2, p3))  # head
    snake_segments.append((p1, p2))
    snake_segments.append((p0, p1))  # tail
    direction = i3
    state["slith_is_paused"] = False
    if dpg.does_item_exist("pause_game_button"):
        dpg.set_item_label("pause_game_button", "Pause")
    food = spawn_food()
    draw_board()
    draw_food()

    if not dpg.does_item_exist("slith_key_press_handler"):
        dpg.add_key_press_handler(tag="slith_key_press_handler", parent="handler_registry", callback=on_key)

    # -----------------------------------------------------------------------------
    # 2.21. Post layout init
    # -----------------------------------------------------------------------------
    def _post_layout_init() -> None:
        """
        Execute the post layout init routine.
        
        Args:
            None.
        
        Returns:
            None: This routine updates state or performs side effects in place.
        """
        try:
            pw, ph = dpg.get_item_rect_size("slith_main_window")
        except Exception:
            pw, ph = state.get("slith_main_win_width", 1280), state.get("slith_main_win_height", 800)
        _apply_sizes(int(pw), int(ph))

    dpg.set_frame_callback(dpg.get_frame_count() + 2, _post_layout_init)

    if not dpg.does_item_exist("slith_parent_resize_handlers"):
        with dpg.item_handler_registry(tag="slith_parent_resize_handlers"):
            dpg.add_item_resize_handler(
                callback=lambda s, a, u: _apply_sizes(*dpg.get_item_rect_size("slith_main_window"))
            )
        dpg.bind_item_handler_registry("slith_main_window", "slith_parent_resize_handlers")

    _schedule_next_frames()
