"""
=====================
lmm_abort.py
=====================

Analysis interruption and cancellation handler.

Provides user confirmation dialogs and clean shutdown routines when the analysis
workflow is manually stopped. Ensures proper thread termination and state reset
without leaving temporary files or inconsistent results.
"""

# =============================================================================
# =============================================================================
# 1. Import module dependencies
# 2. Confirm cancellation

import os
import shutil
import threading
import dearpygui.dearpygui as dpg
from typing import Any


# -----------------------------------------------------------------------------
# 2. Confirm cancellation
# -----------------------------------------------------------------------------
def confirm_cancellation(state: dict[str, Any]) -> None:

    print("Analysis aborted.")
    state["abort_analysis"] = True                     # Signal downstream steps to halt
    if ("analysis_thread" in state 
        and state["analysis_thread"].is_alive()
        and threading.current_thread() is not state["analysis_thread"]):
        state["analysis_thread"].join()
        
    if "timer_running" in state and state["timer_running"]:
        state["timer_running"] = False                 # Request timer thread to stop
        state["timer_thread"].join()                   # Wait for a clean thread shutdown

    dpg.hide_item("stop_button")                       # Hide the stop control
    dpg.show_item("confirm_button")                    # Show the confirm button again

    dpg.set_viewport_title(f"SARgate - {os.path.basename(state['work_dir'])} (Aborted)")

    if os.path.exists(state["work_dir"]):
        shutil.rmtree(state["work_dir"])               # Delete job-specific working folder

    dpg.configure_item("cancel_confirm_popup", show=False)  # Dismiss modal confirmation
    
