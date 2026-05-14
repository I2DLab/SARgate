"""
====================
lmm_workflow.py
====================

Analysis workflow orchestration.

Controls the execution of the complete analysis pipeline once the user confirms
settings. Coordinates data preparation, scaffold extraction, R-group decomposition,
and subsequent analyses, ensuring proper task sequencing and consistent logging.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Confirm and run
# 3. Prepare working directory
# 4. Run script threaded
# 5. Try run script
# 6. Run script

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import time
import threading
import dearpygui.dearpygui as dpg
from typing import Any
from app.utils.callbacks import (
    close_current_job,
    append_to_log,
    change_tab,
    activate_main_tab,
)
from app.lmm.lmm_gui import (
    update_execution_time,
    update_library_preparation_status,
)
from app.lmm.lmm_read_local import (
    read_sdf,
    create_from_csv,
    create_from_smi_or_txt,   
)
from app.lmm.lmm_read_database import (
    create_sdf_from_chembl,
    create_sdf_from_pubchem,
    merge_fetched_sdfs
)
from app.lmm.lmm_preparation import library_preparation
from app.lmm.lmm_substructure import scaffold_analysis
from app.lmm.lmm_decomposition import r_groups_decomposition
from app.lmm.lmm_finalize import (
    save_properties_dictionaries,
    open_overview_tab
)
from app.lmm.lmm_abort import confirm_cancellation


# -----------------------------------------------------------------------------
# 2. Confirm and run
# -----------------------------------------------------------------------------
def confirm_and_run(state: dict[str, Any]) -> None:
    """
    Callback function triggered by the "RUN" button.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    close_current_job(state)

    selected_file_name = state["selected_file_name"]
    checkbox_states = state["checkbox_states"]
    output_dir = state["output_dir"]
    custom_job_name = checkbox_states.get("Job name", "").strip()

    # When creating an SDF from an online database, predefine the resulting file name.
    if checkbox_states["Input source"] == "Database":
        selected_file_name = f"{checkbox_states['Job name']}.sdf"
        state["selected_file_name"] = selected_file_name

    # Proceed only if a supported file is selected (.sdf, .csv, .tsv, .xlsx, .smi, .txt).
    if selected_file_name != "" and selected_file_name.endswith((".sdf", ".csv", ".tsv", ".xlsx", ".smi", ".txt")):
        extension = checkbox_states.get("File extension", "")
        if extension in ["csv", "tsv", "xlsx"] and not state.get("smiles_column"):
            append_to_log(state, "❌ No SMILES column selected for the chosen table file.")
            update_library_preparation_status(
                "   Error: select and confirm a SMILES column before running the analysis",
                state,
                separator=True,
            )
            return

        locked_text_tabs = state.get("locked_text_tabs")
        if isinstance(locked_text_tabs, set):
            locked_text_tabs.discard("analysis_tab")
        top_nav_button_enabled = state.get("top_nav_button_enabled")
        if isinstance(top_nav_button_enabled, dict):
            top_nav_button_enabled["analysis_nav_button"] = True
        refresh_text_tab_themes = state.get("refresh_text_tab_themes")
        if callable(refresh_text_tab_themes):
            try:
                refresh_text_tab_themes()
            except Exception:
                pass
        refresh_top_nav_selection = state.get("refresh_top_nav_selection")
        if callable(refresh_top_nav_selection):
            try:
                refresh_top_nav_selection()
            except Exception:
                pass

        if state["current_tab"] != "analysis_tab":
            if dpg.does_item_exist("tab_bar") and dpg.does_item_exist("analysis_tab"):
                try:
                    dpg.set_value("tab_bar", "analysis_tab")
                except Exception:
                    pass
            activate_main_tab("analysis_tab", state)
            if dpg.does_item_exist("tab_bar"):
                try:
                    change_tab(state)
                except Exception:
                    pass

        if custom_job_name:
            # Use custom name provided by the user
            base_work_dir = os.path.join(output_dir, custom_job_name)
        else:
            # Use the input file name when no custom job name is provided.
            if selected_file_name.endswith((".sdf", ".csv", ".tsv", ".smi", ".txt")):
                base_work_dir = os.path.join(output_dir, selected_file_name[:-4])
            elif selected_file_name.endswith(".xlsx"):
                base_work_dir = os.path.join(output_dir, selected_file_name[:-5])
        work_dir = base_work_dir
        counter = 2
        while os.path.exists(work_dir):
            work_dir = f"{base_work_dir}({counter})"
            counter += 1
        os.makedirs(work_dir, exist_ok=True)
        state["work_dir"] = work_dir
        state["report_dir"] = os.path.join(work_dir, "reports")
        os.makedirs(state["report_dir"], exist_ok=True)

        # Update run/stop/loading controls to reflect the running state.
        dpg.hide_item("confirm_button")
        dpg.show_item("stop_button")

        # Update the viewport title with the working directory name.
        dpg.set_viewport_title(f"SARgate - {os.path.basename(state['work_dir'])} (Running ...)")
        # Reset the abort flag at the start of a new run.
        state["abort_analysis"] = False 
        # Launch the threaded analysis and also print contextual information to console.
        append_to_log(state, f"Launching analysis for file '{selected_file_name}'")
        append_to_log(state, f"     Working directory created: {state['work_dir']}")
        run_script_threaded(state)

    else:
        append_to_log(state, "❌ No file selected.")


# -----------------------------------------------------------------------------
# 3. Prepare working directory
# -----------------------------------------------------------------------------
def prepare_working_directory(state: dict[str, Any]) -> None:
    """
    Creates all required subdirectories and starts the execution timer.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    # Define subfolders to store subsets, images, reports, and summaries.
    state["subset_dir"] =     os.path.join(state["work_dir"], "sdf_files")
    state["image_dir"] =      os.path.join(state["work_dir"], "images")
    state["report_dir"] =     os.path.join(state["work_dir"], "reports")
    state["summary_dir"] =    os.path.join(state["work_dir"], "summary")

    os.makedirs(state["subset_dir"], exist_ok=True)
    os.makedirs(state["image_dir"], exist_ok=True)
    os.makedirs(state["report_dir"], exist_ok=True)
    os.makedirs(state["summary_dir"], exist_ok=True)

    # Record the start time, open the execution-time window, and spawn the timer thread.
    state["start_time"] = time.time()
    state["timer_running"] = True
    state["timer_thread"] = threading.Thread(
        target=lambda: update_execution_time(state),
        daemon=True
    )
    state["timer_thread"].start()

    # Write a header with the basic run context and user-defined settings.
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(os.path.join(state["report_dir"], "log.txt"), "w") as f:
        f.write("                    SARgate Analysis Log\n")
        f.write("\n========================================\n\n")
        f.write(f"Selected file: {state['selected_file_name']}\n")
        f.write(f"Working directory: {state['work_dir']}\n")
        f.write(f"Launch date/time: {current_time}\n")
        f.write("\n========================================\n\n")
        f.write("                    USER SETTINGS\n\n")
        for option, value in state["checkbox_states"].items():
            f.write(f"{option}: {value}\n")
        


# -----------------------------------------------------------------------------
# 4. Run script threaded
# -----------------------------------------------------------------------------
def run_script_threaded(state: dict[str, Any]) -> None:
    """
    Launches the analysis in a separate thread.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    thread = threading.Thread(target=run_script, args=(state,), daemon=True)
    state["analysis_thread"] = thread
    thread.start()


# -----------------------------------------------------------------------------
# 5. Try run script
# -----------------------------------------------------------------------------
def try_run_script(state: dict[str, Any]) -> None:
    """
    Wrapper to run the main analysis function with error handling.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    try:
        run_script(state)

    # On exception, restore UI state, show error window, and confirm cancellation ===
    except Exception as e:
        dpg.hide_item("stop_button")
        dpg.show_item("confirm_button")
        confirm_cancellation(state)

        if dpg.does_item_exist("cover_layer"):
            dpg.delete_item("cover_layer")
            
        with dpg.window(label="Analysis Error", tag="error_window", modal=True, no_resize=True, 
            no_collapse=True, autosize=True, on_close=dpg.delete_item("error_window")):
            dpg.add_text(f"An error occurred during the analysis:\n\n{e}")
            dpg.add_button(label="OK", width=75, height=25, callback=lambda: dpg.delete_item("error_window"))
        


# -----------------------------------------------------------------------------
# 6. Run script
# -----------------------------------------------------------------------------
def run_script(state: dict[str, Any]) -> None:
    """
    Executes the full chemoinformatics workflow:.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    if state.get("abort_analysis", False):
        return

    checkbox_states = state["checkbox_states"]

    if dpg.does_item_exist("analysis_tab"):
        if not dpg.is_item_shown("analysis_tab"):
            dpg.show_item("analysis_tab")

    prepare_working_directory(state)

    # Respect early cancellation before starting the input stage.
    if state.get("abort_analysis", False):
        confirm_cancellation(state)
        return
    
    append_to_log(state, f"\n========================================\n\n                    INPUT READING\n")

    if checkbox_states["Input source"] == "Database":
        try:
            if checkbox_states["Database to search"] == "ChEMBL":
                # Path 1a: Create SDF file from ChEMBL API
                append_to_log(state, f"Creating SDF file from ChEMBL API")
                create_sdf_from_chembl(state)
            elif checkbox_states["Database to search"] == "PubChem":
                # Path 1b: Create SDF file from PubChem API
                append_to_log(state, f"Creating SDF file from PubChem API")
                create_sdf_from_pubchem(state)
            elif checkbox_states["Database to search"] == "ChEMBL and PubChem":
                # Path 1c: Create SDF file from both ChEMBL and PubChem APIs
                append_to_log(state, f"Creating SDF file from both ChEMBL and PubChem APIs")
                create_sdf_from_pubchem(state)
                if state.get("abort_analysis", False):
                    confirm_cancellation(state)
                    return
                
                create_sdf_from_chembl(state)
                if state.get("abort_analysis", False):
                    confirm_cancellation(state)
                    return
                
                merge_fetched_sdfs(state)
                
        except:
            append_to_log(state, "❌ Invalid target ID.")
            update_library_preparation_status("   Error: invalid target ID", state, separator=True)
            confirm_cancellation(state)
            return

    elif checkbox_states["Input source"] == "Local":
        append_to_log(state, f"Reading local file '{state['selected_file_name']}'")
        extension = checkbox_states["File extension"]
        # Path 2a: Read local SDF file
        if extension == "sdf":
            read_sdf(state)
        # Path 2b: Create SDF file from local CSV/TSV/XLSX
        elif extension in ["csv", "tsv", "xlsx"]:
            create_from_csv(state)
        # Path 2c: Create SDF file from local SMI or TXT
        elif extension in ("smi", "txt"):
            create_from_smi_or_txt(state)

    # Respect early cancellation before processing the library.
    if state.get("abort_analysis", False):
        confirm_cancellation(state)
        return
    
    append_to_log(state, f"\n========================================\n\n                    LIBRARY PREPARATION\n")
    library_preparation(state)

    # Respect early cancellation before launching substructure analysis.
    if state.get("abort_analysis", False):
        confirm_cancellation(state)
        return
    if "supplier" not in state:
        append_to_log(state, "Analysis aborted.")
        confirm_cancellation(state)
        return

    append_to_log(state, f"\n========================================\n\n                    SUBSTRUCTURE ANALYSIS\n")
    scaffold_analysis(state)

    # Respect early cancellation before decomposition stage.
    if state.get("abort_analysis", False):
        confirm_cancellation(state)
        return

    append_to_log(state, f"\n========================================\n\n                    R-GROUP DECOMPOSITION\n")
    r_groups_decomposition(state)
    
    # Respect early cancellation before final saving and UI teardown.
    if state.get("abort_analysis", False):
        confirm_cancellation(state)
        return
    
    # Persist computed properties and stop the timer thread.
    save_properties_dictionaries(state)
    state["timer_running"] = False
    state["timer_thread"].join()
    dpg.hide_item("stop_button")
    dpg.show_item("confirm_button")

    append_to_log(state, f"\n========================================\n\n                    ANALYSIS COMPLETED\n")
    execution_time_label = dpg.get_item_configuration("execution_time_text").get("label", "00:00:00")
    append_to_log(state, f"Execution time: {execution_time_label}\n")

    open_overview_tab(state)
